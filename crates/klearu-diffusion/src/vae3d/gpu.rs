//! Metal GPU path for the Wan 3D causal VAE.
//!
//! Mirrors [`super`], which `vae_reference_matches_diffusers` pins against
//! diffusers to ~2e-6 — so agreeing with the CPU path here is a real guarantee.
//!
//! # Causal conv3d without a conv3d kernel
//!
//! A `[kt, kh, kw]` convolution is a sum of `kt` 2D convolutions, one per
//! temporal tap, each applied to a different frame and accumulated:
//!
//! ```text
//! out[t] = sum over a in 0..kt of  conv2d(x[t + a - pad_left], W[:, :, a])
//! ```
//!
//! So we run `kt` **batched** 2D convolutions over all frames at once and fold
//! each result in at its own frame offset. That reuses the existing MPSGraph
//! conv2d — which is well optimised — instead of adding a 3D kernel, and costs
//! `kt` dispatches per conv rather than one per (frame, tap).
//!
//! Causality falls out of the offsets: the left padding is `2 * pad_t` frames
//! on the past side only, so taps that would reach past the start contribute
//! nothing.

use metal::Buffer;

use crate::metal_backend::{
    add_shifted_f16_gpu, bias_add_f16_gpu, blit_copy_f16, dup_up_3d_f16_gpu, eadd_f16_gpu,
    interleave_time_f16_gpu, mpsgraph_conv2d, mpsgraph_conv2d_half, nearest_upsample_2x_f16_gpu,
    extract_frame_pc_f16_gpu, insert_frame_pc_f16_gpu, rms_norm_channels_f16_gpu,
    sgemm_f16_a_btrans_buf, sgemm_f16_buf, silu_f16_gpu, softmax_rows_scaled_f16_gpu,
    transpose_outer_f16_gpu, upload_f32_as_half_buffer, upload_f32_as_f16_buffer, zero_buffer_gpu, GpuTensor,
};

use super::blocks::{Resample, ResampleMode};

use super::{CausalConv3d, RmsNormChannel, Vol};

/// Upload a `[C, T, H, W]` volume to the GPU as bf16.
/// Storage type for decoder activations.
///
/// IEEE f16 by default (10 mantissa bits); `KLEARU_VAE_BF16=1` reverts to
/// bfloat16 (8 bits) for A/B measurement.
pub fn decoder_uses_half() -> bool {
    // Cached: see `wan::gpu::dit_uses_half` — `getenv` in an allocation hot
    // path serialises on a global lock.
    thread_local! {
        static HALF: bool = std::env::var_os("KLEARU_VAE_BF16").is_none();
    }
    HALF.with(|v| *v)
}

pub fn upload_vol(v: &Vol) -> GpuTensor {
    let shape = vec![v.c, v.t, v.h, v.w];
    if decoder_uses_half() {
        GpuTensor::upload_f32_as_half(shape, &v.data)
    } else {
        GpuTensor::upload_f32_as_f16(shape, &v.data)
    }
}

/// Allocate a decoder tensor in whichever storage the decoder is using.
fn new_decoder_tensor(shape: Vec<usize>) -> GpuTensor {
    if decoder_uses_half() { GpuTensor::new_half(shape) } else { GpuTensor::new_f16(shape) }
}

/// The storage tag matching [`new_decoder_tensor`], for wrapping a buffer a
/// convolution already returned.
fn decoder_dtype() -> crate::metal_backend::GpuDtype {
    use crate::metal_backend::GpuDtype;
    if decoder_uses_half() { GpuDtype::Half } else { GpuDtype::F16 }
}

/// Convolution in whichever storage the decoder is using.
#[allow(clippy::too_many_arguments)]
fn mpsgraph_conv2d_dyn(
    input: &Buffer, weight: &[f32], weight_id: u64,
    n: usize, in_c: usize, h_in: usize, w_in: usize,
    out_c: usize, kh: usize, kw: usize, stride: usize, pad: usize,
) -> (Buffer, usize, usize) {
    if decoder_uses_half() {
        crate::metal_backend::mpsgraph_conv2d_half(
            input, weight, weight_id, n, in_c, h_in, w_in, out_c, kh, kw, stride, pad)
    } else {
        crate::metal_backend::mpsgraph_conv2d(
            input, weight, weight_id, n, in_c, h_in, w_in, out_c, kh, kw, stride, pad)
    }
}

/// Upload a decoder weight in whichever storage the decoder is using.
fn upload_decoder_weight(v: &[f32]) -> Buffer {
    if decoder_uses_half() {
        crate::metal_backend::upload_f32_as_half_buffer(v)
    } else {
        crate::metal_backend::upload_f32_as_f16_buffer(v)
    }
}

/// Download a GPU volume back into a [`Vol`].
pub fn download_vol(g: &GpuTensor, c: usize, t: usize, h: usize, w: usize) -> Vol {
    Vol::new(g.download_to_f32(), c, t, h, w)
}

/// Wan's channel norm on the GPU.
pub struct GpuRmsNormChannel {
    pub gamma: Buffer,
    pub channels: usize,
}

impl GpuRmsNormChannel {
    pub fn from_cpu(n: &RmsNormChannel) -> Self {
        assert!(n.bias.is_none(), "Wan VAE norms have no bias");
        Self {
            gamma: upload_decoder_weight(&n.gamma),
            channels: n.gamma.len(),
        }
    }

    /// In place over a `[C, T, H, W]` volume.
    pub fn forward(&self, x: &mut GpuTensor, t: usize, h: usize, w: usize) {
        rms_norm_channels_f16_gpu(x, &self.gamma, self.channels, t * h * w);
    }
}

/// A causal 3D convolution evaluated as `kt` batched 2D convolutions.
///
/// Symmetric spatial padding covers the whole decoder. The encoder's
/// `downsample2d` needs an asymmetric `ZeroPad2d((0,1,0,1))`, which MPSGraph
/// cannot express directly; it is materialised with `pad_end_1_f16_gpu` before
/// the convolution, which then runs with zero padding.
pub struct GpuCausalConv3d {
    /// Per temporal tap, the `[out_c, in_c, kh, kw]` slice as f32 —
    /// `mpsgraph_conv2d` takes its weights on the host and caches the graph.
    pub taps: Vec<Vec<f32>>,
    /// One process-unique id per tap, for the uploaded-weight cache. Assigned
    /// at construction rather than derived from the slice address: convs are
    /// created and dropped freely (the encoder builds them per call), and a
    /// freed address handed to the next allocation made the cache serve one
    /// conv's weights to another. That produced confidently wrong output with
    /// no error — cos_sim 0.01 from a stack whose every part tested clean.
    pub tap_ids: Vec<u64>,
    pub in_c: usize,
    pub out_c: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride_hw: usize,
    /// Temporal stride. Only `Downsample3d`'s time conv uses > 1.
    pub stride_t: usize,
    pub pad_hw: usize,
    /// Extra padding on the *end* of H and W only. The encoder's downsample
    /// convs are `ZeroPad2d((0,1,0,1))`, which MPSGraph's symmetric padding
    /// cannot express, so it is materialised with `pad_end_1_f16_gpu`.
    pub pad_end_hw: (usize, usize),
    pub pad_t: usize,
    /// Per-output-channel bias, resident on the GPU.
    pub bias: Option<Buffer>,
}

impl Drop for GpuCausalConv3d {
    fn drop(&mut self) {
        // Release this conv's uploaded weights. Without it, anything that
        // rebuilds convs grows the upload cache by the model's full weight
        // size every time.
        for id in &self.tap_ids {
            crate::metal_backend::release_weight_id(*id);
        }
    }
}

impl GpuCausalConv3d {
    pub fn from_cpu(c: &CausalConv3d) -> Self {
        let (kt, kh, kw) = c.kernel;
        assert_eq!(c.stride.1, c.stride.2, "non-square spatial stride");
        assert!(
            c.pad_end_hw == (0, 0) || c.pad_end_hw == (1, 1),
            "only ZeroPad2d((0,1,0,1)) end-padding is implemented, got {:?}",
            c.pad_end_hw
        );

        // Weight is [out, in, kt, kh, kw]; slice out each temporal tap.
        let spatial = kh * kw;
        let per_in = kt * spatial;
        let mut taps = Vec::with_capacity(kt);
        for a in 0..kt {
            let mut w = Vec::with_capacity(c.out_c * c.in_c * spatial);
            for o in 0..c.out_c {
                for i in 0..c.in_c {
                    let base = (o * c.in_c + i) * per_in + a * spatial;
                    w.extend_from_slice(&c.weight[base..base + spatial]);
                }
            }
            taps.push(w);
        }
        let tap_ids =
            (0..taps.len()).map(|_| crate::metal_backend::next_weight_id()).collect();
        Self {
            taps,
            tap_ids,
            in_c: c.in_c,
            out_c: c.out_c,
            kh,
            kw,
            stride_hw: c.stride.1,
            stride_t: c.stride.0,
            pad_hw: c.pad.1,
            pad_end_hw: c.pad_end_hw,
            pad_t: c.pad.0,
            bias: c.bias.as_ref().map(|b| upload_decoder_weight(b)),
        }
    }

    /// Output temporal length for an input of `t` frames (kernel 3 / pad 1
    /// preserves it).
    pub fn out_frames(&self, t: usize) -> usize {
        let left = 2 * self.pad_t;
        (left + t).saturating_sub(self.taps.len()) / self.stride_t + 1
    }

    /// Whole-clip form: no cached frames from a previous chunk.
    pub fn forward(
        &self, x: &GpuTensor, t: usize, h: usize, w: usize,
    ) -> (GpuTensor, usize, usize, usize) {
        self.forward_cached(x, t, h, w, None)
    }

    /// `x`: `[in_c, T, H, W]`. Returns `[out_c, T_out, H_out, W_out]`.
    ///
    /// `cache` is the previous chunk's trailing frames as `[in_c, Tc, H, W]`.
    /// They are prepended and the past-side padding shrinks by the same
    /// amount, exactly as on the CPU — so a chunked run reproduces a
    /// whole-clip run.
    ///
    /// The concatenation happens in `[T, C, P]` space, where frames are
    /// contiguous and it is two blits; in the native `[C, T, P]` layout it
    /// would be a strided copy per channel.
    pub fn forward_cached(
        &self,
        x: &GpuTensor,
        t: usize,
        h: usize,
        w: usize,
        cache: Option<&GpuTensor>,
    ) -> (GpuTensor, usize, usize, usize) {
        let p = h * w;
        let frame_in = self.in_c * p;
        let cache_t = match cache {
            Some(c) if self.pad_t > 0 => (c.elements() / frame_in).min(2 * self.pad_t),
            _ => 0,
        };
        let eff_t = cache_t + t;
        let left = 2 * self.pad_t - cache_t;

        let t_out = (left + eff_t).saturating_sub(self.taps.len()) / self.stride_t + 1;
        // End padding is materialised below, so from here on the convolution
        // sees a slightly larger plane with symmetric (zero) padding.
        let h_pad = h + self.pad_end_hw.0;
        let w_pad = w + self.pad_end_hw.1;
        let h_out = (h_pad + 2 * self.pad_hw - self.kh) / self.stride_hw + 1;
        let w_out = (w_pad + 2 * self.pad_hw - self.kw) / self.stride_hw + 1;

        // One frame in, one frame out — every conv of a single-image decode or
        // encode, and the first latent frame of a video. Exactly one tap lands
        // (the one at `a == left`; the general loop below would discard the
        // rest), and `[C, 1, P]` is the same memory as `[1, C, P]`, so the
        // transposes, the zeroed accumulator and the other taps' convolutions
        // are all pure overhead. At 1024x1024 that is two full-resolution
        // convolutions thrown away per 3x3x3 conv, plus three tensor copies.
        // The result is bit-identical: the general path adds this same conv
        // output into zeros and copies it twice.
        if eff_t == 1 && t_out == 1 && left < self.taps.len() {
            let padded;
            let input = if self.pad_end_hw == (0, 0) {
                &x.buffer
            } else {
                padded = crate::metal_backend::pad_end_1_f16_gpu(x, self.in_c, h, w);
                &padded.buffer
            };
            let (y, _, _) = mpsgraph_conv2d_dyn(
                input,
                &self.taps[left],
                self.tap_ids[left],
                1, self.in_c, h_pad, w_pad,
                self.out_c, self.kh, self.kw,
                self.stride_hw, self.pad_hw,
            );
            let mut out = GpuTensor {
                buffer: y,
                shape: vec![self.out_c, 1, h_out, w_out],
                dtype: decoder_dtype(),
            };
            if let Some(b) = &self.bias {
                bias_add_f16_gpu(&mut out, b, h_out * w_out, self.out_c);
            }
            return (out, 1, h_out, w_out);
        }

        // mpsgraph_conv2d batches over frames, so view [C, T, HW] as
        // [T, C, HW]. Transposing on the GPU — a host round trip here would
        // cost more than the convolution.
        let x_tchw = transpose_outer_f16_gpu(x, self.in_c, t, p);
        let input = if cache_t > 0 {
            let c = cache.unwrap();
            let c_t = c.elements() / frame_in;
            let c_tchw = transpose_outer_f16_gpu(c, self.in_c, c_t, p);
            let joined = new_decoder_tensor(vec![eff_t, self.in_c, p]);
            // Take the *last* cache_t frames when more were supplied.
            blit_copy_f16(&c_tchw.buffer, (c_t - cache_t) * frame_in, &joined.buffer, 0,
                          cache_t * frame_in);
            blit_copy_f16(&x_tchw.buffer, 0, &joined.buffer, cache_t * frame_in, t * frame_in);
            joined
        } else {
            x_tchw
        };

        // Materialise the asymmetric end padding, in [T, C, H, W] layout where
        // each (t, c) is an independent plane.
        let input = if self.pad_end_hw == (0, 0) {
            input
        } else {
            crate::metal_backend::pad_end_1_f16_gpu(&input, eff_t * self.in_c, h, w)
        };

        let acc = new_decoder_tensor(vec![t_out * self.out_c * h_out * w_out]);
        zero_buffer_gpu(&acc.buffer, acc.elements() * 2);
        let frame_out = self.out_c * h_out * w_out;

        // Input frame `s` feeds output frame `(s + left - a) / stride_t`, and
        // only when that divides exactly — with stride 1 every frame lands,
        // with stride 2 every other one does.
        let lands = |s: usize, a: usize| -> Option<usize> {
            let shifted = s as isize + left as isize - a as isize;
            if shifted < 0 || shifted % self.stride_t as isize != 0 {
                return None;
            }
            let dst = (shifted / self.stride_t as isize) as usize;
            (dst < t_out).then_some(dst)
        };
        for (a, tap) in self.taps.iter().enumerate() {
            // A tap none of whose frames land contributes nothing; skip its
            // convolution rather than compute and discard it.
            if !(0..eff_t).any(|s| lands(s, a).is_some()) {
                continue;
            }
            let (y, _, _) = mpsgraph_conv2d_dyn(
                &input.buffer,
                tap,
                self.tap_ids[a],
                eff_t, self.in_c, h_pad, w_pad,
                self.out_c, self.kh, self.kw,
                self.stride_hw, self.pad_hw,
            );
            for s in 0..eff_t {
                let Some(dst) = lands(s, a) else { continue };
                add_shifted_f16_gpu(
                    &acc.buffer, dst * frame_out,
                    &y, s * frame_out,
                    frame_out,
                    acc.dtype,
                );
            }
        }

        // Back to [C, T, H, W], then add bias per output channel. In a [C, P]
        // layout `bias_add_strided_f16` indexes as (gid / P) % C, which is
        // exactly the channel — so `period = P`, `bias_len = C`.
        let mut out = transpose_outer_f16_gpu(&acc, t_out, self.out_c, h_out * w_out);
        if let Some(b) = &self.bias {
            bias_add_f16_gpu(&mut out, b, t_out * h_out * w_out, self.out_c);
        }
        (out, t_out, h_out, w_out)
    }
}

/// A VAE residual block on the GPU: the decoder's most repeated unit
/// (14 of them, counting the mid block).
pub struct GpuResidualBlock {
    pub norm1: GpuRmsNormChannel,
    pub conv1: GpuCausalConv3d,
    pub norm2: GpuRmsNormChannel,
    pub conv2: GpuCausalConv3d,
    /// Present only when the channel count changes; a 1x1x1 conv with no
    /// temporal extent, so it takes no cache slot.
    pub shortcut: Option<GpuCausalConv3d>,
    pub out_c: usize,
}

impl GpuResidualBlock {
    pub fn from_cpu(b: &super::blocks::ResidualBlock) -> Self {
        Self {
            norm1: GpuRmsNormChannel::from_cpu(&b.norm1),
            conv1: GpuCausalConv3d::from_cpu(&b.conv1),
            norm2: GpuRmsNormChannel::from_cpu(&b.norm2),
            conv2: GpuCausalConv3d::from_cpu(&b.conv2),
            shortcut: b.shortcut.as_ref().map(GpuCausalConv3d::from_cpu),
            out_c: b.conv2.out_c,
        }
    }

    /// Cacheless form, for whole-clip use.
    pub fn forward(
        &self,
        x: &GpuTensor,
        t: usize,
        h: usize,
        w: usize,
        caches: Option<(&GpuTensor, &GpuTensor)>,
    ) -> GpuTensor {
        let residual = match &self.shortcut {
            Some(sc) => sc.forward(x, t, h, w).0,
            None => x.clone_data(),
        };

        let mut y = x.clone_data();
        self.norm1.forward(&mut y, t, h, w);
        silu_f16_gpu(&mut y);
        let (mut y, t1, h1, w1) =
            self.conv1.forward_cached(&y, t, h, w, caches.map(|c| c.0));

        self.norm2.forward(&mut y, t1, h1, w1);
        silu_f16_gpu(&mut y);
        let (mut y, ..) = self.conv2.forward_cached(&y, t1, h1, w1, caches.map(|c| c.1));

        eadd_f16_gpu(&mut y, &residual);
        y
    }

    /// Chunked form. conv1 then conv2, in the CPU walk order so both paths
    /// address the same cache slots.
    pub fn forward_slot(
        &self,
        x: &GpuTensor,
        t: usize,
        h: usize,
        w: usize,
        cache: &mut GpuFeatCache,
    ) -> (GpuTensor, usize, usize, usize) {
        let residual = match &self.shortcut {
            Some(sc) => sc.forward(x, t, h, w).0,
            None => x.clone_data(),
        };

        let mut y = x.clone_data();
        self.norm1.forward(&mut y, t, h, w);
        silu_f16_gpu(&mut y);
        let (mut y, t1, h1, w1) = self.conv1.forward_slot(&y, t, h, w, cache);

        self.norm2.forward(&mut y, t1, h1, w1);
        silu_f16_gpu(&mut y);
        let (mut y, t2, h2, w2) = self.conv2.forward_slot(&y, t1, h1, w1, cache);

        eadd_f16_gpu(&mut y, &residual);
        (y, t2, h2, w2)
    }
}

/// SiLU in place.
pub fn silu_gpu(x: &mut GpuTensor) {
    silu_f16_gpu(x);
}

// The host round-trip helpers this file used to carry (`transpose_ct`,
// `transpose_tc`, `add_channel_bias`, `zero`) are gone: each one cost a
// GPU->CPU->GPU copy per convolution, which at decoder resolutions is far more
// than the convolution itself. They are now `transpose_outer_f16_gpu`,
// `bias_add_f16_gpu` and `zero_buffer_gpu`.


/// One GPU cache slot, mirroring [`super::blocks::Slot`].
///
/// Not `Clone`: `GpuTensor` owns its buffer and only offers a deep
/// `clone_data`. Slots are taken with `mem::replace` and put back, so a slot's
/// contents are never duplicated.
pub enum GpuSlot {
    Empty,
    /// First chunk seen by a layer that skips its temporal conv once.
    Rep,
    Frames {
        data: GpuTensor,
        t: usize,
    },
}

/// Per-conv trailing-frame cache, resident on the GPU.
///
/// Kept on-device deliberately: at the decoder's last stage a slot is 2 frames
/// x 256 channels x 1376x768, about 1 GB in bf16. Round-tripping that to the
/// host per convolution per chunk would dwarf the convolutions.
///
/// `idx` walks slots in a fixed order and **must be rewound before each
/// chunk** — forgetting it makes every chunk re-read a fresh `Empty` slot and
/// silently take the first-chunk path again.
pub struct GpuFeatCache {
    pub slots: Vec<GpuSlot>,
    pub idx: usize,
    /// False for [`Self::single_frame`]: trailing frames are never stored.
    retain: bool,
}

impl GpuFeatCache {
    pub fn new(n: usize) -> Self {
        Self { slots: (0..n).map(|_| GpuSlot::Empty).collect(), idx: 0, retain: true }
    }

    /// A cache for a one-chunk run — a single image — that never stores the
    /// trailing frames a following chunk would read.
    ///
    /// The first chunk's semantics are untouched (every slot still reads
    /// `Empty`, `upsample3d` still skips its temporal conv, `downsample3d`
    /// still passes through), so the output is bit-identical to [`Self::new`].
    /// What it saves is memory: every causal conv otherwise keeps a copy of its
    /// input, which for a 1024x1024 decode is ~1.5 GB of activations held to
    /// serve a next chunk that never comes. Reusing one for a second chunk
    /// would silently decode it as a first chunk, so `rewind` refuses.
    pub fn single_frame(n: usize) -> Self {
        Self { retain: false, ..Self::new(n) }
    }

    pub fn rewind(&mut self) {
        assert!(
            self.retain || self.idx == 0,
            "a single_frame GpuFeatCache serves exactly one chunk"
        );
        self.idx = 0;
    }

    /// Whether trailing frames should be kept for a following chunk.
    pub fn retains(&self) -> bool {
        self.retain
    }

    fn next(&mut self) -> usize {
        let i = self.idx;
        while self.slots.len() <= i {
            self.slots.push(GpuSlot::Empty);
        }
        self.idx += 1;
        i
    }
}

impl GpuCausalConv3d {
    /// Run through a cache slot, matching `blocks::cached_conv` exactly: the
    /// new cache is taken from the input *before* the convolution, back-filled
    /// from the previous slot when this chunk is shorter than `CACHE_T`, and
    /// the conv runs against the *previous* slot's frames.
    pub fn forward_slot(
        &self,
        x: &GpuTensor,
        t: usize,
        h: usize,
        w: usize,
        cache: &mut GpuFeatCache,
    ) -> (GpuTensor, usize, usize, usize) {
        let p = h * w;
        let i = cache.next();
        if !cache.retains() {
            // Nothing will read this slot again: run against it as a first
            // chunk (it is still `Empty`) and store nothing.
            debug_assert!(matches!(cache.slots[i], GpuSlot::Empty));
            return self.forward_cached(x, t, h, w, None);
        }
        let prev = std::mem::replace(&mut cache.slots[i], GpuSlot::Empty);

        let mut cache_x = tail_frames_gpu(x, self.in_c, t, p, super::CACHE_T);
        let mut cache_t = super::CACHE_T.min(t);
        if cache_t < 2 {
            if let GpuSlot::Frames { data, t: pt } = &prev {
                let last = tail_frames_gpu(data, self.in_c, *pt, p, 1);
                cache_x = concat_time_gpu(&last, 1, &cache_x, cache_t, self.in_c, p);
                cache_t += 1;
            }
        }
        let prev_frames = match &prev {
            GpuSlot::Frames { data, .. } => Some(data),
            _ => None,
        };
        let out = self.forward_cached(x, t, h, w, prev_frames);
        cache.slots[i] = GpuSlot::Frames { data: cache_x, t: cache_t };
        out
    }
}

/// Concatenate two `[C, *, P]` volumes along time.
fn concat_time_gpu(
    a: &GpuTensor, at: usize, b: &GpuTensor, bt: usize, c: usize, p: usize,
) -> GpuTensor {
    let out = new_decoder_tensor(vec![c, at + bt, p]);
    for ci in 0..c {
        blit_copy_f16(&a.buffer, ci * at * p, &out.buffer, ci * (at + bt) * p, at * p);
        blit_copy_f16(&b.buffer, ci * bt * p, &out.buffer, (ci * (at + bt) + at) * p, bt * p);
    }
    out
}

/// Spatial resample plus, for the 3d modes, a temporal convolution.
///
/// Only the upsampling modes are implemented: the decoder is the path that
/// matters for text-to-video, and the encoder's `downsample2d` needs the
/// asymmetric `ZeroPad2d((0,1,0,1))` that [`GpuCausalConv3d`] does not support.
pub struct GpuResample {
    pub mode: ResampleMode,
    pub conv: GpuCausalConv3d,
    pub time_conv: Option<GpuCausalConv3d>,
    pub dim: usize,
}

impl GpuResample {
    pub fn from_cpu(r: &Resample) -> Self {
        let time_conv = r.time_conv.as_ref().map(GpuCausalConv3d::from_cpu);
        Self {
            mode: r.mode,
            conv: GpuCausalConv3d::from_cpu(&r.conv),
            time_conv,
            dim: r.dim,
        }
    }

    /// On the first chunk `upsample3d` skips its temporal conv entirely (the
    /// reference's `"Rep"` sentinel), which is why the first latent frame
    /// yields one output frame rather than `factor_t`.
    pub fn forward(
        &self,
        x: &GpuTensor,
        t: usize,
        h: usize,
        w: usize,
        cache: &mut GpuFeatCache,
    ) -> (GpuTensor, usize, usize, usize) {
        // Downsampling is the encoder direction: a plain strided conv, with the
        // asymmetric end padding handled inside `GpuCausalConv3d`. The temporal
        // half of Downsample3d is a no-op on the first chunk (the reference
        // seeds its cache and passes through), which is the only case this
        // path serves — `GpuEncoder3d` asserts a single chunk.
        if matches!(self.mode, ResampleMode::Downsample2d | ResampleMode::Downsample3d) {
            let (mut y, mut yt, yh, yw) = self.conv.forward(x, t, h, w);
            if self.mode == ResampleMode::Downsample3d {
                // Downsample3d consumes one cache slot on the CPU even on the
                // first chunk, where it only seeds it. Skipping it shifts every
                // later slot index by one, so a conv reads a slot another conv
                // wrote — a stack that was exact for three blocks and then
                // collapsed to cos_sim 0.73 in the fourth.
                let i = cache.next();
                if !cache.retains() {
                    // First (and only) chunk: pass through, seed nothing.
                    return (y, yt, yh, yw);
                }
                let prev = std::mem::replace(&mut cache.slots[i], GpuSlot::Empty);
                let keep = tail_frames_gpu(&y, self.dim, yt, yh * yw, 1);
                if let GpuSlot::Frames { data, t: pt } = &prev {
                    // The reference concatenates the cached frame itself rather
                    // than routing it through padding: this conv has pad_t = 0.
                    let last = tail_frames_gpu(data, self.dim, *pt, yh * yw, 1);
                    let joined = concat_time_gpu(&last, 1, &y, yt, self.dim, yh * yw);
                    let tc = self.time_conv.as_ref().expect("Downsample3d time conv");
                    let (o, ot, ..) = tc.forward(&joined, yt + 1, yh, yw);
                    y = o;
                    yt = ot;
                }
                cache.slots[i] = GpuSlot::Frames { data: keep, t: 1 };
            }
            return (y, yt, yh, yw);
        }

        let mut cur = x.clone_data();
        let mut cur_t = t;

        if self.mode == ResampleMode::Upsample3d {
            let i = cache.next();
            let slot = std::mem::replace(&mut cache.slots[i], GpuSlot::Empty);
            match slot {
                GpuSlot::Empty => cache.slots[i] = GpuSlot::Rep,
                slot => {
                    let tc = self.time_conv.as_ref().unwrap();
                    let prev = match &slot {
                        GpuSlot::Frames { data, .. } => Some(data),
                        _ => None,
                    };
                    let cache_x = tail_frames_gpu(x, self.dim, t, h * w, super::CACHE_T);
                    let (conv_out, ct, ..) =
                        tc.forward_cached(&cur, cur_t, h, w, prev);
                    // [2c, t] -> [c, 2t]: the channel halves interleave into
                    // consecutive frames.
                    cur = interleave_time_f16_gpu(&conv_out, self.dim, ct, h * w);
                    cur_t = ct * 2;
                    cache.slots[i] =
                        GpuSlot::Frames { data: cache_x, t: super::CACHE_T.min(t) };
                }
            }
        }

        // Spatial 2x nearest, then the per-frame conv. The volume is [C, T, P]
        // and the upsample is purely spatial, so each (c, t) plane is
        // independent — pass C*T planes.
        let up = nearest_upsample_2x_f16_gpu(&cur, 1, self.dim * cur_t, h, w);
        self.conv.forward(&up, cur_t, h * 2, w * 2)
    }
}

/// The last `n` frames of a `[C, T, P]` volume, as a fresh tensor.
fn tail_frames_gpu(x: &GpuTensor, c: usize, t: usize, p: usize, n: usize) -> GpuTensor {
    let n = n.min(t);
    let out = new_decoder_tensor(vec![c, n, p]);
    for ci in 0..c {
        blit_copy_f16(
            &x.buffer, (ci * t + (t - n)) * p,
            &out.buffer, ci * n * p,
            n * p,
        );
    }
    out
}

/// `DupUp3D` shortcut on the GPU.
#[allow(clippy::too_many_arguments)]
pub fn dup_up_gpu(
    x: &GpuTensor, in_c: usize, out_c: usize,
    t: usize, h: usize, w: usize, ft: usize, fs: usize, first_chunk: bool,
) -> GpuTensor {
    dup_up_3d_f16_gpu(x, in_c, out_c, t, h, w, ft, fs, first_chunk)
}

/// Score-matrix elements the mid-block attention materialises at once.
///
/// 32M is 64 MB of bf16 scores (128 MB of f32 matmul scratch). Wan 2.2's
/// bottleneck at 704x1280 is 44x80 = 3520 positions — 12.4M scores — so it
/// stays one pass. The 8x VAE is where this bites: a 1024x1024 image is
/// 16384 positions, 268M scores (0.5 GB bf16, 1 GB f32 scratch), and 2048x2048
/// would be 4.3G scores — 8.6 GB before scratch.
const ATTN_MAX_SCORES: usize = 32 << 20;

/// The VAE mid block's single-head spatial attention, on the GPU.
///
/// `head_dim` here is the full channel count (1024 in the Wan 2.2 decoder, 384
/// in Wan 2.1 / Qwen-Image), well outside the flash kernel's supported dims —
/// so this uses explicit matmuls. With Wan 2.2's 16x compression the score
/// matrix is small; with the 8x VAE it is not, so queries are processed in
/// row blocks that keep it under [`ATTN_MAX_SCORES`]. Softmax is per row, so
/// blocking changes nothing numerically.
pub struct GpuAttentionBlock {
    pub norm: GpuRmsNormChannel,
    pub to_qkv: GpuCausalConv3d,
    pub proj: GpuCausalConv3d,
    pub dim: usize,
}

impl GpuAttentionBlock {
    pub fn from_cpu(a: &super::blocks::AttentionBlock) -> Self {
        Self {
            norm: GpuRmsNormChannel::from_cpu(&a.norm),
            to_qkv: GpuCausalConv3d::from_cpu(&a.to_qkv),
            proj: GpuCausalConv3d::from_cpu(&a.proj),
            dim: a.dim,
        }
    }

    pub fn forward(&self, x: &GpuTensor, t: usize, h: usize, w: usize) -> GpuTensor {
        self.forward_blocked(x, t, h, w, ATTN_MAX_SCORES)
    }

    /// [`Self::forward`] with an explicit score budget, so tests can force
    /// the row-blocked path at sizes the CPU reference can check.
    pub fn forward_blocked(
        &self,
        x: &GpuTensor,
        t: usize,
        h: usize,
        w: usize,
        max_scores: usize,
    ) -> GpuTensor {
        let d = self.dim;
        let p = h * w;
        let scale = 1.0 / (d as f32).sqrt();
        // Query rows per block. Every block is exactly this tall, the last one
        // aligned to the bottom and overlapping its predecessor (overlapping
        // rows are recomputed to the same values) — one matmul shape per
        // call, the same trick `sgemm_f16_a_btrans_buf` uses above
        // `MPS_MAX_ROWS`.
        let rows = (max_scores / p).clamp(1, p);

        let mut y = x.clone_data();
        self.norm.forward(&mut y, t, h, w);
        // 1x1x1 conv == a per-position linear over channels.
        let (qkv, ..) = self.to_qkv.forward(&y, t, h, w);

        let attn = new_decoder_tensor(vec![d, t, p]);
        for ti in 0..t {
            // [p, d] per frame: q, k, v are the three channel thirds.
            let q = extract_frame_pc_f16_gpu(&qkv, 0, d, ti, t, p);
            let k = extract_frame_pc_f16_gpu(&qkv, d, d, ti, t, p);
            let v = extract_frame_pc_f16_gpu(&qkv, 2 * d, d, ti, t, p);

            // scores[p, p] = q[p, d] . k[p, d]^T
            // The mid-block attention runs on the bf16 MPS matmuls, which have
            // no half variant, so this stage converts in and out. Everything
            // it touches must be bf16 — a `new_half` buffer here would be
            // written as bf16 and then read as half, which the compiler cannot
            // catch because the storage type is a runtime field.
            //
            // Only convert when the decoder is actually running half storage;
            // in bf16 mode these are already bfloat and converting would
            // reinterpret the bits. Release builds drop the debug assert that
            // would otherwise catch it.
            let to_bf16 = |t: &GpuTensor| -> Option<GpuTensor> {
                t.dtype.is_half().then(|| crate::metal_backend::tensor_half_to_bf16(t))
            };
            let qb = to_bf16(&q);
            let kb = to_bf16(&k);
            let vb = to_bf16(&v);
            let (qb, kb, vb) = (
                qb.as_ref().unwrap_or(&q),
                kb.as_ref().unwrap_or(&k),
                vb.as_ref().unwrap_or(&v),
            );
            let outb = GpuTensor::new_f16(vec![p, d]);
            if rows == p {
                let mut scores = GpuTensor::new_f16(vec![p, p]);
                sgemm_f16_a_btrans_buf(&scores.buffer, &qb.buffer, &kb.buffer, p, p, d);
                softmax_rows_scaled_f16_gpu(&mut scores, p, p, scale);
                // out[p, d] = scores[p, p] . v[p, d]
                sgemm_f16_buf(&outb.buffer, &scores.buffer, &vb.buffer, p, d, p);
            } else {
                // Row blocks of q are contiguous in [p, d], so each is one
                // blit in and one blit out; k and v are shared by all blocks.
                let qc = GpuTensor::new_f16(vec![rows, d]);
                let mut sc = GpuTensor::new_f16(vec![rows, p]);
                let oc = GpuTensor::new_f16(vec![rows, d]);
                let mut start = 0usize;
                loop {
                    let last = start + rows >= p;
                    let r0 = if last { p - rows } else { start };
                    blit_copy_f16(&qb.buffer, r0 * d, &qc.buffer, 0, rows * d);
                    sgemm_f16_a_btrans_buf(&sc.buffer, &qc.buffer, &kb.buffer, rows, p, d);
                    let bad_rows = |what: &str, g: &GpuTensor, n: usize| {
                        // `KLEARU_VAE_PROBE=1`: rows of this block that came out non-finite.
                        if std::env::var_os("KLEARU_VAE_PROBE").is_none() {
                            return;
                        }
                        let v = g.download_to_f32();
                        let rs: Vec<usize> = (0..rows).filter(|r| v[r * n..(r + 1) * n].iter().any(|a| !a.is_finite())).collect();
                        if let (Some(a), Some(b)) = (rs.first(), rs.last()) {
                            println!("[vae probe]   block at row {r0}: {what} {} bad rows, {a}..={b}", rs.len());
                        }
                    };
                    bad_rows("q.k^T", &sc, p);
                    softmax_rows_scaled_f16_gpu(&mut sc, rows, p, scale);
                    sgemm_f16_buf(&oc.buffer, &sc.buffer, &vb.buffer, rows, d, p);
                    bad_rows("scores.v", &oc, d);
                    blit_copy_f16(&oc.buffer, 0, &outb.buffer, r0 * d, rows * d);
                    if last {
                        break;
                    }
                    start += rows;
                }
            }
            let out = if decoder_uses_half() {
                crate::metal_backend::tensor_bf16_to_half(&outb)
            } else {
                outb
            };
            insert_frame_pc_f16_gpu(&out, &attn, d, ti, t, p);
        }

        let (mut out, ..) = self.proj.forward(&attn, t, h, w);
        eadd_f16_gpu(&mut out, x);
        out
    }
}

/// `resnet -> (attention -> resnet) * num_layers`.
pub struct GpuMidBlock {
    pub resnets: Vec<GpuResidualBlock>,
    pub attentions: Vec<GpuAttentionBlock>,
}

impl GpuMidBlock {
    pub fn from_cpu(m: &super::blocks::MidBlock) -> Self {
        Self {
            resnets: m.resnets.iter().map(GpuResidualBlock::from_cpu).collect(),
            attentions: m.attentions.iter().map(GpuAttentionBlock::from_cpu).collect(),
        }
    }

    pub fn forward(&self, x: &GpuTensor, t: usize, h: usize, w: usize) -> GpuTensor {
        let mut y = self.resnets[0].forward(x, t, h, w, None);
        for (attn, resnet) in self.attentions.iter().zip(&self.resnets[1..]) {
            y = attn.forward(&y, t, h, w);
            y = resnet.forward(&y, t, h, w, None);
        }
        y
    }
}

/// Decoder stage: `num_res_blocks + 1` residual blocks, an optional
/// upsampler, and a `DupUp3D` shortcut added on top when upsampling.
pub struct GpuResidualUpBlock {
    pub resnets: Vec<GpuResidualBlock>,
    pub upsampler: Option<GpuResample>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub factor_t: usize,
    pub factor_s: usize,
    pub has_shortcut: bool,
}

impl GpuResidualUpBlock {
    pub fn from_cpu(b: &super::blocks::ResidualUpBlock, in_dim: usize) -> Self {
        Self {
            resnets: b.resnets.iter().map(GpuResidualBlock::from_cpu).collect(),
            upsampler: b.upsampler.as_ref().map(GpuResample::from_cpu),
            in_dim,
            out_dim: b.out_dim,
            factor_t: b.factor_t,
            factor_s: b.factor_s,
            has_shortcut: b.has_shortcut,
        }
    }

    pub fn forward(
        &self,
        x: &GpuTensor,
        t: usize,
        h: usize,
        w: usize,
        cache: &mut GpuFeatCache,
        first_chunk: bool,
    ) -> (GpuTensor, usize, usize, usize) {
        let (mut y, mut yt, mut yh, mut yw) = (x.clone_data(), t, h, w);
        for r in &self.resnets {
            let (o, ot, oh, ow) = r.forward_slot(&y, yt, yh, yw, cache);
            y = o;
            yt = ot;
            yh = oh;
            yw = ow;
        }
        if let Some(u) = &self.upsampler {
            let (o, ot, oh, ow) = u.forward(&y, yt, yh, yw, cache);
            y = o;
            yt = ot;
            yh = oh;
            yw = ow;
        }
        if self.has_shortcut {
            let sc = dup_up_3d_f16_gpu(
                x, self.in_dim, self.out_dim, t, h, w, self.factor_t, self.factor_s, first_chunk,
            );
            assert_eq!(
                sc.elements(),
                self.out_dim * yt * yh * yw,
                "DupUp3D shortcut shape disagrees with the main path"
            );
            eadd_f16_gpu(&mut y, &sc);
        }
        (y, yt, yh, yw)
    }
}

/// The encoder's residual + downsample stage, mirroring `GpuResidualUpBlock`.
/// Covers both [`super::blocks::StageKind`]s: the plain (Wan 2.1) stage has
/// no `AvgDown3D` shortcut and may interleave attention after each resnet.
pub struct GpuResidualDownBlock {
    pub resnets: Vec<GpuResidualBlock>,
    pub attentions: Vec<GpuAttentionBlock>,
    pub downsampler: Option<GpuResample>,
    pub out_dim: usize,
    pub factor_t: usize,
    pub factor_s: usize,
    pub has_shortcut: bool,
}

impl GpuResidualDownBlock {
    pub fn from_cpu(b: &super::blocks::ResidualDownBlock) -> Self {
        Self {
            resnets: b.resnets.iter().map(GpuResidualBlock::from_cpu).collect(),
            attentions: b.attentions.iter().map(GpuAttentionBlock::from_cpu).collect(),
            downsampler: b.downsampler.as_ref().map(GpuResample::from_cpu),
            out_dim: b.out_dim,
            factor_t: b.factor_t,
            factor_s: b.factor_s,
            has_shortcut: b.kind == super::blocks::StageKind::Residual,
        }
    }

    /// `in_c` is the block's input channel count, needed by the shortcut.
    pub fn forward(
        &self,
        x: &GpuTensor,
        in_c: usize,
        t: usize,
        h: usize,
        w: usize,
        cache: &mut GpuFeatCache,
    ) -> (GpuTensor, usize, usize, usize) {
        let (mut y, mut yt, mut yh, mut yw) = (x.clone_data(), t, h, w);
        for (i, r) in self.resnets.iter().enumerate() {
            let (o, ot, oh, ow) = r.forward_slot(&y, yt, yh, yw, cache);
            y = o;
            yt = ot;
            yh = oh;
            yw = ow;
            if let Some(a) = self.attentions.get(i) {
                y = a.forward(&y, yt, yh, yw);
            }
        }
        if let Some(d) = &self.downsampler {
            let (o, ot, oh, ow) = d.forward(&y, yt, yh, yw, cache);
            y = o;
            yt = ot;
            yh = oh;
            yw = ow;
        }
        // The averaged shortcut the CPU block adds on top. Omitting it is not
        // catastrophic — it cost only cos_sim 0.997 at the first block — which
        // is exactly why it is worth a line here rather than a debugging
        // session later.
        if self.has_shortcut {
            let (sc, ..) = crate::metal_backend::avg_down_3d_f16_gpu(
                x, in_c, t, h, w, self.out_dim, self.factor_t, self.factor_s,
            );
            eadd_f16_gpu(&mut y, &sc);
        }
        (y, yt, yh, yw)
    }
}

/// The VAE encoder on the GPU, for **single-chunk** input.
///
/// This exists to make start/end frame conditioning usable. On the CPU path
/// encoding one image costs 102 s at 320x192 and 409 s at 640x352, scaling
/// linearly in pixels — roughly 25 minutes at the native 1280x720, in front of
/// a generation that itself takes about an hour.
///
/// Single chunk only, and that is not a shortcut being papered over: the causal
/// VAE's first chunk is exactly one pixel frame, which is what a conditioning
/// image is. Multi-chunk encoding additionally needs Downsample3d's temporal
/// conv, whose stride-2 time axis the GPU conv does not implement; that path
/// stays on the CPU and `WanVae::encode` is still the reference.
pub struct GpuEncoder3d {
    pub conv_in: GpuCausalConv3d,
    pub down_blocks: Vec<GpuResidualDownBlock>,
    pub mid_block: GpuMidBlock,
    pub norm_out: GpuRmsNormChannel,
    pub conv_out: GpuCausalConv3d,
    /// The 1x1x1 conv that turns encoder features into the posterior's mean
    /// and log-variance. `WanVae::encode` applies it after the encoder, so
    /// producing "the encoder output" without it is not what any caller wants.
    pub quant_conv: GpuCausalConv3d,
    pub convs: usize,
}

impl GpuEncoder3d {
    pub fn from_cpu(e: &super::model::Encoder3d, quant_conv: &super::CausalConv3d) -> Self {
        Self {
            conv_in: GpuCausalConv3d::from_cpu(&e.conv_in),
            down_blocks: e.down_blocks.iter().map(GpuResidualDownBlock::from_cpu).collect(),
            mid_block: GpuMidBlock::from_cpu(&e.mid_block),
            norm_out: GpuRmsNormChannel::from_cpu(&e.norm_out),
            conv_out: GpuCausalConv3d::from_cpu(&e.conv_out),
            quant_conv: GpuCausalConv3d::from_cpu(quant_conv),
            convs: e.convs(),
        }
    }

    /// Encode `[3, T, H, W]` pixels in `[-1, 1]` to `[2 * z_dim, T', H', W']`
    /// moments — the same thing `WanVae::encode` returns, `quant_conv` included.
    pub fn forward(
        &self,
        x: &GpuTensor,
        t: usize,
        h: usize,
        w: usize,
    ) -> (GpuTensor, usize, usize, usize) {
        let mut cache = GpuFeatCache::new(self.convs);
        self.forward_chunk(x, t, h, w, &mut cache)
    }

    /// One chunk, against a caller-owned cache.
    ///
    /// The causal VAE encodes 1 frame then 4 at a time, carrying state between
    /// chunks. `forward` is the single-chunk case; a multi-frame clip rewinds
    /// this cache per chunk, exactly as `WanVae::encode` does on the CPU.
    pub fn forward_chunk(
        &self,
        x: &GpuTensor,
        t: usize,
        h: usize,
        w: usize,
        cache: &mut GpuFeatCache,
    ) -> (GpuTensor, usize, usize, usize) {
        let (mut y, mut yt, mut yh, mut yw) = self.conv_in.forward_slot(x, t, h, w, cache);
        let mut in_c = self.conv_in.out_c;
        for d in &self.down_blocks {
            let (o, ot, oh, ow) = d.forward(&y, in_c, yt, yh, yw, cache);
            y = o;
            yt = ot;
            yh = oh;
            yw = ow;
            in_c = d.out_dim;
        }

        // The mid block walks its resnets with the cache, as the CPU does.
        let mut m = self.mid_block.resnets[0].forward_slot(&y, yt, yh, yw, cache).0;
        for (attn, resnet) in self.mid_block.attentions.iter().zip(&self.mid_block.resnets[1..]) {
            m = attn.forward(&m, yt, yh, yw);
            m = resnet.forward_slot(&m, yt, yh, yw, cache).0;
        }

        self.norm_out.forward(&mut m, yt, yh, yw);
        silu_f16_gpu(&mut m);
        let (feat, ft, fh, fw) = self.conv_out.forward_slot(&m, yt, yh, yw, cache);
        // `quant_conv` has no temporal padding, so it takes no cache slot.
        self.quant_conv.forward(&feat, ft, fh, fw)
    }
}

/// GPU decoder: `conv_in -> mid -> up blocks -> norm -> silu -> conv_out`.
pub struct GpuDecoder3d {
    pub conv_in: GpuCausalConv3d,
    pub mid_block: GpuMidBlock,
    pub up_blocks: Vec<GpuResidualUpBlock>,
    pub norm_out: GpuRmsNormChannel,
    pub conv_out: GpuCausalConv3d,
    pub convs: usize,
}

impl GpuDecoder3d {
    pub fn from_cpu(d: &super::model::Decoder3d) -> Self {
        // Each up block needs its input width for the DupUp3D shortcut. Read it
        // off the stage's first resnet rather than assuming the previous
        // stage's `out_dim`: that holds for Wan 2.2, but a plain (Wan 2.1)
        // upsampler halves the width on the way out.
        let up_blocks = d
            .up_blocks
            .iter()
            .map(|b| GpuResidualUpBlock::from_cpu(b, b.resnets[0].conv1.in_c))
            .collect();
        Self {
            conv_in: GpuCausalConv3d::from_cpu(&d.conv_in),
            mid_block: GpuMidBlock::from_cpu(&d.mid_block),
            up_blocks,
            norm_out: GpuRmsNormChannel::from_cpu(&d.norm_out),
            conv_out: GpuCausalConv3d::from_cpu(&d.conv_out),
            convs: d.convs(),
        }
    }

    /// Decode one latent chunk. The cache must be rewound by the caller before
    /// each chunk; `first_chunk` is true only for the first call after a reset.
    pub fn forward(
        &self,
        z: &GpuTensor,
        t: usize,
        h: usize,
        w: usize,
        cache: &mut GpuFeatCache,
        first_chunk: bool,
    ) -> (GpuTensor, usize, usize, usize) {
        // `KLEARU_VAE_SYNC=1` flushes between stages. Diagnostic: a per-stage
        // bisection (which downloads, and so flushes, after every stage) scores
        // 0.9999995 against the CPU decoder, while the same network run through
        // `decode_all` without those flushes scores 0.99990. If synchronising
        // recovers the accuracy, the residue is an ordering bug rather than
        // arithmetic.
        let sync = std::env::var_os("KLEARU_VAE_SYNC").is_some();
        let sync_pt = || if sync { crate::metal_backend::flush() };
        // `KLEARU_VAE_PROBE=1`: after each stage, its non-finite count, peak
        // magnitude and first bad row — to find where a decode goes bad.
        let probe = std::env::var_os("KLEARU_VAE_PROBE").is_some();
        let probe_pt = |what: &str, g: &GpuTensor, xh: usize, xw: usize| {
            if !probe {
                return;
            }
            let v = g.download_to_f32();
            let plane = xh * xw;
            let bad: Vec<usize> = (0..v.len()).filter(|&i| !v[i].is_finite()).collect();
            let peak = v.iter().filter(|a| a.is_finite()).fold(0.0f32, |m, a| m.max(a.abs()));
            let row = bad.iter().map(|&i| (i % plane) / xw).min();
            println!("[vae probe] {what:<14} {xh}x{xw}: {} non-finite, peak {peak:.1}, first bad row {row:?}", bad.len());
        };

        let (mut x, mut xt, mut xh, mut xw) = self.conv_in.forward_slot(z, t, h, w, cache);
        sync_pt();
        probe_pt("conv_in", &x, xh, xw);

        // The mid block runs uncached on the CPU side too: its resnets are
        // walked with `Some(cache)` there, so mirror that here.
        let mut y = self.mid_block.resnets[0].forward_slot(&x, xt, xh, xw, cache).0;
        sync_pt();
        probe_pt("mid.resnet0", &y, xh, xw);
        for (attn, resnet) in self.mid_block.attentions.iter().zip(&self.mid_block.resnets[1..]) {
            y = attn.forward(&y, xt, xh, xw);
            sync_pt();
            probe_pt("mid.attention", &y, xh, xw);
            y = resnet.forward_slot(&y, xt, xh, xw, cache).0;
            sync_pt();
            probe_pt("mid.resnet", &y, xh, xw);
        }
        x = y;

        for (i, b) in self.up_blocks.iter().enumerate() {
            let (o, ot, oh, ow) = b.forward(&x, xt, xh, xw, cache, first_chunk);
            x = o;
            xt = ot;
            xh = oh;
            xw = ow;
            sync_pt();
            probe_pt(&format!("up_block{i}"), &x, xh, xw);
        }

        self.norm_out.forward(&mut x, xt, xh, xw);
        silu_f16_gpu(&mut x);
        self.conv_out.forward_slot(&x, xt, xh, xw, cache)
    }

    /// Full chunked decode of a `[z_dim, T, H, W]` latent, one frame at a
    /// time, matching `WanVae::decode` minus the patchify/clamp tail.
    pub fn decode_all(&self, z: &Vol) -> Vol {
        // VAE activations run large — the 2D VAE decoder flips this for the
        // same reason. Without it the sgemm scratch is IEEE f16, whose output
        // cast clamps at 65504 and loses precision well before that.
        crate::metal_backend::set_use_f32_sgemm(true);
        // f32 convolutions are available (`KLEARU_VAE_F32_CONV=1`) but OFF by
        // default: they were tried against the GPU-vs-diffusers gap and changed
        // nothing (cos_sim 0.99845 f32 vs 0.99849 f16), for 2x the conv memory
        // and time. So f16 accumulation over a 3x3x1024 sum is NOT that gap,
        // however plausible it looked.
        crate::metal_backend::set_use_f32_conv(
            std::env::var_os("KLEARU_VAE_F32_CONV").is_some(),
        );
        let mut cache = GpuFeatCache::new(self.convs);
        let mut out: Option<Vol> = None;
        let report = std::env::var_os("KLEARU_VAE_MEM").is_some();
        for ti in 0..z.t {
            if report {
                crate::metal_backend::gpu_memory_report(&format!("vae frame {ti}"));
            }
            // Drop the pooled scratch from the previous frame before building
            // this one's. The pool is keyed by exact byte size and never
            // shrinks on its own; at 704x1280 each frame's conv/sgemm scratch
            // runs to gigabytes, and the sizes differ enough frame to frame
            // that nothing gets reused. Measured without this: GPU allocation
            // climbed 0 -> 7.4 -> 22.9 -> 38.1 GB over four frames and the
            // process was killed. 49 frames would need 700+ GB.
            crate::metal_backend::drain_pool();
            // Release conv graphs not used by the previous frame. Without this
            // the cache keeps a full set of MPSGraphs for each of the three
            // batch-size regimes the causal cache produces (frame 0, frame 1,
            // steady state) — 294 graphs holding full-resolution intermediates.
            crate::metal_backend::conv_graph_end_generation();
            cache.rewind();
            let frame = frame_of(z, ti);
            let (g, gt, gh, gw) =
                self.forward(&upload_vol(&frame), 1, z.h, z.w, &mut cache, ti == 0);
            let dec = download_vol(&g, g.elements() / (gt * gh * gw), gt, gh, gw);
            out = Some(match out {
                None => dec,
                Some(prev) => Vol::concat_time(&prev, &dec),
            });
        }
        crate::metal_backend::set_use_f32_sgemm(false);
        crate::metal_backend::set_use_f32_conv(false);
        out.expect("decode_all: empty latent")
    }
}

// ---------------------------------------------------------------------------
// Single-image driver
// ---------------------------------------------------------------------------

/// Sets the VAE's GPU precision modes for one decode/encode and puts them back
/// afterwards, the way [`GpuDecoder3d::decode_all`] does around a clip.
struct VaeGpuScope;

impl VaeGpuScope {
    fn enter() -> Self {
        // The mid-block attention's matmuls see large activations; with an
        // f16 sgemm scratch their output cast clamps at 65504. See
        // `decode_all` for the measurements behind both switches.
        crate::metal_backend::set_use_f32_sgemm(true);
        crate::metal_backend::set_use_f32_conv(std::env::var_os("KLEARU_VAE_F32_CONV").is_some());
        // Start from an empty buffer pool and retire conv graphs that the
        // previous call did not use, so a run of images at changing sizes
        // keeps at most two sizes' graphs alive rather than one per size.
        crate::metal_backend::drain_pool();
        crate::metal_backend::conv_graph_end_generation();
        VaeGpuScope
    }
}

impl Drop for VaeGpuScope {
    fn drop(&mut self) {
        crate::metal_backend::set_use_f32_sgemm(false);
        crate::metal_backend::set_use_f32_conv(false);
        // Hand the scratch back before whatever runs next (the transformer,
        // for the next image) allocates its own.
        crate::metal_backend::drain_pool();
    }
}

/// The Wan VAE as a single-**image** autoencoder on the GPU.
///
/// Qwen-Image and Krea 2 use the Wan 2.1 VAE on pictures as one-frame videos.
/// One frame is exactly the causal VAE's first chunk, so this runs one chunk
/// with a [`GpuFeatCache::single_frame`] cache: `upsample3d` skips its temporal
/// conv, `downsample3d` passes through, and no causal conv keeps a copy of its
/// input for a next chunk. The result equals [`super::WanVae::decode`] /
/// [`super::WanVae::encode`] at `T = 1`.
///
/// Build it once and keep it across images. Construction slices every conv
/// weight into per-tap host copies and uploads happen on first use; and the
/// backend's bf16 weight cache is keyed by host address, so a rebuilt copy
/// is only reclaimed if a later allocation happens to reuse the address.
///
/// # Memory and time
///
/// Measured on the Qwen-Image VAE (decoder 73.3M params, encoder 53.6M) by
/// `qwen_image_vae_gpu_cost_by_resolution`:
///
/// * **Host:** the per-tap f32 weight copies the convs keep — 293 MB for the
///   decoder, 214 MB for the encoder. The CPU [`super::WanVae`] this was built
///   from is not referenced afterwards and can be dropped (another 507 MB).
///   [`Self::decoder_only`] skips the encoder for text-to-image.
/// * **GPU, resident:** weights upload on first use, and a single frame only
///   ever touches the current-frame tap of each causal conv, so after a
///   decode and an encode the device holds ~0.19 GB (half copies plus the
///   backend's bf16 cache).
/// * **GPU, per call:** decode peaks ~1.4 GB above resident at 512x768 and
///   ~3.3 GB at 1024x1024; encode ~2.4 GB at 1024x1024. It scales with pixel
///   count (~3.2 GB per megapixel — extrapolated, ~13 GB at 2048x2048), lives
///   inside the full-resolution convolutions (flushing between stages or
///   between resnets does not lower it), and is released when the call
///   returns. The mid-block attention is blocked to stay under
///   [`ATTN_MAX_SCORES`], so it is not the limit.
/// * **Time** at 1024x1024: decode ~1.0 s, encode ~0.7 s, once warm. The
///   first call at a new size adds graph builds and weight uploads (~0.6 s at
///   512x768); and each call retires conv graphs from sizes the previous call
///   did not use, so alternating between sizes rebuilds every time.
///
/// # Threading
///
/// The Metal backend's conv-graph, weight and buffer caches are thread-local,
/// so use this from the thread that runs the rest of the GPU pipeline.
pub struct GpuImageVae {
    decoder: Option<GpuDecoder3d>,
    encoder: Option<GpuEncoder3d>,
    /// 1x1x1 over `z_dim` channels — `z_dim^2` MACs per latent pixel — so it
    /// stays on the host, as in `GpuWanPipeline::decode`.
    post_quant_conv: CausalConv3d,
    z_dim: usize,
    patch_size: usize,
    spatial_compression: usize,
}

impl GpuImageVae {
    /// Decoder and encoder.
    pub fn from_vae(vae: &super::WanVae) -> Self {
        Self::build(vae, true, true)
    }

    /// Decoder only — the text-to-image case. [`Self::encode`] panics.
    pub fn decoder_only(vae: &super::WanVae) -> Self {
        Self::build(vae, true, false)
    }

    /// Encoder only. [`Self::decode`] panics.
    pub fn encoder_only(vae: &super::WanVae) -> Self {
        Self::build(vae, false, true)
    }

    fn build(vae: &super::WanVae, decoder: bool, encoder: bool) -> Self {
        Self {
            decoder: decoder.then(|| GpuDecoder3d::from_cpu(&vae.decoder)),
            encoder: encoder.then(|| GpuEncoder3d::from_cpu(&vae.encoder, &vae.quant_conv)),
            post_quant_conv: vae.post_quant_conv.clone(),
            z_dim: vae.cfg.z_dim,
            patch_size: vae.cfg.patch_size,
            spatial_compression: vae.cfg.spatial_compression(),
        }
    }

    pub fn has_decoder(&self) -> bool {
        self.decoder.is_some()
    }

    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    /// Latent channels.
    pub fn z_dim(&self) -> usize {
        self.z_dim
    }

    /// Pixels per latent cell along each axis (8 for Wan 2.1 / Qwen-Image).
    pub fn spatial_compression(&self) -> usize {
        self.spatial_compression
    }

    /// Decode one latent `[z_dim, h, w]` to RGB `[3, h*s, w*s]` in `[-1, 1]`,
    /// `s` = [`Self::spatial_compression`].
    ///
    /// `z` is the VAE's own latent space: callers that keep latents
    /// normalised (`(z - latents_mean) / latents_std`, as the Qwen-Image and
    /// Krea pipelines do) must undo that first. The VAE does not.
    pub fn decode(&self, z: &[f32], h: usize, w: usize) -> Vec<f32> {
        let dec = self.decoder.as_ref().expect("GpuImageVae was built without a decoder");
        assert_eq!(
            z.len(),
            self.z_dim * h * w,
            "decode: expected a [{}, {h}, {w}] latent",
            self.z_dim
        );
        let zq = self.post_quant_conv.forward(&Vol::new(z.to_vec(), self.z_dim, 1, h, w), None);

        let _scope = VaeGpuScope::enter();
        let mut cache = GpuFeatCache::single_frame(dec.convs);
        let (g, gt, gh, gw) = dec.forward(&upload_vol(&zq), 1, h, w, &mut cache, true);
        let raw = download_vol(&g, g.elements() / (gt * gh * gw), gt, gh, gw);
        drop(g);

        let mut out = super::unpatchify(&raw, self.patch_size).data;
        for v in out.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        out
    }

    /// Encode RGB `[3, h, w]` in `[-1, 1]` to the posterior **mean**
    /// `[z_dim, h/s, w/s]` — the first `z_dim` channels after `quant_conv`,
    /// un-normalised (diffusers' `latent_dist.mode()`). `h` and `w` must be
    /// multiples of [`Self::spatial_compression`].
    pub fn encode(&self, rgb: &[f32], h: usize, w: usize) -> Vec<f32> {
        let enc = self.encoder.as_ref().expect("GpuImageVae was built without an encoder");
        assert_eq!(rgb.len(), 3 * h * w, "encode: expected [3, {h}, {w}] pixels");
        let s = self.spatial_compression;
        assert!(
            h.is_multiple_of(s) && w.is_multiple_of(s),
            "encode: {w}x{h} is not a multiple of the VAE's {s}px cell"
        );
        let x = super::patchify(&Vol::new(rgb.to_vec(), 3, 1, h, w), self.patch_size);

        let _scope = VaeGpuScope::enter();
        let mut cache = GpuFeatCache::single_frame(enc.convs);
        let (g, _, oh, ow) = enc.forward_chunk(&upload_vol(&x), 1, x.h, x.w, &mut cache);
        let mut moments = g.download_to_f32();
        // Moments are [2 * z_dim, 1, h', w'], mean first: diffusers'
        // DiagonalGaussianDistribution chunks them along channels.
        moments.truncate(self.z_dim * oh * ow);
        moments
    }
}

/// One frame of a volume as its own volume.
fn frame_of(x: &Vol, t: usize) -> Vol {
    let mut out = Vol::zeros(x.c, 1, x.h, x.w);
    for c in 0..x.c {
        for h in 0..x.h {
            for w in 0..x.w {
                let v = x.at(c, t, h, w);
                let i = out.idx(c, 0, h, w);
                out.data[i] = v;
            }
        }
    }
    out
}
