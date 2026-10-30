//! 3D causal video VAE (`AutoencoderKLWan`) — primitives.
//!
//! The 2D [`crate::vae`] cannot serve video: it has no temporal axis, and the
//! temporal axis here is *causal*, so a frame may only ever see frames at or
//! before it. That is what allows a long clip to be decoded in chunks — each
//! chunk carries a small cache of trailing activations forward instead of
//! re-running the whole timeline.
//!
//! Shapes are `[C, T, H, W]` throughout ([`Vol`]); batch is always 1, since a
//! video VAE decodes one clip at a time.
//!
//! Wan 2.2 (`is_residual: true`, `z_dim: 48`) uses the residual down/up blocks
//! built from these primitives, not the classic resample path. Compression is
//! 16x spatial / 4x temporal: a `patch_size: 2` pixel patchify (3 -> 12
//! channels) followed by three spatial stages, with `temperal_downsample:
//! [false, true, true]` giving the 4x in time.
//!
//! Wan 2.1 (`z_dim: 16`, no patchify, 8x spatial) uses the classic path, and
//! so does the Qwen-Image VAE that Krea 2 ships — a fine-tune with the same
//! module tree. The image models run it on one-frame "videos";
//! [`gpu::GpuImageVae`] is the driver for that.

pub mod blocks;
pub mod config;
#[cfg(feature = "metal")]
pub mod gpu;
pub mod model;
pub mod weight;

pub use blocks::{cached_conv, FeatCache, ResampleMode, Slot, StageKind};
pub use config::WanVaeConfig;
#[cfg(feature = "metal")]
pub use gpu::GpuImageVae;
pub use model::{Decoder3d, Encoder3d, WanVae};

/// Trailing frames each causal conv carries into the next chunk. Matches
/// `CACHE_T` in the reference.
pub const CACHE_T: usize = 2;

/// A `[C, T, H, W]` volume. Batch is implicitly 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Vol {
    pub data: Vec<f32>,
    pub c: usize,
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl Vol {
    pub fn zeros(c: usize, t: usize, h: usize, w: usize) -> Self {
        Self { data: vec![0.0; c * t * h * w], c, t, h, w }
    }

    pub fn new(data: Vec<f32>, c: usize, t: usize, h: usize, w: usize) -> Self {
        assert_eq!(data.len(), c * t * h * w, "Vol data length does not match [{c},{t},{h},{w}]");
        Self { data, c, t, h, w }
    }

    /// Elements per channel.
    #[inline]
    pub fn plane(&self) -> usize {
        self.t * self.h * self.w
    }

    #[inline]
    pub fn idx(&self, c: usize, t: usize, h: usize, w: usize) -> usize {
        ((c * self.t + t) * self.h + h) * self.w + w
    }

    #[inline]
    pub fn at(&self, c: usize, t: usize, h: usize, w: usize) -> f32 {
        self.data[self.idx(c, t, h, w)]
    }

    /// The last `n` frames, or all of them when shorter. This is the slice a
    /// causal conv hands to the next chunk.
    pub fn tail_frames(&self, n: usize) -> Vol {
        let n = n.min(self.t);
        let start = self.t - n;
        let mut out = Vol::zeros(self.c, n, self.h, self.w);
        for c in 0..self.c {
            for t in 0..n {
                for h in 0..self.h {
                    for w in 0..self.w {
                        let v = self.at(c, start + t, h, w);
                        let i = out.idx(c, t, h, w);
                        out.data[i] = v;
                    }
                }
            }
        }
        out
    }

    /// Concatenate along the temporal axis.
    pub fn concat_time(a: &Vol, b: &Vol) -> Vol {
        assert_eq!((a.c, a.h, a.w), (b.c, b.h, b.w), "concat_time: non-temporal dims differ");
        let mut out = Vol::zeros(a.c, a.t + b.t, a.h, a.w);
        for c in 0..a.c {
            for t in 0..a.t + b.t {
                for h in 0..a.h {
                    for w in 0..a.w {
                        let v = if t < a.t { a.at(c, t, h, w) } else { b.at(c, t - a.t, h, w) };
                        let i = out.idx(c, t, h, w);
                        out.data[i] = v;
                    }
                }
            }
        }
        out
    }
}

/// SiLU, in place.
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

// ---------------------------------------------------------------------------
// Causal 3D convolution
// ---------------------------------------------------------------------------

/// 3D convolution that is causal in time.
///
/// Spatial padding is symmetric, but the temporal axis is padded **only on the
/// past side, by `2 * pad_t`** — the reference builds
/// `(pad_w, pad_w, pad_h, pad_h, 2 * pad_t, 0)` and zeroes `nn.Conv3d`'s own
/// padding. With the usual `kernel 3, pad 1` that means two zero frames in
/// front and none behind, so output frame `t` sees input frames `t-2 ..= t`
/// and the temporal length is preserved.
///
/// For chunked decoding, `forward` takes the previous chunk's trailing frames
/// as `cache`: they are prepended and the left padding is reduced by the same
/// amount, so a chunked run reproduces the whole-clip result exactly.
#[derive(Clone, Debug)]
pub struct CausalConv3d {
    /// `[out_c, in_c, kt, kh, kw]`, row-major.
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub in_c: usize,
    pub out_c: usize,
    /// `(kt, kh, kw)`
    pub kernel: (usize, usize, usize),
    /// `(st, sh, sw)`
    pub stride: (usize, usize, usize),
    /// `(pt, ph, pw)` as passed to the constructor. Temporal is applied as
    /// `2 * pt` on the past side only.
    pub pad: (usize, usize, usize),
    /// Extra zero padding on the far (bottom / right) spatial edges, on top of
    /// the symmetric `pad`. Wan's `downsample2d` prefixes its stride-2 conv
    /// with `nn.ZeroPad2d((0, 1, 0, 1))`, which is not expressible as
    /// symmetric padding.
    pub pad_end_hw: (usize, usize),
}

impl CausalConv3d {
    pub fn new(
        in_c: usize,
        out_c: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        pad: (usize, usize, usize),
        bias: bool,
    ) -> Self {
        let (kt, kh, kw) = kernel;
        Self {
            weight: vec![0.0; out_c * in_c * kt * kh * kw],
            bias: if bias { Some(vec![0.0; out_c]) } else { None },
            in_c,
            out_c,
            kernel,
            stride,
            pad,
            pad_end_hw: (0, 0),
        }
    }

    /// Per-frame 2D convolution: a 3D conv with a temporal kernel of 1 and no
    /// temporal padding is exactly `nn.Conv2d` applied frame by frame, and the
    /// `[out, in, 1, kh, kw]` weight layout matches `nn.Conv2d`'s
    /// `[out, in, kh, kw]` element for element.
    pub fn conv2d(
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: usize,
        pad: usize,
        bias: bool,
    ) -> Self {
        Self::new(in_c, out_c, (1, k, k), (1, stride, stride), (0, pad, pad), bias)
    }

    /// `conv2d` with `nn.ZeroPad2d((0, 1, 0, 1))` folded in — the bottom/right
    /// edges gain one zero row/column.
    pub fn conv2d_pad_end(in_c: usize, out_c: usize, k: usize, stride: usize) -> Self {
        let mut c = Self::conv2d(in_c, out_c, k, stride, 0, true);
        c.pad_end_hw = (1, 1);
        c
    }

    /// Number of trailing frames this conv must hand to the next chunk.
    pub fn cache_frames(&self) -> usize {
        if self.pad.0 == 0 { 0 } else { CACHE_T }
    }

    /// `x`: `[in_c, T, H, W]`. `cache`: the previous chunk's trailing frames,
    /// prepended before padding.
    pub fn forward(&self, x: &Vol, cache: Option<&Vol>) -> Vol {
        assert_eq!(x.c, self.in_c, "CausalConv3d: expected {} input channels, got {}", self.in_c, x.c);
        let (kt, kh, kw) = self.kernel;
        let (st, sh, sw) = self.stride;
        let (pt, ph, pw) = self.pad;

        // Temporal padding is 2*pt on the past side; cached frames stand in
        // for that many of the pad frames.
        let cache_t = match cache {
            Some(c) if 2 * pt > 0 => {
                assert_eq!((c.c, c.h, c.w), (x.c, x.h, x.w), "cache dims must match input");
                c.t.min(2 * pt)
            }
            _ => 0,
        };
        let left_pad_t = 2 * pt - cache_t;
        let eff_t = cache_t + x.t;

        let out_t = (left_pad_t + eff_t).saturating_sub(kt) / st + 1;
        let out_h = (x.h + 2 * ph + self.pad_end_hw.0).saturating_sub(kh) / sh + 1;
        let out_w = (x.w + 2 * pw + self.pad_end_hw.1).saturating_sub(kw) / sw + 1;
        let mut out = Vol::zeros(self.out_c, out_t, out_h, out_w);

        // Read the virtual padded input: (cache ++ x) with `left_pad_t` zero
        // frames in front and symmetric zero borders in H/W.
        let get = |ci: usize, tp: isize, hp: isize, wp: isize| -> f32 {
            let te = tp - left_pad_t as isize;
            let he = hp - ph as isize;
            let we = wp - pw as isize;
            if te < 0 || he < 0 || we < 0 {
                return 0.0;
            }
            let (te, he, we) = (te as usize, he as usize, we as usize);
            if te >= eff_t || he >= x.h || we >= x.w {
                return 0.0;
            }
            if te < cache_t {
                let c = cache.unwrap();
                // Use the *last* cache_t frames when more were supplied.
                c.at(ci, c.t - cache_t + te, he, we)
            } else {
                x.at(ci, te - cache_t, he, we)
            }
        };

        let k_stride_in = kt * kh * kw;
        for oc in 0..self.out_c {
            let wbase_oc = oc * self.in_c * k_stride_in;
            for ot in 0..out_t {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = self.bias.as_ref().map_or(0.0, |b| b[oc]);
                        for ic in 0..self.in_c {
                            let wbase = wbase_oc + ic * k_stride_in;
                            for a in 0..kt {
                                for b in 0..kh {
                                    for d in 0..kw {
                                        let v = get(
                                            ic,
                                            (ot * st + a) as isize,
                                            (oh * sh + b) as isize,
                                            (ow * sw + d) as isize,
                                        );
                                        if v != 0.0 {
                                            acc += v * self.weight[wbase + (a * kh + b) * kw + d];
                                        }
                                    }
                                }
                            }
                        }
                        let i = out.idx(oc, ot, oh, ow);
                        out.data[i] = acc;
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

/// Wan's channel RMS norm.
///
/// The reference writes it as `F.normalize(x, dim=1) * sqrt(dim) * gamma`,
/// i.e. L2-normalise across channels then rescale — algebraically the usual
/// `x / rms(x)`, but the L2 form matters because `F.normalize` clamps the norm
/// at `eps = 1e-12` rather than adding eps under the square root.
#[derive(Clone, Debug)]
pub struct RmsNormChannel {
    /// Per-channel scale, length `dim`.
    pub gamma: Vec<f32>,
    /// Per-channel bias; the reference omits it unless `bias=True`.
    pub bias: Option<Vec<f32>>,
}

/// `F.normalize`'s default epsilon.
const NORMALIZE_EPS: f32 = 1e-12;

impl RmsNormChannel {
    pub fn new(dim: usize, bias: bool) -> Self {
        Self {
            gamma: vec![1.0; dim],
            bias: if bias { Some(vec![0.0; dim]) } else { None },
        }
    }

    pub fn forward(&self, x: &Vol) -> Vol {
        assert_eq!(x.c, self.gamma.len(), "RmsNormChannel: dim mismatch");
        let scale = (x.c as f32).sqrt();
        let mut out = x.clone();
        let plane = x.plane();
        for p in 0..plane {
            let mut sum = 0.0_f32;
            for c in 0..x.c {
                let v = x.data[c * plane + p];
                sum += v * v;
            }
            let inv = 1.0 / sum.sqrt().max(NORMALIZE_EPS);
            for c in 0..x.c {
                let v = x.data[c * plane + p] * inv * scale * self.gamma[c]
                    + self.bias.as_ref().map_or(0.0, |b| b[c]);
                out.data[c * plane + p] = v;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Shortcut resampling (the `is_residual` path)
// ---------------------------------------------------------------------------

/// `AvgDown3D`: strided average pool that folds the pooled positions into
/// channels, then averages channel groups down to `out_c`.
///
/// The temporal axis is padded on the **past** side to a multiple of
/// `factor_t`, keeping the layer causal.
pub fn avg_down_3d(x: &Vol, out_c: usize, factor_t: usize, factor_s: usize) -> Vol {
    let factor = factor_t * factor_s * factor_s;
    assert_eq!(
        (x.c * factor) % out_c,
        0,
        "avg_down_3d: {}*{factor} not divisible by {out_c}",
        x.c
    );
    let group_size = x.c * factor / out_c;
    assert_eq!(x.h % factor_s, 0, "avg_down_3d: H not divisible by factor_s");
    assert_eq!(x.w % factor_s, 0, "avg_down_3d: W not divisible by factor_s");

    let pad_t = (factor_t - x.t % factor_t) % factor_t;
    let t_p = x.t + pad_t;
    let (ot, oh, ow) = (t_p / factor_t, x.h / factor_s, x.w / factor_s);
    let mut out = Vol::zeros(out_c, ot, oh, ow);

    for oc in 0..out_c {
        for t in 0..ot {
            for h in 0..oh {
                for w in 0..ow {
                    let mut acc = 0.0_f32;
                    for g in 0..group_size {
                        // Channel after the fold: c * factor + (it*fs + ih)*fs + iw.
                        let flat = oc * group_size + g;
                        let c = flat / factor;
                        let rem = flat % factor;
                        let it = rem / (factor_s * factor_s);
                        let ih = (rem / factor_s) % factor_s;
                        let iw = rem % factor_s;
                        // Padding sits in front, so shift the source index back.
                        let ts = (t * factor_t + it) as isize - pad_t as isize;
                        if ts < 0 {
                            continue; // padded frame contributes 0
                        }
                        acc += x.at(c, ts as usize, h * factor_s + ih, w * factor_s + iw);
                    }
                    let i = out.idx(oc, t, h, w);
                    out.data[i] = acc / group_size as f32;
                }
            }
        }
    }
    out
}

/// `DupUp3D`: duplicate channels, then unfold them into time and space.
///
/// `first_chunk` drops the leading `factor_t - 1` frames, which is how the
/// decoder keeps the first chunk's frame count aligned (a 4x-compressed
/// latent's first frame expands to one output frame, not `factor_t`).
pub fn dup_up_3d(
    x: &Vol,
    out_c: usize,
    factor_t: usize,
    factor_s: usize,
    first_chunk: bool,
) -> Vol {
    let factor = factor_t * factor_s * factor_s;
    assert_eq!(
        (out_c * factor) % x.c,
        0,
        "dup_up_3d: {out_c}*{factor} not divisible by {}",
        x.c
    );
    let repeats = out_c * factor / x.c;

    let (ft, fs) = (factor_t, factor_s);
    let full_t = x.t * ft;
    let drop = if first_chunk { ft - 1 } else { 0 };
    let mut out = Vol::zeros(out_c, full_t - drop, x.h * fs, x.w * fs);

    for oc in 0..out_c {
        for t in 0..x.t {
            for it in 0..ft {
                let ot = t * ft + it;
                if ot < drop {
                    continue;
                }
                for h in 0..x.h {
                    for ih in 0..fs {
                        for w in 0..x.w {
                            for iw in 0..fs {
                                // Channel of the repeat-interleaved tensor.
                                let flat = ((oc * ft + it) * fs + ih) * fs + iw;
                                let src_c = flat / repeats;
                                let v = x.at(src_c, t, h, w);
                                let i = out.idx(oc, ot - drop, h * fs + ih, w * fs + iw);
                                out.data[i] = v;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Pixel patchify
// ---------------------------------------------------------------------------

/// Fold `p x p` pixel patches into channels: `[C, T, H, W]` -> `[C*p*p, T, H/p, W/p]`.
///
/// The channel ordering is **width-major**: `c * p^2 + iw * p + ih`, from the
/// reference's `permute(0, 1, 6, 4, 2, 3, 5)`, which puts the width-within-patch
/// index outside the height-within-patch index. Easy to get backwards; the
/// round-trip test pins it.
pub fn patchify(x: &Vol, p: usize) -> Vol {
    if p == 1 {
        return x.clone();
    }
    assert_eq!(x.h % p, 0, "patchify: H={} not divisible by {p}", x.h);
    assert_eq!(x.w % p, 0, "patchify: W={} not divisible by {p}", x.w);
    let (oh, ow) = (x.h / p, x.w / p);
    let mut out = Vol::zeros(x.c * p * p, x.t, oh, ow);
    for c in 0..x.c {
        for ih in 0..p {
            for iw in 0..p {
                let oc = c * p * p + iw * p + ih;
                for t in 0..x.t {
                    for h in 0..oh {
                        for w in 0..ow {
                            let v = x.at(c, t, h * p + ih, w * p + iw);
                            let i = out.idx(oc, t, h, w);
                            out.data[i] = v;
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`patchify`].
pub fn unpatchify(x: &Vol, p: usize) -> Vol {
    if p == 1 {
        return x.clone();
    }
    assert_eq!(x.c % (p * p), 0, "unpatchify: C={} not divisible by {}", x.c, p * p);
    let c_out = x.c / (p * p);
    let mut out = Vol::zeros(c_out, x.t, x.h * p, x.w * p);
    for c in 0..c_out {
        for ih in 0..p {
            for iw in 0..p {
                let ic = c * p * p + iw * p + ih;
                for t in 0..x.t {
                    for h in 0..x.h {
                        for w in 0..x.w {
                            let v = x.at(ic, t, h, w);
                            let i = out.idx(c, t, h * p + ih, w * p + iw);
                            out.data[i] = v;
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
