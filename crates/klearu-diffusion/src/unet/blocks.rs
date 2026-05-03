//! UNet down/mid/up blocks. Each composes ResnetBlock2D + Transformer2DModel
//! + Up/Downsample as configured.
//!
//! Block taxonomy (HF Diffusers names):
//!   - DownBlock2D: N ResnetBlock2D + Downsample (no attention)
//!   - CrossAttnDownBlock2D: N (ResnetBlock2D + Transformer2DModel) + Downsample
//!   - UNetMidBlock2DCrossAttn: ResnetBlock2D + Transformer2DModel + ResnetBlock2D
//!   - UpBlock2D: N ResnetBlock2D + Upsample (no attention)
//!   - CrossAttnUpBlock2D: N (ResnetBlock2D + Transformer2DModel) + Upsample
//!
//! The down path emits skip-connections (one per resnet+attn pair); the up
//! path consumes them via channel-wise concatenation before each resnet.

use crate::layers::{Downsample, Upsample};
use crate::unet::{ResnetBlock2D, Transformer2DModel};

/// Concatenate two NCHW tensors along the channel axis.
/// `a` is [N, Ca, H, W], `b` is [N, Cb, H, W]; result is [N, Ca+Cb, H, W].
pub fn cat_channels(a: &[f32], ca: usize, b: &[f32], cb: usize, n: usize, h: usize, w: usize) -> Vec<f32> {
    let total_c = ca + cb;
    let hw = h * w;
    let mut out = vec![0.0f32; n * total_c * hw];
    for ni in 0..n {
        let dst_n = ni * total_c * hw;
        let src_a_n = ni * ca * hw;
        let src_b_n = ni * cb * hw;
        out[dst_n..dst_n + ca * hw].copy_from_slice(&a[src_a_n..src_a_n + ca * hw]);
        out[dst_n + ca * hw..dst_n + total_c * hw].copy_from_slice(&b[src_b_n..src_b_n + cb * hw]);
    }
    out
}

/// CrossAttnDownBlock2D (with cross-attention via Transformer2DModel) or
/// DownBlock2D (no attention).
pub struct DownBlock2D {
    pub resnets: Vec<ResnetBlock2D>,
    pub attentions: Option<Vec<Transformer2DModel>>, // None for plain DownBlock2D
    pub downsample: Option<Downsample>,
    pub channels_out: usize,
}

impl DownBlock2D {
    /// Forward; returns (output, skip_states, skip_channels, h_out, w_out).
    /// `skip_channels[i]` is the channel count for `skip_states[i]`. We
    /// return these explicitly so the caller doesn't have to back-compute
    /// from spatial dims (which differ between resnet skips and the
    /// downsample skip). All this block's skips share `channels_out`.
    pub fn forward(
        &self,
        input: &[f32],
        n: usize,
        h: usize,
        w: usize,
        time_emb: &[f32],
        text_emb: &[f32],
        text_seq: usize,
    ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<usize>, usize, usize) {
        let mut x = input.to_vec();
        let mut h_cur = h;
        let mut w_cur = w;
        let mut skips = Vec::new();
        let mut skip_channels = Vec::new();
        for (i, resnet) in self.resnets.iter().enumerate() {
            x = resnet.forward(&x, n, h_cur, w_cur, time_emb);
            if let Some(attns) = &self.attentions {
                x = attns[i].forward(&x, n, h_cur, w_cur, text_emb, text_seq);
            }
            skips.push(x.clone());
            skip_channels.push(self.channels_out);
        }
        if let Some(down) = &self.downsample {
            let (next, hn, wn) = down.forward(&x, n, h_cur, w_cur);
            x = next; h_cur = hn; w_cur = wn;
            // The downsample output is also a skip (Diffusers convention).
            // Channel count is preserved by Downsample (same as channels_out).
            skips.push(x.clone());
            skip_channels.push(self.channels_out);
        }
        (x, skips, skip_channels, h_cur, w_cur)
    }

    /// GPU-resident forward. Same semantics as `forward` but with all data
    /// flowing through GpuTensors. Returns skip states as cloned GpuTensors.
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        input: &crate::metal_backend::GpuTensor,
        n: usize, h: usize, w: usize,
        time_emb: &[f32],
        text_emb: &[f32],
        text_seq: usize,
    ) -> (crate::metal_backend::GpuTensor,
          Vec<crate::metal_backend::GpuTensor>,
          Vec<usize>,
          usize, usize) {
        let mut x = input.clone_data();
        let mut h_cur = h;
        let mut w_cur = w;
        let mut skips = Vec::new();
        let mut skip_channels = Vec::new();
        let trace = std::env::var_os("KLEARU_UNET_TRACE").is_some();
        let trace_stat = |label: &str, t: &crate::metal_backend::GpuTensor| {
            if trace {
                let f32 = t.download_to_f32();
                let mut nan = 0; let mut inf = 0;
                let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
                let mut sum_abs = 0.0f64;
                for &v in &f32 {
                    if v.is_nan() { nan += 1; }
                    else if v.is_infinite() { inf += 1; }
                    else { mn = mn.min(v); mx = mx.max(v); sum_abs += v.abs() as f64; }
                }
                eprintln!("      [db] {label:<32} shape={:?}, min={mn:.3}, max={mx:.3}, mean_abs={:.4}, NaN={nan}, Inf={inf}",
                          t.shape, sum_abs / f32.len() as f64);
            }
        };
        for (i, resnet) in self.resnets.iter().enumerate() {
            x = resnet.forward_gpu(&x, n, h_cur, w_cur, time_emb);
            trace_stat(&format!("resnet[{i}]"), &x);
            if let Some(attns) = &self.attentions {
                x = attns[i].forward_gpu(&x, n, h_cur, w_cur, text_emb, text_seq);
                trace_stat(&format!("transformer[{i}]"), &x);
            }
            skips.push(x.clone_data());
            skip_channels.push(self.channels_out);
        }
        if let Some(down) = &self.downsample {
            let (next, hn, wn) = down.forward_gpu(&x, n, h_cur, w_cur);
            x = next; h_cur = hn; w_cur = wn;
            trace_stat("downsample", &x);
            skips.push(x.clone_data());
            skip_channels.push(self.channels_out);
        }
        (x, skips, skip_channels, h_cur, w_cur)
    }
}

/// UNetMidBlock2DCrossAttn: Resnet + Transformer + Resnet.
pub struct MidBlock {
    pub resnet_1: ResnetBlock2D,
    pub attn: Transformer2DModel,
    pub resnet_2: ResnetBlock2D,
}

impl MidBlock {
    pub fn forward(
        &self,
        input: &[f32],
        n: usize,
        h: usize,
        w: usize,
        time_emb: &[f32],
        text_emb: &[f32],
        text_seq: usize,
    ) -> Vec<f32> {
        let mut x = self.resnet_1.forward(input, n, h, w, time_emb);
        x = self.attn.forward(&x, n, h, w, text_emb, text_seq);
        x = self.resnet_2.forward(&x, n, h, w, time_emb);
        x
    }

    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        input: &crate::metal_backend::GpuTensor,
        n: usize, h: usize, w: usize,
        time_emb: &[f32],
        text_emb: &[f32],
        text_seq: usize,
    ) -> crate::metal_backend::GpuTensor {
        let mut x = self.resnet_1.forward_gpu(input, n, h, w, time_emb);
        x = self.attn.forward_gpu(&x, n, h, w, text_emb, text_seq);
        x = self.resnet_2.forward_gpu(&x, n, h, w, time_emb);
        x
    }
}

/// CrossAttnUpBlock2D or UpBlock2D. Each step concatenates with the
/// matching skip from the down path before its resnet.
pub struct UpBlock2D {
    pub resnets: Vec<ResnetBlock2D>,
    pub attentions: Option<Vec<Transformer2DModel>>,
    pub upsample: Option<Upsample>,
    pub channels_out: usize,
    /// Channel counts of the skip connections this up-block consumes
    /// (in pop order — last-pushed first). Used to size the concat.
    pub skip_channels: Vec<usize>,
}

impl UpBlock2D {
    /// Forward; pops one skip per resnet from `skip_states_in_reverse`
    /// (the bottom of the stack). Returns (output, h_out, w_out).
    pub fn forward(
        &self,
        input: &[f32],
        n: usize,
        h: usize,
        w: usize,
        skip_states: &mut Vec<Vec<f32>>,
        skip_channels: &mut Vec<usize>,
        time_emb: &[f32],
        text_emb: &[f32],
        text_seq: usize,
    ) -> (Vec<f32>, usize, usize) {
        let mut x = input.to_vec();
        // Recover the current channel count from the buffer length rather
        // than tracking it through self.skip_channels — that struct field
        // is the static expected-skip-list, not a runtime channel counter.
        let mut x_c = if n * h * w > 0 { x.len() / (n * h * w) } else { 0 };
        for (i, resnet) in self.resnets.iter().enumerate() {
            let skip = skip_states.pop().expect("ran out of skip states");
            let sc = skip_channels.pop().expect("ran out of skip channels");
            x = cat_channels(&x, x_c, &skip, sc, n, h, w);
            x_c += sc;
            x = resnet.forward(&x, n, h, w, time_emb);
            x_c = resnet.out_channels;
            if let Some(attns) = &self.attentions {
                x = attns[i].forward(&x, n, h, w, text_emb, text_seq);
            }
        }
        let _ = x_c;
        let mut h_out = h;
        let mut w_out = w;
        if let Some(up) = &self.upsample {
            let (nx, hn, wn) = up.forward(&x, n, h_out, w_out);
            x = nx; h_out = hn; w_out = wn;
        }
        (x, h_out, w_out)
    }

    /// GPU-resident forward. Pops one skip per resnet from the GpuTensor
    /// skip stack and concats it to the working tensor before each resnet.
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        input: &crate::metal_backend::GpuTensor,
        n: usize, h: usize, w: usize,
        skip_states: &mut Vec<crate::metal_backend::GpuTensor>,
        skip_channels: &mut Vec<usize>,
        time_emb: &[f32],
        text_emb: &[f32],
        text_seq: usize,
    ) -> (crate::metal_backend::GpuTensor, usize, usize) {
        use crate::metal_backend::*;
        let mut x = input.clone_data();
        let mut x_c = if n * h * w > 0 { x.elements() / (n * h * w) } else { 0 };
        for (i, resnet) in self.resnets.iter().enumerate() {
            let skip = skip_states.pop().expect("ran out of skip states");
            let sc = skip_channels.pop().expect("ran out of skip channels");
            x = cat_channels_f16_gpu(&x, x_c, &skip, sc, n, h, w);
            x_c += sc;
            x = resnet.forward_gpu(&x, n, h, w, time_emb);
            x_c = resnet.out_channels;
            if let Some(attns) = &self.attentions {
                x = attns[i].forward_gpu(&x, n, h, w, text_emb, text_seq);
            }
        }
        let _ = x_c;
        let mut h_out = h;
        let mut w_out = w;
        if let Some(up) = &self.upsample {
            let (nx, hn, wn) = up.forward_gpu(&x, n, h_out, w_out);
            x = nx; h_out = hn; w_out = wn;
        }
        (x, h_out, w_out)
    }
}
