//! Transformer2DModel — applies N transformer blocks to a flattened
//! latent grid, with cross-attention to text.
//!
//! Sequence per BasicTransformerBlock:
//!   - LayerNorm → SelfAttention (residual)
//!   - LayerNorm → CrossAttention (residual)
//!   - LayerNorm → FeedForward (GeGLU) (residual)
//!
//! The outer Transformer2DModel:
//!   - GroupNorm input
//!   - Linear or Conv 1×1 input projection
//!   - reshape [B,C,H,W] → [B,H*W,C]
//!   - N basic blocks
//!   - reshape back
//!   - Linear or Conv 1×1 output projection
//!   - residual + input

use crate::error::Result;
use crate::layers::{Attention, GroupNorm, LayerNorm, Linear, Conv2d, gelu_inplace};
use crate::weight::{ComponentTensors, load_conv2d, load_group_norm, load_layer_norm, load_linear};

/// Reshape [N, C, H*W] → [N, H*W, C]. Spatially transposes the (C, HW) slab
/// for each batch. Parallelised across batches.
fn nchw_to_nhwc(src: &[f32], n: usize, c: usize, hw: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut dst = vec![0.0f32; n * hw * c];
    dst.par_chunks_mut(hw * c).enumerate().for_each(|(ni, dst_n)| {
        let src_n = &src[ni * c * hw..(ni + 1) * c * hw];
        for ci in 0..c {
            let src_row = &src_n[ci * hw..(ci + 1) * hw];
            for j in 0..hw {
                dst_n[j * c + ci] = src_row[j];
            }
        }
    });
    dst
}

/// Reshape [N, H*W, C] → [N, C, H*W]. Inverse of `nchw_to_nhwc`.
fn nhwc_to_nchw(src: &[f32], n: usize, c: usize, hw: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut dst = vec![0.0f32; n * c * hw];
    dst.par_chunks_mut(c * hw).enumerate().for_each(|(ni, dst_n)| {
        let src_n = &src[ni * hw * c..(ni + 1) * hw * c];
        for ci in 0..c {
            let dst_row = &mut dst_n[ci * hw..(ci + 1) * hw];
            for j in 0..hw {
                dst_row[j] = src_n[j * c + ci];
            }
        }
    });
    dst
}

pub struct BasicTransformerBlock {
    pub norm1: LayerNorm,
    pub self_attn: Attention,
    pub norm2: LayerNorm,
    pub cross_attn: Attention,
    pub norm3: LayerNorm,
    pub ff: FeedForwardGeGLU,
    pub dim: usize,
}

impl BasicTransformerBlock {
    pub fn new(dim: usize, num_heads: usize, head_dim: usize, cross_attention_dim: usize) -> Self {
        Self {
            norm1: LayerNorm::new(dim, 1e-5),
            self_attn: Attention::new(dim, dim, num_heads, head_dim, true),
            norm2: LayerNorm::new(dim, 1e-5),
            cross_attn: Attention::new(dim, cross_attention_dim, num_heads, head_dim, true),
            norm3: LayerNorm::new(dim, 1e-5),
            ff: FeedForwardGeGLU::new(dim, 4 * dim),
            dim,
        }
    }

    /// Forward over a [n × seq × dim] tensor (already reshaped from spatial).
    pub fn forward(
        &self,
        x: &mut Vec<f32>,
        n: usize,
        seq: usize,
        text_emb: &[f32],
        text_seq: usize,
    ) {
        let trace = std::env::var_os("KLEARU_BTB_TRACE").is_some();
        // When n==2 (CFG batch), also logs the per-element mean|batch1 - batch0|
        // — the cond-vs-uncond signal at this point in the network. Should
        // grow at each cross-attn site (where text gets mixed in) and
        // propagate forward.
        let half = if n == 2 { x.len() / 2 } else { 0 };
        let stat = |label: &str, v: &[f32]| {
            if trace {
                let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
                let mut nan = 0; let mut inf = 0; let mut sum_abs = 0.0f64;
                for &x in v {
                    if x.is_nan() { nan += 1; }
                    else if x.is_infinite() { inf += 1; }
                    else { mn = mn.min(x); mx = mx.max(x); sum_abs += x.abs() as f64; }
                }
                let extra = if n == 2 && v.len() == 2 * half && half > 0 {
                    let (a, b) = v.split_at(half);
                    let mut diff = 0.0f64;
                    for (u, c) in a.iter().zip(b.iter()) {
                        let d = (c - u).abs();
                        if d.is_finite() { diff += d as f64; }
                    }
                    format!(", mean|c-u|={:.5}", diff / half as f64)
                } else { String::new() };
                eprintln!("        [btb] {label:<24} len={}, min={mn:.3}, max={mx:.3}, mean_abs={:.4}, NaN={nan}, Inf={inf}{extra}",
                          v.len(), sum_abs / v.len() as f64);
            }
        };
        stat("input", x);

        // Self-attention
        let mut x_norm = x.clone();
        self.norm1.forward_inplace(&mut x_norm);
        stat("after norm1", &x_norm);
        let mut sa_out = vec![0.0f32; x.len()];
        self.self_attn.forward(&x_norm, &x_norm, n, seq, seq, false, &mut sa_out);
        stat("after self_attn", &sa_out);
        for i in 0..x.len() { x[i] += sa_out[i]; }
        stat("after self_attn+res", x);

        // Cross-attention
        let mut x_norm = x.clone();
        self.norm2.forward_inplace(&mut x_norm);
        stat("after norm2", &x_norm);
        let mut ca_out = vec![0.0f32; x.len()];
        self.cross_attn.forward(&x_norm, text_emb, n, seq, text_seq, false, &mut ca_out);
        stat("after cross_attn", &ca_out);
        for i in 0..x.len() { x[i] += ca_out[i]; }
        stat("after cross_attn+res", x);

        // FF
        let mut x_norm = x.clone();
        self.norm3.forward_inplace(&mut x_norm);
        stat("after norm3", &x_norm);
        let ff_out = self.ff.forward(&x_norm);
        stat("after ff", &ff_out);
        for i in 0..x.len() { x[i] += ff_out[i]; }
        stat("after ff+res", x);
    }

    /// GPU-resident forward. `x` is fp16 GpuTensor `[N, seq, dim]`;
    /// `text_emb` is fp16 GpuTensor `[N, text_seq, cross_attn_dim]`.
    /// Returns the updated x as a fresh GpuTensor.
    ///
    /// Chains: norm1 → self_attn → residual → norm2 → cross_attn → residual
    ///       → norm3 → GeGLU FF → residual. All ops on GPU.
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        x_gpu: &crate::metal_backend::GpuTensor,
        n: usize,
        seq: usize,
        text_emb_gpu: &crate::metal_backend::GpuTensor,
        text_seq: usize,
    ) -> crate::metal_backend::GpuTensor {
        use crate::metal_backend::*;
        let dim = self.norm1.normalized_shape;
        // KLEARU_DISABLE_FUSED_NORMS=1 reverts to a clone + in-place LN
        // (separate buffers); default uses the fused out-of-place kernel.
        let disable_fused = std::env::var_os("KLEARU_DISABLE_FUSED_NORMS").is_some();
        let mut x = x_gpu.clone_data();

        // norm1 + self-attn + residual.
        let n1_g = weight_f16_buffer(&self.norm1.gamma);
        let n1_b = weight_f16_buffer(&self.norm1.beta);
        let x_norm = if disable_fused {
            let mut t = x.clone_data();
            layer_norm_f16_gpu(&mut t, &n1_g, &n1_b, dim, self.norm1.eps);
            t
        } else {
            layer_norm_f16_gpu_out(&x, &n1_g, &n1_b, dim, self.norm1.eps)
        };
        let sa_out = self.self_attn.forward_gpu_self(&x_norm, n, seq);
        eadd_f16_gpu(&mut x, &sa_out);
        drop(sa_out); drop(x_norm);

        // norm2 + cross-attn + residual.
        let n2_g = weight_f16_buffer(&self.norm2.gamma);
        let n2_b = weight_f16_buffer(&self.norm2.beta);
        let x_norm = if disable_fused {
            let mut t = x.clone_data();
            layer_norm_f16_gpu(&mut t, &n2_g, &n2_b, dim, self.norm2.eps);
            t
        } else {
            layer_norm_f16_gpu_out(&x, &n2_g, &n2_b, dim, self.norm2.eps)
        };
        let ca_out = self.cross_attn.forward_gpu(&x_norm, text_emb_gpu, n, seq, text_seq, false);
        eadd_f16_gpu(&mut x, &ca_out);
        drop(ca_out); drop(x_norm);

        // norm3 + GeGLU FF + residual.
        let n3_g = weight_f16_buffer(&self.norm3.gamma);
        let n3_b = weight_f16_buffer(&self.norm3.beta);
        let x_norm = if disable_fused {
            let mut t = x.clone_data();
            layer_norm_f16_gpu(&mut t, &n3_g, &n3_b, dim, self.norm3.eps);
            t
        } else {
            layer_norm_f16_gpu_out(&x, &n3_g, &n3_b, dim, self.norm3.eps)
        };
        let n_rows = n * seq;
        // KLEARU_ENABLE_MPSGRAPH_GEGLU=1 routes the feed-forward through
        // an MPSGraph-fused FFN. The default split-kernel path is faster
        // at this scale (per-call MPSGraph dispatch overhead exceeds the
        // fusion benefit) and more numerically stable.
        let ff_out = if std::env::var_os("KLEARU_ENABLE_MPSGRAPH_GEGLU").is_some() {
            mpsgraph_geglu_ffn_f16_gpu(
                &x_norm,
                &self.ff.proj_in.weight, self.ff.proj_in.bias.as_deref(),
                &self.ff.proj_out.weight, self.ff.proj_out.bias.as_deref(),
                n_rows, dim, self.ff.hidden, dim,
            )
        } else {
            let h = self.ff.proj_in.forward_gpu(&x_norm);
            let gated = geglu_split_f16_gpu(&h, n_rows, self.ff.hidden);
            self.ff.proj_out.forward_gpu(&gated)
        };
        drop(x_norm);
        eadd_f16_gpu(&mut x, &ff_out);
        x
    }
}

impl BasicTransformerBlock {
    /// Load weights at HF Diffusers prefix:
    ///   `<prefix>.norm1.{weight,bias}` LayerNorm
    ///   `<prefix>.attn1.{to_q,to_k,to_v,to_out.0}.{weight,bias?}` self-attn
    ///   `<prefix>.norm2.{weight,bias}`
    ///   `<prefix>.attn2.{to_q,to_k,to_v,to_out.0}.{weight,bias?}` cross-attn
    ///   `<prefix>.norm3.{weight,bias}`
    ///   `<prefix>.ff.net.0.proj.{weight,bias}` GeGLU input (proj_in)
    ///   `<prefix>.ff.net.2.{weight,bias}` GeGLU output (proj_out)
    pub fn load_from(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        load_layer_norm(comp, &format!("{prefix}.norm1"), &mut self.norm1)?;
        load_linear(comp, &format!("{prefix}.attn1.to_q"), &mut self.self_attn.to_q)?;
        load_linear(comp, &format!("{prefix}.attn1.to_k"), &mut self.self_attn.to_k)?;
        load_linear(comp, &format!("{prefix}.attn1.to_v"), &mut self.self_attn.to_v)?;
        load_linear(comp, &format!("{prefix}.attn1.to_out.0"), &mut self.self_attn.to_out)?;
        load_layer_norm(comp, &format!("{prefix}.norm2"), &mut self.norm2)?;
        load_linear(comp, &format!("{prefix}.attn2.to_q"), &mut self.cross_attn.to_q)?;
        load_linear(comp, &format!("{prefix}.attn2.to_k"), &mut self.cross_attn.to_k)?;
        load_linear(comp, &format!("{prefix}.attn2.to_v"), &mut self.cross_attn.to_v)?;
        load_linear(comp, &format!("{prefix}.attn2.to_out.0"), &mut self.cross_attn.to_out)?;
        load_layer_norm(comp, &format!("{prefix}.norm3"), &mut self.norm3)?;
        load_linear(comp, &format!("{prefix}.ff.net.0.proj"), &mut self.ff.proj_in)?;
        load_linear(comp, &format!("{prefix}.ff.net.2"), &mut self.ff.proj_out)?;
        Ok(())
    }
}

impl Transformer2DModel {
    /// Load at prefix `<...>.attentions.<n>` style.
    /// Names: `<prefix>.norm.{weight,bias}` GroupNorm
    ///        `<prefix>.proj_in.{weight,bias}` (Linear or Conv2d 1×1)
    ///        `<prefix>.transformer_blocks.<i>.{...}`
    ///        `<prefix>.proj_out.{weight,bias}`
    pub fn load_from(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        load_group_norm(comp, &format!("{prefix}.norm"), &mut self.norm)?;
        match &mut self.proj_in {
            ProjIn::Conv(c) => load_conv2d(comp, &format!("{prefix}.proj_in"), c)?,
            ProjIn::Linear(l) => load_linear(comp, &format!("{prefix}.proj_in"), l)?,
        }
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            blk.load_from(comp, &format!("{prefix}.transformer_blocks.{i}"))?;
        }
        match &mut self.proj_out {
            ProjOut::Conv(c) => load_conv2d(comp, &format!("{prefix}.proj_out"), c)?,
            ProjOut::Linear(l) => load_linear(comp, &format!("{prefix}.proj_out"), l)?,
        }
        Ok(())
    }
}

/// Feed-forward with GeGLU activation:
///   x → Linear(dim → 2*hidden) → split into [a, b] → a · gelu(b) → Linear(hidden → dim)
pub struct FeedForwardGeGLU {
    pub proj_in: Linear,  // [dim → 2*hidden]
    pub proj_out: Linear, // [hidden → dim]
    pub dim: usize,
    pub hidden: usize,
}

impl FeedForwardGeGLU {
    pub fn new(dim: usize, hidden: usize) -> Self {
        Self {
            proj_in: Linear::new(dim, 2 * hidden, true),
            proj_out: Linear::new(hidden, dim, true),
            dim,
            hidden,
        }
    }
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let trace = std::env::var_os("KLEARU_BTB_TRACE").is_some();
        let stat = |label: &str, v: &[f32]| {
            if trace {
                let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
                let mut nan = 0; let mut inf = 0; let mut sum_abs = 0.0f64;
                for &x in v {
                    if x.is_nan() { nan += 1; }
                    else if x.is_infinite() { inf += 1; }
                    else { mn = mn.min(x); mx = mx.max(x); sum_abs += x.abs() as f64; }
                }
                let extra = if v.len() % 2 == 0 {
                    let half = v.len() / 2;
                    let (a, b) = v.split_at(half);
                    let mut diff = 0.0f64;
                    for (u, c) in a.iter().zip(b.iter()) {
                        let d = (c - u).abs();
                        if d.is_finite() { diff += d as f64; }
                    }
                    format!(", mean|c-u|={:.5}", diff / half as f64)
                } else { String::new() };
                eprintln!("          [ff]  {label:<24} len={}, min={mn:.3}, max={mx:.3}, mean_abs={:.4}, NaN={nan}, Inf={inf}{extra}",
                          v.len(), sum_abs / v.len() as f64);
            }
        };
        let n = x.len() / self.dim;
        let hidden = self.hidden;
        stat("input", x);
        let mut h = vec![0.0f32; n * 2 * hidden];
        self.proj_in.forward_batch(x, &mut h);
        stat("after proj_in", &h);

        // GeGLU: each row of h is [a (hidden) | b (hidden)]; output is a · gelu(b).
        // Split into separate `a` and `b` slabs so we can batch GELU across all
        // n*hidden elements in one pass (hits Metal kernel threshold on SDXL
        // where hidden ≥ 5120).
        let mut a_buf = vec![0.0f32; n * hidden];
        let mut b_buf = vec![0.0f32; n * hidden];
        h.par_chunks(2 * hidden)
            .zip(a_buf.par_chunks_mut(hidden).zip(b_buf.par_chunks_mut(hidden)))
            .for_each(|(h_row, (a_row, b_row))| {
                a_row.copy_from_slice(&h_row[..hidden]);
                b_row.copy_from_slice(&h_row[hidden..]);
            });
        stat("a (linear half)", &a_buf);
        stat("b (gate half)", &b_buf);

        gelu_inplace(&mut b_buf);
        stat("gelu(b)", &b_buf);

        // gated = a * gelu(b), elementwise (parallel).
        let mut gated = a_buf;
        gated.par_iter_mut().zip(b_buf.par_iter()).for_each(|(g, b)| *g *= b);
        stat("a * gelu(b)", &gated);

        let mut out = vec![0.0f32; n * self.dim];
        self.proj_out.forward_batch(&gated, &mut out);
        stat("after proj_out", &out);
        out
    }
}

/// The full Transformer2DModel: norm + proj_in + N blocks + proj_out + residual.
pub struct Transformer2DModel {
    pub norm: GroupNorm,
    pub proj_in: ProjIn,
    pub blocks: Vec<BasicTransformerBlock>,
    pub proj_out: ProjOut,
    pub channels: usize,
}

/// SD 1.5 uses Conv 1×1 for proj_in/out; SDXL uses Linear (when use_linear_projection=true).
pub enum ProjIn {
    Conv(Conv2d),
    Linear(Linear),
}
pub enum ProjOut {
    Conv(Conv2d),
    Linear(Linear),
}

impl Transformer2DModel {
    pub fn new(
        channels: usize,
        num_heads: usize,
        head_dim: usize,
        cross_attention_dim: usize,
        num_layers: usize,
        use_linear_projection: bool,
        groups: usize,
    ) -> Self {
        let blocks = (0..num_layers)
            .map(|_| BasicTransformerBlock::new(channels, num_heads, head_dim, cross_attention_dim))
            .collect();
        let proj_in = if use_linear_projection {
            ProjIn::Linear(Linear::new(channels, channels, true))
        } else {
            ProjIn::Conv(Conv2d::new(channels, channels, 1, 1, 0, true))
        };
        let proj_out = if use_linear_projection {
            ProjOut::Linear(Linear::new(channels, channels, true))
        } else {
            ProjOut::Conv(Conv2d::new(channels, channels, 1, 1, 0, true))
        };
        Self {
            norm: GroupNorm::new(groups, channels, 1e-6),
            proj_in,
            blocks,
            proj_out,
            channels,
        }
    }

    /// Forward [N, C, H, W] → [N, C, H, W] residual.
    /// `text_emb` is [n × text_seq × cross_attention_dim].
    pub fn forward(
        &self,
        input: &[f32],
        n: usize,
        h: usize,
        w: usize,
        text_emb: &[f32],
        text_seq: usize,
    ) -> Vec<f32> {
        let c = self.channels;
        let hw = h * w;

        // norm + proj_in
        let mut x = input.to_vec();
        self.norm.forward_inplace(&mut x, n, h, w);
        let mut x_proj = match &self.proj_in {
            ProjIn::Conv(c) => {
                let mut out = Vec::new();
                c.forward(&x, n, h, w, &mut out);
                out
            }
            ProjIn::Linear(l) => {
                // Linear in spatial form: rearrange [N,C,HW] → [N,HW,C], linear, back.
                let nhwc = nchw_to_nhwc(&x, n, c, hw);
                let mut proj = vec![0.0f32; n * hw * c];
                l.forward_batch(&nhwc, &mut proj);
                nhwc_to_nchw(&proj, n, c, hw)
            }
        };

        // Reshape [N, C, H*W] → [N, H*W, C] (sequence-first for attention)
        let mut seq = nchw_to_nhwc(&x_proj, n, c, hw);

        // Apply each transformer block in sequence.
        for block in &self.blocks {
            block.forward(&mut seq, n, hw, text_emb, text_seq);
        }

        // Reshape back [N, H*W, C] → [N, C, H, W]
        x_proj = nhwc_to_nchw(&seq, n, c, hw);

        // proj_out + residual.
        let mut out_proj = match &self.proj_out {
            ProjOut::Conv(c) => {
                let mut out = Vec::new();
                c.forward(&x_proj, n, h, w, &mut out);
                out
            }
            ProjOut::Linear(l) => {
                let nhwc = nchw_to_nhwc(&x_proj, n, c, hw);
                let mut proj = vec![0.0f32; n * hw * c];
                l.forward_batch(&nhwc, &mut proj);
                nhwc_to_nchw(&proj, n, c, hw)
            }
        };

        for (a, b) in out_proj.iter_mut().zip(input.iter()) {
            *a += b;
        }
        out_proj
    }

    /// CPU bridge fallback: download → CPU forward → upload. Selected
    /// when `KLEARU_DISABLE_GPU_TRANSFORMER=1` is set.
    #[cfg(feature = "metal")]
    fn forward_gpu_via_cpu_bridge(
        &self,
        input: &crate::metal_backend::GpuTensor,
        n: usize, h: usize, w: usize,
        text_emb: &[f32],
        text_seq: usize,
    ) -> crate::metal_backend::GpuTensor {
        use crate::metal_backend::GpuTensor;
        let cpu_in = input.download_to_f32();
        let cpu_out = self.forward(&cpu_in, n, h, w, text_emb, text_seq);
        GpuTensor::upload_f32_as_f16(input.shape.clone(), &cpu_out)
    }

    /// Full GPU-resident forward. All ops chain on GPU buffers — no CPU
    /// round-trips.
    ///
    /// Sequence (mirrors `forward`):
    ///   1. norm (outer GroupNorm) on input
    ///   2. proj_in (Conv 1×1 OR Linear)
    ///   3. NCHW → NHWC reshape
    ///   4. N BasicTransformerBlock::forward_gpu
    ///   5. NHWC → NCHW reshape
    ///   6. proj_out
    ///   7. residual + input
    ///
    /// `KLEARU_DISABLE_GPU_TRANSFORMER=1` falls back to the CPU bridge
    /// (download → CPU forward → upload).
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        input: &crate::metal_backend::GpuTensor,
        n: usize, h: usize, w: usize,
        text_emb: &[f32],
        text_seq: usize,
    ) -> crate::metal_backend::GpuTensor {
        if std::env::var_os("KLEARU_DISABLE_GPU_TRANSFORMER").is_some() {
            return self.forward_gpu_via_cpu_bridge(input, n, h, w, text_emb, text_seq);
        }
        use crate::metal_backend::*;
        let c = self.channels;
        let hw = h * w;

        // norm (in-place on a clone so the residual `input` is preserved).
        let mut x = input.clone_data();
        let no_g = weight_f16_buffer(&self.norm.gamma);
        let no_b = weight_f16_buffer(&self.norm.beta);
        groupnorm_f16_gpu(&mut x, &no_g, &no_b,
                          n, c, h, w,
                          self.norm.num_groups, self.norm.eps);

        // proj_in: Conv 1×1 (SD 1.5) OR Linear (SDXL with use_linear_projection).
        let mut x_proj = match &self.proj_in {
            ProjIn::Conv(conv) => {
                let (out, _, _) = conv.forward_gpu(&x, n, h, w);
                out
            }
            ProjIn::Linear(lin) => {
                // Linear path: NCHW → NHWC reshape, Linear, NHWC → NCHW.
                let nhwc = nchw_to_nhwc_f16_gpu(&x, n, c, hw);
                let proj = lin.forward_gpu(&nhwc);  // [n*hw, c]
                drop(nhwc);
                // proj shape is [n*hw, c]; reshape to [n, hw, c] then to [n, c, h, w].
                let proj = proj.reshape(vec![n, hw, c]);
                let nchw = nhwc_to_nchw_f16_gpu(&proj, n, c, hw);
                drop(proj);
                // After nhwc_to_nchw the shape is [n, c, hw] — reshape to [n, c, h, w].
                nchw.reshape(vec![n, c, h, w])
            }
        };
        drop(x);

        // Reshape [N, C, H*W] → [N, H*W, C] for transformer-block sequence layout.
        let mut seq = nchw_to_nhwc_f16_gpu(&x_proj, n, c, hw);

        // Upload text_emb via the cached path. text_emb is stable across all
        // ~25 timesteps and is referenced by every Transformer2DModel — the
        // cache turns this into a single conversion + reuse.
        let text_emb_gpu = GpuTensor::upload_f32_as_f16_cached(
            vec![n, text_seq, text_emb.len() / (n.max(1) * text_seq.max(1))],
            text_emb,
        );

        for block in &self.blocks {
            seq = block.forward_gpu(&seq, n, hw, &text_emb_gpu, text_seq);
        }

        // Reshape back [N, H*W, C] → [N, C, H, W].
        x_proj = nhwc_to_nchw_f16_gpu(&seq, n, c, hw).reshape(vec![n, c, h, w]);
        drop(seq);

        // proj_out: Conv 1×1 OR Linear, then residual add with input.
        let mut out = match &self.proj_out {
            ProjOut::Conv(conv) => {
                let (out, _, _) = conv.forward_gpu(&x_proj, n, h, w);
                out
            }
            ProjOut::Linear(lin) => {
                let nhwc = nchw_to_nhwc_f16_gpu(&x_proj, n, c, hw);
                let proj = lin.forward_gpu(&nhwc).reshape(vec![n, hw, c]);
                drop(nhwc);
                let nchw = nhwc_to_nchw_f16_gpu(&proj, n, c, hw);
                drop(proj);
                nchw.reshape(vec![n, c, h, w])
            }
        };
        drop(x_proj);

        // Residual: out += input.
        eadd_f16_gpu(&mut out, input);
        out
    }
}
