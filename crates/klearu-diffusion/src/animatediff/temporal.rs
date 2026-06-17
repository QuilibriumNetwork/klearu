//! Temporal self-attention block for AnimateDiff.

use crate::layers::attention::Attention;
use crate::layers::group_norm::GroupNorm;
use crate::layers::layer_norm::LayerNorm;
use crate::layers::linear::Linear;

/// One temporal transformer block: norm → self-attn → norm → ff.
pub struct TemporalTransformerBlock {
    pub norm1: LayerNorm,
    pub attn1: Attention,     // temporal self-attention
    pub norm2: LayerNorm,
    pub attn2: Attention,     // second temporal self-attention
    pub norm3: LayerNorm,
    pub ff_net_0: Linear,     // GEGLU: projects to 2×ff_inner, then splits
    pub ff_net_2: Linear,     // projects ff_inner back to inner_dim
    /// Learned positional embedding, shape `[max_frames, inner_dim]`.
    /// Loaded from `pos_embed.pe` in the adapter weights; replaces the
    /// sinusoidal table when present.
    pub pos_embed: Option<Vec<f32>>,
    pub pos_embed_max_frames: usize,
    pub inner_dim: usize,
    pub ff_inner_dim: usize,
}

impl TemporalTransformerBlock {
    pub fn new(inner_dim: usize, num_heads: usize, ff_inner_dim: usize) -> Self {
        let head_dim = inner_dim / num_heads;
        Self {
            norm1: LayerNorm::new(inner_dim, 1e-5),
            attn1: Attention::new(inner_dim, inner_dim, num_heads, head_dim, true),
            norm2: LayerNorm::new(inner_dim, 1e-5),
            attn2: Attention::new(inner_dim, inner_dim, num_heads, head_dim, true),
            norm3: LayerNorm::new(inner_dim, 1e-5),
            ff_net_0: Linear::new(inner_dim, ff_inner_dim * 2, true),
            ff_net_2: Linear::new(ff_inner_dim, inner_dim, true),
            pos_embed: None,
            pos_embed_max_frames: 0,
            inner_dim,
            ff_inner_dim,
        }
    }

    /// Forward on `[n_seq, num_frames, inner_dim]` flattened to
    /// `[n_seq * num_frames * inner_dim]`. Each of `n_seq` independent
    /// sequences attends across `num_frames`.
    pub fn forward(&self, x: &mut [f32], n_seq: usize, num_frames: usize) {
        let d = self.inner_dim;
        let total = n_seq * num_frames * d;
        assert_eq!(x.len(), total);

        fn blk_nan(name: &str, data: &[f32]) {
            let nan = data.iter().filter(|v| v.is_nan()).count();
            let inf = data.iter().filter(|v| v.is_infinite()).count();
            if nan + inf > 0 {
                let mut mx = 0.0f32;
                for &v in data { if v.is_finite() { mx = mx.max(v.abs()); } }
                eprintln!("[BLOCK TRACE] {name}: NaN={nan} Inf={inf}/{} max_abs={mx:.4}", data.len());
            }
        }

        // Apply learned positional embedding before attention.
        if let Some(pe) = &self.pos_embed {
            let max_f = self.pos_embed_max_frames;
            for s in 0..n_seq {
                for f in 0..num_frames.min(max_f) {
                    let off = s * num_frames * d + f * d;
                    let pe_off = f * d;
                    for j in 0..d {
                        x[off + j] += pe[pe_off + j];
                    }
                }
            }
            blk_nan("after_pos_embed", x);
        }

        // Norm1 + attn1 (temporal self-attention) + residual.
        let mut normed = x.to_vec();
        for i in 0..(n_seq * num_frames) {
            let off = i * d;
            self.norm1.forward_inplace(&mut normed[off..off + d]);
        }
        blk_nan("after_norm1", &normed);
        let mut attn_out = vec![0.0f32; total];
        self.attn1.forward(
            &normed, &normed,
            n_seq, num_frames, num_frames,
            false, &mut attn_out,
        );
        blk_nan("after_attn1", &attn_out);
        for i in 0..total { x[i] += attn_out[i]; }
        blk_nan("after_attn1_residual", x);

        // Norm2 + attn2 (second self-attention) + residual.
        let mut normed2 = x.to_vec();
        for i in 0..(n_seq * num_frames) {
            let off = i * d;
            self.norm2.forward_inplace(&mut normed2[off..off + d]);
        }
        blk_nan("after_norm2", &normed2);
        let mut attn2_out = vec![0.0f32; total];
        self.attn2.forward(
            &normed2, &normed2,
            n_seq, num_frames, num_frames,
            false, &mut attn2_out,
        );
        blk_nan("after_attn2", &attn2_out);
        for i in 0..total { x[i] += attn2_out[i]; }

        // Norm3 + GEGLU feedforward + residual.
        let residual: Vec<f32> = x.to_vec();
        for i in 0..(n_seq * num_frames) {
            let off = i * d;
            self.norm3.forward_inplace(&mut x[off..off + d]);
        }
        // Project to 2×ff_inner via GEGLU.
        let half = self.ff_inner_dim;
        let tokens = n_seq * num_frames;
        let mut projected = vec![0.0f32; tokens * half * 2];
        self.ff_net_0.forward_batch(x, &mut projected);
        blk_nan("after_ff_net_0", &projected);
        // GEGLU split: val * GELU(gate)
        let mut gated = vec![0.0f32; tokens * half];
        for i in 0..tokens {
            for j in 0..half {
                let val = projected[i * half * 2 + j];
                let gate = projected[i * half * 2 + half + j];
                let gelu = gate * 0.5 * (1.0 + fast_erf(gate * std::f32::consts::FRAC_1_SQRT_2));
                gated[i * half + j] = val * gelu;
            }
        }
        blk_nan("after_geglu", &gated);
        // Project back to inner_dim.
        let mut ff_out = vec![0.0f32; total];
        self.ff_net_2.forward_batch(&gated, &mut ff_out);
        blk_nan("after_ff_net_2", &ff_out);
        for i in 0..total {
            x[i] = residual[i] + ff_out[i]; // residual
        }
        blk_nan("after_ff_residual", x);
    }
}

impl TemporalTransformerBlock {
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        x: &mut crate::metal_backend::GpuTensor,
        n_seq: usize,
        num_frames: usize,
    ) {
        use crate::metal_backend::*;
        let d = self.inner_dim;

        if let Some(pe) = &self.pos_embed {
            let pe_buf = weight_f16_buffer(pe);
            pos_embed_add_f16_gpu(x, &pe_buf, num_frames, self.pos_embed_max_frames, d);
        }

        // norm1 + attn1 (self) + residual
        let n1_g = weight_f16_buffer(&self.norm1.gamma);
        let n1_b = weight_f16_buffer(&self.norm1.beta);
        let x_norm = layer_norm_f16_gpu_out(x, &n1_g, &n1_b, d, self.norm1.eps);
        let sa1 = self.attn1.forward_gpu(
            &x_norm, &x_norm, n_seq, num_frames, num_frames, false);
        drop(x_norm);
        eadd_f16_gpu(x, &sa1);
        drop(sa1);

        // norm2 + attn2 (self) + residual
        let n2_g = weight_f16_buffer(&self.norm2.gamma);
        let n2_b = weight_f16_buffer(&self.norm2.beta);
        let x_norm2 = layer_norm_f16_gpu_out(x, &n2_g, &n2_b, d, self.norm2.eps);
        let sa2 = self.attn2.forward_gpu(
            &x_norm2, &x_norm2, n_seq, num_frames, num_frames, false);
        drop(x_norm2);
        eadd_f16_gpu(x, &sa2);
        drop(sa2);

        // norm3 + GEGLU FF + residual
        let n3_g = weight_f16_buffer(&self.norm3.gamma);
        let n3_b = weight_f16_buffer(&self.norm3.beta);
        let x_norm3 = layer_norm_f16_gpu_out(x, &n3_g, &n3_b, d, self.norm3.eps);
        let projected = self.ff_net_0.forward_gpu(&x_norm3);
        drop(x_norm3);
        let tokens = n_seq * num_frames;
        let gated = geglu_split_f16_gpu(&projected, tokens, self.ff_inner_dim);
        drop(projected);
        let ff_out = self.ff_net_2.forward_gpu(&gated);
        drop(gated);
        eadd_f16_gpu(x, &ff_out);
    }
}

/// Fast erf approximation (Abramowitz & Stegun, max error ~1.5e-7).
fn fast_erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Full temporal transformer at one UNet injection site:
/// GroupNorm → proj_in → N blocks → proj_out, with sinusoidal pos-embed.
pub struct TemporalTransformer {
    pub norm: GroupNorm,
    pub proj_in: Linear,
    pub blocks: Vec<TemporalTransformerBlock>,
    pub proj_out: Linear,
    pub channels: usize,
    pub inner_dim: usize,
    pub pos_embed: Vec<f32>,
    pub max_frames: usize,
}

impl TemporalTransformer {
    pub fn new(
        channels: usize,
        inner_dim: usize,
        num_heads: usize,
        num_layers: usize,
        ff_inner_dim: usize,
        max_frames: usize,
    ) -> Self {
        let blocks = (0..num_layers)
            .map(|_| TemporalTransformerBlock::new(inner_dim, num_heads, ff_inner_dim))
            .collect();
        // proj_in/proj_out: identity matrix init. Many AnimateDiff
        // adapters don't include these weights (the 405-param SDXL beta
        // adapter has only block-level params). Zero-init would make
        // proj_in(x) = 0, collapsing all temporal information. Identity
        // init makes them pass-through when not overridden by adapter.
        let mut proj_in = Linear::new(channels, inner_dim, true);
        for i in 0..channels.min(inner_dim) {
            proj_in.weight[i * channels + i] = 1.0;
        }
        let mut proj_out = Linear::new(inner_dim, channels, true);
        for i in 0..inner_dim.min(channels) {
            proj_out.weight[i * inner_dim + i] = 1.0;
        }
        // Verify identity init.
        let pi_diag: f32 = (0..channels.min(inner_dim)).map(|i| proj_in.weight[i * channels + i]).sum();
        let po_diag: f32 = (0..inner_dim.min(channels)).map(|i| proj_out.weight[i * inner_dim + i]).sum();
        let pi_sum: f64 = proj_in.weight.iter().map(|v| *v as f64).sum();
        let po_sum: f64 = proj_out.weight.iter().map(|v| *v as f64).sum();
        eprintln!("[TEMPORAL INIT] c={channels} d={inner_dim} proj_in: diag_sum={pi_diag} total_sum={pi_sum:.1} len={}, proj_out: diag_sum={po_diag} total_sum={po_sum:.1} len={}",
            proj_in.weight.len(), proj_out.weight.len());
        Self {
            norm: GroupNorm::new(32, channels, 1e-6),
            proj_in,
            blocks,
            proj_out,
            channels,
            inner_dim,
            pos_embed: build_sinusoidal_pos_embed(max_frames, inner_dim),
            max_frames,
        }
    }

    /// Apply temporal attention to `x` of shape
    /// `[batch * num_frames, channels, h, w]`.
    pub fn forward(
        &self,
        x: &mut [f32],
        batch: usize,
        num_frames: usize,
        h: usize,
        w: usize,
    ) {
        let c = self.channels;
        let d = self.inner_dim;
        let bf = batch * num_frames;
        assert_eq!(x.len(), bf * c * h * w);

        fn nan_stats(name: &str, data: &[f32]) {
            let nan = data.iter().filter(|v| v.is_nan()).count();
            let inf = data.iter().filter(|v| v.is_infinite()).count();
            if nan + inf > 0 {
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                for &v in data { if v.is_finite() { mn = mn.min(v); mx = mx.max(v); } }
                eprintln!("[TEMPORAL TRACE] {name}: NaN={nan} Inf={inf} / {} | finite range=[{mn:.4}, {mx:.4}]",
                    data.len());
            }
        }

        nan_stats("input", x);

        // GroupNorm per sample.
        let mut normed = x.to_vec();
        for s in 0..bf {
            let off = s * c * h * w;
            self.norm.forward_inplace(&mut normed[off..off + c * h * w], c, h, w);
        }
        nan_stats("after_groupnorm", &normed);

        // Reshape [bf, c, h, w] → [b*h*w, f, c].
        let bhw = batch * h * w;
        let mut temporal = vec![0.0f32; bhw * num_frames * c];
        for b in 0..batch {
            for f in 0..num_frames {
                let src_s = b * num_frames + f;
                for hi in 0..h {
                    for wi in 0..w {
                        let sp = b * h * w + hi * w + wi;
                        for ci in 0..c {
                            temporal[sp * num_frames * c + f * c + ci] =
                                normed[src_s * c * h * w + ci * h * w + hi * w + wi];
                        }
                    }
                }
            }
        }

        // proj_in: [bhw * f, c] → [bhw * f, d].
        let tokens = bhw * num_frames;
        let mut proj = vec![0.0f32; tokens * d];
        self.proj_in.forward_batch(&temporal, &mut proj);
        nan_stats("after_proj_in", &proj);

        // Temporal transformer blocks.
        for (bi, block) in self.blocks.iter().enumerate() {
            block.forward(&mut proj, bhw, num_frames);
            nan_stats(&format!("after_block_{bi}"), &proj);
        }

        // proj_out: [bhw * f, d] → [bhw * f, c].
        let mut proj_back = vec![0.0f32; tokens * c];
        self.proj_out.forward_batch(&proj, &mut proj_back);
        nan_stats("after_proj_out", &proj_back);

        // Reshape [b*h*w, f, c] → [bf, c, h, w] and add to x (residual).
        for b in 0..batch {
            for f in 0..num_frames {
                let dst_s = b * num_frames + f;
                for hi in 0..h {
                    for wi in 0..w {
                        let sp = b * h * w + hi * w + wi;
                        for ci in 0..c {
                            x[dst_s * c * h * w + ci * h * w + hi * w + wi] +=
                                proj_back[sp * num_frames * c + f * c + ci];
                        }
                    }
                }
            }
        }
    }
}

impl TemporalTransformer {
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        x: &mut crate::metal_backend::GpuTensor,
        batch: usize,
        num_frames: usize,
        h: usize,
        w: usize,
    ) {
        use crate::metal_backend::*;
        let c = self.channels;

        // Reshape [B*F, C, H, W] → [B*H*W, F, C] directly — no outer
        // GroupNorm (the adapter doesn't include norm weights, and the
        // reference AnimateDiff impl feeds features directly to temporal).
        let temporal = temporal_reshape_to_f16_gpu(x, batch, num_frames, c, h, w);

        let mut proj = self.proj_in.forward_gpu(&temporal);
        drop(temporal);

        let bhw = batch * h * w;
        for block in &self.blocks {
            block.forward_gpu(&mut proj, bhw, num_frames);
        }

        let proj_back = self.proj_out.forward_gpu(&proj);
        drop(proj);

        let delta = temporal_reshape_from_f16_gpu(&proj_back, batch, num_frames, c, h, w);
        drop(proj_back);

        eadd_f16_gpu(x, &delta);
    }
}

fn build_sinusoidal_pos_embed(max_len: usize, dim: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; max_len * dim];
    let half = dim / 2;
    for pos in 0..max_len {
        for i in 0..half {
            let freq = 1.0 / (10000.0_f32).powf(i as f32 / half as f32);
            let angle = pos as f32 * freq;
            table[pos * dim + i] = angle.sin();
            table[pos * dim + half + i] = angle.cos();
        }
    }
    table
}
