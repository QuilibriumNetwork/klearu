//! ResnetBlock2D — SD's basic conv-residual block.
//!
//! Sequence:
//!   h = GroupNorm(input) → SiLU → Conv 3x3 (in→out)
//!   h = h + Linear(SiLU(time_emb))   ← time conditioning, broadcast across HW
//!   h = GroupNorm(h) → SiLU → Conv 3x3 (out→out)
//!   shortcut = (input if in==out else Conv 1x1(in→out))
//!   return h + shortcut

use crate::error::Result;
use crate::layers::{Conv2d, GroupNorm, Linear, silu_inplace};
use crate::weight::{ComponentTensors, load_conv2d, load_group_norm, load_linear};

pub struct ResnetBlock2D {
    pub norm1: GroupNorm,
    pub conv1: Conv2d,
    pub time_emb_proj: Option<Linear>, // None for VAE which has no time conditioning
    pub norm2: GroupNorm,
    pub conv2: Conv2d,
    pub conv_shortcut: Option<Conv2d>, // 1×1 conv when in_channels != out_channels
    pub in_channels: usize,
    pub out_channels: usize,
    pub time_embed_dim: Option<usize>,
}

impl ResnetBlock2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_embed_dim: Option<usize>,
        groups: usize,
        eps: f32,
    ) -> Self {
        let time_emb_proj = time_embed_dim.map(|d| Linear::new(d, out_channels, true));
        let conv_shortcut = if in_channels != out_channels {
            Some(Conv2d::new(in_channels, out_channels, 1, 1, 0, true))
        } else {
            None
        };
        Self {
            norm1: GroupNorm::new(groups, in_channels, eps),
            conv1: Conv2d::new(in_channels, out_channels, 3, 1, 1, true),
            time_emb_proj,
            norm2: GroupNorm::new(groups, out_channels, eps),
            conv2: Conv2d::new(out_channels, out_channels, 3, 1, 1, true),
            conv_shortcut,
            in_channels,
            out_channels,
            time_embed_dim,
        }
    }

    /// Forward. Input [N, in_C, H, W]; returns [N, out_C, H, W].
    /// `time_emb` is [N × time_embed_dim] (broadcast per-batch); pass empty
    /// slice when `time_embed_dim = None` (VAE).
    pub fn forward(
        &self,
        input: &[f32],
        n: usize,
        h: usize,
        w: usize,
        time_emb: &[f32],
    ) -> Vec<f32> {
        use rayon::prelude::*;
        let oc = self.out_channels;

        // h = GroupNorm(input) → SiLU → Conv1
        let mut x = input.to_vec();
        self.norm1.forward_inplace(&mut x, n, h, w);
        silu_inplace(&mut x);
        let mut h1 = Vec::new();
        let (h_out, w_out) = self.conv1.forward(&x, n, h, w, &mut h1);

        // Add time embedding (broadcast over spatial). Sequential per-batch
        // (n is 1 or 2 with CFG) but the inner per-channel × per-spatial
        // loop is parallelised — for a 64×64×320 ResnetBlock that's 2.6M
        // adds, easily worth the par dispatch.
        if let (Some(proj), Some(_)) = (&self.time_emb_proj, self.time_embed_dim) {
            let te_per_n = time_emb.len() / n.max(1);
            let hw_out = h_out * w_out;
            for ni in 0..n {
                let mut t = vec![0.0f32; oc];
                let mut t_silu = vec![0.0f32; te_per_n];
                t_silu.copy_from_slice(&time_emb[ni * te_per_n..(ni + 1) * te_per_n]);
                silu_inplace(&mut t_silu);
                proj.forward(&t_silu, &mut t);
                let h_off = ni * oc * hw_out;
                // Parallelise across output channels — each channel writes a
                // disjoint hw_out slab. par_chunks_mut gives disjoint mutable
                // access without unsafe.
                h1[h_off..h_off + oc * hw_out]
                    .par_chunks_mut(hw_out)
                    .enumerate()
                    .for_each(|(c, slab)| {
                        let bias = t[c];
                        for v in slab.iter_mut() { *v += bias; }
                    });
            }
        }

        // h = GroupNorm → SiLU → Conv2
        self.norm2.forward_inplace(&mut h1, n, h_out, w_out);
        silu_inplace(&mut h1);
        let mut h2 = Vec::new();
        self.conv2.forward(&h1, n, h_out, w_out, &mut h2);

        // shortcut + h2
        let shortcut = if let Some(sc) = &self.conv_shortcut {
            let mut sc_out = Vec::new();
            sc.forward(input, n, h, w, &mut sc_out);
            sc_out
        } else {
            input.to_vec()
        };
        #[cfg(feature = "metal")]
        if h2.len() >= 1 << 14 {
            crate::metal_backend::eadd_metal(&mut h2, &shortcut);
        } else {
            for (a, b) in h2.iter_mut().zip(shortcut.iter()) { *a += b; }
        }
        #[cfg(not(feature = "metal"))]
        for (a, b) in h2.iter_mut().zip(shortcut.iter()) { *a += b; }
        h2
    }

    /// GPU-resident forward. Input is fp16 GpuTensor `[N, in_C, H, W]`.
    /// All ops (norm1, silu, conv1, time-embed inject, norm2, silu, conv2,
    /// shortcut, eadd) chain on GPU — no CPU↔GPU round-trips between them.
    /// `time_emb` is a CPU `[N, time_embed_dim]` buffer (computed once
    /// per UNet step); the projection through `time_emb_proj` is small
    /// (~1M FLOPs) so we keep it on CPU and upload the per-batch result.
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        input: &crate::metal_backend::GpuTensor,
        n: usize, h: usize, w: usize,
        time_emb: &[f32],
    ) -> crate::metal_backend::GpuTensor {
        use crate::metal_backend::*;
        use crate::layers::silu_inplace;
        let oc = self.out_channels;
        // KLEARU_DISABLE_FUSED_NORMS=1 falls back to the unfused path
        // (clone + groupnorm_inplace + silu_inplace) for diagnostics.
        let disable_fused = std::env::var_os("KLEARU_DISABLE_FUSED_NORMS").is_some();

        // norm1 → silu (fused or split per env flag).
        let n1_gamma = weight_f16_buffer(&self.norm1.gamma);
        let n1_beta  = weight_f16_buffer(&self.norm1.beta);
        let x = if disable_fused {
            let mut tmp = input.clone_data();
            groupnorm_f16_gpu(&mut tmp, &n1_gamma, &n1_beta,
                n, self.in_channels, h, w,
                self.norm1.num_groups, self.norm1.eps);
            silu_f16_gpu(&mut tmp);
            tmp
        } else {
            groupnorm_silu_f16_gpu_out(input, &n1_gamma, &n1_beta,
                n, self.in_channels, h, w,
                self.norm1.num_groups, self.norm1.eps)
        };

        // conv1: → [N, oc, h_out, w_out].
        let (mut h1, h_out, w_out) = self.conv1.forward_gpu(&x, n, h, w);
        drop(x);

        // Time-embedding projection on CPU (small matmul). Per-batch SiLU+Linear.
        if let (Some(proj), Some(_)) = (&self.time_emb_proj, self.time_embed_dim) {
            let per_batch = time_emb.len() / n.max(1);
            let mut t_all = vec![0.0f32; n * oc];
            for ni in 0..n {
                let mut t_silu = vec![0.0f32; per_batch];
                t_silu.copy_from_slice(&time_emb[ni*per_batch..(ni+1)*per_batch]);
                silu_inplace(&mut t_silu);
                let mut t = vec![0.0f32; oc];
                proj.forward(&t_silu, &mut t);
                t_all[ni*oc..(ni+1)*oc].copy_from_slice(&t);
            }
            let t_buf = upload_f32_as_f16_buffer(&t_all);
            bias_add_nc_to_nchw_f16_gpu(&mut h1, &t_buf, oc, h_out * w_out);
            // `t_buf` is fresh-allocated (not pooled); Metal retains it for
            // the lifetime of the bias_add cmd. Just drop our handle — the
            // buffer frees itself once the GPU is done.
            drop(t_buf);
        }

        // norm2 → silu → conv2.
        let n2_gamma = weight_f16_buffer(&self.norm2.gamma);
        let n2_beta  = weight_f16_buffer(&self.norm2.beta);
        if disable_fused {
            groupnorm_f16_gpu(&mut h1, &n2_gamma, &n2_beta,
                              n, oc, h_out, w_out,
                              self.norm2.num_groups, self.norm2.eps);
            silu_f16_gpu(&mut h1);
        } else {
            groupnorm_silu_f16_gpu_inplace(&mut h1, &n2_gamma, &n2_beta,
                              n, oc, h_out, w_out,
                              self.norm2.num_groups, self.norm2.eps);
        }
        let (mut h2, _, _) = self.conv2.forward_gpu(&h1, n, h_out, w_out);
        drop(h1);

        // Shortcut: 1×1 conv if channels mismatch, else clone of input.
        let shortcut = if let Some(sc) = &self.conv_shortcut {
            let (sc_out, _, _) = sc.forward_gpu(input, n, h, w);
            sc_out
        } else {
            input.clone_data()
        };
        eadd_f16_gpu(&mut h2, &shortcut);
        h2
    }
}

impl ResnetBlock2D {
    /// Load weights from a ComponentTensors at the given prefix.
    /// Names follow HF Diffusers convention:
    ///   `<prefix>.norm1.{weight,bias}`, `<prefix>.conv1.{weight,bias}`,
    ///   `<prefix>.time_emb_proj.{weight,bias}`,
    ///   `<prefix>.norm2.{weight,bias}`, `<prefix>.conv2.{weight,bias}`,
    ///   `<prefix>.conv_shortcut.{weight,bias}` (when in != out)
    pub fn load_from(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        load_group_norm(comp, &format!("{prefix}.norm1"), &mut self.norm1)?;
        load_conv2d(comp, &format!("{prefix}.conv1"), &mut self.conv1)?;
        if let Some(p) = &mut self.time_emb_proj {
            load_linear(comp, &format!("{prefix}.time_emb_proj"), p)?;
        }
        load_group_norm(comp, &format!("{prefix}.norm2"), &mut self.norm2)?;
        load_conv2d(comp, &format!("{prefix}.conv2"), &mut self.conv2)?;
        if let Some(s) = &mut self.conv_shortcut {
            load_conv2d(comp, &format!("{prefix}.conv_shortcut"), s)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "metal")]
    #[test]
    fn resnet_forward_gpu_matches_cpu() {
        // Deterministic non-trivial weights so the test actually checks math.
        let in_c = 8; let out_c = 16;
        let mut r = ResnetBlock2D::new(in_c, out_c, Some(32), 4, 1e-5);
        for (i, w) in r.norm1.gamma.iter_mut().enumerate() { *w = 1.0 + (i as f32 * 0.01); }
        for (i, w) in r.norm1.beta.iter_mut().enumerate()  { *w = (i as f32 * 0.005) - 0.02; }
        for (i, w) in r.conv1.weight.iter_mut().enumerate() {
            *w = ((i * 17) % 23) as f32 / 23.0 - 0.5;
        }
        if let Some(b) = &mut r.conv1.bias {
            for (i, v) in b.iter_mut().enumerate() { *v = (i as f32 * 0.01) - 0.05; }
        }
        if let Some(p) = &mut r.time_emb_proj {
            for (i, w) in p.weight.iter_mut().enumerate() {
                *w = ((i * 13) % 19) as f32 / 19.0 - 0.5;
            }
            if let Some(b) = &mut p.bias {
                for (i, v) in b.iter_mut().enumerate() { *v = (i as f32 * 0.02) - 0.1; }
            }
        }
        for (i, w) in r.norm2.gamma.iter_mut().enumerate() { *w = 1.0 + (i as f32 * 0.005); }
        for (i, w) in r.norm2.beta.iter_mut().enumerate()  { *w = (i as f32 * 0.003) - 0.01; }
        for (i, w) in r.conv2.weight.iter_mut().enumerate() {
            *w = ((i * 11) % 17) as f32 / 17.0 - 0.5;
        }
        if let Some(b) = &mut r.conv2.bias {
            for (i, v) in b.iter_mut().enumerate() { *v = (i as f32 * 0.02) - 0.1; }
        }
        if let Some(sc) = &mut r.conv_shortcut {
            for (i, w) in sc.weight.iter_mut().enumerate() {
                *w = ((i * 7) % 11) as f32 / 11.0 - 0.5;
            }
            if let Some(b) = &mut sc.bias {
                for (i, v) in b.iter_mut().enumerate() { *v = (i as f32 * 0.03) - 0.15; }
            }
        }

        let n = 1; let h = 8; let w = 8;
        let input: Vec<f32> = (0..n*in_c*h*w)
            .map(|i| ((i * 5) % 13) as f32 / 13.0 - 0.5).collect();
        let time_emb: Vec<f32> = (0..n*32)
            .map(|i| ((i * 3) % 7) as f32 / 7.0 - 0.5).collect();

        // CPU reference.
        let out_cpu = r.forward(&input, n, h, w, &time_emb);

        // GPU-resident path.
        let in_gpu = crate::metal_backend::GpuTensor::upload_f32_as_f16(
            vec![n, in_c, h, w], &input);
        let out_gpu_t = r.forward_gpu(&in_gpu, n, h, w, &time_emb);
        let out_gpu = out_gpu_t.download_to_f32();

        assert_eq!(out_gpu.len(), out_cpu.len());
        let mut max_diff = 0.0f32;
        for (g, cpu) in out_gpu.iter().zip(out_cpu.iter()) {
            let d = (g - cpu).abs();
            if d > max_diff { max_diff = d; }
        }
        // Many fp16 ops chained — tolerance includes accumulated error.
        assert!(max_diff < 0.5,
            "ResnetBlock::forward_gpu diverges from CPU: max_diff={max_diff}");
    }

    #[test]
    fn resnet_runs_with_zero_weights() {
        // With all zero weights and bias, ResnetBlock2D acts as a residual:
        // output = shortcut(input) = input (when in==out).
        let r = ResnetBlock2D::new(8, 8, Some(16), 4, 1e-5);
        let input = vec![1.0f32; 1 * 8 * 4 * 4];
        let time_emb = vec![0.0f32; 16];
        let out = r.forward(&input, 1, 4, 4, &time_emb);
        // Output is shortcut + h2. shortcut(zero conv) = 0; h2(zero conv) = 0.
        // Wait — shortcut here is identity (in==out, no conv_shortcut). So output ≈ input.
        for v in &out { assert!((*v - 1.0).abs() < 1e-3, "got {v}"); }
    }
}
