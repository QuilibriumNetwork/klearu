//! Building-block layers for the HiDream-O1 Qwen3-VL decoder.
//!
//! This is a CPU reference implementation (matmuls via `crate::blas`). It
//! mirrors the math of `models/qwen3_vl_transformers.py` exactly so we can
//! validate layer-by-layer against a Python activation trace (cosine
//! similarity) before swapping the hot matmuls/attention onto the Metal
//! backend. Correctness first, speed second — the same workflow that nailed
//! the SD 1.5 port.

use crate::weight::loader::ComponentTensors;
use crate::Result;

/// Dense linear layer, weight stored row-major `[out, in]` (HF convention).
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self {
            weight: vec![0.0; in_features * out_features],
            bias: if bias { Some(vec![0.0; out_features]) } else { None },
            in_features,
            out_features,
        }
    }

    /// `out[n, out_f] = x[n, in_f] · weightᵀ` (+ bias). weight is `[out, in]`
    /// row-major, exactly what `sgemm_a_btrans` consumes.
    pub fn forward(&self, x: &[f32], n_tokens: usize, out: &mut [f32]) {
        debug_assert_eq!(x.len(), n_tokens * self.in_features);
        debug_assert_eq!(out.len(), n_tokens * self.out_features);
        crate::blas::sgemm_a_btrans(n_tokens, self.out_features, self.in_features, x, &self.weight, out);
        if let Some(b) = &self.bias {
            use rayon::prelude::*;
            out.par_chunks_mut(self.out_features).for_each(|row| {
                for (r, bv) in row.iter_mut().zip(b.iter()) {
                    *r += bv;
                }
            });
        }
    }

    pub fn forward_vec(&self, x: &[f32], n_tokens: usize) -> Vec<f32> {
        let mut out = vec![0.0; n_tokens * self.out_features];
        self.forward(x, n_tokens, &mut out);
        out
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        self.weight = comp.get_f32(&format!("{prefix}.weight"))?;
        if self.bias.is_some() {
            self.bias = Some(comp.get_f32(&format!("{prefix}.bias"))?);
        }
        Ok(())
    }
}

/// Plain RMSNorm: `x / sqrt(mean(x²) + eps) · weight`. (Qwen3-VL uses the
/// standard form — NOT the `(1 + weight)` variant that Qwen3.5 uses.)
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { weight: vec![1.0; dim], eps }
    }

    /// Normalize each row of length `dim` over the last axis.
    pub fn apply(&self, x: &mut [f32], n_tokens: usize) {
        let dim = self.weight.len();
        debug_assert_eq!(x.len(), n_tokens * dim);
        for t in 0..n_tokens {
            let row = &mut x[t * dim..(t + 1) * dim];
            let mut s = 0.0_f64;
            for &v in row.iter() {
                s += (v as f64) * (v as f64);
            }
            let inv = ((s / dim as f64) + self.eps as f64).sqrt().recip() as f32;
            for (xi, &g) in row.iter_mut().zip(self.weight.iter()) {
                *xi = *xi * inv * g;
            }
        }
    }

    /// Per-head RMSNorm over `head_dim` (Qwen3-VL q_norm/k_norm). `x` is laid
    /// out `[n_tokens, num_heads, head_dim]`; the scale is shared `[head_dim]`.
    pub fn apply_per_head(&self, x: &mut [f32], n_tokens: usize, num_heads: usize) {
        let dh = self.weight.len();
        debug_assert_eq!(x.len(), n_tokens * num_heads * dh);
        for i in 0..n_tokens * num_heads {
            let off = i * dh;
            let row = &mut x[off..off + dh];
            let mut s = 0.0_f64;
            for &v in row.iter() {
                s += (v as f64) * (v as f64);
            }
            let inv = ((s / dh as f64) + self.eps as f64).sqrt().recip() as f32;
            for (xi, &g) in row.iter_mut().zip(self.weight.iter()) {
                *xi = *xi * inv * g;
            }
        }
    }

    pub fn load(&mut self, comp: &ComponentTensors, name: &str) -> Result<()> {
        self.weight = comp.get_f32(name)?;
        Ok(())
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Exact GELU via erf (matches PyTorch `nn.GELU()` — used by the vision
/// mergers).
#[inline]
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// `gelu_pytorch_tanh` — the tanh approximation (used by the vision MLP).
#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_56; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

fn erf(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26.
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    if x < 0.0 {
        -y
    } else {
        y
    }
}

/// Affine LayerNorm over the last `dim`: `(x-μ)/sqrt(σ²+eps)·weight + bias`.
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { weight: vec![1.0; dim], bias: vec![0.0; dim], eps }
    }

    pub fn apply(&self, x: &mut [f32], n_rows: usize) {
        let dim = self.weight.len();
        debug_assert_eq!(x.len(), n_rows * dim);
        for r in 0..n_rows {
            let row = &mut x[r * dim..(r + 1) * dim];
            let mut mean = 0.0_f64;
            for &v in row.iter() {
                mean += v as f64;
            }
            mean /= dim as f64;
            let mut var = 0.0_f64;
            for &v in row.iter() {
                let d = v as f64 - mean;
                var += d * d;
            }
            var /= dim as f64;
            let inv = ((var + self.eps as f64).sqrt().recip()) as f32;
            let m = mean as f32;
            for ((xi, &g), &b) in row.iter_mut().zip(self.weight.iter()).zip(self.bias.iter()) {
                *xi = (*xi - m) * inv * g + b;
            }
        }
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        self.weight = comp.get_f32(&format!("{prefix}.weight"))?;
        self.bias = comp.get_f32(&format!("{prefix}.bias"))?;
        Ok(())
    }
}

// ============================================================================
// Generation-specific modules (bolted onto the Qwen3-VL backbone)
// ============================================================================

/// `BottleneckPatchEmbed` (the reference `x_embedder`): embeds a flattened
/// `PATCH_DIM` (=3072) image patch into the model hidden size through a
/// bottleneck. `proj2(proj1(x))`, proj1 has no bias.
pub struct BottleneckPatchEmbed {
    pub proj1: Linear, // PATCH_DIM -> hidden/4, no bias
    pub proj2: Linear, // hidden/4 -> hidden, bias
}

impl BottleneckPatchEmbed {
    pub fn new(patch_dim: usize, bottleneck: usize, hidden: usize) -> Self {
        Self {
            proj1: Linear::new(patch_dim, bottleneck, false),
            proj2: Linear::new(bottleneck, hidden, true),
        }
    }

    pub fn forward(&self, x: &[f32], n_tokens: usize) -> Vec<f32> {
        let mid = self.proj1.forward_vec(x, n_tokens);
        self.proj2.forward_vec(&mid, n_tokens)
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        self.proj1.load(comp, &format!("{prefix}.proj1"))?;
        self.proj2.load(comp, &format!("{prefix}.proj2"))?;
        Ok(())
    }
}

/// Sinusoidal timestep embedding (GLIDE/DiT form). `t` may be fractional.
/// Produces `dim` values: `[cos(t·freqs), sin(t·freqs)]`.
pub fn timestep_embedding(t: f32, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0_f32; dim];
    let log_mp = max_period.ln();
    for i in 0..half {
        let freq = (-log_mp * (i as f32) / (half as f32)).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

/// `TimestepEmbedder` (the reference `t_embedder1`): sinusoidal(freq_size) →
/// Linear → SiLU → Linear. NOTE the reference multiplies the input by 1000
/// before the sinusoid (`timestep_embedding(t * 1000, ...)`).
pub struct TimestepEmbedder {
    pub lin1: Linear, // freq_size -> hidden
    pub lin2: Linear, // hidden -> hidden
    pub freq_size: usize,
}

impl TimestepEmbedder {
    pub fn new(freq_size: usize, hidden: usize) -> Self {
        Self {
            lin1: Linear::new(freq_size, hidden, true),
            lin2: Linear::new(hidden, hidden, true),
            freq_size,
        }
    }

    /// Embed one scalar timestep into a `hidden`-dim vector.
    pub fn forward(&self, t: f32) -> Vec<f32> {
        let freq = timestep_embedding(t * 1000.0, self.freq_size, 10000.0);
        let mut h = self.lin1.forward_vec(&freq, 1);
        for v in h.iter_mut() {
            *v = silu(*v);
        }
        self.lin2.forward_vec(&h, 1)
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        // Reference: nn.Sequential(Linear, SiLU, Linear) → mlp.0 / mlp.2.
        self.lin1.load(comp, &format!("{prefix}.mlp.0"))?;
        self.lin2.load(comp, &format!("{prefix}.mlp.2"))?;
        Ok(())
    }
}

/// `FinalLayer` (the reference `final_layer2`): hidden → `PATCH_DIM` linear
/// (zero-init at training start) that unembeds a hidden vector back to a raw
/// pixel patch.
pub struct FinalLayer {
    pub linear: Linear,
}

impl FinalLayer {
    pub fn new(hidden: usize, patch_dim: usize) -> Self {
        Self { linear: Linear::new(hidden, patch_dim, true) }
    }

    pub fn forward(&self, x: &[f32], n_tokens: usize) -> Vec<f32> {
        self.linear.forward_vec(x, n_tokens)
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        self.linear.load(comp, &format!("{prefix}.linear"))
    }
}
