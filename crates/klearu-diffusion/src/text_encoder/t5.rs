//! T5-XXL encoder for Flux text conditioning.
//!
//! Architecture (encoder-only, mirrors HuggingFace `T5EncoderModel`):
//!   - Embedding (shared input/output table; here we only need input):
//!       shared.weight  [vocab_size, d_model]
//!   - 24 encoder blocks:
//!       block.<i>.layer.0.{layer_norm,SelfAttention.{q,k,v,o}}
//!       block.<i>.layer.1.{layer_norm,DenseReluDense.{wi_0,wi_1,wo}}
//!     Block 0's SelfAttention also carries `relative_attention_bias.weight`
//!     [num_buckets, num_heads]; later layers reuse it via
//!     `share_relative_attention_bias`.
//!   - final_layer_norm
//!
//! Notable differences vs LLaMA/Qwen-style transformers:
//!   - **RMSNorm without offset**:  norm(x) = x / sqrt(mean(x²) + eps) · γ
//!     (no β, no centering — T5's "T5LayerNorm").
//!   - **Linear without bias** everywhere.
//!   - **Relative position bias** added to the attention logits, not to Q/K
//!     directly. Bucketed (32 buckets up to distance 128).
//!   - **Gated GELU FFN**:  gelu(W_i0 · x) ⊙ (W_i1 · x) → W_o, no bias.
//!     T5-XXL uses `feed_forward_proj = "gated-gelu"`. (Older T5 used ReLU.)
//!
//! T5-XXL parameters:
//!   d_model=4096, d_ff=10240, d_kv=64 (head_dim), num_heads=64,
//!   num_layers=24, vocab_size=32128, max_seq_len=512 (Flux pads to ≤512;
//!   Flux uses 256 for dev, 512 for some workflows).

use crate::error::{DiffusionError, Result};
use crate::weight::ComponentTensors;

/// T5-XXL encoder configuration.
#[derive(Debug, Clone)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub d_kv: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_eps: f32,
    pub max_seq_len: usize,
}

impl T5Config {
    pub fn t5_xxl() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_eps: 1e-6,
            max_seq_len: 512,
        }
    }
}

/// T5LayerNorm — RMSNorm without offset, no β.
pub struct T5LayerNorm {
    pub gamma: Vec<f32>, // [d_model]
    pub eps: f32,
}

impl T5LayerNorm {
    pub fn new(d: usize, eps: f32) -> Self {
        Self { gamma: vec![1.0; d], eps }
    }

    /// Apply in place: x ← x / sqrt(mean(x²) + eps) · γ
    pub fn forward_inplace(&self, x: &mut [f32], n_tokens: usize) {
        let d = self.gamma.len();
        debug_assert_eq!(x.len(), n_tokens * d);
        for t in 0..n_tokens {
            let row = &mut x[t * d..(t + 1) * d];
            // mean(x²)
            let mut s = 0.0_f64;
            for &v in row.iter() { s += (v as f64) * (v as f64); }
            let inv = ((s / d as f64) + self.eps as f64).sqrt().recip() as f32;
            for (xi, &g) in row.iter_mut().zip(self.gamma.iter()) {
                *xi = *xi * inv * g;
            }
        }
    }
}

/// Linear without bias.
pub struct LinearNoBias {
    pub weight: Vec<f32>, // [out, in]
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearNoBias {
    pub fn new(in_f: usize, out_f: usize) -> Self {
        Self { weight: vec![0.0; in_f * out_f], in_features: in_f, out_features: out_f }
    }

    /// out [n_tokens, out_features] = x [n_tokens, in_features] · W^T
    pub fn forward(&self, x: &[f32], n_tokens: usize, out: &mut [f32]) {
        let inf = self.in_features;
        let outf = self.out_features;
        debug_assert_eq!(x.len(), n_tokens * inf);
        debug_assert_eq!(out.len(), n_tokens * outf);
        crate::blas::sgemm_a_btrans(
            n_tokens, outf, inf,
            x, &self.weight, out,
        );
    }
}

/// Gated GELU FFN: out = (gelu(W_i0 · x) ⊙ (W_i1 · x)) · W_o
pub struct T5FF {
    pub layer_norm: T5LayerNorm,
    pub wi_0: LinearNoBias, // gate
    pub wi_1: LinearNoBias, // up
    pub wo: LinearNoBias,   // down
}

impl T5FF {
    pub fn new(cfg: &T5Config) -> Self {
        Self {
            layer_norm: T5LayerNorm::new(cfg.d_model, cfg.layer_norm_eps),
            wi_0: LinearNoBias::new(cfg.d_model, cfg.d_ff),
            wi_1: LinearNoBias::new(cfg.d_model, cfg.d_ff),
            wo: LinearNoBias::new(cfg.d_ff, cfg.d_model),
        }
    }

    pub fn forward_residual(&self, x: &mut [f32], n_tokens: usize) {
        let d = self.layer_norm.gamma.len();
        let dff = self.wi_0.out_features;
        let mut x_norm = x.to_vec();
        self.layer_norm.forward_inplace(&mut x_norm, n_tokens);
        let mut g = vec![0.0_f32; n_tokens * dff];
        let mut u = vec![0.0_f32; n_tokens * dff];
        self.wi_0.forward(&x_norm, n_tokens, &mut g);
        self.wi_1.forward(&x_norm, n_tokens, &mut u);
        // gelu(g) ⊙ u — T5 uses the exact (erf-based) GELU.
        for i in 0..g.len() {
            g[i] = gelu_exact(g[i]) * u[i];
        }
        let mut down = vec![0.0_f32; n_tokens * d];
        self.wo.forward(&g, n_tokens, &mut down);
        for i in 0..x.len() { x[i] += down[i]; }
    }
}

/// Self-attention with optional relative position bias. Block 0 has its
/// own bias table; later blocks share it (passed in from outside).
pub struct T5SelfAttn {
    pub layer_norm: T5LayerNorm,
    pub q: LinearNoBias,
    pub k: LinearNoBias,
    pub v: LinearNoBias,
    pub o: LinearNoBias,
    /// Only populated on block 0 of a T5 stack (`is_first_block`). Shape
    /// [num_buckets, num_heads]. Other blocks call `forward_residual`
    /// passing a borrow of block-0's table.
    pub relative_attention_bias: Option<Vec<f32>>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_buckets: usize,
    pub max_distance: usize,
}

impl T5SelfAttn {
    pub fn new(cfg: &T5Config, is_first_block: bool) -> Self {
        let inner = cfg.num_heads * cfg.d_kv;
        Self {
            layer_norm: T5LayerNorm::new(cfg.d_model, cfg.layer_norm_eps),
            q: LinearNoBias::new(cfg.d_model, inner),
            k: LinearNoBias::new(cfg.d_model, inner),
            v: LinearNoBias::new(cfg.d_model, inner),
            o: LinearNoBias::new(inner, cfg.d_model),
            relative_attention_bias: if is_first_block {
                Some(vec![0.0; cfg.relative_attention_num_buckets * cfg.num_heads])
            } else {
                None
            },
            num_heads: cfg.num_heads,
            head_dim: cfg.d_kv,
            num_buckets: cfg.relative_attention_num_buckets,
            max_distance: cfg.relative_attention_max_distance,
        }
    }

    /// Compute the relative-position bucket for one (i, j) pair.
    /// Mirrors HF `T5Attention._relative_position_bucket(bidirectional=True)`.
    fn rel_bucket(&self, i: i64, j: i64) -> usize {
        let mut relative_position = j - i;
        let num_buckets = self.num_buckets as i64;
        let mut ret: i64 = 0;
        // bidirectional → split first half across pos vs neg.
        let half = num_buckets / 2;
        if relative_position > 0 {
            ret += half;
        } else {
            relative_position = -relative_position;
        }
        // Now relative_position is in the range [0, +inf).
        let max_exact = half / 2;
        let is_small = relative_position < max_exact;
        let val_if_large = max_exact
            + (((relative_position as f32 / max_exact as f32).ln()
                / ((self.max_distance as f32 / max_exact as f32).ln()))
                * ((half - max_exact) as f32)) as i64;
        let val_if_large = val_if_large.min(half - 1);
        let bucket = if is_small { relative_position } else { val_if_large };
        ret += bucket;
        ret.clamp(0, num_buckets - 1) as usize
    }

    /// Compute the [num_heads, seq, seq] bias matrix. Cached across layers
    /// in `T5Encoder::forward`.
    pub fn compute_bias(&self, seq_len: usize) -> Vec<f32> {
        let table = self.relative_attention_bias.as_ref()
            .expect("compute_bias called on a non-first block");
        let mut out = vec![0.0_f32; self.num_heads * seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let bucket = self.rel_bucket(i as i64, j as i64);
                for h in 0..self.num_heads {
                    out[h * seq_len * seq_len + i * seq_len + j] =
                        table[bucket * self.num_heads + h];
                }
            }
        }
        out
    }

    /// Self-attention residual update.  `bias` is precomputed [num_heads,
    /// seq_len, seq_len] from `compute_bias`. `attn_mask` is None for
    /// causal-free encoder attention; pass Some(&[bool; seq_len]) where
    /// `false` means "padded" to mask out padding tokens (set to -inf).
    pub fn forward_residual(
        &self,
        x: &mut [f32],
        seq_len: usize,
        bias: &[f32],
        attn_mask: Option<&[bool]>,
    ) {
        let d = self.layer_norm.gamma.len();
        let inner = self.num_heads * self.head_dim;
        let h = self.num_heads;
        let dh = self.head_dim;

        let mut x_norm = x.to_vec();
        self.layer_norm.forward_inplace(&mut x_norm, seq_len);

        let mut q = vec![0.0_f32; seq_len * inner];
        let mut k = vec![0.0_f32; seq_len * inner];
        let mut v = vec![0.0_f32; seq_len * inner];
        self.q.forward(&x_norm, seq_len, &mut q);
        self.k.forward(&x_norm, seq_len, &mut k);
        self.v.forward(&x_norm, seq_len, &mut v);

        // T5 attention does NOT scale by 1/sqrt(head_dim) — the relative
        // bias absorbs that effect, by design.
        let mut out = vec![0.0_f32; seq_len * inner];
        for hi in 0..h {
            let bias_h = &bias[hi * seq_len * seq_len..(hi + 1) * seq_len * seq_len];
            for qi in 0..seq_len {
                // Compute scores[qi, kj] for all kj.
                let mut scores = vec![0.0_f32; seq_len];
                let q_off = qi * inner + hi * dh;
                for kj in 0..seq_len {
                    let k_off = kj * inner + hi * dh;
                    let mut s = 0.0_f32;
                    for d_i in 0..dh { s += q[q_off + d_i] * k[k_off + d_i]; }
                    s += bias_h[qi * seq_len + kj];
                    if let Some(mask) = attn_mask {
                        if !mask[kj] { s = f32::NEG_INFINITY; }
                    }
                    scores[kj] = s;
                }
                // softmax
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for s in scores.iter_mut() {
                    *s = if max == f32::NEG_INFINITY { 0.0 } else { (*s - max).exp() };
                    sum += *s;
                }
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for s in scores.iter_mut() { *s *= inv; }
                }
                let out_off = qi * inner + hi * dh;
                for kj in 0..seq_len {
                    let v_off = kj * inner + hi * dh;
                    let w = scores[kj];
                    if w == 0.0 { continue; }
                    for d_i in 0..dh { out[out_off + d_i] += w * v[v_off + d_i]; }
                }
            }
        }

        let mut o = vec![0.0_f32; seq_len * d];
        self.o.forward(&out, seq_len, &mut o);
        for i in 0..x.len() { x[i] += o[i]; }
    }
}

pub struct T5Block {
    pub attn: T5SelfAttn,
    pub ff: T5FF,
}

pub struct T5Encoder {
    pub config: T5Config,
    /// `shared.weight` — input token embedding [vocab_size, d_model].
    pub embed: Vec<f32>,
    pub blocks: Vec<T5Block>,
    pub final_layer_norm: T5LayerNorm,
}

impl T5Encoder {
    pub fn from_config(cfg: T5Config) -> Self {
        let blocks: Vec<T5Block> = (0..cfg.num_layers)
            .map(|i| T5Block {
                attn: T5SelfAttn::new(&cfg, i == 0),
                ff: T5FF::new(&cfg),
            })
            .collect();
        let final_layer_norm = T5LayerNorm::new(cfg.d_model, cfg.layer_norm_eps);
        let embed = vec![0.0; cfg.vocab_size * cfg.d_model];
        Self { config: cfg, embed, blocks, final_layer_norm }
    }

    /// Load weights from an HF-format component. Uses the standard
    /// `T5EncoderModel` naming (`shared.weight`, `encoder.block.<i>.layer.{0,1}.*`,
    /// `encoder.final_layer_norm.weight`).
    pub fn load_from(&mut self, comp: &ComponentTensors) -> Result<()> {
        let t0 = std::time::Instant::now();
        let cfg = self.config.clone();
        eprintln!("[flux]   t5 embedding…");
        // Embedding.
        self.embed = comp.get_f32("shared.weight")
            .or_else(|_| comp.get_f32("encoder.embed_tokens.weight"))?;
        if self.embed.len() != cfg.vocab_size * cfg.d_model {
            return Err(DiffusionError::ShapeMismatch {
                expected: format!("{} elems for [vocab, d_model]",
                    cfg.vocab_size * cfg.d_model),
                got: format!("{}", self.embed.len()),
            });
        }
        // Blocks.
        let n = self.blocks.len();
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            let p = format!("encoder.block.{i}");
            // Pre-print so user sees progress *before* the slow tensor reads.
            if (i + 1) % 4 == 0 || i + 1 == n || i == 0 {
                eprintln!("[flux]   t5 block {}/{n}  ({:.1}s elapsed)",
                    i + 1, t0.elapsed().as_secs_f32());
            }
            // layer.0 = SelfAttention
            blk.attn.layer_norm.gamma = comp.get_f32(&format!("{p}.layer.0.layer_norm.weight"))?;
            blk.attn.q.weight = comp.get_f32(&format!("{p}.layer.0.SelfAttention.q.weight"))?;
            blk.attn.k.weight = comp.get_f32(&format!("{p}.layer.0.SelfAttention.k.weight"))?;
            blk.attn.v.weight = comp.get_f32(&format!("{p}.layer.0.SelfAttention.v.weight"))?;
            blk.attn.o.weight = comp.get_f32(&format!("{p}.layer.0.SelfAttention.o.weight"))?;
            if i == 0 {
                let bias = comp.get_f32(&format!(
                    "{p}.layer.0.SelfAttention.relative_attention_bias.weight"
                ))?;
                blk.attn.relative_attention_bias = Some(bias);
            }
            // layer.1 = DenseReluDense (T5-XXL gated-gelu uses wi_0 + wi_1).
            blk.ff.layer_norm.gamma = comp.get_f32(&format!("{p}.layer.1.layer_norm.weight"))?;
            blk.ff.wi_0.weight = comp.get_f32(&format!("{p}.layer.1.DenseReluDense.wi_0.weight"))?;
            blk.ff.wi_1.weight = comp.get_f32(&format!("{p}.layer.1.DenseReluDense.wi_1.weight"))?;
            blk.ff.wo.weight = comp.get_f32(&format!("{p}.layer.1.DenseReluDense.wo.weight"))?;
        }
        self.final_layer_norm.gamma = comp.get_f32("encoder.final_layer_norm.weight")?;
        eprintln!("[flux]   t5 loaded in {:.1}s", t0.elapsed().as_secs_f32());
        Ok(())
    }

    /// Run the full encoder forward.  `tokens`: [seq_len] u32 token IDs.
    /// `attn_mask`: [seq_len] booleans (true = real, false = padding).
    /// Output: [seq_len, d_model].
    pub fn forward(&self, tokens: &[u32], attn_mask: Option<&[bool]>) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let seq_len = tokens.len();
        let d = self.config.d_model;
        let n = self.blocks.len();
        let progress = std::env::var("KLEARU_QUIET").is_err();

        // Embed.
        let mut x = vec![0.0_f32; seq_len * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let row = &self.embed[(tok as usize) * d..(tok as usize + 1) * d];
            x[t * d..(t + 1) * d].copy_from_slice(row);
        }
        if progress {
            eprintln!("[flux]   t5.forward: seq_len={seq_len}, d={d}, {n} layers");
        }

        // Compute relative position bias once (block 0 owns the table).
        let bias = self.blocks[0].attn.compute_bias(seq_len);

        // Run each block: self-attn (with bias) + FF.
        for (i, blk) in self.blocks.iter().enumerate() {
            blk.attn.forward_residual(&mut x, seq_len, &bias, attn_mask);
            blk.ff.forward_residual(&mut x, seq_len);
            if progress {
                eprintln!("[flux]   t5.forward: layer {}/{n}  ({:.1}s elapsed)",
                    i + 1, t0.elapsed().as_secs_f32());
            }
        }

        // Final layer norm.
        self.final_layer_norm.forward_inplace(&mut x, seq_len);
        if progress {
            eprintln!("[flux]   t5.forward: done in {:.1}s", t0.elapsed().as_secs_f32());
        }
        x
    }
}

/// Exact GELU using erf. T5 uses this (not the tanh approximation).
fn gelu_exact(x: f32) -> f32 {
    // 0.5 · x · (1 + erf(x / sqrt(2)))
    0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
}

/// erf approximation accurate to ~1.5e-7 (Abramowitz & Stegun 7.1.26).
fn erf_approx(x: f32) -> f32 {
    let a1 = 0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 = 1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 = 1.061405429_f32;
    let p = 0.3275911_f32;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> T5Config {
        T5Config {
            vocab_size: 16,
            d_model: 8,
            d_ff: 16,
            d_kv: 4,
            num_heads: 2,
            num_layers: 2,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            layer_norm_eps: 1e-6,
            max_seq_len: 32,
        }
    }

    #[test]
    fn rel_bucket_symmetric_around_origin() {
        let cfg = tiny_cfg();
        let attn = T5SelfAttn::new(&cfg, true);
        // Distance 0 → bucket 0.
        assert_eq!(attn.rel_bucket(5, 5), 0);
        // Positive vs negative offsets land in different halves.
        let pos1 = attn.rel_bucket(0, 1);
        let neg1 = attn.rel_bucket(1, 0);
        assert_ne!(pos1, neg1);
        assert!(pos1 < cfg.relative_attention_num_buckets);
        assert!(neg1 < cfg.relative_attention_num_buckets);
    }

    #[test]
    fn forward_runs_with_zero_weights() {
        let mut enc = T5Encoder::from_config(tiny_cfg());
        // Set embedding to small nonzero so we don't hit log(0) in any path.
        for v in enc.embed.iter_mut() { *v = 0.01; }
        // Make all gammas 1.0 so layer norms scale to ones.
        for blk in enc.blocks.iter_mut() {
            for v in blk.attn.layer_norm.gamma.iter_mut() { *v = 1.0; }
            for v in blk.ff.layer_norm.gamma.iter_mut() { *v = 1.0; }
        }
        for v in enc.final_layer_norm.gamma.iter_mut() { *v = 1.0; }
        let tokens = [1_u32, 2, 3, 4];
        let out = enc.forward(&tokens, None);
        assert_eq!(out.len(), 4 * 8);
        // No NaNs.
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
