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
/// Which GELU the gated feed-forward uses.
///
/// HuggingFace's `T5Config` rewrites `feed_forward_proj: "gated-gelu"` into
/// `dense_act_fn: "gelu_new"`, and `ACT2FN["gelu_new"]` is the **tanh**
/// approximation, not erf.
///
/// NOTE: [`T5Config::t5_xxl`] still selects [`GeluKind::Erf`], which is very
/// likely wrong for Flux's T5 as well — but changing it would perturb existing
/// Flux output, so it is left alone pending a reference comparison. New models
/// (umT5) use the correct `Tanh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluKind {
    /// Exact GELU via erf.
    Erf,
    /// `gelu_new` / `NewGELUActivation`: the tanh approximation.
    Tanh,
}

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
    /// umT5 gives **every** block its own `relative_attention_bias`; plain T5
    /// has one in block 0 that all later blocks share.
    pub relative_bias_per_layer: bool,
    pub dense_act: GeluKind,
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
            relative_bias_per_layer: false,
            dense_act: GeluKind::Erf,
        }
    }

    /// `google/umt5-xxl`, the Wan text encoder.
    ///
    /// Same block shape as T5-XXL but a much larger vocabulary, and — the
    /// defining difference — a `relative_attention_bias` table in every one of
    /// the 24 blocks rather than only the first.
    pub fn umt5_xxl() -> Self {
        Self {
            vocab_size: 256_384,
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_eps: 1e-6,
            max_seq_len: 512,
            relative_bias_per_layer: true,
            dense_act: GeluKind::Tanh,
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
    pub act: GeluKind,
}

impl T5FF {
    pub fn new(cfg: &T5Config) -> Self {
        Self {
            layer_norm: T5LayerNorm::new(cfg.d_model, cfg.layer_norm_eps),
            wi_0: LinearNoBias::new(cfg.d_model, cfg.d_ff),
            wi_1: LinearNoBias::new(cfg.d_model, cfg.d_ff),
            wo: LinearNoBias::new(cfg.d_ff, cfg.d_model),
            act: cfg.dense_act,
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
            g[i] = match self.act {
                GeluKind::Erf => gelu_exact(g[i]),
                GeluKind::Tanh => gelu_tanh(g[i]),
            } * u[i];
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
                attn: T5SelfAttn::new(&cfg, i == 0 || cfg.relative_bias_per_layer),
                ff: T5FF::new(&cfg),
            })
            .collect();
        let final_layer_norm = T5LayerNorm::new(cfg.d_model, cfg.layer_norm_eps);
        let embed = vec![0.0; cfg.vocab_size * cfg.d_model];
        Self { config: cfg, embed, blocks, final_layer_norm }
    }

    /// Every tensor name this encoder will ask for, in load order.
    ///
    /// Useful for validating a checkpoint up front, and as a structural
    /// assertion: `google/umt5-xxl` ships exactly 242 tensors, which is 24
    /// blocks x 10 plus `shared.weight` and `encoder.final_layer_norm.weight`.
    /// Plain T5-XXL has 219 — the 23 fewer are the per-block
    /// `relative_attention_bias` tables it does not have.
    pub fn tensor_names(&self) -> Vec<String> {
        let cfg = &self.config;
        let mut out = vec!["shared.weight".to_string()];
        for i in 0..cfg.num_layers {
            let p = format!("encoder.block.{i}");
            out.push(format!("{p}.layer.0.layer_norm.weight"));
            for n in ["q", "k", "v", "o"] {
                out.push(format!("{p}.layer.0.SelfAttention.{n}.weight"));
            }
            if i == 0 || cfg.relative_bias_per_layer {
                out.push(format!("{p}.layer.0.SelfAttention.relative_attention_bias.weight"));
            }
            out.push(format!("{p}.layer.1.layer_norm.weight"));
            for n in ["wi_0", "wi_1", "wo"] {
                out.push(format!("{p}.layer.1.DenseReluDense.{n}.weight"));
            }
        }
        out.push("encoder.final_layer_norm.weight".to_string());
        out
    }

    /// Load weights from an HF-format component. Uses the standard
    /// `T5EncoderModel` naming (`shared.weight`, `encoder.block.<i>.layer.{0,1}.*`,
    /// `encoder.final_layer_norm.weight`).
    pub fn load_from(&mut self, comp: &ComponentTensors) -> Result<()> {
        let t0 = std::time::Instant::now();
        let cfg = self.config.clone();
        eprintln!("[t5] embedding…");
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
                eprintln!("[t5] block {}/{n}  ({:.1}s elapsed)",
                    i + 1, t0.elapsed().as_secs_f32());
            }
            // layer.0 = SelfAttention
            blk.attn.layer_norm.gamma = comp.get_f32(&format!("{p}.layer.0.layer_norm.weight"))?;
            blk.attn.q.weight = comp.get_f32(&format!("{p}.layer.0.SelfAttention.q.weight"))?;
            blk.attn.k.weight = comp.get_f32(&format!("{p}.layer.0.SelfAttention.k.weight"))?;
            blk.attn.v.weight = comp.get_f32(&format!("{p}.layer.0.SelfAttention.v.weight"))?;
            blk.attn.o.weight = comp.get_f32(&format!("{p}.layer.0.SelfAttention.o.weight"))?;
            if i == 0 || cfg.relative_bias_per_layer {
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
        eprintln!("[t5] loaded in {:.1}s", t0.elapsed().as_secs_f32());
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
            eprintln!("[t5] forward: seq_len={seq_len}, d={d}, {n} layers");
        }

        // Plain T5 computes the bias once from block 0 and shares it; umT5
        // has a table per block, so it must be recomputed each layer.
        let shared_bias = if self.config.relative_bias_per_layer {
            None
        } else {
            Some(self.blocks[0].attn.compute_bias(seq_len))
        };

        // Run each block: self-attn (with bias) + FF.
        for (i, blk) in self.blocks.iter().enumerate() {
            let per_layer;
            let bias = match &shared_bias {
                Some(b) => b,
                None => {
                    per_layer = blk.attn.compute_bias(seq_len);
                    &per_layer
                }
            };
            blk.attn.forward_residual(&mut x, seq_len, bias, attn_mask);
            blk.ff.forward_residual(&mut x, seq_len);
            if progress {
                eprintln!("[t5] forward: layer {}/{n}  ({:.1}s elapsed)",
                    i + 1, t0.elapsed().as_secs_f32());
            }
        }

        // Final layer norm.
        self.final_layer_norm.forward_inplace(&mut x, seq_len);
        if progress {
            eprintln!("[t5] forward: done in {:.1}s", t0.elapsed().as_secs_f32());
        }
        x
    }
}

/// `gelu_new` / `NewGELUActivation`: the tanh approximation. This is what
/// HuggingFace actually uses for `feed_forward_proj: "gated-gelu"`.
fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_885; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// Exact GELU using erf.
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
            relative_bias_per_layer: false,
            dense_act: GeluKind::Erf,
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
    // -- umT5 -------------------------------------------------------------

    #[test]
    fn umt5_config_differs_from_t5_where_it_should() {
        let u = T5Config::umt5_xxl();
        let t = T5Config::t5_xxl();
        // Same block geometry...
        assert_eq!((u.d_model, u.d_ff, u.d_kv, u.num_heads, u.num_layers), (4096, 10240, 64, 64, 24));
        assert_eq!((u.d_model, u.d_ff, u.num_layers), (t.d_model, t.d_ff, t.num_layers));
        // ...but a far larger vocabulary, a bias table per block, and the
        // tanh GELU that `gated-gelu` actually maps to.
        assert_eq!(u.vocab_size, 256_384);
        assert!(u.relative_bias_per_layer);
        assert!(!t.relative_bias_per_layer);
        assert_eq!(u.dense_act, GeluKind::Tanh);
    }

    #[test]
    fn umt5_tensor_inventory_matches_the_real_checkpoint() {
        // google/umt5-xxl ships 242 tensors (from its safetensors index).
        let enc = T5Encoder::from_config(T5Config::umt5_xxl());
        let names = enc.tensor_names();
        assert_eq!(names.len(), 242);
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "duplicate tensor name");

        // A bias table in every block is the umT5 signature.
        for i in 0..24 {
            let n = format!("encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight");
            assert!(names.contains(&n), "missing per-block bias at {i}");
        }

        // Plain T5-XXL has one, in block 0 only — 23 tensors fewer.
        let t5 = T5Encoder::from_config(T5Config::t5_xxl());
        let t5_names = t5.tensor_names();
        assert_eq!(t5_names.len(), 242 - 23);
        assert!(t5_names
            .iter()
            .filter(|n| n.contains("relative_attention_bias"))
            .count()
            == 1);
    }

    #[test]
    fn umt5_allocates_a_bias_table_per_block() {
        let enc = T5Encoder::from_config(T5Config::umt5_xxl());
        assert!(
            enc.blocks.iter().all(|b| b.attn.relative_attention_bias.is_some()),
            "every umT5 block needs its own relative_attention_bias"
        );
        let t5 = T5Encoder::from_config(T5Config::t5_xxl());
        assert!(t5.blocks[0].attn.relative_attention_bias.is_some());
        assert!(
            t5.blocks[1..].iter().all(|b| b.attn.relative_attention_bias.is_none()),
            "plain T5 shares block 0's table"
        );
    }

    #[test]
    fn umt5_uses_each_blocks_own_bias() {
        // Give block 1 a distinctive bias table and check it changes the
        // output. Under T5's shared-bias path block 1's table is never read,
        // so this is what separates the two behaviours.
        let mut cfg = tiny_cfg();
        cfg.relative_bias_per_layer = true;
        let mut enc = T5Encoder::from_config(cfg.clone());
        for (i, b) in enc.blocks.iter_mut().enumerate() {
            b.attn.q.weight.iter_mut().enumerate().for_each(|(k, w)| {
                *w = ((k + i) as f32 * 0.3).sin() * 0.2
            });
            b.attn.v.weight.iter_mut().enumerate().for_each(|(k, w)| {
                *w = ((k + i) as f32 * 0.7).cos() * 0.2
            });
            b.attn.o.weight.iter_mut().for_each(|w| *w = 0.05);
        }
        for (i, e) in enc.embed.iter_mut().enumerate() {
            *e = (i as f32 * 0.11).sin();
        }
        let tokens = [1u32, 2, 3, 4];
        let before = enc.forward(&tokens, None);

        let table = enc.blocks[1].attn.relative_attention_bias.as_mut().unwrap();
        for (k, v) in table.iter_mut().enumerate() {
            *v = (k as f32) * 0.5;
        }
        let after = enc.forward(&tokens, None);

        let worst = before.iter().zip(&after).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(worst > 1e-4, "block 1's own bias table was ignored: {worst}");
    }

    #[test]
    fn the_two_gelus_are_close_but_not_equal() {
        // `gated-gelu` maps to `gelu_new` (tanh) in HuggingFace, not erf.
        // The gap is small, which is exactly why using the wrong one is easy
        // to miss — see the note on `GeluKind`.
        let mut worst = 0.0_f32;
        for i in -400..400 {
            let x = i as f32 * 0.01;
            worst = worst.max((gelu_exact(x) - gelu_tanh(x)).abs());
        }
        assert!(worst > 1e-5, "the two should genuinely differ: {worst}");
        assert!(worst < 2e-3, "but only slightly: {worst}");
    }
    /// Load the **real** 11 GB umT5-XXL encoder and run it.
    ///
    /// Validates against the shipped checkpoint what the config alone cannot:
    /// that the 242-tensor inventory is right, that the per-block
    /// `relative_attention_bias` really is present in all 24 blocks (the
    /// difference from plain T5), and that a bf16 checkpoint decodes.
    ///
    ///   KLEARU_UMT5=/abs/path/to/text_encoder \
    ///   cargo test -p klearu-diffusion --lib --release \
    ///     -- --ignored --nocapture real_umt5
    #[test]
    #[ignore]
    fn real_umt5_loads_and_encodes() {
        use crate::weight::ComponentTensors;

        let Ok(dir) = std::env::var("KLEARU_UMT5") else {
            eprintln!("set KLEARU_UMT5 to the text_encoder/ directory; skipping");
            return;
        };
        let dir = std::path::PathBuf::from(dir);

        // The shipped config must agree with `umt5_xxl()`, which is hardcoded.
        let raw = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
        let j: serde_json::Value = serde_json::from_str(&raw).expect("parse config.json");
        let cfg = T5Config::umt5_xxl();
        let get = |k: &str| j[k].as_u64().unwrap_or_else(|| panic!("missing {k}")) as usize;
        assert_eq!(get("d_model"), cfg.d_model);
        assert_eq!(get("d_ff"), cfg.d_ff);
        assert_eq!(get("d_kv"), cfg.d_kv);
        assert_eq!(get("num_heads"), cfg.num_heads);
        assert_eq!(get("num_layers"), cfg.num_layers);
        assert_eq!(get("vocab_size"), cfg.vocab_size);
        assert_eq!(get("relative_attention_num_buckets"), cfg.relative_attention_num_buckets);
        assert_eq!(j["model_type"].as_str(), Some("umt5"));
        assert_eq!(j["feed_forward_proj"].as_str(), Some("gated-gelu"));
        println!(
            "config: umt5 {} layers, d_model {}, d_ff {}, {} heads x {}, vocab {}",
            cfg.num_layers, cfg.d_model, cfg.d_ff, cfg.num_heads, cfg.d_kv, cfg.vocab_size
        );

        let t0 = std::time::Instant::now();
        let comp = ComponentTensors::open_dir(&dir).expect("open checkpoint");
        println!("checkpoint holds {} tensors", comp.tensors.len());

        let mut enc = T5Encoder::from_config(cfg.clone());
        let names = enc.tensor_names();
        println!("model requests {} tensors", names.len());
        let missing: Vec<_> = names.iter().filter(|n| !comp.has(n)).cloned().collect();
        assert!(
            missing.is_empty(),
            "missing {}: {:?}",
            missing.len(),
            &missing[..missing.len().min(8)]
        );
        // The umT5 signature: a bias table in every block, not just block 0.
        for i in 0..cfg.num_layers {
            let n = format!(
                "encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"
            );
            assert!(comp.has(&n), "checkpoint lacks per-block bias at {i}");
        }
        println!("inventory matches, per-block bias present in all {} blocks", cfg.num_layers);

        enc.load_from(&comp).expect("load real weights");
        println!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

        // Arbitrary in-vocabulary ids: real text needs the SentencePiece
        // tokenizer, which is a separate piece.
        let tokens: Vec<u32> = vec![259, 1001, 2048, 77, 3, 512, 9, 1];
        let t1 = std::time::Instant::now();
        let out = enc.forward(&tokens, None);
        println!("encoded {} tokens in {:.1}s", tokens.len(), t1.elapsed().as_secs_f32());

        assert_eq!(out.len(), tokens.len() * cfg.d_model);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite hidden states");
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        let var = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / out.len() as f32;
        println!("hidden states: mean={mean:.4} std={:.4}", var.sqrt());
        assert!(var.sqrt() > 1e-3, "hidden states are nearly constant");

        // Different tokens must give different states — otherwise the
        // embedding table or the bias wiring is wrong.
        let other: Vec<u32> = vec![7, 42, 900, 31337, 5, 6, 8, 1];
        let out2 = enc.forward(&other, None);
        let diff = out
            .iter()
            .zip(&out2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("max |delta| between two token sequences: {diff:.4}");
        assert!(diff > 1e-2, "encoder output does not depend on the tokens");
    }
    /// Compare klearu's umT5 against transformers on the **real 11 GB
    /// checkpoint** — the last Wan component never validated numerically.
    ///
    /// Bad text conditioning would hand the DiT garbage to cross-attend to,
    /// which is exactly what an incoherent generation looks like, so this
    /// closes the remaining gap.
    ///
    ///   <python> scripts/dump_umt5_reference.py \
    ///       --model models/wan22-ti2v-5b/text_encoder --out <dir>/umt5_ref.safetensors
    ///   KLEARU_UMT5=/abs/text_encoder KLEARU_UMT5_REF=<dir>/umt5_ref.safetensors \
    ///   cargo test -p klearu-diffusion --lib --release \
    ///     -- --ignored --nocapture umt5_matches_transformers
    #[test]
    #[ignore]
    fn umt5_matches_transformers() {
        use crate::weight::ComponentTensors;

        let (Ok(dir), Ok(ref_path)) =
            (std::env::var("KLEARU_UMT5"), std::env::var("KLEARU_UMT5_REF"))
        else {
            eprintln!("set KLEARU_UMT5 and KLEARU_UMT5_REF; skipping");
            return;
        };
        let ref_dir = std::path::Path::new(&ref_path).parent().unwrap().to_path_buf();
        let r = ComponentTensors::open_dir(&ref_dir).expect("open reference");

        let mut enc = T5Encoder::from_config(T5Config::umt5_xxl());
        enc.load_from(&ComponentTensors::open_dir(std::path::Path::new(&dir)).unwrap())
            .expect("load real umT5");

        let compare = |label: &str, ids_key: &str, out_key: &str, n_real: usize| -> f64 {
            let ids: Vec<u32> =
                r.get_f32(ids_key).unwrap().iter().map(|v| *v as u32).collect();
            let want = r.get_f32(out_key).unwrap();
            let mask: Vec<bool> = (0..ids.len()).map(|i| i < n_real).collect();
            let got = enc.forward(&ids, Some(&mask));
            assert_eq!(got.len(), want.len(), "{label}: length");

            // Only the real positions are meaningful: what a masked-out
            // position produces is unconstrained, and the pipeline zeroes it.
            let d = enc.config.d_model;
            let (g, w) = (&got[..n_real * d], &want[..n_real * d]);
            let worst = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            let dot: f64 = g.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum();
            let na: f64 = g.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = w.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (na * nb + 1e-12);
            println!("{label:10} max_abs_diff={worst:.4e}  cos_sim={cos:.8}");
            cos
        };

        let c1 = compare("nopad", "in.ids_nopad", "out.nopad", 8);
        let c2 = compare("padded", "in.ids_padded", "out.padded", 8);

        assert!(c1 > 0.9999, "umT5 diverged from transformers (nopad): {c1}");
        assert!(c2 > 0.9999, "umT5 diverged from transformers (padded): {c2}");
    }
}
