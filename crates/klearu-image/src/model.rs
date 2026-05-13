//! Image transformer: decoder-only autoregressive model over a mixed
//! sequence of text BPE tokens (prefix) followed by image VQ tokens.
//!
//! Phase 1 design: dense baseline, no sparsity yet. The point is to have
//! a known-good architecture we can swap pieces into as we sparsify in
//! Phase 2+. Token mixing across modalities is via a unified embedding
//! lookup: text tokens occupy ids `[0, vocab_text)`, image tokens occupy
//! ids `[vocab_text, vocab_text + vocab_image)`. The output head only
//! produces logits over the image-token range — we never predict text.
//!
//! Position encoding (Phase 1 simplification): single learned absolute
//! position embedding table covering the entire sequence. RoPE / 2D
//! grid embeddings are Phase 2+ refinements.
//!
//! Sequence layout per training sample:
//!   [BOS] text_tokens... [SEP_IMAGE] img_tokens... [EOS]
//! Cross-entropy is computed on the image-token positions only.

use crate::error::{ImageGenError, Result};

/// Hyperparameters for the image transformer.
#[derive(Debug, Clone)]
pub struct ImageTransformerConfig {
    /// Text BPE vocabulary size (e.g., 32000 for SmolLM tokenizer).
    pub vocab_text: usize,
    /// Image VQ vocabulary size (e.g., 8192 for MUSE / RQ-VAE tokenizers).
    pub vocab_image: usize,
    /// Hidden dim (model width). 512 for the 50M-param baseline.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads (head_dim = hidden_size / num_heads).
    pub num_heads: usize,
    /// SwiGLU MLP intermediate size. Typically 4 * hidden_size, but a
    /// 2.67× multiplier (matching LLaMA family) is a tighter param fit
    /// when params are tight.
    pub mlp_intermediate: usize,
    /// Max text-token prefix length.
    pub max_text_len: usize,
    /// Image token grid: width and height.
    pub image_grid_h: usize,
    pub image_grid_w: usize,
    /// Special token ids in the unified vocab.
    pub bos_token: u32,
    pub sep_image_token: u32,
    pub eos_token: u32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base for text-position encoding.
    pub rope_theta: f32,
}

impl ImageTransformerConfig {
    /// Default 50M-param baseline (12 layers × 512 hidden × 8 heads).
    /// Image tokens at [16, 16] = 256 per image. Text prefix max 64 tokens.
    pub fn baseline_50m() -> Self {
        let vocab_text = 32_000;
        let vocab_image = 8192;
        // Reserve three slots after text vocab for BOS / SEP_IMAGE / EOS.
        let bos = vocab_text as u32;
        let sep = vocab_text as u32 + 1;
        let eos = vocab_text as u32 + 2;
        Self {
            vocab_text,
            vocab_image,
            hidden_size: 512,
            num_layers: 12,
            num_heads: 8,
            mlp_intermediate: 1408, // 2.67× LLaMA convention
            max_text_len: 64,
            image_grid_h: 16,
            image_grid_w: 16,
            bos_token: bos,
            sep_image_token: sep,
            eos_token: eos,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    /// Total unified-vocab size. Includes 3 special tokens (BOS, SEP, EOS)
    /// inserted after the text-vocab range, then the image-token range.
    pub fn unified_vocab_size(&self) -> usize {
        self.vocab_text + 3 + self.vocab_image
    }

    /// Start of the image-token range in the unified vocab. Image VQ
    /// codeword `c` maps to unified id `image_id_offset() + c`.
    pub fn image_id_offset(&self) -> u32 {
        (self.vocab_text + 3) as u32
    }

    /// Max sequence length the model sees: text prefix + image tokens +
    /// the three special tokens.
    pub fn max_seq_len(&self) -> usize {
        self.max_text_len + self.image_grid_h * self.image_grid_w + 3
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

/// Linear without bias — standard for modern decoder transformers.
pub struct LinearNoBias {
    pub weight: Vec<f32>, // [out, in]
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearNoBias {
    pub fn new(in_f: usize, out_f: usize) -> Self {
        Self {
            weight: vec![0.0; in_f * out_f],
            in_features: in_f,
            out_features: out_f,
        }
    }
}

/// RMSNorm: y = x / sqrt(mean(x²) + eps) · γ
pub struct RmsNorm {
    pub gamma: Vec<f32>, // [dim]
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { gamma: vec![1.0; dim], eps }
    }
}

/// One transformer block: pre-norm self-attention + SwiGLU MLP, with
/// residual connections.
pub struct ImageBlock {
    pub norm_attn: RmsNorm,
    pub q_proj: LinearNoBias,
    pub k_proj: LinearNoBias,
    pub v_proj: LinearNoBias,
    pub o_proj: LinearNoBias,
    pub norm_mlp: RmsNorm,
    /// SwiGLU has two parallel input projections (gate + up) and one
    /// output projection (down).
    pub mlp_gate: LinearNoBias,
    pub mlp_up: LinearNoBias,
    pub mlp_down: LinearNoBias,
}

impl ImageBlock {
    pub fn new(cfg: &ImageTransformerConfig) -> Self {
        let d = cfg.hidden_size;
        let m = cfg.mlp_intermediate;
        Self {
            norm_attn: RmsNorm::new(d, cfg.rms_norm_eps),
            q_proj: LinearNoBias::new(d, d),
            k_proj: LinearNoBias::new(d, d),
            v_proj: LinearNoBias::new(d, d),
            o_proj: LinearNoBias::new(d, d),
            norm_mlp: RmsNorm::new(d, cfg.rms_norm_eps),
            mlp_gate: LinearNoBias::new(d, m),
            mlp_up: LinearNoBias::new(d, m),
            mlp_down: LinearNoBias::new(m, d),
        }
    }
}

/// The full image transformer model.
pub struct ImageTransformer {
    pub config: ImageTransformerConfig,
    /// Unified token embedding table. Rows 0..vocab_text = text BPE,
    /// vocab_text..vocab_text+3 = BOS/SEP/EOS, then vocab_image rows for
    /// VQ codewords.
    pub embed: Vec<f32>, // [unified_vocab_size, hidden_size]
    /// Learned absolute position embeddings for the full sequence.
    /// Shape: [max_seq_len, hidden_size]. Added to token embeddings at
    /// every position before the first transformer block.
    pub pos_embed: Vec<f32>,
    pub blocks: Vec<ImageBlock>,
    pub final_norm: RmsNorm,
    /// Output projection. Produces logits over the image vocab only
    /// (vocab_image entries) — we never predict text. Tied to the
    /// image-portion of the embedding when `tie_lm_head = true`
    /// (default at init; weights overwritten by `load_from`).
    pub lm_head: LinearNoBias,
}

impl ImageTransformer {
    pub fn from_config(config: ImageTransformerConfig) -> Self {
        let unified = config.unified_vocab_size();
        let d = config.hidden_size;
        let blocks: Vec<_> = (0..config.num_layers)
            .map(|_| ImageBlock::new(&config))
            .collect();
        let max_seq = config.max_seq_len();
        let lm_head = LinearNoBias::new(d, config.vocab_image);
        Self {
            embed: vec![0.0; unified * d],
            pos_embed: vec![0.0; max_seq * d],
            blocks,
            final_norm: RmsNorm::new(d, 1e-5),
            lm_head,
            config,
        }
    }

    /// Param count, useful for budgeting.
    pub fn param_count(&self) -> usize {
        let cfg = &self.config;
        let d = cfg.hidden_size;
        let m = cfg.mlp_intermediate;
        let block_params = 4 * d * d        // q, k, v, o
            + 3 * d * m                     // gate, up, down
            + 2 * d;                        // two RMSNorms
        let embed = cfg.unified_vocab_size() * d;
        let pos_embed = cfg.max_seq_len() * d;
        let lm_head = d * cfg.vocab_image;
        let final_norm = d;
        embed + pos_embed + block_params * cfg.num_layers + final_norm + lm_head
    }

    /// Map a VQ codeword (0..vocab_image) into the unified vocab.
    pub fn image_token_to_id(&self, codeword: u32) -> u32 {
        self.config.image_id_offset() + codeword
    }

    /// Decode a unified id back to a VQ codeword (or None if it isn't an
    /// image token).
    pub fn id_to_image_token(&self, id: u32) -> Option<u32> {
        let off = self.config.image_id_offset();
        if id >= off && id < off + self.config.vocab_image as u32 {
            Some(id - off)
        } else { None }
    }
}

// ============================================================================
// Forward pass primitives (Phase 1 dense baseline).
// ============================================================================
//
// All matmuls go through `klearu_diffusion::blas::sgemm_a_btrans` which
// auto-picks Accelerate sgemm under the `accelerate` workspace feature
// (default for klearu-diffusion). At our scale (50M params, seq~320)
// total per-step compute is dominated by the four Linears in each block.

use klearu_diffusion::blas::{sgemm_a_btrans, sgemm_row_major};

/// y[t, ·] = x[t, ·] · W^T   for x [n_tokens, in] · W [out, in]^T → y [n_tokens, out]
fn linear_no_bias_forward(
    w: &LinearNoBias, x: &[f32], n_tokens: usize, out: &mut [f32],
) {
    debug_assert_eq!(x.len(), n_tokens * w.in_features);
    debug_assert_eq!(out.len(), n_tokens * w.out_features);
    sgemm_a_btrans(
        n_tokens, w.out_features, w.in_features,
        x, &w.weight, out,
    );
}

/// RMSNorm applied per row (each row is one token's hidden vector).
/// Routes to `klearu_diffusion::metal_backend::rms_norm_metal` under
/// `feature = "metal"` when the row count × dim is large enough to amortise
/// the GPU-launch overhead; otherwise (or on non-Metal builds) falls back
/// to the CPU loop.
fn rms_norm_apply(x: &mut [f32], norm: &RmsNorm, n_tokens: usize) {
    let d = norm.gamma.len();
    debug_assert_eq!(x.len(), n_tokens * d);
    #[cfg(feature = "metal")]
    {
        // Empirical break-even on M3 Ultra: ~16 rows × 512 dim. Below that the
        // upload/launch cost dominates the actual reduction.
        if n_tokens * d >= 8 * 1024 {
            klearu_diffusion::metal_backend::rms_norm_metal(x, &norm.gamma, d, norm.eps);
            return;
        }
    }
    for t in 0..n_tokens {
        let row = &mut x[t * d..(t + 1) * d];
        let mut s = 0.0_f64;
        for &v in row.iter() { s += (v as f64) * (v as f64); }
        let inv = ((s / d as f64) + norm.eps as f64).sqrt().recip() as f32;
        for (xi, &g) in row.iter_mut().zip(norm.gamma.iter()) {
            *xi = *xi * inv * g;
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// SwiGLU MLP forward: `down(silu(gate(x)) ⊙ up(x))`.
/// Returns the residual to add back to the input (caller does `x += swiglu(x)`).
fn swiglu_forward(blk: &ImageBlock, x: &[f32], n_tokens: usize) -> Vec<f32> {
    let d = blk.norm_mlp.gamma.len();
    let m = blk.mlp_gate.out_features;

    // Pre-norm.
    let mut xn = x.to_vec();
    rms_norm_apply(&mut xn, &blk.norm_mlp, n_tokens);

    let mut gate = vec![0.0_f32; n_tokens * m];
    let mut up = vec![0.0_f32; n_tokens * m];
    linear_no_bias_forward(&blk.mlp_gate, &xn, n_tokens, &mut gate);
    linear_no_bias_forward(&blk.mlp_up, &xn, n_tokens, &mut up);
    // SwiGLU: silu(gate) * up, elementwise.
    for i in 0..gate.len() {
        gate[i] = silu(gate[i]) * up[i];
    }
    let mut out = vec![0.0_f32; n_tokens * d];
    linear_no_bias_forward(&blk.mlp_down, &gate, n_tokens, &mut out);
    out
}

/// Multi-head self-attention with causal mask. Q, K, V projections,
/// scaled-dot-product, output projection. Returns the residual to add
/// back to the input.
fn attention_forward(
    blk: &ImageBlock, x: &[f32], n_tokens: usize,
    cfg: &ImageTransformerConfig,
) -> Vec<f32> {
    let d = cfg.hidden_size;
    let h = cfg.num_heads;
    let dh = cfg.head_dim();
    debug_assert_eq!(d, h * dh);

    let mut xn = x.to_vec();
    rms_norm_apply(&mut xn, &blk.norm_attn, n_tokens);

    let mut q = vec![0.0_f32; n_tokens * d];
    let mut k = vec![0.0_f32; n_tokens * d];
    let mut v = vec![0.0_f32; n_tokens * d];
    linear_no_bias_forward(&blk.q_proj, &xn, n_tokens, &mut q);
    linear_no_bias_forward(&blk.k_proj, &xn, n_tokens, &mut k);
    linear_no_bias_forward(&blk.v_proj, &xn, n_tokens, &mut v);

    // Under the `metal` feature: if the sequence is long enough that the
    // GPU flash-attention kernel beats two sgemms + CPU softmax, dispatch
    // there in one shot. Layout conversion: pack [n_tokens, h*dh] into the
    // kernel's expected [1, h, n_tokens, dh] (head-major).
    #[cfg(feature = "metal")]
    if n_tokens >= 256 {
        let mut q_perm = vec![0.0_f32; h * n_tokens * dh];
        let mut k_perm = vec![0.0_f32; h * n_tokens * dh];
        let mut v_perm = vec![0.0_f32; h * n_tokens * dh];
        for hi in 0..h {
            for t in 0..n_tokens {
                let src = t * d + hi * dh;
                let dst = hi * n_tokens * dh + t * dh;
                q_perm[dst..dst + dh].copy_from_slice(&q[src..src + dh]);
                k_perm[dst..dst + dh].copy_from_slice(&k[src..src + dh]);
                v_perm[dst..dst + dh].copy_from_slice(&v[src..src + dh]);
            }
        }
        let scale_g = 1.0_f32 / (dh as f32).sqrt();
        let mut out_perm = vec![0.0_f32; h * n_tokens * dh];
        klearu_diffusion::metal_backend::flash_attention_causal_metal(
            &q_perm, &k_perm, &v_perm, &mut out_perm,
            1, h, n_tokens, n_tokens, dh, scale_g,
        );
        let mut attn_out = vec![0.0_f32; n_tokens * d];
        for hi in 0..h {
            for t in 0..n_tokens {
                let src = hi * n_tokens * dh + t * dh;
                let dst = t * d + hi * dh;
                attn_out[dst..dst + dh].copy_from_slice(&out_perm[src..src + dh]);
            }
        }
        let mut o = vec![0.0_f32; n_tokens * d];
        linear_no_bias_forward(&blk.o_proj, &attn_out, n_tokens, &mut o);
        return o;
    }

    // Per-head SDPA, expressed as two sgemms that auto-route to Metal/Accelerate:
    //   1. scores = Q_h · K_h^T   (sgemm_a_btrans: m=n, n=n, k=dh)
    //   2. attn_h = scores · V_h  (sgemm_row_major: m=n, n=dh, k=n)
    // The causal mask + per-row softmax happen on CPU between the two —
    // the n×n score matrix per head is small (e.g. 320×320 = ~100k f32).
    let scale = 1.0_f32 / (dh as f32).sqrt();
    let mut attn_out = vec![0.0_f32; n_tokens * d];
    let mut q_h = vec![0.0_f32; n_tokens * dh];
    let mut k_h = vec![0.0_f32; n_tokens * dh];
    let mut v_h = vec![0.0_f32; n_tokens * dh];
    let mut scores = vec![0.0_f32; n_tokens * n_tokens];
    let mut attn_h = vec![0.0_f32; n_tokens * dh];
    for hi in 0..h {
        // Gather head slices into contiguous [n, dh] buffers.
        for t in 0..n_tokens {
            let src_off = t * d + hi * dh;
            let dst_off = t * dh;
            q_h[dst_off..dst_off + dh].copy_from_slice(&q[src_off..src_off + dh]);
            k_h[dst_off..dst_off + dh].copy_from_slice(&k[src_off..src_off + dh]);
            v_h[dst_off..dst_off + dh].copy_from_slice(&v[src_off..src_off + dh]);
        }
        // scores = Q_h · K_h^T.
        sgemm_a_btrans(n_tokens, n_tokens, dh, &q_h, &k_h, &mut scores);
        // Scale + causal mask + softmax per row.
        for qi in 0..n_tokens {
            let row = &mut scores[qi * n_tokens..(qi + 1) * n_tokens];
            for v in row.iter_mut() { *v *= scale; }
            // Mask j > qi.
            for j in (qi + 1)..n_tokens { row[j] = f32::NEG_INFINITY; }
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for s in row.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for s in row.iter_mut() { *s *= inv; }
            }
        }
        // attn_h = scores · V_h.
        sgemm_row_major(
            n_tokens, dh, n_tokens,
            1.0,
            &scores, n_tokens,
            &v_h, dh,
            0.0,
            &mut attn_h, dh,
        );
        // Scatter attn_h back into the [n, h*dh] output buffer.
        for t in 0..n_tokens {
            let dst_off = t * d + hi * dh;
            let src_off = t * dh;
            attn_out[dst_off..dst_off + dh].copy_from_slice(&attn_h[src_off..src_off + dh]);
        }
    }

    let mut o = vec![0.0_f32; n_tokens * d];
    linear_no_bias_forward(&blk.o_proj, &attn_out, n_tokens, &mut o);
    o
}

impl ImageTransformer {
    /// Forward pass.
    ///
    /// Input: `token_ids` is the full sequence in unified-vocab ids.
    /// Output: `[n_tokens, vocab_image]` logits row-major. Caller picks
    /// the rows corresponding to image-token-prediction positions and
    /// computes cross-entropy against the next-image-token targets.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let n = token_ids.len();
        let d = self.config.hidden_size;
        let max = self.config.max_seq_len();
        if n == 0 {
            return Err(ImageGenError::ShapeMismatch {
                expected: "non-empty token sequence".into(),
                got: "0 tokens".into(),
            });
        }
        if n > max {
            return Err(ImageGenError::ShapeMismatch {
                expected: format!("≤ max_seq_len = {max}"),
                got: format!("{n}"),
            });
        }
        let unified = self.config.unified_vocab_size();

        // 1. Token-embedding lookup + position embedding add.
        let mut x = vec![0.0_f32; n * d];
        for (i, &tid) in token_ids.iter().enumerate() {
            if (tid as usize) >= unified {
                return Err(ImageGenError::ShapeMismatch {
                    expected: format!("token id < unified vocab ({unified})"),
                    got: format!("id={tid} at position {i}"),
                });
            }
            let emb_off = (tid as usize) * d;
            let pos_off = i * d;
            let dst = &mut x[i * d..(i + 1) * d];
            for k in 0..d {
                dst[k] = self.embed[emb_off + k] + self.pos_embed[pos_off + k];
            }
        }

        // 2. Transformer blocks (each = pre-norm attention + pre-norm SwiGLU).
        for blk in &self.blocks {
            let attn = attention_forward(blk, &x, n, &self.config);
            for (xi, ai) in x.iter_mut().zip(attn.iter()) { *xi += ai; }
            let mlp = swiglu_forward(blk, &x, n);
            for (xi, mi) in x.iter_mut().zip(mlp.iter()) { *xi += mi; }
        }

        // 3. Final RMSNorm.
        rms_norm_apply(&mut x, &self.final_norm, n);

        // 4. LM head: project every position to image-vocab logits.
        let mut logits = vec![0.0_f32; n * self.config.vocab_image];
        linear_no_bias_forward(&self.lm_head, &x, n, &mut logits);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_50m_param_count_in_target_range() {
        let cfg = ImageTransformerConfig::baseline_50m();
        let model = ImageTransformer::from_config(cfg);
        let p = model.param_count();
        // We're shooting for ~50M; allow 30–80M range as the design space.
        assert!(p >= 30_000_000 && p <= 80_000_000,
            "param count {p} outside target 30–80M");
    }

    #[test]
    fn forward_runs_with_zero_weights() {
        // Tiny model so the test runs quickly.
        let cfg = ImageTransformerConfig {
            vocab_text: 100,
            vocab_image: 64,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            mlp_intermediate: 64,
            max_text_len: 8,
            image_grid_h: 4,
            image_grid_w: 4,
            bos_token: 100,
            sep_image_token: 101,
            eos_token: 102,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };
        let model = ImageTransformer::from_config(cfg.clone());
        // [BOS, text*3, SEP, img*4] = 9 tokens.
        let seq: Vec<u32> = vec![
            cfg.bos_token,
            5, 6, 7,
            cfg.sep_image_token,
            cfg.image_id_offset(),
            cfg.image_id_offset() + 1,
            cfg.image_id_offset() + 2,
            cfg.image_id_offset() + 3,
        ];
        let logits = model.forward(&seq).expect("forward");
        assert_eq!(logits.len(), seq.len() * cfg.vocab_image);
        // Zero weights → zero hidden → zero logits, but exp(0) is 1; the
        // important property is no NaN/Inf and the right shape.
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn unified_vocab_round_trip() {
        let cfg = ImageTransformerConfig::baseline_50m();
        let model = ImageTransformer::from_config(cfg);
        for cw in [0, 1, 4095, 8191] {
            let id = model.image_token_to_id(cw);
            assert_eq!(model.id_to_image_token(id), Some(cw));
        }
        // BOS/SEP/EOS are NOT image tokens.
        assert_eq!(model.id_to_image_token(model.config.bos_token), None);
        assert_eq!(model.id_to_image_token(model.config.sep_image_token), None);
        assert_eq!(model.id_to_image_token(model.config.eos_token), None);
    }
}
