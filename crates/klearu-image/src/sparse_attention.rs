//! LSH-routed sparse self-attention.
//!
//! Replaces dense O(n²) attention with: for each query, retrieve the
//! top-K candidate keys via LSH (signed random projections of the
//! head's K dimension), then compute scores + softmax + weighted sum
//! only over those candidates. The math is mathematically equivalent
//! to dense attention IF LSH returns all causally-valid keys; in
//! practice it returns a subset, but a high-quality one.
//!
//! Cost comparison at our baseline (n=320, h=8, dh=64):
//!   Dense:  O(n²·d) ≈ 320·320·512 ≈ 50M ops per layer's attention.
//!   Sparse: O(n·K·dh + n·num_tables) ≈ 320·64·64 + 320·8 ≈ 1.3M ops.
//!   ~40× speedup on the scoring step. LSH index build adds O(n·num_tables)
//!   per-layer overhead — small at this scale.
//!
//! Backward pass: we cache the candidate sets used during forward and
//! reuse them during backward. The unselected positions contribute
//! zero gradient (straight-through approximation). This is the same
//! trick used by Routing Transformers and Reformer.

use klearu_core::config::{HashFunctionType, LshConfig, BucketType};
use klearu_core::lsh::{create_lsh_index, LshCandidate};

use crate::error::{ImageGenError, Result};
use crate::model::{ImageBlock, ImageTransformer, ImageTransformerConfig, LinearNoBias, RmsNorm};

/// Configuration knobs for sparse attention.
#[derive(Debug, Clone)]
pub struct SparseAttnConfig {
    pub hash_function: HashFunctionType,
    pub num_hashes: usize,
    pub num_tables: usize,
    pub bucket_capacity: usize,
    /// Random seed for the hash projections.
    pub seed: u64,
}

impl Default for SparseAttnConfig {
    fn default() -> Self {
        Self {
            hash_function: HashFunctionType::SimHash,
            num_hashes: 6,        // 2^6 = 64 buckets per table
            num_tables: 8,        // recall × tables
            bucket_capacity: 64,
            seed: 0xa7c12b30_c4decafb,
        }
    }
}

/// Per-attention-call output of the sparse SDPA. Returns the same shape
/// as `attention_out_pre_oproj` from the dense path so callers can
/// drop it in identically. Also returns the per-head candidate sets
/// for backward.
pub struct SparseAttnOut {
    /// `[n_tokens, hidden_size]` — output before O projection.
    pub out_pre_o: Vec<f32>,
    /// `[num_heads]` Vec<Vec<usize>>: candidates[head][query_idx]
    /// returns the kept key indices for backward reuse. Kept SORTED.
    pub candidates: Vec<Vec<Vec<usize>>>,
    /// `[num_heads, n_tokens, max_candidates]` flat softmax weights
    /// aligned with `candidates[head][query_idx]`. Used during
    /// backward to avoid recomputing the softmax.
    pub attn_weights_per_head: Vec<Vec<Vec<f32>>>,
}

/// LSH-routed multi-head self-attention forward.
///
/// `q`, `k`, `v`: shape `[n_tokens, hidden_size]` (each head's slice
///                is at `[t * hidden_size + h * head_dim..][..head_dim]`).
/// Causal: query at position i sees only keys at positions ≤ i.
pub fn sparse_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    model_cfg: &ImageTransformerConfig,
    sparse_cfg: &SparseAttnConfig,
) -> SparseAttnOut {
    let d = model_cfg.hidden_size;
    let h = model_cfg.num_heads;
    let dh = model_cfg.head_dim();
    debug_assert_eq!(d, h * dh);

    let scale = 1.0_f32 / (dh as f32).sqrt();
    let mut out_pre_o = vec![0.0_f32; n * d];
    let mut all_cands: Vec<Vec<Vec<usize>>> = Vec::with_capacity(h);
    let mut all_weights: Vec<Vec<Vec<f32>>> = Vec::with_capacity(h);

    for hi in 0..h {
        // 1. Build a per-head LSH index over this head's keys.
        let lsh_config = LshConfig {
            hash_function: sparse_cfg.hash_function,
            bucket_type: BucketType::Fifo,
            num_tables: sparse_cfg.num_tables,
            range_pow: sparse_cfg.num_hashes,
            num_hashes: sparse_cfg.num_hashes,
            bucket_capacity: sparse_cfg.bucket_capacity,
            rebuild_interval_base: 0,
            rebuild_decay: 1.0,
        };
        // Per-head seed: mix in head index so different heads get
        // different projections.
        let seed = sparse_cfg.seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(hi as u64);
        let mut idx = create_lsh_index(&lsh_config, dh, seed);

        for kj in 0..n {
            let off = kj * d + hi * dh;
            idx.insert(kj as u32, &k[off..off + dh]);
        }

        let mut head_cands: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut head_weights: Vec<Vec<f32>> = Vec::with_capacity(n);

        // 2. Per-query: retrieve candidates, filter to causal, score, softmax.
        for qi in 0..n {
            let q_off = qi * d + hi * dh;
            let q_vec = &q[q_off..q_off + dh];

            // Query LSH. Get candidate counts; we keep all returned + the
            // current position (qi). The current position is always
            // attended to in practice (a query attending to itself);
            // LSH should retrieve it but we add it defensively.
            let cands: Vec<LshCandidate> = idx.query_with_counts(q_vec);
            let mut keep: Vec<usize> = cands.iter()
                .map(|c| c.neuron_id as usize)
                .filter(|&j| j <= qi)
                .collect();
            // Ensure qi itself is included (self-attention) — LSH may
            // miss it under aggressive bucketing.
            if !keep.contains(&qi) { keep.push(qi); }
            // Sort for deterministic backward layout.
            keep.sort_unstable();
            keep.dedup();

            // 3. Score candidates with scaled dot product.
            let m = keep.len();
            let mut scores = vec![0.0_f32; m];
            for (idx_in_keep, &kj) in keep.iter().enumerate() {
                let k_off = kj * d + hi * dh;
                let mut s = 0.0_f32;
                for di in 0..dh { s += q_vec[di] * k[k_off + di]; }
                scores[idx_in_keep] = s * scale;
            }
            // 4. Softmax across the kept candidates.
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for s in scores.iter_mut() { *s *= inv; }
            }
            // 5. Weighted sum of V across kept candidates.
            let out_off = qi * d + hi * dh;
            for (idx_in_keep, &kj) in keep.iter().enumerate() {
                let v_off = kj * d + hi * dh;
                let w = scores[idx_in_keep];
                if w == 0.0 { continue; }
                for di in 0..dh { out_pre_o[out_off + di] += w * v[v_off + di]; }
            }
            head_weights.push(scores);
            head_cands.push(keep);
        }

        all_cands.push(head_cands);
        all_weights.push(head_weights);
    }

    SparseAttnOut { out_pre_o, candidates: all_cands, attn_weights_per_head: all_weights }
}

/// SDPA backward over the cached candidate sets. The "unselected positions
/// contribute zero gradient" approximation: positions not in
/// `cache.candidates[head][qi]` are treated as exactly zero-weighted, which
/// makes their grad contribution exactly zero. Mathematically equivalent to
/// the dense backward IF the LSH retrieved every causally-valid key; lossy
/// otherwise (the same approximation the forward uses).
///
/// Returns `(grad_q, grad_k, grad_v)`, each `[n, hidden_size]`.
pub fn sparse_attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cache: &SparseAttnOut,
    grad_out_pre_o: &[f32],
    n: usize,
    cfg: &ImageTransformerConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = cfg.hidden_size;
    let h = cfg.num_heads;
    let dh = cfg.head_dim();
    let scale = 1.0_f32 / (dh as f32).sqrt();

    let mut grad_q = vec![0.0_f32; n * d];
    let mut grad_k = vec![0.0_f32; n * d];
    let mut grad_v = vec![0.0_f32; n * d];

    for hi in 0..h {
        for qi in 0..n {
            let kept = &cache.candidates[hi][qi];
            let aw = &cache.attn_weights_per_head[hi][qi];
            debug_assert_eq!(kept.len(), aw.len());

            // grad_attn[idx] = Σ_dh grad_out[qi, h, dh] * v[kept[idx], h, dh]
            let out_off = qi * d + hi * dh;
            let mut grad_attn = vec![0.0_f32; kept.len()];
            for (idx, &kj) in kept.iter().enumerate() {
                let v_off = kj * d + hi * dh;
                let mut s = 0.0_f32;
                for di in 0..dh { s += grad_out_pre_o[out_off + di] * v[v_off + di]; }
                grad_attn[idx] = s;
            }

            // grad_v[kept[idx]] += aw[idx] · grad_out[qi]
            for (idx, &kj) in kept.iter().enumerate() {
                let w = aw[idx];
                if w == 0.0 { continue; }
                let v_off = kj * d + hi * dh;
                for di in 0..dh {
                    grad_v[v_off + di] += w * grad_out_pre_o[out_off + di];
                }
            }

            // Softmax backward (restricted to kept):
            //   grad_score[idx] = aw[idx] · (grad_attn[idx] − Σ_l aw[l] · grad_attn[l])
            let mut row_dot = 0.0_f32;
            for idx in 0..kept.len() { row_dot += aw[idx] * grad_attn[idx]; }
            for (idx, &kj) in kept.iter().enumerate() {
                let g_score = aw[idx] * (grad_attn[idx] - row_dot);
                let scaled = g_score * scale;
                let q_off = qi * d + hi * dh;
                let k_off = kj * d + hi * dh;
                for di in 0..dh {
                    grad_q[q_off + di] += scaled * k[k_off + di];
                    grad_k[k_off + di] += scaled * q[q_off + di];
                }
            }
        }
    }

    (grad_q, grad_k, grad_v)
}

// ============================================================================
// Drop-in inference forward: uses the dense ImageTransformer's weights but
// runs SDPA via LSH-routed sparse attention.
// ============================================================================
//
// Same model weights as the dense forward — only the attention math changes.
// No new training required; this is a pure inference-speed optimization that
// keeps the existing checkpoint compatibility intact.

/// Naive linear forward (re-implemented locally to avoid a cycle with model.rs).
fn linear_forward_local(w: &LinearNoBias, x: &[f32], n: usize, y: &mut [f32]) {
    use klearu_diffusion::blas::sgemm_a_btrans;
    sgemm_a_btrans(n, w.out_features, w.in_features, x, &w.weight, y);
}

/// RMSNorm in place.
fn rms_norm_inplace(x: &mut [f32], norm: &RmsNorm, n: usize) {
    let d = norm.gamma.len();
    for t in 0..n {
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

/// One block forward using sparse attention. SwiGLU MLP path is unchanged.
fn block_forward_sparse_attn(
    blk: &ImageBlock,
    x: &[f32],
    n: usize,
    cfg: &ImageTransformerConfig,
    sparse_cfg: &SparseAttnConfig,
) -> Vec<f32> {
    let d = cfg.hidden_size;
    let mp = cfg.mlp_intermediate;

    // Pre-attention norm.
    let mut xn1 = x.to_vec();
    rms_norm_inplace(&mut xn1, &blk.norm_attn, n);

    let mut q = vec![0.0_f32; n * d];
    let mut k = vec![0.0_f32; n * d];
    let mut v = vec![0.0_f32; n * d];
    linear_forward_local(&blk.q_proj, &xn1, n, &mut q);
    linear_forward_local(&blk.k_proj, &xn1, n, &mut k);
    linear_forward_local(&blk.v_proj, &xn1, n, &mut v);

    // ── Sparse SDPA replaces the dense scoring loop.
    let sa = sparse_attention_forward(&q, &k, &v, n, cfg, sparse_cfg);

    let mut o_out = vec![0.0_f32; n * d];
    linear_forward_local(&blk.o_proj, &sa.out_pre_o, n, &mut o_out);
    // Residual.
    let mut attn_residual = vec![0.0_f32; n * d];
    for i in 0..attn_residual.len() { attn_residual[i] = x[i] + o_out[i]; }

    // Dense SwiGLU MLP (unchanged from the dense path).
    let mut xn2 = attn_residual.clone();
    rms_norm_inplace(&mut xn2, &blk.norm_mlp, n);
    let mut gate = vec![0.0_f32; n * mp];
    let mut up = vec![0.0_f32; n * mp];
    linear_forward_local(&blk.mlp_gate, &xn2, n, &mut gate);
    linear_forward_local(&blk.mlp_up, &xn2, n, &mut up);
    for i in 0..gate.len() { gate[i] = silu(gate[i]) * up[i]; }
    let mut down = vec![0.0_f32; n * d];
    linear_forward_local(&blk.mlp_down, &gate, n, &mut down);
    let mut out = vec![0.0_f32; n * d];
    for i in 0..out.len() { out[i] = attn_residual[i] + down[i]; }
    out
}

/// Forward pass of the dense `ImageTransformer` model, but with the
/// attention layers replaced by sparse LSH-routed attention. Same
/// weights, same output shape — only attention math changes.
pub fn forward_with_sparse_attention(
    model: &ImageTransformer,
    token_ids: &[u32],
    sparse_cfg: &SparseAttnConfig,
) -> Result<Vec<f32>> {
    let cfg = &model.config;
    let n = token_ids.len();
    let d = cfg.hidden_size;
    if n == 0 || n > cfg.max_seq_len() {
        return Err(ImageGenError::ShapeMismatch {
            expected: format!("1..={} tokens", cfg.max_seq_len()),
            got: format!("{n}"),
        });
    }
    let unified = cfg.unified_vocab_size();

    // Embed + position add.
    let mut x = vec![0.0_f32; n * d];
    for (i, &tid) in token_ids.iter().enumerate() {
        if (tid as usize) >= unified {
            return Err(ImageGenError::ShapeMismatch {
                expected: format!("token id < {unified}"),
                got: format!("id={tid}"),
            });
        }
        let eoff = (tid as usize) * d;
        let poff = i * d;
        let dst = &mut x[i * d..(i + 1) * d];
        for k in 0..d {
            dst[k] = model.embed[eoff + k] + model.pos_embed[poff + k];
        }
    }

    // Per-block forward with sparse attention.
    for blk in &model.blocks {
        x = block_forward_sparse_attn(blk, &x, n, cfg, sparse_cfg);
    }

    // Final norm + LM head.
    rms_norm_inplace(&mut x, &model.final_norm, n);
    let v = cfg.vocab_image;
    let mut logits = vec![0.0_f32; n * v];
    linear_forward_local(&model.lm_head, &x, n, &mut logits);
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> ImageTransformerConfig {
        ImageTransformerConfig {
            vocab_text: 100, vocab_image: 16,
            hidden_size: 32, num_layers: 1, num_heads: 4,
            mlp_intermediate: 64,
            max_text_len: 8,
            image_grid_h: 4, image_grid_w: 4,
            bos_token: 100, sep_image_token: 101, eos_token: 102,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
        }
    }

    #[test]
    fn sparse_attention_runs_with_finite_output() {
        let cfg = tiny_cfg();
        let n = 20;
        let d = cfg.hidden_size;
        // Random-ish Q/K/V.
        let mut state = 1_u64;
        let mut rand = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let q: Vec<f32> = (0..n * d).map(|_| rand() * 0.5).collect();
        let k: Vec<f32> = (0..n * d).map(|_| rand() * 0.5).collect();
        let v: Vec<f32> = (0..n * d).map(|_| rand() * 0.5).collect();

        let sc = SparseAttnConfig::default();
        let out = sparse_attention_forward(&q, &k, &v, n, &cfg, &sc);
        assert_eq!(out.out_pre_o.len(), n * d);
        assert!(out.out_pre_o.iter().all(|x| x.is_finite()),
            "sparse attention produced NaN/Inf");
        // Each query's candidate set must be non-empty and contain the
        // query position itself (the safety net we add).
        for hi in 0..cfg.num_heads {
            for qi in 0..n {
                let c = &out.candidates[hi][qi];
                assert!(!c.is_empty(), "head {hi} query {qi} has empty candidates");
                assert!(c.contains(&qi), "head {hi} query {qi} missing self");
                for &j in c { assert!(j <= qi, "causal violation: head {hi} qi {qi} kj {j}"); }
            }
        }
        // Softmax weights per (head, query) should sum to ~1.0.
        for hi in 0..cfg.num_heads {
            for qi in 0..n {
                let ws = &out.attn_weights_per_head[hi][qi];
                let sum: f32 = ws.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4,
                    "head {hi} qi {qi}: softmax sum {sum}");
            }
        }
    }

    #[test]
    fn forward_with_sparse_attention_close_to_dense() {
        // Drop-in equivalence test: sparse-attention forward should produce
        // close-to-dense output with high-recall LSH config.
        let cfg = tiny_cfg();
        let mut model = ImageTransformer::from_config(cfg.clone());
        // Tiny random-ish init.
        let mut s = 7_u64;
        let mut rand = || -> f32 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) * 0.1 - 0.05
        };
        for v in model.embed.iter_mut() { *v = rand(); }
        for v in model.pos_embed.iter_mut() { *v = rand(); }
        for blk in model.blocks.iter_mut() {
            for v in blk.q_proj.weight.iter_mut() { *v = rand(); }
            for v in blk.k_proj.weight.iter_mut() { *v = rand(); }
            for v in blk.v_proj.weight.iter_mut() { *v = rand(); }
            for v in blk.o_proj.weight.iter_mut() { *v = rand(); }
            for v in blk.mlp_gate.weight.iter_mut() { *v = rand(); }
            for v in blk.mlp_up.weight.iter_mut() { *v = rand(); }
            for v in blk.mlp_down.weight.iter_mut() { *v = rand(); }
        }
        for v in model.lm_head.weight.iter_mut() { *v = rand(); }

        let seq = vec![
            cfg.bos_token, 5, 6,
            cfg.sep_image_token,
            cfg.image_id_offset(),
            cfg.image_id_offset() + 1,
        ];
        let dense = model.forward(&seq).expect("dense");
        let sparse_cfg = SparseAttnConfig {
            num_tables: 32, num_hashes: 2, bucket_capacity: 64,
            ..SparseAttnConfig::default()
        };
        let sparse = forward_with_sparse_attention(&model, &seq, &sparse_cfg)
            .expect("sparse");
        assert_eq!(dense.len(), sparse.len());
        // Cosine similarity between sparse-attn and dense logits.
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        for (a, b) in dense.iter().zip(sparse.iter()) {
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64) * (*a as f64);
            nb += (*b as f64) * (*b as f64);
        }
        let cos = (dot / (na.sqrt() * nb.sqrt())) as f32;
        assert!(cos > 0.85,
            "sparse-attn forward should be close to dense at high recall: cos={cos}");
    }

    #[test]
    fn sparse_backward_matches_dense_at_high_recall() {
        // At high LSH recall the candidate set covers all causally-valid keys,
        // so sparse backward ≡ dense backward. Verify by direct cosine
        // similarity with a hand-rolled dense backward.
        let cfg = tiny_cfg();
        let n = 12;
        let d = cfg.hidden_size;
        let h = cfg.num_heads;
        let dh = cfg.head_dim();
        let scale = 1.0_f32 / (dh as f32).sqrt();

        let mut s = 0xABCD_1234_u64;
        let mut rand = || -> f32 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let q: Vec<f32> = (0..n * d).map(|_| rand() * 0.5).collect();
        let k: Vec<f32> = (0..n * d).map(|_| rand() * 0.5).collect();
        let v: Vec<f32> = (0..n * d).map(|_| rand() * 0.5).collect();
        let grad_out: Vec<f32> = (0..n * d).map(|_| rand() * 0.2).collect();

        // High-recall sparse forward.
        let sc = SparseAttnConfig {
            num_tables: 32, num_hashes: 1, bucket_capacity: 128,
            ..SparseAttnConfig::default()
        };
        let cache = sparse_attention_forward(&q, &k, &v, n, &cfg, &sc);
        let (gq_s, gk_s, gv_s) = sparse_attention_backward(
            &q, &k, &v, &cache, &grad_out, n, &cfg);

        // Dense reference: redo forward + standard SDPA backward.
        let mut dense_aw = vec![0.0_f32; h * n * n];
        let mut dense_out = vec![0.0_f32; n * d];
        for hi in 0..h {
            for qi in 0..n {
                let mut scores = vec![0.0_f32; qi + 1];
                let q_off = qi * d + hi * dh;
                for kj in 0..=qi {
                    let k_off = kj * d + hi * dh;
                    let mut sum = 0.0_f32;
                    for di in 0..dh { sum += q[q_off + di] * k[k_off + di]; }
                    scores[kj] = sum * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
                let inv = 1.0 / sum;
                for s in scores.iter_mut() { *s *= inv; }
                for (kj, &w) in scores.iter().enumerate() {
                    dense_aw[hi * n * n + qi * n + kj] = w;
                }
                let out_off = qi * d + hi * dh;
                for kj in 0..=qi {
                    let v_off = kj * d + hi * dh;
                    let w = scores[kj];
                    for di in 0..dh { dense_out[out_off + di] += w * v[v_off + di]; }
                }
            }
        }
        let mut gq_d = vec![0.0_f32; n * d];
        let mut gk_d = vec![0.0_f32; n * d];
        let mut gv_d = vec![0.0_f32; n * d];
        for hi in 0..h {
            let aw_base = hi * n * n;
            for qi in 0..n {
                let out_off = qi * d + hi * dh;
                let mut grad_attn = vec![0.0_f32; n];
                for kj in 0..=qi {
                    let v_off = kj * d + hi * dh;
                    let mut sum = 0.0_f32;
                    for di in 0..dh { sum += grad_out[out_off + di] * v[v_off + di]; }
                    grad_attn[kj] = sum;
                }
                for kj in 0..=qi {
                    let w = dense_aw[aw_base + qi * n + kj];
                    let v_off = kj * d + hi * dh;
                    for di in 0..dh { gv_d[v_off + di] += w * grad_out[out_off + di]; }
                }
                let mut row_dot = 0.0_f32;
                for kj in 0..=qi {
                    row_dot += dense_aw[aw_base + qi * n + kj] * grad_attn[kj];
                }
                for kj in 0..=qi {
                    let aij = dense_aw[aw_base + qi * n + kj];
                    let g_score = aij * (grad_attn[kj] - row_dot);
                    let scaled = g_score * scale;
                    let q_off = qi * d + hi * dh;
                    let k_off = kj * d + hi * dh;
                    for di in 0..dh {
                        gq_d[q_off + di] += scaled * k[k_off + di];
                        gk_d[k_off + di] += scaled * q[q_off + di];
                    }
                }
            }
        }

        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for (x, y) in a.iter().zip(b.iter()) {
                dot += (*x as f64) * (*y as f64);
                na += (*x as f64) * (*x as f64);
                nb += (*y as f64) * (*y as f64);
            }
            (dot / (na.sqrt() * nb.sqrt())) as f32
        };
        let cq = cos(&gq_s, &gq_d);
        let ck = cos(&gk_s, &gk_d);
        let cv = cos(&gv_s, &gv_d);
        assert!(cq > 0.85, "grad_q cosine vs dense too low: {cq}");
        assert!(ck > 0.85, "grad_k cosine vs dense too low: {ck}");
        assert!(cv > 0.85, "grad_v cosine vs dense too low: {cv}");
    }

    #[test]
    fn sparse_attention_recovery_with_more_tables() {
        // With LOTS of tables and few hashes (coarse buckets), LSH should
        // return nearly all causally-valid keys → sparse output should
        // approach dense. Test this convergence loosely.
        let cfg = tiny_cfg();
        let n = 16;
        let d = cfg.hidden_size;
        let mut state = 42_u64;
        let mut rand = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let q: Vec<f32> = (0..n * d).map(|_| rand() * 0.5).collect();
        let k: Vec<f32> = (0..n * d).map(|_| rand() * 0.5).collect();
        let v: Vec<f32> = (0..n * d).map(|_| rand() * 0.5).collect();

        // High-recall config: 32 tables, 2 hashes (4 buckets each).
        let sc = SparseAttnConfig {
            num_tables: 32, num_hashes: 2, bucket_capacity: 64,
            ..SparseAttnConfig::default()
        };
        let out = sparse_attention_forward(&q, &k, &v, n, &cfg, &sc);

        // Compute dense reference for comparison.
        let h = cfg.num_heads;
        let dh = cfg.head_dim();
        let scale = 1.0_f32 / (dh as f32).sqrt();
        let mut dense = vec![0.0_f32; n * d];
        for hi in 0..h {
            for qi in 0..n {
                let mut scores = vec![0.0_f32; qi + 1];
                let q_off = qi * d + hi * dh;
                for kj in 0..=qi {
                    let k_off = kj * d + hi * dh;
                    let mut s = 0.0_f32;
                    for di in 0..dh { s += q[q_off + di] * k[k_off + di]; }
                    scores[kj] = s * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
                let inv = 1.0 / sum;
                for s in scores.iter_mut() { *s *= inv; }
                let out_off = qi * d + hi * dh;
                for kj in 0..=qi {
                    let v_off = kj * d + hi * dh;
                    let w = scores[kj];
                    for di in 0..dh { dense[out_off + di] += w * v[v_off + di]; }
                }
            }
        }
        // Cosine similarity between sparse and dense outputs.
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        for (a, b) in out.out_pre_o.iter().zip(dense.iter()) {
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64) * (*a as f64);
            nb += (*b as f64) * (*b as f64);
        }
        let cos = (dot / (na.sqrt() * nb.sqrt())) as f32;
        // With high-recall settings, cosine should be > 0.85. Looser than
        // 1.0 because some keys may still be missed; LSH is approximate.
        assert!(cos > 0.85,
            "sparse vs dense cosine similarity {cos} too low with high-recall config");
    }
}
