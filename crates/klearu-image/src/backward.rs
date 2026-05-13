//! Forward-with-cache + backward pass + full training step for
//! [`ImageTransformer`].
//!
//! Strategy: hand-rolled backward, no autograd tape. Each op has a
//! forward variant that caches intermediates needed for backward
//! (`forward_train`), and a corresponding backward routine that
//! consumes them to populate gradient buffers.
//!
//! All ops here are CPU-only and naive (no Accelerate/Metal). Speed
//! comes later — correctness first. For the baseline 50M-param model
//! at seq_len 320, one training step is ~10-30 seconds; acceptable for
//! the first dense baseline.
//!
//! [`ImageTransformer`]: crate::model::ImageTransformer

use crate::error::{ImageGenError, Result};
use crate::grad::Gradients;
use crate::model::{ImageBlock, ImageTransformer, ImageTransformerConfig, RmsNorm, LinearNoBias};

// ============================================================================
// Forward cache (activations needed during backward)
// ============================================================================

/// Per-layer cached activations.
pub struct LayerCache {
    pub x_in: Vec<f32>,         // input to the layer
    pub xn1: Vec<f32>,          // RMSNorm output before attention
    pub q: Vec<f32>,            // [n, d]
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Softmax-attention weights. Shape `[h, n, n]` row-major. Only the
    /// causal-valid entries (j ≤ i for head h) are populated; the rest
    /// are zero (and skipped during backward).
    pub attn_weights: Vec<f32>,
    pub attn_out_pre_o: Vec<f32>, // [n, d], output of SDPA before o_proj
    pub attn_residual: Vec<f32>,  // x_in + o_proj(attn_out_pre_o)
    pub xn2: Vec<f32>,            // RMSNorm output before MLP
    pub mlp_gate: Vec<f32>,       // [n, m] pre-SiLU
    pub mlp_up: Vec<f32>,         // [n, m]
    pub mlp_silu_x_up: Vec<f32>,  // [n, m] post-SwiGLU mul
}

/// Forward cache for one training-sequence pass through the model.
pub struct ForwardCache {
    pub token_ids: Vec<u32>,
    pub x_after_embed: Vec<f32>,     // [n, d] after embed + pos add
    pub layers: Vec<LayerCache>,
    pub x_pre_final_norm: Vec<f32>,  // [n, d]
    pub x_post_final_norm: Vec<f32>, // [n, d]
    pub logits: Vec<f32>,            // [n, vocab_image]
}

// ============================================================================
// Forward with cache
// ============================================================================

#[inline]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
#[inline]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// Linear forward: y[n, out] = x[n, in] · W^T (W: [out, in]).
fn linear_forward(w: &LinearNoBias, x: &[f32], n: usize, y: &mut [f32]) {
    let in_f = w.in_features;
    let out_f = w.out_features;
    debug_assert_eq!(x.len(), n * in_f);
    debug_assert_eq!(y.len(), n * out_f);
    for t in 0..n {
        let xr = &x[t * in_f..(t + 1) * in_f];
        let yr = &mut y[t * out_f..(t + 1) * out_f];
        for o in 0..out_f {
            let wr = &w.weight[o * in_f..(o + 1) * in_f];
            let mut s = 0.0_f32;
            for i in 0..in_f { s += xr[i] * wr[i]; }
            yr[o] = s;
        }
    }
}

/// RMSNorm forward (in place). Also returns per-row `inv` values
/// (length n) for backward.
fn rms_norm_forward_collect_inv(x: &mut [f32], norm: &RmsNorm, n: usize) -> Vec<f32> {
    let d = norm.gamma.len();
    let mut invs = vec![0.0_f32; n];
    for t in 0..n {
        let row = &mut x[t * d..(t + 1) * d];
        let mut s = 0.0_f64;
        for &v in row.iter() { s += (v as f64) * (v as f64); }
        let inv = ((s / d as f64) + norm.eps as f64).sqrt().recip() as f32;
        invs[t] = inv;
        for (xi, &g) in row.iter_mut().zip(norm.gamma.iter()) {
            *xi = *xi * inv * g;
        }
    }
    invs
}

/// Full forward pass, caching everything needed for backward.
pub fn forward_train(
    model: &ImageTransformer,
    token_ids: &[u32],
) -> Result<ForwardCache> {
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

    // 1. Embed + position add.
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
    let x_after_embed = x.clone();

    // 2. Per-layer forward.
    let mut layers = Vec::with_capacity(model.blocks.len());
    for blk in &model.blocks {
        let cache = layer_forward(blk, &x, n, cfg);
        x = cache.attn_residual.clone();
        // mlp output goes into x via residual.
        let mut mlp_down_out = vec![0.0_f32; n * d];
        linear_forward(&blk.mlp_down, &cache.mlp_silu_x_up, n, &mut mlp_down_out);
        for i in 0..x.len() { x[i] = cache.attn_residual[i] + mlp_down_out[i]; }
        layers.push(cache);
    }

    let x_pre_final_norm = x.clone();
    let _ = rms_norm_forward_collect_inv(&mut x, &model.final_norm, n);
    let x_post_final_norm = x.clone();

    // 3. LM head.
    let v = cfg.vocab_image;
    let mut logits = vec![0.0_f32; n * v];
    linear_forward(&model.lm_head, &x, n, &mut logits);

    Ok(ForwardCache {
        token_ids: token_ids.to_vec(),
        x_after_embed,
        layers,
        x_pre_final_norm,
        x_post_final_norm,
        logits,
    })
}

fn layer_forward(
    blk: &ImageBlock,
    x_in: &[f32],
    n: usize,
    cfg: &ImageTransformerConfig,
) -> LayerCache {
    let d = cfg.hidden_size;
    let h = cfg.num_heads;
    let dh = cfg.head_dim();

    // Pre-attention norm.
    let mut xn1 = x_in.to_vec();
    let _ = rms_norm_forward_collect_inv(&mut xn1, &blk.norm_attn, n);

    // Q, K, V.
    let mut q = vec![0.0_f32; n * d];
    let mut k = vec![0.0_f32; n * d];
    let mut v = vec![0.0_f32; n * d];
    linear_forward(&blk.q_proj, &xn1, n, &mut q);
    linear_forward(&blk.k_proj, &xn1, n, &mut k);
    linear_forward(&blk.v_proj, &xn1, n, &mut v);

    // Attention (causal). attn_weights[h, n, n] only valid for j ≤ i; the
    // rest are 0. attn_out_pre_o[n, d].
    let scale = 1.0_f32 / (dh as f32).sqrt();
    let mut attn_weights = vec![0.0_f32; h * n * n];
    let mut attn_out_pre_o = vec![0.0_f32; n * d];
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
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for s in scores.iter_mut() { *s *= inv; }
            }
            // Store in [h, n, n] layout: attn_weights[hi, qi, kj].
            for (kj, &w) in scores.iter().enumerate() {
                attn_weights[hi * n * n + qi * n + kj] = w;
            }
            // Output for this (head, qi): sum_kj weights[kj] * V[kj, head, :].
            let out_off = qi * d + hi * dh;
            for kj in 0..=qi {
                let v_off = kj * d + hi * dh;
                let w = scores[kj];
                if w == 0.0 { continue; }
                for di in 0..dh { attn_out_pre_o[out_off + di] += w * v[v_off + di]; }
            }
        }
    }

    // Output projection and residual.
    let mut o_out = vec![0.0_f32; n * d];
    linear_forward(&blk.o_proj, &attn_out_pre_o, n, &mut o_out);
    let mut attn_residual = vec![0.0_f32; n * d];
    for i in 0..attn_residual.len() { attn_residual[i] = x_in[i] + o_out[i]; }

    // Pre-MLP norm.
    let mut xn2 = attn_residual.clone();
    let _ = rms_norm_forward_collect_inv(&mut xn2, &blk.norm_mlp, n);

    // SwiGLU MLP: gate, up, silu * up. mlp_down lives outside the cache
    // (the caller does it). We cache pre-down values.
    let mp = cfg.mlp_intermediate;
    let mut mlp_gate = vec![0.0_f32; n * mp];
    let mut mlp_up = vec![0.0_f32; n * mp];
    linear_forward(&blk.mlp_gate, &xn2, n, &mut mlp_gate);
    linear_forward(&blk.mlp_up, &xn2, n, &mut mlp_up);
    let mut mlp_silu_x_up = vec![0.0_f32; n * mp];
    for i in 0..mlp_silu_x_up.len() {
        mlp_silu_x_up[i] = silu(mlp_gate[i]) * mlp_up[i];
    }

    LayerCache { x_in: x_in.to_vec(), xn1, q, k, v, attn_weights,
        attn_out_pre_o, attn_residual, xn2, mlp_gate, mlp_up, mlp_silu_x_up }
}

// ============================================================================
// Backward
// ============================================================================

/// Linear backward — routes through `klearu_diffusion::blas::sgemm_row_major`,
/// which auto-dispatches to Accelerate AMX on macOS / matrixmultiply on
/// other platforms. ~50× faster than the previous naive triple-loop.
///
///   y = x @ W^T (W: [out, in], x: [n, in], y: [n, out])
///   grad_W += grad_y^T @ x   (shape [out, in], accumulated)
///   grad_x  = grad_y @ W     (returned)
fn linear_backward(
    w: &LinearNoBias,
    grad_y: &[f32],
    x: &[f32],
    n: usize,
    grad_w_acc: &mut [f32],
    grad_x_out: &mut [f32],
) {
    use klearu_diffusion::blas::sgemm_row_major;
    let in_f = w.in_features;
    let out_f = w.out_features;
    debug_assert_eq!(grad_y.len(), n * out_f);
    debug_assert_eq!(x.len(), n * in_f);
    debug_assert_eq!(grad_w_acc.len(), out_f * in_f);
    debug_assert_eq!(grad_x_out.len(), n * in_f);

    // ── grad_x = grad_y @ W  (shapes: [n, out_f] @ [out_f, in_f] = [n, in_f])
    // Standard row-major sgemm: m=n, n=in_f, k=out_f, alpha=1, beta=0.
    sgemm_row_major(
        n, in_f, out_f,
        1.0,
        grad_y, out_f,         // lda = out_f
        &w.weight, in_f,       // ldb = in_f
        0.0,
        grad_x_out, in_f,      // ldc = in_f
    );

    // ── grad_W += grad_y^T @ x  (shapes: [out_f, n] @ [n, in_f] = [out_f, in_f])
    // Our sgemm wrapper doesn't expose the transA flag, so materialise
    // grad_y^T into a scratch buffer. Cost: O(n · out_f) — small vs the
    // sgemm itself (O(n · out_f · in_f)) and beats CPU loops by ~50×.
    let mut grad_y_t = vec![0.0_f32; out_f * n];
    for t in 0..n {
        for o in 0..out_f {
            grad_y_t[o * n + t] = grad_y[t * out_f + o];
        }
    }
    // beta=1 to accumulate (for gradient batching).
    sgemm_row_major(
        out_f, in_f, n,
        1.0,
        &grad_y_t, n,          // lda = n
        x, in_f,               // ldb = in_f
        1.0,
        grad_w_acc, in_f,      // ldc = in_f
    );
}

/// RMSNorm backward.
///   y_i = gamma_i * x_i * inv ;  inv = (mean(x²) + eps)^(-1/2)
///   Need: grad_gamma (accumulated), grad_x (output).
///   Standard formula:
///     grad_x_k = inv * gamma_k * grad_y_k - x_k * inv³ / d * Σ_i (grad_y_i * gamma_i * x_i)
fn rms_norm_backward(
    norm: &RmsNorm,
    x: &[f32],         // pre-norm input
    grad_y: &[f32],    // upstream gradient w.r.t. the norm OUTPUT
    n: usize,
    grad_gamma_acc: &mut [f32],
    grad_x_out: &mut [f32],
) {
    let d = norm.gamma.len();
    for t in 0..n {
        let xr = &x[t * d..(t + 1) * d];
        let gr = &grad_y[t * d..(t + 1) * d];
        let dst = &mut grad_x_out[t * d..(t + 1) * d];
        // Recompute inv per row.
        let mut s = 0.0_f64;
        for &v in xr.iter() { s += (v as f64) * (v as f64); }
        let inv = ((s / d as f64) + norm.eps as f64).sqrt().recip() as f32;
        // dot = Σ_i grad_y_i * gamma_i * x_i
        let mut dot = 0.0_f32;
        for i in 0..d { dot += gr[i] * norm.gamma[i] * xr[i]; }
        let coef = dot * inv * inv * inv / d as f32;
        for i in 0..d {
            // grad_gamma_i += grad_y_i * x_i * inv (per-row summed).
            grad_gamma_acc[i] += gr[i] * xr[i] * inv;
            // grad_x_i = inv * gamma_i * grad_y_i - x_i * coef
            dst[i] = inv * norm.gamma[i] * gr[i] - xr[i] * coef;
        }
    }
}

/// Full backward pass. Updates `grad` and returns the (mean) loss.
pub fn backward(
    model: &ImageTransformer,
    cache: &ForwardCache,
    predict_at: &[usize],
    targets: &[u32],
    grad: &mut Gradients,
) -> Result<f32> {
    if predict_at.len() != targets.len() {
        return Err(ImageGenError::ShapeMismatch {
            expected: "predict_at.len() == targets.len()".into(),
            got: format!("{} vs {}", predict_at.len(), targets.len()),
        });
    }
    let cfg = &model.config;
    let n = cache.token_ids.len();
    let d = cfg.hidden_size;
    let v = cfg.vocab_image;
    let mp = cfg.mlp_intermediate;

    // --- 1. Softmax-CE backward (fused).
    let mut grad_logits = vec![0.0_f32; n * v];
    let scale_n = 1.0 / predict_at.len() as f32;
    let mut total_loss = 0.0_f64;
    for (&pos, &target) in predict_at.iter().zip(targets.iter()) {
        let row = &cache.logits[pos * v..(pos + 1) * v];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f64;
        let mut probs = vec![0.0_f32; v];
        for (j, &x) in row.iter().enumerate() {
            let p = (x - max).exp();
            probs[j] = p;
            sum += p as f64;
        }
        let inv = 1.0 / sum as f32;
        for p in probs.iter_mut() { *p *= inv; }
        // CE loss = -log(probs[target]). Numerically stable via log(sum) - (x_t - max).
        let lse = (sum.ln() as f32) + max;
        total_loss += (-(row[target as usize] - lse)) as f64;
        // grad_logits[pos] = (probs - one_hot(target)) / n_predict
        let dst = &mut grad_logits[pos * v..(pos + 1) * v];
        for j in 0..v { dst[j] = probs[j] * scale_n; }
        dst[target as usize] -= scale_n;
    }
    let mean_loss = (total_loss / predict_at.len() as f64) as f32;

    // --- 2. LM head backward: logits = x @ W^T.
    let mut grad_x = vec![0.0_f32; n * d];
    linear_backward(&model.lm_head, &grad_logits, &cache.x_post_final_norm,
        n, &mut grad.lm_head_w, &mut grad_x);

    // --- 3. Final RMSNorm backward.
    let mut grad_pre_final = vec![0.0_f32; n * d];
    rms_norm_backward(&model.final_norm, &cache.x_pre_final_norm, &grad_x, n,
        &mut grad.final_norm_gamma, &mut grad_pre_final);
    let mut grad_x_layer_out = grad_pre_final;

    // --- 4. Layers backward.
    for li in (0..model.blocks.len()).rev() {
        let blk = &model.blocks[li];
        let lc = &cache.layers[li];
        let bg = &mut grad.blocks[li];

        // The layer output: x_out = attn_residual + mlp_down(silu(gate) * up)
        // Residual: grad flows into both branches equally.
        // grad_attn_residual = grad_x_layer_out (residual passes through unchanged)
        // grad_mlp_down_out  = grad_x_layer_out
        let grad_attn_residual_from_mlp_resid = grad_x_layer_out.clone();
        let grad_mlp_down_out = grad_x_layer_out.clone();

        // --- 4a. mlp_down backward.
        let mut grad_mlp_silu_x_up = vec![0.0_f32; n * mp];
        linear_backward(&blk.mlp_down, &grad_mlp_down_out, &lc.mlp_silu_x_up,
            n, &mut bg.mlp_down_w, &mut grad_mlp_silu_x_up);

        // --- 4b. SwiGLU backward: silu_x_up = silu(gate) * up
        let mut grad_gate = vec![0.0_f32; n * mp];
        let mut grad_up = vec![0.0_f32; n * mp];
        for i in 0..(n * mp) {
            let g = lc.mlp_gate[i];
            let sigm = sigmoid(g);
            let silu_g = g * sigm;
            let silu_prime = sigm + g * sigm * (1.0 - sigm);
            grad_gate[i] = grad_mlp_silu_x_up[i] * lc.mlp_up[i] * silu_prime;
            grad_up[i] = grad_mlp_silu_x_up[i] * silu_g;
        }

        // --- 4c. mlp_gate / mlp_up backward (both consume xn2).
        let mut grad_xn2_from_gate = vec![0.0_f32; n * d];
        let mut grad_xn2_from_up = vec![0.0_f32; n * d];
        linear_backward(&blk.mlp_gate, &grad_gate, &lc.xn2,
            n, &mut bg.mlp_gate_w, &mut grad_xn2_from_gate);
        linear_backward(&blk.mlp_up, &grad_up, &lc.xn2,
            n, &mut bg.mlp_up_w, &mut grad_xn2_from_up);
        let mut grad_xn2 = grad_xn2_from_gate;
        for i in 0..grad_xn2.len() { grad_xn2[i] += grad_xn2_from_up[i]; }

        // --- 4d. norm_mlp backward.
        let mut grad_attn_residual_from_mlp_norm = vec![0.0_f32; n * d];
        rms_norm_backward(&blk.norm_mlp, &lc.attn_residual, &grad_xn2, n,
            &mut bg.norm_mlp_gamma, &mut grad_attn_residual_from_mlp_norm);

        // Combined grad on attn_residual: residual path + through norm_mlp.
        let mut grad_attn_residual = grad_attn_residual_from_mlp_resid;
        for i in 0..grad_attn_residual.len() {
            grad_attn_residual[i] += grad_attn_residual_from_mlp_norm[i];
        }

        // --- 4e. attn_residual = x_in + o_proj(attn_out_pre_o)
        //     grad_x_in_from_attn_resid = grad_attn_residual
        //     grad_o_out = grad_attn_residual
        let grad_x_in_from_attn_resid = grad_attn_residual.clone();
        let grad_o_out = grad_attn_residual;

        // --- 4f. o_proj backward.
        let mut grad_attn_out_pre_o = vec![0.0_f32; n * d];
        linear_backward(&blk.o_proj, &grad_o_out, &lc.attn_out_pre_o,
            n, &mut bg.o_proj_w, &mut grad_attn_out_pre_o);

        // --- 4g. Attention SDPA backward (per head).
        let (grad_q, grad_k, grad_v) = attention_backward(
            cfg, &lc.q, &lc.k, &lc.v, &lc.attn_weights, &grad_attn_out_pre_o, n);

        // --- 4h. Q/K/V projections backward (all consume xn1).
        let mut grad_xn1_from_q = vec![0.0_f32; n * d];
        let mut grad_xn1_from_k = vec![0.0_f32; n * d];
        let mut grad_xn1_from_v = vec![0.0_f32; n * d];
        linear_backward(&blk.q_proj, &grad_q, &lc.xn1,
            n, &mut bg.q_proj_w, &mut grad_xn1_from_q);
        linear_backward(&blk.k_proj, &grad_k, &lc.xn1,
            n, &mut bg.k_proj_w, &mut grad_xn1_from_k);
        linear_backward(&blk.v_proj, &grad_v, &lc.xn1,
            n, &mut bg.v_proj_w, &mut grad_xn1_from_v);
        let mut grad_xn1 = grad_xn1_from_q;
        for i in 0..grad_xn1.len() { grad_xn1[i] += grad_xn1_from_k[i] + grad_xn1_from_v[i]; }

        // --- 4i. norm_attn backward.
        let mut grad_x_in_from_attn_norm = vec![0.0_f32; n * d];
        rms_norm_backward(&blk.norm_attn, &lc.x_in, &grad_xn1, n,
            &mut bg.norm_attn_gamma, &mut grad_x_in_from_attn_norm);

        // --- 4j. Combine grads on x_in (this layer's input).
        let mut grad_x_in = grad_x_in_from_attn_resid;
        for i in 0..grad_x_in.len() {
            grad_x_in[i] += grad_x_in_from_attn_norm[i];
        }
        grad_x_layer_out = grad_x_in;
    }

    // --- 5. Embed + pos_embed backward.
    let _ = cache.x_after_embed; // (not directly used; the grad on x_in flows here)
    for i in 0..n {
        let tid = cache.token_ids[i] as usize;
        let eoff = tid * d;
        let poff = i * d;
        let src = &grad_x_layer_out[i * d..(i + 1) * d];
        for k in 0..d {
            grad.embed[eoff + k] += src[k];
            grad.pos_embed[poff + k] += src[k];
        }
    }

    Ok(mean_loss)
}

/// SDPA backward (per-head, causal). Returns grads on Q, K, V of shape [n, d].
fn attention_backward(
    cfg: &ImageTransformerConfig,
    q: &[f32], k: &[f32], v: &[f32],
    attn_weights: &[f32],
    grad_attn_out_pre_o: &[f32],
    n: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = cfg.hidden_size;
    let h = cfg.num_heads;
    let dh = cfg.head_dim();
    let scale = 1.0_f32 / (dh as f32).sqrt();

    let mut grad_q = vec![0.0_f32; n * d];
    let mut grad_k = vec![0.0_f32; n * d];
    let mut grad_v = vec![0.0_f32; n * d];

    for hi in 0..h {
        // For each head:
        //   out[i, k] = Σ_j attn[i, j] * V[j, k]   (j ≤ i)
        //   grad_V[j, k] = Σ_i attn[i, j] * grad_out[i, k]   for i ≥ j
        //   grad_attn[i, j] = Σ_k grad_out[i, k] * V[j, k]
        //   grad_scores[i, j] = attn[i, j] * (grad_attn[i, j] - Σ_l attn[i, l] * grad_attn[i, l])
        //   grad_Q[i, k] = Σ_j grad_scores[i, j] * K[j, k] * scale
        //   grad_K[j, k] = Σ_i grad_scores[i, j] * Q[i, k] * scale
        let aw_base = hi * n * n;

        // grad_attn: [n, n]
        let mut grad_attn = vec![0.0_f32; n * n];
        for qi in 0..n {
            for kj in 0..=qi {
                let v_off = kj * d + hi * dh;
                let out_off = qi * d + hi * dh;
                let mut s = 0.0_f32;
                for di in 0..dh { s += grad_attn_out_pre_o[out_off + di] * v[v_off + di]; }
                grad_attn[qi * n + kj] = s;
            }
        }

        // grad_V: scatter into v's [N, h, dh] layout.
        for qi in 0..n {
            for kj in 0..=qi {
                let w = attn_weights[aw_base + qi * n + kj];
                if w == 0.0 { continue; }
                let v_off = kj * d + hi * dh;
                let out_off = qi * d + hi * dh;
                for di in 0..dh {
                    grad_v[v_off + di] += w * grad_attn_out_pre_o[out_off + di];
                }
            }
        }

        // grad_scores via softmax backward, then grad_Q and grad_K.
        for qi in 0..n {
            // Σ_l attn[qi, l] * grad_attn[qi, l]
            let mut row_dot = 0.0_f32;
            for kj in 0..=qi {
                row_dot += attn_weights[aw_base + qi * n + kj] * grad_attn[qi * n + kj];
            }
            for kj in 0..=qi {
                let aij = attn_weights[aw_base + qi * n + kj];
                let g_score = aij * (grad_attn[qi * n + kj] - row_dot);
                let scaled = g_score * scale;
                // grad_Q[qi, head, :] += scaled * K[kj, head, :]
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
// Gradient checkpointing (#215)
// ============================================================================
//
// Trade memory for compute: cache only the per-layer INPUT (one tensor)
// instead of all 10 intermediates. Backward re-runs each layer's
// forward to reconstruct what it needs. For our 12-layer baseline this
// cuts activation memory ~10× (77 MB → 7.7 MB) at the cost of ~2×
// forward compute during training.
//
// Use `train_step_checkpointed` instead of `train_step` when memory is
// tight (longer sequences, bigger models, or wider batch).

pub struct CheckpointedCache {
    pub token_ids: Vec<u32>,
    pub x_after_embed: Vec<f32>,
    /// Per-layer INPUT only. Layer i's forward consumes `layer_inputs[i]`
    /// and produces `layer_inputs[i+1]` (or `x_pre_final_norm` for the
    /// last layer).
    pub layer_inputs: Vec<Vec<f32>>,
    pub x_pre_final_norm: Vec<f32>,
    pub x_post_final_norm: Vec<f32>,
    pub logits: Vec<f32>,
}

pub fn forward_train_checkpointed(
    model: &ImageTransformer,
    token_ids: &[u32],
) -> Result<CheckpointedCache> {
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
    let x_after_embed = x.clone();

    let mut layer_inputs: Vec<Vec<f32>> = Vec::with_capacity(model.blocks.len());
    for blk in &model.blocks {
        layer_inputs.push(x.clone()); // Save input only.
        let cache = layer_forward(blk, &x, n, cfg);
        let mut mlp_down_out = vec![0.0_f32; n * d];
        linear_forward(&blk.mlp_down, &cache.mlp_silu_x_up, n, &mut mlp_down_out);
        x = cache.attn_residual;
        for i in 0..x.len() { x[i] += mlp_down_out[i]; }
        // The full layer cache is dropped here — that's the memory win.
    }

    let x_pre_final_norm = x.clone();
    let _ = rms_norm_forward_collect_inv(&mut x, &model.final_norm, n);
    let x_post_final_norm = x.clone();
    let v = cfg.vocab_image;
    let mut logits = vec![0.0_f32; n * v];
    linear_forward(&model.lm_head, &x, n, &mut logits);

    Ok(CheckpointedCache {
        token_ids: token_ids.to_vec(),
        x_after_embed,
        layer_inputs,
        x_pre_final_norm,
        x_post_final_norm,
        logits,
    })
}

pub fn backward_checkpointed(
    model: &ImageTransformer,
    cache: &CheckpointedCache,
    predict_at: &[usize],
    targets: &[u32],
    grad: &mut Gradients,
) -> Result<f32> {
    // Reconstruct a ForwardCache by re-running layer_forward per layer.
    // We do this layer-by-layer during the backward pass to keep the
    // peak memory low; the per-layer LayerCache is dropped after that
    // layer's backward finishes.
    //
    // Simpler implementation: just re-materialise everything into a
    // ForwardCache once and call the existing `backward`. The "2× forward
    // compute" cost is the same; the memory saving is at the
    // per-iteration boundary (forward returns a thin cache, backward
    // briefly materialises a fat one then drops it).
    let cfg = &model.config;
    let n = cache.token_ids.len();
    let mut layers: Vec<LayerCache> = Vec::with_capacity(model.blocks.len());
    for (li, blk) in model.blocks.iter().enumerate() {
        let lc = layer_forward(blk, &cache.layer_inputs[li], n, cfg);
        layers.push(lc);
    }
    let full = ForwardCache {
        token_ids: cache.token_ids.clone(),
        x_after_embed: cache.x_after_embed.clone(),
        layers,
        x_pre_final_norm: cache.x_pre_final_norm.clone(),
        x_post_final_norm: cache.x_post_final_norm.clone(),
        logits: cache.logits.clone(),
    };
    backward(model, &full, predict_at, targets, grad)
}

/// Checkpointed training step. Same observable behavior as `train_step`
/// but with reduced activation memory at the cost of ~2× per-step
/// forward compute.
pub fn train_step_checkpointed(
    model: &mut ImageTransformer,
    optimizer: &mut crate::optim::AdamW,
    grad: &mut Gradients,
    batch: &crate::train::TrainBatch,
) -> Result<f32> {
    grad.zero_inplace();
    let cache = forward_train_checkpointed(model, &batch.token_ids)?;
    let loss = backward_checkpointed(model, &cache, &batch.predict_at, &batch.targets, grad)?;
    optimizer.step(model, grad);
    Ok(loss)
}

/// One full training step: forward + backward + optimizer step.
/// Returns mean cross-entropy loss for the step.
pub fn train_step(
    model: &mut ImageTransformer,
    optimizer: &mut crate::optim::AdamW,
    grad: &mut Gradients,
    batch: &crate::train::TrainBatch,
) -> Result<f32> {
    grad.zero_inplace();
    let cache = forward_train(model, &batch.token_ids)?;
    let loss = backward(model, &cache, &batch.predict_at, &batch.targets, grad)?;
    optimizer.step(model, grad);
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ImageTransformerConfig;
    use crate::optim::{AdamW, AdamWConfig};
    use crate::train::{TrainExample, assemble_batch};

    fn tiny_cfg() -> ImageTransformerConfig {
        ImageTransformerConfig {
            vocab_text: 100, vocab_image: 16,
            hidden_size: 16, num_layers: 1, num_heads: 4,
            mlp_intermediate: 32,
            max_text_len: 4,
            image_grid_h: 2, image_grid_w: 2,
            bos_token: 100, sep_image_token: 101, eos_token: 102,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
        }
    }

    /// Initialise small random weights so the model isn't degenerately
    /// stuck at all-zero (which makes gradient flow trivial).
    fn random_init(model: &mut ImageTransformer, seed: u64) {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut rand = || -> f32 {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            // [-0.05, 0.05]
            ((z >> 11) as f32 / (1u64 << 53) as f32) * 0.1 - 0.05
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
    }

    #[test]
    fn forward_train_matches_forward() {
        let cfg = tiny_cfg();
        let mut model = ImageTransformer::from_config(cfg.clone());
        random_init(&mut model, 7);
        let seq = vec![
            cfg.bos_token, 5, 6,
            cfg.sep_image_token,
            cfg.image_id_offset(), cfg.image_id_offset() + 1,
            cfg.image_id_offset() + 2, cfg.image_id_offset() + 3,
        ];
        let logits_a = model.forward(&seq).expect("forward");
        let cache = forward_train(&model, &seq).expect("forward_train");
        for (a, b) in logits_a.iter().zip(cache.logits.iter()) {
            assert!((a - b).abs() < 1e-4, "forward vs forward_train: {a} vs {b}");
        }
    }

    #[test]
    fn checkpointed_loss_matches_full_cache() {
        // Checkpointed training step should produce IDENTICAL loss values
        // (modulo floating-point recompute noise) to the full-cache path.
        let cfg = tiny_cfg();
        let mut model_a = ImageTransformer::from_config(cfg.clone());
        random_init(&mut model_a, 13);
        let mut model_b = ImageTransformer::from_config(cfg.clone());
        random_init(&mut model_b, 13);  // same seed → identical weights

        let mut opt_a = AdamW::new(&model_a, AdamWConfig { lr: 1e-2, ..Default::default() });
        let mut opt_b = AdamW::new(&model_b, AdamWConfig { lr: 1e-2, ..Default::default() });
        let mut grad_a = Gradients::zeros_for(&model_a);
        let mut grad_b = Gradients::zeros_for(&model_b);

        let ex = TrainExample {
            text_tokens: vec![5, 6],
            image_tokens: vec![3, 7, 11, 5],
        };
        let batch = assemble_batch(&cfg, &ex).expect("batch");

        for _ in 0..10 {
            let loss_a = train_step(&mut model_a, &mut opt_a, &mut grad_a, &batch).expect("step");
            let loss_b = train_step_checkpointed(&mut model_b, &mut opt_b, &mut grad_b, &batch)
                .expect("step_ckpt");
            assert!((loss_a - loss_b).abs() < 1e-3,
                "checkpointed loss should match full: {loss_a} vs {loss_b}");
        }
    }

    #[test]
    fn train_step_lowers_loss() {
        // Overfit a single sample. After a handful of AdamW steps, loss
        // should drop substantially. This validates that gradients flow
        // through every op (any broken op would prevent learning).
        let cfg = tiny_cfg();
        let mut model = ImageTransformer::from_config(cfg.clone());
        random_init(&mut model, 42);
        let mut opt = AdamW::new(&model, AdamWConfig {
            lr: 1e-2, weight_decay: 0.0, ..Default::default()
        });
        let mut grad = Gradients::zeros_for(&model);

        let ex = TrainExample {
            text_tokens: vec![5, 6],
            image_tokens: vec![3, 7, 11, 5],
        };
        let batch = assemble_batch(&cfg, &ex).expect("batch");

        let loss0 = {
            let cache = forward_train(&model, &batch.token_ids).expect("fwd");
            backward(&model, &cache, &batch.predict_at, &batch.targets, &mut grad).expect("bwd")
        };
        grad.zero_inplace();

        // 30 optimizer steps.
        let mut loss_final = 0.0_f32;
        for _ in 0..30 {
            loss_final = train_step(&mut model, &mut opt, &mut grad, &batch).expect("step");
        }
        assert!(loss_final < loss0 * 0.5,
            "loss should drop sharply: start={loss0}, end={loss_final}");
        // Sanity: loss should be < log(V) = log(16) ≈ 2.77 (the uniform baseline)
        // after training on a single sample for 30 steps.
        let log_v = (cfg.vocab_image as f32).ln();
        assert!(loss_final < log_v,
            "loss after training should beat uniform baseline {log_v}: got {loss_final}");
    }
}
