//! Mixture-of-Experts FFN with K-of-N expert routing.
//!
//! Replaces the dense SwiGLU MLP with N small experts, of which only the
//! top-K activate per token. Inspired by Switch Transformer / GShard.
//!
//! Layout per layer:
//!   Router:  Linear(hidden, num_experts)  — produces per-token expert scores.
//!   Experts: N × {gate: Linear(hidden, expert_intermediate),
//!                 up:   Linear(hidden, expert_intermediate),
//!                 down: Linear(expert_intermediate, hidden)}
//!
//! Per-token forward:
//!   1. logits = router(x)                       [n_experts]
//!   2. probs = softmax(logits)
//!   3. top_k_idx, top_k_w = topk_renorm(probs, K)
//!   4. y = Σ_{i∈top_k_idx} top_k_w[i] · expert[i](x)   (SwiGLU per expert)
//!
//! Per-batch (for the load-balancing auxiliary loss):
//!   - mean_prob_per_expert = mean over tokens of probs[expert]
//!   - frac_assigned_per_expert = mean over tokens of (1 if top-K, else 0)
//!   - aux_loss = N · Σ (mean_prob × frac_assigned)
//!   This pushes the router toward uniform expert usage.
//!
//! Sizing target (matched to the dense baseline FFN):
//!   Dense  : 3 · hidden · mlp_intermediate                        params
//!   MoE    : N · 3 · hidden · expert_intermediate + hidden · N    params
//!   With N=16, expert_intermediate = mlp_intermediate / K (K=4),
//!   the MoE has ~4× the total FFN params (more capacity), but only
//!   K/N = 1/4 of them activate per token. Effective FLOPs are ~1/K = 25%
//!   of dense; capacity is up to 4× dense.

use crate::error::Result;
use crate::model::{ImageTransformerConfig, LinearNoBias, RmsNorm};

/// Configuration for the MoE FFN.
#[derive(Debug, Clone)]
pub struct MoeFfnConfig {
    pub hidden_size: usize,
    /// Total experts. 16 is a good default — small enough that router
    /// softmax stays cheap, large enough that K=4 routes have variety.
    pub num_experts: usize,
    /// Active experts per token.
    pub top_k: usize,
    /// Each expert's MLP intermediate width.
    pub expert_intermediate: usize,
    /// Load-balancing aux loss coefficient (added to the model's main
    /// loss). 0.01 is typical (Switch Transformer paper).
    pub aux_loss_coeff: f32,
}

impl MoeFfnConfig {
    /// Matched to the dense baseline (`mlp_intermediate = 1408`,
    /// `hidden = 512`). 16 experts × intermediate=352, top-4 routing.
    /// Total expert params ≈ 4× dense FFN; per-token compute ≈ 1/4 dense.
    pub fn baseline_for(hidden_size: usize, dense_mlp_intermediate: usize) -> Self {
        Self {
            hidden_size,
            num_experts: 16,
            top_k: 4,
            expert_intermediate: dense_mlp_intermediate / 4,
            aux_loss_coeff: 0.01,
        }
    }
}

/// One SwiGLU expert. Shape-identical to the existing dense FFN inside
/// `ImageBlock`, just narrower (`expert_intermediate` vs `mlp_intermediate`).
pub struct MoeExpert {
    pub gate: LinearNoBias,
    pub up: LinearNoBias,
    pub down: LinearNoBias,
}

impl MoeExpert {
    pub fn new(hidden: usize, intermediate: usize) -> Self {
        Self {
            gate: LinearNoBias::new(hidden, intermediate),
            up:   LinearNoBias::new(hidden, intermediate),
            down: LinearNoBias::new(intermediate, hidden),
        }
    }
}

/// MoE FFN: router + N experts.
pub struct MoeFfn {
    pub config: MoeFfnConfig,
    pub router: LinearNoBias,   // Linear(hidden, num_experts)
    pub experts: Vec<MoeExpert>,
}

impl MoeFfn {
    pub fn from_config(config: MoeFfnConfig) -> Self {
        let router = LinearNoBias::new(config.hidden_size, config.num_experts);
        let experts: Vec<MoeExpert> = (0..config.num_experts)
            .map(|_| MoeExpert::new(config.hidden_size, config.expert_intermediate))
            .collect();
        Self { config, router, experts }
    }

    /// Total parameters in the MoE FFN.
    pub fn param_count(&self) -> usize {
        let cfg = &self.config;
        let d = cfg.hidden_size;
        let m = cfg.expert_intermediate;
        let router = d * cfg.num_experts;
        let per_expert = d * m + d * m + m * d;
        router + cfg.num_experts * per_expert
    }
}

/// Output of `moe_forward`.
pub struct MoeOut {
    /// `[n_tokens, hidden_size]` — the combined expert outputs.
    pub out: Vec<f32>,
    /// `[n_tokens, top_k]` — per-token selected expert indices.
    pub picks: Vec<Vec<usize>>,
    /// `[n_tokens, top_k]` — per-token renormalised expert weights
    /// (parallel to `picks`, sums to 1 along the K axis).
    pub weights: Vec<Vec<f32>>,
    /// Load-balancing auxiliary loss. Caller adds this (× coeff) to the
    /// total training loss.
    pub aux_loss: f32,
}

#[inline]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// Naive linear: y[n, out] = x[n, in] · W^T.
fn linear_forward(w: &LinearNoBias, x: &[f32], n: usize, y: &mut [f32]) {
    let in_f = w.in_features;
    let out_f = w.out_features;
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

/// MoE forward.
///
/// `x`: `[n_tokens, hidden_size]`.
pub fn moe_forward(ffn: &MoeFfn, x: &[f32], n: usize) -> Result<MoeOut> {
    let cfg = &ffn.config;
    let d = cfg.hidden_size;
    let m = cfg.expert_intermediate;
    let n_e = cfg.num_experts;
    let k = cfg.top_k;
    debug_assert_eq!(x.len(), n * d);
    debug_assert!(k <= n_e, "top_k must be ≤ num_experts");

    // 1. Router: produce per-token expert scores.
    let mut logits = vec![0.0_f32; n * n_e];
    linear_forward(&ffn.router, x, n, &mut logits);

    // 2. Softmax across experts per token.
    let mut probs = vec![0.0_f32; n * n_e];
    for t in 0..n {
        let row = &logits[t * n_e..(t + 1) * n_e];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        let dst = &mut probs[t * n_e..(t + 1) * n_e];
        for (i, &v) in row.iter().enumerate() {
            let p = (v - max).exp();
            dst[i] = p;
            sum += p;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in dst.iter_mut() { *p *= inv; }
        }
    }

    // 3. Top-K selection per token + renormalize.
    let mut picks: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(n);
    for t in 0..n {
        let row = &probs[t * n_e..(t + 1) * n_e];
        // Partial-sort into descending order. n_e is small (~16) so a
        // full sort is fine.
        let mut idx: Vec<usize> = (0..n_e).collect();
        idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top_idx: Vec<usize> = idx[..k].to_vec();
        let top_p: Vec<f32> = top_idx.iter().map(|&i| row[i]).collect();
        let sum: f32 = top_p.iter().sum();
        let top_w: Vec<f32> = if sum > 0.0 {
            top_p.iter().map(|p| p / sum).collect()
        } else {
            vec![1.0 / k as f32; k] // degenerate fallback
        };
        picks.push(top_idx);
        weights.push(top_w);
    }

    // 4. Per-token: run the chosen K experts, combine weighted.
    let mut out = vec![0.0_f32; n * d];
    // Per-token expert evaluation. For each (token, expert) pair we
    // compute a single SwiGLU forward on that token's hidden vector.
    // Naive but correct.
    let mut gate_buf = vec![0.0_f32; m];
    let mut up_buf = vec![0.0_f32; m];
    let mut comb = vec![0.0_f32; m];
    let mut down_buf = vec![0.0_f32; d];
    let x_slice = |t: usize| -> &[f32] { &x[t * d..(t + 1) * d] };
    for t in 0..n {
        let xt = x_slice(t);
        for ki in 0..k {
            let e = picks[t][ki];
            let w = weights[t][ki];
            let exp = &ffn.experts[e];
            // gate = exp.gate(xt)
            for o in 0..m {
                let wr = &exp.gate.weight[o * d..(o + 1) * d];
                let mut s = 0.0_f32;
                for i in 0..d { s += xt[i] * wr[i]; }
                gate_buf[o] = s;
            }
            // up = exp.up(xt)
            for o in 0..m {
                let wr = &exp.up.weight[o * d..(o + 1) * d];
                let mut s = 0.0_f32;
                for i in 0..d { s += xt[i] * wr[i]; }
                up_buf[o] = s;
            }
            // comb = silu(gate) * up
            for i in 0..m { comb[i] = silu(gate_buf[i]) * up_buf[i]; }
            // down = exp.down(comb)
            for o in 0..d {
                let wr = &exp.down.weight[o * m..(o + 1) * m];
                let mut s = 0.0_f32;
                for i in 0..m { s += comb[i] * wr[i]; }
                down_buf[o] = s;
            }
            // Combined: out[t] += w * down
            let dst = &mut out[t * d..(t + 1) * d];
            for o in 0..d { dst[o] += w * down_buf[o]; }
        }
    }

    // 5. Load-balancing aux loss (Switch Transformer formulation).
    //   mean_p[e] = (1/n) · Σ_t probs[t, e]
    //   frac[e]   = (1/n) · #{t : e ∈ picks[t]}
    //   aux_loss  = N · Σ_e (mean_p[e] · frac[e])
    let mut mean_p = vec![0.0_f32; n_e];
    let mut frac = vec![0.0_f32; n_e];
    for t in 0..n {
        for e in 0..n_e { mean_p[e] += probs[t * n_e + e]; }
        for &e in &picks[t] { frac[e] += 1.0; }
    }
    let inv_n = 1.0 / n as f32;
    for e in 0..n_e {
        mean_p[e] *= inv_n;
        frac[e] *= inv_n;
    }
    let mut aux = 0.0_f32;
    for e in 0..n_e { aux += mean_p[e] * frac[e]; }
    let aux_loss = n_e as f32 * aux;

    Ok(MoeOut { out, picks, weights, aux_loss })
}

// ============================================================================
// Backward pass
// ============================================================================
//
// Layout of the cached intermediates per (token t, slot k = 0..K-1):
//   - router logits, post-softmax probs, picks, weights (renormalised)
//   - top_p (raw probs of the chosen experts, pre-renorm) + top_sum (Σ top_p)
//   - per slot: pre-SiLU gate output, up output, silu(gate)·up, expert(x_t)
//
// Gradient flow:
//   y_t = Σ_k w_k · ŷ_k                                  (combined expert output)
//   ŷ_k = down_e(silu(gate_e(x_t)) · up_e(x_t))         (SwiGLU)
//   w_k = p_k / Σ_j p_j                                  (renormalised over picks)
//   p_e = softmax(router(x_t))[e]                        (post-softmax)
//
// We derive d/dw_k → d/dp_k → d/dlogits[t,e] → d/dx_t via standard softmax /
// linear backward. The router gradient also receives a contribution from the
// load-balancing aux loss (treating `frac` as constant — Switch Transformer
// convention).

/// Per-token, per-slot cached intermediates needed for backward.
#[derive(Clone)]
pub struct MoeSlotCache {
    pub gate_pre: Vec<f32>,    // [m] pre-SiLU
    pub up_pre:   Vec<f32>,    // [m]
    pub combo:    Vec<f32>,    // [m] silu(gate) · up
    pub y_hat:    Vec<f32>,    // [d] expert(x_t)
}

/// Full cache for one MoE forward pass — input plus everything backward needs.
pub struct MoeFwdCache {
    pub x: Vec<f32>,               // [n, d]
    pub logits: Vec<f32>,          // [n, n_e] raw router output
    pub probs: Vec<f32>,           // [n, n_e] post-softmax
    pub picks: Vec<Vec<usize>>,    // [n][k]
    pub weights: Vec<Vec<f32>>,    // [n][k] renormalised
    pub top_p: Vec<Vec<f32>>,      // [n][k] raw probs at picks (pre-renorm)
    pub top_sum: Vec<f32>,         // [n] Σ_{k} top_p[t,k]
    pub slots: Vec<Vec<MoeSlotCache>>,  // [n][k]
    pub aux_loss: f32,
    pub frac: Vec<f32>,            // [n_e] expert assignment fraction
    pub out: MoeOut,
}

/// Gradient buffer for one expert.
pub struct MoeExpertGrad {
    pub gate_w: Vec<f32>,
    pub up_w:   Vec<f32>,
    pub down_w: Vec<f32>,
}

impl MoeExpertGrad {
    pub fn zeros_for(e: &MoeExpert) -> Self {
        Self {
            gate_w: vec![0.0; e.gate.weight.len()],
            up_w:   vec![0.0; e.up.weight.len()],
            down_w: vec![0.0; e.down.weight.len()],
        }
    }
    pub fn zero_inplace(&mut self) {
        for v in self.gate_w.iter_mut() { *v = 0.0; }
        for v in self.up_w.iter_mut() { *v = 0.0; }
        for v in self.down_w.iter_mut() { *v = 0.0; }
    }
}

/// Gradient buffer for an MoE FFN (router + N experts).
pub struct MoeFfnGrad {
    pub router_w: Vec<f32>,
    pub experts: Vec<MoeExpertGrad>,
}

impl MoeFfnGrad {
    pub fn zeros_for(ffn: &MoeFfn) -> Self {
        Self {
            router_w: vec![0.0; ffn.router.weight.len()],
            experts: ffn.experts.iter().map(MoeExpertGrad::zeros_for).collect(),
        }
    }
    pub fn zero_inplace(&mut self) {
        for v in self.router_w.iter_mut() { *v = 0.0; }
        for e in self.experts.iter_mut() { e.zero_inplace(); }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// Training-time forward: same outputs as `moe_forward` plus the cache.
pub fn moe_forward_train(ffn: &MoeFfn, x: &[f32], n: usize) -> Result<MoeFwdCache> {
    let cfg = &ffn.config;
    let d = cfg.hidden_size;
    let m = cfg.expert_intermediate;
    let n_e = cfg.num_experts;
    let k = cfg.top_k;
    debug_assert_eq!(x.len(), n * d);

    let mut logits = vec![0.0_f32; n * n_e];
    linear_forward(&ffn.router, x, n, &mut logits);

    let mut probs = vec![0.0_f32; n * n_e];
    for t in 0..n {
        let row = &logits[t * n_e..(t + 1) * n_e];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        let dst = &mut probs[t * n_e..(t + 1) * n_e];
        for (i, &v) in row.iter().enumerate() {
            let p = (v - max).exp();
            dst[i] = p;
            sum += p;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in dst.iter_mut() { *p *= inv; }
        }
    }

    let mut picks: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut top_p: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut top_sum: Vec<f32> = Vec::with_capacity(n);
    for t in 0..n {
        let row = &probs[t * n_e..(t + 1) * n_e];
        let mut idx: Vec<usize> = (0..n_e).collect();
        idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top_idx: Vec<usize> = idx[..k].to_vec();
        let tp: Vec<f32> = top_idx.iter().map(|&i| row[i]).collect();
        let sum: f32 = tp.iter().sum();
        let tw: Vec<f32> = if sum > 0.0 {
            tp.iter().map(|p| p / sum).collect()
        } else {
            vec![1.0 / k as f32; k]
        };
        picks.push(top_idx);
        top_p.push(tp);
        top_sum.push(if sum > 0.0 { sum } else { 1.0 });
        weights.push(tw);
    }

    let mut out = vec![0.0_f32; n * d];
    let mut slots: Vec<Vec<MoeSlotCache>> = Vec::with_capacity(n);
    for t in 0..n {
        let xt = &x[t * d..(t + 1) * d];
        let mut token_slots: Vec<MoeSlotCache> = Vec::with_capacity(k);
        for ki in 0..k {
            let e = picks[t][ki];
            let w = weights[t][ki];
            let exp = &ffn.experts[e];
            let mut gate = vec![0.0_f32; m];
            let mut up = vec![0.0_f32; m];
            for o in 0..m {
                let wr = &exp.gate.weight[o * d..(o + 1) * d];
                let mut s = 0.0_f32;
                for i in 0..d { s += xt[i] * wr[i]; }
                gate[o] = s;
            }
            for o in 0..m {
                let wr = &exp.up.weight[o * d..(o + 1) * d];
                let mut s = 0.0_f32;
                for i in 0..d { s += xt[i] * wr[i]; }
                up[o] = s;
            }
            let mut combo = vec![0.0_f32; m];
            for i in 0..m { combo[i] = silu(gate[i]) * up[i]; }
            let mut y_hat = vec![0.0_f32; d];
            for o in 0..d {
                let wr = &exp.down.weight[o * m..(o + 1) * m];
                let mut s = 0.0_f32;
                for i in 0..m { s += combo[i] * wr[i]; }
                y_hat[o] = s;
            }
            let dst = &mut out[t * d..(t + 1) * d];
            for o in 0..d { dst[o] += w * y_hat[o]; }
            token_slots.push(MoeSlotCache { gate_pre: gate, up_pre: up, combo, y_hat });
        }
        slots.push(token_slots);
    }

    // Aux loss + frac (parallel to moe_forward).
    let mut mean_p = vec![0.0_f32; n_e];
    let mut frac = vec![0.0_f32; n_e];
    for t in 0..n {
        for e in 0..n_e { mean_p[e] += probs[t * n_e + e]; }
        for &e in &picks[t] { frac[e] += 1.0; }
    }
    let inv_n = 1.0 / n as f32;
    for e in 0..n_e { mean_p[e] *= inv_n; frac[e] *= inv_n; }
    let mut aux = 0.0_f32;
    for e in 0..n_e { aux += mean_p[e] * frac[e]; }
    let aux_loss = n_e as f32 * aux;

    let mout = MoeOut {
        out: out.clone(),
        picks: picks.clone(),
        weights: weights.clone(),
        aux_loss,
    };

    Ok(MoeFwdCache {
        x: x.to_vec(), logits, probs, picks, weights, top_p, top_sum,
        slots, aux_loss, frac, out: mout,
    })
}

/// MoE backward.
///
/// - `grad_out`: `[n, d]`, gradient w.r.t. `cache.out.out`.
/// - `grad_aux_scalar`: gradient w.r.t. `cache.out.aux_loss` (usually
///   `aux_loss_coeff` if the caller adds `coeff · aux_loss` to the
///   training objective; pass 0.0 to disable).
/// - `grad`: accumulator for router + expert weight grads (must be
///   shape-matched via `MoeFfnGrad::zeros_for`).
/// - `grad_x_out`: `[n, d]` output gradient on the input.
pub fn moe_backward(
    ffn: &MoeFfn,
    cache: &MoeFwdCache,
    grad_out: &[f32],
    grad_aux_scalar: f32,
    grad: &mut MoeFfnGrad,
    grad_x_out: &mut [f32],
) {
    let cfg = &ffn.config;
    let d = cfg.hidden_size;
    let m = cfg.expert_intermediate;
    let n_e = cfg.num_experts;
    let k = cfg.top_k;
    let n = cache.x.len() / d;
    debug_assert_eq!(grad_out.len(), n * d);
    debug_assert_eq!(grad_x_out.len(), n * d);
    for v in grad_x_out.iter_mut() { *v = 0.0; }

    // Per-token grad on logits accumulator (n × n_e), zeroed.
    let mut grad_logits = vec![0.0_f32; n * n_e];

    for t in 0..n {
        let xt = &cache.x[t * d..(t + 1) * d];
        let dy = &grad_out[t * d..(t + 1) * d];

        // grad w.r.t. renormalised weights w_k: grad_w[k] = Σ_o dy[o] · y_hat_k[o]
        // grad w.r.t. expert output y_hat_k: grad_y_hat_k = w_k · dy
        let mut grad_w = vec![0.0_f32; k];
        for ki in 0..k {
            let yh = &cache.slots[t][ki].y_hat;
            let mut s = 0.0_f32;
            for o in 0..d { s += dy[o] * yh[o]; }
            grad_w[ki] = s;
        }

        // For each slot, propagate grad through expert (SwiGLU + 3 linears).
        let mut grad_xt_total = vec![0.0_f32; d];
        for ki in 0..k {
            let e_idx = cache.picks[t][ki];
            let w = cache.weights[t][ki];
            let slot = &cache.slots[t][ki];
            let exp = &ffn.experts[e_idx];
            let eg = &mut grad.experts[e_idx];

            // grad_y_hat = w · dy   (length d)
            let mut grad_y_hat = vec![0.0_f32; d];
            for o in 0..d { grad_y_hat[o] = w * dy[o]; }

            // down: y_hat[o] = Σ_i combo[i] · down_w[o, i]
            //   grad_down_w[o, i] += grad_y_hat[o] · combo[i]
            //   grad_combo[i]      = Σ_o grad_y_hat[o] · down_w[o, i]
            let mut grad_combo = vec![0.0_f32; m];
            for o in 0..d {
                let wr = &exp.down.weight[o * m..(o + 1) * m];
                let gw = &mut eg.down_w[o * m..(o + 1) * m];
                let gy = grad_y_hat[o];
                for i in 0..m {
                    gw[i] += gy * slot.combo[i];
                    grad_combo[i] += gy * wr[i];
                }
            }

            // SwiGLU: combo[i] = silu(gate[i]) · up[i]
            //   grad_gate[i] = grad_combo[i] · up[i] · silu'(gate[i])
            //   grad_up[i]   = grad_combo[i] · silu(gate[i])
            //   silu'(x) = σ(x) + x·σ(x)·(1 − σ(x))
            let mut grad_gate = vec![0.0_f32; m];
            let mut grad_up   = vec![0.0_f32; m];
            for i in 0..m {
                let g = slot.gate_pre[i];
                let sg = sigmoid(g);
                let silu_g = g * sg;
                let silu_prime = sg + g * sg * (1.0 - sg);
                grad_gate[i] = grad_combo[i] * slot.up_pre[i] * silu_prime;
                grad_up[i]   = grad_combo[i] * silu_g;
            }

            // gate: gate[o] = Σ_i x[i] · gate_w[o, i]
            //   grad_gate_w[o, i] += grad_gate[o] · x[i]
            //   grad_x[i]         += Σ_o grad_gate[o] · gate_w[o, i]
            for o in 0..m {
                let wr = &exp.gate.weight[o * d..(o + 1) * d];
                let gw = &mut eg.gate_w[o * d..(o + 1) * d];
                let gg = grad_gate[o];
                for i in 0..d {
                    gw[i] += gg * xt[i];
                    grad_xt_total[i] += gg * wr[i];
                }
            }
            for o in 0..m {
                let wr = &exp.up.weight[o * d..(o + 1) * d];
                let gw = &mut eg.up_w[o * d..(o + 1) * d];
                let gu = grad_up[o];
                for i in 0..d {
                    gw[i] += gu * xt[i];
                    grad_xt_total[i] += gu * wr[i];
                }
            }
        }

        // Renorm backward: w_k = p_k / S  where S = Σ_j p_j (over picks).
        //   ∂w_a/∂p_b = δ_ab/S − w_a/S                    (a, b both in picks)
        //   grad_p_b  = (grad_w_b − Σ_a grad_w_a · w_a) / S
        let s_inv = 1.0 / cache.top_sum[t];
        let mut dot_gw_w = 0.0_f32;
        for ki in 0..k { dot_gw_w += grad_w[ki] * cache.weights[t][ki]; }
        let mut grad_top_p = vec![0.0_f32; k];
        for ki in 0..k {
            grad_top_p[ki] = (grad_w[ki] - dot_gw_w) * s_inv;
        }

        // Scatter into grad_probs[t, ·]: only the picked experts receive
        // gradient from this branch.
        let mut grad_probs_row = vec![0.0_f32; n_e];
        for ki in 0..k {
            grad_probs_row[cache.picks[t][ki]] = grad_top_p[ki];
        }

        // Aux-loss contribution to grad_probs:
        //   aux_loss = N · Σ_e mean_p[e] · frac[e]
        //   mean_p[e] = (1/n) Σ_t probs[t, e]
        //   ∂aux/∂probs[t, e] = (N/n) · frac[e]
        if grad_aux_scalar != 0.0 {
            let scale = grad_aux_scalar * n_e as f32 / n as f32;
            for e in 0..n_e { grad_probs_row[e] += scale * cache.frac[e]; }
        }

        // Softmax backward: grad_logits[t, e] = p[e] · (grad_p[e] − Σ_f p[f] · grad_p[f])
        let p_row = &cache.probs[t * n_e..(t + 1) * n_e];
        let mut p_dot_g = 0.0_f32;
        for e in 0..n_e { p_dot_g += p_row[e] * grad_probs_row[e]; }
        let glog_row = &mut grad_logits[t * n_e..(t + 1) * n_e];
        for e in 0..n_e {
            glog_row[e] = p_row[e] * (grad_probs_row[e] - p_dot_g);
        }

        // Stash this token's contribution to grad_x via experts.
        let gx_dst = &mut grad_x_out[t * d..(t + 1) * d];
        for i in 0..d { gx_dst[i] += grad_xt_total[i]; }
    }

    // Router backward: logits[t, e] = Σ_i x[t, i] · router_w[e, i]
    //   grad_router_w[e, i] += Σ_t grad_logits[t, e] · x[t, i]
    //   grad_x[t, i]        += Σ_e grad_logits[t, e] · router_w[e, i]
    for t in 0..n {
        let xt = &cache.x[t * d..(t + 1) * d];
        let gl = &grad_logits[t * n_e..(t + 1) * n_e];
        let gx = &mut grad_x_out[t * d..(t + 1) * d];
        for e in 0..n_e {
            let g = gl[e];
            if g == 0.0 { continue; }
            let wr = &ffn.router.weight[e * d..(e + 1) * d];
            let gw = &mut grad.router_w[e * d..(e + 1) * d];
            for i in 0..d {
                gw[i] += g * xt[i];
                gx[i] += g * wr[i];
            }
        }
    }
}

// ============================================================================
// MoE-flavored transformer: separate type because MoE requires new weights
// (router + N experts) that don't map to a dense checkpoint.
// ============================================================================

use klearu_diffusion::blas::sgemm_a_btrans;

#[inline]
fn linear_fwd(w: &LinearNoBias, x: &[f32], n: usize, y: &mut [f32]) {
    sgemm_a_btrans(n, w.out_features, w.in_features, x, &w.weight, y);
}

#[inline]
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

/// MoE-flavored ImageBlock: dense attention + MoE FFN.
pub struct MoeImageBlock {
    pub norm_attn: RmsNorm,
    pub q_proj: LinearNoBias,
    pub k_proj: LinearNoBias,
    pub v_proj: LinearNoBias,
    pub o_proj: LinearNoBias,
    pub norm_mlp: RmsNorm,
    pub moe: MoeFfn,
}

impl MoeImageBlock {
    pub fn new(model_cfg: &ImageTransformerConfig, moe_cfg: MoeFfnConfig) -> Self {
        let d = model_cfg.hidden_size;
        Self {
            norm_attn: RmsNorm::new(d, model_cfg.rms_norm_eps),
            q_proj: LinearNoBias::new(d, d),
            k_proj: LinearNoBias::new(d, d),
            v_proj: LinearNoBias::new(d, d),
            o_proj: LinearNoBias::new(d, d),
            norm_mlp: RmsNorm::new(d, model_cfg.rms_norm_eps),
            moe: MoeFfn::from_config(moe_cfg),
        }
    }
}

/// MoE-flavored transformer. Same I/O contract as `ImageTransformer`.
pub struct MoeImageTransformer {
    pub config: ImageTransformerConfig,
    pub moe_config: MoeFfnConfig,
    pub embed: Vec<f32>,
    pub pos_embed: Vec<f32>,
    pub blocks: Vec<MoeImageBlock>,
    pub final_norm: RmsNorm,
    pub lm_head: LinearNoBias,
}

impl MoeImageTransformer {
    pub fn from_config(config: ImageTransformerConfig, moe_config: MoeFfnConfig) -> Self {
        let d = config.hidden_size;
        let blocks: Vec<_> = (0..config.num_layers)
            .map(|_| MoeImageBlock::new(&config, moe_config.clone()))
            .collect();
        Self {
            embed: vec![0.0; config.unified_vocab_size() * d],
            pos_embed: vec![0.0; config.max_seq_len() * d],
            blocks,
            final_norm: RmsNorm::new(d, 1e-5),
            lm_head: LinearNoBias::new(d, config.vocab_image),
            config,
            moe_config,
        }
    }

    /// Forward pass. Returns `(logits, total_aux_loss)`. `total_aux_loss`
    /// is the sum across all blocks; multiply by `moe_config.aux_loss_coeff`
    /// before adding to the main CE loss during training.
    pub fn forward(&self, token_ids: &[u32]) -> Result<(Vec<f32>, f32)> {
        let cfg = &self.config;
        let n = token_ids.len();
        let d = cfg.hidden_size;
        if n == 0 || n > cfg.max_seq_len() {
            return Err(crate::error::ImageGenError::ShapeMismatch {
                expected: format!("1..={} tokens", cfg.max_seq_len()),
                got: format!("{n}"),
            });
        }
        let unified = cfg.unified_vocab_size();

        // Embed + positional.
        let mut x = vec![0.0_f32; n * d];
        for (i, &tid) in token_ids.iter().enumerate() {
            if (tid as usize) >= unified {
                return Err(crate::error::ImageGenError::ShapeMismatch {
                    expected: format!("token id < {unified}"),
                    got: format!("id={tid}"),
                });
            }
            let eoff = (tid as usize) * d;
            let poff = i * d;
            let dst = &mut x[i * d..(i + 1) * d];
            for k in 0..d {
                dst[k] = self.embed[eoff + k] + self.pos_embed[poff + k];
            }
        }

        let mut total_aux = 0.0_f32;
        for blk in &self.blocks {
            // Pre-attention norm.
            let mut xn1 = x.clone();
            rms_norm_inplace(&mut xn1, &blk.norm_attn, n);
            // Attention (dense, causal).
            let h = cfg.num_heads;
            let dh = cfg.head_dim();
            let scale = 1.0_f32 / (dh as f32).sqrt();
            let mut q = vec![0.0_f32; n * d];
            let mut k = vec![0.0_f32; n * d];
            let mut v = vec![0.0_f32; n * d];
            linear_fwd(&blk.q_proj, &xn1, n, &mut q);
            linear_fwd(&blk.k_proj, &xn1, n, &mut k);
            linear_fwd(&blk.v_proj, &xn1, n, &mut v);

            let mut attn_out = vec![0.0_f32; n * d];
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
                    let out_off = qi * d + hi * dh;
                    for kj in 0..=qi {
                        let v_off = kj * d + hi * dh;
                        let w = scores[kj];
                        if w == 0.0 { continue; }
                        for di in 0..dh { attn_out[out_off + di] += w * v[v_off + di]; }
                    }
                }
            }
            let mut o = vec![0.0_f32; n * d];
            linear_fwd(&blk.o_proj, &attn_out, n, &mut o);
            for i in 0..x.len() { x[i] += o[i]; }

            // Pre-MLP norm + MoE.
            let mut xn2 = x.clone();
            rms_norm_inplace(&mut xn2, &blk.norm_mlp, n);
            let moe_out = moe_forward(&blk.moe, &xn2, n)?;
            for i in 0..x.len() { x[i] += moe_out.out[i]; }
            total_aux += moe_out.aux_loss;
        }

        // Final norm + LM head.
        rms_norm_inplace(&mut x, &self.final_norm, n);
        let v = cfg.vocab_image;
        let mut logits = vec![0.0_f32; n * v];
        linear_fwd(&self.lm_head, &x, n, &mut logits);
        Ok((logits, total_aux))
    }

    /// Param count: dense attention (unchanged from ImageTransformer) +
    /// MoE FFN (typically ~4× dense FFN, but only K/N active per token).
    pub fn param_count(&self) -> usize {
        let cfg = &self.config;
        let d = cfg.hidden_size;
        let attn_per_block = 4 * d * d + 2 * d; // q,k,v,o + 2 RMS norms
        let moe_per_block = self.blocks[0].moe.param_count();
        let embed = cfg.unified_vocab_size() * d;
        let pos_embed = cfg.max_seq_len() * d;
        let lm_head = d * cfg.vocab_image;
        let final_norm = d;
        embed + pos_embed
            + (attn_per_block + moe_per_block) * cfg.num_layers
            + final_norm + lm_head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_count_baseline() {
        let cfg = MoeFfnConfig::baseline_for(512, 1408);
        let ffn = MoeFfn::from_config(cfg.clone());
        let p = ffn.param_count();
        // 16 experts × (2·512·352 + 352·512) + 512·16 ≈ 8.65M
        assert!(p > 5_000_000 && p < 15_000_000,
            "expected ~8M params, got {p}");
        // Dense equivalent: 3 · 512 · 1408 ≈ 2.16M.
        // MoE total is ~4× dense, by design (more capacity).
        let dense = 3 * 512 * 1408;
        assert!(p > dense * 3, "MoE total {p} should be >3× dense {dense}");
    }

    #[test]
    fn forward_shapes_and_softmax_sums() {
        let cfg = MoeFfnConfig {
            hidden_size: 32, num_experts: 4, top_k: 2,
            expert_intermediate: 16, aux_loss_coeff: 0.01,
        };
        let mut ffn = MoeFfn::from_config(cfg.clone());
        // Seed router weights with a small known pattern so we get
        // non-degenerate picks (zero weights → all experts tied).
        for (i, w) in ffn.router.weight.iter_mut().enumerate() {
            *w = ((i as f32) % 7.0 - 3.0) * 0.1;
        }
        // Seed expert weights so they don't all output zero.
        for (ei, exp) in ffn.experts.iter_mut().enumerate() {
            let mark = (ei as f32) * 0.01 + 0.01;
            for w in exp.gate.weight.iter_mut() { *w = mark; }
            for w in exp.up.weight.iter_mut() { *w = mark; }
            for w in exp.down.weight.iter_mut() { *w = mark; }
        }

        let n = 6;
        let x: Vec<f32> = (0..n * cfg.hidden_size)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();

        let r = moe_forward(&ffn, &x, n).expect("moe");
        assert_eq!(r.out.len(), n * cfg.hidden_size);
        assert!(r.out.iter().all(|x| x.is_finite()));
        assert_eq!(r.picks.len(), n);
        for ti in 0..n {
            assert_eq!(r.picks[ti].len(), cfg.top_k);
            assert_eq!(r.weights[ti].len(), cfg.top_k);
            let sum: f32 = r.weights[ti].iter().sum();
            assert!((sum - 1.0).abs() < 1e-4,
                "token {ti} top_k weights should sum to 1, got {sum}");
            // No duplicate picks.
            let mut sorted = r.picks[ti].clone();
            sorted.sort();
            for w in sorted.windows(2) {
                assert!(w[0] != w[1], "duplicate pick in {:?}", r.picks[ti]);
            }
        }
        assert!(r.aux_loss.is_finite() && r.aux_loss > 0.0,
            "aux_loss should be positive: {}", r.aux_loss);
    }

    #[test]
    fn moe_transformer_forward_runs() {
        let model_cfg = ImageTransformerConfig {
            vocab_text: 100, vocab_image: 16,
            hidden_size: 32, num_layers: 2, num_heads: 4,
            mlp_intermediate: 64, max_text_len: 4,
            image_grid_h: 2, image_grid_w: 2,
            bos_token: 100, sep_image_token: 101, eos_token: 102,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
        };
        let moe_cfg = MoeFfnConfig {
            hidden_size: 32, num_experts: 4, top_k: 2,
            expert_intermediate: 16, aux_loss_coeff: 0.01,
        };
        let model = MoeImageTransformer::from_config(model_cfg.clone(), moe_cfg);
        let seq = vec![
            model_cfg.bos_token,
            5, 6,
            model_cfg.sep_image_token,
            model_cfg.image_id_offset(),
            model_cfg.image_id_offset() + 1,
            model_cfg.image_id_offset() + 2,
            model_cfg.image_id_offset() + 3,
        ];
        let (logits, aux) = model.forward(&seq).expect("moe forward");
        assert_eq!(logits.len(), seq.len() * model_cfg.vocab_image);
        assert!(logits.iter().all(|x| x.is_finite()));
        assert!(aux.is_finite() && aux >= 0.0,
            "aux_loss should be finite & non-negative: {aux}");
    }

    #[test]
    fn moe_backward_overfits_single_target() {
        // Overfit a 2-token batch on a synthetic MSE target through the
        // MoE FFN. Loss should drop sharply — proves grads flow through
        // every op (router softmax, top-K renorm, all three expert
        // matmuls, SwiGLU).
        let cfg = MoeFfnConfig {
            hidden_size: 8, num_experts: 4, top_k: 2,
            expert_intermediate: 16, aux_loss_coeff: 0.0,  // disable aux for cleaner signal
        };
        let mut ffn = MoeFfn::from_config(cfg.clone());
        // Init non-trivially.
        let mut s = 0xDEADBEEFu64;
        let mut rand = |scale: f32| -> f32 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32 * 2.0 - 1.0) * scale
        };
        // Router stays small so we don't lock into one expert pick.
        for w in ffn.router.weight.iter_mut() { *w = rand(0.1); }
        // Expert weights initialised larger so the composed forward output
        // has non-trivial magnitude (Xavier-like for 8→16→8 stack).
        for e in ffn.experts.iter_mut() {
            for w in e.gate.weight.iter_mut() { *w = rand(0.5); }
            for w in e.up.weight.iter_mut()   { *w = rand(0.5); }
            for w in e.down.weight.iter_mut() { *w = rand(0.5); }
        }

        let n = 2;
        let x: Vec<f32> = (0..n * cfg.hidden_size)
            .map(|i| (i as f32 * 0.3).sin() * 0.5)
            .collect();
        let target: Vec<f32> = (0..n * cfg.hidden_size)
            .map(|i| if i % 2 == 0 { 0.7 } else { -0.3 })
            .collect();

        let lr = 0.3_f32;
        let mut loss_first = f32::NAN;
        let mut loss_last = 0.0;
        for step in 0..150 {
            let fc = moe_forward_train(&ffn, &x, n).expect("fwd");
            // MSE loss vs target.
            let mut loss = 0.0_f32;
            let mut grad_out = vec![0.0_f32; n * cfg.hidden_size];
            for i in 0..(n * cfg.hidden_size) {
                let diff = fc.out.out[i] - target[i];
                loss += diff * diff;
                grad_out[i] = 2.0 * diff / (n * cfg.hidden_size) as f32;
            }
            loss /= (n * cfg.hidden_size) as f32;
            if step == 0 { loss_first = loss; }
            loss_last = loss;

            let mut g = MoeFfnGrad::zeros_for(&ffn);
            let mut grad_x = vec![0.0_f32; n * cfg.hidden_size];
            moe_backward(&ffn, &fc, &grad_out, 0.0, &mut g, &mut grad_x);
            // SGD update.
            for (w, gw) in ffn.router.weight.iter_mut().zip(g.router_w.iter()) {
                *w -= lr * gw;
            }
            for (e, eg) in ffn.experts.iter_mut().zip(g.experts.iter()) {
                for (w, gw) in e.gate.weight.iter_mut().zip(eg.gate_w.iter()) { *w -= lr * gw; }
                for (w, gw) in e.up.weight.iter_mut().zip(eg.up_w.iter()) { *w -= lr * gw; }
                for (w, gw) in e.down.weight.iter_mut().zip(eg.down_w.iter()) { *w -= lr * gw; }
            }
        }
        assert!(loss_last < loss_first * 0.5,
            "MoE overfit failed: start={loss_first}, end={loss_last}");
    }

    #[test]
    fn aux_loss_minimised_at_uniform_routing() {
        // When the router output is uniform (all experts equally likely),
        // and top-K is forced, aux_loss = N · Σ (1/N · K/N) = K. With
        // imbalanced routing it grows beyond K (Switch Transformer's
        // proof: minimum aux_loss for uniform sampling = K).
        let cfg = MoeFfnConfig {
            hidden_size: 16, num_experts: 4, top_k: 2,
            expert_intermediate: 8, aux_loss_coeff: 0.01,
        };
        let ffn = MoeFfn::from_config(cfg.clone());
        // Zero router weights → uniform softmax → ties broken in index
        // order → ALL tokens pick the SAME top-K (e.g., [0, 1]).
        // This is the WORST imbalance — aux_loss should be N · (K/N · 1/N + …)
        // = N · 2 · (1/N · K/N) where the K experts get frac=1, the rest 0.
        // = 2 · (1/N · 1) = 2/N per active expert × N experts in sum.
        // Concretely: mean_p[0]=mean_p[1]=…=1/N for all; frac[0]=frac[1]=1, rest=0.
        // aux = N · Σ mean_p[e] · frac[e] = N · (1/N · 1 + 1/N · 1) = 2.
        // So at K=2, aux_loss = 2 = K. Verify.
        let n = 8;
        let x = vec![1.0_f32; n * cfg.hidden_size];
        let r = moe_forward(&ffn, &x, n).expect("moe");
        // K=2, so the lower bound on aux_loss is K=2 (achieved when
        // routing is perfectly uniform AND every token picks the same
        // K experts deterministically — our degenerate zero-weight case).
        assert!((r.aux_loss - cfg.top_k as f32).abs() < 0.5,
            "uniform routing should give aux_loss ≈ K = {}, got {}",
            cfg.top_k, r.aux_loss);
    }
}
