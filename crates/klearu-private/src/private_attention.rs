//! Private (2PC) attention forward pass.
//!
//! Since weights are public, Q/K/V projections are local.
//! Q is revealed to compute attention scores via Q·K.
//! Partial attention scores (Q_plain · K_share) are exchanged and
//! reconstructed so both parties apply identical softmax.
//! Weighted V sum uses public softmax weights on V shares (local).
//!
//! Qwen3.5 gated attention: q_proj is doubled (query + gate interleaved
//! per head), Q/K get per-head RMSNorm before RoPE, and output is gated
//! by sigmoid(gate). Both Q and K are revealed for norm, so attention
//! scores are fully public (no partial score exchange needed).

use klearu_llm::model::attention::Attention;
use klearu_llm::model::gated_deltanet::{DeltaNetState, GatedDeltaNet};
use klearu_llm::model::kv_cache::KvCache;
use klearu_llm::model::rope::RotaryEmbedding;
use klearu_mpc::fixed_point::{from_fixed, from_fixed64};
use klearu_mpc::linear::{
    shared_linear_forward, shared_linear_forward_64, shared_linear_forward_f32_input,
    shared_linear_forward_f32_input_64,
};
use klearu_mpc::transport::Transport;
use klearu_mpc::{SharedVec, SharedVec64};
use std::io;

/// In-place RMSNorm with (1 + weight) scaling (for Qwen3.5 Q/K norm).
fn rms_norm_one_plus(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for (xi, &wi) in x.iter_mut().zip(weight.iter()) {
        *xi = *xi * inv_rms * (1.0 + wi);
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reveal f32 shares: send this party's f32 values as bits, receive other party's,
/// reconstruct plaintext by adding.
fn reveal_f32(
    my_f32: &[f32],
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let bits: Vec<u32> = my_f32.iter().map(|&v| v.to_bits()).collect();
    transport.send_u32_slice(&bits)?;
    let other_bits = transport.recv_u32_slice(bits.len())?;
    Ok(my_f32
        .iter()
        .zip(other_bits.iter())
        .map(|(&my, &ob)| my + f32::from_bits(ob))
        .collect())
}

/// Compute softmax scores and weighted V sum for one head.
/// `scores` are fully public attention scores for this head over seq_len positions.
/// Accumulates weighted V into `output[head_offset..head_offset + head_dim]`.
fn softmax_weighted_v(
    scores: &[f32],
    kv_h: usize,
    head_dim: usize,
    head_offset: usize,
    kv_cache: &KvCache,
    output: &mut [f32],
) {
    let seq_len = scores.len();
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f32 = exp_scores.iter().sum();
    for s in &mut exp_scores {
        *s /= sum;
    }
    for t in 0..seq_len {
        let v_slice = kv_cache.v_at(kv_h, t);
        let w = exp_scores[t];
        for d in 0..head_dim {
            output[head_offset + d] += w * v_slice[d];
        }
    }
}

/// Private attention forward pass (Q16.16).
///
/// Handles both standard and Qwen3.5 gated attention.
///
/// Leakage: Q vectors, attention scores. For gated attention: also K vectors and gate values.
pub fn private_attention_forward(
    party: u8,
    attention: &Attention,
    x_share: &SharedVec,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    let hidden_size = attention.q_proj.in_features();
    let num_heads = attention.num_heads();
    let num_kv_heads = attention.num_kv_heads();
    let head_dim = attention.head_dim();
    let gqa_group_size = num_heads / num_kv_heads;
    let q_out = num_heads * head_dim;
    let kv_out = num_kv_heads * head_dim;

    // K/V projections (same for both paths)
    let k_share = shared_linear_forward(
        party, attention.k_proj.weights.as_raw_slice(), hidden_size, kv_out, x_share, &[], transport,
    )?;
    let v_share = shared_linear_forward(
        party, attention.v_proj.weights.as_raw_slice(), hidden_size, kv_out, x_share, &[], transport,
    )?;

    let v_cache_f32: Vec<f32> = v_share.0.iter().map(|&v| from_fixed(v)).collect();

    if attention.output_gate {
        // --- Gated attention path (Qwen3.5) ---

        // 1. Q projection (doubled output), de-interleave per head
        let q_out_doubled = q_out * 2;
        let q_gate_share = shared_linear_forward(
            party, attention.q_proj.weights.as_raw_slice(), hidden_size, q_out_doubled, x_share, &[], transport,
        )?;

        let mut q_vals = vec![0u32; q_out];
        let mut gate_vals = vec![0u32; q_out];
        for h in 0..num_heads {
            let src = h * head_dim * 2;
            let dst = h * head_dim;
            q_vals[dst..dst + head_dim].copy_from_slice(&q_gate_share.0[src..src + head_dim]);
            gate_vals[dst..dst + head_dim].copy_from_slice(&q_gate_share.0[src + head_dim..src + head_dim * 2]);
        }

        // 2. Reveal Q, apply per-head RMSNorm
        let q_f32: Vec<f32> = q_vals.iter().map(|&v| from_fixed(v)).collect();
        let mut q_plain = reveal_f32(&q_f32, transport)?;
        if let Some(ref qw) = attention.q_norm_weight {
            for h in 0..num_heads {
                let off = h * head_dim;
                rms_norm_one_plus(&mut q_plain[off..off + head_dim], qw, attention.qk_norm_eps);
            }
        }

        // 3. Reveal K, apply per-head RMSNorm
        let k_f32: Vec<f32> = k_share.0.iter().map(|&v| from_fixed(v)).collect();
        let mut k_plain = reveal_f32(&k_f32, transport)?;
        if let Some(ref kw) = attention.k_norm_weight {
            for h in 0..num_kv_heads {
                let off = h * head_dim;
                rms_norm_one_plus(&mut k_plain[off..off + head_dim], kw, attention.qk_norm_eps);
            }
        }

        // 4. RoPE on plaintext Q and K
        for h in 0..num_heads {
            let off = h * head_dim;
            rope.apply(&mut q_plain[off..off + head_dim], position);
        }
        for h in 0..num_kv_heads {
            let off = h * head_dim;
            rope.apply(&mut k_plain[off..off + head_dim], position);
        }

        // 5. Append to KV cache (K is public, V is shared)
        kv_cache.append(&k_plain, &v_cache_f32);
        let seq_len = kv_cache.current_len();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 6. Compute attention scores (fully public: Q·K, no exchange needed)
        let mut output_f32 = vec![0.0f32; q_out];
        for h in 0..num_heads {
            let kv_h = h / gqa_group_size;
            let q_offset = h * head_dim;
            let scores: Vec<f32> = (0..seq_len)
                .map(|t| {
                    let k_slice = kv_cache.k_at(kv_h, t);
                    let mut s = 0.0f32;
                    for d in 0..head_dim {
                        s += q_plain[q_offset + d] * k_slice[d];
                    }
                    s * scale
                })
                .collect();
            softmax_weighted_v(&scores, kv_h, head_dim, h * head_dim, kv_cache, &mut output_f32);
        }

        // 7. Reveal gate → sigmoid → multiply output shares
        let gate_f32: Vec<f32> = gate_vals.iter().map(|&v| from_fixed(v)).collect();
        let gate_plain = reveal_f32(&gate_f32, transport)?;
        for i in 0..q_out {
            output_f32[i] *= sigmoid(gate_plain[i]);
        }

        // 8. O projection
        Ok(shared_linear_forward_f32_input(
            attention.o_proj.weights.as_raw_slice(), q_out, hidden_size, &output_f32,
        ))
    } else {
        // --- Standard attention path ---

        // 1. Q projection
        let q_share = shared_linear_forward(
            party, attention.q_proj.weights.as_raw_slice(), hidden_size, q_out, x_share, &[], transport,
        )?;

        // 2. RoPE on shares
        let mut q_f32: Vec<f32> = q_share.0.iter().map(|&v| from_fixed(v)).collect();
        let mut k_f32: Vec<f32> = k_share.0.iter().map(|&v| from_fixed(v)).collect();
        for h in 0..num_heads {
            let off = h * head_dim;
            rope.apply(&mut q_f32[off..off + head_dim], position);
        }
        for h in 0..num_kv_heads {
            let off = h * head_dim;
            rope.apply(&mut k_f32[off..off + head_dim], position);
        }

        // 3. Append K/V shares to cache
        kv_cache.append(&k_f32, &v_cache_f32);
        let seq_len = kv_cache.current_len();

        // 4. Reveal Q
        let q_plain = reveal_f32(&q_f32, transport)?;

        // 5. Compute partial scores: Q_plain · K_share
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_scores = num_heads * seq_len;
        let mut partial_scores = Vec::with_capacity(total_scores);
        for h in 0..num_heads {
            let kv_h = h / gqa_group_size;
            let q_offset = h * head_dim;
            for t in 0..seq_len {
                let k_slice = kv_cache.k_at(kv_h, t);
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_plain[q_offset + d] * k_slice[d];
                }
                partial_scores.push(score * scale);
            }
        }

        // 6. Exchange partial scores → true scores
        let partial_bits: Vec<u32> = partial_scores.iter().map(|&s| s.to_bits()).collect();
        transport.send_u32_slice(&partial_bits)?;
        let other_bits = transport.recv_u32_slice(total_scores)?;
        let true_scores: Vec<f32> = partial_scores
            .iter()
            .zip(other_bits.iter())
            .map(|(&my, &ob)| my + f32::from_bits(ob))
            .collect();

        // 7. Softmax + weighted V sum
        let mut output_f32 = vec![0.0f32; q_out];
        for h in 0..num_heads {
            let kv_h = h / gqa_group_size;
            let score_offset = h * seq_len;
            let scores = &true_scores[score_offset..score_offset + seq_len];
            softmax_weighted_v(scores, kv_h, head_dim, h * head_dim, kv_cache, &mut output_f32);
        }

        // 8. O projection
        Ok(shared_linear_forward_f32_input(
            attention.o_proj.weights.as_raw_slice(), q_out, hidden_size, &output_f32,
        ))
    }
}

/// Secure (Q32.32) attention forward pass.
///
/// Handles both standard and Qwen3.5 gated attention.
///
/// Q is revealed (same leakage). For gated attention: K and gate also revealed.
pub fn private_attention_forward_secure(
    party: u8,
    attention: &Attention,
    x_share: &SharedVec64,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let hidden_size = attention.q_proj.in_features();
    let num_heads = attention.num_heads();
    let num_kv_heads = attention.num_kv_heads();
    let head_dim = attention.head_dim();
    let gqa_group_size = num_heads / num_kv_heads;
    let q_out = num_heads * head_dim;
    let kv_out = num_kv_heads * head_dim;

    // K/V projections (same for both paths)
    let k_share = shared_linear_forward_64(
        party, attention.k_proj.weights.as_raw_slice(), hidden_size, kv_out, x_share, &[], transport,
    )?;
    let v_share = shared_linear_forward_64(
        party, attention.v_proj.weights.as_raw_slice(), hidden_size, kv_out, x_share, &[], transport,
    )?;

    let v_cache_f32: Vec<f32> = v_share.0.iter().map(|&v| from_fixed64(v)).collect();

    if attention.output_gate {
        // --- Gated attention path (Qwen3.5) ---

        // 1. Q projection (doubled), de-interleave per head
        let q_out_doubled = q_out * 2;
        let q_gate_share = shared_linear_forward_64(
            party, attention.q_proj.weights.as_raw_slice(), hidden_size, q_out_doubled, x_share, &[], transport,
        )?;

        let mut q_vals = vec![0u64; q_out];
        let mut gate_vals = vec![0u64; q_out];
        for h in 0..num_heads {
            let src = h * head_dim * 2;
            let dst = h * head_dim;
            q_vals[dst..dst + head_dim].copy_from_slice(&q_gate_share.0[src..src + head_dim]);
            gate_vals[dst..dst + head_dim].copy_from_slice(&q_gate_share.0[src + head_dim..src + head_dim * 2]);
        }

        // 2. Reveal Q, apply per-head RMSNorm
        let q_f32: Vec<f32> = q_vals.iter().map(|&v| from_fixed64(v)).collect();
        let mut q_plain = reveal_f32(&q_f32, transport)?;
        if let Some(ref qw) = attention.q_norm_weight {
            for h in 0..num_heads {
                let off = h * head_dim;
                rms_norm_one_plus(&mut q_plain[off..off + head_dim], qw, attention.qk_norm_eps);
            }
        }

        // 3. Reveal K, apply per-head RMSNorm
        let k_f32: Vec<f32> = k_share.0.iter().map(|&v| from_fixed64(v)).collect();
        let mut k_plain = reveal_f32(&k_f32, transport)?;
        if let Some(ref kw) = attention.k_norm_weight {
            for h in 0..num_kv_heads {
                let off = h * head_dim;
                rms_norm_one_plus(&mut k_plain[off..off + head_dim], kw, attention.qk_norm_eps);
            }
        }

        // 4. RoPE on plaintext Q and K
        for h in 0..num_heads {
            let off = h * head_dim;
            rope.apply(&mut q_plain[off..off + head_dim], position);
        }
        for h in 0..num_kv_heads {
            let off = h * head_dim;
            rope.apply(&mut k_plain[off..off + head_dim], position);
        }

        // 5. Append to KV cache (K public, V shared)
        kv_cache.append(&k_plain, &v_cache_f32);
        let seq_len = kv_cache.current_len();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 6. Compute attention scores (fully public)
        let mut output_f32 = vec![0.0f32; q_out];
        for h in 0..num_heads {
            let kv_h = h / gqa_group_size;
            let q_offset = h * head_dim;
            let scores: Vec<f32> = (0..seq_len)
                .map(|t| {
                    let k_slice = kv_cache.k_at(kv_h, t);
                    let mut s = 0.0f32;
                    for d in 0..head_dim {
                        s += q_plain[q_offset + d] * k_slice[d];
                    }
                    s * scale
                })
                .collect();
            softmax_weighted_v(&scores, kv_h, head_dim, h * head_dim, kv_cache, &mut output_f32);
        }

        // 7. Reveal gate → sigmoid → multiply output shares
        let gate_f32: Vec<f32> = gate_vals.iter().map(|&v| from_fixed64(v)).collect();
        let gate_plain = reveal_f32(&gate_f32, transport)?;
        for i in 0..q_out {
            output_f32[i] *= sigmoid(gate_plain[i]);
        }

        // 8. O projection: f32 input → Q32.32 output
        Ok(shared_linear_forward_f32_input_64(
            attention.o_proj.weights.as_raw_slice(), q_out, hidden_size, &output_f32,
        ))
    } else {
        // --- Standard attention path ---

        // 1. Q projection
        let q_share = shared_linear_forward_64(
            party, attention.q_proj.weights.as_raw_slice(), hidden_size, q_out, x_share, &[], transport,
        )?;

        // 2. RoPE on shares
        let mut q_f32: Vec<f32> = q_share.0.iter().map(|&v| from_fixed64(v)).collect();
        let mut k_f32: Vec<f32> = k_share.0.iter().map(|&v| from_fixed64(v)).collect();
        for h in 0..num_heads {
            let off = h * head_dim;
            rope.apply(&mut q_f32[off..off + head_dim], position);
        }
        for h in 0..num_kv_heads {
            let off = h * head_dim;
            rope.apply(&mut k_f32[off..off + head_dim], position);
        }

        // 3. Append K/V shares to cache
        kv_cache.append(&k_f32, &v_cache_f32);
        let seq_len = kv_cache.current_len();

        // 4. Reveal Q
        let q_plain = reveal_f32(&q_f32, transport)?;

        // 5. Partial scores: Q_plain · K_share
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_scores = num_heads * seq_len;
        let mut partial_scores = Vec::with_capacity(total_scores);
        for h in 0..num_heads {
            let kv_h = h / gqa_group_size;
            let q_offset = h * head_dim;
            for t in 0..seq_len {
                let k_slice = kv_cache.k_at(kv_h, t);
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_plain[q_offset + d] * k_slice[d];
                }
                partial_scores.push(score * scale);
            }
        }

        // 6. Exchange partial scores → true scores
        let partial_bits: Vec<u32> = partial_scores.iter().map(|&s| s.to_bits()).collect();
        transport.send_u32_slice(&partial_bits)?;
        let other_bits = transport.recv_u32_slice(total_scores)?;
        let true_scores: Vec<f32> = partial_scores
            .iter()
            .zip(other_bits.iter())
            .map(|(&my, &ob)| my + f32::from_bits(ob))
            .collect();

        // 7. Softmax + weighted V sum
        let mut output_f32 = vec![0.0f32; q_out];
        for h in 0..num_heads {
            let kv_h = h / gqa_group_size;
            let score_offset = h * seq_len;
            let scores = &true_scores[score_offset..score_offset + seq_len];
            softmax_weighted_v(scores, kv_h, head_dim, h * head_dim, kv_cache, &mut output_f32);
        }

        // 8. O projection
        Ok(shared_linear_forward_f32_input_64(
            attention.o_proj.weights.as_raw_slice(), q_out, hidden_size, &output_f32,
        ))
    }
}

// --- GatedDeltaNet secure forward helpers ---

/// SiLU activation: x * sigmoid(x)
#[inline]
fn dn_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus: ln(1 + exp(x)), numerically stable
#[inline]
fn dn_softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// L2-normalize a vector in-place: x_i = x_i / sqrt(sum(x^2) + eps).
#[inline]
fn dn_l2_normalize(x: &mut [f32]) {
    let eps = 1e-6f32;
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_norm = 1.0 / (sum_sq + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv_norm;
    }
}

/// Reveal Q32.32 shares as f32: convert to f32, exchange, reconstruct by adding.
fn reveal_fixed64_as_f32(
    share: &[u64],
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let f32_vals: Vec<f32> = share.iter().map(|&v| from_fixed64(v)).collect();
    reveal_f32(&f32_vals, transport)
}

/// Secure (Q32.32) GatedDeltaNet forward pass.
///
/// Linear projections are done under MPC (public weights × Q32.32 shares).
/// Only the lower-dimensional projection outputs (QKV, gates, Z) are revealed,
/// never the full hidden state.
///
/// Leakage: QKV projections, gate inputs (a, b), output gate Z.
pub fn private_deltanet_forward_secure(
    party: u8,
    dn: &GatedDeltaNet,
    x_share: &SharedVec64,
    state: &mut DeltaNetState,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let hidden_size = dn.out_proj.out_features();
    let num_heads = dn.num_heads;
    let key_dim = dn.key_dim;
    let value_dim = dn.value_dim;
    let conv_dim = dn.conv_dim;
    let kernel_size = dn.kernel_size;
    let qkv_dim = conv_dim; // = num_heads*key_dim*2 + num_heads*value_dim
    let z_dim = num_heads * value_dim;

    // 1. MPC linear projections (local, no communication — public weights × Q32.32 shares)
    let qkv_share = shared_linear_forward_64(
        party, dn.in_proj_qkv.weights.as_raw_slice(), hidden_size, qkv_dim, x_share, &[], transport,
    )?;
    let z_share = shared_linear_forward_64(
        party, dn.in_proj_z.weights.as_raw_slice(), hidden_size, z_dim, x_share, &[], transport,
    )?;
    let a_share = shared_linear_forward_64(
        party, dn.in_proj_a.weights.as_raw_slice(), hidden_size, num_heads, x_share, &[], transport,
    )?;
    let b_share = shared_linear_forward_64(
        party, dn.in_proj_b.weights.as_raw_slice(), hidden_size, num_heads, x_share, &[], transport,
    )?;

    // 2. Reveal QKV (needed for conv + SiLU which are nonlinear)
    let qkv_plain = reveal_fixed64_as_f32(&qkv_share.0, transport)?;

    // 3. Conv + SiLU (plaintext, both parties identical)
    let conv_pos = state.conv_pos;
    for ch in 0..conv_dim {
        state.conv_state[ch * kernel_size + conv_pos] = qkv_plain[ch];
    }
    state.conv_pos = (conv_pos + 1) % kernel_size;

    let mut conv_out = vec![0.0f32; qkv_dim];
    for ch in 0..conv_dim {
        let mut sum = 0.0f32;
        let state_base = ch * kernel_size;
        let weight_base = ch * kernel_size;
        for k in 0..kernel_size {
            let ring_idx = (state.conv_pos + k) % kernel_size;
            sum += state.conv_state[state_base + ring_idx]
                * dn.conv_weight[weight_base + k];
        }
        conv_out[ch] = dn_silu(sum);
    }

    // 4. Split q/k/v, L2-normalize, scale Q
    let q_total = num_heads * key_dim;
    let k_total = num_heads * key_dim;
    let mut q_data = conv_out[..q_total].to_vec();
    let mut k_data = conv_out[q_total..q_total + k_total].to_vec();
    let v_data = &conv_out[q_total + k_total..];

    let inv_sqrt_dk = 1.0 / (key_dim as f32).sqrt();
    for h in 0..num_heads {
        let offset = h * key_dim;
        dn_l2_normalize(&mut q_data[offset..offset + key_dim]);
        dn_l2_normalize(&mut k_data[offset..offset + key_dim]);
        for i in 0..key_dim {
            q_data[offset + i] *= inv_sqrt_dk;
        }
    }

    // 5. Reveal a, b → compute alpha/beta gates
    let a_plain = reveal_fixed64_as_f32(&a_share.0, transport)?;
    let b_plain = reveal_fixed64_as_f32(&b_share.0, transport)?;

    let mut alpha = vec![0.0f32; num_heads];
    for h in 0..num_heads {
        let a = dn.a_log[h].exp() * dn_softplus(a_plain[h] + dn.dt_bias[h]);
        alpha[h] = (-a).exp();
    }

    let mut beta = vec![0.0f32; num_heads];
    for h in 0..num_heads {
        beta[h] = sigmoid(b_plain[h]);
    }

    // 6. Per-head recurrence (plaintext, both parties identical)
    let mut output_heads = vec![0.0f32; z_dim];

    for h in 0..num_heads {
        let q_h = &q_data[h * key_dim..(h + 1) * key_dim];
        let k_h = &k_data[h * key_dim..(h + 1) * key_dim];
        let v_h = &v_data[h * value_dim..(h + 1) * value_dim];

        // Access state.ssm_state directly (public field)
        let s_offset = h * key_dim * value_dim;
        let s = &mut state.ssm_state[s_offset..s_offset + key_dim * value_dim];

        // S = alpha * S
        let a = alpha[h];
        for val in s.iter_mut() {
            *val *= a;
        }

        // err_j = v_j - sum_i(S[i,j] * k_i)
        let mut err = vec![0.0f32; value_dim];
        for j in 0..value_dim {
            let mut dot = 0.0f32;
            for i in 0..key_dim {
                dot += s[i * value_dim + j] * k_h[i];
            }
            err[j] = v_h[j] - dot;
        }

        // S[i,j] += beta * k_i * err_j
        let b = beta[h];
        for i in 0..key_dim {
            for j in 0..value_dim {
                s[i * value_dim + j] += b * k_h[i] * err[j];
            }
        }

        // o_j = sum_i(S[i,j] * q_i)
        let o_h = &mut output_heads[h * value_dim..(h + 1) * value_dim];
        for j in 0..value_dim {
            let mut dot = 0.0f32;
            for i in 0..key_dim {
                dot += s[i * value_dim + j] * q_h[i];
            }
            o_h[j] = dot;
        }
    }

    // 7. Reveal Z → output gate: rms_norm(o) * silu(z)
    let z_plain = reveal_fixed64_as_f32(&z_share.0, transport)?;

    for h in 0..num_heads {
        let o_h = &mut output_heads[h * value_dim..(h + 1) * value_dim];

        // RMSNorm with weight scaling
        let mut sum_sq = 0.0f32;
        for &v in o_h.iter() {
            sum_sq += v * v;
        }
        let rms = (sum_sq / value_dim as f32 + dn.rms_norm_eps).sqrt();
        let inv_rms = 1.0 / rms;

        let z_h = &z_plain[h * value_dim..(h + 1) * value_dim];
        for j in 0..value_dim {
            o_h[j] = o_h[j] * inv_rms * dn.norm_weight[j] * dn_silu(z_h[j]);
        }
    }

    // 8. Output projection + re-share
    // output_heads is public (identical on both parties), so only party 0
    // contributes to avoid double-counting in the residual add.
    if party == 0 {
        Ok(shared_linear_forward_f32_input_64(
            dn.out_proj.weights.as_raw_slice(), z_dim, hidden_size, &output_heads,
        ))
    } else {
        Ok(SharedVec64::zeros(hidden_size))
    }
}
