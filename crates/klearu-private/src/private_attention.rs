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
use klearu_llm::model::kv_cache::{KvCache, KvCache64};
use klearu_llm::model::rope::RotaryEmbedding;
use klearu_mpc::beaver::TripleGenerator128;
use klearu_mpc::fixed_point::{from_fixed, from_fixed64, FRAC_BITS_64};
use klearu_mpc::linear::{
    shared_linear_forward, shared_linear_forward_64_pq,
    shared_linear_forward_f32_input, shared_linear_forward_f32_input_64,
};
use klearu_mpc::multiply::beaver_dot_product_64;
use klearu_mpc::activation::softmax_shared_64;
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
/// `exp_buffer` is a reusable scratch buffer to avoid per-head allocation.
fn softmax_weighted_v(
    scores: &[f32],
    exp_buffer: &mut Vec<f32>,
    kv_h: usize,
    head_dim: usize,
    head_offset: usize,
    kv_cache: &KvCache,
    output: &mut [f32],
) {
    let seq_len = scores.len();
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    exp_buffer.clear();
    exp_buffer.extend(scores.iter().map(|&s| (s - max_score).exp()));
    let sum: f32 = exp_buffer.iter().sum();
    for s in exp_buffer.iter_mut() {
        *s /= sum;
    }
    for t in 0..seq_len {
        let v_slice = kv_cache.v_at(kv_h, t);
        let w = exp_buffer[t];
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
        let mut exp_buffer = Vec::with_capacity(seq_len);
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
            softmax_weighted_v(&scores, &mut exp_buffer, kv_h, head_dim, h * head_dim, kv_cache, &mut output_f32);
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
        let mut exp_buffer = Vec::with_capacity(seq_len);
        for h in 0..num_heads {
            let kv_h = h / gqa_group_size;
            let score_offset = h * seq_len;
            let scores = &true_scores[score_offset..score_offset + seq_len];
            softmax_weighted_v(scores, &mut exp_buffer, kv_h, head_dim, h * head_dim, kv_cache, &mut output_f32);
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
    _party: u8,
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

    // K/V projections using pre-quantized weights (same for both paths)
    let k_share = shared_linear_forward_64_pq(
        &attention.k_proj.q32_weights(), hidden_size, kv_out, x_share,
    );
    let v_share = shared_linear_forward_64_pq(
        &attention.v_proj.q32_weights(), hidden_size, kv_out, x_share,
    );

    let v_cache_f32: Vec<f32> = v_share.0.iter().map(|&v| from_fixed64(v)).collect();

    if attention.output_gate {
        // --- Gated attention path (Qwen3.5) ---

        // 1. Q projection (doubled), de-interleave per head
        let q_out_doubled = q_out * 2;
        let q_gate_share = shared_linear_forward_64_pq(
            &attention.q_proj.q32_weights(), hidden_size, q_out_doubled, x_share,
        );

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
        let mut exp_buffer = Vec::with_capacity(seq_len);
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
            softmax_weighted_v(&scores, &mut exp_buffer, kv_h, head_dim, h * head_dim, kv_cache, &mut output_f32);
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

        // 1. Q projection using pre-quantized weights
        let q_share = shared_linear_forward_64_pq(
            &attention.q_proj.q32_weights(), hidden_size, q_out, x_share,
        );

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
        let mut exp_buffer = Vec::with_capacity(seq_len);
        for h in 0..num_heads {
            let kv_h = h / gqa_group_size;
            let score_offset = h * seq_len;
            let scores = &true_scores[score_offset..score_offset + seq_len];
            softmax_weighted_v(scores, &mut exp_buffer, kv_h, head_dim, h * head_dim, kv_cache, &mut output_f32);
        }

        // 8. O projection
        Ok(shared_linear_forward_f32_input_64(
            attention.o_proj.weights.as_raw_slice(), q_out, hidden_size, &output_f32,
        ))
    }
}

/// Apply RoPE to Q32.32 shares in-place (no communication needed).
///
/// RoPE is a linear transformation: `x_rot = x*cos + rotate(x)*sin`.
/// Since cos/sin are public and × share is local, this is a pure local op.
fn rope_on_shares_64(share: &mut [u64], position: usize, rope: &RotaryEmbedding) {
    let half = rope.half_rotary_dim();
    let cos = rope.cos_at(position);
    let sin = rope.sin_at(position);

    for i in 0..half {
        let x0 = share[i] as i64;
        let x1 = share[half + i] as i64;
        let c = cos[i] as f64;
        let s = sin[i] as f64;

        // new_x0 = x0*cos - x1*sin
        let new_x0 = (x0 as f64 * c - x1 as f64 * s).round() as i64 as u64;
        // new_x1 = x1*cos + x0*sin
        let new_x1 = (x1 as f64 * c + x0 as f64 * s).round() as i64 as u64;

        share[i] = new_x0;
        share[half + i] = new_x1;
    }
    // Elements beyond rotary_dim are left unchanged.
}

/// No-reveal attention forward pass (Q32.32).
///
/// Q, K, and attention scores are NEVER revealed individually.
/// Only softmax inputs are revealed (attention pattern leakage via score reveal).
///
/// Protocol:
/// 1. QKV projections: local (public weights × share)
/// 2. RoPE on Q, K shares: local (public cos/sin × share)
/// 3. Append K, V to KvCache64
/// 4. Attention scores Q·K^T: Beaver dot products (one round-trip per batch)
/// 5. Softmax: reveal scores → compute softmax publicly (leaks attention pattern)
/// 6. Weighted V: public softmax × V_share (local)
/// 7. O projection: local
///
/// Leakage: attention scores only (not Q, K, V individually).
pub fn private_attention_forward_noreveal(
    party: u8,
    attention: &Attention,
    x_share: &SharedVec64,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache64,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let hidden_size = attention.q_proj.in_features();
    let num_heads = attention.num_heads();
    let num_kv_heads = attention.num_kv_heads();
    let head_dim = attention.head_dim();
    let gqa_group_size = num_heads / num_kv_heads;
    let q_out = num_heads * head_dim;
    let kv_out = num_kv_heads * head_dim;

    // 1. QKV projections (local, no communication)
    let q_share = shared_linear_forward_64_pq(
        &attention.q_proj.q32_weights(), hidden_size, q_out, x_share,
    );
    let k_share = shared_linear_forward_64_pq(
        &attention.k_proj.q32_weights(), hidden_size, kv_out, x_share,
    );
    let v_share = shared_linear_forward_64_pq(
        &attention.v_proj.q32_weights(), hidden_size, kv_out, x_share,
    );

    // 2. RoPE on Q and K shares (local, no communication)
    let mut q_data = q_share.0;
    let mut k_data = k_share.0;
    for h in 0..num_heads {
        let off = h * head_dim;
        rope_on_shares_64(&mut q_data[off..off + head_dim], position, rope);
    }
    for h in 0..num_kv_heads {
        let off = h * head_dim;
        rope_on_shares_64(&mut k_data[off..off + head_dim], position, rope);
    }

    // 3. Append K, V to cache
    kv_cache.append(&k_data, &v_share.0);
    let seq_len = kv_cache.current_len();

    // 4. Attention scores: Q·K^T via Beaver dot products
    // For each head h, position t: score[h][t] = Q_h · K_h[t] / sqrt(head_dim)
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total_dots = num_heads * seq_len;
    let triples_needed = total_dots * head_dim;
    let dot_triples = triples.generate(triples_needed);

    let mut all_scores = Vec::with_capacity(total_dots);
    let mut triple_idx = 0;
    for h in 0..num_heads {
        let kv_h = h / gqa_group_size;
        let q_h = &q_data[h * head_dim..(h + 1) * head_dim];

        for t in 0..seq_len {
            let k_t = kv_cache.k_at(kv_h, t);
            let score_share = beaver_dot_product_64(
                party, q_h, k_t,
                &dot_triples[triple_idx..triple_idx + head_dim],
                transport,
            )?;
            // Scale by 1/sqrt(head_dim): public × share = local
            let scaled = (score_share as i64 as f64 * scale as f64).round() as i64 as u64;
            all_scores.push(scaled);
            triple_idx += head_dim;
        }
    }

    // 5. Softmax per head (reveals scores → leaks attention pattern)
    let softmax_triples_needed = 3 * seq_len; // per head
    let mut softmax_weights_all = Vec::with_capacity(total_dots);
    for h in 0..num_heads {
        let head_scores = &all_scores[h * seq_len..(h + 1) * seq_len];
        let sm_triples = triples.generate(softmax_triples_needed);
        let sm = softmax_shared_64(party, head_scores, &sm_triples, transport)?;
        softmax_weights_all.extend_from_slice(&sm);
    }

    // 6. Weighted V: public softmax × V_share (local)
    // softmax weights are re-shared (party 0 has values, party 1 has zeros).
    // So weighted V = sum_t softmax[t] * V_share[t] is local for each party.
    let mut output = vec![0u64; q_out];
    for h in 0..num_heads {
        let kv_h = h / gqa_group_size;
        let head_offset = h * head_dim;
        for t in 0..seq_len {
            let w = softmax_weights_all[h * seq_len + t];
            let v_t = kv_cache.v_at(kv_h, t);
            // w is Q32.32, v_t is Q32.32 share → product is Q64.64 → shift right 32
            for d in 0..head_dim {
                let w_val = w as i64;
                let v_val = v_t[d] as i64;
                let prod = ((w_val as i128 * v_val as i128) >> FRAC_BITS_64) as i64;
                output[head_offset + d] = (output[head_offset + d] as i64).wrapping_add(prod) as u64;
            }
        }
    }

    // 7. O projection (local, no communication)
    let o_share = shared_linear_forward_64_pq(
        &attention.o_proj.q32_weights(), q_out, hidden_size, &SharedVec64(output),
    );

    Ok(o_share)
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

/// Reveal Q32.32 shares as f32: exchange u64 shares directly, reconstruct in u64
/// (exact wrapping add), then convert to f32 once. Avoids double f32 conversion
/// and the precision loss from adding two f32 approximations.
fn reveal_u64_as_f32(
    share: &[u64],
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    transport.send_u64_slice(share)?;
    let other = transport.recv_u64_slice(share.len())?;
    Ok(share.iter().zip(other.iter())
        .map(|(&a, &b)| from_fixed64(a.wrapping_add(b)))
        .collect())
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
    let num_key_heads = dn.num_key_heads;
    let num_value_heads = dn.num_value_heads;
    let group_size = num_value_heads / num_key_heads;
    let key_dim = dn.key_dim;
    let value_dim = dn.value_dim;
    let conv_dim = dn.conv_dim;
    let kernel_size = dn.kernel_size;
    let qkv_dim = conv_dim; // = num_key_heads*key_dim*2 + num_value_heads*value_dim
    let z_dim = num_value_heads * value_dim;

    // 1. MPC linear projections using pre-quantized weights (local, no communication)
    let qkv_share = shared_linear_forward_64_pq(
        &dn.in_proj_qkv.q32_weights(), hidden_size, qkv_dim, x_share,
    );
    let z_share = shared_linear_forward_64_pq(
        &dn.in_proj_z.q32_weights(), hidden_size, z_dim, x_share,
    );
    let a_share = shared_linear_forward_64_pq(
        &dn.in_proj_a.q32_weights(), hidden_size, num_value_heads, x_share,
    );
    let b_share = shared_linear_forward_64_pq(
        &dn.in_proj_b.q32_weights(), hidden_size, num_value_heads, x_share,
    );

    // 2. Reveal QKV (needed for conv + SiLU which are nonlinear)
    let qkv_plain = reveal_u64_as_f32(&qkv_share.0, transport)?;

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

    // 4. Split q/k/v (q,k indexed by key-head; v indexed by value-head), L2-normalize, scale Q
    let q_total = num_key_heads * key_dim;
    let k_total = num_key_heads * key_dim;
    let mut q_data = conv_out[..q_total].to_vec();
    let mut k_data = conv_out[q_total..q_total + k_total].to_vec();
    let v_data = &conv_out[q_total + k_total..];

    let inv_sqrt_dk = 1.0 / (key_dim as f32).sqrt();
    for h in 0..num_key_heads {
        let offset = h * key_dim;
        dn_l2_normalize(&mut q_data[offset..offset + key_dim]);
        dn_l2_normalize(&mut k_data[offset..offset + key_dim]);
        for i in 0..key_dim {
            q_data[offset + i] *= inv_sqrt_dk;
        }
    }

    // 5. Reveal a, b in a single batch → compute alpha/beta gates (saves 1 round trip).
    //    a, b are per-value-head.
    let mut ab_shares = Vec::with_capacity(num_value_heads * 2);
    ab_shares.extend_from_slice(&a_share.0);
    ab_shares.extend_from_slice(&b_share.0);
    transport.send_u64_slice(&ab_shares)?;
    let ab_other = transport.recv_u64_slice(num_value_heads * 2)?;
    let a_plain: Vec<f32> = (0..num_value_heads)
        .map(|i| from_fixed64(ab_shares[i].wrapping_add(ab_other[i])))
        .collect();
    let b_plain: Vec<f32> = (0..num_value_heads)
        .map(|i| from_fixed64(
            ab_shares[num_value_heads + i].wrapping_add(ab_other[num_value_heads + i])
        ))
        .collect();

    let mut alpha = vec![0.0f32; num_value_heads];
    for h in 0..num_value_heads {
        let a = dn.a_log[h].exp() * dn_softplus(a_plain[h] + dn.dt_bias[h]);
        alpha[h] = (-a).exp();
    }

    let mut beta = vec![0.0f32; num_value_heads];
    for h in 0..num_value_heads {
        beta[h] = sigmoid(b_plain[h]);
    }

    // 6. Per-value-head recurrence (each value head h_v shares q/k with key head h_v / group_size).
    let mut output_heads = vec![0.0f32; z_dim];

    for h_v in 0..num_value_heads {
        let h_k = h_v / group_size;
        let q_h = &q_data[h_k * key_dim..(h_k + 1) * key_dim];
        let k_h = &k_data[h_k * key_dim..(h_k + 1) * key_dim];
        let v_h = &v_data[h_v * value_dim..(h_v + 1) * value_dim];

        let s_offset = h_v * key_dim * value_dim;
        let s = &mut state.ssm_state[s_offset..s_offset + key_dim * value_dim];

        // S = alpha * S
        let a = alpha[h_v];
        for val in s.iter_mut() {
            *val *= a;
        }

        // err = v - S^T·k  (restructured for contiguous inner-loop access)
        let mut err: Vec<f32> = v_h.to_vec();
        for i in 0..key_dim {
            let k_i = k_h[i];
            let s_row = &s[i * value_dim..(i + 1) * value_dim];
            for j in 0..value_dim {
                err[j] -= s_row[j] * k_i;
            }
        }

        // S[i,j] += beta * k_i * err_j
        let b = beta[h_v];
        for i in 0..key_dim {
            let bk = b * k_h[i];
            let s_row = &mut s[i * value_dim..(i + 1) * value_dim];
            for j in 0..value_dim {
                s_row[j] += bk * err[j];
            }
        }

        // o = S^T·q
        let o_h = &mut output_heads[h_v * value_dim..(h_v + 1) * value_dim];
        o_h.fill(0.0);
        for i in 0..key_dim {
            let q_i = q_h[i];
            let s_row = &s[i * value_dim..(i + 1) * value_dim];
            for j in 0..value_dim {
                o_h[j] += s_row[j] * q_i;
            }
        }
    }

    // 7. Reveal Z → output gate: rms_norm(o) * silu(z)
    let z_plain = reveal_u64_as_f32(&z_share.0, transport)?;

    for h in 0..num_value_heads {
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
