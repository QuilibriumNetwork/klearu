//! Secure attention primitives for Q32.32 2-party computation.
//!
//! Provides Beaver-triple-based attention score computation and softmax
//! that never reveals Q or K vectors individually. Only the attention
//! pattern (Q·K^T dot products) is revealed for softmax.

use crate::beaver::BeaverTriple128;
use crate::fixed_point::{from_fixed64, to_fixed64};
use crate::multiply::beaver_dot_product_64;
use crate::sharing::SharedVec64;
use crate::transport::Transport;
use std::io;

/// Compute attention scores Q·K^T entirely in shares using Beaver dot products.
///
/// Q_share: `[num_tokens * head_dim]` (row-major, per head)
/// K_share: `[num_tokens * head_dim]` (row-major, per head)
/// Returns: `[num_tokens * num_tokens]` Q32.32 share of score matrix
///
/// Cost: num_tokens² × head_dim Beaver128 triples.
/// Communication: 2 bulk u128 exchanges per batch of dot products.
pub fn beaver_attention_scores_64(
    party: u8,
    q_share: &[u64],
    k_share: &[u64],
    num_tokens: usize,
    head_dim: usize,
    triples: &[BeaverTriple128],
    transport: &mut impl Transport,
) -> io::Result<Vec<u64>> {
    assert_eq!(q_share.len(), num_tokens * head_dim);
    assert_eq!(k_share.len(), num_tokens * head_dim);
    let total_dots = num_tokens * num_tokens;
    assert!(
        triples.len() >= total_dots * head_dim,
        "need {} triples, got {}",
        total_dots * head_dim,
        triples.len()
    );

    let mut scores = Vec::with_capacity(total_dots);
    let mut triple_offset = 0;

    for qi in 0..num_tokens {
        for ki in 0..num_tokens {
            let q_row = &q_share[qi * head_dim..(qi + 1) * head_dim];
            let k_row = &k_share[ki * head_dim..(ki + 1) * head_dim];
            let dot_triples = &triples[triple_offset..triple_offset + head_dim];

            let score = beaver_dot_product_64(party, q_row, k_row, dot_triples, transport)?;
            scores.push(score);
            triple_offset += head_dim;
        }
    }

    Ok(scores)
}

/// Secure softmax + weighted V computation.
///
/// Protocol:
/// 1. Reveal attention scores (leaks attention pattern, NOT Q or K individually)
/// 2. Apply scale + softmax in plaintext (both parties compute identically)
/// 3. Weighted V: public softmax weights × V_share (local, no communication)
///
/// score_share: `[num_tokens * num_tokens]` Q32.32 share of Q·K^T
/// v_share: `[num_tokens * head_dim]` Q32.32 share of V
/// scale: typically `head_dim^(-0.5)`
///
/// Returns: `[num_tokens * head_dim]` Q32.32 share of attention output
pub fn softmax_weighted_v_64(
    _party: u8,
    score_share: &[u64],
    v_share: &[u64],
    num_tokens: usize,
    head_dim: usize,
    scale: f32,
    transport: &mut impl Transport,
) -> io::Result<Vec<u64>> {
    let total_scores = num_tokens * num_tokens;
    assert_eq!(score_share.len(), total_scores);
    assert_eq!(v_share.len(), num_tokens * head_dim);

    // Step 1: Reveal scores
    transport.send_u64_slice(score_share)?;
    let other_scores = transport.recv_u64_slice(total_scores)?;

    let mut scores_plain = vec![0.0f32; total_scores];
    for i in 0..total_scores {
        scores_plain[i] = from_fixed64(score_share[i].wrapping_add(other_scores[i])) * scale;
    }

    // Step 2: Softmax per row
    for qi in 0..num_tokens {
        let row = &mut scores_plain[qi * num_tokens..(qi + 1) * num_tokens];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }

    // Step 3: Weighted V (public softmax × V_share = local)
    let mut output = vec![0u64; num_tokens * head_dim];
    for qi in 0..num_tokens {
        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for vi in 0..num_tokens {
                let weight = scores_plain[qi * num_tokens + vi];
                let v_val = v_share[vi * head_dim + d];
                acc += weight as f64 * v_val as i64 as f64;
            }
            output[qi * head_dim + d] = acc.round() as i64 as u64;
        }
    }

    Ok(output)
}

/// Full self-attention in Q32.32 without revealing Q or K.
///
/// Protocol:
/// 1. QKV projection: public weights × shared input → local (no communication)
/// 2. Per head: Beaver attention scores Q_share · K_share^T
/// 3. Reveal scores, compute softmax publicly (leaks attention pattern only)
/// 4. Weighted V: public softmax × V_share (local)
/// 5. Output projection: public weights × shared output → local
///
/// `qkv_weight`: `[3*dim, dim]` row-major, `qkv_bias`: `[3*dim]`
/// `proj_weight`: `[dim, dim]` row-major, `proj_bias`: `[dim]`
///
/// Returns: `[num_tokens * dim]` Q32.32 share of attention output
pub fn self_attention_secure_64(
    party: u8,
    tokens_share: &[u64],
    num_tokens: usize,
    dim: usize,
    num_heads: usize,
    qkv_weight: &[f32],
    qkv_bias: &[f32],
    proj_weight: &[f32],
    proj_bias: &[f32],
    triples: &mut dyn FnMut(usize) -> Vec<BeaverTriple128>,
    transport: &mut impl Transport,
) -> io::Result<Vec<u64>> {
    let head_dim = dim / num_heads;
    let qkv_dim = dim * 3;
    let empty_triples = vec![];

    // Step 1: QKV projection (local, public weight × share)
    let mut qkv_buf = vec![0u64; num_tokens * qkv_dim];
    for t in 0..num_tokens {
        let tok = SharedVec64(tokens_share[t * dim..(t + 1) * dim].to_vec());
        let qkv_tok = crate::linear::shared_linear_forward_64(
            party, qkv_weight, dim, qkv_dim, &tok, &empty_triples, transport,
        )?;
        qkv_buf[t * qkv_dim..(t + 1) * qkv_dim].copy_from_slice(&qkv_tok.0);
        if party == 0 {
            for d in 0..qkv_dim {
                qkv_buf[t * qkv_dim + d] =
                    qkv_buf[t * qkv_dim + d].wrapping_add(to_fixed64(qkv_bias[d]));
            }
        }
    }

    // Step 2-4: Per-head attention
    let scale = (head_dim as f32).powf(-0.5);
    let mut attn_output = vec![0u64; num_tokens * dim];

    for head in 0..num_heads {
        let ho = head * head_dim;

        // Extract per-head Q, K, V shares
        let mut q_head = vec![0u64; num_tokens * head_dim];
        let mut k_head = vec![0u64; num_tokens * head_dim];
        let mut v_head = vec![0u64; num_tokens * head_dim];
        for t in 0..num_tokens {
            for d in 0..head_dim {
                q_head[t * head_dim + d] = qkv_buf[t * qkv_dim + ho + d];
                k_head[t * head_dim + d] = qkv_buf[t * qkv_dim + dim + ho + d];
                v_head[t * head_dim + d] = qkv_buf[t * qkv_dim + 2 * dim + ho + d];
            }
        }

        // Beaver attention scores (Q·K^T in shares)
        let score_triples = triples(num_tokens * num_tokens * head_dim);
        let score_shares = beaver_attention_scores_64(
            party, &q_head, &k_head, num_tokens, head_dim, &score_triples, transport,
        )?;

        // Reveal scores + softmax + weighted V
        let head_out =
            softmax_weighted_v_64(party, &score_shares, &v_head, num_tokens, head_dim, scale, transport)?;

        // Scatter into multi-head output
        for t in 0..num_tokens {
            for d in 0..head_dim {
                attn_output[t * dim + ho + d] = head_out[t * head_dim + d];
            }
        }
    }

    // Step 5: Output projection (local)
    let mut proj_output = vec![0u64; num_tokens * dim];
    for t in 0..num_tokens {
        let tok = SharedVec64(attn_output[t * dim..(t + 1) * dim].to_vec());
        let proj_out = crate::linear::shared_linear_forward_64(
            party, proj_weight, dim, dim, &tok, &empty_triples, transport,
        )?;
        for d in 0..dim {
            let mut val = proj_out.0[d];
            if party == 0 {
                val = val.wrapping_add(to_fixed64(proj_bias[d]));
            }
            proj_output[t * dim + d] = val;
        }
    }

    Ok(proj_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::{dummy_triple_pair_128, TripleGenerator128};
    use crate::fixed_point::{from_fixed64, to_fixed64};
    use crate::transport::memory_transport_pair;

    #[test]
    fn test_beaver_attention_scores_64() {
        // 2 tokens, head_dim=2
        // Q = [[1, 0], [0, 1]], K = [[1, 0], [0, 1]]
        // Expected scores: [[1, 0], [0, 1]]
        let q_vals = [1.0f32, 0.0, 0.0, 1.0];
        let k_vals = [1.0f32, 0.0, 0.0, 1.0];

        let q0: Vec<u64> = q_vals.iter().map(|&v| to_fixed64(v)).collect();
        let k0: Vec<u64> = k_vals.iter().map(|&v| to_fixed64(v)).collect();
        let q1 = vec![0u64; 4];
        let k1 = vec![0u64; 4];

        let num_tokens = 2;
        let head_dim = 2;
        let total_triples = num_tokens * num_tokens * head_dim;

        let (mut gen0, mut gen1) = dummy_triple_pair_128(1000);
        let t0 = gen0.generate(total_triples);
        let t1 = gen1.generate(total_triples);

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            beaver_attention_scores_64(1, &q1, &k1, num_tokens, head_dim, &t1, &mut trans_b).unwrap()
        });

        let scores0 = beaver_attention_scores_64(0, &q0, &k0, num_tokens, head_dim, &t0, &mut trans_a).unwrap();
        let scores1 = handle.join().unwrap();

        let expected = [1.0f32, 0.0, 0.0, 1.0];
        for i in 0..4 {
            let result = from_fixed64(scores0[i].wrapping_add(scores1[i]));
            assert!(
                (result - expected[i]).abs() < 0.5,
                "score[{i}]: got {result}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn test_softmax_weighted_v_64() {
        // 2 tokens, head_dim=2
        // Scores (pre-scale): identity → softmax rows = [softmax(1,0), softmax(0,1)]
        let score0: Vec<u64> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .map(|&v| to_fixed64(v))
            .collect();
        let v0: Vec<u64> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| to_fixed64(v))
            .collect();
        let score1 = vec![0u64; 4];
        let v1 = vec![0u64; 4];

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            softmax_weighted_v_64(1, &score1, &v1, 2, 2, 1.0, &mut trans_b).unwrap()
        });

        let out0 = softmax_weighted_v_64(0, &score0, &v0, 2, 2, 1.0, &mut trans_a).unwrap();
        let out1 = handle.join().unwrap();

        // With scale=1.0, scores [1,0] → softmax ≈ [0.731, 0.269]
        // output[0] = 0.731 * V[0] + 0.269 * V[1]
        for i in 0..4 {
            let result = from_fixed64(out0[i].wrapping_add(out1[i]));
            assert!(result.is_finite(), "output[{i}] not finite: {result}");
        }
    }

    #[test]
    fn test_self_attention_secure_64_runs() {
        let dim = 4;
        let num_heads = 2;
        let num_tokens = 2;

        // Random-ish but deterministic weights
        let qkv_weight: Vec<f32> = (0..12 * 4).map(|i| (i as f32 * 0.01) - 0.24).collect();
        let qkv_bias = vec![0.0f32; 12];
        let proj_weight: Vec<f32> = (0..4 * 4).map(|i| (i as f32 * 0.02) - 0.16).collect();
        let proj_bias = vec![0.0f32; 4];

        let tokens: Vec<f32> = (0..num_tokens * dim).map(|i| (i as f32 * 0.1) - 0.2).collect();
        let tokens0: Vec<u64> = tokens.iter().map(|&v| to_fixed64(v)).collect();
        let tokens1 = vec![0u64; tokens0.len()];

        let (mut gen0, mut gen1) = dummy_triple_pair_128(50000);

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let qkv_w2 = qkv_weight.clone();
        let qkv_b2 = qkv_bias.clone();
        let proj_w2 = proj_weight.clone();
        let proj_b2 = proj_bias.clone();

        let handle = std::thread::spawn(move || {
            self_attention_secure_64(
                1, &tokens1, num_tokens, dim, num_heads,
                &qkv_w2, &qkv_b2, &proj_w2, &proj_b2,
                &mut |n| gen1.generate(n),
                &mut trans_b,
            )
            .unwrap()
        });

        let out0 = self_attention_secure_64(
            0, &tokens0, num_tokens, dim, num_heads,
            &qkv_weight, &qkv_bias, &proj_weight, &proj_bias,
            &mut |n| gen0.generate(n),
            &mut trans_a,
        )
        .unwrap();
        let out1 = handle.join().unwrap();

        assert_eq!(out0.len(), num_tokens * dim);
        for i in 0..out0.len() {
            let val = from_fixed64(out0[i].wrapping_add(out1[i]));
            assert!(val.is_finite(), "output[{i}] not finite");
        }
    }
}
