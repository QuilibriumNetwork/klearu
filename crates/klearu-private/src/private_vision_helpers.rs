//! Shared MPC helpers for vision model private inference.
//!
//! Reusable primitives extracted from `private_davit.rs` and extended
//! for all vision architectures. Operates on Q32.32 (u64) shares.

use klearu_mpc::beaver::TripleGenerator128;
use klearu_mpc::fixed_point::{from_fixed64, to_fixed64, FRAC_BITS_64, SCALE_64};
use klearu_mpc::sharing::{SharedVec, SharedVec64};
use klearu_mpc::transport::Transport;
use std::io;

// ---------------------------------------------------------------------------
// Image sharing
// ---------------------------------------------------------------------------

/// Create a Q16.16 shared image for low-security private inference.
///
/// Party 0 holds the full Q16.16 image as its share, party 1 holds zeros.
pub fn shared_image(party: u8, image: &[f32]) -> SharedVec {
    use klearu_mpc::fixed_point::to_fixed;
    if party == 0 {
        SharedVec(image.iter().map(|&v| to_fixed(v)).collect())
    } else {
        SharedVec(vec![0u32; image.len()])
    }
}

/// Create a Q32.32 shared image for high-security private inference.
///
/// Party 0 holds the full Q32.32 image as its share, party 1 holds zeros.
pub fn shared_image_64(party: u8, image: &[f32]) -> SharedVec64 {
    if party == 0 {
        SharedVec64(image.iter().map(|&v| to_fixed64(v)).collect())
    } else {
        SharedVec64(vec![0u64; image.len()])
    }
}

// ---------------------------------------------------------------------------
// Layout conversion
// ---------------------------------------------------------------------------

/// Convert `[C, H, W]` u64 shares to `[N, C]` token layout where N = H*W.
pub fn channel_first_to_tokens_64(chw: &[u64], c: usize, h: usize, w: usize) -> Vec<u64> {
    let n = h * w;
    let mut tokens = vec![0u64; n * c];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let token_idx = y * w + x;
                tokens[token_idx * c + ch] = chw[ch * n + y * w + x];
            }
        }
    }
    tokens
}

/// Convert `[N, C]` token layout back to `[C, H, W]` u64 shares.
pub fn tokens_to_channel_first_64(tokens: &[u64], chw: &mut [u64], c: usize, h: usize, w: usize) {
    let n = h * w;
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let token_idx = y * w + x;
                chw[ch * n + y * w + x] = tokens[token_idx * c + ch];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reveal helpers
// ---------------------------------------------------------------------------

/// Reveal u64 Q32.32 shares as f32 vector.
pub fn reveal_u64_as_f32(shares: &[u64], transport: &mut impl Transport) -> io::Result<Vec<f32>> {
    let n = shares.len();
    transport.send_u64_slice(shares)?;
    let other = transport.recv_u64_slice(n)?;
    Ok((0..n)
        .map(|i| from_fixed64(shares[i].wrapping_add(other[i])))
        .collect())
}

/// Reveal u32 Q16.16 shares as f32 vector.
pub fn reveal_u32_as_f32(shares: &[u32], transport: &mut impl Transport) -> io::Result<Vec<f32>> {
    use klearu_mpc::fixed_point::from_fixed;
    let n = shares.len();
    transport.send_u32_slice(shares)?;
    let other = transport.recv_u32_slice(n)?;
    Ok((0..n)
        .map(|i| from_fixed(shares[i].wrapping_add(other[i])))
        .collect())
}

// ---------------------------------------------------------------------------
// Normalization helpers
// ---------------------------------------------------------------------------

/// Apply LayerNorm2d to Q32.32 shares in `[C, H, W]` layout.
///
/// Normalizes the channel dimension at each spatial position.
pub fn layernorm_2d_shared_64(
    party: u8,
    data: &mut [u64],
    c: usize,
    h: usize,
    w: usize,
    weights: &[f32],
    bias: &[f32],
    eps: f32,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    let n = h * w;
    assert_eq!(data.len(), c * n);
    assert_eq!(weights.len(), c);
    assert_eq!(bias.len(), c);

    for y in 0..h {
        for x in 0..w {
            let spatial_idx = y * w + x;
            let mut token = Vec::with_capacity(c);
            for ch in 0..c {
                token.push(data[ch * n + spatial_idx]);
            }
            let mut token_share = SharedVec64(token);
            klearu_mpc::normalization::layernorm_shared_64(
                party,
                &mut token_share,
                weights,
                bias,
                eps,
                triples,
                transport,
            )?;
            for ch in 0..c {
                data[ch * n + spatial_idx] = token_share.0[ch];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// MLP helpers
// ---------------------------------------------------------------------------

/// Secure MLP with GELU reveal: fc1 → GELU(reveal) → fc2.
///
/// Reveals GELU intermediate (designed leakage, same as DaViT pattern).
pub fn mlp_secure_reveal(
    party: u8,
    x_share: &SharedVec64,
    fc1_weight: &[f32],
    fc1_bias: &[f32],
    fc1_in: usize,
    fc1_out: usize,
    fc2_weight: &[f32],
    fc2_bias: &[f32],
    fc2_out: usize,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let empty_triples = vec![];

    // fc1: local linear
    let mut fc1_result = klearu_mpc::linear::shared_linear_forward_64(
        party,
        fc1_weight,
        fc1_in,
        fc1_out,
        x_share,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for i in 0..fc1_out {
            fc1_result.0[i] = fc1_result.0[i].wrapping_add(to_fixed64(fc1_bias[i]));
        }
    }

    // GELU (reveals intermediate)
    let gelu_out = klearu_mpc::activation::gelu_reveal_64(party, &fc1_result, transport)?;

    // fc2: local linear
    let mut fc2_result = klearu_mpc::linear::shared_linear_forward_64(
        party,
        fc2_weight,
        fc1_out,
        fc2_out,
        &gelu_out,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for i in 0..fc2_out {
            fc2_result.0[i] = fc2_result.0[i].wrapping_add(to_fixed64(fc2_bias[i]));
        }
    }

    Ok(fc2_result)
}

/// Secure MLP with GELU polynomial (NO reveal): fc1 → GELU_poly → fc2.
///
/// Uses polynomial GELU approximation. Requires 2 Beaver128 triples per hidden element.
pub fn mlp_secure_no_reveal(
    party: u8,
    x_share: &SharedVec64,
    fc1_weight: &[f32],
    fc1_bias: &[f32],
    fc1_in: usize,
    fc1_out: usize,
    fc2_weight: &[f32],
    fc2_bias: &[f32],
    fc2_out: usize,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let empty_triples = vec![];

    // fc1: local linear
    let mut fc1_result = klearu_mpc::linear::shared_linear_forward_64(
        party,
        fc1_weight,
        fc1_in,
        fc1_out,
        x_share,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for i in 0..fc1_out {
            fc1_result.0[i] = fc1_result.0[i].wrapping_add(to_fixed64(fc1_bias[i]));
        }
    }

    // GELU polynomial (no reveal, 2 triples per element)
    let gelu_triples = triples.generate(2 * fc1_out);
    let gelu_out =
        klearu_mpc::activation::gelu_approx_shared_64(party, &fc1_result, &gelu_triples, transport)?;

    // fc2: local linear
    let mut fc2_result = klearu_mpc::linear::shared_linear_forward_64(
        party,
        fc2_weight,
        fc1_out,
        fc2_out,
        &gelu_out,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for i in 0..fc2_out {
            fc2_result.0[i] = fc2_result.0[i].wrapping_add(to_fixed64(fc2_bias[i]));
        }
    }

    Ok(fc2_result)
}

/// Secure SwiGLU MLP (NO reveal): gate_proj + up_proj → SiLU_poly(gate) × up → down_proj.
///
/// Uses polynomial SiLU and Beaver multiply for gate×up. 2 triples per hidden element.
pub fn swiglu_mlp_secure(
    party: u8,
    x_share: &SharedVec64,
    gate_weight: &[f32],
    gate_bias: &[f32],
    up_weight: &[f32],
    up_bias: &[f32],
    down_weight: &[f32],
    down_bias: &[f32],
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let empty_triples = vec![];

    // Gate projection (local)
    let mut gate_result = klearu_mpc::linear::shared_linear_forward_64(
        party,
        gate_weight,
        in_dim,
        hidden_dim,
        x_share,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for i in 0..hidden_dim {
            gate_result.0[i] = gate_result.0[i].wrapping_add(to_fixed64(gate_bias[i]));
        }
    }

    // Up projection (local)
    let mut up_result = klearu_mpc::linear::shared_linear_forward_64(
        party,
        up_weight,
        in_dim,
        hidden_dim,
        x_share,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for i in 0..hidden_dim {
            up_result.0[i] = up_result.0[i].wrapping_add(to_fixed64(up_bias[i]));
        }
    }

    // SwiGLU: silu_poly(gate) * up (2 triples per element)
    let swiglu_triples = triples.generate(2 * hidden_dim);
    let swiglu_out = klearu_mpc::activation::swiglu_shared_64(
        party,
        &gate_result,
        &up_result,
        &swiglu_triples,
        transport,
    )?;

    // Down projection (local)
    let mut down_result = klearu_mpc::linear::shared_linear_forward_64(
        party,
        down_weight,
        hidden_dim,
        out_dim,
        &swiglu_out,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for i in 0..out_dim {
            down_result.0[i] = down_result.0[i].wrapping_add(to_fixed64(down_bias[i]));
        }
    }

    Ok(down_result)
}

// ---------------------------------------------------------------------------
// Scaling helpers
// ---------------------------------------------------------------------------

/// LayerScale: public gamma × share. Purely local, no communication.
pub fn layer_scale_shared_64(shares: &mut [u64], gamma: &[f32]) {
    let n = shares.len();
    assert_eq!(n, gamma.len());
    for i in 0..n {
        let g = gamma[i] as f64;
        let x = shares[i] as i64 as f64;
        shares[i] = (g * x).round() as i64 as u64;
    }
}

/// GRN (Global Response Normalization) on Q32.32 shares.
///
/// Reveals scalar L2 norm per position (1 scalar leak). Then applies
/// normalization and affine transform locally.
pub fn grn_shared_64(
    party: u8,
    shares: &mut [u64],
    dim: usize,
    gamma: &[f32],
    beta: &[f32],
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    assert_eq!(shares.len(), dim);
    assert_eq!(gamma.len(), dim);
    assert_eq!(beta.len(), dim);

    // Compute sum(x²) via Beaver squaring
    let sq_triples = triples.generate(dim);

    let d_shares: Vec<u128> = (0..dim)
        .map(|k| (shares[k] as i64 as i128 as u128).wrapping_sub(sq_triples[k].a))
        .collect();
    let e_shares: Vec<u128> = (0..dim)
        .map(|k| (shares[k] as i64 as i128 as u128).wrapping_sub(sq_triples[k].b))
        .collect();

    let mut de_concat = Vec::with_capacity(2 * dim);
    de_concat.extend_from_slice(&d_shares);
    de_concat.extend_from_slice(&e_shares);
    transport.send_u128_slice(&de_concat)?;
    let de_others = transport.recv_u128_slice(2 * dim)?;
    let (d_others, e_others) = de_others.split_at(dim);

    let mut sum_sq_share = 0u64;
    for k in 0..dim {
        let d = d_shares[k].wrapping_add(d_others[k]);
        let e = e_shares[k].wrapping_add(e_others[k]);

        let mut z = sq_triples[k].c;
        z = z.wrapping_add(sq_triples[k].a.wrapping_mul(e));
        z = z.wrapping_add(d.wrapping_mul(sq_triples[k].b));
        if party == 0 {
            z = z.wrapping_add(d.wrapping_mul(e));
        }

        let x_sq_share = ((z as i128) >> FRAC_BITS_64) as i64 as u64;
        sum_sq_share = sum_sq_share.wrapping_add(x_sq_share);
    }

    // Reveal L2 norm (single scalar leak)
    transport.send_u64(sum_sq_share)?;
    let sum_sq_other = transport.recv_u64()?;
    let sum_sq = from_fixed64(sum_sq_share.wrapping_add(sum_sq_other)) as f64;
    let l2_norm = sum_sq.sqrt();

    // GRN formula: x_grn = x * (l2_norm / (mean_l2 + eps)) but simplified:
    // GRN(x) = gamma * x * Gx + beta + x, where Gx = x / (||x||_2 + eps)
    // For shares: party computes gamma[i] * (share[i] * norm_factor) + party0: beta
    let norm_factor = if l2_norm > 1e-10 { 1.0 / l2_norm } else { 0.0 };

    for k in 0..dim {
        let x_normed = norm_factor * shares[k] as i64 as f64;
        // x * Gx = x * x_normed (but x_normed is from reveal, so it's public * share)
        let grn_contrib = (gamma[k] as f64 * x_normed).round() as i64 as u64;
        shares[k] = shares[k].wrapping_add(grn_contrib);
        if party == 0 {
            shares[k] = shares[k].wrapping_add(to_fixed64(beta[k]));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Attention helpers (common patterns)
// ---------------------------------------------------------------------------

/// Self-attention with Q/K/V reveal pattern (lower security, same as DaViT).
///
/// Reveals Q vectors and K vectors for score computation.
/// Used by models that opt for the reveal-Q/K pattern.
pub fn self_attention_reveal_qk(
    party: u8,
    tokens_share: &[u64],
    num_tokens: usize,
    dim: usize,
    num_heads: usize,
    qkv_weight: &[f32],
    qkv_bias: &[f32],
    proj_weight: &[f32],
    proj_bias: &[f32],
    transport: &mut impl Transport,
) -> io::Result<Vec<u64>> {
    let head_dim = dim / num_heads;
    let qkv_dim = dim * 3;
    let empty_triples = vec![];

    // QKV projection (local)
    let mut qkv_buf = vec![0u64; num_tokens * qkv_dim];
    for t in 0..num_tokens {
        let tok = SharedVec64(tokens_share[t * dim..(t + 1) * dim].to_vec());
        let qkv_tok = klearu_mpc::linear::shared_linear_forward_64(
            party,
            qkv_weight,
            dim,
            qkv_dim,
            &tok,
            &empty_triples,
            transport,
        )?;
        qkv_buf[t * qkv_dim..(t + 1) * qkv_dim].copy_from_slice(&qkv_tok.0);
        if party == 0 {
            for d in 0..qkv_dim {
                qkv_buf[t * qkv_dim + d] =
                    qkv_buf[t * qkv_dim + d].wrapping_add(to_fixed64(qkv_bias[d]));
            }
        }
    }

    // Reveal Q
    let q_shares: Vec<u64> = (0..num_tokens)
        .flat_map(|t| qkv_buf[t * qkv_dim..t * qkv_dim + dim].to_vec())
        .collect();
    let q_plain = reveal_u64_as_f32(&q_shares, transport)?;

    // Reveal K
    let k_flat: Vec<u64> = (0..num_tokens)
        .flat_map(|t| qkv_buf[t * qkv_dim + dim..t * qkv_dim + 2 * dim].to_vec())
        .collect();
    let k_plain = reveal_u64_as_f32(&k_flat, transport)?;

    // Compute attention scores and softmax per head
    let scale = (head_dim as f32).powf(-0.5);
    let mut scores = vec![0.0f32; num_heads * num_tokens * num_tokens];

    for head in 0..num_heads {
        let ho = head * head_dim;
        for qi in 0..num_tokens {
            for ki in 0..num_tokens {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_plain[qi * dim + ho + d] * k_plain[ki * dim + ho + d];
                }
                scores[head * num_tokens * num_tokens + qi * num_tokens + ki] = dot * scale;
            }
        }

        // Softmax per row
        for qi in 0..num_tokens {
            let base = head * num_tokens * num_tokens + qi * num_tokens;
            let row = &mut scores[base..base + num_tokens];
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
    }

    // Weighted V
    let mut attn_output = vec![0u64; num_tokens * dim];
    for head in 0..num_heads {
        let ho = head * head_dim;
        for qi in 0..num_tokens {
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for vi in 0..num_tokens {
                    let v_share = qkv_buf[vi * qkv_dim + 2 * dim + ho + d];
                    let weight =
                        scores[head * num_tokens * num_tokens + qi * num_tokens + vi];
                    acc += weight as f64 * v_share as i64 as f64;
                }
                attn_output[qi * dim + ho + d] = acc.round() as i64 as u64;
            }
        }
    }

    // Output projection (local)
    let mut proj_output = vec![0u64; num_tokens * dim];
    for t in 0..num_tokens {
        let tok = SharedVec64(attn_output[t * dim..(t + 1) * dim].to_vec());
        let proj_out = klearu_mpc::linear::shared_linear_forward_64(
            party,
            proj_weight,
            dim,
            dim,
            &tok,
            &empty_triples,
            transport,
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

// ---------------------------------------------------------------------------
// Global average pooling
// ---------------------------------------------------------------------------

/// Global average pool over spatial dimensions in Q32.32.
///
/// Input: `[C, H, W]` or `[N, C]` tokens. Returns `[C]` pooled.
pub fn global_avg_pool_64(tokens: &[u64], num_tokens: usize, dim: usize) -> Vec<u64> {
    assert_eq!(tokens.len(), num_tokens * dim);
    let inv_n = 1.0 / num_tokens as f64;
    let inv_n_q32 = (inv_n * SCALE_64).round() as i64;

    let mut pooled = vec![0u64; dim];
    for d in 0..dim {
        let mut sum = 0i128;
        for t in 0..num_tokens {
            sum += tokens[t * dim + d] as i64 as i128;
        }
        pooled[d] = (((sum) * (inv_n_q32 as i128)) >> FRAC_BITS_64) as i64 as u64;
    }
    pooled
}

/// Reveal final logits from Q32.32 shares.
pub fn reveal_logits(
    shares: &SharedVec64,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    reveal_u64_as_f32(&shares.0, transport)
}

// ---------------------------------------------------------------------------
// Sparse MPC helpers
// ---------------------------------------------------------------------------

/// Sparse MLP in Q32.32: compute only selected fc1 neurons.
///
/// 1. Sparse fc1: only the `active_neurons` rows are computed
/// 2. GELU polynomial on sparse subset (2 triples per active neuron)
/// 3. Scatter-back through fc2: for each output dim, accumulate contributions
///    only from active neuron columns
///
/// With 50% sparsity: fc1 compute halved, GELU triples halved.
pub fn mlp_sparse_secure(
    party: u8,
    x_share: &SharedVec64,
    fc1_weight: &[f32],
    fc1_bias: &[f32],
    fc1_in: usize,
    fc1_total_out: usize,
    active_neurons: &[usize],
    fc2_weight: &[f32],
    fc2_bias: &[f32],
    fc2_out: usize,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let k = active_neurons.len();
    let empty_triples = vec![];

    // Sparse fc1: only compute selected output neurons
    let mut fc1_result = klearu_mpc::linear::shared_linear_forward_sparse_64(
        party,
        fc1_weight,
        fc1_in,
        fc1_total_out,
        active_neurons,
        x_share,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for (i, &neuron_idx) in active_neurons.iter().enumerate() {
            fc1_result.0[i] = fc1_result.0[i].wrapping_add(to_fixed64(fc1_bias[neuron_idx]));
        }
    }

    // GELU polynomial on sparse subset (2 triples per element)
    let gelu_triples = triples.generate(2 * k);
    let gelu_out =
        klearu_mpc::activation::gelu_approx_shared_64(party, &fc1_result, &gelu_triples, transport)?;

    // Scatter-back through fc2: for each output dim, accumulate from active columns only
    let fc2_stride = if fc1_total_out > 0 {
        fc2_weight.len() / fc2_out
    } else {
        fc1_total_out
    };

    let mut output = vec![0u64; fc2_out];
    for out_d in 0..fc2_out {
        let row_offset = out_d * fc2_stride;
        let mut acc = 0i128;
        for (sparse_idx, &neuron_idx) in active_neurons.iter().enumerate() {
            let w_q32 = (fc2_weight[row_offset + neuron_idx] as f64 * SCALE_64).round() as i64;
            acc += (w_q32 as i128) * (gelu_out.0[sparse_idx] as i64 as i128);
        }
        let mut val = ((acc >> FRAC_BITS_64) as i64) as u64;
        if party == 0 {
            val = val.wrapping_add(to_fixed64(fc2_bias[out_d]));
        }
        output[out_d] = val;
    }

    Ok(SharedVec64(output))
}

/// Sparse SwiGLU MLP in Q32.32: only selected hidden neurons.
///
/// 1. Sparse gate_proj and up_proj for active neurons only
/// 2. SiLU polynomial on gate subset + Beaver gate×up
/// 3. Scatter-back through down_proj from active neurons
pub fn swiglu_mlp_sparse_secure(
    party: u8,
    x_share: &SharedVec64,
    gate_weight: &[f32],
    gate_bias: &[f32],
    up_weight: &[f32],
    up_bias: &[f32],
    down_weight: &[f32],
    down_bias: &[f32],
    in_dim: usize,
    hidden_total: usize,
    active_neurons: &[usize],
    out_dim: usize,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let k = active_neurons.len();
    let empty_triples = vec![];

    // Sparse gate projection
    let mut gate_result = klearu_mpc::linear::shared_linear_forward_sparse_64(
        party,
        gate_weight,
        in_dim,
        hidden_total,
        active_neurons,
        x_share,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for (i, &neuron_idx) in active_neurons.iter().enumerate() {
            gate_result.0[i] = gate_result.0[i].wrapping_add(to_fixed64(gate_bias[neuron_idx]));
        }
    }

    // Sparse up projection
    let mut up_result = klearu_mpc::linear::shared_linear_forward_sparse_64(
        party,
        up_weight,
        in_dim,
        hidden_total,
        active_neurons,
        x_share,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for (i, &neuron_idx) in active_neurons.iter().enumerate() {
            up_result.0[i] = up_result.0[i].wrapping_add(to_fixed64(up_bias[neuron_idx]));
        }
    }

    // SwiGLU: silu_poly(gate) × up (2 triples per element)
    let swiglu_triples = triples.generate(2 * k);
    let swiglu_out = klearu_mpc::activation::swiglu_shared_64(
        party,
        &gate_result,
        &up_result,
        &swiglu_triples,
        transport,
    )?;

    // Scatter-back through down_proj
    let down_stride = if hidden_total > 0 {
        down_weight.len() / out_dim
    } else {
        hidden_total
    };

    let mut output = vec![0u64; out_dim];
    for out_d in 0..out_dim {
        let row_offset = out_d * down_stride;
        let mut acc = 0i128;
        for (sparse_idx, &neuron_idx) in active_neurons.iter().enumerate() {
            let w_q32 = (down_weight[row_offset + neuron_idx] as f64 * SCALE_64).round() as i64;
            acc += (w_q32 as i128) * (swiglu_out.0[sparse_idx] as i64 as i128);
        }
        let mut val = ((acc >> FRAC_BITS_64) as i64) as u64;
        if party == 0 {
            val = val.wrapping_add(to_fixed64(down_bias[out_d]));
        }
        output[out_d] = val;
    }

    Ok(SharedVec64(output))
}

/// Sparse secure self-attention: only compute for active heads.
///
/// 1. QKV projection: public weights × shared input (local, all heads computed)
/// 2. Beaver attention scores: only for active heads
/// 3. Reveal scores, softmax (only active heads)
/// 4. Weighted V (only active heads, local)
/// 5. Output projection (local, zero inactive head positions)
///
/// With 50% head sparsity: Beaver triples for scores halved.
pub fn self_attention_sparse_secure_64(
    party: u8,
    tokens_share: &[u64],
    num_tokens: usize,
    dim: usize,
    num_heads: usize,
    active_heads: &[usize],
    qkv_weight: &[f32],
    qkv_bias: &[f32],
    proj_weight: &[f32],
    proj_bias: &[f32],
    triples_fn: &mut dyn FnMut(usize) -> Vec<klearu_mpc::beaver::BeaverTriple128>,
    transport: &mut impl Transport,
) -> io::Result<Vec<u64>> {
    let head_dim = dim / num_heads;
    let qkv_dim = dim * 3;
    let empty_triples = vec![];

    // QKV projection (local, all heads — cheaper than selective projection)
    let mut qkv_buf = vec![0u64; num_tokens * qkv_dim];
    for t in 0..num_tokens {
        let tok = SharedVec64(tokens_share[t * dim..(t + 1) * dim].to_vec());
        let qkv_tok = klearu_mpc::linear::shared_linear_forward_64(
            party,
            qkv_weight,
            dim,
            qkv_dim,
            &tok,
            &empty_triples,
            transport,
        )?;
        qkv_buf[t * qkv_dim..(t + 1) * qkv_dim].copy_from_slice(&qkv_tok.0);
        if party == 0 {
            for d in 0..qkv_dim {
                qkv_buf[t * qkv_dim + d] =
                    qkv_buf[t * qkv_dim + d].wrapping_add(to_fixed64(qkv_bias[d]));
            }
        }
    }

    // Only process active heads
    let mut attn_output = vec![0u64; num_tokens * dim];

    for &head in active_heads {
        let ho = head * head_dim;

        // Extract Q, K shares for this head
        let mut q_share = vec![0u64; num_tokens * head_dim];
        let mut k_share = vec![0u64; num_tokens * head_dim];

        for t in 0..num_tokens {
            for d in 0..head_dim {
                q_share[t * head_dim + d] = qkv_buf[t * qkv_dim + ho + d];
                k_share[t * head_dim + d] = qkv_buf[t * qkv_dim + dim + ho + d];
            }
        }

        // Beaver attention scores for this head
        let head_triples = triples_fn(num_tokens * num_tokens * head_dim);
        let score_shares = klearu_mpc::attention_mpc::beaver_attention_scores_64(
            party,
            &q_share,
            &k_share,
            num_tokens,
            head_dim,
            &head_triples,
            transport,
        )?;

        // Extract V shares for this head
        let mut v_share = vec![0u64; num_tokens * head_dim];
        for t in 0..num_tokens {
            for d in 0..head_dim {
                v_share[t * head_dim + d] = qkv_buf[t * qkv_dim + 2 * dim + ho + d];
            }
        }

        // Reveal scores, softmax, weighted V
        let scale = (head_dim as f32).powf(-0.5);
        let weighted_v = klearu_mpc::attention_mpc::softmax_weighted_v_64(
            party,
            &score_shares,
            &v_share,
            num_tokens,
            head_dim,
            scale,
            transport,
        )?;

        // Place in head position
        for t in 0..num_tokens {
            for d in 0..head_dim {
                attn_output[t * dim + ho + d] = weighted_v[t * head_dim + d];
            }
        }
    }

    // Output projection (local)
    let mut proj_output = vec![0u64; num_tokens * dim];
    for t in 0..num_tokens {
        let tok = SharedVec64(attn_output[t * dim..(t + 1) * dim].to_vec());
        let proj_out = klearu_mpc::linear::shared_linear_forward_64(
            party,
            proj_weight,
            dim,
            dim,
            &tok,
            &empty_triples,
            transport,
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
