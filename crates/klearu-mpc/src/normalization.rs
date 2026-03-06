use crate::beaver::TripleGenerator128;
use crate::fixed_point::{from_fixed, from_fixed64, FRAC_BITS_64, SCALE_64};
use crate::sharing::{SharedVec, SharedVec64};
use crate::transport::Transport;
use std::io;

/// RMSNorm under secret sharing using reveal-and-correct.
///
/// 1. Reveal the full hidden state x (leaks x, acceptable in semi-honest model
///    since Q and gate values are already revealed)
/// 2. Compute inv_rms = 1/sqrt(mean(x^2) + eps) locally in f32
/// 3. Locally scale each share by inv_rms * weight
pub fn rmsnorm_shared(
    _party: u8,
    x_share: &mut SharedVec,
    weights: &[f32],
    eps: f32,
    transport: &mut impl Transport,
) -> io::Result<()> {
    let n = x_share.len();
    assert_eq!(n, weights.len());

    // Reveal x: exchange shares and reconstruct in f32.
    // This avoids the u32 overflow that occurs when computing sum(x_i^2) in
    // Q16.16 fixed-point (overflows when RMS > sqrt(65536/n), ~10.7 for n=576).
    transport.send_u32_slice(&x_share.0)?;
    let other_x = transport.recv_u32_slice(n)?;

    let x_plain: Vec<f32> = x_share.0.iter().zip(other_x.iter())
        .map(|(&a, &b)| from_fixed(a.wrapping_add(b)))
        .collect();

    // Compute scaling factor in f32 — no overflow possible.
    let sum_sq: f32 = x_plain.iter().map(|v| v * v).sum();
    let mean_sq = sum_sq / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    // Scale each share locally using f64 mixed-precision (avoids Q16.16 quantization of scale)
    for i in 0..n {
        let scale = (inv_rms * weights[i]) as f64;
        let x = x_share.0[i] as i32 as f64;
        x_share.0[i] = (scale * x).round() as i64 as i32 as u32;
    }

    Ok(())
}

/// Privacy-preserving RMSNorm under Q32.32 secret sharing.
///
/// Uses Beaver-triple-based squaring to compute sum(x²) without revealing x.
/// Only the scalar sum(x²) is revealed (single value), not the full hidden state.
///
/// Protocol:
/// 1. For each element k, generate a squaring triple (a, a, a²)
/// 2. Compute d[k] = x_share[k] - a[k], exchange d
/// 3. z[k] = c[k] + 2*a[k]*d + [party 0: d²]  (share of x[k]²)
/// 4. Sum z[k] locally, reveal scalar sum(x²) (single value leak)
/// 5. Compute scale = 1/sqrt(sum_sq/n + eps) in f32
/// 6. Scale each share: share[k] = (scale * weight[k]) * share[k] (i128 precision)
pub fn rmsnorm_shared_64(
    party: u8,
    x_share: &mut SharedVec64,
    weights: &[f32],
    eps: f32,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    let n = x_share.len();
    assert_eq!(n, weights.len());

    // Generate n squaring triples in Z_{2^128}.
    // Each triple (a, b, c) where c = a*b mod 2^128.
    // We use x as both inputs: d = x - a, e = x - b.
    let sq_triples = triples.generate(n);

    // d[k] = x_share[k] - triple.a (in Z_{2^128}, sign-extending u64→u128)
    let d_shares: Vec<u128> = (0..n)
        .map(|k| (x_share.0[k] as i64 as i128 as u128).wrapping_sub(sq_triples[k].a))
        .collect();
    let e_shares: Vec<u128> = (0..n)
        .map(|k| (x_share.0[k] as i64 as i128 as u128).wrapping_sub(sq_triples[k].b))
        .collect();

    // Exchange d and e in a single batch (2 transport calls instead of 4)
    let mut de_concat = Vec::with_capacity(2 * n);
    de_concat.extend_from_slice(&d_shares);
    de_concat.extend_from_slice(&e_shares);
    transport.send_u128_slice(&de_concat)?;
    let de_others = transport.recv_u128_slice(2 * n)?;
    let (d_others, e_others) = de_others.split_at(n);

    // Beaver squaring: z[k] = c + a*e + d*b + [party 0: d*e]  (all in Z_{2^128})
    // z[k] is a Q32.32 × Q32.32 product → truncate >> 32 to get Q32.32 share of x²
    // Accumulate sum(x²) shares as u64
    let mut sum_sq_share = 0u64;
    for k in 0..n {
        let d = d_shares[k].wrapping_add(d_others[k]);
        let e = e_shares[k].wrapping_add(e_others[k]);

        let mut z = sq_triples[k].c;
        z = z.wrapping_add(sq_triples[k].a.wrapping_mul(e));
        z = z.wrapping_add(d.wrapping_mul(sq_triples[k].b));
        if party == 0 {
            z = z.wrapping_add(d.wrapping_mul(e));
        }

        // Truncate from Q64.64 → Q32.32
        let x_sq_share = ((z as i128) >> FRAC_BITS_64) as i64 as u64;
        sum_sq_share = sum_sq_share.wrapping_add(x_sq_share);
    }

    // Reveal sum(x²) — single scalar leak.
    transport.send_u64(sum_sq_share)?;
    let sum_sq_other = transport.recv_u64()?;
    let sum_sq_q32 = sum_sq_share.wrapping_add(sum_sq_other);
    let sum_sq = from_fixed64(sum_sq_q32) as f64;

    // Compute normalization scale in f64
    let mean_sq = sum_sq / n as f64;
    let inv_rms = 1.0 / (mean_sq + eps as f64).sqrt();

    // Scale each share: output[k] = (inv_rms * weight[k]) * x_share[k]
    // Use i128 to multiply Q32.32 scale × Q32.32 share, then truncate >> 32
    for k in 0..n {
        let scale = inv_rms * weights[k] as f64;
        let scale_q32 = (scale * SCALE_64).round() as i64;
        let x = x_share.0[k] as i64;
        x_share.0[k] = (((scale_q32 as i128) * (x as i128)) >> FRAC_BITS_64) as i64 as u64;
    }

    Ok(())
}

/// Privacy-preserving LayerNorm under Q32.32 secret sharing.
///
/// Uses Beaver-triple-based squaring to compute variance without revealing x.
/// Only the mean scalar and sum((x-mean)²) scalar are revealed.
///
/// Protocol:
/// 1. Each party sums their share locally → exchange u64 scalars → reconstruct mean.
///    **Leaks only the mean scalar.**
/// 2. Party 0 subtracts to_fixed64(mean) from share. Now shares represent (x - mean).
/// 3. Beaver squaring on centered shares to get shares of (x-mean)².
///    Sum locally → exchange scalar sum → reconstruct sum(x_centered²).
///    **Leaks only the variance scalar.**
/// 4. Compute inv_std = 1/sqrt(var + eps) locally.
/// 5. Scale shares: share[i] = (inv_std * weight[i]) * centered_share[i].
///    Party 0 also adds to_fixed64(bias[i]).
pub fn layernorm_shared_64(
    party: u8,
    x_share: &mut SharedVec64,
    weights: &[f32],
    bias: &[f32],
    eps: f32,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    let n = x_share.len();
    assert_eq!(n, weights.len());
    assert_eq!(n, bias.len());

    // Step 1: Compute mean via scalar exchange.
    // Each party sums their share locally (in Q32.32).
    let local_sum: u64 = x_share.0.iter().fold(0u64, |acc, &v| acc.wrapping_add(v));
    transport.send_u64(local_sum)?;
    let other_sum = transport.recv_u64()?;
    let total_sum = local_sum.wrapping_add(other_sum);
    let mean = from_fixed64(total_sum) as f64 / n as f64;

    // Step 2: Party 0 subtracts mean from share to center.
    let mean_q32 = (mean * SCALE_64).round() as i64 as u64;
    if party == 0 {
        for v in x_share.0.iter_mut() {
            *v = v.wrapping_sub(mean_q32);
        }
    }

    // Step 3: Beaver squaring on centered shares.
    let sq_triples = triples.generate(n);

    let d_shares: Vec<u128> = (0..n)
        .map(|k| (x_share.0[k] as i64 as i128 as u128).wrapping_sub(sq_triples[k].a))
        .collect();
    let e_shares: Vec<u128> = (0..n)
        .map(|k| (x_share.0[k] as i64 as i128 as u128).wrapping_sub(sq_triples[k].b))
        .collect();

    let mut de_concat = Vec::with_capacity(2 * n);
    de_concat.extend_from_slice(&d_shares);
    de_concat.extend_from_slice(&e_shares);
    transport.send_u128_slice(&de_concat)?;
    let de_others = transport.recv_u128_slice(2 * n)?;
    let (d_others, e_others) = de_others.split_at(n);

    let mut sum_sq_share = 0u64;
    for k in 0..n {
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

    // Reveal sum((x-mean)²) — single scalar leak.
    transport.send_u64(sum_sq_share)?;
    let sum_sq_other = transport.recv_u64()?;
    let sum_sq_q32 = sum_sq_share.wrapping_add(sum_sq_other);
    let sum_sq = from_fixed64(sum_sq_q32) as f64;

    // Step 4: Compute normalization scale.
    let var = sum_sq / n as f64;
    let inv_std = 1.0 / (var + eps as f64).sqrt();

    // Step 5: Scale each centered share: output[k] = (inv_std * weight[k]) * x_centered[k] + bias[k]
    let bias_scale = SCALE_64;
    for k in 0..n {
        let scale = inv_std * weights[k] as f64;
        let scale_q32 = (scale * SCALE_64).round() as i64;
        let x = x_share.0[k] as i64;
        let mut out = (((scale_q32 as i128) * (x as i128)) >> FRAC_BITS_64) as i64 as u64;
        if party == 0 {
            let bias_q32 = (bias[k] as f64 * bias_scale).round() as i64 as u64;
            out = out.wrapping_add(bias_q32);
        }
        x_share.0[k] = out;
    }

    Ok(())
}

/// LayerNorm under secret sharing using reveal-and-correct.
///
/// Like `rmsnorm_shared`, reveals the full hidden state to compute the
/// normalization statistics (mean and variance) in plaintext, then locally
/// applies the scaling to each party's share.
///
/// LayerNorm differs from RMSNorm:
/// 1. Mean subtraction: `x_centered = x - mean(x)` (party 0 subtracts, party 1 doesn't)
/// 2. Additive bias: `output = x_norm * weight + bias` (party 0 adds, party 1 doesn't)
///
/// Protocol:
/// 1. Reveal x (exchange shares, reconstruct in f32)
/// 2. Compute mean, var, inv_std in plaintext
/// 3. Party 0: `share[i] = (share[i] - to_fixed(mean)) * inv_std * weight[i] + to_fixed(bias[i])`
/// 4. Party 1: `share[i] = share[i] * inv_std * weight[i]`
///
/// Uses f64 mixed-precision for scale multiplication to avoid Q16.16 quantization.
pub fn layernorm_shared(
    party: u8,
    x_share: &mut SharedVec,
    weights: &[f32],
    bias: &[f32],
    eps: f32,
    transport: &mut impl Transport,
) -> io::Result<()> {
    use crate::fixed_point::to_fixed;

    let n = x_share.len();
    assert_eq!(n, weights.len());
    assert_eq!(n, bias.len());

    // Reveal x: exchange shares and reconstruct in f32.
    transport.send_u32_slice(&x_share.0)?;
    let other_x = transport.recv_u32_slice(n)?;

    let x_plain: Vec<f32> = x_share.0.iter().zip(other_x.iter())
        .map(|(&a, &b)| from_fixed(a.wrapping_add(b)))
        .collect();

    // Compute normalization statistics in f32
    let mean = x_plain.iter().sum::<f32>() / n as f32;
    let var = x_plain.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();

    // Apply normalization to shares:
    // Conceptually: output = (x - mean) * inv_std * weight + bias
    //
    // For additive shares: x = share_0 + share_1
    // So: (x - mean) = (share_0 - mean) + share_1  [party 0 subtracts mean]
    // And: scale * centered_share gives share of scale * (x - mean)
    // Party 0 also adds the bias term.
    if party == 0 {
        for i in 0..n {
            let scale = (inv_std * weights[i]) as f64;
            // Subtract mean from share, then scale
            let centered = x_share.0[i].wrapping_sub(to_fixed(mean));
            let scaled = (scale * centered as i32 as f64).round() as i64 as i32 as u32;
            // Add bias (only party 0)
            x_share.0[i] = scaled.wrapping_add(to_fixed(bias[i]));
        }
    } else {
        for i in 0..n {
            let scale = (inv_std * weights[i]) as f64;
            let x = x_share.0[i] as i32 as f64;
            x_share.0[i] = (scale * x).round() as i64 as i32 as u32;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::dummy_triple_pair_128;
    use crate::fixed_point::{from_fixed, to_fixed, from_fixed64, to_fixed64};
    use crate::transport::memory_transport_pair;

    fn rmsnorm_plaintext(x: &mut [f32], weights: &[f32], eps: f32) {
        let n = x.len();
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..n {
            x[i] *= inv_rms * weights[i];
        }
    }

    #[test]
    fn test_rmsnorm_shared_matches_plaintext() {
        let x_vals = [1.0f32, 2.0, -1.0, 0.5];
        let weights = [1.0f32, 1.0, 1.0, 1.0];
        let eps = 1e-5f32;

        let mut x_plain = x_vals.to_vec();
        rmsnorm_plaintext(&mut x_plain, &weights, eps);

        let x_fixed: Vec<u32> = x_vals.iter().map(|&v| to_fixed(v)).collect();
        let n = x_fixed.len();

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec(x_fixed);
        let mut share1 = SharedVec(vec![0u32; n]);

        let weights_clone = weights.to_vec();
        let handle = std::thread::spawn(move || {
            rmsnorm_shared(1, &mut share1, &weights_clone, eps, &mut trans_b).unwrap();
            share1
        });

        rmsnorm_shared(0, &mut share0, &weights, eps, &mut trans_a).unwrap();
        let share1 = handle.join().unwrap();

        for i in 0..n {
            let result = from_fixed(share0.0[i].wrapping_add(share1.0[i]));
            assert!(
                (result - x_plain[i]).abs() < 0.01,
                "rmsnorm[{}]: shared={}, plain={}", i, result, x_plain[i]
            );
        }
    }

    /// Test with hidden_size=576 and RMS=8. The old Beaver-triple approach
    /// overflowed u32 here. The reveal-and-correct approach handles this fine.
    #[test]
    fn test_rmsnorm_shared_large_rms() {
        let n = 576;
        let x_vals: Vec<f32> = (0..n)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * (6.0 + (i % 5) as f32)
            })
            .collect();

        let rms: f32 = (x_vals.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
        eprintln!("Input RMS = {rms}");
        assert!(rms > 7.5, "Test setup: need RMS > 7.5, got {rms}");

        let weights: Vec<f32> = vec![1.0; n];
        let eps = 1e-5f32;

        let mut x_plain = x_vals.clone();
        rmsnorm_plaintext(&mut x_plain, &weights, eps);

        let x_fixed: Vec<u32> = x_vals.iter().map(|&v| to_fixed(v)).collect();

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec(x_fixed);
        let mut share1 = SharedVec(vec![0u32; n]);

        let weights_clone = weights.clone();
        let handle = std::thread::spawn(move || {
            rmsnorm_shared(1, &mut share1, &weights_clone, eps, &mut trans_b).unwrap();
            share1
        });

        rmsnorm_shared(0, &mut share0, &weights, eps, &mut trans_a).unwrap();
        let share1 = handle.join().unwrap();

        let mut max_diff = 0.0f32;
        for i in 0..n {
            let result = from_fixed(share0.0[i].wrapping_add(share1.0[i]));
            assert!(
                result.is_finite(),
                "rmsnorm[{}]: got NaN/Inf", i
            );
            let diff = (result - x_plain[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        eprintln!("Large-RMS test: max diff = {max_diff}");
        assert!(
            max_diff < 0.01,
            "Large-RMS rmsnorm diverged: max_diff={max_diff}"
        );
    }

    #[test]
    fn test_rmsnorm_shared_with_weights() {
        let x_vals = [1.0f32, 2.0, -1.0, 0.5];
        let weights = [0.5f32, 2.0, 1.5, 0.8];
        let eps = 1e-5f32;

        let mut x_plain = x_vals.to_vec();
        rmsnorm_plaintext(&mut x_plain, &weights, eps);

        let x_fixed: Vec<u32> = x_vals.iter().map(|&v| to_fixed(v)).collect();
        let n = x_fixed.len();

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec(x_fixed);
        let mut share1 = SharedVec(vec![0u32; n]);

        let weights_clone = weights.to_vec();
        let handle = std::thread::spawn(move || {
            rmsnorm_shared(1, &mut share1, &weights_clone, eps, &mut trans_b).unwrap();
            share1
        });

        rmsnorm_shared(0, &mut share0, &weights, eps, &mut trans_a).unwrap();
        let share1 = handle.join().unwrap();

        for i in 0..n {
            let result = from_fixed(share0.0[i].wrapping_add(share1.0[i]));
            assert!(
                (result - x_plain[i]).abs() < 0.01,
                "rmsnorm[{}]: shared={}, plain={}", i, result, x_plain[i]
            );
        }
    }

    // --- Q32.32 RMSNorm tests ---

    #[test]
    fn test_rmsnorm_shared_64_matches_plaintext() {
        let x_vals = [1.0f32, 2.0, -1.0, 0.5];
        let weights = [1.0f32, 1.0, 1.0, 1.0];
        let eps = 1e-5f32;

        let mut x_plain = x_vals.to_vec();
        rmsnorm_plaintext(&mut x_plain, &weights, eps);

        let x_fixed: Vec<u64> = x_vals.iter().map(|&v| to_fixed64(v)).collect();
        let n = x_fixed.len();

        let (mut gen0, mut gen1) = dummy_triple_pair_128(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec64(x_fixed);
        let mut share1 = SharedVec64(vec![0u64; n]);

        let weights_clone = weights.to_vec();
        let handle = std::thread::spawn(move || {
            rmsnorm_shared_64(1, &mut share1, &weights_clone, eps, &mut gen1, &mut trans_b).unwrap();
            share1
        });

        rmsnorm_shared_64(0, &mut share0, &weights, eps, &mut gen0, &mut trans_a).unwrap();
        let share1 = handle.join().unwrap();

        for i in 0..n {
            let result = from_fixed64(share0.0[i].wrapping_add(share1.0[i]));
            assert!(
                (result - x_plain[i]).abs() < 0.05,
                "Q32 rmsnorm[{}]: shared={}, plain={}", i, result, x_plain[i]
            );
        }
    }

    #[test]
    fn test_rmsnorm_shared_64_with_weights() {
        let x_vals = [1.0f32, 2.0, -1.0, 0.5];
        let weights = [0.5f32, 2.0, 1.5, 0.8];
        let eps = 1e-5f32;

        let mut x_plain = x_vals.to_vec();
        rmsnorm_plaintext(&mut x_plain, &weights, eps);

        let x_fixed: Vec<u64> = x_vals.iter().map(|&v| to_fixed64(v)).collect();
        let n = x_fixed.len();

        let (mut gen0, mut gen1) = dummy_triple_pair_128(2000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec64(x_fixed);
        let mut share1 = SharedVec64(vec![0u64; n]);

        let weights_clone = weights.to_vec();
        let handle = std::thread::spawn(move || {
            rmsnorm_shared_64(1, &mut share1, &weights_clone, eps, &mut gen1, &mut trans_b).unwrap();
            share1
        });

        rmsnorm_shared_64(0, &mut share0, &weights, eps, &mut gen0, &mut trans_a).unwrap();
        let share1 = handle.join().unwrap();

        for i in 0..n {
            let result = from_fixed64(share0.0[i].wrapping_add(share1.0[i]));
            assert!(
                (result - x_plain[i]).abs() < 0.05,
                "Q32 rmsnorm[{}]: shared={}, plain={}", i, result, x_plain[i]
            );
        }
    }

    #[test]
    fn test_rmsnorm_shared_64_large_rms() {
        let n = 576;
        let x_vals: Vec<f32> = (0..n)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * (6.0 + (i % 5) as f32)
            })
            .collect();

        let rms: f32 = (x_vals.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
        assert!(rms > 7.5, "Test setup: need RMS > 7.5, got {rms}");

        let weights: Vec<f32> = vec![1.0; n];
        let eps = 1e-5f32;

        let mut x_plain = x_vals.clone();
        rmsnorm_plaintext(&mut x_plain, &weights, eps);

        let x_fixed: Vec<u64> = x_vals.iter().map(|&v| to_fixed64(v)).collect();

        let (mut gen0, mut gen1) = dummy_triple_pair_128(3000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec64(x_fixed);
        let mut share1 = SharedVec64(vec![0u64; n]);

        let weights_clone = weights.clone();
        let handle = std::thread::spawn(move || {
            rmsnorm_shared_64(1, &mut share1, &weights_clone, eps, &mut gen1, &mut trans_b).unwrap();
            share1
        });

        rmsnorm_shared_64(0, &mut share0, &weights, eps, &mut gen0, &mut trans_a).unwrap();
        let share1 = handle.join().unwrap();

        let mut max_diff = 0.0f32;
        for i in 0..n {
            let result = from_fixed64(share0.0[i].wrapping_add(share1.0[i]));
            assert!(result.is_finite(), "Q32 rmsnorm[{}]: got NaN/Inf", i);
            let diff = (result - x_plain[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        eprintln!("Q32 Large-RMS test: max diff = {max_diff}");
        assert!(
            max_diff < 0.05,
            "Q32 Large-RMS rmsnorm diverged: max_diff={max_diff}"
        );
    }

    // --- LayerNorm tests ---

    fn layernorm_plaintext(x: &mut [f32], weights: &[f32], bias: &[f32], eps: f32) {
        let n = x.len();
        let mean = x.iter().sum::<f32>() / n as f32;
        for v in x.iter_mut() {
            *v -= mean;
        }
        let var = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..n {
            x[i] = x[i] * inv_std * weights[i] + bias[i];
        }
    }

    #[test]
    fn test_layernorm_shared_matches_plaintext() {
        let x_vals = [1.0f32, 2.0, -1.0, 0.5];
        let weights = [1.0f32, 1.0, 1.0, 1.0];
        let bias = [0.0f32, 0.0, 0.0, 0.0];
        let eps = 1e-5f32;

        let mut x_plain = x_vals.to_vec();
        layernorm_plaintext(&mut x_plain, &weights, &bias, eps);

        let x_fixed: Vec<u32> = x_vals.iter().map(|&v| to_fixed(v)).collect();
        let n = x_fixed.len();

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec(x_fixed);
        let mut share1 = SharedVec(vec![0u32; n]);

        let w_clone = weights.to_vec();
        let b_clone = bias.to_vec();
        let handle = std::thread::spawn(move || {
            layernorm_shared(1, &mut share1, &w_clone, &b_clone, eps, &mut trans_b).unwrap();
            share1
        });

        layernorm_shared(0, &mut share0, &weights, &bias, eps, &mut trans_a).unwrap();
        let share1 = handle.join().unwrap();

        for i in 0..n {
            let result = from_fixed(share0.0[i].wrapping_add(share1.0[i]));
            assert!(
                (result - x_plain[i]).abs() < 0.01,
                "layernorm[{}]: shared={}, plain={}", i, result, x_plain[i]
            );
        }
    }

    #[test]
    fn test_layernorm_shared_with_bias() {
        let x_vals = [1.0f32, 3.0, 5.0, 7.0];
        let weights = [0.5f32, 2.0, 1.5, 0.8];
        let bias = [0.1f32, -0.2, 0.3, -0.1];
        let eps = 1e-5f32;

        let mut x_plain = x_vals.to_vec();
        layernorm_plaintext(&mut x_plain, &weights, &bias, eps);

        let x_fixed: Vec<u32> = x_vals.iter().map(|&v| to_fixed(v)).collect();
        let n = x_fixed.len();

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec(x_fixed);
        let mut share1 = SharedVec(vec![0u32; n]);

        let w_clone = weights.to_vec();
        let b_clone = bias.to_vec();
        let handle = std::thread::spawn(move || {
            layernorm_shared(1, &mut share1, &w_clone, &b_clone, eps, &mut trans_b).unwrap();
            share1
        });

        layernorm_shared(0, &mut share0, &weights, &bias, eps, &mut trans_a).unwrap();
        let share1 = handle.join().unwrap();

        for i in 0..n {
            let result = from_fixed(share0.0[i].wrapping_add(share1.0[i]));
            assert!(
                (result - x_plain[i]).abs() < 0.01,
                "layernorm[{}]: shared={}, plain={}", i, result, x_plain[i]
            );
        }
    }

    // --- Q32.32 LayerNorm tests ---

    #[test]
    fn test_layernorm_shared_64_matches_plaintext() {
        let x_vals = [1.0f32, 2.0, -1.0, 0.5];
        let weights = [1.0f32, 1.0, 1.0, 1.0];
        let bias = [0.0f32, 0.0, 0.0, 0.0];
        let eps = 1e-5f32;

        let mut x_plain = x_vals.to_vec();
        layernorm_plaintext(&mut x_plain, &weights, &bias, eps);

        let x_fixed: Vec<u64> = x_vals.iter().map(|&v| to_fixed64(v)).collect();
        let n = x_fixed.len();

        let (mut gen0, mut gen1) = dummy_triple_pair_128(4000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec64(x_fixed);
        let mut share1 = SharedVec64(vec![0u64; n]);

        let w_clone = weights.to_vec();
        let b_clone = bias.to_vec();
        let handle = std::thread::spawn(move || {
            layernorm_shared_64(1, &mut share1, &w_clone, &b_clone, eps, &mut gen1, &mut trans_b).unwrap();
            share1
        });

        layernorm_shared_64(0, &mut share0, &weights, &bias, eps, &mut gen0, &mut trans_a).unwrap();
        let share1 = handle.join().unwrap();

        for i in 0..n {
            let result = from_fixed64(share0.0[i].wrapping_add(share1.0[i]));
            assert!(
                (result - x_plain[i]).abs() < 0.05,
                "Q32 layernorm[{}]: shared={}, plain={}", i, result, x_plain[i]
            );
        }
    }

    #[test]
    fn test_layernorm_shared_64_with_bias() {
        let x_vals = [1.0f32, 3.0, 5.0, 7.0];
        let weights = [0.5f32, 2.0, 1.5, 0.8];
        let bias = [0.1f32, -0.2, 0.3, -0.1];
        let eps = 1e-5f32;

        let mut x_plain = x_vals.to_vec();
        layernorm_plaintext(&mut x_plain, &weights, &bias, eps);

        let x_fixed: Vec<u64> = x_vals.iter().map(|&v| to_fixed64(v)).collect();
        let n = x_fixed.len();

        let (mut gen0, mut gen1) = dummy_triple_pair_128(5000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec64(x_fixed);
        let mut share1 = SharedVec64(vec![0u64; n]);

        let w_clone = weights.to_vec();
        let b_clone = bias.to_vec();
        let handle = std::thread::spawn(move || {
            layernorm_shared_64(1, &mut share1, &w_clone, &b_clone, eps, &mut gen1, &mut trans_b).unwrap();
            share1
        });

        layernorm_shared_64(0, &mut share0, &weights, &bias, eps, &mut gen0, &mut trans_a).unwrap();
        let share1 = handle.join().unwrap();

        for i in 0..n {
            let result = from_fixed64(share0.0[i].wrapping_add(share1.0[i]));
            assert!(
                (result - x_plain[i]).abs() < 0.05,
                "Q32 layernorm[{}]: shared={}, plain={}", i, result, x_plain[i]
            );
        }
    }
}
