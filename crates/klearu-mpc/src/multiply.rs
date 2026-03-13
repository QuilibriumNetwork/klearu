use crate::beaver::{BeaverTriple, BeaverTriple128};
use crate::fixed_point::{FRAC_BITS, FRAC_BITS_64};
use crate::transport::Transport;
use std::io;

/// Beaver multiplication for fixed-point secret-shared values.
///
/// The protocol operates in Z_{2^64} to avoid overflow when multiplying
/// fixed-point values (which are u32 values embedded in u64).
///
/// Input: shares of x and y as u32 (embedded in u64 for the protocol).
/// Output: share of (x * y) >> FRAC_BITS as u32.
///
/// Protocol:
/// 1. d = x - a, e = y - b (reveal d and e)
/// 2. z = c + a*e + d*b + [party 0: d*e]   (all in Z_{2^64})
/// 3. Truncate z >> FRAC_BITS and return as u32
pub fn beaver_multiply(
    party: u8,
    x_share: u32,
    y_share: u32,
    triple: &BeaverTriple,
    transport: &mut impl Transport,
) -> io::Result<u32> {
    // Embed u32 shares into u64 (sign-extend for correct signed arithmetic)
    let x = x_share as i32 as i64 as u64;
    let y = y_share as i32 as i64 as u64;

    let d_share = x.wrapping_sub(triple.a);
    let e_share = y.wrapping_sub(triple.b);

    transport.send_u64(d_share)?;
    transport.send_u64(e_share)?;
    let d_other = transport.recv_u64()?;
    let e_other = transport.recv_u64()?;

    let d = d_share.wrapping_add(d_other);
    let e = e_share.wrapping_add(e_other);

    let mut z = triple.c;
    z = z.wrapping_add(triple.a.wrapping_mul(e));
    z = z.wrapping_add(d.wrapping_mul(triple.b));
    if party == 0 {
        z = z.wrapping_add(d.wrapping_mul(e));
    }

    // Truncate: arithmetic right shift by FRAC_BITS, take low 32 bits
    let result = ((z as i64) >> FRAC_BITS) as i32 as u32;
    Ok(result)
}

/// Fixed-point Beaver dot product.
pub fn beaver_dot_product(
    party: u8,
    x_shares: &[u32],
    y_shares: &[u32],
    triples: &[BeaverTriple],
    transport: &mut impl Transport,
) -> io::Result<u32> {
    let n = x_shares.len();
    assert_eq!(n, y_shares.len());
    assert_eq!(n, triples.len());

    let d_shares: Vec<u64> = x_shares.iter().zip(triples.iter())
        .map(|(&x, t)| (x as i32 as i64 as u64).wrapping_sub(t.a))
        .collect();
    let e_shares: Vec<u64> = y_shares.iter().zip(triples.iter())
        .map(|(&y, t)| (y as i32 as i64 as u64).wrapping_sub(t.b))
        .collect();

    transport.send_u64_slice(&d_shares)?;
    transport.send_u64_slice(&e_shares)?;
    let d_others = transport.recv_u64_slice(n)?;
    let e_others = transport.recv_u64_slice(n)?;

    let mut result = 0u32;
    for i in 0..n {
        let d = d_shares[i].wrapping_add(d_others[i]);
        let e = e_shares[i].wrapping_add(e_others[i]);

        let mut z = triples[i].c;
        z = z.wrapping_add(triples[i].a.wrapping_mul(e));
        z = z.wrapping_add(d.wrapping_mul(triples[i].b));
        if party == 0 {
            z = z.wrapping_add(d.wrapping_mul(e));
        }

        let truncated = ((z as i64) >> FRAC_BITS) as i32 as u32;
        result = result.wrapping_add(truncated);
    }

    Ok(result)
}

/// Server-weight multiply: server knows weight w, input x is secret-shared.
pub fn server_weight_multiply(
    party: u8,
    weight: u32,
    x_share: u32,
    triple: &BeaverTriple,
    transport: &mut impl Transport,
) -> io::Result<u32> {
    let w = if party == 1 { weight as i32 as i64 as u64 } else { 0u64 };
    let x = x_share as i32 as i64 as u64;

    let d_share = w.wrapping_sub(triple.a);
    let e_share = x.wrapping_sub(triple.b);

    transport.send_u64(d_share)?;
    transport.send_u64(e_share)?;
    let d_other = transport.recv_u64()?;
    let e_other = transport.recv_u64()?;

    let d = d_share.wrapping_add(d_other);
    let e = e_share.wrapping_add(e_other);

    let mut z = triple.c;
    z = z.wrapping_add(triple.a.wrapping_mul(e));
    z = z.wrapping_add(d.wrapping_mul(triple.b));
    if party == 0 {
        z = z.wrapping_add(d.wrapping_mul(e));
    }

    Ok(((z as i64) >> FRAC_BITS) as i32 as u32)
}

// --- Q32.32 Beaver multiply (u64 in/out, Z_{2^128} protocol, >>32 truncation) ---

/// Beaver multiplication for Q32.32 fixed-point secret-shared values.
///
/// Input/output are u64 shares. The protocol works in Z_{2^128} using
/// BeaverTriple128 (u128 fields) to avoid overflow. The d and e masks
/// are u128 (u64 share embedded via sign-extension). After the Beaver
/// formula, truncate >> 32 to get Q32.32 output as u64.
pub fn beaver_multiply_64(
    party: u8,
    x_share: u64,
    y_share: u64,
    triple: &BeaverTriple128,
    transport: &mut impl Transport,
) -> io::Result<u64> {
    // Embed u64 Q32.32 shares into u128 (sign-extend)
    let x = x_share as i64 as i128 as u128;
    let y = y_share as i64 as i128 as u128;

    let d_share = x.wrapping_sub(triple.a);
    let e_share = y.wrapping_sub(triple.b);

    transport.send_u128(d_share)?;
    transport.send_u128(e_share)?;
    let d_other = transport.recv_u128()?;
    let e_other = transport.recv_u128()?;

    let d = d_share.wrapping_add(d_other);
    let e = e_share.wrapping_add(e_other);

    let mut z = triple.c;
    z = z.wrapping_add(triple.a.wrapping_mul(e));
    z = z.wrapping_add(d.wrapping_mul(triple.b));
    if party == 0 {
        z = z.wrapping_add(d.wrapping_mul(e));
    }

    // Truncate: arithmetic right shift by FRAC_BITS_64, take low 64 bits
    let result = ((z as i128) >> FRAC_BITS_64) as i64 as u64;
    Ok(result)
}

/// Q32.32 Beaver dot product.
pub fn beaver_dot_product_64(
    party: u8,
    x_shares: &[u64],
    y_shares: &[u64],
    triples: &[BeaverTriple128],
    transport: &mut impl Transport,
) -> io::Result<u64> {
    let n = x_shares.len();
    assert_eq!(n, y_shares.len());
    assert_eq!(n, triples.len());

    let d_shares: Vec<u128> = x_shares.iter().zip(triples.iter())
        .map(|(&x, t)| (x as i64 as i128 as u128).wrapping_sub(t.a))
        .collect();
    let e_shares: Vec<u128> = y_shares.iter().zip(triples.iter())
        .map(|(&y, t)| (y as i64 as i128 as u128).wrapping_sub(t.b))
        .collect();

    transport.send_u128_slice(&d_shares)?;
    transport.send_u128_slice(&e_shares)?;
    let d_others = transport.recv_u128_slice(n)?;
    let e_others = transport.recv_u128_slice(n)?;

    let mut result = 0u64;
    for i in 0..n {
        let d = d_shares[i].wrapping_add(d_others[i]);
        let e = e_shares[i].wrapping_add(e_others[i]);

        let mut z = triples[i].c;
        z = z.wrapping_add(triples[i].a.wrapping_mul(e));
        z = z.wrapping_add(d.wrapping_mul(triples[i].b));
        if party == 0 {
            z = z.wrapping_add(d.wrapping_mul(e));
        }

        let truncated = ((z as i128) >> FRAC_BITS_64) as i64 as u64;
        result = result.wrapping_add(truncated);
    }

    Ok(result)
}

/// Batched element-wise Q32.32 Beaver multiply.
///
/// Returns `z[i] = x[i] * y[i]` for each element. Uses a single round-trip:
/// concatenates all d and e shares, sends once, receives once.
///
/// Cost: n BeaverTriple128, 1 send + 1 recv of 2n u128 values.
pub fn beaver_multiply_elementwise_64(
    party: u8,
    x_shares: &[u64],
    y_shares: &[u64],
    triples: &[BeaverTriple128],
    transport: &mut impl Transport,
) -> io::Result<Vec<u64>> {
    let n = x_shares.len();
    assert_eq!(n, y_shares.len());
    assert_eq!(n, triples.len());

    // Compute d = x - a, e = y - b for all elements
    let mut de_concat = Vec::with_capacity(2 * n);
    for i in 0..n {
        de_concat.push((x_shares[i] as i64 as i128 as u128).wrapping_sub(triples[i].a));
    }
    for i in 0..n {
        de_concat.push((y_shares[i] as i64 as i128 as u128).wrapping_sub(triples[i].b));
    }

    // Single round-trip
    transport.send_u128_slice(&de_concat)?;
    let de_others = transport.recv_u128_slice(2 * n)?;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let d = de_concat[i].wrapping_add(de_others[i]);
        let e = de_concat[n + i].wrapping_add(de_others[n + i]);

        let mut z = triples[i].c;
        z = z.wrapping_add(triples[i].a.wrapping_mul(e));
        z = z.wrapping_add(d.wrapping_mul(triples[i].b));
        if party == 0 {
            z = z.wrapping_add(d.wrapping_mul(e));
        }

        result.push(((z as i128) >> FRAC_BITS_64) as i64 as u64);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::{dummy_triple_pair, dummy_triple_pair_128, TripleGenerator, TripleGenerator128};
    use crate::fixed_point::{from_fixed, to_fixed, from_fixed64, to_fixed64};
    use crate::transport::memory_transport_pair;

    fn run_beaver_multiply(x: f32, y: f32) -> f32 {
        let x_fixed = to_fixed(x);
        let y_fixed = to_fixed(y);

        let (mut gen0, mut gen1) = dummy_triple_pair(100);
        let t0 = gen0.generate(1).pop().unwrap();
        let t1 = gen1.generate(1).pop().unwrap();

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            beaver_multiply(1, 0u32, y_fixed, &t1, &mut trans_b).unwrap()
        });

        let z0 = beaver_multiply(0, x_fixed, 0u32, &t0, &mut trans_a).unwrap();
        let z1 = handle.join().unwrap();

        from_fixed(z0.wrapping_add(z1))
    }

    #[test]
    fn test_beaver_multiply_simple() {
        let result = run_beaver_multiply(3.0, 4.0);
        assert!((result - 12.0).abs() < 1.0, "3*4 should be ~12, got {}", result);
    }

    #[test]
    fn test_beaver_multiply_negative() {
        let result = run_beaver_multiply(-2.0, 3.0);
        assert!((result - (-6.0)).abs() < 1.0, "-2*3 should be ~-6, got {}", result);
    }

    #[test]
    fn test_beaver_multiply_small() {
        let result = run_beaver_multiply(0.5, 0.5);
        assert!((result - 0.25).abs() < 1.0, "0.5*0.5 should be ~0.25, got {}", result);
    }

    #[test]
    fn test_beaver_dot_product() {
        let x_vals = [1.0f32, 2.0, 3.0];
        let y_vals = [4.0f32, 5.0, 6.0];
        let expected = 32.0;

        let x_fixed: Vec<u32> = x_vals.iter().map(|&v| to_fixed(v)).collect();
        let y_fixed: Vec<u32> = y_vals.iter().map(|&v| to_fixed(v)).collect();

        let (mut gen0, mut gen1) = dummy_triple_pair(200);
        let t0 = gen0.generate(3);
        let t1 = gen1.generate(3);

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let x0 = x_fixed;
        let y0 = vec![0u32; 3];
        let x1 = vec![0u32; 3];
        let y1 = y_fixed;

        let handle = std::thread::spawn(move || {
            beaver_dot_product(1, &x1, &y1, &t1, &mut trans_b).unwrap()
        });

        let z0 = beaver_dot_product(0, &x0, &y0, &t0, &mut trans_a).unwrap();
        let z1 = handle.join().unwrap();

        let result = from_fixed(z0.wrapping_add(z1));
        assert!((result - expected).abs() < 1.0, "dot product should be ~{}, got {}", expected, result);
    }

    #[test]
    fn test_server_weight_multiply() {
        let w_fixed = to_fixed(3.0);
        let x_fixed = to_fixed(4.0);

        let (mut gen0, mut gen1) = dummy_triple_pair(300);
        let t0 = gen0.generate(1).pop().unwrap();
        let t1 = gen1.generate(1).pop().unwrap();

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            server_weight_multiply(1, w_fixed, 0u32, &t1, &mut trans_b).unwrap()
        });

        let z0 = server_weight_multiply(0, 0, x_fixed, &t0, &mut trans_a).unwrap();
        let z1 = handle.join().unwrap();

        let result = from_fixed(z0.wrapping_add(z1));
        assert!((result - 12.0).abs() < 1.0, "weight*input should be ~12, got {}", result);
    }

    // --- Q32.32 Beaver multiply tests ---

    fn run_beaver_multiply_64(x: f32, y: f32) -> f32 {
        let x_fixed = to_fixed64(x);
        let y_fixed = to_fixed64(y);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(400);
        let t0 = gen0.generate(1).pop().unwrap();
        let t1 = gen1.generate(1).pop().unwrap();

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            beaver_multiply_64(1, 0u64, y_fixed, &t1, &mut trans_b).unwrap()
        });

        let z0 = beaver_multiply_64(0, x_fixed, 0u64, &t0, &mut trans_a).unwrap();
        let z1 = handle.join().unwrap();

        from_fixed64(z0.wrapping_add(z1))
    }

    #[test]
    fn test_beaver_multiply_64_simple() {
        let result = run_beaver_multiply_64(3.0, 4.0);
        assert!((result - 12.0).abs() < 1.0, "Q32 3*4 should be ~12, got {}", result);
    }

    #[test]
    fn test_beaver_multiply_64_negative() {
        let result = run_beaver_multiply_64(-2.0, 3.0);
        assert!((result - (-6.0)).abs() < 1.0, "Q32 -2*3 should be ~-6, got {}", result);
    }

    #[test]
    fn test_beaver_dot_product_64() {
        let x_vals = [1.0f32, 2.0, 3.0];
        let y_vals = [4.0f32, 5.0, 6.0];
        let expected = 32.0;

        let x_fixed: Vec<u64> = x_vals.iter().map(|&v| to_fixed64(v)).collect();
        let y_fixed: Vec<u64> = y_vals.iter().map(|&v| to_fixed64(v)).collect();

        let (mut gen0, mut gen1) = dummy_triple_pair_128(500);
        let t0 = gen0.generate(3);
        let t1 = gen1.generate(3);

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let x0 = x_fixed;
        let y0 = vec![0u64; 3];
        let x1 = vec![0u64; 3];
        let y1 = y_fixed;

        let handle = std::thread::spawn(move || {
            beaver_dot_product_64(1, &x1, &y1, &t1, &mut trans_b).unwrap()
        });

        let z0 = beaver_dot_product_64(0, &x0, &y0, &t0, &mut trans_a).unwrap();
        let z1 = handle.join().unwrap();

        let result = from_fixed64(z0.wrapping_add(z1));
        assert!((result - expected).abs() < 1.0, "Q32 dot product should be ~{}, got {}", expected, result);
    }

    #[test]
    fn test_beaver_multiply_elementwise_64() {
        let x_vals = [3.0f32, -2.0, 0.5];
        let y_vals = [4.0f32, 3.0, 0.5];
        let expected = [12.0f32, -6.0, 0.25];

        let x_fixed: Vec<u64> = x_vals.iter().map(|&v| to_fixed64(v)).collect();
        let y_fixed: Vec<u64> = y_vals.iter().map(|&v| to_fixed64(v)).collect();

        let (mut gen0, mut gen1) = dummy_triple_pair_128(600);
        let t0 = gen0.generate(3);
        let t1 = gen1.generate(3);

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let x0 = x_fixed;
        let y0 = vec![0u64; 3];
        let x1 = vec![0u64; 3];
        let y1 = y_fixed;

        let handle = std::thread::spawn(move || {
            beaver_multiply_elementwise_64(1, &x1, &y1, &t1, &mut trans_b).unwrap()
        });

        let z0 = beaver_multiply_elementwise_64(0, &x0, &y0, &t0, &mut trans_a).unwrap();
        let z1 = handle.join().unwrap();

        for i in 0..3 {
            let result = from_fixed64(z0[i].wrapping_add(z1[i]));
            assert!(
                (result - expected[i]).abs() < 1.0,
                "elementwise[{i}]: expected {}, got {}", expected[i], result
            );
        }
    }
}
