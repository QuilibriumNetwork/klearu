use crate::fixed_point::{to_fixed, FRAC_BITS_64, SCALE_64};
use crate::multiply::{beaver_multiply, beaver_multiply_64};
use crate::beaver::{BeaverTriple, BeaverTriple128};
use crate::sharing::{SharedVec, SharedVec64};
use crate::transport::Transport;
use std::io;

/// SiLU polynomial approximation coefficients (fitted over [-3, 3]).
///
/// silu(x) ≈ c0 + c1*x + c2*x^2
///
/// Max error over [-3, 3]: ~0.12
const SILU_C0: f32 = 0.07;
const SILU_C1: f32 = 0.5;
const SILU_C2: f32 = 0.157;

/// Evaluate SiLU polynomial approximation on shared values.
///
/// Needs 1 triple per element (for x^2).
pub fn silu_approx_shared(
    party: u8,
    x_share: &SharedVec,
    triples: &[BeaverTriple],
    transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    let n = x_share.len();
    assert!(triples.len() >= n);

    // Compute x^2 shares
    let mut x2_shares = Vec::with_capacity(n);
    for i in 0..n {
        let x2 = beaver_multiply(party, x_share.0[i], x_share.0[i], &triples[i], transport)?;
        x2_shares.push(x2);
    }

    let c0_fp = to_fixed(SILU_C0);
    let c1_fp = to_fixed(SILU_C1);
    let c2_fp = to_fixed(SILU_C2);

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = 0u32;

        // c0 (only party 0 adds public constants)
        if party == 0 {
            val = val.wrapping_add(c0_fp);
        }

        // c1 * x_share (public constant * share = local widening multiply + truncate)
        val = val.wrapping_add(
            ((c1_fp as i32 as i64 * x_share.0[i] as i32 as i64) >> 16) as i32 as u32
        );

        // c2 * x2_share
        val = val.wrapping_add(
            ((c2_fp as i32 as i64 * x2_shares[i] as i32 as i64) >> 16) as i32 as u32
        );

        result.push(val);
    }

    Ok(SharedVec(result))
}

/// SwiGLU: silu_approx(gate) * up, element-wise.
///
/// Needs 2 triples per element: 1 for SiLU's x^2 + 1 for gate*up multiply.
pub fn swiglu_shared(
    party: u8,
    gate_share: &SharedVec,
    up_share: &SharedVec,
    triples: &[BeaverTriple],
    transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    let n = gate_share.len();
    assert_eq!(n, up_share.len());
    assert!(triples.len() >= 2 * n);

    let activated = silu_approx_shared(party, gate_share, &triples[..n], transport)?;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let val = beaver_multiply(
            party, activated.0[i], up_share.0[i], &triples[n + i], transport,
        )?;
        result.push(val);
    }

    Ok(SharedVec(result))
}

// ---------------------------------------------------------------------------
// Q32.32 (u64/BeaverTriple128) polynomial approximations
// ---------------------------------------------------------------------------

/// GELU polynomial approximation coefficients (least-squares degree-3 fit over [-2, 2]).
///
/// gelu(x) ≈ g0 + g1*x + g2*x^2 + g3*x^3
///
/// Max error over [-2, 2]: ~0.1 at boundaries. Sufficient for MPC where
/// fixed-point quantization noise is comparable. LayerNorm keeps activations
/// well within [-2, 2] in practice.
///
/// Uses 2 Beaver128 triples per element (for x² and x³).
const GELU_G0: f64 = 0.0506;
const GELU_G1: f64 = 0.5;
const GELU_G2: f64 = 0.2507;
const GELU_G3: f64 = 0.0;

/// Reference GELU polynomial approximation (plaintext).
pub fn gelu_poly_approx(x: f32) -> f32 {
    (GELU_G0 + GELU_G1 * x as f64 + GELU_G2 * (x * x) as f64 + GELU_G3 * (x * x * x) as f64) as f32
}

/// Evaluate GELU polynomial approximation on Q32.32 shared values.
///
/// Needs 2 triples per element (for x^2 and x^3).
/// No reveal — purely polynomial computation on shares.
pub fn gelu_approx_shared_64(
    party: u8,
    x_share: &SharedVec64,
    triples: &[BeaverTriple128],
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let n = x_share.len();
    assert!(triples.len() >= 2 * n);

    // Compute x^2 shares (first n triples)
    let mut x2_shares = Vec::with_capacity(n);
    for i in 0..n {
        let x2 = beaver_multiply_64(party, x_share.0[i], x_share.0[i], &triples[i], transport)?;
        x2_shares.push(x2);
    }

    // Compute x^3 = x * x^2 (next n triples)
    let mut x3_shares = Vec::with_capacity(n);
    for i in 0..n {
        let x3 = beaver_multiply_64(party, x_share.0[i], x2_shares[i], &triples[n + i], transport)?;
        x3_shares.push(x3);
    }

    let g0_q32 = (GELU_G0 * SCALE_64).round() as i64;
    let g1_q32 = (GELU_G1 * SCALE_64).round() as i64;
    let g2_q32 = (GELU_G2 * SCALE_64).round() as i64;
    let g3_q32 = (GELU_G3 * SCALE_64).round() as i64;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = 0u64;

        // g0 (only party 0 adds public constants)
        if party == 0 {
            val = val.wrapping_add(g0_q32 as u64);
        }

        // g1 * x_share (public constant × share: i128 widening)
        val = val.wrapping_add(
            (((g1_q32 as i128) * (x_share.0[i] as i64 as i128)) >> FRAC_BITS_64) as i64 as u64,
        );

        // g2 * x2_share
        val = val.wrapping_add(
            (((g2_q32 as i128) * (x2_shares[i] as i64 as i128)) >> FRAC_BITS_64) as i64 as u64,
        );

        // g3 * x3_share
        val = val.wrapping_add(
            (((g3_q32 as i128) * (x3_shares[i] as i64 as i128)) >> FRAC_BITS_64) as i64 as u64,
        );

        result.push(val);
    }

    Ok(SharedVec64(result))
}

/// Q32.32 SiLU polynomial approximation on shared values.
///
/// silu(x) ≈ c0 + c1*x + c2*x^2
///
/// Needs 1 triple per element (for x^2).
pub fn silu_approx_shared_64(
    party: u8,
    x_share: &SharedVec64,
    triples: &[BeaverTriple128],
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let n = x_share.len();
    assert!(triples.len() >= n);

    // Compute x^2 shares
    let mut x2_shares = Vec::with_capacity(n);
    for i in 0..n {
        let x2 = beaver_multiply_64(party, x_share.0[i], x_share.0[i], &triples[i], transport)?;
        x2_shares.push(x2);
    }

    let c0_q32 = (SILU_C0 as f64 * SCALE_64).round() as i64;
    let c1_q32 = (SILU_C1 as f64 * SCALE_64).round() as i64;
    let c2_q32 = (SILU_C2 as f64 * SCALE_64).round() as i64;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = 0u64;

        if party == 0 {
            val = val.wrapping_add(c0_q32 as u64);
        }

        val = val.wrapping_add(
            (((c1_q32 as i128) * (x_share.0[i] as i64 as i128)) >> FRAC_BITS_64) as i64 as u64,
        );

        val = val.wrapping_add(
            (((c2_q32 as i128) * (x2_shares[i] as i64 as i128)) >> FRAC_BITS_64) as i64 as u64,
        );

        result.push(val);
    }

    Ok(SharedVec64(result))
}

/// Q32.32 SwiGLU: silu_approx_64(gate) * up, element-wise.
///
/// Needs 2 triples per element: 1 for SiLU's x^2 + 1 for gate*up multiply.
pub fn swiglu_shared_64(
    party: u8,
    gate_share: &SharedVec64,
    up_share: &SharedVec64,
    triples: &[BeaverTriple128],
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let n = gate_share.len();
    assert_eq!(n, up_share.len());
    assert!(triples.len() >= 2 * n);

    let activated = silu_approx_shared_64(party, gate_share, &triples[..n], transport)?;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let val = beaver_multiply_64(
            party, activated.0[i], up_share.0[i], &triples[n + i], transport,
        )?;
        result.push(val);
    }

    Ok(SharedVec64(result))
}

/// Reference SiLU: x * sigmoid(x).
pub fn silu_exact(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Reference SiLU polynomial approximation (plaintext).
pub fn silu_poly_approx(x: f32) -> f32 {
    SILU_C0 + SILU_C1 * x + SILU_C2 * x * x
}

/// GELU activation (tanh approximation).
pub fn gelu_exact(x: f32) -> f32 {
    let coeff = (2.0f32 / std::f32::consts::PI).sqrt();
    x * 0.5 * (1.0 + (coeff * (x + 0.044715 * x * x * x)).tanh())
}

/// GELU under Q32.32 secret sharing using reveal-and-compute.
///
/// Reveals x (u64 shares), computes GELU in f32, then re-shares:
/// party 0 gets to_fixed64(result), party 1 gets 0.
pub fn gelu_reveal_64(
    party: u8,
    x_share: &SharedVec64,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    use crate::fixed_point::{from_fixed64, to_fixed64};
    use crate::sharing::SharedVec64;

    let n = x_share.len();

    // Reveal x
    transport.send_u64_slice(&x_share.0)?;
    let other = transport.recv_u64_slice(n)?;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let x_plain = from_fixed64(x_share.0[i].wrapping_add(other[i]));
        let g = gelu_exact(x_plain);
        if party == 0 {
            result.push(to_fixed64(g));
        } else {
            result.push(0u64);
        }
    }

    Ok(SharedVec64(result))
}

/// GELU under secret sharing using reveal-and-compute.
///
/// Reveals x, computes GELU in f32, then re-shares:
/// party 0 gets to_fixed(result), party 1 gets 0.
pub fn gelu_reveal(
    party: u8,
    x_share: &SharedVec,
    transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    use crate::fixed_point::{from_fixed, to_fixed};

    let n = x_share.len();

    // Reveal x
    transport.send_u32_slice(&x_share.0)?;
    let other = transport.recv_u32_slice(n)?;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let x_plain = from_fixed(x_share.0[i].wrapping_add(other[i]));
        let g = gelu_exact(x_plain);
        if party == 0 {
            result.push(to_fixed(g));
        } else {
            result.push(0u32);
        }
    }

    Ok(SharedVec(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::{dummy_triple_pair, dummy_triple_pair_128, TripleGenerator, TripleGenerator128};
    use crate::fixed_point::{from_fixed, to_fixed, from_fixed64, to_fixed64};
    use crate::transport::memory_transport_pair;

    #[test]
    fn test_silu_poly_approx_accuracy() {
        // Max error over [-3, 3] should be < 0.15
        for i in -30..=30 {
            let x = i as f32 / 10.0;
            let exact = silu_exact(x);
            let approx = silu_poly_approx(x);
            assert!(
                (exact - approx).abs() < 0.15,
                "SiLU approx at x={}: exact={}, approx={}, err={}",
                x, exact, approx, (exact - approx).abs()
            );
        }
    }

    #[test]
    fn test_silu_shared() {
        for &x in &[0.0f32, 1.0, -1.0, 2.0] {
            let x_fixed = to_fixed(x);
            let expected = silu_poly_approx(x);

            let (mut gen0, mut gen1) = dummy_triple_pair(500);
            let t0 = gen0.generate(1);
            let t1 = gen1.generate(1);

            let (mut trans_a, mut trans_b) = memory_transport_pair();

            let share0 = SharedVec(vec![x_fixed]);
            let share1 = SharedVec(vec![0u32]);

            let handle = std::thread::spawn(move || {
                silu_approx_shared(1, &share1, &t1, &mut trans_b).unwrap()
            });

            let result0 = silu_approx_shared(0, &share0, &t0, &mut trans_a).unwrap();
            let result1 = handle.join().unwrap();

            let result = from_fixed(result0.0[0].wrapping_add(result1.0[0]));
            assert!(
                (result - expected).abs() < 1.5,
                "shared SiLU({}) = {}, expected {} (exact silu: {})",
                x, result, expected, silu_exact(x)
            );
        }
    }

    #[test]
    fn test_swiglu_shared() {
        let gate = 1.0f32;
        let up = 2.0f32;
        let expected = silu_poly_approx(gate) * up;

        let (mut gen0, mut gen1) = dummy_triple_pair(600);
        let t0 = gen0.generate(2);
        let t1 = gen1.generate(2);

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let gate0 = SharedVec(vec![to_fixed(gate)]);
        let up0 = SharedVec(vec![to_fixed(up)]);
        let gate1 = SharedVec(vec![0u32]);
        let up1 = SharedVec(vec![0u32]);

        let handle = std::thread::spawn(move || {
            swiglu_shared(1, &gate1, &up1, &t1, &mut trans_b).unwrap()
        });

        let result0 = swiglu_shared(0, &gate0, &up0, &t0, &mut trans_a).unwrap();
        let result1 = handle.join().unwrap();

        let result = from_fixed(result0.0[0].wrapping_add(result1.0[0]));
        assert!(
            (result - expected).abs() < 2.0,
            "shared SwiGLU({}, {}) = {}, expected {}",
            gate, up, result, expected
        );
    }

    #[test]
    fn test_gelu_reveal() {
        for &x in &[0.0f32, 1.0, -1.0, 2.0, -0.5] {
            let expected = gelu_exact(x);
            let x_fixed = to_fixed(x);

            let (mut trans_a, mut trans_b) = memory_transport_pair();

            let share0 = SharedVec(vec![x_fixed]);
            let share1 = SharedVec(vec![0u32]);

            let handle = std::thread::spawn(move || {
                gelu_reveal(1, &share1, &mut trans_b).unwrap()
            });

            let result0 = gelu_reveal(0, &share0, &mut trans_a).unwrap();
            let result1 = handle.join().unwrap();

            let result = from_fixed(result0.0[0].wrapping_add(result1.0[0]));
            assert!(
                (result - expected).abs() < 0.01,
                "gelu_reveal({x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_gelu_reveal_64() {
        for &x in &[0.0f32, 1.0, -1.0, 2.0, -0.5] {
            let expected = gelu_exact(x);
            let x_fixed = to_fixed64(x);

            let (mut trans_a, mut trans_b) = memory_transport_pair();

            let share0 = SharedVec64(vec![x_fixed]);
            let share1 = SharedVec64(vec![0u64]);

            let handle = std::thread::spawn(move || {
                gelu_reveal_64(1, &share1, &mut trans_b).unwrap()
            });

            let result0 = gelu_reveal_64(0, &share0, &mut trans_a).unwrap();
            let result1 = handle.join().unwrap();

            let result = from_fixed64(result0.0[0].wrapping_add(result1.0[0]));
            assert!(
                (result - expected).abs() < 0.01,
                "gelu_reveal_64({x}) = {result}, expected {expected}"
            );
        }
    }

    // --- Q32.32 polynomial approximation tests ---

    #[test]
    fn test_gelu_poly_approx_accuracy() {
        // Degree-3 polynomial is accurate in [-2, 2]; max error ~0.1 at boundaries.
        // In practice LayerNorm keeps activations well within this range.
        for i in -20..=20 {
            let x = i as f32 / 10.0;
            let exact = gelu_exact(x);
            let approx = gelu_poly_approx(x);
            assert!(
                (exact - approx).abs() < 0.15,
                "GELU poly at x={}: exact={}, approx={}, err={}",
                x, exact, approx, (exact - approx).abs()
            );
        }
    }

    #[test]
    fn test_gelu_approx_shared_64() {
        for &x in &[0.0f32, 1.0, -1.0, 2.0, -0.5] {
            let expected = gelu_poly_approx(x);
            let x_fixed = to_fixed64(x);

            let (mut gen0, mut gen1) = dummy_triple_pair_128(700);
            let t0 = gen0.generate(2);
            let t1 = gen1.generate(2);

            let (mut trans_a, mut trans_b) = memory_transport_pair();

            let share0 = SharedVec64(vec![x_fixed]);
            let share1 = SharedVec64(vec![0u64]);

            let handle = std::thread::spawn(move || {
                gelu_approx_shared_64(1, &share1, &t1, &mut trans_b).unwrap()
            });

            let result0 = gelu_approx_shared_64(0, &share0, &t0, &mut trans_a).unwrap();
            let result1 = handle.join().unwrap();

            let result = from_fixed64(result0.0[0].wrapping_add(result1.0[0]));
            assert!(
                (result - expected).abs() < 0.5,
                "gelu_approx_shared_64({x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_silu_approx_shared_64() {
        for &x in &[0.0f32, 1.0, -1.0, 2.0] {
            let expected = silu_poly_approx(x);
            let x_fixed = to_fixed64(x);

            let (mut gen0, mut gen1) = dummy_triple_pair_128(800);
            let t0 = gen0.generate(1);
            let t1 = gen1.generate(1);

            let (mut trans_a, mut trans_b) = memory_transport_pair();

            let share0 = SharedVec64(vec![x_fixed]);
            let share1 = SharedVec64(vec![0u64]);

            let handle = std::thread::spawn(move || {
                silu_approx_shared_64(1, &share1, &t1, &mut trans_b).unwrap()
            });

            let result0 = silu_approx_shared_64(0, &share0, &t0, &mut trans_a).unwrap();
            let result1 = handle.join().unwrap();

            let result = from_fixed64(result0.0[0].wrapping_add(result1.0[0]));
            assert!(
                (result - expected).abs() < 1.5,
                "silu_approx_shared_64({x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_swiglu_shared_64() {
        let gate = 1.0f32;
        let up = 2.0f32;
        let expected = silu_poly_approx(gate) * up;

        let (mut gen0, mut gen1) = dummy_triple_pair_128(900);
        let t0 = gen0.generate(2);
        let t1 = gen1.generate(2);

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let gate0 = SharedVec64(vec![to_fixed64(gate)]);
        let up0 = SharedVec64(vec![to_fixed64(up)]);
        let gate1 = SharedVec64(vec![0u64]);
        let up1 = SharedVec64(vec![0u64]);

        let handle = std::thread::spawn(move || {
            swiglu_shared_64(1, &gate1, &up1, &t1, &mut trans_b).unwrap()
        });

        let result0 = swiglu_shared_64(0, &gate0, &up0, &t0, &mut trans_a).unwrap();
        let result1 = handle.join().unwrap();

        let result = from_fixed64(result0.0[0].wrapping_add(result1.0[0]));
        assert!(
            (result - expected).abs() < 2.0,
            "swiglu_shared_64({gate}, {up}) = {result}, expected {expected}"
        );
    }
}
