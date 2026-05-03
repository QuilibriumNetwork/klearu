/// Number of fractional bits in the fixed-point representation (Q16.16).
pub const FRAC_BITS: u32 = 16;

/// Scale factor: 2^FRAC_BITS = 65536.
pub const SCALE: f32 = 65536.0;

// --- Q32.32 fixed-point (u64 shares) ---

/// Number of fractional bits in Q32.32 representation.
pub const FRAC_BITS_64: u32 = 32;

/// Scale factor for Q32.32: 2^32 = 4294967296.
pub const SCALE_64: f64 = 4294967296.0;

/// Convert an f32 to fixed-point u32 representation.
///
/// The value is scaled by 2^16 and cast to u32 (wrapping for negative values).
/// Range: approximately [-32768, 32767] with precision ~1.5e-5.
pub fn to_fixed(x: f32) -> u32 {
    (x * SCALE).round() as i32 as u32
}

/// Convert a fixed-point u32 back to f32.
pub fn from_fixed(x: u32) -> f32 {
    (x as i32 as f32) / SCALE
}

/// Truncate a fixed-point product: arithmetic right-shift by FRAC_BITS.
///
/// After multiplying two fixed-point values, the result has 2*FRAC_BITS
/// fractional bits. This shifts right to restore the correct representation.
///
/// For secret shares: truncating individual shares introduces a 1-bit error
/// in the LSB, which is acceptable for inference.
pub fn truncate(x: u32) -> u32 {
    // Arithmetic right shift (preserving sign)
    ((x as i32) >> FRAC_BITS) as u32
}

/// Fixed-point multiply: multiply two fixed-point values and truncate.
///
/// Uses widening multiplication to avoid overflow.
pub fn fixed_mul(a: u32, b: u32) -> u32 {
    let product = (a as i32 as i64).wrapping_mul(b as i32 as i64);
    (product >> FRAC_BITS) as i32 as u32
}

// --- Q32.32 functions ---

/// Convert an f32 to Q32.32 fixed-point u64 representation.
///
/// The value is scaled by 2^32. Range: approximately [-2^31, 2^31-1] with
/// precision ~2.3e-10.
pub fn to_fixed64(x: f32) -> u64 {
    ((x as f64) * SCALE_64).round() as i64 as u64
}

/// Convert a Q32.32 fixed-point u64 back to f32.
pub fn from_fixed64(x: u64) -> f32 {
    ((x as i64) as f64 / SCALE_64) as f32
}

/// Truncate a Q32.32 fixed-point product: arithmetic right-shift by FRAC_BITS_64.
pub fn truncate64(x: u64) -> u64 {
    ((x as i64) >> FRAC_BITS_64) as u64
}

/// Q32.32 fixed-point multiply: multiply two Q32.32 values and truncate.
///
/// Uses i128 widening multiplication to avoid overflow.
pub fn fixed_mul64(a: u64, b: u64) -> u64 {
    let product = (a as i64 as i128).wrapping_mul(b as i64 as i128);
    (product >> FRAC_BITS_64) as i64 as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_positive() {
        for v in [0.0f32, 1.0, 0.5, 0.25, 100.0, 3.25, 0.001] {
            let fixed = to_fixed(v);
            let back = from_fixed(fixed);
            assert!((back - v).abs() < 2e-5, "round-trip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_round_trip_negative() {
        for v in [-1.0f32, -0.5, -100.0, -0.001] {
            let fixed = to_fixed(v);
            let back = from_fixed(fixed);
            assert!((back - v).abs() < 2e-5, "round-trip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_zero() {
        assert_eq!(to_fixed(0.0), 0);
        assert_eq!(from_fixed(0), 0.0);
    }

    #[test]
    fn test_one() {
        let one = to_fixed(1.0);
        assert_eq!(one, 1 << FRAC_BITS);
        assert_eq!(from_fixed(one), 1.0);
    }

    #[test]
    fn test_negative_one() {
        let neg_one = to_fixed(-1.0);
        assert_eq!(neg_one, (-65536i32) as u32);
        assert_eq!(from_fixed(neg_one), -1.0);
    }

    #[test]
    fn test_fixed_mul() {
        let a = to_fixed(3.0);
        let b = to_fixed(4.0);
        let result = from_fixed(fixed_mul(a, b));
        assert!((result - 12.0).abs() < 2e-5, "3*4 should be 12, got {}", result);
    }

    #[test]
    fn test_fixed_mul_negative() {
        let a = to_fixed(-2.0);
        let b = to_fixed(3.0);
        let result = from_fixed(fixed_mul(a, b));
        assert!((result - (-6.0)).abs() < 2e-5, "-2*3 should be -6, got {}", result);
    }

    #[test]
    fn test_wrapping_addition() {
        let a = to_fixed(5.0);
        let b = to_fixed(-3.0);
        let sum = a.wrapping_add(b);
        let result = from_fixed(sum);
        assert!((result - 2.0).abs() < 2e-5);
    }

    // --- Q32.32 tests ---

    #[test]
    fn test_q32_round_trip_positive() {
        for v in [0.0f32, 1.0, 0.5, 0.25, 100.0, 3.25, 0.001] {
            let fixed = to_fixed64(v);
            let back = from_fixed64(fixed);
            assert!((back - v).abs() < 1e-6, "Q32 round-trip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_q32_round_trip_negative() {
        for v in [-1.0f32, -0.5, -100.0, -0.001] {
            let fixed = to_fixed64(v);
            let back = from_fixed64(fixed);
            assert!((back - v).abs() < 1e-6, "Q32 round-trip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_q32_zero() {
        assert_eq!(to_fixed64(0.0), 0);
        assert_eq!(from_fixed64(0), 0.0);
    }

    #[test]
    fn test_q32_one() {
        let one = to_fixed64(1.0);
        assert_eq!(one, 1u64 << FRAC_BITS_64);
        assert_eq!(from_fixed64(one), 1.0);
    }

    #[test]
    fn test_q32_fixed_mul() {
        let a = to_fixed64(3.0);
        let b = to_fixed64(4.0);
        let result = from_fixed64(fixed_mul64(a, b));
        assert!((result - 12.0).abs() < 1e-4, "Q32 3*4 should be 12, got {}", result);
    }

    #[test]
    fn test_q32_fixed_mul_negative() {
        let a = to_fixed64(-2.0);
        let b = to_fixed64(3.0);
        let result = from_fixed64(fixed_mul64(a, b));
        assert!((result - (-6.0)).abs() < 1e-4, "Q32 -2*3 should be -6, got {}", result);
    }

    #[test]
    fn test_q32_wrapping_addition() {
        let a = to_fixed64(5.0);
        let b = to_fixed64(-3.0);
        let sum = a.wrapping_add(b);
        let result = from_fixed64(sum);
        assert!((result - 2.0).abs() < 1e-6);
    }
}
