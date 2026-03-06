/// GELU activation (tanh approximation, matches PyTorch default).
///
/// gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: f32) -> f32 {
    let coeff = (2.0f32 / std::f32::consts::PI).sqrt();
    x * 0.5 * (1.0 + (coeff * (x + 0.044715 * x * x * x)).tanh())
}

/// Apply GELU activation in-place.
pub fn gelu_inplace(x: &mut [f32]) {
    let coeff = (2.0f32 / std::f32::consts::PI).sqrt();
    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        *v = *v * 0.5 * (1.0 + (coeff * (*v + 0.044715 * x3)).tanh());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_known_values() {
        // gelu(0) = 0
        assert!((gelu(0.0)).abs() < 1e-6);

        // gelu(1) ≈ 0.841
        assert!((gelu(1.0) - 0.841).abs() < 0.002, "gelu(1)={}", gelu(1.0));

        // gelu(-1) ≈ -0.159
        assert!((gelu(-1.0) - (-0.159)).abs() < 0.002, "gelu(-1)={}", gelu(-1.0));

        // gelu(x) ≈ x for large positive x
        assert!((gelu(5.0) - 5.0).abs() < 0.01);

        // gelu(x) ≈ 0 for large negative x
        assert!(gelu(-5.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_inplace() {
        let mut x = vec![0.0, 1.0, -1.0, 2.0];
        let expected: Vec<f32> = x.iter().map(|&v| gelu(v)).collect();
        gelu_inplace(&mut x);
        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gelu_monotonic_positive() {
        // GELU should be monotonically increasing for x > ~-0.75
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let y = (i + 1) as f32 * 0.1;
            assert!(gelu(y) >= gelu(x), "gelu({}) < gelu({})", y, x);
        }
    }
}
