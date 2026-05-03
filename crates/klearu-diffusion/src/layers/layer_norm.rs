//! LayerNorm — normalises over the last dimension, then scales+shifts by γ, β.

pub struct LayerNorm {
    pub gamma: Vec<f32>, // scale, length = normalized_shape
    pub beta: Vec<f32>,  // shift, length = normalized_shape
    pub eps: f32,
    pub normalized_shape: usize,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize, eps: f32) -> Self {
        Self {
            gamma: vec![1.0; normalized_shape],
            beta: vec![0.0; normalized_shape],
            eps,
            normalized_shape,
        }
    }

    /// Apply over the last dimension of the flat input. `x.len()` must be a
    /// multiple of `normalized_shape`. Each chunk is normalised independently.
    pub fn forward_inplace(&self, x: &mut [f32]) {
        let n = self.normalized_shape;
        debug_assert_eq!(x.len() % n, 0);
        // Route big LayerNorms to Metal. Sub-threshold (CLIP final-LN at
        // 77×768=59K) inline scalar loop only.
        #[cfg(feature = "metal")]
        if x.len() >= 1 << 14 {
            crate::metal_backend::layer_norm_metal(x, &self.gamma, &self.beta, n, self.eps);
            return;
        }
        let inv_n = 1.0 / n as f32;
        for chunk in x.chunks_mut(n) {
            let mean: f32 = chunk.iter().sum::<f32>() * inv_n;
            let var: f32 = chunk.iter().map(|v| (v - mean).powi(2)).sum::<f32>() * inv_n;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for (i, v) in chunk.iter_mut().enumerate() {
                *v = (*v - mean) * inv_std * self.gamma[i] + self.beta[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_centres_and_scales() {
        let ln = LayerNorm::new(4, 1e-5);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        ln.forward_inplace(&mut x);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean ≈ 0, got {mean}");
        assert!((var - 1.0).abs() < 1e-3, "var ≈ 1, got {var}");
    }

    #[test]
    fn ln_per_row() {
        let ln = LayerNorm::new(3, 1e-5);
        // 2 rows, normalise each independently
        let mut x = vec![1.0, 2.0, 3.0, 100.0, 200.0, 300.0];
        ln.forward_inplace(&mut x);
        // Both rows should normalise to roughly the same shape: -1.22, 0, 1.22
        assert!((x[0] - x[3]).abs() < 1e-3);
        assert!((x[1] - x[4]).abs() < 1e-3);
        assert!((x[2] - x[5]).abs() < 1e-3);
    }
}
