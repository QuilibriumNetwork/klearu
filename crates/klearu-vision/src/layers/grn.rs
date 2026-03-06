/// Global Response Normalization (ConvNeXt V2).
///
/// Forward: `x = x * (gamma * norm(x) / (mean_norm + eps)) + beta + x`.
/// Where `norm(x)` is L2 norm per-spatial-position across channels,
/// and `mean_norm` is the spatial average of these norms.

pub struct GRN {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

impl GRN {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: vec![0.0f32; dim],
            beta: vec![0.0f32; dim],
        }
    }

    pub fn dim(&self) -> usize {
        self.gamma.len()
    }

    /// In-place GRN on a single `[dim]` token.
    pub fn forward(&self, x: &mut [f32]) {
        let dim = self.gamma.len();
        debug_assert_eq!(x.len(), dim);

        // L2 norm of the vector
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        // For single-token case, mean_norm = norm
        let gx = if norm > 1e-8 { 1.0 } else { 0.0 };

        for i in 0..dim {
            x[i] = x[i] + x[i] * self.gamma[i] * gx + self.beta[i];
        }
    }

    /// In-place GRN over `[num_tokens, dim]` tokens.
    pub fn forward_batch(&self, x: &mut [f32], num_tokens: usize) {
        let dim = self.gamma.len();
        debug_assert_eq!(x.len(), num_tokens * dim);

        // Compute per-token L2 norms
        let mut norms = vec![0.0f32; num_tokens];
        for t in 0..num_tokens {
            let token = &x[t * dim..(t + 1) * dim];
            norms[t] = token.iter().map(|v| v * v).sum::<f32>().sqrt();
        }

        // Mean norm across tokens
        let mean_norm = norms.iter().sum::<f32>() / num_tokens as f32 + 1e-8;

        for t in 0..num_tokens {
            let gx = norms[t] / mean_norm;
            let token = &mut x[t * dim..(t + 1) * dim];
            for i in 0..dim {
                token[i] = token[i] + token[i] * self.gamma[i] * gx + self.beta[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grn() {
        let grn = GRN::new(4);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        grn.forward(&mut x);
        // With gamma=0 and beta=0, output = x + x*0*gx + 0 = x
        assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_grn_batch() {
        let grn = GRN::new(4);
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        grn.forward_batch(&mut x, 2);
        for &v in &x {
            assert!(v.is_finite());
        }
    }
}
