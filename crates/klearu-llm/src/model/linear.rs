use klearu_accel::memory::ContiguousWeightStore;
use klearu_accel::simd::dense_dot_dense_simd;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Bias-free linear layer backed by ContiguousWeightStore.
///
/// Weight layout: `[out_features × in_features]` (row-major, each row is one output neuron).
/// Uses SIMD-accelerated dot products for forward pass.
pub struct Linear {
    pub weights: ContiguousWeightStore,
    /// Pre-quantized weights for Q32.32 MPC: each f32 weight → `(w as f64 * 2^32).round() as i64`.
    /// Layout mirrors `weights` (same stride). Call `sync_q32()` after modifying weights.
    pub weights_q32: Vec<i64>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let store = ContiguousWeightStore::new(out_features, in_features);
        // On WASM, don't pre-allocate q32 to save memory (~2× reduction).
        // q32 weights are computed on-the-fly via q32_weights() instead.
        #[cfg(not(target_arch = "wasm32"))]
        let weights_q32 = vec![0i64; store.as_raw_slice().len()];
        #[cfg(target_arch = "wasm32")]
        let weights_q32 = Vec::new();
        Self {
            weights: store,
            weights_q32,
            in_features,
            out_features,
        }
    }

    /// Synchronize `weights_q32` from current `weights`.
    /// Must be called after modifying weights and before MPC inference.
    /// No-op on WASM (q32 computed on-the-fly instead).
    pub fn sync_q32(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let raw = self.weights.as_raw_slice();
            self.weights_q32.resize(raw.len(), 0);
            for (dst, &src) in self.weights_q32.iter_mut().zip(raw.iter()) {
                *dst = (src as f64 * 4294967296.0).round() as i64;
            }
        }
    }

    /// Get Q32.32 pre-quantized weights. Returns cached weights on native,
    /// computes on-the-fly on WASM (slower but saves ~2× memory).
    pub fn q32_weights(&self) -> std::borrow::Cow<'_, [i64]> {
        if !self.weights_q32.is_empty() {
            std::borrow::Cow::Borrowed(&self.weights_q32)
        } else {
            let raw = self.weights.as_raw_slice();
            let q32: Vec<i64> = raw.iter()
                .map(|&v| (v as f64 * 4294967296.0).round() as i64)
                .collect();
            std::borrow::Cow::Owned(q32)
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Minimum total FLOPs (out × in) before switching to rayon parallelism.
    #[cfg(feature = "parallel")]
    const PARALLEL_THRESHOLD: usize = 4_000_000;

    /// Dense forward: `output[i] = dot(weights[i], input)` for all output neurons.
    /// Auto-parallelizes with rayon when `out_features * in_features >= 4M`
    /// and the `parallel` feature is enabled.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_features);
        debug_assert!(output.len() >= self.out_features);

        #[cfg(feature = "parallel")]
        {
            let total_work = self.out_features * self.in_features;
            if total_work >= Self::PARALLEL_THRESHOLD {
                let in_feat = self.in_features;
                let weights = &self.weights;
                output[..self.out_features]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, out)| {
                        let w = weights.get_weights(i);
                        *out = dense_dot_dense_simd(&w[..in_feat], input);
                    });
                return;
            }
        }

        for (i, out) in output.iter_mut().enumerate().take(self.out_features) {
            let w = self.weights.get_weights(i);
            *out = dense_dot_dense_simd(&w[..self.in_features], input);
        }
    }

    /// Sparse forward: only compute selected output indices.
    pub fn forward_sparse(&self, input: &[f32], indices: &[usize], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_features);

        for (out_idx, &neuron_idx) in indices.iter().enumerate() {
            let w = self.weights.get_weights(neuron_idx);
            output[out_idx] = dense_dot_dense_simd(&w[..self.in_features], input);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let mut linear = Linear::new(4, 3);

        // Set weights: row 0 = [1,0,0,0], row 1 = [0,1,0,0], row 2 = [0,0,1,0]
        linear.weights.set_weights(0, &[1.0, 0.0, 0.0, 0.0]);
        linear.weights.set_weights(1, &[0.0, 1.0, 0.0, 0.0]);
        linear.weights.set_weights(2, &[0.0, 0.0, 1.0, 0.0]);

        let input = vec![10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0; 3];
        linear.forward(&input, &mut output);

        assert!((output[0] - 10.0).abs() < 1e-5);
        assert!((output[1] - 20.0).abs() < 1e-5);
        assert!((output[2] - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_dot_product() {
        let mut linear = Linear::new(4, 2);
        linear.weights.set_weights(0, &[1.0, 2.0, 3.0, 4.0]);
        linear.weights.set_weights(1, &[4.0, 3.0, 2.0, 1.0]);

        let input = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 2];
        linear.forward(&input, &mut output);

        assert!((output[0] - 10.0).abs() < 1e-4);
        assert!((output[1] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_linear_sparse_matches_dense() {
        let mut linear = Linear::new(4, 4);
        for i in 0..4 {
            let mut w = vec![0.0; 4];
            w[i] = 1.0;
            linear.weights.set_weights(i, &w);
        }

        let input = vec![10.0, 20.0, 30.0, 40.0];

        let mut dense_out = vec![0.0; 4];
        linear.forward(&input, &mut dense_out);

        let indices = vec![1, 3];
        let mut sparse_out = vec![0.0; 2];
        linear.forward_sparse(&input, &indices, &mut sparse_out);

        assert!((sparse_out[0] - dense_out[1]).abs() < 1e-5);
        assert!((sparse_out[1] - dense_out[3]).abs() < 1e-5);
    }
}
