use klearu_accel::memory::ContiguousWeightStore;
use klearu_accel::simd::dense_dot_dense_simd;
use rayon::prelude::*;

/// Minimum output rows before Rayon parallelism kicks in.
/// Below this threshold the thread-pool overhead exceeds the compute savings.
const PAR_ROW_THRESHOLD: usize = 512;
const PAR_MIN_CHUNK: usize = 64;

/// Bias-free linear layer backed by ContiguousWeightStore.
///
/// Weight layout: `[out_features × in_features]` (row-major, each row is one output neuron).
/// Uses SIMD-accelerated dot products and Rayon parallelism for forward pass.
pub struct Linear {
    pub weights: ContiguousWeightStore,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: ContiguousWeightStore::new(out_features, in_features),
            in_features,
            out_features,
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Dense forward: `output[i] = dot(weights[i], input)` for all output neurons.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_features);
        debug_assert!(output.len() >= self.out_features);

        let in_features = self.in_features;
        let weights = &self.weights;
        let dst = &mut output[..self.out_features];

        if dst.len() >= PAR_ROW_THRESHOLD {
            dst.par_iter_mut()
                .with_min_len(PAR_MIN_CHUNK)
                .enumerate()
                .for_each(|(i, out)| {
                    let w = weights.get_weights(i);
                    *out = dense_dot_dense_simd(&w[..in_features], input);
                });
        } else {
            for (i, out) in dst.iter_mut().enumerate() {
                let w = weights.get_weights(i);
                *out = dense_dot_dense_simd(&w[..in_features], input);
            }
        }
    }

    /// Sparse forward: only compute selected output indices.
    pub fn forward_sparse(&self, input: &[f32], indices: &[usize], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_features);

        let in_features = self.in_features;
        let weights = &self.weights;
        let dst = &mut output[..indices.len()];

        if dst.len() >= PAR_ROW_THRESHOLD {
            dst.par_iter_mut()
                .with_min_len(PAR_MIN_CHUNK)
                .zip(indices.par_iter())
                .for_each(|(out, &neuron_idx)| {
                    let w = weights.get_weights(neuron_idx);
                    *out = dense_dot_dense_simd(&w[..in_features], input);
                });
        } else {
            for (out, &neuron_idx) in dst.iter_mut().zip(indices.iter()) {
                let w = weights.get_weights(neuron_idx);
                *out = dense_dot_dense_simd(&w[..in_features], input);
            }
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
