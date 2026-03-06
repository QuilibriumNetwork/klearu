use klearu_accel::memory::ContiguousWeightStore;
use klearu_accel::simd::dense_dot_dense_simd;

/// Linear layer with bias, backed by ContiguousWeightStore.
///
/// Weight layout: `[out_features × in_features]` (row-major).
pub struct LinearBias {
    pub weights: ContiguousWeightStore,
    pub bias: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl LinearBias {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let store = ContiguousWeightStore::new(out_features, in_features);
        Self {
            weights: store,
            bias: vec![0.0; out_features],
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

    /// Dense forward: `output[i] = dot(weights[i], input) + bias[i]`.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_features);
        debug_assert!(output.len() >= self.out_features);

        for (i, out) in output.iter_mut().enumerate().take(self.out_features) {
            let w = self.weights.get_weights(i);
            *out = dense_dot_dense_simd(&w[..self.in_features], input) + self.bias[i];
        }
    }

    /// Sparse forward: only compute selected output indices.
    pub fn forward_sparse(&self, input: &[f32], indices: &[usize], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_features);

        for (out_idx, &neuron_idx) in indices.iter().enumerate() {
            let w = self.weights.get_weights(neuron_idx);
            output[out_idx] = dense_dot_dense_simd(&w[..self.in_features], input)
                + self.bias[neuron_idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_bias_forward() {
        let mut lb = LinearBias::new(3, 2);
        lb.weights.set_weights(0, &[1.0, 0.0, 0.0]);
        lb.weights.set_weights(1, &[0.0, 1.0, 0.0]);
        lb.bias = vec![10.0, 20.0];

        let input = vec![5.0, 7.0, 9.0];
        let mut output = vec![0.0; 2];
        lb.forward(&input, &mut output);

        assert!((output[0] - 15.0).abs() < 1e-5); // 5 + 10
        assert!((output[1] - 27.0).abs() < 1e-5); // 7 + 20
    }

    #[test]
    fn test_linear_bias_sparse() {
        let mut lb = LinearBias::new(3, 4);
        for i in 0..4 {
            let mut w = vec![0.0; 3];
            w[i % 3] = 1.0;
            lb.weights.set_weights(i, &w);
        }
        lb.bias = vec![1.0, 2.0, 3.0, 4.0];

        let input = vec![10.0, 20.0, 30.0];
        let indices = vec![1, 3];
        let mut output = vec![0.0; 2];
        lb.forward_sparse(&input, &indices, &mut output);

        // neuron 1: w=[0,1,0], bias=2 → 20+2=22
        assert!((output[0] - 22.0).abs() < 1e-5);
        // neuron 3: w=[0,1,0] (3%3=0, wait: i=3, i%3=0), bias=4 → 10+4=14
        assert!((output[1] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_bias_dot_product() {
        let mut lb = LinearBias::new(4, 1);
        lb.weights.set_weights(0, &[1.0, 2.0, 3.0, 4.0]);
        lb.bias = vec![0.5];

        let input = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 1];
        lb.forward(&input, &mut output);

        // dot = 10, + 0.5 = 10.5
        assert!((output[0] - 10.5).abs() < 1e-4);
    }
}
