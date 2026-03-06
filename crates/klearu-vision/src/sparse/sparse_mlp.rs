use crate::layers::{LinearBias, gelu_inplace};

/// Sparse MLP forward: select top-k neurons from fc1, compute GELU,
/// then scatter-back through fc2.
///
/// Same pattern as klearu-llm sparse MLP but with GELU instead of SiLU.
pub fn forward_sparse_mlp(
    fc1: &LinearBias,
    fc2: &LinearBias,
    input: &[f32],
    active_neurons: &[usize],
) -> Vec<f32> {
    let k = active_neurons.len();
    let dim = fc2.out_features();

    // Sparse fc1: only compute selected neurons
    let mut fc1_out = vec![0.0f32; k];
    fc1.forward_sparse(input, active_neurons, &mut fc1_out);

    // GELU activation on sparse intermediate
    gelu_inplace(&mut fc1_out);

    // Scatter-back via fc2: for each output dim, dot product with selected fc2 columns
    let mut output = vec![0.0f32; dim];
    for out_d in 0..dim {
        let w_row = fc2.weights.get_weights(out_d);
        let mut sum = fc2.bias[out_d];
        for (sparse_idx, &neuron_idx) in active_neurons.iter().enumerate() {
            if neuron_idx < fc2.in_features() {
                sum += w_row[neuron_idx] * fc1_out[sparse_idx];
            }
        }
        output[out_d] = sum;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_mlp_all_neurons_matches_dense() {
        let mut fc1 = LinearBias::new(4, 8);
        let mut fc2 = LinearBias::new(8, 4);

        // Set some non-zero weights
        for i in 0..8 {
            let mut w = vec![0.0; 4];
            w[i % 4] = 0.1 * (i as f32 + 1.0);
            fc1.weights.set_weights(i, &w);
            fc1.bias[i] = 0.01 * i as f32;
        }
        for i in 0..4 {
            let mut w = vec![0.0; 8];
            w[i] = 0.2;
            w[(i + 1) % 8] = 0.1;
            fc2.weights.set_weights(i, &w);
            fc2.bias[i] = 0.05;
        }

        let input = vec![1.0, 2.0, 3.0, 4.0];

        // Dense forward
        let mut dense_fc1 = vec![0.0f32; 8];
        fc1.forward(&input, &mut dense_fc1);
        gelu_inplace(&mut dense_fc1);
        let mut dense_out = vec![0.0f32; 4];
        fc2.forward(&dense_fc1, &mut dense_out);

        // Sparse with all neurons selected
        let all_neurons: Vec<usize> = (0..8).collect();
        let sparse_out = forward_sparse_mlp(&fc1, &fc2, &input, &all_neurons);

        for i in 0..4 {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-4,
                "dim[{i}]: dense={}, sparse={}", dense_out[i], sparse_out[i]
            );
        }
    }
}
