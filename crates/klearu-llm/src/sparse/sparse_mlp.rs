use crate::model::mlp::Mlp;

/// Sparse MLP: only compute selected neurons in the intermediate layer.
pub fn forward_sparse(mlp: &Mlp, input: &[f32], active_neurons: &[usize]) -> Vec<f32> {
    let k = active_neurons.len();

    // Sparse gate and up projections (only selected neurons)
    let mut gate_sparse = vec![0.0f32; k];
    let mut up_sparse = vec![0.0f32; k];

    mlp.gate_proj
        .forward_sparse(input, active_neurons, &mut gate_sparse);
    mlp.up_proj
        .forward_sparse(input, active_neurons, &mut up_sparse);

    // SiLU(gate) * up for active neurons only
    for (g, u) in gate_sparse.iter_mut().zip(up_sparse.iter()) {
        *g = silu(*g) * u;
    }

    // Down projection: compute using sparse input.
    // down_proj has shape [hidden_size × intermediate_size]
    // output[j] = sum_i (down_proj[j][active_neurons[i]] * gate_sparse[i])
    let hidden_size = mlp.down_proj.out_features();
    let mut output = vec![0.0f32; hidden_size];

    for (j, out) in output.iter_mut().enumerate() {
        let w = mlp.down_proj.weights.get_weights(j);
        let mut sum = 0.0f32;
        for (idx, &neuron) in active_neurons.iter().enumerate() {
            sum += w[neuron] * gate_sparse[idx];
        }
        *out = sum;
    }

    output
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_mlp_output_shape() {
        let mlp = Mlp::new(8, 16);
        let input = vec![0.1; 8];
        let active = vec![0, 2, 5, 10];
        let output = forward_sparse(&mlp, &input, &active);
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_sparse_mlp_all_neurons_matches_dense() {
        let mut mlp = Mlp::new(4, 8);

        // Set small known weights
        for i in 0..8 {
            let mut w = vec![0.0; 4];
            w[i % 4] = 0.1;
            mlp.gate_proj.weights.set_weights(i, &w);
            mlp.up_proj.weights.set_weights(i, &w);
        }
        for j in 0..4 {
            let mut w = vec![0.0; 8];
            w[j] = 0.1;
            mlp.down_proj.weights.set_weights(j, &w);
        }

        let input = vec![1.0; 4];

        let dense_out = mlp.forward(&input);
        let all_neurons: Vec<usize> = (0..8).collect();
        let sparse_out = forward_sparse(&mlp, &input, &all_neurons);

        for (d, s) in dense_out.iter().zip(sparse_out.iter()) {
            assert!(
                (d - s).abs() < 1e-5,
                "Sparse should match dense when all neurons active: {d} vs {s}"
            );
        }
    }
}
