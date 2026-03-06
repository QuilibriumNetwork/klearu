use super::predictor_store::VisionPredictorStore;

/// Per-block activation records for calibration.
pub struct BlockActivations {
    /// Input to this block (pre-LayerNorm hidden state): `[seq_len, dim]`.
    pub input: Vec<f32>,
    /// MLP intermediate (fc1 output, pre-GELU): `[seq_len, mlp_hidden]`.
    pub mlp_intermediate: Vec<f32>,
    /// Attention output per head (before output projection): `[num_heads, seq_len, head_dim]`.
    pub attn_head_outputs: Vec<f32>,
    /// Block dimension.
    pub dim: usize,
    /// MLP hidden dimension.
    pub mlp_hidden: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Sequence length.
    pub seq_len: usize,
}

/// Calibrate a VisionPredictorStore from recorded block activations.
///
/// For each block:
/// - MLP predictor: trained to predict which fc1 neurons have top-k magnitude
/// - Head predictor: trained to predict which heads produce top-k norm output
///
/// `all_activations`: one `Vec<BlockActivations>` per calibration image.
/// Each inner Vec has one entry per block.
pub fn calibrate_predictors(
    store: &mut VisionPredictorStore,
    all_activations: &[Vec<BlockActivations>],
    lr: f32,
    epochs: usize,
) {
    let num_blocks = store.num_blocks();

    for epoch in 0..epochs {
        let _ = epoch;
        for activations in all_activations {
            assert_eq!(activations.len(), num_blocks);
            for (block_idx, block_act) in activations.iter().enumerate() {
                // --- MLP predictor training ---
                // Target: importance of each fc1 neuron (mean absolute activation across tokens)
                let mlp_hidden = block_act.mlp_hidden;
                let seq_len = block_act.seq_len;
                let mut neuron_importance = vec![0.0f32; mlp_hidden];
                for t in 0..seq_len {
                    for n in 0..mlp_hidden {
                        neuron_importance[n] +=
                            block_act.mlp_intermediate[t * mlp_hidden + n].abs();
                    }
                }
                // Normalize to [0, 1]
                let max_imp = neuron_importance
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                if max_imp > 0.0 {
                    for v in &mut neuron_importance {
                        *v /= max_imp;
                    }
                }

                // Use mean input across tokens as predictor input
                let dim = block_act.dim;
                let mut mean_input = vec![0.0f32; dim];
                for t in 0..seq_len {
                    for d in 0..dim {
                        mean_input[d] += block_act.input[t * dim + d];
                    }
                }
                let inv_seq = 1.0 / seq_len as f32;
                for v in &mut mean_input {
                    *v *= inv_seq;
                }

                store.mlp_predictors[block_idx].train_step(
                    &mean_input,
                    &neuron_importance,
                    lr,
                );

                // --- Head predictor training ---
                let num_heads = block_act.num_heads;
                let head_dim = dim / num_heads;
                let mut head_importance = vec![0.0f32; num_heads];
                for h in 0..num_heads {
                    let mut norm_sq = 0.0f32;
                    for t in 0..seq_len {
                        for d in 0..head_dim {
                            let val = block_act.attn_head_outputs
                                [h * seq_len * head_dim + t * head_dim + d];
                            norm_sq += val * val;
                        }
                    }
                    head_importance[h] = norm_sq.sqrt();
                }
                let max_head = head_importance
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                if max_head > 0.0 {
                    for v in &mut head_importance {
                        *v /= max_head;
                    }
                }

                store.head_predictors[block_idx].train_step(
                    &mean_input,
                    &head_importance,
                    lr,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrate_runs() {
        let mut store = VisionPredictorStore::new(2, 8, 16, 2, 4, 0.5, 0.5, 42);

        let block_acts: Vec<BlockActivations> = (0..2)
            .map(|_| BlockActivations {
                input: vec![0.1f32; 4 * 8],
                mlp_intermediate: vec![0.5f32; 4 * 16],
                attn_head_outputs: vec![0.3f32; 2 * 4 * 4],
                dim: 8,
                mlp_hidden: 16,
                num_heads: 2,
                seq_len: 4,
            })
            .collect();

        calibrate_predictors(&mut store, &[block_acts], 0.01, 3);
    }
}
