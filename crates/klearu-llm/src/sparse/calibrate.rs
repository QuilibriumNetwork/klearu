use klearu_dejavu::predictor::SparsityPredictor;

use crate::config::LlmConfig;
use crate::model::Model;
use crate::model::block::AttentionLayer;
use crate::sparse::predictor_store::PredictorStore;

/// Head importance scores collected during calibration.
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// `[num_layers][num_heads]` -- L2 norm of each head's attention output.
    pub head_importance: Vec<Vec<f32>>,
    /// `[num_layers][intermediate_size]` -- mean absolute value of each neuron's contribution.
    pub neuron_importance: Vec<Vec<f32>>,
    pub num_samples: usize,
}

impl CalibrationData {
    pub fn new(config: &LlmConfig) -> Self {
        Self {
            head_importance: vec![vec![0.0; config.num_heads]; config.num_layers],
            neuron_importance: vec![vec![0.0; config.intermediate_size]; config.num_layers],
            num_samples: 0,
        }
    }

    /// Accumulate importance from a single calibration forward pass.
    ///
    /// This is a simplified calibration that runs dense inference and records
    /// the norms of each head's/neuron's contribution to the output.
    pub fn calibrate_sequence(&mut self, model: &mut Model, token_ids: &[u32]) {
        let config = model.config.clone();
        let hidden_size = config.hidden_size;

        model.reset_kv_caches();

        for (pos, &token_id) in token_ids.iter().enumerate() {
            let mut hidden = vec![0.0f32; hidden_size];
            model.embedding.forward(token_id, &mut hidden);

            let mut norm_buf = vec![0.0f32; hidden_size];

            let mut attn_out = vec![0.0f32; hidden_size];

            for layer_idx in 0..config.num_layers {
                norm_buf.copy_from_slice(&hidden);
                model.layers[layer_idx].attn_norm.forward(&mut norm_buf);

                // Full attention (dispatch based on layer type)
                match &model.layers[layer_idx].attention {
                    AttentionLayer::Standard(attn) => {
                        attn.forward_into(
                            &norm_buf,
                            pos,
                            &model.rope,
                            model.kv_caches.layer_mut(layer_idx),
                            &mut attn_out,
                        );
                    }
                    AttentionLayer::GatedDeltaNet(dn) => {
                        let state = model.deltanet_states.layer_mut(layer_idx)
                            .expect("DeltaNet state missing for linear attention layer");
                        dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                    }
                };

                // Approximate head importance: L2 norm of the full attention output
                // divided equally across heads (simplified).
                let attn_l2: f32 = attn_out.iter().map(|x| x * x).sum::<f32>().sqrt();
                let per_head = attn_l2 / config.num_heads as f32;
                for h in 0..config.num_heads {
                    self.head_importance[layer_idx][h] += per_head;
                }

                for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                    *h += a;
                }

                norm_buf.copy_from_slice(&hidden);
                model.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

                let mlp_out = model.layers[layer_idx].mlp.forward(&norm_buf);

                // Approximate neuron importance: use the output's L2 contribution.
                let mlp_l2: f32 = mlp_out.iter().map(|x| x * x).sum::<f32>().sqrt();
                let per_neuron = mlp_l2 / config.intermediate_size as f32;
                for n in 0..config.intermediate_size {
                    self.neuron_importance[layer_idx][n] += per_neuron;
                }

                for (h, m) in hidden.iter_mut().zip(mlp_out.iter()) {
                    *h += m;
                }
            }

            self.num_samples += 1;
        }
    }

    /// Normalize accumulated importance by number of samples.
    pub fn finalize(&mut self) {
        if self.num_samples == 0 {
            return;
        }
        let inv = 1.0 / self.num_samples as f32;
        for layer in &mut self.head_importance {
            for v in layer.iter_mut() {
                *v *= inv;
            }
        }
        for layer in &mut self.neuron_importance {
            for v in layer.iter_mut() {
                *v *= inv;
            }
        }
    }

    /// Train sparsity predictors from the calibration importance scores.
    ///
    /// Each layer gets a head predictor (output_dim = num_heads) and a neuron
    /// predictor (output_dim = intermediate_size). The predictors are small MLPs
    /// trained to predict the normalized importance scores from a synthetic input.
    ///
    /// Call `finalize()` before this method.
    pub fn train_predictors(
        &self,
        config: &LlmConfig,
        hidden_dim: usize,
        lr: f32,
        epochs: usize,
    ) -> PredictorStore {
        let input_dim = config.hidden_size;
        let mut store = PredictorStore::new(config.num_layers);

        for layer_idx in 0..config.num_layers {
            // Normalize head importance to [0, 1] range for training targets
            let head_targets = normalize_scores(&self.head_importance[layer_idx]);
            let neuron_targets = normalize_scores(&self.neuron_importance[layer_idx]);

            // Use a synthetic input derived from the importance scores themselves.
            // In practice, calibration would store actual hidden states; here we
            // use a simple uniform input so the predictor learns the bias pattern.
            let input = vec![1.0f32 / input_dim as f32; input_dim];

            // Train head predictor
            let mut head_pred = SparsityPredictor::new(
                input_dim,
                hidden_dim,
                config.num_heads,
                (layer_idx as u64) * 2,
            );
            for _ in 0..epochs {
                head_pred.train_step(&input, &head_targets, lr);
            }
            store.set_head_predictor(layer_idx, head_pred);

            // Train neuron predictor
            let mut neuron_pred = SparsityPredictor::new(
                input_dim,
                hidden_dim,
                config.intermediate_size,
                (layer_idx as u64) * 2 + 1,
            );
            for _ in 0..epochs {
                neuron_pred.train_step(&input, &neuron_targets, lr);
            }
            store.set_neuron_predictor(layer_idx, neuron_pred);
        }

        store
    }
}

/// Normalize scores to [0, 1] by dividing by max (or return zeros if max is 0).
fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return vec![0.0; scores.len()];
    }
    scores.iter().map(|&s| s / max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_data_creation() {
        let config = LlmConfig {
            num_heads: 4,
            num_layers: 2,
            intermediate_size: 8,
            ..LlmConfig::default()
        };
        let data = CalibrationData::new(&config);
        assert_eq!(data.head_importance.len(), 2);
        assert_eq!(data.head_importance[0].len(), 4);
        assert_eq!(data.neuron_importance[0].len(), 8);
    }

    #[test]
    fn test_train_predictors() {
        let config = LlmConfig {
            num_heads: 4,
            num_layers: 2,
            intermediate_size: 8,
            hidden_size: 16,
            num_kv_heads: 2,
            head_dim: 4,
            ..LlmConfig::default()
        };
        let mut data = CalibrationData::new(&config);
        // Simulate some importance data
        data.head_importance[0] = vec![10.0, 5.0, 2.0, 1.0];
        data.head_importance[1] = vec![8.0, 6.0, 4.0, 2.0];
        data.neuron_importance[0] = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        data.neuron_importance[1] = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        data.num_samples = 10;
        data.finalize();

        let store = data.train_predictors(&config, 8, 0.01, 50);
        assert_eq!(store.num_layers(), 2);

        // Predictor should return valid indices
        let input = vec![0.1f32; 16];
        let heads = store.predict_heads(0, &input, 2);
        assert_eq!(heads.len(), 2);
        assert!(heads.iter().all(|&h| h < 4));
    }

    #[test]
    fn test_calibration_finalize() {
        let config = LlmConfig {
            num_heads: 2,
            num_layers: 1,
            intermediate_size: 4,
            ..LlmConfig::default()
        };
        let mut data = CalibrationData::new(&config);
        data.head_importance[0] = vec![10.0, 20.0];
        data.neuron_importance[0] = vec![1.0, 2.0, 3.0, 4.0];
        data.num_samples = 5;

        data.finalize();

        assert!((data.head_importance[0][0] - 2.0).abs() < 1e-5);
        assert!((data.head_importance[0][1] - 4.0).abs() < 1e-5);
        assert!((data.neuron_importance[0][2] - 0.6).abs() < 1e-5);
    }
}
