pub mod calibrate;
pub mod predictor_store;
pub mod sparse_attention;
pub mod sparse_mlp;

use crate::model::Model;
use crate::model::block::AttentionLayer;

use predictor_store::PredictorStore;

/// Model wrapper that applies Deja Vu sparsity during inference.
pub struct SparseModel {
    pub model: Model,
    pub predictor_store: PredictorStore,
    pub head_sparsity: f32,   // fraction of heads to keep (e.g. 0.5)
    pub neuron_sparsity: f32, // fraction of neurons to keep (e.g. 0.5)
}

impl SparseModel {
    pub fn new(model: Model, predictor_store: PredictorStore, head_sparsity: f32, neuron_sparsity: f32) -> Self {
        Self {
            model,
            predictor_store,
            head_sparsity,
            neuron_sparsity,
        }
    }

    /// Sparse decode: predict important heads/neurons, then only compute those.
    ///
    /// When predictors are not calibrated, delegates to the dense
    /// `Model::forward_decode` to avoid numerical divergence from the
    /// manual layer loop reimplementation.
    pub fn forward_decode_sparse(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        // If no predictors are calibrated, delegate to the dense path directly.
        // This avoids numerical divergence between the sparse layer loop and
        // the dense Model::forward_decode implementation.
        if !self.predictor_store.is_calibrated() {
            return self.model.forward_decode(token_id, position);
        }

        let config = &self.model.config;
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;

        let mut hidden = vec![0.0f32; hidden_size];
        self.model.embedding.forward(token_id, &mut hidden);

        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; config.intermediate_size];
        let mut up_buf = vec![0.0f32; config.intermediate_size];

        for layer_idx in 0..config.num_layers {
            // Pre-attention norm
            norm_buf.copy_from_slice(&hidden);
            self.model.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            // Dispatch attention based on layer type
            match &self.model.layers[layer_idx].attention {
                AttentionLayer::Standard(attn) => {
                    // Skip sparse for gated attention (Qwen3.5 full attention with
                    // output gate, Q/K norms, doubled q_proj) — forward_sparse doesn't
                    // handle these features.
                    if attn.output_gate {
                        attn.forward_into(
                            &norm_buf,
                            position,
                            &self.model.rope,
                            self.model.kv_caches.layer_mut(layer_idx),
                            &mut attn_out,
                        );
                    } else {
                        // Predict important heads
                        let num_heads_to_keep =
                            ((config.num_heads as f32) * self.head_sparsity).ceil() as usize;
                        let head_indices = self
                            .predictor_store
                            .predict_heads(layer_idx, &norm_buf, num_heads_to_keep);

                        if head_indices.len() < config.num_heads {
                            let sparse_out = sparse_attention::forward_sparse(
                                attn,
                                &norm_buf,
                                position,
                                &self.model.rope,
                                self.model.kv_caches.layer_mut(layer_idx),
                                &head_indices,
                            );
                            attn_out.copy_from_slice(&sparse_out);
                        } else {
                            attn.forward_into(
                                &norm_buf,
                                position,
                                &self.model.rope,
                                self.model.kv_caches.layer_mut(layer_idx),
                                &mut attn_out,
                            );
                        }
                    }
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    // GatedDeltaNet has no sparse implementation — always dense
                    let state = self.model.deltanet_states.layer_mut(layer_idx)
                        .expect("DeltaNet state missing for linear attention layer");
                    dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                }
            };

            for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            // Pre-MLP norm
            norm_buf.copy_from_slice(&hidden);
            self.model.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

            // Predict important neurons
            let num_neurons_to_keep =
                ((config.intermediate_size as f32) * self.neuron_sparsity).ceil() as usize;
            let neuron_indices = self
                .predictor_store
                .predict_neurons(layer_idx, &norm_buf, num_neurons_to_keep);

            // Sparse MLP
            if neuron_indices.len() < config.intermediate_size {
                let sparse_out = sparse_mlp::forward_sparse(
                    &self.model.layers[layer_idx].mlp,
                    &norm_buf,
                    &neuron_indices,
                );
                mlp_out.copy_from_slice(&sparse_out);
            } else {
                self.model.layers[layer_idx].mlp.forward_into(
                    &norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf,
                );
            }

            for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
                *h += m;
            }
        }

        self.model.final_norm.forward(&mut hidden);

        let mut logits = vec![0.0f32; vocab_size];
        match &self.model.lm_head {
            Some(head) => head.forward(&hidden, &mut logits),
            None => self.model.embedding.lm_head_forward(&hidden, &mut logits),
        }

        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlmConfig;

    #[test]
    fn test_sparse_model_creation() {
        let config = LlmConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            num_layers: 2,
            max_seq_len: 16,
            tie_word_embeddings: true,
            ..LlmConfig::default()
        };
        let model = Model::new(config.clone());
        let store = PredictorStore::new(config.num_layers);
        let sparse = SparseModel::new(model, store, 0.5, 0.5);
        assert_eq!(sparse.head_sparsity, 0.5);
    }
}
