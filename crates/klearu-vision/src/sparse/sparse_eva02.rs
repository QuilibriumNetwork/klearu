use crate::layers::SwiGluMlp;
use crate::model::eva02::EVA02Model;
use crate::model::vit::PoolType;
use crate::sparse::predictor_store::VisionPredictorStore;

/// EVA-02 model with SLIDE-based sparsity for SwiGLU MLPs.
///
/// Attention remains dense (EVA-02 uses RoPE which complicates head sparsity),
/// while MLP sparsity targets the SwiGLU gate/up projections for the majority
/// of compute savings.
pub struct SparseEVA02Model {
    pub model: EVA02Model,
    pub predictors: VisionPredictorStore,
}

impl SparseEVA02Model {
    pub fn new(model: EVA02Model, predictors: VisionPredictorStore) -> Self {
        Self { model, predictors }
    }

    /// Forward pass with dense attention and sparse SwiGLU MLP.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        let embed_dim = self.model.config.embed_dim;
        let image_size = self.model.config.image_size;

        let patch_tokens = self.model.patch_embed.forward(image, image_size, image_size);
        let num_patches = patch_tokens.len() / embed_dim;

        let has_cls = self.model.config.pool_type == PoolType::Cls;
        let (mut tokens, seq_len) = if has_cls {
            let seq_len = 1 + num_patches;
            let mut tokens = vec![0.0f32; seq_len * embed_dim];
            tokens[..embed_dim].copy_from_slice(&self.model.cls_token);
            tokens[embed_dim..].copy_from_slice(&patch_tokens);
            (tokens, seq_len)
        } else {
            (patch_tokens, num_patches)
        };

        // Add position embedding
        if !self.model.pos_embed.is_empty() {
            for (t, p) in tokens.iter_mut().zip(self.model.pos_embed.iter()) {
                *t += p;
            }
        }

        // Transformer blocks with sparse SwiGLU MLP
        for (block_idx, block) in self.model.blocks.iter().enumerate() {
            let mlp_hidden = block.mlp.gate_proj.out_features();

            // Compute mean token for predictor input
            let mut mean_token = vec![0.0f32; embed_dim];
            for t in 0..seq_len {
                for d in 0..embed_dim {
                    mean_token[d] += tokens[t * embed_dim + d];
                }
            }
            let inv_seq = 1.0 / seq_len as f32;
            for v in &mut mean_token {
                *v *= inv_seq;
            }

            // --- Dense attention with sub-norm ---
            // x = x + sub_norm(attn(norm1(x)))
            let mut normed = tokens.clone();
            for t in 0..seq_len {
                block.norm1.forward(&mut normed[t * embed_dim..(t + 1) * embed_dim]);
            }
            let mut attn_out = block.attn.forward(&normed, seq_len);
            for t in 0..seq_len {
                block.sub_norm.forward(&mut attn_out[t * embed_dim..(t + 1) * embed_dim]);
            }
            for (x, a) in tokens.iter_mut().zip(attn_out.iter()) {
                *x += a;
            }

            // --- Sparse SwiGLU MLP ---
            // x = x + sparse_swiglu(norm2(x))
            let active_neurons = self.predictors.select_neurons(block_idx, &mean_token, mlp_hidden);

            for t in 0..seq_len {
                let token = &mut tokens[t * embed_dim..(t + 1) * embed_dim];
                let mut normed_token = token.to_vec();
                block.norm2.forward(&mut normed_token);

                let mlp_out = forward_sparse_swiglu(&block.mlp, &normed_token, &active_neurons);
                for d in 0..embed_dim {
                    token[d] += mlp_out[d];
                }
            }
        }

        // Final norm
        for t in 0..seq_len {
            self.model.norm.forward(&mut tokens[t * embed_dim..(t + 1) * embed_dim]);
        }

        // Pool
        let pooled = match self.model.config.pool_type {
            PoolType::Cls => tokens[..embed_dim].to_vec(),
            PoolType::Mean => {
                let mut mean = vec![0.0f32; embed_dim];
                for t in 0..seq_len {
                    for d in 0..embed_dim {
                        mean[d] += tokens[t * embed_dim + d];
                    }
                }
                let inv = 1.0 / seq_len as f32;
                for v in &mut mean {
                    *v *= inv;
                }
                mean
            }
        };

        // Classification head
        let mut logits = vec![0.0f32; self.model.head.out_features()];
        self.model.head.forward(&pooled, &mut logits);
        logits
    }
}

/// Sparse SwiGLU MLP forward: compute gate_proj and up_proj only for active neurons,
/// apply SiLU(gate) * up, then scatter-back through down_proj.
fn forward_sparse_swiglu(
    mlp: &SwiGluMlp,
    input: &[f32],
    active_neurons: &[usize],
) -> Vec<f32> {
    let k = active_neurons.len();
    let out_dim = mlp.down_proj.out_features();

    // Sparse gate_proj and up_proj: only compute active hidden neurons
    let mut gate = vec![0.0f32; k];
    let mut up = vec![0.0f32; k];
    mlp.gate_proj.forward_sparse(input, active_neurons, &mut gate);
    mlp.up_proj.forward_sparse(input, active_neurons, &mut up);

    // SiLU(gate) * up
    for (g, u) in gate.iter_mut().zip(up.iter()) {
        *g = *g * (1.0 / (1.0 + (-*g).exp())) * u;
    }

    // Scatter-back via down_proj: for each output dim, gather from active columns
    let mut output = vec![0.0f32; out_dim];
    for out_d in 0..out_dim {
        let w_row = mlp.down_proj.weights.get_weights(out_d);
        let mut sum = mlp.down_proj.bias[out_d];
        for (sparse_idx, &neuron_idx) in active_neurons.iter().enumerate() {
            if neuron_idx < mlp.down_proj.in_features() {
                sum += w_row[neuron_idx] * gate[sparse_idx];
            }
        }
        output[out_d] = sum;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ViTConfig;

    #[test]
    fn test_sparse_eva02_runs() {
        let config = ViTConfig {
            image_size: 32,
            in_channels: 3,
            patch_size: 8,
            embed_dim: 16,
            num_heads: 2,
            num_layers: 1,
            mlp_ratio: 2.0,
            num_classes: 5,
            layer_norm_eps: 1e-5,
            pool_type: PoolType::Cls,
        };

        let model = EVA02Model::new(config);
        let mlp_hidden = model.blocks[0].mlp.gate_proj.out_features();
        let predictors = VisionPredictorStore::new(1, 16, mlp_hidden, 2, 8, 0.5, 0.5, 42);
        let sparse_model = SparseEVA02Model::new(model, predictors);

        let input = vec![0.1f32; 3 * 32 * 32];
        let logits = sparse_model.forward(&input);
        assert_eq!(logits.len(), 5);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sparse_swiglu_all_neurons_matches_dense() {
        let mlp = SwiGluMlp::new(8, 16);
        let input = vec![0.5f32; 8];

        // Dense forward
        let mut dense_out = vec![0.0f32; 8];
        mlp.forward(&input, &mut dense_out);

        // Sparse with all neurons selected
        let all_neurons: Vec<usize> = (0..16).collect();
        let sparse_out = forward_sparse_swiglu(&mlp, &input, &all_neurons);

        assert_eq!(dense_out.len(), sparse_out.len());
        for i in 0..dense_out.len() {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-4,
                "dim[{i}]: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i],
            );
        }
    }

    #[test]
    fn test_sparse_eva02_full_sparsity_matches_dense() {
        let config = ViTConfig {
            image_size: 32,
            in_channels: 3,
            patch_size: 8,
            embed_dim: 16,
            num_heads: 2,
            num_layers: 1,
            mlp_ratio: 2.0,
            num_classes: 5,
            layer_norm_eps: 1e-5,
            pool_type: PoolType::Cls,
        };

        let model = EVA02Model::new(config);
        let dense_out = model.forward(&vec![0.1f32; 3 * 32 * 32]);

        let model2 = EVA02Model::new(config);
        let mlp_hidden = model2.blocks[0].mlp.gate_proj.out_features();
        let predictors = VisionPredictorStore::new(1, 16, mlp_hidden, 2, 8, 1.0, 1.0, 42);
        let sparse_model = SparseEVA02Model::new(model2, predictors);
        let sparse_out = sparse_model.forward(&vec![0.1f32; 3 * 32 * 32]);

        assert_eq!(dense_out.len(), sparse_out.len());
        // With sparsity=1.0 (keep all neurons), sparse MLP should match dense
        for i in 0..dense_out.len() {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-3,
                "logit[{i}]: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i],
            );
        }
    }
}
