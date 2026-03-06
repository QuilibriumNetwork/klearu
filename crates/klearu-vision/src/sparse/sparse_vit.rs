use crate::model::vit::{PoolType, ViTModel};
use crate::sparse::predictor_store::VisionPredictorStore;
use crate::sparse::sparse_attention::select_active_heads;
use crate::sparse::sparse_mlp::forward_sparse_mlp;

/// ViT model with SLIDE-based sparsity for MLPs and attention heads.
pub struct SparseViTModel {
    pub model: ViTModel,
    pub predictors: VisionPredictorStore,
}

impl SparseViTModel {
    pub fn new(model: ViTModel, predictors: VisionPredictorStore) -> Self {
        Self { model, predictors }
    }

    /// Forward pass with sparse MLP and sparse attention heads.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        let embed_dim = self.model.config.embed_dim;
        let image_size = self.model.config.image_size;

        let patch_tokens = self.model.patch_embed.forward(image, image_size, image_size);
        let num_patches = patch_tokens.len() / embed_dim;

        let (mut tokens, seq_len) = if self.model.pool_type == PoolType::Cls {
            let seq_len = 1 + num_patches;
            let mut tokens = vec![0.0f32; seq_len * embed_dim];
            tokens[..embed_dim].copy_from_slice(&self.model.cls_token);
            tokens[embed_dim..].copy_from_slice(&patch_tokens);
            (tokens, seq_len)
        } else {
            (patch_tokens, num_patches)
        };

        // Add position embedding
        for (t, p) in tokens.iter_mut().zip(self.model.pos_embed.iter()) {
            *t += p;
        }

        // Transformer blocks with sparsity
        let mlp_hidden = self.model.blocks[0].mlp_fc1.out_features();

        for (block_idx, block) in self.model.blocks.iter().enumerate() {
            // Pre-norm for predictor input (use mean token)
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

            // Attention (with head sparsity prediction)
            let mut normed = tokens.clone();
            for t in 0..seq_len {
                block.norm1.forward(&mut normed[t * embed_dim..(t + 1) * embed_dim]);
            }

            let head_scores = self.predictors.head_predictors[block_idx].predict(&mean_token);
            let active_heads = select_active_heads(
                &head_scores,
                block.attn.num_heads,
                self.predictors.head_sparsity,
            );
            let attn_out = forward_sparse_attention(
                &block.attn,
                &normed,
                seq_len,
                &active_heads,
            );
            for (x, a) in tokens.iter_mut().zip(attn_out.iter()) {
                *x += a;
            }

            // MLP (with neuron sparsity prediction)
            let active_neurons = self.predictors.select_neurons(block_idx, &mean_token, mlp_hidden);

            for t in 0..seq_len {
                let token = &mut tokens[t * embed_dim..(t + 1) * embed_dim];
                let mut normed_token = token.to_vec();
                block.norm2.forward(&mut normed_token);

                let mlp_out = forward_sparse_mlp(
                    &block.mlp_fc1,
                    &block.mlp_fc2,
                    &normed_token,
                    &active_neurons,
                );
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
        let pooled = match self.model.pool_type {
            PoolType::Cls => tokens[..embed_dim].to_vec(),
            PoolType::Mean => {
                let mut avg = vec![0.0f32; embed_dim];
                for t in 0..num_patches {
                    let offset = if self.model.pool_type == PoolType::Cls {
                        (t + 1) * embed_dim
                    } else {
                        t * embed_dim
                    };
                    for d in 0..embed_dim {
                        avg[d] += tokens[offset + d];
                    }
                }
                let inv = 1.0 / num_patches as f32;
                for v in &mut avg {
                    *v *= inv;
                }
                avg
            }
        };

        let mut logits = vec![0.0f32; self.model.head.out_features()];
        self.model.head.forward(&pooled, &mut logits);
        logits
    }
}

/// Sparse attention: only compute Q/K/V and attention for active heads.
fn forward_sparse_attention(
    attn: &crate::model::vit::ViTSelfAttention,
    input: &[f32],
    seq_len: usize,
    active_heads: &[usize],
) -> Vec<f32> {
    let dim = attn.num_heads * attn.head_dim;
    let head_dim = attn.head_dim;

    // Full QKV projection (dense — projections are cheap)
    let mut qkv = vec![0.0f32; seq_len * 3 * dim];
    for t in 0..seq_len {
        attn.qkv.forward(
            &input[t * dim..(t + 1) * dim],
            &mut qkv[t * 3 * dim..(t + 1) * 3 * dim],
        );
    }

    // Only compute attention for active heads
    let mut output = vec![0.0f32; seq_len * dim];

    for &h in active_heads {
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Extract Q, K, V for this head
        let mut q = vec![0.0f32; seq_len * head_dim];
        let mut k = vec![0.0f32; seq_len * head_dim];
        let mut v = vec![0.0f32; seq_len * head_dim];

        for t in 0..seq_len {
            let qkv_base = t * 3 * dim;
            for d in 0..head_dim {
                q[t * head_dim + d] = qkv[qkv_base + h * head_dim + d];
                k[t * head_dim + d] = qkv[qkv_base + dim + h * head_dim + d];
                v[t * head_dim + d] = qkv[qkv_base + 2 * dim + h * head_dim + d];
            }
        }

        // Attention scores
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }

        // Softmax per row
        for i in 0..seq_len {
            let row = &mut scores[i * seq_len..(i + 1) * seq_len];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }

        // Weighted V
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for j in 0..seq_len {
                    val += scores[i * seq_len + j] * v[j * head_dim + d];
                }
                output[i * dim + h * head_dim + d] = val;
            }
        }
    }

    // Output projection
    let mut proj_out = vec![0.0f32; seq_len * dim];
    for t in 0..seq_len {
        attn.proj.forward(
            &output[t * dim..(t + 1) * dim],
            &mut proj_out[t * dim..(t + 1) * dim],
        );
    }

    proj_out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ViTConfig;

    #[test]
    fn test_sparse_vit_runs() {
        let config = ViTConfig {
            image_size: 32,
            in_channels: 3,
            patch_size: 8,
            embed_dim: 16,
            num_heads: 2,
            num_layers: 2,
            mlp_ratio: 2.0,
            num_classes: 5,
            layer_norm_eps: 1e-5,
            pool_type: PoolType::Cls,
        };

        let model = ViTModel::new(config);
        let predictors = VisionPredictorStore::new(2, 16, 32, 2, 8, 0.5, 0.5, 42);
        let sparse_model = SparseViTModel::new(model, predictors);

        let input = vec![0.1f32; 3 * 32 * 32];
        let logits = sparse_model.forward(&input);
        assert_eq!(logits.len(), 5);
    }

    #[test]
    fn test_sparse_vit_full_sparsity_matches_dense() {
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

        let model = ViTModel::new(config);
        let dense_out = model.forward(&vec![0.1f32; 3 * 32 * 32]);

        let predictors = VisionPredictorStore::new(1, 16, 32, 2, 8, 1.0, 1.0, 42);
        let model2 = ViTModel::new(config);
        let sparse_model = SparseViTModel::new(model2, predictors);
        let sparse_out = sparse_model.forward(&vec![0.1f32; 3 * 32 * 32]);

        assert_eq!(dense_out.len(), sparse_out.len());
        // With sparsity=1.0 (keep all), results should match dense forward
        for i in 0..dense_out.len() {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-3,
                "logit[{i}]: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i]
            );
        }
    }
}
