use crate::config::HieraConfig;
use crate::model::hiera::HieraModel;
use crate::sparse::predictor_store::VisionPredictorStore;
use crate::sparse::sparse_attention::select_active_heads;
use crate::sparse::sparse_mlp::forward_sparse_mlp;

/// Hiera model with SLIDE-based sparsity for MLPs and attention heads.
///
/// Block indices are flattened across stages: stage 0 block 0 = index 0,
/// stage 0 block 1 = index 1, ..., stage 1 block 0 = sum(depths[0..1]), etc.
pub struct SparseHieraModel {
    pub model: HieraModel,
    pub predictors: VisionPredictorStore,
}

impl SparseHieraModel {
    pub fn new(model: HieraModel, predictors: VisionPredictorStore) -> Self {
        Self { model, predictors }
    }

    /// Build a `VisionPredictorStore` for Hiera with per-stage dimensions.
    ///
    /// Since Hiera has varying embed_dims and num_heads across stages, we
    /// construct each block's predictor with the correct input/output dims
    /// rather than using `VisionPredictorStore::new` (which assumes uniform dims).
    pub fn build_predictors(
        config: &HieraConfig,
        predictor_hidden: usize,
        neuron_sparsity: f32,
        head_sparsity: f32,
        seed: u64,
    ) -> VisionPredictorStore {
        use klearu_dejavu::predictor::SparsityPredictor;

        let total_blocks: usize = config.depths.iter().sum();
        let mut mlp_predictors = Vec::with_capacity(total_blocks);
        let mut head_predictors = Vec::with_capacity(total_blocks);

        let mut flat_idx: u64 = 0;
        for s in 0..4 {
            let dim = config.embed_dims[s];
            let mlp_hidden = config.mlp_hidden_dim(s);
            let num_heads = config.num_heads[s];

            for _ in 0..config.depths[s] {
                mlp_predictors.push(SparsityPredictor::new(
                    dim,
                    predictor_hidden,
                    mlp_hidden,
                    seed + flat_idx,
                ));
                head_predictors.push(SparsityPredictor::new(
                    dim,
                    predictor_hidden,
                    num_heads,
                    seed + total_blocks as u64 + flat_idx,
                ));
                flat_idx += 1;
            }
        }

        VisionPredictorStore {
            mlp_predictors,
            head_predictors,
            neuron_sparsity,
            head_sparsity,
        }
    }

    /// Forward pass with sparse MLP and sparse attention heads.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        let features = self.forward_features(image);
        let mut logits = vec![0.0f32; self.model.config.num_classes];
        self.model.head.forward(&features, &mut logits);
        logits
    }

    /// Forward features with sparse blocks, returning pooled features before the head.
    pub fn forward_features(&self, image: &[f32]) -> Vec<f32> {
        let image_size = self.model.config.image_size;
        let dim0 = self.model.config.embed_dims[0];

        // Patch embedding (dense)
        let (grid_h, grid_w) = self.model.patch_embed.output_dims(image_size, image_size);
        let num_patches = grid_h * grid_w;
        let mut conv_out = vec![0.0f32; dim0 * num_patches];
        self.model.patch_embed.forward(image, image_size, image_size, &mut conv_out);

        // Transpose [C, H, W] -> [H*W, C]
        let mut tokens = vec![0.0f32; num_patches * dim0];
        for c in 0..dim0 {
            for p in 0..num_patches {
                tokens[p * dim0 + c] = conv_out[c * num_patches + p];
            }
        }

        let mut h = grid_h;
        let mut w = grid_w;
        let mut dim = dim0;

        let mut flat_block_idx = 0usize;

        for s in 0..4 {
            // Token merge (dense, no sparsity)
            if let Some(ref merge) = self.model.merges[s] {
                tokens = merge.forward(&tokens, h, w, dim);
                h /= 2;
                w /= 2;
                dim = self.model.config.embed_dims[s];
            }

            let seq_len = h * w;
            let mlp_hidden = self.model.config.mlp_hidden_dim(s);
            let num_heads = self.model.config.num_heads[s];

            for block in &self.model.stages[s] {
                // Compute mean token for predictor input
                let mut mean_token = vec![0.0f32; dim];
                for t in 0..seq_len {
                    for d in 0..dim {
                        mean_token[d] += tokens[t * dim + d];
                    }
                }
                let inv_seq = 1.0 / seq_len as f32;
                for v in &mut mean_token {
                    *v *= inv_seq;
                }

                // --- Sparse attention ---
                let mut normed = tokens.clone();
                for t in 0..seq_len {
                    block.norm1.forward(&mut normed[t * dim..(t + 1) * dim]);
                }

                let head_scores = self.predictors.head_predictors[flat_block_idx].predict(&mean_token);
                let active_heads = select_active_heads(
                    &head_scores,
                    num_heads,
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

                // --- Sparse MLP ---
                let active_neurons = self.predictors.select_neurons(
                    flat_block_idx,
                    &mean_token,
                    mlp_hidden,
                );

                for t in 0..seq_len {
                    let token = &mut tokens[t * dim..(t + 1) * dim];
                    let mut normed_token = token.to_vec();
                    block.norm2.forward(&mut normed_token);

                    let mlp_out = forward_sparse_mlp(
                        &block.mlp_fc1,
                        &block.mlp_fc2,
                        &normed_token,
                        &active_neurons,
                    );
                    for d in 0..dim {
                        token[d] += mlp_out[d];
                    }
                }

                flat_block_idx += 1;
            }
        }

        // Global average pool
        let seq_len = h * w;
        let mut pooled = vec![0.0f32; dim];
        for t in 0..seq_len {
            for d in 0..dim {
                pooled[d] += tokens[t * dim + d];
            }
        }
        let inv = 1.0 / seq_len as f32;
        for v in pooled.iter_mut() {
            *v *= inv;
        }

        self.model.final_norm.forward(&mut pooled);
        pooled
    }
}

/// Sparse attention: only compute Q/K/V and attention for active heads.
///
/// Same implementation as in sparse_vit.rs, reused here for HieraBlock.attn
/// which is also a `ViTSelfAttention`.
fn forward_sparse_attention(
    attn: &crate::model::vit::ViTSelfAttention,
    input: &[f32],
    seq_len: usize,
    active_heads: &[usize],
) -> Vec<f32> {
    let dim = attn.num_heads * attn.head_dim;
    let head_dim = attn.head_dim;

    // Full QKV projection (dense -- projections are cheap)
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

    fn tiny_test_config() -> HieraConfig {
        HieraConfig {
            image_size: 32,
            in_channels: 3,
            patch_size: 4,
            embed_dims: [8, 16, 32, 64],
            num_heads: [1, 2, 4, 8],
            depths: [1, 1, 1, 1],
            mask_unit_size: 8,
            mlp_ratio: 2.0,
            num_classes: 5,
        }
    }

    #[test]
    fn test_sparse_hiera_runs() {
        let config = tiny_test_config();
        let predictors = SparseHieraModel::build_predictors(&config, 4, 0.5, 0.5, 42);
        let model = HieraModel::new(config);
        let sparse_model = SparseHieraModel::new(model, predictors);

        let input = vec![0.1f32; 3 * 32 * 32];
        let logits = sparse_model.forward(&input);
        assert_eq!(logits.len(), 5);
        for &v in &logits {
            assert!(v.is_finite(), "logit is not finite: {}", v);
        }
    }

    #[test]
    fn test_sparse_hiera_full_sparsity_matches_dense() {
        let config = tiny_test_config();

        // Dense forward
        let model = HieraModel::new(config.clone());
        let input = vec![0.1f32; 3 * 32 * 32];
        let dense_out = model.forward(&input);

        // Sparse forward with sparsity=1.0 (keep all)
        let predictors = SparseHieraModel::build_predictors(&config, 4, 1.0, 1.0, 42);
        let model2 = HieraModel::new(config);
        let sparse_model = SparseHieraModel::new(model2, predictors);
        let sparse_out = sparse_model.forward(&input);

        assert_eq!(dense_out.len(), sparse_out.len());
        for i in 0..dense_out.len() {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-3,
                "logit[{i}]: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i],
            );
        }
    }

    #[test]
    fn test_sparse_hiera_predictor_counts() {
        let config = tiny_test_config();
        let total_blocks: usize = config.depths.iter().sum(); // 1+1+1+1 = 4
        let predictors = SparseHieraModel::build_predictors(&config, 4, 0.5, 0.5, 42);

        assert_eq!(predictors.mlp_predictors.len(), total_blocks);
        assert_eq!(predictors.head_predictors.len(), total_blocks);
    }
}
