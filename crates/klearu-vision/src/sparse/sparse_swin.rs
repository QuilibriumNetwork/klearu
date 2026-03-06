use crate::model::swin::SwinModel;
use crate::sparse::predictor_store::VisionPredictorStore;
use crate::sparse::sparse_mlp::forward_sparse_mlp;

/// Swin Transformer model with SLIDE-based MLP sparsity.
///
/// Attention is kept fully dense (window attention with shifted windows and
/// relative position bias is too tightly coupled to partition geometry for
/// per-head pruning to be practical). Only the MLP inside each SwinBlock is
/// sparsified: a per-block `SparsityPredictor` selects the top-k fc1 neurons,
/// and `forward_sparse_mlp` does the rest.
///
/// Block indices are flattened across stages: block 0 is `stages[0].blocks[0]`,
/// etc.
pub struct SparseSwinModel {
    pub model: SwinModel,
    pub predictors: VisionPredictorStore,
}

impl SparseSwinModel {
    pub fn new(model: SwinModel, predictors: VisionPredictorStore) -> Self {
        Self { model, predictors }
    }

    /// Forward pass with sparse MLPs and dense window attention.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        let image_size = self.model.config.image_size;
        let dim0 = self.model.config.embed_dims[0];

        // ── Patch embedding ──────────────────────────────────────────
        let (grid_h, grid_w) = self.model.patch_embed.output_dims(image_size, image_size);
        let num_patches = grid_h * grid_w;
        let mut conv_out = vec![0.0f32; dim0 * num_patches];
        self.model
            .patch_embed
            .forward(image, image_size, image_size, &mut conv_out);

        // Transpose [C, H, W] → [H*W, C]
        let mut tokens = vec![0.0f32; num_patches * dim0];
        for c in 0..dim0 {
            for p in 0..num_patches {
                tokens[p * dim0 + c] = conv_out[c * num_patches + p];
            }
        }

        // Patch norm
        for t in 0..num_patches {
            self.model
                .patch_norm
                .forward(&mut tokens[t * dim0..(t + 1) * dim0]);
        }

        // ── Stages ───────────────────────────────────────────────────
        let mut h = grid_h;
        let mut w = grid_w;
        let mut dim = dim0;
        let mut block_idx: usize = 0;

        for s in 0..4 {
            // Downsample (if present)
            if let Some(ref ds) = self.model.stages[s].downsample {
                tokens = ds.forward(&tokens, h, w, dim);
                h /= 2;
                w /= 2;
                dim *= 2;
            }

            let mlp_hidden = self.model.config.mlp_hidden_dim(s);

            for block in &self.model.stages[s].blocks {
                let n = h * w;

                // ── Attention (dense — window attention kept intact) ─
                let mut normed = tokens.clone();
                for t in 0..n {
                    block.norm1.forward(&mut normed[t * dim..(t + 1) * dim]);
                }
                let attn_out = if block.shift_size > 0 {
                    block.attn.forward_shifted(&normed, h, w, block.shift_size)
                } else {
                    block.attn.forward(&normed, h, w)
                };
                for (x, a) in tokens.iter_mut().zip(attn_out.iter()) {
                    *x += a;
                }

                // ── MLP (sparse) ─────────────────────────────────────
                // Compute a mean token as predictor input.
                let mut mean_token = vec![0.0f32; dim];
                for t in 0..n {
                    for d in 0..dim {
                        mean_token[d] += tokens[t * dim + d];
                    }
                }
                let inv_n = 1.0 / n as f32;
                for v in &mut mean_token {
                    *v *= inv_n;
                }

                let active_neurons =
                    self.predictors
                        .select_neurons(block_idx, &mean_token, mlp_hidden);

                for t in 0..n {
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

                block_idx += 1;
            }
        }

        // ── Global average pool + final norm + head ──────────────────
        let n = h * w;
        let mut pooled = vec![0.0f32; dim];
        for t in 0..n {
            for d in 0..dim {
                pooled[d] += tokens[t * dim + d];
            }
        }
        let inv = 1.0 / n as f32;
        for v in pooled.iter_mut() {
            *v *= inv;
        }

        self.model.final_norm.forward(&mut pooled);

        let mut logits = vec![0.0f32; self.model.config.num_classes];
        self.model.head.forward(&pooled, &mut logits);
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SwinConfig;
    use klearu_dejavu::predictor::SparsityPredictor;

    /// Build a `VisionPredictorStore` whose per-block predictors match the
    /// varying embed dims / MLP hidden dims of a Swin config.
    fn make_swin_predictors(config: &SwinConfig, neuron_sparsity: f32) -> VisionPredictorStore {
        let mut mlp_predictors = Vec::new();
        let mut head_predictors = Vec::new();
        let predictor_hidden = 8;
        let mut seed: u64 = 42;

        for s in 0..4 {
            let dim = config.embed_dims[s];
            let mlp_hidden = config.mlp_hidden_dim(s);
            let num_heads = config.num_heads[s];
            for _ in 0..config.depths[s] {
                mlp_predictors.push(SparsityPredictor::new(
                    dim,
                    predictor_hidden,
                    mlp_hidden,
                    seed,
                ));
                seed += 1;
                head_predictors.push(SparsityPredictor::new(
                    dim,
                    predictor_hidden,
                    num_heads,
                    seed,
                ));
                seed += 1;
            }
        }

        VisionPredictorStore {
            mlp_predictors,
            head_predictors,
            neuron_sparsity,
            head_sparsity: 1.0, // ignored for Swin (attention is always dense)
        }
    }

    fn tiny_test_config() -> SwinConfig {
        SwinConfig {
            image_size: 56,
            in_channels: 3,
            patch_size: 4,
            embed_dims: [8, 16, 32, 64],
            num_heads: [2, 2, 4, 8],
            depths: [1, 1, 1, 1],
            window_size: 7,
            mlp_ratio: 2.0,
            num_classes: 5,
            layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn test_sparse_swin_runs() {
        let config = tiny_test_config();
        let model = SwinModel::new(config.clone());
        let predictors = make_swin_predictors(&config, 0.5);
        let sparse_model = SparseSwinModel::new(model, predictors);

        let input = vec![0.1f32; 3 * 56 * 56];
        let logits = sparse_model.forward(&input);
        assert_eq!(logits.len(), 5);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sparse_swin_full_sparsity_matches_dense() {
        let config = tiny_test_config();

        let model_dense = SwinModel::new(config.clone());
        let dense_out = model_dense.forward(&vec![0.1f32; 3 * 56 * 56]);

        let model_sparse = SwinModel::new(config.clone());
        let predictors = make_swin_predictors(&config, 1.0);
        let sparse_model = SparseSwinModel::new(model_sparse, predictors);
        let sparse_out = sparse_model.forward(&vec![0.1f32; 3 * 56 * 56]);

        assert_eq!(dense_out.len(), sparse_out.len());
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
