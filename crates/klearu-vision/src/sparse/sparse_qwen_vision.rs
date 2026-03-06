/// Sparse forward wrapper for the Qwen Vision Encoder.
///
/// Applies SLIDE-based sparsity to the transformer blocks:
/// - Sparse attention via selective head computation
/// - Sparse MLP via `forward_sparse_mlp`
///
/// Patch embedding, position embedding, and PatchMerger remain dense
/// (fixed-structure operations that are not worth sparsifying).

use crate::model::qwen_vision::QwenVisionEncoder;
use crate::sparse::predictor_store::VisionPredictorStore;
use crate::sparse::sparse_attention::select_active_heads;
use crate::sparse::sparse_mlp::forward_sparse_mlp;

/// Qwen Vision Encoder with SLIDE-based sparsity for MLPs and attention heads.
pub struct SparseQwenVisionEncoder {
    pub model: QwenVisionEncoder,
    pub predictors: VisionPredictorStore,
}

impl SparseQwenVisionEncoder {
    pub fn new(model: QwenVisionEncoder, predictors: VisionPredictorStore) -> Self {
        Self { model, predictors }
    }

    /// Forward pass with sparse transformer blocks.
    ///
    /// Input: `[in_channels, H, W]` (channel-first, H and W must be multiples of
    /// `patch_size * spatial_merge_size`).
    ///
    /// Returns: `[num_merged_tokens, out_hidden_size]` (flattened token embeddings).
    pub fn forward(&self, image: &[f32], h: usize, w: usize) -> Vec<f32> {
        let hidden = self.model.config.hidden_size;
        let tp = self.model.config.temporal_patch_size;

        // For single images, duplicate frame to fill temporal dimension
        // [C, H, W] -> [C*temporal, H, W]
        let temporal_input = if tp == 2 {
            let spatial = h * w;
            let c = self.model.config.in_channels;
            let mut dup = vec![0.0f32; c * tp * spatial];
            for ch in 0..c {
                for frame in 0..tp {
                    dup[(ch * tp + frame) * spatial..(ch * tp + frame + 1) * spatial]
                        .copy_from_slice(&image[ch * spatial..(ch + 1) * spatial]);
                }
            }
            dup
        } else {
            image.to_vec()
        };

        // Conv2d patch embedding
        let (grid_h, grid_w) = self.model.patch_embed.output_dims(h, w);
        let num_patches = grid_h * grid_w;
        let conv_channels = self.model.patch_embed.out_channels;
        let mut conv_out = vec![0.0f32; conv_channels * num_patches];
        self.model.patch_embed.forward(&temporal_input, h, w, &mut conv_out);

        // Transpose from [C, grid_h, grid_w] to [num_patches, hidden]
        let mut tokens = vec![0.0f32; num_patches * hidden];
        for c in 0..hidden {
            for p in 0..num_patches {
                tokens[p * hidden + c] = conv_out[c * num_patches + p];
            }
        }

        // Add position embeddings
        let pos_ids = compute_position_ids(
            grid_h,
            grid_w,
            self.model.config.num_position_embeddings,
        );
        for (p, &pos_id) in pos_ids.iter().enumerate() {
            let pos_offset = pos_id * hidden;
            for d in 0..hidden {
                tokens[p * hidden + d] += self.model.pos_embed[pos_offset + d];
            }
        }

        // Transformer blocks with sparsity
        let seq_len = num_patches;
        let mlp_hidden = self.model.blocks[0].mlp_fc1.out_features();

        for (block_idx, block) in self.model.blocks.iter().enumerate() {
            // Compute mean token for predictor input
            let mut mean_token = vec![0.0f32; hidden];
            for t in 0..seq_len {
                for d in 0..hidden {
                    mean_token[d] += tokens[t * hidden + d];
                }
            }
            let inv_seq = 1.0 / seq_len as f32;
            for v in &mut mean_token {
                *v *= inv_seq;
            }

            // --- Sparse attention ---
            let mut normed = tokens.clone();
            for t in 0..seq_len {
                block.norm1.forward(&mut normed[t * hidden..(t + 1) * hidden]);
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

            // --- Sparse MLP ---
            let active_neurons =
                self.predictors.select_neurons(block_idx, &mean_token, mlp_hidden);

            for t in 0..seq_len {
                let token = &mut tokens[t * hidden..(t + 1) * hidden];
                let mut normed_token = token.to_vec();
                block.norm2.forward(&mut normed_token);

                let mlp_out = forward_sparse_mlp(
                    &block.mlp_fc1,
                    &block.mlp_fc2,
                    &normed_token,
                    &active_neurons,
                );
                for d in 0..hidden {
                    token[d] += mlp_out[d];
                }
            }
        }

        // PatchMerger (dense -- fixed grouping + projection, not worth sparsifying)
        self.model.merger.forward(&tokens, grid_h, grid_w, hidden)
    }
}

/// Sparse attention: only compute Q/K/V and attention for active heads.
///
/// Same pattern as `sparse_vit::forward_sparse_attention` but extracted here
/// to take `ViTSelfAttention` from the qwen blocks.
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

/// Compute position IDs for a grid of patches.
///
/// For a `grid_h x grid_w` patch grid, maps each position to an index into the
/// learned position embedding table. Uses nearest-neighbor interpolation when
/// the grid doesn't match the base grid size (sqrt(num_position_embeddings)).
fn compute_position_ids(
    grid_h: usize,
    grid_w: usize,
    num_position_embeddings: usize,
) -> Vec<usize> {
    let base_size = (num_position_embeddings as f64).sqrt() as usize;
    let num_patches = grid_h * grid_w;
    let mut ids = Vec::with_capacity(num_patches);

    if grid_h <= base_size && grid_w <= base_size {
        // Direct indexing -- center the grid within the base grid
        let offset_y = (base_size - grid_h) / 2;
        let offset_x = (base_size - grid_w) / 2;
        for y in 0..grid_h {
            for x in 0..grid_w {
                let id = (offset_y + y) * base_size + (offset_x + x);
                ids.push(id.min(num_position_embeddings - 1));
            }
        }
    } else {
        // Nearest-neighbor interpolation for oversized grids
        for y in 0..grid_h {
            for x in 0..grid_w {
                let src_y = (y * base_size / grid_h).min(base_size - 1);
                let src_x = (x * base_size / grid_w).min(base_size - 1);
                ids.push(src_y * base_size + src_x);
            }
        }
    }

    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QwenVisionConfig;

    fn tiny_config() -> QwenVisionConfig {
        QwenVisionConfig {
            hidden_size: 16,
            in_channels: 3,
            patch_size: 4,
            temporal_patch_size: 2,
            depth: 1,
            num_heads: 2,
            intermediate_size: 32,
            spatial_merge_size: 2,
            out_hidden_size: 32,
            num_position_embeddings: 64,
            layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn test_sparse_qwen_vision_runs() {
        let config = tiny_config();
        let model = QwenVisionEncoder::new(config);

        // hidden=16, mlp_hidden=32, num_heads=2, head_dim=8
        let predictors = VisionPredictorStore::new(
            1,  // num_blocks (depth=1)
            16, // input_dim (hidden_size)
            32, // mlp_hidden_dim (intermediate_size)
            2,  // num_heads
            8,  // predictor_hidden
            0.5, // neuron_sparsity
            0.5, // head_sparsity
            42,  // seed
        );
        let sparse_model = SparseQwenVisionEncoder::new(model, predictors);

        // h=w=16, patch_size=4 -> 4x4 grid, merge_size=2 -> 2x2 = 4 merged tokens
        let input = vec![0.1f32; 3 * 16 * 16];
        let output = sparse_model.forward(&input, 16, 16);

        // 4 merged tokens x 32 out_hidden_size
        assert_eq!(output.len(), 4 * 32);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_sparse_qwen_vision_full_sparsity_matches_dense() {
        let config = tiny_config();

        let model = QwenVisionEncoder::new(config.clone());
        let input = vec![0.1f32; 3 * 16 * 16];
        let dense_out = model.forward(&input, 16, 16);

        let model2 = QwenVisionEncoder::new(config);
        let predictors = VisionPredictorStore::new(
            1,   // num_blocks
            16,  // input_dim
            32,  // mlp_hidden_dim
            2,   // num_heads
            8,   // predictor_hidden
            1.0, // neuron_sparsity = keep all
            1.0, // head_sparsity = keep all
            42,  // seed
        );
        let sparse_model = SparseQwenVisionEncoder::new(model2, predictors);
        let sparse_out = sparse_model.forward(&input, 16, 16);

        assert_eq!(dense_out.len(), sparse_out.len());
        // With sparsity=1.0 (keep all), results should match dense forward
        for i in 0..dense_out.len() {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-3,
                "output[{i}]: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i],
            );
        }
    }

    #[test]
    fn test_sparse_qwen_vision_different_sparsity_levels() {
        let config = tiny_config();

        // Sparse at 50%
        let model = QwenVisionEncoder::new(config.clone());
        let predictors_half = VisionPredictorStore::new(1, 16, 32, 2, 8, 0.5, 0.5, 42);
        let sparse_half = SparseQwenVisionEncoder::new(model, predictors_half);

        // Sparse at 100%
        let model2 = QwenVisionEncoder::new(config);
        let predictors_full = VisionPredictorStore::new(1, 16, 32, 2, 8, 1.0, 1.0, 42);
        let sparse_full = SparseQwenVisionEncoder::new(model2, predictors_full);

        let input = vec![0.1f32; 3 * 16 * 16];
        let out_half = sparse_half.forward(&input, 16, 16);
        let out_full = sparse_full.forward(&input, 16, 16);

        assert_eq!(out_half.len(), out_full.len());
        for &v in &out_half {
            assert!(v.is_finite());
        }
        for &v in &out_full {
            assert!(v.is_finite());
        }
    }
}
