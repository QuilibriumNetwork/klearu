/// Qwen3.5 Vision Encoder.
///
/// Architecture: Conv2d patch embed → position embed → N ViT blocks → PatchMerger.
///
/// The patch embedding uses a Conv3d kernel `[temporal_patch_size, patch_size, patch_size]`,
/// treated as Conv2d by merging temporal into input channels: `[C*temporal, H, W]`.
/// For single images, the frame is duplicated to fill the temporal dimension.
///
/// PatchMerger groups `spatial_merge_size × spatial_merge_size` adjacent tokens into one,
/// projecting to the LLM's hidden dimension.

use crate::config::QwenVisionConfig;
use crate::layers::{Conv2d, LayerNorm, LinearBias, gelu_inplace};
use crate::model::vit::ViTBlock;

/// Groups adjacent spatial tokens and projects to LLM hidden dimension.
///
/// Forward: reshape tokens to `[grid_h/merge, grid_w/merge, merge²*hidden]`
/// → LayerNorm → fc1 → GELU → fc2.
pub struct PatchMerger {
    pub norm: LayerNorm,
    pub fc1: LinearBias,
    pub fc2: LinearBias,
    pub merge_size: usize,
}

impl PatchMerger {
    pub fn new(hidden_size: usize, merge_size: usize, out_hidden_size: usize, eps: f32) -> Self {
        let merged_dim = hidden_size * merge_size * merge_size;
        Self {
            norm: LayerNorm::new(hidden_size, eps),
            fc1: LinearBias::new(merged_dim, merged_dim),
            fc2: LinearBias::new(merged_dim, out_hidden_size),
            merge_size,
        }
    }

    /// Forward: `[grid_h * grid_w, hidden_size]` → `[merged_tokens, out_hidden_size]`.
    pub fn forward(
        &self,
        tokens: &[f32],
        grid_h: usize,
        grid_w: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let ms = self.merge_size;
        let merged_h = grid_h / ms;
        let merged_w = grid_w / ms;
        let num_merged = merged_h * merged_w;
        let merged_dim = hidden_size * ms * ms;
        let out_dim = self.fc2.out_features();

        let mut merged = vec![0.0f32; num_merged * merged_dim];

        // Norm each sub-token, then group ms×ms into single merged tokens
        for my in 0..merged_h {
            for mx in 0..merged_w {
                let merged_idx = my * merged_w + mx;
                for sy in 0..ms {
                    for sx in 0..ms {
                        let gy = my * ms + sy;
                        let gx = mx * ms + sx;
                        let token_idx = gy * grid_w + gx;
                        let merge_offset = (sy * ms + sx) * hidden_size;
                        let mut sub_token = tokens[token_idx * hidden_size..(token_idx + 1) * hidden_size].to_vec();
                        self.norm.forward(&mut sub_token);
                        let dst_start = merged_idx * merged_dim + merge_offset;
                        merged[dst_start..dst_start + hidden_size].copy_from_slice(&sub_token);
                    }
                }
            }
        }

        // fc1 → GELU → fc2
        let mut fc1_buf = vec![0.0f32; merged_dim];
        let mut output = vec![0.0f32; num_merged * out_dim];

        for t in 0..num_merged {
            let token = &merged[t * merged_dim..(t + 1) * merged_dim];
            self.fc1.forward(token, &mut fc1_buf);
            gelu_inplace(&mut fc1_buf);
            self.fc2.forward(&fc1_buf, &mut output[t * out_dim..(t + 1) * out_dim]);
        }

        output
    }
}

/// Qwen3.5 Vision Encoder.
///
/// Pipeline: duplicate frame → Conv2d patches → add pos_embed → N ViTBlocks → PatchMerger.
pub struct QwenVisionEncoder {
    pub config: QwenVisionConfig,
    pub patch_embed: Conv2d,
    /// Learned position embeddings: `[num_position_embeddings, hidden_size]`.
    pub pos_embed: Vec<f32>,
    pub blocks: Vec<ViTBlock>,
    pub merger: PatchMerger,
}

impl QwenVisionEncoder {
    pub fn new(config: QwenVisionConfig) -> Self {
        let hidden = config.hidden_size;
        let mlp_hidden = config.intermediate_size;
        let eps = config.layer_norm_eps;

        // Conv2d patch embed: kernel is conceptually Conv3d [temporal, patch, patch]
        // but treated as Conv2d with in_channels = in_channels * temporal_patch_size
        let conv_in_channels = config.in_channels * config.temporal_patch_size;
        let patch_embed = Conv2d::new(
            conv_in_channels,
            hidden,
            config.patch_size,
            config.patch_size,
            config.patch_size,
            config.patch_size,
            0,
            0,
            1,
            true,
        );

        let pos_embed = vec![0.0f32; config.num_position_embeddings * hidden];

        let blocks = (0..config.depth)
            .map(|_| ViTBlock::new(hidden, config.num_heads, mlp_hidden, eps))
            .collect();

        let merger = PatchMerger::new(hidden, config.spatial_merge_size, config.out_hidden_size, eps);

        Self {
            config,
            patch_embed,
            pos_embed,
            blocks,
            merger,
        }
    }

    /// Forward pass for a single image.
    ///
    /// Input: `[in_channels, H, W]` (channel-first, H and W must be multiples of
    /// `patch_size * spatial_merge_size`).
    ///
    /// Returns: `[num_merged_tokens, out_hidden_size]` (flattened token embeddings).
    pub fn forward(&self, image: &[f32], h: usize, w: usize) -> Vec<f32> {
        let hidden = self.config.hidden_size;
        let tp = self.config.temporal_patch_size;

        // For single images, duplicate frame to fill temporal dimension
        // [C, H, W] → [C*temporal, H, W]
        let temporal_input = if tp == 2 {
            let spatial = h * w;
            let c = self.config.in_channels;
            let mut expanded = vec![0.0f32; c * tp * spatial];
            for ch in 0..c {
                for t in 0..tp {
                    let dst_ch = ch * tp + t;
                    expanded[dst_ch * spatial..(dst_ch + 1) * spatial]
                        .copy_from_slice(&image[ch * spatial..(ch + 1) * spatial]);
                }
            }
            expanded
        } else {
            image.to_vec()
        };

        // Conv2d patch embedding
        let (grid_h, grid_w) = self.patch_embed.output_dims(h, w);
        let num_patches = grid_h * grid_w;
        let conv_channels = self.patch_embed.out_channels;
        let mut conv_out = vec![0.0f32; conv_channels * num_patches];
        self.patch_embed.forward(&temporal_input, h, w, &mut conv_out);

        // Transpose from [C, grid_h, grid_w] to [num_patches, hidden]
        let mut tokens = vec![0.0f32; num_patches * hidden];
        for c in 0..hidden {
            for p in 0..num_patches {
                tokens[p * hidden + c] = conv_out[c * num_patches + p];
            }
        }

        // Add position embeddings (index by spatial position)
        let pos_ids = compute_position_ids(grid_h, grid_w, self.config.num_position_embeddings);
        for (p, &pos_id) in pos_ids.iter().enumerate() {
            let pos_offset = pos_id * hidden;
            for d in 0..hidden {
                tokens[p * hidden + d] += self.pos_embed[pos_offset + d];
            }
        }

        // Transformer blocks
        for block in &self.blocks {
            block.forward(&mut tokens, num_patches, hidden);
        }

        // PatchMerger: group spatial_merge_size² tokens → project to out_hidden_size
        self.merger.forward(&tokens, grid_h, grid_w, hidden)
    }

    /// Forward pass returning raw patch tokens before merging (for advanced use).
    pub fn forward_features(&self, image: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let hidden = self.config.hidden_size;
        let tp = self.config.temporal_patch_size;

        let temporal_input = if tp == 2 {
            let spatial = h * w;
            let c = self.config.in_channels;
            let mut expanded = vec![0.0f32; c * tp * spatial];
            for ch in 0..c {
                for t in 0..tp {
                    let dst_ch = ch * tp + t;
                    expanded[dst_ch * spatial..(dst_ch + 1) * spatial]
                        .copy_from_slice(&image[ch * spatial..(ch + 1) * spatial]);
                }
            }
            expanded
        } else {
            image.to_vec()
        };

        let (grid_h, grid_w) = self.patch_embed.output_dims(h, w);
        let num_patches = grid_h * grid_w;
        let conv_channels = self.patch_embed.out_channels;
        let mut conv_out = vec![0.0f32; conv_channels * num_patches];
        self.patch_embed.forward(&temporal_input, h, w, &mut conv_out);

        let mut tokens = vec![0.0f32; num_patches * hidden];
        for c in 0..hidden {
            for p in 0..num_patches {
                tokens[p * hidden + c] = conv_out[c * num_patches + p];
            }
        }

        let pos_ids = compute_position_ids(grid_h, grid_w, self.config.num_position_embeddings);
        for (p, &pos_id) in pos_ids.iter().enumerate() {
            let pos_offset = pos_id * hidden;
            for d in 0..hidden {
                tokens[p * hidden + d] += self.pos_embed[pos_offset + d];
            }
        }

        for block in &self.blocks {
            block.forward(&mut tokens, num_patches, hidden);
        }

        (tokens, grid_h, grid_w)
    }
}

/// Compute position IDs for a grid of patches.
///
/// For a `grid_h × grid_w` patch grid, maps each position to an index into the
/// learned position embedding table. Uses bilinear interpolation when the grid
/// doesn't match the base grid size (sqrt(num_position_embeddings)).
fn compute_position_ids(grid_h: usize, grid_w: usize, num_position_embeddings: usize) -> Vec<usize> {
    let base_size = (num_position_embeddings as f64).sqrt() as usize;
    let num_patches = grid_h * grid_w;
    let mut ids = Vec::with_capacity(num_patches);

    if grid_h <= base_size && grid_w <= base_size {
        // Direct indexing — position (y, x) maps to y * base_size + x
        // Center the grid within the base grid
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

    fn tiny_qwen_config() -> QwenVisionConfig {
        QwenVisionConfig {
            depth: 2,
            hidden_size: 16,
            num_heads: 4,
            intermediate_size: 32,
            patch_size: 8,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            out_hidden_size: 32,
            num_position_embeddings: 64, // 8x8 base grid
            layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn test_patch_merger() {
        let pm = PatchMerger::new(16, 2, 32, 1e-5);
        // 4x4 grid → 2x2 merged = 4 tokens
        let tokens = vec![0.1f32; 16 * 16]; // 16 tokens × 16 dim
        let output = pm.forward(&tokens, 4, 4, 16);
        assert_eq!(output.len(), 4 * 32); // 4 merged tokens × 32 out_dim
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_qwen_vision_forward() {
        let config = tiny_qwen_config();
        let encoder = QwenVisionEncoder::new(config);

        // 32x32 image: patch_size=8 → 4x4 grid, merge_size=2 → 2x2 = 4 merged tokens
        let image = vec![0.1f32; 3 * 32 * 32];
        let output = encoder.forward(&image, 32, 32);
        assert_eq!(output.len(), 4 * 32); // 4 merged tokens × 32 out_dim
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_qwen_vision_forward_features() {
        let config = tiny_qwen_config();
        let encoder = QwenVisionEncoder::new(config);

        let image = vec![0.1f32; 3 * 32 * 32];
        let (tokens, gh, gw) = encoder.forward_features(&image, 32, 32);
        assert_eq!(gh, 4);
        assert_eq!(gw, 4);
        assert_eq!(tokens.len(), 16 * 16); // 16 patches × 16 hidden
    }

    #[test]
    fn test_position_ids() {
        let ids = compute_position_ids(4, 4, 64); // 8x8 base
        assert_eq!(ids.len(), 16);
        // All IDs should be valid
        for &id in &ids {
            assert!(id < 64);
        }
    }

    #[test]
    fn test_position_ids_oversized() {
        let ids = compute_position_ids(16, 16, 64); // 8x8 base, oversized
        assert_eq!(ids.len(), 256);
        for &id in &ids {
            assert!(id < 64);
        }
    }
}
