/// Swin Transformer model.
///
/// Architecture: 4 stages with shifted window attention.
/// - Stage 0: Patch embed (4×4 conv stride 4)
/// - Stages 1-3: PatchMerge downsample → blocks
/// - Each block alternates shift_size=0 and shift_size=window_size/2.
///
/// Reuses `WindowAttention` from `window_attention.rs` which already supports
/// relative position bias and shifted windows.

use crate::config::SwinConfig;
use crate::layers::{Conv2d, LayerNorm, LinearBias, PatchMerge, gelu_inplace};
use crate::model::window_attention::WindowAttention;

/// Swin Transformer block: norm → window attention → residual → norm → MLP → residual.
pub struct SwinBlock {
    pub norm1: LayerNorm,
    pub attn: WindowAttention,
    pub norm2: LayerNorm,
    pub mlp_fc1: LinearBias,
    pub mlp_fc2: LinearBias,
    pub shift_size: usize,
}

impl SwinBlock {
    pub fn new(dim: usize, num_heads: usize, window_size: usize, shift_size: usize, mlp_hidden: usize, eps: f32) -> Self {
        Self {
            norm1: LayerNorm::new(dim, eps),
            attn: WindowAttention::new_with_relative_position_bias(dim, num_heads, window_size),
            norm2: LayerNorm::new(dim, eps),
            mlp_fc1: LinearBias::new(dim, mlp_hidden),
            mlp_fc2: LinearBias::new(mlp_hidden, dim),
            shift_size,
        }
    }

    /// Forward: `[H*W, dim]` → `[H*W, dim]` (in-place on token buffer).
    pub fn forward(&self, tokens: &mut [f32], h: usize, w: usize, dim: usize) {
        let n = h * w;

        // x = x + attn(norm1(x))
        let mut normed = tokens.to_vec();
        for t in 0..n {
            self.norm1.forward(&mut normed[t * dim..(t + 1) * dim]);
        }
        let attn_out = if self.shift_size > 0 {
            self.attn.forward_shifted(&normed, h, w, self.shift_size)
        } else {
            self.attn.forward(&normed, h, w)
        };
        for (x, a) in tokens.iter_mut().zip(attn_out.iter()) {
            *x += a;
        }

        // x = x + mlp(norm2(x))
        let mut mlp_buf = vec![0.0f32; self.mlp_fc1.out_features()];
        let mut mlp_out = vec![0.0f32; dim];
        for t in 0..n {
            let token = &mut tokens[t * dim..(t + 1) * dim];
            let mut normed_token = token.to_vec();
            self.norm2.forward(&mut normed_token);
            self.mlp_fc1.forward(&normed_token, &mut mlp_buf);
            gelu_inplace(&mut mlp_buf);
            self.mlp_fc2.forward(&mlp_buf, &mut mlp_out);
            for d in 0..dim {
                token[d] += mlp_out[d];
            }
        }
    }
}

/// Single Swin stage: optional PatchMerge + N SwinBlocks.
pub struct SwinStage {
    pub downsample: Option<PatchMerge>,
    pub blocks: Vec<SwinBlock>,
}

/// Swin Transformer model.
pub struct SwinModel {
    pub config: SwinConfig,
    pub patch_embed: Conv2d,
    pub patch_norm: LayerNorm,
    pub stages: Vec<SwinStage>,
    pub final_norm: LayerNorm,
    pub head: LinearBias,
}

impl SwinModel {
    pub fn new(config: SwinConfig) -> Self {
        let eps = config.layer_norm_eps;

        // Patch embedding: 4×4 conv stride 4
        let patch_embed = Conv2d::new(
            config.in_channels, config.embed_dims[0],
            config.patch_size, config.patch_size,
            config.patch_size, config.patch_size,
            0, 0, 1, true,
        );
        let patch_norm = LayerNorm::new(config.embed_dims[0], eps);

        let mut stages = Vec::with_capacity(4);
        for s in 0..4 {
            let dim = config.embed_dims[s];
            let num_heads = config.num_heads[s];
            let ws = config.window_size;
            let mlp_hidden = config.mlp_hidden_dim(s);

            let downsample = if s > 0 {
                Some(PatchMerge::new(config.embed_dims[s - 1], eps))
            } else {
                None
            };

            let blocks = (0..config.depths[s])
                .map(|i| {
                    let shift = if i % 2 == 1 { ws / 2 } else { 0 };
                    SwinBlock::new(dim, num_heads, ws, shift, mlp_hidden, eps)
                })
                .collect();

            stages.push(SwinStage { downsample, blocks });
        }

        let final_norm = LayerNorm::new(config.embed_dims[3], eps);
        let head = LinearBias::new(config.embed_dims[3], config.num_classes);

        Self {
            config,
            patch_embed,
            patch_norm,
            stages,
            final_norm,
            head,
        }
    }

    /// Forward pass for classification.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        let features = self.forward_features(image);
        let mut logits = vec![0.0f32; self.config.num_classes];
        self.head.forward(&features, &mut logits);
        logits
    }

    /// Forward to get pooled features.
    pub fn forward_features(&self, image: &[f32]) -> Vec<f32> {
        let image_size = self.config.image_size;
        let dim0 = self.config.embed_dims[0];

        // Patch embedding
        let (grid_h, grid_w) = self.patch_embed.output_dims(image_size, image_size);
        let num_patches = grid_h * grid_w;
        let mut conv_out = vec![0.0f32; dim0 * num_patches];
        self.patch_embed.forward(image, image_size, image_size, &mut conv_out);

        // Transpose [C, H, W] → [H*W, C]
        let mut tokens = vec![0.0f32; num_patches * dim0];
        for c in 0..dim0 {
            for p in 0..num_patches {
                tokens[p * dim0 + c] = conv_out[c * num_patches + p];
            }
        }

        // Patch norm
        for t in 0..num_patches {
            self.patch_norm.forward(&mut tokens[t * dim0..(t + 1) * dim0]);
        }

        let mut h = grid_h;
        let mut w = grid_w;
        let mut dim = dim0;

        for s in 0..4 {
            // Downsample
            if let Some(ref ds) = self.stages[s].downsample {
                tokens = ds.forward(&tokens, h, w, dim);
                h /= 2;
                w /= 2;
                dim *= 2;
            }

            // Blocks
            for block in &self.stages[s].blocks {
                block.forward(&mut tokens, h, w, dim);
            }
        }

        // Global average pool
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

        self.final_norm.forward(&mut pooled);
        pooled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SwinConfig;

    fn tiny_test_config() -> SwinConfig {
        // Use image_size=56 for small test (divisible by 4*7)
        SwinConfig {
            image_size: 56,
            in_channels: 3,
            patch_size: 4,
            embed_dims: [8, 16, 32, 64],
            num_heads: [2, 2, 4, 8],
            depths: [1, 1, 1, 1],
            window_size: 7,
            mlp_ratio: 2.0,
            num_classes: 10,
            layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn test_swin_block() {
        let block = SwinBlock::new(8, 2, 7, 0, 16, 1e-5);
        let h = 14;
        let w = 14;
        let mut tokens = vec![0.1f32; h * w * 8];
        block.forward(&mut tokens, h, w, 8);
        for &v in &tokens {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_swin_forward() {
        let config = tiny_test_config();
        let model = SwinModel::new(config);
        let image = vec![0.1f32; 3 * 56 * 56];
        let logits = model.forward(&image);
        assert_eq!(logits.len(), 10);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }
}
