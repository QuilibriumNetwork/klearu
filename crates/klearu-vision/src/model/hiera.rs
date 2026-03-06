/// Hiera (Hierarchical ViT) model — SAM2 backbone.
///
/// Architecture:
/// - Patch embed (conv stride patch_size)
/// - 4 stages with mask-unit attention (simple windowed, no relative bias)
/// - Token merging via 2×2 pooling between stages
/// - No shifted windows

use crate::config::HieraConfig;
use crate::layers::{Conv2d, LayerNorm, LinearBias, gelu_inplace};
use crate::model::vit::ViTSelfAttention;

/// Hiera transformer block: norm → attention → residual → norm → MLP → residual.
pub struct HieraBlock {
    pub norm1: LayerNorm,
    pub attn: ViTSelfAttention,
    pub norm2: LayerNorm,
    pub mlp_fc1: LinearBias,
    pub mlp_fc2: LinearBias,
}

impl HieraBlock {
    pub fn new(dim: usize, num_heads: usize, mlp_hidden: usize, eps: f32) -> Self {
        Self {
            norm1: LayerNorm::new(dim, eps),
            attn: ViTSelfAttention::new(dim, num_heads),
            norm2: LayerNorm::new(dim, eps),
            mlp_fc1: LinearBias::new(dim, mlp_hidden),
            mlp_fc2: LinearBias::new(mlp_hidden, dim),
        }
    }

    /// Forward: `[seq_len, dim]` tokens in-place.
    pub fn forward(&self, tokens: &mut [f32], seq_len: usize, dim: usize) {
        // x = x + attn(norm1(x))
        let mut normed = tokens.to_vec();
        for t in 0..seq_len {
            self.norm1.forward(&mut normed[t * dim..(t + 1) * dim]);
        }
        let attn_out = self.attn.forward(&normed, seq_len);
        for (x, a) in tokens.iter_mut().zip(attn_out.iter()) {
            *x += a;
        }

        // x = x + mlp(norm2(x))
        let mut mlp_buf = vec![0.0f32; self.mlp_fc1.out_features()];
        let mut mlp_out = vec![0.0f32; dim];
        for t in 0..seq_len {
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

/// Token merging: 2×2 pooling with projection to double channels.
pub struct HieraTokenMerge {
    pub proj: LinearBias,
}

impl HieraTokenMerge {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            proj: LinearBias::new(4 * in_dim, out_dim),
        }
    }

    /// Forward: `[H*W, in_dim]` → `[H/2*W/2, out_dim]`.
    pub fn forward(&self, tokens: &[f32], h: usize, w: usize, in_dim: usize) -> Vec<f32> {
        let new_h = h / 2;
        let new_w = w / 2;
        let num_out = new_h * new_w;
        let merged_dim = 4 * in_dim;
        let out_dim = self.proj.out_features();

        let mut output = vec![0.0f32; num_out * out_dim];

        for y in 0..new_h {
            for x in 0..new_w {
                let out_idx = y * new_w + x;
                let mut merged = vec![0.0f32; merged_dim];

                // Gather 2×2 tokens
                let offsets = [
                    (2 * y) * w + (2 * x),
                    (2 * y) * w + (2 * x + 1),
                    (2 * y + 1) * w + (2 * x),
                    (2 * y + 1) * w + (2 * x + 1),
                ];
                for (i, &off) in offsets.iter().enumerate() {
                    merged[i * in_dim..(i + 1) * in_dim]
                        .copy_from_slice(&tokens[off * in_dim..(off + 1) * in_dim]);
                }

                self.proj.forward(&merged, &mut output[out_idx * out_dim..(out_idx + 1) * out_dim]);
            }
        }

        output
    }
}

/// Hiera model.
pub struct HieraModel {
    pub config: HieraConfig,
    pub patch_embed: Conv2d,
    pub stages: Vec<Vec<HieraBlock>>,
    pub merges: Vec<Option<HieraTokenMerge>>,
    pub final_norm: LayerNorm,
    pub head: LinearBias,
}

impl HieraModel {
    pub fn new(config: HieraConfig) -> Self {
        let patch_embed = Conv2d::new(
            config.in_channels, config.embed_dims[0],
            config.patch_size, config.patch_size,
            config.patch_size, config.patch_size,
            0, 0, 1, true,
        );

        let mut stages = Vec::with_capacity(4);
        let mut merges = Vec::with_capacity(4);

        for s in 0..4 {
            if s > 0 {
                merges.push(Some(HieraTokenMerge::new(
                    config.embed_dims[s - 1], config.embed_dims[s],
                )));
            } else {
                merges.push(None);
            }

            let mlp_hidden = config.mlp_hidden_dim(s);
            let blocks = (0..config.depths[s])
                .map(|_| HieraBlock::new(config.embed_dims[s], config.num_heads[s], mlp_hidden, 1e-6))
                .collect();
            stages.push(blocks);
        }

        let final_norm = LayerNorm::new(config.embed_dims[3], 1e-6);
        let head = LinearBias::new(config.embed_dims[3], config.num_classes);

        Self {
            config,
            patch_embed,
            stages,
            merges,
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

        let mut h = grid_h;
        let mut w = grid_w;
        let mut dim = dim0;

        for s in 0..4 {
            // Token merge (downsample)
            if let Some(ref merge) = self.merges[s] {
                tokens = merge.forward(&tokens, h, w, dim);
                h /= 2;
                w /= 2;
                dim = self.config.embed_dims[s];
            }

            // Blocks
            let n = h * w;
            for block in &self.stages[s] {
                block.forward(&mut tokens, n, dim);
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
    use crate::config::HieraConfig;

    fn tiny_test_config() -> HieraConfig {
        HieraConfig {
            image_size: 32,
            in_channels: 3,
            patch_size: 4,
            embed_dims: [8, 16, 32, 64],
            num_heads: [1, 2, 4, 8],
            depths: [1, 1, 1, 1],
            mask_unit_size: 4,
            mlp_ratio: 2.0,
            num_classes: 10,
        }
    }

    #[test]
    fn test_hiera_block() {
        let block = HieraBlock::new(8, 2, 16, 1e-6);
        let mut tokens = vec![0.1f32; 16 * 8]; // 16 tokens × 8 dim
        block.forward(&mut tokens, 16, 8);
        for &v in &tokens {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_hiera_forward() {
        let config = tiny_test_config();
        let model = HieraModel::new(config);
        let image = vec![0.1f32; 3 * 32 * 32];
        let logits = model.forward(&image);
        assert_eq!(logits.len(), 10);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }
}
