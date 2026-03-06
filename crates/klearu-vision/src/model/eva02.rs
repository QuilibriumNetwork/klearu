/// EVA-02 Vision Transformer.
///
/// ViT variant with:
/// - SwiGLU MLP instead of standard MLP
/// - Sub-layer norm after attention (pre/post norm)
/// - 2D RoPE for positional encoding
///
/// Reuses `ViTSelfAttention` for the attention mechanism
/// and `RoPE2d` for positional encoding.

use crate::config::ViTConfig;
use crate::layers::{LayerNorm, LinearBias, SwiGluMlp, RoPE2d};
use crate::model::vit::{ViTPatchEmbed, ViTSelfAttention, PoolType};

/// EVA-02 transformer block with SwiGLU MLP and sub-layer norm.
pub struct EVA02Block {
    pub norm1: LayerNorm,
    pub attn: ViTSelfAttention,
    pub sub_norm: LayerNorm,
    pub norm2: LayerNorm,
    pub mlp: SwiGluMlp,
}

impl EVA02Block {
    pub fn new(embed_dim: usize, num_heads: usize, mlp_hidden: usize, eps: f32) -> Self {
        Self {
            norm1: LayerNorm::new(embed_dim, eps),
            attn: ViTSelfAttention::new(embed_dim, num_heads),
            sub_norm: LayerNorm::new(embed_dim, eps),
            norm2: LayerNorm::new(embed_dim, eps),
            mlp: SwiGluMlp::new(embed_dim, mlp_hidden),
        }
    }

    /// Forward pass on `[seq_len, embed_dim]` tokens (in-place).
    pub fn forward(&self, tokens: &mut [f32], seq_len: usize, embed_dim: usize) {
        // x = x + sub_norm(attn(norm1(x)))
        let mut normed = tokens.to_vec();
        for t in 0..seq_len {
            self.norm1.forward(&mut normed[t * embed_dim..(t + 1) * embed_dim]);
        }
        let mut attn_out = self.attn.forward(&normed, seq_len);
        for t in 0..seq_len {
            self.sub_norm.forward(&mut attn_out[t * embed_dim..(t + 1) * embed_dim]);
        }
        for (x, a) in tokens.iter_mut().zip(attn_out.iter()) {
            *x += a;
        }

        // x = x + mlp(norm2(x))
        let mut mlp_out = vec![0.0f32; embed_dim];
        for t in 0..seq_len {
            let token = &mut tokens[t * embed_dim..(t + 1) * embed_dim];
            let mut normed_token = token.to_vec();
            self.norm2.forward(&mut normed_token);
            self.mlp.forward(&normed_token, &mut mlp_out);
            for d in 0..embed_dim {
                token[d] += mlp_out[d];
            }
        }
    }
}

/// EVA-02 Vision Transformer model.
pub struct EVA02Model {
    pub config: ViTConfig,
    pub patch_embed: ViTPatchEmbed,
    pub cls_token: Vec<f32>,
    pub pos_embed: Vec<f32>,
    pub blocks: Vec<EVA02Block>,
    pub norm: LayerNorm,
    pub head: LinearBias,
    pub rope: Option<RoPE2d>,
}

impl EVA02Model {
    pub fn new(config: ViTConfig) -> Self {
        let embed_dim = config.embed_dim;
        let num_patches = config.num_patches();
        // EVA-02 typically uses 2/3 of the standard MLP ratio for SwiGLU
        let mlp_hidden = ((embed_dim as f32 * config.mlp_ratio * 2.0 / 3.0) as usize + 7) & !7;
        let eps = config.layer_norm_eps;

        let patch_embed = ViTPatchEmbed::new(
            config.in_channels, embed_dim, config.patch_size, config.image_size,
        );

        let has_cls = config.pool_type == PoolType::Cls;
        let pos_len = if has_cls { 1 + num_patches } else { num_patches };
        let cls_token = vec![0.0f32; embed_dim];
        let pos_embed = vec![0.0f32; pos_len * embed_dim];

        let blocks = (0..config.num_layers)
            .map(|_| EVA02Block::new(embed_dim, config.num_heads, mlp_hidden, eps))
            .collect();

        let norm = LayerNorm::new(embed_dim, eps);
        let head = LinearBias::new(embed_dim, config.num_classes);

        let grid_size = config.image_size / config.patch_size;
        let rope = Some(RoPE2d::new(grid_size, grid_size, config.head_dim(), 10000.0));

        Self {
            config,
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
            rope,
        }
    }

    /// Forward pass for classification.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        let features = self.forward_features(image);
        let mut logits = vec![0.0f32; self.config.num_classes];
        self.head.forward(&features, &mut logits);
        logits
    }

    /// Forward pass to extract features.
    pub fn forward_features(&self, image: &[f32]) -> Vec<f32> {
        let embed_dim = self.config.embed_dim;
        let image_size = self.config.image_size;

        let patch_tokens = self.patch_embed.forward(image, image_size, image_size);
        let num_patches = patch_tokens.len() / embed_dim;

        let has_cls = self.config.pool_type == PoolType::Cls;
        let (mut tokens, seq_len) = if has_cls {
            let seq_len = 1 + num_patches;
            let mut tokens = vec![0.0f32; seq_len * embed_dim];
            tokens[..embed_dim].copy_from_slice(&self.cls_token);
            tokens[embed_dim..].copy_from_slice(&patch_tokens);
            (tokens, seq_len)
        } else {
            (patch_tokens, num_patches)
        };

        // Add position embedding
        if !self.pos_embed.is_empty() {
            for (t, p) in tokens.iter_mut().zip(self.pos_embed.iter()) {
                *t += p;
            }
        }

        // Transformer blocks
        for block in &self.blocks {
            block.forward(&mut tokens, seq_len, embed_dim);
        }

        // Final norm
        for t in 0..seq_len {
            self.norm.forward(&mut tokens[t * embed_dim..(t + 1) * embed_dim]);
        }

        // Pool
        match self.config.pool_type {
            PoolType::Cls => tokens[..embed_dim].to_vec(),
            PoolType::Mean => {
                let mut mean = vec![0.0f32; embed_dim];
                for t in 0..seq_len {
                    for d in 0..embed_dim {
                        mean[d] += tokens[t * embed_dim + d];
                    }
                }
                let inv = 1.0 / seq_len as f32;
                for v in mean.iter_mut() {
                    *v *= inv;
                }
                mean
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ViTConfig;

    #[test]
    fn test_eva02_forward() {
        let config = ViTConfig {
            image_size: 32,
            in_channels: 3,
            patch_size: 8,
            embed_dim: 16,
            num_heads: 4,
            num_layers: 2,
            mlp_ratio: 4.0,
            num_classes: 10,
            layer_norm_eps: 1e-5,
            pool_type: PoolType::Cls,
        };
        let model = EVA02Model::new(config);
        let image = vec![0.1f32; 3 * 32 * 32];
        let logits = model.forward(&image);
        assert_eq!(logits.len(), 10);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }
}
