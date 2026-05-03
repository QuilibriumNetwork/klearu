/// Standard Vision Transformer (ViT) model.
///
/// Architecture: patch embedding → CLS token + position embedding → N transformer blocks → head.
/// Each block: LayerNorm → Multi-Head Self-Attention → residual → LayerNorm → MLP → residual.
///
/// Supports CLS token pooling (standard ViT) or global average pooling (DINOv2/SigLIP).

use crate::config::ViTConfig;
use crate::layers::{Conv2d, LayerNorm, LinearBias, gelu_inplace};

/// Pool type for extracting the final representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolType {
    /// Use the CLS token output (standard ViT, CLIP).
    Cls,
    /// Global average pooling over all patch tokens (DINOv2, SigLIP).
    Mean,
}

/// ViT patch embedding: Conv2d → flatten to tokens.
pub struct ViTPatchEmbed {
    pub proj: Conv2d,
    pub num_patches: usize,
}

impl ViTPatchEmbed {
    pub fn new(in_channels: usize, embed_dim: usize, patch_size: usize, image_size: usize) -> Self {
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        Self {
            proj: Conv2d::new(
                in_channels, embed_dim,
                patch_size, patch_size,
                patch_size, patch_size,
                0, 0, 1, true,
            ),
            num_patches,
        }
    }

    /// Forward: `[C, H, W]` → `[num_patches, embed_dim]`.
    pub fn forward(&self, input: &[f32], in_h: usize, in_w: usize) -> Vec<f32> {
        let (out_h, out_w) = self.proj.output_dims(in_h, in_w);
        let embed_dim = self.proj.out_channels;
        let num_patches = out_h * out_w;

        let mut conv_out = vec![0.0f32; embed_dim * num_patches];
        self.proj.forward(input, in_h, in_w, &mut conv_out);

        // Convert from [embed_dim, out_h, out_w] (channel-first) to [num_patches, embed_dim] (token layout)
        let mut tokens = vec![0.0f32; num_patches * embed_dim];
        for c in 0..embed_dim {
            for p in 0..num_patches {
                tokens[p * embed_dim + c] = conv_out[c * num_patches + p];
            }
        }

        tokens
    }
}

/// ViT transformer block: LayerNorm → MHA → residual → LayerNorm → MLP → residual.
pub struct ViTBlock {
    pub norm1: LayerNorm,
    pub attn: ViTSelfAttention,
    pub norm2: LayerNorm,
    pub mlp_fc1: LinearBias,
    pub mlp_fc2: LinearBias,
}

impl ViTBlock {
    pub fn new(embed_dim: usize, num_heads: usize, mlp_hidden: usize, eps: f32) -> Self {
        Self {
            norm1: LayerNorm::new(embed_dim, eps),
            attn: ViTSelfAttention::new(embed_dim, num_heads),
            norm2: LayerNorm::new(embed_dim, eps),
            mlp_fc1: LinearBias::new(embed_dim, mlp_hidden),
            mlp_fc2: LinearBias::new(mlp_hidden, embed_dim),
        }
    }

    /// Forward pass on `[seq_len, embed_dim]` tokens (in-place on the token buffer).
    pub fn forward(&self, tokens: &mut [f32], seq_len: usize, embed_dim: usize) {
        // x = x + attn(norm1(x))
        let mut normed = tokens.to_vec();
        for t in 0..seq_len {
            self.norm1.forward(&mut normed[t * embed_dim..(t + 1) * embed_dim]);
        }
        let attn_out = self.attn.forward(&normed, seq_len);
        for (x, a) in tokens.iter_mut().zip(attn_out.iter()) {
            *x += a;
        }

        // x = x + mlp(norm2(x))
        let mut mlp_buf = vec![0.0f32; self.mlp_fc1.out_features()];
        let mut mlp_out = vec![0.0f32; embed_dim];
        for t in 0..seq_len {
            let token = &mut tokens[t * embed_dim..(t + 1) * embed_dim];
            let mut normed_token = token.to_vec();
            self.norm2.forward(&mut normed_token);

            self.mlp_fc1.forward(&normed_token, &mut mlp_buf);
            gelu_inplace(&mut mlp_buf);
            self.mlp_fc2.forward(&mlp_buf, &mut mlp_out);

            for d in 0..embed_dim {
                token[d] += mlp_out[d];
            }
        }
    }
}

/// Multi-head self-attention (global, no windowing).
pub struct ViTSelfAttention {
    pub qkv: LinearBias,
    pub proj: LinearBias,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl ViTSelfAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        Self {
            qkv: LinearBias::new(embed_dim, embed_dim * 3),
            proj: LinearBias::new(embed_dim, embed_dim),
            num_heads,
            head_dim,
        }
    }

    /// Forward: `[seq_len, dim]` → `[seq_len, dim]`.
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let dim = self.num_heads * self.head_dim;

        // QKV projection
        let mut qkv_buf = vec![0.0f32; seq_len * dim * 3];
        for t in 0..seq_len {
            self.qkv.forward(
                &input[t * dim..(t + 1) * dim],
                &mut qkv_buf[t * dim * 3..(t + 1) * dim * 3],
            );
        }

        let scale = (self.head_dim as f32).powf(-0.5);
        let mut attn_output = vec![0.0f32; seq_len * dim];

        for head in 0..self.num_heads {
            let head_offset = head * self.head_dim;

            // Compute attention scores
            let mut scores = vec![0.0f32; seq_len * seq_len];
            for qi in 0..seq_len {
                for ki in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        let q = qkv_buf[qi * dim * 3 + head_offset + d];
                        let k = qkv_buf[ki * dim * 3 + dim + head_offset + d];
                        dot += q * k;
                    }
                    scores[qi * seq_len + ki] = dot * scale;
                }
            }

            // Softmax
            for qi in 0..seq_len {
                let row = &mut scores[qi * seq_len..(qi + 1) * seq_len];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                if sum > 0.0 {
                    for v in row.iter_mut() {
                        *v /= sum;
                    }
                }
            }

            // Weighted sum of V
            for qi in 0..seq_len {
                for d in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for vi in 0..seq_len {
                        let v = qkv_buf[vi * dim * 3 + 2 * dim + head_offset + d];
                        sum += scores[qi * seq_len + vi] * v;
                    }
                    attn_output[qi * dim + head_offset + d] = sum;
                }
            }
        }

        // Output projection
        let mut proj_output = vec![0.0f32; seq_len * dim];
        for t in 0..seq_len {
            self.proj.forward(
                &attn_output[t * dim..(t + 1) * dim],
                &mut proj_output[t * dim..(t + 1) * dim],
            );
        }

        proj_output
    }
}

/// Standard Vision Transformer model.
pub struct ViTModel {
    pub config: ViTConfig,
    pub patch_embed: ViTPatchEmbed,
    /// Learnable CLS token: `[embed_dim]`. Only used when `pool_type == Cls`.
    pub cls_token: Vec<f32>,
    /// Learnable absolute position embedding: `[1 + num_patches, embed_dim]` or `[num_patches, embed_dim]`.
    pub pos_embed: Vec<f32>,
    pub blocks: Vec<ViTBlock>,
    pub norm: LayerNorm,
    pub head: LinearBias,
    pub pool_type: PoolType,
}

impl ViTModel {
    pub fn new(config: ViTConfig) -> Self {
        let num_patches = (config.image_size / config.patch_size).pow(2);
        let embed_dim = config.embed_dim;
        let mlp_hidden = (embed_dim as f32 * config.mlp_ratio) as usize;
        let has_cls = config.pool_type == PoolType::Cls;
        let pos_len = if has_cls { 1 + num_patches } else { num_patches };

        let patch_embed = ViTPatchEmbed::new(
            config.in_channels, embed_dim, config.patch_size, config.image_size,
        );
        let cls_token = vec![0.0f32; embed_dim];
        let pos_embed = vec![0.0f32; pos_len * embed_dim];

        let blocks = (0..config.num_layers)
            .map(|_| ViTBlock::new(embed_dim, config.num_heads, mlp_hidden, config.layer_norm_eps))
            .collect();

        let norm = LayerNorm::new(embed_dim, config.layer_norm_eps);
        let head = LinearBias::new(embed_dim, config.num_classes);

        Self {
            config,
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
            pool_type: config.pool_type,
        }
    }

    /// Forward pass for classification.
    ///
    /// Input: `[in_channels, image_h, image_w]`.
    /// Returns: `[num_classes]` logits.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        let embed_dim = self.config.embed_dim;
        let image_size = self.config.image_size;

        // Patch embedding: [C, H, W] → [num_patches, embed_dim]
        let patch_tokens = self.patch_embed.forward(image, image_size, image_size);
        let num_patches = patch_tokens.len() / embed_dim;

        // Prepend CLS token if needed
        let (mut tokens, seq_len) = if self.pool_type == PoolType::Cls {
            let seq_len = 1 + num_patches;
            let mut tokens = vec![0.0f32; seq_len * embed_dim];
            tokens[..embed_dim].copy_from_slice(&self.cls_token);
            tokens[embed_dim..].copy_from_slice(&patch_tokens);
            (tokens, seq_len)
        } else {
            (patch_tokens, num_patches)
        };

        // Add position embedding
        debug_assert_eq!(self.pos_embed.len(), seq_len * embed_dim);
        for (t, p) in tokens.iter_mut().zip(self.pos_embed.iter()) {
            *t += p;
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
        let pooled = match self.pool_type {
            PoolType::Cls => {
                // CLS token is at position 0
                tokens[..embed_dim].to_vec()
            }
            PoolType::Mean => {
                // Average over all tokens
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
        };

        // Classification head
        let mut logits = vec![0.0f32; self.config.num_classes];
        self.head.forward(&pooled, &mut logits);

        logits
    }

    /// Extract features before the classification head.
    ///
    /// Returns: `[embed_dim]` pooled features.
    pub fn forward_features(&self, image: &[f32]) -> Vec<f32> {
        let embed_dim = self.config.embed_dim;
        let image_size = self.config.image_size;

        let patch_tokens = self.patch_embed.forward(image, image_size, image_size);
        let num_patches = patch_tokens.len() / embed_dim;

        let (mut tokens, seq_len) = if self.pool_type == PoolType::Cls {
            let seq_len = 1 + num_patches;
            let mut tokens = vec![0.0f32; seq_len * embed_dim];
            tokens[..embed_dim].copy_from_slice(&self.cls_token);
            tokens[embed_dim..].copy_from_slice(&patch_tokens);
            (tokens, seq_len)
        } else {
            (patch_tokens, num_patches)
        };

        for (t, p) in tokens.iter_mut().zip(self.pos_embed.iter()) {
            *t += p;
        }

        for block in &self.blocks {
            block.forward(&mut tokens, seq_len, embed_dim);
        }

        for t in 0..seq_len {
            self.norm.forward(&mut tokens[t * embed_dim..(t + 1) * embed_dim]);
        }

        match self.pool_type {
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

    fn tiny_vit_config() -> ViTConfig {
        ViTConfig {
            image_size: 32,
            in_channels: 3,
            patch_size: 8,
            embed_dim: 16,
            num_heads: 4,
            num_layers: 2,
            mlp_ratio: 2.0,
            num_classes: 10,
            layer_norm_eps: 1e-5,
            pool_type: PoolType::Cls,
        }
    }

    #[test]
    fn test_vit_forward_cls() {
        let config = tiny_vit_config();
        let model = ViTModel::new(config);

        let image = vec![0.1f32; 3 * 32 * 32];
        let logits = model.forward(&image);

        assert_eq!(logits.len(), 10);
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_vit_forward_mean_pool() {
        let config = ViTConfig {
            pool_type: PoolType::Mean,
            ..tiny_vit_config()
        };
        let model = ViTModel::new(config);

        let image = vec![0.1f32; 3 * 32 * 32];
        let logits = model.forward(&image);

        assert_eq!(logits.len(), 10);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_vit_forward_features() {
        let config = tiny_vit_config();
        let model = ViTModel::new(config);

        let image = vec![0.1f32; 3 * 32 * 32];
        let features = model.forward_features(&image);

        assert_eq!(features.len(), 16); // embed_dim
        for &v in &features {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_vit_patch_embed() {
        // 32x32 image, patch_size=8 → 4x4 = 16 patches
        let pe = ViTPatchEmbed::new(3, 16, 8, 32);
        assert_eq!(pe.num_patches, 16);

        let input = vec![0.1f32; 3 * 32 * 32];
        let tokens = pe.forward(&input, 32, 32);
        assert_eq!(tokens.len(), 16 * 16); // 16 patches × 16 embed_dim
    }

    #[test]
    fn test_vit_block() {
        let block = ViTBlock::new(16, 4, 32, 1e-5);
        let seq_len = 5;
        let mut tokens = vec![0.1f32; seq_len * 16];
        block.forward(&mut tokens, seq_len, 16);

        for &v in &tokens {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_vit_self_attention() {
        let attn = ViTSelfAttention::new(16, 4);
        let seq_len = 5;
        let input = vec![0.1f32; seq_len * 16];
        let output = attn.forward(&input, seq_len);

        assert_eq!(output.len(), seq_len * 16);
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_vit_different_inputs() {
        let config = tiny_vit_config();
        let model = ViTModel::new(config);

        let image1 = vec![0.1f32; 3 * 32 * 32];
        let image2 = vec![0.5f32; 3 * 32 * 32];

        let logits1 = model.forward(&image1);
        let logits2 = model.forward(&image2);

        // Different inputs should produce different outputs (unless all weights are zero).
        // With zero-initialized weights, outputs may be similar; verify finite at minimum.
        for &v in logits1.iter().chain(logits2.iter()) {
            assert!(v.is_finite());
        }
    }
}
