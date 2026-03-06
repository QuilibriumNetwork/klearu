use crate::layers::{Conv2d, LayerNorm};

/// Convolutional patch embedding (DaViT stem).
///
/// `Conv2d(3, embed_dim, 7, stride=4, pad=3)` → `LayerNorm2d(embed_dim)`.
/// Reduces spatial resolution by 4x.
pub struct PatchEmbed {
    pub conv: Conv2d,
    pub norm: LayerNorm,
}

impl PatchEmbed {
    pub fn new(in_channels: usize, embed_dim: usize, eps: f32) -> Self {
        Self {
            conv: Conv2d::new(in_channels, embed_dim, 7, 7, 4, 4, 3, 3, 1, true),
            norm: LayerNorm::new(embed_dim, eps),
        }
    }

    /// Forward pass.
    ///
    /// Input: `[in_channels, image_h, image_w]`.
    /// Returns: `(features, out_h, out_w)` where features is `[embed_dim, out_h, out_w]`.
    pub fn forward(&self, input: &[f32], in_h: usize, in_w: usize) -> (Vec<f32>, usize, usize) {
        let (out_h, out_w) = self.conv.output_dims(in_h, in_w);
        let embed_dim = self.norm.dim();
        let out_size = embed_dim * out_h * out_w;

        let mut features = vec![0.0f32; out_size];
        self.conv.forward(input, in_h, in_w, &mut features);

        // Apply LayerNorm2d (normalizes over channel dimension at each spatial position)
        self.norm.forward_2d(&mut features, out_h, out_w);

        (features, out_h, out_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_embed_dims_224() {
        let pe = PatchEmbed::new(3, 96, 1e-5);
        let input = vec![0.0f32; 3 * 224 * 224];
        let (features, h, w) = pe.forward(&input, 224, 224);
        assert_eq!(h, 56);
        assert_eq!(w, 56);
        assert_eq!(features.len(), 96 * 56 * 56);
    }

    #[test]
    fn test_patch_embed_dims_28() {
        let pe = PatchEmbed::new(3, 8, 1e-5);
        let input = vec![0.1f32; 3 * 28 * 28];
        let (features, h, w) = pe.forward(&input, 28, 28);
        assert_eq!(h, 7);
        assert_eq!(w, 7);
        assert_eq!(features.len(), 8 * 7 * 7);

        // Check finite
        for &v in &features {
            assert!(v.is_finite());
        }
    }
}
