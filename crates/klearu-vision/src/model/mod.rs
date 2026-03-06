pub mod patch_embed;
pub mod cpe;
pub mod window_attention;
pub mod channel_attention;
pub mod davit_block;
pub mod davit_stage;
pub mod classification_head;
pub mod vit;
pub mod qwen_vision;
pub mod eva02;
pub mod convnext;
pub mod swin;
pub mod hiera;

use crate::config::DaViTConfig;
use davit_stage::DaViTStage;
use classification_head::ClassificationHead;
use patch_embed::PatchEmbed;

/// DaViT (Dual Attention Vision Transformer) model.
pub struct DaViTModel {
    pub config: DaViTConfig,
    pub stem: PatchEmbed,
    pub stages: Vec<DaViTStage>,
    pub head: ClassificationHead,
}

impl DaViTModel {
    pub fn new(config: DaViTConfig) -> Self {
        let stem = PatchEmbed::new(config.in_channels, config.embed_dims[0], config.layer_norm_eps);

        let mut stages = Vec::with_capacity(4);
        for s in 0..4 {
            let has_downsample = s > 0;
            let prev_dim = if s > 0 { config.embed_dims[s - 1] } else { config.embed_dims[0] };
            let dim = config.embed_dims[s];
            let num_heads = config.num_heads[s];
            let depth = config.depths[s];
            let mlp_hidden = config.mlp_hidden_dim(s);
            let window_size = config.window_size;
            let eps = config.layer_norm_eps;

            stages.push(DaViTStage::new(
                has_downsample,
                prev_dim,
                dim,
                num_heads,
                depth,
                mlp_hidden,
                window_size,
                eps,
            ));
        }

        let head = ClassificationHead::new(
            config.embed_dims[3],
            config.num_classes,
            config.layer_norm_eps,
        );

        Self { config, stem, stages, head }
    }

    /// Forward pass for image classification.
    ///
    /// Input: `[in_channels, image_h, image_w]` (channel-first).
    /// Returns: `[num_classes]` logits.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        let (mut features, mut h, mut w) = self.stem.forward(
            image,
            self.config.image_size,
            self.config.image_size,
        );

        for stage in &self.stages {
            let result = stage.forward(&features, h, w);
            features = result.0;
            h = result.1;
            w = result.2;
        }

        self.head.forward(&features, self.config.embed_dims[3], h, w)
    }

    /// Extract intermediate feature maps at each stage resolution.
    ///
    /// Returns a `Vec` of `(features, h, w)` tuples, one per stage.
    /// `features` is `[embed_dim, h, w]` channel-first.
    ///
    /// Useful for detection/segmentation heads that need multi-scale features.
    pub fn forward_features(&self, image: &[f32]) -> Vec<(Vec<f32>, usize, usize)> {
        let (mut features, mut h, mut w) = self.stem.forward(
            image,
            self.config.image_size,
            self.config.image_size,
        );

        let mut stage_outputs = Vec::with_capacity(self.stages.len());

        for stage in &self.stages {
            let result = stage.forward(&features, h, w);
            features = result.0;
            h = result.1;
            w = result.2;
            stage_outputs.push((features.clone(), h, w));
        }

        stage_outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_test_config() -> DaViTConfig {
        // image_size=224 → stem(7x7,s4,p3) → 56x56
        // stage0: no downsample → 56x56 (window=7 fits: 56/7=8)
        // stage1: downsample → 28x28 (window=7 fits: 28/7=4)
        // stage2: downsample → 14x14 (window=7 fits: 14/7=2)
        // stage3: downsample → 7x7 (window=7 fits: 7/7=1)
        DaViTConfig {
            image_size: 224,
            in_channels: 3,
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
    fn test_davit_model_forward_finite() {
        let config = tiny_test_config();
        let model = DaViTModel::new(config.clone());

        let image_size = config.image_size;
        let input = vec![0.1f32; 3 * image_size * image_size];
        let logits = model.forward(&input);

        assert_eq!(logits.len(), config.num_classes);
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_davit_forward_correct_shape() {
        let config = tiny_test_config();
        let model = DaViTModel::new(config.clone());

        let size = 3 * config.image_size * config.image_size;
        let input = vec![0.1f32; size];

        let logits = model.forward(&input);
        assert_eq!(logits.len(), config.num_classes);

        // All logits should be finite
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] = {v}");
        }
    }

    #[test]
    fn test_davit_forward_features() {
        let config = tiny_test_config();
        let model = DaViTModel::new(config.clone());

        let input = vec![0.1f32; 3 * config.image_size * config.image_size];
        let features = model.forward_features(&input);

        assert_eq!(features.len(), 4); // 4 stages

        // Stage 0: 56x56, dim=8
        assert_eq!(features[0].1, 56);
        assert_eq!(features[0].2, 56);
        assert_eq!(features[0].0.len(), 8 * 56 * 56);

        // Stage 1: 28x28, dim=16
        assert_eq!(features[1].1, 28);
        assert_eq!(features[1].2, 28);
        assert_eq!(features[1].0.len(), 16 * 28 * 28);

        // Stage 2: 14x14, dim=32
        assert_eq!(features[2].1, 14);
        assert_eq!(features[2].2, 14);
        assert_eq!(features[2].0.len(), 32 * 14 * 14);

        // Stage 3: 7x7, dim=64
        assert_eq!(features[3].1, 7);
        assert_eq!(features[3].2, 7);
        assert_eq!(features[3].0.len(), 64 * 7 * 7);

        // All values finite
        for (stage_features, _, _) in &features {
            for &v in stage_features {
                assert!(v.is_finite());
            }
        }
    }
}
