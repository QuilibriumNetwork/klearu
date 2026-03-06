use crate::layers::{LayerNorm, LinearBias};

/// Classification head: GlobalAvgPool → LayerNorm → Linear → logits.
pub struct ClassificationHead {
    pub norm: LayerNorm,
    pub fc: LinearBias,
}

impl ClassificationHead {
    pub fn new(embed_dim: usize, num_classes: usize, eps: f32) -> Self {
        Self {
            norm: LayerNorm::new(embed_dim, eps),
            fc: LinearBias::new(embed_dim, num_classes),
        }
    }

    /// Forward pass.
    ///
    /// Input: `[C, H, W]` (channel-first features).
    /// Returns: `[num_classes]` logits.
    pub fn forward(&self, features: &[f32], num_channels: usize, h: usize, w: usize) -> Vec<f32> {
        debug_assert_eq!(features.len(), num_channels * h * w);
        let n = h * w;

        // Global average pooling: mean over spatial dims per channel
        let mut pooled = vec![0.0f32; num_channels];
        for c in 0..num_channels {
            let base = c * n;
            let sum: f32 = features[base..base + n].iter().sum();
            pooled[c] = sum / n as f32;
        }

        // LayerNorm
        self.norm.forward(&mut pooled);

        // Linear → logits
        let mut logits = vec![0.0f32; self.fc.out_features()];
        self.fc.forward(&pooled, &mut logits);

        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_head() {
        let head = ClassificationHead::new(8, 10, 1e-5);
        let features = vec![0.1f32; 8 * 7 * 7];
        let logits = head.forward(&features, 8, 7, 7);
        assert_eq!(logits.len(), 10);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_classification_head_different_inputs() {
        let mut head = ClassificationHead::new(4, 5, 1e-5);
        // Set non-zero weights so different inputs produce different outputs
        for i in 0..5 {
            let mut w = vec![0.0; 4];
            w[i % 4] = 1.0;
            head.fc.weights.set_weights(i, &w);
        }

        let features1 = vec![0.1f32; 4 * 4 * 4];
        let mut features2 = vec![0.1f32; 4 * 4 * 4];
        // Make channel 0 values different
        for i in 0..16 {
            features2[i] = 0.5;
        }

        let logits1 = head.forward(&features1, 4, 4, 4);
        let logits2 = head.forward(&features2, 4, 4, 4);

        let any_diff = logits1.iter().zip(logits2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff);
    }
}
