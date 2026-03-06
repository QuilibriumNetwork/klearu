/// ConvNeXt model (V1 and V2).
///
/// Architecture:
/// - Stem: 4×4 conv stride 4 + LayerNorm
/// - 4 stages with depthwise 7×7 conv → LayerNorm → fc1 → GELU → fc2 → LayerScale/GRN
/// - Downsampling between stages: LayerNorm + 2×2 conv stride 2

use crate::config::ConvNextConfig;
use crate::layers::{Conv2d, LayerNorm, LinearBias, LayerScale, GRN, gelu_inplace};

/// Single ConvNeXt block.
pub struct ConvNextBlock {
    pub dwconv: Conv2d,
    pub norm: LayerNorm,
    pub fc1: LinearBias,
    pub fc2: LinearBias,
    pub layer_scale: Option<LayerScale>,
    pub grn: Option<GRN>,
}

impl ConvNextBlock {
    pub fn new(dim: usize, is_v2: bool, layer_scale_init: f32, eps: f32) -> Self {
        Self {
            dwconv: Conv2d::new(dim, dim, 7, 7, 1, 1, 3, 3, dim, true),
            norm: LayerNorm::new(dim, eps),
            fc1: LinearBias::new(dim, 4 * dim),
            fc2: LinearBias::new(4 * dim, dim),
            layer_scale: if !is_v2 { Some(LayerScale::new(dim, layer_scale_init)) } else { None },
            grn: if is_v2 { Some(GRN::new(4 * dim)) } else { None },
        }
    }

    /// Forward: `[C, H, W]` → `[C, H, W]` with residual.
    pub fn forward(&self, input: &mut [f32], dim: usize, h: usize, w: usize) {
        let spatial = h * w;
        let mut conv_out = vec![0.0f32; dim * spatial];
        self.dwconv.forward(input, h, w, &mut conv_out);

        // Permute to [H*W, C], apply norm + MLP, permute back
        let mut fc1_buf = vec![0.0f32; 4 * dim];
        let mut fc2_buf = vec![0.0f32; dim];

        for p in 0..spatial {
            // Gather channel values for this spatial position
            let mut token = vec![0.0f32; dim];
            for c in 0..dim {
                token[c] = conv_out[c * spatial + p];
            }

            self.norm.forward(&mut token);
            self.fc1.forward(&token, &mut fc1_buf);
            gelu_inplace(&mut fc1_buf);

            if let Some(ref grn) = self.grn {
                grn.forward(&mut fc1_buf);
            }

            self.fc2.forward(&fc1_buf, &mut fc2_buf);

            if let Some(ref ls) = self.layer_scale {
                ls.forward(&mut fc2_buf);
            }

            // Residual and scatter back
            for c in 0..dim {
                input[c * spatial + p] += fc2_buf[c];
            }
        }
    }
}

/// ConvNeXt downsampling layer: LayerNorm (channel-wise) + 2×2 conv stride 2.
pub struct ConvNextDownsample {
    pub norm: LayerNorm,
    pub conv: Conv2d,
}

impl ConvNextDownsample {
    pub fn new(in_dim: usize, out_dim: usize, eps: f32) -> Self {
        Self {
            norm: LayerNorm::new(in_dim, eps),
            conv: Conv2d::new(in_dim, out_dim, 2, 2, 2, 2, 0, 0, 1, true),
        }
    }

    /// Forward: `[in_dim, H, W]` → `[out_dim, H/2, W/2]`.
    pub fn forward(&self, input: &[f32], _in_dim: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        // Apply LayerNorm per spatial position (channel-wise)
        let mut normed = input.to_vec();
        self.norm.forward_2d(&mut normed, h, w);

        let (out_h, out_w) = self.conv.output_dims(h, w);
        let out_dim = self.conv.out_channels;
        let mut output = vec![0.0f32; out_dim * out_h * out_w];
        self.conv.forward(&normed, h, w, &mut output);

        (output, out_h, out_w)
    }
}

/// ConvNeXt model.
pub struct ConvNextModel {
    pub config: ConvNextConfig,
    pub stem_conv: Conv2d,
    pub stem_norm: LayerNorm,
    pub stages: Vec<Vec<ConvNextBlock>>,
    pub downsamples: Vec<Option<ConvNextDownsample>>,
    pub final_norm: LayerNorm,
    pub head: LinearBias,
}

impl ConvNextModel {
    pub fn new(config: ConvNextConfig) -> Self {
        let eps = config.layer_norm_eps;

        // Stem: 4×4 conv stride 4
        let stem_conv = Conv2d::new(config.in_channels, config.dims[0], 4, 4, 4, 4, 0, 0, 1, true);
        let stem_norm = LayerNorm::new(config.dims[0], eps);

        let mut stages = Vec::with_capacity(4);
        let mut downsamples = Vec::with_capacity(4);

        for s in 0..4 {
            // Downsample between stages (except first)
            if s > 0 {
                downsamples.push(Some(ConvNextDownsample::new(
                    config.dims[s - 1], config.dims[s], eps,
                )));
            } else {
                downsamples.push(None);
            }

            let blocks = (0..config.depths[s])
                .map(|_| ConvNextBlock::new(config.dims[s], config.is_v2, config.layer_scale_init, eps))
                .collect();
            stages.push(blocks);
        }

        let final_norm = LayerNorm::new(config.dims[3], eps);
        let head = LinearBias::new(config.dims[3], config.num_classes);

        Self {
            config,
            stem_conv,
            stem_norm,
            stages,
            downsamples,
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

    /// Forward pass to extract pooled features.
    pub fn forward_features(&self, image: &[f32]) -> Vec<f32> {
        let image_size = 224; // Assume standard size
        let (stem_h, stem_w) = self.stem_conv.output_dims(image_size, image_size);
        let mut features = vec![0.0f32; self.config.dims[0] * stem_h * stem_w];
        self.stem_conv.forward(image, image_size, image_size, &mut features);

        // Stem norm (channel-wise)
        self.stem_norm.forward_2d(&mut features, stem_h, stem_w);

        let mut h = stem_h;
        let mut w = stem_w;
        let mut dim = self.config.dims[0];

        for s in 0..4 {
            // Downsample
            if let Some(ref ds) = self.downsamples[s] {
                let (new_features, new_h, new_w) = ds.forward(&features, dim, h, w);
                features = new_features;
                h = new_h;
                w = new_w;
                dim = self.config.dims[s];
            }

            // Stage blocks
            for block in &self.stages[s] {
                block.forward(&mut features, dim, h, w);
            }
        }

        // Global average pool: [C, H, W] → [C]
        let spatial = h * w;
        let mut pooled = vec![0.0f32; dim];
        for c in 0..dim {
            let mut sum = 0.0f32;
            for p in 0..spatial {
                sum += features[c * spatial + p];
            }
            pooled[c] = sum / spatial as f32;
        }

        self.final_norm.forward(&mut pooled);
        pooled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConvNextConfig;

    fn tiny_test_config() -> ConvNextConfig {
        ConvNextConfig {
            in_channels: 3,
            dims: [8, 16, 32, 64],
            depths: [1, 1, 1, 1],
            num_classes: 10,
            layer_scale_init: 1e-6,
            layer_norm_eps: 1e-5,
            is_v2: false,
        }
    }

    #[test]
    fn test_convnext_block() {
        let block = ConvNextBlock::new(8, false, 1e-6, 1e-5);
        let h = 7;
        let w = 7;
        let mut features = vec![0.1f32; 8 * h * w];
        block.forward(&mut features, 8, h, w);
        for &v in &features {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_convnext_forward() {
        let config = tiny_test_config();
        let model = ConvNextModel::new(config);
        let image = vec![0.1f32; 3 * 224 * 224];
        let logits = model.forward(&image);
        assert_eq!(logits.len(), 10);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }
}
