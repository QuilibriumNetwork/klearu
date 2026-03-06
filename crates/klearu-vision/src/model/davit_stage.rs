use crate::layers::{Conv2d, LayerNorm};
use super::davit_block::{SpatialBlock, ChannelBlock};

/// Downsample module: LayerNorm2d → Conv2d(stride=2).
pub struct Downsample {
    pub norm: LayerNorm,
    pub conv: Conv2d,
}

impl Downsample {
    pub fn new(in_dim: usize, out_dim: usize, eps: f32) -> Self {
        Self {
            norm: LayerNorm::new(in_dim, eps),
            conv: Conv2d::new(in_dim, out_dim, 2, 2, 2, 2, 0, 0, 1, true),
        }
    }

    /// Forward: LayerNorm2d → Conv2d (halves spatial dims, changes channels).
    ///
    /// Input: `[in_dim, H, W]`. Returns `(output, H/2, W/2)` with `output = [out_dim, H/2, W/2]`.
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let in_dim = self.norm.dim();
        debug_assert_eq!(input.len(), in_dim * h * w);

        let mut normed = input.to_vec();
        self.norm.forward_2d(&mut normed, h, w);

        let (out_h, out_w) = self.conv.output_dims(h, w);
        let out_dim = self.conv.out_channels;
        let mut output = vec![0.0f32; out_dim * out_h * out_w];
        self.conv.forward(&normed, h, w, &mut output);

        (output, out_h, out_w)
    }
}

/// A DaViT stage: optional downsample + N dual-attention blocks.
///
/// Each "block" consists of one SpatialBlock followed by one ChannelBlock.
pub struct DaViTStage {
    pub downsample: Option<Downsample>,
    pub blocks: Vec<(SpatialBlock, ChannelBlock)>,
}

impl DaViTStage {
    pub fn new(
        has_downsample: bool,
        prev_dim: usize,
        dim: usize,
        num_heads: usize,
        depth: usize,
        mlp_hidden: usize,
        window_size: usize,
        eps: f32,
    ) -> Self {
        let downsample = if has_downsample {
            Some(Downsample::new(prev_dim, dim, eps))
        } else {
            None
        };

        let blocks = (0..depth)
            .map(|_| {
                let spatial = SpatialBlock::new(dim, num_heads, mlp_hidden, window_size, eps);
                let channel = ChannelBlock::new(dim, num_heads, mlp_hidden, eps);
                (spatial, channel)
            })
            .collect();

        Self { downsample, blocks }
    }

    /// Forward pass.
    ///
    /// Input: `[C_prev, H, W]`. Returns `(output, out_h, out_w)`.
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let (mut x, cur_h, cur_w) = if let Some(ref ds) = self.downsample {
            ds.forward(input, h, w)
        } else {
            (input.to_vec(), h, w)
        };

        for (spatial, channel) in &self.blocks {
            spatial.forward(&mut x, cur_h, cur_w);
            channel.forward(&mut x, cur_h, cur_w);
        }

        (x, cur_h, cur_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample() {
        let ds = Downsample::new(8, 16, 1e-5);
        let input = vec![0.1f32; 8 * 14 * 14];
        let (output, h, w) = ds.forward(&input, 14, 14);
        assert_eq!(h, 7);
        assert_eq!(w, 7);
        assert_eq!(output.len(), 16 * 7 * 7);
    }

    #[test]
    fn test_stage_without_downsample() {
        let stage = DaViTStage::new(false, 8, 8, 2, 1, 16, 7, 1e-5);
        let input = vec![0.1f32; 8 * 7 * 7];
        let (output, h, w) = stage.forward(&input, 7, 7);
        assert_eq!(h, 7);
        assert_eq!(w, 7);
        assert_eq!(output.len(), 8 * 7 * 7);
    }

    #[test]
    fn test_stage_with_downsample() {
        let stage = DaViTStage::new(true, 8, 16, 2, 1, 32, 7, 1e-5);
        let input = vec![0.1f32; 8 * 14 * 14];
        let (output, h, w) = stage.forward(&input, 14, 14);
        assert_eq!(h, 7);
        assert_eq!(w, 7);
        assert_eq!(output.len(), 16 * 7 * 7);
        for &v in &output {
            assert!(v.is_finite());
        }
    }
}
