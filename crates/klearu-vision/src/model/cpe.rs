use crate::layers::Conv2d;

/// Conditional Position Encoding via depthwise convolution.
///
/// Applies `x = x + depthwise_conv3x3(x)` as a residual position encoding.
pub struct ConvPosEnc {
    pub proj: Conv2d,
}

impl ConvPosEnc {
    pub fn new(dim: usize) -> Self {
        Self {
            proj: Conv2d::new(dim, dim, 3, 3, 1, 1, 1, 1, dim, true),
        }
    }

    /// Forward: residual add `x += conv(x)`.
    ///
    /// Input/output layout: `[C, H, W]` (channel-first, modified in-place).
    pub fn forward(&self, x: &mut [f32], h: usize, w: usize) {
        let c = self.proj.out_channels;
        debug_assert_eq!(x.len(), c * h * w);

        let mut conv_out = vec![0.0f32; x.len()];
        self.proj.forward(x, h, w, &mut conv_out);

        for (xv, cv) in x.iter_mut().zip(conv_out.iter()) {
            *xv += cv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpe_same_dims() {
        let cpe = ConvPosEnc::new(4);
        let mut x = vec![1.0f32; 4 * 7 * 7];
        cpe.forward(&mut x, 7, 7);
        // Should still be finite and non-zero (residual preserves)
        for &v in &x {
            assert!(v.is_finite());
        }
        // At least some values should differ from 1.0 (bias is 0, but kernel is 0 too for new)
        // With zero weights and zero bias, conv output is all 0, so x stays at 1.0
        assert!((x[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpe_preserves_shape() {
        let cpe = ConvPosEnc::new(8);
        let mut x = vec![0.5f32; 8 * 14 * 14];
        let orig_len = x.len();
        cpe.forward(&mut x, 14, 14);
        assert_eq!(x.len(), orig_len);
    }
}
