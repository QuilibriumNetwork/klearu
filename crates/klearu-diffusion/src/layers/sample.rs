//! Up/down sampling for the U-Net spatial pathway.
//!
//! - Upsample: nearest-neighbor 2× then a 3×3 conv (preserves channels).
//! - Downsample: 3×3 conv with stride=2 (halves H/W).

use crate::layers::Conv2d;

/// Upsample by 2x using nearest-neighbor, then apply a 3×3 conv.
pub struct Upsample {
    pub conv: Conv2d,
    pub channels: usize,
}

impl Upsample {
    pub fn new(channels: usize) -> Self {
        Self {
            conv: Conv2d::new(channels, channels, 3, 1, 1, true),
            channels,
        }
    }

    /// Forward. Input [N, C, H, W] → output [N, C, 2H, 2W].
    pub fn forward(&self, input: &[f32], n: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let c = self.channels;
        let h_up = h * 2;
        let w_up = w * 2;
        let mut up = vec![0.0f32; n * c * h_up * w_up];
        // Nearest-neighbor: out[..., 2h+i, 2w+j] = in[..., h, w] for i,j in {0,1}.
        for ni in 0..n {
            for ci in 0..c {
                let in_off = ni * c * h * w + ci * h * w;
                let out_off = ni * c * h_up * w_up + ci * h_up * w_up;
                for hi in 0..h {
                    for wi in 0..w {
                        let v = input[in_off + hi * w + wi];
                        let oh = 2 * hi;
                        let ow = 2 * wi;
                        up[out_off + (oh) * w_up + ow] = v;
                        up[out_off + (oh) * w_up + ow + 1] = v;
                        up[out_off + (oh + 1) * w_up + ow] = v;
                        up[out_off + (oh + 1) * w_up + ow + 1] = v;
                    }
                }
            }
        }
        let mut out = Vec::new();
        let (ho, wo) = self.conv.forward(&up, n, h_up, w_up, &mut out);
        (out, ho, wo)
    }

    /// GPU-resident forward: GPU nearest-neighbor 2× → GPU Conv2d.
    /// Input/output stay on GPU as fp16.
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        input: &crate::metal_backend::GpuTensor,
        n: usize, h: usize, w: usize,
    ) -> (crate::metal_backend::GpuTensor, usize, usize) {
        use crate::metal_backend::*;
        let up = nearest_upsample_2x_f16_gpu(input, n, self.channels, h, w);
        self.conv.forward_gpu(&up, n, h * 2, w * 2)
    }
}

/// Downsample by 2x via 3×3 conv with stride 2, padding 1.
pub struct Downsample {
    pub conv: Conv2d,
    pub channels: usize,
}

impl Downsample {
    pub fn new(channels: usize) -> Self {
        Self {
            conv: Conv2d::new(channels, channels, 3, 2, 1, true),
            channels,
        }
    }

    pub fn forward(&self, input: &[f32], n: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let mut out = Vec::new();
        let (ho, wo) = self.conv.forward(input, n, h, w, &mut out);
        (out, ho, wo)
    }

    /// GPU-resident forward: just a stride-2 GPU conv. Halves H/W.
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        input: &crate::metal_backend::GpuTensor,
        n: usize, h: usize, w: usize,
    ) -> (crate::metal_backend::GpuTensor, usize, usize) {
        self.conv.forward_gpu(input, n, h, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "metal")]
    #[test]
    fn upsample_forward_gpu_matches_cpu() {
        let mut u = Upsample::new(4);
        for (i, w) in u.conv.weight.iter_mut().enumerate() {
            *w = ((i * 11) % 17) as f32 / 17.0 - 0.5;
        }
        if let Some(b) = &mut u.conv.bias {
            for (i, v) in b.iter_mut().enumerate() { *v = (i as f32 * 0.05) - 0.1; }
        }

        let n = 1; let h = 4; let w = 4;
        let input: Vec<f32> = (0..n*4*h*w)
            .map(|i| ((i * 7) % 13) as f32 / 13.0 - 0.5).collect();

        let (cpu_out, h_cpu, w_cpu) = u.forward(&input, n, h, w);

        let in_gpu = crate::metal_backend::GpuTensor::upload_f32_as_f16(
            vec![n, 4, h, w], &input);
        let (gpu_t, h_gpu, w_gpu) = u.forward_gpu(&in_gpu, n, h, w);
        assert_eq!((h_cpu, w_cpu), (h_gpu, w_gpu));
        let gpu_out = gpu_t.download_to_f32();

        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 0.2, "Upsample::forward_gpu diverges: max_diff={max_diff}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn downsample_forward_gpu_matches_cpu() {
        let mut d = Downsample::new(4);
        for (i, w) in d.conv.weight.iter_mut().enumerate() {
            *w = ((i * 13) % 19) as f32 / 19.0 - 0.5;
        }
        if let Some(b) = &mut d.conv.bias {
            for (i, v) in b.iter_mut().enumerate() { *v = (i as f32 * 0.04) - 0.08; }
        }

        let n = 1; let h = 8; let w = 8;
        let input: Vec<f32> = (0..n*4*h*w)
            .map(|i| ((i * 5) % 11) as f32 / 11.0 - 0.5).collect();

        let (cpu_out, h_cpu, w_cpu) = d.forward(&input, n, h, w);

        let in_gpu = crate::metal_backend::GpuTensor::upload_f32_as_f16(
            vec![n, 4, h, w], &input);
        let (gpu_t, h_gpu, w_gpu) = d.forward_gpu(&in_gpu, n, h, w);
        assert_eq!((h_cpu, w_cpu), (h_gpu, w_gpu));
        let gpu_out = gpu_t.download_to_f32();

        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            let d = (g - c).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 0.2, "Downsample::forward_gpu diverges: max_diff={max_diff}");
    }

    #[test]
    fn upsample_doubles_dims() {
        let u = Upsample::new(2);
        let input = vec![0.0f32; 1 * 2 * 4 * 4];
        let (_, h, w) = u.forward(&input, 1, 4, 4);
        assert_eq!((h, w), (8, 8));
    }

    #[test]
    fn downsample_halves_dims() {
        let d = Downsample::new(2);
        let input = vec![0.0f32; 1 * 2 * 8 * 8];
        let (_, h, w) = d.forward(&input, 1, 8, 8);
        assert_eq!((h, w), (4, 4));
    }
}
