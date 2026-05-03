//! Dense linear layer (fully connected). y = x · Wᵀ + b.
//!
//! Weight layout: row-major [out_features, in_features], matching
//! HuggingFace / PyTorch convention. The forward computes one output
//! row at a time as a dot product across the input.

pub struct Linear {
    pub weight: Vec<f32>, // [out, in]
    pub bias: Option<Vec<f32>>, // [out] or None
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, has_bias: bool) -> Self {
        Self {
            weight: vec![0.0; in_features * out_features],
            bias: if has_bias { Some(vec![0.0; out_features]) } else { None },
            in_features,
            out_features,
        }
    }

    /// y[..out_features] = x · Wᵀ + b for a single sample.
    pub fn forward(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_features);
        debug_assert_eq!(y.len(), self.out_features);
        for o in 0..self.out_features {
            let mut s = match &self.bias {
                Some(b) => b[o],
                None => 0.0,
            };
            let row = &self.weight[o * self.in_features..(o + 1) * self.in_features];
            for i in 0..self.in_features {
                s += row[i] * x[i];
            }
            y[o] = s;
        }
    }

    /// Apply to a batch: x is [n × in], y is [n × out].
    /// Uses sgemm: y = x · Wᵀ (then add bias broadcast per-row).
    pub fn forward_batch(&self, x: &[f32], y: &mut [f32]) {
        let n = x.len() / self.in_features;
        debug_assert_eq!(y.len(), n * self.out_features);
        // y = x · Wᵀ where x is [n × in] and W is [out × in]; output is [n × out].
        crate::blas::sgemm_a_btrans(
            n, self.out_features, self.in_features,
            x, &self.weight, y,
        );
        // Bias broadcast — parallelise across rows. For SD attention layers
        // this is 8192 rows × 320-1280 features per call; sequential was a
        // measurable cost in the per-step budget.
        if let Some(b) = &self.bias {
            use rayon::prelude::*;
            y.par_chunks_mut(self.out_features).for_each(|row| {
                for (rv, bv) in row.iter_mut().zip(b.iter()) { *rv += bv; }
            });
        }
    }

    /// GPU-resident batched forward. Input is an fp16 GpuTensor `[n × in]`;
    /// output is a freshly-allocated fp16 GpuTensor `[n × out]`. Uses MPS
    /// fp16 sgemm with the cached f16 weight buffer; bias is added via a
    /// strided fp16 kernel. Keeps everything on GPU — no CPU round trip.
    #[cfg(feature = "metal")]
    pub fn forward_gpu(&self, x: &crate::metal_backend::GpuTensor)
        -> crate::metal_backend::GpuTensor
    {
        use crate::metal_backend::*;
        debug_assert_eq!(x.dtype, GpuDtype::F16);
        let n = x.elements() / self.in_features;
        let mut out = GpuTensor::new_f16(vec![n, self.out_features]);

        // Weight goes through the f16 weight cache — converted once, reused
        // across all timesteps.
        let w_buf = weight_f16_buffer(&self.weight);
        sgemm_f16_a_btrans_buf(&out.buffer, &x.buffer, &w_buf,
                               n, self.out_features, self.in_features);
        if let Some(bias) = &self.bias {
            let b_buf = weight_f16_buffer(bias);
            // Linear bias: each row gets bias added per column → period=1.
            bias_add_f16_gpu(&mut out, &b_buf, 1, self.out_features);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_identity() {
        let mut l = Linear::new(3, 3, false);
        // Identity weight: I_3
        l.weight[0*3+0] = 1.0;
        l.weight[1*3+1] = 1.0;
        l.weight[2*3+2] = 1.0;
        let x = vec![5.0, 6.0, 7.0];
        let mut y = vec![0.0; 3];
        l.forward(&x, &mut y);
        assert_eq!(y, x);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn linear_forward_gpu_matches_cpu() {
        // Random-ish but deterministic Linear with bias.
        let in_f = 32; let out_f = 16;
        let mut l = Linear::new(in_f, out_f, true);
        for (i, w) in l.weight.iter_mut().enumerate() {
            *w = ((i * 31) % 13) as f32 / 13.0 - 0.5;
        }
        if let Some(b) = &mut l.bias {
            for (i, v) in b.iter_mut().enumerate() {
                *v = (i as f32 * 0.1) - 0.5;
            }
        }

        let n = 8;
        let x: Vec<f32> = (0..n*in_f).map(|i| ((i * 7) % 11) as f32 / 11.0 - 0.5).collect();

        // CPU reference.
        let mut y_cpu = vec![0.0f32; n * out_f];
        l.forward_batch(&x, &mut y_cpu);

        // GPU-resident path.
        let x_gpu = crate::metal_backend::GpuTensor::upload_f32_as_f16(vec![n, in_f], &x);
        let y_gpu_t = l.forward_gpu(&x_gpu);
        let y_gpu = y_gpu_t.download_to_f32();

        let mut max_diff = 0.0f32;
        for (g, c) in y_gpu.iter().zip(y_cpu.iter()) {
            let d = (g - c).abs();
            if d > max_diff { max_diff = d; }
        }
        // fp16 quantization at this scale gives ~0.05 absolute error in worst case.
        assert!(max_diff < 0.1, "Linear::forward_gpu diverges from CPU: max_diff={max_diff}");
    }

    #[test]
    fn linear_with_bias() {
        let mut l = Linear::new(2, 2, true);
        // [[1, 2], [3, 4]] with bias [0.5, 0.5]
        l.weight = vec![1.0, 2.0, 3.0, 4.0];
        l.bias = Some(vec![0.5, 0.5]);
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];
        l.forward(&x, &mut y);
        // y[0] = 1*1 + 2*1 + 0.5 = 3.5
        // y[1] = 3*1 + 4*1 + 0.5 = 7.5
        assert!((y[0] - 3.5).abs() < 1e-6);
        assert!((y[1] - 7.5).abs() < 1e-6);
    }
}
