/// INT8 weight-only quantized linear layer (W8A32).
///
/// Weights are stored as per-channel symmetric INT8 with f32 scale factors.
/// Activations remain in f32. Dequantization happens on the fly during matmul.
///
/// Quantization: `w_i8[row][col] = round(w_f32[row][col] / scale[row])`
/// where `scale[row] = max_abs(w_f32[row]) / 127`.
///
/// Forward: `output[row] = sum(w_i8[row][col] * input[col]) * scale[row] + bias[row]`

/// Per-channel symmetric INT8 quantized linear layer.
pub struct QuantizedLinear {
    weights_i8: Vec<i8>,
    scales: Vec<f32>,
    bias: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl QuantizedLinear {
    /// Quantize from f32 weights.
    ///
    /// `weights` is row-major `[out_features, in_features]`.
    pub fn from_f32(
        weights: &[f32],
        bias: &[f32],
        in_features: usize,
        out_features: usize,
    ) -> Self {
        assert_eq!(weights.len(), out_features * in_features);
        assert_eq!(bias.len(), out_features);

        let mut weights_i8 = vec![0i8; out_features * in_features];
        let mut scales = vec![0.0f32; out_features];

        for row in 0..out_features {
            let row_start = row * in_features;
            let row_slice = &weights[row_start..row_start + in_features];

            let max_abs = row_slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales[row] = scale;

            let inv_scale = 1.0 / scale;
            for (i, &w) in row_slice.iter().enumerate() {
                weights_i8[row_start + i] = (w * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
        }

        Self {
            weights_i8,
            scales,
            bias: bias.to_vec(),
            in_features,
            out_features,
        }
    }

    /// Quantize from a `LinearBias` layer.
    pub fn from_linear_bias(linear: &super::LinearBias) -> Self {
        let in_f = linear.in_features();
        let out_f = linear.out_features();
        let mut flat_weights = vec![0.0f32; out_f * in_f];
        for row in 0..out_f {
            let w = linear.weights.get_weights(row);
            flat_weights[row * in_f..(row + 1) * in_f].copy_from_slice(&w[..in_f]);
        }
        Self::from_f32(&flat_weights, &linear.bias, in_f, out_f)
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Forward pass: dequantize on the fly.
    ///
    /// `output[row] = sum(w_i8[row][col] * input[col]) * scale[row] + bias[row]`
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_features);
        debug_assert!(output.len() >= self.out_features);

        for row in 0..self.out_features {
            let row_start = row * self.in_features;
            let scale = self.scales[row];
            let mut acc = 0.0f32;

            // Process in chunks for better cache behavior
            let w_row = &self.weights_i8[row_start..row_start + self.in_features];
            for (w, x) in w_row.iter().zip(input.iter()) {
                acc += *w as f32 * *x;
            }

            output[row] = acc * scale + self.bias[row];
        }
    }

    /// Sparse forward: only compute selected output indices.
    pub fn forward_sparse(&self, input: &[f32], indices: &[usize], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_features);

        for (out_idx, &neuron_idx) in indices.iter().enumerate() {
            let row_start = neuron_idx * self.in_features;
            let scale = self.scales[neuron_idx];
            let mut acc = 0.0f32;

            let w_row = &self.weights_i8[row_start..row_start + self.in_features];
            for (w, x) in w_row.iter().zip(input.iter()) {
                acc += *w as f32 * *x;
            }

            output[out_idx] = acc * scale + self.bias[neuron_idx];
        }
    }

    /// Memory usage in bytes (INT8 weights + f32 scales + f32 bias).
    pub fn memory_bytes(&self) -> usize {
        self.weights_i8.len() + self.scales.len() * 4 + self.bias.len() * 4
    }

    /// Compression ratio compared to f32 weights.
    pub fn compression_ratio(&self) -> f32 {
        let f32_bytes = (self.in_features * self.out_features + self.out_features) * 4;
        f32_bytes as f32 / self.memory_bytes() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_identity() {
        // Identity-like: w = [[1,0], [0,1]], bias = [0,0]
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![0.0, 0.0];
        let ql = QuantizedLinear::from_f32(&weights, &bias, 2, 2);

        let input = vec![3.0, 7.0];
        let mut output = vec![0.0; 2];
        ql.forward(&input, &mut output);

        // INT8 quantization of identity should be exact (1.0/scale * scale = 1.0)
        assert!((output[0] - 3.0).abs() < 0.1, "output[0]={}", output[0]);
        assert!((output[1] - 7.0).abs() < 0.1, "output[1]={}", output[1]);
    }

    #[test]
    fn test_quantized_with_bias() {
        let weights = vec![2.0, 0.0, 0.0, 3.0];
        let bias = vec![1.0, -1.0];
        let ql = QuantizedLinear::from_f32(&weights, &bias, 2, 2);

        let input = vec![5.0, 4.0];
        let mut output = vec![0.0; 2];
        ql.forward(&input, &mut output);

        // Expected: [2*5+1, 3*4-1] = [11, 11]
        assert!((output[0] - 11.0).abs() < 0.5, "output[0]={}", output[0]);
        assert!((output[1] - 11.0).abs() < 0.5, "output[1]={}", output[1]);
    }

    #[test]
    fn test_quantized_accuracy() {
        // Random-ish weights, check that quantization error is small
        let in_f = 64;
        let out_f = 16;
        let weights: Vec<f32> = (0..in_f * out_f)
            .map(|i| (i as f32 * 0.37).sin() * 0.5)
            .collect();
        let bias = vec![0.0f32; out_f];
        let ql = QuantizedLinear::from_f32(&weights, &bias, in_f, out_f);

        let input: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.1).cos()).collect();

        // Compute f32 reference
        let mut ref_output = vec![0.0f32; out_f];
        for row in 0..out_f {
            let mut sum = 0.0f32;
            for col in 0..in_f {
                sum += weights[row * in_f + col] * input[col];
            }
            ref_output[row] = sum + bias[row];
        }

        let mut q_output = vec![0.0f32; out_f];
        ql.forward(&input, &mut q_output);

        // Error should be small relative to output magnitude
        let max_diff: f32 = ref_output.iter().zip(q_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_val: f32 = ref_output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        // Relative error should be < 1%
        assert!(
            max_diff / max_val.max(1e-6) < 0.02,
            "max_diff={max_diff}, max_val={max_val}, relative={}",
            max_diff / max_val.max(1e-6)
        );
    }

    #[test]
    fn test_quantized_sparse_forward() {
        let weights = vec![
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
            1.0, 1.0, 1.0,
        ];
        let bias = vec![0.0; 4];
        let ql = QuantizedLinear::from_f32(&weights, &bias, 3, 4);

        let input = vec![10.0, 20.0, 30.0];
        let indices = vec![1, 3]; // neuron 1 and 3
        let mut output = vec![0.0; 2];
        ql.forward_sparse(&input, &indices, &mut output);

        // neuron 1: 2*20 = 40
        assert!((output[0] - 40.0).abs() < 1.0, "output[0]={}", output[0]);
        // neuron 3: 10+20+30 = 60
        assert!((output[1] - 60.0).abs() < 1.0, "output[1]={}", output[1]);
    }

    #[test]
    fn test_compression_ratio() {
        let weights = vec![0.0f32; 1024 * 768];
        let bias = vec![0.0f32; 1024];
        let ql = QuantizedLinear::from_f32(&weights, &bias, 768, 1024);

        // INT8 weights = 1024*768 bytes, scales = 1024*4, bias = 1024*4
        // F32 weights = 1024*768*4, bias = 1024*4
        // Ratio ≈ 3.97x
        assert!(ql.compression_ratio() > 3.5);
    }

    #[test]
    fn test_from_linear_bias() {
        let mut lb = crate::layers::LinearBias::new(4, 2);
        lb.weights.set_weights(0, &[1.0, 2.0, 3.0, 4.0]);
        lb.weights.set_weights(1, &[5.0, 6.0, 7.0, 8.0]);
        lb.bias = vec![0.1, 0.2];

        let ql = QuantizedLinear::from_linear_bias(&lb);

        let input = vec![1.0, 1.0, 1.0, 1.0];
        let mut output_f32 = vec![0.0; 2];
        let mut output_i8 = vec![0.0; 2];
        lb.forward(&input, &mut output_f32);
        ql.forward(&input, &mut output_i8);

        for (a, b) in output_f32.iter().zip(output_i8.iter()) {
            assert!((a - b).abs() < 0.5, "f32={a}, i8={b}");
        }
    }
}
