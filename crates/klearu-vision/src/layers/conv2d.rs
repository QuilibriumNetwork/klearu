use klearu_accel::simd::dense_dot_dense_simd;

/// 2D convolution supporting regular and depthwise modes.
///
/// Weight layout: `[out_channels, in_channels/groups, kernel_h, kernel_w]` flattened.
/// Input/output layout: channel-first `[C, H, W]`.
pub struct Conv2d {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub groups: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        groups: usize,
        bias: bool,
    ) -> Self {
        let channels_per_group = in_channels / groups;
        let weight_len = out_channels * channels_per_group * kernel_h * kernel_w;
        Self {
            weight: vec![0.0; weight_len],
            bias: if bias { Some(vec![0.0; out_channels]) } else { None },
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            groups,
        }
    }

    /// Compute output spatial dimensions.
    pub fn output_dims(&self, in_h: usize, in_w: usize) -> (usize, usize) {
        let padded_h = in_h + 2 * self.padding_h;
        let padded_w = in_w + 2 * self.padding_w;
        assert!(
            padded_h >= self.kernel_h && padded_w >= self.kernel_w,
            "Input too small for conv: padded ({padded_h}, {padded_w}) < kernel ({}, {})",
            self.kernel_h, self.kernel_w,
        );
        let out_h = (padded_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (padded_w - self.kernel_w) / self.stride_w + 1;
        (out_h, out_w)
    }

    /// Forward pass. Input: `[in_channels, in_h, in_w]`. Output: `[out_channels, out_h, out_w]`.
    /// Returns `(out_h, out_w)`.
    pub fn forward(&self, input: &[f32], in_h: usize, in_w: usize, output: &mut [f32]) -> (usize, usize) {
        let (out_h, out_w) = self.output_dims(in_h, in_w);
        let out_spatial = out_h * out_w;

        debug_assert_eq!(input.len(), self.in_channels * in_h * in_w);
        debug_assert!(output.len() >= self.out_channels * out_spatial);

        let channels_per_group = self.in_channels / self.groups;
        let out_channels_per_group = self.out_channels / self.groups;
        let kernel_size = channels_per_group * self.kernel_h * self.kernel_w;

        if self.groups == self.out_channels && self.groups == self.in_channels {
            // Depthwise convolution: each channel independently
            self.forward_depthwise(input, in_h, in_w, out_h, out_w, output);
        } else if kernel_size >= 32 {
            // Regular convolution with SIMD (im2col-style per pixel)
            self.forward_regular_simd(
                input, in_h, in_w, out_h, out_w,
                channels_per_group, out_channels_per_group, kernel_size,
                output,
            );
        } else {
            // Regular convolution, scalar
            self.forward_regular_scalar(
                input, in_h, in_w, out_h, out_w,
                channels_per_group, out_channels_per_group, kernel_size,
                output,
            );
        }

        // Add bias
        if let Some(ref bias) = self.bias {
            for oc in 0..self.out_channels {
                let b = bias[oc];
                let base = oc * out_spatial;
                for i in 0..out_spatial {
                    output[base + i] += b;
                }
            }
        }

        (out_h, out_w)
    }

    fn forward_depthwise(
        &self,
        input: &[f32],
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        output: &mut [f32],
    ) {
        let out_spatial = out_h * out_w;
        // weight layout for depthwise: [out_channels, 1, kH, kW]
        let kernel_elems = self.kernel_h * self.kernel_w;

        for c in 0..self.out_channels {
            let w_base = c * kernel_elems;
            let in_base = c * in_h * in_w;
            let out_base = c * out_spatial;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for kh in 0..self.kernel_h {
                        let ih = oh * self.stride_h + kh;
                        let ih = ih as isize - self.padding_h as isize;
                        if ih < 0 || ih as usize >= in_h {
                            continue;
                        }
                        for kw in 0..self.kernel_w {
                            let iw = ow * self.stride_w + kw;
                            let iw = iw as isize - self.padding_w as isize;
                            if iw < 0 || iw as usize >= in_w {
                                continue;
                            }
                            let w = self.weight[w_base + kh * self.kernel_w + kw];
                            let x = input[in_base + ih as usize * in_w + iw as usize];
                            sum += w * x;
                        }
                    }
                    output[out_base + oh * out_w + ow] = sum;
                }
            }
        }
    }

    fn forward_regular_simd(
        &self,
        input: &[f32],
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        channels_per_group: usize,
        out_channels_per_group: usize,
        kernel_size: usize,
        output: &mut [f32],
    ) {
        let out_spatial = out_h * out_w;
        let mut patch = vec![0.0f32; kernel_size];

        for g in 0..self.groups {
            let in_c_start = g * channels_per_group;
            let out_c_start = g * out_channels_per_group;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    // Extract patch (im2col for one pixel)
                    self.extract_patch(input, in_h, in_w, in_c_start, channels_per_group, oh, ow, &mut patch);

                    // Compute dot product with each output filter
                    for oc_local in 0..out_channels_per_group {
                        let oc = out_c_start + oc_local;
                        let w_base = oc * kernel_size;
                        let w_slice = &self.weight[w_base..w_base + kernel_size];
                        let val = dense_dot_dense_simd(w_slice, &patch);
                        output[oc * out_spatial + oh * out_w + ow] = val;
                    }
                }
            }
        }
    }

    fn forward_regular_scalar(
        &self,
        input: &[f32],
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        channels_per_group: usize,
        out_channels_per_group: usize,
        kernel_size: usize,
        output: &mut [f32],
    ) {
        let out_spatial = out_h * out_w;

        for g in 0..self.groups {
            let in_c_start = g * channels_per_group;
            let out_c_start = g * out_channels_per_group;

            for oc_local in 0..out_channels_per_group {
                let oc = out_c_start + oc_local;
                let w_base = oc * kernel_size;

                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let mut w_idx = 0;
                        for ic in 0..channels_per_group {
                            let c = in_c_start + ic;
                            for kh in 0..self.kernel_h {
                                let ih = (oh * self.stride_h + kh) as isize - self.padding_h as isize;
                                for kw in 0..self.kernel_w {
                                    let iw = (ow * self.stride_w + kw) as isize - self.padding_w as isize;
                                    if ih >= 0 && (ih as usize) < in_h && iw >= 0 && (iw as usize) < in_w {
                                        let x = input[c * in_h * in_w + ih as usize * in_w + iw as usize];
                                        sum += self.weight[w_base + w_idx] * x;
                                    }
                                    w_idx += 1;
                                }
                            }
                        }
                        output[oc * out_spatial + oh * out_w + ow] = sum;
                    }
                }
            }
        }
    }

    fn extract_patch(
        &self,
        input: &[f32],
        in_h: usize,
        in_w: usize,
        in_c_start: usize,
        channels_per_group: usize,
        oh: usize,
        ow: usize,
        patch: &mut [f32],
    ) {
        let mut idx = 0;
        for ic in 0..channels_per_group {
            let c = in_c_start + ic;
            for kh in 0..self.kernel_h {
                let ih = (oh * self.stride_h + kh) as isize - self.padding_h as isize;
                for kw in 0..self.kernel_w {
                    let iw = (ow * self.stride_w + kw) as isize - self.padding_w as isize;
                    if ih >= 0 && (ih as usize) < in_h && iw >= 0 && (iw as usize) < in_w {
                        patch[idx] = input[c * in_h * in_w + ih as usize * in_w + iw as usize];
                    } else {
                        patch[idx] = 0.0;
                    }
                    idx += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_output_dims() {
        // Stem conv: 7x7, stride=4, pad=3
        let conv = Conv2d::new(3, 96, 7, 7, 4, 4, 3, 3, 1, false);
        assert_eq!(conv.output_dims(224, 224), (56, 56));

        // Downsample: 2x2, stride=2, no padding
        let conv = Conv2d::new(96, 192, 2, 2, 2, 2, 0, 0, 1, false);
        assert_eq!(conv.output_dims(56, 56), (28, 28));

        // CPE: 3x3, stride=1, pad=1 (same)
        let conv = Conv2d::new(96, 96, 3, 3, 1, 1, 1, 1, 96, false);
        assert_eq!(conv.output_dims(56, 56), (56, 56));
    }

    #[test]
    fn test_conv2d_identity_1x1() {
        // 1x1 conv with identity weight should pass through each channel
        let mut conv = Conv2d::new(2, 2, 1, 1, 1, 1, 0, 0, 1, false);
        // w[0] = [1, 0], w[1] = [0, 1]
        conv.weight = vec![1.0, 0.0, 0.0, 1.0];

        let input = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0: 2x2
            5.0, 6.0, 7.0, 8.0, // channel 1: 2x2
        ];
        let mut output = vec![0.0; 8];
        let (oh, ow) = conv.forward(&input, 2, 2, &mut output);
        assert_eq!((oh, ow), (2, 2));
        assert_eq!(&output[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&output[4..8], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_conv2d_depthwise_3x3() {
        // Depthwise 3x3, stride=1, pad=1 (same dims)
        let mut conv = Conv2d::new(2, 2, 3, 3, 1, 1, 1, 1, 2, false);
        // All-ones kernel for channel 0, all-zeros for channel 1
        conv.weight = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // ch 0
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // ch 1
        ];

        // Input: 2 channels, 3x3 each, all ones
        let input = vec![1.0; 2 * 3 * 3];
        let mut output = vec![0.0; 2 * 3 * 3];
        let (oh, ow) = conv.forward(&input, 3, 3, &mut output);
        assert_eq!((oh, ow), (3, 3));

        // Center pixel of channel 0 sees 9 ones
        assert!((output[4] - 9.0).abs() < 1e-5, "center={}", output[4]);
        // Corner of channel 0 sees 4 ones (3x3 with pad=1)
        assert!((output[0] - 4.0).abs() < 1e-5, "corner={}", output[0]);
        // Channel 1 should be all zeros
        for i in 9..18 {
            assert!((output[i]).abs() < 1e-5, "ch1[{}]={}", i - 9, output[i]);
        }
    }

    #[test]
    fn test_conv2d_with_bias() {
        let mut conv = Conv2d::new(1, 1, 1, 1, 1, 1, 0, 0, 1, true);
        conv.weight = vec![2.0];
        conv.bias = Some(vec![3.0]);

        let input = vec![5.0];
        let mut output = vec![0.0; 1];
        conv.forward(&input, 1, 1, &mut output);
        assert!((output[0] - 13.0).abs() < 1e-5); // 2*5 + 3 = 13
    }

    #[test]
    fn test_conv2d_stride2() {
        // 2x2 conv, stride=2, no padding: downsamples by 2
        let mut conv = Conv2d::new(1, 1, 2, 2, 2, 2, 0, 0, 1, false);
        conv.weight = vec![1.0, 1.0, 1.0, 1.0]; // sum kernel

        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut output = vec![0.0; 4];
        let (oh, ow) = conv.forward(&input, 4, 4, &mut output);
        assert_eq!((oh, ow), (2, 2));
        // Top-left 2x2: 1+2+5+6=14
        assert!((output[0] - 14.0).abs() < 1e-5);
        // Top-right 2x2: 3+4+7+8=22
        assert!((output[1] - 22.0).abs() < 1e-5);
    }
}
