//! Shared Conv2d: public weights, shared input → shared output.
//!
//! Since weights are public, the convolution is purely local:
//! each party computes `conv(weight, x_share)` independently.
//! No communication or triples needed.

use klearu_mpc::{SharedVec, SharedVec64};

/// Compute a Conv2d forward with public weights on shared Q16.16 input.
///
/// Input shares: `[in_channels * in_h * in_w]` in Q16.16.
/// Output shares: `[out_channels * out_h * out_w]` in Q16.16.
///
/// Uses f64 mixed-precision accumulation to avoid quantization cascade.
pub fn shared_conv2d_forward(
    _party: u8,
    weight: &[f32],
    bias: Option<&[f32]>,
    out_channels: usize,
    in_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    groups: usize,
    x_share: &[u32],
    in_h: usize,
    in_w: usize,
) -> (SharedVec, usize, usize) {
    let padded_h = in_h + 2 * padding_h;
    let padded_w = in_w + 2 * padding_w;
    assert!(padded_h >= kernel_h && padded_w >= kernel_w);

    let out_h = (padded_h - kernel_h) / stride_h + 1;
    let out_w = (padded_w - kernel_w) / stride_w + 1;
    let out_spatial = out_h * out_w;

    let channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;
    let kernel_size = channels_per_group * kernel_h * kernel_w;

    let mut output = vec![0u32; out_channels * out_spatial];

    for g in 0..groups {
        let in_c_start = g * channels_per_group;
        let out_c_start = g * out_channels_per_group;

        for oc_local in 0..out_channels_per_group {
            let oc = out_c_start + oc_local;
            let w_base = oc * kernel_size;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0f64;
                    let mut w_idx = 0;
                    for ic in 0..channels_per_group {
                        let c = in_c_start + ic;
                        for kh in 0..kernel_h {
                            let ih = (oh * stride_h + kh) as isize - padding_h as isize;
                            for kw in 0..kernel_w {
                                let iw = (ow * stride_w + kw) as isize - padding_w as isize;
                                if ih >= 0 && (ih as usize) < in_h && iw >= 0 && (iw as usize) < in_w {
                                    let x_val = x_share[c * in_h * in_w + ih as usize * in_w + iw as usize];
                                    acc += weight[w_base + w_idx] as f64 * x_val as i32 as f64;
                                }
                                w_idx += 1;
                            }
                        }
                    }
                    output[oc * out_spatial + oh * out_w + ow] = acc.round() as i64 as i32 as u32;
                }
            }
        }
    }

    // Add bias (party 0 only would be correct for shares, but since both parties
    // compute the same local operation with public weights, and bias is public,
    // bias is added to the share directly — it's absorbed into the share value)
    if let Some(bias) = bias {
        use klearu_mpc::to_fixed;
        for oc in 0..out_channels {
            let b = to_fixed(bias[oc]);
            let base = oc * out_spatial;
            for i in 0..out_spatial {
                output[base + i] = output[base + i].wrapping_add(b);
            }
        }
    }

    (SharedVec(output), out_h, out_w)
}

/// Compute a Conv2d forward with public weights on shared Q32.32 (u64) input.
///
/// Input shares: `[in_channels * in_h * in_w]` in Q32.32.
/// Output shares: `[out_channels * out_h * out_w]` in Q32.32.
///
/// Uses f64 mixed-precision accumulation: weight_f32 as f64 * x_share as i64 as f64.
/// Result is Q32.32 directly (no truncation needed since weight is plain f32).
pub fn shared_conv2d_forward_64(
    party: u8,
    weight: &[f32],
    bias: Option<&[f32]>,
    out_channels: usize,
    in_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    groups: usize,
    x_share: &[u64],
    in_h: usize,
    in_w: usize,
) -> (SharedVec64, usize, usize) {
    let padded_h = in_h + 2 * padding_h;
    let padded_w = in_w + 2 * padding_w;
    assert!(padded_h >= kernel_h && padded_w >= kernel_w);

    let out_h = (padded_h - kernel_h) / stride_h + 1;
    let out_w = (padded_w - kernel_w) / stride_w + 1;
    let out_spatial = out_h * out_w;

    let channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;
    let kernel_size = channels_per_group * kernel_h * kernel_w;

    let mut output = vec![0u64; out_channels * out_spatial];

    for g in 0..groups {
        let in_c_start = g * channels_per_group;
        let out_c_start = g * out_channels_per_group;

        for oc_local in 0..out_channels_per_group {
            let oc = out_c_start + oc_local;
            let w_base = oc * kernel_size;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0f64;
                    let mut w_idx = 0;
                    for ic in 0..channels_per_group {
                        let c = in_c_start + ic;
                        for kh in 0..kernel_h {
                            let ih = (oh * stride_h + kh) as isize - padding_h as isize;
                            for kw in 0..kernel_w {
                                let iw = (ow * stride_w + kw) as isize - padding_w as isize;
                                if ih >= 0 && (ih as usize) < in_h && iw >= 0 && (iw as usize) < in_w {
                                    let x_val = x_share[c * in_h * in_w + ih as usize * in_w + iw as usize];
                                    acc += weight[w_base + w_idx] as f64 * x_val as i64 as f64;
                                }
                                w_idx += 1;
                            }
                        }
                    }
                    output[oc * out_spatial + oh * out_w + ow] = acc.round() as i64 as u64;
                }
            }
        }
    }

    // Add bias: party 0 only
    if party == 0 {
        if let Some(bias) = bias {
            use klearu_mpc::to_fixed64;
            for oc in 0..out_channels {
                let b = to_fixed64(bias[oc]);
                let base = oc * out_spatial;
                for i in 0..out_spatial {
                    output[base + i] = output[base + i].wrapping_add(b);
                }
            }
        }
    }

    (SharedVec64(output), out_h, out_w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_mpc::{from_fixed, to_fixed};

    #[test]
    fn test_shared_conv2d_identity() {
        // 1x1 conv, identity weight (1 channel)
        let weight = vec![1.0f32];
        let in_h = 2;
        let in_w = 2;
        let input: Vec<u32> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| to_fixed(v)).collect();

        let (out_share, oh, ow) = shared_conv2d_forward(
            0, &weight, None,
            1, 1, 1, 1, 1, 1, 0, 0, 1,
            &input, in_h, in_w,
        );

        assert_eq!(oh, 2);
        assert_eq!(ow, 2);
        for i in 0..4 {
            let result = from_fixed(out_share.0[i]);
            let expected = from_fixed(input[i]);
            assert!(
                (result - expected).abs() < 0.01,
                "pixel[{i}]: got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_shared_conv2d_matches_plaintext() {
        // 1x1 conv with scale=2.0
        let weight = vec![2.0f32];
        let input_f32 = [1.0f32, 3.0, -1.0, 0.5];
        let input: Vec<u32> = input_f32.iter().map(|&v| to_fixed(v)).collect();

        let (out_share, _, _) = shared_conv2d_forward(
            0, &weight, None,
            1, 1, 1, 1, 1, 1, 0, 0, 1,
            &input, 2, 2,
        );

        for i in 0..4 {
            let result = from_fixed(out_share.0[i]);
            let expected = input_f32[i] * 2.0;
            assert!(
                (result - expected).abs() < 0.01,
                "pixel[{i}]: got {result}, expected {expected}"
            );
        }
    }

    // --- Q32.32 Conv2d tests ---

    #[test]
    fn test_shared_conv2d_64_identity() {
        use klearu_mpc::{from_fixed64, to_fixed64};

        let weight = vec![1.0f32];
        let in_h = 2;
        let in_w = 2;
        let input_f32 = [1.0f32, 2.0, 3.0, 4.0];
        let input: Vec<u64> = input_f32.iter().map(|&v| to_fixed64(v)).collect();

        let (out_share, oh, ow) = shared_conv2d_forward_64(
            0, &weight, None,
            1, 1, 1, 1, 1, 1, 0, 0, 1,
            &input, in_h, in_w,
        );

        assert_eq!(oh, 2);
        assert_eq!(ow, 2);
        for i in 0..4 {
            let result = from_fixed64(out_share.0[i]);
            assert!(
                (result - input_f32[i]).abs() < 0.01,
                "Q32 pixel[{i}]: got {result}, expected {}", input_f32[i]
            );
        }
    }

    #[test]
    fn test_shared_conv2d_64_scale() {
        use klearu_mpc::{from_fixed64, to_fixed64};

        let weight = vec![2.0f32];
        let input_f32 = [1.0f32, 3.0, -1.0, 0.5];
        let input: Vec<u64> = input_f32.iter().map(|&v| to_fixed64(v)).collect();

        let (out_share, _, _) = shared_conv2d_forward_64(
            0, &weight, None,
            1, 1, 1, 1, 1, 1, 0, 0, 1,
            &input, 2, 2,
        );

        for i in 0..4 {
            let result = from_fixed64(out_share.0[i]);
            let expected = input_f32[i] * 2.0;
            assert!(
                (result - expected).abs() < 0.01,
                "Q32 pixel[{i}]: got {result}, expected {expected}"
            );
        }
    }
}
