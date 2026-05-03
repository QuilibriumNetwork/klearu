//! 2D Convolution. NCHW row-major.
//!
//! Direct implementation (im2col + matmul would be faster but we want
//! correctness first; SD's 3×3 stride-1 padding-1 is the dominant case
//! and direct conv keeps memory low).

use std::cell::RefCell;

// Thread-local scratch buffer for im2col. Reused across all Conv2d::forward
// calls on a thread to eliminate ~3500 large allocations per SD generation.
// (Per-call sizes can reach ~200 MB for the largest convs in SDXL UNet.)
thread_local! {
    static IM2COL_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

pub struct Conv2d {
    pub weight: Vec<f32>,     // [out_c, in_c, kh, kw]
    pub bias: Option<Vec<f32>>, // [out_c]
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        has_bias: bool,
    ) -> Self {
        Self {
            weight: vec![0.0; out_channels * in_channels * kernel_size * kernel_size],
            bias: if has_bias { Some(vec![0.0; out_channels]) } else { None },
            in_channels,
            out_channels,
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride,
            padding,
        }
    }

    /// Compute output spatial dimensions.
    pub fn output_dims(&self, h_in: usize, w_in: usize) -> (usize, usize) {
        let h_out = (h_in + 2 * self.padding).saturating_sub(self.kernel_h) / self.stride + 1;
        let w_out = (w_in + 2 * self.padding).saturating_sub(self.kernel_w) / self.stride + 1;
        (h_out, w_out)
    }

    /// Forward via im2col + SGEMM. Per-batch loop is sequential because
    /// SGEMM internally parallelises with rayon-equivalent threads anyway.
    /// For batch sizes >1 we'd parallelise here, but SD inference is batch=1
    /// so this isn't the bottleneck.
    ///
    /// Reshape the input patches into a [C_in·K_h·K_w, H_out·W_out] matrix
    /// per batch, then output[oc, *] = weight[oc, :] · im2col, computed as
    /// a single matmul:  W[C_out, K] · X[K, H_out·W_out] = Y[C_out, H_out·W_out]
    /// with K = C_in · K_h · K_w. This is ~50-100× faster than the direct
    /// six-loop convolution for typical SD shapes.
    pub fn forward(
        &self,
        input: &[f32],
        n: usize,
        h_in: usize,
        w_in: usize,
        output: &mut Vec<f32>,
    ) -> (usize, usize) {
        let (h_out, w_out) = self.output_dims(h_in, w_in);
        output.clear();
        output.resize(n * self.out_channels * h_out * w_out, 0.0);

        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let kk = self.in_channels * kh * kw;
        let hw_out = h_out * w_out;
        let pad = self.padding as isize;
        let stride = self.stride;

        // Per-batch im2col + sgemm. The im2col matrix is built into a
        // thread-local scratch buffer that's reused across all Conv2d
        // forward calls (resized as needed; never freed).
        IM2COL_SCRATCH.with_borrow_mut(|col| {
            let needed = kk * hw_out;
            col.clear();
            col.resize(needed, 0.0);

            for ni in 0..n {
                if ni > 0 {
                    // Reset for next batch (we've already cleared on first pass).
                    for v in col.iter_mut() { *v = 0.0; }
                }
                let in_n_offset = ni * self.in_channels * h_in * w_in;
                for ic in 0..self.in_channels {
                    let in_c_offset = in_n_offset + ic * h_in * w_in;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let row = ic * kh * kw + ki * kw + kj;
                            let row_offset = row * hw_out;
                            for oh in 0..h_out {
                                let ih = oh as isize * stride as isize - pad + ki as isize;
                                if ih < 0 || ih >= h_in as isize {
                                    continue;
                                }
                                let in_h_offset = in_c_offset + (ih as usize) * w_in;
                                let dst_h_offset = row_offset + oh * w_out;
                                for ow in 0..w_out {
                                    let iw = ow as isize * stride as isize - pad + kj as isize;
                                    if iw < 0 || iw >= w_in as isize { continue; }
                                    col[dst_h_offset + ow] = input[in_h_offset + iw as usize];
                                }
                            }
                        }
                    }
                }

                let out_n_offset = ni * self.out_channels * hw_out;

                // For multi-threaded sgemm backends (Accelerate / MPS), chunking
                // hurts: 8 sgemm calls × ~250μs MPS dispatch overhead × redundant
                // reads of the full 47MB im2col matrix per chunk = 8× memory
                // bandwidth waste. Single sgemm lets the backend parallelise
                // internally and reads `col` once.
                //
                // For the unaccelerated matrixmultiply path (default features),
                // chunking still gives meaningful multi-core parallelism since
                // matrixmultiply is single-threaded.
                let oc_total = self.out_channels;
                let weight = &self.weight;
                let col_ref: &[f32] = col;

                #[cfg(any(feature = "accelerate", feature = "metal"))]
                {
                    crate::blas::sgemm_row_major(
                        oc_total, hw_out, kk,
                        1.0,
                        weight, kk,
                        col_ref, hw_out,
                        0.0,
                        &mut output[out_n_offset..out_n_offset + oc_total * hw_out], hw_out,
                    );
                }

                #[cfg(not(any(feature = "accelerate", feature = "metal")))]
                {
                    let target_chunks = 8usize;
                    let chunk_size = ((oc_total + target_chunks - 1) / target_chunks).max(1);
                    use rayon::prelude::*;
                    output[out_n_offset..out_n_offset + oc_total * hw_out]
                        .par_chunks_mut(chunk_size * hw_out)
                        .enumerate()
                        .for_each(|(chunk_i, out_chunk)| {
                            let oc_start = chunk_i * chunk_size;
                            let chunk_oc = out_chunk.len() / hw_out;
                            let weight_chunk = &weight[oc_start * kk..(oc_start + chunk_oc) * kk];
                            crate::blas::sgemm_row_major(
                                chunk_oc, hw_out, kk,
                                1.0,
                                weight_chunk, kk,
                                col_ref, hw_out,
                                0.0,
                                out_chunk, hw_out,
                            );
                        });
                }

                if let Some(b) = &self.bias {
                    use rayon::prelude::*;
                    output[out_n_offset..out_n_offset + oc_total * hw_out]
                        .par_chunks_mut(hw_out)
                        .enumerate()
                        .for_each(|(oc, row)| {
                            let bias = b[oc];
                            for j in 0..hw_out {
                                row[j] += bias;
                            }
                        });
                }
            }
        });
        (h_out, w_out)
    }

    /// GPU-resident im2col + sgemm + bias-add. Input is an fp16 GpuTensor
    /// `[N × C_in × H_in × W_in]`; output is a freshly-allocated fp16 GpuTensor
    /// `[N × C_out × H_out × W_out]`. Returns `(output, h_out, w_out)`.
    ///
    /// Uses the GPU im2col kernel (no CPU round-trip), MPS fp16 sgemm with
    /// the cached f16 weight buffer, and the strided fp16 bias-add kernel.
    /// Per-batch loop is sequential — for SD inference batch is 1 (or 2 with
    /// CFG), and each MPS sgemm internally saturates the GPU.
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        input: &crate::metal_backend::GpuTensor,
        n: usize, h_in: usize, w_in: usize,
    ) -> (crate::metal_backend::GpuTensor, usize, usize) {
        use crate::metal_backend::*;
        debug_assert_eq!(input.dtype, GpuDtype::F16);

        // MPSGraph path (opt-in via KLEARU_MPSGRAPH_CONV=1). Apple's
        // optimised conv kernel — replaces the per-batch im2col + sgemm
        // loop with a single fused conv. Cached graph per (weight, shape).
        if std::env::var_os("KLEARU_MPSGRAPH_CONV").is_some() {
            let (out_buf, h_out, w_out) = mpsgraph_conv2d(
                &input.buffer, &self.weight,
                n, self.in_channels, h_in, w_in,
                self.out_channels, self.kernel_h, self.kernel_w,
                self.stride, self.padding,
            );
            let mut out = GpuTensor {
                buffer: out_buf,
                shape: vec![n, self.out_channels, h_out, w_out],
                dtype: GpuDtype::F16,
            };
            if let Some(bias) = &self.bias {
                let b_buf = weight_f16_buffer(bias);
                bias_add_f16_gpu(&mut out, &b_buf, h_out * w_out, self.out_channels);
            }
            return (out, h_out, w_out);
        }

        let h_out = (h_in + 2 * self.padding - self.kernel_h) / self.stride + 1;
        let w_out = (w_in + 2 * self.padding - self.kernel_w) / self.stride + 1;
        let kk = self.in_channels * self.kernel_h * self.kernel_w;
        let hw_in = h_in * w_in;
        let hw_out = h_out * w_out;

        let out = GpuTensor::new_f16(vec![n, self.out_channels, h_out, w_out]);
        let w_buf = weight_f16_buffer(&self.weight);

        // Per-batch im2col → sgemm.
        for ni in 0..n {
            // im2col input: a view of input[ni] as [C_in, H_in, W_in].
            // We dispatch GPU im2col directly on the offset region of the
            // shared input buffer by encoding the slice's offset implicitly
            // via a temporary "view buffer" — Metal doesn't support buffer
            // offsets on set_buffer when contents() is used by kernels in
            // shape-implicit mode. Easiest portable path: make a small
            // staging buffer that mirrors the per-batch slice.
            //
            // For batch=1 (typical), this is a no-op aliasing case.
            let input_per_batch = if n == 1 {
                // Just reuse the input buffer directly — no copy needed.
                None
            } else {
                // Alias by absolute offset is awkward through the half-typed
                // device pointer in MSL; instead we set_buffer with an offset.
                Some(())
            };
            // Actual dispatch: regardless of batch, we can use set_buffer's
            // offset parameter to point at the per-batch slice.
            let col_buf = acquire_f16_buffer(kk * hw_out);
            let col_bytes = (kk * hw_out) * 2;
            let _ = (input_per_batch, col_bytes);

            // im2col with batched offset:
            im2col_per_batch_dispatch(
                &input.buffer, ni, hw_in,
                &col_buf,
                self.in_channels, h_in, w_in,
                self.kernel_h, self.kernel_w,
                self.stride, self.padding,
                h_out, w_out,
            );

            // sgemm: out_per_batch = weight · col.
            // weight [out_c, kk], col [kk, hw_out] → out [out_c, hw_out].
            sgemm_f16_buf_with_offsets(
                &out.buffer, ni * self.out_channels * hw_out,
                &w_buf, 0,
                &col_buf, 0,
                self.out_channels, hw_out, kk,
            );

            release_pool_buffer(col_buf);
        }

        // Bias-add: shape [N, C_out, HW_out]. period = HW_out, bias_len = C_out.
        if let Some(bias) = &self.bias {
            let b_buf = weight_f16_buffer(bias);
            let mut out_mut = out;
            bias_add_f16_gpu(&mut out_mut, &b_buf, hw_out, self.out_channels);
            return (out_mut, h_out, w_out);
        }

        (out, h_out, w_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "metal")]
    #[test]
    fn conv2d_forward_gpu_matches_cpu_3x3() {
        // 3×3 stride=1 padding=1 conv, in_channels=4, out_channels=8, H=W=8.
        let mut c = Conv2d::new(4, 8, 3, 1, 1, true);
        for (i, w) in c.weight.iter_mut().enumerate() {
            *w = ((i * 19) % 23) as f32 / 23.0 - 0.5;
        }
        if let Some(b) = &mut c.bias {
            for (i, v) in b.iter_mut().enumerate() {
                *v = (i as f32 * 0.05) - 0.2;
            }
        }
        let n = 1; let h_in = 8; let w_in = 8;
        let input: Vec<f32> = (0..n*4*h_in*w_in)
            .map(|i| ((i * 7) % 11) as f32 / 11.0 - 0.5).collect();

        // CPU reference.
        let mut out_cpu = Vec::new();
        let (h_out, w_out) = c.forward(&input, n, h_in, w_in, &mut out_cpu);

        // GPU-resident path.
        let in_gpu = crate::metal_backend::GpuTensor::upload_f32_as_f16(
            vec![n, c.in_channels, h_in, w_in], &input);
        let (out_gpu_t, h_g, w_g) = c.forward_gpu(&in_gpu, n, h_in, w_in);
        assert_eq!((h_g, w_g), (h_out, w_out));
        let out_gpu = out_gpu_t.download_to_f32();

        let mut max_diff = 0.0f32;
        for (g, cpu) in out_gpu.iter().zip(out_cpu.iter()) {
            let d = (g - cpu).abs();
            if d > max_diff { max_diff = d; }
        }
        // fp16 with kk=36 accumulations + 0.5-magnitude inputs: ~0.1 worst-case.
        assert!(max_diff < 0.2,
            "Conv2d::forward_gpu diverges from CPU: max_diff={max_diff}");
    }

    #[test]
    fn conv_1x1_is_channel_linear() {
        // 1×1 conv with identity weight and zero bias should be a per-pixel
        // channel-wise linear projection. Set out=in=2 with identity weight.
        let mut c = Conv2d::new(2, 2, 1, 1, 0, false);
        // weight [out=2, in=2, 1, 1] = identity
        c.weight = vec![1.0, 0.0, 0.0, 1.0];
        // input [N=1, C=2, H=1, W=1]: [[3, 5]]
        let input = vec![3.0, 5.0];
        let mut out = Vec::new();
        let (h, w) = c.forward(&input, 1, 1, 1, &mut out);
        assert_eq!((h, w), (1, 1));
        assert_eq!(out, vec![3.0, 5.0]);
    }

    #[test]
    fn conv_3x3_padding_1_preserves_spatial() {
        // 3×3 conv with stride=1, padding=1: output H/W same as input.
        let c = Conv2d::new(1, 1, 3, 1, 1, false);
        let input = vec![0.0; 16]; // 1×1×4×4
        let mut out = Vec::new();
        let (h, w) = c.forward(&input, 1, 4, 4, &mut out);
        assert_eq!((h, w), (4, 4));
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn conv_stride_2_downsamples() {
        let c = Conv2d::new(1, 1, 3, 2, 1, false);
        let input = vec![0.0; 16]; // 1×1×4×4
        let mut out = Vec::new();
        let (h, w) = c.forward(&input, 1, 4, 4, &mut out);
        assert_eq!((h, w), (2, 2));
    }

    #[test]
    fn conv_constant_input_constant_output() {
        // All-1s input, all-1s 3×3 weight, no bias, no padding → output is 9 at center.
        // With padding=1, edges sum fewer cells.
        let mut c = Conv2d::new(1, 1, 3, 1, 1, false);
        c.weight = vec![1.0; 9];
        let input = vec![1.0; 9]; // 1×1×3×3
        let mut out = Vec::new();
        c.forward(&input, 1, 3, 3, &mut out);
        // Center pixel sees 9 ones → 9. Edges see fewer.
        assert_eq!(out[1*3+1], 9.0);
        // Corner sees 4 ones (2×2 window of valid input).
        assert_eq!(out[0], 4.0);
    }
}
