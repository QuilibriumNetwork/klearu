use crate::layers::LinearBias;

/// Channel group attention.
///
/// Unlike window attention which attends over spatial positions within windows,
/// channel attention transposes and attends over the head_dim dimension,
/// treating channels as the sequence dimension. This captures global spatial context.
pub struct ChannelAttention {
    pub qkv: LinearBias,
    pub proj: LinearBias,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl ChannelAttention {
    pub fn new(dim: usize, num_heads: usize) -> Self {
        let head_dim = dim / num_heads;
        Self {
            qkv: LinearBias::new(dim, dim * 3),
            proj: LinearBias::new(dim, dim),
            num_heads,
            head_dim,
        }
    }

    /// Forward pass.
    ///
    /// Input: `[N, C]` where N = H*W spatial tokens.
    /// Output: `[N, C]`.
    ///
    /// Channel attention transposes Q/K/V so that head_dim positions become
    /// the sequence dimension (length = head_dim), and N spatial positions
    /// become the feature dimension. Scale factor is `N^(-0.5)`.
    pub fn forward(&self, input: &[f32], n: usize) -> Vec<f32> {
        let dim = self.num_heads * self.head_dim;
        debug_assert_eq!(input.len(), n * dim);

        // QKV projection: [N, dim] → [N, 3*dim]
        let mut qkv_buf = vec![0.0f32; n * dim * 3];
        for t in 0..n {
            self.qkv.forward(
                &input[t * dim..(t + 1) * dim],
                &mut qkv_buf[t * dim * 3..(t + 1) * dim * 3],
            );
        }

        // Reshape to [num_heads, head_dim, N] for each of Q/K/V
        // Then compute attention over head_dim dimension with scale = N^(-0.5)
        let scale = (n as f32).powf(-0.5);
        let mut attn_output = vec![0.0f32; n * dim];

        for head in 0..self.num_heads {
            let head_offset = head * self.head_dim;

            // Channel attention: attention matrix is [head_dim, head_dim]
            // scores[d1, d2] = sum_n(Q[n, d1] * K[n, d2]) * scale
            let mut scores = vec![0.0f32; self.head_dim * self.head_dim];

            for d1 in 0..self.head_dim {
                for d2 in 0..self.head_dim {
                    let mut dot = 0.0f32;
                    for t in 0..n {
                        let q = qkv_buf[t * dim * 3 + head_offset + d1];
                        let k = qkv_buf[t * dim * 3 + dim + head_offset + d2];
                        dot += q * k;
                    }
                    scores[d1 * self.head_dim + d2] = dot * scale;
                }
            }

            // Softmax per row (over d2)
            for d1 in 0..self.head_dim {
                let row = &mut scores[d1 * self.head_dim..(d1 + 1) * self.head_dim];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                if sum > 0.0 {
                    for v in row.iter_mut() {
                        *v /= sum;
                    }
                }
            }

            // Weighted sum: output[n, d1] = sum_d2(scores[d1, d2] * V[n, d2])
            for t in 0..n {
                for d1 in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for d2 in 0..self.head_dim {
                        let v = qkv_buf[t * dim * 3 + 2 * dim + head_offset + d2];
                        sum += scores[d1 * self.head_dim + d2] * v;
                    }
                    attn_output[t * dim + head_offset + d1] = sum;
                }
            }
        }

        // Output projection
        let mut proj_output = vec![0.0f32; n * dim];
        for t in 0..n {
            self.proj.forward(
                &attn_output[t * dim..(t + 1) * dim],
                &mut proj_output[t * dim..(t + 1) * dim],
            );
        }

        proj_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_attention_output_shape() {
        let ca = ChannelAttention::new(8, 2);
        let n = 49; // 7x7
        let input = vec![0.1f32; n * 8];
        let output = ca.forward(&input, n);
        assert_eq!(output.len(), n * 8);
    }

    #[test]
    fn test_channel_attention_finite() {
        let ca = ChannelAttention::new(8, 2);
        let n = 16;
        let input = vec![0.1f32; n * 8];
        let output = ca.forward(&input, n);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
    }
}
