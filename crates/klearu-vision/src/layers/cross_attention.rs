/// Cross-attention layer for VLM adapters and DETR-like decoders.
///
/// Q from one sequence, K/V from another.

use crate::layers::LinearBias;

pub struct CrossAttention {
    pub q_proj: LinearBias,
    pub k_proj: LinearBias,
    pub v_proj: LinearBias,
    pub out_proj: LinearBias,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl CrossAttention {
    pub fn new(q_dim: usize, kv_dim: usize, num_heads: usize) -> Self {
        let head_dim = q_dim / num_heads;
        Self {
            q_proj: LinearBias::new(q_dim, q_dim),
            k_proj: LinearBias::new(kv_dim, q_dim),
            v_proj: LinearBias::new(kv_dim, q_dim),
            out_proj: LinearBias::new(q_dim, q_dim),
            num_heads,
            head_dim,
        }
    }

    /// Forward: `query [q_len, q_dim]`, `kv [kv_len, kv_dim]` → `[q_len, q_dim]`.
    pub fn forward(
        &self,
        query: &[f32],
        q_len: usize,
        kv: &[f32],
        kv_len: usize,
    ) -> Vec<f32> {
        let dim = self.num_heads * self.head_dim;

        // Project Q, K, V
        let mut q_buf = vec![0.0f32; q_len * dim];
        let mut k_buf = vec![0.0f32; kv_len * dim];
        let mut v_buf = vec![0.0f32; kv_len * dim];

        for t in 0..q_len {
            self.q_proj.forward(&query[t * dim..(t + 1) * dim], &mut q_buf[t * dim..(t + 1) * dim]);
        }
        let kv_dim = self.k_proj.in_features();
        for t in 0..kv_len {
            self.k_proj.forward(&kv[t * kv_dim..(t + 1) * kv_dim], &mut k_buf[t * dim..(t + 1) * dim]);
            self.v_proj.forward(&kv[t * kv_dim..(t + 1) * kv_dim], &mut v_buf[t * dim..(t + 1) * dim]);
        }

        let scale = (self.head_dim as f32).powf(-0.5);
        let mut attn_out = vec![0.0f32; q_len * dim];

        for head in 0..self.num_heads {
            let h_off = head * self.head_dim;

            // Scores: Q @ K^T
            let mut scores = vec![0.0f32; q_len * kv_len];
            for qi in 0..q_len {
                for ki in 0..kv_len {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_buf[qi * dim + h_off + d] * k_buf[ki * dim + h_off + d];
                    }
                    scores[qi * kv_len + ki] = dot * scale;
                }
            }

            // Softmax
            for qi in 0..q_len {
                let row = &mut scores[qi * kv_len..(qi + 1) * kv_len];
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

            // Weighted sum of V
            for qi in 0..q_len {
                for d in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for vi in 0..kv_len {
                        sum += scores[qi * kv_len + vi] * v_buf[vi * dim + h_off + d];
                    }
                    attn_out[qi * dim + h_off + d] = sum;
                }
            }
        }

        // Output projection
        let mut output = vec![0.0f32; q_len * dim];
        for t in 0..q_len {
            self.out_proj.forward(&attn_out[t * dim..(t + 1) * dim], &mut output[t * dim..(t + 1) * dim]);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_attention() {
        let ca = CrossAttention::new(16, 8, 4);
        let query = vec![0.1f32; 3 * 16]; // 3 queries
        let kv = vec![0.1f32; 5 * 8];     // 5 KV tokens
        let output = ca.forward(&query, 3, &kv, 5);
        assert_eq!(output.len(), 3 * 16);
        for &v in &output {
            assert!(v.is_finite());
        }
    }
}
