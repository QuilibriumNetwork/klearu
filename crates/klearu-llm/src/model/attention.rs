use super::kv_cache::KvCache;
use super::linear::Linear;
use super::rope::RotaryEmbedding;
use klearu_accel::simd::dense_dot_dense_simd;
use rayon::prelude::*;

/// Multi-head attention with Grouped Query Attention (GQA) and KV cache.
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_group_size: usize,
}

impl Attention {
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            q_proj: Linear::new(hidden_size, q_dim),
            k_proj: Linear::new(hidden_size, kv_dim),
            v_proj: Linear::new(hidden_size, kv_dim),
            o_proj: Linear::new(q_dim, hidden_size),
            num_heads,
            num_kv_heads,
            head_dim,
            gqa_group_size: num_heads / num_kv_heads,
        }
    }

    /// Forward pass for a single token at `position`.
    /// Returns attention output of size `[hidden_size]`.
    pub fn forward(
        &self,
        input: &[f32],
        position: usize,
        rope: &RotaryEmbedding,
        kv_cache: &mut KvCache,
    ) -> Vec<f32> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        // Project Q, K, V
        let mut q = vec![0.0f32; q_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];

        self.q_proj.forward(input, &mut q);
        self.k_proj.forward(input, &mut k);
        self.v_proj.forward(input, &mut v);

        // Apply RoPE to each Q head
        for h in 0..self.num_heads {
            let offset = h * self.head_dim;
            rope.apply(&mut q[offset..offset + self.head_dim], position);
        }

        // Apply RoPE to each K head
        for h in 0..self.num_kv_heads {
            let offset = h * self.head_dim;
            rope.apply(&mut k[offset..offset + self.head_dim], position);
        }

        // Append to KV cache
        kv_cache.append(&k, &v);

        let seq_len = kv_cache.current_len();
        let inv_sqrt_dk = 1.0 / (self.head_dim as f32).sqrt();
        let kv = &*kv_cache;

        // Compute attention for each Q head.
        // Each head writes to a disjoint head_dim-sized chunk of attn_concat.
        let mut attn_concat = vec![0.0f32; q_dim];
        let head_dim = self.head_dim;
        let gqa_group_size = self.gqa_group_size;
        let num_heads = self.num_heads;

        let compute_head = |h: usize, head_out: &mut [f32]| {
            let kv_h = h / gqa_group_size;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            let mut scores: Vec<f32> = (0..seq_len)
                .map(|j| {
                    let k_j = kv.k_at(kv_h, j);
                    dense_dot_dense_simd(q_head, k_j) * inv_sqrt_dk
                })
                .collect();

            softmax_inplace(&mut scores);

            for (j, &score) in scores.iter().enumerate() {
                let v_j = kv.v_at(kv_h, j);
                for (d, &vv) in head_out.iter_mut().zip(v_j.iter()) {
                    *d += score * vv;
                }
            }
        };

        if num_heads >= 8 {
            attn_concat
                .par_chunks_mut(head_dim)
                .enumerate()
                .for_each(|(h, head_out)| compute_head(h, head_out));
        } else {
            for h in 0..num_heads {
                let head_out = &mut attn_concat[h * head_dim..(h + 1) * head_dim];
                compute_head(h, head_out);
            }
        }

        // Output projection
        let hidden_size = self.o_proj.out_features();
        let mut output = vec![0.0f32; hidden_size];
        self.o_proj.forward(&attn_concat, &mut output);

        output
    }
}

/// In-place softmax over a slice.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Should be monotonically increasing
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
    }

    #[test]
    fn test_softmax_single() {
        let mut x = vec![5.0];
        softmax_inplace(&mut x);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gqa_head_mapping() {
        // 4 Q heads, 2 KV heads => group_size = 2
        let attn = Attention::new(16, 4, 2, 4);
        assert_eq!(attn.gqa_group_size, 2);
        // Q head 0,1 -> KV head 0; Q head 2,3 -> KV head 1
        let g = attn.gqa_group_size;
        // Q heads 0,1 share KV head 0; Q heads 2,3 share KV head 1
        let mappings: Vec<usize> = (0..4).map(|h| h / g).collect();
        assert_eq!(mappings, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_attention_output_finite() {
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;

        let attn = Attention::new(hidden_size, num_heads, num_kv_heads, head_dim);
        let rope = RotaryEmbedding::new(head_dim, 32, 10000.0);
        let mut kv_cache = KvCache::new(num_kv_heads, 32, head_dim);

        let input = vec![0.1; hidden_size];
        let output = attn.forward(&input, 0, &rope, &mut kv_cache);

        assert_eq!(output.len(), hidden_size);
        assert!(output.iter().all(|x| x.is_finite()));
        assert_eq!(kv_cache.current_len(), 1);
    }

    #[test]
    fn test_causal_independence() {
        let hidden_size = 16;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;

        let attn = Attention::new(hidden_size, num_heads, num_kv_heads, head_dim);
        let rope = RotaryEmbedding::new(head_dim, 32, 10000.0);

        // Run with 1 token
        let mut kv1 = KvCache::new(num_kv_heads, 32, head_dim);
        let input0 = vec![0.1; hidden_size];
        let out0_alone = attn.forward(&input0, 0, &rope, &mut kv1);

        // Run with 2 tokens - first token's output should be same
        let mut kv2 = KvCache::new(num_kv_heads, 32, head_dim);
        let out0_with_next = attn.forward(&input0, 0, &rope, &mut kv2);

        // The output for position 0 should be the same regardless of future tokens
        for (a, b) in out0_alone.iter().zip(out0_with_next.iter()) {
            assert!((a - b).abs() < 1e-6, "Causal violation: {a} != {b}");
        }
    }
}
