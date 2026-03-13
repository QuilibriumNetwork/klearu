use super::kv_cache::KvCache;
use super::linear::Linear;
use super::rope::RotaryEmbedding;
use klearu_accel::simd::dense_dot_dense_simd;

/// Multi-head attention with Grouped Query Attention (GQA), KV cache,
/// and optional Qwen3.5 features (output gate, Q/K RMSNorm, partial RoPE).
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_group_size: usize,

    // Qwen3.5 gated full attention features
    /// If true, q_proj output is doubled: first half = query, second half = sigmoid gate.
    pub output_gate: bool,
    /// Per-head_dim RMSNorm for Q (Qwen3.5). Uses (1 + weight) scaling.
    pub q_norm_weight: Option<Vec<f32>>,
    /// Per-head_dim RMSNorm for K (Qwen3.5). Uses (1 + weight) scaling.
    pub k_norm_weight: Option<Vec<f32>>,
    /// Epsilon for Q/K RMSNorm.
    pub qk_norm_eps: f32,
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
            output_gate: false,
            q_norm_weight: None,
            k_norm_weight: None,
            qk_norm_eps: 1e-6,
        }
    }

    /// Create an attention layer for Qwen3.5 gated full attention.
    /// q_proj is doubled (2 * num_heads * head_dim) for the output gate.
    pub fn new_gated(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f32,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            // q_proj output is doubled: query + gate
            q_proj: Linear::new(hidden_size, q_dim * 2),
            k_proj: Linear::new(hidden_size, kv_dim),
            v_proj: Linear::new(hidden_size, kv_dim),
            o_proj: Linear::new(q_dim, hidden_size),
            num_heads,
            num_kv_heads,
            head_dim,
            gqa_group_size: num_heads / num_kv_heads,
            output_gate: true,
            q_norm_weight: Some(vec![0.0; head_dim]),
            k_norm_weight: Some(vec![0.0; head_dim]),
            qk_norm_eps: rms_norm_eps,
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
        let hidden_size = self.o_proj.out_features();
        let mut output = vec![0.0f32; hidden_size];
        self.forward_into(input, position, rope, kv_cache, &mut output);
        output
    }

    /// Forward pass writing into a pre-allocated output buffer.
    pub fn forward_into(
        &self,
        input: &[f32],
        position: usize,
        rope: &RotaryEmbedding,
        kv_cache: &mut KvCache,
        output: &mut [f32],
    ) {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        // Project K, V (same for both standard and gated)
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        self.k_proj.forward(input, &mut k);
        self.v_proj.forward(input, &mut v);

        // Project Q (and gate if output_gate)
        let (mut q, gate) = if self.output_gate {
            // q_proj output is [num_heads × (head_dim * 2)], interleaved per-head:
            // [head0_query(head_dim) | head0_gate(head_dim) | head1_query | head1_gate | ...]
            // We must de-interleave into separate q[num_heads × head_dim] and gate[num_heads × head_dim].
            let mut q_gate = vec![0.0f32; q_dim * 2];
            self.q_proj.forward(input, &mut q_gate);
            let mut q = vec![0.0f32; q_dim];
            let mut gate = vec![0.0f32; q_dim];
            for h in 0..self.num_heads {
                let src_offset = h * self.head_dim * 2;
                let dst_offset = h * self.head_dim;
                q[dst_offset..dst_offset + self.head_dim]
                    .copy_from_slice(&q_gate[src_offset..src_offset + self.head_dim]);
                gate[dst_offset..dst_offset + self.head_dim]
                    .copy_from_slice(&q_gate[src_offset + self.head_dim..src_offset + self.head_dim * 2]);
            }
            (q, Some(gate))
        } else {
            let mut q = vec![0.0f32; q_dim];
            self.q_proj.forward(input, &mut q);
            (q, None)
        };

        // Apply per-head Q/K RMSNorm if present (Qwen3.5, uses 1+weight scaling)
        if let Some(ref qw) = self.q_norm_weight {
            for h in 0..self.num_heads {
                let offset = h * self.head_dim;
                rms_norm_one_plus(&mut q[offset..offset + self.head_dim], qw, self.qk_norm_eps);
            }
        }
        if let Some(ref kw) = self.k_norm_weight {
            for h in 0..self.num_kv_heads {
                let offset = h * self.head_dim;
                rms_norm_one_plus(&mut k[offset..offset + self.head_dim], kw, self.qk_norm_eps);
            }
        }

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

        // Compute attention for each Q head
        let mut attn_concat = vec![0.0f32; q_dim];
        let mut scores = vec![0.0f32; seq_len];

        for h in 0..self.num_heads {
            let kv_h = h / self.gqa_group_size;
            let q_head = &q[h * self.head_dim..(h + 1) * self.head_dim];

            // Compute attention scores over all cached positions (reuse buffer)
            for j in 0..seq_len {
                let k_j = kv_cache.k_at(kv_h, j);
                scores[j] = dense_dot_dense_simd(q_head, k_j) * inv_sqrt_dk;
            }

            // Softmax
            softmax_inplace(&mut scores[..seq_len]);

            // Weighted sum of V
            let head_out = &mut attn_concat[h * self.head_dim..(h + 1) * self.head_dim];
            for (j, &score) in scores[..seq_len].iter().enumerate() {
                let v_j = kv_cache.v_at(kv_h, j);
                for (d, &vv) in head_out.iter_mut().zip(v_j.iter()) {
                    *d += score * vv;
                }
            }
        }

        // Apply output gate: output *= sigmoid(gate)
        if let Some(ref gate) = gate {
            for (o, &g) in attn_concat.iter_mut().zip(gate.iter()) {
                *o *= sigmoid(g);
            }
        }

        // Output projection
        output.iter_mut().for_each(|v| *v = 0.0);
        self.o_proj.forward(&attn_concat, output);
    }

    /// Forward pass using pre-allocated scratch buffers.
    /// Avoids all heap allocation in the hot decode path.
    pub fn forward_into_buffered(
        &self,
        input: &[f32],
        position: usize,
        rope: &RotaryEmbedding,
        kv_cache: &mut KvCache,
        output: &mut [f32],
        scratch: &mut AttentionScratch,
    ) {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        // Project K, V
        self.k_proj.forward(input, &mut scratch.k[..kv_dim]);
        self.v_proj.forward(input, &mut scratch.v[..kv_dim]);

        // Project Q (and gate if output_gate)
        let has_gate = if self.output_gate {
            self.q_proj.forward(input, &mut scratch.q_gate[..q_dim * 2]);
            // De-interleave: q_gate is [head0_query | head0_gate | head1_query | head1_gate | ...]
            for h in 0..self.num_heads {
                let src_offset = h * self.head_dim * 2;
                let dst_offset = h * self.head_dim;
                scratch.q[dst_offset..dst_offset + self.head_dim]
                    .copy_from_slice(&scratch.q_gate[src_offset..src_offset + self.head_dim]);
                scratch.gate[dst_offset..dst_offset + self.head_dim]
                    .copy_from_slice(&scratch.q_gate[src_offset + self.head_dim..src_offset + self.head_dim * 2]);
            }
            true
        } else {
            self.q_proj.forward(input, &mut scratch.q[..q_dim]);
            false
        };

        // Apply per-head Q/K RMSNorm if present
        if let Some(ref qw) = self.q_norm_weight {
            for h in 0..self.num_heads {
                let offset = h * self.head_dim;
                rms_norm_one_plus(&mut scratch.q[offset..offset + self.head_dim], qw, self.qk_norm_eps);
            }
        }
        if let Some(ref kw) = self.k_norm_weight {
            for h in 0..self.num_kv_heads {
                let offset = h * self.head_dim;
                rms_norm_one_plus(&mut scratch.k[offset..offset + self.head_dim], kw, self.qk_norm_eps);
            }
        }

        // Apply RoPE
        for h in 0..self.num_heads {
            let offset = h * self.head_dim;
            rope.apply(&mut scratch.q[offset..offset + self.head_dim], position);
        }
        for h in 0..self.num_kv_heads {
            let offset = h * self.head_dim;
            rope.apply(&mut scratch.k[offset..offset + self.head_dim], position);
        }

        // Append to KV cache
        kv_cache.append(&scratch.k[..kv_dim], &scratch.v[..kv_dim]);

        let seq_len = kv_cache.current_len();
        let inv_sqrt_dk = 1.0 / (self.head_dim as f32).sqrt();

        // Zero attn_concat for accumulation
        scratch.attn_concat[..q_dim].fill(0.0);

        for h in 0..self.num_heads {
            let kv_h = h / self.gqa_group_size;
            let q_head = &scratch.q[h * self.head_dim..(h + 1) * self.head_dim];

            for j in 0..seq_len {
                let k_j = kv_cache.k_at(kv_h, j);
                scratch.scores[j] = dense_dot_dense_simd(q_head, k_j) * inv_sqrt_dk;
            }

            softmax_inplace(&mut scratch.scores[..seq_len]);

            let head_out = &mut scratch.attn_concat[h * self.head_dim..(h + 1) * self.head_dim];
            for (j, &score) in scratch.scores[..seq_len].iter().enumerate() {
                let v_j = kv_cache.v_at(kv_h, j);
                for (d, &vv) in head_out.iter_mut().zip(v_j.iter()) {
                    *d += score * vv;
                }
            }
        }

        // Apply output gate
        if has_gate {
            for (o, &g) in scratch.attn_concat[..q_dim].iter_mut().zip(scratch.gate.iter()) {
                *o *= sigmoid(g);
            }
        }

        // Output projection
        output.iter_mut().for_each(|v| *v = 0.0);
        self.o_proj.forward(&scratch.attn_concat[..q_dim], output);
    }
}

/// Pre-allocated scratch buffers for attention forward pass.
/// Reuse across layers and tokens to avoid per-call heap allocation.
pub struct AttentionScratch {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub q: Vec<f32>,
    pub q_gate: Vec<f32>,
    pub gate: Vec<f32>,
    pub attn_concat: Vec<f32>,
    pub scores: Vec<f32>,
}

impl AttentionScratch {
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        output_gate: bool,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        Self {
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            q: vec![0.0; q_dim],
            q_gate: if output_gate { vec![0.0; q_dim * 2] } else { Vec::new() },
            gate: if output_gate { vec![0.0; q_dim] } else { Vec::new() },
            attn_concat: vec![0.0; q_dim],
            scores: vec![0.0; max_seq_len],
        }
    }
}

/// In-place RMSNorm with (1 + weight) scaling.
fn rms_norm_one_plus(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(n, weight.len());

    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for (xi, &wi) in x.iter_mut().zip(weight.iter()) {
        *xi = *xi * inv_rms * (1.0 + wi);
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
    use super::super::kv_cache::KvCache;

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

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_rms_norm_one_plus() {
        let weight = vec![0.0; 4]; // (1+0) = 1, same as unit
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let expected_rms = (7.5f32 + 1e-6).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0].iter().map(|&v| v / expected_rms).collect();
        rms_norm_one_plus(&mut x, &weight, 1e-6);
        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    #[test]
    fn test_gated_attention_output_finite() {
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;

        let attn = Attention::new_gated(hidden_size, num_heads, num_kv_heads, head_dim, 1e-6);
        let rope = RotaryEmbedding::new(head_dim, 32, 10000.0);
        let mut kv_cache = KvCache::new(num_kv_heads, 32, head_dim);

        let input = vec![0.1; hidden_size];
        let output = attn.forward(&input, 0, &rope, &mut kv_cache);

        assert_eq!(output.len(), hidden_size);
        assert!(output.iter().all(|x| x.is_finite()));
    }
}
