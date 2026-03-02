use crate::model::attention::Attention;
use crate::model::kv_cache::KvCache;
use crate::model::rope::RotaryEmbedding;
use klearu_accel::simd::dense_dot_dense_simd;
use rayon::prelude::*;

/// Sparse attention: only compute selected heads, zero out the rest.
pub fn forward_sparse(
    attn: &Attention,
    input: &[f32],
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    active_heads: &[usize],
) -> Vec<f32> {
    let num_heads = attn.num_heads();
    let num_kv_heads = attn.num_kv_heads();
    let head_dim = attn.head_dim();
    let gqa_group_size = num_heads / num_kv_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Project Q, K, V (full projection - could optimize later to only project active heads)
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];

    attn.q_proj.forward(input, &mut q);
    attn.k_proj.forward(input, &mut k);
    attn.v_proj.forward(input, &mut v);

    // Apply RoPE
    for h in 0..num_heads {
        rope.apply(&mut q[h * head_dim..(h + 1) * head_dim], position);
    }
    for h in 0..num_kv_heads {
        rope.apply(&mut k[h * head_dim..(h + 1) * head_dim], position);
    }

    // Append to KV cache (always, even for sparse)
    kv_cache.append(&k, &v);
    let seq_len = kv_cache.current_len();
    let inv_sqrt_dk = 1.0 / (head_dim as f32).sqrt();
    let kv = &*kv_cache;

    let mut attn_concat = vec![0.0f32; q_dim];

    let compute_head = |h: usize| -> Vec<f32> {
        let kv_h = h / gqa_group_size;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];

        let mut scores: Vec<f32> = (0..seq_len)
            .map(|j| {
                let k_j = kv.k_at(kv_h, j);
                dense_dot_dense_simd(q_head, k_j) * inv_sqrt_dk
            })
            .collect();

        softmax_inplace(&mut scores);

        let mut head_out = vec![0.0f32; head_dim];
        for (j, &score) in scores.iter().enumerate() {
            let v_j = kv.v_at(kv_h, j);
            for (d, &vv) in head_out.iter_mut().zip(v_j.iter()) {
                *d += score * vv;
            }
        }
        head_out
    };

    let valid_heads: Vec<usize> = active_heads.iter().copied().filter(|&h| h < num_heads).collect();

    if valid_heads.len() >= 8 {
        let head_results: Vec<(usize, Vec<f32>)> = valid_heads
            .par_iter()
            .map(|&h| (h, compute_head(h)))
            .collect();
        for (h, head_out) in head_results {
            attn_concat[h * head_dim..(h + 1) * head_dim].copy_from_slice(&head_out);
        }
    } else {
        for &h in &valid_heads {
            let head_out = compute_head(h);
            attn_concat[h * head_dim..(h + 1) * head_dim].copy_from_slice(&head_out);
        }
    }

    // Output projection
    let hidden_size = attn.o_proj.out_features();
    let mut output = vec![0.0f32; hidden_size];
    attn.o_proj.forward(&attn_concat, &mut output);

    output
}

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
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_attention_subset() {
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;

        let attn = Attention::new(hidden_size, num_heads, num_kv_heads, head_dim);
        let rope = RotaryEmbedding::new(head_dim, 32, 10000.0);
        let mut kv_cache = KvCache::new(num_kv_heads, 32, head_dim);

        let input = vec![0.1; hidden_size];

        // Use only 2 of 4 heads
        let active_heads = vec![0, 2];
        let output = forward_sparse(&attn, &input, 0, &rope, &mut kv_cache, &active_heads);

        assert_eq!(output.len(), hidden_size);
        assert!(output.iter().all(|x| x.is_finite()));
    }
}
