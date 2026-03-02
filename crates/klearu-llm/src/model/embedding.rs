use klearu_accel::memory::ContiguousWeightStore;
use klearu_accel::simd::dense_dot_dense_simd;
use rayon::prelude::*;

/// Token embedding table backed by ContiguousWeightStore.
///
/// Layout: `[vocab_size × hidden_size]`.
/// Also serves as LM head when `tie_word_embeddings = true`.
pub struct Embedding {
    pub weights: ContiguousWeightStore,
    vocab_size: usize,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            weights: ContiguousWeightStore::new(vocab_size, hidden_size),
            vocab_size,
            hidden_size,
        }
    }

    /// Look up embedding for a single token ID.
    pub fn forward(&self, token_id: u32, output: &mut [f32]) {
        let row = self.weights.get_weights(token_id as usize);
        output[..self.hidden_size].copy_from_slice(&row[..self.hidden_size]);
    }

    /// Use embedding weights as LM head (tied embeddings).
    /// Computes `logits[i] = dot(hidden, embedding[i])` for each vocab token.
    pub fn lm_head_forward(&self, hidden: &[f32], logits: &mut [f32]) {
        let hidden_size = self.hidden_size;
        let weights = &self.weights;
        let dst = &mut logits[..self.vocab_size];

        if dst.len() >= 512 {
            dst.par_iter_mut()
                .with_min_len(64)
                .enumerate()
                .for_each(|(i, logit)| {
                    let row = weights.get_weights(i);
                    *logit = dense_dot_dense_simd(&row[..hidden_size], hidden);
                });
        } else {
            for (i, logit) in dst.iter_mut().enumerate() {
                let row = weights.get_weights(i);
                *logit = dense_dot_dense_simd(&row[..hidden_size], hidden);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup() {
        let mut emb = Embedding::new(4, 3);
        emb.weights.set_weights(0, &[1.0, 0.0, 0.0]);
        emb.weights.set_weights(1, &[0.0, 1.0, 0.0]);
        emb.weights.set_weights(2, &[0.0, 0.0, 1.0]);
        emb.weights.set_weights(3, &[1.0, 1.0, 1.0]);

        let mut out = vec![0.0; 3];
        emb.forward(2, &mut out);
        assert_eq!(&out, &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_tied_lm_head() {
        let mut emb = Embedding::new(3, 4);
        emb.weights.set_weights(0, &[1.0, 0.0, 0.0, 0.0]);
        emb.weights.set_weights(1, &[0.0, 1.0, 0.0, 0.0]);
        emb.weights.set_weights(2, &[0.0, 0.0, 1.0, 0.0]);

        let hidden = vec![0.5, 0.3, 0.8, 0.0];
        let mut logits = vec![0.0; 3];
        emb.lm_head_forward(&hidden, &mut logits);

        assert!((logits[0] - 0.5).abs() < 1e-5);
        assert!((logits[1] - 0.3).abs() < 1e-5);
        assert!((logits[2] - 0.8).abs() < 1e-5);
    }
}
