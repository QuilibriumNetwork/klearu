/// 2D Rotary Position Embedding for vision transformers.
///
/// Splits head_dim into two halves: the first half encodes row position,
/// the second half encodes column position. Used by EVA-02, DINOv3.
///
/// Each pair of consecutive dimensions (d, d+1) is rotated by an angle
/// determined by the spatial position and frequency band.

/// Precomputed 2D RoPE tables.
pub struct RoPE2d {
    /// Cosine values: `[max_tokens, head_dim / 2]`.
    cos: Vec<f32>,
    /// Sine values: `[max_tokens, head_dim / 2]`.
    sin: Vec<f32>,
    max_h: usize,
    max_w: usize,
    head_dim: usize,
    num_pairs: usize,
}

impl RoPE2d {
    /// Create a 2D RoPE table.
    ///
    /// - `max_h`, `max_w`: maximum spatial dimensions (e.g., 14x14 for 224/16 patches)
    /// - `head_dim`: dimension per attention head (must be even)
    /// - `theta`: frequency base (default 10000.0)
    pub fn new(max_h: usize, max_w: usize, head_dim: usize, theta: f32) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even");
        let num_pairs = head_dim / 2;
        let half_pairs = num_pairs / 2; // pairs for row, pairs for col
        let max_tokens = max_h * max_w;

        let mut cos = vec![0.0f32; max_tokens * num_pairs];
        let mut sin = vec![0.0f32; max_tokens * num_pairs];

        for y in 0..max_h {
            for x in 0..max_w {
                let token = y * max_w + x;
                let base = token * num_pairs;

                // First half of pairs: row position encoding
                for p in 0..half_pairs {
                    let freq = 1.0 / theta.powf(2.0 * p as f32 / half_pairs as f32);
                    let angle = y as f32 * freq;
                    cos[base + p] = angle.cos();
                    sin[base + p] = angle.sin();
                }

                // Second half of pairs: column position encoding
                for p in 0..half_pairs {
                    let freq = 1.0 / theta.powf(2.0 * p as f32 / half_pairs as f32);
                    let angle = x as f32 * freq;
                    cos[base + half_pairs + p] = angle.cos();
                    sin[base + half_pairs + p] = angle.sin();
                }

                // If odd number of half-pairs, last pair gets col encoding
                if num_pairs % 2 == 1 {
                    let p = num_pairs - 1;
                    let freq = 1.0 / theta.powf(2.0 * (half_pairs) as f32 / half_pairs as f32);
                    let angle = x as f32 * freq;
                    cos[base + p] = angle.cos();
                    sin[base + p] = angle.sin();
                }
            }
        }

        Self { cos, sin, max_h, max_w, head_dim, num_pairs }
    }

    /// Apply 2D RoPE to Q and K vectors for a single token position.
    ///
    /// `q` and `k` are slices of length `head_dim` for one head.
    /// `row`, `col` are the spatial position of the token.
    pub fn apply(&self, q: &mut [f32], k: &mut [f32], row: usize, col: usize) {
        debug_assert_eq!(q.len(), self.head_dim);
        debug_assert_eq!(k.len(), self.head_dim);
        debug_assert!(row < self.max_h && col < self.max_w);

        let token = row * self.max_w + col;
        let base = token * self.num_pairs;

        for p in 0..self.num_pairs {
            let c = self.cos[base + p];
            let s = self.sin[base + p];
            let d0 = p * 2;
            let d1 = p * 2 + 1;

            let q0 = q[d0];
            let q1 = q[d1];
            q[d0] = q0 * c - q1 * s;
            q[d1] = q0 * s + q1 * c;

            let k0 = k[d0];
            let k1 = k[d1];
            k[d0] = k0 * c - k1 * s;
            k[d1] = k0 * s + k1 * c;
        }
    }

    /// Apply 2D RoPE to a batch of Q and K vectors.
    ///
    /// `q` and `k` are `[num_tokens, head_dim]` for one head.
    /// `h`, `w` are the spatial grid dimensions (num_tokens = h * w).
    pub fn apply_batch(&self, q: &mut [f32], k: &mut [f32], h: usize, w: usize) {
        let hd = self.head_dim;
        debug_assert_eq!(q.len(), h * w * hd);
        debug_assert_eq!(k.len(), h * w * hd);

        for row in 0..h {
            for col in 0..w {
                let t = row * w + col;
                self.apply(
                    &mut q[t * hd..(t + 1) * hd],
                    &mut k[t * hd..(t + 1) * hd],
                    row,
                    col,
                );
            }
        }
    }

    pub fn max_h(&self) -> usize { self.max_h }
    pub fn max_w(&self) -> usize { self.max_w }
    pub fn head_dim(&self) -> usize { self.head_dim }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope2d_identity_at_origin() {
        let rope = RoPE2d::new(8, 8, 4, 10000.0);
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q_orig = q.clone();
        let k_orig = k.clone();

        // At position (0, 0), all angles are 0 → cos=1, sin=0 → identity
        rope.apply(&mut q, &mut k, 0, 0);
        for (a, b) in q.iter().zip(q_orig.iter()) {
            assert!((a - b).abs() < 1e-6, "q mismatch at origin");
        }
        for (a, b) in k.iter().zip(k_orig.iter()) {
            assert!((a - b).abs() < 1e-6, "k mismatch at origin");
        }
    }

    #[test]
    fn test_rope2d_different_positions_differ() {
        let rope = RoPE2d::new(8, 8, 4, 10000.0);

        let mut q1 = vec![1.0, 2.0, 3.0, 4.0];
        let mut k1 = vec![1.0, 2.0, 3.0, 4.0];
        rope.apply(&mut q1, &mut k1, 1, 0);

        let mut q2 = vec![1.0, 2.0, 3.0, 4.0];
        let mut k2 = vec![1.0, 2.0, 3.0, 4.0];
        rope.apply(&mut q2, &mut k2, 0, 1);

        // Position (1,0) and (0,1) should produce different results
        let any_diff = q1.iter().zip(q2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "Different positions should produce different embeddings");
    }

    #[test]
    fn test_rope2d_preserves_norm() {
        let rope = RoPE2d::new(8, 8, 4, 10000.0);
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q_norm_before: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let k_norm_before: f32 = k.iter().map(|v| v * v).sum::<f32>().sqrt();

        rope.apply(&mut q, &mut k, 3, 5);

        let q_norm_after: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let k_norm_after: f32 = k.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!((q_norm_before - q_norm_after).abs() < 1e-4, "RoPE should preserve Q norm");
        assert!((k_norm_before - k_norm_after).abs() < 1e-4, "RoPE should preserve K norm");
    }

    #[test]
    fn test_rope2d_batch() {
        let rope = RoPE2d::new(4, 4, 4, 10000.0);
        let h = 2;
        let w = 2;
        let hd = 4;
        let mut q = vec![1.0f32; h * w * hd];
        let mut k = vec![1.0f32; h * w * hd];

        rope.apply_batch(&mut q, &mut k, h, w);

        // All values should be finite
        for &v in q.iter().chain(k.iter()) {
            assert!(v.is_finite());
        }
    }
}
