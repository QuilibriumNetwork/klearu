/// Rotary Positional Embeddings (RoPE) with optional partial rotation.
///
/// Precomputes `cos` and `sin` tables for all positions and frequency bands.
/// Applies rotation to pairs of elements: `(x0, x1) -> (x0*cos - x1*sin, x1*cos + x0*sin)`.
///
/// When `rotary_dim < head_dim`, only the first `rotary_dim` elements are rotated;
/// the remaining elements pass through unchanged (Qwen3.5 partial RoPE).
pub struct RotaryEmbedding {
    cos_cache: Vec<f32>, // [max_seq_len × half_rotary_dim]
    sin_cache: Vec<f32>,
    half_rotary_dim: usize,
    head_dim: usize,
}

impl RotaryEmbedding {
    /// Create RoPE where the entire head_dim is rotated (standard LLaMA).
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        Self::new_partial(head_dim, head_dim, max_seq_len, theta)
    }

    /// Create RoPE with partial rotation.
    /// `rotary_dim` must be even and <= `head_dim`.
    pub fn new_partial(head_dim: usize, rotary_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        debug_assert!(rotary_dim <= head_dim);
        debug_assert!(rotary_dim % 2 == 0);

        let half_rotary_dim = rotary_dim / 2;
        let mut cos_cache = vec![0.0f32; max_seq_len * half_rotary_dim];
        let mut sin_cache = vec![0.0f32; max_seq_len * half_rotary_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_rotary_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / rotary_dim as f32);
                let angle = pos as f32 * freq;
                cos_cache[pos * half_rotary_dim + i] = angle.cos();
                sin_cache[pos * half_rotary_dim + i] = angle.sin();
            }
        }

        Self {
            cos_cache,
            sin_cache,
            half_rotary_dim,
            head_dim,
        }
    }

    /// Get the cos table entry for a given position.
    /// Returns `&[f32]` of length `half_rotary_dim`.
    pub fn cos_at(&self, position: usize) -> &[f32] {
        let base = position * self.half_rotary_dim;
        &self.cos_cache[base..base + self.half_rotary_dim]
    }

    /// Get the sin table entry for a given position.
    /// Returns `&[f32]` of length `half_rotary_dim`.
    pub fn sin_at(&self, position: usize) -> &[f32] {
        let base = position * self.half_rotary_dim;
        &self.sin_cache[base..base + self.half_rotary_dim]
    }

    /// The number of dimension pairs that are rotated.
    pub fn half_rotary_dim(&self) -> usize {
        self.half_rotary_dim
    }

    /// The head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Apply RoPE to a single head's Q or K vector in-place.
    /// `x` has length `head_dim`.
    /// Only the first `rotary_dim` elements are rotated; the rest pass through.
    pub fn apply(&self, x: &mut [f32], position: usize) {
        debug_assert!(x.len() >= self.head_dim);
        let base = position * self.half_rotary_dim;
        for i in 0..self.half_rotary_dim {
            let cos = self.cos_cache[base + i];
            let sin = self.sin_cache[base + i];
            let x0 = x[i];
            let x1 = x[self.half_rotary_dim + i];
            x[i] = x0 * cos - x1 * sin;
            x[self.half_rotary_dim + i] = x1 * cos + x0 * sin;
        }
        // Elements beyond rotary_dim are left unchanged.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_preserves_magnitude() {
        let rope = RotaryEmbedding::new(8, 16, 10000.0);
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mag_before: f32 = x.iter().map(|v| v * v).sum();

        rope.apply(&mut x, 5);

        let mag_after: f32 = x.iter().map(|v| v * v).sum();
        assert!(
            (mag_before - mag_after).abs() < 1e-4,
            "RoPE should preserve magnitude: {mag_before} vs {mag_after}"
        );
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        let rope = RotaryEmbedding::new(4, 8, 10000.0);
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = original.clone();

        rope.apply(&mut x, 0);

        // At position 0, all angles are 0, so cos=1, sin=0
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "Position 0 should be identity");
        }
    }

    #[test]
    fn test_rope_known_angle() {
        // head_dim=2, half_dim=1, theta=1.0
        // freq[0] = 1/1^(0/2) = 1.0
        // At position 1: angle = 1.0, cos(1)=0.5403, sin(1)=0.8415
        let rope = RotaryEmbedding::new(2, 4, 1.0);
        let mut x = vec![1.0, 0.0];
        rope.apply(&mut x, 1);

        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();
        assert!((x[0] - cos1).abs() < 1e-5);
        assert!((x[1] - sin1).abs() < 1e-5);
    }

    #[test]
    fn test_rope_different_positions_differ() {
        let rope = RotaryEmbedding::new(4, 8, 10000.0);
        let original = vec![1.0, 2.0, 3.0, 4.0];

        let mut x1 = original.clone();
        rope.apply(&mut x1, 1);

        let mut x2 = original.clone();
        rope.apply(&mut x2, 2);

        let diff: f32 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "Different positions should produce different outputs");
    }

    #[test]
    fn test_rope_partial_rotation() {
        // head_dim=8, rotary_dim=4 => only first 4 elements rotated
        let rope = RotaryEmbedding::new_partial(8, 4, 16, 10000.0);
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let pass_through = x[4..].to_vec();

        rope.apply(&mut x, 5);

        // First 4 elements should have changed
        let rotated_mag: f32 = x[..4].iter().map(|v| v * v).sum();
        let orig_mag: f32 = [1.0f32, 2.0, 3.0, 4.0].iter().map(|v| v * v).sum();
        assert!((rotated_mag - orig_mag).abs() < 1e-4, "Rotation preserves magnitude");

        // Last 4 elements should be unchanged
        assert_eq!(&x[4..], &pass_through[..], "Pass-through dims unchanged");
    }

    #[test]
    fn test_rope_partial_preserves_magnitude() {
        let rope = RotaryEmbedding::new_partial(8, 4, 16, 10000.0);
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mag_before: f32 = x.iter().map(|v| v * v).sum();

        rope.apply(&mut x, 3);

        let mag_after: f32 = x.iter().map(|v| v * v).sum();
        assert!(
            (mag_before - mag_after).abs() < 1e-4,
            "Partial RoPE should preserve total magnitude"
        );
    }
}
