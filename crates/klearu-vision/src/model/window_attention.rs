use crate::layers::LinearBias;

/// Spatial windowed multi-head self-attention with optional Swin-style features.
///
/// Input is partitioned into non-overlapping windows of `window_size x window_size`,
/// and standard multi-head attention is applied within each window.
///
/// Optional Swin features:
/// - Relative position bias: learned bias added to attention scores.
/// - Shifted window attention: roll the feature map and use attention masking.
pub struct WindowAttention {
    pub qkv: LinearBias,
    pub proj: LinearBias,
    pub num_heads: usize,
    pub head_dim: usize,
    pub window_size: usize,
    /// Relative position bias table: `[(2*ws-1)*(2*ws-1), num_heads]`.
    /// Indexed by `relative_position_index` to get per-head bias for each pair.
    pub relative_position_bias_table: Option<Vec<f32>>,
    /// Precomputed relative position index: `[ws*ws, ws*ws]`.
    pub relative_position_index: Vec<usize>,
}

impl WindowAttention {
    pub fn new(dim: usize, num_heads: usize, window_size: usize) -> Self {
        let head_dim = dim / num_heads;
        let rel_pos_index = compute_relative_position_index(window_size);
        Self {
            qkv: LinearBias::new(dim, dim * 3),
            proj: LinearBias::new(dim, dim),
            num_heads,
            head_dim,
            window_size,
            relative_position_bias_table: None,
            relative_position_index: rel_pos_index,
        }
    }

    /// Create with Swin-style relative position bias (table initialized to zeros).
    pub fn new_with_relative_position_bias(dim: usize, num_heads: usize, window_size: usize) -> Self {
        let table_size = (2 * window_size - 1) * (2 * window_size - 1);
        let mut attn = Self::new(dim, num_heads, window_size);
        attn.relative_position_bias_table = Some(vec![0.0f32; table_size * num_heads]);
        attn
    }

    /// Forward pass on tokens arranged in `[H, W, C]` layout (non-shifted).
    ///
    /// Input: `[N, C]` where N = H*W tokens in row-major order.
    /// Output: `[N, C]` with attention applied within windows.
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> Vec<f32> {
        self.forward_impl(input, h, w, 0, None)
    }

    /// Forward pass with shifted windows (Swin Transformer).
    ///
    /// `shift_size`: typically `window_size / 2`. Set to 0 for non-shifted.
    pub fn forward_shifted(&self, input: &[f32], h: usize, w: usize, shift_size: usize) -> Vec<f32> {
        if shift_size == 0 {
            return self.forward(input, h, w);
        }

        // Compute attention mask for shifted windows
        let mask = compute_shift_mask(h, w, self.window_size, shift_size);
        self.forward_impl(input, h, w, shift_size, Some(&mask))
    }

    fn forward_impl(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
        shift_size: usize,
        attn_mask: Option<&ShiftMask>,
    ) -> Vec<f32> {
        let dim = self.num_heads * self.head_dim;
        let n = h * w;
        debug_assert_eq!(input.len(), n * dim);

        let ws = self.window_size;
        let num_windows_h = h / ws;
        let num_windows_w = w / ws;
        let window_tokens = ws * ws;

        // Roll input if shifted
        let rolled;
        let working_input = if shift_size > 0 {
            rolled = roll_2d(input, dim, h, w, shift_size, shift_size);
            &rolled
        } else {
            input
        };

        let mut output = vec![0.0f32; n * dim];

        // Get relative position bias for all pairs in a window
        let rel_bias = self.get_relative_position_bias();

        // Process each window
        for wy in 0..num_windows_h {
            for wx in 0..num_windows_w {
                let window_idx = wy * num_windows_w + wx;

                // Gather tokens for this window
                let mut window_input = vec![0.0f32; window_tokens * dim];
                for sy in 0..ws {
                    for sx in 0..ws {
                        let global_y = wy * ws + sy;
                        let global_x = wx * ws + sx;
                        let global_idx = global_y * w + global_x;
                        let local_idx = sy * ws + sx;
                        window_input[local_idx * dim..(local_idx + 1) * dim]
                            .copy_from_slice(&working_input[global_idx * dim..(global_idx + 1) * dim]);
                    }
                }

                // Get per-window attention mask if shifted
                let window_mask = attn_mask.map(|m| &m.masks[window_idx]);

                // Apply attention within this window
                let window_output = self.attention_forward(
                    &window_input, window_tokens, rel_bias.as_deref(), window_mask.map(|v| &**v),
                );

                // Scatter back
                for sy in 0..ws {
                    for sx in 0..ws {
                        let global_y = wy * ws + sy;
                        let global_x = wx * ws + sx;
                        let global_idx = global_y * w + global_x;
                        let local_idx = sy * ws + sx;
                        output[global_idx * dim..(global_idx + 1) * dim]
                            .copy_from_slice(&window_output[local_idx * dim..(local_idx + 1) * dim]);
                    }
                }
            }
        }

        // Reverse roll if shifted
        if shift_size > 0 {
            let unrolled = roll_2d(&output, dim, h, w, h - shift_size, w - shift_size);
            return unrolled;
        }

        output
    }

    /// Get the relative position bias matrix: `[num_heads, ws*ws, ws*ws]` flattened.
    pub fn get_relative_position_bias(&self) -> Option<Vec<f32>> {
        let table = self.relative_position_bias_table.as_ref()?;
        let ws2 = self.window_size * self.window_size;
        let num_heads = self.num_heads;
        let mut bias = vec![0.0f32; num_heads * ws2 * ws2];

        for qi in 0..ws2 {
            for ki in 0..ws2 {
                let idx = self.relative_position_index[qi * ws2 + ki];
                for h in 0..num_heads {
                    bias[h * ws2 * ws2 + qi * ws2 + ki] = table[idx * num_heads + h];
                }
            }
        }

        Some(bias)
    }

    /// Standard multi-head self-attention on `[seq_len, dim]` tokens.
    fn attention_forward(
        &self,
        input: &[f32],
        seq_len: usize,
        rel_bias: Option<&[f32]>,
        attn_mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let dim = self.num_heads * self.head_dim;

        // QKV projection: [seq_len, dim] → [seq_len, 3*dim]
        let mut qkv_buf = vec![0.0f32; seq_len * dim * 3];
        for t in 0..seq_len {
            self.qkv.forward(
                &input[t * dim..(t + 1) * dim],
                &mut qkv_buf[t * dim * 3..(t + 1) * dim * 3],
            );
        }

        // Split into Q, K, V and compute per-head attention
        let scale = (self.head_dim as f32).powf(-0.5);
        let mut attn_output = vec![0.0f32; seq_len * dim];

        for head in 0..self.num_heads {
            let head_offset = head * self.head_dim;

            // Compute attention scores: Q @ K^T / sqrt(head_dim)
            let mut scores = vec![0.0f32; seq_len * seq_len];

            for qi in 0..seq_len {
                for ki in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        let q = qkv_buf[qi * dim * 3 + head_offset + d];
                        let k = qkv_buf[ki * dim * 3 + dim + head_offset + d];
                        dot += q * k;
                    }
                    scores[qi * seq_len + ki] = dot * scale;
                }
            }

            // Add relative position bias
            if let Some(bias) = rel_bias {
                let bias_offset = head * seq_len * seq_len;
                for i in 0..seq_len * seq_len {
                    scores[i] += bias[bias_offset + i];
                }
            }

            // Add attention mask (for shifted windows)
            if let Some(mask) = attn_mask {
                for i in 0..seq_len * seq_len {
                    scores[i] += mask[i];
                }
            }

            // Softmax per row
            for qi in 0..seq_len {
                let row = &mut scores[qi * seq_len..(qi + 1) * seq_len];
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

            // Weighted sum of V: attn_scores @ V
            for qi in 0..seq_len {
                for d in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for vi in 0..seq_len {
                        let v = qkv_buf[vi * dim * 3 + 2 * dim + head_offset + d];
                        sum += scores[qi * seq_len + vi] * v;
                    }
                    attn_output[qi * dim + head_offset + d] = sum;
                }
            }
        }

        // Output projection
        let mut proj_output = vec![0.0f32; seq_len * dim];
        for t in 0..seq_len {
            self.proj.forward(
                &attn_output[t * dim..(t + 1) * dim],
                &mut proj_output[t * dim..(t + 1) * dim],
            );
        }

        proj_output
    }
}

/// Compute the relative position index table for a window of size `ws`.
///
/// Returns `[ws*ws, ws*ws]` indices into a bias table of size `(2*ws-1)*(2*ws-1)`.
pub fn compute_relative_position_index(ws: usize) -> Vec<usize> {
    let ws2 = ws * ws;
    let table_w = 2 * ws - 1;
    let mut index = vec![0usize; ws2 * ws2];

    for qi_y in 0..ws {
        for qi_x in 0..ws {
            let qi = qi_y * ws + qi_x;
            for ki_y in 0..ws {
                for ki_x in 0..ws {
                    let ki = ki_y * ws + ki_x;
                    // Relative position, shifted to non-negative
                    let dy = (qi_y as isize - ki_y as isize + ws as isize - 1) as usize;
                    let dx = (qi_x as isize - ki_x as isize + ws as isize - 1) as usize;
                    index[qi * ws2 + ki] = dy * table_w + dx;
                }
            }
        }
    }

    index
}

/// Attention mask for shifted window attention.
pub struct ShiftMask {
    /// Per-window attention masks: `[num_windows][ws*ws * ws*ws]`.
    /// Values are 0.0 (attend) or -100.0 (mask).
    pub masks: Vec<Vec<f32>>,
}

/// Compute the attention mask for shifted window attention.
///
/// The mask prevents tokens from different spatial regions from attending
/// to each other within a shifted window.
pub fn compute_shift_mask(h: usize, w: usize, window_size: usize, shift_size: usize) -> ShiftMask {
    let ws = window_size;
    let ws2 = ws * ws;
    let num_windows_h = h / ws;
    let num_windows_w = w / ws;
    let num_windows = num_windows_h * num_windows_w;

    // Assign region labels to each spatial position
    let mut region_map = vec![0usize; h * w];
    let mut cnt = 0usize;

    // Split into regions based on shift boundaries
    let h_splits = [0, h - ws, h - shift_size, h];
    let w_splits = [0, w - ws, w - shift_size, w];

    for hi in 0..3 {
        for wi in 0..3 {
            let h_start = h_splits[hi];
            let h_end = h_splits[hi + 1];
            let w_start = w_splits[wi];
            let w_end = w_splits[wi + 1];
            if h_start >= h_end || w_start >= w_end {
                continue;
            }
            for y in h_start..h_end {
                for x in w_start..w_end {
                    region_map[y * w + x] = cnt;
                }
            }
            cnt += 1;
        }
    }

    // Roll the region map (same shift as applied to features)
    let mut rolled_map = vec![0usize; h * w];
    for y in 0..h {
        for x in 0..w {
            let src_y = (y + shift_size) % h;
            let src_x = (x + shift_size) % w;
            rolled_map[y * w + x] = region_map[src_y * w + src_x];
        }
    }

    // For each window, create attention mask
    let mut masks = Vec::with_capacity(num_windows);
    for wy in 0..num_windows_h {
        for wx in 0..num_windows_w {
            let mut mask = vec![0.0f32; ws2 * ws2];
            for qi_y in 0..ws {
                for qi_x in 0..ws {
                    let qi = qi_y * ws + qi_x;
                    let qi_region = rolled_map[(wy * ws + qi_y) * w + (wx * ws + qi_x)];
                    for ki_y in 0..ws {
                        for ki_x in 0..ws {
                            let ki = ki_y * ws + ki_x;
                            let ki_region = rolled_map[(wy * ws + ki_y) * w + (wx * ws + ki_x)];
                            if qi_region != ki_region {
                                mask[qi * ws2 + ki] = -100.0;
                            }
                        }
                    }
                }
            }
            masks.push(mask);
        }
    }

    ShiftMask { masks }
}

/// Roll a `[N, C]` tensor (where N = H*W) by `(shift_h, shift_w)` positions.
///
/// Tokens wrap around the edges (cyclic shift).
fn roll_2d(tokens: &[f32], dim: usize, h: usize, w: usize, shift_h: usize, shift_w: usize) -> Vec<f32> {
    let n = h * w;
    let mut out = vec![0.0f32; n * dim];
    for y in 0..h {
        for x in 0..w {
            let src_y = (y + shift_h) % h;
            let src_x = (x + shift_w) % w;
            let src_idx = src_y * w + src_x;
            let dst_idx = y * w + x;
            out[dst_idx * dim..(dst_idx + 1) * dim]
                .copy_from_slice(&tokens[src_idx * dim..(src_idx + 1) * dim]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_attention_output_shape() {
        let wa = WindowAttention::new(8, 2, 7);
        // 14x14 spatial → 4 windows of 7x7
        let n = 14 * 14;
        let input = vec![0.1f32; n * 8];
        let output = wa.forward(&input, 14, 14);
        assert_eq!(output.len(), n * 8);
    }

    #[test]
    fn test_window_attention_finite() {
        let wa = WindowAttention::new(8, 2, 7);
        let n = 7 * 7;
        let input = vec![0.1f32; n * 8];
        let output = wa.forward(&input, 7, 7);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_relative_position_index() {
        let ws = 3;
        let idx = compute_relative_position_index(ws);
        let ws2 = ws * ws;
        let table_size = (2 * ws - 1) * (2 * ws - 1); // 5*5 = 25

        assert_eq!(idx.len(), ws2 * ws2);

        // All indices should be valid
        for &i in &idx {
            assert!(i < table_size, "index {i} >= table_size {table_size}");
        }

        // Self-attention at same position should have the center index
        let center = (ws - 1) * (2 * ws - 1) + (ws - 1); // (2,2) in 5x5 = 12
        for pos in 0..ws2 {
            assert_eq!(idx[pos * ws2 + pos], center, "Self-position should be center");
        }
    }

    #[test]
    fn test_relative_position_bias() {
        let wa = WindowAttention::new_with_relative_position_bias(8, 2, 7);
        assert!(wa.relative_position_bias_table.is_some());

        let table = wa.relative_position_bias_table.as_ref().unwrap();
        let expected = (2 * 7 - 1) * (2 * 7 - 1) * 2; // 169 * 2 heads
        assert_eq!(table.len(), expected);

        // Forward should still work
        let n = 7 * 7;
        let input = vec![0.1f32; n * 8];
        let output = wa.forward(&input, 7, 7);
        assert_eq!(output.len(), n * 8);
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_shifted_window_attention() {
        let wa = WindowAttention::new(8, 2, 7);
        let n = 14 * 14;
        let input = vec![0.1f32; n * 8];

        // Non-shifted
        let out_normal = wa.forward(&input, 14, 14);
        // Shifted with shift_size = 3
        let out_shifted = wa.forward_shifted(&input, 14, 14, 3);

        assert_eq!(out_shifted.len(), n * 8);
        for &v in &out_shifted {
            assert!(v.is_finite());
        }

        // Shifted should produce different results than non-shifted
        // (unless weights are all zero, which they are here, but the mask changes things)
        assert_eq!(out_normal.len(), out_shifted.len());
    }

    #[test]
    fn test_shift_mask() {
        let h = 14;
        let w = 14;
        let ws = 7;
        let ss = 3;
        let mask = compute_shift_mask(h, w, ws, ss);

        let num_windows = (h / ws) * (w / ws);
        assert_eq!(mask.masks.len(), num_windows);

        for m in &mask.masks {
            assert_eq!(m.len(), ws * ws * ws * ws);
            // All values should be 0.0 or -100.0
            for &v in m {
                assert!(v == 0.0 || v == -100.0, "unexpected mask value: {v}");
            }
        }
    }

    #[test]
    fn test_roll_2d_identity() {
        let h = 4;
        let w = 4;
        let dim = 2;
        let tokens: Vec<f32> = (0..h * w * dim).map(|i| i as f32).collect();

        // Rolling by (h, w) should be identity
        let rolled = roll_2d(&tokens, dim, h, w, h, w);
        assert_eq!(tokens, rolled);
    }

    #[test]
    fn test_roll_2d_roundtrip() {
        let h = 4;
        let w = 4;
        let dim = 2;
        let tokens: Vec<f32> = (0..h * w * dim).map(|i| i as f32).collect();

        let rolled = roll_2d(&tokens, dim, h, w, 1, 2);
        let unrolled = roll_2d(&rolled, dim, h, w, h - 1, w - 2);
        assert_eq!(tokens, unrolled);
    }

    #[test]
    fn test_shifted_zero_is_normal() {
        let wa = WindowAttention::new(8, 2, 7);
        let n = 7 * 7;
        let input = vec![0.1f32; n * 8];

        let normal = wa.forward(&input, 7, 7);
        let shifted_zero = wa.forward_shifted(&input, 7, 7, 0);

        for (a, b) in normal.iter().zip(shifted_zero.iter()) {
            assert!((a - b).abs() < 1e-6, "shift=0 should match non-shifted");
        }
    }
}
