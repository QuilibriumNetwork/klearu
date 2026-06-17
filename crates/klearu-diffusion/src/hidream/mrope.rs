//! Interleaved multimodal RoPE (mRoPE) for Qwen3-VL.
//!
//! Qwen3-VL carries a 3-axis position id per token `(t, h, w)`: text tokens
//! advance all three together; image tokens encode their patch-grid row/col.
//! The rotary frequencies over the half-head-dim are *interleaved* across the
//! three axes per the `mrope_section` split (default `[24,20,20]`, summing to
//! head_dim/2).
//!
//! Reference `apply_interleaved_mrope`:
//!   start with all axis-0 (T); then for axis d∈{1,2} (H,W) overwrite the
//!   strided slice `idx = offset(d) .. section[d]*3 step 3` with axis-d freqs.
//! With `[24,20,20]` over 64 half-dims this yields the pattern
//!   T H W T H W … (first 60 indices) then T T T T (indices 60..63),
//! giving exactly 24 T / 20 H / 20 W frequencies.

/// For each half-head-dim index, which position axis (0=T, 1=H, 2=W) supplies
/// its rotation angle. `half_dim` must equal head_dim/2; `section` sums to it.
pub fn axis_map(section: &[usize], half_dim: usize) -> Vec<usize> {
    let mut axis = vec![0usize; half_dim]; // default: T
    for (d, offset) in [(1usize, 1usize), (2usize, 2usize)] {
        let length = section[d] * 3;
        let mut k = offset;
        while k < length && k < half_dim {
            axis[k] = d;
            k += 3;
        }
    }
    axis
}

/// Inverse frequencies `theta^(-2i/head_dim)` for i in 0..half_dim.
pub fn inv_freqs(head_dim: usize, theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| theta.powf(-(2.0 * i as f32) / head_dim as f32))
        .collect()
}

/// Precomputed cos/sin tables for a sequence.
///
/// `cos`/`sin` are `[seq_len, head_dim]`, ready to apply with `rotate_half`
/// (the second half mirrors the first, matching `emb = cat(freqs, freqs)`).
pub struct RopeTables {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub head_dim: usize,
}

/// Build cos/sin for every token from its 3-axis position ids.
///
/// `position_ids`: row-major `[3, seq_len]` (axis-major: t row, h row, w row).
pub fn build_tables(
    position_ids: &[i64],
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    section: &[usize],
) -> RopeTables {
    let half = head_dim / 2;
    let invf = inv_freqs(head_dim, theta);
    let axis = axis_map(section, half);

    let mut cos = vec![0.0_f32; seq_len * head_dim];
    let mut sin = vec![0.0_f32; seq_len * head_dim];
    for t in 0..seq_len {
        for k in 0..half {
            let pos = position_ids[axis[k] * seq_len + t] as f32;
            let angle = pos * invf[k];
            let (s, c) = angle.sin_cos();
            // emb = cat(freqs, freqs): index k and k+half share the angle.
            cos[t * head_dim + k] = c;
            cos[t * head_dim + half + k] = c;
            sin[t * head_dim + k] = s;
            sin[t * head_dim + half + k] = s;
        }
    }
    RopeTables { cos, sin, head_dim }
}

/// Apply RoPE in place to `x` laid out `[seq_len, num_heads, head_dim]` using
/// precomputed `cos`/`sin` (`[seq_len, head_dim]`). `rotate_half` convention:
/// `(x·cos) + (rotate_half(x)·sin)` where `rotate_half([a,b]) = [-b,a]` over
/// the two half-dim halves.
pub fn apply(x: &mut [f32], tables: &RopeTables, seq_len: usize, num_heads: usize) {
    let dh = tables.head_dim;
    let half = dh / 2;
    debug_assert_eq!(x.len(), seq_len * num_heads * dh);
    for t in 0..seq_len {
        let cos = &tables.cos[t * dh..(t + 1) * dh];
        let sin = &tables.sin[t * dh..(t + 1) * dh];
        for h in 0..num_heads {
            let off = (t * num_heads + h) * dh;
            let row = &mut x[off..off + dh];
            // rotate_half: first half pairs with negative of second half.
            for i in 0..half {
                let a = row[i];
                let b = row[half + i];
                row[i] = a * cos[i] - b * sin[i];
                row[half + i] = b * cos[half + i] + a * sin[half + i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_map_2420_20() {
        // Qwen3-VL default: [24,20,20] over 64 half-dims.
        let axis = axis_map(&[24, 20, 20], 64);
        assert_eq!(axis.len(), 64);
        // Pattern T H W T H W … for the first 60, then T T T T.
        assert_eq!(axis[0], 0);
        assert_eq!(axis[1], 1);
        assert_eq!(axis[2], 2);
        assert_eq!(axis[3], 0);
        assert_eq!(axis[58], 1);
        assert_eq!(axis[59], 2);
        assert_eq!(axis[60], 0);
        assert_eq!(axis[63], 0);
        // Counts must match the section split.
        let t = axis.iter().filter(|&&a| a == 0).count();
        let h = axis.iter().filter(|&&a| a == 1).count();
        let w = axis.iter().filter(|&&a| a == 2).count();
        assert_eq!((t, h, w), (24, 20, 20));
    }

    #[test]
    fn rope_position_zero_is_identity() {
        // All positions 0 → cos=1, sin=0 → x unchanged.
        let head_dim = 8;
        let seq = 2;
        let pos = vec![0i64; 3 * seq];
        let tables = build_tables(&pos, seq, head_dim, 10000.0, &[2, 1, 1]);
        let mut x: Vec<f32> = (0..seq * head_dim).map(|i| i as f32).collect();
        let orig = x.clone();
        apply(&mut x, &tables, seq, 1);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_preserves_norm() {
        let head_dim = 8;
        let seq = 3;
        // Distinct positions per axis.
        let pos = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let tables = build_tables(&pos, seq, head_dim, 10000.0, &[2, 1, 1]);
        let mut x: Vec<f32> = (0..seq * head_dim).map(|i| (i as f32 * 0.3).sin()).collect();
        let before: f32 = x.iter().map(|v| v * v).sum();
        apply(&mut x, &tables, seq, 1);
        let after: f32 = x.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-4, "norm changed: {before} -> {after}");
    }
}
