//! 3D rotary position embedding for the Wan DiT.
//!
//! Each attention head's channels are split across the three axes — time,
//! height, width — with **unequal** budgets: height and width each get
//! `2 * (head_dim / 6)` and time takes the remainder. For `head_dim` 128 that
//! is `(44, 42, 42)`: time is wider than either spatial axis, and the split
//! does not divide evenly.
//!
//! Each axis derives its frequencies from *its own* width, so the same channel
//! index means a different frequency depending on which axis owns it.
//!
//! Rotation is applied to **adjacent pairs** `(x[2i], x[2i+1])`, not to a
//! half-split of the head — the reference builds its tables with
//! `repeat_interleave_real=True` and then reads `cos[..., 0::2]` against
//! `sin[..., 1::2]`.

use super::config::WanDitConfig;

/// Cos/sin tables for one axis: `[max_len, dim]`, row-major.
#[derive(Debug, Clone)]
pub struct AxisTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub dim: usize,
    pub max_len: usize,
}

impl AxisTable {
    /// `dim` must be even. Frequencies are `theta^(-2i/dim)`, each repeated
    /// across the pair it rotates.
    fn build(dim: usize, max_len: usize, theta: f32) -> Self {
        assert_eq!(dim % 2, 0, "rope axis dim must be even, got {dim}");
        let mut cos = vec![0.0_f32; max_len * dim];
        let mut sin = vec![0.0_f32; max_len * dim];
        for pos in 0..max_len {
            for i in 0..dim / 2 {
                // f64 for the table: theta^(-2i/dim) loses meaningful
                // precision in f32 at the long end of the frequency range.
                let inv = (theta as f64).powf(-((2 * i) as f64) / dim as f64);
                let ang = pos as f64 * inv;
                let (s, c) = ang.sin_cos();
                cos[pos * dim + 2 * i] = c as f32;
                cos[pos * dim + 2 * i + 1] = c as f32;
                sin[pos * dim + 2 * i] = s as f32;
                sin[pos * dim + 2 * i + 1] = s as f32;
            }
        }
        Self { cos, sin, dim, max_len }
    }

    fn row(&self, pos: usize) -> (&[f32], &[f32]) {
        assert!(pos < self.max_len, "rope position {pos} exceeds table length {}", self.max_len);
        let a = pos * self.dim;
        (&self.cos[a..a + self.dim], &self.sin[a..a + self.dim])
    }
}

/// The three axis tables for one model.
#[derive(Debug, Clone)]
pub struct RopeTables {
    pub t: AxisTable,
    pub h: AxisTable,
    pub w: AxisTable,
    pub head_dim: usize,
}

impl RopeTables {
    pub fn new(cfg: &WanDitConfig, theta: f32) -> Self {
        let (dt, dh, dw) = cfg.rope_dims();
        let n = cfg.rope_max_seq_len;
        Self {
            t: AxisTable::build(dt, n, theta),
            h: AxisTable::build(dh, n, theta),
            w: AxisTable::build(dw, n, theta),
            head_dim: cfg.attention_head_dim,
        }
    }

    /// Cos/sin for the token at grid position `(f, y, x)`, each `head_dim`
    /// long: the time slice, then height, then width.
    pub fn token(&self, f: usize, y: usize, x: usize) -> (Vec<f32>, Vec<f32>) {
        let mut cos = Vec::with_capacity(self.head_dim);
        let mut sin = Vec::with_capacity(self.head_dim);
        for (tab, pos) in [(&self.t, f), (&self.h, y), (&self.w, x)] {
            let (c, s) = tab.row(pos);
            cos.extend_from_slice(c);
            sin.extend_from_slice(s);
        }
        debug_assert_eq!(cos.len(), self.head_dim);
        (cos, sin)
    }
}

/// Rotate one head's channels in place.
///
/// Pairs are adjacent: `(x[2i], x[2i+1])`. `cos` is read at even indices and
/// `sin` at odd — both yield the i-th frequency because the tables repeat each
/// value across its pair, and matching the reference's indexing exactly keeps
/// this correct if the tables ever stop being interleaved.
pub fn apply_rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    debug_assert_eq!(x.len(), cos.len());
    debug_assert_eq!(x.len(), sin.len());
    for i in 0..x.len() / 2 {
        let (x1, x2) = (x[2 * i], x[2 * i + 1]);
        let c = cos[2 * i];
        let s = sin[2 * i + 1];
        x[2 * i] = x1 * c - x2 * s;
        x[2 * i + 1] = x1 * s + x2 * c;
    }
}
