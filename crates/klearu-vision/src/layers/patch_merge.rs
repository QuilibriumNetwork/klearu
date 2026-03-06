/// Patch merging layer (Swin Transformer downsampling).
///
/// Groups 2x2 adjacent tokens and projects to 2x the channel dimension.
/// `[H, W, C]` → `[H/2, W/2, 2*C]`.

use crate::layers::{LayerNorm, LinearBias};

pub struct PatchMerge {
    pub norm: LayerNorm,
    pub reduction: LinearBias,
}

impl PatchMerge {
    /// `dim`: input channel dimension. Output dimension is `2 * dim`.
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            norm: LayerNorm::new(4 * dim, eps),
            reduction: LinearBias::new(4 * dim, 2 * dim),
        }
    }

    /// Forward: `[H*W, dim]` → `[H/2*W/2, 2*dim]`.
    pub fn forward(&self, input: &[f32], h: usize, w: usize, dim: usize) -> Vec<f32> {
        let new_h = h / 2;
        let new_w = w / 2;
        let num_out = new_h * new_w;
        let merged_dim = 4 * dim;
        let out_dim = 2 * dim;

        let mut merged = vec![0.0f32; num_out * merged_dim];

        // Gather 2x2 patches: concat [top-left, top-right, bottom-left, bottom-right]
        for y in 0..new_h {
            for x in 0..new_w {
                let out_idx = y * new_w + x;
                let offsets = [
                    (2 * y) * w + (2 * x),         // top-left
                    (2 * y) * w + (2 * x + 1),     // top-right
                    (2 * y + 1) * w + (2 * x),     // bottom-left
                    (2 * y + 1) * w + (2 * x + 1), // bottom-right
                ];
                for (i, &off) in offsets.iter().enumerate() {
                    merged[out_idx * merged_dim + i * dim..out_idx * merged_dim + (i + 1) * dim]
                        .copy_from_slice(&input[off * dim..(off + 1) * dim]);
                }
            }
        }

        // LayerNorm → Linear reduction
        let mut output = vec![0.0f32; num_out * out_dim];
        for t in 0..num_out {
            let token = &mut merged[t * merged_dim..(t + 1) * merged_dim];
            self.norm.forward(token);
            self.reduction.forward(token, &mut output[t * out_dim..(t + 1) * out_dim]);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_merge() {
        let pm = PatchMerge::new(8, 1e-5);
        // 4x4 grid → 2x2 grid
        let input = vec![0.1f32; 16 * 8];
        let output = pm.forward(&input, 4, 4, 8);
        assert_eq!(output.len(), 4 * 16); // 4 tokens × 16 dim
        for &v in &output {
            assert!(v.is_finite());
        }
    }
}
