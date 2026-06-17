//! Pixel ↔ patch-token conversion for HiDream-O1.
//!
//! The model works directly on raw pixels: a `[3, H, W]` image is split into
//! non-overlapping `PATCH_SIZE × PATCH_SIZE` patches flattened to
//! `(C·p·p)`-dim tokens. Matches the reference einops:
//!   `B C (H p1) (W p2) -> B (H W) (C p1 p2)`   (p1 = p2 = PATCH_SIZE = 32)
//! with the patch grid enumerated row-major (h outer, w inner).

use super::config::{IN_CHANNELS, PATCH_DIM, PATCH_SIZE};

/// `[C, H, W]` (row-major, C-major) → `[image_len, PATCH_DIM]`.
pub fn patchify(img: &[f32], h: usize, w: usize) -> Vec<f32> {
    let p = PATCH_SIZE;
    let c = IN_CHANNELS;
    debug_assert_eq!(img.len(), c * h * w);
    debug_assert_eq!(h % p, 0);
    debug_assert_eq!(w % p, 0);
    let hp = h / p;
    let wp = w / p;
    let mut out = vec![0.0_f32; hp * wp * PATCH_DIM];
    for prow in 0..hp {
        for pcol in 0..wp {
            let j = prow * wp + pcol;
            let base = j * PATCH_DIM;
            for ci in 0..c {
                for py in 0..p {
                    let src_row = (prow * p + py) * w + pcol * p;
                    let src = ci * h * w + src_row;
                    let dst = base + (ci * p + py) * p;
                    out[dst..dst + p].copy_from_slice(&img[src..src + p]);
                }
            }
        }
    }
    out
}

/// `[image_len, PATCH_DIM]` → `[C, H, W]`. Inverse of [`patchify`].
pub fn unpatchify(patches: &[f32], h: usize, w: usize) -> Vec<f32> {
    let p = PATCH_SIZE;
    let c = IN_CHANNELS;
    let hp = h / p;
    let wp = w / p;
    debug_assert_eq!(patches.len(), hp * wp * PATCH_DIM);
    let mut img = vec![0.0_f32; c * h * w];
    for prow in 0..hp {
        for pcol in 0..wp {
            let j = prow * wp + pcol;
            let base = j * PATCH_DIM;
            for ci in 0..c {
                for py in 0..p {
                    let dst_row = (prow * p + py) * w + pcol * p;
                    let dst = ci * h * w + dst_row;
                    let src = base + (ci * p + py) * p;
                    img[dst..dst + p].copy_from_slice(&patches[src..src + p]);
                }
            }
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_roundtrip() {
        let (h, w) = (64, 96); // 2x3 patch grid
        let img: Vec<f32> = (0..IN_CHANNELS * h * w).map(|i| (i as f32 * 0.001).sin()).collect();
        let patches = patchify(&img, h, w);
        assert_eq!(patches.len(), (h / PATCH_SIZE) * (w / PATCH_SIZE) * PATCH_DIM);
        let back = unpatchify(&patches, h, w);
        assert_eq!(img, back);
    }

    #[test]
    fn patchify_known_layout() {
        // A single 32x32 patch: token 0 holds [C, p, p] contiguous.
        let (h, w) = (PATCH_SIZE, PATCH_SIZE);
        let img: Vec<f32> = (0..IN_CHANNELS * h * w).map(|i| i as f32).collect();
        let patches = patchify(&img, h, w);
        // First pixel of channel 0 → patch[0]; channel 1 starts at p*p.
        assert_eq!(patches[0], img[0]);
        assert_eq!(patches[PATCH_SIZE * PATCH_SIZE], img[h * w]); // channel 1, (0,0)
    }
}
