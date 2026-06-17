//! Qwen2-VL / Qwen3-VL image preprocessing for the reference-editing vision
//! encoder. Produces `pixel_values [n_patches, C*T*patch*patch]` in merge-block
//! order plus the `(t, h, w)` grid (patch units), matching
//! `Qwen2VLImageProcessorFast`.
//!
//! Pipeline: smart-resize the image to multiples of `patch_size*merge_size`
//! within the pixel budget → bilinear resample → normalize to `[-1, 1]`
//! (mean/std = 0.5) → patchify with the Qwen merge-block reshape (each
//! `merge×merge` spatial block's tokens are contiguous), repeating the single
//! frame `temporal_patch_size` times.

/// HiDream-O1's `preprocessor_config.json`: patch 16, temporal 2, merge 2,
/// mean/std 0.5, pixel budget [65536, 16777216].
pub const VLM_PATCH: usize = 16;
pub const VLM_TEMPORAL: usize = 2;
pub const VLM_MERGE: usize = 2;
pub const VLM_MIN_PIXELS: usize = 65536;
pub const VLM_MAX_PIXELS: usize = 16_777_216;

/// Qwen2-VL `smart_resize`: round (h, w) to multiples of `factor`, keeping the
/// area within `[min_px, max_px]` and preserving aspect ratio. Returns the
/// resized `(h, w)` in pixels.
pub fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    min_px: usize,
    max_px: usize,
) -> (usize, usize) {
    let (hf, wf) = (h as f64, w as f64);
    let round_to = |x: f64| -> usize { ((x / factor as f64).round() as usize).max(1) * factor };
    let mut hb = round_to(hf);
    let mut wb = round_to(wf);
    if hb * wb > max_px {
        let beta = (hf * wf / max_px as f64).sqrt();
        hb = (((hf / beta) / factor as f64).floor() as usize).max(1) * factor;
        wb = (((wf / beta) / factor as f64).floor() as usize).max(1) * factor;
    } else if hb * wb < min_px {
        let beta = (min_px as f64 / (hf * wf)).sqrt();
        hb = ((hf * beta / factor as f64).ceil() as usize).max(1) * factor;
        wb = ((wf * beta / factor as f64).ceil() as usize).max(1) * factor;
    }
    (hb, wb)
}

/// Bilinear resample a CHW float image `[c, h, w]` to `[c, out_h, out_w]` using
/// half-pixel centers (PyTorch/`torchvision` `align_corners=False` convention).
pub fn resize_chw_bilinear(
    src: &[f32],
    c: usize,
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0_f32; c * out_h * out_w];
    let sy = h as f32 / out_h as f32;
    let sx = w as f32 / out_w as f32;
    for ch in 0..c {
        let sp = &src[ch * h * w..(ch + 1) * h * w];
        let dp = &mut dst[ch * out_h * out_w..(ch + 1) * out_h * out_w];
        for oy in 0..out_h {
            let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let wy = fy - y0 as f32;
            for ox in 0..out_w {
                let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let wx = fx - x0 as f32;
                let v00 = sp[y0 * w + x0];
                let v01 = sp[y0 * w + x1];
                let v10 = sp[y1 * w + x0];
                let v11 = sp[y1 * w + x1];
                let top = v00 + (v01 - v00) * wx;
                let bot = v10 + (v11 - v10) * wx;
                dp[oy * out_w + ox] = top + (bot - top) * wy;
            }
        }
    }
    dst
}

/// Patchify an already-resized, already-normalized CHW float image into the
/// Qwen merge-block token layout. Mirrors `Qwen2VLImageProcessor`'s reshape:
/// reshape `[gt, T, C, gh/m, m, p, gw/m, m, p]` → transpose
/// `(0,3,6,4,7,2,1,5,8)` → flatten to `[gt*gh*gw, C*T*p*p]`. The single frame is
/// repeated `temporal` times. Returns `(pixel_values, (t, h, w))` (grid in
/// patch units).
pub fn patchify_merge_block(
    chw: &[f32],
    c: usize,
    h: usize,
    w: usize,
    patch: usize,
    temporal: usize,
    merge: usize,
) -> (Vec<f32>, (usize, usize, usize)) {
    let gh = h / patch;
    let gw = w / patch;
    let grid_t = 1usize;
    let feat = c * temporal * patch * patch;
    let mut out = vec![0.0_f32; grid_t * gh * gw * feat];
    let mut tok = 0usize;
    for ghb in 0..gh / merge {
        for gwb in 0..gw / merge {
            for mh in 0..merge {
                for mw in 0..merge {
                    let gh_idx = ghb * merge + mh;
                    let gw_idx = gwb * merge + mw;
                    let base = tok * feat;
                    let mut f = 0usize;
                    for ch in 0..c {
                        let plane = &chw[ch * h * w..(ch + 1) * h * w];
                        for _t in 0..temporal {
                            for ph in 0..patch {
                                let y = gh_idx * patch + ph;
                                for pw in 0..patch {
                                    let x = gw_idx * patch + pw;
                                    out[base + f] = plane[y * w + x];
                                    f += 1;
                                }
                            }
                        }
                    }
                    tok += 1;
                }
            }
        }
    }
    (out, (grid_t, gh, gw))
}

/// Full preprocessing for one reference image given a CHW float image in
/// `[0, 1]`. Smart-resizes, resamples, normalizes to `[-1, 1]`, and patchifies.
/// Returns `(pixel_values, (t, h, w))`.
pub fn preprocess_vlm_image(chw01: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, (usize, usize, usize)) {
    let factor = VLM_PATCH * VLM_MERGE;
    let (rh, rw) = smart_resize(h, w, factor, VLM_MIN_PIXELS, VLM_MAX_PIXELS);
    let resized = resize_chw_bilinear(chw01, c, h, w, rh, rw);
    let mut norm = resized;
    for v in norm.iter_mut() {
        *v = (*v - 0.5) / 0.5; // mean/std = 0.5 → [-1, 1]
    }
    patchify_merge_block(&norm, c, rh, rw, VLM_PATCH, VLM_TEMPORAL, VLM_MERGE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_rounds_to_factor() {
        // Within budget: round to nearest multiple of 32.
        assert_eq!(smart_resize(384, 384, 32, VLM_MIN_PIXELS, VLM_MAX_PIXELS), (384, 384));
        assert_eq!(smart_resize(300, 500, 32, VLM_MIN_PIXELS, VLM_MAX_PIXELS), (288, 512));
        // Below min_pixels (256² = 65536): scale up.
        let (h, w) = smart_resize(64, 64, 32, VLM_MIN_PIXELS, VLM_MAX_PIXELS);
        assert!(h * w >= VLM_MIN_PIXELS && h % 32 == 0 && w % 32 == 0);
    }

    #[test]
    fn patchify_token_and_feature_order() {
        // 1 channel, 2x2 patches (patch=1, merge=2, temporal=1) so the layout is
        // trivially checkable: values 0,1,2,3 at (y,x) row-major.
        let chw = vec![0.0, 1.0, 2.0, 3.0]; // [1,2,2]
        let (out, grid) = patchify_merge_block(&chw, 1, 2, 2, 1, 1, 2);
        assert_eq!(grid, (1, 2, 2));
        // One merge block, order (mh, mw): (0,0)=val0, (0,1)=val1, (1,0)=val2, (1,1)=val3.
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0]);
    }
}
