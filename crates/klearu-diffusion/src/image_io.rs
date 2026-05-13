//! Image I/O — convert VAE-decoded tensors to PNG files.
//!
//! VAE output is f32 [N, 3, H, W] in roughly [-1, 1]. We clamp/scale
//! to [0, 255] u8 and write a PNG via the `image` crate (cli feature).

use std::path::Path;

use crate::error::{DiffusionError, Result};

/// Save a single-image VAE output [3, H, W] as a PNG.
/// Pixel values are clamped from [-1, 1] (VAE convention) into [0, 255].
#[cfg(feature = "cli")]
pub fn save_png(image_chw: &[f32], h: usize, w: usize, path: &Path) -> Result<()> {
    if image_chw.len() != 3 * h * w {
        return Err(DiffusionError::ShapeMismatch {
            expected: format!("3*{h}*{w} = {}", 3 * h * w),
            got: format!("{}", image_chw.len()),
        });
    }
    // CHW → HWC u8 with [-1, 1] → [0, 255].
    let mut buf = vec![0u8; 3 * h * w];
    for hi in 0..h {
        for wi in 0..w {
            for ci in 0..3 {
                let v = image_chw[ci * h * w + hi * w + wi];
                let scaled = ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                buf[(hi * w + wi) * 3 + ci] = scaled;
            }
        }
    }
    let img = image::RgbImage::from_raw(w as u32, h as u32, buf)
        .ok_or_else(|| DiffusionError::Unsupported("failed to construct RgbImage".into()))?;
    img.save(path)
        .map_err(|e| DiffusionError::Unsupported(format!("PNG save: {e}")))?;
    Ok(())
}

#[cfg(not(feature = "cli"))]
pub fn save_png(_image_chw: &[f32], _h: usize, _w: usize, _path: &Path) -> Result<()> {
    Err(DiffusionError::Unsupported(
        "PNG output requires the 'cli' feature (build with --features cli)".into(),
    ))
}

/// Load a PNG (or any format `image` supports) into a `[3, H, W]` `Vec<f32>`
/// in **`[-1, 1]`** (matching VAE input/output convention). The returned
/// shape is `expected_h × expected_w`; if the image's native dimensions
/// don't match, this errors — caller must pre-resize.
///
/// Grayscale inputs are broadcast to all three channels. RGBA discards
/// alpha. RGB is taken as-is.
#[cfg(feature = "cli")]
pub fn load_image_as_chw_f32(
    path: &Path,
    expected_h: usize,
    expected_w: usize,
) -> Result<Vec<f32>> {
    let img = image::open(path)
        .map_err(|e| DiffusionError::Unsupported(format!("read image {path:?}: {e}")))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if h != expected_h || w != expected_w {
        return Err(DiffusionError::ShapeMismatch {
            expected: format!("{expected_h}×{expected_w} (resize before loading)"),
            got: format!("{h}×{w}"),
        });
    }
    let mut out = vec![0.0_f32; 3 * h * w];
    for hi in 0..h {
        for wi in 0..w {
            let p = rgb.get_pixel(wi as u32, hi as u32);
            for c in 0..3 {
                // [0, 255] → [-1, 1]
                out[c * h * w + hi * w + wi] = (p[c] as f32) / 127.5 - 1.0;
            }
        }
    }
    Ok(out)
}

#[cfg(not(feature = "cli"))]
pub fn load_image_as_chw_f32(
    _path: &Path, _expected_h: usize, _expected_w: usize,
) -> Result<Vec<f32>> {
    Err(DiffusionError::Unsupported(
        "image input requires the 'cli' feature (build with --features cli)".into(),
    ))
}

/// Load a mask image into a `[H, W]` `Vec<f32>` in **`[0, 1]`**. White
/// (255) maps to 1.0 (= "regenerate this pixel"), black (0) maps to 0.0
/// (= "preserve original"). Soft greys interpolate. RGB inputs are
/// averaged across channels; alpha is ignored.
#[cfg(feature = "cli")]
pub fn load_mask(
    path: &Path,
    expected_h: usize,
    expected_w: usize,
) -> Result<Vec<f32>> {
    let img = image::open(path)
        .map_err(|e| DiffusionError::Unsupported(format!("read mask {path:?}: {e}")))?;
    let gray = img.to_luma8();
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if h != expected_h || w != expected_w {
        return Err(DiffusionError::ShapeMismatch {
            expected: format!("{expected_h}×{expected_w} (resize mask before loading)"),
            got: format!("{h}×{w}"),
        });
    }
    let mut out = vec![0.0_f32; h * w];
    for hi in 0..h {
        for wi in 0..w {
            out[hi * w + wi] = gray.get_pixel(wi as u32, hi as u32)[0] as f32 / 255.0;
        }
    }
    Ok(out)
}

#[cfg(not(feature = "cli"))]
pub fn load_mask(_path: &Path, _expected_h: usize, _expected_w: usize) -> Result<Vec<f32>> {
    Err(DiffusionError::Unsupported(
        "mask input requires the 'cli' feature (build with --features cli)".into(),
    ))
}

/// Box-average downsample a single-channel mask `[H, W]` by integer
/// factor `f`. Used to project pixel-resolution masks down to latent
/// resolution (typically 8× for SD/SDXL/Flux VAEs). Output is `[H/f, W/f]`.
///
/// We intentionally average rather than nearest-sample so soft masks
/// stay soft after downsampling (a 50% gradient at pixel resolution
/// stays a 50% gradient at latent resolution).
pub fn downsample_mask(mask: &[f32], h: usize, w: usize, factor: usize) -> Result<Vec<f32>> {
    if !h.is_multiple_of(factor) || !w.is_multiple_of(factor) {
        return Err(DiffusionError::ShapeMismatch {
            expected: format!("h={h}, w={w} divisible by factor {factor}"),
            got: format!("h%{factor}={}, w%{factor}={}", h % factor, w % factor),
        });
    }
    let oh = h / factor;
    let ow = w / factor;
    let mut out = vec![0.0_f32; oh * ow];
    let inv = 1.0 / (factor * factor) as f32;
    for oh_i in 0..oh {
        for ow_i in 0..ow {
            let mut s = 0.0_f32;
            for dy in 0..factor {
                for dx in 0..factor {
                    let hi = oh_i * factor + dy;
                    let wi = ow_i * factor + dx;
                    s += mask[hi * w + wi];
                }
            }
            out[oh_i * ow + ow_i] = s * inv;
        }
    }
    Ok(out)
}
