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
