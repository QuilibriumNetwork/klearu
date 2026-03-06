/// Image preprocessing pipeline for vision models.
///
/// Supports standard timm evaluation pipeline:
/// 1. Resize shorter edge to `resize_size` (preserving aspect ratio)
/// 2. Center-crop to `input_size x input_size`
/// 3. Normalize with per-channel mean/std

/// Resize strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Resize shorter edge to target, preserve aspect ratio (default timm).
    ShortEdge,
    /// Resize longer edge to target, preserve aspect ratio.
    LongEdge,
    /// Resize to exact dimensions (may distort aspect ratio).
    Exact,
}

/// Interpolation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Bilinear,
    Bicubic,
}

/// Preprocessing configuration.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Target input size after cropping (e.g., 224).
    pub input_size: usize,
    /// Size to resize to before cropping (e.g., 256).
    pub resize_size: usize,
    /// Crop percentage: `resize_size = input_size / crop_pct` (e.g., 0.875).
    pub crop_pct: f32,
    /// Per-channel mean for normalization.
    pub mean: [f32; 3],
    /// Per-channel std for normalization.
    pub std: [f32; 3],
    /// Interpolation method for resizing.
    pub interpolation: Interpolation,
    /// Resize strategy.
    pub resize_mode: ResizeMode,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            input_size: 224,
            resize_size: 256,
            crop_pct: 0.875,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            interpolation: Interpolation::Bicubic,
            resize_mode: ResizeMode::ShortEdge,
        }
    }
}

impl PreprocessConfig {
    /// Standard ImageNet config (matching most timm models).
    pub fn imagenet() -> Self {
        Self::default()
    }

    /// CLIP/SigLIP normalization constants.
    pub fn clip() -> Self {
        Self {
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
            ..Self::default()
        }
    }

    /// Qwen3.5 VL normalization (mean=0.5, std=0.5, no crop).
    pub fn qwen() -> Self {
        Self {
            input_size: 224,
            resize_size: 224,
            crop_pct: 1.0,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            interpolation: Interpolation::Bicubic,
            resize_mode: ResizeMode::ShortEdge,
        }
    }

    /// Inception-style normalization (mean=0.5, std=0.5).
    pub fn inception() -> Self {
        Self {
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            ..Self::default()
        }
    }

    /// Update resize_size from crop_pct.
    pub fn with_crop_pct(mut self, crop_pct: f32) -> Self {
        self.crop_pct = crop_pct;
        self.resize_size = (self.input_size as f32 / crop_pct).ceil() as usize;
        self
    }

    /// Parse from a timm `pretrained_cfg` JSON object.
    ///
    /// Expected fields (all optional, with defaults):
    /// - `input_size`: `[3, H, W]` array
    /// - `crop_pct`: float
    /// - `mean`: `[r, g, b]`
    /// - `std`: `[r, g, b]`
    /// - `interpolation`: `"bilinear"` or `"bicubic"`
    pub fn from_pretrained_cfg(cfg: &serde_json::Value) -> Self {
        let mut result = Self::default();

        if let Some(input_size) = cfg.get("input_size").and_then(|v| v.as_array()) {
            // timm format: [channels, height, width]
            if input_size.len() == 3 {
                if let Some(h) = input_size[1].as_u64() {
                    result.input_size = h as usize;
                }
            }
        }

        if let Some(crop_pct) = cfg.get("crop_pct").and_then(|v| v.as_f64()) {
            result.crop_pct = crop_pct as f32;
        }

        result.resize_size = (result.input_size as f32 / result.crop_pct).ceil() as usize;

        if let Some(mean) = cfg.get("mean").and_then(|v| v.as_array()) {
            if mean.len() == 3 {
                for (i, v) in mean.iter().enumerate() {
                    if let Some(f) = v.as_f64() {
                        result.mean[i] = f as f32;
                    }
                }
            }
        }

        if let Some(std) = cfg.get("std").and_then(|v| v.as_array()) {
            if std.len() == 3 {
                for (i, v) in std.iter().enumerate() {
                    if let Some(f) = v.as_f64() {
                        result.std[i] = f as f32;
                    }
                }
            }
        }

        if let Some(interp) = cfg.get("interpolation").and_then(|v| v.as_str()) {
            result.interpolation = match interp {
                "bilinear" => Interpolation::Bilinear,
                "bicubic" => Interpolation::Bicubic,
                _ => Interpolation::Bicubic,
            };
        }

        result
    }

    /// Parse from a full timm config.json, extracting `pretrained_cfg` if present.
    pub fn from_model_config(json: &str) -> Self {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(json) {
            if let Some(cfg) = v.get("pretrained_cfg") {
                return Self::from_pretrained_cfg(cfg);
            }
        }
        Self::default()
    }
}

/// Normalize a `[C, H, W]` tensor in-place with per-channel mean and std.
pub fn normalize(tensor: &mut [f32], channels: usize, h: usize, w: usize, mean: &[f32], std: &[f32]) {
    let spatial = h * w;
    debug_assert_eq!(tensor.len(), channels * spatial);
    for c in 0..channels {
        let m = mean[c];
        let s = std[c];
        let base = c * spatial;
        for i in 0..spatial {
            tensor[base + i] = (tensor[base + i] - m) / s;
        }
    }
}

/// Center-crop a `[C, H, W]` tensor to `[C, crop_h, crop_w]`.
///
/// Returns the cropped tensor.
pub fn center_crop(tensor: &[f32], channels: usize, h: usize, w: usize, crop_h: usize, crop_w: usize) -> Vec<f32> {
    assert!(crop_h <= h && crop_w <= w, "Crop size must be <= input size");
    let y_offset = (h - crop_h) / 2;
    let x_offset = (w - crop_w) / 2;

    let mut out = vec![0.0f32; channels * crop_h * crop_w];
    for c in 0..channels {
        for y in 0..crop_h {
            for x in 0..crop_w {
                let src_idx = c * h * w + (y + y_offset) * w + (x + x_offset);
                let dst_idx = c * crop_h * crop_w + y * crop_w + x;
                out[dst_idx] = tensor[src_idx];
            }
        }
    }
    out
}

/// Horizontal flip of a `[C, H, W]` tensor.
pub fn horizontal_flip(tensor: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tensor.len()];
    for c in 0..channels {
        for y in 0..h {
            for x in 0..w {
                let src = c * h * w + y * w + x;
                let dst = c * h * w + y * w + (w - 1 - x);
                out[dst] = tensor[src];
            }
        }
    }
    out
}

/// Convert raw RGB bytes `[H, W, 3]` (row-major) to `[3, H, W]` f32 tensor scaled to [0, 1].
pub fn rgb_bytes_to_chw(rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
    let mut tensor = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel_idx = (y * w + x) * 3;
            for c in 0..3 {
                tensor[c * h * w + y * w + x] = rgb[pixel_idx + c] as f32 / 255.0;
            }
        }
    }
    tensor
}

/// Full preprocessing pipeline on a `[C, H, W]` f32 tensor (already in [0,1] range).
///
/// Applies center crop (if tensor is larger than input_size) and normalization.
pub fn preprocess_tensor(
    tensor: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    config: &PreprocessConfig,
) -> Vec<f32> {
    let target = config.input_size;
    let mut result = if h > target || w > target {
        center_crop(tensor, channels, h, w, target, target)
    } else {
        tensor.to_vec()
    };
    normalize(&mut result, channels, target, target, &config.mean, &config.std);
    result
}

/// Compute the target resolution for Qwen3.5 VL dynamic resolution.
///
/// Maintains aspect ratio, rounds both dimensions to nearest multiple of
/// `factor = patch_size * merge_size`, and stays within `min_pixels..max_pixels`.
///
/// Returns `(new_h, new_w)`.
pub fn qwen_smart_resize(
    h: usize,
    w: usize,
    patch_size: usize,
    merge_size: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let factor = patch_size * merge_size;
    let pixels = h * w;

    // Clamp total pixels
    let (mut new_h, mut new_w) = if pixels < min_pixels {
        let scale = (min_pixels as f64 / pixels as f64).sqrt();
        ((h as f64 * scale) as usize, (w as f64 * scale) as usize)
    } else if pixels > max_pixels {
        let scale = (max_pixels as f64 / pixels as f64).sqrt();
        ((h as f64 * scale) as usize, (w as f64 * scale) as usize)
    } else {
        (h, w)
    };

    // Round to nearest multiple of factor
    new_h = ((new_h as f64 / factor as f64).round() as usize).max(1) * factor;
    new_w = ((new_w as f64 / factor as f64).round() as usize).max(1) * factor;

    (new_h, new_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let mut tensor = vec![0.5f32; 3 * 4 * 4];
        normalize(&mut tensor, 3, 4, 4, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]);
        for &v in &tensor {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_center_crop() {
        // 2 channels, 4x4 → crop to 2x2
        let tensor: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let cropped = center_crop(&tensor, 2, 4, 4, 2, 2);
        assert_eq!(cropped.len(), 2 * 2 * 2);
        // Channel 0, center 2x2 of 4x4: rows 1-2, cols 1-2
        assert_eq!(cropped[0], 5.0);  // (1,1)
        assert_eq!(cropped[1], 6.0);  // (1,2)
        assert_eq!(cropped[2], 9.0);  // (2,1)
        assert_eq!(cropped[3], 10.0); // (2,2)
    }

    #[test]
    fn test_horizontal_flip() {
        // 1 channel, 2x3
        let tensor = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let flipped = horizontal_flip(&tensor, 1, 2, 3);
        assert_eq!(flipped, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_horizontal_flip_roundtrip() {
        let tensor = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let flipped = horizontal_flip(&tensor, 1, 2, 3);
        let restored = horizontal_flip(&flipped, 1, 2, 3);
        assert_eq!(tensor, restored);
    }

    #[test]
    fn test_rgb_bytes_to_chw() {
        // 2x2 image: red, green, blue, white
        let rgb = vec![
            255, 0, 0,     0, 255, 0,
            0, 0, 255,     255, 255, 255,
        ];
        let tensor = rgb_bytes_to_chw(&rgb, 2, 2);
        assert_eq!(tensor.len(), 3 * 2 * 2);
        // Channel 0 (R): 1.0, 0.0, 0.0, 1.0
        assert!((tensor[0] - 1.0).abs() < 1e-5);
        assert!((tensor[1] - 0.0).abs() < 1e-5);
        assert!((tensor[2] - 0.0).abs() < 1e-5);
        assert!((tensor[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_from_pretrained_cfg() {
        let json = serde_json::json!({
            "input_size": [3, 384, 384],
            "crop_pct": 1.0,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "interpolation": "bicubic"
        });
        let config = PreprocessConfig::from_pretrained_cfg(&json);
        assert_eq!(config.input_size, 384);
        assert_eq!(config.resize_size, 384); // crop_pct=1.0 → same size
        assert_eq!(config.mean, [0.5, 0.5, 0.5]);
        assert!(matches!(config.interpolation, Interpolation::Bicubic));
    }

    #[test]
    fn test_default_config() {
        let config = PreprocessConfig::default();
        assert_eq!(config.input_size, 224);
        assert_eq!(config.resize_size, 256);
        assert!((config.crop_pct - 0.875).abs() < 1e-5);
    }

    #[test]
    fn test_with_crop_pct() {
        let config = PreprocessConfig::default().with_crop_pct(1.0);
        assert_eq!(config.resize_size, 224); // 224 / 1.0 = 224
    }

    #[test]
    fn test_qwen_config() {
        let config = PreprocessConfig::qwen();
        assert_eq!(config.mean, [0.5, 0.5, 0.5]);
        assert_eq!(config.std, [0.5, 0.5, 0.5]);
        assert_eq!(config.crop_pct, 1.0);
    }

    #[test]
    fn test_qwen_smart_resize() {
        // 224x224 with factor=32 → stays 224x224 (already multiple of 32 → rounds to 224)
        let (h, w) = qwen_smart_resize(224, 224, 16, 2, 256 * 256, 1280 * 1280);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);

        // Very small image gets upscaled
        let (h, w) = qwen_smart_resize(50, 50, 16, 2, 256 * 256, 1280 * 1280);
        assert!(h * w >= 256 * 256 / 2); // approximately at min
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);

        // Very large image gets downscaled
        let (h, w) = qwen_smart_resize(2000, 2000, 16, 2, 256 * 256, 1280 * 1280);
        assert!(h * w <= 1280 * 1280 * 2); // approximately at max
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }
}
