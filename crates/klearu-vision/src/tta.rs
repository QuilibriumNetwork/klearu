/// Test-time augmentation (TTA) for vision models.
///
/// Run the model on multiple augmented views of the same image and average
/// the logits for improved accuracy.

use crate::preprocess::horizontal_flip;

/// Horizontal-flip TTA: run model on original + horizontally flipped image, average logits.
///
/// `image` is `[C, H, W]` channel-first tensor.
/// `forward` is a closure that runs the model and returns logits.
pub fn tta_horizontal_flip<F>(image: &[f32], channels: usize, h: usize, w: usize, forward: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    let logits_orig = forward(image);
    let flipped = horizontal_flip(image, channels, h, w);
    let logits_flip = forward(&flipped);

    logits_orig
        .iter()
        .zip(logits_flip.iter())
        .map(|(a, b)| (a + b) * 0.5)
        .collect()
}

/// Multi-view TTA: run model on multiple augmented views and average logits.
///
/// `augmentations` produces the augmented images. Each must be `[C, H, W]`.
pub fn tta_multi_view<F>(views: &[Vec<f32>], forward: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    assert!(!views.is_empty(), "Need at least one view");

    let first = forward(&views[0]);
    let mut sum = first;

    for view in &views[1..] {
        let logits = forward(view);
        for (s, l) in sum.iter_mut().zip(logits.iter()) {
            *s += l;
        }
    }

    let inv = 1.0 / views.len() as f32;
    for v in sum.iter_mut() {
        *v *= inv;
    }
    sum
}

/// Generate 5-crop views (center + 4 corners) of a `[C, H, W]` tensor.
///
/// Returns 5 tensors of size `[C, crop_h, crop_w]`.
pub fn five_crop(tensor: &[f32], channels: usize, h: usize, w: usize, crop_h: usize, crop_w: usize) -> Vec<Vec<f32>> {
    assert!(crop_h <= h && crop_w <= w);

    let extract = |y_off: usize, x_off: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; channels * crop_h * crop_w];
        for c in 0..channels {
            for y in 0..crop_h {
                for x in 0..crop_w {
                    out[c * crop_h * crop_w + y * crop_w + x] =
                        tensor[c * h * w + (y + y_off) * w + (x + x_off)];
                }
            }
        }
        out
    };

    vec![
        extract((h - crop_h) / 2, (w - crop_w) / 2), // center
        extract(0, 0),                                  // top-left
        extract(0, w - crop_w),                         // top-right
        extract(h - crop_h, 0),                         // bottom-left
        extract(h - crop_h, w - crop_w),                // bottom-right
    ]
}

/// Generate 10-crop views (5 crops + their horizontal flips).
pub fn ten_crop(tensor: &[f32], channels: usize, h: usize, w: usize, crop_h: usize, crop_w: usize) -> Vec<Vec<f32>> {
    let crops = five_crop(tensor, channels, h, w, crop_h, crop_w);
    let mut views = Vec::with_capacity(10);
    for crop in &crops {
        views.push(crop.clone());
        views.push(horizontal_flip(crop, channels, crop_h, crop_w));
    }
    views
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tta_horizontal_flip() {
        let image = vec![1.0f32; 3 * 4 * 4];
        let logits = tta_horizontal_flip(&image, 3, 4, 4, |_img| vec![1.0, 2.0, 3.0]);
        assert_eq!(logits.len(), 3);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_tta_multi_view() {
        let views = vec![
            vec![0.0f32; 12],
            vec![1.0f32; 12],
        ];
        let logits = tta_multi_view(&views, |_| vec![1.0, 3.0]);
        assert_eq!(logits, vec![1.0, 3.0]);
    }

    #[test]
    fn test_five_crop() {
        // 1 channel, 6x6 → crop 4x4
        let tensor: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let crops = five_crop(&tensor, 1, 6, 6, 4, 4);
        assert_eq!(crops.len(), 5);
        for crop in &crops {
            assert_eq!(crop.len(), 16);
        }
        // Center crop starts at (1, 1)
        assert_eq!(crops[0][0], 7.0);  // (1,1)
        // Top-left starts at (0, 0)
        assert_eq!(crops[1][0], 0.0);
        // Top-right starts at (0, 2)
        assert_eq!(crops[2][0], 2.0);
    }

    #[test]
    fn test_ten_crop() {
        let tensor = vec![0.0f32; 3 * 6 * 6];
        let crops = ten_crop(&tensor, 3, 6, 6, 4, 4);
        assert_eq!(crops.len(), 10);
        for crop in &crops {
            assert_eq!(crop.len(), 3 * 4 * 4);
        }
    }
}
