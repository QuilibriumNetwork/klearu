//! DaViT inference CLI.
//!
//! Usage:
//!   davit-infer <model_dir> [image_path] [--tta]
//!
//! Supports PNG/JPEG/etc. images with proper preprocessing:
//! - Resize shorter edge (bicubic) → center crop → ImageNet normalize.
//! - Preprocessing params auto-detected from timm pretrained_cfg if available.
//! - Optional `--tta` flag for horizontal-flip test-time augmentation.

mod imagenet_classes;

use std::path::PathBuf;
use std::time::Instant;

use klearu_vision::preprocess::{
    PreprocessConfig, Interpolation, ResizeMode,
    center_crop, normalize, rgb_bytes_to_chw,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: davit-infer <model_dir> [image_path] [--tta]");
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let mut image_path: Option<PathBuf> = None;
    let mut use_tta = false;

    for arg in &args[2..] {
        if arg == "--tta" {
            use_tta = true;
        } else if image_path.is_none() {
            image_path = Some(PathBuf::from(arg));
        }
    }

    // Load preprocessing config from model directory
    let preprocess_config = load_preprocess_config(&model_dir);
    eprintln!(
        "Preprocessing: input_size={}, resize_size={}, crop_pct={:.3}, interp={:?}",
        preprocess_config.input_size, preprocess_config.resize_size,
        preprocess_config.crop_pct, preprocess_config.interpolation,
    );

    // Load model
    eprintln!("Loading model from {:?}...", model_dir);
    let t0 = Instant::now();
    let model = klearu_vision::weight::load_davit_model(&model_dir)
        .expect("Failed to load model");
    eprintln!("Model loaded in {:.2}s", t0.elapsed().as_secs_f32());
    eprintln!(
        "  embed_dims={:?}, depths={:?}, classes={}",
        model.config.embed_dims, model.config.depths, model.config.num_classes,
    );

    let image_size = preprocess_config.input_size;

    // Load or generate image
    let image = if let Some(ref path) = image_path {
        eprintln!("Loading image from {:?}...", path);
        load_and_preprocess(path, &preprocess_config)
    } else {
        eprintln!("No image provided, using constant 0.5 (sanity check)");
        let mut img = vec![0.5f32; 3 * image_size * image_size];
        normalize(&mut img, 3, image_size, image_size, &preprocess_config.mean, &preprocess_config.std);
        img
    };

    // Run inference
    let t1 = Instant::now();
    let logits = if use_tta {
        eprintln!("Running forward pass with horizontal-flip TTA...");
        klearu_vision::tta::tta_horizontal_flip(&image, 3, image_size, image_size, |img| {
            model.forward(img)
        })
    } else {
        eprintln!("Running forward pass...");
        model.forward(&image)
    };
    let elapsed = t1.elapsed();
    eprintln!("Forward pass: {:.1}ms\n", elapsed.as_secs_f64() * 1000.0);

    // Top-5 with class names
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let classes = &imagenet_classes::IMAGENET_CLASSES;
    println!("Top-5 predictions:");
    for (rank, (class_id, score)) in indexed.iter().take(5).enumerate() {
        let prob = softmax_prob(*score, &logits);
        let name = if *class_id < classes.len() {
            classes[*class_id]
        } else {
            "?"
        };
        println!(
            "  #{}: {:>30} (class {:>4}) | logit {:>8.4} | prob {:.4}",
            rank + 1, name, class_id, score, prob
        );
    }
}

/// Load preprocessing config from the model directory's config.json.
fn load_preprocess_config(model_dir: &PathBuf) -> PreprocessConfig {
    let config_path = model_dir.join("config.json");
    if let Ok(json_str) = std::fs::read_to_string(&config_path) {
        PreprocessConfig::from_model_config(&json_str)
    } else {
        PreprocessConfig::default()
    }
}

/// Load an image file with proper preprocessing pipeline.
///
/// 1. Resize shorter edge to `resize_size` (bicubic or bilinear)
/// 2. Center-crop to `input_size x input_size`
/// 3. Normalize with mean/std
fn load_and_preprocess(path: &PathBuf, config: &PreprocessConfig) -> Vec<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to open image {:?}: {}", path, e));

    let filter = match config.interpolation {
        Interpolation::Bicubic => image::imageops::FilterType::CatmullRom,
        Interpolation::Bilinear => image::imageops::FilterType::Triangle,
    };

    let target = config.input_size;

    let resized = match config.resize_mode {
        ResizeMode::ShortEdge => {
            let (orig_w, orig_h) = (img.width(), img.height());
            let shorter = orig_w.min(orig_h) as f32;
            let scale = config.resize_size as f32 / shorter;
            let new_w = (orig_w as f32 * scale).round() as u32;
            let new_h = (orig_h as f32 * scale).round() as u32;
            img.resize_exact(new_w, new_h, filter)
        }
        ResizeMode::LongEdge => {
            let (orig_w, orig_h) = (img.width(), img.height());
            let longer = orig_w.max(orig_h) as f32;
            let scale = config.resize_size as f32 / longer;
            let new_w = (orig_w as f32 * scale).round() as u32;
            let new_h = (orig_h as f32 * scale).round() as u32;
            img.resize_exact(new_w, new_h, filter)
        }
        ResizeMode::Exact => {
            img.resize_exact(
                config.resize_size as u32,
                config.resize_size as u32,
                filter,
            )
        }
    };

    let rgb = resized.to_rgb8();
    let (rw, rh) = (rgb.width() as usize, rgb.height() as usize);

    // Convert to [C, H, W] f32 tensor
    let mut tensor = rgb_bytes_to_chw(rgb.as_raw(), rh, rw);

    // Center crop if needed
    let tensor = if rh > target || rw > target {
        center_crop(&tensor, 3, rh, rw, target, target)
    } else if rh == target && rw == target {
        tensor
    } else {
        // If resize_size == input_size (crop_pct=1.0), just use as-is
        tensor
    };

    let mut result = tensor;
    normalize(&mut result, 3, target, target, &config.mean, &config.std);
    result
}

fn softmax_prob(logit: f32, all_logits: &[f32]) -> f32 {
    let max = all_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp = (logit - max).exp();
    let sum: f32 = all_logits.iter().map(|&v| (v - max).exp()).sum();
    exp / sum
}
