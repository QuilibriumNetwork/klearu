/// Integration tests that load real model weights and run inference.
///
/// These tests are marked `#[ignore]` because they require downloaded model
/// weights. Run with: `cargo test -p klearu-vision --test model_loading -- --ignored`
use std::path::Path;

fn workspace_root() -> &'static Path {
    // Integration tests are run from the workspace root
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
}

// ─── DaViT ──────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_load_davit_tiny() {
    let model_dir = workspace_root().join("davit_tiny");
    if !model_dir.exists() {
        eprintln!("Skipping: davit_tiny/ not found");
        return;
    }
    let model = klearu_vision::weight::load_davit_model(&model_dir).unwrap();
    let image_size = model.config.image_size;
    let input = vec![0.5f32; 3 * image_size * image_size];
    let logits = model.forward(&input);
    assert_eq!(logits.len(), model.config.num_classes);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
    }
    // Print top-5 for manual inspection
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("DaViT tiny top-5:");
    for (idx, score) in indexed.iter().take(5) {
        eprintln!("  class {idx}: {score:.4}");
    }
}

// ─── ViT ────────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_load_vit_tiny() {
    let model_dir = workspace_root().join("vit_tiny");
    if !model_dir.exists() {
        eprintln!("Skipping: vit_tiny/ not found");
        return;
    }
    let model = klearu_vision::weight::load_vit_model(&model_dir).unwrap();
    let image_size = model.config.image_size;
    let input = vec![0.5f32; 3 * image_size * image_size];
    let logits = model.forward(&input);
    assert_eq!(logits.len(), model.config.num_classes);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
    }
    eprintln!("ViT tiny top-5:");
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (idx, score) in indexed.iter().take(5) {
        eprintln!("  class {idx}: {score:.4}");
    }
}

// ─── DINOv2 ─────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_load_dinov2_small() {
    let model_dir = workspace_root().join("dinov2_small");
    if !model_dir.exists() {
        eprintln!("Skipping: dinov2_small/ not found");
        return;
    }
    let model = klearu_vision::weight::load_dinov2_model(&model_dir).unwrap();
    let image_size = model.config.image_size;
    let input = vec![0.5f32; 3 * image_size * image_size];
    // DINOv2 has num_classes=0 (feature extractor), forward still works
    let logits = model.forward(&input);
    eprintln!("DINOv2 small: image_size={image_size}, embed_dim={}, output_len={}",
        model.config.embed_dim, logits.len());
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }
}

// ─── EVA-02 ─────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_load_eva02_tiny() {
    let model_dir = workspace_root().join("eva02_tiny");
    if !model_dir.exists() {
        eprintln!("Skipping: eva02_tiny/ not found");
        return;
    }
    let model = klearu_vision::weight::load_eva02_model(&model_dir).unwrap();
    let image_size = model.config.image_size;
    let input = vec![0.5f32; 3 * image_size * image_size];
    let logits = model.forward(&input);
    assert_eq!(logits.len(), model.config.num_classes);
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
    }
    eprintln!("EVA-02 tiny top-5:");
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (idx, score) in indexed.iter().take(5) {
        eprintln!("  class {idx}: {score:.4}");
    }
}

// ─── SigLIP ─────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_load_siglip_base() {
    let model_dir = workspace_root().join("siglip_base");
    if !model_dir.exists() {
        eprintln!("Skipping: siglip_base/ not found");
        return;
    }
    let model = klearu_vision::weight::load_siglip_model(&model_dir).unwrap();
    let image_size = model.config.image_size;
    let input = vec![0.5f32; 3 * image_size * image_size];
    let logits = model.forward(&input);
    eprintln!("SigLIP base: image_size={image_size}, embed_dim={}, output_len={}",
        model.config.embed_dim, logits.len());
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }
}

// ─── Qwen3.5 Vision Encoder ────────────────────────────────────────────────

#[test]
#[ignore]
fn test_load_qwen_vision() {
    let model_dir = workspace_root().join("Qwen3.5-0.8B");
    if !model_dir.exists() {
        eprintln!("Skipping: Qwen3.5-0.8B/ not found");
        return;
    }
    let encoder = klearu_vision::weight::load_qwen_vision_from_dir(&model_dir).unwrap();
    let config = &encoder.config;
    eprintln!("Qwen vision: hidden_size={}, depth={}, out_hidden_size={}",
        config.hidden_size, config.depth, config.out_hidden_size);
    // Create a small test image (patch_size=16, spatial_merge_size=2)
    // Minimum grid for position embedding: at least 2x2 merged patches → 4x4 patches → 64x64 image
    let test_size = 64;
    let input = vec![0.5f32; 3 * test_size * test_size];
    let output = encoder.forward(&input, test_size, test_size);
    eprintln!("Qwen vision output length: {}", output.len());
    for (i, &v) in output.iter().enumerate().take(10) {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }
}
