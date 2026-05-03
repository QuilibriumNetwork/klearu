//! DaViT MPC inference benchmark / comparison.
//!
//! Usage:
//!   davit-mpc-infer <model_dir> [image_path]
//!
//! Runs all three modes on the same image and compares:
//! 1. Plaintext (direct `model.forward()`)
//! 2. Low-security MPC (reveal image, plaintext forward)
//! 3. High-security MPC (Q32.32 shares throughout)
//!
//! Requires the `vision` and `cli` features.

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: davit-mpc-infer <model_dir> [image_path]");
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let image_path: Option<PathBuf> = args.get(2).map(PathBuf::from);

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

    let image_size = model.config.image_size;

    // Load or generate image
    let image = if let Some(ref path) = image_path {
        eprintln!("Loading image from {:?}...", path);
        load_and_preprocess(path, image_size)
    } else {
        eprintln!("No image provided, using constant 0.5 (sanity check)");
        vec![0.5f32; 3 * image_size * image_size]
    };

    // === 1. Plaintext ===
    eprintln!("\n=== Plaintext forward pass ===");
    let t1 = Instant::now();
    let plaintext_logits = model.forward(&image);
    eprintln!("  Time: {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);
    print_top5("  Plaintext", &plaintext_logits);

    // Two independent copies of the model are required, one per MPC party.
    let model0 = klearu_vision::weight::load_davit_model(&model_dir)
        .expect("Failed to load model (party 0)");
    let model1 = klearu_vision::weight::load_davit_model(&model_dir)
        .expect("Failed to load model (party 1)");

    // === 2. Low-security MPC ===
    {
        use klearu_mpc::beaver::dummy_triple_pair;
        use klearu_mpc::transport::memory_transport_pair;
        use klearu_private::private_davit::{private_davit_forward, shared_image};

        eprintln!("\n=== Low-security MPC (reveal image, plaintext forward) ===");

        let share0 = shared_image(0, &image);
        let share1 = shared_image(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair(100);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let t2 = Instant::now();

        let handle = std::thread::spawn(move || {
            private_davit_forward(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let logits0 = private_davit_forward(0, &model0, &share0, &mut gen0, &mut trans_a)
            .expect("party 0 failed");
        let logits1 = handle.join().expect("party 1 panicked").expect("party 1 failed");

        let elapsed = t2.elapsed();
        eprintln!("  Time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);

        // Verify parties agree
        let mut party_diff = 0.0f32;
        for i in 0..logits0.len() {
            let d = (logits0[i] - logits1[i]).abs();
            if d > party_diff { party_diff = d; }
        }
        eprintln!("  Party agreement: max diff = {:.6}", party_diff);

        // Compare to plaintext
        let mut max_diff = 0.0f32;
        for i in 0..logits0.len() {
            let d = (logits0[i] - plaintext_logits[i]).abs();
            if d > max_diff { max_diff = d; }
        }
        eprintln!("  vs Plaintext: max logit diff = {:.6}", max_diff);
        print_top5("  Low-sec MPC", &logits0);
    }

    // === 3. High-security MPC ===
    {
        use klearu_mpc::beaver::dummy_triple_pair_128;
        use klearu_mpc::transport::memory_transport_pair;
        use klearu_private::private_davit::{private_davit_forward_secure, shared_image_64};

        eprintln!("\n=== High-security MPC (Q32.32 shares throughout) ===");

        // Reload models for this test
        let model0 = klearu_vision::weight::load_davit_model(&model_dir)
            .expect("Failed to load model (party 0)");
        let model1 = klearu_vision::weight::load_davit_model(&model_dir)
            .expect("Failed to load model (party 1)");

        let share0 = shared_image_64(0, &image);
        let share1 = shared_image_64(1, &image);

        // DaViT tiny needs a lot of triples for all the LayerNorm squaring operations
        let (mut gen0, mut gen1) = dummy_triple_pair_128(5_000_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let t3 = Instant::now();

        let handle = std::thread::spawn(move || {
            private_davit_forward_secure(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let logits0 = private_davit_forward_secure(0, &model0, &share0, &mut gen0, &mut trans_a)
            .expect("party 0 failed");
        let logits1 = handle.join().expect("party 1 panicked").expect("party 1 failed");

        let elapsed = t3.elapsed();
        eprintln!("  Time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);

        // Verify parties agree
        let mut party_diff = 0.0f32;
        for i in 0..logits0.len() {
            let d = (logits0[i] - logits1[i]).abs();
            if d > party_diff { party_diff = d; }
        }
        eprintln!("  Party agreement: max diff = {:.6}", party_diff);

        // Compare to plaintext
        let mut max_diff = 0.0f32;
        for i in 0..logits0.len() {
            let d = (logits0[i] - plaintext_logits[i]).abs();
            if d > max_diff { max_diff = d; }
        }
        eprintln!("  vs Plaintext: max logit diff = {:.6}", max_diff);
        print_top5("  High-sec MPC", &logits0);
    }
}

fn load_and_preprocess(path: &PathBuf, target_size: usize) -> Vec<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to open image {:?}: {}", path, e));

    let resized = img.resize_exact(
        target_size as u32,
        target_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();

    let mean = [0.485f32, 0.456, 0.406];
    let std_dev = [0.229f32, 0.224, 0.225];

    let mut tensor = vec![0.0f32; 3 * target_size * target_size];
    for y in 0..target_size {
        for x in 0..target_size {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                tensor[c * target_size * target_size + y * target_size + x] =
                    (val - mean[c]) / std_dev[c];
            }
        }
    }

    tensor
}

fn print_top5(label: &str, logits: &[f32]) {
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_logit = indexed[0].1;
    let sum_exp: f32 = logits.iter().map(|&v| (v - max_logit).exp()).sum();

    println!("{} top-5:", label);
    for (rank, (class_id, score)) in indexed.iter().take(5).enumerate() {
        let prob = (*score - max_logit).exp() / sum_exp;
        println!(
            "    #{}: class {:>4} | logit {:>8.4} | prob {:.4}",
            rank + 1, class_id, score, prob
        );
    }
}
