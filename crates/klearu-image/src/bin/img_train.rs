//! img_train — train a klearu-image model from a JSONL manifest.
//!
//! Usage:
//!     img_train \
//!         --manifest data.jsonl \
//!         --out checkpoint.safetensors \
//!         --steps 10000 \
//!         --lr 3e-4 \
//!         --log-every 100 \
//!         [--resume previous.safetensors]
//!
//! The manifest must be JSONL with one `{caption: string, image_tokens: [u32; 256]}`
//! per line. Pre-tokenize images with `scripts/build_codebook.py` (for the
//! codebook) + a separate offline encode-and-dump script.
//!
//! Text BPE tokenization is intentionally NOT baked in: callers pass
//! pre-tokenized text via the `text_tokens` field of the manifest (or
//! use the demo path that BPE-encodes captions on the fly via a
//! tokenizer.json file passed via `--bpe`).

use std::path::PathBuf;
use std::process::ExitCode;

use klearu_image::backward::train_step;
use klearu_image::checkpoint::{save_with_optimizer, load_with_optimizer};
use klearu_image::error::Result as ImgResult;
use klearu_image::grad::Gradients;
use klearu_image::model::{ImageTransformer, ImageTransformerConfig};
use klearu_image::optim::{AdamW, AdamWConfig};
use klearu_image::train::{assemble_batch, read_manifest, TrainExample};
use tokenizers::Tokenizer;

struct Args {
    manifest: PathBuf,
    out: PathBuf,
    steps: usize,
    lr: f32,
    log_every: usize,
    seed: u64,
    resume: Option<PathBuf>,
    bpe: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    let mut args = Args {
        manifest: PathBuf::new(),
        out: PathBuf::from("klearu_image.safetensors"),
        steps: 10000,
        lr: 3e-4,
        log_every: 100,
        seed: 42,
        resume: None,
        bpe: None,
    };
    let mut i = 1;
    while i < raw.len() {
        let arg = &raw[i];
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            raw.get(*i).cloned().ok_or_else(|| format!("flag {arg} expects a value"))
        };
        match arg.as_str() {
            "--manifest" => args.manifest = PathBuf::from(next(&mut i)?),
            "--out" => args.out = PathBuf::from(next(&mut i)?),
            "--steps" => args.steps = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--lr" => args.lr = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--log-every" => args.log_every = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--seed" => args.seed = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--resume" => args.resume = Some(PathBuf::from(next(&mut i)?)),
            "--bpe" => args.bpe = Some(PathBuf::from(next(&mut i)?)),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other:?}")),
        }
        i += 1;
    }
    if args.manifest.as_os_str().is_empty() {
        return Err("--manifest is required".into());
    }
    Ok(args)
}

fn print_help() {
    eprintln!("img_train — train klearu-image model");
    eprintln!();
    eprintln!("  --manifest FILE     JSONL with image_tokens + caption (required)");
    eprintln!("  --out FILE          Output checkpoint path (default klearu_image.safetensors)");
    eprintln!("  --steps N           Number of optimizer steps (default 10000)");
    eprintln!("  --lr F              Learning rate (default 3e-4)");
    eprintln!("  --log-every N       Print loss every N steps (default 100)");
    eprintln!("  --seed N            RNG seed for init / shuffle (default 42)");
    eprintln!("  --resume FILE       Resume from existing checkpoint");
    eprintln!("  --bpe FILE          Path to a tokenizer.json for BPE-encoding captions.");
    eprintln!("                      Omit to expect manifest entries to carry pre-BPE'd text.");
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => { eprintln!("error: {e}"); return ExitCode::from(2); }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
    }
}

fn run(args: Args) -> ImgResult<()> {
    let cfg = ImageTransformerConfig::baseline_50m();
    let (mut model, mut opt) = if let Some(path) = &args.resume {
        eprintln!("[img_train] resuming from {}", path.display());
        load_with_optimizer(path)?
    } else {
        let mut model = ImageTransformer::from_config(cfg.clone());
        init_random(&mut model, args.seed);
        let opt = AdamW::new(&model, AdamWConfig {
            lr: args.lr, ..Default::default()
        });
        (model, opt)
    };
    eprintln!("[img_train] model: {} params, {} layers, hidden={}",
        model.param_count(), model.config.num_layers, model.config.hidden_size);

    let manifest = read_manifest(&args.manifest)?;
    eprintln!("[img_train] manifest: {} entries", manifest.len());
    if manifest.is_empty() {
        return Err(klearu_image::error::ImageGenError::Config(
            "empty manifest".into()));
    }

    let mut grad = Gradients::zeros_for(&model);

    // BPE setup. If --bpe is provided, captions are encoded inline.
    let bpe = if let Some(path) = &args.bpe {
        eprintln!("[img_train] loading BPE tokenizer from {}", path.display());
        let tok = Tokenizer::from_file(path)
            .map_err(|e| klearu_image::error::ImageGenError::Unsupported(
                format!("tokenizer: {e}")))?;
        Some(tok)
    } else {
        eprintln!("[img_train] no --bpe: using empty text prefix (image-only training)");
        None
    };

    let n_entries = manifest.len();
    let start = std::time::Instant::now();
    let mut loss_running = 0.0_f64;
    let mut loss_count = 0_usize;
    for step in 0..args.steps {
        let idx = ((step as u64).wrapping_mul(2654435761) as usize) % n_entries;
        let entry = &manifest[idx];
        // Tokenize caption via BPE if available, else empty prefix.
        let text_tokens: Vec<u32> = if let Some(tok) = &bpe {
            match tok.encode(entry.caption.as_str(), false) {
                Ok(enc) => {
                    let mut ids: Vec<u32> = enc.get_ids().iter().copied()
                        .filter(|&t| (t as usize) < model.config.vocab_text)
                        .collect();
                    if ids.len() > model.config.max_text_len {
                        ids.truncate(model.config.max_text_len);
                    }
                    ids
                }
                Err(_) => Vec::new(),
            }
        } else { Vec::new() };
        let example = TrainExample {
            text_tokens,
            image_tokens: entry.image_tokens.clone(),
        };
        let batch = assemble_batch(&model.config, &example)?;
        let loss = train_step(&mut model, &mut opt, &mut grad, &batch)?;
        loss_running += loss as f64;
        loss_count += 1;
        if (step + 1) % args.log_every == 0 || step == 0 {
            let avg = (loss_running / loss_count as f64) as f32;
            let elapsed = start.elapsed().as_secs_f32();
            eprintln!("[img_train] step {}/{}  loss={:.4}  ({:.1}s elapsed, {:.2}s/step)",
                step + 1, args.steps, avg, elapsed, elapsed / (step + 1) as f32);
            loss_running = 0.0;
            loss_count = 0;
        }
        // Save checkpoint every 1000 steps.
        if (step + 1) % 1000 == 0 {
            save_with_optimizer(&model, &opt, &args.out)?;
            eprintln!("[img_train] saved checkpoint → {}", args.out.display());
        }
    }
    save_with_optimizer(&model, &opt, &args.out)?;
    eprintln!("[img_train] final checkpoint → {} (+ .optim.safetensors)",
        args.out.display());
    Ok(())
}

/// Initialise model weights with small random values. Uses splitmix for
/// determinism; replace with klearu-llm's xavier/kaiming when wiring
/// the proper init pass.
fn init_random(model: &mut ImageTransformer, seed: u64) {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut rand = move || -> f32 {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z >> 11) as f32 / (1u64 << 53) as f32) * 0.04 - 0.02
    };
    for v in model.embed.iter_mut() { *v = rand(); }
    for v in model.pos_embed.iter_mut() { *v = rand(); }
    for blk in model.blocks.iter_mut() {
        for v in blk.q_proj.weight.iter_mut() { *v = rand(); }
        for v in blk.k_proj.weight.iter_mut() { *v = rand(); }
        for v in blk.v_proj.weight.iter_mut() { *v = rand(); }
        for v in blk.o_proj.weight.iter_mut() { *v = rand(); }
        for v in blk.mlp_gate.weight.iter_mut() { *v = rand(); }
        for v in blk.mlp_up.weight.iter_mut() { *v = rand(); }
        for v in blk.mlp_down.weight.iter_mut() { *v = rand(); }
    }
    for v in model.lm_head.weight.iter_mut() { *v = rand(); }
}
