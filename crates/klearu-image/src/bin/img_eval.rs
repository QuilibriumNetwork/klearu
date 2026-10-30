//! Read-only held-out evaluation. Never updates weights or optimizer state.
use std::{path::PathBuf, time::Instant};

use clap::Parser;
use klearu_image::{
    checkpoint::load_model,
    conditioning,
    train::{assemble_batch, cross_entropy_loss, read_manifest, TrainExample},
};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(about = "Evaluate a frozen image transformer without training or saving it")]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    bpe: Option<PathBuf>,
    #[arg(long)]
    unconditional: bool,
    #[arg(long, default_value_t = 128)]
    limit: usize,
    /// Add a cat-versus-dog label-swap control; requires literal class captions.
    #[arg(long)]
    opposite_class: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    if args.limit == 0 {
        return Err("--limit must be positive".into());
    }
    let started = Instant::now();
    let model = load_model(&args.checkpoint)?;
    let supplied = args.bpe.as_ref().map(Tokenizer::from_file).transpose()?;
    let tok = conditioning::resolve(&model.conditioning, supplied, args.unconditional)?;
    let mut data = Vec::new();
    let mut opposite_text = Vec::new();
    for entry in read_manifest(&args.manifest)? {
        if args.opposite_class {
            let label = match entry.caption.as_str() { "cat" => "dog", "dog" => "cat", _ => return Err("Class contrast requires cat/dog captions".into()) };
            let tokenizer = tok.as_ref().ok_or("Class contrast requires conditioning")?;
            opposite_text.push(conditioning::encode(tokenizer, &model.config, label)?);
        }
        let text_tokens = if let Some(tok) = &tok {
            match entry.text_tokens {
                Some(ids) => ids,
                None => conditioning::encode(tok, &model.config, &entry.caption)?,
            }
        } else {
            Vec::new()
        };
        if tok.is_some() && text_tokens.is_empty() {
            return Err("conditional example has no text tokens".into());
        }
        let example = TrainExample {
            text_tokens,
            image_tokens: entry.image_tokens,
        };
        assemble_batch(&model.config, &example)?;
        data.push(example);
    }
    let n = data.len().min(args.limit);
    if n == 0 {
        return Err("empty evaluation manifest".into());
    }
    let modes = if args.opposite_class { vec!["correct", "shuffled", "empty", "opposite_class"] } else { vec!["correct", "shuffled", "empty"] };
    let mut sums = vec![0.0_f64; modes.len()];
    let mut correct = vec![0_usize; modes.len()];
    let mut predictions = 0;
    for (i, example) in data.iter().take(n).enumerate() {
        for mode in 0..modes.len() {
            let text_tokens = match mode {
                0 => example.text_tokens.clone(),
                1 => data[(i + 1) % data.len()].text_tokens.clone(),
                2 => Vec::new(),
                _ => opposite_text[i].clone(),
            };
            let batch = assemble_batch(
                &model.config,
                &TrainExample {
                    text_tokens,
                    image_tokens: example.image_tokens.clone(),
                },
            )?;
            let logits = model.forward(&batch.token_ids)?;
            let (loss, hits) = cross_entropy_loss(&model, &logits, &batch)?;
            if !loss.is_finite() {
                return Err("non-finite evaluation loss".into());
            }
            sums[mode] += loss as f64;
            correct[mode] += hits;
            if mode == 0 {
                predictions += batch.targets.len();
            }
        }
    }
    let mut losses = serde_json::Map::new();
    let mut accuracy = serde_json::Map::new();
    for (i, mode) in modes.iter().enumerate() {
        losses.insert(mode.to_string(), serde_json::json!(sums[i] / n as f64));
        accuracy.insert(
            mode.to_string(),
            serde_json::json!(correct[i] as f64 / predictions as f64),
        );
    }
    let report = serde_json::json!({
        "examples":n, "predictions":predictions, "cross_entropy":losses,
        "token_accuracy":accuracy, "seconds":started.elapsed().as_secs_f64(),
        "checkpoint_sha256":format!("{:x}", Sha256::digest(std::fs::read(&args.checkpoint)?)),
        "manifest_sha256":format!("{:x}", Sha256::digest(std::fs::read(&args.manifest)?)),
        "shuffled_caption_method":"cyclic next manifest row; equal captions may occur",
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
