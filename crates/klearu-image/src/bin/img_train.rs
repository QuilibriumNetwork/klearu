//! Single-machine training with explicit conditioning and bounded activation memory.
use clap::Parser;
use klearu_image::{
    backward::{backward, forward_train},
    checkpoint::{load_with_optimizer, save_with_optimizer},
    conditioning::{self, Conditioning},
    grad::Gradients,
    model::{ImageTransformer, ImageTransformerConfig},
    optim::{AdamW, AdamWConfig},
    train::{assemble_batch, cross_entropy_loss, read_manifest, TrainExample},
    ImageGenError, Result,
};
use rand::{seq::SliceRandom, SeedableRng};
use std::{path::PathBuf, time::Instant};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(about = "Train klearu-image on one machine; --steps counts additional optimizer updates")]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long, default_value = "klearu_image.safetensors")]
    out: PathBuf,
    #[arg(long, default_value_t = 1000)]
    steps: usize,
    /// New runs default to 3e-4; resume keeps stored LR unless explicitly overridden.
    #[arg(long)]
    lr: Option<f32>,
    #[arg(long, default_value_t = 100)]
    log_every: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Tokenizer is embedded in new checkpoints. Required for new conditional runs,
    /// including manifests that already contain text_tokens.
    #[arg(long, conflicts_with = "unconditional")]
    bpe: Option<PathBuf>,
    #[arg(long)]
    unconditional: bool,
    #[arg(long, default_value="pilot", value_parser=["pilot","baseline"])]
    preset: String,
    /// Image codebook size for a fresh run; resume uses its checkpoint configuration.
    #[arg(long, conflicts_with = "resume", value_parser = clap::value_parser!(u32).range(2..))]
    image_vocab: Option<u32>,
    /// Give image positions fixed raster slots independent of caption length.
    #[arg(long, conflicts_with = "resume")]
    stable_image_positions: bool,
    /// Sequential microbatches: one image's activations in memory at a time.
    #[arg(long, default_value_t = 8)]
    accumulate: usize,
    #[arg(long, default_value_t = 1.0)]
    clip_norm: f32,
    #[arg(long)]
    validation: Option<PathBuf>,
    #[arg(long, default_value_t = 100)]
    eval_every: usize,
    #[arg(long, default_value_t = 128)]
    eval_limit: usize,
    #[arg(long, default_value_t = 500)]
    save_every: usize,
}
fn error(e: impl std::fmt::Display) -> ImageGenError {
    ImageGenError::Config(e.to_string())
}
fn main() -> std::process::ExitCode {
    match run(Args::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
fn examples(
    path: &PathBuf,
    model: &ImageTransformer,
    tok: Option<&Tokenizer>,
) -> Result<Vec<TrainExample>> {
    read_manifest(path)?
        .into_iter()
        .enumerate()
        .map(|(i, e)| {
            let text_tokens = if let Some(tok) = tok {
                match e.text_tokens {
                    Some(ids) => ids,
                    None => conditioning::encode(tok, &model.config, &e.caption)?,
                }
            } else {
                Vec::new()
            };
            if tok.is_some() && text_tokens.is_empty() {
                return Err(error(format!(
                    "{} row {} has empty conditional text",
                    path.display(),
                    i + 1
                )));
            }
            let ex = TrainExample {
                text_tokens,
                image_tokens: e.image_tokens,
            };
            assemble_batch(&model.config, &ex)?; // Validate ALL IDs and lengths before training.
            Ok(ex)
        })
        .collect()
}
fn evaluate(model: &ImageTransformer, data: &[TrainExample], limit: usize) -> Result<()> {
    let n = data.len().min(limit);
    if n == 0 {
        return Err(error("empty validation set"));
    }
    let mut sums = [0.0_f64; 3];
    for (i, ex) in data.iter().take(n).enumerate() {
        for mode in 0..3 {
            let text = match mode {
                0 => ex.text_tokens.clone(),
                1 => data[(i + 1) % data.len()].text_tokens.clone(),
                _ => Vec::new(),
            };
            let b = assemble_batch(
                &model.config,
                &TrainExample {
                    text_tokens: text,
                    image_tokens: ex.image_tokens.clone(),
                },
            )?;
            let logits = model.forward(&b.token_ids)?;
            sums[mode] += cross_entropy_loss(model, &logits, &b)?.0 as f64;
        }
    }
    eprintln!(
        "[validation] examples={n} CE correct={:.4} shuffled={:.4} empty={:.4}",
        sums[0] / n as f64,
        sums[1] / n as f64,
        sums[2] / n as f64
    );
    Ok(())
}
fn run(args: Args) -> Result<()> {
    if args.steps == 0
        || args.accumulate == 0
        || args.log_every == 0
        || args.save_every == 0
        || args.eval_every == 0
        || args.eval_limit == 0
        || args.lr.is_some_and(|lr| !lr.is_finite() || lr <= 0.0)
        || !args.clip_norm.is_finite()
        || args.clip_norm <= 0.0
    {
        return Err(error(
            "steps, intervals, accumulation, learning rate and clip norm must be positive",
        ));
    }
    let supplied = args
        .bpe
        .as_ref()
        .map(|p| Tokenizer::from_file(p).map_err(error))
        .transpose()?;
    let (mut model, mut opt, tok) = if let Some(path) = &args.resume {
        let (mut m, o) = load_with_optimizer(path)?;
        let tok = conditioning::resolve(&m.conditioning, supplied, args.unconditional)?;
        if let Some(t) = &tok {
            conditioning::validate(t, &m.config)?;
        }
        m.conditioning = tok
            .as_ref()
            .map(conditioning::identity)
            .transpose()?
            .unwrap_or(Conditioning::Unconditional);
        (m, o, tok)
    } else {
        let tok = conditioning::resolve(&Conditioning::Legacy, supplied, args.unconditional)?;
        let mut cfg = if args.preset == "pilot" {
            ImageTransformerConfig::pilot()
        } else {
            ImageTransformerConfig::baseline_50m()
        };
        cfg.vocab_text = tok.as_ref().map(conditioning::vocab_size).unwrap_or(1);
        cfg.stable_image_positions = args.stable_image_positions;
        if let Some(vocab) = args.image_vocab {
            cfg.vocab_image = vocab as usize;
        }
        cfg.bos_token = cfg.vocab_text as u32;
        cfg.sep_image_token = cfg.bos_token + 1;
        cfg.eos_token = cfg.bos_token + 2;
        let mut m = ImageTransformer::from_config(cfg);
        m.init_random(args.seed);
        m.conditioning = tok
            .as_ref()
            .map(conditioning::identity)
            .transpose()?
            .unwrap_or(Conditioning::Unconditional);
        let o = AdamW::new(
            &m,
            AdamWConfig {
                lr: args.lr.unwrap_or(3e-4),
                ..Default::default()
            },
        );
        (m, o, tok)
    };
    if let Some(lr) = args.lr {
        opt.config.lr = lr;
    }
    // Explicit run settings let a resumed job preserve data order; changing these
    // settings requires starting a new run rather than silently replaying samples.
    use sha2::{Digest, Sha256};
    let manifest_sha256 = format!("{:x}", Sha256::digest(std::fs::read(&args.manifest)?));
    let settings = serde_json::json!({"seed":args.seed,"accumulate":args.accumulate,
        "manifest":std::fs::canonicalize(&args.manifest)?,"manifest_sha256":manifest_sha256,"clip_norm":args.clip_norm});
    if let Some(path) = &args.resume {
        let run_path = path.with_extension("run.json");
        if run_path.exists() {
            let old: serde_json::Value =
                serde_json::from_slice(&std::fs::read(run_path)?).map_err(error)?;
            if old != settings {
                return Err(error(
                    "resume settings differ (seed, accumulation, manifest or clipping)",
                ));
            }
        } else {
            eprintln!("[img_train] legacy checkpoint has no run settings; data order cannot be recovered exactly");
        }
        eprintln!(
            "[img_train] resumed optimizer at step {}; lr={}",
            opt.step, opt.config.lr
        );
    }
    let data = examples(&args.manifest, &model, tok.as_ref())?;
    if data.is_empty() {
        return Err(error("empty manifest"));
    }
    let val = args
        .validation
        .as_ref()
        .map(|p| examples(p, &model, tok.as_ref()))
        .transpose()?;
    eprintln!(
        "[img_train] params={} examples={} accumulate={} parameter/gradient/Adam={:.2} GiB",
        model.param_count(),
        data.len(),
        args.accumulate,
        model.param_count() as f64 * 16.0 / 1073741824.0
    );
    if let Some(v) = &val {
        evaluate(&model, v, args.eval_limit)?;
    }
    let mut grad = Gradients::zeros_for(&model);
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut epoch_cached = None;
    let start = Instant::now();
    for local in 0..args.steps {
        grad.zero_inplace();
        let mut loss = 0.0;
        for micro in 0..args.accumulate {
            let sample = opt
                .step
                .checked_mul(args.accumulate as u64)
                .and_then(|x| x.checked_add(micro as u64))
                .ok_or_else(|| error("sample counter overflow"))?;
            let epoch = sample / data.len() as u64;
            if epoch_cached != Some(epoch) {
                order.iter_mut().enumerate().for_each(|(i, x)| *x = i);
                order.shuffle(&mut rand::rngs::StdRng::seed_from_u64(
                    args.seed.wrapping_add(epoch),
                ));
                epoch_cached = Some(epoch);
            }
            let ex = &data[order[(sample % data.len() as u64) as usize]];
            let batch = assemble_batch(&model.config, ex)?;
            let cache = forward_train(&model, &batch.token_ids)?;
            loss += backward(&model, &cache, &batch.predict_at, &batch.targets, &mut grad)?;
        }
        let norm = grad.average_and_clip(args.accumulate, args.clip_norm)?;
        if !loss.is_finite() {
            return Err(error("non-finite loss"));
        }
        opt.step(&mut model, &grad);
        if local == 0 || (local + 1) % args.log_every == 0 {
            eprintln!(
                "[img_train] step={} CE={:.4} grad_norm={:.3} elapsed={:.1}s seconds/example={:.4}",
                opt.step,
                loss / args.accumulate as f32,
                norm,
                start.elapsed().as_secs_f64(),
                start.elapsed().as_secs_f64() / ((local + 1) * args.accumulate) as f64
            );
        }
        if opt.step % args.eval_every as u64 == 0 {
            if let Some(v) = &val {
                evaluate(&model, v, args.eval_limit)?;
            }
        }
        if (local + 1) % args.save_every == 0 || local + 1 == args.steps {
            save_with_optimizer(&model, &opt, &args.out)?;
            std::fs::write(
                args.out.with_extension("run.json"),
                serde_json::to_vec_pretty(&settings).map_err(error)?,
            )?;
        }
    }
    Ok(())
}
