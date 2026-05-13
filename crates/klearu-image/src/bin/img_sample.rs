//! img_sample — generate an image from a trained klearu-image checkpoint.
//!
//! Usage:
//!     img_sample \
//!         --checkpoint klearu_image.safetensors \
//!         --vae sd-v1-5-pruned-emaonly.safetensors \
//!         --codebook codebook.safetensors \
//!         --bpe tokenizer.json \
//!         --prompt "a black cat curled up" \
//!         --temperature 0.9 --top-k 64 --top-p 0.95 \
//!         --seed 42 \
//!         --out out.png
//!
//! Pipeline:
//!   prompt → BPE tokens → ImageTransformer.forward (autoregressive) →
//!   256 codewords → SdVaeQuantizedTokenizer.decode → 128×128 RGB → PNG

use std::path::PathBuf;
use std::process::ExitCode;

use klearu_diffusion::config::VAEConfig;
use klearu_diffusion::image_io::save_png;
use klearu_diffusion::vae::AutoencoderKL;
use klearu_diffusion::weight::SingleFileLoader;

use klearu_image::checkpoint::load_model;
use klearu_image::error::Result as ImgResult;
use klearu_image::sample::{sample_image_tokens, SampleConfig};
use klearu_image::tokenizer::{SdVaeQuantizedTokenizer, ImageTokenizer, VqTokenizerConfig};
use tokenizers::Tokenizer;

struct Args {
    checkpoint: PathBuf,
    vae: Option<PathBuf>,
    codebook: Option<PathBuf>,
    bpe: Option<PathBuf>,
    prompt: String,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    out: PathBuf,
    tokens_out: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    let mut args = Args {
        checkpoint: PathBuf::new(),
        vae: None,
        codebook: None,
        bpe: None,
        prompt: String::new(),
        temperature: 0.9,
        top_k: 64,
        top_p: 0.95,
        seed: 42,
        out: PathBuf::from("klearu_image.png"),
        tokens_out: None,
    };
    let mut i = 1;
    while i < raw.len() {
        let arg = &raw[i];
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            raw.get(*i).cloned().ok_or_else(|| format!("flag {arg} expects a value"))
        };
        match arg.as_str() {
            "--checkpoint" => args.checkpoint = PathBuf::from(next(&mut i)?),
            "--vae" => args.vae = Some(PathBuf::from(next(&mut i)?)),
            "--codebook" => args.codebook = Some(PathBuf::from(next(&mut i)?)),
            "--bpe" => args.bpe = Some(PathBuf::from(next(&mut i)?)),
            "--tokens-out" => args.tokens_out = Some(PathBuf::from(next(&mut i)?)),
            "--prompt" => args.prompt = next(&mut i)?,
            "--temperature" => args.temperature = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--top-k" => args.top_k = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--top-p" => args.top_p = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--seed" => args.seed = next(&mut i)?.parse().map_err(|e| format!("{e}"))?,
            "--out" => args.out = PathBuf::from(next(&mut i)?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other:?}")),
        }
        i += 1;
    }
    if args.checkpoint.as_os_str().is_empty() {
        return Err("--checkpoint is required".into());
    }
    if args.prompt.is_empty() {
        return Err("--prompt is required".into());
    }
    // Two modes: --tokens-out (skip Rust decode, dump JSON) or
    // --vae + --codebook (Rust-side decode via SD VAE + k-means codebook).
    if args.tokens_out.is_none() {
        if args.vae.is_none() || args.codebook.is_none() {
            return Err(
                "--tokens-out FILE  OR  (--vae FILE --codebook FILE) is required".into()
            );
        }
    }
    Ok(args)
}

fn print_help() {
    eprintln!("img_sample — generate an image from a trained checkpoint");
    eprintln!();
    eprintln!("  --checkpoint FILE    klearu-image checkpoint (required)");
    eprintln!("  --vae FILE           SD VAE single-file safetensors");
    eprintln!("  --codebook FILE      k-means codebook safetensors");
    eprintln!("                       (--vae + --codebook required UNLESS --tokens-out)");
    eprintln!("  --tokens-out FILE    Write generated tokens as JSON and skip Rust");
    eprintln!("                       decode. Pair with scripts/vqvae_decode.py for");
    eprintln!("                       the trained-VQ-VAE pipeline.");
    eprintln!("  --bpe FILE           tokenizer.json for BPE encoding the prompt");
    eprintln!("                       (omit to send the prompt with empty text tokens)");
    eprintln!("  --prompt \"...\"       Text prompt (required)");
    eprintln!("  --temperature F      Softmax temperature (default 0.9)");
    eprintln!("  --top-k N            Restrict to top-K codewords per step (default 64)");
    eprintln!("  --top-p F            Nucleus threshold (default 0.95)");
    eprintln!("  --seed N             RNG seed (default 42)");
    eprintln!("  --out FILE           Output PNG (Rust-decode mode only)");
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => { eprintln!("error: {e}"); return ExitCode::from(2); }
    };
    eprintln!("[img_sample] checkpoint: {}", args.checkpoint.display());
    if let Some(p) = &args.vae { eprintln!("[img_sample] vae:        {}", p.display()); }
    if let Some(p) = &args.codebook { eprintln!("[img_sample] codebook:   {}", p.display()); }
    if let Some(p) = &args.tokens_out { eprintln!("[img_sample] tokens-out:{}", p.display()); }
    eprintln!("[img_sample] prompt:     {:?}", args.prompt);
    eprintln!("[img_sample] sampling:   T={}, top-k={}, top-p={}, seed={}",
        args.temperature, args.top_k, args.top_p, args.seed);
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
    }
}

fn run(args: Args) -> ImgResult<()> {
    // 1. Load the trained image transformer.
    eprintln!("[img_sample] loading checkpoint …");
    let model = load_model(&args.checkpoint)?;
    eprintln!("[img_sample]   ✓ {} params, {} layers, hidden={}",
        model.param_count(), model.config.num_layers, model.config.hidden_size);

    // 2. Tokenize the prompt (or empty if no BPE provided).
    let text_tokens: Vec<u32> = if let Some(bpe_path) = &args.bpe {
        eprintln!("[img_sample] loading BPE tokenizer …");
        let tok = Tokenizer::from_file(bpe_path)
            .map_err(|e| klearu_image::error::ImageGenError::Unsupported(
                format!("tokenizer: {e}")))?;
        let enc = tok.encode(args.prompt.as_str(), false)
            .map_err(|e| klearu_image::error::ImageGenError::Unsupported(
                format!("bpe encode: {e}")))?;
        let mut ids: Vec<u32> = enc.get_ids().iter().copied()
            .filter(|&t| (t as usize) < model.config.vocab_text)
            .collect();
        if ids.len() > model.config.max_text_len {
            ids.truncate(model.config.max_text_len);
        }
        eprintln!("[img_sample]   ✓ {} text tokens", ids.len());
        ids
    } else {
        eprintln!("[img_sample] no --bpe: sampling with empty text prefix");
        Vec::new()
    };

    // 3. Sample tokens (mode-independent).
    eprintln!("[img_sample] sampling {} image tokens …",
        model.config.image_grid_h * model.config.image_grid_w);
    let sc = SampleConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };
    let t0 = std::time::Instant::now();
    let tokens = sample_image_tokens(&model, &text_tokens, &sc)?;
    eprintln!("[img_sample]   ✓ {} tokens in {:.2}s",
        tokens.len(), t0.elapsed().as_secs_f32());

    // 4a. tokens-out mode: dump JSON for an external decoder (e.g. the
    //     trained VQ-VAE — `scripts/vqvae_decode.py`).
    if let Some(tokens_path) = &args.tokens_out {
        let record = serde_json::json!({
            "image_tokens": tokens,
            "prompt": args.prompt,
            "grid_h": model.config.image_grid_h,
            "grid_w": model.config.image_grid_w,
            "vocab_image": model.config.vocab_image,
        });
        let json = serde_json::to_string_pretty(&record)
            .map_err(|e| klearu_image::error::ImageGenError::Unsupported(
                format!("serialize: {e}")))?;
        std::fs::write(tokens_path, json)?;
        eprintln!("[img_sample] wrote tokens → {} (decode with scripts/vqvae_decode.py)",
            tokens_path.display());
        return Ok(());
    }

    // 4b. Rust-decode mode: SD VAE + k-means codebook tokenizer.
    let vae_path = args.vae.as_ref().expect("checked in parse_args");
    let codebook_path = args.codebook.as_ref().expect("checked in parse_args");
    eprintln!("[img_sample] loading SD VAE …");
    let vae_cfg = sd15_vae_config();
    let mut vae = AutoencoderKL::from_config(vae_cfg);
    let loader = SingleFileLoader::open(vae_path)
        .map_err(|e| klearu_image::error::ImageGenError::Unsupported(
            format!("vae open: {e}")))?;
    let comp = loader.component("vae")
        .map_err(|e| klearu_image::error::ImageGenError::Unsupported(
            format!("vae component: {e}")))?;
    vae.load_from(&comp)
        .map_err(|e| klearu_image::error::ImageGenError::Unsupported(
            format!("vae load: {e}")))?;
    eprintln!("[img_sample]   ✓ vae loaded");

    let tk_cfg = VqTokenizerConfig {
        image_size: model.config.image_grid_w * 8,
        token_grid: model.config.image_grid_w,
        vocab_size: model.config.vocab_image,
        latent_channels: 4,
        vae_scaling_factor: 0.18215,
    };
    let tokenizer = SdVaeQuantizedTokenizer::new(tk_cfg.clone(), vae, codebook_path)?;
    eprintln!("[img_sample]   ✓ tokenizer ready ({}×{} grid, K={})",
        tk_cfg.token_grid, tk_cfg.token_grid, tk_cfg.vocab_size);

    eprintln!("[img_sample] decoding image …");
    let rgb = tokenizer.decode(&tokens)?;
    let h = tk_cfg.image_size;
    let w = tk_cfg.image_size;
    save_png(&rgb, h, w, &args.out)
        .map_err(|e| klearu_image::error::ImageGenError::Unsupported(
            format!("save_png: {e}")))?;
    eprintln!("[img_sample] wrote {}", args.out.display());
    Ok(())
}

/// SD 1.5 VAE shape — matches `klearu_diffusion::config::VAEConfig`'s SD 1.5
/// defaults. Inlined so we don't pull the whole SD config struct.
fn sd15_vae_config() -> VAEConfig {
    VAEConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 4,
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        act_fn: "silu".into(),
        norm_num_groups: 32,
        sample_size: 512,
        scaling_factor: 0.18215,
        shift_factor: 0.0,
    }
}
