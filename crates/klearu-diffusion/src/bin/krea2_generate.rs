//! krea2_generate — Krea 2 text-to-image (and img2img / inpaint).
//!
//!   1. Encode the prompt (Qwen-Image chat template through Qwen3-VL-4B,
//!      12 tapped hidden states, then the transformer's text fusion)
//!   2. Euler on the flow ODE with the 12.9B single-stream transformer
//!   3. Decode with the Qwen-Image VAE -> PNG
//!
//! Turbo (default 8 steps, no CFG) and Raw (28 steps, CFG 4.5 in Krea's
//! `cond + g·(cond − uncond)` convention) are detected from the checkpoint.
//!
//! Requires `--features metal`.

use std::path::PathBuf;
use std::process::ExitCode;

#[cfg(feature = "metal")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use klearu_diffusion::image_io::save_png;
    use klearu_diffusion::krea2::runner::{Krea2GenOpts, Krea2LoadSpec, Krea2Runner, Krea2Variant};
    use std::sync::atomic::AtomicBool;

    let args = Args::parse();
    let variant = match args.variant.as_deref() {
        Some("turbo") => Some(Krea2Variant::Turbo),
        Some("raw") => Some(Krea2Variant::Raw),
        Some(other) => return Err(format!("--variant must be turbo or raw, not {other}").into()),
        None => None,
    };
    let spec = Krea2LoadSpec {
        checkpoint: args.model_path.clone(),
        text_encoder: args.text_encoder.clone(),
        tokenizer: args.tokenizer.clone(),
        vae: args.vae.clone(),
        variant,
        // One image per process: nothing to gain from keeping it resident.
        offload_text_encoder: true,
    };
    println!("[krea2] model: {}", spec.checkpoint.display());
    let t_load = std::time::Instant::now();
    let runner = Krea2Runner::load_with_progress(&spec, |m| println!("[krea2] {m}…"))?;
    println!("[krea2] {:?} loaded in {:.1}s", runner.variant, t_load.elapsed().as_secs_f32());
    // Allocated vs the device's working-set limit: MPS returns NaN instead of
    // failing when overcommitted, so this is the first thing to check after one.
    klearu_diffusion::metal_backend::gpu_memory_report("after load");

    let read = |p: &Option<PathBuf>| p.as_ref().map(std::fs::read).transpose();
    let opts = Krea2GenOpts {
        prompt: args.prompt,
        negative_prompt: args.negative,
        width: args.width,
        height: args.height,
        steps: args.steps.unwrap_or(0),
        guidance: args.guidance,
        seed: args.seed,
        input_image_bytes: read(&args.init)?,
        mask_bytes: read(&args.mask)?,
        strength: args.strength,
    };
    let abort = AtomicBool::new(false);
    let t_gen = std::time::Instant::now();
    let img = runner.generate(
        &opts,
        |step, total| println!("[krea2] step {step}/{total}  ({:.1}s)", t_gen.elapsed().as_secs_f32()),
        &abort,
    )?;
    println!("[krea2] generated in {:.1}s", t_gen.elapsed().as_secs_f32());
    klearu_diffusion::metal_backend::gpu_memory_report("after generate");
    save_png(&img.chw, img.height, img.width, &args.out)?;
    println!("[krea2] saved {}x{} → {}", img.width, img.height, args.out.display());
    Ok(())
}

struct Args {
    model_path: PathBuf,
    text_encoder: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    vae: Option<PathBuf>,
    variant: Option<String>,
    prompt: String,
    negative: String,
    width: usize,
    height: usize,
    steps: Option<usize>,
    guidance: Option<f32>,
    seed: u64,
    out: PathBuf,
    init: Option<PathBuf>,
    mask: Option<PathBuf>,
    strength: f32,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            model_path: PathBuf::from("models/Krea-2-Turbo"),
            text_encoder: None,
            tokenizer: None,
            vae: None,
            variant: None,
            prompt: "a fox walking in the snow".to_string(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: None,
            guidance: None,
            seed: 0,
            out: PathBuf::from("krea2_out.png"),
            init: None,
            mask: None,
            strength: 0.75,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().expect("missing value");
            match flag.as_str() {
                "--model-path" => a.model_path = PathBuf::from(val()),
                "--text-encoder" => a.text_encoder = Some(PathBuf::from(val())),
                "--tokenizer" => a.tokenizer = Some(PathBuf::from(val())),
                "--vae" => a.vae = Some(PathBuf::from(val())),
                "--variant" => a.variant = Some(val()),
                "--prompt" => a.prompt = val(),
                "--negative" => a.negative = val(),
                "--width" => a.width = val().parse().unwrap(),
                "--height" => a.height = val().parse().unwrap(),
                "--steps" => a.steps = Some(val().parse().unwrap()),
                "--guidance" => a.guidance = Some(val().parse().unwrap()),
                "--seed" => a.seed = val().parse().unwrap(),
                "--out" => a.out = PathBuf::from(val()),
                "--init" => a.init = Some(PathBuf::from(val())),
                "--mask" => a.mask = Some(PathBuf::from(val())),
                "--strength" => a.strength = val().parse().unwrap(),
                other => eprintln!("[krea2] ignoring unknown flag {other}"),
            }
        }
        a
    }
}

#[cfg(not(feature = "metal"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    Err("krea2_generate requires --features metal".into())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[krea2] error: {e}");
            ExitCode::FAILURE
        }
    }
}
