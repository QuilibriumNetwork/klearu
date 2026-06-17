//! hidream_generate — HiDream-O1-Image text-to-image generation.
//!
//! Pixel-native flow-matching diffusion on a Qwen3-VL-8B backbone (no VAE).
//!   1. Tokenize prompt (Qwen3-VL chat template + `<|boi|>` + `<|tms|>`)
//!   2. Load the 8B transformer into GPU bf16 buffers
//!   3. Sample noise, run the CFG denoise loop (Dev: 28-step flash, guidance 0)
//!   4. Unpatchify → save PNG
//!
//! Requires `--features metal` (8B on CPU is impractical).

use std::path::PathBuf;
use std::process::ExitCode;

#[cfg(feature = "metal")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use klearu_diffusion::hidream::runner::{HiDreamGenOpts, HiDreamRunner};
    use klearu_diffusion::image_io::save_png;
    use std::sync::atomic::AtomicBool;

    let args = Args::parse();
    println!("[hidream] model: {}", args.model_path.display());

    // Load the model + tokenizer via the same runner the desktop app uses.
    let t_load = std::time::Instant::now();
    println!("[hidream] loading weights (this converts ~8B params to bf16)…");
    let runner = HiDreamRunner::load(&args.model_path)?;
    println!("[hidream] weights loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    // Dev is guidance-distilled → no CFG (guidance 0 → flash); full uses CFG.
    let guidance = if args.model_type == "dev" { 0.0 } else { args.guidance };
    let input_image_bytes = match &args.init {
        Some(p) => Some(std::fs::read(p)?),
        None => None,
    };
    let mask_bytes = match &args.mask {
        Some(p) => Some(std::fs::read(p)?),
        None => None,
    };
    let ref_image_bytes: Vec<Vec<u8>> =
        args.refs.iter().map(std::fs::read).collect::<std::io::Result<_>>()?;
    let opts = HiDreamGenOpts {
        prompt: args.prompt,
        negative_prompt: String::new(),
        height: args.height,
        width: args.width,
        steps: args.steps.unwrap_or(if args.model_type == "full" { 50 } else { 28 }),
        guidance,
        seed: args.seed,
        input_image_bytes,
        mask_bytes,
        strength: args.strength,
        ref_image_bytes,
        // Binary respects explicit --width/--height unless --snap is given.
        snap_resolution: args.snap,
    };
    println!(
        "[hidream] {}x{}, guidance {} ({})",
        args.width, args.height, guidance,
        if guidance > 1.0 { "full / UniPC + CFG" } else { "dev / flash" },
    );

    let abort = AtomicBool::new(false);
    let t_gen = std::time::Instant::now();
    let img = runner.generate(
        &opts,
        |step, total| {
            if step == total || step % 5 == 0 {
                println!("[hidream] step {step}/{total}");
            }
        },
        &abort,
    )?;
    println!("[hidream] generated in {:.1}s", t_gen.elapsed().as_secs_f32());

    save_png(&img.chw, img.height, img.width, &args.out)?;
    println!("[hidream] saved {}x{} → {}", img.width, img.height, args.out.display());
    Ok(())
}

struct Args {
    model_path: PathBuf,
    prompt: String,
    width: usize,
    height: usize,
    steps: Option<usize>,
    guidance: f32,
    shift: f32,
    seed: u64,
    model_type: String,
    out: PathBuf,
    init: Option<PathBuf>,
    mask: Option<PathBuf>,
    strength: f32,
    refs: Vec<PathBuf>,
    snap: bool,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            model_path: PathBuf::from("models/HiDream-O1-Image-Dev"),
            prompt: "a photorealistic red apple on a wooden table".to_string(),
            width: 2048,
            height: 2048,
            steps: None,
            guidance: 5.0, // full-model default; dev forces 0.0
            shift: 3.0,
            seed: 32,
            model_type: "dev".to_string(),
            out: PathBuf::from("hidream_out.png"),
            init: None,
            mask: None,
            strength: 0.6,
            refs: Vec::new(),
            snap: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().expect("missing value");
            match flag.as_str() {
                "--model-path" => a.model_path = PathBuf::from(val()),
                "--prompt" => a.prompt = val(),
                "--width" => a.width = val().parse().unwrap(),
                "--height" => a.height = val().parse().unwrap(),
                "--steps" => a.steps = Some(val().parse().unwrap()),
                "--guidance" => a.guidance = val().parse().unwrap(),
                "--shift" => a.shift = val().parse().unwrap(),
                "--seed" => a.seed = val().parse().unwrap(),
                "--model-type" => a.model_type = val(),
                "--out" => a.out = PathBuf::from(val()),
                "--init" => a.init = Some(PathBuf::from(val())),
                "--mask" => a.mask = Some(PathBuf::from(val())),
                "--strength" => a.strength = val().parse().unwrap(),
                "--ref" => a.refs.push(PathBuf::from(val())),
                "--snap" => a.snap = true,
                other => eprintln!("[hidream] ignoring unknown flag {other}"),
            }
        }
        a
    }
}

#[cfg(not(feature = "metal"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    Err("hidream_generate requires --features metal (8B inference on CPU is impractical)".into())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[hidream] error: {e}");
            ExitCode::FAILURE
        }
    }
}
