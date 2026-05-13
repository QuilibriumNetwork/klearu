//! flux_generate — Flux (flux1-dev / flux1-schnell) image generation.
//!
//! Pipeline:
//!   1. Tokenize prompt with CLIP-L tokenizer (max 77) → CLIP-L pooled vector
//!   2. Tokenize prompt with T5 tokenizer (256 / 512) → T5-XXL seq embedding
//!   3. Sample noise latent at σ=1 [16, h_lat, w_lat]
//!   4. FlowMatchScheduler with Flux shift schedule
//!   5. Loop: v = transformer.forward(...); latent += Δσ · v
//!   6. VAE.unscale_latent → VAE decode → save PNG
//!
//! Required flags map to the four BFL release files. CLI flags can name
//! them individually or `--checkpoint <dir>` lets `discover_flux_paths`
//! find them automatically.

use std::path::PathBuf;
use std::process::ExitCode;

use klearu_diffusion::config::{CLIPTextConfig, VAEConfig};
use klearu_diffusion::image_io::{downsample_mask, load_image_as_chw_f32, load_mask, save_png};
use klearu_diffusion::scheduler::{FlowMatchConfig, FlowMatchScheduler, Scheduler};
use klearu_diffusion::text_encoder::CLIPTextModel;
use klearu_diffusion::text_encoder::t5::{T5Config, T5Encoder};
use klearu_diffusion::tokenizer::{CLIPTokenizer, T5Tokenizer};
use klearu_diffusion::unet::{FluxConfig, FluxTransformer};
use klearu_diffusion::vae::AutoencoderKL;
use klearu_diffusion::weight::{
    FluxCheckpoint, FluxPaths, FluxVariant, SDFormat, SingleFileLoader,
    component_with_prefix, detect_variant_single_file, discover_flux_paths,
    transformer_component_from_loader, vae_component_from_loader,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    Auto,
    Dev,
    Schnell,
}

fn parse_variant(s: &str) -> Variant {
    match s.to_ascii_lowercase().as_str() {
        "dev" | "flux-dev" | "flux1-dev" => Variant::Dev,
        "schnell" | "flux-schnell" | "flux1-schnell" => Variant::Schnell,
        _ => Variant::Auto,
    }
}

struct Args {
    /// Single-file all-in-one Flux bundle (ComfyUI / Civitai packaging:
    /// one safetensors with `model.diffusion_model.*`, `vae.*`,
    /// `text_encoders.clip_l.transformer.*`, `text_encoders.t5xxl.transformer.*`).
    /// Mutually exclusive with --checkpoint / per-file flags.
    checkpoint_file: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    transformer: Option<PathBuf>,
    vae: Option<PathBuf>,
    clip_l: Option<PathBuf>,
    clip_l_tokenizer: Option<PathBuf>,
    t5: Option<PathBuf>,
    t5_tokenizer: Option<PathBuf>,
    prompt: String,
    width: usize,
    height: usize,
    steps: Option<usize>,
    seed: u64,
    guidance: f32,
    variant: Variant,
    out: PathBuf,
    /// Optional path to a `reference_flux.safetensors` produced by
    /// `scripts/dump_flux_reference.py`. When set, the binary compares its
    /// own intermediates against the reference using cosine similarity at
    /// each captured checkpoint (T5 output, CLIP pooled, final latent,
    /// decoded image). Used for bug-hunting against diffusers — same
    /// pattern that surfaced the four SD 1.5 bugs.
    reference: Option<PathBuf>,
    /// Interactive mode: read prompts from stdin in a loop. The `--prompt`
    /// flag becomes optional (used only for non-REPL one-shot runs).
    repl: bool,
    /// Number of images to generate per prompt. Each gets its own seed
    /// (`seed`, `seed+1`, …) and output filename suffix (`out_0.png`,
    /// `out_1.png`, …) when `batch > 1`.
    batch: usize,
    /// Inpainting / img2img: input image to VAE-encode as starting latent.
    input: Option<PathBuf>,
    /// Optional mask (white=regenerate, black=preserve). Same dims as
    /// `--input`. If absent, regenerates the whole image (img2img mode).
    mask: Option<PathBuf>,
    /// 0..=1; 0 = no change to input, 1 = full regen from pure noise.
    strength: f32,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        checkpoint_file: None,
        checkpoint_dir: None,
        transformer: None, vae: None, clip_l: None, clip_l_tokenizer: None,
        t5: None, t5_tokenizer: None,
        prompt: String::new(),
        width: 1024, height: 1024,
        steps: None,
        seed: 42,
        guidance: 3.5,
        variant: Variant::Auto,
        out: PathBuf::from("flux_out.png"),
        reference: None,
        repl: false,
        batch: 1,
        input: None,
        mask: None,
        strength: 0.85,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        let next_path = |it: &mut dyn Iterator<Item=String>| -> Result<PathBuf, String> {
            it.next().map(PathBuf::from)
                .ok_or_else(|| format!("flag {a} expects a path"))
        };
        match a.as_str() {
            "--checkpoint" => args.checkpoint_dir = Some(next_path(&mut iter)?),
            "--checkpoint-file" => args.checkpoint_file = Some(next_path(&mut iter)?),
            "--transformer" => args.transformer = Some(next_path(&mut iter)?),
            "--vae" => args.vae = Some(next_path(&mut iter)?),
            "--clip-l" => args.clip_l = Some(next_path(&mut iter)?),
            "--clip-l-tokenizer" => args.clip_l_tokenizer = Some(next_path(&mut iter)?),
            "--t5" => args.t5 = Some(next_path(&mut iter)?),
            "--t5-tokenizer" => args.t5_tokenizer = Some(next_path(&mut iter)?),
            "--prompt" => args.prompt = iter.next().ok_or("--prompt requires a value")?,
            "--width" => args.width = iter.next().ok_or("--width n")?.parse().map_err(|e| format!("{e}"))?,
            "--height" => args.height = iter.next().ok_or("--height n")?.parse().map_err(|e| format!("{e}"))?,
            "--steps" => args.steps = Some(iter.next().ok_or("--steps n")?.parse().map_err(|e| format!("{e}"))?),
            "--seed" => args.seed = iter.next().ok_or("--seed u64")?.parse().map_err(|e| format!("{e}"))?,
            "--guidance" => args.guidance = iter.next().ok_or("--guidance f32")?.parse().map_err(|e| format!("{e}"))?,
            "--variant" => args.variant = parse_variant(&iter.next().ok_or("--variant <auto|dev|schnell>")?),
            "--out" => args.out = next_path(&mut iter)?,
            "--repl" => args.repl = true,
            "--batch" => args.batch = iter.next().ok_or("--batch n")?.parse().map_err(|e| format!("{e}"))?,
            "--ref" | "--reference" => args.reference = Some(next_path(&mut iter)?),
            "--input" => args.input = Some(next_path(&mut iter)?),
            "--mask" => args.mask = Some(next_path(&mut iter)?),
            "--strength" => args.strength = iter.next().ok_or("--strength f32")?.parse().map_err(|e| format!("{e}"))?,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown flag {a:?} — try --help")),
        }
    }
    if args.batch == 0 {
        return Err("--batch must be ≥ 1".into());
    }
    if !args.repl && args.prompt.is_empty() {
        return Err("--prompt is required (or use --repl for interactive mode)".into());
    }
    Ok(args)
}

fn print_help() {
    eprintln!("flux_generate — Flux image generation");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --prompt \"...\"        Text prompt");
    eprintln!();
    eprintln!("File inputs (pick ONE of these three styles):");
    eprintln!("  --checkpoint-file FILE  Single all-in-one Flux bundle (ComfyUI / Civitai");
    eprintln!("                          packaging: one .safetensors with transformer + VAE +");
    eprintln!("                          CLIP-L + T5 under different top-level prefixes).");
    eprintln!("  --checkpoint DIR        Directory with the four BFL release files: flux1-*.safetensors,");
    eprintln!("                          ae.safetensors, clip-l.safetensors, t5xxl_*.safetensors");
    eprintln!("  --transformer FILE      flux1-dev.safetensors or flux1-schnell.safetensors");
    eprintln!("  --vae FILE              ae.safetensors");
    eprintln!("  --clip-l FILE           CLIP-L weights (HF safetensors)");
    eprintln!("  --clip-l-tokenizer P    Path to CLIP-L tokenizer (file or dir)");
    eprintln!("  --t5 FILE               T5-XXL weights (HF safetensors)");
    eprintln!("  --t5-tokenizer FILE     T5 tokenizer.json");
    eprintln!();
    eprintln!("Generation:");
    eprintln!("  --width N               Output width  (default 1024)");
    eprintln!("  --height N              Output height (default 1024)");
    eprintln!("  --steps N               Inference steps (default: 50 dev / 4 schnell)");
    eprintln!("  --seed N                RNG seed (default 42)");
    eprintln!("  --guidance F            Distilled guidance for flux1-dev (default 3.5)");
    eprintln!("  --variant V             auto | dev | schnell (default auto)");
    eprintln!("  --out FILE              Output PNG (default flux_out.png)");
    eprintln!("  --repl                  Interactive: read prompts from stdin until EOF/`exit`.");
    eprintln!("                          Each line = one prompt; weights stay loaded between");
    eprintln!("                          prompts. In REPL --batch sets per-prompt count.");
    eprintln!("                          Inline overrides per prompt: `prompt :: --steps 30`");
    eprintln!("                          (only --steps / --seed / --guidance / --batch / --out).");
    eprintln!("  --batch N               Generate N images per prompt (default 1). Filenames");
    eprintln!("                          get a `_<i>` suffix when N > 1; seeds increment by 1");
    eprintln!("                          across the batch.");
    eprintln!();
    eprintln!("Inpainting / img2img:");
    eprintln!("  --input PNG             VAE-encode this image as the starting latent");
    eprintln!("                          (instead of pure noise). Image dims must match");
    eprintln!("                          --width × --height.");
    eprintln!("  --mask PNG              Grayscale mask, same dims as --input. White=regenerate,");
    eprintln!("                          black=preserve. If omitted when --input is given, the");
    eprintln!("                          whole image is regenerated (img2img mode).");
    eprintln!("  --strength F            How much to denoise (0..=1, default 0.85).");
    eprintln!();
    eprintln!("  --ref FILE              Compare against a reference_flux.safetensors produced");
    eprintln!("                          by scripts/dump_flux_reference.py. Reports cos-sim at");
    eprintln!("                          each captured checkpoint (T5 / CLIP / final latent /");
    eprintln!("                          decoded image).");
}

/// Reference tensors loaded from a `--ref` safetensors file (produced by
/// `scripts/dump_flux_reference.py`). All fields are optional: a partial
/// dump (e.g., just `t5_embed`) still validates as far as it goes.
struct FluxRefTensors {
    t5_embed: Option<Vec<f32>>,
    clip_pooled: Option<Vec<f32>>,
    latent_init: Option<Vec<f32>>,
    latent_final: Option<Vec<f32>>,
    decoded_image: Option<Vec<f32>>,
}

impl FluxRefTensors {
    fn load(path: &std::path::Path) -> Result<Self, String> {
        use safetensors::SafeTensors;
        let buf = std::fs::read(path).map_err(|e| format!("read ref: {e}"))?;
        let st = SafeTensors::deserialize(&buf).map_err(|e| format!("parse ref: {e}"))?;
        let opt = |name: &str| -> Option<Vec<f32>> {
            let t = st.tensor(name).ok()?;
            let bytes = t.data();
            if !bytes.len().is_multiple_of(4) { return None; }
            Some(bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        Ok(Self {
            t5_embed: opt("t5_embed"),
            clip_pooled: opt("clip_pooled"),
            latent_init: opt("latent_init"),
            latent_final: opt("latent_final"),
            decoded_image: opt("decoded_image"),
        })
    }
}

/// Cosine similarity between two equal-length vectors, ignoring NaN
/// elements (counts them in the report instead).
fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return f32::NAN; }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() { continue; }
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 { return 0.0; }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn report_cos(name: &str, a: &[f32], b: &[f32]) {
    let same_len = a.len() == b.len();
    let len_note = if same_len {
        format!("len={}", a.len())
    } else {
        format!("len mismatch: ours={} ref={}", a.len(), b.len())
    };
    let cs = if same_len { cos_sim(a, b) } else { f32::NAN };
    eprintln!("  [ref] {name:<24} {len_note}   cos_sim={cs:.6}");
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Per-prompt overrides parsed from REPL inline trailing flags.
#[derive(Default, Clone)]
struct PromptOverrides {
    steps: Option<usize>,
    seed: Option<u64>,
    guidance: Option<f32>,
    batch: Option<usize>,
    out: Option<PathBuf>,
    input: Option<PathBuf>,
    mask: Option<PathBuf>,
    strength: Option<f32>,
}

/// Parse `"prompt text :: --steps 30 --seed 7"` into (prompt, overrides).
/// Lines without `::` produce no overrides.
fn parse_prompt_line(line: &str) -> Result<(String, PromptOverrides), String> {
    let mut overrides = PromptOverrides::default();
    let (prompt, tail) = match line.split_once("::") {
        Some((p, t)) => (p.trim().to_string(), t.trim()),
        None => return Ok((line.trim().to_string(), overrides)),
    };
    let toks: Vec<&str> = tail.split_whitespace().collect();
    let mut i = 0;
    while i < toks.len() {
        let flag = toks[i];
        let val = toks.get(i + 1)
            .ok_or_else(|| format!("inline override {flag} expects a value"))?;
        match flag {
            "--steps" => overrides.steps = Some(val.parse().map_err(|e| format!("--steps: {e}"))?),
            "--seed" => overrides.seed = Some(val.parse().map_err(|e| format!("--seed: {e}"))?),
            "--guidance" => overrides.guidance = Some(val.parse().map_err(|e| format!("--guidance: {e}"))?),
            "--batch" => overrides.batch = Some(val.parse().map_err(|e| format!("--batch: {e}"))?),
            "--out" => overrides.out = Some(PathBuf::from(val)),
            "--input" => overrides.input = Some(PathBuf::from(val)),
            "--mask" => overrides.mask = Some(PathBuf::from(val)),
            "--strength" => overrides.strength = Some(val.parse().map_err(|e| format!("--strength: {e}"))?),
            _ => return Err(format!("unknown inline override flag {flag}")),
        }
        i += 2;
    }
    Ok((prompt, overrides))
}

/// Bundle of loaded models — populated once at startup, reused across
/// every prompt in REPL or batch mode.
struct LoadedModels {
    transformer: FluxTransformer,
    cfg: FluxConfig,
    variant: FluxVariant,
    vae: AutoencoderKL,
    clip_l: CLIPTextModel,
    t5: T5Encoder,
    clip_tok: CLIPTokenizer,
    t5_tok: T5Tokenizer,
}

/// Build the output filename for batch index `i`. If `batch == 1` returns
/// `base` unchanged; otherwise inserts `_<i>` before the extension.
fn batch_filename(base: &PathBuf, i: usize, batch: usize) -> PathBuf {
    if batch <= 1 { return base.clone(); }
    let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("flux_out");
    let ext = base.extension().and_then(|s| s.to_str()).unwrap_or("png");
    let parent = base.parent().unwrap_or(std::path::Path::new("."));
    parent.join(format!("{stem}_{i}.{ext}"))
}

fn run(args: Args) -> Result<(), String> {
    // ===== Open components (branch on single-file vs multi-file) =====
    // Each branch produces (variant, transformer_comp, vae_comp, clip_comp, t5_comp).
    // We use Box<dyn Drop>-style placeholders by hoisting all four ComponentTensors
    // into the same scope.
    let (variant_detected, comp_t, comp_v, comp_c, comp_t5) =
        if let Some(file) = &args.checkpoint_file {
            eprintln!("[flux] single-file checkpoint: {}", file.display());
            let loader = SingleFileLoader::open(file)
                .map_err(|e| format!("open single-file checkpoint: {e}"))?;
            if loader.variant != SDFormat::Flux {
                return Err(format!(
                    "{} does not look like a Flux checkpoint (variant={:?})",
                    file.display(), loader.variant
                ));
            }
            let v = detect_variant_single_file(&loader);

            // For each of the four components, source it from whichever is
            // available: explicit --vae / --clip-l / --t5 flag wins over
            // anything bundled in the single-file checkpoint. Transformer
            // is always taken from --checkpoint-file (that's the whole
            // point of the flag).
            let comp_t = transformer_component_from_loader(&loader)
                .map_err(|e| format!("transformer comp: {e}"))?;

            let comp_v = if let Some(p) = &args.vae {
                eprintln!("[flux] vae:         {} (per-flag override)", p.display());
                let l = SingleFileLoader::open(p).map_err(|e| format!("open vae: {e}"))?;
                let c = vae_component_from_loader(&l).map_err(|e| format!("vae comp: {e}"))?;
                drop(l); c
            } else {
                vae_component_from_loader(&loader)
                    .map_err(|e| format!(
                        "vae comp: {e}\n\
                         Hint: --checkpoint-file does not appear to contain a VAE. \
                         Pass --vae <ae.safetensors> separately."))?
            };

            let comp_c = if let Some(p) = &args.clip_l {
                eprintln!("[flux] clip-l:      {} (per-flag override)", p.display());
                let l = SingleFileLoader::open(p).map_err(|e| format!("open clip-l: {e}"))?;
                let c = component_with_prefix(&l, "")
                    .map_err(|e| format!("clip-l comp: {e}"))?;
                drop(l); c
            } else {
                component_with_prefix(&loader, "text_encoders.clip_l.transformer.")
                    .map_err(|e| format!(
                        "clip-l comp: {e}\n\
                         Hint: --checkpoint-file does not appear to contain CLIP-L. \
                         Pass --clip-l <clip_l.safetensors> separately."))?
            };

            let comp_t5 = if let Some(p) = &args.t5 {
                eprintln!("[flux] t5:          {} (per-flag override)", p.display());
                let l = SingleFileLoader::open(p).map_err(|e| format!("open t5: {e}"))?;
                let c = component_with_prefix(&l, "")
                    .map_err(|e| format!("t5 comp: {e}"))?;
                drop(l); c
            } else {
                component_with_prefix(&loader, "text_encoders.t5xxl.transformer.")
                    .map_err(|e| format!(
                        "t5 comp: {e}\n\
                         Hint: --checkpoint-file does not appear to contain T5-XXL. \
                         Pass --t5 <t5xxl_fp16.safetensors> separately."))?
            };

            drop(loader);
            (v, comp_t, comp_v, comp_c, comp_t5)
        } else {
            let paths = resolve_paths(&args)?;
            eprintln!("[flux] transformer: {}", paths.transformer.display());
            eprintln!("[flux] vae:         {}", paths.vae.display());
            eprintln!("[flux] clip-l:      {}", paths.clip_l.display());
            eprintln!("[flux] t5:          {}", paths.t5.display());

            let ckpt = FluxCheckpoint::open(&paths)
                .map_err(|e| format!("open checkpoint: {e}"))?;
            let v = ckpt.variant;
            let comp_t  = ckpt.transformer_component().map_err(|e| format!("transformer component: {e}"))?;
            let comp_v  = ckpt.vae_component().map_err(|e| format!("vae component: {e}"))?;
            let comp_c  = ckpt.clip_l_component().map_err(|e| format!("clip-l component: {e}"))?;
            let comp_t5 = ckpt.t5_component().map_err(|e| format!("t5 component: {e}"))?;
            (v, comp_t, comp_v, comp_c, comp_t5)
        };

    let variant = match args.variant {
        Variant::Auto => variant_detected,
        Variant::Dev => FluxVariant::Dev,
        Variant::Schnell => FluxVariant::Schnell,
    };
    eprintln!("[flux] variant detected: {variant:?}");

    // ===== Load transformer =====
    let cfg = match variant {
        FluxVariant::Dev => FluxConfig::flux_dev(),
        FluxVariant::Schnell => FluxConfig::flux_schnell(),
    };
    eprintln!("[flux] loading transformer ({} double + {} single blocks)…",
        cfg.num_double_blocks, cfg.num_single_blocks);
    let mut transformer = FluxTransformer::from_config(cfg.clone());
    transformer.load_from(&comp_t)
        .map_err(|e| format!("load transformer: {e}"))?;
    drop(comp_t);

    // ===== Load VAE (16-channel) =====
    eprintln!("[flux] loading VAE…");
    let vae_cfg = VAEConfig::flux();
    let mut vae = AutoencoderKL::from_config(vae_cfg);
    vae.load_from(&comp_v)
        .map_err(|e| format!("load vae: {e}"))?;
    drop(comp_v);

    // ===== Load CLIP-L (for pooled vector) =====
    eprintln!("[flux] loading CLIP-L…");
    let clip_cfg = CLIPTextConfig {
        vocab_size: 49408,
        hidden_size: 768,
        intermediate_size: 3072,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        max_position_embeddings: 77,
        layer_norm_eps: 1e-5,
        hidden_act: "quick_gelu".into(),
    };
    let mut clip_l = CLIPTextModel::from_config(clip_cfg);
    clip_l.load_from(&comp_c)
        .map_err(|e| format!("load clip-l: {e}"))?;
    drop(comp_c);

    // ===== Load T5-XXL =====
    eprintln!("[flux] loading T5-XXL (this is the slow part — 4.7B params)…");
    let t5_cfg = T5Config::t5_xxl();
    let mut t5 = T5Encoder::from_config(t5_cfg.clone());
    t5.load_from(&comp_t5)
        .map_err(|e| format!("load t5: {e}"))?;
    drop(comp_t5);

    // ===== Tokenizers =====
    let clip_tok_path = args.clip_l_tokenizer.as_ref()
        .ok_or("--clip-l-tokenizer is required (path to CLIP-L tokenizer file or dir)")?;
    let clip_tok = CLIPTokenizer::from_file(clip_tok_path)
        .map_err(|e| format!("clip tokenizer: {e}"))?;
    let t5_tok_path = args.t5_tokenizer.as_ref()
        .ok_or("--t5-tokenizer is required (path to T5 tokenizer.json)")?;
    let t5_max = match variant {
        FluxVariant::Schnell => 256,
        FluxVariant::Dev => 512,
    };
    let t5_tok = T5Tokenizer::from_file(t5_tok_path, t5_max)
        .map_err(|e| format!("t5 tokenizer: {e}"))?;

    if !args.width.is_multiple_of(16) || !args.height.is_multiple_of(16) {
        return Err(format!(
            "--width and --height must be multiples of 16; got {}×{}",
            args.width, args.height
        ));
    }

    let models = LoadedModels {
        transformer, cfg: cfg.clone(), variant,
        vae, clip_l, t5, clip_tok, t5_tok,
    };

    // ===== Dispatch =====
    if args.repl {
        run_repl(&args, &models)
    } else {
        run_one_prompt(&args, &models, &args.prompt, &PromptOverrides::default())
    }
}

/// Generate one batch (count = effective_batch) for one prompt.
/// `eff_*` come from `PromptOverrides` resolved against the global Args.
fn run_one_prompt(
    args: &Args,
    models: &LoadedModels,
    prompt: &str,
    over: &PromptOverrides,
) -> Result<(), String> {
    if prompt.is_empty() {
        return Err("empty prompt".into());
    }

    // ===== Reference (optional) =====
    let reference = match &args.reference {
        Some(p) => Some(FluxRefTensors::load(p)?),
        None => None,
    };

    // ===== Tokenize + encode prompt (once per prompt; reused across batch) =====
    eprintln!("[flux] encoding prompt: {:?}", prompt);
    let clip_ids = models.clip_tok.encode_padded(prompt)
        .map_err(|e| format!("clip encode: {e}"))?;
    let pooled = models.clip_l.pooled_forward(&clip_ids)
        .map_err(|e| format!("clip-l forward: {e}"))?;

    let (t5_ids, t5_mask) = models.t5_tok.encode_padded(prompt)
        .map_err(|e| format!("t5 encode: {e}"))?;
    let t5_embed = models.t5.forward(&t5_ids, Some(&t5_mask));
    let l_txt = t5_ids.len();

    if let Some(r) = &reference {
        if let Some(ref_t5) = &r.t5_embed {
            report_cos("t5_embed", &t5_embed, ref_t5);
        }
        if let Some(ref_clip) = &r.clip_pooled {
            report_cos("clip_pooled", &pooled, ref_clip);
        }
    }

    let lat_h = args.height / 8;
    let lat_w = args.width / 8;
    let in_c = models.cfg.in_channels;
    let img_seq_len = (lat_h / models.cfg.patch_size) * (lat_w / models.cfg.patch_size);

    // Resolved per-batch parameters.
    let n_steps = over.steps.or(args.steps).unwrap_or(match models.variant {
        FluxVariant::Schnell => 4,
        FluxVariant::Dev => 50,
    });
    let base_seed = over.seed.unwrap_or(args.seed);
    let guidance = over.guidance.unwrap_or(args.guidance);
    let batch = over.batch.unwrap_or(args.batch);
    let out_base = over.out.clone().unwrap_or_else(|| args.out.clone());

    // ===== Inpainting / img2img setup (once per prompt; reused across batch) =====
    let input_path: Option<&PathBuf> = over.input.as_ref().or(args.input.as_ref());
    let mask_path:  Option<&PathBuf> = over.mask.as_ref().or(args.mask.as_ref());
    let strength: f32 = over.strength.unwrap_or(args.strength).clamp(0.0, 1.0);

    let (init_latent_x0, mask_latent): (Option<Vec<f32>>, Option<Vec<f32>>) = if let Some(p) = input_path {
        eprintln!("[flux/inpaint] loading --input {}", p.display());
        let img_pixels = load_image_as_chw_f32(p, args.height, args.width)
            .map_err(|e| format!("input image: {e}"))?;
        let enc_t0 = std::time::Instant::now();
        eprintln!("[flux/inpaint] VAE encoding {}×{} image…",
            args.height, args.width);
        let force_cpu = std::env::var_os("KLEARU_FORCE_CPU_VAE_ENCODE").is_some();
        let unscaled = if force_cpu {
            models.vae.encode(&img_pixels, 1, args.height, args.width)
                .map_err(|e| format!("vae.encode: {e}"))?
        } else {
            #[cfg(feature = "metal")]
            { models.vae.encode_gpu(&img_pixels, 1, args.height, args.width)
                .map_err(|e| format!("vae.encode_gpu: {e}"))? }
            #[cfg(not(feature = "metal"))]
            { models.vae.encode(&img_pixels, 1, args.height, args.width)
                .map_err(|e| format!("vae.encode: {e}"))? }
        };
        // Surface NaN/Inf loudly — don't silently degrade.
        if unscaled.iter().any(|v| !v.is_finite()) {
            return Err("[flux/inpaint] GPU encode produced NaN/Inf. \
                Reproduce with KLEARU_VAE_TRACE=1 to identify the layer. \
                Workaround: KLEARU_FORCE_CPU_VAE_ENCODE=1".to_string());
        }
        eprintln!("[flux/inpaint] VAE encode done in {:.1}s", enc_t0.elapsed().as_secs_f32());
        let mut x0 = unscaled;
        models.vae.scale_latent(&mut x0); // (x − shift) · scale
        let mask = if let Some(mp) = mask_path {
            eprintln!("[flux/inpaint] loading --mask {}", mp.display());
            let raw = load_mask(mp, args.height, args.width)
                .map_err(|e| format!("mask: {e}"))?;
            Some(downsample_mask(&raw, args.height, args.width, 8)
                .map_err(|e| format!("downsample_mask: {e}"))?)
        } else {
            eprintln!("[flux/inpaint] no --mask: regenerating whole image (img2img)");
            None
        };
        (Some(x0), mask)
    } else { (None, None) };
    let start_step: usize = if input_path.is_some() {
        ((1.0 - strength) * n_steps as f32).floor() as usize
    } else { 0 };
    if input_path.is_some() {
        eprintln!("[flux/inpaint] strength={strength}, start_step={start_step}/{n_steps}");
    }

    for i in 0..batch {
        let seed = base_seed.wrapping_add(i as u64);
        eprintln!("[flux] batch {}/{}: seed={seed}, steps={n_steps}, guidance={guidance}",
            i + 1, batch);
        let mut latent = sample_gaussian(&[in_c, lat_h, lat_w], seed);

        // Optionally seed-replace the initial latent from the reference
        // dump so per-step comparisons isolate transformer/VAE bugs from
        // RNG differences.
        if i == 0 {
            if let Some(r) = &reference {
                if let Some(ref_init) = &r.latent_init {
                    if ref_init.len() == latent.len() {
                        eprintln!("[flux] seeding initial latent from --ref");
                        latent.copy_from_slice(ref_init);
                    } else {
                        eprintln!("[flux] (warn) ref latent_init shape mismatch \
                            (ours={}, ref={}); using sampled noise",
                            latent.len(), ref_init.len());
                    }
                }
            }
        }

        let sched_cfg = match models.variant {
            FluxVariant::Dev => FlowMatchConfig::flux_dev(),
            FluxVariant::Schnell => FlowMatchConfig::flux_schnell(),
        };
        let mut scheduler = FlowMatchScheduler::new(sched_cfg);
        scheduler.set_image_seq_len(img_seq_len);
        scheduler.set_timesteps(n_steps);

        // Inpainting / img2img: replace pure-noise latent with the noised
        // version of the encoded input at the start_step's σ. Flow-matching
        // uses linear interpolation: x_σ = (1 − σ) · x_0 + σ · noise.
        if let Some(x0) = &init_latent_x0 {
            let start_sigma = scheduler.sigma_at(start_step);
            for i in 0..latent.len() {
                latent[i] = (1.0 - start_sigma) * x0[i] + start_sigma * latent[i];
            }
        } else {
            let init_sigma = scheduler.init_noise_sigma();
            if (init_sigma - 1.0).abs() > 1e-3 {
                for v in latent.iter_mut() { *v *= init_sigma; }
            }
        }

        let g = if models.cfg.guidance_embeds { guidance } else { 0.0 };
        let sample_start = std::time::Instant::now();
        for step in start_step..n_steps {
            let t = scheduler.timestep_f32(step);
            let step_t0 = std::time::Instant::now();
            eprintln!("[flux]   step {}/{n_steps} starting (σ={:.4}, t={:.1})…",
                step + 1, scheduler.sigma_at(step), t);
            let v = models.transformer.forward(
                &latent, lat_h, lat_w,
                &t5_embed, l_txt,
                &pooled, t, g,
            );
            latent = scheduler.step(&v, step, &latent);

            // Per-step blend: anchor the unmasked region to the input
            // image's diffusion trajectory so it doesn't drift. Flow-matching
            // q_sample at next σ is `(1−σ) · x_0 + σ · noise`. We re-sample
            // noise per step (RePaint convention) for the unmasked side.
            if let Some(x0) = &init_latent_x0 {
                let next_sigma = if step + 1 < scheduler.sigmas().len() {
                    scheduler.sigma_at(step + 1)
                } else { 0.0 };
                let noise = sample_gaussian(&[in_c, lat_h, lat_w],
                    seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(step as u64 + 1));
                let lat_pixel = lat_h * lat_w;
                for j in 0..latent.len() {
                    let m = match &mask_latent {
                        Some(mk) => mk[j % lat_pixel],
                        None => 1.0,
                    };
                    let unmasked = (1.0 - next_sigma) * x0[j] + next_sigma * noise[j];
                    latent[j] = m * latent[j] + (1.0 - m) * unmasked;
                }
            }

            eprintln!("[flux]   step {}/{n_steps} done in {:.1}s (total {:.1}s)",
                step + 1,
                step_t0.elapsed().as_secs_f32(),
                sample_start.elapsed().as_secs_f32());
        }

        eprintln!("[flux] decoding via VAE…");
        models.vae.unscale_latent(&mut latent);

        if i == 0 {
            if let Some(r) = &reference {
                if let Some(ref_lat) = &r.latent_final {
                    report_cos("latent_final", &latent, ref_lat);
                }
            }
        }

        let img = models.vae.decode_with_dims(&latent, 1, lat_h, lat_w);

        if i == 0 {
            if let Some(r) = &reference {
                if let Some(ref_img) = &r.decoded_image {
                    report_cos("decoded_image", &img, ref_img);
                }
            }
        }

        let h = args.height; let w = args.width;
        let out_path = batch_filename(&out_base, i, batch);
        save_png(&img, h, w, &out_path)
            .map_err(|e| format!("save png: {e}"))?;
        eprintln!("[flux] saved {} ({}×{})", out_path.display(), w, h);
    }
    Ok(())
}

fn run_repl(args: &Args, models: &LoadedModels) -> Result<(), String> {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    eprintln!();
    eprintln!("[flux] REPL ready. Type a prompt and press Enter.");
    eprintln!("[flux] Inline overrides: `your prompt :: --steps 30 --seed 7 --batch 4 --out img.png`");
    eprintln!("[flux] Empty line repeats the previous prompt; `exit` or Ctrl-D to quit.");
    eprintln!();
    let mut last_line: Option<String> = None;
    loop {
        print!("flux> ");
        stdout.flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => { eprintln!(); break; }   // EOF
            Ok(_) => {}
            Err(e) => return Err(format!("stdin: {e}")),
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if let Some(prev) = &last_line {
                let l = prev.clone();
                if let Err(e) = handle_repl_line(args, models, &l) {
                    eprintln!("[flux] error: {e}");
                }
            }
            continue;
        }
        if matches!(trimmed, "exit" | "quit" | ":q") { break; }
        last_line = Some(trimmed.to_string());
        if let Err(e) = handle_repl_line(args, models, trimmed) {
            eprintln!("[flux] error: {e}");
        }
    }
    Ok(())
}

fn handle_repl_line(args: &Args, models: &LoadedModels, line: &str) -> Result<(), String> {
    let (prompt, overrides) = parse_prompt_line(line)?;
    if prompt.is_empty() {
        return Err("empty prompt".into());
    }
    run_one_prompt(args, models, &prompt, &overrides)
}

/// Resolve the four file paths from CLI args. Either `--checkpoint <dir>`
/// auto-discovers all four, or every individual flag must be provided.
fn resolve_paths(args: &Args) -> Result<FluxPaths, String> {
    if let Some(dir) = &args.checkpoint_dir {
        if let Some(p) = discover_flux_paths(dir) {
            return Ok(p);
        }
        return Err(format!(
            "could not auto-discover Flux files in {} \
             (expected flux1-*.safetensors, ae.safetensors, clip-l.safetensors, t5xxl_*.safetensors)",
            dir.display()
        ));
    }
    Ok(FluxPaths {
        transformer: args.transformer.clone()
            .ok_or("--transformer or --checkpoint is required")?,
        vae: args.vae.clone()
            .ok_or("--vae or --checkpoint is required")?,
        clip_l: args.clip_l.clone()
            .ok_or("--clip-l or --checkpoint is required")?,
        t5: args.t5.clone()
            .ok_or("--t5 or --checkpoint is required")?,
    })
}

/// Sample standard-normal noise of the given shape using a deterministic
/// xoshiro-like PRNG seeded by `seed`. Box-Muller transform.
fn sample_gaussian(shape: &[usize], seed: u64) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut next_u32 = move || -> u32 {
        // splitmix64 step.
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        z as u32
    };
    let mut to_unit = || -> f32 {
        // Avoid 0.0 (log of 0 in BM): map to (0, 1].
        let u = next_u32();
        ((u as f64 + 1.0) / (u32::MAX as f64 + 2.0)) as f32
    };
    while out.len() < n {
        let u1 = to_unit();
        let u2 = to_unit();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out
}
