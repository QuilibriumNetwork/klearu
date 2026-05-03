//! sd_generate — Stable Diffusion image generation (SD 1.5 or SDXL).
//!
//! Branches on the detected variant. With weights uninitialised (no
//! checkpoint loading wired yet), output is noise — the pipeline shape
//! is exercised end-to-end so once weight loading lands, this binary
//! becomes the working e2e demo.

use std::path::PathBuf;
use std::process::ExitCode;

use klearu_diffusion::config::{CheckpointConfig, SDVariant};
use klearu_diffusion::image_io::save_png;
use klearu_diffusion::scheduler::{DDIMScheduler, Scheduler};
use klearu_diffusion::text_encoder::{CLIPTextModel, SDXLDualTextEncoder};
use klearu_diffusion::tokenizer::CLIPTokenizer;
use klearu_diffusion::unet::{SDXLAdditionalConditioning, UNet2DConditionModel};
use klearu_diffusion::vae::AutoencoderKL;
use klearu_diffusion::weight::{ComponentTensors, SDFormat, SingleFileLoader};

/// Shallow-clone a CLIPTextModel by re-constructing from config and copying
/// loaded weights. We can't `derive(Clone)` because the underlying Linear
/// has Vec data. Allocates ~hidden_size² × num_layers floats.
fn text_l_clone(src: &CLIPTextModel) -> CLIPTextModel {
    let mut copy = CLIPTextModel::from_config(src.config.clone());
    copy.token_embedding.clone_from(&src.token_embedding);
    copy.position_embedding.clone_from(&src.position_embedding);
    copy.final_layer_norm.gamma.clone_from(&src.final_layer_norm.gamma);
    copy.final_layer_norm.beta.clone_from(&src.final_layer_norm.beta);
    for (dst, s) in copy.layers.iter_mut().zip(src.layers.iter()) {
        dst.layer_norm1.gamma.clone_from(&s.layer_norm1.gamma);
        dst.layer_norm1.beta.clone_from(&s.layer_norm1.beta);
        dst.self_attn.to_q.weight.clone_from(&s.self_attn.to_q.weight);
        dst.self_attn.to_k.weight.clone_from(&s.self_attn.to_k.weight);
        dst.self_attn.to_v.weight.clone_from(&s.self_attn.to_v.weight);
        dst.self_attn.to_out.weight.clone_from(&s.self_attn.to_out.weight);
        dst.self_attn.to_out.bias.clone_from(&s.self_attn.to_out.bias);
        dst.layer_norm2.gamma.clone_from(&s.layer_norm2.gamma);
        dst.layer_norm2.beta.clone_from(&s.layer_norm2.beta);
        dst.mlp.fc1.weight.clone_from(&s.mlp.fc1.weight);
        dst.mlp.fc1.bias.clone_from(&s.mlp.fc1.bias);
        dst.mlp.fc2.weight.clone_from(&s.mlp.fc2.weight);
        dst.mlp.fc2.bias.clone_from(&s.mlp.fc2.bias);
    }
    copy
}

fn count_cross_attn_blocks(unet: &UNet2DConditionModel) -> usize {
    let mut n = 0usize;
    for db in &unet.down_blocks {
        if let Some(attns) = &db.attentions {
            for t2d in attns { n += t2d.blocks.len(); }
        }
    }
    n += unet.mid_block.attn.blocks.len();
    for ub in &unet.up_blocks {
        if let Some(attns) = &ub.attentions {
            for t2d in attns { n += t2d.blocks.len(); }
        }
    }
    n
}

/// Reference tensors loaded from a `--ref` safetensors file (produced by
/// `scripts/dump_sd_reference.py`). Used to compare klearu's intermediate
/// outputs against diffusers at the same prompt/seed/step.
struct RefTensors {
    text_emb_cond: Vec<f32>,
    text_emb_uncond: Vec<f32>,
    latent_init: Vec<f32>,
    /// Per-step eps for [step0, step12, step24]: (uncond, cond).
    eps_at_step: std::collections::HashMap<usize, (Vec<f32>, Vec<f32>)>,
    latent_final: Vec<f32>,
    vae_image: Vec<f32>,
}

impl RefTensors {
    fn load(path: &std::path::Path) -> Result<Self, String> {
        use safetensors::SafeTensors;
        let buf = std::fs::read(path).map_err(|e| format!("read ref: {e}"))?;
        let st = SafeTensors::deserialize(&buf).map_err(|e| format!("parse ref: {e}"))?;
        let tensor = |name: &str| -> Result<Vec<f32>, String> {
            let t = st.tensor(name).map_err(|e| format!("missing tensor {name}: {e}"))?;
            let bytes = t.data();
            if !bytes.len().is_multiple_of(4) {
                return Err(format!("{name}: byte length {} not multiple of 4", bytes.len()));
            }
            Ok(bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let opt = |name: &str| -> Option<Vec<f32>> { tensor(name).ok() };
        let mut eps_at_step = std::collections::HashMap::new();
        for s in [0usize, 12, 24] {
            if let (Some(u), Some(c)) = (
                opt(&format!("eps_uncond_step{s}")),
                opt(&format!("eps_cond_step{s}")),
            ) {
                eps_at_step.insert(s, (u, c));
            }
        }
        Ok(Self {
            text_emb_cond: tensor("text_emb_cond")?,
            text_emb_uncond: tensor("text_emb_uncond")?,
            latent_init: tensor("latent_init")?,
            eps_at_step,
            latent_final: tensor("latent_final").unwrap_or_default(),
            vae_image: tensor("vae_image").unwrap_or_default(),
        })
    }
}

/// Element-wise comparison: prints mean|ours-ref|, max|diff|, cos_sim, and
/// magnitude ratio. The first metric is the most diagnostic — values >1% of
/// either tensor's mean_abs indicate divergence.
fn compare_tensors(label: &str, ours: &[f32], reference: &[f32]) {
    if ours.len() != reference.len() {
        eprintln!("  [REF] {label:<24} ✗ length mismatch: ours={}, ref={}", ours.len(), reference.len());
        return;
    }
    let mut sum_ours = 0.0f64; let mut sum_ref = 0.0f64;
    let mut sum_diff = 0.0f64; let mut max_diff = 0.0f32;
    let mut dot = 0.0f64; let mut sq_ours = 0.0f64; let mut sq_ref = 0.0f64;
    let mut nan_or_inf_ours = 0; let mut nan_or_inf_ref = 0;
    for (&o, &r) in ours.iter().zip(reference.iter()) {
        if !o.is_finite() { nan_or_inf_ours += 1; continue; }
        if !r.is_finite() { nan_or_inf_ref += 1; continue; }
        sum_ours += o.abs() as f64;
        sum_ref += r.abs() as f64;
        let d = (o - r).abs();
        sum_diff += d as f64;
        if d > max_diff { max_diff = d; }
        dot += (o as f64) * (r as f64);
        sq_ours += (o as f64) * (o as f64);
        sq_ref += (r as f64) * (r as f64);
    }
    let n = ours.len() as f64;
    let mean_ours = sum_ours / n;
    let mean_ref = sum_ref / n;
    let mean_diff = sum_diff / n;
    let cos = if sq_ours > 0.0 && sq_ref > 0.0 { dot / (sq_ours.sqrt() * sq_ref.sqrt()) } else { 0.0 };
    let mag_ratio = if mean_ref > 0.0 { mean_ours / mean_ref } else { 0.0 };
    let mark = if (mean_diff / mean_ref.max(1e-8)) < 0.01 { "✓" } else { "✗" };
    eprintln!("  [REF] {label:<24} {mark} mean|ours|={mean_ours:.4}, mean|ref|={mean_ref:.4}, mag_ratio={mag_ratio:.4}, mean|diff|={mean_diff:.5}, max|diff|={max_diff:.4}, cos={cos:.5}");
    if nan_or_inf_ours > 0 || nan_or_inf_ref > 0 {
        eprintln!("    ⚠ non-finite: ours={nan_or_inf_ours}, ref={nan_or_inf_ref}");
    }
}

fn deterministic_noise(n_floats: usize, seed: u64) -> Vec<f32> {
    // Simple splitmix64-derived gaussian (Box-Muller). Enough for testing
    // pipeline shape; real generation replaces with the user's preferred RNG.
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let mut out = vec![0.0f32; n_floats];
    let mut i = 0;
    while i + 1 < n_floats {
        let u1 = (next_u64() >> 11) as f32 / ((1u64 << 53) as f32);
        let u2 = (next_u64() >> 11) as f32 / ((1u64 << 53) as f32);
        let r = (-2.0 * u1.max(1e-9).ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out[i] = r * theta.cos();
        out[i + 1] = r * theta.sin();
        i += 2;
    }
    if i < n_floats {
        let u1 = (next_u64() >> 11) as f32 / ((1u64 << 53) as f32);
        out[i] = (-2.0 * u1.max(1e-9).ln()).sqrt();
    }
    out
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut checkpoint: Option<PathBuf> = None;
    let mut single_file: Option<PathBuf> = None;
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut tokenizer_2_path: Option<PathBuf> = None;
    let mut force_variant: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut negative_prompt: Option<String> = None;
    let mut cfg_scale: f32 = 7.5;
    let mut steps: usize = 25;
    let mut out_path: Option<PathBuf> = None;
    // `--seed N` pins the seed; otherwise we randomise per generation.
    // In `--repl` each iteration also gets a fresh random seed (so the same
    // prompt yields a different sample each time the user re-types it).
    let mut explicit_seed: Option<u64> = None;
    // GPU residence default-on. The bugs that previously broke it (GELU NaN,
    // CLIP qkv_bias, DDIM steps_offset/set_alpha_to_one, VAE post_quant_conv)
    // are all fixed and verified element-wise vs diffusers. Use
    // `--no-gpu-residence` for the legacy per-call-Metal CPU fallback.
    let mut gpu_residence: bool = true;
    let mut ref_path: Option<PathBuf> = None;
    let mut repl: bool = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => { i += 1; checkpoint = Some(PathBuf::from(&args[i])); }
            "--single-file" => { i += 1; single_file = Some(PathBuf::from(&args[i])); }
            "--tokenizer" => { i += 1; tokenizer_path = Some(PathBuf::from(&args[i])); }
            "--tokenizer-2" => { i += 1; tokenizer_2_path = Some(PathBuf::from(&args[i])); }
            "--variant" => { i += 1; force_variant = Some(args[i].clone()); }
            "--prompt" => { i += 1; prompt = Some(args[i].clone()); }
            "--negative-prompt" => { i += 1; negative_prompt = Some(args[i].clone()); }
            "--cfg-scale" => { i += 1; cfg_scale = args[i].parse().expect("f32"); }
            "--steps" => { i += 1; steps = args[i].parse().expect("usize"); }
            "--out" => { i += 1; out_path = Some(PathBuf::from(&args[i])); }
            "--seed" => { i += 1; explicit_seed = Some(args[i].parse().expect("u64")); }
            "--gpu-residence" => { gpu_residence = true; }
            "--no-gpu-residence" => { gpu_residence = false; }
            "--ref" => { i += 1; ref_path = Some(PathBuf::from(&args[i])); }
            "--repl" => { repl = true; }
            "--help" | "-h" => {
                eprintln!("usage:");
                eprintln!("  sd_generate --checkpoint DIR --prompt \"...\" --out OUT.png");
                eprintln!("    [--negative-prompt \"...\"] [--cfg-scale 7.5]");
                eprintln!("    [--steps 25] [--seed 42]");
                eprintln!("  sd_generate --single-file FILE.safetensors --tokenizer tokenizer.json");
                eprintln!("    --prompt \"...\" --out OUT.png");
                eprintln!("    [--variant sd15|sdxl] [--tokenizer-2 tokenizer_2.json]");
                eprintln!("    [--negative-prompt \"...\"] [--cfg-scale 7.5] [--steps] [--seed]");
                eprintln!("    [--gpu-residence  (experimental: full GPU residence path)]");
                return ExitCode::SUCCESS;
            }
            other => { eprintln!("unknown flag: {other}"); return ExitCode::FAILURE; }
        }
        i += 1;
    }
    if !repl && prompt.is_none() {
        eprintln!("--prompt required (or --repl for interactive mode)");
        return ExitCode::FAILURE;
    }
    if !repl && out_path.is_none() {
        eprintln!("--out required (or --repl for interactive mode)");
        return ExitCode::FAILURE;
    }
    let initial_prompt = prompt.unwrap_or_default();
    let initial_out_path = out_path.unwrap_or_else(|| PathBuf::from("img.png"));
    let negative_prompt = negative_prompt.unwrap_or_default();

    // Load reference tensors for diffusers comparison, if --ref given. When
    // present we override our noise with the reference's so inputs match.
    let reference: Option<RefTensors> = match ref_path {
        Some(p) => {
            eprintln!("loading reference tensors from {}", p.display());
            match RefTensors::load(&p) {
                Ok(r) => {
                    eprintln!("  ✓ ref loaded: text_emb_cond={}, latent_init={}, eps_checkpoints={:?}",
                              r.text_emb_cond.len(), r.latent_init.len(),
                              {let mut k: Vec<_> = r.eps_at_step.keys().copied().collect(); k.sort(); k});
                    Some(r)
                }
                Err(e) => { eprintln!("ref: {e}"); return ExitCode::FAILURE; }
            }
        }
        None => None,
    };

    // Resolve config and weight source. Two paths:
    //   (A) --checkpoint DIR  (Diffusers folder format with per-component subdirs)
    //   (B) --single-file FILE.safetensors  (CompVis/A1111 all-in-one)
    enum WeightSrc {
        Folder(PathBuf),
        Single(SingleFileLoader),
    }
    let (cfg, weight_src, fake_root) = match (checkpoint.clone(), single_file.clone()) {
        (Some(p), None) => {
            let cfg = match CheckpointConfig::from_dir(&p) {
                Ok(c) => c,
                Err(e) => { eprintln!("config: {e}"); return ExitCode::FAILURE; }
            };
            (cfg, WeightSrc::Folder(p.clone()), p)
        }
        (None, Some(sf)) => {
            eprintln!("opening single-file checkpoint {}", sf.display());
            let loader = match SingleFileLoader::open(&sf) {
                Ok(l) => l,
                Err(e) => { eprintln!("single-file: {e}"); return ExitCode::FAILURE; }
            };
            // Surface modelspec metadata if the publisher (Civitai / A1111) included it.
            if let Some(t) = &loader.metadata.title {
                eprintln!("modelspec title: {t:?}");
            }
            if let Some(a) = &loader.metadata.architecture {
                eprintln!("modelspec architecture: {a:?}");
            }
            if let Some(p) = &loader.metadata.prediction_type {
                eprintln!("modelspec prediction_type: {p:?}");
            }
            if let Some(r) = &loader.metadata.resolution {
                eprintln!("modelspec resolution: {r:?}");
            }
            if let Some(l) = &loader.metadata.license {
                eprintln!("modelspec license: {l:?}");
            }
            eprintln!("detected variant: {:?}", loader.variant);
            // Pick config defaults based on detected (or forced) variant.
            let variant = match force_variant.as_deref() {
                Some("sd15") => SDFormat::Sd15,
                Some("sdxl") => SDFormat::Sdxl,
                _ => loader.variant,
            };
            let fake_root = sf.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
            let mut cfg = match variant {
                SDFormat::Sd15 => CheckpointConfig::sd15_default(fake_root.clone()),
                SDFormat::Sdxl => CheckpointConfig::sdxl_default(fake_root.clone()),
                SDFormat::Unknown => {
                    eprintln!("could not detect SD variant; pass --variant sd15 or --variant sdxl");
                    return ExitCode::FAILURE;
                }
            };
            // Override defaults from modelspec when the publisher specified them.
            if let Some(p) = &loader.metadata.prediction_type {
                cfg.scheduler.prediction_type = p.clone();
                eprintln!("(scheduler.prediction_type overridden to {:?} from modelspec)", p);
            }
            if let Some((w, h)) = loader.metadata.resolution_pixels() {
                // SD's UNet sample_size is the latent side (image / 8).
                let new_sample = (w.max(h) / 8) as usize;
                if new_sample != cfg.unet.sample_size {
                    eprintln!("(unet.sample_size {} → {} from modelspec resolution {}x{})",
                        cfg.unet.sample_size, new_sample, w, h);
                    cfg.unet.sample_size = new_sample;
                    cfg.vae.sample_size = (w.max(h)) as usize;
                }
            }
            (cfg, WeightSrc::Single(loader), fake_root)
        }
        (Some(_), Some(_)) => {
            eprintln!("--checkpoint and --single-file are mutually exclusive");
            return ExitCode::FAILURE;
        }
        (None, None) => {
            eprintln!("must specify --checkpoint DIR or --single-file FILE.safetensors");
            return ExitCode::FAILURE;
        }
    };
    let variant = cfg.variant();
    eprintln!("variant: {:?}", variant);
    eprintln!("prompt: {initial_prompt:?}{}", if repl { " (and stdin in REPL mode)" } else { "" });
    let seed_label: String = match explicit_seed {
        Some(s) => format!("{s} (pinned)"),
        None => "random per-generation".into(),
    };
    eprintln!("steps: {steps}, seed: {seed_label}");
    // Time-derived RNG seed source. Per-iteration we derive a unique 64-bit
    // seed: nanosecond clock XOR'd with iteration counter so back-to-back
    // generations don't collide.
    fn random_seed() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH)
            .map(|d| (d.as_nanos() as u64).wrapping_mul(0x9E3779B97F4A7C15))
            .unwrap_or(42)
    }

    // Construct components and load weights — folder or single-file source.
    let load_component = |name: &str| -> Option<ComponentTensors> {
        match &weight_src {
            WeightSrc::Folder(root) => {
                ComponentTensors::open_dir(&root.join(name)).ok()
            }
            WeightSrc::Single(loader) => loader.component(name).ok(),
        }
    };

    let mut text_l = CLIPTextModel::from_config(cfg.text_encoder.clone());
    eprintln!("loading text_encoder weights...");
    if let Some(c) = load_component("text_encoder") {
        match text_l.load_from(&c) {
            Ok(()) => eprintln!("  ✓ text_encoder loaded"),
            Err(e) => eprintln!("  ⚠ text_encoder: {e} (continuing with zero weights)"),
        }
    } else {
        eprintln!("  ⚠ text_encoder unavailable (continuing with zero weights)");
    }

    let mut dual = cfg.text_encoder_2.as_ref().map(|t2| {
        SDXLDualTextEncoder::from_configs(cfg.text_encoder.clone(), t2.clone())
    });
    if let Some(d) = &mut dual {
        d.clip_l = text_l_clone(&text_l);
        eprintln!("loading text_encoder_2 (CLIP-G) weights...");
        if let Some(c) = load_component("text_encoder_2") {
            match d.clip_g.load_from(&c) {
                Ok(()) => eprintln!("  ✓ text_encoder_2 loaded"),
                Err(e) => eprintln!("  ⚠ text_encoder_2: {e}"),
            }
        } else {
            eprintln!("  ⚠ text_encoder_2 unavailable");
        }
    }

    let mut unet = UNet2DConditionModel::from_config(cfg.unet.clone());
    eprintln!("loading unet weights (largest, may take a moment)...");
    if let Some(c) = load_component("unet") {
        match unet.load_from(&c) {
            Ok(()) => eprintln!("  ✓ unet loaded"),
            Err(e) => eprintln!("  ⚠ unet: {e}"),
        }
    } else {
        eprintln!("  ⚠ unet unavailable");
    }

    let mut vae = AutoencoderKL::from_config(cfg.vae.clone());
    eprintln!("loading vae weights...");
    if let Some(c) = load_component("vae") {
        match vae.load_from(&c) {
            Ok(()) => eprintln!("  ✓ vae loaded"),
            Err(e) => eprintln!("  ⚠ vae: {e}"),
        }
    } else {
        eprintln!("  ⚠ vae unavailable");
    }
    let _ = fake_root; // suppress unused
    let mut scheduler = DDIMScheduler::new(&cfg.scheduler);
    scheduler.set_timesteps(steps);

    // Tokenize prompt with the CLIP BPE tokenizer (real, BOS+EOS padded to 77).
    // For folder mode, tokenizer.json is auto-discovered under <checkpoint>/tokenizer/.
    // For single-file mode, the user must pass --tokenizer.
    let tok_l_path = match (&weight_src, &tokenizer_path) {
        (_, Some(p)) => p.clone(),
        (WeightSrc::Folder(root), None) => root.join("tokenizer").join("tokenizer.json"),
        (WeightSrc::Single(_), None) => {
            eprintln!("--single-file requires --tokenizer FILE.json (the CLIP tokenizer.json)");
            eprintln!("(typically downloaded alongside the checkpoint, or from any matching SD repo)");
            return ExitCode::FAILURE;
        }
    };
    let tok_l = match CLIPTokenizer::from_file(&tok_l_path) {
        Ok(t) => t,
        Err(e) => { eprintln!("CLIP-L tokenizer load: {e}"); return ExitCode::FAILURE; }
    };

    // Per-prompt loop. Single-shot mode runs once and exits; --repl loops
    // reading prompts from stdin and auto-naming outputs `img_NNN.png`.
    // Models, tokenizer, and scheduler stay loaded across iterations.
    let mut iteration: usize = 0;
    if repl {
        eprintln!("\nREPL mode: type a prompt and press Enter. Empty line, 'exit', 'quit', or Ctrl-D to stop.");
    }
    let exit_code: ExitCode = 'main: loop {
        // Resolve prompt + output path for this iteration.
        let prompt: String;
        let out_path: PathBuf;
        if iteration == 0 && (!initial_prompt.is_empty() || !repl) {
            prompt = initial_prompt.clone();
            out_path = initial_out_path.clone();
        } else if !repl {
            break 'main ExitCode::SUCCESS;
        } else {
            use std::io::Write;
            eprint!("\nprompt> ");
            std::io::stderr().flush().ok();
            let mut line = String::new();
            match std::io::stdin().read_line(&mut line) {
                Ok(0) | Err(_) => break 'main ExitCode::SUCCESS,
                Ok(_) => {}
            }
            let p = line.trim();
            if p.is_empty() || p == "exit" || p == "quit" || p == ":q" {
                break 'main ExitCode::SUCCESS;
            }
            prompt = p.to_string();
            // Auto-name output. Place beside the initial --out path so
            // the user's chosen directory is respected.
            let parent = initial_out_path.parent().unwrap_or(std::path::Path::new("."));
            out_path = parent.join(format!("img_{iteration:03}.png"));
        }
        iteration += 1;
        eprintln!("\n=== generating: {prompt:?} → {} ===", out_path.display());

        let ids_l = match tok_l.encode_padded(&prompt) {
        Ok(v) => v,
        Err(e) => { eprintln!("CLIP-L encode: {e}"); if repl { continue 'main; } break 'main ExitCode::FAILURE; }
    };
    eprintln!("CLIP-L tokens: {} (first 12: {:?})", ids_l.len(), &ids_l[..12.min(ids_l.len())]);
    // Find the position of the first 49407 (EOS terminator) — everything
    // after it is pad. If padding starts at position 1, CLIP-L tokenizer
    // produced no content tokens (suspicious unless prompt is empty).
    let l_eos = ids_l.iter().position(|&t| t == 49407).unwrap_or(ids_l.len());
    eprintln!("  CLIP-L content length: {l_eos} tokens (excl. BOS, before pad)");
    let ids_g: Vec<u32> = if matches!(variant, SDVariant::Sdxl) {
        let tok_g_path: Option<PathBuf> = match (&weight_src, &tokenizer_2_path) {
            (_, Some(p)) => Some(p.clone()),
            (WeightSrc::Folder(root), None) => Some(root.join("tokenizer_2").join("tokenizer.json")),
            (WeightSrc::Single(_), None) => {
                eprintln!("(SDXL detected but no --tokenizer-2; reusing CLIP-L ids for CLIP-G)");
                None
            }
        };
        match tok_g_path {
            Some(p) => match CLIPTokenizer::from_file(&p) {
                Ok(t) => {
                    // SDXL CLIP-G pads with token id 0, NOT 49407 (EOS) — see
                    // ComfyUI sdxl_clip.py SDXLClipGTokenizer(pad_with_end=False).
                    let pad_g = if matches!(variant, SDVariant::Sdxl) { 0u32 } else { 49407u32 };
                    t.encode_padded_with(&prompt, pad_g).unwrap_or_else(|_| ids_l.clone())
                }
                Err(_) => {
                    eprintln!("(no tokenizer_2 found; reusing CLIP-L ids for CLIP-G)");
                    ids_l.clone()
                }
            },
            None => ids_l.clone(),
        }
    } else {
        ids_l.clone()
    };
    // CLIP-G is SDXL-only — skip the diagnostic for SD 1.5.
    if matches!(variant, SDVariant::Sdxl) {
        eprintln!("CLIP-G tokens: {} (first 12: {:?})", ids_g.len(), &ids_g[..12.min(ids_g.len())]);
        if ids_g == ids_l {
            eprintln!("  ⚠ CLIP-G ids identical to CLIP-L (no separate tokenizer_2). For SDXL,");
            eprintln!("    CLIP-G should pad with id=0; CLIP-L pads with id=49407. Without a");
            eprintln!("    proper tokenizer_2, CLIP-G is fed CLIP-L's EOS-padded ids and produces");
            eprintln!("    out-of-distribution hidden states → cross-attention is degenerate.");
        } else {
            let g_pad_count = ids_g.iter().rev().take_while(|&&t| t == 0).count();
            let l_pad_count = ids_l.iter().rev().take_while(|&&t| t == 49407).count();
            eprintln!("  CLIP-G uses pad=0 ({g_pad_count} pads); CLIP-L uses pad=49407 ({l_pad_count} pads)");
        }
    }
    let placeholder_ids = ids_l;

    // Helper: print min/max/mean-abs/finite-count for a tensor.
    fn stats(name: &str, x: &[f32]) {
        let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
        let mut sum_abs = 0.0f64; let mut nan = 0usize; let mut inf = 0usize;
        for &v in x {
            if v.is_nan() { nan += 1; }
            else if v.is_infinite() { inf += 1; }
            else { mn = mn.min(v); mx = mx.max(v); sum_abs += v.abs() as f64; }
        }
        let mean_abs = sum_abs / x.len() as f64;
        eprintln!("  {name:<22} len={}, min={mn:.4}, max={mx:.4}, mean_abs={mean_abs:.6}, NaN={nan}, Inf={inf}",
                  x.len());
    }

    // Encode the conditional (positive) prompt.
    let (text_emb, pooled_g) = match variant {
        SDVariant::Sdxl => {
            match &dual {
                Some(d) => {
                    let r = d.forward(&placeholder_ids, &ids_g);
                    match r {
                        Ok((emb, p)) => (emb, Some(p)),
                        Err(e) => { eprintln!("text encode: {e}"); if repl { continue 'main; } break 'main ExitCode::FAILURE; }
                    }
                }
                None => { eprintln!("SDXL detected but no dual encoder"); if repl { continue 'main; } break 'main ExitCode::FAILURE; }
            }
        }
        _ => {
            match text_l.forward(&placeholder_ids) {
                Ok(emb) => (emb, None),
                Err(e) => { eprintln!("text encode: {e}"); if repl { continue 'main; } break 'main ExitCode::FAILURE; }
            }
        }
    };

    eprintln!("text encode complete:");
    stats("  text_emb (cond)", &text_emb);
    if let Some(p) = &pooled_g { stats("  pooled_g (cond)", p); }

    // Encode the unconditional / negative prompt for classifier-free guidance.
    // SD/SDXL checkpoints are trained with CFG dropout (10% empty prompt) and
    // require CFG at inference for coherent samples — without it the output is
    // essentially noise.
    let do_cfg = cfg_scale > 1.0 + 1e-6;
    let (text_emb_uncond, pooled_g_uncond) = if do_cfg {
        let neg_ids_l = match tok_l.encode_padded(&negative_prompt) {
            Ok(v) => v,
            Err(e) => { eprintln!("CLIP-L encode (neg): {e}"); if repl { continue 'main; } break 'main ExitCode::FAILURE; }
        };
        let neg_ids_g: Vec<u32> = if matches!(variant, SDVariant::Sdxl) {
            // Reuse same tokenizer-2 path resolution as the positive branch.
            let tok_g_path: Option<PathBuf> = match (&weight_src, &tokenizer_2_path) {
                (_, Some(p)) => Some(p.clone()),
                (WeightSrc::Folder(root), None) => Some(root.join("tokenizer_2").join("tokenizer.json")),
                (WeightSrc::Single(_), None) => None,
            };
            match tok_g_path {
                Some(p) => match CLIPTokenizer::from_file(&p) {
                    Ok(t) => {
                        let pad_g = if matches!(variant, SDVariant::Sdxl) { 0u32 } else { 49407u32 };
                        t.encode_padded_with(&negative_prompt, pad_g).unwrap_or_else(|_| neg_ids_l.clone())
                    }
                    Err(_) => neg_ids_l.clone(),
                },
                None => neg_ids_l.clone(),
            }
        } else {
            neg_ids_l.clone()
        };
        match variant {
            SDVariant::Sdxl => {
                match &dual {
                    Some(d) => match d.forward(&neg_ids_l, &neg_ids_g) {
                        Ok((e, p)) => (Some(e), Some(p)),
                        Err(e) => { eprintln!("uncond text encode: {e}"); if repl { continue 'main; } break 'main ExitCode::FAILURE; }
                    },
                    None => { eprintln!("SDXL but no dual encoder"); if repl { continue 'main; } break 'main ExitCode::FAILURE; }
                }
            }
            _ => match text_l.forward(&neg_ids_l) {
                Ok(emb) => (Some(emb), None),
                Err(e) => { eprintln!("uncond text encode: {e}"); if repl { continue 'main; } break 'main ExitCode::FAILURE; }
            },
        }
    } else {
        (None, None)
    };

    if let Some(uemb) = &text_emb_uncond {
        stats("  text_emb (uncond)", uemb);
        // Compare to cond: if mean|cond - uncond| is tiny, CLIP isn't differentiating
        // — almost certainly a tokenizer / CLIP-G pad-token / weight-loading issue.
        let mut diff_sum = 0.0f64;
        let mut max_diff = 0.0f32;
        for (c, u) in text_emb.iter().zip(uemb.iter()) {
            let d = (c - u).abs();
            diff_sum += d as f64;
            if d > max_diff { max_diff = d; }
        }
        let mean_diff = diff_sum / text_emb.len() as f64;
        eprintln!("  text_emb cond-vs-uncond: mean|diff|={mean_diff:.6}, max|diff|={max_diff:.4}");
        if mean_diff < 1e-3 {
            eprintln!("  ⚠ CLIP outputs are nearly identical for cond/uncond. Cross-attn will not differentiate.");
        }
    }

    // Reference comparison: how close is our CLIP output to diffusers'?
    if let Some(r) = &reference {
        compare_tensors("text_emb_cond", &text_emb, &r.text_emb_cond);
        if let Some(u) = &text_emb_uncond {
            compare_tensors("text_emb_uncond", u, &r.text_emb_uncond);
        }
    }
    if let Some(pu) = &pooled_g_uncond {
        if let Some(p) = &pooled_g {
            let mut sum = 0.0f64;
            for (c, u) in p.iter().zip(pu.iter()) { sum += (c - u).abs() as f64; }
            eprintln!("  pooled_g cond-vs-uncond: mean|diff|={:.6}", sum / p.len() as f64);
        }
    }

    // Initial noisy latent.
    let latent_h = cfg.unet.sample_size;
    let latent_w = cfg.unet.sample_size;
    let latent_c = cfg.unet.in_channels;
    // Pin to --seed only on the very first iteration; subsequent REPL
    // iterations always randomise so the same prompt yields a different
    // sample. (`--ref` overrides — uses diffusers' fixed noise for
    // numerical comparison.)
    let seed: u64 = match (explicit_seed, iteration == 1) {
        (Some(s), true) => s,
        _ => random_seed().wrapping_add(iteration as u64),
    };
    eprintln!("seed for this iteration: {seed}");
    let mut latent = if let Some(r) = &reference {
        // Use diffusers' noise so input matches. This isolates the UNet
        // comparison from any RNG-driven divergence.
        eprintln!("  [REF] overriding latent_init from reference (diffusers noise)");
        r.latent_init.clone()
    } else {
        deterministic_noise(latent_c * latent_h * latent_w, seed)
    };
    eprintln!("starting denoise: {} steps, latent {}×{}×{}, cfg_scale={cfg_scale}",
        scheduler.timesteps().len(), latent_c, latent_h, latent_w);

    let text_seq_cond = text_emb.len() / cfg.unet.cross_attention_dim;

    // Build batched [uncond | cond] inputs when CFG is on, so the UNet runs
    // once per step at n=2 instead of twice at n=1. ~Halves wall-clock per
    // step (+ small overhead from larger tensors).
    let (text_emb_run, pooled_run, n_batch) = if do_cfg {
        let mut emb = Vec::with_capacity(text_emb.len() * 2);
        emb.extend_from_slice(text_emb_uncond.as_ref().unwrap());
        emb.extend_from_slice(&text_emb);
        let pooled = match (&pooled_g_uncond, &pooled_g) {
            (Some(u), Some(c)) => {
                let mut p = Vec::with_capacity(u.len() + c.len());
                p.extend_from_slice(u);
                p.extend_from_slice(c);
                Some(p)
            }
            _ => None,
        };
        (emb, pooled, 2usize)
    } else {
        (text_emb.clone(), pooled_g.clone(), 1usize)
    };

    // Precompute cross-attention K, V from the (now possibly batched) text
    // embedding. The cache holds K, V for [n_batch × text_seq × inner].
    eprintln!("precomputing cross-attention K, V from text embedding ({} cross-attn blocks, n_batch={n_batch})...",
        count_cross_attn_blocks(&unet));
    unet.precompute_text_kv(&text_emb_run, text_seq_cond);
    #[cfg(feature = "metal")]
    if gpu_residence {
        // Also populate the GPU-resident cache. forward_gpu reads from this
        // (skipping K, V matmul on every step) — ~3500 matmuls saved per
        // inference for SDXL. Done in addition to the CPU cache so both
        // forward_gpu and forward_sdxl share the same precompute call.
        unet.precompute_text_kv_gpu(&text_emb_run, text_seq_cond);
        eprintln!("  ✓ GPU KV cache populated");
    }
    eprintln!("  ✓ KV cache populated");

    let single_latent_floats = latent_c * latent_h * latent_w;
    // Pre-allocate the UNet input buffer once, outside the loop. Per-step
    // we just refresh contents — saves a fresh 5-20MB allocation + zero-init
    // every step.
    let mut latent_input: Vec<f32> = Vec::with_capacity(
        if do_cfg { 2 } else { 1 } * single_latent_floats);

    for (step_idx, &t) in scheduler.timesteps().to_vec().iter().enumerate() {
        let step_start = std::time::Instant::now();
        let t_f = t as f32;

        // Replicate the latent across batch when CFG is on. Reuse buffer.
        latent_input.clear();
        latent_input.extend_from_slice(&latent);
        if do_cfg { latent_input.extend_from_slice(&latent); }

        // Default path (CPU + per-call Metal kernels): every kernel call
        // round-trips CPU↔GPU but the math is well-tested.
        // `--gpu-residence` switches to forward_sdxl_gpu / forward_gpu which
        // chains ops on the GPU end-to-end. Currently being debugged (see
        // the black-output regression on SDXL).
        let eps_batched = {
            #[cfg(feature = "metal")]
            {
                if gpu_residence {
                    match variant {
                        SDVariant::Sdxl => unet.forward_sdxl_gpu(
                            &latent_input, t_f, &text_emb_run,
                            pooled_run.as_ref().expect("SDXL requires pooled"),
                            SDXLAdditionalConditioning::default_1024(),
                        ),
                        _ => unet.forward_gpu(&latent_input, t_f, &text_emb_run),
                    }
                } else {
                    match variant {
                        SDVariant::Sdxl => unet.forward_sdxl(
                            &latent_input, t_f, &text_emb_run,
                            pooled_run.as_ref().expect("SDXL requires pooled"),
                            SDXLAdditionalConditioning::default_1024(),
                        ),
                        _ => unet.forward(&latent_input, t_f, &text_emb_run),
                    }
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = gpu_residence;
                match variant {
                    SDVariant::Sdxl => unet.forward_sdxl(
                        &latent_input, t_f, &text_emb_run,
                        pooled_run.as_ref().expect("SDXL requires pooled"),
                        SDXLAdditionalConditioning::default_1024(),
                    ),
                    _ => unet.forward(&latent_input, t_f, &text_emb_run),
                }
            }
        };
        let eps_batched = match eps_batched {
            Ok(e) => e,
            Err(e) => { eprintln!("UNet at step {step_idx}: {e}"); if repl { continue 'main; } break 'main ExitCode::FAILURE; }
        };

        let log_this_step = step_idx == 0
            || step_idx == scheduler.timesteps().len() / 2
            || step_idx + 1 == scheduler.timesteps().len();
        if log_this_step {
            eprintln!("step {step_idx}/{} (t={t}):", scheduler.timesteps().len());
            stats("eps_batched", &eps_batched);
            if do_cfg {
                let (eps_u, eps_c) = eps_batched.split_at(single_latent_floats);
                stats("eps_uncond", eps_u);
                stats("eps_cond  ", eps_c);
                let mut diff_sum = 0.0f64;
                for (u, c) in eps_u.iter().zip(eps_c.iter()) {
                    diff_sum += (c - u).abs() as f64;
                }
                let diff_mean = diff_sum / eps_u.len() as f64;
                eprintln!("  cross-attn signal:     mean|cond-uncond|={diff_mean:.6} (should be > 0.01 for prompt-adherent output)");

                // Reference comparison: at each checkpoint (steps 0, 12, 24)
                // diffusers' eps is available. Compare element-wise — divergence
                // at step N narrows the issue to between steps 0 and N (UNet,
                // scheduler, or accumulated state).
                if let Some(r) = &reference {
                    if let Some((u, c)) = r.eps_at_step.get(&step_idx) {
                        compare_tensors(
                            &format!("eps_uncond_step{step_idx}"), eps_u, u,
                        );
                        compare_tensors(
                            &format!("eps_cond_step{step_idx}  "), eps_c, c,
                        );
                    }
                }

                // Sanity check: at high t (early steps), the latent IS noise (since
                // α_bar_t ≈ 0 means x_t ≈ ε), so a correctly-trained model should
                // predict eps ≈ latent. We compare cosine similarity and mean|diff|.
                // If eps is uncorrelated with latent, the model isn't actually
                // denoising — it's outputting random-looking noise of right magnitude.
                let mut dot = 0.0f64; let mut nl2 = 0.0f64; let mut ne2 = 0.0f64;
                let mut diff_abs = 0.0f64;
                for (l, e) in latent.iter().zip(eps_u.iter()) {
                    dot += (*l as f64) * (*e as f64);
                    nl2 += (*l as f64).powi(2);
                    ne2 += (*e as f64).powi(2);
                    diff_abs += (l - e).abs() as f64;
                }
                let cos = if nl2 > 0.0 && ne2 > 0.0 { dot / (nl2.sqrt() * ne2.sqrt()) } else { 0.0 };
                let diff_mean = diff_abs / latent.len() as f64;
                eprintln!("  eps-vs-latent (uncond):  cos_sim={cos:.4}, mean|latent-eps|={diff_mean:.4}");
                eprintln!("    (at high-t, eps_perfect ≈ latent → cos≈1.0; cos≈0 means model isn't denoising)");
            }
        }

        // Combine: when do_cfg, eps_batched is [eps_uncond | eps_cond].
        let eps: Vec<f32> = if do_cfg {
            let (eps_u, eps_c) = eps_batched.split_at(single_latent_floats);
            eps_u.iter().zip(eps_c.iter())
                .map(|(u, c)| u + cfg_scale * (c - u))
                .collect()
        } else {
            eps_batched
        };

        if log_this_step && do_cfg {
            stats("eps_post_cfg", &eps);
        }

        latent = scheduler.step(&eps, step_idx, &latent);
        if log_this_step {
            stats("latent_after", &latent);
        }
        // Per-step latent magnitude — for tracking the denoising trajectory.
        // Expected: std ≈ sqrt(α_bar_t * 0.033 + (1-α_bar_t)), starting ≈1.0
        // and ending ≈0.18 for SD 1.5.
        let mut sa = 0.0f64; let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
        for &v in &latent { sa += v.abs() as f64; if v < mn { mn = v; } if v > mx { mx = v; } }
        let lm = sa / latent.len() as f64;
        eprintln!("  s{step_idx} t={t}: latent mean|.|={lm:.4}, range=[{mn:.2}, {mx:.2}]");
        if step_idx % 5 == 0 || step_idx + 1 == scheduler.timesteps().len() {
            let elapsed = step_start.elapsed();
            eprintln!("  step {step_idx}/{} (t={t}) [{:.1}s]", scheduler.timesteps().len(), elapsed.as_secs_f32());
        }
    }

    // Compare final latent (pre-VAE-scale) to diffusers reference.
    if let Some(r) = &reference {
        if !r.latent_final.is_empty() {
            compare_tensors("latent_final", &latent, &r.latent_final);
        }
    }

    // Decode latent → image.
    let scale = cfg.vae.scaling_factor;
    if scale != 0.0 {
        for v in latent.iter_mut() { *v /= scale; }
    }
    stats("latent_pre_vae", &latent);
    let img = {
        #[cfg(feature = "metal")]
        { if gpu_residence { vae.decode_with_dims_gpu(&latent, 1, latent_h, latent_w) }
          else { vae.decode_with_dims(&latent, 1, latent_h, latent_w) } }
        #[cfg(not(feature = "metal"))]
        { vae.decode_with_dims(&latent, 1, latent_h, latent_w) }
    };
    stats("vae_output    ", &img);
    if let Some(r) = &reference {
        if !r.vae_image.is_empty() {
            compare_tensors("vae_image", &img, &r.vae_image);
        }
    }
    let img_h = latent_h * 8;
    let img_w = latent_w * 8;
    eprintln!("decoded image: {img_h}×{img_w}");
    eprintln!("  (vae_output is the f32 image in [-1,1] convention; PNG conversion clips:");
    eprintln!("   black = vae output ≤ -1.0, white = vae output ≥ +1.0)");

    if let Err(e) = save_png(&img, img_h, img_w, &out_path) {
        eprintln!("save PNG: {e}");
        if repl { continue 'main; } break 'main ExitCode::FAILURE;
    }
    eprintln!("wrote {}", out_path.display());
    // End of per-prompt iteration. The 'main loop continues — in single-shot
    // mode it'll exit on next iteration; in --repl it reads next prompt.
    }; // close 'main: loop

    exit_code
}
