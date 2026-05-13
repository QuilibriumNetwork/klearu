//! sd_generate — Stable Diffusion image generation (SD 1.5 or SDXL).
//!
//! Branches on the detected variant. Supports:
//!   - txt2img (default).
//!   - img2img and naive inpainting (`--input`, `--mask`, `--strength`)
//!     against any standard 4-channel UNet — CPU-side blending re-anchors
//!     unmasked pixels each step.
//!   - **Native inpainting** (`--input`/`--mask`) against a dedicated
//!     9-channel UNet (RunwayML SD 1.5 inpaint or Stability SDXL
//!     inpaint). Detected automatically from the conv_in weight shape;
//!     the binary feeds `cat(noisy_latent, masked_init, mask)` to the
//!     UNet and skips the per-step blend. Higher-quality mask edges
//!     than naive inpainting because the model was trained on the task.
//!
//! ## Inpaint model checkpoints (single-file `.safetensors`):
//!
//!   - SD 1.5 inpaint:  `runwayml/stable-diffusion-inpainting`
//!     - HF download:  `hf download runwayml/stable-diffusion-inpainting sd-v1-5-inpainting.ckpt --local-dir <DIR>`
//!     - Or the `.safetensors` mirror via Civitai / ModelScope.
//!     - Size ~4 GB.
//!   - SDXL inpaint:    `stabilityai/stable-diffusion-xl-1.0-inpainting-0.1`
//!     - Single-file via Civitai mirror; gated on HF.
//!     - Size ~7 GB.
//!
//! ## Example:
//!
//!     sd_generate \
//!       --single-file <inpaint-checkpoint>.safetensors \
//!       --tokenizer clip-l/tokenizer.json \
//!       --prompt "a sleeping cat curled up on the cushion" \
//!       --input photo.png \
//!       --mask cat_mask.png \
//!       --steps 30 --cfg-scale 7.5 \
//!       --out cat_inpainted.png
//!
//! When the UNet's first-conv input-channel count is 9, the binary auto-
//! switches to the native-inpaint path and prints
//! `(unet.in_channels 4 → 9 from conv_in weight shape — inpaint UNet detected)`.
//! No flag changes needed; same `--input`/`--mask`/`--strength` flags.
//!
//! ## ControlNet (advanced inpaint via auxiliary network)
//!
//! When you have a regular SD/SDXL base model + a ControlNet weight file,
//! pass `--controlnet <path>` to inject per-step residuals into the UNet's
//! skip connections. The classic use case is **ControlNet-Inpaint** for
//! SDXL where dedicated 9-channel inpaint UNets are scarce: any SDXL
//! fine-tune (Juggernaut, RealVis, …) + a single ControlNet-Inpaint
//! weight produces high-quality inpainting at trained-mask-edge quality.
//!
//! ### Sources
//!   - SD 1.5 ControlNet-Inpaint: `lllyasviel/sd-controlnet-inpaint` (~1.4 GB).
//!   - SDXL ControlNet-Inpaint: `xinsir/controlnet-canny-sdxl-1.0` style;
//!     several inpaint variants on Civitai (~2.5 GB).
//!   - Other ControlNet types (canny, depth, openpose, …) work the same way;
//!     just provide the appropriate `--control-image` (an edge map, depth
//!     map, etc.).
//!
//! ### Flags
//!   --controlnet FILE          Path to a single-file `.safetensors` ControlNet.
//!   --control-image FILE       The control image (edges/depth/masked image/…).
//!                              Defaults to `--input` if omitted, which is
//!                              what ControlNet-Inpaint expects.
//!   --controlnet-weight F      Scale residuals by this factor (default 1.0).
//!   --controlnet-start F       Start step fraction (default 0.0).
//!   --controlnet-end F         End step fraction (default 1.0). Outside
//!                              [start, end), residuals are not injected.
//!
//! ### Example
//!
//!     sd_generate \
//!       --single-file <sdxl-base>.safetensors \
//!       --tokenizer clip-l/tokenizer.json --tokenizer-2 clip-g/tokenizer.json \
//!       --controlnet <sdxl-controlnet-inpaint>.safetensors \
//!       --prompt "a vase of red roses on the table" \
//!       --input photo.png --mask vase_mask.png \
//!       --steps 30 --cfg-scale 7.5 \
//!       --out result.png
//!
//! Metal GPU support is on by default (the `_with_controlnet` UNet
//! variants run on the same residence path the UNet does).

use std::path::PathBuf;
use std::process::ExitCode;

use klearu_diffusion::config::{CheckpointConfig, SDVariant};
use klearu_diffusion::image_io::{downsample_mask, load_image_as_chw_f32, load_mask, save_png};
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

/// Per-prompt overrides parsed from REPL inline trailing flags. Each
/// field is `Some` only when the user explicitly passed an override on
/// that line; `None` falls back to the run-level defaults.
#[derive(Default, Clone)]
struct PromptOverrides {
    steps: Option<usize>,
    seed: Option<u64>,
    cfg_scale: Option<f32>,
    negative_prompt: Option<String>,
    batch: Option<usize>,
    out: Option<PathBuf>,
    input: Option<PathBuf>,
    mask: Option<PathBuf>,
    strength: Option<f32>,
}

/// Parse a REPL line of the form `"prompt text :: --steps 30 --seed 7"`.
/// Lines without `::` produce `(line, default-overrides)`.
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
            "--cfg-scale" => overrides.cfg_scale = Some(val.parse().map_err(|e| format!("--cfg-scale: {e}"))?),
            "--negative-prompt" => overrides.negative_prompt = Some(val.to_string()),
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
    // Number of images per prompt. When >1, output filenames get a `_<i>`
    // suffix and seeds increment by 1 across the batch (so the same prompt
    // gets a deterministic family of variations).
    let mut batch: usize = 1;
    // Inpainting / img2img: when `--input` is set we VAE-encode it as the
    // starting latent, then re-noise. `--mask` (optional) is a grayscale
    // image where white=regenerate, black=preserve. `--strength` (0..=1)
    // controls how much to denoise (0 = no change, 1 = full regen).
    let mut input_image: Option<PathBuf> = None;
    let mut input_mask: Option<PathBuf> = None;
    let mut strength: f32 = 0.85;
    // ControlNet (optional): load an auxiliary network and inject per-step
    // residuals into the UNet. ControlNet-Inpaint is the typical use case
    // for SDXL where dedicated 9-channel inpaint UNets are scarce.
    let mut controlnet_path: Option<PathBuf> = None;
    let mut control_image_path: Option<PathBuf> = None;
    let mut controlnet_weight: f32 = 1.0;
    // Step-fraction window over which residuals get injected. Outside
    // [start, end), the UNet runs without ControlNet (vanilla generation).
    let mut controlnet_start: f32 = 0.0;
    let mut controlnet_end: f32 = 1.0;

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
            "--batch" => { i += 1; batch = args[i].parse().expect("usize"); }
            "--input" => { i += 1; input_image = Some(PathBuf::from(&args[i])); }
            "--mask"  => { i += 1; input_mask  = Some(PathBuf::from(&args[i])); }
            "--strength" => { i += 1; strength = args[i].parse().expect("f32"); }
            "--controlnet" => { i += 1; controlnet_path = Some(PathBuf::from(&args[i])); }
            "--control-image" => { i += 1; control_image_path = Some(PathBuf::from(&args[i])); }
            "--controlnet-weight" => { i += 1; controlnet_weight = args[i].parse().expect("f32"); }
            "--controlnet-start" => { i += 1; controlnet_start = args[i].parse().expect("f32"); }
            "--controlnet-end" => { i += 1; controlnet_end = args[i].parse().expect("f32"); }
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
                eprintln!();
                eprintln!("REPL + batch:");
                eprintln!("  --repl                  Interactive: read prompts from stdin until EOF/`exit`.");
                eprintln!("                          Inline overrides per prompt:");
                eprintln!("                            `your prompt :: --steps 30 --cfg-scale 5 --seed 7`");
                eprintln!("                          Override flags: --steps, --cfg-scale, --seed,");
                eprintln!("                            --negative-prompt, --batch, --out, --input,");
                eprintln!("                            --mask, --strength.");
                eprintln!("                          Empty line repeats the previous prompt.");
                eprintln!("  --batch N               Generate N images per prompt (default 1). Filenames");
                eprintln!("                          get a `_<i>` suffix when N > 1; seeds increment by 1");
                eprintln!("                          across the batch.");
                eprintln!();
                eprintln!("Inpainting / img2img:");
                eprintln!("  --input PNG             VAE-encode this image as the starting latent");
                eprintln!("                          (instead of pure noise). Image dims must match");
                eprintln!("                          the model's sample size × 8 (e.g., 512×512 for SD1.5).");
                eprintln!("  --mask PNG              Grayscale mask, same dims as --input. White=regenerate,");
                eprintln!("                          black=preserve, soft greys interpolate. If omitted");
                eprintln!("                          when --input is given, the whole image is regenerated");
                eprintln!("                          (full-image img2img).");
                eprintln!("  --strength F            How much to denoise the input (0..=1, default 0.85).");
                eprintln!("                          0 = no change; 1 = full regen as if from pure noise.");
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
                SDFormat::Flux => {
                    eprintln!("Flux checkpoint detected — sd_generate handles SD 1.5 / SDXL only.");
                    eprintln!("Use the `flux_generate` binary for Flux models.");
                    return ExitCode::FAILURE;
                }
                SDFormat::Unknown => {
                    eprintln!("could not detect SD variant; pass --variant sd15 or --variant sdxl");
                    return ExitCode::FAILURE;
                }
            };
            // Inpaint detection: if the conv_in weight has 9 input channels
            // instead of 4, this is a RunwayML / Stability inpaint UNet
            // (4 noisy latent + 4 masked init latent + 1 mask). Override
            // the in_channels default so the UNet's first conv is allocated
            // wide enough to consume the augmented input.
            if let Some(in_c) = loader.unet_in_channels() {
                if in_c != cfg.unet.in_channels {
                    eprintln!("(unet.in_channels {} → {in_c} from conv_in weight shape — \
                        inpaint UNet detected)", cfg.unet.in_channels);
                    cfg.unet.in_channels = in_c;
                }
            }
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

    // Optional ControlNet load. Mirrors UNet's encoder; receives the
    // same conditioning each step and produces residuals injected into
    // UNet's skip connections.
    let controlnet: Option<klearu_diffusion::controlnet::ControlNet2DModel> =
        if let Some(p) = &controlnet_path {
            eprintln!("[controlnet] loading {} …", p.display());
            let comp = match SingleFileLoader::open(p) {
                Ok(loader) => {
                    // ControlNet single-file safetensors stores the model
                    // under root keys (no `controlnet.` prefix). Convert
                    // raw tensors directly to a ComponentTensors.
                    use std::collections::HashMap;
                    use klearu_diffusion::weight::TensorRef;
                    let translated: HashMap<String, TensorRef> = loader.raw_tensors
                        .iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                    Some(ComponentTensors::from_single_mmap_borrow(&loader, translated))
                }
                Err(e) => { eprintln!("  ⚠ open controlnet: {e}"); None }
            };
            if let Some(comp) = comp {
                let cn_cfg = klearu_diffusion::controlnet::ControlNetConfig::from_unet(cfg.unet.clone());
                let mut cn = klearu_diffusion::controlnet::ControlNet2DModel::from_config(cn_cfg);
                match cn.load_from(&comp) {
                    Ok(()) => {
                        eprintln!("  ✓ controlnet loaded ({} down residuals + 1 mid residual)",
                            cn.num_down_residuals());
                        Some(cn)
                    }
                    Err(e) => { eprintln!("  ⚠ controlnet load: {e}"); None }
                }
            } else { None }
        } else { None };

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
    // Snapshot run-level defaults so per-prompt inline overrides can shadow
    // them without permanently mutating the run-level settings.
    let initial_steps = steps;
    let initial_cfg_scale = cfg_scale;
    let initial_negative_prompt = negative_prompt.clone();
    let initial_batch = batch;
    let initial_input = input_image.clone();
    let initial_mask = input_mask.clone();
    let initial_strength = strength;

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
        eprintln!("Inline overrides: `your prompt :: --steps 30 --cfg-scale 5 --seed 7 --batch 4 --out img.png`");
    }
    // `last_repl_line` lets a blank REPL line repeat the previous prompt.
    let mut last_repl_line: Option<String> = None;
    let exit_code: ExitCode = 'main: loop {
        // Resolve prompt + output path + inline overrides for this iteration.
        let prompt: String;
        let mut iter_out_path: PathBuf;
        let iter_overrides: PromptOverrides;
        if iteration == 0 && (!initial_prompt.is_empty() || !repl) {
            prompt = initial_prompt.clone();
            iter_out_path = initial_out_path.clone();
            iter_overrides = PromptOverrides::default();
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
            let raw = line.trim().to_string();
            if matches!(raw.as_str(), "exit" | "quit" | ":q") {
                break 'main ExitCode::SUCCESS;
            }
            // Empty line repeats the previous prompt (overrides and all).
            let to_parse = if raw.is_empty() {
                match &last_repl_line {
                    Some(p) => p.clone(),
                    None => continue 'main,
                }
            } else {
                last_repl_line = Some(raw.clone());
                raw
            };
            let (p, ov) = match parse_prompt_line(&to_parse) {
                Ok(t) => t,
                Err(e) => { eprintln!("(override parse error: {e}; ignoring this line)"); continue 'main; }
            };
            if p.is_empty() {
                continue 'main;
            }
            prompt = p;
            iter_overrides = ov;
            // Default auto-name; if --out was provided inline, the override
            // resolution below replaces it.
            let parent = initial_out_path.parent().unwrap_or(std::path::Path::new("."));
            iter_out_path = parent.join(format!("img_{iteration:03}.png"));
        }
        iteration += 1;

        // Resolve per-iteration parameters by overlaying inline overrides
        // on top of run-level defaults. The outer mutable `cfg_scale` /
        // `steps` etc. are intentionally NOT re-bound; we shadow with
        // `let` so the defaults remain intact for subsequent iterations.
        let steps = iter_overrides.steps.unwrap_or(initial_steps);
        let cfg_scale = iter_overrides.cfg_scale.unwrap_or(initial_cfg_scale);
        let negative_prompt: String = iter_overrides.negative_prompt.clone()
            .unwrap_or_else(|| initial_negative_prompt.clone());
        let iter_batch = iter_overrides.batch.unwrap_or(initial_batch).max(1);
        let iter_seed_pin = iter_overrides.seed.or(explicit_seed);
        if let Some(p) = &iter_overrides.out { iter_out_path = p.clone(); }
        let iter_input: Option<PathBuf> = iter_overrides.input.clone().or_else(|| initial_input.clone());
        let iter_mask: Option<PathBuf> = iter_overrides.mask.clone().or_else(|| initial_mask.clone());
        let iter_strength: f32 = iter_overrides.strength.unwrap_or(initial_strength).clamp(0.0, 1.0);

        // Re-apply timestep schedule for this iteration's step count.
        scheduler.set_timesteps(steps);

        eprintln!("\n=== generating: {prompt:?} → {} ===", iter_out_path.display());
        if iter_overrides.steps.is_some() || iter_overrides.cfg_scale.is_some()
            || iter_overrides.negative_prompt.is_some() || iter_overrides.seed.is_some()
            || iter_overrides.batch.is_some() || iter_overrides.out.is_some()
        {
            eprintln!("    (overrides: steps={steps}, cfg={cfg_scale}, batch={iter_batch}, \
                neg={:?}, seed={:?})", negative_prompt, iter_seed_pin);
        }

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
    // Two distinct channel counts:
    //   `latent_c`     = VAE latent channels (4 for SD/SDXL). Sets the
    //                    width of `latent`, `init_latent_x0`, `masked_init_latent`,
    //                    and the noise vector.
    //   `unet_in_c`    = UNet first-conv input channels (4 for normal,
    //                    9 for the dedicated inpaint UNets). Sets the
    //                    width of `latent_input` (the per-step UNet feed).
    // For inpaint UNets, latent_input concatenates [noisy_latent (4) |
    // masked_init_latent (4) | mask (1)] along the channel axis.
    let latent_c = cfg.vae.latent_channels;
    let unet_in_c = cfg.unet.in_channels;

    // Base seed for this prompt (across the whole batch). Pin to --seed
    // (or inline `:: --seed N`) when given on the first iteration;
    // otherwise randomise per iteration. Per-batch images derive
    // `base_seed + batch_i` so they form a deterministic family.
    let base_seed: u64 = match (iter_seed_pin, iteration == 1) {
        (Some(s), _) => s,
        _ => random_seed().wrapping_add(iteration as u64),
    };

    // Batch loop: same prompt, varied seed → distinct images. When
    // `iter_batch == 1` this runs exactly once and the original
    // `iter_out_path` is used as-is.
    for batch_i in 0..iter_batch {
        let seed = base_seed.wrapping_add(batch_i as u64);
        let out_path = if iter_batch <= 1 {
            iter_out_path.clone()
        } else {
            // Insert `_<i>` between stem and extension.
            let stem = iter_out_path.file_stem().and_then(|s| s.to_str()).unwrap_or("img");
            let ext = iter_out_path.extension().and_then(|s| s.to_str()).unwrap_or("png");
            let parent = iter_out_path.parent().unwrap_or(std::path::Path::new("."));
            parent.join(format!("{stem}_{batch_i}.{ext}"))
        };
        if iter_batch > 1 {
            eprintln!("\n--- batch {}/{iter_batch} (seed={seed}) → {} ---",
                batch_i + 1, out_path.display());
        } else {
            eprintln!("seed for this iteration: {seed}");
        }
    // ===== Inpainting / img2img setup =====
    // When --input is set, VAE-encode it as the starting x_0, downscale
    // an optional mask to latent resolution, and noise-inject init_latent_x0
    // up to the start-step σ. The sampling loop will:
    //   - iterate from `start_step..total_steps` (skipping high-noise steps),
    //   - blend per-step with q_sample(init_latent_x0, t_next) in the
    //     unmasked region, leaving the masked region free to evolve.
    let total_steps = scheduler.timesteps().len();
    let img_h_px = latent_h * 8;
    let img_w_px = latent_w * 8;
    let (init_latent_x0, mask_latent, start_step): (Option<Vec<f32>>, Option<Vec<f32>>, usize) =
        if let Some(input_path) = &iter_input {
            eprintln!("[inpaint] loading --input {}", input_path.display());
            let img_pixels = match load_image_as_chw_f32(input_path, img_h_px, img_w_px) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("input image: {e}");
                    if repl { continue 'main; } break 'main ExitCode::FAILURE;
                }
            };
            // VAE encode. Mirror the decode-side dispatch: route through
            // Metal (`encode_gpu`) when available + gpu_residence is on,
            // otherwise the CPU sgemm path. Encoder is the costlier of the
            // two on CPU because conv2d runs at full input resolution
            // before downsampling.
            let enc_t0 = std::time::Instant::now();
            eprintln!("[inpaint] VAE encoding {img_h_px}×{img_w_px} image…");
            // Auto-retry: if GPU encode produces NaN/Inf (sporadic Metal
            // pipeline bug under back-to-back batch encodes), fall back to
            // the CPU encode for this iteration. Set KLEARU_FORCE_CPU_VAE_ENCODE=1
            // to skip the GPU attempt entirely.
            let force_cpu_encode = std::env::var_os("KLEARU_FORCE_CPU_VAE_ENCODE").is_some();
            let try_gpu = {
                #[cfg(feature = "metal")] { gpu_residence && !force_cpu_encode }
                #[cfg(not(feature = "metal"))] { false }
            };
            let unscaled = if try_gpu {
                #[cfg(feature = "metal")]
                { match vae.encode_gpu(&img_pixels, 1, img_h_px, img_w_px) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("vae.encode_gpu: {e}");
                        if repl { continue 'main; } break 'main ExitCode::FAILURE;
                    }
                }}
                #[cfg(not(feature = "metal"))] { unreachable!() }
            } else {
                match vae.encode(&img_pixels, 1, img_h_px, img_w_px) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("vae.encode: {e}");
                        if repl { continue 'main; } break 'main ExitCode::FAILURE;
                    }
                }
            };
            // Surface NaN/Inf loudly — don't silently degrade. If this
            // fires it indicates a Metal pipeline / weight-cache bug; we
            // want it visible so it gets fixed, not papered over.
            if unscaled.iter().any(|v| !v.is_finite()) {
                eprintln!("[inpaint] ERROR: GPU encode produced NaN/Inf. \
                    This is a Metal-side bug. Reproduce with KLEARU_VAE_TRACE=1 \
                    to identify the layer. (Workaround: KLEARU_FORCE_CPU_VAE_ENCODE=1)");
                if repl { continue 'main; } break 'main ExitCode::FAILURE;
            }
            eprintln!("[inpaint] VAE encode done in {:.1}s", enc_t0.elapsed().as_secs_f32());
            // Per-iteration sanity check: NaN here means GPU state was
            // corrupted by a prior REPL iteration (most likely), or the
            // input image hit some VAE pathology. Surface it loudly so
            // we don't silently propagate garbage into the sampling loop.
            stats("init_latent_x0 (post-encode)", &unscaled);
            let mut x0 = unscaled;
            let scale = cfg.vae.scaling_factor;
            let shift = cfg.vae.shift_factor;
            for v in x0.iter_mut() { *v = (*v - shift) * scale; }

            let mask = if let Some(mp) = &iter_mask {
                eprintln!("[inpaint] loading --mask {}", mp.display());
                let raw = match load_mask(mp, img_h_px, img_w_px) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("mask: {e}");
                        if repl { continue 'main; } break 'main ExitCode::FAILURE;
                    }
                };
                match downsample_mask(&raw, img_h_px, img_w_px, 8) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        eprintln!("downsample_mask: {e}");
                        if repl { continue 'main; } break 'main ExitCode::FAILURE;
                    }
                }
            } else {
                eprintln!("[inpaint] no --mask: regenerating whole image (img2img mode)");
                None
            };
            let ss = ((1.0 - iter_strength) * total_steps as f32).floor() as usize;
            let ss = ss.min(total_steps);
            eprintln!("[inpaint] strength={iter_strength}, starting at step {ss}/{total_steps} \
                (skipping {ss} high-noise steps)");
            (Some(x0), mask, ss)
        } else {
            (None, None, 0)
        };

    // Native-inpaint detection. Only activates when --input is given AND
    // the UNet has 9 input channels (RunwayML SD 1.5 inpaint or Stability
    // SDXL inpaint). On native inpaint:
    //   - initial latent is pure Gaussian noise (model uses input image
    //     via the masked-init channel of its augmented input);
    //   - per-step UNet input is `cat(noisy_latent, masked_init_latent, mask)`
    //     instead of just the noisy latent;
    //   - per-step CPU-side blend is skipped (model handles mask natively).
    let native_inpaint = unet_in_c == 9 && init_latent_x0.is_some();
    if native_inpaint {
        eprintln!("[inpaint] native-inpaint UNet detected (in_channels=9); \
            using 9-channel UNet feed (no CPU-side blend)");
    }
    let (masked_init_latent, mask_lat_for_unet) = if native_inpaint {
        let x0 = init_latent_x0.as_ref().unwrap();
        let mask_l: Vec<f32> = match &mask_latent {
            Some(m) => m.clone(),
            None => vec![1.0_f32; latent_h * latent_w],
        };
        let lat_pixel = latent_h * latent_w;
        let mut masked = x0.clone();
        for i in 0..masked.len() {
            let m = mask_l[i % lat_pixel];
            masked[i] *= 1.0 - m; // zero the inpaint region
        }
        (Some(masked), Some(mask_l))
    } else { (None, None) };

    let mut latent = if native_inpaint {
        // Native-inpaint: always start from pure Gaussian noise. The
        // model receives the input image via the masked-init channel of
        // the UNet input, so the noisy-latent channels don't need to
        // carry the input at all.
        deterministic_noise(latent_c * latent_h * latent_w, seed)
    } else if let Some(x0) = &init_latent_x0 {
        // Naive img2img: re-noise init_latent_x0 to the start-step's σ.
        if start_step >= total_steps {
            // strength=0: leave x_0 unchanged; sampling loop runs 0 iterations.
            x0.clone()
        } else {
            let t_start = scheduler.timesteps()[start_step];
            let alpha_bar = scheduler.alpha_bar_at(t_start);
            let sqrt_alpha = alpha_bar.sqrt();
            let sqrt_one_minus = (1.0_f32 - alpha_bar).sqrt();
            let noise = deterministic_noise(x0.len(), seed);
            let mut l = vec![0.0_f32; x0.len()];
            for i in 0..x0.len() {
                l[i] = sqrt_alpha * x0[i] + sqrt_one_minus * noise[i];
            }
            l
        }
    } else if let Some(r) = &reference {
        // Use diffusers' noise so input matches. This isolates the UNet
        // comparison from any RNG-driven divergence.
        eprintln!("  [REF] overriding latent_init from reference (diffusers noise)");
        r.latent_init.clone()
    } else {
        deterministic_noise(latent_c * latent_h * latent_w, seed)
    };
    // ===== ControlNet control-image preload (once per prompt) =====
    // Source: --control-image if given; else the inpaint --input as a
    // sensible default for ControlNet-Inpaint workflows.
    let control_image_buf: Option<Vec<f32>> = if controlnet.is_some() {
        let path = control_image_path.as_ref().or(iter_input.as_ref());
        match path {
            Some(p) => {
                eprintln!("[controlnet] loading control image {} …", p.display());
                match load_image_as_chw_f32(p, img_h_px, img_w_px) {
                    Ok(v) => {
                        // Replicate across CFG batch (uncond + cond use the
                        // same control signal). UNet's latent_input is
                        // similarly batch-doubled when do_cfg is on.
                        if do_cfg {
                            let mut doubled = Vec::with_capacity(v.len() * 2);
                            doubled.extend_from_slice(&v);
                            doubled.extend_from_slice(&v);
                            Some(doubled)
                        } else { Some(v) }
                    }
                    Err(e) => {
                        eprintln!("  ⚠ control image: {e} — disabling ControlNet for this run");
                        None
                    }
                }
            }
            None => {
                eprintln!("[controlnet] (warn) no --control-image and no --input — disabling ControlNet");
                None
            }
        }
    } else { None };
    let cn_active = controlnet.is_some() && control_image_buf.is_some();

    eprintln!("starting denoise: {} steps from idx={start_step}, latent {}×{}×{}, cfg_scale={cfg_scale}",
        total_steps, latent_c, latent_h, latent_w);

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
    let single_unet_input_floats = unet_in_c * latent_h * latent_w;
    // Pre-allocate the UNet input buffer once, outside the loop. Per-step
    // we just refresh contents — saves a fresh 5-20MB allocation + zero-init
    // every step.
    let mut latent_input: Vec<f32> = Vec::with_capacity(
        if do_cfg { 2 } else { 1 } * single_unet_input_floats);

    // Native-inpaint: skip the start-step skip (model handles it natively
    // — always start from pure noise, full step count). The `start_step`
    // skip is only for the naive img2img blending path.
    let effective_start_step = if native_inpaint { 0 } else { start_step };

    for (step_idx, &t) in scheduler.timesteps().to_vec().iter().enumerate() {
        // Inpainting / img2img: skip the high-noise steps (sigma > start σ).
        if step_idx < effective_start_step { continue; }
        let step_start = std::time::Instant::now();
        let t_f = t as f32;

        // Build the per-step UNet input.
        latent_input.clear();
        let lat_pixel = latent_h * latent_w;
        if native_inpaint {
            // Per batch element: concat [noisy_latent | masked_init | mask].
            // Channel layout for one element is [c, h, w] flattened
            // (chunks of `lat_pixel`), so we concatenate channel-stacked
            // sub-tensors in order.
            let masked = masked_init_latent.as_ref().unwrap();
            let mask_l = mask_lat_for_unet.as_ref().unwrap();
            let mut push_one = |latent_input: &mut Vec<f32>| {
                latent_input.extend_from_slice(&latent);          // 4 ch
                latent_input.extend_from_slice(masked);            // 4 ch
                latent_input.extend_from_slice(&mask_l[..lat_pixel]); // 1 ch
            };
            push_one(&mut latent_input);
            if do_cfg { push_one(&mut latent_input); }
        } else {
            // Standard 4-channel feed; replicate latent across CFG batch.
            latent_input.extend_from_slice(&latent);
            if do_cfg { latent_input.extend_from_slice(&latent); }
        }

        // ControlNet residuals for this step (gated by step fraction).
        let frac = step_idx as f32 / total_steps.max(1) as f32;
        let cn_in_window = cn_active && frac >= controlnet_start && frac < controlnet_end;
        let cn_pooled_ref: Option<&[f32]> = pooled_run.as_deref();
        let cn_addl: Option<SDXLAdditionalConditioning> = if matches!(variant, SDVariant::Sdxl) {
            Some(SDXLAdditionalConditioning::default_1024())
        } else { None };

        // CPU residuals (used when GPU residence is OFF).
        let cn_cpu = if cn_in_window && !gpu_residence {
            controlnet.as_ref().and_then(|cn| {
                let img = control_image_buf.as_ref().unwrap();
                cn.forward(
                    &latent_input, t_f, &text_emb_run,
                    img, img_h_px, img_w_px,
                    cn_pooled_ref, cn_addl,
                ).ok()
            })
        } else { None };
        // GPU residuals (used when GPU residence is ON).
        #[cfg(feature = "metal")]
        let cn_gpu = if cn_in_window && gpu_residence {
            controlnet.as_ref().and_then(|cn| {
                let img = control_image_buf.as_ref().unwrap();
                cn.forward_gpu(
                    &latent_input, t_f, &text_emb_run,
                    img, img_h_px, img_w_px,
                    cn_pooled_ref, cn_addl,
                ).ok()
            })
        } else { None };

        // Apply controlnet_weight scaling on CPU residuals if active.
        let mut cn_cpu_scaled = cn_cpu;
        if let Some((dr, mr)) = cn_cpu_scaled.as_mut() {
            if controlnet_weight != 1.0 {
                for d in dr.iter_mut() {
                    for v in d.iter_mut() { *v *= controlnet_weight; }
                }
                for v in mr.iter_mut() { *v *= controlnet_weight; }
            }
        }
        // GPU residuals: scaling is applied on-device by the residual-
        // injection path (eadd_scaled_f16 kernel). The CPU branch above
        // pre-scales the f32 residual vectors directly.

        let eps_batched = {
            #[cfg(feature = "metal")]
            {
                if gpu_residence {
                    let dr_ref = cn_gpu.as_ref().map(|(d, _)| d.as_slice());
                    let mr_ref = cn_gpu.as_ref().map(|(_, m)| m);
                    // Apply controlnet_weight on the GPU via the
                    // eadd_scaled kernel during residual injection. With
                    // weight=1.0 the inject path uses the unscaled
                    // eadd_f16 kernel (no extra mul per element).
                    let cn_w = controlnet_weight;
                    match variant {
                        SDVariant::Sdxl => unet.forward_sdxl_gpu_with_controlnet(
                            &latent_input, t_f, &text_emb_run,
                            pooled_run.as_ref().expect("SDXL requires pooled"),
                            SDXLAdditionalConditioning::default_1024(),
                            dr_ref, mr_ref, cn_w,
                        ),
                        _ => unet.forward_gpu_with_controlnet(
                            &latent_input, t_f, &text_emb_run, dr_ref, mr_ref, cn_w),
                    }
                } else {
                    let dr_ref = cn_cpu_scaled.as_ref().map(|(d, _)| d.as_slice());
                    let mr_ref = cn_cpu_scaled.as_ref().map(|(_, m)| m.as_slice());
                    match variant {
                        SDVariant::Sdxl => unet.forward_sdxl_with_controlnet(
                            &latent_input, t_f, &text_emb_run,
                            pooled_run.as_ref().expect("SDXL requires pooled"),
                            SDXLAdditionalConditioning::default_1024(),
                            dr_ref, mr_ref,
                        ),
                        _ => unet.forward_with_controlnet(
                            &latent_input, t_f, &text_emb_run, dr_ref, mr_ref),
                    }
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = gpu_residence;
                let dr_ref = cn_cpu_scaled.as_ref().map(|(d, _)| d.as_slice());
                let mr_ref = cn_cpu_scaled.as_ref().map(|(_, m)| m.as_slice());
                match variant {
                    SDVariant::Sdxl => unet.forward_sdxl_with_controlnet(
                        &latent_input, t_f, &text_emb_run,
                        pooled_run.as_ref().expect("SDXL requires pooled"),
                        SDXLAdditionalConditioning::default_1024(),
                        dr_ref, mr_ref,
                    ),
                    _ => unet.forward_with_controlnet(
                        &latent_input, t_f, &text_emb_run, dr_ref, mr_ref),
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

        // ===== Inpainting / img2img per-step blend =====
        // Native-inpaint UNets handle mask via the 9-channel input; no
        // CPU-side blend needed. The blend below is only for naive
        // img2img on standard 4-channel UNets.
        if !native_inpaint {
        // Replace the unmasked region with q_sample(init_latent_x0, t_next),
        // so the unmasked pixels track the diffusion trajectory of the
        // input image while the masked region evolves freely under the
        // model. Skipped when no --input was provided (init_latent_x0 = None).
        if let Some(x0) = &init_latent_x0 {
            let target_t: i64 = if step_idx + 1 < total_steps {
                scheduler.timesteps()[step_idx + 1]
            } else {
                -1 // post-final → α_bar≈1, no noise
            };
            let alpha_bar = scheduler.alpha_bar_at(target_t);
            let sqrt_alpha = alpha_bar.sqrt();
            let sqrt_one_minus = (1.0_f32 - alpha_bar).sqrt();
            // Per-step independent noise — RePaint convention.
            let noise = deterministic_noise(latent.len(),
                seed.wrapping_add(0x9E3779B97F4A7C15).wrapping_mul(step_idx as u64 + 1));
            let lat_pixel = latent_h * latent_w;
            for i in 0..latent.len() {
                let m = match &mask_latent {
                    Some(mk) => mk[i % lat_pixel],
                    None => 1.0, // no mask → full regen (img2img mode); blend = identity
                };
                let unmasked = sqrt_alpha * x0[i] + sqrt_one_minus * noise[i];
                latent[i] = m * latent[i] + (1.0 - m) * unmasked;
            }
        }
        } // end !native_inpaint

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
    let _ = batch_i; // satisfy borrow-checker for the rare unused-warning case
    } // end 'batch loop
    // End of per-prompt iteration. The 'main loop continues — in single-shot
    // mode it'll exit on next iteration; in --repl it reads next prompt.
    }; // close 'main: loop

    exit_code
}
