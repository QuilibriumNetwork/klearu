//! Self-contained Krea 2 runner for the desktop app: load once, then turn
//! prompts into images on the Metal GPU, in memory (inputs as bytes, output as
//! planar f32), so it works incognito like `FluxRunner` and `HiDreamRunner`.
//!
//! Generation follows the reference sampler (`krea-2/sampling.py`) exactly:
//! Euler on the flow ODE from σ = 1 to 0 over the shifted schedule, with
//! Krea's guidance convention `v = cond + g·(cond − uncond)`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use super::config::{Krea2Config, LATENT_CHANNELS, PIXELS_PER_TOKEN};
use super::gpu::GpuKrea2Transformer;
use super::model::{pack_latents, unpack_latents};
use super::names;
use super::text::Krea2TextEncoder;
use crate::image_io::{downsample_mask, load_image_bytes_as_chw_f32, load_mask_bytes};
use crate::scheduler::{FlowMatchConfig, FlowMatchScheduler, Scheduler};
use crate::vae3d::{GpuImageVae, WanVae, WanVaeConfig};
use crate::weight::loader::ComponentTensors;
use crate::{DiffusionError, Result};

/// The two released checkpoints. They share every component but the
/// transformer, and sample differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Krea2Variant {
    /// Distilled: 8 steps, no CFG, timestep shift fixed at μ = 1.15.
    Turbo,
    /// Base: resolution-dependent shift, CFG.
    Raw,
}

impl Krea2Variant {
    pub fn default_steps(self) -> usize {
        match self {
            Krea2Variant::Turbo => 8,
            Krea2Variant::Raw => 28,
        }
    }

    /// Default guidance in Krea's convention (`cond + g·(cond − uncond)`).
    pub fn default_guidance(self) -> f32 {
        match self {
            Krea2Variant::Turbo => 0.0,
            Krea2Variant::Raw => 4.5,
        }
    }

    fn schedule(self) -> FlowMatchConfig {
        match self {
            Krea2Variant::Turbo => FlowMatchConfig::krea2_turbo(),
            Krea2Variant::Raw => FlowMatchConfig::krea2_raw(),
        }
    }
}

/// Where to find the model. `checkpoint` is a Krea 2 directory in diffusers
/// layout (`model_index.json`, `transformer/`, `text_encoder/`, `tokenizer/`,
/// `vae/`) or a single-file transformer (`turbo.safetensors`/`raw.safetensors`).
/// Components the checkpoint lacks — all of them for a single file, and the
/// shared ones for a Raw download that skipped them — come from the overrides,
/// or failing that from any other Krea 2 snapshot in the Hugging Face cache.
#[derive(Debug, Clone, Default)]
pub struct Krea2LoadSpec {
    pub checkpoint: PathBuf,
    /// A `text_encoder/` directory (Qwen3-VL-4B).
    pub text_encoder: Option<PathBuf>,
    /// A `tokenizer.json`.
    pub tokenizer: Option<PathBuf>,
    /// A `vae/` directory (Qwen-Image VAE).
    pub vae: Option<PathBuf>,
    /// `None` = detect (`model_index.json`'s `is_distilled`, else the file name).
    pub variant: Option<Krea2Variant>,
    /// Load the text encoder for each generation and free it before
    /// denoising: ~7 GB less at peak, a few seconds more per image. Right for
    /// one-shot use; a resident app generating repeatedly should keep it.
    pub offload_text_encoder: bool,
}

/// Knobs for one generation.
#[derive(Debug, Clone)]
pub struct Krea2GenOpts {
    pub prompt: String,
    /// Used only with CFG. Empty is the reference's default.
    pub negative_prompt: String,
    /// Rounded down to multiples of 16. Turbo is trained for 1024–2048, Raw
    /// for up to ~1280.
    pub width: usize,
    pub height: usize,
    /// `0` = the variant's default.
    pub steps: usize,
    /// Krea's convention: `v = cond + g·(cond − uncond)`, so `g` here equals
    /// standard CFG at `g + 1`; `0` disables CFG. `None` = variant default.
    pub guidance: Option<f32>,
    pub seed: u64,
    /// PNG/JPEG bytes for img2img; with `mask_bytes` (white = regenerate),
    /// inpainting.
    pub input_image_bytes: Option<Vec<u8>>,
    pub mask_bytes: Option<Vec<u8>>,
    /// img2img denoise strength `0..=1`.
    pub strength: f32,
}

impl Default for Krea2GenOpts {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 0,
            guidance: None,
            seed: 0,
            input_image_bytes: None,
            mask_bytes: None,
            strength: 0.75,
        }
    }
}

/// A rendered image, planar `[3, H, W]` f32 in `[-1, 1]`.
pub struct Krea2Image {
    pub chw: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

/// True for a Krea 2 directory or a single-file Krea 2 transformer.
pub fn is_krea2_checkpoint(path: &Path) -> bool {
    if path.is_dir() {
        let class_is = |file: PathBuf, want: &str| -> bool {
            std::fs::read_to_string(file)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .and_then(|v| v.get("_class_name").and_then(|c| c.as_str()).map(|c| c == want))
                .unwrap_or(false)
        };
        return class_is(path.join("model_index.json"), "Krea2Pipeline")
            || class_is(path.join("transformer/config.json"), "Krea2Transformer2DModel");
    }
    path.extension().and_then(|e| e.to_str()) == Some("safetensors")
        && ComponentTensors::open_file(path)
            .map(|c| names::is_krea2_transformer(&c))
            .unwrap_or(false)
}

/// A loaded Krea 2: transformer and VAE GPU-resident, the text encoder
/// resident or loaded on demand (see [`Krea2LoadSpec::offload_text_encoder`]).
pub struct Krea2Runner {
    pub variant: Krea2Variant,
    transformer: GpuKrea2Transformer,
    text: TextEncoder,
    vae: GpuImageVae,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

enum TextEncoder {
    Resident(Box<Krea2TextEncoder>),
    OnDemand { dir: PathBuf, tokenizer: PathBuf },
}

impl Krea2Runner {
    /// Load every component. The transformer streams to the GPU one tensor at
    /// a time (~25 GB of bf16); the text encoder adds ~7 GB.
    pub fn load(spec: &Krea2LoadSpec) -> Result<Self> {
        Self::load_with_progress(spec, |_| {})
    }

    /// As [`Self::load`], reporting coarse progress messages.
    pub fn load_with_progress(spec: &Krea2LoadSpec, mut on_progress: impl FnMut(&str)) -> Result<Self> {
        let layout = |m: String| DiffusionError::Layout(m);
        let ck = &spec.checkpoint;
        let dir = ck.is_dir().then(|| ck.clone());

        // ---- variant ----
        let variant = spec.variant.unwrap_or_else(|| detect_variant(ck));

        // ---- transformer ----
        on_progress("loading Krea 2 transformer");
        let (cfg, mut comp) = match &dir {
            Some(d) => {
                let cfg_text = std::fs::read_to_string(d.join("transformer/config.json"))
                    .map_err(|e| layout(format!("{}: transformer/config.json: {e}", d.display())))?;
                (Krea2Config::from_json(&cfg_text)?, ComponentTensors::open_dir(&d.join("transformer"))?)
            }
            None => (Krea2Config::default(), ComponentTensors::open_file(ck)?),
        };
        names::canonicalize(&mut comp);
        let transformer = GpuKrea2Transformer::load(cfg, &comp, |i, n| {
            tracing::debug!("krea2 transformer block {i}/{n} uploaded");
        })?;
        drop(comp);

        // ---- text encoder + tokenizer ----
        on_progress("loading Qwen3-VL text encoder");
        let te_dir = resolve_component(spec.text_encoder.as_ref(), dir.as_deref(), "text_encoder", has_safetensors)
            .ok_or_else(|| layout("no Krea 2 text_encoder/ found; set the text encoder path".into()))?;
        let tok = resolve_component(spec.tokenizer.as_ref(), dir.as_deref(), "tokenizer", |p| {
            p.join("tokenizer.json").is_file()
        })
        .map(|p| if p.is_dir() { p.join("tokenizer.json") } else { p })
        .ok_or_else(|| layout("no Krea 2 tokenizer/tokenizer.json found; set the tokenizer path".into()))?;
        let text = if spec.offload_text_encoder {
            TextEncoder::OnDemand { dir: te_dir, tokenizer: tok }
        } else {
            TextEncoder::Resident(Box::new(Krea2TextEncoder::load(&te_dir, &tok)?))
        };

        // ---- VAE ----
        on_progress("loading Qwen-Image VAE");
        let vae_dir = resolve_component(spec.vae.as_ref(), dir.as_deref(), "vae", has_safetensors)
            .ok_or_else(|| layout("no Krea 2 vae/ found; set the VAE path".into()))?;
        let vae_cfg_text = std::fs::read_to_string(vae_dir.join("config.json"))
            .map_err(|e| layout(format!("{}: config.json: {e}", vae_dir.display())))?;
        let vae_cfg = WanVaeConfig::from_json(&vae_cfg_text)?;
        if vae_cfg.latents_mean.len() != LATENT_CHANNELS || vae_cfg.latents_std.len() != LATENT_CHANNELS {
            return Err(layout(format!(
                "{} is not the 16-channel Qwen-Image VAE (latents_mean/std missing)",
                vae_dir.display()
            )));
        }
        let (latents_mean, latents_std) = (vae_cfg.latents_mean.clone(), vae_cfg.latents_std.clone());
        let vae_cpu = WanVae::load(vae_cfg, &ComponentTensors::open_dir(&vae_dir)?)?;
        let vae = GpuImageVae::from_vae(&vae_cpu);
        drop(vae_cpu);

        tracing::info!(?variant, "Krea 2 loaded");
        Ok(Self { variant, transformer, text, vae, latents_mean, latents_std })
    }

    /// Render one image. `on_step(step, total)` reports denoising progress;
    /// setting `abort` stops early and decodes what there is.
    pub fn generate(
        &self,
        opts: &Krea2GenOpts,
        mut on_step: impl FnMut(usize, usize),
        abort: &AtomicBool,
    ) -> Result<Krea2Image> {
        let unsup = |e: String| DiffusionError::Unsupported(e);
        let w = (opts.width / PIXELS_PER_TOKEN).max(1) * PIXELS_PER_TOKEN;
        let h = (opts.height / PIXELS_PER_TOKEN).max(1) * PIXELS_PER_TOKEN;
        let (lat_h, lat_w) = (h / 8, w / 8);
        let p = self.transformer.cfg.patch();
        let grid = (lat_h / p, lat_w / p);
        let c = LATENT_CHANNELS;

        let steps = if opts.steps == 0 { self.variant.default_steps() } else { opts.steps };
        let guidance = opts.guidance.unwrap_or_else(|| self.variant.default_guidance()).max(0.0);
        let use_cfg = guidance > 0.0;

        // ---- prompt -> text tokens (timestep-independent; once per branch) ----
        let t0 = std::time::Instant::now();
        let (cond, n, uncond) = self.text_tokens(opts, use_cfg)?;
        let d = self.transformer.cfg.hidden();
        // Both branches padded to one length so the second reuses the
        // first's buffers (padding is masked out exactly; see `gpu`).
        let n_max = (cond.len() / d).max(uncond.as_ref().map_or(0, |u| u.len() / d));
        let ctx_c = self.transformer.prepare(&cond, grid, n_max);
        let ctx_u = uncond.as_ref().map(|u| self.transformer.prepare(u, grid, n_max));
        tracing::info!(
            tokens = n, cfg = use_cfg, secs = t0.elapsed().as_secs_f32(),
            "krea2 prompt encoded"
        );

        // ---- schedule ----
        let mut sched = FlowMatchScheduler::new(self.variant.schedule());
        sched.set_image_seq_len(grid.0 * grid.1);
        sched.set_timesteps(steps);
        let sigmas = sched.sigmas().to_vec();

        // ---- img2img / inpaint ----
        let strength = opts.strength.clamp(0.0, 1.0);
        let (x0, mask): (Option<Vec<f32>>, Option<Vec<f32>>) = match &opts.input_image_bytes {
            Some(bytes) => {
                let rgb = load_image_bytes_as_chw_f32(bytes, h, w).map_err(|e| unsup(format!("input image: {e}")))?;
                let barrier = crate::metal_backend::set_mps_barrier(true);
                let z = self.vae.encode(&rgb, h, w);
                crate::metal_backend::set_mps_barrier(barrier);
                let z = self.normalize(z, lat_h * lat_w);
                let mask = match &opts.mask_bytes {
                    Some(mb) => {
                        let raw = load_mask_bytes(mb, h, w).map_err(|e| unsup(format!("mask: {e}")))?;
                        let m = downsample_mask(&raw, h, w, 8).map_err(|e| unsup(format!("mask: {e}")))?;
                        // Same mask for every latent channel, packed like the latent.
                        let full: Vec<f32> = (0..c).flat_map(|_| m.iter().copied()).collect();
                        Some(pack_latents(&full, c, lat_h, lat_w, p))
                    }
                    None => None,
                };
                (Some(pack_latents(&z, c, lat_h, lat_w, p)), mask)
            }
            None => (None, None),
        };
        let start = if x0.is_some() { ((1.0 - strength) * steps as f32).floor() as usize } else { 0 };

        // ---- initial latent ----
        let noise = |seed: u64| pack_latents(&crate::flux_runner::sample_gaussian(&[c, lat_h, lat_w], seed), c, lat_h, lat_w, p);
        let mut x = noise(opts.seed);
        if let Some(x0) = &x0 {
            let s = sigmas[start];
            x.iter_mut().zip(x0).for_each(|(v, &a)| *v = (1.0 - s) * a + s * *v);
        }

        // ---- Euler on the flow ODE ----
        let t_loop = std::time::Instant::now();
        for i in start..steps {
            let (s, s_next) = (sigmas[i], sigmas[i + 1]);
            let mut v = self.transformer.forward(&x, s, &ctx_c);
            finite_at("velocity", i, &v)?;
            if let Some(cu) = &ctx_u {
                let vu = self.transformer.forward(&x, s, cu);
                finite_at("unconditional velocity", i, &vu)?;
                v.iter_mut().zip(&vu).for_each(|(a, b)| *a += guidance * (*a - b));
            }
            x.iter_mut().zip(&v).for_each(|(a, b)| *a += (s_next - s) * b);

            if let (Some(x0), Some(m)) = (&x0, &mask) {
                // Keep the unmasked region on the input's own noisy trajectory.
                let nz = noise(opts.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64 + 1));
                for j in 0..x.len() {
                    let keep = (1.0 - s_next) * x0[j] + s_next * nz[j];
                    x[j] = m[j] * x[j] + (1.0 - m[j]) * keep;
                }
            }
            on_step(i + 1, steps);
            if abort.load(Ordering::Relaxed) {
                break;
            }
        }
        tracing::info!(
            steps = steps - start, secs = t_loop.elapsed().as_secs_f32(), w, h,
            "krea2 denoise done"
        );

        // ---- decode ----
        // Diagnostic: `KLEARU_KREA2_DUMP_LATENT=<file>` saves the final packed
        // latent (raw little-endian f32, `[tokens, 64]`) to decode elsewhere.
        if let Some(path) = std::env::var_os("KLEARU_KREA2_DUMP_LATENT") {
            let raw: Vec<u8> = x.iter().flat_map(|a| a.to_le_bytes()).collect();
            if let Err(e) = std::fs::write(&path, raw) {
                tracing::warn!("krea2: could not dump the latent: {e}");
            }
        }
        let chw = self.decode_packed(&x, lat_h, lat_w);
        // The decode's clamp passes NaN through and PNG encoding turns it
        // black, so a bad decode would otherwise arrive as a plausible image.
        finite("decoded image", &chw)?;
        Ok(Krea2Image { chw, height: h, width: w })
    }

    /// A packed `[tokens, 64]` latent, as the denoise loop leaves it, to RGB
    /// `[3, lat_h * 8, lat_w * 8]` in `[-1, 1]`.
    pub(crate) fn decode_packed(&self, x: &[f32], lat_h: usize, lat_w: usize) -> Vec<f32> {
        let p = self.transformer.cfg.patch();
        let z = unpack_latents(x, LATENT_CHANNELS, lat_h, lat_w, p);
        let z = self.denormalize(z, lat_h * lat_w);
        // The VAE's matmuls run on the same MPS path, fenced as the
        // transformer's are (see `metal_backend::set_mps_barrier`).
        let barrier = crate::metal_backend::set_mps_barrier(true);
        let chw = self.vae.decode(&z, lat_h, lat_w);
        crate::metal_backend::set_mps_barrier(barrier);
        chw
    }

    /// Prompt (and, with CFG, negative prompt) to fused text tokens, loading
    /// the text encoder first if it is not resident. Returns
    /// `(cond, n_cond, uncond)`.
    pub(crate) fn text_tokens(
        &self,
        opts: &Krea2GenOpts,
        use_cfg: bool,
    ) -> Result<(Vec<f32>, usize, Option<Vec<f32>>)> {
        match &self.text {
            TextEncoder::Resident(te) => self.encode_prompts(te, opts, use_cfg),
            TextEncoder::OnDemand { dir, tokenizer } => {
                let te = Krea2TextEncoder::load(dir, tokenizer)?;
                self.encode_prompts(&te, opts, use_cfg)
                // `te` drops here, before the transformer needs the room.
            }
        }
    }

    fn encode_prompts(
        &self,
        te: &Krea2TextEncoder,
        opts: &Krea2GenOpts,
        use_cfg: bool,
    ) -> Result<(Vec<f32>, usize, Option<Vec<f32>>)> {
        let (taps, n) = te.encode(&opts.prompt)?;
        finite("prompt encoder taps", &taps)?;
        let cond = self.transformer.encode_text(&taps, n);
        finite("prompt text tokens", &cond)?;
        let uncond = if use_cfg {
            let (taps_u, n_u) = te.encode(&opts.negative_prompt)?;
            finite("negative-prompt encoder taps", &taps_u)?;
            let u = self.transformer.encode_text(&taps_u, n_u);
            finite("negative-prompt text tokens", &u)?;
            Some(u)
        } else {
            None
        };
        Ok((cond, n, uncond))
    }

    /// VAE latent -> transformer space: `(z - mean) / std` per channel.
    fn normalize(&self, mut z: Vec<f32>, plane: usize) -> Vec<f32> {
        for (ch, chunk) in z.chunks_mut(plane).enumerate() {
            let (m, s) = (self.latents_mean[ch], self.latents_std[ch]);
            chunk.iter_mut().for_each(|v| *v = (*v - m) / s);
        }
        z
    }

    /// Transformer space -> VAE latent: `z * std + mean`.
    fn denormalize(&self, mut z: Vec<f32>, plane: usize) -> Vec<f32> {
        for (ch, chunk) in z.chunks_mut(plane).enumerate() {
            let (m, s) = (self.latents_mean[ch], self.latents_std[ch]);
            chunk.iter_mut().for_each(|v| *v = *v * s + m);
        }
        z
    }
}

/// Refuse to carry non-finite values forward. A NaN anywhere in the loop ends
/// as an all-black image with no hint of where it came from; this turns it
/// into an error that names the stage. The cause seen here was MPS matmuls
/// garbling rows when an operand crossed a 4 GiB boundary of GPU address
/// space (now prevented by `metal_backend::acquire_mps_buffer`); another GPU
/// job competing for the device is the other suspect.
fn finite(what: &str, v: &[f32]) -> Result<()> {
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    if bad > 0 {
        return Err(DiffusionError::Unsupported(format!(
            "krea2: {what} has {bad}/{} non-finite values (is another GPU job running?)",
            v.len()
        )));
    }
    Ok(())
}

fn finite_at(what: &str, step: usize, v: &[f32]) -> Result<()> {
    finite(&format!("{what} at step {}", step + 1), v)
}

/// Turbo or Raw: `model_index.json`'s `is_distilled` when there is one,
/// otherwise the file name, otherwise Raw (the reference's default).
pub fn detect_variant(checkpoint: &Path) -> Krea2Variant {
    let index = if checkpoint.is_dir() {
        Some(checkpoint.join("model_index.json"))
    } else {
        checkpoint.parent().map(|p| p.join("model_index.json"))
    };
    if let Some(distilled) = index
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get("is_distilled").and_then(|d| d.as_bool()))
    {
        return if distilled { Krea2Variant::Turbo } else { Krea2Variant::Raw };
    }
    let name = checkpoint.to_string_lossy().to_lowercase();
    if name.contains("turbo") { Krea2Variant::Turbo } else { Krea2Variant::Raw }
}

fn has_safetensors(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| rd.flatten().any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors")))
        .unwrap_or(false)
}

/// Pick a component directory: the explicit override, then the checkpoint's
/// own subfolder, then the same subfolder of any Krea 2 snapshot in the
/// Hugging Face cache (the shared components are byte-identical between
/// Turbo and Raw, so a Raw download can borrow Turbo's).
fn resolve_component(
    explicit: Option<&PathBuf>,
    dir: Option<&Path>,
    sub: &str,
    usable: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.clone());
    }
    if let Some(d) = dir {
        let own = d.join(sub);
        if usable(&own) {
            return Some(own);
        }
    }
    hf_cache_dirs()
        .into_iter()
        .flat_map(|hub| krea2_snapshots(&hub))
        .map(|snap| snap.join(sub))
        .find(|p| usable(p))
}

fn hf_cache_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        v.push(PathBuf::from(p));
    }
    if let Some(p) = std::env::var_os("HF_HOME") {
        v.push(PathBuf::from(p).join("hub"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        v.push(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    v
}

fn krea2_snapshots(hub: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(hub) else { return Vec::new() };
    let mut out = Vec::new();
    for repo in rd.flatten() {
        let name = repo.file_name().to_string_lossy().to_lowercase();
        if !name.starts_with("models--krea--krea-2") {
            continue;
        }
        if let Ok(snaps) = std::fs::read_dir(repo.path().join("snapshots")) {
            out.extend(snaps.flatten().map(|s| s.path()));
        }
    }
    out
}
