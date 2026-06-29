//! Self-contained Flux runner: load a Flux checkpoint (single-file bundle or
//! directory) + the CLIP-L/T5 tokenizers once, then turn prompts into images
//! on the Metal GPU. The orchestration mirrors the `flux_generate` binary but
//! returns the image in-memory (no file I/O) and takes image inputs as byte
//! buffers — so it works incognito and embeds in the desktop app, the same way
//! `HiDreamRunner` does.
//!
//! Gated on `feature = "metal"` (the GPU transformer is the whole point).

#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::{CLIPTextConfig, VAEConfig};
use crate::image_io::{downsample_mask, load_image_bytes_as_chw_f32, load_mask_bytes};
use crate::scheduler::{FlowMatchConfig, FlowMatchScheduler, Scheduler};
use crate::text_encoder::t5::{T5Config, T5Encoder};
use crate::text_encoder::CLIPTextModel;
use crate::tokenizer::{CLIPTokenizer, T5Tokenizer};
use crate::unet::flux::FluxConfig;
use crate::unet::flux_gpu::GpuFluxTransformer;
use crate::vae::AutoencoderKL;
use crate::weight::flux_checkpoint::{
    component_with_prefix, detect_variant_single_file, discover_flux_paths,
    transformer_component_from_loader, vae_component_from_loader, FluxCheckpoint, FluxPaths,
    FluxVariant,
};
use crate::weight::{SDFormat, SingleFileLoader};
use crate::{DiffusionError, Result};

/// Where to source the Flux model + tokenizers. `checkpoint` is either an
/// all-in-one `.safetensors` bundle or a directory of the four BFL files.
pub struct FluxLoadSpec {
    pub checkpoint: PathBuf,
    pub clip_tokenizer: PathBuf,
    pub t5_tokenizer: PathBuf,
    /// Optional per-component overrides (used when the bundle lacks a part).
    pub vae: Option<PathBuf>,
    pub clip_l: Option<PathBuf>,
    pub t5: Option<PathBuf>,
    /// `None` = auto-detect dev vs schnell.
    pub variant: Option<FluxVariant>,
}

/// Knobs for one generation (mirrors `HiDreamGenOpts`).
pub struct FluxGenOpts {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    /// `0` = the variant default (dev 50, schnell 4).
    pub steps: usize,
    pub guidance: f32,
    pub seed: u64,
    pub input_image_bytes: Option<Vec<u8>>,
    pub mask_bytes: Option<Vec<u8>>,
    pub strength: f32,
}

/// A rendered image in planar `[3, H, W]` f32, normalized to `[-1, 1]`.
pub struct FluxImage {
    pub chw: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

/// A loaded Flux model — the GPU transformer plus VAE / CLIP-L / T5 / tokenizers.
pub struct FluxRunner {
    transformer: GpuFluxTransformer,
    cfg: FluxConfig,
    variant: FluxVariant,
    vae: AutoencoderKL,
    clip_l: CLIPTextModel,
    t5: T5Encoder,
    clip_tok: CLIPTokenizer,
    t5_tok: T5Tokenizer,
}

/// True if `path` looks like a Flux checkpoint — a single-file bundle whose
/// loader reports `SDFormat::Flux`, or a directory `discover_flux_paths` accepts.
pub fn is_flux_checkpoint(path: &Path) -> bool {
    if path.is_dir() {
        return discover_flux_paths(path).is_some();
    }
    SingleFileLoader::open(path)
        .map(|l| l.variant == SDFormat::Flux)
        .unwrap_or(false)
}

impl FluxRunner {
    /// Load everything once. The 12B transformer streams straight to GPU bf16
    /// (per-block, low peak RAM); T5-XXL (4.7B) is the slow load.
    pub fn load(spec: &FluxLoadSpec) -> Result<Self> {
        let unsup = |e: String| DiffusionError::Unsupported(e);
        // ---- open the four components (single-file vs directory) ----
        let (variant_detected, comp_t, comp_v, comp_c, comp_t5) = if spec.checkpoint.is_dir() {
            let paths: FluxPaths = discover_flux_paths(&spec.checkpoint)
                .ok_or_else(|| unsup(format!("not a Flux directory: {}", spec.checkpoint.display())))?;
            let ckpt = FluxCheckpoint::open(&paths).map_err(|e| unsup(format!("open flux dir: {e}")))?;
            let v = ckpt.variant;
            (
                v,
                ckpt.transformer_component().map_err(|e| unsup(format!("transformer: {e}")))?,
                ckpt.vae_component().map_err(|e| unsup(format!("vae: {e}")))?,
                ckpt.clip_l_component().map_err(|e| unsup(format!("clip-l: {e}")))?,
                ckpt.t5_component().map_err(|e| unsup(format!("t5: {e}")))?,
            )
        } else {
            let loader = SingleFileLoader::open(&spec.checkpoint)
                .map_err(|e| unsup(format!("open single-file: {e}")))?;
            if loader.variant != SDFormat::Flux {
                return Err(unsup(format!(
                    "{} is not a Flux checkpoint (variant {:?})",
                    spec.checkpoint.display(),
                    loader.variant
                )));
            }
            let v = detect_variant_single_file(&loader);
            let comp_t = transformer_component_from_loader(&loader)
                .map_err(|e| unsup(format!("transformer: {e}")))?;
            // Each component: explicit override path wins, else from the bundle.
            let from_override = |p: &Path, prefix: &str| -> Result<crate::weight::loader::ComponentTensors> {
                let l = SingleFileLoader::open(p).map_err(|e| unsup(format!("open {}: {e}", p.display())))?;
                component_with_prefix(&l, prefix).map_err(|e| unsup(format!("{}: {e}", p.display())))
            };
            let comp_v = match &spec.vae {
                Some(p) => from_override(p, "")?,
                None => vae_component_from_loader(&loader)
                    .map_err(|e| unsup(format!("vae (pass a separate VAE if the bundle lacks one): {e}")))?,
            };
            let comp_c = match &spec.clip_l {
                Some(p) => from_override(p, "")?,
                None => component_with_prefix(&loader, "text_encoders.clip_l.transformer.")
                    .map_err(|e| unsup(format!("clip-l: {e}")))?,
            };
            let comp_t5 = match &spec.t5 {
                Some(p) => from_override(p, "")?,
                None => component_with_prefix(&loader, "text_encoders.t5xxl.transformer.")
                    .map_err(|e| unsup(format!("t5: {e}")))?,
            };
            (v, comp_t, comp_v, comp_c, comp_t5)
        };

        let variant = spec.variant.unwrap_or(variant_detected);
        let cfg = match variant {
            FluxVariant::Dev => FluxConfig::flux_dev(),
            FluxVariant::Schnell => FluxConfig::flux_schnell(),
        };

        // ---- transformer straight to GPU ----
        let transformer = GpuFluxTransformer::load(cfg.clone(), &comp_t)?;
        drop(comp_t);

        // ---- VAE (16-channel) ----
        let mut vae = AutoencoderKL::from_config(VAEConfig::flux());
        vae.load_from(&comp_v).map_err(|e| unsup(format!("load vae: {e}")))?;
        drop(comp_v);

        // ---- CLIP-L (pooled vector) ----
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
        clip_l.load_from(&comp_c).map_err(|e| unsup(format!("load clip-l: {e}")))?;
        drop(comp_c);

        // ---- T5-XXL ----
        let mut t5 = T5Encoder::from_config(T5Config::t5_xxl());
        t5.load_from(&comp_t5).map_err(|e| unsup(format!("load t5: {e}")))?;
        drop(comp_t5);

        // ---- tokenizers ----
        let clip_tok = CLIPTokenizer::from_file(&spec.clip_tokenizer)
            .map_err(|e| unsup(format!("clip tokenizer: {e}")))?;
        let t5_max = match variant {
            FluxVariant::Schnell => 256,
            FluxVariant::Dev => 512,
        };
        let t5_tok = T5Tokenizer::from_file(&spec.t5_tokenizer, t5_max)
            .map_err(|e| unsup(format!("t5 tokenizer: {e}")))?;

        Ok(Self { transformer, cfg, variant, vae, clip_l, t5, clip_tok, t5_tok })
    }

    /// Render one image. `on_step(step, total)` reports progress; `abort` stops
    /// the denoise loop early (returns the partial decode).
    pub fn generate(
        &self,
        opts: &FluxGenOpts,
        mut on_step: impl FnMut(usize, usize),
        abort: &AtomicBool,
    ) -> Result<FluxImage> {
        let unsup = |e: String| DiffusionError::Unsupported(e);
        // Flux works in multiples of 16 px (8× VAE × 2× patch).
        let w = (opts.width / 16).max(1) * 16;
        let h = (opts.height / 16).max(1) * 16;

        // ---- encode prompt ----
        let clip_ids = self.clip_tok.encode_padded(&opts.prompt).map_err(|e| unsup(format!("clip encode: {e}")))?;
        let pooled = self.clip_l.pooled_forward(&clip_ids).map_err(|e| unsup(format!("clip-l forward: {e}")))?;
        let (t5_ids, t5_mask) = self.t5_tok.encode_padded(&opts.prompt).map_err(|e| unsup(format!("t5 encode: {e}")))?;
        let t5_embed = self.t5.forward(&t5_ids, Some(&t5_mask));
        let l_txt = t5_ids.len();

        let lat_h = h / 8;
        let lat_w = w / 8;
        let in_c = self.cfg.in_channels;
        let img_seq_len = (lat_h / self.cfg.patch_size) * (lat_w / self.cfg.patch_size);

        let n_steps = if opts.steps == 0 {
            match self.variant {
                FluxVariant::Schnell => 4,
                FluxVariant::Dev => 50,
            }
        } else {
            opts.steps
        };
        let guidance = if self.cfg.guidance_embeds { opts.guidance } else { 0.0 };

        // ---- img2img / inpaint setup ----
        let strength = opts.strength.clamp(0.0, 1.0);
        let (init_x0, mask_latent): (Option<Vec<f32>>, Option<Vec<f32>>) =
            if let Some(bytes) = &opts.input_image_bytes {
                let pixels = load_image_bytes_as_chw_f32(bytes, h, w).map_err(|e| unsup(format!("input image: {e}")))?;
                let unscaled = self.vae.encode_gpu(&pixels, 1, h, w).map_err(|e| unsup(format!("vae encode: {e}")))?;
                if unscaled.iter().any(|v| !v.is_finite()) {
                    return Err(unsup("vae encode produced NaN/Inf".into()));
                }
                let mut x0 = unscaled;
                self.vae.scale_latent(&mut x0);
                let mask = match &opts.mask_bytes {
                    Some(mb) => {
                        let raw = load_mask_bytes(mb, h, w).map_err(|e| unsup(format!("mask: {e}")))?;
                        Some(downsample_mask(&raw, h, w, 8).map_err(|e| unsup(format!("downsample mask: {e}")))?)
                    }
                    None => None,
                };
                (Some(x0), mask)
            } else {
                (None, None)
            };
        let start_step = if init_x0.is_some() {
            ((1.0 - strength) * n_steps as f32).floor() as usize
        } else {
            0
        };

        // ---- initial latent ----
        let mut latent = sample_gaussian(&[in_c, lat_h, lat_w], opts.seed);
        let sched_cfg = match self.variant {
            FluxVariant::Dev => FlowMatchConfig::flux_dev(),
            FluxVariant::Schnell => FlowMatchConfig::flux_schnell(),
        };
        let mut scheduler = FlowMatchScheduler::new(sched_cfg);
        scheduler.set_image_seq_len(img_seq_len);
        scheduler.set_timesteps(n_steps);

        if let Some(x0) = &init_x0 {
            let s = scheduler.sigma_at(start_step);
            for (l, &x) in latent.iter_mut().zip(x0.iter()) {
                *l = (1.0 - s) * x + s * *l;
            }
        } else {
            let init_sigma = scheduler.init_noise_sigma();
            if (init_sigma - 1.0).abs() > 1e-3 {
                for v in latent.iter_mut() { *v *= init_sigma; }
            }
        }

        // ---- denoise loop ----
        for step in start_step..n_steps {
            let t = scheduler.timestep_f32(step);
            let v = self.transformer.forward(&latent, lat_h, lat_w, &t5_embed, l_txt, &pooled, t, guidance);
            latent = scheduler.step(&v, step, &latent);

            if let Some(x0) = &init_x0 {
                let next_sigma = if step + 1 < scheduler.sigmas().len() { scheduler.sigma_at(step + 1) } else { 0.0 };
                let noise = sample_gaussian(
                    &[in_c, lat_h, lat_w],
                    opts.seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(step as u64 + 1),
                );
                let lat_pixel = lat_h * lat_w;
                for j in 0..latent.len() {
                    let m = mask_latent.as_ref().map(|mk| mk[j % lat_pixel]).unwrap_or(1.0);
                    let unmasked = (1.0 - next_sigma) * x0[j] + next_sigma * noise[j];
                    latent[j] = m * latent[j] + (1.0 - m) * unmasked;
                }
            }

            on_step(step + 1, n_steps);
            if abort.load(Ordering::Relaxed) {
                break;
            }
        }

        // ---- decode (GPU) ----
        self.vae.unscale_latent(&mut latent);
        let chw = self.vae.decode_with_dims_gpu(&latent, 1, lat_h, lat_w);
        Ok(FluxImage { chw, height: h, width: w })
    }
}

/// Seeded standard-normal noise (splitmix64 + Box-Muller) — matches the
/// `flux_generate` binary so seeds reproduce.
fn sample_gaussian(shape: &[usize], seed: u64) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut next_u32 = move || -> u32 {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        z as u32
    };
    let mut to_unit = || -> f32 {
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
