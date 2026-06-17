//! Reusable SD 1.5 / SDXL pipeline.
//!
//! Splits the work that used to live in `sd_generate`'s `main()` into two
//! halves so the desktop shell — and any other caller — can load a
//! checkpoint once and run many generations without per-call subprocess
//! overhead:
//!
//!   1. [`SdResidents::load`] — opens the checkpoint, builds the CLIP
//!      encoders, UNet, VAE, optional ControlNet, scheduler, and
//!      tokenizers, and returns a single struct that owns all the
//!      long-lived state.
//!   2. [`generate`] — runs one full generation against `&mut SdResidents`,
//!      reusing the loaded models. Driven by [`GenerateRequest`]; emits
//!      per-step progress via a callback and accepts a shared
//!      [`AtomicBool`] for cooperative cancellation.
//!
//! The binary keeps its CLI parsing, REPL UI, and PNG IO; everything else
//! goes through here.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use crate::config::{CheckpointConfig, SDVariant};
use crate::controlnet::ControlNet2DModel;
use crate::error::{DiffusionError, Result};
use crate::scheduler::DDIMScheduler;
use crate::text_encoder::{CLIPTextModel, SDXLDualTextEncoder};
use crate::tokenizer::CLIPTokenizer;
use crate::unet::UNet2DConditionModel;
use crate::vae::AutoencoderKL;
use crate::weight::{ComponentTensors, SDFormat, SingleFileLoader};

// ---- load -----------------------------------------------------------------

/// Where the weights came from. Held by [`SdResidents`] so the
/// per-prompt path can re-fetch components if it ever needs to (it
/// generally doesn't — everything's eager-loaded in [`SdResidents::load`]).
pub enum WeightSrc {
    /// Diffusers folder layout: subdirs `text_encoder/`, `unet/`, `vae/`, …
    Folder(PathBuf),
    /// Single-file CompVis / A1111 `.safetensors` (UNet + VAE + CLIP-L,
    /// plus CLIP-G for SDXL, all keyed by their original key namespaces).
    Single(Box<SingleFileLoader>),
}

#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Either a directory (Diffusers layout) or a `.safetensors` file
    /// (single-file CompVis layout). The constructor sniffs which it is.
    pub checkpoint: PathBuf,
    /// `tokenizer.json` for CLIP-L. For folder-mode checkpoints we can
    /// auto-discover this under `<checkpoint>/tokenizer/`; for single-file
    /// checkpoints the caller must supply it.
    pub tokenizer_l: Option<PathBuf>,
    /// `tokenizer.json` for CLIP-G. Only used for SDXL; auto-discoverable
    /// in folder mode at `<checkpoint>/tokenizer_2/`.
    pub tokenizer_g: Option<PathBuf>,
    /// Optional `.safetensors` ControlNet to load alongside the UNet.
    pub controlnet: Option<PathBuf>,
    /// Override variant detection. Accepts `"sd15"` or `"sdxl"`.
    pub force_variant: Option<String>,
    /// Enables the Metal "GPU residence" path in the UNet/VAE when true.
    pub gpu_residence: bool,
}

/// Everything that's expensive to construct — held by value so callers
/// can `&mut` it once and run many generations against the same residents.
pub struct SdResidents {
    pub variant: SDVariant,
    pub cfg: CheckpointConfig,
    pub weight_src: WeightSrc,
    pub text_l: CLIPTextModel,
    pub dual: Option<SDXLDualTextEncoder>,
    pub unet: UNet2DConditionModel,
    pub vae: AutoencoderKL,
    pub controlnet: Option<ControlNet2DModel>,
    pub tok_l: CLIPTokenizer,
    pub tok_g: Option<CLIPTokenizer>,
    pub scheduler: DDIMScheduler,
    pub gpu_residence: bool,
}

impl SdResidents {
    /// Open the checkpoint and load every component. Mirrors what
    /// `sd_generate`'s main() does today (incl. all the modelspec
    /// metadata-driven config overrides). Emits the same `eprintln!`
    /// progress so the existing CLI logs are unchanged.
    pub fn load(opts: &LoadOptions) -> Result<Self> {
        let p = &opts.checkpoint;
        if p.as_os_str().is_empty() {
            return Err(DiffusionError::Unsupported(
                "LoadOptions.checkpoint is empty".into(),
            ));
        }

        // Decide folder vs single-file from the path itself.
        let is_dir = p.is_dir();
        let (cfg, weight_src, fake_root) = if is_dir {
            let cfg = CheckpointConfig::from_dir(p)
                .map_err(|e| DiffusionError::Unsupported(format!("config: {e}")))?;
            (cfg, WeightSrc::Folder(p.clone()), p.clone())
        } else {
            eprintln!("opening single-file checkpoint {}", p.display());
            let loader = SingleFileLoader::open(p)
                .map_err(|e| DiffusionError::Unsupported(format!("single-file: {e}")))?;
            if let Some(t) = &loader.metadata.title {
                eprintln!("modelspec title: {t:?}");
            }
            if let Some(a) = &loader.metadata.architecture {
                eprintln!("modelspec architecture: {a:?}");
            }
            if let Some(pp) = &loader.metadata.prediction_type {
                eprintln!("modelspec prediction_type: {pp:?}");
            }
            if let Some(r) = &loader.metadata.resolution {
                eprintln!("modelspec resolution: {r:?}");
            }
            if let Some(l) = &loader.metadata.license {
                eprintln!("modelspec license: {l:?}");
            }
            eprintln!("detected variant: {:?}", loader.variant);

            let variant = match opts.force_variant.as_deref() {
                Some("sd15") => SDFormat::Sd15,
                Some("sdxl") => SDFormat::Sdxl,
                Some(other) => {
                    return Err(DiffusionError::Unsupported(format!(
                        "force_variant {other:?} not recognised (use sd15|sdxl)"
                    )));
                }
                None => loader.variant,
            };
            let fake_root = p.parent().unwrap_or(Path::new(".")).to_path_buf();
            let mut cfg = match variant {
                SDFormat::Sd15 => CheckpointConfig::sd15_default(fake_root.clone()),
                SDFormat::Sdxl => CheckpointConfig::sdxl_default(fake_root.clone()),
                SDFormat::Flux => {
                    return Err(DiffusionError::Unsupported(
                        "Flux checkpoint — use the flux_generate binary".into(),
                    ));
                }
                SDFormat::Unknown => {
                    return Err(DiffusionError::Unsupported(
                        "could not detect SD variant; set LoadOptions.force_variant".into(),
                    ));
                }
            };
            // Inpaint detection: 9-channel conv_in → swap in_channels so
            // the UNet's first conv is allocated wide enough.
            if let Some(in_c) = loader.unet_in_channels() {
                if in_c != cfg.unet.in_channels {
                    eprintln!(
                        "(unet.in_channels {} → {in_c} from conv_in weight shape — \
                         inpaint UNet detected)",
                        cfg.unet.in_channels
                    );
                    cfg.unet.in_channels = in_c;
                }
            }
            // Modelspec overrides.
            if let Some(pt) = &loader.metadata.prediction_type {
                cfg.scheduler.prediction_type = pt.clone();
                eprintln!(
                    "(scheduler.prediction_type overridden to {:?} from modelspec)",
                    pt
                );
            }
            if let Some((w, h)) = loader.metadata.resolution_pixels() {
                let new_sample = (w.max(h) / 8) as usize;
                if new_sample != cfg.unet.sample_size {
                    eprintln!(
                        "(unet.sample_size {} → {} from modelspec resolution {}x{})",
                        cfg.unet.sample_size, new_sample, w, h
                    );
                    cfg.unet.sample_size = new_sample;
                    cfg.vae.sample_size = (w.max(h)) as usize;
                }
            }
            (cfg, WeightSrc::Single(Box::new(loader)), fake_root)
        };

        let variant = cfg.variant();
        eprintln!("variant: {:?}", variant);
        let _ = fake_root; // referenced when sd_generate-style folder load tracks roots

        // Component loader closure: same as sd_generate. Folder mode reads
        // per-subdir; single-file mode pulls a named component from the
        // mmapped loader.
        let load_component = |name: &str| -> Option<ComponentTensors> {
            match &weight_src {
                WeightSrc::Folder(root) => ComponentTensors::open_dir(&root.join(name)).ok(),
                WeightSrc::Single(loader) => loader.component(name).ok(),
            }
        };

        // CLIP-L.
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

        // CLIP-G (SDXL only). Re-uses the just-loaded CLIP-L via a manual
        // deep copy so we don't load it twice.
        let mut dual = cfg
            .text_encoder_2
            .as_ref()
            .map(|t2| SDXLDualTextEncoder::from_configs(cfg.text_encoder.clone(), t2.clone()));
        if let Some(d) = &mut dual {
            d.clip_l = clone_clip_text_model(&text_l);
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

        // UNet.
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

        // Optional ControlNet.
        let controlnet: Option<ControlNet2DModel> = if let Some(cp) = &opts.controlnet {
            eprintln!("[controlnet] loading {} …", cp.display());
            let comp = match SingleFileLoader::open(cp) {
                Ok(loader) => {
                    use crate::weight::TensorRef;
                    use std::collections::HashMap;
                    let translated: HashMap<String, TensorRef> = loader
                        .raw_tensors
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    Some(ComponentTensors::from_single_mmap_borrow(&loader, translated))
                }
                Err(e) => {
                    eprintln!("  ⚠ open controlnet: {e}");
                    None
                }
            };
            if let Some(comp) = comp {
                let cn_cfg =
                    crate::controlnet::ControlNetConfig::from_unet(cfg.unet.clone());
                let mut cn = ControlNet2DModel::from_config(cn_cfg);
                match cn.load_from(&comp) {
                    Ok(()) => {
                        eprintln!(
                            "  ✓ controlnet loaded ({} down residuals + 1 mid residual)",
                            cn.num_down_residuals()
                        );
                        Some(cn)
                    }
                    Err(e) => {
                        eprintln!("  ⚠ controlnet load: {e}");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // VAE.
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

        let scheduler = DDIMScheduler::new(&cfg.scheduler);

        // Tokenizers — paths are auto-discoverable for folder layouts;
        // single-file callers must pass them through LoadOptions.
        let tok_l_path = match (&weight_src, &opts.tokenizer_l) {
            (_, Some(p)) => p.clone(),
            (WeightSrc::Folder(root), None) => root.join("tokenizer").join("tokenizer.json"),
            (WeightSrc::Single(_), None) => {
                return Err(DiffusionError::Unsupported(
                    "single-file checkpoint requires LoadOptions.tokenizer_l".into(),
                ));
            }
        };
        let tok_l = CLIPTokenizer::from_file(&tok_l_path)
            .map_err(|e| DiffusionError::Unsupported(format!("CLIP-L tokenizer: {e}")))?;

        // CLIP-G tokenizer (SDXL only). Auto-discover in folder mode; for
        // single-file mode it's optional — the per-prompt path will handle
        // a missing one by falling back to CLIP-L chunks for CLIP-G.
        let tok_g: Option<CLIPTokenizer> = if matches!(variant, SDVariant::Sdxl) {
            let path = match (&weight_src, &opts.tokenizer_g) {
                (_, Some(p)) => Some(p.clone()),
                (WeightSrc::Folder(root), None) => {
                    Some(root.join("tokenizer_2").join("tokenizer.json"))
                }
                (WeightSrc::Single(_), None) => None,
            };
            match path {
                Some(p) => CLIPTokenizer::from_file(&p).ok(),
                None => None,
            }
        } else {
            None
        };

        Ok(SdResidents {
            variant,
            cfg,
            weight_src,
            text_l,
            dual,
            unet,
            vae,
            controlnet,
            tok_l,
            tok_g,
            scheduler,
            gpu_residence: opts.gpu_residence,
        })
    }
}

/// Compile-time probe for the Metal GPU backend. Lets callers display
/// "metal=true/false" at startup without having to feature-gate their own
/// crate against ours.
#[cfg(feature = "metal")]
pub const HAS_METAL: bool = true;
#[cfg(not(feature = "metal"))]
pub const HAS_METAL: bool = false;

// ---- generate -------------------------------------------------------------

use std::sync::atomic::Ordering;

use crate::scheduler::Scheduler;
use crate::unet::SDXLAdditionalConditioning;

/// Per-prompt configuration for [`generate`].
///
/// Bytes-everywhere by design: no filesystem paths in this struct. The
/// CLI binary reads its `--input` / `--mask` / `--control-image` files
/// into bytes before calling; the desktop shell hands over the bytes
/// it already has in memory. This keeps the library callable in
/// incognito mode without ever touching disk.
#[derive(Debug, Clone, Default)]
pub struct GenerateRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub steps: usize,
    pub cfg_scale: f32,
    pub seed: u64,
    /// Bytes of the source PNG/JPEG for img2img / inpaint. **Not yet
    /// supported by the library path** — set this and `generate` returns
    /// an error pointing at the inline sd_generate fallback.
    pub input_image_bytes: Option<Vec<u8>>,
    /// Bytes of the grayscale mask. Companion to `input_image_bytes`.
    pub mask_bytes: Option<Vec<u8>>,
    /// 0..=1 denoise strength when `input_image_bytes` is set.
    pub strength: f32,
    /// Bytes of the ControlNet conditioning image (edge/depth/etc).
    /// When `None` and the residents loaded a ControlNet, the residents'
    /// CN is silently ignored for this request — pass `Some` here to
    /// engage it.
    pub control_image_bytes: Option<Vec<u8>>,
    pub controlnet_weight: f32,
    pub controlnet_start: f32,
    pub controlnet_end: f32,
    /// Additive per-position bias added to every token of the cond
    /// embedding (length must equal `cross_attention_dim`). The CLIP-
    /// space user steering vector (Refactor 1's `image_clip_*` user
    /// vector) lands here.
    pub cond_bias: Option<Vec<f32>>,
}

/// Streaming progress event. Emitted at each denoise step.
#[derive(Debug, Clone)]
pub struct GenerateProgress {
    pub step: usize,
    pub total: usize,
    pub elapsed_s: f32,
}

/// Result of one generation. `rgb_chw` is the VAE decoder's f32 output in
/// `[-1, 1]` convention, shape `[3 * height * width]` in CHW order — same
/// thing `image_io::save_png` consumes. Callers can route it straight to
/// PNG, base64-encode it, or copy it into a shared buffer.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub rgb_chw: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

/// Run one full generation against an already-loaded [`SdResidents`].
/// The function reuses the resident CLIP / UNet / VAE, so back-to-back
/// calls amortize the multi-GB load cost.
///
/// Cancellation: poll the shared `AtomicBool` between steps; setting it
/// to `true` causes `generate` to return [`DiffusionError::Aborted`]
/// after the current UNet forward finishes (we don't kill mid-forward
/// because that would leak partial GPU state).
///
/// **Coverage**: this entry currently handles txt2img end-to-end,
/// including ControlNet residual injection when `req.control_image_bytes`
/// is supplied. The img2img / inpaint paths (per-step CPU blend, native
/// 9-channel UNet) live inline in `sd_generate` for now and will be
/// migrated here as a follow-up. Setting `input_image_bytes` returns
/// [`DiffusionError::Unsupported`] with a clear message.
pub fn generate(
    residents: &mut SdResidents,
    req: &GenerateRequest,
    mut on_progress: impl FnMut(GenerateProgress),
    abort: &std::sync::atomic::AtomicBool,
) -> Result<GeneratedImage> {
    // A bare mask with no source image makes no sense.
    if req.input_image_bytes.is_none() && req.mask_bytes.is_some() {
        return Err(DiffusionError::Unsupported(
            "mask_bytes set without input_image_bytes — mask requires a source image".into(),
        ));
    }

    let t_start = std::time::Instant::now();
    // Destructure rather than clone — CheckpointConfig isn't Clone (its
    // schedule arrays are owned `Vec<f32>`) and we only need the scalars.
    let cad = residents.cfg.unet.cross_attention_dim;
    let latent_h = residents.cfg.unet.sample_size;
    let latent_w = latent_h; // square latents for SD/SDXL
    let latent_c = residents.cfg.vae.latent_channels;
    let unet_in_c = residents.cfg.unet.in_channels;
    let vae_scale = residents.cfg.vae.scaling_factor;
    let img_h = latent_h * 8;
    let img_w = latent_w * 8;

    // 1. Chunked tokenize + CLIP encode (cond + uncond).
    let chunks_l = residents
        .tok_l
        .encode_chunks(&req.prompt)
        .map_err(|e| DiffusionError::Unsupported(format!("CLIP-L encode: {e}")))?;
    let n_chunks = chunks_l.len();
    let chunks_g: Vec<Vec<u32>> = if residents.variant == SDVariant::Sdxl {
        match residents.tok_g.as_ref() {
            Some(t) => match t.encode_chunks_with(&req.prompt, 0) {
                Ok(c) if c.len() == n_chunks => c,
                _ => chunks_l.clone(),
            },
            None => chunks_l.clone(),
        }
    } else {
        chunks_l.clone()
    };

    let (mut text_emb, pooled_g) = match residents.variant {
        SDVariant::Sdxl => {
            let d = residents
                .dual
                .as_ref()
                .ok_or_else(|| DiffusionError::Unsupported("SDXL but no dual encoder".into()))?;
            let (emb, p) = d
                .forward_chunked(&chunks_l, &chunks_g)
                .map_err(|e| DiffusionError::Unsupported(format!("text encode: {e}")))?;
            (emb, Some(p))
        }
        _ => {
            let emb = residents
                .text_l
                .forward_chunked(&chunks_l, None)
                .map_err(|e| DiffusionError::Unsupported(format!("text encode: {e}")))?;
            (emb, None)
        }
    };

    // 2. CLIP-space cond-bias — broadcast over every position. Uncond
    //    intentionally stays unbiased so CFG keeps the signal.
    if let Some(bias) = &req.cond_bias {
        if bias.len() != cad {
            return Err(DiffusionError::Unsupported(format!(
                "cond_bias length {} != cross_attention_dim {}",
                bias.len(),
                cad
            )));
        }
        let positions = text_emb.len() / cad;
        for i in 0..positions {
            let row = &mut text_emb[i * cad..(i + 1) * cad];
            for (h, b) in row.iter_mut().zip(bias.iter()) {
                *h += *b;
            }
        }
    }

    let do_cfg = req.cfg_scale > 1.0 + 1e-6;
    let (text_emb_uncond, pooled_g_uncond) = if do_cfg {
        let mut neg_chunks_l = residents
            .tok_l
            .encode_chunks(&req.negative_prompt)
            .map_err(|e| DiffusionError::Unsupported(format!("CLIP-L neg: {e}")))?;
        let empty_l = residents.tok_l.encode_chunks("").unwrap_or_default();
        while neg_chunks_l.len() < n_chunks {
            neg_chunks_l.push(empty_l[0].clone());
        }
        neg_chunks_l.truncate(n_chunks);
        let neg_chunks_g: Vec<Vec<u32>> = if residents.variant == SDVariant::Sdxl {
            match residents.tok_g.as_ref() {
                Some(t) => {
                    let mut c = t
                        .encode_chunks_with(&req.negative_prompt, 0)
                        .unwrap_or_else(|_| neg_chunks_l.clone());
                    let empty_g = t.encode_chunks_with("", 0).unwrap_or_default();
                    while c.len() < n_chunks && !empty_g.is_empty() {
                        c.push(empty_g[0].clone());
                    }
                    c.truncate(n_chunks);
                    if c.len() != n_chunks {
                        neg_chunks_l.clone()
                    } else {
                        c
                    }
                }
                None => neg_chunks_l.clone(),
            }
        } else {
            neg_chunks_l.clone()
        };
        match residents.variant {
            SDVariant::Sdxl => {
                let d = residents.dual.as_ref().unwrap();
                let (e, p) = d
                    .forward_chunked(&neg_chunks_l, &neg_chunks_g)
                    .map_err(|e| DiffusionError::Unsupported(format!("neg encode: {e}")))?;
                (Some(e), Some(p))
            }
            _ => {
                let e = residents
                    .text_l
                    .forward_chunked(&neg_chunks_l, None)
                    .map_err(|e| DiffusionError::Unsupported(format!("neg encode: {e}")))?;
                (Some(e), None)
            }
        }
    } else {
        (None, None)
    };

    // 3. Scheduler init.
    residents.scheduler.set_timesteps(req.steps);
    let total_steps = residents.scheduler.timesteps().len();
    let vae_shift = residents.cfg.vae.shift_factor;

    // 3a. Img2img / inpaint setup — VAE-encode the source image (if any)
    //     and downsample the mask to latent resolution. The init_latent_x0
    //     becomes the "denoising anchor" for naive img2img; for native-
    //     inpaint UNets it's also fed via the 9-channel input.
    let (init_latent_x0, mask_latent, start_step): (Option<Vec<f32>>, Option<Vec<f32>>, usize) =
        if let Some(bytes) = &req.input_image_bytes {
            let img_pixels =
                crate::image_io::load_image_bytes_as_chw_f32(bytes, img_h, img_w)?;
            // CPU VAE encode. sd_generate has an optional GPU encode path
            // with NaN auto-retry; for the library we keep it simple and
            // CPU-only. The desktop wrapper can still set gpu_residence
            // for UNet + VAE-decode.
            // Prefer the Metal encoder when the residents were loaded
            // with gpu_residence + the metal feature is on. Falls back to
            // CPU on failure or when Metal isn't available — same dance
            // sd_generate does inline.
            let unscaled = {
                #[cfg(feature = "metal")]
                {
                    if residents.gpu_residence {
                        match residents.vae.encode_gpu(&img_pixels, 1, img_h, img_w) {
                            Ok(v) => v,
                            Err(_) => residents
                                .vae
                                .encode(&img_pixels, 1, img_h, img_w)
                                .map_err(|e| DiffusionError::Unsupported(
                                    format!("vae encode: {e}")))?,
                        }
                    } else {
                        residents
                            .vae
                            .encode(&img_pixels, 1, img_h, img_w)
                            .map_err(|e| DiffusionError::Unsupported(
                                format!("vae encode: {e}")))?
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    residents
                        .vae
                        .encode(&img_pixels, 1, img_h, img_w)
                        .map_err(|e| DiffusionError::Unsupported(format!("vae encode: {e}")))?
                }
            };
            if unscaled.iter().any(|v| !v.is_finite()) {
                return Err(DiffusionError::Unsupported(
                    "VAE encode produced NaN/Inf on the input image".into(),
                ));
            }
            let mut x0 = unscaled;
            for v in x0.iter_mut() {
                *v = (*v - vae_shift) * vae_scale;
            }
            let mask = if let Some(mb) = &req.mask_bytes {
                let raw = crate::image_io::load_mask_bytes(mb, img_h, img_w)?;
                Some(crate::image_io::downsample_mask(&raw, img_h, img_w, 8)?)
            } else {
                None
            };
            let ss = ((1.0 - req.strength.clamp(0.0, 1.0)) * total_steps as f32).floor() as usize;
            let ss = ss.min(total_steps);
            (Some(x0), mask, ss)
        } else {
            (None, None, 0)
        };

    // 3b. Native-inpaint detection: 9-channel conv_in + we have a source
    //     image → use the 9-channel UNet feed and skip the per-step CPU
    //     blend. The model handles mask via its augmented input directly.
    let native_inpaint = unet_in_c == 9 && init_latent_x0.is_some();
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
            masked[i] *= 1.0 - m;
        }
        (Some(masked), Some(mask_l))
    } else {
        (None, None)
    };

    // 4. Initial latent. Four flavors:
    //    - native-inpaint: pure Gaussian noise (model gets x0 via the
    //      9-channel input)
    //    - naive img2img: re-noise x0 up to the start-step σ
    //    - txt2img: pure Gaussian noise
    let mut latent: Vec<f32> = if native_inpaint {
        deterministic_noise(latent_c * latent_h * latent_w, req.seed)
    } else if let Some(x0) = &init_latent_x0 {
        if start_step >= total_steps {
            // strength = 0: leave x0 unchanged, denoise loop runs 0 iters.
            x0.clone()
        } else {
            let t_start_sched = residents.scheduler.timesteps()[start_step];
            let alpha_bar = residents.scheduler.alpha_bar_at(t_start_sched);
            let sqrt_alpha = alpha_bar.sqrt();
            let sqrt_one_minus = (1.0_f32 - alpha_bar).sqrt();
            let noise = deterministic_noise(x0.len(), req.seed);
            let mut l = vec![0.0_f32; x0.len()];
            for i in 0..x0.len() {
                l[i] = sqrt_alpha * x0[i] + sqrt_one_minus * noise[i];
            }
            l
        }
    } else {
        deterministic_noise(latent_c * latent_h * latent_w, req.seed)
    };

    // 5. ControlNet control image (optional).
    let cn_active = residents.controlnet.is_some() && req.control_image_bytes.is_some();
    let control_image_buf: Option<Vec<f32>> = if cn_active {
        let bytes = req.control_image_bytes.as_ref().unwrap();
        let v = crate::image_io::load_image_bytes_as_chw_f32(bytes, img_h, img_w)?;
        if do_cfg {
            let mut doubled = Vec::with_capacity(v.len() * 2);
            doubled.extend_from_slice(&v);
            doubled.extend_from_slice(&v);
            Some(doubled)
        } else {
            Some(v)
        }
    } else {
        None
    };

    // 6. Build batched [uncond | cond] text input + precompute KV cache.
    let text_seq_cond = text_emb.len() / cad;
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
    let _ = n_batch;
    residents.unet.precompute_text_kv(&text_emb_run, text_seq_cond);
    // Also warm the GPU-resident KV cache so the GPU forward path can
    // skip per-step K/V matmuls (matching what sd_generate does inline).
    #[cfg(feature = "metal")]
    if residents.gpu_residence {
        residents.unet.precompute_text_kv_gpu(&text_emb_run, text_seq_cond);
    }

    let single_latent_floats = latent_c * latent_h * latent_w;
    let single_unet_input_floats = unet_in_c * latent_h * latent_w;
    let mut latent_input: Vec<f32> = Vec::with_capacity(
        if do_cfg { 2 } else { 1 } * single_unet_input_floats,
    );

    // Native-inpaint always starts from full noise; the start_step skip
    // is only meaningful for the naive img2img path.
    let effective_start_step = if native_inpaint { 0 } else { start_step };
    let lat_pixel = latent_h * latent_w;

    // 7. Denoise loop. Three flavours live here: txt2img (no x0, no
    //    native), naive img2img / inpaint (x0 + per-step blend), and
    //    native-inpaint (9-channel UNet feed + no per-step blend).
    let timesteps: Vec<i64> = residents.scheduler.timesteps().to_vec();
    for (step_idx, &t) in timesteps.iter().enumerate() {
        if step_idx < effective_start_step { continue; }
        if abort.load(Ordering::SeqCst) {
            return Err(DiffusionError::Unsupported("aborted".into()));
        }
        let step_start = std::time::Instant::now();
        let t_f = t as f32;

        // Build the per-step UNet input.
        latent_input.clear();
        if native_inpaint {
            // Per CFG element: cat([noisy_latent (4ch) | masked (4ch) | mask (1ch)]).
            // Channel layout is `[c, h, w]` flattened; we concatenate the
            // three sub-tensors in channel-stacked order.
            let masked = masked_init_latent.as_ref().unwrap();
            let mask_l = mask_lat_for_unet.as_ref().unwrap();
            latent_input.extend_from_slice(&latent);
            latent_input.extend_from_slice(masked);
            latent_input.extend_from_slice(&mask_l[..lat_pixel]);
            if do_cfg {
                latent_input.extend_from_slice(&latent);
                latent_input.extend_from_slice(masked);
                latent_input.extend_from_slice(&mask_l[..lat_pixel]);
            }
        } else {
            latent_input.extend_from_slice(&latent);
            if do_cfg {
                latent_input.extend_from_slice(&latent);
            }
        }

        // ControlNet residuals gated by step fraction.
        let frac = step_idx as f32 / total_steps.max(1) as f32;
        let cn_in_window = cn_active
            && frac >= req.controlnet_start
            && frac < req.controlnet_end;
        let cn_pooled_ref: Option<&[f32]> = pooled_run.as_deref();
        let cn_addl: Option<SDXLAdditionalConditioning> =
            if residents.variant == SDVariant::Sdxl {
                Some(SDXLAdditionalConditioning::default_1024())
            } else {
                None
            };
        let cn_cpu = if cn_in_window {
            residents.controlnet.as_ref().and_then(|cn| {
                let img = control_image_buf.as_ref().unwrap();
                cn.forward(
                    &latent_input,
                    t_f,
                    &text_emb_run,
                    img,
                    img_h,
                    img_w,
                    cn_pooled_ref,
                    cn_addl,
                )
                .ok()
            })
        } else {
            None
        };
        let mut cn_cpu_scaled = cn_cpu;
        if let Some((dr, mr)) = cn_cpu_scaled.as_mut() {
            if req.controlnet_weight != 1.0 {
                for d in dr.iter_mut() {
                    for v in d.iter_mut() {
                        *v *= req.controlnet_weight;
                    }
                }
                for v in mr.iter_mut() {
                    *v *= req.controlnet_weight;
                }
            }
        }

        // UNet forward. Metal path branches on `residents.gpu_residence`
        // and uses the GPU-resident model + KV cache for ~10-50× speedup
        // on Apple Silicon. ControlNet residuals: CPU forwards already
        // hold them as `Vec<Vec<f32>>`; GPU forwards expect them as
        // `GpuTensor`s, so we recompute under the GPU branch when CN is
        // active. (CN is None for the desktop path today — the GPU CN
        // branch is here for forward-compat once we wire it through.)
        let eps_batched = {
            #[cfg(feature = "metal")]
            {
                if residents.gpu_residence {
                    let pooled_for_sdxl = pooled_run.as_ref().ok_or_else(|| {
                        DiffusionError::Unsupported("SDXL requires pooled embedding".into())
                    });
                    match residents.variant {
                        SDVariant::Sdxl => residents.unet.forward_sdxl_gpu_with_controlnet(
                            &latent_input,
                            t_f,
                            &text_emb_run,
                            pooled_for_sdxl?,
                            SDXLAdditionalConditioning::default_1024(),
                            None,
                            None,
                            req.controlnet_weight,
                        ),
                        _ => residents.unet.forward_gpu_with_controlnet(
                            &latent_input,
                            t_f,
                            &text_emb_run,
                            None,
                            None,
                            req.controlnet_weight,
                        ),
                    }
                } else {
                    let dr_ref = cn_cpu_scaled.as_ref().map(|(d, _)| d.as_slice());
                    let mr_ref = cn_cpu_scaled.as_ref().map(|(_, m)| m.as_slice());
                    match residents.variant {
                        SDVariant::Sdxl => residents.unet.forward_sdxl_with_controlnet(
                            &latent_input,
                            t_f,
                            &text_emb_run,
                            pooled_run.as_ref().ok_or_else(|| {
                                DiffusionError::Unsupported("SDXL requires pooled embedding".into())
                            })?,
                            SDXLAdditionalConditioning::default_1024(),
                            dr_ref,
                            mr_ref,
                        ),
                        _ => residents.unet.forward_with_controlnet(
                            &latent_input, t_f, &text_emb_run, dr_ref, mr_ref,
                        ),
                    }
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                let dr_ref = cn_cpu_scaled.as_ref().map(|(d, _)| d.as_slice());
                let mr_ref = cn_cpu_scaled.as_ref().map(|(_, m)| m.as_slice());
                match residents.variant {
                    SDVariant::Sdxl => residents.unet.forward_sdxl_with_controlnet(
                        &latent_input,
                        t_f,
                        &text_emb_run,
                        pooled_run.as_ref().ok_or_else(|| {
                            DiffusionError::Unsupported("SDXL requires pooled embedding".into())
                        })?,
                        SDXLAdditionalConditioning::default_1024(),
                        dr_ref,
                        mr_ref,
                    ),
                    _ => residents.unet.forward_with_controlnet(
                        &latent_input, t_f, &text_emb_run, dr_ref, mr_ref,
                    ),
                }
            }
        }
        .map_err(|e| DiffusionError::Unsupported(format!("UNet step {step_idx}: {e}")))?;

        let eps: Vec<f32> = if do_cfg {
            let (eps_u, eps_c) = eps_batched.split_at(single_latent_floats);
            eps_u
                .iter()
                .zip(eps_c.iter())
                .map(|(u, c)| u + req.cfg_scale * (c - u))
                .collect()
        } else {
            eps_batched
        };

        latent = residents.scheduler.step(&eps, step_idx, &latent);

        // ===== Per-step blend (naive img2img / inpaint only) =====
        // Replace the unmasked region with q_sample(init_latent_x0, t_next)
        // so unmasked pixels track the diffusion trajectory of the source
        // image while the masked region evolves freely. Native-inpaint
        // UNets handle the mask via their 9-channel input and skip this.
        if !native_inpaint {
            if let Some(x0) = &init_latent_x0 {
                let target_t: i64 = if step_idx + 1 < total_steps {
                    residents.scheduler.timesteps()[step_idx + 1]
                } else {
                    -1 // post-final → α_bar ≈ 1, no noise
                };
                let alpha_bar = residents.scheduler.alpha_bar_at(target_t);
                let sqrt_alpha = alpha_bar.sqrt();
                let sqrt_one_minus = (1.0_f32 - alpha_bar).sqrt();
                // Per-step independent noise — RePaint convention. Seed
                // mixes the original seed with the step index so back-to-
                // back batches stay deterministic.
                let noise_seed = req
                    .seed
                    .wrapping_add(0x9E3779B97F4A7C15)
                    .wrapping_mul(step_idx as u64 + 1);
                let noise = deterministic_noise(latent.len(), noise_seed);
                for i in 0..latent.len() {
                    let m = match &mask_latent {
                        Some(mk) => mk[i % lat_pixel],
                        None => 1.0, // no mask → full regen (img2img mode)
                    };
                    let unmasked = sqrt_alpha * x0[i] + sqrt_one_minus * noise[i];
                    latent[i] = m * latent[i] + (1.0 - m) * unmasked;
                }
            }
        }

        let _step_elapsed = step_start.elapsed();
        on_progress(GenerateProgress {
            step: step_idx + 1,
            total: total_steps + 1, // +1 for VAE decode
            elapsed_s: t_start.elapsed().as_secs_f32(),
        });
    }

    // 8. VAE decode — report as the final step so the progress bar
    //    doesn't stall at 100% while the decode runs.
    if vae_scale != 0.0 {
        for v in latent.iter_mut() {
            *v /= vae_scale;
        }
    }
    let img = {
        #[cfg(feature = "metal")]
        {
            if residents.gpu_residence {
                residents.vae.decode_with_dims_gpu(&latent, 1, latent_h, latent_w)
            } else {
                residents.vae.decode_with_dims(&latent, 1, latent_h, latent_w)
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            residents.vae.decode_with_dims(&latent, 1, latent_h, latent_w)
        }
    };

    on_progress(GenerateProgress {
        step: total_steps + 1,
        total: total_steps + 1,
        elapsed_s: t_start.elapsed().as_secs_f32(),
    });

    Ok(GeneratedImage {
        rgb_chw: img,
        height: img_h,
        width: img_w,
    })
}

// ---- video generation (AnimateDiff) ----------------------------------------

/// Result of one video generation: a sequence of frames.
#[derive(Debug, Clone)]
pub struct GeneratedVideo {
    pub frames: Vec<GeneratedImage>,
    pub fps: usize,
}

/// Video generation request.
#[derive(Debug, Clone, Default)]
pub struct VideoRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub steps: usize,
    pub cfg_scale: f32,
    pub seed: u64,
    pub num_frames: usize,
    pub fps: usize,
    pub cond_bias: Option<Vec<f32>>,
}

/// Generate a video clip using AnimateDiff temporal attention.
///
/// The SDXL UNet processes all frames simultaneously:
/// `[batch * num_frames, C, H, W]` for the latent, with temporal
/// attention attending across the frame axis at each spatial position.
/// Returns one `GeneratedImage` per frame.
///
/// **Metal**: the spatial UNet blocks (resnet + spatial attention) run
/// on CPU in this path. A follow-up `generate_video_gpu` variant will
/// pipe spatial blocks through the existing Metal kernels with CPU
/// temporal attention in between.
pub fn generate_video(
    residents: &mut SdResidents,
    motion: &crate::animatediff::MotionModules,
    req: &VideoRequest,
    mut on_progress: impl FnMut(GenerateProgress),
    abort: &std::sync::atomic::AtomicBool,
) -> Result<GeneratedVideo> {
    use std::sync::atomic::Ordering;

    let t_start = std::time::Instant::now();
    let cad = residents.cfg.unet.cross_attention_dim;
    let latent_h = residents.cfg.unet.sample_size;
    let latent_w = latent_h;
    let latent_c = residents.cfg.vae.latent_channels;
    let vae_scale = residents.cfg.vae.scaling_factor;
    let img_h = latent_h * 8;
    let img_w = latent_w * 8;
    let nf = req.num_frames;
    let fps = req.fps.max(1);

    // 1. CLIP encode (same as image path — prompt is shared across frames).
    let chunks_l = residents.tok_l.encode_chunks(&req.prompt)
        .map_err(|e| DiffusionError::Unsupported(format!("CLIP-L encode: {e}")))?;
    let n_chunks = chunks_l.len();
    let chunks_g = match residents.tok_g.as_ref() {
        Some(t) => match t.encode_chunks_with(&req.prompt, 0) {
            Ok(c) if c.len() == n_chunks => c,
            _ => chunks_l.clone(),
        },
        None => chunks_l.clone(),
    };
    let d = residents.dual.as_ref()
        .ok_or_else(|| DiffusionError::Unsupported("AnimateDiff SDXL requires dual encoder".into()))?;
    let (mut text_emb, pooled_g) = d.forward_chunked(&chunks_l, &chunks_g)
        .map_err(|e| DiffusionError::Unsupported(format!("text encode: {e}")))?;
    let pooled = pooled_g;

    // Cond bias.
    if let Some(bias) = &req.cond_bias {
        if bias.len() == cad {
            let positions = text_emb.len() / cad;
            for i in 0..positions {
                let row = &mut text_emb[i * cad..(i + 1) * cad];
                for (h, b) in row.iter_mut().zip(bias.iter()) { *h += *b; }
            }
        }
    }

    // CFG: uncond embedding (negative prompt).
    let do_cfg = req.cfg_scale > 1.0 + 1e-6;

    // For video CFG, we replicate embeddings across `batch * num_frames`.
    // The UNet sees n = cfg_batch * num_frames total samples.
    // batch=1 for both cond and uncond → cfg_batch=2, n = 2 * nf.
    let (text_emb_run, pooled_run, n_unet) = if do_cfg {
        // Encode negative.
        let neg_chunks_l = residents.tok_l.encode_chunks(&req.negative_prompt)
            .unwrap_or_default();
        let mut neg_cl = neg_chunks_l;
        let empty_l = residents.tok_l.encode_chunks("").unwrap_or_default();
        while neg_cl.len() < n_chunks { neg_cl.push(empty_l[0].clone()); }
        neg_cl.truncate(n_chunks);
        let neg_cg = match residents.tok_g.as_ref() {
            Some(t) => {
                let mut c = t.encode_chunks_with(&req.negative_prompt, 0)
                    .unwrap_or_else(|_| neg_cl.clone());
                let eg = t.encode_chunks_with("", 0).unwrap_or_default();
                while c.len() < n_chunks && !eg.is_empty() { c.push(eg[0].clone()); }
                c.truncate(n_chunks);
                if c.len() != n_chunks { neg_cl.clone() } else { c }
            }
            None => neg_cl.clone(),
        };
        let (neg_emb, neg_pooled) = d.forward_chunked(&neg_cl, &neg_cg)
            .map_err(|e| DiffusionError::Unsupported(format!("neg encode: {e}")))?;

        // Replicate embeddings: each batch element (uncond/cond) gets the
        // same text embedding repeated nf times (one per frame in that
        // batch-half). Shape: [2*nf, seq, cad].
        let single_emb_len = text_emb.len();
        let mut run = Vec::with_capacity(single_emb_len * 2 * nf);
        for _ in 0..nf { run.extend_from_slice(&neg_emb); }
        for _ in 0..nf { run.extend_from_slice(&text_emb); }
        // Pooled: just [neg_pooled, cond_pooled] for per-frame usage.
        let mut pr = Vec::with_capacity(neg_pooled.len() + pooled.len());
        pr.extend_from_slice(&neg_pooled);
        pr.extend_from_slice(&pooled);
        (run, pr, 2 * nf)
    } else {
        let mut run = Vec::with_capacity(text_emb.len() * nf);
        for _ in 0..nf { run.extend_from_slice(&text_emb); }
        (run, pooled.clone(), nf)
    };

    // 2. Scheduler.
    residents.scheduler.set_timesteps(req.steps);
    let total_steps = residents.scheduler.timesteps().len();

    // 3. Initial noise: [cfg_batch * nf, latent_c, latent_h, latent_w].
    // For CFG: first nf frames = uncond noise, second nf = cond noise.
    // Both halves use the same noise (CFG convention).
    let single_frame_floats = latent_c * latent_h * latent_w;
    let noise_frames: Vec<Vec<f32>> = (0..nf)
        .map(|f| deterministic_noise(single_frame_floats, req.seed.wrapping_add(f as u64)))
        .collect();
    let mut latent = Vec::with_capacity(n_unet * single_frame_floats);
    if do_cfg {
        // uncond half = same noise
        for f in &noise_frames { latent.extend_from_slice(f); }
        // cond half = same noise
        for f in &noise_frames { latent.extend_from_slice(f); }
    } else {
        for f in &noise_frames { latent.extend_from_slice(f); }
    }

    // 4. Precompute KV cache.
    // For the per-frame GPU path, the KV cache should match n_per_frame
    // (cfg_batch), NOT the full n=cfg_batch*nf. Build a per-frame text
    // slice: [uncond_text, cond_text] (or just [cond_text] without CFG).
    let text_seq = text_emb.len() / cad;
    let text_per_frame: Vec<f32> = if do_cfg {
        let neg_chunks_l2 = residents.tok_l.encode_chunks(&req.negative_prompt).unwrap_or_default();
        let mut ncl = neg_chunks_l2;
        let el = residents.tok_l.encode_chunks("").unwrap_or_default();
        while ncl.len() < n_chunks { ncl.push(el[0].clone()); }
        ncl.truncate(n_chunks);
        let ncg = match residents.tok_g.as_ref() {
            Some(t) => {
                let mut c = t.encode_chunks_with(&req.negative_prompt, 0).unwrap_or_else(|_| ncl.clone());
                let eg = t.encode_chunks_with("", 0).unwrap_or_default();
                while c.len() < n_chunks && !eg.is_empty() { c.push(eg[0].clone()); }
                c.truncate(n_chunks);
                if c.len() != n_chunks { ncl.clone() } else { c }
            }
            None => ncl.clone(),
        };
        let d2 = residents.dual.as_ref().unwrap();
        let (neg_emb, _) = d2.forward_chunked(&ncl, &ncg)
            .map_err(|e| DiffusionError::Unsupported(format!("neg: {e}")))?;
        let mut pf = Vec::with_capacity(neg_emb.len() + text_emb.len());
        pf.extend_from_slice(&neg_emb);
        pf.extend_from_slice(&text_emb);
        pf
    } else {
        text_emb.clone()
    };
    let cfg_batch = if do_cfg { 2 } else { 1 };
    // Build the full bf-sized text embedding for the CPU path (which
    // processes all frames simultaneously) AND the per-frame slice for
    // the GPU path (which processes one frame at a time).
    let text_emb_full: Vec<f32> = {
        let mut full = Vec::with_capacity(text_per_frame.len() * nf);
        // text_per_frame is [cfg_batch, text_seq, cad]. Replicate each
        // cfg-half nf times: [uncond × nf, cond × nf].
        let per_cfg = text_per_frame.len() / cfg_batch;
        for cb in 0..cfg_batch {
            for _ in 0..nf {
                full.extend_from_slice(&text_per_frame[cb * per_cfg..(cb + 1) * per_cfg]);
            }
        }
        full
    };
    let pooled_full: Vec<f32> = {
        let per_cfg = pooled_run.len() / cfg_batch;
        let mut full = Vec::with_capacity(pooled_run.len() * nf);
        for cb in 0..cfg_batch {
            for _ in 0..nf {
                full.extend_from_slice(&pooled_run[cb * per_cfg..(cb + 1) * per_cfg]);
            }
        }
        full
    };

    // KV cache: GPU unified path processes n = cfg_batch * nf samples
    // simultaneously, so precompute for the full batch size.
    #[cfg(feature = "metal")]
    if residents.gpu_residence {
        residents.unet.precompute_text_kv(&text_emb_full, text_seq);
        residents.unet.precompute_text_kv_gpu(&text_emb_full, text_seq);
    }
    #[cfg(feature = "metal")]
    if !residents.gpu_residence {
        residents.unet.precompute_text_kv(&text_emb_full, text_seq);
    }
    #[cfg(not(feature = "metal"))]
    {
        residents.unet.precompute_text_kv(&text_emb_full, text_seq);
    }

    // 5. Denoise loop.
    //
    // `latent` is maintained as [nf, latent_c, latent_h, latent_w] — the
    // cond-only working state. At each step we duplicate it for CFG
    // (building [uncond_nf, cond_nf] = [2*nf, ...]), run the UNet, CFG-
    // combine back to [nf, ...], and scheduler-step.
    //
    // Strip the initial latent down to cond-only (second half for CFG).
    if do_cfg {
        let half = nf * single_frame_floats;
        latent = latent[half..].to_vec(); // keep cond half only
    }

    let timesteps: Vec<i64> = residents.scheduler.timesteps().to_vec();
    for (step_idx, &t) in timesteps.iter().enumerate() {
        if abort.load(Ordering::SeqCst) {
            return Err(DiffusionError::Unsupported("aborted".into()));
        }
        let t_f = t as f32;

        // Duplicate latent for CFG: [uncond=latent, cond=latent].
        let latent_input = if do_cfg {
            let mut doubled = Vec::with_capacity(latent.len() * 2);
            doubled.extend_from_slice(&latent); // uncond half
            doubled.extend_from_slice(&latent); // cond half
            doubled
        } else {
            latent.clone()
        };

        let eps_batched = {
            #[cfg(feature = "metal")]
            {
                if residents.gpu_residence {
                    residents.unet.forward_video_sdxl_gpu(
                        &latent_input, t_f, &text_emb_full, &pooled_full,
                        crate::unet::SDXLAdditionalConditioning::default_1024(),
                        motion, nf,
                    )?
                } else {
                    residents.unet.forward_video_sdxl(
                        &latent_input, t_f, &text_emb_full, &pooled_full,
                        crate::unet::SDXLAdditionalConditioning::default_1024(),
                        motion, nf,
                    )?
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                residents.unet.forward_video_sdxl(
                    &latent_input, t_f, &text_emb_full, &pooled_full,
                    crate::unet::SDXLAdditionalConditioning::default_1024(),
                    motion, nf,
                )?
            }
        };

        // CFG combine: [uncond_nf | cond_nf] → [nf].
        let eps: Vec<f32> = if do_cfg {
            let half = nf * single_frame_floats;
            let (eps_u, eps_c) = eps_batched.split_at(half);
            eps_u.iter().zip(eps_c.iter())
                .map(|(u, c)| u + req.cfg_scale * (c - u))
                .collect()
        } else {
            eps_batched
        };

        // Scheduler step per frame.
        let mut new_latent = Vec::with_capacity(nf * single_frame_floats);
        for f in 0..nf {
            let off = f * single_frame_floats;
            let frame_eps = &eps[off..off + single_frame_floats];
            let frame_lat = &latent[off..off + single_frame_floats];
            let stepped = residents.scheduler.step(frame_eps, step_idx, frame_lat);
            new_latent.extend_from_slice(&stepped);
        }
        latent = new_latent;

        on_progress(GenerateProgress {
            step: step_idx + 1,
            total: total_steps + nf, // include VAE decode in total
            elapsed_s: t_start.elapsed().as_secs_f32(),
        });
    }

    // 6. VAE decode each frame. This is the second-heaviest phase
    //    (after the denoise loop). Report progress so the UI shows
    //    "decoding frame N/F" instead of hanging silently.
    let mut frames = Vec::with_capacity(nf);
    for f in 0..nf {
        if abort.load(Ordering::SeqCst) {
            return Err(DiffusionError::Unsupported("aborted".into()));
        }
        let off = f * single_frame_floats;
        let mut frame_lat = latent[off..off + single_frame_floats].to_vec();
        if vae_scale != 0.0 {
            for v in frame_lat.iter_mut() { *v /= vae_scale; }
        }
        let rgb = {
            #[cfg(feature = "metal")]
            {
                if residents.gpu_residence {
                    residents.vae.decode_with_dims_gpu(&frame_lat, 1, latent_h, latent_w)
                } else {
                    residents.vae.decode_with_dims(&frame_lat, 1, latent_h, latent_w)
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                residents.vae.decode_with_dims(&frame_lat, 1, latent_h, latent_w)
            }
        };
        frames.push(GeneratedImage {
            rgb_chw: rgb,
            height: img_h,
            width: img_w,
        });
        // Progress: denoise steps are 1..total_steps, VAE decode is
        // total_steps+1..total_steps+nf. The UI shows "step N / total"
        // which naturally extends past the denoise count.
        on_progress(GenerateProgress {
            step: total_steps + f + 1,
            total: total_steps + nf,
            elapsed_s: t_start.elapsed().as_secs_f32(),
        });
    }

    Ok(GeneratedVideo { frames, fps })
}

/// Same Gaussian PRNG `sd_generate` uses (SplitMix64 → Box-Muller). The
/// CLI binary has the same function in its own scope; sharing it here so
/// the library and the binary produce byte-identical noise from the same
/// seed.
fn deterministic_noise(n_floats: usize, seed: u64) -> Vec<f32> {
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

/// Manual deep clone of a `CLIPTextModel`. The internal layer types don't
/// implement `Clone` (intentional — they own large `Vec<f32>`s and we want
/// callers to be explicit about the copy). SDXL needs CLIP-L resident in
/// both `SdResidents.text_l` and `SdResidents.dual.unwrap().clip_l`; this
/// lifts the once-loaded CLIP-L into the dual without re-reading weights
/// from disk. Mirrors `text_l_clone` from the sd_generate binary.
pub(crate) fn clone_clip_text_model(src: &CLIPTextModel) -> CLIPTextModel {
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
