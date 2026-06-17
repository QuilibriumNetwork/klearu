//! Self-contained HiDream-O1 runner for embedding in the desktop app.
//!
//! Encapsulates everything needed to turn a prompt into an image: the Qwen3-VL
//! tokenizer + chat template, the 8B GPU model, scheduler selection (flash for
//! the distilled Dev path, UniPC + CFG for the full model), seeded noise, and a
//! progress/abort hook. The caller (desktop engine) only needs to detect a
//! HiDream-O1 directory, [`HiDreamRunner::load`] it once, and call
//! [`HiDreamRunner::generate`] per request.
//!
//! Gated behind `feature = "metal"` (8B inference needs the GPU).
#![cfg(feature = "metal")]

use std::path::Path;
use std::sync::atomic::AtomicBool;

use super::config::{HiDreamConfig, IN_CHANNELS, PATCH_SIZE, TMS_TOKEN_ID};
use super::gpu::GpuHiDreamModel;
use super::image::patchify;
use super::pipeline::{generate_t2i_cb, GenParams, InitImage};
use super::scheduler::{DenoiseScheduler, Schedule, SchedulerKind, UniPCScheduler, DEFAULT_TIMESTEPS};
use super::vision::{VisionConfig, VisionEncoder};
use crate::weight::loader::ComponentTensors;
use crate::{DiffusionError, Result};

/// True if `dir` is a HiDream-O1 model directory (config.json has
/// `model_type: qwen3_vl` and the generation patch-embed tensors via the index).
pub fn is_hidream_dir(dir: &Path) -> bool {
    let cfg_path = dir.join("config.json");
    let Ok(text) = std::fs::read_to_string(&cfg_path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    let is_qwen3vl = v.get("model_type").and_then(|m| m.as_str()) == Some("qwen3_vl");
    // The x_embedder/final_layer2 generation heads distinguish HiDream-O1 from a
    // plain Qwen3-VL VLM (confirm via the safetensors index if present).
    let has_gen_head = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
        .map(|idx| idx.contains("model.x_embedder") && idx.contains("model.final_layer2"))
        .unwrap_or(true); // if no index, trust model_type
    is_qwen3vl && has_gen_head
}

/// Options for one HiDream-O1 generation.
pub struct HiDreamGenOpts {
    pub prompt: String,
    pub negative_prompt: String,
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    /// CFG scale. `> 1` selects the full model's UniPC + CFG path; otherwise the
    /// distilled Dev flash path (guidance 0).
    pub guidance: f32,
    pub seed: u64,
    /// Optional input image (PNG/JPEG bytes) for img2img / inpaint.
    pub input_image_bytes: Option<Vec<u8>>,
    /// Optional grayscale mask (PNG bytes); white = regenerate. Requires
    /// `input_image_bytes`.
    pub mask_bytes: Option<Vec<u8>>,
    /// img2img denoise strength `0..=1` (only used when an input image is set).
    pub strength: f32,
    /// Optional reference images (PNG/JPEG bytes) for native reference editing.
    /// When non-empty the runner routes through the vision-encoder + DeepStack
    /// path instead of plain txt2img.
    pub ref_image_bytes: Vec<Vec<u8>>,
    /// Snap the output to the model's native inference resolution
    /// (`PREDEFINED_RESOLUTIONS`, all ≥ 2048²) by matching the requested aspect
    /// ratio — exactly what the reference does. The model is a 2048²-native
    /// model; generating below this is noticeably softer. Set `false` for fast,
    /// lower-quality previews at the requested size.
    pub snap_resolution: bool,
}

/// HiDream-O1's native inference resolutions (width, height). The reference
/// snaps every request to the closest of these by aspect ratio — the model is
/// trained/tuned for ≥2048² output.
pub const PREDEFINED_RESOLUTIONS: [(usize, usize); 11] = [
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
];

/// Snap `(width, height)` to the closest native resolution by aspect ratio.
pub fn find_closest_resolution(width: usize, height: usize) -> (usize, usize) {
    let img_ratio = width as f32 / height as f32;
    PREDEFINED_RESOLUTIONS
        .iter()
        .copied()
        .min_by(|&(w1, h1), &(w2, h2)| {
            let d1 = (w1 as f32 / h1 as f32 - img_ratio).abs();
            let d2 = (w2 as f32 / h2 as f32 - img_ratio).abs();
            d1.total_cmp(&d2)
        })
        .unwrap()
}

impl HiDreamGenOpts {
    /// txt2img defaults (no input image).
    pub fn txt2img(prompt: String, height: usize, width: usize, steps: usize, guidance: f32, seed: u64) -> Self {
        Self {
            prompt,
            negative_prompt: String::new(),
            height,
            width,
            steps,
            guidance,
            seed,
            input_image_bytes: None,
            mask_bytes: None,
            strength: 1.0,
            ref_image_bytes: Vec::new(),
            snap_resolution: true,
        }
    }
}

/// A rendered image in planar `[C, H, W]` f32, normalized to `[-1, 1]`
/// (`image_io::save_png` maps this to u8).
pub struct HiDreamImage {
    pub chw: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

/// A loaded HiDream-O1 model + tokenizer, ready to render many images.
pub struct HiDreamRunner {
    model: GpuHiDreamModel,
    tokenizer: tokenizers::Tokenizer,
    cfg: HiDreamConfig,
    /// Qwen3-VL vision encoder for reference editing. Loaded lazily on the first
    /// reference-image request (most renders are txt2img and don't need it).
    vision: std::sync::Mutex<Option<std::sync::Arc<VisionEncoder>>>,
    dir: std::path::PathBuf,
}

impl HiDreamRunner {
    /// Load from a HiDream-O1 directory (config.json + sharded safetensors +
    /// tokenizer.json). Converts the 8B weights to GPU bf16 buffers.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg_text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| DiffusionError::Layout(format!("hidream config.json: {e}")))?;
        let cfg = HiDreamConfig::from_json(&cfg_text)?;
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| DiffusionError::Unsupported(format!("hidream tokenizer: {e}")))?;
        let comp = ComponentTensors::open_dir(dir)?;
        let model = GpuHiDreamModel::load(cfg.clone(), &comp)?;
        Ok(Self {
            model,
            tokenizer,
            cfg,
            vision: std::sync::Mutex::new(None),
            dir: dir.to_path_buf(),
        })
    }

    /// Load (and cache) the vision encoder. The 27-layer ViT lives under
    /// `model.visual.*` in the same safetensors shards.
    fn vision_encoder(&self) -> Result<std::sync::Arc<VisionEncoder>> {
        let mut guard = self.vision.lock().unwrap();
        if let Some(v) = guard.as_ref() {
            return Ok(v.clone());
        }
        let vcfg = parse_vision_config(&self.dir)?;
        let mut enc = VisionEncoder::new(vcfg);
        let comp = ComponentTensors::open_dir(&self.dir)?;
        enc.load(&comp, "model.visual")?;
        let rc = std::sync::Arc::new(enc);
        *guard = Some(rc.clone());
        Ok(rc)
    }

    /// Build the Qwen3-VL prompt token sequence:
    /// `<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<|boi|><|tms|>`.
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        let s = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|>"
        );
        let enc = self
            .tokenizer
            .encode(s, false)
            .map_err(|e| DiffusionError::Unsupported(format!("hidream encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Build the reference-editing prompt token sequence. Mirrors the reference
    /// `processor.apply_chat_template` + processor image-pad expansion: each
    /// reference image becomes `<|vision_start|> + image_token×merged + <|vision_end|>`
    /// (the merged token count = grid_h·grid_w / merge²), spliced numerically so
    /// we don't depend on the exact `<|image_pad|>` string. `merged_counts` is
    /// per reference image, in order.
    ///
    /// `<|im_start|>user\n {refs…} {prompt} {pad} <|im_end|>\n<|im_start|>assistant\n<|boi|><|tms|>`
    ///
    /// `min_len` pads the user-text region (newline filler before the
    /// assistant/boi/tms suffix) so the returned sequence is at least `min_len`
    /// tokens. Used to make the cond and uncond forwards of one CFG run share a
    /// sequence length — that keeps the GPU buffer pool warm (forwards of
    /// differing length force fresh allocations every step, which both stalls
    /// the pipeline and risks cross-size buffer-reuse corruption). Pass 0 for no
    /// padding (e.g. to measure the natural length first).
    #[cfg_attr(not(test), allow(dead_code))]
    fn tokenize_ref(&self, prompt: &str, merged_counts: &[usize], min_len: usize) -> Result<Vec<u32>> {
        let enc = |s: &str| -> Result<Vec<u32>> {
            self.tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| DiffusionError::Unsupported(format!("hidream encode: {e}")))
        };
        let image_token = self.cfg.image_token_id;
        let vstart = enc("<|vision_start|>")?;
        let vend = enc("<|vision_end|>")?;
        let suffix = enc("<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|>")?;
        let mut ids = enc("<|im_start|>user\n")?;
        for &count in merged_counts {
            ids.extend_from_slice(&vstart);
            ids.extend(std::iter::repeat(image_token).take(count));
            ids.extend_from_slice(&vend);
        }
        ids.extend(enc(prompt)?);
        // Pad the pre-suffix (user-text) region up to `min_len - suffix.len()`
        // with a neutral newline token. Newline is benign text context (never
        // the image_token), so it doesn't disturb the vision/gen structure.
        let target_pre = min_len.saturating_sub(suffix.len());
        if ids.len() < target_pre {
            let nl = enc("\n")?;
            let pad_tok = nl.first().copied().unwrap_or(0);
            ids.resize(target_pre, pad_tok);
        }
        ids.extend_from_slice(&suffix);
        Ok(ids)
    }

    /// Render an image. `on_step(step, total)` reports progress; `abort` stops
    /// the denoise loop early (the partial result is returned).
    pub fn generate(
        &self,
        opts: &HiDreamGenOpts,
        on_step: impl FnMut(usize, usize),
        abort: &AtomicBool,
    ) -> Result<HiDreamImage> {
        // Reference editing routes through the vision-encoder + DeepStack path.
        #[cfg(feature = "image-io")]
        if !opts.ref_image_bytes.is_empty() {
            return self.generate_ref(opts, on_step, abort);
        }
        // Resolution. The model is 2048²-native: by default snap to the closest
        // trained resolution by aspect ratio (what the reference does) for sharp
        // output; otherwise just patch-align the requested size (fast preview).
        let (w, h) = if opts.snap_resolution {
            find_closest_resolution(opts.width, opts.height)
        } else {
            (
                (opts.width / PATCH_SIZE).max(1) * PATCH_SIZE,
                (opts.height / PATCH_SIZE).max(1) * PATCH_SIZE,
            )
        };

        let use_cfg = opts.guidance > 1.0;
        let cond = self.tokenize(&opts.prompt)?;
        let uncond = if use_cfg {
            let neg = if opts.negative_prompt.trim().is_empty() { " " } else { &opts.negative_prompt };
            Some(self.tokenize(neg)?)
        } else {
            None
        };

        // Scheduler: UniPC + CFG for the full model (guidance > 1), else the
        // distilled flash path (28-step DEFAULT_TIMESTEPS, guidance 0).
        let mut scheduler: Box<dyn DenoiseScheduler> = if use_cfg {
            let steps = if opts.steps == 0 { 50 } else { opts.steps };
            Box::new(UniPCScheduler::new(steps, 3.0))
        } else {
            Box::new(Schedule::from_timesteps(SchedulerKind::Flash, &DEFAULT_TIMESTEPS))
        };

        let mut params = GenParams::dev(h, w);
        params.guidance_scale = if use_cfg { opts.guidance } else { 0.0 };

        // img2img / inpaint: decode the input (and mask) into patch space.
        let init = self.build_init(opts, h, w)?;

        let mut rng = SeededNormal::new(opts.seed.wrapping_add(1));
        let noise_fn = move |len: usize| rng.normals(len);

        let chw = generate_t2i_cb(
            &self.model,
            &cond,
            uncond.as_deref(),
            &params,
            scheduler.as_mut(),
            init.as_ref(),
            noise_fn,
            on_step,
            abort,
        );
        Ok(HiDreamImage { chw, height: h, width: w })
    }

    /// Decode the optional input image + mask into an [`InitImage`] (patch
    /// space, `[-1,1]`). Returns `None` for txt2img.
    #[cfg(feature = "image-io")]
    fn build_init(&self, opts: &HiDreamGenOpts, h: usize, w: usize) -> Result<Option<InitImage>> {
        let Some(bytes) = &opts.input_image_bytes else {
            return Ok(None);
        };
        let chw = decode_resize_chw(bytes, h, w)?; // [C,H,W] in [-1,1]
        let patches = patchify(&chw, h, w);
        let mask = match &opts.mask_bytes {
            Some(mb) => Some(decode_resize_mask(mb, h, w)?),
            None => None,
        };
        Ok(Some(InitImage { patches, strength: opts.strength.clamp(0.0, 1.0), mask }))
    }

    #[cfg(not(feature = "image-io"))]
    fn build_init(&self, _opts: &HiDreamGenOpts, _h: usize, _w: usize) -> Result<Option<InitImage>> {
        Ok(None)
    }

    /// Reference-image editing. Runs the vision encoder + image processor once
    /// to build the conditioning, then denoises the target with `forward_ref`.
    #[cfg(feature = "image-io")]
    fn generate_ref(
        &self,
        opts: &HiDreamGenOpts,
        on_step: impl FnMut(usize, usize),
        abort: &AtomicBool,
    ) -> Result<HiDreamImage> {
        use super::pipeline::{generate_ref_cb, RefContext};

        // Start the run with a fresh GPU buffer pool so we never recycle buffers
        // left by a prior generation of a different sequence length (cross-size
        // reuse corrupts trailing tokens). Done ONCE per run, not per forward —
        // within a run cond/uncond share a length (see `pad_to`), so the pool
        // stays warm across all denoise steps. (Per-forward draining tanks
        // throughput: every step re-allocates the large attention buffers.)
        crate::metal_backend::drain_pool();

        let (w, h) = if opts.snap_resolution {
            find_closest_resolution(opts.width, opts.height)
        } else {
            ((opts.width / PATCH_SIZE).max(1) * PATCH_SIZE, (opts.height / PATCH_SIZE).max(1) * PATCH_SIZE)
        };
        let (hp, wp) = (h / PATCH_SIZE, w / PATCH_SIZE);
        let tgt_len = hp * wp;
        let k = opts.ref_image_bytes.len();

        // K-dependent reference budgets (reference pipeline.py).
        let max_size = match k {
            1 => h.max(w),
            2 => h.max(w) * 48 / 64,
            3..=4 => h.max(w) / 2,
            5..=8 => h.max(w) * 24 / 64,
            _ => h.max(w) / 4,
        };
        let cond_img_size: usize = if k <= 4 { 384 } else if k <= 8 { 384 * 48 / 64 } else { 384 / 2 };
        let merge = self.cfg_vision_merge()?;

        let enc = self.vision_encoder()?;
        let hidden = self.cfg.text_config.hidden_size;
        let n_deep = enc_deepstack_count(&enc);

        // Per-reference: clean patches (for vinputs) + VLM pixel_values (encoder).
        let mut vision_embeds = Vec::new();
        let mut deepstack: Vec<Vec<f32>> = vec![Vec::new(); n_deep];
        let mut ref_patches = Vec::new();
        let mut merged_counts = Vec::new();
        let mut grids_vlm = Vec::new();
        let mut grids_ref = Vec::new();
        for bytes in &opts.ref_image_bytes {
            let (chw01, ih, iw) = decode_chw01(bytes)?;

            // Clean reference patches: area ≤ max_size², 32-aligned, [-1,1].
            let (rw, rh) = area_aligned(iw, ih, max_size, PATCH_SIZE);
            let mut clean = resize_chw01_to(&chw01, ih, iw, rh, rw);
            for v in clean.iter_mut() {
                *v = *v * 2.0 - 1.0;
            }
            ref_patches.extend(patchify(&clean, rh, rw));
            grids_ref.push((1usize, rh / PATCH_SIZE, rw / PATCH_SIZE));

            // VLM input: shrink to ~cond_img_size, then the Qwen processor.
            let (cw, ch) = calculate_dimensions(cond_img_size, iw as f32 / ih as f32);
            let vlm01 = resize_chw01_to(&chw01, ih, iw, ch, cw);
            let (pixel_values, (gt, gh, gw)) =
                super::vlm_image::preprocess_vlm_image(&vlm01, 3, ch, cw);
            let (emb, ds) = enc.forward(&pixel_values, (gt, gh, gw));
            debug_assert_eq!(emb.len() % hidden, 0);
            vision_embeds.extend(emb);
            for (l, d) in ds.into_iter().enumerate() {
                deepstack[l].extend(d);
            }
            merged_counts.push((gh / merge) * (gw / merge));
            grids_vlm.push((gt, gh / merge, gw / merge));
        }

        let neg = if opts.negative_prompt.trim().is_empty() { " " } else { &opts.negative_prompt };
        // Common sequence length for both CFG branches so every forward in this
        // run is the same size — keeps the GPU buffer pool warm (differing
        // lengths force fresh large-buffer allocations each step, stalling the
        // pipeline, and risk cross-size buffer-reuse corruption).
        let pad_to = self
            .tokenize_ref(&opts.prompt, &merged_counts, 0)?
            .len()
            .max(self.tokenize_ref(neg, &merged_counts, 0)?.len());

        let build_ctx = |prompt: &str| -> Result<RefContext> {
            let text_ids = self.tokenize_ref(prompt, &merged_counts, pad_to)?;
            // Padded sequence for position computation: text + target + ref runs.
            // Each skip=1 (target / pixel-ref) block begins with a vision_start
            // marker that `build_position_ids_multi` absorbs into the grid (the
            // reference's `vt[0] = vision_start`); the block is then
            // `[vision_start, image_token×(n-1)]` for an n-token grid.
            let img = self.cfg.image_token_id;
            let vstart = self.cfg.vision_start_token_id;
            let mut padded = text_ids.clone();
            let mut push_gen_block = |padded: &mut Vec<u32>, n: usize| {
                padded.push(vstart);
                padded.extend(std::iter::repeat(img).take(n.saturating_sub(1)));
            };
            push_gen_block(&mut padded, tgt_len);
            for g in &grids_ref {
                push_gen_block(&mut padded, g.0 * g.1 * g.2);
            }
            let mut grids: Vec<(usize, usize, usize)> = grids_vlm.clone();
            grids.push((1, hp, wp));
            grids.extend(grids_ref.iter().copied());
            let mut skip = vec![0usize; k];
            skip.push(1); // target
            skip.extend(std::iter::repeat(1usize).take(k)); // pixel refs
            let positions = super::model::build_position_ids_multi(
                &padded, &grids, &skip, img, super::model::ROPE_FIX_POINT,
            );
            Ok(RefContext {
                text_ids,
                vision_embeds: vision_embeds.clone(),
                deepstack: deepstack.clone(),
                ref_patches: ref_patches.clone(),
                positions,
                tgt_len,
                image_token_id: img,
                tms_token_id: TMS_TOKEN_ID,
            })
        };

        if std::env::var("HIDREAM_DEBUG").is_ok() {
            super::pipeline::stats("vision_embeds", &vision_embeds);
            for (l, d) in deepstack.iter().enumerate() {
                super::pipeline::stats(&format!("deepstack[{l}]"), d);
            }
            super::pipeline::stats("ref_patches", &ref_patches);
            eprintln!(
                "[ref] k={k} tgt_len={tgt_len} merged_counts={merged_counts:?} grids_vlm={grids_vlm:?} grids_ref={grids_ref:?} vlm_tokens={} hidden={hidden}",
                vision_embeds.len() / hidden,
            );
        }

        // Reference editing only manifests on the CFG path — the distilled flash
        // path (guidance ≤ 1) denoises to a flat gray (the conditioning needs the
        // cond/uncond contrast to express the edit). Force a guidance floor so
        // editing works out of the box.
        let guidance = if opts.guidance > 1.0 { opts.guidance } else { 4.0 };
        if opts.guidance <= 1.0 {
            eprintln!("[ref] editing requires CFG; using guidance {guidance} (flash path produces flat output)");
        }
        let cond = build_ctx(&opts.prompt)?;
        if std::env::var("HIDREAM_DEBUG").is_ok() {
            eprintln!(
                "[ref] text_ids.len={} seq={} positions.len/3={}",
                cond.text_ids.len(),
                cond.text_ids.len() + tgt_len + ref_patches.len() / super::config::PATCH_DIM,
                cond.positions.len() / 3,
            );
        }
        let uncond = Some(build_ctx(neg)?);

        let steps = if opts.steps == 0 { 50 } else { opts.steps };
        let mut scheduler: Box<dyn DenoiseScheduler> = Box::new(UniPCScheduler::new(steps, 3.0));
        let mut params = GenParams::dev(h, w);
        params.guidance_scale = guidance;

        let mut rng = SeededNormal::new(opts.seed.wrapping_add(1));
        let noise_fn = move |len: usize| rng.normals(len);

        let chw = generate_ref_cb(
            &self.model,
            &cond,
            uncond.as_ref(),
            &params,
            scheduler.as_mut(),
            noise_fn,
            on_step,
            abort,
        );
        Ok(HiDreamImage { chw, height: h, width: w })
    }

    fn cfg_vision_merge(&self) -> Result<usize> {
        Ok(parse_vision_config(&self.dir)?.spatial_merge_size)
    }
}

/// Parse the `vision_config` sub-object of a HiDream-O1 `config.json`.
fn parse_vision_config(dir: &Path) -> Result<VisionConfig> {
    let text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| DiffusionError::Layout(format!("hidream config.json: {e}")))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| DiffusionError::Layout(format!("hidream config.json: {e}")))?;
    let vc = v.get("vision_config").ok_or_else(|| {
        DiffusionError::Layout("hidream config.json missing vision_config".into())
    })?;
    let g = |k: &str| -> Result<u64> {
        vc.get(k)
            .and_then(|x| x.as_u64())
            .ok_or_else(|| DiffusionError::Layout(format!("vision_config.{k}")))
    };
    let deep: Vec<usize> = vc
        .get("deepstack_visual_indexes")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_else(|| vec![8, 16, 24]);
    Ok(VisionConfig {
        depth: g("depth")? as usize,
        hidden_size: g("hidden_size")? as usize,
        num_heads: g("num_heads")? as usize,
        intermediate_size: g("intermediate_size")? as usize,
        patch_size: g("patch_size")? as usize,
        temporal_patch_size: g("temporal_patch_size")? as usize,
        spatial_merge_size: g("spatial_merge_size")? as usize,
        in_channels: g("in_channels")? as usize,
        out_hidden_size: g("out_hidden_size")? as usize,
        num_position_embeddings: g("num_position_embeddings")? as usize,
        deepstack_visual_indexes: deep,
    })
}

#[cfg(feature = "image-io")]
fn enc_deepstack_count(enc: &VisionEncoder) -> usize {
    enc.cfg.deepstack_visual_indexes.len()
}

/// Decode image bytes into planar `[C, H, W]` f32 in `[0, 1]` plus `(h, w)`.
#[cfg(feature = "image-io")]
fn decode_chw01(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize)> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| DiffusionError::Unsupported(format!("decode ref image: {e}")))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = vec![0.0_f32; IN_CHANNELS * h * w];
    for hi in 0..h {
        for wi in 0..w {
            let px = img.get_pixel(wi as u32, hi as u32);
            for c in 0..IN_CHANNELS {
                out[c * h * w + hi * w + wi] = px[c] as f32 / 255.0;
            }
        }
    }
    Ok((out, h, w))
}

/// Bilinear-resize a CHW `[0,1]` image to `out_h × out_w`.
#[cfg(feature = "image-io")]
fn resize_chw01_to(src: &[f32], h: usize, w: usize, out_h: usize, out_w: usize) -> Vec<f32> {
    super::vlm_image::resize_chw_bilinear(src, IN_CHANNELS, h, w, out_h, out_w)
}

/// Largest `(w, h)` with `w·h ≤ side²`, both multiples of `align`, aspect of
/// `(in_w, in_h)`. Approximates the reference `resize_pilimage` sizing.
#[cfg(feature = "image-io")]
fn area_aligned(in_w: usize, in_h: usize, side: usize, align: usize) -> (usize, usize) {
    let scale = ((side as f64 * side as f64) / (in_w as f64 * in_h as f64)).sqrt();
    let mut nw = ((in_w as f64 * scale) as usize / align) * align;
    let mut nh = ((in_h as f64 * scale) as usize / align) * align;
    nw = nw.max(align);
    nh = nh.max(align);
    (nw, nh)
}

/// Reference `calculate_dimensions`: `(w, h)` with area ≈ `max_size²`, aspect
/// `ratio = w/h`, both floored to multiples of 32.
#[cfg(feature = "image-io")]
fn calculate_dimensions(max_size: usize, ratio: f32) -> (usize, usize) {
    let m = max_size as f32;
    let width = (m * m * ratio).sqrt();
    let height = width / ratio;
    let w = ((width as usize) / 32) * 32;
    let h = ((height as usize) / 32) * 32;
    (w.max(32), h.max(32))
}

/// Decode image bytes, resize to `h×w`, return planar `[C,H,W]` in `[-1,1]`.
#[cfg(feature = "image-io")]
fn decode_resize_chw(bytes: &[u8], h: usize, w: usize) -> Result<Vec<f32>> {
    use image::imageops::FilterType;
    let img = image::load_from_memory(bytes)
        .map_err(|e| DiffusionError::Unsupported(format!("decode input image: {e}")))?
        .resize_exact(w as u32, h as u32, FilterType::Lanczos3)
        .to_rgb8();
    let mut out = vec![0.0_f32; IN_CHANNELS * h * w];
    for hi in 0..h {
        for wi in 0..w {
            let px = img.get_pixel(wi as u32, hi as u32);
            for c in 0..IN_CHANNELS {
                out[c * h * w + hi * w + wi] = (px[c] as f32) / 127.5 - 1.0;
            }
        }
    }
    Ok(out)
}

/// Decode a grayscale mask, resize to `h×w`, and return a patch-layout mask
/// `[image_len, PATCH_DIM]` where 1 = regenerate (white) and 0 = keep (black),
/// broadcast across the 3 channels.
#[cfg(feature = "image-io")]
fn decode_resize_mask(bytes: &[u8], h: usize, w: usize) -> Result<Vec<f32>> {
    use image::imageops::FilterType;
    let img = image::load_from_memory(bytes)
        .map_err(|e| DiffusionError::Unsupported(format!("decode mask: {e}")))?
        .resize_exact(w as u32, h as u32, FilterType::Triangle)
        .to_luma8();
    // Broadcast the grayscale regenerate-weight to [C,H,W], then patchify.
    let mut chw = vec![0.0_f32; IN_CHANNELS * h * w];
    for hi in 0..h {
        for wi in 0..w {
            let m = img.get_pixel(wi as u32, hi as u32)[0] as f32 / 255.0;
            for c in 0..IN_CHANNELS {
                chw[c * h * w + hi * w + wi] = m;
            }
        }
    }
    Ok(patchify(&chw, h, w))
}

/// Seeded standard-normal generator (Box-Muller over SplitMix64).
struct SeededNormal {
    state: u64,
}

impl SeededNormal {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f64 + 1.0) as f32 / (1u64 << 53) as f32
    }
    fn normals(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let u1 = self.unit();
            let u2 = self.unit();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            out.push(r * theta.cos());
            if out.len() < n {
                out.push(r * theta.sin());
            }
        }
        out
    }
}
