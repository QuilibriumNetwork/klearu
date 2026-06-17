//! CLIP text encoder, loaded for image-feedback Welford updates.
//!
//! The desktop shell needs to encode prompts in the same conditioning
//! space sd_generate uses (`cross_attention_dim`: 768 for SD 1.5, 2048 for
//! SDXL after concatenating CLIP-L 768 + CLIP-G 1280). Mirroring
//! sd_generate's loading sequence exactly — `SingleFileLoader::open` →
//! `CheckpointConfig::*_default` → build `CLIPTextModel` from config →
//! `load_from(component)` — keeps the engine wire-compatible with whatever
//! checkpoint the user pointed at in settings.
//!
//! For feedback we average the per-token output to a single `[cad]`-d
//! vector and hand it to the Welford state. That vector is then written
//! out (scaled by α) as the `--cond-bias` sidecar sd_generate consumes.

use std::path::{Path, PathBuf};

use klearu_diffusion::config::{CheckpointConfig, SDVariant};
use klearu_diffusion::text_encoder::{CLIPTextModel, SDXLDualTextEncoder};
use klearu_diffusion::tokenizer::CLIPTokenizer;
use klearu_diffusion::weight::{SDFormat, SingleFileLoader};
use serde::{Deserialize, Serialize};

use crate::error::{DesktopError, Result};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ClipFormat { Sd15, Sdxl }

impl ClipFormat {
    /// Per-format namespace for the user-vector row name.
    pub fn user_vector_key(self) -> &'static str {
        match self { Self::Sd15 => "image_clip_sd15", Self::Sdxl => "image_clip_sdxl" }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipStatus {
    pub loaded: bool,
    pub format: Option<ClipFormat>,
    pub cad: Option<usize>,
    pub checkpoint: Option<String>,
}

struct LoadedSd15 {
    text_l: CLIPTextModel,
    tok_l: CLIPTokenizer,
}

struct LoadedSdxl {
    dual: SDXLDualTextEncoder,
    tok_l: CLIPTokenizer,
    tok_g: CLIPTokenizer,
}

enum LoadedKind {
    Sd15(LoadedSd15),
    Sdxl(LoadedSdxl),
}

struct Loaded {
    kind: LoadedKind,
    format: ClipFormat,
    cad: usize,
    checkpoint: PathBuf,
}

#[derive(Default)]
pub struct ClipEngine {
    loaded: Option<Loaded>,
}

impl ClipEngine {
    pub fn new() -> Self { Self::default() }

    pub fn status(&self) -> ClipStatus {
        match &self.loaded {
            Some(l) => ClipStatus {
                loaded: true,
                format: Some(l.format),
                cad: Some(l.cad),
                checkpoint: Some(l.checkpoint.to_string_lossy().into_owned()),
            },
            None => ClipStatus { loaded: false, format: None, cad: None, checkpoint: None },
        }
    }

    pub fn loaded(&self) -> bool { self.loaded.is_some() }
    pub fn cad(&self) -> Option<usize> { self.loaded.as_ref().map(|l| l.cad) }
    pub fn format(&self) -> Option<ClipFormat> { self.loaded.as_ref().map(|l| l.format) }
    pub fn unload(&mut self) { self.loaded = None; }

    /// Load CLIP encoder(s) from a single-file safetensors checkpoint.
    /// `tok_l_path` is required; `tok_g_path` is only needed for SDXL.
    pub fn load(
        &mut self,
        checkpoint: &Path,
        tok_l_path: &Path,
        tok_g_path: Option<&Path>,
    ) -> Result<ClipStatus> {
        tracing::info!(?checkpoint, "loading CLIP encoders");
        let loader = SingleFileLoader::open(checkpoint)
            .map_err(|e| DesktopError::engine(format!("open checkpoint: {e}")))?;
        let fake_root = checkpoint.parent().unwrap_or(Path::new(".")).to_path_buf();
        // Use loader.variant (metadata + tensor-name heuristic) rather than
        // loader.metadata.architecture_variant() (metadata string only) —
        // checkpoints lacking architecture metadata still get detected via
        // the tensor-name fallback inside SingleFileLoader::open.
        let (cfg, format, variant) = match loader.variant {
            SDFormat::Sd15 => (CheckpointConfig::sd15_default(fake_root), ClipFormat::Sd15, SDVariant::Sd15),
            SDFormat::Sdxl => (CheckpointConfig::sdxl_default(fake_root), ClipFormat::Sdxl, SDVariant::Sdxl),
            other => {
                return Err(DesktopError::bad_input(format!(
                    "checkpoint format {other:?} not supported for CLIP-space feedback (need SD 1.5 or SDXL)"
                )));
            }
        };

        let tok_l = CLIPTokenizer::from_file(tok_l_path)
            .map_err(|e| DesktopError::engine(format!("load CLIP-L tokenizer: {e}")))?;

        let mut text_l = CLIPTextModel::from_config(cfg.text_encoder.clone());
        let comp_l = loader
            .component("text_encoder")
            .map_err(|e| DesktopError::engine(format!("checkpoint text_encoder: {e}")))?;
        text_l
            .load_from(&comp_l)
            .map_err(|e| DesktopError::engine(format!("load CLIP-L weights: {e}")))?;

        let kind = match variant {
            SDVariant::Sd15 => LoadedKind::Sd15(LoadedSd15 { text_l, tok_l }),
            SDVariant::Unknown => {
                return Err(DesktopError::bad_input(
                    "checkpoint architecture is `unknown` — pick a known SD 1.5 or SDXL .safetensors",
                ));
            }
            SDVariant::Sdxl => {
                let cfg_g = cfg.text_encoder_2.clone().ok_or_else(|| {
                    DesktopError::engine("SDXL checkpoint missing text_encoder_2 default")
                })?;
                let tok_g_path = tok_g_path.ok_or_else(|| {
                    DesktopError::bad_input("SDXL needs `image.tokenizer_g` (CLIP-G tokenizer.json)")
                })?;
                let tok_g = CLIPTokenizer::from_file(tok_g_path)
                    .map_err(|e| DesktopError::engine(format!("load CLIP-G tokenizer: {e}")))?;

                let mut dual = SDXLDualTextEncoder::from_configs(cfg.text_encoder.clone(), cfg_g);
                // Reuse the just-loaded CLIP-L instead of re-loading it.
                dual.clip_l = text_l;
                let comp_g = loader
                    .component("text_encoder_2")
                    .map_err(|e| DesktopError::engine(format!("checkpoint text_encoder_2: {e}")))?;
                dual.clip_g
                    .load_from(&comp_g)
                    .map_err(|e| DesktopError::engine(format!("load CLIP-G weights: {e}")))?;
                LoadedKind::Sdxl(LoadedSdxl { dual, tok_l, tok_g })
            }
        };

        let cad = cfg.unet.cross_attention_dim;
        self.loaded = Some(Loaded {
            kind,
            format,
            cad,
            checkpoint: checkpoint.to_path_buf(),
        });
        Ok(self.status())
    }

    /// Encode `text` and return a single `[cad]`-d vector — the per-token
    /// mean of the CLIP output. Used as a Welford sample per feedback event.
    /// Long prompts truncate to one chunk (77 tokens) since the running mean
    /// makes exact length unimportant.
    pub fn encode_mean(&self, text: &str) -> Result<Vec<f32>> {
        let Some(loaded) = self.loaded.as_ref() else {
            return Err(DesktopError::NotLoaded("clip"));
        };
        let emb = match &loaded.kind {
            LoadedKind::Sd15(l) => {
                let chunks = l
                    .tok_l
                    .encode_chunks(text)
                    .map_err(|e| DesktopError::engine(format!("CLIP-L encode: {e}")))?;
                let first = chunks.into_iter().next().ok_or_else(|| {
                    DesktopError::engine("CLIP-L tokenizer returned zero chunks")
                })?;
                l.text_l
                    .forward_chunked(&[first], None)
                    .map_err(|e| DesktopError::engine(format!("CLIP-L forward: {e}")))?
            }
            LoadedKind::Sdxl(s) => {
                let chunks_l = s
                    .tok_l
                    .encode_chunks(text)
                    .map_err(|e| DesktopError::engine(format!("CLIP-L encode: {e}")))?;
                let chunks_g = s
                    .tok_g
                    .encode_chunks_with(text, 0) // CLIP-G pad token is 0
                    .map_err(|e| DesktopError::engine(format!("CLIP-G encode: {e}")))?;
                let first_l = chunks_l.into_iter().next().ok_or_else(|| {
                    DesktopError::engine("CLIP-L tokenizer returned zero chunks")
                })?;
                let first_g = chunks_g.into_iter().next().ok_or_else(|| {
                    DesktopError::engine("CLIP-G tokenizer returned zero chunks")
                })?;
                let (emb, _pooled) = s
                    .dual
                    .forward_chunked(&[first_l], &[first_g])
                    .map_err(|e| DesktopError::engine(format!("SDXL dual encode: {e}")))?;
                emb
            }
        };

        let cad = loaded.cad;
        if emb.len() % cad != 0 {
            return Err(DesktopError::engine(format!(
                "CLIP output {} not a multiple of cad {}", emb.len(), cad
            )));
        }
        let positions = emb.len() / cad;
        let mut mean = vec![0.0f32; cad];
        for i in 0..positions {
            let row = &emb[i * cad..(i + 1) * cad];
            for (m, v) in mean.iter_mut().zip(row) { *m += *v; }
        }
        if positions > 0 {
            let inv = 1.0 / positions as f32;
            for m in mean.iter_mut() { *m *= inv; }
        }
        Ok(mean)
    }
}
