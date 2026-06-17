//! HiDream-O1-Image configuration.
//!
//! HiDream-O1-Image is a pixel-native flow-matching diffusion model built on a
//! Qwen3-VL-8B-Instruct transformer backbone. It ships as a HuggingFace
//! directory whose `config.json` carries the standard Qwen3-VL nested layout:
//!
//!   {
//!     "model_type": "qwen3_vl",
//!     "image_token_id": 151655,
//!     "video_token_id": 151656,
//!     "vision_start_token_id": 151652,
//!     "text_config": { "hidden_size": 4096, "num_hidden_layers": 36, ... },
//!     "vision_config": { ... }       // only needed for reference-image editing
//!   }
//!
//! We parse the fields the generation forward pass needs and tolerate unknown
//! ones. The image-generation–specific hyperparameters (patch size, the
//! `<tms>` timestep-token id) are not in the HF config and are pinned as
//! constants below, matching the reference `models/qwen3_vl_transformers.py`.

use serde::Deserialize;

/// Generation patch size — each image token is a `PATCH_SIZE × PATCH_SIZE × 3`
/// raw-pixel patch. NOTE: this is the *generation* patch size (32), distinct
/// from the vision encoder's patch size (16) used only for ref-image editing.
pub const PATCH_SIZE: usize = 32;

/// Channels of the generated image (RGB).
pub const IN_CHANNELS: usize = 3;

/// Flattened patch dimension fed to the patch embedder: 32·32·3 = 3072.
pub const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS;

/// Token id whose embedding is overwritten by the timestep embedding.
/// Pinned in the reference (`self.tms_token_id = 151673`).
pub const TMS_TOKEN_ID: u32 = 151673;

/// Timestep sinusoidal frequency embedding size (before the MLP).
pub const FREQ_EMBED_SIZE: usize = 256;

/// Text-backbone (Qwen3-VL language model) hyperparameters needed by the
/// decoder. Parsed from `text_config` in the model's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Whether q/k/v/o projections carry a bias (Qwen3-VL: false).
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    /// Interleaved mRoPE section split over half the head_dim, e.g. [24,20,20].
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_rms_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    // Qwen3-VL default; overridden by config.json when present.
    5_000_000.0
}
fn default_max_position() -> usize {
    262_144
}

impl TextConfig {
    /// Resolve head_dim, falling back to hidden_size / num_heads when the
    /// config omits it (Qwen3-VL specifies head_dim=128 explicitly).
    pub fn head_dim(&self) -> usize {
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }

    /// mRoPE section, defaulting to [24,20,20] (Qwen3-VL) — sums to head_dim/2.
    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .and_then(|r| r.mrope_section.clone())
            .unwrap_or_else(|| vec![24, 20, 20])
    }

    /// Grouped-query repeat factor (query heads per KV head).
    pub fn kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Full HiDream-O1 config: the top-level token ids plus the text backbone.
#[derive(Debug, Clone, Deserialize)]
pub struct HiDreamConfig {
    #[serde(default = "default_image_token")]
    pub image_token_id: u32,
    #[serde(default = "default_video_token")]
    pub video_token_id: u32,
    #[serde(default = "default_vision_start")]
    pub vision_start_token_id: u32,
    pub text_config: TextConfig,
}

fn default_image_token() -> u32 {
    151655
}
fn default_video_token() -> u32 {
    151656
}
fn default_vision_start() -> u32 {
    151652
}

impl HiDreamConfig {
    /// Parse from the raw contents of a model `config.json`.
    pub fn from_json(s: &str) -> crate::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}
