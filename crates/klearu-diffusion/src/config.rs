//! HF Diffusers configuration types.
//!
//! Stable Diffusion checkpoints in the Diffusers format ship as a
//! directory like:
//!
//!   sd-1-5/
//!     model_index.json          ← top-level component manifest
//!     unet/
//!       config.json             ← UNet hyperparameters
//!       diffusion_pytorch_model.safetensors
//!     vae/
//!       config.json
//!       diffusion_pytorch_model.safetensors
//!     text_encoder/
//!       config.json
//!       model.safetensors
//!     tokenizer/
//!       tokenizer.json
//!     scheduler/
//!       scheduler_config.json
//!
//! These types parse the per-component config files into typed Rust
//! structs. We're tolerant of unknown fields (`#[serde(default)]`).

use serde::Deserialize;

/// model_index.json — names each component's class so the loader knows
/// which subdirectory to load and how to dispatch construction.
#[derive(Debug, Deserialize)]
pub struct ModelIndex {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    #[serde(rename = "_diffusers_version", default)]
    pub diffusers_version: Option<String>,
    /// Each remaining field is `[<library>, <ClassName>]`. Standard
    /// SD 1.5 components: text_encoder, tokenizer, unet, vae, scheduler,
    /// safety_checker, feature_extractor.
    #[serde(flatten)]
    pub components: std::collections::HashMap<String, serde_json::Value>,
}

/// UNet config — tracks the hyperparameters that determine architecture.
///
/// Supports both SD 1.5 (cross_attention_dim=768, fixed transformer
/// layers per block = 1) and SDXL (cross_attention_dim=2048,
/// transformer_layers_per_block per-block array, addition_time_embed
/// for size/crop conditioning, addition_embed_type="text_time").
#[derive(Debug, Deserialize, Clone)]
pub struct UNetConfig {
    #[serde(default = "default_sample_size")]
    pub sample_size: usize, // 64 for SD 1.5, 128 for SDXL
    #[serde(default = "default_in_channels")]
    pub in_channels: usize, // 4
    #[serde(default = "default_out_channels")]
    pub out_channels: usize, // 4
    #[serde(default = "default_layers_per_block")]
    pub layers_per_block: usize, // 2
    #[serde(default = "default_block_out_channels")]
    pub block_out_channels: Vec<usize>, // SD15: [320,640,1280,1280]; SDXL: [320,640,1280]
    #[serde(default = "default_down_block_types")]
    pub down_block_types: Vec<String>,
    #[serde(default = "default_up_block_types")]
    pub up_block_types: Vec<String>,
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: AttentionHeadDim,
    #[serde(default = "default_cross_attention_dim")]
    pub cross_attention_dim: usize, // SD15: 768; SDXL: 2048 (CLIP-L 768 + CLIP-G 1280 concat)
    #[serde(default = "default_act_fn")]
    pub act_fn: String, // "silu"
    #[serde(default = "default_norm_num_groups")]
    pub norm_num_groups: usize, // 32
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32, // 1e-5
    #[serde(default = "default_use_linear_projection")]
    pub use_linear_projection: bool,
    #[serde(default = "default_only_cross_attention")]
    pub only_cross_attention: OnlyCrossAttention,
    #[serde(default = "default_dual_cross_attention")]
    pub dual_cross_attention: bool,
    #[serde(default = "default_class_embed_type")]
    pub class_embed_type: Option<String>,
    #[serde(default)]
    pub time_embedding_type: Option<String>,

    // SDXL-specific fields. All optional / defaulted to keep SD 1.5 parsing clean.

    /// Per-block transformer layer count. SD 1.5 = [1,1,1,1]; SDXL = [1,2,10] etc.
    /// Stored as Single(n) for uniform, PerBlock(vec) for SDXL.
    #[serde(default)]
    pub transformer_layers_per_block: Option<TransformerLayersPerBlock>,
    /// "text_time" on SDXL (text embedding + time-id encoding); None on SD 1.5.
    #[serde(default)]
    pub addition_embed_type: Option<String>,
    /// SDXL only: dimension of the time-id sinusoidal embedding (256).
    #[serde(default)]
    pub addition_time_embed_dim: Option<usize>,
    /// SDXL only: number of "additional condition" features, typically 6
    /// (original_size_h, original_size_w, crop_top, crop_left, target_h, target_w).
    #[serde(default)]
    pub addition_embed_type_num_heads: Option<usize>,
    /// SDXL only: projection dim for the additional embedding (1280 + 6×256 = 2816 → 1280).
    #[serde(default)]
    pub projection_class_embeddings_input_dim: Option<usize>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum TransformerLayersPerBlock {
    Single(usize),
    PerBlock(Vec<usize>),
}

impl UNetConfig {
    /// True when this UNet was built for SDXL (vs SD 1.5 / 2.x).
    /// Heuristic: cross_attention_dim of 2048 + addition_embed_type="text_time".
    pub fn is_sdxl(&self) -> bool {
        self.cross_attention_dim == 2048
            && matches!(self.addition_embed_type.as_deref(), Some("text_time"))
    }

    /// Returns the per-block transformer layer count, expanded to one
    /// entry per down/up block.
    pub fn transformer_layers_expanded(&self) -> Vec<usize> {
        let n_blocks = self.block_out_channels.len();
        match &self.transformer_layers_per_block {
            Some(TransformerLayersPerBlock::Single(n)) => vec![*n; n_blocks],
            Some(TransformerLayersPerBlock::PerBlock(v)) => v.clone(),
            None => vec![1; n_blocks], // SD 1.5 default
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum AttentionHeadDim {
    Single(usize),
    PerBlock(Vec<usize>),
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum OnlyCrossAttention {
    Bool(bool),
    PerBlock(Vec<bool>),
}

fn default_sample_size() -> usize { 64 }
fn default_in_channels() -> usize { 4 }
fn default_out_channels() -> usize { 4 }
fn default_layers_per_block() -> usize { 2 }
fn default_block_out_channels() -> Vec<usize> { vec![320, 640, 1280, 1280] }
fn default_down_block_types() -> Vec<String> {
    vec![
        "CrossAttnDownBlock2D".into(),
        "CrossAttnDownBlock2D".into(),
        "CrossAttnDownBlock2D".into(),
        "DownBlock2D".into(),
    ]
}
fn default_up_block_types() -> Vec<String> {
    vec![
        "UpBlock2D".into(),
        "CrossAttnUpBlock2D".into(),
        "CrossAttnUpBlock2D".into(),
        "CrossAttnUpBlock2D".into(),
    ]
}
fn default_attention_head_dim() -> AttentionHeadDim { AttentionHeadDim::Single(8) }
fn default_cross_attention_dim() -> usize { 768 }
fn default_act_fn() -> String { "silu".into() }
fn default_norm_num_groups() -> usize { 32 }
fn default_norm_eps() -> f32 { 1e-5 }
fn default_use_linear_projection() -> bool { false }
fn default_only_cross_attention() -> OnlyCrossAttention { OnlyCrossAttention::Bool(false) }
fn default_dual_cross_attention() -> bool { false }
fn default_class_embed_type() -> Option<String> { None }

/// VAE (AutoencoderKL) config.
#[derive(Debug, Deserialize, Clone)]
pub struct VAEConfig {
    #[serde(default = "default_vae_in_channels")]
    pub in_channels: usize,    // 3 (RGB)
    #[serde(default = "default_vae_out_channels")]
    pub out_channels: usize,   // 3
    #[serde(default = "default_vae_latent_channels")]
    pub latent_channels: usize, // 4
    #[serde(default = "default_vae_block_out_channels")]
    pub block_out_channels: Vec<usize>, // [128, 256, 512, 512]
    #[serde(default = "default_vae_layers_per_block")]
    pub layers_per_block: usize, // 2
    #[serde(default = "default_vae_act_fn")]
    pub act_fn: String, // "silu"
    #[serde(default = "default_vae_norm_num_groups")]
    pub norm_num_groups: usize, // 32
    #[serde(default = "default_vae_sample_size")]
    pub sample_size: usize, // 512
    #[serde(default = "default_vae_scaling_factor")]
    pub scaling_factor: f32, // 0.18215 for SD 1.5
}

fn default_vae_in_channels() -> usize { 3 }
fn default_vae_out_channels() -> usize { 3 }
fn default_vae_latent_channels() -> usize { 4 }
fn default_vae_block_out_channels() -> Vec<usize> { vec![128, 256, 512, 512] }
fn default_vae_layers_per_block() -> usize { 2 }
fn default_vae_act_fn() -> String { "silu".into() }
fn default_vae_norm_num_groups() -> usize { 32 }
fn default_vae_sample_size() -> usize { 512 }
fn default_vae_scaling_factor() -> f32 { 0.18215 }

/// CLIP text encoder config (used as text_encoder/config.json).
#[derive(Debug, Deserialize, Clone)]
pub struct CLIPTextConfig {
    #[serde(default = "default_clip_vocab_size")]
    pub vocab_size: usize, // 49408
    #[serde(default = "default_clip_hidden_size")]
    pub hidden_size: usize, // 768 for L, 1024 for G (SDXL)
    #[serde(default = "default_clip_intermediate_size")]
    pub intermediate_size: usize, // 3072
    #[serde(default = "default_clip_num_hidden_layers")]
    pub num_hidden_layers: usize, // 12
    #[serde(default = "default_clip_num_attention_heads")]
    pub num_attention_heads: usize, // 12
    #[serde(default = "default_clip_max_position_embeddings")]
    pub max_position_embeddings: usize, // 77 for SD
    #[serde(default = "default_clip_layer_norm_eps")]
    pub layer_norm_eps: f32, // 1e-5
    #[serde(default = "default_clip_hidden_act")]
    pub hidden_act: String, // "quick_gelu"
}

fn default_clip_vocab_size() -> usize { 49408 }
fn default_clip_hidden_size() -> usize { 768 }
fn default_clip_intermediate_size() -> usize { 3072 }
fn default_clip_num_hidden_layers() -> usize { 12 }
fn default_clip_num_attention_heads() -> usize { 12 }
fn default_clip_max_position_embeddings() -> usize { 77 }
fn default_clip_layer_norm_eps() -> f32 { 1e-5 }
fn default_clip_hidden_act() -> String { "quick_gelu".into() }

/// Scheduler config (PNDM/DDIM/Euler/etc.).
#[derive(Debug, Deserialize, Clone)]
pub struct SchedulerConfig {
    #[serde(rename = "_class_name", default)]
    pub class_name: String, // "PNDMScheduler" / "DDIMScheduler" / ...
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize, // 1000
    #[serde(default = "default_beta_start")]
    pub beta_start: f32, // 0.00085
    #[serde(default = "default_beta_end")]
    pub beta_end: f32, // 0.012
    #[serde(default = "default_beta_schedule")]
    pub beta_schedule: String, // "scaled_linear"
    #[serde(default = "default_prediction_type")]
    pub prediction_type: String, // "epsilon"
    #[serde(default = "default_clip_sample")]
    pub clip_sample: bool, // false for SD 1.5 PNDM
}

fn default_num_train_timesteps() -> usize { 1000 }
fn default_beta_start() -> f32 { 0.00085 }
fn default_beta_end() -> f32 { 0.012 }
fn default_beta_schedule() -> String { "scaled_linear".into() }
fn default_prediction_type() -> String { "epsilon".into() }
fn default_clip_sample() -> bool { false }

/// Full SD checkpoint, after parsing model_index.json + each component's config.
///
/// Handles both SD 1.5 (single text_encoder) and SDXL (text_encoder + text_encoder_2).
pub struct CheckpointConfig {
    pub root: std::path::PathBuf,
    pub model_index: ModelIndex,
    pub unet: UNetConfig,
    pub vae: VAEConfig,
    /// Always present. CLIP-L for both SD 1.5 and SDXL.
    pub text_encoder: CLIPTextConfig,
    /// Only present on SDXL. CLIP-G (OpenCLIP bigG/14, hidden_size=1280).
    pub text_encoder_2: Option<CLIPTextConfig>,
    pub scheduler: SchedulerConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SDVariant {
    Sd15,
    Sdxl,
    Unknown,
}

impl CheckpointConfig {
    pub fn from_dir(root: &std::path::Path) -> crate::Result<Self> {
        let model_index_path = root.join("model_index.json");
        let model_index_text = std::fs::read_to_string(&model_index_path)?;
        let model_index: ModelIndex = serde_json::from_str(&model_index_text)?;

        let unet_path = root.join("unet").join("config.json");
        let unet: UNetConfig = serde_json::from_str(&std::fs::read_to_string(&unet_path)?)?;

        let vae_path = root.join("vae").join("config.json");
        let vae: VAEConfig = serde_json::from_str(&std::fs::read_to_string(&vae_path)?)?;

        let text_path = root.join("text_encoder").join("config.json");
        let text_encoder: CLIPTextConfig = serde_json::from_str(&std::fs::read_to_string(&text_path)?)?;

        let text_encoder_2: Option<CLIPTextConfig> = {
            let p2 = root.join("text_encoder_2").join("config.json");
            if p2.exists() {
                Some(serde_json::from_str(&std::fs::read_to_string(&p2)?)?)
            } else {
                None
            }
        };

        let sched_path = root.join("scheduler").join("scheduler_config.json");
        let scheduler: SchedulerConfig = serde_json::from_str(&std::fs::read_to_string(&sched_path)?)?;

        Ok(Self {
            root: root.to_path_buf(),
            model_index,
            unet,
            vae,
            text_encoder,
            text_encoder_2,
            scheduler,
        })
    }

    pub fn variant(&self) -> SDVariant {
        if self.unet.is_sdxl() && self.text_encoder_2.is_some() {
            SDVariant::Sdxl
        } else if self.text_encoder_2.is_none() && self.unet.cross_attention_dim == 768 {
            SDVariant::Sd15
        } else {
            SDVariant::Unknown
        }
    }

    /// Default config for SD 1.5 (used when loading a single-file
    /// `.safetensors` checkpoint where per-component config.json files
    /// are absent).
    pub fn sd15_default(root: std::path::PathBuf) -> Self {
        Self {
            root,
            model_index: ModelIndex {
                class_name: "StableDiffusionPipeline".into(),
                diffusers_version: None,
                components: std::collections::HashMap::new(),
            },
            unet: UNetConfig {
                sample_size: 64,
                in_channels: 4,
                out_channels: 4,
                layers_per_block: 2,
                block_out_channels: vec![320, 640, 1280, 1280],
                down_block_types: default_down_block_types(),
                up_block_types: default_up_block_types(),
                attention_head_dim: AttentionHeadDim::Single(8),
                cross_attention_dim: 768,
                act_fn: "silu".into(),
                norm_num_groups: 32,
                norm_eps: 1e-5,
                use_linear_projection: false,
                only_cross_attention: OnlyCrossAttention::Bool(false),
                dual_cross_attention: false,
                class_embed_type: None,
                time_embedding_type: None,
                transformer_layers_per_block: None,
                addition_embed_type: None,
                addition_time_embed_dim: None,
                addition_embed_type_num_heads: None,
                projection_class_embeddings_input_dim: None,
            },
            vae: VAEConfig {
                in_channels: 3,
                out_channels: 3,
                latent_channels: 4,
                block_out_channels: vec![128, 256, 512, 512],
                layers_per_block: 2,
                act_fn: "silu".into(),
                norm_num_groups: 32,
                sample_size: 512,
                scaling_factor: 0.18215,
            },
            text_encoder: CLIPTextConfig {
                vocab_size: 49408,
                hidden_size: 768,
                intermediate_size: 3072,
                num_hidden_layers: 12,
                num_attention_heads: 12,
                max_position_embeddings: 77,
                layer_norm_eps: 1e-5,
                hidden_act: "quick_gelu".into(),
            },
            text_encoder_2: None,
            scheduler: SchedulerConfig {
                class_name: "DDIMScheduler".into(),
                num_train_timesteps: 1000,
                beta_start: 0.00085,
                beta_end: 0.012,
                beta_schedule: "scaled_linear".into(),
                prediction_type: "epsilon".into(),
                clip_sample: false,
            },
        }
    }

    /// Default config for SDXL.
    pub fn sdxl_default(root: std::path::PathBuf) -> Self {
        let mut cfg = Self::sd15_default(root);
        // UNet differences
        cfg.unet.sample_size = 128;
        cfg.unet.block_out_channels = vec![320, 640, 1280];
        cfg.unet.down_block_types = vec![
            "DownBlock2D".into(),
            "CrossAttnDownBlock2D".into(),
            "CrossAttnDownBlock2D".into(),
        ];
        cfg.unet.up_block_types = vec![
            "CrossAttnUpBlock2D".into(),
            "CrossAttnUpBlock2D".into(),
            "UpBlock2D".into(),
        ];
        cfg.unet.attention_head_dim = AttentionHeadDim::PerBlock(vec![5, 10, 20]);
        cfg.unet.cross_attention_dim = 2048;
        cfg.unet.use_linear_projection = true;
        cfg.unet.transformer_layers_per_block = Some(TransformerLayersPerBlock::PerBlock(vec![1, 2, 10]));
        cfg.unet.addition_embed_type = Some("text_time".into());
        cfg.unet.addition_time_embed_dim = Some(256);
        cfg.unet.projection_class_embeddings_input_dim = Some(2816);
        // VAE: SDXL uses 0.13025 scaling factor (instead of 0.18215)
        cfg.vae.scaling_factor = 0.13025;
        cfg.vae.sample_size = 1024;
        // CLIP-G as text_encoder_2
        cfg.text_encoder_2 = Some(CLIPTextConfig {
            vocab_size: 49408,
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            hidden_act: "gelu".into(), // CLIP-G uses true GELU not quick_gelu
        });
        cfg
    }
}
