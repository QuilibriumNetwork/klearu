//! HiDream-O1-Image: pixel-native flow-matching diffusion on a Qwen3-VL-8B
//! transformer backbone.
//!
//! Unlike SD/SDXL (UNet) or Flux (MMDiT), HiDream-O1 has **no VAE** and **no
//! separate text encoders**: text tokens and noised image-patch tokens share a
//! single Qwen3-VL sequence, and the transformer predicts the clean image
//! (x₀) directly. Velocity is recovered in the sampling loop and stepped with a
//! flow-matching scheduler.
//!
//! Module layout:
//!   - [`config`]  — parse the HF `config.json` (Qwen3-VL text backbone + ids)
//!   - [`layers`]  — Linear / RMSNorm + the generation modules (patch embed,
//!                   timestep embedder, final unembed)
//!   - [`mrope`]   — interleaved multimodal RoPE
//!   - `model`     — the decoder + t2i forward (added next)
//!   - `pipeline`  — patchify + denoise loop (Phase 2)
//!
//! Status: Phase 1 in progress (CPU reference; GPU swap to follow).

pub mod config;
#[cfg(feature = "metal")]
pub mod gpu;
pub mod image;
pub mod layers;
pub mod model;
pub mod mrope;
pub mod pipeline;
#[cfg(feature = "metal")]
pub mod runner;
pub mod scheduler;
pub mod vision;
pub mod vlm_image;

pub use config::{HiDreamConfig, TextConfig, PATCH_DIM, PATCH_SIZE, TMS_TOKEN_ID};
pub use model::HiDreamModel;
