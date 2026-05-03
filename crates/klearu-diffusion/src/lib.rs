//! klearu-diffusion: Stable Diffusion checkpoint loading + inference.
//!
//! End-to-end text-to-image generation for Stable Diffusion 1.5 and SDXL,
//! including all components:
//!   - `scheduler::ddim`: DDIM scheduler.
//!   - `vae::AutoencoderKL`: 4-channel latent ↔ RGB encoder/decoder.
//!   - `unet::UNet2DConditionModel`: ResNet + attention denoising network.
//!   - `text_encoder::CLIPTextModel`: prompt → embedding for cross-attention.
//!   - `weight::inventory` / `weight::single_file`: load tensors from
//!     HF Diffusers layouts or single-file `.safetensors` checkpoints.
//!   - `config`: parse HF Diffusers config.json files.
//!
//! # Targets
//!
//! - **SD 1.5**: 512×512 default, latent 64×64×4, CLIP-L (768-d),
//!   UNet ~860M params.
//! - **SDXL**: dual text encoders (CLIP-L + CLIP-G), 1024×1024
//!   default, larger UNet (~2.6B), optional refiner stage.
//!
//! # Connection to the rest of klearu
//!
//! This crate stays separate from klearu-vision to keep classification /
//! encoder-only models decoupled from the generative pipeline. VAE-encoder
//! activations are consumable by other klearu crates for cross-modal work.

pub mod blas;
pub mod config;
#[cfg(feature = "metal")]
pub mod metal_backend;
pub mod error;
pub mod image_io;
pub mod layers;
pub mod scheduler;
pub mod text_encoder;
pub mod tokenizer;
pub mod unet;
pub mod vae;
pub mod weight;

pub use tokenizer::{CLIPTokenizer, CLIP_BOS_ID, CLIP_EOS_ID, CLIP_MAX_LENGTH};

pub use error::{DiffusionError, Result};
