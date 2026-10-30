//! Krea 2 — Krea's open-weights text-to-image model (June 2026).
//!
//! A 12.9B single-stream diffusion transformer conditioned on 12 tapped
//! hidden states of Qwen3-VL-4B, decoding through the Qwen-Image VAE (the
//! Wan 2.1 architecture, see [`crate::vae3d`]). Two checkpoints share every
//! component but the transformer: **Raw** (base; ~28–52 steps with CFG) and
//! **Turbo** (distilled; 8 steps, no CFG, fixed timestep shift).
//!
//! - [`config`]: transformer hyperparameters.
//! - [`names`]: diffusers vs reference tensor naming.
//! - [`model`]: f32 CPU reference — the oracle for the GPU path.
//! - [`gpu`]: the Metal transformer.
//! - [`text`]: prompt template + Qwen3-VL encoder.
//! - [`runner`]: end-to-end generation for the desktop app.
//!
//! Two conventions differ from the rest of the crate and are easy to get
//! wrong: guidance is `cond + g·(cond − uncond)` — standard CFG at scale
//! `g + 1` — and the rotary base is 1000, not 10000.

pub mod config;
#[cfg(feature = "metal")]
pub mod gpu;
pub mod model;
pub mod names;
#[cfg(feature = "metal")]
pub mod runner;
pub mod text;
#[cfg(test)]
mod tests;

pub use config::Krea2Config;
