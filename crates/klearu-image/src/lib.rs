//! klearu-image: transformer-based image generation, designed for CPU training.
//!
//! Dense autoregressive baseline: text prefix → 256 image codewords → decoder.
//! Train with `img_train`; generate with `img_sample` using KV-cached decoding.
//! `accelerate` enables Apple BLAS for training and inference contractions.
//! `metal` is experimental here: this transformer is not GPU-resident.
//!
//! Sparse attention and MoE are separate experimental paths, not CLI defaults.
//! CPU trainability and image quality require measurement, not sparsity claims.
//! See TRAINING.md for the single-M1-Max corpus, memory and evaluation plan.
//!
mod attention;

pub mod backward;
pub mod conditioning;
pub mod checkpoint;
pub mod distributed;
pub mod error;
pub mod grad;
pub mod model;
pub mod moe;
pub mod optim;
pub mod planner;
pub mod sample;
pub mod sparse_attention;
pub mod tokenizer;
pub mod train;

pub use error::{ImageGenError, Result};
