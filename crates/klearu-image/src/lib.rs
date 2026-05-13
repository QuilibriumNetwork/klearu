//! klearu-image: transformer-based image generation, designed for CPU training.
//!
//! # Acceleration features
//!
//! - `accelerate` (workspace feature): macOS Accelerate.framework's AMX
//!   coprocessor handles every matmul in both forward and backward via
//!   `klearu_diffusion::blas::sgemm_*`. Typical 50× speedup over naive
//!   triple-loops on M-series. **Recommended default for CPU training.**
//! - `metal` (workspace feature): Metal/MPS path is wired through for the
//!   tokenizer's SD VAE encode/decode (`klearu_diffusion::vae` →
//!   `decode_with_dims_gpu`) and is available as a fallback for the
//!   transformer matmuls (opt-in via `KLEARU_FORCE_MPS_FP16_SYNC`,
//!   matching the convention from the diffusion crate's blas dispatcher).
//!   Full GPU-residence for the transformer (RMSNorm GPU kernel + causal
//!   SDPA + GPU MoE routing) is a separate follow-up — at our 50M-param
//!   baseline scale, Accelerate matches Metal performance, so the CPU
//!   path is the recommended training target. Metal becomes a measurable
//!   win at hidden_size ≥ 1024 + sequence_length ≥ 1024.
//!
//! Architectural shape (intentionally analogous to ChatGPT Images 2.0):
//!
//!   text prompt ──┐                                 ┌── coarse RGB ──┐
//!                 ├─▶  text BPE  ──┐                │                ├──▶  PNG
//!   [reasoning    │                ▼                │                │
//!    planner]  ───┤             ┌─────────────┐    │                │
//!                 │             │ Autoregress │    │   ┌────────┐  │
//!                 └──────────▶  │ transformer │ ──▶│   │ image  │  │
//!                               │ over image  │    │   │tokenizer│  │
//!                               │ tokens      │    │   │ decoder │ │
//!                               └─────────────┘    │   └────────┘ │
//!                                                  └────────────────┘
//!
//! Phase plan (see TODO files in repo for milestone tracking):
//!   - Phase 0: load a pretrained VQ-VAE / RQ-VAE tokenizer (256×256 → [16,16] int tokens).
//!   - Phase 1: dense decoder-only transformer over image tokens. ~50M params.
//!   - Phase 2: sparse attention via klearu_core LSH + MoE FFN via dejavu predictor.
//!   - Phase 3: reasoning planner prefix from a small klearu-llm model.
//!   - Phase 4: optional diffusion decoder for super-resolution refinement.
//!
//! The point of this crate is to be CPU-trainable end-to-end. Sparseness
//! is what makes that feasible: dense at this param scale would take
//! orders of magnitude longer.

pub mod backward;
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
