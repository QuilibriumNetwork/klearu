//! AnimateDiff — temporal attention modules for video generation.
//!
//! AnimateDiff works by injecting lightweight temporal self-attention
//! layers after each spatial attention block in an existing frozen
//! SD/SDXL UNet. The spatial path is untouched (your checkpoint's image
//! quality is preserved); the temporal path attends across frames at
//! each spatial position so motion is coherent.
//!
//! ## Architecture
//!
//! ```text
//! For each (resnet, spatial_attn) pair in the UNet:
//!   x = resnet(x)                     // [B*F, C, H, W]
//!   x = spatial_attn(x)               // per-frame spatial attention
//!   x = temporal_attn(x, num_frames)  // cross-frame temporal attention ← NEW
//! ```
//!
//! The temporal attention reshapes `[B*F, C, H, W]` →
//! `[B*H*W, F, C]` (each spatial position is a separate sequence of F
//! tokens), runs multi-head self-attention across F, then reshapes back.
//!
//! ## Weight format
//!
//! The SDXL motion adapter (`mm_sdxl_v10_beta.safetensors`) stores
//! weights under keys like:
//!
//!   `down_blocks.{i}.motion_modules.{j}.temporal_transformer.{...}`
//!   `mid_block.motion_modules.0.temporal_transformer.{...}`
//!   `up_blocks.{i}.motion_modules.{j}.temporal_transformer.{...}`

pub mod temporal;
pub mod motion_module;
pub mod weight;

pub use motion_module::MotionModules;
pub use temporal::TemporalTransformerBlock;
