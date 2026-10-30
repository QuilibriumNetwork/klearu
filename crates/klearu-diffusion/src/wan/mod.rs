//! Wan DiT (`WanTransformer3DModel`) — the video diffusion transformer.
//!
//! Wan2.2-TI2V-5B: 30 blocks, `dim` 3072 (24 heads x 128), `ffn_dim` 14336,
//! 48 latent channels, patch `[1, 2, 2]`. Each block is
//! self-attention (with 3D RoPE) -> cross-attention to the umT5 text states ->
//! GELU feed-forward, all AdaLN-modulated from the timestep.
//!
//! # TI2V-5B modulates per token
//!
//! Wan 2.1 and the 14B models produce one `[6, dim]` modulation per sample.
//! TI2V-5B instead takes a timestep **per token** — `timestep` arrives as
//! `[batch, seq_len]` — and produces `[batch, seq_len, 6, dim]`. That is how
//! it conditions on an input image: the frames being kept get timestep 0 while
//! the rest get `t`, in the same forward pass. Any port that assumes one
//! modulation vector per sample will silently break image-to-video.
//!
//! The reference branches on `temb.ndim` to choose between the two, both in
//! each block and in the final output norm.

pub mod config;
pub mod names;
#[cfg(feature = "metal")]
pub mod gpu;
pub mod model;
pub mod pipeline;
pub mod rope;
pub mod text;
pub mod weight;

pub use config::WanDitConfig;
pub use names::{tensor_names, tensor_numel};
pub use model::{Modulation, RmsNorm, RopeArg, WanAttention, WanBlock, WanDit};
pub use pipeline::{
    apply_cfg, denormalize_latents, expand_timesteps, wan_sigmas, WanGenConfig, WanPipeline,
};
#[cfg(feature = "metal")]
pub use pipeline::GpuWanPipeline;
pub use rope::{apply_rope, AxisTable, RopeTables};
pub use text::{prompt_clean, WanTextEncoder, MAX_PROMPT_TOKENS};

#[cfg(test)]
mod tests;
