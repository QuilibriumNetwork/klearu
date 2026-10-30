//! `AutoencoderKLWan` configuration, parsed from a Diffusers `vae/config.json`.

use serde::Deserialize;

use crate::Result;

/// Wan video VAE hyperparameters.
///
/// Wan 2.1 (`z_dim` 16) and Wan 2.2 (`z_dim` 48) share this struct; they differ
/// by [`WanVaeConfig::is_residual`], which selects the residual down/up blocks
/// with `AvgDown3D`/`DupUp3D` shortcuts over the classic resample path.
///
/// The Qwen-Image VAE (`AutoencoderKLQwenImage`, also Krea 2's) is a fine-tune
/// of Wan 2.1 with the identical module tree, so its `vae/config.json` parses
/// here too. It differs only in spelling: `input_channels` for `in_channels`,
/// and no `is_residual` / `patch_size` / `decoder_base_dim` keys at all — whose
/// defaults below are exactly the Wan 2.1 values.
#[derive(Debug, Clone, Deserialize)]
pub struct WanVaeConfig {
    /// Encoder width at the first stage.
    pub base_dim: usize,
    /// Decoder width at its first stage. Wan 2.2 makes the decoder wider than
    /// the encoder (256 vs 160); falls back to `base_dim` when absent.
    #[serde(default)]
    pub decoder_base_dim: Option<usize>,
    /// Latent channels.
    pub z_dim: usize,
    /// Width multiplier per stage.
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// Stages that carry an attention block. Empty for Wan 2.2 — there is no
    /// attention anywhere in this VAE, which removes an entire class of work
    /// from the port.
    #[serde(default)]
    pub attn_scales: Vec<f32>,
    /// Which downsample stages also halve time. `[false, true, true]` gives
    /// the 4x temporal compression.
    pub temperal_downsample: Vec<bool>,
    #[serde(default)]
    pub dropout: f32,
    /// Channels the VAE itself sees: RGB after the pixel patchify, so
    /// `3 * patch_size^2` = 12 for Wan 2.2 and 3 for Wan 2.1.
    /// `AutoencoderKLQwenImage` calls it `input_channels` — same meaning, and
    /// without the alias its config fails to parse rather than defaulting.
    #[serde(alias = "input_channels", default = "three")]
    pub in_channels: usize,
    #[serde(default)]
    pub out_channels: usize,
    /// Pixel-space patch fold applied before the encoder.
    #[serde(default = "one")]
    pub patch_size: usize,
    /// Wan 2.2's residual down/up blocks. Absent (false) for Wan 2.1 and
    /// Qwen-Image, which use plain resnet stages — see
    /// [`super::blocks::StageKind`].
    #[serde(default)]
    pub is_residual: bool,
    #[serde(default)]
    pub clip_output: bool,
    /// Per-latent-channel normalisation applied *outside* the VAE: the
    /// transformer works in `(latent - mean) / std` space.
    #[serde(default)]
    pub latents_mean: Vec<f32>,
    #[serde(default)]
    pub latents_std: Vec<f32>,
}

fn one() -> usize {
    1
}

fn three() -> usize {
    3
}

impl WanVaeConfig {
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Decoder first-stage width, defaulting to the encoder's.
    pub fn decoder_dim(&self) -> usize {
        self.decoder_base_dim.unwrap_or(self.base_dim)
    }

    /// Total spatial compression: one halving per stage after the first, times
    /// the pixel patchify. 16x for Wan 2.2, 8x for Wan 2.1 / Qwen-Image.
    pub fn spatial_compression(&self) -> usize {
        self.patch_size * (1 << (self.dim_mult.len() - 1))
    }

    /// Total temporal compression: 2x per `temperal_downsample` stage. 4x for
    /// Wan 2.2.
    pub fn temporal_compression(&self) -> usize {
        1 << self.temperal_downsample.iter().filter(|d| **d).count()
    }

    /// Encoder channel width entering each stage.
    pub fn stage_dims(&self, base: usize) -> Vec<usize> {
        self.dim_mult.iter().map(|m| base * m).collect()
    }
}
