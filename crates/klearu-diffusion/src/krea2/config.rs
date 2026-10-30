//! `Krea2Transformer2DModel` hyperparameters, parsed from `transformer/config.json`.
//!
//! Every field defaults to the released 12.9B checkpoint, so a config that
//! omits a key still describes the real model rather than something smaller.

use serde::Deserialize;

use crate::Result;

/// Latent channels of the Qwen-Image VAE the transformer works against.
pub const LATENT_CHANNELS: usize = 16;

/// VAE spatial compression (8x) times the transformer's 2x2 patch: one token
/// covers a 16x16 pixel square, so image sides must be multiples of this.
pub const PIXELS_PER_TOKEN: usize = 16;

/// Encoder hidden-state taps fed to the text fusion stage. Indices into the
/// HF `hidden_states` tuple, where 0 is the embedding output and `i` is the
/// residual stream after decoder layer `i - 1` — so the deepest tap (35) needs
/// only 35 of Qwen3-VL-4B's 36 layers.
pub const TEXT_SELECT_LAYERS: [usize; 12] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];

#[derive(Debug, Clone, Deserialize)]
pub struct Krea2Config {
    /// Packed patch width: `LATENT_CHANNELS * patch^2` = 64.
    #[serde(default = "d_in_channels")]
    pub in_channels: usize,
    #[serde(default = "d_num_layers")]
    pub num_layers: usize,
    #[serde(default = "d_head_dim")]
    pub attention_head_dim: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    /// Grouped-query attention: 48 query heads share 12 key/value heads.
    #[serde(default = "d_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "d_timestep_dim")]
    pub timestep_embed_dim: usize,
    /// Width of the text encoder's hidden states (Qwen3-VL-4B: 2560).
    #[serde(default = "d_text_dim")]
    pub text_hidden_dim: usize,
    /// Number of tapped encoder layers stacked per token.
    #[serde(default = "d_text_layers")]
    pub num_text_layers: usize,
    #[serde(default = "d_text_heads")]
    pub text_num_attention_heads: usize,
    #[serde(default = "d_text_heads")]
    pub text_num_key_value_heads: usize,
    #[serde(default = "d_text_intermediate")]
    pub text_intermediate_size: usize,
    #[serde(default = "d_two")]
    pub num_layerwise_text_blocks: usize,
    #[serde(default = "d_two")]
    pub num_refiner_text_blocks: usize,
    /// How the head dimension splits across the (t, h, w) rotary axes.
    #[serde(default = "d_axes")]
    pub axes_dims_rope: [usize; 3],
    /// 1000, not Flux's 10000.
    #[serde(default = "d_theta")]
    pub rope_theta: f64,
    #[serde(default = "d_eps")]
    pub norm_eps: f32,
}

fn d_in_channels() -> usize { 64 }
fn d_num_layers() -> usize { 28 }
fn d_head_dim() -> usize { 128 }
fn d_heads() -> usize { 48 }
fn d_kv_heads() -> usize { 12 }
fn d_intermediate() -> usize { 16384 }
fn d_timestep_dim() -> usize { 256 }
fn d_text_dim() -> usize { 2560 }
fn d_text_layers() -> usize { 12 }
fn d_text_heads() -> usize { 20 }
fn d_text_intermediate() -> usize { 6912 }
fn d_two() -> usize { 2 }
fn d_axes() -> [usize; 3] { [32, 48, 48] }
fn d_theta() -> f64 { 1000.0 }
fn d_eps() -> f32 { 1e-5 }

impl Default for Krea2Config {
    fn default() -> Self {
        serde_json::from_str("{}").expect("all Krea2Config fields have defaults")
    }
}

impl Krea2Config {
    pub fn from_json(s: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(s)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(crate::DiffusionError::Layout(m));
        if self.axes_dims_rope.iter().sum::<usize>() != self.attention_head_dim {
            return bad(format!(
                "axes_dims_rope {:?} must sum to attention_head_dim {}",
                self.axes_dims_rope, self.attention_head_dim
            ));
        }
        if self.axes_dims_rope.iter().any(|a| a % 2 != 0) {
            return bad(format!("axes_dims_rope {:?} must all be even", self.axes_dims_rope));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return bad(format!(
                "{} query heads do not divide into {} kv heads",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if !self.text_hidden_dim.is_multiple_of(self.text_num_attention_heads) {
            return bad(format!(
                "text_hidden_dim {} is not divisible by {} heads",
                self.text_hidden_dim, self.text_num_attention_heads
            ));
        }
        if self.in_channels != LATENT_CHANNELS * self.patch() * self.patch() {
            return bad(format!(
                "in_channels {} is not {LATENT_CHANNELS} x a square patch",
                self.in_channels
            ));
        }
        Ok(())
    }

    /// Transformer width: 48 x 128 = 6144.
    pub fn hidden(&self) -> usize {
        self.attention_head_dim * self.num_attention_heads
    }

    /// Side of the square latent patch (2).
    pub fn patch(&self) -> usize {
        ((self.in_channels / LATENT_CHANNELS) as f64).sqrt().round() as usize
    }

    pub fn text_head_dim(&self) -> usize {
        self.text_hidden_dim / self.text_num_attention_heads
    }

    /// A miniature config with the same structure, for CPU/GPU parity tests.
    /// Head dims stay at 32 so the tiled flash-attention pipeline is exercised.
    pub fn tiny() -> Self {
        Self {
            in_channels: 64,
            num_layers: 2,
            attention_head_dim: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 192,
            timestep_embed_dim: 32,
            text_hidden_dim: 64,
            num_text_layers: 3,
            text_num_attention_heads: 2,
            text_num_key_value_heads: 2,
            text_intermediate_size: 96,
            num_layerwise_text_blocks: 2,
            num_refiner_text_blocks: 2,
            axes_dims_rope: [8, 12, 12],
            rope_theta: 1000.0,
            norm_eps: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_released_transformer_config() {
        let text = r#"{
          "_class_name": "Krea2Transformer2DModel",
          "_diffusers_version": "0.39.0.dev0",
          "attention_head_dim": 128,
          "axes_dims_rope": [32, 48, 48],
          "in_channels": 64,
          "intermediate_size": 16384,
          "norm_eps": 1e-05,
          "num_attention_heads": 48,
          "num_key_value_heads": 12,
          "num_layers": 28,
          "num_layerwise_text_blocks": 2,
          "num_refiner_text_blocks": 2,
          "num_text_layers": 12,
          "rope_theta": 1000.0,
          "text_hidden_dim": 2560,
          "text_intermediate_size": 6912,
          "text_num_attention_heads": 20,
          "text_num_key_value_heads": 20,
          "timestep_embed_dim": 256
        }"#;
        let cfg = Krea2Config::from_json(text).unwrap();
        assert_eq!(cfg.hidden(), 6144);
        assert_eq!(cfg.patch(), 2);
        assert_eq!(cfg.text_head_dim(), 128);
        assert_eq!(cfg.num_attention_heads / cfg.num_key_value_heads, 4);
    }

    #[test]
    fn defaults_describe_the_released_model() {
        let cfg = Krea2Config::default();
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.rope_theta, 1000.0);
        assert_eq!(cfg.axes_dims_rope, [32, 48, 48]);
    }

    #[test]
    fn tiny_config_is_valid() {
        Krea2Config::tiny().validate().unwrap();
    }
}
