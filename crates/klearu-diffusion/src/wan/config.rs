//! `WanTransformer3DModel` configuration, from a Diffusers
//! `transformer/config.json`.

use serde::Deserialize;

use crate::Result;

/// Wan DiT hyperparameters.
///
/// Wan2.2-TI2V-5B: 30 layers, `dim` 3072 (24 heads x 128), `ffn_dim` 14336,
/// 48 in/out channels, patch `[1, 2, 2]`.
#[derive(Debug, Clone, Deserialize)]
pub struct WanDitConfig {
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub ffn_dim: usize,
    /// Width of the text-encoder hidden states fed to cross-attention
    /// (umT5-XXL: 4096).
    pub text_dim: usize,
    /// Sinusoidal timestep-embedding width before the MLP.
    pub freq_dim: usize,
    /// `(p_t, p_h, p_w)` — `[1, 2, 2]`, which on top of the VAE's 16x spatial
    /// gives 32x effective.
    pub patch_size: Vec<usize>,
    #[serde(default = "default_eps")]
    pub eps: f32,
    /// Always `"rms_norm_across_heads"` for Wan: `norm_q`/`norm_k` normalise
    /// the whole `inner_dim` jointly, before the tensor is split into heads.
    #[serde(default)]
    pub qk_norm: Option<String>,
    /// Whether the cross-attention norm carries affine parameters. True for
    /// Wan, which is why only `norm2` has weights in the checkpoint.
    #[serde(default)]
    pub cross_attn_norm: bool,
    #[serde(default = "default_rope_len")]
    pub rope_max_seq_len: usize,
    /// I2V image conditioning; `None` for TI2V-5B, so no `add_k_proj` tensors.
    #[serde(default)]
    pub added_kv_proj_dim: Option<usize>,
    #[serde(default)]
    pub image_dim: Option<usize>,
    #[serde(default)]
    pub pos_embed_seq_len: Option<usize>,
}

fn default_eps() -> f32 {
    1e-6
}
fn default_rope_len() -> usize {
    1024
}

impl WanDitConfig {
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// `num_attention_heads * attention_head_dim` — the residual-stream width.
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn patch(&self) -> (usize, usize, usize) {
        (self.patch_size[0], self.patch_size[1], self.patch_size[2])
    }

    /// Width of `time_proj`'s output: six modulation vectors (shift, scale and
    /// gate, for attention and for the feed-forward).
    pub fn time_proj_dim(&self) -> usize {
        6 * self.inner_dim()
    }

    /// RoPE frequency budget per axis, `(t, h, w)`.
    ///
    /// Height and width each get `2 * (head_dim / 6)` and time takes the
    /// remainder, so for `head_dim` 128 that is `(44, 42, 42)` — note time is
    /// *wider* than the spatial axes, and the split is not even.
    pub fn rope_dims(&self) -> (usize, usize, usize) {
        let d = self.attention_head_dim;
        let hw = 2 * (d / 6);
        (d - 2 * hw, hw, hw)
    }
}
