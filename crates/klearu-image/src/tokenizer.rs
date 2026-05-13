//! Image tokenizer wrapper.
//!
//! Phase 0 strategy: don't implement a VQ-VAE from scratch. Instead:
//!
//! 1. Reuse the existing SD VAE from `klearu-diffusion::vae` to produce
//!    continuous 4-channel latents at 1/8 resolution.
//! 2. **Discretize** each spatial position's 4-d latent vector by
//!    nearest-neighbour lookup into a learned codebook of K cluster
//!    centroids. K-means is computed *offline* (see `scripts/build_codebook.py`)
//!    and the codebook stored as a safetensors file with two tensors:
//!       - `codebook`: `[K, 4]` f32 — cluster centroids.
//!       - `usage_counts`: `[K]` f32 — optional, just for diagnostics.
//!
//! This is a "Linear-VQ" approximation: not as good as a trained VQ-VAE
//! decoder reconstructing from indices, but:
//!   - **Decode quality is bounded by VAE only** (the codebook noise is
//!     ~1/K of the 4-d latent space — barely visible at K=4096+).
//!   - **Zero new model training needed** — just k-means once.
//!   - **Encoder and decoder are already implemented + GPU-accelerated**
//!     via klearu-diffusion's `AutoencoderKL::{encode, decode}`.
//!
//! Target geometry (matches `ImageTransformerConfig::baseline_50m`):
//!   - Input image: 128×128 RGB
//!   - VAE latent: 16×16×4
//!   - Token grid: 16×16 = 256 tokens per image
//!   - Vocab: K (e.g., 8192) discrete codewords
//!
//! For Phase 2+ (a quality bump), the right upgrade is to swap this for
//! a real trained VQ-VAE; the trait interface stays the same.

use std::collections::HashMap;
use std::path::Path;

use klearu_diffusion::vae::AutoencoderKL;
use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::error::{ImageGenError, Result};

/// Configuration for an image tokenizer instance.
#[derive(Debug, Clone)]
pub struct VqTokenizerConfig {
    /// Square image side length the encoder expects.
    pub image_size: usize,
    /// Token-grid side after encoder downsampling.
    pub token_grid: usize,
    /// Number of discrete codewords.
    pub vocab_size: usize,
    /// Latent channels in the encoder's bottleneck.
    pub latent_channels: usize,
    /// VAE scaling factor applied to the latent after `encode` (SD's
    /// 0.18215 convention). The quantizer operates on the SCALED latent
    /// so that the codebook lives in the same space as the model's
    /// expected input.
    pub vae_scaling_factor: f32,
}

impl VqTokenizerConfig {
    /// Matches `ImageTransformerConfig::baseline_50m`:
    /// 128×128 RGB → 16×16 grid → 8192-codeword vocab on SD VAE's 4-d latents.
    pub fn baseline() -> Self {
        Self {
            image_size: 128,
            token_grid: 16,
            vocab_size: 8192,
            latent_channels: 4,
            vae_scaling_factor: 0.18215,
        }
    }
}

/// Trait surface the rest of klearu-image consumes. The implementation
/// type is `SdVaeQuantizedTokenizer` (below); other implementations can
/// be plugged in later (real VQ-VAE, Cosmos, etc.).
pub trait ImageTokenizer {
    fn config(&self) -> &VqTokenizerConfig;

    /// Encode a `[3, H, W]` RGB image (HW = image_size×image_size, values
    /// in [-1, 1]) to `[token_grid × token_grid]` codeword integers.
    fn encode(&self, image: &[f32]) -> Result<Vec<u32>>;

    /// Decode codeword integers back to a `[3, H, W]` RGB image in [-1, 1].
    fn decode(&self, tokens: &[u32]) -> Result<Vec<f32>>;
}

/// SD VAE + offline-computed k-means codebook → discrete image tokens.
pub struct SdVaeQuantizedTokenizer {
    config: VqTokenizerConfig,
    /// Reuses klearu-diffusion's VAE end-to-end. Owns its weights.
    vae: AutoencoderKL,
    /// Codebook: K rows × `latent_channels` columns, row-major.
    codebook: Vec<f32>,
}

impl SdVaeQuantizedTokenizer {
    /// Construct from a loaded VAE + a codebook safetensors file.
    ///
    /// The codebook file must contain a `codebook` tensor of shape
    /// `[K, latent_channels]` (f32). Optionally also `usage_counts` of
    /// shape `[K]` — that's diagnostics only and is ignored at load.
    pub fn new(config: VqTokenizerConfig, vae: AutoencoderKL, codebook_path: &Path) -> Result<Self> {
        let codebook = load_codebook(codebook_path, config.vocab_size, config.latent_channels)?;
        Ok(Self { config, vae, codebook })
    }

    /// Quantize a single latent vector (length = `latent_channels`) to
    /// the nearest codeword. Returns the index.
    fn quantize_one(&self, v: &[f32]) -> u32 {
        let d = self.config.latent_channels;
        let k = self.config.vocab_size;
        debug_assert_eq!(v.len(), d);
        let mut best = 0_u32;
        let mut best_d2 = f32::INFINITY;
        for ki in 0..k {
            let row = &self.codebook[ki * d..(ki + 1) * d];
            let mut d2 = 0.0_f32;
            for c in 0..d {
                let diff = v[c] - row[c];
                d2 += diff * diff;
            }
            if d2 < best_d2 {
                best_d2 = d2;
                best = ki as u32;
            }
        }
        best
    }
}

impl ImageTokenizer for SdVaeQuantizedTokenizer {
    fn config(&self) -> &VqTokenizerConfig { &self.config }

    fn encode(&self, image: &[f32]) -> Result<Vec<u32>> {
        let cfg = &self.config;
        let img_n = cfg.image_size * cfg.image_size * 3;
        if image.len() != img_n {
            return Err(ImageGenError::ShapeMismatch {
                expected: format!("3·{0}·{0} = {1} f32s", cfg.image_size, img_n),
                got: format!("{}", image.len()),
            });
        }

        // VAE-encode: image → latent mean. [4, lat_h, lat_w] flat.
        let lat_h = cfg.image_size / 8;
        let lat_w = cfg.image_size / 8;
        if lat_h != cfg.token_grid || lat_w != cfg.token_grid {
            return Err(ImageGenError::Config(format!(
                "image_size/8 = {lat_h}×{lat_w} does not match token_grid = {0}",
                cfg.token_grid,
            )));
        }
        let latent = self.vae.encode(image, 1, cfg.image_size, cfg.image_size)
            .map_err(|e| ImageGenError::Unsupported(format!("vae.encode: {e}")))?;
        // Apply VAE scaling so the codebook lives in the canonical space.
        let scaled: Vec<f32> = latent.iter().map(|x| x * cfg.vae_scaling_factor).collect();

        // Per-spatial-position quantize. Latent layout: [c, h, w] with c=4.
        let c = cfg.latent_channels;
        let n_pos = cfg.token_grid * cfg.token_grid;
        let mut tokens = vec![0_u32; n_pos];
        let mut vec_buf = vec![0.0_f32; c];
        for pos in 0..n_pos {
            // Gather the c-d vector at this spatial position.
            for ci in 0..c {
                vec_buf[ci] = scaled[ci * n_pos + pos];
            }
            tokens[pos] = self.quantize_one(&vec_buf);
        }
        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_pos = cfg.token_grid * cfg.token_grid;
        if tokens.len() != n_pos {
            return Err(ImageGenError::ShapeMismatch {
                expected: format!("{n_pos} tokens"),
                got: format!("{}", tokens.len()),
            });
        }

        // Reconstruct scaled latent from codebook lookups.
        let c = cfg.latent_channels;
        let mut scaled = vec![0.0_f32; c * n_pos];
        for (pos, &tok) in tokens.iter().enumerate() {
            if (tok as usize) >= cfg.vocab_size {
                return Err(ImageGenError::ShapeMismatch {
                    expected: format!("codeword < vocab_size = {}", cfg.vocab_size),
                    got: format!("tok={tok} at pos {pos}"),
                });
            }
            let row = &self.codebook[(tok as usize) * c..(tok as usize + 1) * c];
            for ci in 0..c {
                scaled[ci * n_pos + pos] = row[ci];
            }
        }
        // Un-scale and decode via VAE.
        let inv = if cfg.vae_scaling_factor != 0.0 { 1.0 / cfg.vae_scaling_factor } else { 1.0 };
        let latent: Vec<f32> = scaled.iter().map(|x| x * inv).collect();
        Ok(self.vae.decode_with_dims(&latent, 1, cfg.token_grid, cfg.token_grid))
    }
}

/// Load `codebook` tensor from a safetensors file. Errors if missing or
/// has the wrong shape.
fn load_codebook(path: &Path, vocab_size: usize, latent_channels: usize) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let names: HashMap<String, ()> = st.names().iter().map(|n| (n.to_string(), ())).collect();
    if !names.contains_key("codebook") {
        return Err(ImageGenError::MissingTensor("codebook".into()));
    }
    let t = st.tensor("codebook")?;
    let expected_shape = vec![vocab_size, latent_channels];
    if t.shape() != expected_shape.as_slice() {
        return Err(ImageGenError::ShapeMismatch {
            expected: format!("{:?}", expected_shape),
            got: format!("{:?}", t.shape()),
        });
    }
    let bytes = t.data();
    let dtype = t.dtype();
    let n = vocab_size * latent_channels;
    let mut out = Vec::with_capacity(n);
    use safetensors::Dtype;
    match dtype {
        Dtype::F32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        Dtype::F16 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
        }
        Dtype::BF16 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
        }
        other => {
            return Err(ImageGenError::Unsupported(format!("codebook dtype {other:?}")));
        }
    }
    if out.len() != n {
        return Err(ImageGenError::ShapeMismatch {
            expected: format!("{n} codebook elements"),
            got: format!("{}", out.len()),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trip() {
        let cfg = VqTokenizerConfig::baseline();
        assert_eq!(cfg.image_size, 128);
        assert_eq!(cfg.token_grid, 16);
        assert_eq!(cfg.vocab_size, 8192);
        assert_eq!(cfg.latent_channels, 4);
        // image_size / 8 must equal token_grid for the SD VAE's 8× downsample.
        assert_eq!(cfg.image_size / 8, cfg.token_grid);
    }
}
