//! SDXL TextTimeEmbedding.
//!
//! Conceptually:
//!   1. The pooled CLIP-G embedding (1280-d) is taken as-is.
//!   2. Six size/crop conditioning floats (orig_h, orig_w, crop_top,
//!      crop_left, target_h, target_w) are sinusoidally encoded
//!      to 256-d each → 1536-d total.
//!   3. Concatenate (1280 + 1536) → 2816-d.
//!   4. MLP: Linear(2816 → 1280) → SiLU → Linear(1280 → 1280).
//!   5. Add to the time embedding (also 1280-d).

use crate::error::Result;
use crate::layers::{Linear, silu_inplace};
use crate::unet::time_embedding::sinusoidal_embedding;
use crate::weight::{ComponentTensors, load_linear};

pub struct TextTimeEmbedding {
    pub linear_1: Linear,  // [2816 → 1280]
    pub linear_2: Linear,  // [1280 → 1280]
    pub time_embed_dim: usize,
    pub time_id_dim: usize, // per-id sinusoidal dim, e.g. 256
    pub pooled_text_dim: usize, // 1280
}

impl TextTimeEmbedding {
    pub fn new(pooled_text_dim: usize, time_id_dim: usize, time_embed_dim: usize) -> Self {
        let in_dim = pooled_text_dim + 6 * time_id_dim;
        Self {
            linear_1: Linear::new(in_dim, time_embed_dim, true),
            linear_2: Linear::new(time_embed_dim, time_embed_dim, true),
            time_embed_dim,
            time_id_dim,
            pooled_text_dim,
        }
    }

    /// Compute the additional embedding to add to the time embedding.
    /// `pooled_text` is the CLIP-G pooled embedding [pooled_text_dim].
    /// `time_ids` is the 6-tuple [orig_h, orig_w, crop_top, crop_left, target_h, target_w].
    /// Load at `<prefix>.linear_1.{weight,bias}` etc. SDXL stores this under `add_embedding.*`.
    pub fn load_from(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        load_linear(comp, &format!("{prefix}.linear_1"), &mut self.linear_1)?;
        load_linear(comp, &format!("{prefix}.linear_2"), &mut self.linear_2)?;
        Ok(())
    }

    pub fn forward(&self, pooled_text: &[f32], time_ids: &[f32; 6]) -> Vec<f32> {
        // Sinusoidal-encode each id into time_id_dim, concatenate.
        let mut concat = Vec::with_capacity(self.pooled_text_dim + 6 * self.time_id_dim);
        concat.extend_from_slice(pooled_text);
        for &id in time_ids {
            concat.extend_from_slice(&sinusoidal_embedding(id, self.time_id_dim, 10000.0));
        }
        debug_assert_eq!(concat.len(), self.pooled_text_dim + 6 * self.time_id_dim);

        let mut h = vec![0.0f32; self.time_embed_dim];
        self.linear_1.forward(&concat, &mut h);
        silu_inplace(&mut h);
        let mut out = vec![0.0f32; self.time_embed_dim];
        self.linear_2.forward(&h, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_time_embed_correct_dims() {
        let tt = TextTimeEmbedding::new(1280, 256, 1280);
        let pooled = vec![0.1f32; 1280];
        let ids = [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0];
        let out = tt.forward(&pooled, &ids);
        assert_eq!(out.len(), 1280);
    }
}
