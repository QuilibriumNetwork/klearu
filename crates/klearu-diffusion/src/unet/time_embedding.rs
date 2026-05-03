//! Time embedding: sinusoidal encoding of the diffusion timestep, then
//! a 2-layer MLP. Used in both the SD 1.5 and SDXL UNets.
//!
//! Sinusoidal half: timestep_embedding(t, dim=320) gives 320 floats
//! mixing sin/cos of t·exp(-i·log(max_period)/half_dim) for i ∈ [0, half_dim).
//!
//! MLP: Linear(320 → 1280) → SiLU → Linear(1280 → 1280).

use crate::error::Result;
use crate::layers::{Linear, silu_inplace};
use crate::weight::{ComponentTensors, load_linear};

pub struct TimeEmbedding {
    pub dim: usize,         // 320 for SD 1.5; SDXL UNet uses 320 too then projects to 1280
    pub time_embed_dim: usize, // 1280
    pub linear_1: Linear,   // [dim → time_embed_dim]
    pub linear_2: Linear,   // [time_embed_dim → time_embed_dim]
    pub max_period: f32,
}

impl TimeEmbedding {
    pub fn new(dim: usize, time_embed_dim: usize) -> Self {
        Self {
            dim,
            time_embed_dim,
            linear_1: Linear::new(dim, time_embed_dim, true),
            linear_2: Linear::new(time_embed_dim, time_embed_dim, true),
            max_period: 10000.0,
        }
    }

    /// Compute the time embedding for a single timestep.
    /// Returns a Vec<f32> of length `time_embed_dim`.
    pub fn forward(&self, timestep: f32) -> Vec<f32> {
        let mut sinusoidal = sinusoidal_embedding(timestep, self.dim, self.max_period);
        let mut h = vec![0.0f32; self.time_embed_dim];
        self.linear_1.forward(&sinusoidal, &mut h);
        silu_inplace(&mut h);
        let mut out = vec![0.0f32; self.time_embed_dim];
        self.linear_2.forward(&h, &mut out);
        sinusoidal.clear();
        out
    }

    /// Load weights at `<prefix>.linear_1.{weight,bias}` and `<prefix>.linear_2.{weight,bias}`.
    pub fn load_from(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        load_linear(comp, &format!("{prefix}.linear_1"), &mut self.linear_1)?;
        load_linear(comp, &format!("{prefix}.linear_2"), &mut self.linear_2)?;
        Ok(())
    }
}

/// Standard sinusoidal positional encoding for diffusion timesteps.
/// Returns a vector of length `dim`. `dim` should be even.
pub fn sinusoidal_embedding(t: f32, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    let log_period = max_period.ln();
    for i in 0..half {
        let freq = (-log_period * (i as f32) / (half as f32)).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_at_zero() {
        let e = sinusoidal_embedding(0.0, 16, 10000.0);
        // cos(0)=1, sin(0)=0 ⇒ first half all 1, second half all 0.
        for i in 0..8 { assert!((e[i] - 1.0).abs() < 1e-6); }
        for i in 8..16 { assert!(e[i].abs() < 1e-6); }
    }

    #[test]
    fn time_embed_returns_correct_length() {
        let te = TimeEmbedding::new(320, 1280);
        let v = te.forward(500.0);
        assert_eq!(v.len(), 1280);
    }
}
