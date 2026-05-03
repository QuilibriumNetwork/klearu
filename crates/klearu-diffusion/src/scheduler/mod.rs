//! Diffusion schedulers — the "denoising loop" math.
//!
//! A scheduler turns the UNet's noise prediction into the next-step
//! latent. Different schedulers (DDIM, Euler, DPM-Solver, PNDM, etc.)
//! make different tradeoffs between quality and step count.
//!
//! This module starts with **DDIM** because it is:
//!   - Deterministic given the seed (η=0).
//!   - Has well-known formulas with no ODE-solver bookkeeping.
//!   - Works at 20-50 inference steps for SD 1.5.

pub mod ddim;
pub use ddim::DDIMScheduler;

/// Common interface for any scheduler used in the SD pipeline.
pub trait Scheduler {
    /// Compute the timesteps to use during inference, given the
    /// number of steps requested. Returns timesteps in *descending*
    /// order (we denoise from high noise to low noise).
    fn set_timesteps(&mut self, num_inference_steps: usize);

    /// Returns the current schedule (descending timesteps).
    fn timesteps(&self) -> &[i64];

    /// Apply one denoising step. Given the current latent `sample`
    /// and the UNet's predicted noise `model_output`, produce the
    /// next-step latent. `step_index` is the position into
    /// `timesteps()` (0-indexed).
    fn step(&self, model_output: &[f32], step_index: usize, sample: &[f32]) -> Vec<f32>;

    /// Optional initial noise scaling (some schedulers like Euler
    /// scale the initial latent by sqrt(σ_max² + 1)). Default no-op.
    fn scale_model_input(&self, sample: &mut [f32], _step_index: usize) {
        let _ = sample;
    }

    /// `init_noise_sigma`: the noise scale at the first inference step.
    /// For DDIM this is just 1.0; Euler/Heun would scale by sqrt(σ_max² + 1).
    fn init_noise_sigma(&self) -> f32 { 1.0 }
}
