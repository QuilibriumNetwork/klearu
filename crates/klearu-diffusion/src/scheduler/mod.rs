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
pub mod flow_match;
pub mod samplers;
pub use ddim::{DDIMScheduler, PredictionType};
pub use flow_match::{FlowMatchConfig, FlowMatchScheduler};
pub use samplers::{build_sd_scheduler, SamplerKind};

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

    /// Cumulative product ᾱ at training timestep `t` (t<0 → the final-step
    /// value). Shared by every SD scheduler (same noise schedule); used by the
    /// img2img/inpaint q_sample. Default assumes a leading integer schedule and
    /// is overridden with the real table.
    fn alpha_bar_at(&self, _t: i64) -> f32 { 1.0 }

    /// The (possibly fractional) timestep to feed the UNet at `step_index`.
    /// Karras/sigma schedules produce fractional timesteps; the default just
    /// casts `timesteps()[step_index]`.
    fn timestep_f32(&self, step_index: usize) -> f32 {
        self.timesteps().get(step_index).copied().unwrap_or(0) as f32
    }

    /// Some samplers (ancestral, LCM) inject fresh noise each step. When true,
    /// the pipeline supplies per-step noise via [`Scheduler::step_with_noise`].
    fn needs_noise(&self) -> bool { false }

    /// Step variant that accepts per-step standard-normal noise (same shape as
    /// `sample`). Non-stochastic schedulers ignore it. Default → `step`.
    fn step_with_noise(&self, model_output: &[f32], step_index: usize, sample: &[f32], _noise: &[f32]) -> Vec<f32> {
        self.step(model_output, step_index, sample)
    }
}
