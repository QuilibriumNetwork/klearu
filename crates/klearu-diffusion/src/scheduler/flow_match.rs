//! Flow-matching scheduler (rectified flow).
//!
//! Used by Flux. The model predicts velocity v ≈ noise − x_0 along the
//! linear interpolation `x_t = (1−σ) · x_0 + σ · noise` with σ in [0, 1].
//!
//! Forward step (Euler):
//!
//!   x_{t−Δ} = x_t + (σ_{t−Δ} − σ_t) · v    (Δσ < 0, since σ decreases)
//!
//! Equivalently, since the trajectory is straight in (x, σ)-space:
//!
//!   x_{t−Δ} = x_t − (σ_t − σ_{t−Δ}) · v
//!
//! Schedule: a base linear σ schedule from 1 → 0 in `num_inference_steps+1`
//! points, optionally warped by Flux's "shift" parameter `μ` which depends
//! on image sequence length (latent H × W / 4 — i.e., post-patchify token
//! count).
//!
//! For flux1-dev: 50-step default with image-dependent shift.
//! For flux1-schnell: 1–4 step distilled, no shift (`μ = 0`).

use super::Scheduler;

/// Configuration for the flow-matching scheduler.
#[derive(Debug, Clone)]
pub struct FlowMatchConfig {
    /// Whether to apply the Flux dynamic shift. `true` for flux1-dev,
    /// `false` for flux1-schnell.
    pub use_dynamic_shifting: bool,
    /// Base shift used when `use_dynamic_shifting=false`. Flux-schnell
    /// passes `1.0` (no shift).
    pub base_shift: f32,
    /// Maximum shift used at the upper anchor of the dynamic schedule.
    pub max_shift: f32,
    /// Lower anchor for image_seq_len when computing dynamic shift
    /// (default: 256, matching BFL `time_shift` formula).
    pub base_image_seq_len: usize,
    /// Upper anchor for image_seq_len.
    pub max_image_seq_len: usize,
}

impl FlowMatchConfig {
    pub fn flux_dev() -> Self {
        Self {
            use_dynamic_shifting: true,
            base_shift: 0.5,
            max_shift: 1.15,
            base_image_seq_len: 256,
            max_image_seq_len: 4096,
        }
    }
    pub fn flux_schnell() -> Self {
        Self {
            use_dynamic_shifting: false,
            base_shift: 1.0,
            max_shift: 1.0,
            base_image_seq_len: 256,
            max_image_seq_len: 4096,
        }
    }
}

pub struct FlowMatchScheduler {
    config: FlowMatchConfig,
    /// Sigma schedule, length = num_inference_steps + 1, descending from
    /// 1.0 to 0.0 (with shift applied if configured). σ at step i is
    /// `sigmas[i]`; the next step is `sigmas[i+1]`. Final sigma is 0.0.
    sigmas: Vec<f32>,
    /// Timesteps in *descending* model-input units. For flow-matching,
    /// timestep = σ × num_train_timesteps (typically 1000), as a float
    /// sentinel passed to the model. Stored as i64 for `Scheduler` trait
    /// compatibility (rounded; the model itself takes a float).
    timesteps: Vec<i64>,
    /// Unrounded sigmas-as-timesteps for the model (kept alongside
    /// `timesteps` so the consumer can fetch the float σ directly via
    /// `sigma_at`).
    raw_timesteps: Vec<f32>,
    /// Resolution-dependent: image_seq_len = (H/16) × (W/16) for Flux's
    /// 16-pixel patches. Set via `set_image_seq_len` before
    /// `set_timesteps` to enable dynamic shifting.
    image_seq_len: Option<usize>,
}

impl FlowMatchScheduler {
    pub fn new(config: FlowMatchConfig) -> Self {
        Self {
            config,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            raw_timesteps: Vec::new(),
            image_seq_len: None,
        }
    }

    /// Set the latent token count (after patchify) so dynamic shifting can
    /// pick μ. For Flux this is `(latent_h / 2) × (latent_w / 2)` since
    /// Flux applies a 2×2 patch on top of the 16-channel latent, plus
    /// optional sequence packing.
    pub fn set_image_seq_len(&mut self, n: usize) {
        self.image_seq_len = Some(n);
    }

    /// The unrounded model-input timestep (σ × 1000) at step `i`. Use this
    /// instead of `timesteps()` to avoid the i64 rounding in the
    /// `Scheduler` trait.
    pub fn timestep_f32(&self, i: usize) -> f32 {
        self.raw_timesteps[i]
    }

    /// The current σ at step `i`.
    pub fn sigma_at(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    /// All sigmas (descending, length num_inference_steps + 1).
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Compute Flux's dynamic shift μ from the image_seq_len. Linear
    /// interpolation between (base_image_seq_len → base_shift) and
    /// (max_image_seq_len → max_shift), no clamp on out-of-range.
    fn dynamic_mu(&self, image_seq_len: usize) -> f32 {
        let x = image_seq_len as f32;
        let x1 = self.config.base_image_seq_len as f32;
        let x2 = self.config.max_image_seq_len as f32;
        let y1 = self.config.base_shift;
        let y2 = self.config.max_shift;
        let m = (y2 - y1) / (x2 - x1);
        let b = y1 - m * x1;
        m * x + b
    }

    /// Apply the time-shift map  σ' = exp(μ) / (exp(μ) + 1/σ − 1)  to a
    /// base linear σ. With μ=0 this is the identity. Larger μ pushes σ
    /// toward 1 (more time spent at high noise).
    fn time_shift(mu: f32, sigma: f32) -> f32 {
        let e = mu.exp();
        e / (e + (1.0 / sigma.max(1e-6)) - 1.0)
    }
}

impl Scheduler for FlowMatchScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) {
        if num_inference_steps == 0 {
            self.sigmas.clear();
            self.timesteps.clear();
            self.raw_timesteps.clear();
            return;
        }

        let mu = if self.config.use_dynamic_shifting {
            self.dynamic_mu(self.image_seq_len.unwrap_or(self.config.base_image_seq_len))
        } else {
            // Static-shift mode: μ is `base_shift` directly. ComfyUI's
            // schnell config sets shift=1.0 and applies `flux_time_shift(1.0, 1.0, t)`
            // unconditionally — this is *not* the identity transform. We
            // mirror that by always feeding `base_shift` into the time_shift
            // map in static mode. (Earlier we incorrectly forced μ=0 here,
            // producing a flatter σ trajectory than ComfyUI on schnell.)
            self.config.base_shift
        };

        // Base linear σ schedule: 1 → 0 in N+1 points.
        let n = num_inference_steps;
        let mut sigmas: Vec<f32> = (0..=n)
            .map(|i| 1.0 - (i as f32) / (n as f32))
            .collect();

        // Apply the time-shift map. The map has a singularity at σ=0
        // (1/σ blows up), so the final σ=0 entry stays as-is. For μ≈0
        // the map is the identity, so we skip the loop in that case.
        if mu.abs() > 1e-6 {
            for s in sigmas.iter_mut() {
                if *s > 0.0 {
                    *s = Self::time_shift(mu, *s);
                }
            }
        }

        // Compute model-input timesteps (σ × 1000). We expose both
        // rounded i64 (Scheduler trait) and raw f32 (timestep_f32).
        // Last entry (σ=0) becomes timestep 0.
        let raw: Vec<f32> = sigmas.iter().map(|s| s * 1000.0).collect();
        let rounded: Vec<i64> = raw.iter().map(|&s| s.round() as i64).collect();

        // The Scheduler trait expects timesteps to be the *step input*
        // (descending). We drop the trailing σ=0 entry (no model call at
        // the final boundary; that's the output point).
        self.sigmas = sigmas;
        self.raw_timesteps = raw[..n].to_vec();
        self.timesteps = rounded[..n].to_vec();
    }

    fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    /// Euler step.  For flow-matching with model output v:
    ///   x_{t−Δ} = x_t + (σ_{i+1} − σ_i) · v
    /// (σ_{i+1} − σ_i) is negative, so this subtracts in the velocity
    /// direction along the straight-line trajectory.
    fn step(&self, model_output: &[f32], step_index: usize, sample: &[f32]) -> Vec<f32> {
        let sigma_t = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];
        let dsigma = sigma_next - sigma_t;
        let mut out = Vec::with_capacity(sample.len());
        for i in 0..sample.len() {
            out.push(sample[i] + dsigma * model_output[i]);
        }
        out
    }

    fn init_noise_sigma(&self) -> f32 {
        // First σ in the schedule (typically 1.0; lifted toward >1 only
        // when shift > 0 with very small image_seq_len, which doesn't
        // happen in practice — shift maps σ=1 to σ=1 always).
        self.sigmas.first().copied().unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schnell_applies_static_time_shift() {
        // ComfyUI's schnell config uses shift=1.0 and applies
        // `flux_time_shift(1.0, 1.0, t) = e / (e + 1/t − 1)` to every σ.
        // Verify our schnell preset reproduces that mapping (i.e., the
        // schedule is *not* a flat 1.0 → 0.0 linear interp; it's warped
        // by the time-shift map).
        let mut s = FlowMatchScheduler::new(FlowMatchConfig::flux_schnell());
        s.set_timesteps(4);
        assert_eq!(s.sigmas.len(), 5);
        // Boundary sigmas are pinned: σ_0=1.0 (time_shift maps 1.0→1.0)
        // and σ_4=0.0 (left as-is to avoid the 1/σ singularity).
        assert!((s.sigmas[0] - 1.0).abs() < 1e-5);
        assert!(s.sigmas[4].abs() < 1e-5);
        // Mid-schedule σ should match `flux_time_shift(1.0, 1.0, base)`
        // rather than the unshifted linear value. Base linear at i=2 is
        // 0.5; warped value is e / (e + 1/0.5 − 1) = e / (e + 1) ≈ 0.7311.
        let expected = std::f32::consts::E / (std::f32::consts::E + 1.0);
        assert!((s.sigmas[2] - expected).abs() < 1e-5,
            "σ_2={} (expected ≈ {expected})", s.sigmas[2]);
    }

    #[test]
    fn dev_dynamic_shift_pushes_higher_noise() {
        let mut s = FlowMatchScheduler::new(FlowMatchConfig::flux_dev());
        s.set_image_seq_len(4096); // matches max_image_seq_len → μ=max_shift=1.15
        s.set_timesteps(50);
        // σ should still start at 1.0 and end at 0.0…
        assert!((s.sigmas[0] - 1.0).abs() < 1e-3);
        assert!((s.sigmas.last().unwrap() - 0.0).abs() < 1e-3);
        // …but mid-schedule σ should be *higher* than the un-shifted linear
        // value (1 − 25/50 = 0.5).
        let mid = s.sigmas[25];
        assert!(mid > 0.5,
            "shifted σ_mid={mid} should be > 0.5 with μ>0");
    }

    #[test]
    fn euler_step_along_velocity() {
        let mut s = FlowMatchScheduler::new(FlowMatchConfig::flux_schnell());
        s.set_timesteps(4);
        let sample = vec![1.0, 0.5, -0.5, -1.0];
        let v = vec![1.0, 1.0, 1.0, 1.0];
        let next = s.step(&v, 0, &sample);
        // x_{t-Δ} = x_t + (σ_{i+1} − σ_i) · v. Use the actual sigmas in
        // the schedule (which are warped by the schnell time_shift).
        let dsigma = s.sigma_at(1) - s.sigma_at(0);
        for i in 0..sample.len() {
            assert!((next[i] - (sample[i] + dsigma * v[i])).abs() < 1e-6,
                "i={i}: next={}, expected={}", next[i], sample[i] + dsigma * v[i]);
        }
    }

    #[test]
    fn full_trajectory_to_x0_when_v_is_constant() {
        // Sanity: integrating constant v from σ=1 to σ=0 over N Euler
        // steps gives  x_final = x_0 − v  (since Δσ_total = -1).
        let mut s = FlowMatchScheduler::new(FlowMatchConfig::flux_schnell());
        s.set_timesteps(10);
        let mut x = vec![5.0_f32];
        let v = vec![2.0_f32];
        for i in 0..10 {
            x = s.step(&v, i, &x);
        }
        assert!((x[0] - (5.0 - 2.0)).abs() < 1e-5);
    }
}
