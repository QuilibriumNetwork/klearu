//! Flow-matching schedulers for HiDream-O1.
//!
//! The denoising loop (see the reference `models/pipeline.py::generate_image`)
//! works on the patchified latent `z`. Each step the model predicts x₀; the
//! loop forms velocity `v = (x₀ − z)/σ`, applies CFG, and passes
//! `model_output = −v_guided` to `step`.
//!
//! Two schedulers are implemented:
//!   - [`SchedulerKind::Flash`] — the distilled "Dev" path (re-noising Euler;
//!     `DEFAULT_TIMESTEPS`, 28 steps, shift 1.0).
//!   - [`SchedulerKind::FlowMatchEuler`] — deterministic flow-match Euler.
//! (The full model's UniPC multistep solver is a separate, larger port.)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerKind {
    Flash,
    FlowMatchEuler,
}

/// Common interface for the denoise loop. Flash/Euler are stateless;
/// UniPC is a stateful multistep predictor-corrector.
pub trait DenoiseScheduler {
    /// The per-step timestep values (length = number of inference steps).
    fn timesteps(&self) -> &[f32];
    /// σ used to recover velocity at `step` (= `t/1000`, clamped).
    fn velocity_sigma(&self, step: usize) -> f32 {
        (self.timesteps()[step] / NUM_TRAIN_TIMESTEPS).max(T_EPS)
    }
    /// The flow-matching σ boundary at `step` (0..=n_steps; `sigma_at(n)` = 0).
    /// Used for img2img/inpaint init and known-region renoising.
    fn sigma_at(&self, step: usize) -> f32;
    /// True if the scheduler consumes fresh per-step noise (Flash).
    fn needs_noise(&self) -> bool {
        false
    }
    /// Advance one step. `model_output = −v_guided`; `sample` = current latent.
    fn step(
        &mut self,
        step: usize,
        model_output: &[f32],
        sample: &[f32],
        noise: Option<&[f32]>,
        s_noise: f32,
        noise_clip_std: f32,
    ) -> Vec<f32>;
}

/// The 28-step schedule the distilled Dev model ships with (reference
/// `DEFAULT_TIMESTEPS`).
pub const DEFAULT_TIMESTEPS: [f32; 28] = [
    999.0, 987.0, 974.0, 960.0, 945.0, 929.0, 913.0, 895.0, 877.0, 857.0, 836.0, 814.0, 790.0,
    764.0, 737.0, 707.0, 675.0, 640.0, 602.0, 560.0, 515.0, 464.0, 409.0, 347.0, 278.0, 199.0,
    110.0, 8.0,
];

const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
const T_EPS: f32 = 0.001;

/// A discrete flow-matching schedule: the timestep at each step plus the
/// sigma boundaries (`sigmas.len() == timesteps.len() + 1`, terminal 0).
pub struct Schedule {
    pub kind: SchedulerKind,
    pub timesteps: Vec<f32>,
    pub sigmas: Vec<f32>,
}

impl Schedule {
    /// Build from an explicit timestep list (the Dev path): `σ = t / 1000`,
    /// with a terminal 0 appended.
    pub fn from_timesteps(kind: SchedulerKind, timesteps: &[f32]) -> Self {
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| t / NUM_TRAIN_TIMESTEPS).collect();
        sigmas.push(0.0);
        Self { kind, timesteps: timesteps.to_vec(), sigmas }
    }

    /// Build a linspaced schedule with a flow-match `shift` (full-model path
    /// when no explicit timestep list is supplied):
    ///   `σ_raw = linspace(1, ~0, n)`, `σ = shift·σ / (1 + (shift−1)·σ)`.
    pub fn linspaced(kind: SchedulerKind, num_steps: usize, shift: f32) -> Self {
        let sigma_max = 1.0_f32;
        let sigma_min = 1.0 / NUM_TRAIN_TIMESTEPS;
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        let mut timesteps = Vec::with_capacity(num_steps);
        for i in 0..num_steps {
            let frac = if num_steps > 1 { i as f32 / (num_steps - 1) as f32 } else { 0.0 };
            let s = sigma_max + (sigma_min - sigma_max) * frac;
            let s = shift * s / (1.0 + (shift - 1.0) * s);
            sigmas.push(s);
            timesteps.push(s * NUM_TRAIN_TIMESTEPS);
        }
        sigmas.push(0.0);
        Self { kind, timesteps, sigmas }
    }

    pub fn len(&self) -> usize {
        self.timesteps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timesteps.is_empty()
    }

    /// The sigma used to recover velocity at step `i`: `σ = clamp(t/1000, eps)`.
    pub fn velocity_sigma(&self, step: usize) -> f32 {
        (self.timesteps[step] / NUM_TRAIN_TIMESTEPS).max(T_EPS)
    }

    /// Advance the sample one step.
    ///
    /// - `model_output = −v_guided` (as the loop computes it)
    /// - `sample` = current latent `z`
    /// - `noise` (Flash only): fresh Gaussian noise the same length as `sample`
    /// - `s_noise`, `noise_clip_std`: Flash re-noising controls
    pub fn step(
        &self,
        step: usize,
        model_output: &[f32],
        sample: &[f32],
        noise: Option<&[f32]>,
        s_noise: f32,
        noise_clip_std: f32,
    ) -> Vec<f32> {
        let sigma = self.sigmas[step];
        let sigma_next = self.sigmas[step + 1];
        match self.kind {
            SchedulerKind::FlowMatchEuler => {
                // Deterministic: x ← x + (σ_next − σ)·model_output.
                let dt = sigma_next - sigma;
                sample.iter().zip(model_output).map(|(&x, &m)| x + dt * m).collect()
            }
            SchedulerKind::Flash => {
                // denoised (x₀) = sample − model_output·σ, then re-noise to σ_next.
                let denoised: Vec<f32> =
                    sample.iter().zip(model_output).map(|(&x, &m)| x - m * sigma).collect();
                if sigma_next <= 0.0 {
                    return denoised;
                }
                let noise = noise.expect("Flash scheduler needs fresh noise");
                // Optional symmetric clamp at noise_clip_std · std(noise).
                let clip = if noise_clip_std > 0.0 {
                    let n = noise.len() as f32;
                    let mean = noise.iter().sum::<f32>() / n;
                    let var = noise.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
                    Some(noise_clip_std * var.sqrt())
                } else {
                    None
                };
                denoised
                    .iter()
                    .zip(noise)
                    .map(|(&d, &z)| {
                        let z = match clip {
                            Some(c) => z.clamp(-c, c),
                            None => z,
                        };
                        sigma_next * z * s_noise + (1.0 - sigma_next) * d
                    })
                    .collect()
            }
        }
    }
}

impl DenoiseScheduler for Schedule {
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
    fn sigma_at(&self, step: usize) -> f32 {
        self.sigmas[step.min(self.sigmas.len() - 1)]
    }
    fn needs_noise(&self) -> bool {
        self.kind == SchedulerKind::Flash
    }
    fn step(
        &mut self,
        step: usize,
        model_output: &[f32],
        sample: &[f32],
        noise: Option<&[f32]>,
        s_noise: f32,
        noise_clip_std: f32,
    ) -> Vec<f32> {
        Schedule::step(self, step, model_output, sample, noise, s_noise, noise_clip_std)
    }
}

/// `σ → (α=1−σ, σ)` for flow matching.
#[inline]
fn shift_sigma(s: f32, shift: f32) -> f32 {
    shift * s / (1.0 + (shift - 1.0) * s)
}

/// FlowUniPC multistep predictor-corrector (`solver_order=2`, `predict_x0`,
/// `bh2`, `flow_prediction`) — the full HiDream-O1 model's scheduler. Ported
/// from the reference `models/fm_solvers_unipc.py` for the order-2 case.
pub struct UniPCScheduler {
    pub timesteps: Vec<f32>,
    pub sigmas: Vec<f32>, // len = n+1, terminal 0
    solver_order: usize,
    /// Converted x₀ predictions, newest last (len ≤ solver_order).
    model_outputs: Vec<Vec<f32>>,
    lower_order_nums: usize,
    last_sample: Option<Vec<f32>>,
    this_order: usize,
}

impl UniPCScheduler {
    /// Build with the reference's (double-shifted) sigma schedule.
    pub fn new(num_steps: usize, shift: f32) -> Self {
        // __init__: sigma_max/min from a shifted 1−α ramp.
        let sigma_max = shift_sigma(0.999, shift);
        let sigma_min = shift_sigma(0.0, shift); // = 0
        // set_timesteps: linspace(sigma_max, sigma_min, n+1)[:-1], then shift again.
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        let mut timesteps = Vec::with_capacity(num_steps);
        for i in 0..num_steps {
            let frac = i as f32 / num_steps as f32; // n+1 points, drop last
            let s_lin = sigma_max + (sigma_min - sigma_max) * frac;
            let s = shift_sigma(s_lin, shift);
            sigmas.push(s);
            timesteps.push(s * NUM_TRAIN_TIMESTEPS);
        }
        sigmas.push(0.0); // final_sigmas_type = "zero"
        Self {
            timesteps,
            sigmas,
            solver_order: 2,
            model_outputs: Vec::new(),
            lower_order_nums: 0,
            last_sample: None,
            this_order: 1,
        }
    }

    fn lambda(sigma: f32) -> f32 {
        // λ = log(α) − log(σ), α = 1 − σ.
        (1.0 - sigma).ln() - sigma.ln()
    }

    /// Predictor (UniP, B(h)) for order 1 or 2.
    fn uni_p(&self, step: usize, m0: &[f32]) -> Vec<f32> {
        let sigma_t = self.sigmas[step + 1];
        let sigma_s0 = self.sigmas[step];
        let alpha_t = 1.0 - sigma_t;
        let h = Self::lambda(sigma_t) - Self::lambda(sigma_s0);
        let hh = -h; // predict_x0
        let h_phi_1 = hh.exp_m1(); // e^hh − 1
        let b_h = hh.exp_m1(); // bh2

        let n = m0.len();
        let mut x_t = vec![0.0_f32; n];
        // x_t_ = σ_t/σ_s0 · x − α_t·h_phi_1·m0
        let a = sigma_t / sigma_s0;
        let sample = self.last_sample.as_ref().unwrap();
        for i in 0..n {
            x_t[i] = a * sample[i] - alpha_t * h_phi_1 * m0[i];
        }
        if self.this_order >= 2 && self.model_outputs.len() >= 2 {
            // order 2: rks = [rk1, 1], rhos_p = [0.5]; D1 = (m1 − m0)/rk1.
            let m1 = &self.model_outputs[self.model_outputs.len() - 2];
            let sigma_s1 = self.sigmas[step - 1];
            let rk1 = (Self::lambda(sigma_s1) - Self::lambda(sigma_s0)) / h;
            let coef = -alpha_t * b_h * 0.5 / rk1;
            for i in 0..n {
                x_t[i] += coef * (m1[i] - m0[i]);
            }
        }
        x_t
    }

    /// Corrector (UniC, B(h)) for order 1 or 2. Updates `sample` in place spirit.
    fn uni_c(&self, step: usize, model_t: &[f32], m0: &[f32], this_sample: &[f32]) -> Vec<f32> {
        let sigma_t = self.sigmas[step];
        let sigma_s0 = self.sigmas[step - 1];
        let alpha_t = 1.0 - sigma_t;
        let h = Self::lambda(sigma_t) - Self::lambda(sigma_s0);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();

        let n = model_t.len();
        let x = self.last_sample.as_ref().unwrap();
        let mut x_t = vec![0.0_f32; n];
        let a = sigma_t / sigma_s0;
        // x_t_ = σ_t/σ_s0 · x − α_t·h_phi_1·m0
        for i in 0..n {
            x_t[i] = a * x[i] - alpha_t * h_phi_1 * m0[i];
        }
        let _ = this_sample; // UniPC corrector recomputes from last_sample, not this_sample
        if self.this_order >= 2 && self.model_outputs.len() >= 2 {
            // order 2: solve 2×2 R·ρ = b. R=[[1,1],[rk1,1]], b=[b0,b1].
            let m1 = &self.model_outputs[self.model_outputs.len() - 2];
            let sigma_s1 = self.sigmas[step - 2];
            let rk1 = (Self::lambda(sigma_s1) - Self::lambda(sigma_s0)) / h;
            // h_phi_k progression for b (bh2): see reference loop.
            let mut h_phi_k = h_phi_1 / hh - 1.0;
            let b0 = h_phi_k * 1.0 / b_h; // i=1, factorial_i=1
            let factorial_2 = 2.0;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_2;
            let b1 = h_phi_k * factorial_2 / b_h; // i=2
            // Solve [[1,1],[rk1,1]] [r0,r1]^T = [b0,b1].
            let det = 1.0 - rk1; // 1*1 - 1*rk1
            let r0 = (b0 - b1) / det;
            let r1 = (b1 - rk1 * b0) / det;
            let d1_coef = -alpha_t * b_h * r0 / rk1; // corr_res term: r0 · (m1−m0)/rk1
            let dt_coef = -alpha_t * b_h * r1; // ρ_last · (model_t − m0)
            for i in 0..n {
                x_t[i] += d1_coef * (m1[i] - m0[i]) + dt_coef * (model_t[i] - m0[i]);
            }
        } else {
            // order 1: rhos_c = [0.5], no D1s; D1_t = model_t − m0.
            let coef = -alpha_t * b_h * 0.5;
            for i in 0..n {
                x_t[i] += coef * (model_t[i] - m0[i]);
            }
        }
        x_t
    }
}

impl DenoiseScheduler for UniPCScheduler {
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
    fn sigma_at(&self, step: usize) -> f32 {
        self.sigmas[step.min(self.sigmas.len() - 1)]
    }

    fn step(
        &mut self,
        step: usize,
        model_output: &[f32],
        sample: &[f32],
        _noise: Option<&[f32]>,
        _s_noise: f32,
        _noise_clip_std: f32,
    ) -> Vec<f32> {
        // convert_model_output: x0 = sample − σ_t · model_output.
        let sigma = self.sigmas[step];
        let m_convert: Vec<f32> =
            sample.iter().zip(model_output).map(|(&s, &m)| s - sigma * m).collect();

        // Corrector (uses the PREVIOUS step's history before we push).
        let mut sample_corr = sample.to_vec();
        let use_corrector = step > 0 && self.last_sample.is_some() && !self.model_outputs.is_empty();
        if use_corrector {
            let m0 = self.model_outputs[self.model_outputs.len() - 1].clone();
            sample_corr = self.uni_c(step, &m_convert, &m0, sample);
        }

        // Push converted output into history (cap at solver_order).
        self.model_outputs.push(m_convert);
        if self.model_outputs.len() > self.solver_order {
            self.model_outputs.remove(0);
        }

        // this_order: lower_order_final warmup + tail.
        let n_steps = self.timesteps.len();
        let order_final = self.solver_order.min(n_steps - step);
        self.this_order = order_final.min(self.lower_order_nums + 1).max(1);

        // Predictor.
        self.last_sample = Some(sample_corr.clone());
        let m0 = self.model_outputs[self.model_outputs.len() - 1].clone();
        // The predictor uses last_sample (= corrected sample) as x.
        let prev = self.uni_p(step, &m0);

        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }
        prev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_sigmas_from_default() {
        let s = Schedule::from_timesteps(SchedulerKind::Flash, &DEFAULT_TIMESTEPS);
        assert_eq!(s.sigmas.len(), DEFAULT_TIMESTEPS.len() + 1);
        assert!((s.sigmas[0] - 0.999).abs() < 1e-6);
        assert_eq!(*s.sigmas.last().unwrap(), 0.0);
    }

    #[test]
    fn euler_step_moves_toward_x0() {
        // With model_output = -(x0 - z)/sigma, an Euler step to sigma_next=0
        // should land exactly on x0.
        let sched = Schedule { kind: SchedulerKind::FlowMatchEuler, timesteps: vec![500.0], sigmas: vec![0.5, 0.0] };
        let z = vec![1.0_f32, -2.0];
        let x0 = vec![0.2_f32, 0.3];
        let sigma = 0.5;
        let v: Vec<f32> = x0.iter().zip(&z).map(|(&a, &b)| (a - b) / sigma).collect();
        let model_output: Vec<f32> = v.iter().map(|&x| -x).collect();
        let out = sched.step(0, &model_output, &z, None, 1.0, 0.0);
        for (o, x) in out.iter().zip(&x0) {
            assert!((o - x).abs() < 1e-5, "{o} != {x}");
        }
    }

    #[test]
    fn unipc_sigmas_monotonic_and_terminal_zero() {
        let s = UniPCScheduler::new(50, 3.0);
        assert_eq!(s.timesteps.len(), 50);
        assert_eq!(s.sigmas.len(), 51);
        assert_eq!(*s.sigmas.last().unwrap(), 0.0);
        // sigmas strictly decreasing toward 0.
        for w in s.sigmas.windows(2) {
            assert!(w[0] > w[1] - 1e-6, "sigmas not decreasing: {} -> {}", w[0], w[1]);
        }
        assert!(s.sigmas[0] > 0.9 && s.sigmas[0] < 1.0);
    }

    #[test]
    fn unipc_runs_and_converges_to_x0() {
        // With a constant x0 target, the multistep solver should land near it.
        let mut s = UniPCScheduler::new(8, 3.0);
        let n = 16usize;
        let x0_target = vec![0.3_f32; n];
        // z starts as noise; each step the "model" reports model_output = -(x0 - z)/sigma.
        let mut z = vec![1.5_f32; n];
        for step in 0..s.timesteps.len() {
            let sigma = (s.timesteps[step] / 1000.0).max(0.001);
            let model_output: Vec<f32> =
                z.iter().zip(&x0_target).map(|(&zi, &x)| -((x - zi) / sigma)).collect();
            z = s.step(step, &model_output, &z, None, 1.0, 0.0);
            assert!(z.iter().all(|v| v.is_finite()), "step {step} produced non-finite");
        }
        for (zi, xi) in z.iter().zip(&x0_target) {
            assert!((zi - xi).abs() < 1e-2, "did not converge: {zi} vs {xi}");
        }
    }

    #[test]
    fn flash_last_step_returns_x0() {
        let sched = Schedule { kind: SchedulerKind::Flash, timesteps: vec![8.0], sigmas: vec![0.008, 0.0] };
        let z = vec![1.0_f32, -2.0, 0.5];
        let model_output = vec![0.1_f32, 0.2, -0.3];
        let out = sched.step(0, &model_output, &z, None, 1.0, 0.0);
        // sigma_next == 0 → returns denoised = z - model_output*sigma.
        for ((o, &x), &m) in out.iter().zip(&z).zip(&model_output) {
            assert!((o - (x - m * 0.008)).abs() < 1e-6);
        }
    }
}
