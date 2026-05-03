//! DDIM (Song et al. 2020) scheduler — deterministic at η=0.
//!
//! Formula (η=0, "epsilon" prediction type):
//!
//!   x_0_pred = (x_t − sqrt(1 − ᾱ_t) · ε_θ(x_t, t)) / sqrt(ᾱ_t)
//!   x_{t-1} = sqrt(ᾱ_{t-1}) · x_0_pred + sqrt(1 − ᾱ_{t-1}) · ε_θ
//!
//! where ᾱ_t = ∏ (1 − β_s) for s ≤ t, and {β_t} is the beta schedule
//! (typically scaled-linear for SD: β = linspace(√β_start, √β_end, T)²).

use super::Scheduler;
use crate::config::SchedulerConfig;

pub struct DDIMScheduler {
    pub num_train_timesteps: usize,
    pub alpha_bars: Vec<f32>, // length num_train_timesteps; ᾱ_t for t in 0..T
    pub timesteps: Vec<i64>,  // descending; populated by set_timesteps
    pub prediction_type: PredictionType,
}

#[derive(Debug, Clone, Copy)]
pub enum PredictionType {
    Epsilon,   // model predicts noise ε
    VPrediction, // model predicts v = α·ε − σ·x_0 (used in some SD2.x)
    Sample,    // model predicts x_0 directly
}

impl DDIMScheduler {
    pub fn new(config: &SchedulerConfig) -> Self {
        let t = config.num_train_timesteps;
        let betas = compute_betas(&config.beta_schedule, config.beta_start, config.beta_end, t);
        let mut alpha_bars = Vec::with_capacity(t);
        let mut acc = 1.0f32;
        for &b in &betas {
            acc *= 1.0 - b;
            alpha_bars.push(acc);
        }
        let prediction_type = match config.prediction_type.as_str() {
            "epsilon" => PredictionType::Epsilon,
            "v_prediction" => PredictionType::VPrediction,
            "sample" => PredictionType::Sample,
            other => {
                tracing::warn!("unknown prediction_type {other:?}; defaulting to epsilon");
                PredictionType::Epsilon
            }
        };
        Self {
            num_train_timesteps: t,
            alpha_bars,
            timesteps: Vec::new(),
            prediction_type,
        }
    }

    fn alpha_bar_at(&self, t: i64) -> f32 {
        if t < 0 {
            // Match Diffusers SD 1.5 default: set_alpha_to_one=False uses
            // alphas_cumprod[0] (≈0.99915) as the final-step α_bar_prev,
            // not 1.0. Tiny but matters for the last DDIM step's math.
            self.alpha_bars[0]
        } else {
            let idx = (t as usize).min(self.num_train_timesteps - 1);
            self.alpha_bars[idx]
        }
    }
}

impl Scheduler for DDIMScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize) {
        if num_inference_steps == 0 || num_inference_steps > self.num_train_timesteps {
            self.timesteps.clear();
            return;
        }
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        // Match Diffusers SD 1.5 default: timestep_spacing="leading" with
        // steps_offset=1. For 25 steps over 1000 train this gives [961, 921,
        // ..., 1] (reversed), not [960, 920, ..., 0]. The +1 shifts which
        // α_bar values the model is sampled at — small per step, but
        // accumulates over the trajectory.
        let mut ts: Vec<i64> = (0..num_inference_steps)
            .map(|i| (i * step_ratio) as i64 + 1)
            .collect();
        ts.reverse();
        self.timesteps = ts;
    }

    fn timesteps(&self) -> &[i64] { &self.timesteps }

    fn step(&self, model_output: &[f32], step_index: usize, sample: &[f32]) -> Vec<f32> {
        let t = self.timesteps[step_index];
        let prev_t = if step_index + 1 < self.timesteps.len() {
            self.timesteps[step_index + 1]
        } else {
            -1 // before-zero placeholder; alpha_bar_at(-1) = 1.0
        };

        let alpha_bar_t = self.alpha_bar_at(t);
        let alpha_bar_prev = self.alpha_bar_at(prev_t);
        let sqrt_ab_t = alpha_bar_t.sqrt();
        let sqrt_one_minus_ab_t = (1.0 - alpha_bar_t).sqrt();
        let sqrt_ab_prev = alpha_bar_prev.sqrt();
        let sqrt_one_minus_ab_prev = (1.0 - alpha_bar_prev).sqrt();

        // Compute predicted x_0 from the chosen prediction_type.
        let mut pred_x0 = vec![0.0f32; sample.len()];
        let mut pred_eps = vec![0.0f32; sample.len()];
        match self.prediction_type {
            PredictionType::Epsilon => {
                // ε is given; x_0 = (x_t − sqrt(1−ᾱ_t) · ε) / sqrt(ᾱ_t)
                for i in 0..sample.len() {
                    pred_eps[i] = model_output[i];
                    pred_x0[i] = (sample[i] - sqrt_one_minus_ab_t * model_output[i]) / sqrt_ab_t.max(1e-8);
                }
            }
            PredictionType::Sample => {
                for i in 0..sample.len() {
                    pred_x0[i] = model_output[i];
                    pred_eps[i] = (sample[i] - sqrt_ab_t * model_output[i]) / sqrt_one_minus_ab_t.max(1e-8);
                }
            }
            PredictionType::VPrediction => {
                // v = α·ε − σ·x_0  ⇒  x_0 = α·x_t − σ·v ;  ε = α·v + σ·x_t
                let alpha = sqrt_ab_t;
                let sigma = sqrt_one_minus_ab_t;
                for i in 0..sample.len() {
                    pred_x0[i] = alpha * sample[i] - sigma * model_output[i];
                    pred_eps[i] = alpha * model_output[i] + sigma * sample[i];
                }
            }
        }

        // x_{t-1} = sqrt(ᾱ_{t-1}) · pred_x0 + sqrt(1 − ᾱ_{t-1}) · pred_eps  (η=0)
        let mut prev = vec![0.0f32; sample.len()];
        for i in 0..sample.len() {
            prev[i] = sqrt_ab_prev * pred_x0[i] + sqrt_one_minus_ab_prev * pred_eps[i];
        }
        prev
    }
}

fn compute_betas(schedule: &str, beta_start: f32, beta_end: f32, t: usize) -> Vec<f32> {
    match schedule {
        "linear" => (0..t)
            .map(|i| beta_start + (beta_end - beta_start) * (i as f32) / (t.saturating_sub(1).max(1) as f32))
            .collect(),
        "scaled_linear" => {
            // β = linspace(sqrt(β_start), sqrt(β_end), T)^2  — SD's default.
            let s_start = beta_start.sqrt();
            let s_end = beta_end.sqrt();
            (0..t)
                .map(|i| {
                    let s = s_start + (s_end - s_start) * (i as f32) / (t.saturating_sub(1).max(1) as f32);
                    s * s
                })
                .collect()
        }
        "squaredcos_cap_v2" => {
            // Cosine schedule (Nichol & Dhariwal 2021).
            let f = |t: f32| ((t / (t + 1.0)) * std::f32::consts::PI / 2.0).cos().powi(2);
            let alpha_bar = |i: f32| f(i / t as f32) / f(0.0);
            (0..t).map(|i| {
                let a_i = alpha_bar(i as f32);
                let a_ip1 = alpha_bar((i + 1) as f32);
                (1.0 - a_ip1 / a_i).clamp(0.0, 0.999)
            }).collect()
        }
        other => {
            tracing::warn!("unknown beta_schedule {other:?}; using scaled_linear");
            compute_betas("scaled_linear", beta_start, beta_end, t)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sd15_default_config() -> SchedulerConfig {
        SchedulerConfig {
            class_name: "DDIMScheduler".into(),
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".into(),
            prediction_type: "epsilon".into(),
            clip_sample: false,
        }
    }

    #[test]
    fn alpha_bars_monotone_decreasing() {
        let s = DDIMScheduler::new(&sd15_default_config());
        for w in s.alpha_bars.windows(2) {
            assert!(w[1] <= w[0], "ᾱ should decrease over time");
        }
        assert!(s.alpha_bars[0] > 0.99, "ᾱ_0 ≈ 1");
        assert!(*s.alpha_bars.last().unwrap() < 0.01, "ᾱ_T ≈ 0");
    }

    #[test]
    fn timesteps_descending() {
        let mut s = DDIMScheduler::new(&sd15_default_config());
        s.set_timesteps(50);
        assert_eq!(s.timesteps.len(), 50);
        for w in s.timesteps.windows(2) {
            assert!(w[0] > w[1], "timesteps should descend");
        }
        assert_eq!(s.timesteps[0], 980);
        assert_eq!(*s.timesteps.last().unwrap(), 0);
    }

    #[test]
    fn step_at_t0_returns_x0_pred() {
        let mut s = DDIMScheduler::new(&sd15_default_config());
        s.set_timesteps(50);
        let n = 8;
        let sample: Vec<f32> = (0..n).map(|i| (i as f32 - 4.0) * 0.5).collect();
        let eps: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let last = s.timesteps.len() - 1;
        let prev = s.step(&eps, last, &sample);
        // At the last step (t≈0), prev_t = -1 → ᾱ_prev = 1.0, so prev should be ≈ pred_x0.
        let alpha_bar_t = s.alpha_bar_at(s.timesteps[last]);
        for i in 0..n {
            let expected_x0 = (sample[i] - (1.0 - alpha_bar_t).sqrt() * eps[i]) / alpha_bar_t.sqrt();
            assert!((prev[i] - expected_x0).abs() < 1e-3,
                "i={i}: prev={} expected≈{}", prev[i], expected_x0);
        }
    }
}
