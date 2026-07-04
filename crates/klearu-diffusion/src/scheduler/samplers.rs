//! Alternative SD/SDXL samplers: Euler, Euler Ancestral, DPM++ 2M, LCM
//! (UniPC to follow), with an optional Karras sigma schedule.
//!
//! ## Integration (why these are self-contained)
//! The SD pipeline drives the loop in the **DDPM `x_t`** parametrization
//! (`x_t = √ᾱ·x₀ + √(1−ᾱ)·ε`) and feeds `x_t` straight to the UNet — which is
//! exactly what the model expects. The k-diffusion "sigma" samplers work in a
//! rescaled space `x_σ = x_t/√ᾱ` (with `σ = √((1−ᾱ)/ᾱ)`), but their
//! `scale_model_input` divides by `√(σ²+1)`, which *exactly* recovers `x_t`:
//!
//! ```text
//!   x_σ / √(σ²+1) = (x_t/√ᾱ) / √(1/ᾱ) = x_t
//! ```
//!
//! So the UNet always sees `x_t` regardless of sampler. Each sampler here is a
//! black box `(ε, x_t, step) → x_{t−1}` that internally converts `x_t ↔ x_σ`,
//! applies the k-diffusion update in σ-space, and converts back. Nothing in the
//! pipeline's latent handling changes; only the per-step math differs. Karras
//! reparametrizes the σ ladder (`ᾱ = 1/(1+σ²)` derived from the σ's), and the
//! (fractional) UNet timestep for each σ comes from inverting `σ(t)`.

use super::ddim::PredictionType;
use super::Scheduler;
use crate::config::SchedulerConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SamplerKind {
    #[default]
    Ddim,
    Euler,
    EulerA,
    DpmPP2M,
    UniPC,
    Lcm,
}

impl SamplerKind {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().replace([' ', '_', '-'], "").as_str() {
            "ddim" => Self::Ddim,
            "euler" => Self::Euler,
            "eulera" | "eulerancestral" => Self::EulerA,
            "dpmpp2m" | "dpm++2m" | "dpmsolver++" => Self::DpmPP2M,
            "unipc" => Self::UniPC,
            "lcm" => Self::Lcm,
            _ => return None,
        })
    }
    /// Samplers that need LCM-distilled weights to produce sensible output.
    pub fn requires_lcm_weights(self) -> bool {
        matches!(self, Self::Lcm)
    }
}

/// Build a boxed SD scheduler for the given sampler + Karras option.
pub fn build_sd_scheduler(kind: SamplerKind, karras: bool, cfg: &SchedulerConfig) -> Box<dyn Scheduler> {
    match kind {
        SamplerKind::Ddim => Box::new(super::DDIMScheduler::new(cfg)),
        _ => Box::new(SigmaSampler::new(kind, karras, cfg)),
    }
}

// ============================================================================
// Shared noise schedule
// ============================================================================

struct NoiseSchedule {
    num_train: usize,
    alpha_bars: Vec<f32>,     // ᾱ_t, t in 0..T
    train_sigmas: Vec<f32>,   // σ(t) = √((1−ᾱ_t)/ᾱ_t), ascending in t
    prediction_type: PredictionType,
}

impl NoiseSchedule {
    fn new(cfg: &SchedulerConfig) -> Self {
        let t = cfg.num_train_timesteps;
        let betas = super::ddim::compute_betas(&cfg.beta_schedule, cfg.beta_start, cfg.beta_end, t);
        let mut alpha_bars = Vec::with_capacity(t);
        let mut acc = 1.0f32;
        for &b in &betas {
            acc *= 1.0 - b;
            alpha_bars.push(acc);
        }
        let train_sigmas: Vec<f32> = alpha_bars.iter().map(|&ab| ((1.0 - ab) / ab).sqrt()).collect();
        let prediction_type = match cfg.prediction_type.as_str() {
            "v_prediction" => PredictionType::VPrediction,
            "sample" => PredictionType::Sample,
            _ => PredictionType::Epsilon,
        };
        Self { num_train: t, alpha_bars, train_sigmas, prediction_type }
    }

    fn alpha_bar_at(&self, t: i64) -> f32 {
        if t < 0 {
            self.alpha_bars[0]
        } else {
            self.alpha_bars[(t as usize).min(self.num_train - 1)]
        }
    }

    /// Invert σ → (fractional) training timestep by linear interpolation over
    /// the monotonically-increasing `train_sigmas` (diffusers `_sigma_to_t`).
    fn sigma_to_t(&self, sigma: f32) -> f32 {
        let sig = &self.train_sigmas;
        // log-space interpolation, as diffusers does.
        let log_sigma = sigma.max(1e-10).ln();
        // Find the bracketing indices (train_sigmas ascending).
        let mut lo = 0usize;
        for i in 0..sig.len() - 1 {
            if sig[i] <= sigma {
                lo = i;
            } else {
                break;
            }
        }
        let hi = (lo + 1).min(sig.len() - 1);
        let low = sig[lo].max(1e-10).ln();
        let high = sig[hi].max(1e-10).ln();
        let w = if (high - low).abs() < 1e-12 { 0.0 } else { ((log_sigma - low) / (high - low)).clamp(0.0, 1.0) };
        (1.0 - w) * lo as f32 + w * hi as f32
    }
}

// ============================================================================
// σ-space sampler (Euler / Euler-a / DPM++ 2M / LCM)
// ============================================================================

struct SigmaSampler {
    kind: SamplerKind,
    ns: NoiseSchedule,
    /// σ ladder for the run, length `steps + 1` (last entry 0.0).
    sigmas: Vec<f32>,
    /// Integer UNet timesteps (rounded) for `timesteps()` + indexing.
    timesteps_i: Vec<i64>,
    /// Fractional UNet timesteps (Karras) for the time embedding.
    timesteps_f: Vec<f32>,
    /// Multistep state (DPM++ 2M): previous step's denoised (x₀), σ-space.
    prev_denoised: std::cell::RefCell<Option<Vec<f32>>>,
    prev_step: std::cell::RefCell<Option<usize>>,
    karras: bool,
}

const LCM_SIGMA_DATA: f32 = 0.5;
const LCM_TIMESTEP_SCALING: f32 = 10.0;

impl SigmaSampler {
    fn new(kind: SamplerKind, karras: bool, cfg: &SchedulerConfig) -> Self {
        Self {
            kind,
            ns: NoiseSchedule::new(cfg),
            sigmas: Vec::new(),
            timesteps_i: Vec::new(),
            timesteps_f: Vec::new(),
            prev_denoised: std::cell::RefCell::new(None),
            prev_step: std::cell::RefCell::new(None),
            karras,
        }
    }

    /// Convert one model output at σ to the denoised x₀ (σ-space `x_sig`).
    fn denoised(&self, model_output: &[f32], x_sig: &[f32], sigma: f32) -> Vec<f32> {
        let mut d = vec![0.0f32; x_sig.len()];
        match self.ns.prediction_type {
            PredictionType::Epsilon => {
                for i in 0..d.len() {
                    d[i] = x_sig[i] - sigma * model_output[i];
                }
            }
            PredictionType::Sample => d.copy_from_slice(model_output),
            PredictionType::VPrediction => {
                // c_skip = 1/(σ²+1), c_out = −σ/√(σ²+1) (diffusers precondition).
                let s2 = sigma * sigma + 1.0;
                let c_skip = 1.0 / s2;
                let c_out = -sigma / s2.sqrt();
                for i in 0..d.len() {
                    d[i] = c_skip * x_sig[i] + c_out * model_output[i];
                }
            }
        }
        d
    }
}

impl Scheduler for SigmaSampler {
    fn alpha_bar_at(&self, t: i64) -> f32 {
        self.ns.alpha_bar_at(t)
    }

    fn timestep_f32(&self, step_index: usize) -> f32 {
        self.timesteps_f.get(step_index).copied().unwrap_or(0.0)
    }

    fn timesteps(&self) -> &[i64] {
        &self.timesteps_i
    }

    fn needs_noise(&self) -> bool {
        matches!(self.kind, SamplerKind::EulerA | SamplerKind::Lcm)
    }

    fn set_timesteps(&mut self, n: usize) {
        *self.prev_denoised.borrow_mut() = None;
        *self.prev_step.borrow_mut() = None;
        if n == 0 || n > self.ns.num_train {
            self.sigmas.clear();
            self.timesteps_i.clear();
            self.timesteps_f.clear();
            return;
        }

        if self.kind == SamplerKind::Lcm {
            // LCM: a coarse skipping schedule in DDPM-t; keep σ derived from ᾱ.
            let step_ratio = self.ns.num_train / n;
            let mut ts: Vec<i64> = (0..n).map(|i| (self.ns.num_train - 1 - i * step_ratio) as i64).collect();
            ts.sort_unstable_by(|a, b| b.cmp(a));
            self.timesteps_i = ts.clone();
            self.timesteps_f = ts.iter().map(|&t| t as f32).collect();
            self.sigmas = ts.iter().map(|&t| self.ns.train_sigmas[t as usize]).collect();
            self.sigmas.push(0.0);
            return;
        }

        // Leading integer timesteps (SD default), descending.
        let step_ratio = self.ns.num_train / n;
        let mut ts: Vec<i64> = (0..n).map(|i| (i * step_ratio) as i64 + 1).collect();
        ts.reverse();
        // σ from those timesteps.
        let mut sigmas: Vec<f32> = ts.iter().map(|&t| self.ns.train_sigmas[t as usize]).collect();

        if self.karras {
            sigmas = karras_sigmas(&sigmas, n);
            // Recover fractional timesteps by inverting σ(t).
            self.timesteps_f = sigmas.iter().map(|&s| self.ns.sigma_to_t(s)).collect();
            self.timesteps_i = self.timesteps_f.iter().map(|&t| t.round() as i64).collect();
        } else {
            self.timesteps_i = ts;
            self.timesteps_f = self.timesteps_i.iter().map(|&t| t as f32).collect();
        }
        sigmas.push(0.0);
        self.sigmas = sigmas;
    }

    fn step(&self, model_output: &[f32], step_index: usize, sample: &[f32]) -> Vec<f32> {
        self.step_with_noise(model_output, step_index, sample, &[])
    }

    fn step_with_noise(&self, model_output: &[f32], step_index: usize, sample: &[f32], noise: &[f32]) -> Vec<f32> {
        // ---- LCM: DDPM-space consistency step ----
        if self.kind == SamplerKind::Lcm {
            let t = self.timesteps_i[step_index];
            let ab_t = self.ns.alpha_bar_at(t);
            let (sqrt_ab, sqrt_1m) = (ab_t.sqrt(), (1.0 - ab_t).sqrt());
            let scaled = t as f32 * LCM_TIMESTEP_SCALING;
            let sd2 = LCM_SIGMA_DATA * LCM_SIGMA_DATA;
            let c_skip = sd2 / (scaled * scaled + sd2);
            let c_out = scaled / (scaled * scaled + sd2).sqrt();
            let last = step_index + 1 >= self.timesteps_i.len();
            let ab_prev = if last {
                1.0
            } else {
                self.ns.alpha_bar_at(self.timesteps_i[step_index + 1])
            };
            let mut out = vec![0.0f32; sample.len()];
            for i in 0..sample.len() {
                let x0 = (sample[i] - sqrt_1m * model_output[i]) / sqrt_ab.max(1e-8);
                let denoised = c_out * x0 + c_skip * sample[i];
                out[i] = if last {
                    denoised
                } else {
                    let nz = noise.get(i).copied().unwrap_or(0.0);
                    ab_prev.sqrt() * denoised + (1.0 - ab_prev).sqrt() * nz
                };
            }
            return out;
        }

        // ---- σ-space samplers (Euler / Euler-a / DPM++ 2M) ----
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];
        let ab = 1.0 / (1.0 + sigma * sigma);
        let sqrt_ab = ab.sqrt();
        let ab_next = 1.0 / (1.0 + sigma_next * sigma_next);
        let sqrt_ab_next = ab_next.sqrt();

        // x_t (DDPM) → x_σ.
        let x_sig: Vec<f32> = sample.iter().map(|&x| x / sqrt_ab).collect();
        let denoised = self.denoised(model_output, &x_sig, sigma);

        let mut x_next = vec![0.0f32; sample.len()];
        match self.kind {
            SamplerKind::Euler => {
                let dt = sigma_next - sigma;
                for i in 0..x_next.len() {
                    let d = (x_sig[i] - denoised[i]) / sigma;
                    x_next[i] = x_sig[i] + d * dt;
                }
            }
            SamplerKind::EulerA => {
                // Ancestral split.
                let (sigma_down, sigma_up) = ancestral_step(sigma, sigma_next);
                let dt = sigma_down - sigma;
                for i in 0..x_next.len() {
                    let d = (x_sig[i] - denoised[i]) / sigma;
                    let nz = noise.get(i).copied().unwrap_or(0.0);
                    x_next[i] = x_sig[i] + d * dt + nz * sigma_up;
                }
            }
            // UniPC is not yet implemented; fall back to DPM++ 2M (close, and
            // avoids a panic if it's somehow selected). TODO: true UniPC.
            SamplerKind::DpmPP2M | SamplerKind::UniPC => {
                // k-diffusion dpmpp_2m: t = −log σ, h = t_next − t.
                let t = -(sigma.ln());
                let t_next = if sigma_next > 0.0 { -(sigma_next.ln()) } else { f32::INFINITY };
                let h = t_next - t;
                let ratio = sigma_next / sigma;
                let mut prev = self.prev_denoised.borrow_mut();
                let prev_step = *self.prev_step.borrow();
                let contiguous = prev_step == Some(step_index.wrapping_sub(1));
                if prev.is_none() || sigma_next == 0.0 || !contiguous {
                    let em1 = (-h).exp_m1();
                    for i in 0..x_next.len() {
                        x_next[i] = ratio * x_sig[i] - em1 * denoised[i];
                    }
                } else {
                    let old = prev.as_ref().unwrap();
                    // h_last = t − t_prev; but we only stored denoised, not t_prev.
                    // For the standard uniform-in-t case r≈1; we recover r from the
                    // σ ladder using the previous sigma.
                    let sigma_prev = self.sigmas[step_index - 1];
                    let t_prev = -(sigma_prev.ln());
                    let h_last = t - t_prev;
                    let r = h_last / h;
                    let em1 = (-h).exp_m1();
                    for i in 0..x_next.len() {
                        let d = (1.0 + 1.0 / (2.0 * r)) * denoised[i] - (1.0 / (2.0 * r)) * old[i];
                        x_next[i] = ratio * x_sig[i] - em1 * d;
                    }
                }
                *prev = Some(denoised.clone());
                *self.prev_step.borrow_mut() = Some(step_index);
            }
            SamplerKind::Ddim | SamplerKind::Lcm => unreachable!(),
        }

        // x_σ → x_{t-1} (DDPM).
        for v in x_next.iter_mut() {
            *v *= sqrt_ab_next;
        }
        x_next
    }
}

/// Karras σ ladder (Karras et al. 2022, ρ=7) between the min/max of `in_sigmas`.
fn karras_sigmas(in_sigmas: &[f32], n: usize) -> Vec<f32> {
    let rho = 7.0f32;
    let sigma_min = *in_sigmas.last().unwrap();
    let sigma_max = in_sigmas[0];
    let min_inv = sigma_min.powf(1.0 / rho);
    let max_inv = sigma_max.powf(1.0 / rho);
    (0..n)
        .map(|i| {
            let x = i as f32 / (n as f32 - 1.0).max(1.0);
            (max_inv + x * (min_inv - max_inv)).powf(rho)
        })
        .collect()
}

/// Ancestral σ split: returns (σ_down, σ_up).
fn ancestral_step(sigma: f32, sigma_next: f32) -> (f32, f32) {
    if sigma_next <= 0.0 {
        return (0.0, 0.0);
    }
    let up = (sigma_next * sigma_next * (sigma * sigma - sigma_next * sigma_next) / (sigma * sigma))
        .max(0.0)
        .sqrt()
        .min(sigma_next);
    let down = (sigma_next * sigma_next - up * up).max(0.0).sqrt();
    (down, up)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SchedulerConfig;

    fn sd15_cfg() -> SchedulerConfig {
        SchedulerConfig {
            class_name: "".into(),
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".into(),
            prediction_type: "epsilon".into(),
            clip_sample: false,
        }
    }

    fn pseudo_model(x: &[f32], t: f32) -> Vec<f32> {
        x.iter().map(|&v| 0.15 * v - 0.1 * v.tanh() + 0.0003 * (t - 500.0)).collect()
    }

    fn f32_of(st: &safetensors::SafeTensors, name: &str) -> Vec<f32> {
        let t = st.tensor(name).unwrap();
        t.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            d += x as f64 * y as f64;
            na += (x as f64).powi(2);
            nb += (y as f64).powi(2);
        }
        d / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    fn run_sampler(kind: SamplerKind, karras: bool, init: &[f32], n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
        let cfg = sd15_cfg();
        let mut s = SigmaSampler::new(kind, karras, &cfg);
        s.set_timesteps(n);
        let mut latent = init.to_vec();
        let mut xts = Vec::new();
        for i in 0..n {
            xts.push(latent.clone());
            let t = s.timestep_f32(i);
            let eps = pseudo_model(&latent, t);
            latent = s.step(&eps, i, &latent);
        }
        (xts, latent)
    }

    #[test]
    fn samplers_match_diffusers() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../experiments/sd_samplers_ref.safetensors");
        if !path.exists() {
            eprintln!("SKIP: run scripts/dump_sd_samplers_reference.py first");
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let init = f32_of(&st, "init_noise");
        let n = 10;

        let cases = [
            ("euler", SamplerKind::Euler, false),
            ("euler_karras", SamplerKind::Euler, true),
            ("dpmpp2m", SamplerKind::DpmPP2M, false),
            ("dpmpp2m_karras", SamplerKind::DpmPP2M, true),
        ];
        for (name, kind, karras) in cases {
            let (xts, final_l) = run_sampler(kind, karras, &init, n);
            let ref_xt = f32_of(&st, &format!("{name}.xt")); // [n, N_ELEM]
            let ref_final = f32_of(&st, &format!("{name}.final"));
            let ne = init.len();
            // compare each step's x_t
            let mut worst = f64::INFINITY;
            for i in 0..n {
                let c = cos(&xts[i], &ref_xt[i * ne..(i + 1) * ne]);
                worst = worst.min(c);
            }
            let cf = cos(&final_l, &ref_final);
            eprintln!("[sampler {name:16}] worst-step cos={worst:.6} final cos={cf:.6}");
            assert!(worst > 0.999, "{name}: trajectory diverges (worst step cos {worst})");
            assert!(cf > 0.999, "{name}: final diverges (cos {cf})");
        }
    }
}
