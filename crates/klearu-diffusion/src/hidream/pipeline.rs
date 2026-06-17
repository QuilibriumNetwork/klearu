//! HiDream-O1 text-to-image denoising pipeline.
//!
//! Mirrors the no-reference (`ref_patches is None`) path of the reference
//! `models/pipeline.py::generate_image`. Pixel-native: the loop runs entirely
//! on patchified latents and the model predicts x₀ each step.
//!
//! The model and tokenizer are decoupled — [`generate_t2i`] takes pre-tokenized
//! prompt ids and a noise source, so the loop is unit-testable without the
//! 8B checkpoint. The binary wires in the Qwen3-VL tokenizer + weight loading.

use super::image::{patchify, unpatchify};
use super::scheduler::DenoiseScheduler;

/// Default initial-noise scale (reference `noise_scale_start`).
pub const NOISE_SCALE: f32 = 7.5;
const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// Anything that can predict x₀ patches for a denoising step. Implemented by
/// both the CPU [`super::model::HiDreamModel`] and the GPU `GpuHiDreamModel`.
pub trait X0Predictor {
    /// `text_ids`, noised patches `z` `[image_len, PATCH_DIM]`, scalar
    /// `timestep` (= `1 − step_t/1000`), grid `h_patches × w_patches`.
    /// Returns predicted x₀ patches `[image_len, PATCH_DIM]`.
    fn predict_x0(
        &self,
        text_ids: &[u32],
        z: &[f32],
        timestep: f32,
        h_patches: usize,
        w_patches: usize,
    ) -> Vec<f32>;
}

impl X0Predictor for super::model::HiDreamModel {
    fn predict_x0(&self, t: &[u32], z: &[f32], ts: f32, hp: usize, wp: usize) -> Vec<f32> {
        self.forward_t2i(t, z, ts, hp, wp)
    }
}

#[cfg(feature = "metal")]
impl X0Predictor for super::gpu::GpuHiDreamModel {
    fn predict_x0(&self, t: &[u32], z: &[f32], ts: f32, hp: usize, wp: usize) -> Vec<f32> {
        self.forward_t2i(t, z, ts, hp, wp)
    }
}

/// Precomputed conditioning for reference-image editing, reused across every
/// denoise step. The vision encoder + image processor run once to fill this;
/// each step only re-noises the target patches.
pub struct RefContext {
    /// Chat-template token ids (text + per-ref `<vision_start> img… <vision_end>`).
    pub text_ids: Vec<u32>,
    /// Vision-encoder output `[num_vlm_tokens, hidden]`, scattered at the
    /// image-token placeholders.
    pub vision_embeds: Vec<f32>,
    /// DeepStack features (one per tapped vision layer), each
    /// `[num_vlm_tokens, hidden]`, injected into the early decoder layers.
    pub deepstack: Vec<Vec<f32>>,
    /// Clean reference patches `[ref_len, PATCH_DIM]` appended to the target in
    /// `vinputs` (constant across steps).
    pub ref_patches: Vec<f32>,
    /// 3D positions `[3, seq]` from `build_position_ids_multi`.
    pub positions: Vec<i64>,
    /// Target token count (`h_patches·w_patches`); the loop denoises these.
    pub tgt_len: usize,
    pub image_token_id: u32,
    pub tms_token_id: u32,
}

/// Anything that can predict x₀ for the target given the reference conditioning.
pub trait RefX0Predictor {
    /// `z_target` is the current noised target `[tgt_len, PATCH_DIM]`; returns
    /// predicted x₀ for the target (same shape).
    fn predict_x0_ref(&self, ctx: &RefContext, z_target: &[f32], timestep: f32) -> Vec<f32>;
}

fn ref_vinputs(ctx: &RefContext, z_target: &[f32]) -> (Vec<f32>, usize) {
    let ref_len = ctx.ref_patches.len() / super::config::PATCH_DIM;
    let gen_len = ctx.tgt_len + ref_len;
    let mut vinputs = Vec::with_capacity(gen_len * super::config::PATCH_DIM);
    vinputs.extend_from_slice(z_target);
    vinputs.extend_from_slice(&ctx.ref_patches);
    (vinputs, gen_len)
}

impl RefX0Predictor for super::model::HiDreamModel {
    fn predict_x0_ref(&self, ctx: &RefContext, z_target: &[f32], timestep: f32) -> Vec<f32> {
        let (vinputs, gen_len) = ref_vinputs(ctx, z_target);
        self.forward_ref(
            &ctx.text_ids, &ctx.vision_embeds, &ctx.deepstack, &vinputs, gen_len, ctx.tgt_len,
            &ctx.positions, timestep, ctx.image_token_id, ctx.tms_token_id,
        )
    }
}

#[cfg(feature = "metal")]
impl RefX0Predictor for super::gpu::GpuHiDreamModel {
    fn predict_x0_ref(&self, ctx: &RefContext, z_target: &[f32], timestep: f32) -> Vec<f32> {
        let (vinputs, gen_len) = ref_vinputs(ctx, z_target);
        self.forward_ref(
            &ctx.text_ids, &ctx.vision_embeds, &ctx.deepstack, &vinputs, gen_len, ctx.tgt_len,
            &ctx.positions, timestep, ctx.image_token_id, ctx.tms_token_id,
        )
    }
}

/// Reference-editing denoise loop. Mirrors [`generate_t2i_cb`] but predicts x₀
/// from the precomputed reference conditioning each step (the reference clean
/// patches ride along in `vinputs` but are never noised). Returns the final
/// image `[C, H, W]` in `[−1, 1]`.
#[allow(clippy::too_many_arguments)]
pub fn generate_ref_cb<M: RefX0Predictor>(
    model: &M,
    cond: &RefContext,
    uncond: Option<&RefContext>,
    params: &GenParams,
    scheduler: &mut dyn DenoiseScheduler,
    mut noise_fn: impl FnMut(usize) -> Vec<f32>,
    mut on_step: impl FnMut(usize, usize),
    abort: &std::sync::atomic::AtomicBool,
) -> Vec<f32> {
    use std::sync::atomic::Ordering;
    let (h, w) = (params.height, params.width);
    let n_pix = super::config::IN_CHANNELS * h * w;
    let scale = params.noise_scale_start;
    let do_cfg = params.guidance_scale > 1.0 && uncond.is_some();
    let n_steps = scheduler.timesteps().len();

    let mut noise0 = noise_fn(n_pix);
    for v in noise0.iter_mut() {
        *v *= scale;
    }
    let mut z = patchify(&noise0, h, w); // [tgt_len, PATCH_DIM]

    let debug = std::env::var("HIDREAM_DEBUG").is_ok();
    if debug {
        stats("ref init z", &z);
    }

    for step in 0..n_steps {
        let step_t = scheduler.timesteps()[step];
        let t_pixeldit = 1.0 - step_t / NUM_TRAIN_TIMESTEPS;
        let sigma = scheduler.velocity_sigma(step);

        let x0_cond = model.predict_x0_ref(cond, &z, t_pixeldit);
        if debug && step < 2 {
            stats(&format!("ref step {step} x0"), &x0_cond);
        }
        let v_guided: Vec<f32> = if do_cfg {
            let x0_uncond = model.predict_x0_ref(uncond.unwrap(), &z, t_pixeldit);
            x0_cond
                .iter()
                .zip(&x0_uncond)
                .zip(&z)
                .map(|((&c, &u), &zi)| {
                    let vc = (c - zi) / sigma;
                    let vu = (u - zi) / sigma;
                    vu + params.guidance_scale * (vc - vu)
                })
                .collect()
        } else {
            x0_cond.iter().zip(&z).map(|(&c, &zi)| (c - zi) / sigma).collect()
        };
        let model_output: Vec<f32> = v_guided.iter().map(|&v| -v).collect();

        let s_noise = if n_steps > 1 {
            params.noise_scale_start
                + (params.noise_scale_end - params.noise_scale_start) * step as f32
                    / (n_steps - 1) as f32
        } else {
            params.noise_scale_start
        };
        let step_noise = if scheduler.needs_noise() { Some(noise_fn(z.len())) } else { None };
        z = scheduler.step(step, &model_output, &z, step_noise.as_deref(), s_noise, params.noise_clip_std);

        on_step(step + 1, n_steps);
        if abort.load(Ordering::Relaxed) {
            break;
        }
    }
    if debug {
        stats("ref final z", &z);
    }
    unpatchify(&z, h, w)
}

/// Print min/max/mean and NaN/Inf counts of a tensor (diagnostics).
pub fn stats(name: &str, x: &[f32]) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nan = 0usize;
    for &v in x {
        if v.is_nan() || v.is_infinite() {
            nan += 1;
            continue;
        }
        mn = mn.min(v);
        mx = mx.max(v);
        sum += v as f64;
    }
    eprintln!(
        "[stat] {name:18} n={} min={mn:.4} max={mx:.4} mean={:.4} nan/inf={nan}",
        x.len(),
        sum / x.len().max(1) as f64,
    );
}

/// Input image for img2img / inpaint, in patch layout `[image_len, PATCH_DIM]`,
/// normalized to `[-1, 1]` (the model's pixel space).
pub struct InitImage {
    /// The input image as patch tokens `[image_len, PATCH_DIM]`.
    pub patches: Vec<f32>,
    /// Denoise strength in `0..=1`. `1.0` = start from full noise (only
    /// constrained by an inpaint mask); lower keeps more of the input.
    pub strength: f32,
    /// Optional per-element inpaint mask, same shape as `patches`. `1` =
    /// regenerate (masked), `0` = keep (known). When set, the known region is
    /// re-noised onto the forward trajectory each step (RePaint).
    pub mask: Option<Vec<f32>>,
}

/// Knobs for one generation.
pub struct GenParams {
    pub height: usize,
    pub width: usize,
    pub guidance_scale: f32,
    pub noise_scale_start: f32,
    pub noise_scale_end: f32,
    pub noise_clip_std: f32,
}

impl GenParams {
    /// Reference "Dev" defaults: 28-step flash schedule, guidance 0 (no CFG).
    pub fn dev(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            guidance_scale: 0.0,
            noise_scale_start: NOISE_SCALE,
            noise_scale_end: NOISE_SCALE,
            noise_clip_std: 2.5,
        }
    }
}

/// Run the text-to-image denoising loop. Returns the final image as planar
/// `[C, H, W]` f32 in `[−1, 1]` (normalized-image space; `save_png` maps to u8).
///
/// - `cond_ids` / `uncond_ids`: tokenized prompts. `uncond_ids` is only used
///   when `guidance_scale > 1`.
/// - `schedule`: the timestep/sigma schedule (Flash for Dev, Euler, …).
/// - `noise_fn(len)`: returns fresh standard-normal noise of length `len`
///   (the caller owns the RNG / seeding).
pub fn generate_t2i<M: X0Predictor>(
    model: &M,
    cond_ids: &[u32],
    uncond_ids: Option<&[u32]>,
    params: &GenParams,
    scheduler: &mut dyn DenoiseScheduler,
    noise_fn: impl FnMut(usize) -> Vec<f32>,
) -> Vec<f32> {
    generate_t2i_cb(model, cond_ids, uncond_ids, params, scheduler, None, noise_fn, |_, _| {}, &std::sync::atomic::AtomicBool::new(false))
}

/// Like [`generate_t2i`] but with a per-step progress callback `on_step(step,
/// total)` and a cooperative `abort` flag (checked each step; returns the
/// current latent decoded when set).
#[allow(clippy::too_many_arguments)]
pub fn generate_t2i_cb<M: X0Predictor>(
    model: &M,
    cond_ids: &[u32],
    uncond_ids: Option<&[u32]>,
    params: &GenParams,
    scheduler: &mut dyn DenoiseScheduler,
    init: Option<&InitImage>,
    mut noise_fn: impl FnMut(usize) -> Vec<f32>,
    mut on_step: impl FnMut(usize, usize),
    abort: &std::sync::atomic::AtomicBool,
) -> Vec<f32> {
    use std::sync::atomic::Ordering;
    let p = super::config::PATCH_SIZE;
    assert_eq!(params.height % p, 0, "height must be a multiple of {p}");
    assert_eq!(params.width % p, 0, "width must be a multiple of {p}");
    let (h, w) = (params.height, params.width);
    let hp = h / p;
    let wp = w / p;
    let n_pix = super::config::IN_CHANNELS * h * w;
    let scale = params.noise_scale_start;
    let do_cfg = params.guidance_scale > 1.0 && uncond_ids.is_some();
    let n_steps = scheduler.timesteps().len();

    // Fresh noise in patch layout (matches `z` / input patches).
    let mut patch_noise = |nf: &mut dyn FnMut(usize) -> Vec<f32>| patchify(&nf(n_pix), h, w);

    // Initial latent. txt2img: scale·N(0,1). img2img/inpaint: blend the input
    // toward the forward trajectory at the starting σ, and start partway in.
    let (mut z, start_step) = if let Some(init) = init {
        let start_step = if init.strength >= 1.0 {
            0
        } else {
            ((1.0 - init.strength.clamp(0.0, 1.0)) * n_steps as f32).floor() as usize
        };
        let sigma_s = scheduler.sigma_at(start_step);
        let noise = patch_noise(&mut noise_fn);
        let z: Vec<f32> = init
            .patches
            .iter()
            .zip(&noise)
            .map(|(&x, &nz)| (1.0 - sigma_s) * x + sigma_s * scale * nz)
            .collect();
        (z, start_step)
    } else {
        let mut noise0 = noise_fn(n_pix);
        for v in noise0.iter_mut() {
            *v *= scale;
        }
        (patchify(&noise0, h, w), 0)
    };

    let debug = std::env::var("HIDREAM_DEBUG").is_ok();
    if debug {
        stats("init z", &z);
    }

    for step in start_step..n_steps {
        let step_t = scheduler.timesteps()[step];
        let t_pixeldit = 1.0 - step_t / NUM_TRAIN_TIMESTEPS;
        let sigma = scheduler.velocity_sigma(step);

        let x0_cond = model.predict_x0(cond_ids, &z, t_pixeldit, hp, wp);
        if debug && step < 2 {
            stats(&format!("step {step} x0"), &x0_cond);
        }
        let v_guided: Vec<f32> = if do_cfg {
            let x0_uncond = model.predict_x0(uncond_ids.unwrap(), &z, t_pixeldit, hp, wp);
            // v = (x0 − z)/σ for each branch; CFG on velocity.
            x0_cond
                .iter()
                .zip(&x0_uncond)
                .zip(&z)
                .map(|((&c, &u), &zi)| {
                    let vc = (c - zi) / sigma;
                    let vu = (u - zi) / sigma;
                    vu + params.guidance_scale * (vc - vu)
                })
                .collect()
        } else {
            x0_cond.iter().zip(&z).map(|(&c, &zi)| (c - zi) / sigma).collect()
        };
        let model_output: Vec<f32> = v_guided.iter().map(|&v| -v).collect();

        // Flash re-noising scale ramps start→end across the schedule.
        let s_noise = if n_steps > 1 {
            params.noise_scale_start
                + (params.noise_scale_end - params.noise_scale_start) * step as f32
                    / (n_steps - 1) as f32
        } else {
            params.noise_scale_start
        };
        let step_noise = if scheduler.needs_noise() { Some(noise_fn(z.len())) } else { None };
        z = scheduler.step(
            step,
            &model_output,
            &z,
            step_noise.as_deref(),
            s_noise,
            params.noise_clip_std,
        );

        // Inpaint (RePaint): pin the known region to the forward trajectory at
        // the next σ so it stays consistent while the model fills the mask.
        if let Some(init) = init {
            if let Some(mask) = &init.mask {
                let sigma_next = scheduler.sigma_at(step + 1);
                let noise_k = patch_noise(&mut noise_fn);
                for i in 0..z.len() {
                    let known = (1.0 - sigma_next) * init.patches[i] + sigma_next * scale * noise_k[i];
                    z[i] = mask[i] * z[i] + (1.0 - mask[i]) * known;
                }
            }
        }

        on_step(step + 1, n_steps);
        if abort.load(Ordering::Relaxed) {
            break;
        }
    }

    if debug {
        stats("final z", &z);
    }
    // z is the final x₀ in [−1, 1] (normalized-image space). Return it as-is —
    // `save_png` maps [−1, 1] → [0, 255] (the reference's `(z+1)/2*255`). Do NOT
    // pre-map to [0, 1] here or the output is washed out (double conversion).
    unpatchify(&z, h, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hidream::config::{HiDreamConfig, RopeScaling, TextConfig, PATCH_SIZE, TMS_TOKEN_ID};
    use crate::hidream::layers::Linear;
    use crate::hidream::model::HiDreamModel;

    fn fill(l: &mut Linear, seed: f32) {
        for (i, w) in l.weight.iter_mut().enumerate() {
            *w = ((i as f32 * seed).sin()) * 0.05;
        }
    }

    #[test]
    fn denoise_loop_runs_and_is_finite() {
        let cfg = HiDreamConfig {
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            text_config: TextConfig {
                hidden_size: 32,
                num_hidden_layers: 1,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                head_dim: 8,
                intermediate_size: 64,
                vocab_size: 152000,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                attention_bias: false,
                rope_scaling: Some(RopeScaling { mrope_section: Some(vec![2, 1, 1]), rope_type: None }),
                max_position_embeddings: 4096,
            },
        };
        let mut model = HiDreamModel::new(cfg);
        for (i, w) in model.embed_tokens.iter_mut().enumerate() {
            *w = ((i % 11) as f32 - 5.0) * 0.01;
        }
        for layer in model.layers.iter_mut() {
            fill(&mut layer.self_attn.q_proj, 0.11);
            fill(&mut layer.self_attn.k_proj, 0.22);
            fill(&mut layer.self_attn.v_proj, 0.33);
            fill(&mut layer.self_attn.o_proj, 0.44);
            fill(&mut layer.mlp.gate_proj, 0.5);
            fill(&mut layer.mlp.up_proj, 0.6);
            fill(&mut layer.mlp.down_proj, 0.7);
        }
        fill(&mut model.x_embedder.proj1, 0.8);
        fill(&mut model.x_embedder.proj2, 0.9);
        fill(&mut model.final_layer2.linear, 0.15);
        fill(&mut model.t_embedder1.lin1, 0.25);
        fill(&mut model.t_embedder1.lin2, 0.35);

        // Deterministic pseudo-noise (LCG) — no RNG dependency in tests.
        let mut seed = 12345u64;
        let noise_fn = move |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
                })
                .collect()
        };

        let (h, w) = (PATCH_SIZE * 2, PATCH_SIZE * 2); // 64x64 → 4 patch tokens
        let params = GenParams::dev(h, w);
        // 3-step flash schedule for a quick test.
        let mut schedule = super::super::scheduler::Schedule::from_timesteps(
            super::super::scheduler::SchedulerKind::Flash,
            &[999.0, 500.0, 8.0],
        );
        let cond_ids = vec![5u32, 9u32, TMS_TOKEN_ID];

        let img = generate_t2i(&model, &cond_ids, None, &params, &mut schedule, noise_fn);
        assert_eq!(img.len(), 3 * h * w);
        assert!(img.iter().all(|v| v.is_finite()));
    }
}
