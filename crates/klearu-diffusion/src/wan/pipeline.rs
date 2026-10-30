//! Wan text-to-video pipeline: schedule, conditioning, and the denoising loop.
//!
//! Ties together the umT5 text encoder, the DiT, and the 3D causal VAE. The
//! conventions here are all from `WanPipeline` in diffusers and are easy to
//! get wrong in ways that produce plausible-but-wrong video rather than errors.

use crate::hidream::scheduler::{DenoiseScheduler, UniPCScheduler};
use crate::vae3d::{Vol, WanVae};

use super::model::WanDit;

/// Wan's sigma schedule: `linspace(1, 1/1000, steps+1)[:-1]`, shifted once,
/// with a terminal 0 (`final_sigmas_type: "zero"`).
///
/// Note this shifts **once**. `UniPCScheduler::new` applies HiDream's
/// double-shifted ramp, which is a different schedule for the same solver.
///
/// `flow_shift` belongs to the checkpoint, not to the resolution. Read it from
/// `scheduler/scheduler_config.json` with [`flow_shift_from_checkpoint`] rather
/// than assuming: the "5.0 at 720p, 3.0 at 480p" rule is Wan 2.1 T2V-14B's, and
/// applying it to Wan 2.2 TI2V-5B — whose config asks for 5.0 at every
/// resolution — under-shifts the schedule and blows the output distribution out
/// (std 1.49 in normalised latent space against ~1.0, decoding to a video with
/// 35% of its pixels clipped white).
///
/// The first sigma is nudged below 1.0. UniPC's `lambda(s) = ln(1-s) - ln(s)`
/// is `-inf` at exactly 1.0, so leaving it there makes the first two solver
/// updates produce NaN. diffusers applies the same `1e-6` nudge with the
/// comment "to avoid inf torch.log(alpha_si)".
pub fn wan_sigmas(num_steps: usize, flow_shift: f32) -> Vec<f32> {
    const NUM_TRAIN: f32 = 1000.0;
    const EPS: f32 = 1e-6;
    let mut out = Vec::with_capacity(num_steps + 1);
    for i in 0..num_steps {
        // linspace over steps+1 points, dropping the last.
        let frac = i as f32 / num_steps as f32;
        let s = 1.0 + (1.0 / NUM_TRAIN - 1.0) * frac;
        out.push(flow_shift * s / (1.0 + (flow_shift - 1.0) * s));
    }
    if (out[0] - 1.0).abs() < EPS {
        out[0] -= EPS;
    }
    out.push(0.0);
    out
}

/// Read `flow_shift` from a checkpoint's `scheduler/scheduler_config.json`.
///
/// Sampling hyperparameters ship *with* the weights, and guessing them from
/// folklore is how a pipeline ends up quietly mis-tuned while every component
/// still matches its reference numerically.
pub fn flow_shift_from_checkpoint(root: &std::path::Path) -> Option<f32> {
    let txt = std::fs::read_to_string(root.join("scheduler/scheduler_config.json")).ok()?;
    let key = "\"flow_shift\"";
    let i = txt.find(key)? + key.len();
    let rest = &txt[i..];
    let j = rest.find(':')? + 1;
    rest[j..]
        .trim_start()
        .split([',', '}', '\n'])
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Build the per-token timestep vector from a conditioning mask.
///
/// `mask` is `[frames, height, width]` at **latent** resolution: 1 where the
/// token is being denoised, 0 where it is held as conditioning. The reference
/// computes `(mask[:, ::2, ::2] * t).flatten()` — subsampling spatially by the
/// patch size, not temporally (patch_t is 1) — and the result is ordered
/// `(f, h, w)` with width fastest, matching the token order.
///
/// Text-to-video passes an all-ones mask, so every token gets `t`. Image-to-
/// video zeroes the conditioned frames, which is the whole reason TI2V-5B
/// modulates per token.
pub fn expand_timesteps(
    mask: &[f32],
    (f, h, w): (usize, usize, usize),
    (p_t, p_h, p_w): (usize, usize, usize),
    t: f32,
) -> Vec<f32> {
    assert_eq!(mask.len(), f * h * w, "mask must be at latent resolution");
    let (ppf, pph, ppw) = (f / p_t, h / p_h, w / p_w);
    let mut out = Vec::with_capacity(ppf * pph * ppw);
    for fi in 0..ppf {
        for y in 0..pph {
            for x in 0..ppw {
                // Subsample by taking the patch's top-left corner, as `::2` does.
                let m = mask[(fi * p_t * h + y * p_h) * w + x * p_w];
                out.push(m * t);
            }
        }
    }
    out
}

/// Undo the per-channel latent normalisation before VAE decode.
///
/// The transformer works in `(raw - mean) / std` space. The reference spells
/// the inverse as `latents / (1/std) + mean`, i.e. `raw = z * std + mean`.
pub fn denormalize_latents(z: &mut [f32], channels: usize, mean: &[f32], std: &[f32]) {
    assert_eq!(mean.len(), channels);
    assert_eq!(std.len(), channels);
    let plane = z.len() / channels;
    for c in 0..channels {
        for v in z[c * plane..(c + 1) * plane].iter_mut() {
            *v = *v * std[c] + mean[c];
        }
    }
}

/// The forward direction, for image-to-video conditioning.
pub fn normalize_latents(z: &mut [f32], channels: usize, mean: &[f32], std: &[f32]) {
    let plane = z.len() / channels;
    for c in 0..channels {
        for v in z[c * plane..(c + 1) * plane].iter_mut() {
            *v = (*v - mean[c]) / std[c];
        }
    }
}


/// Start and/or end frame conditioning for image-to-video.
///
/// Both ends are optional and independent: neither (text-to-video), start only
/// (the usual image-to-video), end only, or both.
///
/// Conditioning is two things that must agree. `mask` is 0.0 at the latent
/// frames being held and 1.0 elsewhere, which `expand_timesteps` turns into a
/// timestep of 0 for those tokens — telling the model they are already clean.
/// `latents` carries the values to hold them *at*. The mask alone is not
/// enough: without re-pinning after every scheduler step the held frames drift
/// away from the reference image, and the model is then told "this is clean"
/// about something that is not.
#[derive(Debug, Clone)]
pub struct WanConditioning {
    /// Normalised latents `[z_dim, T, H, W]`; only masked positions are read.
    pub latents: Vec<f32>,
    /// Latent-resolution mask `[T, H, W]`: 0.0 = held, 1.0 = denoised.
    pub mask: Vec<f32>,
}

impl WanConditioning {
    /// Build conditioning from optional start/end images, already at the
    /// generation's pixel resolution and in `[-1, 1]`.
    ///
    /// The two ends are encoded differently, and not arbitrarily. Wan's VAE is
    /// causal with a temporal compression of 4: latent frame 0 covers exactly
    /// one pixel frame, every later latent frame covers four. So a start image
    /// encodes as a single frame and lands in slot 0 exactly. An end image has
    /// to represent four pixel frames, so it is encoded as a five-frame clip of
    /// itself and the *second* latent frame is taken — that is the frame whose
    /// cache state and 4-frame span match the slot it is going into.
    ///
    /// End-frame conditioning is off-distribution for TI2V-5B, which was
    /// trained to condition on a *first* frame (first-and-last-frame is a
    /// separate Wan 2.1 checkpoint). Be precise about what that does and does
    /// not mean. The held frame is held exactly either way — that is pinning,
    /// not inference, and measured at cos_sim 0.97 for an end frame against
    /// 1.00 for a start frame. What is off-distribution is the model's ability
    /// to generate motion that *arrives* at a given ending, which no amount of
    /// pinning provides. Expect the last frame to be right and the approach to
    /// it to be weaker than the departure from a start frame.
    pub fn from_images(
        vae: &crate::vae3d::WanVae,
        dims: (usize, usize, usize),
        start: Option<&crate::vae3d::Vol>,
        end: Option<&crate::vae3d::Vol>,
    ) -> Self {
        let (t, h, w) = dims;
        let z_dim = vae.cfg.z_dim;
        let plane = t * h * w;
        let mut latents = vec![0.0_f32; z_dim * plane];
        let mut mask = vec![1.0_f32; plane];

        // The encoder returns mean and log-variance stacked; conditioning uses
        // the distribution mean, not a sample, so the held frame is exactly the
        // reference image rather than one draw from it.
        let take_mean = |moments: &crate::vae3d::Vol, frame: usize| -> Vec<f32> {
            let fp = moments.h * moments.w;
            let mut out = vec![0.0_f32; z_dim * fp];
            for c in 0..z_dim {
                let src = (c * moments.t + frame) * fp;
                out[c * fp..(c + 1) * fp].copy_from_slice(&moments.data[src..src + fp]);
            }
            out
        };
        let mut place = |frame: usize, mut z: Vec<f32>| {
            normalize_latents(
                &mut z,
                z_dim,
                &vae.cfg.latents_mean,
                &vae.cfg.latents_std,
            );
            let fp = h * w;
            for c in 0..z_dim {
                let dst = c * plane + frame * fp;
                latents[dst..dst + fp].copy_from_slice(&z[c * fp..(c + 1) * fp]);
            }
            for m in mask[frame * fp..(frame + 1) * fp].iter_mut() {
                *m = 0.0;
            }
        };

        // Prefer the GPU encoder. It is the difference between seconds and
        // ~25 minutes for a start frame at native resolution, and seconds
        // against roughly three hours for an end frame's five-frame clip.
        // `encode_gpu` returns None rather than guessing when it cannot serve a
        // shape, so the CPU path stays both the fallback and the reference.
        let encode = |v: &crate::vae3d::Vol| -> crate::vae3d::Vol {
            #[cfg(feature = "metal")]
            if let Some(out) = vae.encode_gpu(v) {
                return out;
            }
            vae.encode(v)
        };

        if let Some(img) = start {
            let moments = encode(img);
            place(0, take_mean(&moments, 0));
        }
        if let Some(img) = end {
            if t >= 2 {
                // The last latent frame stands for four pixel frames, so the
                // end image is encoded as a five-frame clip of itself and the
                // *second* latent frame is taken. A single-frame encode into
                // that slot is not equivalent — measured cos_sim 0.97 but with
                // the magnitude wrong (std 0.48 against 0.75), which would pin
                // the frame to something the model was told is clean and is not.
                let clip = repeat_frames(img, 5);
                let moments = encode(&clip);
                place(t - 1, take_mean(&moments, 1));
            }
        }
        Self { latents, mask }
    }
}

/// Repeat a single-frame volume `n` times along time.
fn repeat_frames(img: &crate::vae3d::Vol, n: usize) -> crate::vae3d::Vol {
    assert_eq!(img.t, 1, "repeat_frames expects a single frame");
    let fp = img.h * img.w;
    let mut data = vec![0.0_f32; img.c * n * fp];
    for c in 0..img.c {
        for f in 0..n {
            let dst = (c * n + f) * fp;
            data[dst..dst + fp].copy_from_slice(&img.data[c * fp..(c + 1) * fp]);
        }
    }
    crate::vae3d::Vol { data, c: img.c, t: n, h: img.h, w: img.w }
}

/// Hold the conditioned latent frames at their reference values.
///
/// Called once before the loop and again after every scheduler step. Skipping
/// the post-step call is the subtle failure: the video still generates and the
/// conditioning image still influences it, but the held frame drifts, so the
/// first frame of the output is recognisably *not* the image that was passed in.
pub fn apply_conditioning(latents: &mut [f32], cond: &WanConditioning, channels: usize) {
    let plane = cond.mask.len();
    debug_assert_eq!(latents.len(), channels * plane);
    for (i, &m) in cond.mask.iter().enumerate() {
        if m != 0.0 {
            continue;
        }
        for c in 0..channels {
            latents[c * plane + i] = cond.latents[c * plane + i];
        }
    }
}

/// Wan's own default negative prompt, verbatim from `sample_neg_prompt` in the
/// reference `wan/configs/shared_config.py`.
///
/// It is Chinese because umT5 is multilingual and this is what the model was
/// tuned against. Substituting a short invented English list ("blurry, low
/// quality, static, watermark") is not a neutral choice: at guidance 5.0 the
/// negative branch is half the guidance signal, and this string specifically
/// names oversaturation, overexposure and static frames — the exact artefacts
/// that showed up in early generations here.
pub const DEFAULT_NEGATIVE_PROMPT: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

/// Reference sampling defaults for TI2V-5B, from `wan/configs/wan_ti2v_5B.py`:
/// 50 steps, guidance 5.0, shift 5.0, 24 fps, 121 frames.
pub const REFERENCE_STEPS: usize = 50;
/// Frames per second the model was tuned for. Encoding at 16 plays it slow.
pub const REFERENCE_FPS: usize = 24;

impl WanGenConfig {
    /// Whether step `i` of `total` runs the unconditional pass.
    ///
    /// Half-open on the right so `(0.0, 1.0)` covers every step and a band
    /// never runs one extra by rounding.
    pub fn cfg_applies_at(&self, i: usize, total: usize) -> bool {
        if total == 0 {
            return false;
        }
        let f = i as f32 / total as f32;
        f >= self.cfg_interval.0 && f < self.cfg_interval.1
    }

    /// Forwards a full run costs, for reporting and for sizing an experiment.
    pub fn forward_count(&self) -> usize {
        if self.guidance_scale <= 1.0 {
            return self.num_steps;
        }
        (0..self.num_steps)
            .filter(|i| self.cfg_applies_at(*i, self.num_steps))
            .count()
            + self.num_steps
    }
}

/// Classifier-free guidance, in place on the conditional prediction:
/// `uncond + scale * (cond - uncond)`.
pub fn apply_cfg(cond: &mut [f32], uncond: &[f32], scale: f32) {
    for (c, u) in cond.iter_mut().zip(uncond) {
        *c = u + scale * (*c - u);
    }
}

/// Denoising settings.
#[derive(Debug, Clone)]
pub struct WanGenConfig {
    pub num_steps: usize,
    /// 5.0 at 720p, 3.0 at 480p.
    pub flow_shift: f32,
    /// Values <= 1 skip the negative pass entirely.
    pub guidance_scale: f32,
    /// Fraction of the schedule, as `(start, end)` in [0, 1], over which
    /// classifier-free guidance is applied. `(0.0, 1.0)` is every step.
    ///
    /// Guidance costs a second full forward on every step it runs — half the
    /// generation. It is also not equally useful at every timestep: the early
    /// steps set composition from nearly pure noise and the late ones only
    /// refine detail, so narrowing the band drops forwards for much less
    /// quality than the step count would.
    pub cfg_interval: (f32, f32),
}

impl Default for WanGenConfig {
    fn default() -> Self {
        Self { num_steps: 50, flow_shift: 5.0, guidance_scale: 5.0, cfg_interval: (0.0, 1.0) }
    }
}

/// Text-to-video pipeline over an already-loaded DiT and VAE.
pub struct WanPipeline {
    pub dit: WanDit,
    pub vae: WanVae,
}

impl WanPipeline {
    pub fn new(dit: WanDit, vae: WanVae) -> Self {
        Self { dit, vae }
    }

    /// Run the denoising loop. `latents` is `[z_dim, T, H, W]` of noise in
    /// normalised latent space; returns the denoised latents in the same
    /// space.
    ///
    /// `prompt` and `negative` are `[n_ctx, text_dim]` umT5 hidden states.
    /// `mask` is the per-token conditioning mask (all ones for text-to-video).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        mut latents: Vec<f32>,
        dims: (usize, usize, usize),
        prompt: &[f32],
        negative: Option<&[f32]>,
        n_ctx: usize,
        mask: &[f32],
        cond: Option<&WanConditioning>,
        cfg: &WanGenConfig,
        mut on_step: impl FnMut(usize, usize, &[f32]),
    ) -> Vec<f32> {
        let sigmas = wan_sigmas(cfg.num_steps, cfg.flow_shift);
        let mut sched = UniPCScheduler::from_sigmas(sigmas);
        let timesteps = sched.timesteps.clone();
        // Hold the conditioned frames from the start: step 0 is told they are
        // clean, so they must actually be the reference before it runs.
        if let Some(c) = cond {
            apply_conditioning(&mut latents, c, self.vae.cfg.z_dim);
        }
        let patch = self.dit.cfg.patch();

        for (i, &t) in timesteps.iter().enumerate() {
            let ts = expand_timesteps(mask, dims, patch, t);
            let mut pred = self.dit.forward(&latents, dims, &ts, prompt, n_ctx);
            if let Some(neg) = negative {
                if cfg.guidance_scale > 1.0 && cfg.cfg_applies_at(i, timesteps.len()) {
                    let uncond = self.dit.forward(&latents, dims, &ts, neg, n_ctx);
                    apply_cfg(&mut pred, &uncond, cfg.guidance_scale);
                }
            }
            // `x0 = sample - sigma * model_output`, i.e. diffusers'
            // `flow_prediction`: pass the raw velocity, not its negation.
            latents = sched.step(i, &pred, &latents, None, 1.0, 0.0);
            // Re-pin: the scheduler moves every position, conditioned or not.
            if let Some(c) = cond {
                apply_conditioning(&mut latents, c, self.vae.cfg.z_dim);
            }
            on_step(i + 1, timesteps.len(), &latents);
        }
        latents
    }

    /// Denoised latents -> pixels: denormalise, then chunked VAE decode.
    pub fn decode(&self, latents: &[f32], dims: (usize, usize, usize)) -> Vol {
        let z_dim = self.vae.cfg.z_dim;
        let mut z = latents.to_vec();
        if self.vae.cfg.latents_mean.len() == z_dim && self.vae.cfg.latents_std.len() == z_dim {
            denormalize_latents(
                &mut z,
                z_dim,
                &self.vae.cfg.latents_mean.clone(),
                &self.vae.cfg.latents_std.clone(),
            );
        }
        self.vae.decode(&Vol::new(z, z_dim, dims.0, dims.1, dims.2))
    }
}

// ---------------------------------------------------------------------------
// GPU pipeline
// ---------------------------------------------------------------------------

/// Text-to-video with both the DiT and the VAE decoder on the GPU.
///
/// Holds the CPU DiT alongside the GPU one: the conditioning MLPs (timestep
/// and text embedders) and the patchify/unpatchify reshapes stay on the host,
/// where they cost a few hundred rows once per step against 30 blocks over
/// tens of thousands of tokens.
#[cfg(feature = "metal")]
pub struct GpuWanPipeline {
    pub gpu_dit: super::gpu::GpuWanDit,
    pub vae: WanVae,
    pub gpu_decoder: crate::vae3d::gpu::GpuDecoder3d,
}

#[cfg(feature = "metal")]
impl GpuWanPipeline {
    /// The f32 DiT is consumed: everything the GPU path needs is copied out,
    /// so ~20 GB of host weights is freed on return.
    pub fn new(cpu_dit: WanDit, vae: WanVae) -> Self {
        let gpu_dit = super::gpu::GpuWanDit::from_cpu(&cpu_dit);
        drop(cpu_dit);
        let gpu_decoder = crate::vae3d::gpu::GpuDecoder3d::from_cpu(&vae.decoder);
        Self { gpu_dit, vae, gpu_decoder }
    }

    /// Build without ever materialising the f32 DiT — the memory-efficient
    /// path for real checkpoints.
    pub fn load(
        cfg: super::WanDitConfig,
        dit: &crate::weight::loader::ComponentTensors,
        vae: WanVae,
    ) -> crate::Result<Self> {
        let gpu_dit = super::gpu::GpuWanDit::load(cfg, dit)?;
        let gpu_decoder = crate::vae3d::gpu::GpuDecoder3d::from_cpu(&vae.decoder);
        Ok(Self { gpu_dit, vae, gpu_decoder })
    }

    /// Denoising loop, identical to [`WanPipeline::denoise`] but evaluating
    /// the transformer on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        mut latents: Vec<f32>,
        dims: (usize, usize, usize),
        prompt: &[f32],
        negative: Option<&[f32]>,
        n_ctx: usize,
        mask: &[f32],
        cond: Option<&WanConditioning>,
        cfg: &WanGenConfig,
        mut on_step: impl FnMut(usize, usize, &[f32]),
    ) -> Vec<f32> {
        let sigmas = wan_sigmas(cfg.num_steps, cfg.flow_shift);
        let mut sched = UniPCScheduler::from_sigmas(sigmas);
        let timesteps = sched.timesteps.clone();
        // Hold the conditioned frames from the start: step 0 is told they are
        // clean, so they must actually be the reference before it runs.
        if let Some(c) = cond {
            apply_conditioning(&mut latents, c, self.vae.cfg.z_dim);
        }
        let patch = self.gpu_dit.cfg.patch();

        for (i, &t) in timesteps.iter().enumerate() {
            let ts = expand_timesteps(mask, dims, patch, t);
            let mut pred = self.gpu_dit.forward(&latents, dims, &ts, prompt, n_ctx);
            if let Some(neg) = negative {
                if cfg.guidance_scale > 1.0 && cfg.cfg_applies_at(i, timesteps.len()) {
                    let uncond = self.gpu_dit.forward(&latents, dims, &ts, neg, n_ctx);
                    apply_cfg(&mut pred, &uncond, cfg.guidance_scale);
                }
            }
            latents = sched.step(i, &pred, &latents, None, 1.0, 0.0);
            // Re-pin: the scheduler moves every position, conditioned or not.
            if let Some(c) = cond {
                apply_conditioning(&mut latents, c, self.vae.cfg.z_dim);
            }
            on_step(i + 1, timesteps.len(), &latents);
        }
        latents
    }

    /// Denoised latents -> pixels, decoding on the GPU.
    ///
    /// `post_quant_conv` is a 1x1x1 conv over the latent and stays on the
    /// host; the unpatchify and clamp tail matches `WanVae::decode`.
    pub fn decode(&self, latents: &[f32], dims: (usize, usize, usize)) -> Vol {
        let z_dim = self.vae.cfg.z_dim;
        let mut z = latents.to_vec();
        if self.vae.cfg.latents_mean.len() == z_dim && self.vae.cfg.latents_std.len() == z_dim {
            denormalize_latents(
                &mut z,
                z_dim,
                &self.vae.cfg.latents_mean.clone(),
                &self.vae.cfg.latents_std.clone(),
            );
        }
        let z = Vol::new(z, z_dim, dims.0, dims.1, dims.2);
        let z = self.vae.post_quant_conv.forward(&z, None);
        let raw = self.gpu_decoder.decode_all(&z);
        let mut out = crate::vae3d::unpatchify(&raw, self.vae.cfg.patch_size);
        for v in out.data.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        out
    }
}
