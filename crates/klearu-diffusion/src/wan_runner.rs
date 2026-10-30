//! Self-contained Wan 2.2 TI2V-5B runner: load the checkpoint once, then turn
//! prompts (and optional start/end frames) into video on the Metal GPU.
//!
//! Mirrors [`crate::flux_runner`]: everything is in-memory. Conditioning images
//! arrive as encoded bytes rather than paths and frames come back as pixel
//! buffers, so a caller can run a whole generation without touching disk —
//! which is what incognito mode needs.
//!
//! Gated on `feature = "metal"`; the GPU transformer is the whole point.

#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::pipeline::GeneratedImage;
use crate::vae3d::{Vol, WanVae, WanVaeConfig};
use crate::wan::pipeline::{
    flow_shift_from_checkpoint, GpuWanPipeline, WanConditioning, WanGenConfig,
    DEFAULT_NEGATIVE_PROMPT,
};
use crate::wan::{WanDitConfig, WanTextEncoder};
use crate::weight::ComponentTensors;
use crate::{DiffusionError, Result};

/// A Wan checkpoint directory — the one holding `transformer/`, `vae/`,
/// `text_encoder/`, `tokenizer/` and `scheduler/`.
pub struct WanLoadSpec {
    pub root: PathBuf,
}

/// What to generate.
pub struct WanGenOpts {
    pub prompt: String,
    /// `None` uses Wan's own `sample_neg_prompt`, which is what the model was
    /// tuned against — see [`DEFAULT_NEGATIVE_PROMPT`].
    pub negative_prompt: Option<String>,
    pub steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    /// Pixel frames. Rounded to the VAE's `4k + 1` grid.
    pub num_frames: usize,
    pub width: usize,
    pub height: usize,
    pub fps: usize,
    /// Fraction of the schedule over which classifier-free guidance runs.
    /// `(0.0, 1.0)` applies it everywhere, which is the reference behaviour.
    pub cfg_interval: (f32, f32),
    /// Encoded image bytes (PNG/JPEG/...), resized to the generation size.
    pub start_image: Option<Vec<u8>>,
    pub end_image: Option<Vec<u8>>,
}

impl Default for WanGenOpts {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            steps: 30,
            guidance_scale: 5.0,
            seed: 0,
            // The model's native size and its reference frame rate. Generating
            // below this is the single biggest quality loss available.
            num_frames: 49,
            width: 1280,
            height: 704,
            fps: 24,
            cfg_interval: (0.0, 1.0),
            start_image: None,
            end_image: None,
        }
    }
}

/// Finished video, in memory.
pub struct WanVideo {
    pub frames: Vec<GeneratedImage>,
    pub fps: usize,
    pub width: usize,
    pub height: usize,
}

/// Progress for one denoising step.
pub struct WanProgress {
    pub step: usize,
    pub total: usize,
    pub elapsed_s: f32,
}

pub struct WanRunner {
    text: WanTextEncoder,
    pipe: GpuWanPipeline,
    /// From `scheduler/scheduler_config.json`; 5.0 for TI2V-5B. Hardcoding 3.0
    /// here produces a visibly worse video, so it is read, not assumed.
    flow_shift: f32,
    vae_cpu: WanVae,
}

/// Whether `root` looks like a Wan checkpoint directory.
pub fn is_wan_checkpoint(root: &Path) -> bool {
    root.join("transformer").is_dir()
        && root.join("vae").is_dir()
        && root.join("text_encoder").is_dir()
}

impl WanRunner {
    pub fn load(spec: &WanLoadSpec) -> Result<Self> {
        let root = &spec.root;
        if !is_wan_checkpoint(root) {
            return Err(DiffusionError::Unsupported(format!(
                "{} is not a Wan checkpoint (expected transformer/, vae/ and text_encoder/ inside)",
                root.display()
            )));
        }

        let vae_dir = root.join("vae");
        let vae_cfg = WanVaeConfig::from_json(&read_config(&vae_dir)?)?;
        let vae = WanVae::load(vae_cfg.clone(), &ComponentTensors::open_dir(&vae_dir)?)?;
        // A second CPU copy for encoding conditioning images: the GPU pipeline
        // owns a decoder only, and encoding one still frame is cheap.
        let vae_cpu = WanVae::load(vae_cfg, &ComponentTensors::open_dir(&vae_dir)?)?;

        let te_dir = root.join("text_encoder");
        let text = WanTextEncoder::load(
            &root.join("tokenizer/tokenizer.json"),
            &ComponentTensors::open_dir(&te_dir)?,
        )?;

        let dit_dir = root.join("transformer");
        let dit_cfg = WanDitConfig::from_json(&read_config(&dit_dir)?)?;
        let pipe =
            GpuWanPipeline::load(dit_cfg, &ComponentTensors::open_dir(&dit_dir)?, vae)?;

        let flow_shift = flow_shift_from_checkpoint(root).unwrap_or(5.0);
        Ok(Self { text, pipe, flow_shift, vae_cpu })
    }

    /// Latent dimensions for a pixel size, and the pixel frame count actually
    /// produced.
    ///
    /// The VAE compresses 16x spatially and 4x temporally, the latter causally:
    /// one latent frame covers pixel frame 0, each later one covers four. So
    /// only `4k + 1` frame counts round-trip exactly, and a request for 50
    /// frames yields 49.
    pub fn latent_dims(&self, opts: &WanGenOpts) -> ((usize, usize, usize), usize) {
        let sc = self.vae_cpu.cfg.spatial_compression();
        let tc = self.vae_cpu.cfg.temporal_compression();
        let lt = 1 + opts.num_frames.saturating_sub(1) / tc;
        let lt = lt.max(1);
        (
            (lt, opts.height / sc, opts.width / sc),
            1 + (lt - 1) * tc,
        )
    }

    pub fn generate(
        &self,
        opts: &WanGenOpts,
        mut on_progress: impl FnMut(WanProgress),
        abort: &AtomicBool,
    ) -> Result<WanVideo> {
        let (dims, frames_out) = self.latent_dims(opts);
        let (lt, lh, lw) = dims;
        if lh == 0 || lw == 0 {
            return Err(DiffusionError::Unsupported(format!(
                "{}x{} is smaller than one VAE cell ({}px)",
                opts.width,
                opts.height,
                self.vae_cpu.cfg.spatial_compression()
            )));
        }

        let prompt = self.text.encode(&opts.prompt)?;
        // Only encode the negative branch if guidance will actually use it —
        // umT5 is 5.7B and this is ~30 s of work that `denoise` would discard.
        let negative = if opts.guidance_scale > 1.0 {
            Some(self.text.encode(
                opts.negative_prompt.as_deref().unwrap_or(DEFAULT_NEGATIVE_PROMPT),
            )?)
        } else {
            None
        };
        let n_ctx = crate::wan::MAX_PROMPT_TOKENS;

        // Conditioning frames, if any. Decoded and scaled to the generation
        // size here so the caller never has to know the latent grid.
        let load = |bytes: &Vec<u8>| -> Result<Vol> {
            let chw = crate::image_io::load_image_bytes_resized_chw_f32(
                bytes,
                opts.height,
                opts.width,
            )?;
            Ok(Vol::new(chw, 3, 1, opts.height, opts.width))
        };
        let start = opts.start_image.as_ref().map(load).transpose()?;
        let end = opts.end_image.as_ref().map(load).transpose()?;
        let cond = if start.is_some() || end.is_some() {
            Some(WanConditioning::from_images(
                &self.vae_cpu,
                dims,
                start.as_ref(),
                end.as_ref(),
            ))
        } else {
            None
        };
        let mask = match &cond {
            Some(c) => c.mask.clone(),
            None => vec![1.0_f32; lt * lh * lw],
        };

        let latents = gaussian_latents(
            self.pipe.gpu_dit.cfg.in_channels * lt * lh * lw,
            opts.seed,
        );

        let cfg = WanGenConfig {
            num_steps: opts.steps,
            flow_shift: self.flow_shift,
            guidance_scale: opts.guidance_scale,
            cfg_interval: opts.cfg_interval,
        };
        let t0 = std::time::Instant::now();
        let mut aborted = false;
        let out = self.pipe.denoise(
            latents,
            dims,
            &prompt,
            negative.as_deref(),
            n_ctx,
            &mask,
            cond.as_ref(),
            &cfg,
            |step, total, _latents| {
                if abort.load(Ordering::SeqCst) {
                    aborted = true;
                }
                on_progress(WanProgress {
                    step,
                    total,
                    elapsed_s: t0.elapsed().as_secs_f32(),
                });
            },
        );
        if aborted {
            return Err(DiffusionError::Unsupported("generation aborted".into()));
        }

        let video = self.pipe.decode(&out, dims);
        Ok(WanVideo {
            frames: split_frames(&video, frames_out),
            fps: opts.fps,
            width: video.w,
            height: video.h,
        })
    }
}

fn read_config(dir: &Path) -> Result<String> {
    std::fs::read_to_string(dir.join("config.json")).map_err(|e| {
        DiffusionError::Unsupported(format!("read {}/config.json: {e}", dir.display()))
    })
}

/// Split a `[3, T, H, W]` volume into per-frame CHW images.
fn split_frames(video: &Vol, limit: usize) -> Vec<GeneratedImage> {
    let plane = video.h * video.w;
    let n = video.t.min(limit);
    (0..n)
        .map(|f| {
            let mut rgb_chw = vec![0.0_f32; 3 * plane];
            for c in 0..3 {
                let src = (c * video.t + f) * plane;
                rgb_chw[c * plane..(c + 1) * plane]
                    .copy_from_slice(&video.data[src..src + plane]);
            }
            GeneratedImage { rgb_chw, height: video.h, width: video.w }
        })
        .collect()
}

/// Seeded standard-normal noise, via Box-Muller.
///
/// Diffusion models are trained on normal noise; a uniform draw has the right
/// variance but no tails, which starts the trajectory off-distribution.
fn gaussian_latents(n: usize, seed: u64) -> Vec<f32> {
    let mut s = (seed as u32) ^ 0x9E37_79B9;
    if s == 0 {
        s = 0x1234_5678;
    }
    let mut uniform = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        // Open interval, so ln() below never sees 0.
        ((s >> 8) as f32 + 0.5) / 16_777_216.0
    };
    let mut out = vec![0.0_f32; n];
    for pair in out.chunks_mut(2) {
        let (u1, u2) = (uniform(), uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        pair[0] = r * theta.cos();
        if pair.len() > 1 {
            pair[1] = r * theta.sin();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frame counts are not free-form: the causal VAE only round-trips `4k + 1`.
    ///
    /// Asking for 50 frames and silently getting 49 is fine; asking for 50 and
    /// getting a video whose last frames are garbage is not, which is what
    /// happens if the latent count and the emitted frame count disagree.
    /// End to end: a start frame must actually come out as frame 0.
    ///
    /// Every other test here checks a piece. This one checks the claim the
    /// feature makes to the user — "the video starts from this image" — through
    /// the real runner, the real checkpoint and the real denoise loop. It is
    /// the test that fails if the mask, the encode, the placement or the
    /// re-pinning is wrong, in any combination.
    ///
    /// Two steps at a small size: the point is where frame 0 lands, not
    /// quality.
    #[test]
    #[ignore]
    fn a_start_frame_survives_generation() {
        let (Ok(root), Ok(img_path)) =
            (std::env::var("KLEARU_WAN_ROOT"), std::env::var("KLEARU_START_IMAGE"))
        else {
            eprintln!("set KLEARU_WAN_ROOT and KLEARU_START_IMAGE; skipping");
            return;
        };
        let bytes = std::fs::read(&img_path).expect("read start image");

        let runner = WanRunner::load(&WanLoadSpec { root: PathBuf::from(root) })
            .expect("load wan");
        let opts = WanGenOpts {
            prompt: "a cat on a coastal road".into(),
            steps: 2,
            // Guidance 1.0 skips the negative pass — half the compute, and
            // irrelevant to where the conditioning frame ends up.
            guidance_scale: 1.0,
            seed: 11,
            num_frames: 5,
            width: 320,
            height: 192,
            fps: 24,
            start_image: Some(bytes.clone()),
            ..Default::default()
        };

        let abort = AtomicBool::new(false);
        let video = runner
            .generate(&opts, |p| println!("  step {}/{} {:.1}s", p.step, p.total, p.elapsed_s), &abort)
            .expect("generate");
        assert_eq!(video.frames.len(), 5, "5 frames requested and 5 sit on the 4k+1 grid");

        // What the conditioning image looks like at the generation size.
        let want = crate::image_io::load_image_bytes_resized_chw_f32(
            &bytes,
            opts.height,
            opts.width,
        )
        .expect("load reference");

        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
            dot / (na * nb + 1e-12)
        };
        let first = cos(&video.frames[0].rgb_chw, &want);
        let last = cos(&video.frames[4].rgb_chw, &want);
        println!("frame0 vs start image: cos_sim={first:.4}");
        println!("frame4 vs start image: cos_sim={last:.4}");

        // A VAE round-trip plus two denoising steps is lossy, so this is not an
        // equality check. But an unconditioned frame 0 at 2 steps is mostly
        // noise and lands nowhere near here.
        assert!(first > 0.85, "start frame did not survive generation: cos_sim {first}");
        // And it must be *conditioning*, not the whole video being the image:
        // later frames are free to move away.
        assert!(first > last, "frame 0 should match the start image more than frame 4 does");

        // The other end, on the same loaded runner. End conditioning is
        // off-distribution for TI2V-5B — it was trained to condition on a first
        // frame — so the bar is lower than for the start frame, but the held
        // slot must still be recognisably the image, and must beat frame 0.
        let end_opts = WanGenOpts {
            start_image: None,
            end_image: Some(bytes),
            ..opts
        };
        let video = runner
            .generate(&end_opts, |_| {}, &abort)
            .expect("generate with end frame");
        let first = cos(&video.frames[0].rgb_chw, &want);
        let last = cos(&video.frames[4].rgb_chw, &want);
        println!("end-conditioned frame0: cos_sim={first:.4}");
        println!("end-conditioned frame4: cos_sim={last:.4}");
        assert!(last > 0.85, "end frame did not survive generation: cos_sim {last}");
        assert!(last > first, "frame 4 should match the end image more than frame 0 does");
    }

    #[test]
    fn frame_counts_round_to_the_vae_grid() {
        // (requested, expected latent frames, expected emitted frames)
        for (req, lt, out) in
            [(1, 1, 1), (2, 1, 1), (5, 2, 5), (49, 13, 49), (50, 13, 49), (53, 14, 53)]
        {
            let latent = 1 + (req as usize).saturating_sub(1) / 4;
            let emitted = 1 + (latent - 1) * 4;
            assert_eq!(latent, lt, "latent frames for {req} requested");
            assert_eq!(emitted, out, "emitted frames for {req} requested");
            assert_eq!(emitted % 4, 1, "emitted count must sit on the 4k+1 grid");
            assert!(emitted <= req.max(1), "must never emit more than requested");
        }
    }

    #[test]
    fn gaussian_latents_are_standard_normal_and_seed_dependent() {
        let a = gaussian_latents(20_000, 7);
        let b = gaussian_latents(20_000, 8);
        assert_ne!(a, b, "different seeds must give different noise");
        assert_eq!(a, gaussian_latents(20_000, 7), "same seed must reproduce");

        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.05, "mean {mean} should be ~0");
        assert!((var.sqrt() - 1.0).abs() < 0.05, "std {} should be ~1", var.sqrt());
        // Box-Muller has tails; a uniform draw would not reach here.
        assert!(a.iter().any(|v| v.abs() > 3.0), "normal noise should have tails");
    }

    /// An odd length must not read past the end of the buffer.
    #[test]
    fn gaussian_latents_handles_an_odd_count() {
        let v = gaussian_latents(7, 3);
        assert_eq!(v.len(), 7);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn split_frames_deinterleaves_channels_and_respects_the_limit() {
        // [3, 2, 1, 2]: value encodes channel and frame so a transposed copy
        // would be caught rather than merely a wrong length.
        let (t, h, w) = (2usize, 1usize, 2usize);
        let mut data = vec![0.0_f32; 3 * t * h * w];
        for c in 0..3 {
            for f in 0..t {
                for p in 0..h * w {
                    data[(c * t + f) * h * w + p] = (c * 10 + f) as f32;
                }
            }
        }
        let vol = Vol::new(data, 3, t, h, w);

        let frames = split_frames(&vol, 8);
        assert_eq!(frames.len(), t, "limit above the frame count must not invent frames");
        for (f, frame) in frames.iter().enumerate() {
            assert_eq!(frame.height, h);
            assert_eq!(frame.width, w);
            for c in 0..3 {
                assert_eq!(
                    frame.rgb_chw[c * h * w],
                    (c * 10 + f) as f32,
                    "frame {f} channel {c}"
                );
            }
        }
        assert_eq!(split_frames(&vol, 1).len(), 1, "limit must truncate");
    }
}
