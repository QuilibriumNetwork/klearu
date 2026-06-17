//! Diffusion engine.
//!
//! Two backends, picked per-request:
//!
//!   * **In-process** ([`generate_in_process`]) — calls
//!     `klearu_diffusion::pipeline::generate` directly against
//!     resident-in-memory `SdResidents`. The checkpoint is loaded once
//!     and reused across calls. No disk writes for the conditioning
//!     bias (passed as `Vec<f32>`) and the output bytes can be returned
//!     to the caller without ever touching the filesystem — this is the
//!     path that lets incognito mode generate images. Currently covers
//!     **txt2img only** (`req.input_path` / `mask_path` unset).
//!
//!   * **Subprocess** ([`generate`]) — spawns `sd_generate` for the
//!     paths the in-process pipeline doesn't yet cover (img2img, naive +
//!     native inpaint, full ControlNet workflows). Same stderr-parsing
//!     progress events as before.
//!
//! Callers pick the backend explicitly; the routing decision lives in
//! `commands::image`.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use klearu_diffusion::pipeline::{
    self as kpipe, GenerateProgress as PipelineProgress, GenerateRequest as PipelineReq,
    LoadOptions, SdResidents,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

use crate::error::{DesktopError, Result};

// ---- public API ------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffusionConfig {
    /// Path to the `sd_generate` binary. If `None`, we'll try to auto-locate
    /// (env `KLEARU_SD_GENERATE`, sibling of current exe, target/{release,debug},
    /// then `which`).
    pub sd_generate_path: Option<PathBuf>,
    /// `--single-file FILE.safetensors` — the diffusion checkpoint.
    pub checkpoint: Option<PathBuf>,
    /// `--tokenizer FILE.json` (CLIP-L).
    pub tokenizer_l: Option<PathBuf>,
    /// `--tokenizer-2 FILE.json` (CLIP-G; SDXL only).
    pub tokenizer_g: Option<PathBuf>,
    /// Output directory. Defaults to `~/.klearu/images/`.
    pub output_dir: Option<PathBuf>,
    /// Path to an AnimateDiff motion adapter `.safetensors` for video
    /// generation. E.g. `mm_sdxl_v10_beta.safetensors`.
    pub motion_adapter: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Optional conversation tying — image's row gets this fk for provenance.
    #[serde(default)]
    pub conversation_id: Option<i64>,

    // ---- img2img / inpaint -------------------------------------------------
    // When `input_path` is set sd_generate runs in img2img / inpaint mode.
    // Adding `mask_path` switches to mask-aware inpaint. The combination of
    // (input only) / (input + mask) / (input + mask + 9-ch UNet) is decided
    // by sd_generate based on the checkpoint and the flags it sees.

    /// `--input FILE.png` — source image for img2img / inpaint.
    #[serde(default)]
    pub input_path: Option<PathBuf>,
    /// `--mask FILE.png` — grayscale mask (white = regenerate, black = keep).
    /// Requires `input_path` to be set.
    #[serde(default)]
    pub mask_path: Option<PathBuf>,
    /// `--strength F` — denoise strength in [0,1] (only honored when
    /// `input_path` is set). 0.0 = no change, 1.0 = full regen.
    #[serde(default)]
    pub strength: Option<f32>,

    // ---- reference editing (HiDream-O1) ------------------------------------
    /// Reference images as raw PNG/JPEG bytes (one per image), supplied
    /// directly by the frontend. When non-empty, HiDream routes through the
    /// native reference-editing path (vision encoder + DeepStack). Bytes
    /// rather than paths so it works incognito (no filesystem round-trip).
    #[serde(default)]
    pub ref_image_bytes: Option<Vec<Vec<u8>>>,
    /// Reference images as file paths (REPL / file-based callers). Read into
    /// bytes by the command; ignored when `ref_image_bytes` is set.
    #[serde(default)]
    pub ref_image_paths: Option<Vec<PathBuf>>,

    /// `--cond-bias FILE.bin` — sidecar of `cross_attention_dim` little-
    /// endian f32s added uniformly to every position of the cond CLIP
    /// embedding before cross-attention. Comes from the per-format
    /// `image_clip_*` user vector × α.
    #[serde(default)]
    pub cond_bias_path: Option<PathBuf>,
}

/// Streaming event emitted while a job is running. Forwarded to the
/// frontend over a Tauri Channel by `commands::image::image_generate`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenEvent {
    /// Job accepted; the subprocess has started.
    Started { job_id: u64, out_path: String, pid: Option<u32> },
    /// One denoising step finished. `step` is 1-indexed; `total` is the
    /// announced step count. Both come straight from the subprocess.
    Step { job_id: u64, step: u32, total: u32, elapsed_s: f32 },
    /// Arbitrary subprocess log line (anything we didn't parse).
    Log { job_id: u64, line: String },
    /// Subprocess wrote a PNG. `path` is absolute.
    Done { job_id: u64, path: String },
    /// User asked us to stop the job; we killed the subprocess.
    Aborted { job_id: u64 },
    /// Subprocess exited with a non-zero status or stderr emitted a hard
    /// error. `message` is the most informative line we captured.
    Error { job_id: u64, code: &'static str, message: String },
}

/// In-flight job handle. Kept in `DiffusionEngine::running` so `abort()`
/// can kill it from another thread.
struct RunningJob {
    job_id: u64,
    child: Child,
    /// Cancellation signal. The reader task listens for this and a kill
    /// follows immediately. We don't try to be polite — sd_generate's own
    /// Esc handler doesn't apply when stdin is a pipe.
    cancel: mpsc::Sender<()>,
}

/// Long-lived engine state. Holds the config (read by `generate`), the
/// currently-running job (if any), and — when the in-process path is in
/// use — a resident `SdResidents` loaded from the current checkpoint.
/// Cloning is cheap because everything is behind `Arc`s.
#[derive(Default, Clone)]
pub struct DiffusionEngine {
    config: Arc<Mutex<DiffusionConfig>>,
    running: Arc<Mutex<Option<RunningJob>>>,
    next_id: Arc<Mutex<u64>>,
    /// Resident pipeline state for the in-process path. Lazy-loaded on
    /// first `generate_in_process` call; reloaded when the checkpoint /
    /// tokenizer config changes. `Option<Box<>>` keeps the type tractable
    /// — `SdResidents` is large (gigabytes of weights) and not Clone.
    residents: Arc<Mutex<Option<LoadedResidents>>>,
    /// Cooperative cancel flag for the in-process denoise loop.
    in_proc_abort: Arc<AtomicBool>,
    /// Resident HiDream-O1 runner (macOS/Metal only). Cached separately from
    /// `residents` because it's a distinct model family (Qwen3-VL, no VAE).
    #[cfg(target_os = "macos")]
    hidream: Arc<Mutex<Option<LoadedHiDream>>>,
}

/// A loaded HiDream-O1 runner paired with its checkpoint signature.
#[cfg(target_os = "macos")]
struct LoadedHiDream {
    runner: klearu_diffusion::hidream::runner::HiDreamRunner,
    signature: String,
}

/// `SdResidents` paired with the config signature it was loaded with —
/// when the signature stops matching the engine config, we drop and reload.
struct LoadedResidents {
    inner: SdResidents,
    signature: String,
}

impl DiffusionEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_config(&self, c: DiffusionConfig) {
        *self.config.lock() = c;
    }

    pub fn config(&self) -> DiffusionConfig {
        self.config.lock().clone()
    }

    /// Returns whether a generation is currently running.
    pub fn busy(&self) -> bool {
        self.running.lock().is_some()
    }

    /// Spawn a generation. `events` receives `GenEvent`s as the job
    /// progresses. The future resolves when the subprocess exits (success
    /// or failure) — caller should `tokio::spawn` it.
    pub async fn generate(
        self: &Self,
        req: GenerateRequest,
        events: mpsc::UnboundedSender<GenEvent>,
        data_dir: PathBuf,
    ) -> Result<()> {
        if self.busy() {
            return Err(DesktopError::engine(
                "another generation is already running — call image_abort first",
            ));
        }

        let cfg = self.config.lock().clone();
        let binary = resolve_binary(cfg.sd_generate_path.as_deref())?;
        let checkpoint = cfg
            .checkpoint
            .as_deref()
            .ok_or_else(|| DesktopError::bad_input("image.checkpoint not configured"))?;
        let tokenizer_l = cfg
            .tokenizer_l
            .as_deref()
            .ok_or_else(|| DesktopError::bad_input("image.tokenizer_l not configured"))?;
        let out_dir = cfg
            .output_dir
            .clone()
            .unwrap_or_else(|| data_dir.join("images"));
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| DesktopError::Io(format!("create {}: {e}", out_dir.display())))?;

        let job_id = {
            let mut n = self.next_id.lock();
            *n += 1;
            *n
        };
        let ts = now_secs();
        let seed_for_name = req.seed.unwrap_or(ts as u64);
        let out_path = out_dir.join(format!("{ts}-{seed_for_name}.png"));

        let mut cmd = Command::new(&binary);
        cmd.arg("--single-file").arg(checkpoint);
        cmd.arg("--tokenizer").arg(tokenizer_l);
        if let Some(t2) = &cfg.tokenizer_g {
            cmd.arg("--tokenizer-2").arg(t2);
        }
        cmd.arg("--prompt").arg(&req.prompt);
        if let Some(neg) = req.negative_prompt.as_deref().filter(|s| !s.is_empty()) {
            cmd.arg("--negative-prompt").arg(neg);
        }
        if let Some(s) = req.steps {
            cmd.arg("--steps").arg(s.to_string());
        }
        if let Some(c) = req.cfg_scale {
            cmd.arg("--cfg-scale").arg(c.to_string());
        }
        if let Some(s) = req.seed {
            cmd.arg("--seed").arg(s.to_string());
        }
        if let Some(input) = req.input_path.as_deref() {
            cmd.arg("--input").arg(input);
            if let Some(mask) = req.mask_path.as_deref() {
                cmd.arg("--mask").arg(mask);
            }
            if let Some(s) = req.strength {
                cmd.arg("--strength").arg(s.to_string());
            }
        }
        if let Some(bias) = req.cond_bias_path.as_deref() {
            cmd.arg("--cond-bias").arg(bias);
        }
        cmd.arg("--out").arg(&out_path);
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());
        cmd.kill_on_drop(true);

        tracing::info!(?cmd, "spawning sd_generate");
        let mut child = cmd
            .spawn()
            .map_err(|e| DesktopError::engine(format!("spawn sd_generate: {e}")))?;
        let pid = child.id();
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| DesktopError::engine("subprocess had no stderr pipe"))?;

        let (cancel_tx, mut cancel_rx) = mpsc::channel::<()>(1);

        *self.running.lock() = Some(RunningJob {
            job_id,
            child,
            cancel: cancel_tx,
        });

        let _ = events.send(GenEvent::Started {
            job_id,
            out_path: out_path.to_string_lossy().into_owned(),
            pid,
        });

        // Reader loop: parse stderr lines until EOF or cancellation.
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        let mut last_error: Option<String> = None;

        loop {
            tokio::select! {
                line = lines.next_line() => {
                    match line {
                        Ok(Some(line)) => {
                            if let Some(ev) = parse_progress(job_id, &line) {
                                let _ = events.send(ev);
                            } else if line.to_lowercase().contains("error") {
                                last_error = Some(line.clone());
                                let _ = events.send(GenEvent::Log { job_id, line });
                            } else {
                                let _ = events.send(GenEvent::Log { job_id, line });
                            }
                        }
                        Ok(None) => break,
                        Err(e) => {
                            last_error = Some(format!("stderr read: {e}"));
                            break;
                        }
                    }
                }
                _ = cancel_rx.recv() => {
                    // Kill the child; the next stderr read will EOF.
                    if let Some(mut job) = self.running.lock().take() {
                        let _ = job.child.start_kill();
                    }
                    let _ = events.send(GenEvent::Aborted { job_id });
                    return Ok(());
                }
            }
        }

        // Stderr closed; wait for exit status. If we already aborted above
        // we'd have returned; getting here means the subprocess exited on
        // its own.
        //
        // Important: take the job out from under the parking_lot guard
        // BEFORE awaiting. MutexGuard isn't Send, so holding it across .await
        // would make the whole future !Send and Tauri requires Send futures.
        let job_opt = self.running.lock().take();
        let mut job = match job_opt {
            Some(j) => j,
            None => return Ok(()),
        };
        let status = job
            .child
            .wait()
            .await
            .map_err(|e| DesktopError::engine(format!("wait sd_generate: {e}")))?;

        if status.success() && out_path.exists() {
            let _ = events.send(GenEvent::Done {
                job_id,
                path: out_path.to_string_lossy().into_owned(),
            });
            Ok(())
        } else {
            let message = last_error.unwrap_or_else(|| {
                format!(
                    "sd_generate exited with {}",
                    status
                        .code()
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "signal".to_string())
                )
            });
            let _ = events.send(GenEvent::Error {
                job_id,
                code: "subprocess",
                message: message.clone(),
            });
            Err(DesktopError::engine(message))
        }
    }

    /// Request cancellation of the in-flight job, if any. Covers both
    /// backends:
    ///   * Subprocess: send through the cancel channel; the reader task
    ///     kills the child + drains stderr.
    ///   * In-process: flip the shared `AtomicBool` that
    ///     `pipeline::generate` polls between denoise steps. The current
    ///     UNet forward finishes (we don't kill mid-forward — that would
    ///     leak GPU state), then the loop returns `DiffusionError::Aborted`.
    /// Idempotent: safe to call when nothing is running.
    pub fn abort(&self) {
        if let Some(job) = self.running.lock().as_ref() {
            let _ = job.cancel.try_send(());
            tracing::info!(job_id = job.job_id, "abort requested (subprocess)");
        }
        // In-process flag — set even when subprocess wasn't running, since
        // the in-process path doesn't touch `self.running`. Spurious sets
        // are harmless: `generate_in_process` clears the flag at the
        // start of every new call.
        self.in_proc_abort
            .store(true, std::sync::atomic::Ordering::SeqCst);
        tracing::info!("abort requested (in-process flag set)");
    }

    pub fn current_job_id(&self) -> Option<u64> {
        self.running.lock().as_ref().map(|j| j.job_id)
    }

    /// In-process txt2img generation. Loads the checkpoint into resident
    /// memory on first call (multi-GB; ~10-15s), reuses it on subsequent
    /// calls. Returns the RGB-CHW f32 buffer the VAE produced, plus the
    /// final image dimensions.
    ///
    /// **Does not touch disk.** All inputs (cond_bias, prompts, sampler
    /// settings) come in as values; the output is returned by value. The
    /// caller decides whether to encode to PNG on disk or to a data URI
    /// for in-memory delivery. This is the path that makes image
    /// generation possible inside an incognito session.
    pub async fn generate_in_process(
        &self,
        req: InProcessRequest,
        events: mpsc::UnboundedSender<GenEvent>,
    ) -> Result<InProcessResult> {
        if self.busy() {
            return Err(DesktopError::engine(
                "another generation is already running — call image_abort first",
            ));
        }

        let cfg = self.config.lock().clone();
        let checkpoint = cfg
            .checkpoint
            .clone()
            .ok_or_else(|| DesktopError::bad_input("image.checkpoint not configured"))?;

        let job_id = {
            let mut n = self.next_id.lock();
            *n += 1;
            *n
        };
        self.in_proc_abort
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let _ = events.send(GenEvent::Started {
            job_id,
            out_path: String::new(), // no path for in-process; caller assigns
            pid: None,
        });

        // HiDream-O1 is a directory model (Qwen3-VL, no VAE, self-contained
        // tokenizer) — detect and dispatch to its own runner before the
        // SD/SDXL CLIP-tokenizer requirement.
        #[cfg(target_os = "macos")]
        if klearu_diffusion::hidream::runner::is_hidream_dir(&checkpoint) {
            let signature = checkpoint.display().to_string();
            return self
                .generate_hidream_in_process(checkpoint, signature, req, job_id, events)
                .await;
        }

        let tokenizer_l = cfg
            .tokenizer_l
            .clone()
            .ok_or_else(|| DesktopError::bad_input("image.tokenizer_l not configured"))?;
        let tokenizer_g = cfg.tokenizer_g.clone();
        let load_opts = LoadOptions {
            checkpoint: checkpoint.clone(),
            tokenizer_l: Some(tokenizer_l.clone()),
            tokenizer_g: tokenizer_g.clone(),
            controlnet: None, // ControlNet via the in-process path: future work
            force_variant: None,
            gpu_residence: true,
        };
        let signature = format!(
            "{}|{}|{}",
            checkpoint.display(),
            tokenizer_l.display(),
            tokenizer_g.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
        );

        let residents_handle = self.residents.clone();
        let abort = self.in_proc_abort.clone();
        let events_tx = events.clone();

        // spawn_blocking: pipeline::generate runs the denoise loop and is
        // CPU/GPU-heavy. Off the tauri event-loop thread it goes.
        let total_steps = req.steps;
        let result = tokio::task::spawn_blocking(move || -> Result<InProcessResult> {
            // Acquire (or replace) residents under lock — mutated during
            // generation (KV caches, scheduler timesteps) so we hold it
            // for the duration of one generation. Concurrent in-process
            // calls naturally serialize here, which matches the
            // single-pipeline reality of an open checkpoint.
            let mut guard = residents_handle.lock();
            let needs_load = match guard.as_ref() {
                Some(loaded) => loaded.signature != signature,
                None => true,
            };
            if needs_load {
                *guard = None; // drop old residents first (free big buffers)
                let inner = SdResidents::load(&load_opts)
                    .map_err(|e| DesktopError::engine(format!("load residents: {e}")))?;
                *guard = Some(LoadedResidents { inner, signature });
            }
            let residents = &mut guard.as_mut().unwrap().inner;

            let pipe_req = PipelineReq {
                prompt: req.prompt,
                negative_prompt: req.negative_prompt,
                steps: total_steps,
                cfg_scale: req.cfg_scale,
                seed: req.seed,
                input_image_bytes: req.input_image_bytes,
                mask_bytes: req.mask_bytes,
                strength: req.strength,
                control_image_bytes: req.control_image_bytes,
                controlnet_weight: if req.controlnet_weight == 0.0 { 1.0 } else { req.controlnet_weight },
                controlnet_start: req.controlnet_start,
                controlnet_end: if req.controlnet_end == 0.0 { 1.0 } else { req.controlnet_end },
                cond_bias: req.cond_bias,
            };
            let events_for_cb = events_tx.clone();
            let img = kpipe::generate(
                residents,
                &pipe_req,
                |p: PipelineProgress| {
                    let _ = events_for_cb.send(GenEvent::Step {
                        job_id,
                        step: p.step as u32,
                        total: p.total as u32,
                        elapsed_s: p.elapsed_s,
                    });
                },
                &abort,
            )
            .map_err(|e| DesktopError::engine(format!("pipeline::generate: {e}")))?;
            Ok(InProcessResult {
                rgb_chw: img.rgb_chw,
                height: img.height,
                width: img.width,
                job_id,
            })
        })
        .await
        .map_err(|e| DesktopError::engine(format!("join: {e}")))??;

        let _ = events.send(GenEvent::Done {
            job_id,
            path: String::new(), // populated by the caller after PNG encode
        });
        Ok(result)
    }

    /// In-process generation for a HiDream-O1 checkpoint directory. Loads (and
    /// caches) a `HiDreamRunner`, runs the denoise loop with progress + abort,
    /// and returns the image as planar `[C,H,W]` in `[-1,1]` (same convention as
    /// the SD path, so the caller's PNG encoder works unchanged).
    #[cfg(target_os = "macos")]
    async fn generate_hidream_in_process(
        &self,
        checkpoint: PathBuf,
        signature: String,
        req: InProcessRequest,
        job_id: u64,
        events: mpsc::UnboundedSender<GenEvent>,
    ) -> Result<InProcessResult> {
        use klearu_diffusion::hidream::runner::{HiDreamGenOpts, HiDreamRunner};

        let handle = self.hidream.clone();
        let residents_handle = self.residents.clone();
        let abort = self.in_proc_abort.clone();
        let events_tx = events.clone();
        let result = tokio::task::spawn_blocking(move || -> Result<InProcessResult> {
            let mut guard = handle.lock();
            let needs_load = match guard.as_ref() {
                Some(l) => l.signature != signature,
                None => true,
            };
            if needs_load {
                *guard = None; // free old HiDream weights first
                *residents_handle.lock() = None; // free any SD residents to make room
                let runner = HiDreamRunner::load(&checkpoint)
                    .map_err(|e| DesktopError::engine(format!("load hidream: {e}")))?;
                *guard = Some(LoadedHiDream { runner, signature });
            }
            let runner = &guard.as_ref().unwrap().runner;

            let opts = HiDreamGenOpts {
                prompt: req.prompt,
                negative_prompt: req.negative_prompt,
                // The desktop request has no resolution field yet; 1024² is a
                // good default (2048² is ~4× slower). cfg_scale>1 → full/UniPC.
                height: 1024,
                width: 1024,
                steps: req.steps,
                guidance: req.cfg_scale,
                seed: req.seed,
                // img2img (input image) / inpaint (input + mask) flow through.
                input_image_bytes: req.input_image_bytes,
                mask_bytes: req.mask_bytes,
                strength: req.strength,
                ref_image_bytes: req.ref_image_bytes,
                // Snap to the model's 2048²-native resolution for sharp output
                // (the 1024² request above only sets the square aspect ratio).
                snap_resolution: true,
            };
            let t0 = std::time::Instant::now();
            let img = runner
                .generate(
                    &opts,
                    |step, total| {
                        let _ = events_tx.send(GenEvent::Step {
                            job_id,
                            step: step as u32,
                            total: total as u32,
                            elapsed_s: t0.elapsed().as_secs_f32(),
                        });
                    },
                    &abort,
                )
                .map_err(|e| DesktopError::engine(format!("hidream generate: {e}")))?;
            Ok(InProcessResult { rgb_chw: img.chw, height: img.height, width: img.width, job_id })
        })
        .await
        .map_err(|e| DesktopError::engine(format!("join: {e}")))??;

        let _ = events.send(GenEvent::Done { job_id, path: String::new() });
        Ok(result)
    }

    /// Generate a video clip using AnimateDiff temporal attention.
    /// Requires a motion adapter `.safetensors` to be configured in
    /// settings (`image.motion_adapter`). Returns MP4 bytes.
    pub async fn generate_video_in_process(
        &self,
        req: VideoRequest,
        events: mpsc::UnboundedSender<GenEvent>,
    ) -> Result<VideoResult> {
        if self.busy() {
            return Err(DesktopError::engine("another generation is already running"));
        }
        let cfg = self.config.lock().clone();
        let checkpoint = cfg.checkpoint.clone()
            .ok_or_else(|| DesktopError::bad_input("image.checkpoint not configured"))?;
        let tokenizer_l = cfg.tokenizer_l.clone()
            .ok_or_else(|| DesktopError::bad_input("image.tokenizer_l not configured"))?;
        let tokenizer_g = cfg.tokenizer_g.clone();
        let load_opts = LoadOptions {
            checkpoint: checkpoint.clone(),
            tokenizer_l: Some(tokenizer_l.clone()),
            tokenizer_g: tokenizer_g.clone(),
            controlnet: None,
            force_variant: None,
            gpu_residence: true, // per-frame GPU spatial via proven image-gen path
        };
        let signature = format!("{}|{}|{}",
            checkpoint.display(), tokenizer_l.display(),
            tokenizer_g.as_ref().map(|p| p.display().to_string()).unwrap_or_default());

        let job_id = { let mut n = self.next_id.lock(); *n += 1; *n };
        self.in_proc_abort.store(false, std::sync::atomic::Ordering::SeqCst);
        let _ = events.send(GenEvent::Started {
            job_id, out_path: String::new(), pid: None,
        });

        let residents_handle = self.residents.clone();
        let abort = self.in_proc_abort.clone();
        let events_tx = events.clone();
        let motion_path_opt: Option<std::path::PathBuf> = self.config.lock()
            .motion_adapter.clone();

        let result = tokio::task::spawn_blocking(move || -> Result<VideoResult> {
            let mut guard = residents_handle.lock();
            let needs_load = match guard.as_ref() {
                Some(l) => l.signature != signature,
                None => true,
            };
            if needs_load {
                *guard = None;
                let inner = SdResidents::load(&load_opts)
                    .map_err(|e| DesktopError::engine(format!("load: {e}")))?;
                *guard = Some(LoadedResidents { inner, signature });
            }
            let residents = &mut guard.as_mut().unwrap().inner;

            // Load motion adapter.
            let motion_path = motion_path_opt
                .ok_or_else(|| DesktopError::bad_input(
                    "image.motion_adapter not configured — set the path to mm_sdxl_v10_beta.safetensors"
                ))?;
            let max_frames = req.num_frames.max(32);
            let mut motion = klearu_diffusion::animatediff::MotionModules::sdxl_default(max_frames);
            klearu_diffusion::animatediff::weight::load_motion_adapter(
                &mut motion, &motion_path,
            ).map_err(|e| DesktopError::engine(format!("load motion adapter: {e}")))?;

            let video_req = klearu_diffusion::pipeline::VideoRequest {
                prompt: req.prompt,
                negative_prompt: req.negative_prompt,
                steps: req.steps,
                cfg_scale: req.cfg_scale,
                seed: req.seed,
                num_frames: req.num_frames,
                fps: req.fps,
                cond_bias: req.cond_bias,
            };
            let ev = events_tx.clone();
            let video = klearu_diffusion::pipeline::generate_video(
                residents,
                &motion,
                &video_req,
                |p: klearu_diffusion::pipeline::GenerateProgress| {
                    let _ = ev.send(GenEvent::Step {
                        job_id,
                        step: p.step as u32,
                        total: p.total as u32,
                        elapsed_s: p.elapsed_s,
                    });
                },
                &abort,
            ).map_err(|e| DesktopError::engine(format!("generate_video: {e}")))?;

            // Encode frames → MP4 via ffmpeg (in-memory).
            let mp4 = klearu_diffusion::video_io::encode_mp4(&video.frames, video.fps, None)
                .map_err(|e| DesktopError::engine(format!("encode mp4: {e}")))?;

            Ok(VideoResult {
                mp4_bytes: mp4,
                num_frames: video.frames.len(),
                fps: video.fps,
                width: video.frames.first().map(|f| f.width).unwrap_or(0),
                height: video.frames.first().map(|f| f.height).unwrap_or(0),
                job_id,
            })
        })
        .await
        .map_err(|e| DesktopError::engine(format!("join: {e}")))?
        ?;

        let _ = events.send(GenEvent::Done {
            job_id, path: String::new(),
        });
        Ok(result)
    }

    /// Whether the in-process residents are currently loaded.
    pub fn residents_loaded(&self) -> bool {
        self.residents.lock().is_some()
    }

    /// Drop the in-process residents — frees the GBs of weights. Useful
    /// when the user changes checkpoints or wants to reclaim memory.
    pub fn unload_residents(&self) {
        *self.residents.lock() = None;
    }
}

/// Parameters for video generation via AnimateDiff.
#[derive(Debug, Clone, Default)]
pub struct VideoRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub steps: usize,
    pub cfg_scale: f32,
    pub seed: u64,
    pub num_frames: usize,
    pub fps: usize,
    pub cond_bias: Option<Vec<f32>>,
}

/// Result of a video generation.
#[derive(Debug, Clone)]
pub struct VideoResult {
    /// MP4 bytes (in-memory encoded via ffmpeg).
    pub mp4_bytes: Vec<u8>,
    pub num_frames: usize,
    pub fps: usize,
    pub width: usize,
    pub height: usize,
    pub job_id: u64,
}

/// Parameters for [`DiffusionEngine::generate_in_process`].
///
/// Mirrors `klearu_diffusion::pipeline::GenerateRequest` 1:1 (txt2img,
/// img2img, inpaint, native-inpaint, ControlNet all live here) — the
/// indirection just keeps the desktop crate's public API decoupled from
/// the library crate's, in case either grows fields independently.
#[derive(Debug, Clone, Default)]
pub struct InProcessRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub steps: usize,
    pub cfg_scale: f32,
    pub seed: u64,
    /// PNG/JPEG bytes of the source image (img2img / inpaint). `None`
    /// means txt2img.
    pub input_image_bytes: Option<Vec<u8>>,
    /// PNG bytes of the grayscale mask. Requires `input_image_bytes` set.
    pub mask_bytes: Option<Vec<u8>>,
    /// Denoise strength for img2img mode, 0..=1.
    pub strength: f32,
    /// Reference images (PNG/JPEG bytes) for HiDream native reference editing.
    /// Empty = no reference editing.
    pub ref_image_bytes: Vec<Vec<u8>>,
    /// Bytes of the ControlNet conditioning image.
    pub control_image_bytes: Option<Vec<u8>>,
    pub controlnet_weight: f32,
    pub controlnet_start: f32,
    pub controlnet_end: f32,
    pub cond_bias: Option<Vec<f32>>,
}

/// Returned by [`DiffusionEngine::generate_in_process`]. Bytes-in-memory —
/// the caller writes a PNG to disk or to a data URI as appropriate for
/// the active session (incognito vs normal).
#[derive(Debug, Clone)]
pub struct InProcessResult {
    pub rgb_chw: Vec<f32>,
    pub height: usize,
    pub width: usize,
    pub job_id: u64,
}

// ---- helpers ---------------------------------------------------------------

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Find the `sd_generate` binary. Search order:
///   1. `override` arg (typically from the app's settings)
///   2. env var `KLEARU_SD_GENERATE`
///   3. sibling of the current executable (release builds bundle alongside)
///   4. `target/{release,debug}/sd_generate` walking up from current_exe
///   5. `which sd_generate`
fn resolve_binary(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        if p.exists() { return Ok(p.to_path_buf()); }
        return Err(DesktopError::bad_input(format!(
            "image.sd_generate_path is set but not found: {}",
            p.display()
        )));
    }
    if let Ok(env) = std::env::var("KLEARU_SD_GENERATE") {
        let p = PathBuf::from(env);
        if p.exists() { return Ok(p); }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let sibling = parent.join("sd_generate");
            if sibling.exists() { return Ok(sibling); }
            // When running via `cargo tauri dev` the exe lives at
            // .../target/debug/klearu — sibling sd_generate is at .../target/debug/sd_generate.
            // Walk up looking for `target/{debug,release}/sd_generate`.
            for ancestor in parent.ancestors().take(4) {
                for profile in ["release", "debug"] {
                    let candidate = ancestor.join("target").join(profile).join("sd_generate");
                    if candidate.exists() { return Ok(candidate); }
                }
            }
        }
    }
    // PATH lookup as a last resort.
    if let Ok(out) = std::process::Command::new("which")
        .arg("sd_generate")
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                let p = PathBuf::from(s);
                if p.exists() { return Ok(p); }
            }
        }
    }
    Err(DesktopError::bad_input(
        "sd_generate not found. Configure image.sd_generate_path or set KLEARU_SD_GENERATE.",
    ))
}

/// Parse a single stderr line into a structured event, or return None if
/// the line is informational (caller turns it into a Log event).
///
/// Recognized patterns (per `sd_generate`'s `eraw_println!` output):
///   - `  step N/M (t=...) [Ts]`
///   - `step N/M (t=...):`
///   - `wrote /path/to/file.png`
fn parse_progress(job_id: u64, line: &str) -> Option<GenEvent> {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed.strip_prefix("step ") {
        // "N/M (t=...)..." — parse N/M.
        let slash = rest.find('/')?;
        let after_slash = &rest[slash + 1..];
        let space = after_slash.find(|c: char| !c.is_ascii_digit())?;
        let step_n: u32 = rest[..slash].parse().ok()?;
        let total: u32 = after_slash[..space].parse().ok()?;
        // Extract `[T.Ts]` if present, for "elapsed".
        let elapsed_s = parse_bracketed_secs(line).unwrap_or(0.0);
        return Some(GenEvent::Step {
            job_id,
            step: step_n,
            total,
            elapsed_s,
        });
    }
    if let Some(rest) = trimmed.strip_prefix("wrote ") {
        return Some(GenEvent::Done {
            job_id,
            path: rest.to_string(),
        });
    }
    None
}

fn parse_bracketed_secs(line: &str) -> Option<f32> {
    let lb = line.rfind('[')?;
    let rb = line[lb + 1..].find(']')? + lb + 1;
    let s = &line[lb + 1..rb];
    s.strip_suffix('s')?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_step_line() {
        let ev = parse_progress(7, "  step 12/30 (t=412) [2.3s]").unwrap();
        match ev {
            GenEvent::Step { job_id, step, total, elapsed_s } => {
                assert_eq!(job_id, 7);
                assert_eq!(step, 12);
                assert_eq!(total, 30);
                assert!((elapsed_s - 2.3).abs() < 1e-3);
            }
            _ => panic!("expected Step"),
        }
    }

    #[test]
    fn parses_wrote_line() {
        let ev = parse_progress(1, "wrote /tmp/out.png").unwrap();
        match ev {
            GenEvent::Done { path, .. } => assert_eq!(path, "/tmp/out.png"),
            _ => panic!("expected Done"),
        }
    }

    #[test]
    fn ignores_unrelated_lines() {
        assert!(parse_progress(0, "steps: 30, seed: 42").is_none());
        assert!(parse_progress(0, "[generate] loading UNet").is_none());
    }
}
