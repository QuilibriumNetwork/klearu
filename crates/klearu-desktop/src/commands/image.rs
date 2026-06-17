//! Image generation commands.
//!
//! `image_generate` is the headline command. It spawns the diffusion
//! engine (`sd_generate` subprocess), forwards its progress events over a
//! Tauri Channel, and — on success — records the resulting PNG in the
//! `images` table so the gallery can show it later.

use std::sync::Arc;

use serde::Deserialize;
use tauri::ipc::Channel;
use tauri::State;
use tokio::sync::mpsc;

use crate::engine::clip::ClipFormat;
use crate::engine::diffusion::{GenEvent, GenerateRequest, InProcessRequest};
use crate::error::{DesktopError, Result};
use crate::memory::{ImageRecord, StoredImage};
use crate::state::Workspace;

#[derive(Debug, Deserialize)]
pub struct ImageGenerateArgs {
    #[serde(flatten)]
    pub req: GenerateRequest,
}

#[tauri::command]
pub async fn image_generate(
    workspace: State<'_, Arc<Workspace>>,
    args: ImageGenerateArgs,
    on_event: Channel<GenEvent>,
) -> Result<u64> {
    // Always route through the in-process pipeline now that the library
    // supports txt2img + img2img + naive inpaint + native-inpaint +
    // ControlNet. Input/mask paths are read into byte buffers before the
    // pipeline runs — the bytes flow through `InProcessRequest` and the
    // generated PNG is delivered either to `~/.klearu/images/` (normal
    // mode) or inline as a `data:` URI (incognito mode). Subprocess path
    // is retained for now but no longer reachable via this entry.
    image_generate_in_process(workspace, args, on_event).await
}

async fn image_generate_in_process(
    workspace: State<'_, Arc<Workspace>>,
    args: ImageGenerateArgs,
    on_event: Channel<GenEvent>,
) -> Result<u64> {
    let req = args.req;
    // Read input / mask / control bytes from disk if the caller supplied
    // paths. This converges the file-path API (used by REPL workflows and
    // by Tauri's drag-drop) and the bytes-only API (used by Inpaint
    // panel) into the same in-process call below.
    let input_bytes = match req.input_path.as_ref() {
        Some(p) => Some(std::fs::read(p).map_err(|e| {
            DesktopError::Io(format!("read input {}: {e}", p.display()))
        })?),
        None => None,
    };
    let mask_bytes = match req.mask_path.as_ref() {
        Some(p) => Some(std::fs::read(p).map_err(|e| {
            DesktopError::Io(format!("read mask {}: {e}", p.display()))
        })?),
        None => None,
    };
    let strength = req.strength.unwrap_or(0.85);

    // Reference images: prefer frontend-supplied bytes (incognito-safe, no
    // filesystem); fall back to reading any supplied paths (REPL/file callers).
    let ref_image_bytes: Vec<Vec<u8>> = match req.ref_image_bytes {
        Some(b) => b,
        None => match req.ref_image_paths.as_ref() {
            Some(paths) => paths
                .iter()
                .map(|p| {
                    std::fs::read(p)
                        .map_err(|e| DesktopError::Io(format!("read ref {}: {e}", p.display())))
                })
                .collect::<Result<Vec<_>>>()?,
            None => Vec::new(),
        },
    };

    run_in_process(
        workspace,
        InProcInputs {
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone().unwrap_or_default(),
            steps: req.steps.unwrap_or(25) as usize,
            cfg_scale: req.cfg_scale.unwrap_or(7.5),
            seed: req.seed,
            input_image_bytes: input_bytes,
            mask_bytes,
            strength,
            ref_image_bytes,
            conversation_id: req.conversation_id,
        },
        on_event,
    )
    .await
}

/// Shared driver for the in-process path. Used by both `image_generate`
/// (path-based) and `image_inpaint` (bytes-from-Frontend) — collapses
/// the duplicate PNG-encode / record / emit logic.
struct InProcInputs {
    prompt: String,
    negative_prompt: String,
    steps: usize,
    cfg_scale: f32,
    seed: Option<u64>,
    input_image_bytes: Option<Vec<u8>>,
    mask_bytes: Option<Vec<u8>>,
    strength: f32,
    ref_image_bytes: Vec<Vec<u8>>,
    conversation_id: Option<i64>,
}

async fn run_in_process(
    workspace: State<'_, Arc<Workspace>>,
    inputs: InProcInputs,
    on_event: Channel<GenEvent>,
) -> Result<u64> {
    let (tx, mut rx) = mpsc::unbounded_channel::<GenEvent>();
    let engine = workspace.diffusion.clone();
    let memory = workspace.memory.clone();

    let cond_bias = match prepare_cond_bias_inmem(&workspace).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(?e, "cond-bias prep skipped");
            None
        }
    };
    let seed = inputs.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| (d.as_nanos() as u64).wrapping_mul(0x9E3779B97F4A7C15))
            .unwrap_or(42)
    });
    let in_proc_req = InProcessRequest {
        prompt: inputs.prompt.clone(),
        negative_prompt: inputs.negative_prompt,
        steps: inputs.steps,
        cfg_scale: inputs.cfg_scale,
        seed,
        input_image_bytes: inputs.input_image_bytes,
        mask_bytes: inputs.mask_bytes,
        strength: inputs.strength,
        ref_image_bytes: inputs.ref_image_bytes,
        control_image_bytes: None,
        controlnet_weight: 1.0,
        controlnet_start: 0.0,
        controlnet_end: 1.0,
        cond_bias,
    };

    let on_event_for_task = on_event.clone();
    let forward = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            // Suppress the engine's empty-path Done — we'll emit our own
            // after the PNG-encode + persist step below.
            if !matches!(ev, GenEvent::Done { .. }) {
                let _ = on_event_for_task.send(ev);
            }
        }
    });

    let result = engine.generate_in_process(in_proc_req, tx).await?;
    let _ = forward.await;

    let png_bytes = encode_chw_to_png(&result.rgb_chw, result.height, result.width)?;
    let path = if workspace.is_incognito() {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
        format!("data:image/png;base64,{b64}")
    } else {
        let data_dir = workspace.data_dir();
        let images_dir = data_dir.join("images");
        let _ = std::fs::create_dir_all(&images_dir);
        let p = images_dir.join(format!("{}-{}.png", now_secs(), seed));
        std::fs::write(&p, &png_bytes)
            .map_err(|e| DesktopError::Io(format!("write png: {e}")))?;
        p.to_string_lossy().into_owned()
    };

    let rec = ImageRecord {
        path: &path,
        prompt: &inputs.prompt,
        negative_prompt: None,
        seed: Some(seed as i64),
        steps: Some(inputs.steps as i64),
        cfg_scale: Some(inputs.cfg_scale as f64),
        width: Some(result.width as i64),
        height: Some(result.height as i64),
        checkpoint: None,
        conversation_id: inputs.conversation_id,
        created_at: now_secs(),
    };
    if let Err(e) = memory.image_record(&rec) {
        tracing::warn!(?e, "failed to record image row");
    }

    let _ = on_event.send(GenEvent::Done {
        job_id: result.job_id,
        path,
    });
    Ok(result.job_id)
}

pub(crate) fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// PNG-encode a `[3 * H * W]` f32 CHW buffer in `[-1, 1]` convention.
/// Same conversion `klearu_diffusion::image_io::save_png` does, but
/// emits bytes rather than writing to disk.
pub(crate) fn encode_chw_to_png(chw: &[f32], h: usize, w: usize) -> Result<Vec<u8>> {
    if chw.len() != 3 * h * w {
        return Err(DesktopError::engine(format!(
            "CHW buffer len {} ≠ 3 * {h} * {w}", chw.len()
        )));
    }
    let mut interleaved = vec![0u8; 3 * h * w];
    for hi in 0..h {
        for wi in 0..w {
            for c in 0..3 {
                let v = chw[c * h * w + hi * w + wi];
                let v01 = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
                interleaved[(hi * w + wi) * 3 + c] = (v01 * 255.0).round() as u8;
            }
        }
    }
    use image::{ImageBuffer, Rgb};
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w as u32, h as u32, interleaved)
            .ok_or_else(|| DesktopError::engine("ImageBuffer::from_raw failed"))?;
    let mut bytes = Vec::with_capacity(h * w * 3);
    img.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(|e| DesktopError::engine(format!("png encode: {e}")))?;
    Ok(bytes)
}

/// In-memory variant of `prepare_cond_bias`. Returns the persisted CLIP-
/// space user vector scaled by α, ready to drop into the pipeline
/// request. No sidecar file written.
async fn prepare_cond_bias_inmem(
    workspace: &State<'_, Arc<Workspace>>,
) -> Result<Option<Vec<f32>>> {
    let diff_cfg = workspace.diffusion.config();
    let Some(checkpoint) = diff_cfg.checkpoint.clone() else { return Ok(None); };
    let detect = tokio::task::spawn_blocking(move || -> Result<ClipFormat> {
        use klearu_diffusion::weight::{SDFormat, SingleFileLoader};
        let loader = SingleFileLoader::open(&checkpoint)
            .map_err(|e| DesktopError::engine(format!("open checkpoint: {e}")))?;
        match loader.metadata.architecture_variant() {
            SDFormat::Sd15 => Ok(ClipFormat::Sd15),
            SDFormat::Sdxl => Ok(ClipFormat::Sdxl),
            _ => Err(DesktopError::bad_input("checkpoint not SD 1.5 or SDXL")),
        }
    })
    .await
    .map_err(|e| DesktopError::engine(format!("join: {e}")))?;
    let format = match detect { Ok(f) => f, Err(_) => return Ok(None) };
    let cad = match format { ClipFormat::Sd15 => 768, ClipFormat::Sdxl => 2048 };
    let state = workspace.memory.user_vector_load(format.user_vector_key(), cad)?;
    if state.n_pos + state.n_neg == 0 || state.alpha == 0.0 {
        return Ok(None);
    }
    let scaled: Vec<f32> = state.vec.iter().map(|v| state.alpha * v).collect();
    Ok(Some(scaled))
}

#[tauri::command]
pub fn image_abort(workspace: State<'_, Arc<Workspace>>) -> Option<u64> {
    let id = workspace.diffusion.current_job_id();
    workspace.diffusion.abort();
    id
}

#[tauri::command]
pub fn image_list(workspace: State<'_, Arc<Workspace>>, limit: Option<usize>) -> Result<Vec<StoredImage>> {
    workspace.memory.image_list(limit.unwrap_or(60))
}

// ---- video (AnimateDiff) ---------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct VideoGenerateArgs {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "default_frames")]
    pub num_frames: usize,
    #[serde(default = "default_fps")]
    pub fps: usize,
}
fn default_frames() -> usize { 16 }
fn default_fps() -> usize { 8 }

#[tauri::command]
pub async fn video_generate(
    workspace: State<'_, Arc<Workspace>>,
    args: VideoGenerateArgs,
    on_event: tauri::ipc::Channel<crate::engine::diffusion::GenEvent>,
) -> Result<u64> {
    let (tx, mut rx) = mpsc::unbounded_channel::<crate::engine::diffusion::GenEvent>();
    let engine = workspace.diffusion.clone();
    let memory = workspace.memory.clone();

    let cond_bias = match prepare_cond_bias_inmem(&workspace).await {
        Ok(b) => b,
        Err(e) => { tracing::warn!(?e, "cond-bias skipped"); None }
    };
    let seed = args.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| (d.as_nanos() as u64).wrapping_mul(0x9E3779B97F4A7C15))
            .unwrap_or(42)
    });
    let video_req = crate::engine::diffusion::VideoRequest {
        prompt: args.prompt.clone(),
        negative_prompt: args.negative_prompt.clone().unwrap_or_default(),
        steps: args.steps.unwrap_or(25) as usize,
        cfg_scale: args.cfg_scale.unwrap_or(7.5),
        seed,
        num_frames: args.num_frames,
        fps: args.fps,
        cond_bias,
    };

    let on_event_copy = on_event.clone();
    let forward = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            if !matches!(ev, crate::engine::diffusion::GenEvent::Done { .. }) {
                let _ = on_event_copy.send(ev);
            }
        }
    });

    let result = engine.generate_video_in_process(video_req, tx).await?;
    let _ = forward.await;

    // Save the MP4 to disk (or base64 in incognito).
    let path = if workspace.is_incognito() {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&result.mp4_bytes);
        format!("data:video/mp4;base64,{b64}")
    } else {
        let data_dir = workspace.data_dir();
        let videos_dir = data_dir.join("videos");
        let _ = std::fs::create_dir_all(&videos_dir);
        let p = videos_dir.join(format!("{}-{}.mp4", now_secs(), seed));
        std::fs::write(&p, &result.mp4_bytes)
            .map_err(|e| DesktopError::Io(format!("write mp4: {e}")))?;
        p.to_string_lossy().into_owned()
    };

    let _ = on_event.send(crate::engine::diffusion::GenEvent::Done {
        job_id: result.job_id,
        path: path.clone(),
    });

    // Record in images table for gallery (reusing the same table — it stores
    // the path and the UI can detect `.mp4` extension for video rendering).
    let rec = crate::memory::ImageRecord {
        path: &path,
        prompt: &args.prompt,
        negative_prompt: args.negative_prompt.as_deref(),
        seed: Some(seed as i64),
        steps: args.steps.map(|s| s as i64),
        cfg_scale: args.cfg_scale.map(|c| c as f64),
        width: Some(result.width as i64),
        height: Some(result.height as i64),
        checkpoint: None,
        conversation_id: None,
        created_at: now_secs(),
    };
    let _ = memory.image_record(&rec);

    Ok(result.job_id)
}

#[tauri::command]
pub fn image_delete(workspace: State<'_, Arc<Workspace>>, id: i64, delete_file: bool) -> Result<()> {
    if let Some(path) = workspace.memory.image_delete(id)? {
        if delete_file {
            let _ = std::fs::remove_file(path);
        }
    }
    Ok(())
}

// ---- inpaint ---------------------------------------------------------------

/// Inpaint request — input + mask come in as raw bytes from the canvas
/// editor and stay in memory through the entire pipeline. No temp files
/// are written; the in-process pipeline VAE-encodes the input bytes and
/// applies the mask natively.
#[derive(Debug, Deserialize)]
pub struct ImageInpaintArgs {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub strength: Option<f32>,
    #[serde(default)]
    pub conversation_id: Option<i64>,
    /// PNG bytes of the source image.
    pub input_bytes: Vec<u8>,
    /// PNG bytes of the grayscale mask (white = regenerate, black = keep).
    /// Omit for full-image img2img.
    #[serde(default)]
    pub mask_bytes: Option<Vec<u8>>,
}

#[tauri::command]
pub async fn image_inpaint(
    workspace: State<'_, Arc<Workspace>>,
    args: ImageInpaintArgs,
    on_event: Channel<crate::engine::diffusion::GenEvent>,
) -> Result<u64> {
    // Goes through the same in-process driver as image_generate. Works
    // in incognito — no disk touch end-to-end.
    run_in_process(
        workspace,
        InProcInputs {
            prompt: args.prompt,
            negative_prompt: args.negative_prompt.unwrap_or_default(),
            steps: args.steps.unwrap_or(25) as usize,
            cfg_scale: args.cfg_scale.unwrap_or(7.5),
            seed: args.seed,
            input_image_bytes: Some(args.input_bytes),
            mask_bytes: args.mask_bytes,
            strength: args.strength.unwrap_or(0.85),
            ref_image_bytes: Vec::new(),
            conversation_id: args.conversation_id,
        },
        on_event,
    )
    .await
}
