//! Vision classification commands. `vision_load` warms a DaViT/ViT
//! checkpoint into the engine; `vision_classify` accepts either a path
//! on disk or raw image bytes (base64) and returns top-K predictions.

use std::path::PathBuf;

use serde::Deserialize;
use tauri::State;

use crate::engine::vision::{Prediction, VisionKind, VisionStatus};
use std::sync::Arc;
use crate::error::{DesktopError, Result};
use crate::state::Workspace;

#[derive(Debug, Deserialize)]
pub struct VisionLoadArgs {
    pub model_dir: String,
    /// Override the auto-detected family (`davit`/`vit`). `None` lets the
    /// engine sniff `config.json`.
    #[serde(default)]
    pub force_kind: Option<VisionKind>,
}

#[tauri::command]
pub async fn vision_load(
    workspace: State<'_, Arc<Workspace>>,
    args: VisionLoadArgs,
) -> Result<VisionStatus> {
    let engine = workspace.vision.clone();
    let path = PathBuf::from(args.model_dir);
    let force = args.force_kind;
    tokio::task::spawn_blocking(move || engine.lock().load(&path, force))
        .await
        .map_err(|e| DesktopError::engine(format!("join: {e}")))?
}

#[tauri::command]
pub fn vision_status(workspace: State<'_, Arc<Workspace>>) -> VisionStatus {
    workspace.vision.lock().status()
}

#[tauri::command]
pub fn vision_unload(workspace: State<'_, Arc<Workspace>>) {
    workspace.vision.lock().unload();
}

#[derive(Debug, Deserialize)]
pub struct VisionClassifyArgs {
    /// File path on disk (preferred when the image already lives on disk —
    /// no JSON-array round-trip for the bytes).
    #[serde(default)]
    pub path: Option<String>,
    /// Raw image bytes (PNG/JPEG/etc). Used when the frontend has bytes
    /// from a `<input type="file">` or a paste event.
    #[serde(default)]
    pub bytes: Option<Vec<u8>>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize { 5 }

#[tauri::command]
pub async fn vision_classify(
    workspace: State<'_, Arc<Workspace>>,
    args: VisionClassifyArgs,
) -> Result<Vec<Prediction>> {
    let engine = workspace.vision.clone();

    let bytes: Vec<u8> = if let Some(p) = args.path {
        std::fs::read(&p).map_err(|e| DesktopError::Io(format!("read {p}: {e}")))?
    } else if let Some(b) = args.bytes {
        b
    } else {
        return Err(DesktopError::bad_input("vision_classify requires `path` or `bytes`"));
    };

    let k = args.top_k;
    tokio::task::spawn_blocking(move || engine.lock().classify_bytes(&bytes, k))
        .await
        .map_err(|e| DesktopError::engine(format!("join: {e}")))?
}
