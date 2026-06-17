//! Incognito session lifecycle.
//!
//! On start: open a fresh `:memory:` SQLite + secure_delete pragma, swap
//! the [`MemoryStore`]'s connection to it, record the start time.
//!
//! On stop: drop the [`IncognitoSession`] and swap the persistent disk
//! connection back in. The in-memory SQLite arena is freed back to the
//! system allocator when the connection drops; no file artifacts ever
//! existed.
//!
//! **There is no scratch directory.** Features that fundamentally need a
//! filesystem path — image generation (the `sd_generate` subprocess
//! requires `--out FILE.png`), inpainting (same), distillation (writes a
//! whole dataset) — are refused while incognito is active. They'll be
//! unblocked once the diffusion library refactor (tasks 247/248) makes
//! generation callable in-process with byte buffers instead of paths.

use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::error::Result;
use crate::state::{IncognitoSession, Workspace};

#[derive(Debug, Clone, Serialize)]
pub struct IncognitoStatus {
    pub active: bool,
    pub started_at: Option<i64>,
    /// Free-form notes the UI surfaces alongside the toggle so the user
    /// knows exactly what's NOT being persisted, and what's unavailable.
    pub disabled_features: Vec<&'static str>,
    pub blocked_features: Vec<&'static str>,
}

const INCOGNITO_NON_PERSISTED: &[&str] = &[
    "conversation + message history (in-memory only)",
    "memory entries (in-memory only)",
    "feedback ratings + steering vectors (in-memory only)",
    "settings writes (in-memory only)",
    "image generation — txt2img, img2img, inpaint, native-inpaint, reference editing (in-process)",
    "image gallery rows + PNGs (in-memory; PNGs delivered as data: URIs)",
];

const INCOGNITO_BLOCKED: &[&str] = &[
    "distillation jobs (entire pipeline is file-handoff to encode_images.py)",
    "image training jobs (writes checkpoint + log files)",
];

#[tauri::command]
pub fn incognito_status(workspace: State<'_, Arc<Workspace>>) -> IncognitoStatus {
    let guard = workspace.incognito.read();
    match guard.as_ref() {
        Some(s) => IncognitoStatus {
            active: true,
            started_at: Some(s.started_at),
            disabled_features: INCOGNITO_NON_PERSISTED.to_vec(),
            blocked_features: INCOGNITO_BLOCKED.to_vec(),
        },
        None => IncognitoStatus {
            active: false,
            started_at: None,
            disabled_features: INCOGNITO_NON_PERSISTED.to_vec(),
            blocked_features: INCOGNITO_BLOCKED.to_vec(),
        },
    }
}

#[tauri::command]
pub fn incognito_start(workspace: State<'_, Arc<Workspace>>) -> Result<IncognitoStatus> {
    if workspace.is_incognito() {
        return Ok(incognito_status(workspace));
    }
    // Abort any in-flight diffusion job before swapping — once we flip
    // the flag the next image_generate call would refuse anyway, but a
    // job already mid-flight needs to be killed so its child doesn't
    // keep writing to the regular data dir.
    workspace.diffusion.abort();

    workspace.memory.enter_incognito()?;

    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let session = Arc::new(IncognitoSession { started_at });
    *workspace.incognito.write() = Some(session);

    tracing::info!("incognito session started (in-memory DB, disk writes refused)");
    Ok(incognito_status(workspace))
}

#[tauri::command]
pub fn incognito_stop(workspace: State<'_, Arc<Workspace>>) -> Result<IncognitoStatus> {
    if !workspace.is_incognito() {
        return Ok(incognito_status(workspace));
    }
    workspace.diffusion.abort();

    let session = workspace.incognito.write().take();
    drop(session);

    // Swap MemoryStore back to the persistent disk connection. The
    // in-memory SQLite connection drops here → its arena (which held
    // every conversation, memory entry, feedback row, and steering
    // vector accumulated during the session) is freed to the allocator.
    workspace.memory.exit_incognito()?;

    tracing::info!("incognito session stopped");
    Ok(incognito_status(workspace))
}
