//! LLM lifecycle commands. Generation is in `commands::chat`.

use std::path::PathBuf;

use tauri::State;

use crate::engine::llm::LlmLoadStatus;
use std::sync::Arc;
use crate::error::Result;
use crate::state::Workspace;

#[tauri::command]
pub async fn llm_load(
    workspace: State<'_, Arc<Workspace>>,
    model_dir: String,
) -> Result<LlmLoadStatus> {
    let engine = workspace.llm.clone();
    let path = PathBuf::from(model_dir);
    // Load is heavy (multi-GB safetensors mmap + tokenizer build). Push it
    // off the Tauri event-loop thread so the UI stays responsive.
    tokio::task::spawn_blocking(move || engine.lock().load(&path))
        .await
        .map_err(|e| crate::error::DesktopError::engine(format!("join: {e}")))?
}

#[tauri::command]
pub fn llm_status(workspace: State<'_, Arc<Workspace>>) -> LlmLoadStatus {
    workspace.llm.lock().status()
}

#[tauri::command]
pub fn llm_unload(workspace: State<'_, Arc<Workspace>>) {
    workspace.llm.lock().unload();
}
