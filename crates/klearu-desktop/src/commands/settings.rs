//! Key/value settings commands. The frontend reads/writes app config
//! (image checkpoint paths, output dir, etc) through these. Image-related
//! keys are mirrored into `DiffusionEngine` whenever they change.

use std::path::PathBuf;

use serde::Serialize;
use tauri::State;

use crate::engine::diffusion::DiffusionConfig;
use std::sync::Arc;
use crate::error::Result;
use crate::state::Workspace;

#[derive(Serialize)]
pub struct SettingEntry { pub key: String, pub value: String }

#[tauri::command]
pub fn settings_get(workspace: State<'_, Arc<Workspace>>, key: String) -> Result<Option<String>> {
    workspace.memory.settings_get(&key)
}

#[tauri::command]
pub fn settings_get_all(workspace: State<'_, Arc<Workspace>>) -> Result<Vec<SettingEntry>> {
    let pairs = workspace.memory.settings_get_all()?;
    Ok(pairs
        .into_iter()
        .map(|(k, v)| SettingEntry { key: k, value: v })
        .collect())
}

#[tauri::command]
pub fn settings_set(
    workspace: State<'_, Arc<Workspace>>,
    key: String,
    value: String,
) -> Result<()> {
    workspace.memory.settings_set(&key, &value)?;
    // Image-related keys: mirror into the live engine so the next
    // generation picks them up without an app restart.
    if key.starts_with("image.") {
        let new_cfg = build_diffusion_config(&workspace);
        workspace.diffusion.set_config(new_cfg);
    }
    Ok(())
}

fn build_diffusion_config(workspace: &Workspace) -> DiffusionConfig {
    let get = |k: &str| workspace.memory.settings_get(k).ok().flatten();
    DiffusionConfig {
        sd_generate_path: get("image.sd_generate_path").map(PathBuf::from),
        checkpoint: get("image.checkpoint").map(PathBuf::from),
        tokenizer_l: get("image.tokenizer_l").map(PathBuf::from),
        tokenizer_g: get("image.tokenizer_g").map(PathBuf::from),
        output_dir: get("image.output_dir").map(PathBuf::from),
        motion_adapter: get("image.motion_adapter").map(PathBuf::from),
    }
}
