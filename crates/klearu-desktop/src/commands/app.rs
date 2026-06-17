use std::sync::Arc;
use serde::Serialize;
use tauri::State;

use crate::state::Workspace;

#[derive(Serialize)]
pub struct AppInfo {
    pub name: &'static str,
    pub version: &'static str,
    pub data_dir: String,
    pub features: AppFeatures,
}

#[derive(Serialize)]
pub struct AppFeatures {
    pub chat: bool,
    pub image_generate: bool,
    pub inpaint: bool,
    pub classify: bool,
    pub training: bool,
    pub memory: bool,
}

#[tauri::command]
pub fn app_info(workspace: State<'_, Arc<Workspace>>) -> AppInfo {
    AppInfo {
        name: "Klearu",
        version: env!("CARGO_PKG_VERSION"),
        data_dir: workspace.data_dir().to_string_lossy().into_owned(),
        features: AppFeatures {
            chat: true,            // local LLM available once a checkpoint is loaded
            memory: true,          // sqlite store always available
            image_generate: true,  // subprocess-mode txt2img wired
            inpaint: true,         // canvas-driven inpaint via sd_generate
            classify: true,        // DaViT/ViT classification wired
            training: true,        // job manager + img_train + distill wired
        },
    }
}
