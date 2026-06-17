//! Klearu desktop backend.
//!
//! This crate hosts the Tauri command surface, the long-lived engine state,
//! and the on-disk memory/asset store. The Tauri binary (`web/src-tauri`) is
//! a thin shim that calls [`run`] — all real code lives here so it can be
//! exercised by tests without spinning up a window.
//!
//! Phasing:
//!   - Phase 0: scaffolding; one stub command to prove the wire.
//!   - Phase 1 (current): LLM engine + SQLite memory store + memory CRUD
//!     commands + streamed chat command.
//!   - Phase 2+: engines are added as additional optional fields on
//!     [`state::Workspace`] and loaded lazily.

pub mod commands;
pub mod engine;
pub mod error;
pub mod intent;
pub mod lan_server;
pub mod memory;
pub mod state;
pub mod tools;

use std::sync::Arc;
use state::Workspace;

/// Probes `klearu_diffusion`'s compiled feature set. The metal feature is
/// gated by `cfg(feature = "metal")` *inside* klearu-diffusion, so we
/// can't `cfg!` it directly from here — instead we let the library
/// expose a runtime flag.
fn klearu_diffusion_has_metal() -> bool {
    klearu_diffusion::pipeline::HAS_METAL
}

/// Entry point invoked by `web/src-tauri/src/main.rs`.
///
/// Takes the Tauri context from the binary crate — `tauri::generate_context!`
/// is a proc macro that resolves `tauri.conf.json` relative to the invoking
/// crate's `CARGO_MANIFEST_DIR`, so it has to be called from `web/src-tauri`
/// (where the config lives), not from this lib.
pub fn run(context: tauri::Context) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,klearu_desktop=debug")),
        )
        .with_target(false)
        .try_init();

    // Make the runtime configuration visible at startup so users can
    // confirm at a glance whether the binary is the debug-slow or
    // release-fast variant, and whether Metal acceleration compiled in.
    let metal_enabled = klearu_diffusion_has_metal();
    let profile = if cfg!(debug_assertions) {
        "DEBUG (slow — use `tauri dev --release` or `tauri build`)"
    } else {
        "release"
    };
    tracing::info!(profile, metal_enabled, "klearu-desktop starting");
    eprintln!("[klearu-desktop] profile={profile}, metal={metal_enabled}");

    let workspace = Arc::new(Workspace::new().expect("klearu-desktop: workspace init failed"));

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(workspace)
        .invoke_handler(tauri::generate_handler![
            commands::app::app_info,
            commands::llm::llm_load,
            commands::llm::llm_status,
            commands::llm::llm_unload,
            commands::memory::memory_list,
            commands::memory::memory_upsert,
            commands::memory::memory_delete,
            commands::chat::conversation_create,
            commands::chat::conversation_delete,
            commands::chat::conversations_list,
            commands::chat::conversation_messages,
            commands::chat::chat_send,
            commands::chat::chat_abort,
            commands::image::image_generate,
            commands::image::image_inpaint,
            commands::image::video_generate,
            commands::image::image_abort,
            commands::image::image_list,
            commands::image::image_delete,
            commands::settings::settings_get,
            commands::settings::settings_set,
            commands::settings::settings_get_all,
            commands::vision::vision_load,
            commands::vision::vision_status,
            commands::vision::vision_unload,
            commands::vision::vision_classify,
            commands::job::job_list,
            commands::job::job_get,
            commands::job::job_abort,
            commands::job::job_log_tail,
            commands::job::job_train_image,
            commands::job::job_distill,
            commands::dataset::dataset_list,
            commands::dataset::dataset_upsert,
            commands::dataset::dataset_delete,
            commands::feedback::feedback_set,
            commands::feedback::feedback_get,
            commands::feedback::feedback_list,
            commands::feedback::user_vector_status,
            commands::feedback::user_vector_set_alpha,
            commands::feedback::user_vector_reset,
            commands::feedback::user_vectors_list,
            commands::feedback::user_vector_reset_named,
            commands::feedback::user_vector_set_alpha_named,
            commands::incognito::incognito_status,
            commands::incognito::incognito_start,
            commands::incognito::incognito_stop,
            commands::lan::lan_status,
            commands::lan::lan_start,
            commands::lan::lan_stop,
        ])
        .run(context)
        .expect("klearu-desktop: tauri app failed to start");
}
