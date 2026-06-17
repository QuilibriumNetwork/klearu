//! Lazy-loaded inference engines. Each engine wraps a klearu-* pipeline
//! and exposes the minimal surface the command layer needs (load, status,
//! stream, abort).
//!
//! Engines are held inside Tauri state behind `parking_lot::Mutex` so the
//! command handlers can mutate them; generation runs on a blocking task so
//! it doesn't stall the Tauri event loop.

pub mod clip;
pub mod diffusion;
pub mod jobs;
pub mod llm;
pub mod vision;

pub use clip::ClipEngine;
pub use diffusion::DiffusionEngine;
pub use jobs::JobManager;
pub use llm::LlmEngine;
pub use vision::VisionEngine;
