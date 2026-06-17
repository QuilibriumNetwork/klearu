//! Tauri command surface. One submodule per concern. Every command is a
//! thin async wrapper that pulls [`Workspace`] from Tauri state, delegates
//! to the engine/store layer, and returns either the result or a
//! [`DesktopError`] (serialized to `{code, message}` on the wire).

pub mod app;
pub mod chat;
pub mod dataset;
pub mod feedback;
pub mod image;
pub mod incognito;
pub mod job;
pub mod lan;
pub mod llm;
pub mod memory;
pub mod settings;
pub mod vision;
