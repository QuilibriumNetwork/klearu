//! Tauri Mobile entry point.
//!
//! Tauri 2.x's mobile build needs the app entry exported from a `lib` target;
//! the desktop binary in `main.rs` and this shim both delegate to the real
//! implementation in `klearu_desktop::run`.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    klearu_desktop::run(tauri::generate_context!());
}
