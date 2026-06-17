// Prevents an extra console window from popping up on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    klearu_desktop::run(tauri::generate_context!());
}
