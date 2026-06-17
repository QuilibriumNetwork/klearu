//! Dataset CRUD. v1 supports three kinds:
//!   - `images`: a directory of image files (we count them on import)
//!   - `manifest`: a `.jsonl` file (we count lines)
//!   - `vae-tokens`: a `.jsonl` of pre-encoded VAE tokens (counted like manifest)

use std::path::Path;

use serde::Deserialize;
use tauri::State;

use crate::error::{DesktopError, Result};
use std::sync::Arc;
use crate::memory::{DatasetRow, DatasetUpsert};
use crate::state::Workspace;

#[tauri::command]
pub fn dataset_list(workspace: State<'_, Arc<Workspace>>) -> Result<Vec<DatasetRow>> {
    workspace.memory.dataset_list()
}

#[derive(Debug, Deserialize)]
pub struct DatasetUpsertArgs {
    #[serde(default)]
    pub id: Option<i64>,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub kind: String,
    pub path: String,
}

#[tauri::command]
pub fn dataset_upsert(
    workspace: State<'_, Arc<Workspace>>,
    args: DatasetUpsertArgs,
) -> Result<DatasetRow> {
    let count = count_items(&args.kind, Path::new(&args.path)).unwrap_or(0);
    let upsert = DatasetUpsert {
        id: args.id,
        name: args.name,
        description: args.description,
        kind: args.kind,
        path: args.path,
        item_count: count,
    };
    workspace.memory.dataset_upsert(&upsert)
}

#[tauri::command]
pub fn dataset_delete(workspace: State<'_, Arc<Workspace>>, id: i64) -> Result<()> {
    workspace.memory.dataset_delete(id)
}

fn count_items(kind: &str, path: &Path) -> Result<i64> {
    match kind {
        "images" => count_image_files(path),
        "manifest" | "vae-tokens" => count_jsonl_lines(path),
        _ => Ok(0),
    }
}

fn count_image_files(dir: &Path) -> Result<i64> {
    if !dir.is_dir() {
        return Err(DesktopError::bad_input(format!(
            "{} is not a directory", dir.display()
        )));
    }
    const EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "gif"];
    let mut n: i64 = 0;
    // Single-level scan; deep walks can be added later if a real use-case lands.
    for entry in std::fs::read_dir(dir)
        .map_err(|e| DesktopError::Io(format!("read_dir {}: {e}", dir.display())))?
    {
        let entry = entry.map_err(|e| DesktopError::Io(e.to_string()))?;
        let p = entry.path();
        if let Some(ext) = p.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()) {
            if EXTS.contains(&ext.as_str()) { n += 1; }
        }
    }
    Ok(n)
}

fn count_jsonl_lines(file: &Path) -> Result<i64> {
    use std::io::BufRead;
    if !file.is_file() {
        return Err(DesktopError::bad_input(format!(
            "{} is not a file", file.display()
        )));
    }
    let f = std::fs::File::open(file)
        .map_err(|e| DesktopError::Io(format!("open {}: {e}", file.display())))?;
    let r = std::io::BufReader::new(f);
    let mut n: i64 = 0;
    for _ in r.lines().map_while(|l| l.ok()) { n += 1; }
    Ok(n)
}
