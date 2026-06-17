use std::sync::Arc;
use tauri::State;

use crate::error::Result;
use crate::memory::{MemoryEntry, MemoryKind, MemoryUpsert};
use crate::state::Workspace;

#[tauri::command]
pub fn memory_list(
    workspace: State<'_, Arc<Workspace>>,
    kind: Option<MemoryKind>,
) -> Result<Vec<MemoryEntry>> {
    workspace.memory.list_memories(kind)
}

#[tauri::command]
pub fn memory_upsert(
    workspace: State<'_, Arc<Workspace>>,
    entry: MemoryUpsert,
) -> Result<MemoryEntry> {
    workspace.memory.upsert_memory(entry)
}

#[tauri::command]
pub fn memory_delete(workspace: State<'_, Arc<Workspace>>, id: i64) -> Result<()> {
    workspace.memory.delete_memory(id)
}
