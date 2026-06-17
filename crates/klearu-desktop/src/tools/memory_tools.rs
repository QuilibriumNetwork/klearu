//! Memory-related tools: save / recall / list memories, list conversations.
//!
//! `recall_memory` is the read path the LLM uses to look something up
//! before answering. `save_memory` is the write path — the model decides
//! something is worth remembering and persists it.

use serde_json::{json, Value};

use crate::error::{DesktopError, Result};
use crate::memory::{MemoryKind, MemoryUpsert};
use crate::state::Workspace;

use super::{opt_str, required_str, ArgSpec, Tool, ToolSpec};

pub struct SaveMemory;

impl Tool for SaveMemory {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "save_memory",
            description:
                "Persist a memory entry for use in future conversations. \
                 Use `user` for facts about who the user is, `feedback` for \
                 guidance the user has given (what to do or not do), \
                 `project` for ongoing-work context, `reference` for pointers \
                 to external systems.",
            args: vec![
                ArgSpec { name: "kind", r#type: "string", description: "one of: user, feedback, project, reference", required: true },
                ArgSpec { name: "name", r#type: "string", description: "short-kebab-case-slug identifying this memory", required: true },
                ArgSpec { name: "description", r#type: "string", description: "one-line summary used to decide future relevance", required: true },
                ArgSpec { name: "body", r#type: "string", description: "the memory content (markdown is fine)", required: true },
            ],
        }
    }
    fn invoke(&self, workspace: &Workspace, args: &Value) -> Result<Value> {
        let kind_s = required_str(args, "kind")?;
        let kind = match kind_s.as_str() {
            "user" => MemoryKind::User,
            "feedback" => MemoryKind::Feedback,
            "project" => MemoryKind::Project,
            "reference" => MemoryKind::Reference,
            other => return Err(DesktopError::bad_input(format!(
                "kind must be one of user|feedback|project|reference (got `{other}`)"
            ))),
        };
        let entry = workspace.memory.upsert_memory(MemoryUpsert {
            id: None,
            kind,
            name: required_str(args, "name")?,
            description: required_str(args, "description")?,
            body: required_str(args, "body")?,
        })?;
        Ok(json!({
            "id": entry.id,
            "kind": kind_s,
            "name": entry.name,
        }))
    }
}

pub struct RecallMemory;

impl Tool for RecallMemory {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "recall_memory",
            description:
                "Search persisted memories. Returns matching entries with \
                 their name, description, and body. Use this before \
                 answering anything that depends on prior context the user \
                 has given you.",
            args: vec![
                ArgSpec { name: "query", r#type: "string", description: "substring or phrase to match against name/description/body (case-insensitive)", required: false },
                ArgSpec { name: "kind", r#type: "string", description: "filter to one kind (user/feedback/project/reference); omit for all", required: false },
            ],
        }
    }
    fn invoke(&self, workspace: &Workspace, args: &Value) -> Result<Value> {
        let kind_filter = match opt_str(args, "kind").as_deref() {
            Some("user") => Some(MemoryKind::User),
            Some("feedback") => Some(MemoryKind::Feedback),
            Some("project") => Some(MemoryKind::Project),
            Some("reference") => Some(MemoryKind::Reference),
            Some(other) => return Err(DesktopError::bad_input(format!(
                "kind must be one of user|feedback|project|reference (got `{other}`)"
            ))),
            None => None,
        };
        let query = opt_str(args, "query").map(|s| s.to_lowercase());

        let all = workspace.memory.list_memories(kind_filter)?;
        let matches: Vec<Value> = all
            .into_iter()
            .filter(|m| {
                let Some(q) = &query else { return true };
                m.name.to_lowercase().contains(q)
                    || m.description.to_lowercase().contains(q)
                    || m.body.to_lowercase().contains(q)
            })
            .map(|m| json!({
                "id": m.id,
                "kind": format!("{:?}", m.kind).to_lowercase(),
                "name": m.name,
                "description": m.description,
                "body": m.body,
            }))
            .collect();
        Ok(json!({ "matches": matches }))
    }
}

pub struct ListMemories;

impl Tool for ListMemories {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_memories",
            description: "List all memory entries (just name + kind + description). Useful for an overview before deciding what to recall.",
            args: vec![
                ArgSpec { name: "kind", r#type: "string", description: "optional kind filter", required: false },
            ],
        }
    }
    fn invoke(&self, workspace: &Workspace, args: &Value) -> Result<Value> {
        let kind_filter = match opt_str(args, "kind").as_deref() {
            Some("user") => Some(MemoryKind::User),
            Some("feedback") => Some(MemoryKind::Feedback),
            Some("project") => Some(MemoryKind::Project),
            Some("reference") => Some(MemoryKind::Reference),
            _ => None,
        };
        let all = workspace.memory.list_memories(kind_filter)?;
        let entries: Vec<Value> = all.into_iter().map(|m| json!({
            "id": m.id,
            "kind": format!("{:?}", m.kind).to_lowercase(),
            "name": m.name,
            "description": m.description,
        })).collect();
        Ok(json!({ "entries": entries }))
    }
}

pub struct ListConversations;

impl Tool for ListConversations {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_conversations",
            description: "List prior chat sessions (id + title + updated_at). The current conversation is included.",
            args: vec![],
        }
    }
    fn invoke(&self, workspace: &Workspace, _args: &Value) -> Result<Value> {
        let convs = workspace.memory.list_conversations()?;
        let entries: Vec<Value> = convs.into_iter().map(|c| json!({
            "id": c.id,
            "title": c.title,
            "updated_at": c.updated_at,
        })).collect();
        Ok(json!({ "conversations": entries }))
    }
}
