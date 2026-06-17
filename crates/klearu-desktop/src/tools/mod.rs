//! Tool catalog. The LLM emits tool calls in its reply; the intent router
//! parses them out and dispatches via this registry.
//!
//! A tool is just (name, description, json-schema-like args, invoke fn).
//! The schema is rendered into the system prompt so the model knows what
//! exists and what arguments each tool takes. The body of `invoke` runs
//! synchronously with a `Workspace` reference — heavy work (image gen,
//! training) returns immediately with a "job started" handle and notifies
//! via separate channel events.

use std::sync::Arc;

use serde::Serialize;
use serde_json::{json, Value};

use crate::error::{DesktopError, Result};
use crate::state::Workspace;

mod memory_tools;

pub use memory_tools::*;

/// Description of one tool exposed to the LLM.
#[derive(Debug, Clone, Serialize)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// Arg name → (one-line description, JSON type literal).
    pub args: Vec<ArgSpec>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArgSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub r#type: &'static str,
    pub required: bool,
}

pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn invoke(&self, workspace: &Workspace, args: &Value) -> Result<Value>;
}

/// Build the canonical tool registry. Phase 2 ships memory tools only;
/// later phases append image / classify / training tools to this list.
pub fn registry() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(SaveMemory),
        Arc::new(RecallMemory),
        Arc::new(ListMemories),
        Arc::new(ListConversations),
    ]
}

/// Convenience: look up by name.
pub fn find(name: &str) -> Option<Arc<dyn Tool>> {
    registry().into_iter().find(|t| t.spec().name == name)
}

/// Convert a tool registry into the catalog the LLM sees in its system
/// prompt. Format is intentionally compact and human-readable so the model
/// can ad-lib new arg patterns without a hard schema mismatch.
pub fn catalog_for_prompt(tools: &[Arc<dyn Tool>]) -> String {
    let mut out = String::new();
    for t in tools {
        let s = t.spec();
        out.push_str("- `");
        out.push_str(s.name);
        out.push_str("` — ");
        out.push_str(s.description);
        out.push('\n');
        for a in &s.args {
            out.push_str("    · ");
            out.push_str(a.name);
            out.push_str(" (");
            out.push_str(a.r#type);
            if !a.required {
                out.push_str(", optional");
            }
            out.push_str("): ");
            out.push_str(a.description);
            out.push('\n');
        }
    }
    out
}

/// Helper for tool impls: pull a required string arg out of the args object.
pub(crate) fn required_str(args: &Value, name: &str) -> Result<String> {
    args.get(name)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| DesktopError::bad_input(format!("missing required arg `{name}`")))
}

/// Helper: pull an optional string arg.
pub(crate) fn opt_str(args: &Value, name: &str) -> Option<String> {
    args.get(name).and_then(|v| v.as_str()).map(|s| s.to_string())
}

/// Helper: render a JSON value as a compact preview (truncated).
pub fn preview(v: &Value, max: usize) -> String {
    let s = if v.is_string() {
        v.as_str().unwrap_or("").to_string()
    } else {
        v.to_string()
    };
    if s.chars().count() <= max { s } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}

/// Wrap an `invoke` return so the channel event has a stable JSON shape.
pub fn ok_value(v: Value) -> Value { json!({ "ok": true, "result": v }) }
pub fn err_value(code: &str, message: &str) -> Value {
    json!({ "ok": false, "code": code, "message": message })
}
