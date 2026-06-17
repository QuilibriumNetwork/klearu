//! Intent router glue: builds the system prompt the LLM sees, and parses
//! tool calls back out of the generated text.
//!
//! ## Tool-call syntax
//!
//! Tool calls are written as XML-ish blocks the model emits inline with
//! its normal reply text:
//!
//! ```text
//! <tool name="save_memory">
//! {"kind": "user", "name": "favorite-color", "description": "user told me their favorite color", "body": "Blue."}
//! </tool>
//! ```
//!
//! Multiple blocks are allowed in one reply. The JSON between the tags is
//! the args object. The parser is intentionally lax — it accepts whitespace
//! around the tags and tolerates the model emitting extra prose around or
//! between blocks.
//!
//! ## Why not OpenAI-style function calling?
//!
//! Our models are local checkpoints with no tool-use post-training. A
//! simple sentinel format the model can imitate from in-context examples
//! is more robust than relying on a head that doesn't exist. The format is
//! easy to read by humans skimming the transcript, and stripping it from
//! the rendered text is trivial.

use std::sync::Arc;

use serde_json::Value;

use crate::memory::MemoryEntry;
use crate::tools::{catalog_for_prompt, Tool};

const TOOL_OPEN: &str = "<tool name=\"";
const TOOL_OPEN_END: &str = "\">";
const TOOL_CLOSE: &str = "</tool>";

const BASE_PERSONA: &str =
"You are Klearu — a private local assistant running on the user's own machine.

You may use the following tools when they help:
";

const TOOL_USAGE_RULES: &str =
"Tool-call syntax — emit blocks like this directly in your reply (you may emit prose around them):

    <tool name=\"the_tool_name\">
    {\"key\": \"value\", ...}
    </tool>

Rules:
- Output ONLY a single JSON object between the tags (no markdown fence).
- Use save_memory ONLY when the user shares a stable fact you should keep \
across sessions (preferences, identity, project goals). Don't save chit-chat.
- Use recall_memory BEFORE answering anything that depends on prior context.
- Do not invent tool names. If no tool fits, just answer in plain text.
- After a tool call, wait — the next turn will include the tool's result. \
Don't continue the reply with imaginary results in the same turn.
";

/// Build the system prompt the LLM sees.
///
/// Includes the persona, the tool catalog rendered from the registry, the
/// tool-call format rules, and a memories section listing each persisted
/// entry's name + description + body. Memories are surfaced verbatim so
/// the model can use them without explicitly calling `recall_memory` first.
pub fn build_system_prompt(
    tools: &[Arc<dyn Tool>],
    memories: &[MemoryEntry],
    extra_system: Option<&str>,
) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str(BASE_PERSONA);
    s.push_str(&catalog_for_prompt(tools));
    s.push('\n');
    s.push_str(TOOL_USAGE_RULES);

    if !memories.is_empty() {
        s.push_str("\nPersisted memories about this user and project (read these before answering):\n");
        for m in memories {
            s.push_str(&format!(
                "- [{}] {} — {}\n",
                format!("{:?}", m.kind).to_lowercase(),
                m.name,
                m.description.trim()
            ));
            // Indent body two spaces so it groups visually under the bullet.
            for line in m.body.lines() {
                s.push_str("    ");
                s.push_str(line);
                s.push('\n');
            }
        }
    }

    if let Some(extra) = extra_system.filter(|x| !x.trim().is_empty()) {
        s.push('\n');
        s.push_str(extra);
        s.push('\n');
    }
    s
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub args: Value,
    /// Byte range in the source text — used to strip the block from the
    /// human-readable transcript or replace it with a result preview.
    pub range: std::ops::Range<usize>,
    /// Reason the JSON failed to parse, if any. When `Some`, `args` is `null`.
    pub parse_error: Option<String>,
}

/// Scan `text` for `<tool name="...">…</tool>` blocks. Returns all blocks
/// found, in source order. Malformed blocks (bad JSON) are returned with
/// `parse_error` set so the UI can surface them and the model can self-
/// correct.
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(off) = text[cursor..].find(TOOL_OPEN) {
        let start = cursor + off;
        let after_open = start + TOOL_OPEN.len();
        let name_end = match text[after_open..].find(TOOL_OPEN_END) {
            Some(o) => after_open + o,
            None => break, // unterminated open tag — bail
        };
        let name = text[after_open..name_end].to_string();
        let body_start = name_end + TOOL_OPEN_END.len();
        let close_off = match text[body_start..].find(TOOL_CLOSE) {
            Some(o) => o,
            None => break, // unterminated block — bail
        };
        let body_end = body_start + close_off;
        let block_end = body_end + TOOL_CLOSE.len();
        let raw_body = text[body_start..body_end].trim();

        let (args, parse_error) = match serde_json::from_str::<Value>(raw_body) {
            Ok(v) => (v, None),
            Err(e) => (Value::Null, Some(e.to_string())),
        };

        calls.push(ToolCall {
            name,
            args,
            range: start..block_end,
            parse_error,
        });
        cursor = block_end;
    }
    calls
}

/// Strip all tool-call blocks from `text` so the rendered transcript shows
/// only the model's narrative. The Channel events carry the tool calls /
/// results separately, so dropping them here doesn't lose information.
pub fn strip_tool_calls(text: &str, calls: &[ToolCall]) -> String {
    if calls.is_empty() { return text.to_string(); }
    let mut out = String::with_capacity(text.len());
    let mut prev = 0;
    for c in calls {
        out.push_str(&text[prev..c.range.start]);
        prev = c.range.end;
    }
    out.push_str(&text[prev..]);
    // Collapse blank-line runs left behind by stripped blocks.
    let collapsed = out
        .split('\n')
        .collect::<Vec<_>>()
        .join("\n")
        .replace("\n\n\n", "\n\n");
    collapsed.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_tool_call() {
        let text = r#"Sure! <tool name="save_memory">
{"kind": "user", "name": "x", "description": "y", "body": "z"}
</tool> Done."#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "save_memory");
        assert_eq!(calls[0].args["name"], "x");
        assert!(calls[0].parse_error.is_none());
    }

    #[test]
    fn parses_multiple_blocks_in_order() {
        let text = r#"<tool name="a">{"k":1}</tool>middle<tool name="b">{"k":2}</tool>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[0].args["k"], 1);
        assert_eq!(calls[1].args["k"], 2);
    }

    #[test]
    fn captures_bad_json_with_error() {
        let text = r#"<tool name="x">not-json</tool>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert!(calls[0].parse_error.is_some());
    }

    #[test]
    fn ignores_unterminated_blocks() {
        let text = r#"<tool name="x">{"k":1"#;
        assert!(parse_tool_calls(text).is_empty());
    }

    #[test]
    fn strips_blocks() {
        let text = r#"before <tool name="x">{"k":1}</tool> after"#;
        let calls = parse_tool_calls(text);
        assert_eq!(strip_tool_calls(text, &calls), "before  after");
    }
}
