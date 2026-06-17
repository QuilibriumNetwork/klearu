//! Chat-related commands: conversation CRUD plus streamed generation with
//! intent routing.
//!
//! `chat_send` runs a small ReAct loop: each iteration generates from the
//! LLM, parses out any `<tool name="…">…</tool>` blocks, executes the
//! corresponding tools, and — if any ran — appends `<tool_result …>` blocks
//! and iterates again. Capped at [`MAX_ITERS`] to bound budget per turn.
//!
//! Token events stream over a Tauri `Channel` so the UI can render the
//! reply as it appears; tool calls and results emit their own events so the
//! UI can render them as chips inline.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::ipc::Channel;
use tauri::State;

use crate::engine::llm::{LlmSamplerInput, WireMessage, WireRole};
use std::sync::Arc;
use crate::error::Result;
use crate::intent::{build_system_prompt, parse_tool_calls};
use crate::memory::{Conversation, Message, MessageRole};
use crate::state::Workspace;
use crate::tools::{self, err_value, ok_value};

/// Hard ceiling on tool-result → re-generate iterations within one user turn.
/// Three is enough for "look up A, look up B, answer" without runaway loops.
const MAX_ITERS: usize = 3;

// ---- conversation CRUD -----------------------------------------------------

#[tauri::command]
pub fn conversation_create(
    workspace: State<'_, Arc<Workspace>>,
    title: Option<String>,
) -> Result<Conversation> {
    workspace
        .memory
        .create_conversation(title.as_deref().unwrap_or("New chat"))
}

#[tauri::command]
pub fn conversations_list(workspace: State<'_, Arc<Workspace>>) -> Result<Vec<Conversation>> {
    workspace.memory.list_conversations()
}

#[tauri::command]
pub fn conversation_messages(
    workspace: State<'_, Arc<Workspace>>,
    conversation_id: i64,
) -> Result<Vec<Message>> {
    workspace.memory.conversation_messages(conversation_id)
}

#[tauri::command]
pub fn conversation_delete(workspace: State<'_, Arc<Workspace>>, id: i64) -> Result<()> {
    workspace.memory.delete_conversation(id)
}

// ---- streaming generation --------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    Token { text: String },
    /// One iteration of the ReAct loop finished and we're about to dispatch
    /// `count` tool calls. Useful for the UI to insert a "thinking…" marker
    /// without waiting for individual `tool_result`s.
    IterEnd { iter: usize, count: usize },
    /// The model emitted a tool call. The args are pre-parsed JSON; on
    /// JSON parse failure `args` is null and `parse_error` is set.
    ToolCall { iter: usize, name: String, args: Value, parse_error: Option<String> },
    /// Tool finished. `result` is either `{ok: true, result: …}` or
    /// `{ok: false, code, message}`.
    ToolResult { iter: usize, name: String, result: Value },
    /// Generation finished cleanly. `assistant_message_id` points at the
    /// persisted row. `total_text` is the FULL assistant turn (tool blocks
    /// + tool_result blocks + narrative interleaved).
    Done { assistant_message_id: i64, total_text: String },
    /// Caller invoked `chat_abort`. Whatever text was generated so far is
    /// persisted so the conversation history stays coherent.
    Aborted { assistant_message_id: i64, partial: String },
    /// Engine error (load failure, model crash, etc). No message saved.
    Error { code: String, message: String },
}

#[derive(Debug, Deserialize)]
pub struct ChatSendArgs {
    pub conversation_id: i64,
    pub content: String,
    /// Extra system text appended after the standard persona + tool catalog.
    /// Use sparingly — memory entries are auto-injected and usually enough.
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub sampler: Option<LlmSamplerInput>,
}

#[tauri::command]
pub async fn chat_send(
    workspace: State<'_, Arc<Workspace>>,
    args: ChatSendArgs,
    on_event: Channel<StreamEvent>,
) -> Result<()> {
    // 1. Persist the user turn synchronously so the transcript stays
    //    consistent if generation fails mid-flight.
    workspace.memory.append_message(args.conversation_id, MessageRole::User, &args.content)?;

    // 2. Build the system prompt fresh each turn so newly-saved memories
    //    show up without restarting the app.
    let memories = workspace.memory.list_memories(None).unwrap_or_default();
    let tool_registry = tools::registry();
    let system = build_system_prompt(&tool_registry, &memories, args.system_prompt.as_deref());

    // 3. Snapshot the persisted history (incl. the user message we just wrote).
    let history = workspace.memory.conversation_messages(args.conversation_id)?;
    let base_wire = wire_from_history(&system, &history);

    let sampler = args.sampler.unwrap_or_default_sampler();

    // Load the persisted user steering vector (if any) so this turn's
    // generation reflects accumulated feedback. The vector is only valid
    // when its hidden_size matches the loaded model — otherwise we send
    // an empty vec and the engine no-ops the steering branch.
    let hidden_size = workspace.llm.lock().hidden_size().unwrap_or(0);
    let user_state = if hidden_size > 0 {
        workspace.memory.user_vector_load("default", hidden_size).ok()
    } else { None };
    let (steering_vec, alpha): (Vec<f32>, f32) = match &user_state {
        Some(s) if s.n_pos + s.n_neg > 0 => (s.vec.clone(), s.alpha),
        _ => (Vec::new(), 0.0),
    };

    // 4. ReAct loop. Each iteration: prompt → generate → parse → run tools →
    //    accumulate. Bail when the model emits no tool calls (= done) or
    //    we hit [`MAX_ITERS`].
    let mut accumulated = String::new();
    let mut had_error: Option<(String, String)> = None;
    let mut was_aborted = false;
    // Captured at the end of each iteration; the last iteration's wins
    // since that's the response the user reacts to.
    let final_hidden: std::sync::Arc<parking_lot::Mutex<Option<Vec<f32>>>> =
        std::sync::Arc::new(parking_lot::Mutex::new(None));

    for iter in 0..MAX_ITERS {
        let llm = workspace.llm.clone();
        let ev = on_event.clone();
        let mut wire = base_wire.clone();
        if !accumulated.is_empty() {
            wire.push(WireMessage { role: WireRole::Assistant, content: accumulated.clone() });
            wire.push(WireMessage { role: WireRole::User, content: "[continue]".to_string() });
        }
        let sampler_c = sampler.clone();
        let steering_c = steering_vec.clone();
        let alpha_c = alpha;
        let hidden_capture = final_hidden.clone();

        // Run one generation pass on a blocking thread.
        let gen: std::result::Result<Result<String>, tokio::task::JoinError> =
            tokio::task::spawn_blocking(move || {
                let mut engine = llm.lock();
                engine.stream_reply(
                    wire,
                    sampler_c,
                    &steering_c,
                    alpha_c,
                    |tok| {
                        let _ = ev.send(StreamEvent::Token { text: tok.to_string() });
                    },
                    |h| {
                        *hidden_capture.lock() = Some(h.to_vec());
                    },
                )
            })
            .await;

        let chunk = match gen {
            Err(e) => { had_error = Some(("engine".into(), format!("join: {e}"))); break; }
            Ok(Err(e)) => { had_error = Some((e.code().to_string(), e.to_string())); break; }
            Ok(Ok(text)) => text,
        };

        // 5. Parse tool calls from THIS chunk only. Earlier-iteration blocks
        //    already in `accumulated` were dispatched on their own iteration.
        let calls = parse_tool_calls(&chunk);
        accumulated.push_str(&chunk);

        // Abort can land between iterations or mid-token; either way, save
        // what we have and stop.
        if workspace
            .llm
            .lock()
            .abort_handle()
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            was_aborted = true;
            break;
        }

        if calls.is_empty() {
            break;
        }

        let _ = on_event.send(StreamEvent::IterEnd { iter, count: calls.len() });

        for call in &calls {
            let _ = on_event.send(StreamEvent::ToolCall {
                iter,
                name: call.name.clone(),
                args: call.args.clone(),
                parse_error: call.parse_error.clone(),
            });

            let result = invoke_tool(&workspace, call.parse_error.as_deref(), &call.name, &call.args);
            let _ = on_event.send(StreamEvent::ToolResult {
                iter,
                name: call.name.clone(),
                result: result.clone(),
            });

            // Inline the result so the next iteration's prompt sees it.
            accumulated.push_str("\n<tool_result name=\"");
            accumulated.push_str(&call.name);
            accumulated.push_str("\">\n");
            accumulated.push_str(&serde_json::to_string(&result).unwrap_or_default());
            accumulated.push_str("\n</tool_result>\n");
        }
    }

    // 6. Persist the assistant turn. We always store the FULL text
    //    (tool blocks + results + narrative); the frontend decides what to
    //    render where.
    let final_text = accumulated.trim().to_string();
    let saved = workspace.memory.append_message(
        args.conversation_id,
        MessageRole::Assistant,
        &final_text,
    )?;
    // Persist the last-token hidden alongside the message so subsequent
    // feedback can update the user steering vector. The lookup keys on
    // message_id, so this is harmless when feedback never arrives.
    if let Some(h) = final_hidden.lock().take() {
        let _ = workspace.memory.message_hidden_save(saved.id, &h);
    }

    let payload = if let Some((code, message)) = had_error {
        StreamEvent::Error { code, message }
    } else if was_aborted {
        StreamEvent::Aborted { assistant_message_id: saved.id, partial: final_text }
    } else {
        StreamEvent::Done { assistant_message_id: saved.id, total_text: final_text }
    };
    let _ = on_event.send(payload);
    Ok(())
}

#[tauri::command]
pub fn chat_abort(workspace: State<'_, Arc<Workspace>>) {
    workspace
        .llm
        .lock()
        .abort_handle()
        .store(true, std::sync::atomic::Ordering::SeqCst);
}

// ---- helpers ---------------------------------------------------------------

fn wire_from_history(system: &str, history: &[Message]) -> Vec<WireMessage> {
    let mut wire = Vec::with_capacity(history.len() + 1);
    wire.push(WireMessage { role: WireRole::System, content: system.to_string() });
    for m in history {
        let role = match m.role {
            MessageRole::System => WireRole::System,
            MessageRole::User => WireRole::User,
            MessageRole::Assistant => WireRole::Assistant,
        };
        wire.push(WireMessage { role, content: m.content.clone() });
    }
    wire
}

fn invoke_tool(
    workspace: &Workspace,
    parse_error: Option<&str>,
    name: &str,
    args: &Value,
) -> Value {
    if let Some(e) = parse_error {
        return err_value("bad_json", &format!("tool args weren't valid JSON: {e}"));
    }
    let Some(tool) = tools::find(name) else {
        return err_value("unknown_tool", &format!("no tool named `{name}`"));
    };
    match tool.invoke(workspace, args) {
        Ok(v) => ok_value(v),
        Err(e) => err_value(e.code(), &e.to_string()),
    }
}

trait OrDefaultSampler {
    fn unwrap_or_default_sampler(self) -> LlmSamplerInput;
}
impl OrDefaultSampler for Option<LlmSamplerInput> {
    fn unwrap_or_default_sampler(self) -> LlmSamplerInput {
        self.unwrap_or(LlmSamplerInput {
            temperature: Some(0.7),
            top_k: Some(40),
            top_p: Some(0.95),
            repetition_penalty: Some(1.05),
            max_new_tokens: Some(512),
            seed: None,
        })
    }
}
