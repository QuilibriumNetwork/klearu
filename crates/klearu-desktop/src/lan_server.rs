//! Embedded LAN server.
//!
//! When the user toggles "LAN sharing" on, we spawn an Axum HTTP server
//! on `0.0.0.0:<port>` that:
//!
//!   * Serves the built React SPA from `web/dist/` (or an embedded
//!     fallback page if dist/ doesn't exist).
//!   * Exposes `POST /api/invoke` — a JSON envelope that dispatches to
//!     the same [`Workspace`] methods the Tauri IPC surface uses.
//!   * Exposes `GET /api/asset/<path>` — serves files from the data dir
//!     (images, etc) so `<img src>` on the remote browser works.
//!   * Requires a bearer token (random PIN generated at start) on every
//!     `/api/*` request so nobody on the LAN can poke the endpoints
//!     without the PIN the desktop user sees.
//!
//! Streaming commands (chat_send, image_generate) use Server-Sent Events
//! via `POST /api/stream`. The body is the same JSON envelope; the
//! response is `text/event-stream` with one `data:` line per event.
//!
//! The server holds an `Arc<Workspace>` — same instance the Tauri
//! commands use. All state is shared: memory, engines, incognito flag.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, Query, State as AxState};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::state::Workspace;

// Re-export for the dispatch closures that call command-layer methods.
pub use crate::commands;

/// Active LAN server handle. Dropping it shuts down the server.
pub struct LanHandle {
    pub port: u16,
    pub pin: String,
    pub addr: SocketAddr,
    /// SHA-256 fingerprint of the self-signed TLS certificate, formatted
    /// as `AA:BB:CC:…`. Shown on the desktop so the remote user can
    /// verify the cert presented by the browser matches.
    pub cert_fingerprint: String,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl LanHandle {
    pub fn stop(self) {
        let _ = self.shutdown_tx.send(());
    }
}

#[derive(Clone)]
struct ServerState {
    workspace: Arc<Workspace>,
    pin: String,
    dist_dir: Option<PathBuf>,
    data_dir: PathBuf,
}

/// Start the LAN server. Returns a handle the caller holds to keep it
/// alive; dropping or calling `.stop()` shuts it down.
pub async fn start(
    workspace: Arc<Workspace>,
    port: u16,
) -> Result<LanHandle, crate::error::DesktopError> {
    let pin = generate_pin();
    let data_dir = workspace.data_dir();

    // Try to locate web/dist/ relative to the current exe. During
    // `tauri dev` the exe is in target/debug or target/release; dist/
    // is at ../../web/dist from there.
    let dist_dir = find_dist_dir();

    let state = ServerState {
        workspace,
        pin: pin.clone(),
        dist_dir: dist_dir.clone(),
        data_dir,
    };

    let app = Router::new()
        .route("/api/invoke", post(handle_invoke))
        .route("/api/stream", post(handle_stream))
        .route("/api/asset/{*path}", get(handle_asset))
        .route("/api/ping", get(handle_ping))
        .fallback(get(handle_static))
        .layer(axum::extract::DefaultBodyLimit::max(100 * 1024 * 1024)) // 100 MB for inpaint bytes
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    // Rustls requires a process-level crypto provider to be installed
    // before any TLS operation. `ring` is compiled in via our features;
    // install it as the default (idempotent — second call is a no-op).
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Generate a self-signed TLS cert so the LAN link is HTTPS.
    let (tls_config, cert_fingerprint) = make_self_signed_tls()
        .await
        .map_err(|e| crate::error::DesktopError::Io(format!("TLS setup: {e}")))?;

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let bound_addr = addr;
    tokio::spawn(async move {
        let handle = axum_server::Handle::new();
        let h2 = handle.clone();
        tokio::spawn(async move { let _ = shutdown_rx.await; h2.shutdown(); });
        axum_server::bind_rustls(addr, tls_config)
            .handle(handle)
            .serve(app.into_make_service())
            .await
            .ok();
    });

    tracing::info!(%bound_addr, %cert_fingerprint, "LAN server started (HTTPS, PIN: {pin})");
    Ok(LanHandle {
        port: bound_addr.port(),
        pin,
        addr: bound_addr,
        cert_fingerprint,
        shutdown_tx,
    })
}

// ---- auth -----------------------------------------------------------------

fn check_auth(headers: &HeaderMap, pin: &str) -> Result<(), StatusCode> {
    // Accept `Authorization: Bearer <pin>` or `?pin=<pin>` query param.
    if let Some(auth) = headers.get("authorization") {
        if let Ok(s) = auth.to_str() {
            if s.strip_prefix("Bearer ").unwrap_or("") == pin {
                return Ok(());
            }
        }
    }
    Err(StatusCode::UNAUTHORIZED)
}

#[derive(Deserialize)]
struct PinQuery {
    pin: Option<String>,
}

fn check_auth_or_query(headers: &HeaderMap, query: &PinQuery, pin: &str) -> Result<(), StatusCode> {
    if check_auth(headers, pin).is_ok() {
        return Ok(());
    }
    if let Some(qp) = &query.pin {
        if qp == pin {
            return Ok(());
        }
    }
    Err(StatusCode::UNAUTHORIZED)
}

// ---- handlers -------------------------------------------------------------

#[derive(Deserialize)]
struct InvokeBody {
    command: String,
    #[serde(default)]
    args: serde_json::Value,
}

async fn handle_invoke(
    AxState(state): AxState<ServerState>,
    headers: HeaderMap,
    Json(body): Json<InvokeBody>,
) -> Response {
    if let Err(status) = check_auth(&headers, &state.pin) {
        return status.into_response();
    }
    match dispatch_command(&state.workspace, &body.command, body.args).await {
        Ok(val) => Json(val).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": e.to_string()
        }))).into_response(),
    }
}

async fn handle_stream(
    AxState(state): AxState<ServerState>,
    headers: HeaderMap,
    Json(body): Json<InvokeBody>,
) -> Response {
    if let Err(status) = check_auth(&headers, &state.pin) {
        return status.into_response();
    }
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
    let workspace = state.workspace.clone();
    let command = body.command;
    let args = body.args;

    tokio::spawn(async move {
        let result = dispatch_stream_command(&workspace, &command, args, tx.clone()).await;
        if let Err(e) = result {
            let _ = tx.send(Ok(Event::default().data(
                serde_json::json!({"type": "error", "code": e.code(), "message": e.to_string()}).to_string()
            ))).await;
        }
    });

    let stream = ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn handle_asset(
    AxState(state): AxState<ServerState>,
    headers: HeaderMap,
    Query(query): Query<PinQuery>,
    Path(path): Path<String>,
) -> Response {
    if let Err(status) = check_auth_or_query(&headers, &query, &state.pin) {
        return status.into_response();
    }
    let full = state.data_dir.join(&path);
    if !full.starts_with(&state.data_dir) {
        return StatusCode::FORBIDDEN.into_response();
    }
    match tokio::fs::read(&full).await {
        Ok(bytes) => {
            let content_type = if path.ends_with(".png") {
                "image/png"
            } else if path.ends_with(".jpg") || path.ends_with(".jpeg") {
                "image/jpeg"
            } else {
                "application/octet-stream"
            };
            ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn handle_ping(
    AxState(state): AxState<ServerState>,
    headers: HeaderMap,
) -> Response {
    if let Err(status) = check_auth(&headers, &state.pin) {
        return status.into_response();
    }
    Json(serde_json::json!({"ok": true})).into_response()
}

async fn handle_static(
    AxState(state): AxState<ServerState>,
    req: axum::extract::Request,
) -> Response {
    if let Some(dist) = &state.dist_dir {
        let path = req.uri().path().trim_start_matches('/');
        let file = if path.is_empty() { "index.html" } else { path };
        let full = dist.join(file);
        if full.is_file() {
            if let Ok(bytes) = tokio::fs::read(&full).await {
                let ct = if file.ends_with(".html") { "text/html; charset=utf-8" }
                    else if file.ends_with(".js") { "application/javascript" }
                    else if file.ends_with(".css") { "text/css" }
                    else if file.ends_with(".json") { "application/json" }
                    else if file.ends_with(".wasm") { "application/wasm" }
                    else { "application/octet-stream" };
                return ([(axum::http::header::CONTENT_TYPE, ct)], bytes).into_response();
            }
        }
        // SPA fallback: serve index.html for all unrecognized paths so
        // client-side routing works.
        let index = dist.join("index.html");
        if let Ok(bytes) = tokio::fs::read(&index).await {
            return ([(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")], bytes).into_response();
        }
    }
    Html("<h1>Klearu LAN</h1><p>Build the frontend first: <code>cd web && npm run build</code></p>").into_response()
}

// ---- command dispatch ------------------------------------------------------

async fn dispatch_command(
    workspace: &Workspace,
    command: &str,
    args: serde_json::Value,
) -> Result<serde_json::Value, crate::error::DesktopError> {
    use crate::commands::*;

    macro_rules! sync_cmd {
        ($fn:expr) => {{
            let r = $fn;
            serde_json::to_value(&r).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }};
        ($fn:expr, result) => {{
            let r = $fn?;
            serde_json::to_value(&r).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }};
    }

    match command {
        "app_info" => {
            // app_info takes State<Workspace> — we need to create a fake State.
            // Since we hold Arc<Workspace>, we can construct via a wrapper.
            let info = app::AppInfo {
                name: "Klearu",
                version: env!("CARGO_PKG_VERSION"),
                data_dir: workspace.data_dir().to_string_lossy().into_owned(),
                features: app::AppFeatures {
                    chat: true,
                    memory: true,
                    image_generate: true,
                    inpaint: true,
                    classify: true,
                    training: true,
                },
            };
            serde_json::to_value(&info).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "llm_status" => {
            let s = workspace.llm.lock().status();
            serde_json::to_value(&s).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "memory_list" => {
            let kind: Option<crate::memory::MemoryKind> =
                args.get("kind").and_then(|v| serde_json::from_value(v.clone()).ok());
            let list = workspace.memory.list_memories(kind)?;
            serde_json::to_value(&list).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "conversations_list" => {
            let list = workspace.memory.list_conversations()?;
            serde_json::to_value(&list).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "conversation_messages" => {
            let conv_id = args.get("conversationId")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| crate::error::DesktopError::bad_input("conversationId required"))?;
            let list = workspace.memory.conversation_messages(conv_id)?;
            serde_json::to_value(&list).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "conversation_create" => {
            let title = args.get("title").and_then(|v| v.as_str()).unwrap_or("New chat");
            let c = workspace.memory.create_conversation(title)?;
            serde_json::to_value(&c).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "image_list" => {
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(60) as usize;
            let list = workspace.memory.image_list(limit)?;
            serde_json::to_value(&list).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "image_abort" => {
            workspace.diffusion.abort();
            Ok(serde_json::json!({"ok": true}))
        }
        "incognito_status" => {
            let active = workspace.is_incognito();
            Ok(serde_json::json!({"active": active}))
        }
        "vision_status" => {
            let s = workspace.vision.lock().status();
            serde_json::to_value(&s).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "settings_get" => {
            let key = args.get("key")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::error::DesktopError::bad_input("key required"))?;
            let val = workspace.memory.settings_get(key)?;
            Ok(serde_json::json!(val))
        }
        "settings_set" => {
            let key = args.get("key")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::error::DesktopError::bad_input("key required"))?;
            let value = args.get("value")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::error::DesktopError::bad_input("value required"))?;
            workspace.memory.settings_set(key, value)?;
            if key.starts_with("image.") {
                let get = |k: &str| workspace.memory.settings_get(k).ok().flatten();
                let new_cfg = crate::engine::diffusion::DiffusionConfig {
                    sd_generate_path: get("image.sd_generate_path").map(std::path::PathBuf::from),
                    checkpoint: get("image.checkpoint").map(std::path::PathBuf::from),
                    tokenizer_l: get("image.tokenizer_l").map(std::path::PathBuf::from),
                    tokenizer_g: get("image.tokenizer_g").map(std::path::PathBuf::from),
                    output_dir: get("image.output_dir").map(std::path::PathBuf::from),
                    motion_adapter: get("image.motion_adapter").map(std::path::PathBuf::from),
                };
                workspace.diffusion.set_config(new_cfg);
            }
            Ok(serde_json::json!(null))
        }
        "settings_get_all" => {
            let pairs = workspace.memory.settings_get_all()?;
            let entries: Vec<serde_json::Value> = pairs.into_iter()
                .map(|(k, v)| serde_json::json!({"key": k, "value": v}))
                .collect();
            Ok(serde_json::Value::Array(entries))
        }
        "feedback_list" => {
            let kind = args.get("args")
                .and_then(|a| a.get("target_kind"))
                .and_then(|v| v.as_str());
            let limit = args.get("args")
                .and_then(|a| a.get("limit"))
                .and_then(|v| v.as_u64())
                .unwrap_or(200) as usize;
            let list = workspace.memory.feedback_list(kind, limit)?;
            serde_json::to_value(&list).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "user_vector_status" => {
            let hs = workspace.llm.lock().hidden_size().unwrap_or(0);
            let state = if hs > 0 {
                workspace.memory.user_vector_load("default", hs)?
            } else {
                workspace.memory.user_vector_load("default", 1)?
            };
            let status = crate::commands::feedback::UserVectorStatus::from(state);
            serde_json::to_value(&status).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "llm_load" => {
            let dir = args.get("modelDir")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::error::DesktopError::bad_input("modelDir required"))?;
            let engine = workspace.llm.clone();
            let path = std::path::PathBuf::from(dir);
            let status = tokio::task::spawn_blocking(move || engine.lock().load(&path))
                .await
                .map_err(|e| crate::error::DesktopError::engine(format!("join: {e}")))?
                .map_err(|e| crate::error::DesktopError::engine(format!("{e}")))?;
            serde_json::to_value(&status).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "llm_unload" => {
            workspace.llm.lock().unload();
            Ok(serde_json::json!(null))
        }
        "conversation_delete" => {
            let id = args.get("id").and_then(|v| v.as_i64())
                .ok_or_else(|| crate::error::DesktopError::bad_input("id required"))?;
            workspace.memory.delete_conversation(id)?;
            Ok(serde_json::json!(null))
        }
        "memory_upsert" => {
            let entry: crate::memory::MemoryUpsert =
                serde_json::from_value(args.get("entry").cloned().unwrap_or_default())
                    .map_err(|e| crate::error::DesktopError::bad_input(format!("parse: {e}")))?;
            let saved = workspace.memory.upsert_memory(entry)?;
            serde_json::to_value(&saved).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "memory_delete" => {
            let id = args.get("id").and_then(|v| v.as_i64())
                .ok_or_else(|| crate::error::DesktopError::bad_input("id required"))?;
            workspace.memory.delete_memory(id)?;
            Ok(serde_json::json!(null))
        }
        "image_delete" => {
            let id = args.get("id").and_then(|v| v.as_i64())
                .ok_or_else(|| crate::error::DesktopError::bad_input("id required"))?;
            let delete_file = args.get("deleteFile").and_then(|v| v.as_bool()).unwrap_or(false);
            if let Some(path) = workspace.memory.image_delete(id)? {
                if delete_file { let _ = std::fs::remove_file(path); }
            }
            Ok(serde_json::json!(null))
        }
        "feedback_set" => {
            let a = args.get("args").cloned().unwrap_or_default();
            let target_kind = a.get("target_kind").and_then(|v| v.as_str()).unwrap_or("");
            let target_id = a.get("target_id").and_then(|v| v.as_i64()).unwrap_or(0);
            let rating = a.get("rating").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let note = a.get("note").and_then(|v| v.as_str());
            let row = workspace.memory.feedback_set(target_kind, target_id, rating, note)?;
            serde_json::to_value(&row).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "feedback_get" => {
            let tk = args.get("targetKind").and_then(|v| v.as_str()).unwrap_or("");
            let tid = args.get("targetId").and_then(|v| v.as_i64()).unwrap_or(0);
            let row = workspace.memory.feedback_get(tk, tid)?;
            serde_json::to_value(&row).map_err(|e| crate::error::DesktopError::Other(e.to_string()))
        }
        "chat_abort" => {
            workspace.llm.lock().abort_handle().store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(serde_json::json!(null))
        }
        other => Err(crate::error::DesktopError::bad_input(format!(
            "LAN invoke: unknown command `{other}` (streaming commands use /api/stream)"
        ))),
    }
}

async fn dispatch_stream_command(
    workspace: &Workspace,
    command: &str,
    args: serde_json::Value,
    tx: mpsc::Sender<Result<Event, std::convert::Infallible>>,
) -> Result<(), crate::error::DesktopError> {
    match command {
        "chat_send" => {
            let chat_args: crate::commands::chat::ChatSendArgs =
                serde_json::from_value(args.get("args").cloned().unwrap_or_default())
                    .map_err(|e| crate::error::DesktopError::bad_input(format!("parse args: {e}")))?;

            // Reuse the same logic as the Tauri command but with an SSE
            // sender instead of a Tauri Channel.
            workspace.memory.append_message(
                chat_args.conversation_id,
                crate::memory::MessageRole::User,
                &chat_args.content,
            )?;

            let memories = workspace.memory.list_memories(None).unwrap_or_default();
            let tools = crate::tools::registry();
            let system = crate::intent::build_system_prompt(
                &tools, &memories, chat_args.system_prompt.as_deref(),
            );
            let history = workspace.memory.conversation_messages(chat_args.conversation_id)?;
            let mut wire: Vec<crate::engine::llm::WireMessage> = Vec::with_capacity(history.len() + 1);
            wire.push(crate::engine::llm::WireMessage {
                role: crate::engine::llm::WireRole::System,
                content: system,
            });
            for m in &history {
                wire.push(crate::engine::llm::WireMessage {
                    role: match m.role {
                        crate::memory::MessageRole::System => crate::engine::llm::WireRole::System,
                        crate::memory::MessageRole::User => crate::engine::llm::WireRole::User,
                        crate::memory::MessageRole::Assistant => crate::engine::llm::WireRole::Assistant,
                    },
                    content: m.content.clone(),
                });
            }
            let sampler = chat_args.sampler.unwrap_or(crate::engine::llm::LlmSamplerInput {
                temperature: Some(0.7), top_k: Some(40), top_p: Some(0.95),
                repetition_penalty: Some(1.05), max_new_tokens: Some(512), seed: None,
            });
            let hidden_size = workspace.llm.lock().hidden_size().unwrap_or(0);
            let (steering_vec, alpha) = if hidden_size > 0 {
                match workspace.memory.user_vector_load("default", hidden_size) {
                    Ok(s) if s.n_pos + s.n_neg > 0 => (s.vec, s.alpha),
                    _ => (Vec::new(), 0.0),
                }
            } else { (Vec::new(), 0.0) };

            let llm = workspace.llm.clone();
            let tx2 = tx.clone();
            let conv_id = chat_args.conversation_id;
            let memory = workspace.memory.clone();

            tokio::task::spawn_blocking(move || {
                let mut engine = llm.lock();
                let text = engine.stream_reply(
                    wire, sampler, &steering_vec, alpha,
                    |tok| {
                        let ev = serde_json::json!({"type": "token", "text": tok});
                        let _ = tx2.blocking_send(Ok(Event::default().data(ev.to_string())));
                    },
                    |_hidden| {},
                );
                match text {
                    Ok(text) => {
                        let msg = memory.append_message(conv_id, crate::memory::MessageRole::Assistant, &text);
                        let msg_id = msg.map(|m| m.id).unwrap_or(0);
                        let ev = serde_json::json!({"type": "done", "assistant_message_id": msg_id, "total_text": text});
                        let _ = tx2.blocking_send(Ok(Event::default().data(ev.to_string())));
                    }
                    Err(e) => {
                        let ev = serde_json::json!({"type": "error", "code": "engine", "message": e.to_string()});
                        let _ = tx2.blocking_send(Ok(Event::default().data(ev.to_string())));
                    }
                }
            }).await.map_err(|e| crate::error::DesktopError::engine(format!("join: {e}")))?;
            Ok(())
        }
        "image_generate" => {
            // Frontend sends { args: { prompt, ... } } — unwrap the inner
            // "args" field to get the actual GenerateRequest.
            let inner = args.get("args").cloned().unwrap_or(args.clone());
            let req: crate::engine::diffusion::GenerateRequest =
                serde_json::from_value(inner.clone())
                    .map_err(|e| crate::error::DesktopError::bad_input(format!("parse: {e}")))?;
            // Reference images arrive as a base64 array (remote/mobile) or, on
            // the Tauri IPC fallback, as the `ref_image_bytes` JSON number
            // arrays already deserialized into `req`. Prefer the base64 form.
            let ref_image_bytes: Vec<Vec<u8>> = if let Some(arr) =
                inner.get("ref_image_bytes_b64").and_then(|v| v.as_array())
            {
                use base64::Engine;
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|b64| {
                        base64::engine::general_purpose::STANDARD
                            .decode(b64)
                            .map_err(|e| crate::error::DesktopError::bad_input(format!("ref b64: {e}")))
                    })
                    .collect::<crate::error::Result<Vec<_>>>()?
            } else {
                req.ref_image_bytes.clone().unwrap_or_default()
            };
            let engine = workspace.diffusion.clone();
            let memory = workspace.memory.clone();
            let (etx, mut erx) = mpsc::unbounded_channel::<crate::engine::diffusion::GenEvent>();
            let tx2 = tx.clone();

            // Forward gen events as SSE
            tokio::spawn(async move {
                while let Some(ev) = erx.recv().await {
                    let json = serde_json::to_string(&ev).unwrap_or_default();
                    let _ = tx2.send(Ok(Event::default().data(json))).await;
                }
            });

            let prompt_for_record = req.prompt.clone();
            let neg_for_record = req.negative_prompt.clone();
            let steps_for_record = req.steps.unwrap_or(25) as usize;
            let cfg_for_record = req.cfg_scale.unwrap_or(7.5);
            let seed_for_record = req.seed.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| (d.as_nanos() as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    .unwrap_or(42)
            });
            let in_proc = crate::engine::diffusion::InProcessRequest {
                prompt: req.prompt,
                negative_prompt: req.negative_prompt.unwrap_or_default(),
                steps: steps_for_record,
                cfg_scale: cfg_for_record,
                seed: seed_for_record,
                ref_image_bytes,
                cond_bias: None,
                ..Default::default()
            };
            let result = engine.generate_in_process(in_proc, etx).await?;

            // Encode PNG, then either write to disk or embed as data URI
            // depending on incognito status — same logic as the Tauri
            // command path in commands::image.
            let png = crate::commands::image::encode_chw_to_png(
                &result.rgb_chw, result.height, result.width
            )?;
            let path_str = if workspace.is_incognito() {
                use base64::Engine;
                let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
                format!("data:image/png;base64,{b64}")
            } else {
                let data_dir = workspace.data_dir();
                let images_dir = data_dir.join("images");
                let _ = std::fs::create_dir_all(&images_dir);
                let p = images_dir.join(format!("{}-{}.png",
                    crate::commands::image::now_secs(), result.job_id));
                std::fs::write(&p, &png)
                    .map_err(|e| crate::error::DesktopError::Io(format!("write: {e}")))?;
                p.to_string_lossy().into_owned()
            };

            let rec = crate::memory::ImageRecord {
                path: &path_str,
                prompt: &prompt_for_record,
                negative_prompt: neg_for_record.as_deref(),
                seed: Some(seed_for_record as i64),
                steps: Some(steps_for_record as i64),
                cfg_scale: Some(cfg_for_record as f64),
                width: Some(result.width as i64), height: Some(result.height as i64),
                checkpoint: None, conversation_id: None,
                created_at: crate::commands::image::now_secs(),
            };
            let _ = memory.image_record(&rec);

            let done = serde_json::json!({"type": "done", "job_id": result.job_id, "path": path_str});
            let _ = tx.send(Ok(Event::default().data(done.to_string()))).await;
            Ok(())
        }
        "image_inpaint" => {
            let inner = args.get("args").cloned().unwrap_or(args.clone());
            let prompt: String = inner.get("prompt")
                .and_then(|v| v.as_str()).unwrap_or("").to_string();
            let neg: Option<String> = inner.get("negative_prompt")
                .and_then(|v| v.as_str()).map(|s| s.to_string());
            let steps = inner.get("steps").and_then(|v| v.as_u64()).unwrap_or(25) as usize;
            let cfg_scale = inner.get("cfg_scale").and_then(|v| v.as_f64()).unwrap_or(7.5) as f32;
            let strength = inner.get("strength").and_then(|v| v.as_f64()).unwrap_or(0.85) as f32;
            let seed = inner.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| (d.as_nanos() as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    .unwrap_or(42)
            });
            // Bytes arrive as base64 strings (remote/mobile) or JSON
            // number arrays (Tauri IPC fallback). Try b64 first.
            let input_bytes: Vec<u8> = if let Some(b64) = inner.get("input_bytes_b64").and_then(|v| v.as_str()) {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.decode(b64)
                    .map_err(|e| crate::error::DesktopError::bad_input(format!("input b64: {e}")))?
            } else {
                inner.get("input_bytes")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|n| n.as_u64().map(|n| n as u8)).collect())
                    .unwrap_or_default()
            };
            let mask_bytes: Option<Vec<u8>> = if let Some(b64) = inner.get("mask_bytes_b64").and_then(|v| v.as_str()) {
                use base64::Engine;
                Some(base64::engine::general_purpose::STANDARD.decode(b64)
                    .map_err(|e| crate::error::DesktopError::bad_input(format!("mask b64: {e}")))?)
            } else {
                inner.get("mask_bytes")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|n| n.as_u64().map(|n| n as u8)).collect())
            };

            tracing::info!(
                input_len = input_bytes.len(),
                mask_len = mask_bytes.as_ref().map(|m| m.len()).unwrap_or(0),
                "image_inpaint stream: received bytes"
            );
            if input_bytes.is_empty() {
                return Err(crate::error::DesktopError::bad_input("input_bytes required (0 bytes received — base64 decode may have failed)"));
            }
            // Quick PNG header check — first 8 bytes must be the PNG magic.
            if input_bytes.len() < 8 || &input_bytes[..4] != b"\x89PNG" {
                return Err(crate::error::DesktopError::bad_input(format!(
                    "input_bytes ({} bytes) doesn't start with PNG magic — got {:?}",
                    input_bytes.len(),
                    &input_bytes[..input_bytes.len().min(8)]
                )));
            }

            let engine = workspace.diffusion.clone();
            let memory = workspace.memory.clone();
            let (etx, mut erx) = mpsc::unbounded_channel::<crate::engine::diffusion::GenEvent>();
            let tx2 = tx.clone();
            tokio::spawn(async move {
                while let Some(ev) = erx.recv().await {
                    let json = serde_json::to_string(&ev).unwrap_or_default();
                    let _ = tx2.send(Ok(Event::default().data(json))).await;
                }
            });

            let in_proc = crate::engine::diffusion::InProcessRequest {
                prompt: prompt.clone(),
                negative_prompt: neg.clone().unwrap_or_default(),
                steps,
                cfg_scale,
                seed,
                input_image_bytes: Some(input_bytes),
                mask_bytes,
                strength,
                cond_bias: None,
                ..Default::default()
            };
            let result = engine.generate_in_process(in_proc, etx).await?;

            let png = crate::commands::image::encode_chw_to_png(
                &result.rgb_chw, result.height, result.width
            )?;
            let path_str = if workspace.is_incognito() {
                use base64::Engine;
                let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
                format!("data:image/png;base64,{b64}")
            } else {
                let data_dir = workspace.data_dir();
                let images_dir = data_dir.join("images");
                let _ = std::fs::create_dir_all(&images_dir);
                let p = images_dir.join(format!("{}-{}.png",
                    crate::commands::image::now_secs(), result.job_id));
                std::fs::write(&p, &png)
                    .map_err(|e| crate::error::DesktopError::Io(format!("write: {e}")))?;
                p.to_string_lossy().into_owned()
            };

            let rec = crate::memory::ImageRecord {
                path: &path_str, prompt: &prompt,
                negative_prompt: neg.as_deref(),
                seed: Some(seed as i64), steps: Some(steps as i64),
                cfg_scale: Some(cfg_scale as f64),
                width: Some(result.width as i64), height: Some(result.height as i64),
                checkpoint: None, conversation_id: None,
                created_at: crate::commands::image::now_secs(),
            };
            let _ = memory.image_record(&rec);

            let done = serde_json::json!({"type": "done", "job_id": result.job_id, "path": path_str});
            let _ = tx.send(Ok(Event::default().data(done.to_string()))).await;
            Ok(())
        }
        other => Err(crate::error::DesktopError::bad_input(format!(
            "LAN stream: unknown command `{other}`"
        ))),
    }
}

// ---- helpers ---------------------------------------------------------------

fn generate_pin() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    format!("{:06}", rng.gen_range(0..1_000_000u32))
}

/// Returns `(RustlsConfig, fingerprint_hex)`. The fingerprint is the
/// SHA-256 hash of the DER-encoded certificate, formatted as colon-
/// separated uppercase hex bytes (`AA:BB:CC:…`). The desktop UI shows
/// this so the remote user can compare it with the cert the browser
/// presents on first connect.
async fn make_self_signed_tls() -> Result<(axum_server::tls_rustls::RustlsConfig, String), Box<dyn std::error::Error + Send + Sync>> {
    use rcgen::{CertificateParams, KeyPair};
    let key_pair = KeyPair::generate()?;
    let mut params = CertificateParams::new(vec!["localhost".into()])?;
    params.distinguished_name.push(
        rcgen::DnType::CommonName, rcgen::DnValue::Utf8String("Klearu LAN".into()),
    );
    if let Ok(addrs) = super::commands::lan::local_ipv4_addrs_pub() {
        for ip in addrs {
            params.subject_alt_names.push(
                rcgen::SanType::IpAddress(std::net::IpAddr::V4(ip)),
            );
        }
    }
    let cert = params.self_signed(&key_pair)?;
    // Compute SHA-256 fingerprint from the DER-encoded certificate.
    let der = cert.der();
    let digest = ring::digest::digest(&ring::digest::SHA256, der);
    let fingerprint = digest
        .as_ref()
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(":");
    let cert_pem = cert.pem();
    let key_pem = key_pair.serialize_pem();
    let config = axum_server::tls_rustls::RustlsConfig::from_pem(
        cert_pem.into_bytes().into(),
        key_pem.into_bytes().into(),
    ).await?;
    Ok((config, fingerprint))
}


fn find_dist_dir() -> Option<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().take(6) {
            let candidate = ancestor.join("web").join("dist");
            if candidate.join("index.html").exists() {
                return Some(candidate);
            }
        }
    }
    None
}
