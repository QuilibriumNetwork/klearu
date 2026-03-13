//! Server A: Proxy + Party 0
//!
//! - Accepts thin WebSocket clients via binary MPC protocol
//! - Communicates with Server B over TCP for MPC
//! - Client sends DPF keys (~327 bytes each) for token PIR
//! - Returns Q16.16 logit shares to client
//!
//! Usage:
//!   server-a --model-dir ./SmolLM-135M --ws-port 3000 --server-b-addr 127.0.0.1:9090
//!
//! For HTTPS/WSS (recommended):
//!   server-a --model-dir ./SmolLM-135M --tls-cert cert.pem --tls-key key.pem

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, State, WebSocketUpgrade};
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::{Router, routing::get};
use clap::Parser;
use klearu_dpf::AesPrg;
use klearu_llm::generate::pipeline::{Pipeline, detect_eos_tokens};
use klearu_llm::model::kv_cache::KvCacheStore;
use klearu_mpc::embedding_pir::{
    deserialize_dpf_key, dpf_depth_for_vocab, pir_compute_embedding_share,
    pir_compute_embedding_shares_batch,
};
use klearu_private::private_model::create_deltanet_states;
use klearu_private::tcp_transport::TcpTransport;
use klearu_mpc::transport::Transport;
use rand::Rng;
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

#[path = "../two_party.rs"]
mod two_party;

// Protocol tags
const TAG_SESSION_INIT: u8 = 0x01;
const TAG_SESSION_READY: u8 = 0x02;
const TAG_NEXT_TOKEN: u8 = 0x52;
const TAG_LOGIT_SHARE: u8 = 0x51;
const TAG_EOS_SIGNAL: u8 = 0x42;
const TAG_ERROR: u8 = 0xF0;

#[derive(Parser)]
#[command(name = "server-a", about = "Server A: proxy + Party 0 for 2-server MPC")]
struct Args {
    /// Path to the model directory (config.json, tokenizer.json, *.safetensors)
    #[arg(long)]
    model_dir: PathBuf,

    /// WebSocket port for client connections
    #[arg(long, default_value = "3000")]
    ws_port: u16,

    /// Server B's TCP address (host:port)
    #[arg(long, default_value = "127.0.0.1:9090")]
    server_b_addr: String,

    /// Server B's WebSocket URL for client connections (sent to client in SessionReady)
    #[arg(long, default_value = "ws://localhost:9091/ws")]
    server_b_ws: String,

    /// Maximum new tokens to generate per request
    #[arg(long, default_value = "2048")]
    max_tokens: usize,

    /// Path to static files to serve (built web app)
    #[arg(long)]
    static_dir: Option<PathBuf>,

    /// TLS certificate file (PEM). Enables HTTPS/WSS when both --tls-cert and --tls-key are set.
    #[arg(long)]
    tls_cert: Option<PathBuf>,

    /// TLS private key file (PEM)
    #[arg(long)]
    tls_key: Option<PathBuf>,
}

struct AppState {
    model: klearu_llm::Model,
    eos_token_ids: Vec<u32>,
    server_b_addr: String,
    server_b_ws: String,
    max_tokens: usize,
    model_dir: PathBuf,
    embedding_q16: Vec<u32>,
    prg_key: [u8; 16],
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_client(socket, state))
}

async fn handle_client(mut socket: WebSocket, state: Arc<AppState>) {
    tracing::info!("Client connected via WebSocket");

    // Wait for SessionInit (binary, tag 0x01)
    let init_data = match socket.recv().await {
        Some(Ok(Message::Binary(data))) => data.to_vec(),
        Some(Ok(Message::Close(_))) | None => {
            tracing::info!("Client disconnected before init");
            return;
        }
        other => {
            tracing::warn!("Expected binary SessionInit, got: {:?}", other.map(|r| r.map(|_| "...")));
            return;
        }
    };

    if init_data.is_empty() || init_data[0] != TAG_SESSION_INIT {
        tracing::warn!("Expected SessionInit (0x01), got {:?}", init_data.first());
        let _ = send_error(&mut socket, "Expected SessionInit message").await;
        return;
    }

    // Parse SessionInit: security(1) + maxNew(4) + temp(4) + topK(4) + topP(4) + numPrompt(4) = 21 bytes
    if init_data.len() < 22 {
        let _ = send_error(&mut socket, "SessionInit too short").await;
        return;
    }
    let payload = &init_data[1..];
    let _security = payload[0];
    let max_new = u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]) as usize;
    let _temperature = f32::from_le_bytes([payload[5], payload[6], payload[7], payload[8]]);
    let _top_k = u32::from_le_bytes([payload[9], payload[10], payload[11], payload[12]]);
    let _top_p = f32::from_le_bytes([payload[13], payload[14], payload[15], payload[16]]);
    let num_prompt = u32::from_le_bytes([payload[17], payload[18], payload[19], payload[20]]) as usize;

    let max_new = max_new.min(state.max_tokens);

    tracing::info!("SessionInit: prompt={}, max_new={}", num_prompt, max_new);

    // Generate session_id and triple_seed
    let session_id: u64 = rand::thread_rng().gen();
    let triple_seed: u64 = rand::thread_rng().gen();

    // Connect to Server B via TCP
    let stream = match TcpStream::connect(&state.server_b_addr) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to connect to Server B at {}: {}", state.server_b_addr, e);
            let _ = send_error(&mut socket, &format!("Cannot connect to server B: {}", e)).await;
            return;
        }
    };
    let mut tcp_transport = TcpTransport::new(stream).expect("TcpTransport::new failed");

    // Send session init to Server B: session_id(8) + triple_seed(8) + prg_key(16) + num_prompt(4) + max_new(4) = 40 bytes
    let mut tcp_init = Vec::with_capacity(40);
    tcp_init.extend_from_slice(&session_id.to_le_bytes());
    tcp_init.extend_from_slice(&triple_seed.to_le_bytes());
    tcp_init.extend_from_slice(&state.prg_key);
    tcp_init.extend_from_slice(&(num_prompt as u32).to_le_bytes());
    tcp_init.extend_from_slice(&(max_new as u32).to_le_bytes());
    if let Err(e) = tcp_transport.send(&tcp_init) {
        let _ = send_error(&mut socket, &format!("Failed to init Server B: {}", e)).await;
        return;
    }

    tracing::info!(
        "Connected to Server B, session={:#018x}, seed={:#018x}, prompt={}, max_new={}",
        session_id, triple_seed, num_prompt, max_new,
    );

    // Send SessionReady to client with config JSON
    let config = &state.model.config;
    let prg_key_hex: String = state.prg_key.iter().map(|b| format!("{:02x}", b)).collect();
    let config_json = serde_json::json!({
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "max_seq_len": config.max_seq_len,
        "rms_norm_eps": config.rms_norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
        "model_type": config.model_type,
        "eos_token_ids": state.eos_token_ids,
        "session_id": format!("{:016x}", session_id),
        "triple_seed": format!("{:016x}", triple_seed),
        "server_b_ws": state.server_b_ws,
        "prg_key": prg_key_hex,
    });
    let config_bytes = serde_json::to_vec(&config_json).unwrap();
    let mut ready_msg = Vec::with_capacity(1 + config_bytes.len());
    ready_msg.push(TAG_SESSION_READY);
    ready_msg.extend_from_slice(&config_bytes);
    if socket.send(Message::Binary(ready_msg.into())).await.is_err() {
        return;
    }

    let vocab_size = config.vocab_size;
    let depth = dpf_depth_for_vocab(vocab_size);
    let expected_key_len = 1 + 16 + 4 + (depth as usize) * 17;

    // Channels for WS ↔ blocking thread communication
    let (share_tx, share_rx) = std::sync::mpsc::channel::<Option<Vec<u8>>>();
    let (result_tx, mut result_rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, String>>(4);

    let model_clone = state.clone();
    let blocking_handle = tokio::task::spawn_blocking(move || {
        run_party0_blocking(
            &model_clone.model,
            &model_clone.embedding_q16,
            &model_clone.prg_key,
            num_prompt,
            max_new,
            0, // party
            triple_seed,
            tcp_transport,
            share_rx,
            result_tx,
        );
    });

    // Main loop: forward DPF keys from WS to blocking thread, results back to WS
    let mut steps_done = 0;

    loop {
        tokio::select! {
            // Receive DPF key from client WS
            ws_msg = socket.recv() => {
                match ws_msg {
                    Some(Ok(Message::Binary(data))) => {
                        if data.is_empty() {
                            let _ = share_tx.send(None);
                            break;
                        }
                        match data[0] {
                            TAG_NEXT_TOKEN => {
                                let key_data = data[1..].to_vec();
                                if key_data.len() != expected_key_len {
                                    let _ = send_error(&mut socket, &format!(
                                        "Bad DPF key length: {} (expected {})", key_data.len(), expected_key_len
                                    )).await;
                                    let _ = share_tx.send(None);
                                    break;
                                }
                                if share_tx.send(Some(key_data)).is_err() {
                                    break;
                                }
                            }
                            TAG_EOS_SIGNAL => {
                                tracing::info!("Client sent EOS");
                                let _ = share_tx.send(None);
                                break;
                            }
                            other => {
                                tracing::warn!("Unexpected tag: 0x{:02x}", other);
                                let _ = share_tx.send(None);
                                break;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        let _ = share_tx.send(None);
                        break;
                    }
                    _ => {}
                }
            }
            // Receive logit shares from blocking thread, forward to client
            result = result_rx.recv() => {
                match result {
                    Some(Ok(logit_data)) => {
                        let mut payload = Vec::with_capacity(1 + logit_data.len());
                        payload.push(TAG_LOGIT_SHARE);
                        payload.extend_from_slice(&logit_data);
                        if socket.send(Message::Binary(payload.into())).await.is_err() {
                            let _ = share_tx.send(None);
                            break;
                        }
                        steps_done += 1;
                    }
                    Some(Err(e)) => {
                        let _ = send_error(&mut socket, &e).await;
                        let _ = share_tx.send(None);
                        break;
                    }
                    None => {
                        // Blocking thread finished
                        break;
                    }
                }
            }
        }
    }

    let _ = blocking_handle.await;
    let _ = socket.send(Message::Close(None)).await;
    tracing::info!("Client session ended ({} steps)", steps_done);
}

fn run_party0_blocking(
    model: &klearu_llm::Model,
    embedding_q16: &[u32],
    prg_key: &[u8; 16],
    num_prompt: usize,
    max_new: usize,
    party: u8,
    triple_seed: u64,
    mut tcp_transport: TcpTransport,
    share_rx: std::sync::mpsc::Receiver<Option<Vec<u8>>>,
    result_tx: tokio::sync::mpsc::Sender<Result<Vec<u8>, String>>,
) {
    let config = &model.config;
    let vocab_size = config.vocab_size;
    let hidden_size = config.hidden_size;
    let depth = dpf_depth_for_vocab(vocab_size);

    let prg = AesPrg::new(prg_key);

    let mut kv_caches = KvCacheStore::new(
        config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
    );
    let mut deltanet_states = create_deltanet_states(model);

    let rt = tokio::runtime::Handle::current();

    // === PREFILL (batched) ===
    // Collect all DPF keys first, then batch PIR + single TCP reveal + forward
    let mut prefill_keys = Vec::with_capacity(num_prompt);
    for pos in 0..num_prompt {
        let key_data = match share_rx.recv() {
            Ok(Some(data)) => data,
            _ => {
                tracing::info!("Client disconnected during prefill at pos {}", pos);
                return;
            }
        };
        match deserialize_dpf_key(&key_data, depth) {
            Ok(k) => prefill_keys.push(k),
            Err(e) => {
                let _ = rt.block_on(result_tx.send(Err(format!("Bad DPF key at prefill {}: {}", pos, e))));
                return;
            }
        }
    }

    // Pre-allocate buffers (used for both prefill and decode)
    let mut decode_buffers = two_party::DecodeBuffers::new(model);

    if !prefill_keys.is_empty() {
        // Batch PIR: single pass over embedding table for all tokens
        let shares_q16 = pir_compute_embedding_shares_batch(
            &prg, &prefill_keys, embedding_q16, vocab_size, hidden_size,
        );

        // Batched reveal + prefill: Q16.16 direct path, 1 TCP roundtrip
        match two_party::server_forward_shared_reveal_prefill_q16(
            party, model, &shares_q16, &mut kv_caches, &mut deltanet_states,
            triple_seed, &mut tcp_transport, &mut decode_buffers,
        ) {
            Ok(()) => {
                let payload = decode_buffers.logit_share_bytes(vocab_size).to_vec();
                if rt.block_on(result_tx.send(Ok(payload))).is_err() {
                    return;
                }
            }
            Err(e) => {
                let _ = rt.block_on(result_tx.send(Err(format!("Prefill error: {}", e))));
                return;
            }
        }
    }

    for step in 0..max_new {
        let key_data = match share_rx.recv() {
            Ok(Some(data)) => data,
            Ok(None) => {
                tracing::info!("Client sent EOS");
                return;
            }
            Err(_) => {
                tracing::info!("Client disconnected during decode");
                return;
            }
        };

        let position = num_prompt + step;
        let dpf_key = match deserialize_dpf_key(&key_data, depth) {
            Ok(k) => k,
            Err(e) => {
                let _ = rt.block_on(result_tx.send(Err(format!("Bad DPF key at decode {}: {}", step, e))));
                return;
            }
        };
        let share_q16 = pir_compute_embedding_share(&prg, &dpf_key, embedding_q16, vocab_size, hidden_size);

        match two_party::server_forward_shared_reveal_q16(
            party, model, &share_q16, position, &mut kv_caches, &mut deltanet_states,
            triple_seed, &mut tcp_transport, &mut decode_buffers,
        ) {
            Ok(()) => {
                let payload = decode_buffers.logit_share_bytes(vocab_size).to_vec();
                if rt.block_on(result_tx.send(Ok(payload))).is_err() {
                    return;
                }
            }
            Err(e) => {
                let _ = rt.block_on(result_tx.send(Err(format!("Forward error at decode {}: {}", step, e))));
                return;
            }
        }
    }

    tracing::info!("Party {} session complete", party);
}

/// Serve model metadata files (tokenizer.json, config.json)
async fn model_file(
    State(state): State<Arc<AppState>>,
    Path(filename): Path<String>,
) -> impl IntoResponse {
    let allowed = ["tokenizer.json", "tokenizer_config.json", "config.json"];
    if !allowed.contains(&filename.as_str()) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let path = state.model_dir.join(&filename);
    match tokio::fs::read(&path).await {
        Ok(data) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            data,
        )
            .into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn send_error(socket: &mut WebSocket, msg: &str) -> Result<(), axum::Error> {
    let mut data = vec![TAG_ERROR];
    data.extend_from_slice(msg.as_bytes());
    socket.send(Message::Binary(data.into())).await.map_err(axum::Error::new)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("server_a=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    tracing::info!("Loading model from {:?}", args.model_dir);
    let mut pipeline = Pipeline::from_dir(&args.model_dir).expect("Failed to load model");
    let eos_token_ids = detect_eos_tokens(&args.model_dir);
    tracing::info!("EOS token IDs: {:?}", eos_token_ids);

    tracing::info!(
        "Model loaded: vocab={}, hidden={}, layers={}, heads={}",
        pipeline.model.config.vocab_size,
        pipeline.model.config.hidden_size,
        pipeline.model.config.num_layers,
        pipeline.model.config.num_heads,
    );

    // Sync Q32.32 pre-quantized weights
    two_party::sync_all_q32(&mut pipeline.model);

    // Pre-compute Q16.16 embedding table for DPF-PIR
    tracing::info!("Pre-computing Q16.16 embedding table for DPF-PIR...");
    let embedding_q16 = two_party::prepare_embedding_table(&pipeline.model);
    tracing::info!("Embedding table ready: {} entries", embedding_q16.len());

    // Generate random PRG key for DPF
    let prg_key: [u8; 16] = rand::thread_rng().gen();

    let state = Arc::new(AppState {
        model: pipeline.model,
        eos_token_ids,
        server_b_addr: args.server_b_addr.clone(),
        server_b_ws: args.server_b_ws.clone(),
        max_tokens: args.max_tokens,
        model_dir: args.model_dir,
        embedding_q16,
        prg_key,
    });

    let mut app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(|| async { "ok" }))
        .route("/api/models/{filename}", get(model_file))
        .layer(CorsLayer::permissive())
        .with_state(state);

    if let Some(ref static_dir) = args.static_dir {
        tracing::info!("Serving static files from {:?}", static_dir);
        app = app.fallback_service(ServeDir::new(static_dir));
    }

    let addr = SocketAddr::from(([0, 0, 0, 0], args.ws_port));
    tracing::info!("Server B target: {} (ws: {})", args.server_b_addr, args.server_b_ws);

    match (args.tls_cert, args.tls_key) {
        (Some(cert), Some(key)) => {
            let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert, &key)
                .await
                .expect("Failed to load TLS cert/key");
            tracing::info!("Listening on https://{} (TLS)", addr);
            axum_server::bind_rustls(addr, tls_config)
                .serve(app.into_make_service())
                .await
                .unwrap();
        }
        (None, None) => {
            tracing::warn!("No TLS configured — serving plain HTTP/WS (use --tls-cert and --tls-key for HTTPS/WSS)");
            tracing::info!("Listening on http://{}", addr);
            let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
            axum::serve(listener, app).await.unwrap();
        }
        _ => {
            panic!("Both --tls-cert and --tls-key must be provided together");
        }
    }
}
