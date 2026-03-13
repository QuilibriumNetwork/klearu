//! Server B: Model Owner + Party 1 (Hybrid TCP + WebSocket)
//!
//! - Listens for TCP connections from Server A (MPC transport)
//! - Listens for WebSocket connections from clients (DPF keys)
//! - Sessions matched by session_id
//! - Runs Party 1 of the no-reveal MPC protocol
//! - Returns Q16.16 logit shares to client via WebSocket
//!
//! Usage:
//!   server-b --model-dir ./SmolLM-135M --tcp-port 9090 --ws-port 9091

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::{Router, routing::get};
use clap::Parser;
use klearu_dpf::AesPrg;
use klearu_llm::generate::pipeline::Pipeline;
use klearu_llm::model::kv_cache::KvCacheStore;
use klearu_mpc::embedding_pir::{
    deserialize_dpf_key, dpf_depth_for_vocab, pir_compute_embedding_share,
    pir_compute_embedding_shares_batch,
};
use klearu_private::private_model::create_deltanet_states;
use klearu_private::tcp_transport::TcpTransport;
use klearu_mpc::transport::Transport;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

#[path = "../two_party.rs"]
mod two_party;

// Protocol tags
const TAG_JOIN_SESSION: u8 = 0x03;
const TAG_JOIN_ACK: u8 = 0x04;
const TAG_NEXT_TOKEN: u8 = 0x52;
const TAG_LOGIT_SHARE: u8 = 0x51;
const TAG_EOS_SIGNAL: u8 = 0x42;
const TAG_ERROR: u8 = 0xF0;

#[derive(Parser)]
#[command(name = "server-b", about = "Server B: model partner (Party 1) for 2-server MPC")]
struct Args {
    /// Path to the model directory (config.json, *.safetensors)
    #[arg(long)]
    model_dir: PathBuf,

    /// TCP port to listen on for Server A connections
    #[arg(long, default_value = "9090")]
    tcp_port: u16,

    /// WebSocket port to listen on for client connections
    #[arg(long, default_value = "9091")]
    ws_port: u16,
}

/// A pending session created by TCP init from Server A, awaiting client WS connection.
struct PendingSession {
    triple_seed: u64,
    prg_key: [u8; 16],
    num_prompt: usize,
    max_new: usize,
    tcp_transport: TcpTransport,
}

struct ServerBState {
    model: klearu_llm::Model,
    embedding_q16: Vec<u32>,
    pending_sessions: Mutex<HashMap<u64, PendingSession>>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerBState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_client_ws(socket, state))
}

async fn handle_client_ws(mut socket: WebSocket, state: Arc<ServerBState>) {
    tracing::info!("Client connected via WebSocket");

    // Wait for JoinSession message
    let join_msg = match socket.recv().await {
        Some(Ok(Message::Binary(data))) => data.to_vec(),
        Some(Ok(Message::Close(_))) | None => {
            tracing::info!("Client disconnected before JoinSession");
            return;
        }
        other => {
            tracing::warn!("Expected binary JoinSession, got: {:?}", other.map(|r| r.map(|_| "...")));
            return;
        }
    };

    if join_msg.len() < 9 || join_msg[0] != TAG_JOIN_SESSION {
        let _ = send_error(&mut socket, "Expected JoinSession (0x03) with session_id").await;
        return;
    }

    let session_id = u64::from_le_bytes([
        join_msg[1], join_msg[2], join_msg[3], join_msg[4],
        join_msg[5], join_msg[6], join_msg[7], join_msg[8],
    ]);

    tracing::info!("Client JoinSession: {:#018x}", session_id);

    // Look up pending session
    let pending = {
        let mut sessions = state.pending_sessions.lock().await;
        sessions.remove(&session_id)
    };

    let pending = match pending {
        Some(p) => p,
        None => {
            tracing::warn!("Unknown session_id: {:#018x}", session_id);
            let _ = send_error(&mut socket, "Unknown session_id").await;
            return;
        }
    };

    // Send JoinAck
    if socket.send(Message::Binary(vec![TAG_JOIN_ACK].into())).await.is_err() {
        return;
    }

    tracing::info!(
        "Session matched: {:#018x}, prompt={}, max_new={}",
        session_id, pending.num_prompt, pending.max_new,
    );

    let vocab_size = state.model.config.vocab_size;
    let depth = dpf_depth_for_vocab(vocab_size);
    let expected_key_len = 1 + 16 + 4 + (depth as usize) * 17;

    // Channels for WS ↔ blocking thread
    let (share_tx, share_rx) = std::sync::mpsc::channel::<Option<Vec<u8>>>();
    let (result_tx, mut result_rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, String>>(4);

    let state_clone = state.clone();
    let blocking_handle = tokio::task::spawn_blocking(move || {
        run_party1_blocking(
            &state_clone.model,
            &state_clone.embedding_q16,
            &pending.prg_key,
            pending.num_prompt,
            pending.max_new,
            pending.triple_seed,
            pending.tcp_transport,
            share_rx,
            result_tx,
        );
    });

    // Main loop: forward DPF keys from client WS to blocking thread, results back to WS
    loop {
        tokio::select! {
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
                    }
                    Some(Err(e)) => {
                        let _ = send_error(&mut socket, &e).await;
                        let _ = share_tx.send(None);
                        break;
                    }
                    None => break,
                }
            }
        }
    }

    let _ = blocking_handle.await;
    let _ = socket.send(Message::Close(None)).await;
    tracing::info!("Client session ended for {:#018x}", session_id);
}

fn run_party1_blocking(
    model: &klearu_llm::Model,
    embedding_q16: &[u32],
    prg_key: &[u8; 16],
    num_prompt: usize,
    max_new: usize,
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
            1, model, &shares_q16, &mut kv_caches, &mut deltanet_states,
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
            1, model, &share_q16, position, &mut kv_caches, &mut deltanet_states,
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

    tracing::info!("Party 1 session complete");
}

async fn send_error(socket: &mut WebSocket, msg: &str) -> Result<(), axum::Error> {
    let mut data = vec![TAG_ERROR];
    data.extend_from_slice(msg.as_bytes());
    socket.send(Message::Binary(data.into())).await.map_err(axum::Error::new)
}

/// TCP listener task: accepts connections from Server A, creates pending sessions.
async fn tcp_listener_task(
    tcp_port: u16,
    state: Arc<ServerBState>,
) {
    let addr = SocketAddr::from(([0, 0, 0, 0], tcp_port));
    let listener = std::net::TcpListener::bind(addr).expect("Failed to bind TCP");
    tracing::info!("Server B TCP listening on tcp://{}", addr);

    // Run in blocking thread since std::net::TcpListener is sync
    tokio::task::spawn_blocking(move || {
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let peer = stream.peer_addr().ok();
                    tracing::info!("Server A connected from {:?}", peer);

                    let mut transport = match TcpTransport::new(stream) {
                        Ok(t) => t,
                        Err(e) => {
                            tracing::error!("TcpTransport::new failed: {}", e);
                            continue;
                        }
                    };

                    // Receive session init: session_id(8) + triple_seed(8) + prg_key(16) + num_prompt(4) + max_new(4) = 40 bytes
                    let init_bytes = match transport.recv(40) {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::error!("Failed to receive session init: {}", e);
                            continue;
                        }
                    };

                    let session_id = u64::from_le_bytes([
                        init_bytes[0], init_bytes[1], init_bytes[2], init_bytes[3],
                        init_bytes[4], init_bytes[5], init_bytes[6], init_bytes[7],
                    ]);
                    let triple_seed = u64::from_le_bytes([
                        init_bytes[8], init_bytes[9], init_bytes[10], init_bytes[11],
                        init_bytes[12], init_bytes[13], init_bytes[14], init_bytes[15],
                    ]);
                    let mut prg_key = [0u8; 16];
                    prg_key.copy_from_slice(&init_bytes[16..32]);
                    let num_prompt = u32::from_le_bytes([
                        init_bytes[32], init_bytes[33], init_bytes[34], init_bytes[35],
                    ]) as usize;
                    let max_new = u32::from_le_bytes([
                        init_bytes[36], init_bytes[37], init_bytes[38], init_bytes[39],
                    ]) as usize;

                    tracing::info!(
                        "TCP session init: id={:#018x}, seed={:#018x}, prompt={}, max_new={}",
                        session_id, triple_seed, num_prompt, max_new,
                    );

                    let pending = PendingSession {
                        triple_seed,
                        prg_key,
                        num_prompt,
                        max_new,
                        tcp_transport: transport,
                    };

                    // Store pending session (blocking lock via tokio Handle)
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async {
                        let mut sessions = state.pending_sessions.lock().await;
                        sessions.insert(session_id, pending);
                    });

                    tracing::info!("Session {:#018x} pending, awaiting client WS", session_id);
                }
                Err(e) => {
                    tracing::error!("TCP accept error: {}", e);
                }
            }
        }
    }).await.unwrap();
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("server_b=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    tracing::info!("Loading model from {:?}", args.model_dir);
    let mut pipeline = Pipeline::from_dir(&args.model_dir).expect("Failed to load model");

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

    let state = Arc::new(ServerBState {
        model: pipeline.model,
        embedding_q16,
        pending_sessions: Mutex::new(HashMap::new()),
    });

    // Start TCP listener in background
    let state_tcp = Arc::clone(&state);
    let tcp_port = args.tcp_port;
    tokio::spawn(async move {
        tcp_listener_task(tcp_port, state_tcp).await;
    });

    // Start WS server
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(|| async { "ok" }))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], args.ws_port));
    tracing::info!("Server B WS listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
