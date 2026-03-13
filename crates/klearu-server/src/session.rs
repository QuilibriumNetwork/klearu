use axum::extract::ws::{Message, WebSocket};
use klearu_llm::Model;
use klearu_llm::config::LlmConfig;
use klearu_llm::model::block::AttentionLayer;
use klearu_mpc::beaver::DummyTripleGen128;
use klearu_mpc::transport::Transport;
use klearu_mpc::SharedVec64;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::AppState;

// --- Protocol message tags ---
const TAG_MPC_DATA: u8 = 0x50;
const TAG_LOGIT_SHARE: u8 = 0x51;
const TAG_NEXT_TOKEN: u8 = 0x52;
const TAG_EOS_SIGNAL: u8 = 0x42;
const TAG_ERROR: u8 = 0xF0;

pub struct SessionConfig {
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub num_prompt_tokens: u32,
}

impl SessionConfig {
    /// Parse from bytes after the tag byte.
    /// Layout: security(1) + maxNew(4) + temp(4) + topK(4) + topP(4) + numPromptTokens(4) = 21 bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 21 {
            return Err("Session init too short".into());
        }
        // data[0] = security byte (ignored, always High)
        let max_new_tokens = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
        let temperature = f32::from_le_bytes([data[5], data[6], data[7], data[8]]);
        let top_k = u32::from_le_bytes([data[9], data[10], data[11], data[12]]);
        let top_p = f32::from_le_bytes([data[13], data[14], data[15], data[16]]);
        let num_prompt_tokens = u32::from_le_bytes([data[17], data[18], data[19], data[20]]);

        Ok(Self {
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            num_prompt_tokens,
        })
    }
}

/// Transport implementation that bridges MPC protocol over WebSocket channels.
///
/// Party 1's Transport.send() → TAG_MPC_DATA over WS → client's WASM worker recv
/// Client's WASM worker send → TAG_MPC_DATA over WS → Party 1's Transport.recv()
struct WsTransport {
    /// Send MPC data to the WS bridge (which forwards to client)
    ws_tx: std::sync::mpsc::Sender<Vec<u8>>,
    /// Receive MPC data from the WS bridge (forwarded from client)
    mpc_rx: std::sync::mpsc::Receiver<Vec<u8>>,
    /// Buffered partial reads
    recv_buf: Vec<u8>,
}

impl Transport for WsTransport {
    fn send(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.ws_tx.send(data.to_vec()).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, e.to_string())
        })
    }

    fn recv(&mut self, len: usize) -> std::io::Result<Vec<u8>> {
        while self.recv_buf.len() < len {
            let data = self.mpc_rx.recv().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::UnexpectedEof, e.to_string())
            })?;
            self.recv_buf.extend_from_slice(&data);
        }
        Ok(self.recv_buf.drain(..len).collect())
    }
}

/// Messages from session thread → WebSocket sender
enum SessionToWs {
    /// MPC transport data: prepend TAG_MPC_DATA before sending
    MpcData(Vec<u8>),
    /// Party 1's logit share: prepend TAG_LOGIT_SHARE before sending
    LogitShare(Vec<u8>),
    Error(String),
    Done,
}

/// Messages from WebSocket → session thread (non-MPC)
enum WsToSession {
    /// Client's embedding share for next token
    NextToken(Vec<u8>),
    /// Client signals EOS
    Eos,
    /// Client disconnected or error
    Disconnected,
}

/// Run the private 2PC session. Server is party 1 only.
///
/// The client runs party 0 in a WASM Web Worker. MPC transport data flows
/// bidirectionally over WebSocket with TAG_MPC_DATA (0x50). The server sends
/// its logit share with TAG_LOGIT_SHARE (0x51) after each forward pass.
/// The client sends next token embedding shares with TAG_NEXT_TOKEN (0x52).
pub async fn run_session(
    socket: &mut WebSocket,
    state: &Arc<AppState>,
    config: &SessionConfig,
    triple_seed: u64,
) {
    tracing::info!(
        "Session: {} prompt tokens, max {} new tokens",
        config.num_prompt_tokens,
        config.max_new_tokens,
    );

    // Channels for the WS bridge
    let (session_tx, mut session_rx) = mpsc::channel::<SessionToWs>(32);
    let (ws_tx, ws_rx) = mpsc::channel::<WsToSession>(4);
    // Channel for MPC transport data from WS to party 1 thread
    let (mpc_to_party_tx, mpc_to_party_rx) = std::sync::mpsc::channel::<Vec<u8>>();

    let state_clone = Arc::clone(state);
    let num_prompt_tokens = config.num_prompt_tokens as usize;
    let max_new_tokens = config.max_new_tokens as usize;

    let session_handle = tokio::task::spawn_blocking(move || {
        run_party1_blocking(
            state_clone,
            num_prompt_tokens,
            max_new_tokens,
            triple_seed,
            session_tx,
            ws_rx,
            mpc_to_party_rx,
        );
    });

    // WS bridge: demultiplex incoming by tag, forward outgoing
    loop {
        tokio::select! {
            // Session → WS: send MPC data, logit shares, or errors to client
            msg = session_rx.recv() => {
                match msg {
                    Some(SessionToWs::MpcData(data)) => {
                        let mut payload = Vec::with_capacity(1 + data.len());
                        payload.push(TAG_MPC_DATA);
                        payload.extend_from_slice(&data);
                        if socket.send(Message::Binary(payload.into())).await.is_err() {
                            let _ = ws_tx.send(WsToSession::Disconnected).await;
                            break;
                        }
                    }
                    Some(SessionToWs::LogitShare(data)) => {
                        let mut payload = Vec::with_capacity(1 + data.len());
                        payload.push(TAG_LOGIT_SHARE);
                        payload.extend_from_slice(&data);
                        if socket.send(Message::Binary(payload.into())).await.is_err() {
                            let _ = ws_tx.send(WsToSession::Disconnected).await;
                            break;
                        }
                    }
                    Some(SessionToWs::Error(msg)) => {
                        let mut payload = vec![TAG_ERROR];
                        payload.extend_from_slice(msg.as_bytes());
                        let _ = socket.send(Message::Binary(payload.into())).await;
                        break;
                    }
                    Some(SessionToWs::Done) | None => break,
                }
            }
            // WS → Session: demultiplex by tag
            ws_msg = socket.recv() => {
                match ws_msg {
                    Some(Ok(Message::Binary(data))) => {
                        if data.is_empty() {
                            let _ = ws_tx.send(WsToSession::Disconnected).await;
                            break;
                        }
                        match data[0] {
                            TAG_MPC_DATA => {
                                // Forward MPC transport data to party 1 thread
                                if mpc_to_party_tx.send(data[1..].to_vec()).is_err() {
                                    break;
                                }
                            }
                            TAG_NEXT_TOKEN => {
                                let _ = ws_tx.send(
                                    WsToSession::NextToken(data[1..].to_vec())
                                ).await;
                            }
                            TAG_EOS_SIGNAL => {
                                let _ = ws_tx.send(WsToSession::Eos).await;
                            }
                            other => {
                                tracing::warn!("Unexpected WS tag: 0x{:02x}", other);
                                let _ = ws_tx.send(WsToSession::Disconnected).await;
                                break;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        let _ = ws_tx.send(WsToSession::Disconnected).await;
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    let _ = session_handle.await;
}

/// Run party 1's MPC forward pass in a blocking thread.
fn run_party1_blocking(
    state: Arc<AppState>,
    num_prompt_tokens: usize,
    max_new_tokens: usize,
    triple_seed: u64,
    session_tx: mpsc::Sender<SessionToWs>,
    ws_rx: mpsc::Receiver<WsToSession>,
    mpc_rx: std::sync::mpsc::Receiver<Vec<u8>>,
) {
    let rt = tokio::runtime::Handle::current();
    let pipeline = rt.block_on(state.pipeline.lock());

    let model_config = pipeline.model.config.clone();
    let hidden_size = model_config.hidden_size;

    let mut model = clone_model_from(&pipeline.model, &model_config);
    drop(pipeline);

    model.reset_kv_caches();

    // Create party 1's triple generator with shared seed
    let mut triples = DummyTripleGen128::new(1, triple_seed);

    // Create WsTransport for party 1
    // MPC sends go through session_tx → WS bridge → client
    let (mpc_send_tx, mpc_send_rx) = std::sync::mpsc::channel::<Vec<u8>>();

    // Spawn a relay thread: mpc_send_rx → session_tx as MpcData
    let session_tx_relay = session_tx.clone();
    let rt_relay = rt.clone();
    let relay_handle = std::thread::spawn(move || {
        while let Ok(data) = mpc_send_rx.recv() {
            if rt_relay.block_on(session_tx_relay.send(SessionToWs::MpcData(data))).is_err() {
                break;
            }
        }
    });

    let mut transport = WsTransport {
        ws_tx: mpc_send_tx,
        mpc_rx,
        recv_buf: Vec::new(),
    };

    let mut ws_rx = ws_rx;

    // === PREFILL ===
    for token_idx in 0..num_prompt_tokens {
        // Receive party 1's embedding share from client
        let msg = match rt.block_on(ws_rx.recv()) {
            Some(WsToSession::NextToken(data)) => data,
            Some(WsToSession::Disconnected) | None => {
                let _ = rt.block_on(session_tx.send(SessionToWs::Done));
                drop(transport);
                let _ = relay_handle.join();
                return;
            }
            Some(other) => {
                tracing::warn!("Expected NextToken during prefill, got {:?}",
                    match &other { WsToSession::Eos => "Eos", _ => "other" });
                let _ = rt.block_on(session_tx.send(SessionToWs::Error(
                    "Expected NextToken during prefill".into()
                )));
                let _ = rt.block_on(session_tx.send(SessionToWs::Done));
                drop(transport);
                let _ = relay_handle.join();
                return;
            }
        };

        // Parse party 1's embedding share (hidden_size × 8 bytes u64 LE)
        let expected_len = hidden_size * 8;
        if msg.len() != expected_len {
            let _ = rt.block_on(session_tx.send(SessionToWs::Error(format!(
                "Bad embedding share length: {} (expected {})", msg.len(), expected_len
            ))));
            let _ = rt.block_on(session_tx.send(SessionToWs::Done));
            break;
        }

        let share = parse_share(&msg, hidden_size);
        let position = token_idx;

        // Run party 1's forward pass — MPC transport exchanges happen via WsTransport
        let result = klearu_private::private_model::private_model_forward_secure_no_reveal(
            1, &mut model, &share, position, &mut triples, &mut transport,
        );

        match result {
            Ok(logit_share) => {
                // Only send logit share for last prefill token
                if token_idx == num_prompt_tokens - 1 {
                    let payload = encode_share(&logit_share.0);
                    if rt.block_on(session_tx.send(SessionToWs::LogitShare(payload))).is_err() {
                        break;
                    }
                }
            }
            Err(e) => {
                tracing::error!("Party 1 error at prefill step {}: {}", token_idx, e);
                let _ = rt.block_on(session_tx.send(SessionToWs::Error(format!(
                    "MPC error: {}", e
                ))));
                break;
            }
        }
    }

    // === DECODE LOOP ===
    for step in 0..max_new_tokens {
        // Wait for next embedding share or EOS
        let msg = match rt.block_on(ws_rx.recv()) {
            Some(WsToSession::NextToken(data)) => data,
            Some(WsToSession::Eos) => {
                tracing::info!("Client signaled EOS");
                break;
            }
            Some(WsToSession::Disconnected) | None => {
                tracing::info!("Client disconnected during decode");
                break;
            }
        };

        let expected_len = hidden_size * 8;
        if msg.len() != expected_len {
            let _ = rt.block_on(session_tx.send(SessionToWs::Error(format!(
                "Bad embedding share length: {} (expected {})", msg.len(), expected_len
            ))));
            break;
        }

        let share = parse_share(&msg, hidden_size);
        let position = num_prompt_tokens + step;

        let result = klearu_private::private_model::private_model_forward_secure_no_reveal(
            1, &mut model, &share, position, &mut triples, &mut transport,
        );

        match result {
            Ok(logit_share) => {
                let payload = encode_share(&logit_share.0);
                if rt.block_on(session_tx.send(SessionToWs::LogitShare(payload))).is_err() {
                    break;
                }
            }
            Err(e) => {
                tracing::error!("Party 1 error at decode step {}: {}", step, e);
                let _ = rt.block_on(session_tx.send(SessionToWs::Error(format!(
                    "MPC error: {}", e
                ))));
                break;
            }
        }
    }

    drop(transport);
    let _ = relay_handle.join();
    let _ = rt.block_on(session_tx.send(SessionToWs::Done));
    tracing::info!("Session completed");
}

/// Parse a single embedding share from u64 LE bytes.
fn parse_share(data: &[u8], hidden_size: usize) -> SharedVec64 {
    let mut share = Vec::with_capacity(hidden_size);
    for i in 0..hidden_size {
        let offset = i * 8;
        share.push(u64::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
        ]));
    }
    SharedVec64(share)
}

/// Encode a share vector as u64 LE bytes.
fn encode_share(share: &[u64]) -> Vec<u8> {
    let mut data = Vec::with_capacity(share.len() * 8);
    for &v in share {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data
}

/// Clone model weights from an existing model into a fresh Model instance.
fn clone_model_from(source: &Model, config: &LlmConfig) -> Model {
    let mut model = Model::new(config.clone());

    // Copy embedding weights
    for i in 0..config.vocab_size {
        let src = source.embedding.weights.get_weights(i);
        let dst = model.embedding.weights.get_weights_mut(i);
        dst[..config.hidden_size].copy_from_slice(&src[..config.hidden_size]);
    }

    // Copy layer weights
    for (layer_idx, (dst_layer, src_layer)) in model.layers.iter_mut()
        .zip(source.layers.iter())
        .enumerate()
    {
        // Norms
        dst_layer.attn_norm.weight.copy_from_slice(&src_layer.attn_norm.weight);
        dst_layer.mlp_norm.weight.copy_from_slice(&src_layer.mlp_norm.weight);

        // Attention projections
        match (&mut dst_layer.attention, &src_layer.attention) {
            (AttentionLayer::Standard(dst_attn), AttentionLayer::Standard(src_attn)) => {
                copy_linear(&mut dst_attn.q_proj, &src_attn.q_proj);
                copy_linear(&mut dst_attn.k_proj, &src_attn.k_proj);
                copy_linear(&mut dst_attn.v_proj, &src_attn.v_proj);
                copy_linear(&mut dst_attn.o_proj, &src_attn.o_proj);
            }
            (AttentionLayer::GatedDeltaNet(dst_dn), AttentionLayer::GatedDeltaNet(src_dn)) => {
                copy_linear(&mut dst_dn.in_proj_qkv, &src_dn.in_proj_qkv);
                copy_linear(&mut dst_dn.in_proj_z, &src_dn.in_proj_z);
                copy_linear(&mut dst_dn.in_proj_a, &src_dn.in_proj_a);
                copy_linear(&mut dst_dn.in_proj_b, &src_dn.in_proj_b);
                copy_linear(&mut dst_dn.out_proj, &src_dn.out_proj);
                dst_dn.conv_weight.copy_from_slice(&src_dn.conv_weight);
                dst_dn.dt_bias.copy_from_slice(&src_dn.dt_bias);
                dst_dn.a_log.copy_from_slice(&src_dn.a_log);
                dst_dn.norm_weight.copy_from_slice(&src_dn.norm_weight);
            }
            _ => {
                tracing::warn!("Layer {} attention type mismatch during clone", layer_idx);
            }
        }

        // MLP projections
        copy_linear(&mut dst_layer.mlp.gate_proj, &src_layer.mlp.gate_proj);
        copy_linear(&mut dst_layer.mlp.up_proj, &src_layer.mlp.up_proj);
        copy_linear(&mut dst_layer.mlp.down_proj, &src_layer.mlp.down_proj);
    }

    // Final norm
    model.final_norm.weight.copy_from_slice(&source.final_norm.weight);

    // LM head (if separate from embedding)
    if let (Some(dst_head), Some(src_head)) = (&mut model.lm_head, &source.lm_head) {
        copy_linear(dst_head, src_head);
    }

    // Sync Q32.32 pre-quantized weights for the secure protocol
    sync_all_q32(&mut model);

    model
}

fn copy_linear(dst: &mut klearu_llm::model::linear::Linear, src: &klearu_llm::model::linear::Linear) {
    for i in 0..src.out_features() {
        let src_row = src.weights.get_weights(i);
        let dst_row = dst.weights.get_weights_mut(i);
        let len = src_row.len().min(dst_row.len());
        dst_row[..len].copy_from_slice(&src_row[..len]);
    }
}

fn sync_all_q32(model: &mut Model) {
    if let Some(ref mut head) = model.lm_head {
        head.sync_q32();
    }
    for layer in &mut model.layers {
        match &mut layer.attention {
            AttentionLayer::Standard(attn) => {
                attn.q_proj.sync_q32();
                attn.k_proj.sync_q32();
                attn.v_proj.sync_q32();
                attn.o_proj.sync_q32();
            }
            AttentionLayer::GatedDeltaNet(dn) => {
                dn.in_proj_qkv.sync_q32();
                dn.in_proj_z.sync_q32();
                dn.in_proj_a.sync_q32();
                dn.in_proj_b.sync_q32();
                dn.out_proj.sync_q32();
            }
        }
        layer.mlp.gate_proj.sync_q32();
        layer.mlp.up_proj.sync_q32();
        layer.mlp.down_proj.sync_q32();
    }
}
