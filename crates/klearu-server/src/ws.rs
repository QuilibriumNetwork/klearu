use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use std::sync::Arc;

use crate::session;
use crate::AppState;

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    tracing::info!("WebSocket client connected");

    // Wait for session init message
    let init_msg = match socket.recv().await {
        Some(Ok(Message::Binary(data))) => data,
        Some(Ok(Message::Close(_))) | None => {
            tracing::info!("Client disconnected before init");
            return;
        }
        other => {
            tracing::warn!("Unexpected init message: {:?}", other.map(|r| r.map(|_| "...")));
            return;
        }
    };

    if init_msg.is_empty() || init_msg[0] != 0x01 {
        tracing::warn!("Expected SessionInit (0x01), got {:?}", init_msg.first());
        let _ = send_error(&mut socket, "Expected SessionInit message").await;
        return;
    }

    // Parse session init (security byte is ignored — always High)
    let session_config = match session::SessionConfig::from_bytes(&init_msg[1..]) {
        Ok(c) => c,
        Err(e) => {
            let _ = send_error(&mut socket, &format!("Invalid session init: {}", e)).await;
            return;
        }
    };

    tracing::info!(
        "Session init: max_new={}, temp={}, top_k={}, top_p={}, prompt_tokens={}",
        session_config.max_new_tokens,
        session_config.temperature,
        session_config.top_k,
        session_config.top_p,
        session_config.num_prompt_tokens,
    );

    // Generate triple seed for deterministic DummyTripleGen on both parties
    let triple_seed: u64 = rand::random();

    // Send session ready with model config + triple seed
    {
        let pipeline = state.pipeline.lock().await;
        let config = &pipeline.model.config;
        let config_json = serde_json::to_vec(&serde_json::json!({
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
            "eos_token_id": state.eos_token_id,
            "triple_seed": format!("{:016x}", triple_seed),
            // Include full config for WASM model construction
            "rope_theta": config.rope_theta,
            "attn_output_gate": config.attn_output_gate,
            "layer_types": config.layer_types,
            "linear_num_key_heads": config.linear_num_key_heads,
            "linear_num_value_heads": config.linear_num_value_heads,
            "linear_key_head_dim": config.linear_key_head_dim,
            "linear_value_head_dim": config.linear_value_head_dim,
            "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
        }))
        .unwrap();
        let mut response = vec![0x02]; // SessionReady
        response.extend_from_slice(&config_json);
        if socket.send(Message::Binary(response.into())).await.is_err() {
            return;
        }
    }

    // Run the session (server is party 1 only, party 0 runs in client's WASM worker)
    session::run_session(&mut socket, &state, &session_config, triple_seed).await;

    tracing::info!("WebSocket session ended");
}

async fn send_error(socket: &mut WebSocket, msg: &str) -> Result<(), axum::Error> {
    let mut data = vec![0xF0];
    data.extend_from_slice(msg.as_bytes());
    socket
        .send(Message::Binary(data.into()))
        .await
        .map_err(|e| axum::Error::new(e))
}
