//! WASM entry point for MPC party 0 running in a Web Worker.
//!
//! Party0State holds the model, triple generator, and transport.
//! The main thread controls it via postMessage:
//!   1. Create with config JSON + triple seed + SharedArrayBuffer
//!   2. Load weights from binary blob
//!   3. Run forward_step per token (blocking on MPC transport)

use js_sys::SharedArrayBuffer;
use klearu_llm::config::LlmConfig;
use klearu_llm::Model;
use klearu_llm::model::block::AttentionLayer;
use klearu_llm::model::linear::Linear;
use klearu_mpc::beaver::DummyTripleGen128;
use klearu_mpc::SharedVec64;
use wasm_bindgen::prelude::*;

use crate::wasm_transport::WasmTransport;

/// Party 0 state for MPC inference in a Web Worker.
#[wasm_bindgen]
pub struct Party0State {
    model: Model,
    triples: DummyTripleGen128,
    transport: WasmTransport,
    position: usize,
}

#[wasm_bindgen]
impl Party0State {
    /// Create a new Party0State.
    ///
    /// - `config_json`: JSON string of the model config (LlmConfig fields)
    /// - `triple_seed`: seed for DummyTripleGen128 (party 0)
    /// - `sab`: SharedArrayBuffer for MPC transport recv channel
    /// - `send_callback`: JS function for MPC transport send
    #[wasm_bindgen(constructor)]
    pub fn new(
        config_json: &str,
        triple_seed: f64,
        sab: SharedArrayBuffer,
        send_callback: js_sys::Function,
    ) -> Result<Party0State, JsValue> {
        let config: LlmConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Config parse error: {}", e)))?;

        // Use no-embedding model: minimal embedding (1 row), separate lm_head.
        // Party 0 never uses the embedding table (receives shares as input).
        let model = Model::new_no_embedding(config);
        let triples = DummyTripleGen128::new(0, triple_seed as u64);
        let transport = WasmTransport::new(sab, send_callback);

        Ok(Party0State {
            model,
            triples,
            transport,
            position: 0,
        })
    }

    /// Load model weights from a binary blob (excluding embeddings).
    ///
    /// Embeddings are excluded — they are served separately and unused
    /// during forward pass (takes embedding shares as input).
    ///
    /// Binary format (all f32 LE, no stride padding):
    ///   1. Per layer (num_layers times):
    ///      a. attn_norm weights: hidden_size f32s
    ///      b. mlp_norm weights: hidden_size f32s
    ///      c. Attention projections (Standard or GatedDeltaNet)
    ///      d. MLP: gate_proj, up_proj, down_proj weights
    ///   2. final_norm weights: hidden_size f32s
    ///   3. lm_head weights (if !tie_word_embeddings): vocab_size * hidden_size f32s
    pub fn load_weights(&mut self, weights_bytes: &[u8]) -> Result<(), JsValue> {
        let config = &self.model.config;
        let hidden_size = config.hidden_size;

        let mut cursor = 0usize;

        // 1. Per-layer weights
        for layer_idx in 0..config.num_layers {
            let layer = &mut self.model.layers[layer_idx];

            // attn_norm
            let norm_w = read_f32s_raw(&mut cursor, hidden_size, weights_bytes)?;
            layer.attn_norm.weight.copy_from_slice(&norm_w);

            // mlp_norm
            let norm_w = read_f32s_raw(&mut cursor, hidden_size, weights_bytes)?;
            layer.mlp_norm.weight.copy_from_slice(&norm_w);

            // Attention projections
            match &mut layer.attention {
                AttentionLayer::Standard(attn) => {
                    load_linear(&mut attn.q_proj, &mut cursor, weights_bytes)?;
                    load_linear(&mut attn.k_proj, &mut cursor, weights_bytes)?;
                    load_linear(&mut attn.v_proj, &mut cursor, weights_bytes)?;
                    load_linear(&mut attn.o_proj, &mut cursor, weights_bytes)?;
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    load_linear(&mut dn.in_proj_qkv, &mut cursor, weights_bytes)?;
                    load_linear(&mut dn.in_proj_z, &mut cursor, weights_bytes)?;
                    load_linear(&mut dn.in_proj_a, &mut cursor, weights_bytes)?;
                    load_linear(&mut dn.in_proj_b, &mut cursor, weights_bytes)?;
                    load_linear(&mut dn.out_proj, &mut cursor, weights_bytes)?;

                    // conv_weight, dt_bias, a_log, norm_weight
                    let n = dn.conv_weight.len();
                    let v = read_f32s_raw(&mut cursor, n, weights_bytes)?;
                    dn.conv_weight.copy_from_slice(&v);

                    let n = dn.dt_bias.len();
                    let v = read_f32s_raw(&mut cursor, n, weights_bytes)?;
                    dn.dt_bias.copy_from_slice(&v);

                    let n = dn.a_log.len();
                    let v = read_f32s_raw(&mut cursor, n, weights_bytes)?;
                    dn.a_log.copy_from_slice(&v);

                    let n = dn.norm_weight.len();
                    let v = read_f32s_raw(&mut cursor, n, weights_bytes)?;
                    dn.norm_weight.copy_from_slice(&v);
                }
            }

            // MLP projections
            load_linear(&mut layer.mlp.gate_proj, &mut cursor, weights_bytes)?;
            load_linear(&mut layer.mlp.up_proj, &mut cursor, weights_bytes)?;
            load_linear(&mut layer.mlp.down_proj, &mut cursor, weights_bytes)?;
        }

        // 2. final_norm
        let norm_w = read_f32s_raw(&mut cursor, hidden_size, weights_bytes)?;
        self.model.final_norm.weight.copy_from_slice(&norm_w);

        // 3. lm_head (only present in weights if original model has !tie_word_embeddings)
        // For tie_word_embeddings models, lm_head is loaded separately via
        // load_lm_head_from_embedding().
        if !self.model.config.tie_word_embeddings {
            if let Some(ref mut head) = self.model.lm_head {
                load_linear(head, &mut cursor, weights_bytes)?;
            }
        }

        // Sync Q32.32 pre-quantized weights (no-op on WASM — computed on the fly)
        sync_all_q32(&mut self.model);

        // Reset KV caches
        self.model.reset_kv_caches();

        Ok(())
    }

    /// Load lm_head weights from the embedding table (for tie_word_embeddings models).
    ///
    /// The embedding table is downloaded separately by the main thread.
    /// For tie_word_embeddings models, the embedding IS the lm_head — we load
    /// it into our explicit lm_head Linear.
    ///
    /// `embedding_bytes`: contiguous f32 LE bytes, vocab_size × hidden_size.
    pub fn load_lm_head_from_embedding(&mut self, embedding_bytes: &[u8]) -> Result<(), JsValue> {
        let hidden_size = self.model.config.hidden_size;
        let vocab_size = self.model.config.vocab_size;
        let expected_len = vocab_size * hidden_size * 4;

        if embedding_bytes.len() != expected_len {
            return Err(JsValue::from_str(&format!(
                "Embedding data wrong size: {} (expected {})",
                embedding_bytes.len(), expected_len
            )));
        }

        let head = self.model.lm_head.as_mut()
            .ok_or_else(|| JsValue::from_str("No lm_head allocated"))?;

        let mut cursor = 0usize;
        for i in 0..vocab_size {
            let row = read_f32s_raw(&mut cursor, hidden_size, embedding_bytes)?;
            let dst = head.weights.get_weights_mut(i);
            dst[..hidden_size].copy_from_slice(&row);
        }

        head.sync_q32();
        Ok(())
    }

    /// Run one MPC forward step. Takes party 0's embedding share (u64 LE bytes),
    /// runs private_model_forward_secure_no_reveal, returns logit share (u64 LE bytes).
    pub fn forward_step(&mut self, embedding_share_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
        let hidden_size = self.model.config.hidden_size;

        // Parse embedding share from u64 LE bytes
        if embedding_share_bytes.len() != hidden_size * 8 {
            return Err(JsValue::from_str(&format!(
                "Bad embedding share length: {} (expected {})",
                embedding_share_bytes.len(),
                hidden_size * 8
            )));
        }

        let mut share = Vec::with_capacity(hidden_size);
        for i in 0..hidden_size {
            let off = i * 8;
            share.push(u64::from_le_bytes([
                embedding_share_bytes[off],
                embedding_share_bytes[off + 1],
                embedding_share_bytes[off + 2],
                embedding_share_bytes[off + 3],
                embedding_share_bytes[off + 4],
                embedding_share_bytes[off + 5],
                embedding_share_bytes[off + 6],
                embedding_share_bytes[off + 7],
            ]));
        }

        let input_share = SharedVec64(share);
        let position = self.position;

        let result = klearu_private::private_model::private_model_forward_secure_no_reveal(
            0,
            &mut self.model,
            &input_share,
            position,
            &mut self.triples,
            &mut self.transport,
        )
        .map_err(|e| JsValue::from_str(&format!("MPC forward error: {}", e)))?;

        self.position += 1;

        // Encode logit shares as u64 LE bytes
        let mut out = Vec::with_capacity(result.len() * 8);
        for &v in &result.0 {
            out.extend_from_slice(&v.to_le_bytes());
        }
        Ok(out)
    }

    /// Reset KV caches and position counter for a new conversation.
    pub fn reset(&mut self) {
        self.model.reset_kv_caches();
        self.position = 0;
    }
}

/// Load a Linear layer's weights from the binary cursor.
fn load_linear(linear: &mut Linear, cursor: &mut usize, data: &[u8]) -> Result<(), JsValue> {
    let in_f = linear.in_features();
    let out_f = linear.out_features();

    for i in 0..out_f {
        let row = read_f32s_raw(cursor, in_f, data)?;
        let dst = linear.weights.get_weights_mut(i);
        dst[..in_f].copy_from_slice(&row);
    }
    Ok(())
}

/// Read n f32 values from data at cursor position.
fn read_f32s_raw(cursor: &mut usize, n: usize, data: &[u8]) -> Result<Vec<f32>, JsValue> {
    let byte_len = n * 4;
    if *cursor + byte_len > data.len() {
        return Err(JsValue::from_str(&format!(
            "Weight data too short at offset {}: need {} more bytes, have {}",
            *cursor, byte_len, data.len() - *cursor
        )));
    }
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let off = *cursor + i * 4;
        values.push(f32::from_le_bytes([
            data[off], data[off + 1], data[off + 2], data[off + 3],
        ]));
    }
    *cursor += byte_len;
    Ok(values)
}

/// Sync Q32.32 pre-quantized weights on all Linear projections.
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
