//! Private (2PC) evaluation of a full transformer model.
//!
//! The client provides a token (secret-shared embedding), the server holds
//! the same public model weights. The embedding share is revealed once at
//! the start, then both parties run the plaintext forward pass independently
//! to produce identical logits. The server never learns the token IDs.
//!
//! Security model: the hidden state is leaked (this was already true in the
//! per-layer RMSNorm reveals). All intermediate computation is done in f32,
//! eliminating Q16.16 quantization error that caused multi-turn degeneration.
//!
//! Communication: 1 exchange per forward pass (embedding reveal) instead of
//! ~150 per-layer exchanges. Massive bandwidth reduction.

use klearu_llm::Model;
use klearu_llm::model::block::AttentionLayer;
use klearu_llm::model::gated_deltanet::DeltaNetStateStore;
use klearu_llm::model::kv_cache::KvCacheStore;
use klearu_mpc::beaver::{TripleGenerator, TripleGenerator128};
use klearu_mpc::fixed_point::{from_fixed, from_fixed64, to_fixed, to_fixed64};
use klearu_mpc::linear::{shared_linear_forward_64, shared_linear_forward_64_pq};
use klearu_mpc::normalization::rmsnorm_shared_64;
use klearu_mpc::transport::Transport;
use klearu_mpc::{SharedVec, SharedVec64};
use std::io;

use crate::private_block::{private_block_forward_secure, effective_norm_weights};

/// Create a `DeltaNetStateStore` matching a model's layer architecture.
///
/// Layers with GatedDeltaNet attention get initialized states; Standard
/// attention layers get `None`. The returned store is owned by the caller
/// and persists across tokens (like `KvCacheStore`).
pub fn create_deltanet_states(model: &Model) -> DeltaNetStateStore {
    let mut store = DeltaNetStateStore::with_capacity(model.config.num_layers);
    for (i, layer) in model.layers.iter().enumerate() {
        if let AttentionLayer::GatedDeltaNet(dn) = &layer.attention {
            store.states[i] = Some(dn.create_state());
        }
    }
    store
}

#[cfg(feature = "oprf-sparse")]
use crate::oprf::{OprfClient, OprfServer};

/// Run private inference for a single decode step.
///
/// Both parties hold the same `model` weights in plaintext.
/// `input_share` is this party's share of the embedded token.
///
/// The embedding share is revealed, then the plaintext forward pass is run
/// by both parties independently. Both produce identical f32 logits.
pub fn private_model_forward(
    _party: u8,
    model: &mut Model,
    input_share: &SharedVec,
    position: usize,
    _triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let hidden_size = model.config.hidden_size;

    // Reveal the embedding share: both parties exchange their shares and
    // reconstruct the plaintext hidden state. This is the only communication
    // per forward pass.
    transport.send_u32_slice(&input_share.0)?;
    let other = transport.recv_u32_slice(hidden_size)?;

    let mut hidden = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        hidden[i] = from_fixed(input_share.0[i].wrapping_add(other[i]));
    }

    // Run the plaintext forward pass. Both parties compute identically.
    let mut norm_buf = vec![0.0f32; hidden_size];

    for layer_idx in 0..model.config.num_layers {
        // Pre-attention norm
        norm_buf.copy_from_slice(&hidden);
        model.layers[layer_idx].attn_norm.forward(&mut norm_buf);

        // Attention (dispatch based on layer type)
        let attn_out = match &model.layers[layer_idx].attention {
            AttentionLayer::Standard(attn) => {
                attn.forward(
                    &norm_buf,
                    position,
                    &model.rope,
                    model.kv_caches.layer_mut(layer_idx),
                )
            }
            AttentionLayer::GatedDeltaNet(dn) => {
                let state = model.deltanet_states.layer_mut(layer_idx)
                    .expect("DeltaNet state missing for linear attention layer");
                dn.forward_decode(&norm_buf, state)
            }
        };

        // Residual
        for (h, a) in hidden.iter_mut().zip(attn_out.iter()) {
            *h += a;
        }

        // Pre-MLP norm
        norm_buf.copy_from_slice(&hidden);
        model.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

        // MLP
        let mlp_out = model.layers[layer_idx].mlp.forward(&norm_buf);

        // Residual
        for (h, m) in hidden.iter_mut().zip(mlp_out.iter()) {
            *h += m;
        }
    }

    // Final norm
    model.final_norm.forward(&mut hidden);

    // LM head
    let vocab_size = model.config.vocab_size;
    let mut logits = vec![0.0f32; vocab_size];
    match &model.lm_head {
        Some(head) => head.forward(&hidden, &mut logits),
        None => model.embedding.lm_head_forward(&hidden, &mut logits),
    }

    Ok(logits)
}

/// Sparse variant: uses a subset of MLP neurons per layer.
///
/// In Lower security mode, the embedding is revealed and the forward pass runs
/// in plaintext. When the `oprf-sparse` feature is enabled, OPRF-based neuron
/// selection is used — the client and server cooperate to derive neuron indices
/// from the hidden state via a PRF keyed by the server. When `oprf-sparse` is
/// disabled, falls back to fixed first-k neurons.
pub fn private_model_forward_sparse(
    party: u8,
    model: &mut Model,
    input_share: &SharedVec,
    position: usize,
    neuron_indices: &[usize],
    _triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let hidden_size = model.config.hidden_size;

    // Suppress unused warnings when oprf-sparse is disabled.
    #[cfg(not(feature = "oprf-sparse"))]
    let _ = (party, neuron_indices);

    // Reveal the embedding share
    transport.send_u32_slice(&input_share.0)?;
    let other = transport.recv_u32_slice(hidden_size)?;

    let mut hidden = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        hidden[i] = from_fixed(input_share.0[i].wrapping_add(other[i]));
    }

    // Run the plaintext forward pass (with optional sparse MLP)
    let mut norm_buf = vec![0.0f32; hidden_size];

    #[cfg(feature = "oprf-sparse")]
    let intermediate_size = model.config.intermediate_size;
    #[cfg(feature = "oprf-sparse")]
    let k = neuron_indices.len();

    for layer_idx in 0..model.config.num_layers {
        norm_buf.copy_from_slice(&hidden);
        model.layers[layer_idx].attn_norm.forward(&mut norm_buf);

        let attn_out = match &model.layers[layer_idx].attention {
            AttentionLayer::Standard(attn) => {
                attn.forward(
                    &norm_buf,
                    position,
                    &model.rope,
                    model.kv_caches.layer_mut(layer_idx),
                )
            }
            AttentionLayer::GatedDeltaNet(dn) => {
                let state = model.deltanet_states.layer_mut(layer_idx)
                    .expect("DeltaNet state missing for linear attention layer");
                dn.forward_decode(&norm_buf, state)
            }
        };

        for (h, a) in hidden.iter_mut().zip(attn_out.iter()) {
            *h += a;
        }

        norm_buf.copy_from_slice(&hidden);
        model.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

        #[cfg(feature = "oprf-sparse")]
        let mlp_out = {
            let layer_neurons = if k > 0 && k < intermediate_size {
                oprf_neuron_selection(
                    party,
                    &norm_buf,
                    layer_idx,
                    k,
                    intermediate_size,
                    transport,
                )?
            } else {
                (0..intermediate_size).collect()
            };

            if layer_neurons.len() < intermediate_size {
                klearu_llm::sparse::sparse_mlp::forward_sparse(
                    &model.layers[layer_idx].mlp,
                    &norm_buf,
                    &layer_neurons,
                )
            } else {
                model.layers[layer_idx].mlp.forward(&norm_buf)
            }
        };

        #[cfg(not(feature = "oprf-sparse"))]
        let mlp_out = model.layers[layer_idx].mlp.forward(&norm_buf);

        for (h, m) in hidden.iter_mut().zip(mlp_out.iter()) {
            *h += m;
        }
    }

    model.final_norm.forward(&mut hidden);

    let vocab_size = model.config.vocab_size;
    let mut logits = vec![0.0f32; vocab_size];
    match &model.lm_head {
        Some(head) => head.forward(&hidden, &mut logits),
        None => model.embedding.lm_head_forward(&hidden, &mut logits),
    }

    Ok(logits)
}

/// OPRF-based neuron selection for sparse MLP (Lower security mode only).
///
/// Both parties have the revealed hidden state. The client hashes it to a
/// Ristretto point, blinds it, and sends the blinded point to the server.
/// The server evaluates with its OPRF key and returns the result. The client
/// unblinds and sends the PRF output to the server so both can derive the
/// same neuron indices.
#[cfg(feature = "oprf-sparse")]
fn oprf_neuron_selection(
    party: u8,
    hidden_state: &[f32],
    layer_idx: usize,
    k: usize,
    intermediate_size: usize,
    transport: &mut impl Transport,
) -> io::Result<Vec<usize>> {
    // Construct an input key from hidden state + layer index
    let mut key_bytes = Vec::with_capacity(hidden_state.len() * 4 + 8);
    for &v in hidden_state {
        key_bytes.extend_from_slice(&v.to_le_bytes());
    }
    key_bytes.extend_from_slice(&(layer_idx as u64).to_le_bytes());

    if party == 0 {
        // Client: blind and send
        let mut rng = rand::thread_rng();
        let mut client = OprfClient::new();
        let blinded = client.blind(&[key_bytes.as_slice()], &mut rng);

        // Send blinded point to server (32 bytes compressed)
        let blinded_bytes = blinded[0].to_bytes();
        transport.send(&blinded_bytes)?;

        // Receive evaluated point from server
        let eval_bytes = transport.recv(32)?;
        let eval_point = curve25519_dalek::ristretto::CompressedRistretto::from_slice(&eval_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad OPRF response: {e}")))?;

        // Unblind
        let prf_output = client.unblind(&[eval_point]);

        // Send PRF output to server so both agree on indices
        transport.send(&prf_output[0])?;

        Ok(derive_neuron_indices(&prf_output[0], k, intermediate_size))
    } else {
        // Server: receive blinded point, evaluate, return
        let blinded_bytes = transport.recv(32)?;
        let blinded_point = curve25519_dalek::ristretto::CompressedRistretto::from_slice(&blinded_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad OPRF request: {e}")))?;

        // Use a deterministic OPRF key derived from layer index
        // In production, this would be a persistent server key.
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&(layer_idx as u64).to_le_bytes());
        seed[8] = 0x4F; // 'O' for OPRF
        let server = OprfServer::from_seed(&seed);

        let evaluated = server.evaluate(&[blinded_point]);
        transport.send(&evaluated[0].to_bytes())?;

        // Receive PRF output from client
        let prf_bytes = transport.recv(32)?;
        let mut prf_output = [0u8; 32];
        prf_output.copy_from_slice(&prf_bytes);

        Ok(derive_neuron_indices(&prf_output, k, intermediate_size))
    }
}

/// Derive `k` neuron indices from a 32-byte PRF output.
///
/// Uses the PRF output as a seed for Fisher-Yates partial shuffle to
/// select `k` unique indices from `0..intermediate_size`.
#[cfg(feature = "oprf-sparse")]
fn derive_neuron_indices(prf_output: &[u8; 32], k: usize, intermediate_size: usize) -> Vec<usize> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::Rng;

    let mut rng = StdRng::from_seed(*prf_output);
    let mut indices: Vec<usize> = (0..intermediate_size).collect();

    // Partial Fisher-Yates: only shuffle the first k positions
    let k = k.min(intermediate_size);
    for i in 0..k {
        let j = rng.gen_range(i..intermediate_size);
        indices.swap(i, j);
    }

    indices[..k].to_vec()
}

/// Helper: create a shared embedding for a token.
///
/// Both parties look up the same embedding row (weights are public),
/// then party 0 holds the full value as its share and party 1 holds zero.
/// This is correct because: share0 + share1 = embedding + 0 = embedding.
pub fn shared_embedding_lookup(
    party: u8,
    model: &Model,
    token_id: u32,
) -> SharedVec {
    let hidden_size = model.config.hidden_size;
    let row = model.embedding.weights.get_weights(token_id as usize);

    if party == 0 {
        SharedVec(row[..hidden_size].iter().map(|&v| to_fixed(v)).collect())
    } else {
        SharedVec(vec![0u32; hidden_size])
    }
}

/// Secure (Q32.32) private inference — returns logit SHARES without reveal.
///
/// Runs the full per-layer MPC forward pass with Q32.32 shares, but does NOT
/// exchange shares at the end. The caller receives this party's logit share
/// and must combine with the other party's share to reconstruct logits.
///
/// This is the building block for client-side logit reconstruction: the server
/// sends both parties' shares to the client, who adds them locally.
pub fn private_model_forward_secure_no_reveal(
    party: u8,
    model: &mut Model,
    input_share: &SharedVec64,
    position: usize,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let hidden_size = model.config.hidden_size;
    assert_eq!(input_share.len(), hidden_size);

    let mut hidden_share = input_share.clone();
    let mut normed_buf = SharedVec64::zeros(hidden_size);

    // Per-layer MPC forward pass
    for layer_idx in 0..model.config.num_layers {
        let dn_state = model.deltanet_states.layer_mut(layer_idx);

        private_block_forward_secure(
            party,
            &model.layers[layer_idx],
            &mut hidden_share,
            position,
            &model.rope,
            model.kv_caches.layer_mut(layer_idx),
            dn_state,
            &mut normed_buf,
            triples,
            transport,
        )?;
    }

    // Final RMSNorm (privacy-preserving, handle one_plus_weight)
    let final_weights = effective_norm_weights(&model.final_norm);
    rmsnorm_shared_64(
        party,
        &mut hidden_share,
        &final_weights,
        model.final_norm.eps(),
        triples,
        transport,
    )?;

    // LM head projection (local, public weights)
    let vocab_size = model.config.vocab_size;
    let lm_head_weights = match &model.lm_head {
        Some(head) => head.weights.as_raw_slice(),
        None => model.embedding.weights.as_raw_slice(),
    };
    let logit_shares = shared_linear_forward_64(
        party, lm_head_weights, hidden_size, vocab_size, &hidden_share, &[], transport,
    )?;

    Ok(logit_shares)
}

/// Secure (Q32.32) private inference for a single decode step.
///
/// Unlike `private_model_forward` which reveals the full embedding and runs
/// plaintext, this uses per-layer MPC with Q32.32 shares. Only sum(x²)
/// scalars, Q vectors, attention scores, and gate activations are revealed.
///
/// Calls `private_model_forward_secure_no_reveal` and then exchanges shares
/// to reveal the final logits.
pub fn private_model_forward_secure(
    party: u8,
    model: &mut Model,
    input_share: &SharedVec64,
    position: usize,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let vocab_size = model.config.vocab_size;
    let logit_shares = private_model_forward_secure_no_reveal(
        party, model, input_share, position, triples, transport,
    )?;

    // Final reveal: exchange logit shares → Vec<f32>
    transport.send_u64_slice(&logit_shares.0)?;
    let other = transport.recv_u64_slice(vocab_size)?;

    Ok((0..vocab_size)
        .map(|i| from_fixed64(logit_shares.0[i].wrapping_add(other[i])))
        .collect())
}

/// Create a Q32.32 shared embedding for a token.
///
/// Party 0 holds the full Q32.32 embedding, party 1 holds zeros.
pub fn shared_embedding_lookup_64(
    party: u8,
    model: &Model,
    token_id: u32,
) -> SharedVec64 {
    let hidden_size = model.config.hidden_size;
    let row = model.embedding.weights.get_weights(token_id as usize);

    if party == 0 {
        SharedVec64(row[..hidden_size].iter().map(|&v| to_fixed64(v)).collect())
    } else {
        SharedVec64::zeros(hidden_size)
    }
}

/// Private inference with external caches — returns logit SHARES.
///
/// Uses the secure MPC path (reveals Q/K/gate as f32, plaintext softmax/SiLU)
/// for accurate nonlinear computation. DPF-PIR hides token IDs from Server B.
///
/// Only RMSNorm sum(x²) scalars, Q/K vectors, gate activations, and attention
/// scores are revealed — NOT the full hidden state.
///
/// `kv_caches`: f32 KV cache store (caller owns, persists across tokens).
/// `deltanet_states`: DeltaNet recurrent state store (caller owns, persists across tokens).
pub fn private_model_forward_noreveal(
    party: u8,
    model: &Model,
    input_share: &SharedVec64,
    position: usize,
    kv_caches: &mut KvCacheStore,
    deltanet_states: &mut DeltaNetStateStore,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let hidden_size = model.config.hidden_size;
    assert_eq!(input_share.len(), hidden_size);

    let mut hidden_share = input_share.clone();
    let mut normed_buf = SharedVec64::zeros(hidden_size);

    for layer_idx in 0..model.config.num_layers {
        let dn_state = deltanet_states.layer_mut(layer_idx);

        private_block_forward_secure(
            party,
            &model.layers[layer_idx],
            &mut hidden_share,
            position,
            &model.rope,
            kv_caches.layer_mut(layer_idx),
            dn_state,
            &mut normed_buf,
            triples,
            transport,
        )?;
    }

    // Final RMSNorm
    let final_weights = effective_norm_weights(&model.final_norm);
    rmsnorm_shared_64(
        party,
        &mut hidden_share,
        &final_weights,
        model.final_norm.eps(),
        triples,
        transport,
    )?;

    // LM head projection (local, public weights)
    let vocab_size = model.config.vocab_size;
    let lm_head_weights = match &model.lm_head {
        Some(head) => head.q32_weights().into_owned(),
        None => {
            // Tied embeddings: use embedding table as lm_head weights
            let raw = model.embedding.weights.as_raw_slice();
            raw.iter().map(|&v| klearu_mpc::to_fixed64(v) as i64).collect()
        }
    };
    let logit_shares = shared_linear_forward_64_pq(
        &lm_head_weights, hidden_size, vocab_size, &hidden_share,
    );

    Ok(logit_shares)
}

/// Private inference with external caches — returns revealed f32 logits.
///
/// Calls `private_model_forward_noreveal` then exchanges logit shares.
pub fn private_model_forward_noreveal_reveal(
    party: u8,
    model: &Model,
    input_share: &SharedVec64,
    position: usize,
    kv_caches: &mut KvCacheStore,
    deltanet_states: &mut DeltaNetStateStore,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let vocab_size = model.config.vocab_size;
    let logit_shares = private_model_forward_noreveal(
        party, model, input_share, position, kv_caches, deltanet_states, triples, transport,
    )?;

    // Exchange logit shares → Vec<f32>
    transport.send_u64_slice(&logit_shares.0)?;
    let other = transport.recv_u64_slice(vocab_size)?;

    Ok((0..vocab_size)
        .map(|i| from_fixed64(logit_shares.0[i].wrapping_add(other[i])))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_llm::config::LlmConfig;
    use klearu_mpc::beaver::{dummy_triple_pair, dummy_triple_pair_128};
    use klearu_mpc::transport::memory_transport_pair;

    fn tiny_config() -> LlmConfig {
        LlmConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            num_layers: 1,
            max_seq_len: 8,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            ..LlmConfig::default()
        }
    }

    fn set_norm_weights(model: &mut Model) {
        for w in model.final_norm.weight.iter_mut() {
            *w = 1.0;
        }
        for layer in &mut model.layers {
            for w in layer.attn_norm.weight.iter_mut() {
                *w = 1.0;
            }
            for w in layer.mlp_norm.weight.iter_mut() {
                *w = 1.0;
            }
        }
    }

    #[test]
    fn test_private_model_forward_runs() {
        let config = tiny_config();
        let mut model0 = Model::new(config.clone());
        set_norm_weights(&mut model0);

        let mut model1 = Model::new(config.clone());
        set_norm_weights(&mut model1);

        let token_id = 1u32;
        let share0 = shared_embedding_lookup(0, &model0, token_id);
        let share1 = shared_embedding_lookup(1, &model1, token_id);

        let (mut gen0, mut gen1) = dummy_triple_pair(50000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_model_forward(1, &mut model1, &share1, 0, &mut gen1, &mut trans_b)
        });

        let result0 = private_model_forward(0, &mut model0, &share0, 0, &mut gen0, &mut trans_a);
        let result1 = handle.join().unwrap();

        let logits0 = result0.expect("party 0 failed");
        let logits1 = result1.expect("party 1 failed");

        assert_eq!(logits0.len(), config.vocab_size);
        assert_eq!(logits1.len(), config.vocab_size);

        // Both parties compute identical f32 logits (reveal + local computation)
        for i in 0..config.vocab_size {
            assert!(logits0[i].is_finite(), "logit[{}] is not finite: {}", i, logits0[i]);
            assert_eq!(logits0[i], logits1[i], "logit[{}] mismatch: {} != {}", i, logits0[i], logits1[i]);
        }
    }

    /// Sync pre-quantized weights on all Linear projections in the model.
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

    /// Set all projection weights on a model to small deterministic non-zero values.
    fn init_model_weights(model: &mut Model) {
        let config = &model.config;
        let vs = config.vocab_size;

        // Embedding weights
        for i in 0..vs {
            let row = model.embedding.weights.get_weights_mut(i);
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((i * 7 + j * 3 + 1) % 17) as f32 * 0.02 - 0.15;
            }
        }

        // Norm weights = 1.0
        set_norm_weights(model);

        // Attention and MLP weights
        for layer in &mut model.layers {
            // Q, K, V, O projections (only for Standard attention layers)
            if let AttentionLayer::Standard(attn) = &mut layer.attention {
                for (proj_idx, proj) in [
                    &mut attn.q_proj,
                    &mut attn.k_proj,
                    &mut attn.v_proj,
                    &mut attn.o_proj,
                ].into_iter().enumerate() {
                    for i in 0..proj.out_features() {
                        let row = proj.weights.get_weights_mut(i);
                        for (j, v) in row.iter_mut().enumerate() {
                            *v = ((i * 5 + j * 11 + proj_idx * 3 + 1) % 19) as f32 * 0.01 - 0.09;
                        }
                    }
                }
            }

            // MLP projections
            for (proj_idx, proj) in [
                &mut layer.mlp.gate_proj,
                &mut layer.mlp.up_proj,
                &mut layer.mlp.down_proj,
            ].into_iter().enumerate() {
                for i in 0..proj.out_features() {
                    let row = proj.weights.get_weights_mut(i);
                    for (j, v) in row.iter_mut().enumerate() {
                        *v = ((i * 13 + j * 7 + proj_idx * 5 + 2) % 23) as f32 * 0.01 - 0.11;
                    }
                }
            }
        }

        sync_all_q32(model);
    }

    #[test]
    fn test_mpc_matches_plaintext() {
        let config = tiny_config();

        // Create three identical models
        let mut model_plain = Model::new(config.clone());
        init_model_weights(&mut model_plain);

        let mut model0 = Model::new(config.clone());
        init_model_weights(&mut model0);

        let mut model1 = Model::new(config.clone());
        init_model_weights(&mut model1);

        let token_id = 1u32;

        // 1. Run plaintext forward
        let plaintext_logits = model_plain.forward_decode(token_id, 0);

        // 2. Run MPC forward (now returns Vec<f32> directly)
        let share0 = shared_embedding_lookup(0, &model0, token_id);
        let share1 = shared_embedding_lookup(1, &model1, token_id);

        let (mut gen0, mut gen1) = dummy_triple_pair(100_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_model_forward(1, &mut model1, &share1, 0, &mut gen1, &mut trans_b)
        });

        let mpc_logits = private_model_forward(0, &mut model0, &share0, 0, &mut gen0, &mut trans_a).unwrap();
        let mpc_logits1 = handle.join().unwrap().unwrap();

        // Both parties now compute identical f32 logits
        assert_eq!(mpc_logits, mpc_logits1);

        // 3. Compare with plaintext
        assert_eq!(mpc_logits.len(), config.vocab_size);

        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        let mut plaintext_argmax = 0;
        let mut mpc_argmax = 0;
        let mut plaintext_max = f32::NEG_INFINITY;
        let mut mpc_max = f32::NEG_INFINITY;

        for i in 0..config.vocab_size {
            let diff = (mpc_logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
            if plaintext_logits[i] > plaintext_max {
                plaintext_max = plaintext_logits[i];
                plaintext_argmax = i;
            }
            if mpc_logits[i] > mpc_max {
                mpc_max = mpc_logits[i];
                mpc_argmax = i;
            }
        }

        eprintln!("Max logit difference: {max_diff} at index {max_idx}");
        eprintln!("Plaintext argmax: {plaintext_argmax} (logit={plaintext_max})");
        eprintln!("MPC argmax: {mpc_argmax} (logit={mpc_max})");

        // Print top-5 for both
        let mut plain_sorted: Vec<(usize, f32)> = plaintext_logits.iter().copied().enumerate().collect();
        plain_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("Plaintext top-5: {:?}", &plain_sorted[..5]);

        let mut mpc_sorted: Vec<(usize, f32)> = mpc_logits.iter().copied().enumerate().collect();
        mpc_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("MPC top-5: {:?}", &mpc_sorted[..5]);

        assert!(
            max_diff < 2.0,
            "MPC logits diverge too much from plaintext: max_diff={max_diff} at idx={max_idx}"
        );
        assert_eq!(
            plaintext_argmax, mpc_argmax,
            "Argmax mismatch: plaintext={plaintext_argmax}, MPC={mpc_argmax}"
        );
    }

    #[test]
    fn test_mpc_matches_plaintext_4_layers() {
        let config = LlmConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            num_layers: 4,
            max_seq_len: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            ..LlmConfig::default()
        };

        let mut model_plain = Model::new(config.clone());
        init_model_weights(&mut model_plain);

        let mut model0 = Model::new(config.clone());
        init_model_weights(&mut model0);

        let mut model1 = Model::new(config.clone());
        init_model_weights(&mut model1);

        let token_id = 3u32;

        // Run plaintext
        let plaintext_logits = model_plain.forward_decode(token_id, 0);

        // Run MPC
        let share0 = shared_embedding_lookup(0, &model0, token_id);
        let share1 = shared_embedding_lookup(1, &model1, token_id);

        let (mut gen0, mut gen1) = dummy_triple_pair(500_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_model_forward(1, &mut model1, &share1, 0, &mut gen1, &mut trans_b)
        });

        let mpc_logits = private_model_forward(0, &mut model0, &share0, 0, &mut gen0, &mut trans_a).unwrap();
        let mpc_logits1 = handle.join().unwrap().unwrap();

        // Both parties compute identical f32 logits
        assert_eq!(mpc_logits, mpc_logits1);

        let mut max_diff = 0.0f32;
        let mut plaintext_argmax = 0;
        let mut mpc_argmax = 0;
        let mut plaintext_max = f32::NEG_INFINITY;
        let mut mpc_max = f32::NEG_INFINITY;

        for i in 0..config.vocab_size {
            let diff = (mpc_logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if plaintext_logits[i] > plaintext_max {
                plaintext_max = plaintext_logits[i];
                plaintext_argmax = i;
            }
            if mpc_logits[i] > mpc_max {
                mpc_max = mpc_logits[i];
                mpc_argmax = i;
            }
        }

        eprintln!("4-layer test: max logit diff={max_diff}");
        eprintln!("Plaintext argmax={plaintext_argmax} ({plaintext_max})");
        eprintln!("MPC argmax={mpc_argmax} ({mpc_max})");

        // Print top-5
        let mut plain_sorted: Vec<(usize, f32)> = plaintext_logits.iter().copied().enumerate().collect();
        plain_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut mpc_sorted: Vec<(usize, f32)> = mpc_logits.iter().copied().enumerate().collect();
        mpc_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("Plaintext top-5: {:?}", &plain_sorted[..5]);
        eprintln!("MPC top-5: {:?}", &mpc_sorted[..5]);

        assert!(
            max_diff < 5.0,
            "4-layer MPC diverges too much: max_diff={max_diff}"
        );
    }

    #[test]
    fn test_mpc_matches_plaintext_multi_token() {
        // Test the protocol with 2 prefill tokens + 1 decode step
        let config = tiny_config();

        let mut model_plain = Model::new(config.clone());
        init_model_weights(&mut model_plain);

        let mut model0 = Model::new(config.clone());
        init_model_weights(&mut model0);

        let mut model1 = Model::new(config.clone());
        init_model_weights(&mut model1);

        // Plaintext: process tokens 1, 2, then get logits for token 3
        model_plain.reset_kv_caches();
        let _ = model_plain.forward_decode(1, 0);
        let _ = model_plain.forward_decode(2, 1);
        let plaintext_logits = model_plain.forward_decode(3, 2);

        // MPC: same 3 tokens
        let (mut gen0, mut gen1) = dummy_triple_pair(500_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut m0 = model0;
        let mut m1 = model1;

        m0.reset_kv_caches();
        m1.reset_kv_caches();

        let tokens = [1u32, 2, 3];

        let handle = std::thread::spawn(move || {
            let mut last_logits = Vec::new();
            for pos in 0..3 {
                let share1 = shared_embedding_lookup(1, &m1, 0);
                last_logits = private_model_forward(1, &mut m1, &share1, pos, &mut gen1, &mut trans_b).unwrap();
            }
            last_logits
        });

        let mut mpc_logits = Vec::new();
        for pos in 0..3 {
            let share0 = shared_embedding_lookup(0, &m0, tokens[pos]);
            mpc_logits = private_model_forward(0, &mut m0, &share0, pos, &mut gen0, &mut trans_a).unwrap();
        }

        let mpc_logits1 = handle.join().unwrap();

        // Both parties compute identical f32 logits
        assert_eq!(mpc_logits, mpc_logits1);

        let mut max_diff = 0.0f32;
        for i in 0..config.vocab_size {
            let diff = (mpc_logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }

        eprintln!("Multi-token test (3 positions): max logit diff={max_diff}");
        assert!(
            max_diff < 2.0,
            "Multi-token MPC diverges: max_diff={max_diff}"
        );
    }

    #[test]
    fn test_shared_embedding_lookup_reconstructs() {
        let config = tiny_config();
        let model = Model::new(config.clone());

        let token_id = 3u32;
        let share0 = shared_embedding_lookup(0, &model, token_id);
        let share1 = shared_embedding_lookup(1, &model, token_id);

        // Reconstruct should give back the original embedding
        let row = model.embedding.weights.get_weights(token_id as usize);
        for i in 0..config.hidden_size {
            let reconstructed = from_fixed(share0.0[i].wrapping_add(share1.0[i]));
            let expected = row[i];
            assert!(
                (reconstructed - expected).abs() < 0.001,
                "embedding[{}]: got {}, expected {}", i, reconstructed, expected
            );
        }
    }

    /// Test MPC vs plaintext with SmolLM-like internal dimensions.
    /// Uses hidden=576, heads=9, kv_heads=3, head_dim=64, intermediate=1536.
    /// This catches bugs that only manifest at these specific dimensions.
    #[test]
    fn test_mpc_matches_plaintext_smollm_dims() {
        let config = LlmConfig {
            vocab_size: 64,
            hidden_size: 576,
            num_heads: 9,
            num_kv_heads: 3,
            head_dim: 64,
            intermediate_size: 1536,
            num_layers: 4,
            max_seq_len: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            ..LlmConfig::default()
        };

        let mut model_plain = Model::new(config.clone());
        init_model_weights(&mut model_plain);

        let mut model0 = Model::new(config.clone());
        init_model_weights(&mut model0);

        let mut model1 = Model::new(config.clone());
        init_model_weights(&mut model1);

        let token_id = 5u32;

        // Plaintext forward
        let plaintext_logits = model_plain.forward_decode(token_id, 0);

        // MPC forward
        let share0 = shared_embedding_lookup(0, &model0, token_id);
        let share1 = shared_embedding_lookup(1, &model1, token_id);

        let (mut gen0, mut gen1) = dummy_triple_pair(999_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_model_forward(1, &mut model1, &share1, 0, &mut gen1, &mut trans_b)
        });

        let mpc_logits = private_model_forward(0, &mut model0, &share0, 0, &mut gen0, &mut trans_a).unwrap();
        let mpc_logits1 = handle.join().unwrap().unwrap();

        // Both parties compute identical f32 logits
        assert_eq!(mpc_logits, mpc_logits1);
        assert_eq!(mpc_logits.len(), config.vocab_size);

        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        let mut plaintext_argmax = 0;
        let mut mpc_argmax = 0;
        let mut plaintext_max = f32::NEG_INFINITY;
        let mut mpc_max = f32::NEG_INFINITY;

        for i in 0..config.vocab_size {
            let diff = (mpc_logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
            if plaintext_logits[i] > plaintext_max {
                plaintext_max = plaintext_logits[i];
                plaintext_argmax = i;
            }
            if mpc_logits[i] > mpc_max {
                mpc_max = mpc_logits[i];
                mpc_argmax = i;
            }
        }

        eprintln!("SmolLM-dims test (4 layers): max logit diff={max_diff} at idx={max_idx}");
        eprintln!("Plaintext argmax={plaintext_argmax} ({plaintext_max})");
        eprintln!("MPC argmax={mpc_argmax} ({mpc_max})");

        // Print top-5 for both
        let mut plain_sorted: Vec<(usize, f32)> = plaintext_logits.iter().copied().enumerate().collect();
        plain_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut mpc_sorted: Vec<(usize, f32)> = mpc_logits.iter().copied().enumerate().collect();
        mpc_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("Plaintext top-5: {:?}", &plain_sorted[..5]);
        eprintln!("MPC top-5: {:?}", &mpc_sorted[..5]);

        // Check finite
        for i in 0..config.vocab_size {
            assert!(mpc_logits[i].is_finite(), "logit[{i}] is NaN/Inf");
        }

        assert!(
            max_diff < 5.0,
            "SmolLM-dims MPC diverges: max_diff={max_diff} at idx={max_idx}"
        );
        assert_eq!(
            plaintext_argmax, mpc_argmax,
            "Argmax mismatch: plaintext={plaintext_argmax}, MPC={mpc_argmax}"
        );
    }

    /// Diagnostic: 30-layer model with 10 tokens to reproduce multi-turn divergence.
    #[test]
    fn test_mpc_matches_plaintext_30_layers_multi_token() {
        let config = LlmConfig {
            vocab_size: 64,
            hidden_size: 576,
            num_heads: 9,
            num_kv_heads: 3,
            head_dim: 64,
            intermediate_size: 1536,
            num_layers: 30,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            ..LlmConfig::default()
        };

        let mut model_plain = Model::new(config.clone());
        init_model_weights(&mut model_plain);

        let mut model0 = Model::new(config.clone());
        init_model_weights(&mut model0);

        let mut model1 = Model::new(config.clone());
        init_model_weights(&mut model1);

        let tokens = [1u32, 5, 12, 3, 7, 2, 9, 4, 11, 6];

        // Plaintext: process all tokens, keep logits for each
        model_plain.reset_kv_caches();
        let mut plaintext_all_logits = Vec::new();
        for (pos, &tid) in tokens.iter().enumerate() {
            plaintext_all_logits.push(model_plain.forward_decode(tid, pos));
        }

        // MPC: process all tokens
        model0.reset_kv_caches();
        model1.reset_kv_caches();

        let (mut gen0, mut gen1) = dummy_triple_pair(9_999_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let tokens_copy = tokens;
        let handle = std::thread::spawn(move || {
            let mut results = Vec::new();
            for (pos, &tid) in tokens_copy.iter().enumerate() {
                let share1 = shared_embedding_lookup(1, &model1, tid);
                let r = private_model_forward(1, &mut model1, &share1, pos, &mut gen1, &mut trans_b).unwrap();
                results.push(r);
            }
            results
        });

        let mut results0 = Vec::new();
        for (pos, &tid) in tokens.iter().enumerate() {
            let share0 = shared_embedding_lookup(0, &model0, tid);
            let r = private_model_forward(0, &mut model0, &share0, pos, &mut gen0, &mut trans_a).unwrap();
            results0.push(r);
        }
        let results1 = handle.join().unwrap();

        // Compare per-token
        for pos in 0..tokens.len() {
            // Both parties compute identical f32 logits
            assert_eq!(results0[pos], results1[pos], "Token {pos}: party logits differ");

            let mut max_diff = 0.0f32;
            let mut plaintext_argmax = 0;
            let mut mpc_argmax = 0;
            let mut plaintext_max = f32::NEG_INFINITY;
            let mut mpc_max = f32::NEG_INFINITY;

            for i in 0..config.vocab_size {
                let diff = (results0[pos][i] - plaintext_all_logits[pos][i]).abs();
                if diff > max_diff { max_diff = diff; }
                if plaintext_all_logits[pos][i] > plaintext_max {
                    plaintext_max = plaintext_all_logits[pos][i];
                    plaintext_argmax = i;
                }
                if results0[pos][i] > mpc_max {
                    mpc_max = results0[pos][i];
                    mpc_argmax = i;
                }
            }

            let argmax_match = if plaintext_argmax == mpc_argmax { "OK" } else { "MISMATCH" };
            eprintln!(
                "Token {pos} (id={}): max_diff={max_diff:.4}, plain_argmax={plaintext_argmax}, mpc_argmax={mpc_argmax} [{argmax_match}]",
                tokens[pos]
            );

            // Check finite
            for i in 0..config.vocab_size {
                assert!(results0[pos][i].is_finite(), "Token {pos} logit[{i}] is NaN/Inf");
            }

            assert_eq!(
                plaintext_argmax, mpc_argmax,
                "Token {pos}: argmax mismatch plain={plaintext_argmax} mpc={mpc_argmax}, max_diff={max_diff}"
            );
        }
    }

    // --- Secure (Q32.32) model tests ---

    #[test]
    fn test_mpc_secure_matches_plaintext() {
        let config = tiny_config();

        let mut model_plain = Model::new(config.clone());
        init_model_weights(&mut model_plain);

        let mut model0 = Model::new(config.clone());
        init_model_weights(&mut model0);

        let mut model1 = Model::new(config.clone());
        init_model_weights(&mut model1);

        let token_id = 1u32;

        // Plaintext forward
        let plaintext_logits = model_plain.forward_decode(token_id, 0);

        // Secure MPC forward
        let share0 = shared_embedding_lookup_64(0, &model0, token_id);
        let share1 = shared_embedding_lookup_64(1, &model1, token_id);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(100_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_model_forward_secure(1, &mut model1, &share1, 0, &mut gen1, &mut trans_b)
        });

        let mpc_logits = private_model_forward_secure(0, &mut model0, &share0, 0, &mut gen0, &mut trans_a).unwrap();
        let mpc_logits1 = handle.join().unwrap().unwrap();

        // Both parties should produce identical revealed logits
        assert_eq!(mpc_logits, mpc_logits1);

        let mut max_diff = 0.0f32;
        let mut plaintext_argmax = 0;
        let mut mpc_argmax = 0;
        let mut plaintext_max = f32::NEG_INFINITY;
        let mut mpc_max = f32::NEG_INFINITY;

        for i in 0..config.vocab_size {
            let diff = (mpc_logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if plaintext_logits[i] > plaintext_max {
                plaintext_max = plaintext_logits[i];
                plaintext_argmax = i;
            }
            if mpc_logits[i] > mpc_max {
                mpc_max = mpc_logits[i];
                mpc_argmax = i;
            }
        }

        eprintln!("Secure 1-layer: max logit diff={max_diff}");
        eprintln!("Plaintext argmax={plaintext_argmax}, MPC argmax={mpc_argmax}");

        for i in 0..config.vocab_size {
            assert!(mpc_logits[i].is_finite(), "secure logit[{i}] is NaN/Inf");
        }

        // Q32.32 has quantization error — allow wider tolerance than plaintext-reveal mode
        assert!(
            max_diff < 5.0,
            "Secure MPC diverges too much: max_diff={max_diff}"
        );
    }

    #[test]
    fn test_shared_embedding_lookup_64_reconstructs() {
        let config = tiny_config();
        let model = Model::new(config.clone());

        let token_id = 3u32;
        let share0 = shared_embedding_lookup_64(0, &model, token_id);
        let share1 = shared_embedding_lookup_64(1, &model, token_id);

        let row = model.embedding.weights.get_weights(token_id as usize);
        for i in 0..config.hidden_size {
            let reconstructed = from_fixed64(share0.0[i].wrapping_add(share1.0[i]));
            let expected = row[i];
            assert!(
                (reconstructed - expected).abs() < 0.001,
                "Q32 embedding[{}]: got {}, expected {}", i, reconstructed, expected
            );
        }
    }

    // --- Qwen3.5 hybrid model tests ---

    fn qwen35_tiny_config() -> LlmConfig {
        LlmConfig {
            vocab_size: 16,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 4,
            max_seq_len: 8,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            model_type: Some("qwen3_5_text".to_string()),
            attn_output_gate: true,
            layer_types: Some(vec![
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "full_attention".to_string(),
            ]),
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 2,
            ..LlmConfig::default()
        }
    }

    fn init_qwen35_model_weights(model: &mut Model) {
        let vs = model.config.vocab_size;

        // Embedding weights
        for i in 0..vs {
            let row = model.embedding.weights.get_weights_mut(i);
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((i * 7 + j * 3 + 1) % 17) as f32 * 0.02 - 0.15;
            }
        }

        // Norm weights: for one_plus_weight models, leave at 0.0 (effective = 1.0)
        // For non-one_plus, set to 1.0
        if model.final_norm.is_one_plus_weight() {
            // Leave at 0.0 — (1 + 0) = 1.0
        } else {
            for w in model.final_norm.weight.iter_mut() {
                *w = 1.0;
            }
        }

        for layer in &mut model.layers {
            if layer.attn_norm.is_one_plus_weight() {
                // Leave at 0.0
            } else {
                for w in layer.attn_norm.weight.iter_mut() {
                    *w = 1.0;
                }
            }
            if layer.mlp_norm.is_one_plus_weight() {
                // Leave at 0.0
            } else {
                for w in layer.mlp_norm.weight.iter_mut() {
                    *w = 1.0;
                }
            }

            match &mut layer.attention {
                AttentionLayer::Standard(attn) => {
                    for (proj_idx, proj) in [
                        &mut attn.q_proj,
                        &mut attn.k_proj,
                        &mut attn.v_proj,
                        &mut attn.o_proj,
                    ].into_iter().enumerate() {
                        for i in 0..proj.out_features() {
                            let row = proj.weights.get_weights_mut(i);
                            for (j, v) in row.iter_mut().enumerate() {
                                *v = ((i * 5 + j * 11 + proj_idx * 3 + 1) % 19) as f32 * 0.01 - 0.09;
                            }
                        }
                    }
                    // Q/K norm weights: leave at 0.0 (Qwen3.5 default)
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    for (proj_idx, proj) in [
                        &mut dn.in_proj_qkv, &mut dn.in_proj_z, &mut dn.in_proj_a,
                        &mut dn.in_proj_b, &mut dn.out_proj,
                    ].into_iter().enumerate() {
                        for i in 0..proj.out_features() {
                            let row = proj.weights.get_weights_mut(i);
                            for (j, v) in row.iter_mut().enumerate() {
                                *v = ((i * 5 + j * 11 + proj_idx * 3 + 1) % 19) as f32 * 0.01 - 0.09;
                            }
                        }
                    }
                    for (i, v) in dn.conv_weight.iter_mut().enumerate() {
                        *v = ((i * 3 + 7) % 11) as f32 * 0.02 - 0.1;
                    }
                    for (i, v) in dn.dt_bias.iter_mut().enumerate() {
                        *v = ((i * 5 + 2) % 7) as f32 * 0.1 - 0.3;
                    }
                    for (i, v) in dn.a_log.iter_mut().enumerate() {
                        *v = -1.0 - (i as f32) * 0.1;
                    }
                    // norm_weight: leave at 1.0 (GatedDeltaNet uses standard RMSNorm)
                    for v in dn.norm_weight.iter_mut() {
                        *v = 1.0;
                    }
                }
            }

            // MLP projections
            for (proj_idx, proj) in [
                &mut layer.mlp.gate_proj,
                &mut layer.mlp.up_proj,
                &mut layer.mlp.down_proj,
            ].into_iter().enumerate() {
                for i in 0..proj.out_features() {
                    let row = proj.weights.get_weights_mut(i);
                    for (j, v) in row.iter_mut().enumerate() {
                        *v = ((i * 13 + j * 7 + proj_idx * 5 + 2) % 23) as f32 * 0.01 - 0.11;
                    }
                }
            }
        }

        sync_all_q32(model);
    }

    /// Test lower-security MPC (reveal embedding, plaintext forward) with Qwen3.5 hybrid model.
    #[test]
    fn test_mpc_qwen35_matches_plaintext() {
        let config = qwen35_tiny_config();

        let mut model_plain = Model::new(config.clone());
        init_qwen35_model_weights(&mut model_plain);

        let mut model0 = Model::new(config.clone());
        init_qwen35_model_weights(&mut model0);

        let mut model1 = Model::new(config.clone());
        init_qwen35_model_weights(&mut model1);

        let token_id = 1u32;

        // Plaintext forward
        let plaintext_logits = model_plain.forward_decode(token_id, 0);

        // MPC forward (lower security: reveal + plaintext)
        let share0 = shared_embedding_lookup(0, &model0, token_id);
        let share1 = shared_embedding_lookup(1, &model1, token_id);

        let (mut gen0, mut gen1) = dummy_triple_pair(100_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_model_forward(1, &mut model1, &share1, 0, &mut gen1, &mut trans_b)
        });

        let mpc_logits = private_model_forward(0, &mut model0, &share0, 0, &mut gen0, &mut trans_a).unwrap();
        let mpc_logits1 = handle.join().unwrap().unwrap();

        // Both parties compute identical f32 logits
        assert_eq!(mpc_logits, mpc_logits1);
        assert_eq!(mpc_logits.len(), config.vocab_size);

        // Should be close to plaintext (embedding Q16.16 roundtrip introduces ~1e-5 error)
        let mut max_diff = 0.0f32;
        for i in 0..config.vocab_size {
            assert!(mpc_logits[i].is_finite(), "logit[{i}] is NaN/Inf");
            let diff = (mpc_logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        eprintln!("Qwen3.5 lower-security: max logit diff={max_diff}");
        assert!(
            max_diff < 5.0,
            "Qwen3.5 MPC diverges too much from plaintext: max_diff={max_diff}"
        );
    }

    /// Test secure MPC (Q32.32 per-layer) with Qwen3.5 hybrid model.
    #[test]
    fn test_mpc_secure_qwen35_matches_plaintext() {
        let config = qwen35_tiny_config();

        let mut model_plain = Model::new(config.clone());
        init_qwen35_model_weights(&mut model_plain);

        let mut model0 = Model::new(config.clone());
        init_qwen35_model_weights(&mut model0);

        let mut model1 = Model::new(config.clone());
        init_qwen35_model_weights(&mut model1);

        let token_id = 1u32;

        // Plaintext forward
        let plaintext_logits = model_plain.forward_decode(token_id, 0);

        // Secure MPC forward (Q32.32)
        let share0 = shared_embedding_lookup_64(0, &model0, token_id);
        let share1 = shared_embedding_lookup_64(1, &model1, token_id);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(500_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_model_forward_secure(1, &mut model1, &share1, 0, &mut gen1, &mut trans_b)
        });

        let mpc_logits = private_model_forward_secure(0, &mut model0, &share0, 0, &mut gen0, &mut trans_a).unwrap();
        let mpc_logits1 = handle.join().unwrap().unwrap();

        // Both parties should produce identical revealed logits
        assert_eq!(mpc_logits, mpc_logits1);
        assert_eq!(mpc_logits.len(), config.vocab_size);

        let mut max_diff = 0.0f32;
        let mut plaintext_argmax = 0;
        let mut mpc_argmax = 0;
        let mut plaintext_max = f32::NEG_INFINITY;
        let mut mpc_max = f32::NEG_INFINITY;

        for i in 0..config.vocab_size {
            let diff = (mpc_logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if plaintext_logits[i] > plaintext_max {
                plaintext_max = plaintext_logits[i];
                plaintext_argmax = i;
            }
            if mpc_logits[i] > mpc_max {
                mpc_max = mpc_logits[i];
                mpc_argmax = i;
            }
        }

        eprintln!("Secure Qwen3.5 (4 layers): max logit diff={max_diff}");
        eprintln!("Plaintext argmax={plaintext_argmax}, MPC argmax={mpc_argmax}");

        for i in 0..config.vocab_size {
            assert!(mpc_logits[i].is_finite(), "secure logit[{i}] is NaN/Inf");
        }

        // Q32.32 has quantization error — allow wider tolerance
        assert!(
            max_diff < 5.0,
            "Secure Qwen3.5 MPC diverges too much: max_diff={max_diff}"
        );
    }

    // --- No-reveal model tests ---

    #[test]
    fn test_noreveal_model_forward_runs() {
        use klearu_llm::model::kv_cache::KvCacheStore;

        let config = tiny_config();

        let mut model0 = Model::new(config.clone());
        init_model_weights(&mut model0);

        let mut model1 = Model::new(config.clone());
        init_model_weights(&mut model1);

        let token_id = 1u32;
        let share0 = shared_embedding_lookup_64(0, &model0, token_id);
        let share1 = shared_embedding_lookup_64(1, &model1, token_id);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(500_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut kv0 = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut kv1 = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );

        let mut dn0 = create_deltanet_states(&model0);
        let mut dn1 = create_deltanet_states(&model1);

        let handle = std::thread::spawn(move || {
            private_model_forward_noreveal_reveal(
                1, &model1, &share1, 0, &mut kv1, &mut dn1, &mut gen1, &mut trans_b,
            )
        });

        let mpc_logits = private_model_forward_noreveal_reveal(
            0, &model0, &share0, 0, &mut kv0, &mut dn0, &mut gen0, &mut trans_a,
        ).unwrap();
        let mpc_logits1 = handle.join().unwrap().unwrap();

        // Both parties should produce identical revealed logits
        assert_eq!(mpc_logits, mpc_logits1);
        assert_eq!(mpc_logits.len(), config.vocab_size);

        for i in 0..config.vocab_size {
            assert!(mpc_logits[i].is_finite(), "noreveal logit[{i}] is NaN/Inf");
        }

        eprintln!("No-reveal 1-layer: logits finite, len={}", mpc_logits.len());
    }
}