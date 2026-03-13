//! Shared infrastructure for the 2-server MPC architecture.
//!
//! Server A (Party 0): Proxy facing thin WebSocket clients.
//! Server B (Party 1): Model partner communicating over TCP.
//!
//! Both servers hold the full model weights (public in semi-honest model).
//! DPF-based PIR hides token identities from Server B.
//! The no-reveal MPC protocol protects intermediate computations.

use klearu_dpf::AesPrg;
use klearu_llm::Model;
use klearu_llm::model::attention::AttentionScratch;
use klearu_llm::model::block::AttentionLayer;
use klearu_llm::model::gated_deltanet::DeltaNetStateStore;
use klearu_llm::model::kv_cache::KvCacheStore;
use klearu_mpc::beaver::DummyTripleGen128;
use klearu_mpc::embedding_pir::{
    pir_compute_embedding_share, pir_keygen_and_send, pir_recv_key, q16_to_q32_share,
    quantize_embedding_table,
};
use klearu_mpc::fixed_point::{from_fixed64, to_fixed64};
use klearu_mpc::transport::Transport;
use klearu_mpc::SharedVec64;
use rand::{Rng, SeedableRng};
use klearu_private::private_model::private_model_forward_noreveal;
#[cfg(test)]
use klearu_private::private_model::create_deltanet_states;
use std::io;

/// Pre-quantize a model's embedding table to Q16.16 for DPF-PIR.
pub fn prepare_embedding_table(model: &Model) -> Vec<u32> {
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;

    let mut emb_f32 = Vec::with_capacity(vocab_size * hidden_size);
    for i in 0..vocab_size {
        let row = model.embedding.weights.get_weights(i);
        emb_f32.extend_from_slice(&row[..hidden_size]);
    }

    quantize_embedding_table(&emb_f32, vocab_size, hidden_size)
}

/// Sync pre-quantized Q32.32 weights on all Linear projections in the model.
pub fn sync_all_q32(model: &mut Model) {
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

/// Server A (Party 0) per-token step:
/// 1. Generate DPF keys, send key_1 to Server B
/// 2. Compute embedding share from key_0
/// 3. Run no-reveal MPC forward pass
/// 4. Receive Server B's logit share, reconstruct logits
pub fn server_a_forward_token(
    model: &Model,
    token_id: u32,
    position: usize,
    kv_caches: &mut KvCacheStore,
    deltanet_states: &mut DeltaNetStateStore,
    embedding_q16: &[u32],
    prg: &AesPrg,
    triples: &mut DummyTripleGen128,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;

    // 1. DPF-PIR: generate keys, send key_1 to Server B
    let key_0 = pir_keygen_and_send(prg, token_id, vocab_size, transport)?;

    // 2. Compute embedding share from key_0
    let share_q16 = pir_compute_embedding_share(prg, &key_0, embedding_q16, vocab_size, hidden_size);
    let share_q32 = q16_to_q32_share(&share_q16);
    let embedding_share = SharedVec64(share_q32);

    // 3. No-reveal MPC forward pass (Party 0)
    let logit_share = private_model_forward_noreveal(
        0, model, &embedding_share, position, kv_caches, deltanet_states, triples, transport,
    )?;

    // 4. Exchange logit shares → reconstruct f32 logits
    transport.send_u64_slice(&logit_share.0)?;
    let other = transport.recv_u64_slice(vocab_size)?;

    Ok((0..vocab_size)
        .map(|i| from_fixed64(logit_share.0[i].wrapping_add(other[i])))
        .collect())
}

/// Server B (Party 1) per-token step:
/// 1. Receive DPF key from Server A
/// 2. Compute embedding share from key_1
/// 3. Run no-reveal MPC forward pass
/// 4. Send logit share to Server A
pub fn server_b_forward_token(
    model: &Model,
    position: usize,
    kv_caches: &mut KvCacheStore,
    deltanet_states: &mut DeltaNetStateStore,
    embedding_q16: &[u32],
    prg: &AesPrg,
    triples: &mut DummyTripleGen128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;

    // 1. Receive DPF key from Server A
    let key_1 = pir_recv_key(vocab_size, transport)?;

    // 2. Compute embedding share from key_1
    let share_q16 = pir_compute_embedding_share(prg, &key_1, embedding_q16, vocab_size, hidden_size);
    let share_q32 = q16_to_q32_share(&share_q16);
    let embedding_share = SharedVec64(share_q32);

    // 3. No-reveal MPC forward pass (Party 1)
    let logit_share = private_model_forward_noreveal(
        1, model, &embedding_share, position, kv_caches, deltanet_states, triples, transport,
    )?;

    // 4. Exchange logit shares (Server B sends, then receives Server A's)
    transport.send_u64_slice(&logit_share.0)?;
    let _other = transport.recv_u64_slice(vocab_size)?;

    Ok(())
}

/// Server A (Party 0) forward with pre-computed embedding share.
/// No DPF-PIR, no logit exchange — returns raw logit shares.
pub fn server_a_forward_shared(
    model: &Model,
    share_a: &SharedVec64,
    position: usize,
    kv_caches: &mut KvCacheStore,
    deltanet_states: &mut DeltaNetStateStore,
    triples: &mut DummyTripleGen128,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    private_model_forward_noreveal(
        0, model, share_a, position, kv_caches, deltanet_states, triples, transport,
    )
}

/// Server B (Party 1) forward with pre-computed embedding share.
/// No DPF-PIR, no logit exchange — returns raw logit shares.
pub fn server_b_forward_shared(
    model: &Model,
    share_b: &SharedVec64,
    position: usize,
    kv_caches: &mut KvCacheStore,
    deltanet_states: &mut DeltaNetStateStore,
    triples: &mut DummyTripleGen128,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    private_model_forward_noreveal(
        1, model, share_b, position, kv_caches, deltanet_states, triples, transport,
    )
}

/// Server forward with reveal: exchange Q32.32 shares → plaintext forward → logit shares.
///
/// Avoids the Q32.32 quantization cascade by revealing the embedding between
/// servers, running the forward pass in f32, then creating additive Q32.32 logit
/// shares using a deterministic random mask (so the client can reconstruct).
///
/// Security: both servers see the plaintext embedding and hidden state, but
/// neither sees the plaintext input tokens (only the client knows the token→embedding
/// mapping). Neither server sees the reconstructed logits (only their share).
pub fn server_forward_shared_reveal(
    party: u8,
    model: &Model,
    input_share: &SharedVec64,
    position: usize,
    kv_caches: &mut KvCacheStore,
    deltanet_states: &mut DeltaNetStateStore,
    triple_seed: u64,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let hidden_size = model.config.hidden_size;
    assert_eq!(input_share.len(), hidden_size);

    // 1. Reveal embedding: exchange Q32.32 shares over TCP
    transport.send_u64_slice(&input_share.0)?;
    let other = transport.recv_u64_slice(hidden_size)?;

    // Reconstruct plaintext embedding
    let mut hidden = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        hidden[i] = from_fixed64(input_share.0[i].wrapping_add(other[i]));
    }

    // 2. Plaintext forward pass (both parties compute identical results)
    // Pre-allocate all scratch buffers (reused across layers, avoids per-layer allocation)
    let intermediate_size = model.config.intermediate_size;
    let mut norm_buf = vec![0.0f32; hidden_size];
    let mut attn_out = vec![0.0f32; hidden_size];
    let mut mlp_out = vec![0.0f32; hidden_size];
    let mut gate_buf = vec![0.0f32; intermediate_size];
    let mut up_buf = vec![0.0f32; intermediate_size];

    for layer_idx in 0..model.config.num_layers {
        norm_buf.copy_from_slice(&hidden);
        model.layers[layer_idx].attn_norm.forward(&mut norm_buf);

        match &model.layers[layer_idx].attention {
            AttentionLayer::Standard(attn) => {
                attn.forward_into(
                    &norm_buf,
                    position,
                    &model.rope,
                    kv_caches.layer_mut(layer_idx),
                    &mut attn_out,
                );
            }
            AttentionLayer::GatedDeltaNet(dn) => {
                let state = deltanet_states.layer_mut(layer_idx)
                    .expect("DeltaNet state missing");
                dn.forward_decode_into(&norm_buf, state, &mut attn_out);
            }
        }

        for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
            *h += a;
        }

        norm_buf.copy_from_slice(&hidden);
        model.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

        model.layers[layer_idx].mlp.forward_into(&norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf);

        for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
            *h += m;
        }
    }

    model.final_norm.forward(&mut hidden);

    let vocab_size = model.config.vocab_size;
    let mut logits = vec![0.0f32; vocab_size];
    match &model.lm_head {
        Some(head) => head.forward(&hidden, &mut logits),
        None => model.embedding.lm_head_forward(&hidden, &mut logits),
    };

    // 3. Create logit shares using deterministic random mask.
    // Both servers generate the same mask from triple_seed + position.
    let mask_seed = triple_seed
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(position as u64);
    let mut mask_rng = rand::rngs::StdRng::seed_from_u64(mask_seed);

    let mut logit_shares = Vec::with_capacity(vocab_size);
    for &logit in &logits {
        let q32 = to_fixed64(logit);
        let mask: u64 = mask_rng.gen();
        if party == 0 {
            logit_shares.push(q32.wrapping_sub(mask));
        } else {
            logit_shares.push(mask);
        }
    }

    Ok(SharedVec64(logit_shares))
}

/// Batched prefill: exchange ALL embedding shares in one TCP roundtrip, then run
/// forward passes through all layers for each position without TCP, skipping
/// lm_head for all but the last position.
///
/// This is dramatically faster than calling `server_forward_shared_reveal` per token:
/// - 1 TCP roundtrip instead of N
/// - No lm_head computation for N-1 tokens (saves ~155M FLOPs per skipped token)
/// - Forward passes run without TCP synchronization between them
pub fn server_forward_shared_reveal_prefill(
    party: u8,
    model: &Model,
    input_shares: &[SharedVec64],
    kv_caches: &mut KvCacheStore,
    deltanet_states: &mut DeltaNetStateStore,
    triple_seed: u64,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let n = input_shares.len();
    assert!(n > 0, "Empty prefill");
    let hidden_size = model.config.hidden_size;

    // 1. Exchange all embedding shares in one TCP message
    let mut all_shares = Vec::with_capacity(n * hidden_size);
    for share in input_shares {
        assert_eq!(share.len(), hidden_size);
        all_shares.extend_from_slice(&share.0);
    }
    transport.send_u64_slice(&all_shares)?;
    let all_other = transport.recv_u64_slice(n * hidden_size)?;

    // 2. Reconstruct all plaintext embeddings
    let mut embeddings = vec![vec![0.0f32; hidden_size]; n];
    for t in 0..n {
        for i in 0..hidden_size {
            let idx = t * hidden_size + i;
            embeddings[t][i] = from_fixed64(all_shares[idx].wrapping_add(all_other[idx]));
        }
    }

    // 3. Run forward pass for each position (building KV cache)
    // Pre-allocate all scratch buffers (reused across layers and positions)
    let intermediate_size = model.config.intermediate_size;
    let mut norm_buf = vec![0.0f32; hidden_size];
    let mut attn_out = vec![0.0f32; hidden_size];
    let mut mlp_out = vec![0.0f32; hidden_size];
    let mut gate_buf = vec![0.0f32; intermediate_size];
    let mut up_buf = vec![0.0f32; intermediate_size];

    for pos in 0..n {
        let hidden = &mut embeddings[pos];

        for layer_idx in 0..model.config.num_layers {
            norm_buf.copy_from_slice(hidden);
            model.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            match &model.layers[layer_idx].attention {
                AttentionLayer::Standard(attn) => {
                    attn.forward_into(
                        &norm_buf,
                        pos,
                        &model.rope,
                        kv_caches.layer_mut(layer_idx),
                        &mut attn_out,
                    );
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    let state = deltanet_states
                        .layer_mut(layer_idx)
                        .expect("DeltaNet state missing");
                    dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                }
            }

            for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            norm_buf.copy_from_slice(hidden);
            model.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

            model.layers[layer_idx].mlp.forward_into(&norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf);

            for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
                *h += m;
            }
        }
    }

    // 4. Only compute final_norm + lm_head + logit shares for the LAST token
    let last_hidden = &mut embeddings[n - 1];
    model.final_norm.forward(last_hidden);

    let vocab_size = model.config.vocab_size;
    let mut logits = vec![0.0f32; vocab_size];
    match &model.lm_head {
        Some(head) => head.forward(last_hidden, &mut logits),
        None => model.embedding.lm_head_forward(last_hidden, &mut logits),
    };

    // Create logit shares using deterministic random mask (same as decode)
    let last_pos = n - 1;
    let mask_seed = triple_seed
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(last_pos as u64);
    let mut mask_rng = rand::rngs::StdRng::seed_from_u64(mask_seed);

    let mut logit_shares = Vec::with_capacity(vocab_size);
    for &logit in &logits {
        let q32 = to_fixed64(logit);
        let mask: u64 = mask_rng.gen();
        if party == 0 {
            logit_shares.push(q32.wrapping_sub(mask));
        } else {
            logit_shares.push(mask);
        }
    }

    Ok(SharedVec64(logit_shares))
}

/// Pre-allocated buffers for the decode loop.
/// Created once per session, reused across all decode tokens.
pub struct DecodeBuffers {
    pub hidden: Vec<f32>,
    pub norm_buf: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub mlp_out: Vec<f32>,
    pub gate_buf: Vec<f32>,
    pub up_buf: Vec<f32>,
    pub logits: Vec<f32>,
    pub logit_shares_u32: Vec<u32>,
    pub attn_scratch: AttentionScratch,
    pub recv_q16: Vec<u32>,
}

impl DecodeBuffers {
    pub fn new(model: &Model) -> Self {
        let config = &model.config;
        let has_gated = model.layers.iter().any(|l| {
            matches!(&l.attention, AttentionLayer::Standard(a) if a.output_gate)
        });
        Self {
            hidden: vec![0.0; config.hidden_size],
            norm_buf: vec![0.0; config.hidden_size],
            attn_out: vec![0.0; config.hidden_size],
            mlp_out: vec![0.0; config.hidden_size],
            gate_buf: vec![0.0; config.intermediate_size],
            up_buf: vec![0.0; config.intermediate_size],
            logits: vec![0.0; config.vocab_size],
            logit_shares_u32: vec![0u32; config.vocab_size],
            attn_scratch: AttentionScratch::new(
                config.num_heads, config.num_kv_heads, config.head_dim,
                config.max_seq_len, has_gated,
            ),
            recv_q16: vec![0u32; config.hidden_size],
        }
    }

    /// Get logit share bytes as a zero-copy slice (LE platforms).
    #[cfg(target_endian = "little")]
    pub fn logit_share_bytes(&self, vocab_size: usize) -> &[u8] {
        bytemuck::cast_slice(&self.logit_shares_u32[..vocab_size])
    }

    #[cfg(not(target_endian = "little"))]
    pub fn logit_share_bytes(&self, vocab_size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(vocab_size * 4);
        for &v in &self.logit_shares_u32[..vocab_size] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data
    }
}

/// Optimized server forward with reveal: works directly in Q16.16, no Q32.32 intermediate.
///
/// Compared to `server_forward_shared_reveal`:
/// - Takes Q16.16 shares directly (no q16_to_q32 conversion)
/// - Halves TCP payload (hidden_size × 4 bytes vs × 8 bytes)
/// - Uses pre-allocated buffers (no per-call heap allocation)
/// - Uses attention scratch buffers (no per-layer allocation)
/// - Writes Q16.16 logit shares into `buffers.logit_shares_u32`
pub fn server_forward_shared_reveal_q16(
    party: u8,
    model: &Model,
    input_share_q16: &[u32],
    position: usize,
    kv_caches: &mut KvCacheStore,
    deltanet_states: &mut DeltaNetStateStore,
    triple_seed: u64,
    transport: &mut impl Transport,
    buffers: &mut DecodeBuffers,
) -> io::Result<()> {
    let hidden_size = model.config.hidden_size;
    assert_eq!(input_share_q16.len(), hidden_size);

    // 1. Reveal embedding: exchange Q16.16 shares directly
    transport.send_u32_slice(input_share_q16)?;
    transport.recv_u32_slice_into(&mut buffers.recv_q16[..hidden_size])?;

    // Reconstruct plaintext embedding from Q16.16
    for i in 0..hidden_size {
        let sum = input_share_q16[i].wrapping_add(buffers.recv_q16[i]);
        buffers.hidden[i] = (sum as i32) as f32 / 65536.0;
    }

    // 2. Plaintext forward pass using pre-allocated buffers
    for layer_idx in 0..model.config.num_layers {
        buffers.norm_buf[..hidden_size].copy_from_slice(&buffers.hidden[..hidden_size]);
        model.layers[layer_idx].attn_norm.forward(&mut buffers.norm_buf[..hidden_size]);

        match &model.layers[layer_idx].attention {
            AttentionLayer::Standard(attn) => {
                attn.forward_into_buffered(
                    &buffers.norm_buf,
                    position,
                    &model.rope,
                    kv_caches.layer_mut(layer_idx),
                    &mut buffers.attn_out,
                    &mut buffers.attn_scratch,
                );
            }
            AttentionLayer::GatedDeltaNet(dn) => {
                let state = deltanet_states.layer_mut(layer_idx)
                    .expect("DeltaNet state missing");
                dn.forward_decode_into(&buffers.norm_buf, state, &mut buffers.attn_out);
            }
        }

        for i in 0..hidden_size {
            buffers.hidden[i] += buffers.attn_out[i];
        }

        buffers.norm_buf[..hidden_size].copy_from_slice(&buffers.hidden[..hidden_size]);
        model.layers[layer_idx].mlp_norm.forward(&mut buffers.norm_buf[..hidden_size]);

        model.layers[layer_idx].mlp.forward_into(
            &buffers.norm_buf, &mut buffers.mlp_out, &mut buffers.gate_buf, &mut buffers.up_buf,
        );

        for i in 0..hidden_size {
            buffers.hidden[i] += buffers.mlp_out[i];
        }
    }

    model.final_norm.forward(&mut buffers.hidden[..hidden_size]);

    let vocab_size = model.config.vocab_size;
    match &model.lm_head {
        Some(head) => head.forward(&buffers.hidden, &mut buffers.logits[..vocab_size]),
        None => model.embedding.lm_head_forward(&buffers.hidden, &mut buffers.logits[..vocab_size]),
    };

    // 3. Create Q16.16 logit shares directly
    let mask_seed = triple_seed
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(position as u64);
    let mut mask_rng = rand::rngs::StdRng::seed_from_u64(mask_seed);

    for i in 0..vocab_size {
        let q16 = ((buffers.logits[i] * 65536.0).round() as i32) as u32;
        let mask: u32 = mask_rng.gen();
        buffers.logit_shares_u32[i] = if party == 0 {
            q16.wrapping_sub(mask)
        } else {
            mask
        };
    }

    Ok(())
}

/// Batched prefill with Q16.16 direct path: exchange all Q16.16 shares in one
/// TCP roundtrip, run forward passes for all positions, write Q16.16 logit
/// shares to `buffers.logit_shares_u32`.
///
/// Compared to `server_forward_shared_reveal_prefill`:
/// - Takes Q16.16 shares directly (no q16_to_q32 conversion)
/// - Halves TCP payload (n × hidden_size × 4 bytes vs × 8 bytes)
/// - Uses DecodeBuffers scratch for per-layer computation
/// - Writes Q16.16 logit shares directly (no Q32.32 intermediate)
pub fn server_forward_shared_reveal_prefill_q16(
    party: u8,
    model: &Model,
    input_shares_q16: &[Vec<u32>],
    kv_caches: &mut KvCacheStore,
    deltanet_states: &mut DeltaNetStateStore,
    triple_seed: u64,
    transport: &mut impl Transport,
    buffers: &mut DecodeBuffers,
) -> io::Result<()> {
    let n = input_shares_q16.len();
    assert!(n > 0, "Empty prefill");
    let hidden_size = model.config.hidden_size;

    // 1. Exchange all Q16.16 embedding shares in one TCP message
    let mut all_shares = Vec::with_capacity(n * hidden_size);
    for share in input_shares_q16 {
        assert_eq!(share.len(), hidden_size);
        all_shares.extend_from_slice(share);
    }
    transport.send_u32_slice(&all_shares)?;

    let mut all_other = vec![0u32; n * hidden_size];
    transport.recv_u32_slice_into(&mut all_other)?;

    // 2. Reconstruct all plaintext embeddings from Q16.16
    let mut embeddings = vec![vec![0.0f32; hidden_size]; n];
    for t in 0..n {
        for i in 0..hidden_size {
            let idx = t * hidden_size + i;
            let sum = all_shares[idx].wrapping_add(all_other[idx]);
            embeddings[t][i] = (sum as i32) as f32 / 65536.0;
        }
    }

    // 3. Run forward pass for each position (building KV cache)
    for pos in 0..n {
        let hidden = &mut embeddings[pos];

        for layer_idx in 0..model.config.num_layers {
            buffers.norm_buf[..hidden_size].copy_from_slice(hidden);
            model.layers[layer_idx].attn_norm.forward(&mut buffers.norm_buf[..hidden_size]);

            match &model.layers[layer_idx].attention {
                AttentionLayer::Standard(attn) => {
                    attn.forward_into_buffered(
                        &buffers.norm_buf,
                        pos,
                        &model.rope,
                        kv_caches.layer_mut(layer_idx),
                        &mut buffers.attn_out,
                        &mut buffers.attn_scratch,
                    );
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    let state = deltanet_states
                        .layer_mut(layer_idx)
                        .expect("DeltaNet state missing");
                    dn.forward_decode_into(&buffers.norm_buf, state, &mut buffers.attn_out);
                }
            }

            for (h, &a) in hidden.iter_mut().zip(buffers.attn_out.iter()) {
                *h += a;
            }

            buffers.norm_buf[..hidden_size].copy_from_slice(hidden);
            model.layers[layer_idx].mlp_norm.forward(&mut buffers.norm_buf[..hidden_size]);

            model.layers[layer_idx].mlp.forward_into(
                &buffers.norm_buf, &mut buffers.mlp_out, &mut buffers.gate_buf, &mut buffers.up_buf,
            );

            for (h, &m) in hidden.iter_mut().zip(buffers.mlp_out.iter()) {
                *h += m;
            }
        }
    }

    // 4. Only compute final_norm + lm_head + logit shares for the LAST token
    let last_hidden = &mut embeddings[n - 1];
    model.final_norm.forward(last_hidden);

    let vocab_size = model.config.vocab_size;
    match &model.lm_head {
        Some(head) => head.forward(last_hidden, &mut buffers.logits[..vocab_size]),
        None => model.embedding.lm_head_forward(last_hidden, &mut buffers.logits[..vocab_size]),
    };

    // Create Q16.16 logit shares directly
    let last_pos = n - 1;
    let mask_seed = triple_seed
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(last_pos as u64);
    let mut mask_rng = rand::rngs::StdRng::seed_from_u64(mask_seed);

    for i in 0..vocab_size {
        let q16 = ((buffers.logits[i] * 65536.0).round() as i32) as u32;
        let mask: u32 = mask_rng.gen();
        buffers.logit_shares_u32[i] = if party == 0 {
            q16.wrapping_sub(mask)
        } else {
            mask
        };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_llm::config::LlmConfig;
    use klearu_llm::model::block::AttentionLayer;
    use klearu_private::tcp_transport::tcp_transport_pair;
    use rand::Rng;

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

    fn init_model(config: &LlmConfig) -> Model {
        let mut model = Model::new(config.clone());

        // Set norm weights to 1.0
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

        // Set deterministic weights
        let vs = config.vocab_size;
        for i in 0..vs {
            let row = model.embedding.weights.get_weights_mut(i);
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((i * 7 + j * 3 + 1) % 17) as f32 * 0.02 - 0.15;
            }
        }

        for layer in &mut model.layers {
            if let AttentionLayer::Standard(attn) = &mut layer.attention {
                for (proj_idx, proj) in [
                    &mut attn.q_proj, &mut attn.k_proj,
                    &mut attn.v_proj, &mut attn.o_proj,
                ].into_iter().enumerate() {
                    for i in 0..proj.out_features() {
                        let row = proj.weights.get_weights_mut(i);
                        for (j, v) in row.iter_mut().enumerate() {
                            *v = ((i * 5 + j * 11 + proj_idx * 3 + 1) % 19) as f32 * 0.01 - 0.09;
                        }
                    }
                }
            }

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

        sync_all_q32(&mut model);
        model
    }

    /// End-to-end test: Server A generates DPF keys for a token, both servers
    /// run the no-reveal forward pass, and Server A reconstructs logits.
    #[test]
    fn test_two_party_end_to_end() {
        let config = tiny_config();
        let model_a = init_model(&config);
        let model_b = init_model(&config);

        let embedding_q16_a = prepare_embedding_table(&model_a);
        let embedding_q16_b = prepare_embedding_table(&model_b);

        let prg_a = AesPrg::new(&[42u8; 16]);
        let prg_b = AesPrg::new(&[42u8; 16]);
        let triple_seed = 12345u64;
        let mut triples_a = DummyTripleGen128::new(0, triple_seed);
        let mut triples_b = DummyTripleGen128::new(1, triple_seed);

        let mut kv_a = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut kv_b = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );

        let mut dn_a = create_deltanet_states(&model_a);
        let mut dn_b = create_deltanet_states(&model_b);

        let (mut trans_a, mut trans_b) = tcp_transport_pair().expect("tcp_transport_pair failed");

        let token_id = 3u32;

        let handle = std::thread::spawn(move || {
            server_b_forward_token(
                &model_b, 0, &mut kv_b, &mut dn_b, &embedding_q16_b,
                &prg_b, &mut triples_b, &mut trans_b,
            )
        });

        let logits = server_a_forward_token(
            &model_a, token_id, 0, &mut kv_a, &mut dn_a, &embedding_q16_a,
            &prg_a, &mut triples_a, &mut trans_a,
        ).expect("Server A forward failed");

        handle.join().unwrap().expect("Server B forward failed");

        // Verify logits are finite and have correct length
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
        }

        eprintln!("Two-party e2e logits (first 8): {:?}", &logits[..8.min(logits.len())]);
    }

    /// Multi-token test: prefill 2 tokens, verify both complete without error.
    #[test]
    fn test_two_party_multi_token() {
        let config = tiny_config();
        let model_a = init_model(&config);
        let model_b = init_model(&config);

        let embedding_q16_a = prepare_embedding_table(&model_a);
        let embedding_q16_b = prepare_embedding_table(&model_b);

        let prg_a = AesPrg::new(&[42u8; 16]);
        let prg_b = AesPrg::new(&[42u8; 16]);
        let triple_seed = 99999u64;
        let mut triples_a = DummyTripleGen128::new(0, triple_seed);
        let mut triples_b = DummyTripleGen128::new(1, triple_seed);

        let mut kv_a = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut kv_b = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );

        let mut dn_a = create_deltanet_states(&model_a);
        let mut dn_b = create_deltanet_states(&model_b);

        let (mut trans_a, mut trans_b) = tcp_transport_pair().expect("tcp_transport_pair failed");

        let tokens = [1u32, 5, 3];

        let handle = std::thread::spawn(move || {
            for pos in 0..tokens.len() {
                server_b_forward_token(
                    &model_b, pos, &mut kv_b, &mut dn_b, &embedding_q16_b,
                    &prg_b, &mut triples_b, &mut trans_b,
                ).expect(&format!("Server B failed at pos {pos}"));
            }
        });

        let mut last_logits = Vec::new();
        for (pos, &token_id) in tokens.iter().enumerate() {
            last_logits = server_a_forward_token(
                &model_a, token_id, pos, &mut kv_a, &mut dn_a, &embedding_q16_a,
                &prg_a, &mut triples_a, &mut trans_a,
            ).expect(&format!("Server A failed at pos {pos}"));
        }

        handle.join().unwrap();

        assert_eq!(last_logits.len(), config.vocab_size);
        for (i, &v) in last_logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
        }

        eprintln!("Multi-token e2e logits: {:?}", &last_logits[..8.min(last_logits.len())]);
    }

    /// Test share-based forward: client creates shares, both servers process,
    /// client reconstructs logits from shares.
    #[test]
    fn test_shared_forward_reconstruction() {
        use klearu_mpc::fixed_point::{to_fixed64, from_fixed64};

        let config = tiny_config();
        let model_a = init_model(&config);
        let model_b = init_model(&config);

        let triple_seed = 54321u64;
        let mut triples_a = DummyTripleGen128::new(0, triple_seed);
        let mut triples_b = DummyTripleGen128::new(1, triple_seed);

        let mut kv_a = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut kv_b = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );

        let mut dn_a = create_deltanet_states(&model_a);
        let mut dn_b = create_deltanet_states(&model_b);

        let (mut trans_a, mut trans_b) = tcp_transport_pair().expect("tcp_transport_pair failed");

        // Simulate client creating shares from embedding row for token 3
        let token_id = 3usize;
        let hidden_size = config.hidden_size;
        let embedding_row: Vec<f32> = (0..hidden_size)
            .map(|j| model_a.embedding.weights.get_weights(token_id)[j])
            .collect();

        // Create Q32.32 shares: share_a + share_b = embedding in Q32.32
        let mut share_a_vals = Vec::with_capacity(hidden_size);
        let mut share_b_vals = Vec::with_capacity(hidden_size);
        let mut rng = rand::thread_rng();
        for &v in &embedding_row {
            let q32 = to_fixed64(v);
            let rb: u64 = rng.gen();
            share_a_vals.push(q32.wrapping_sub(rb));
            share_b_vals.push(rb);
        }
        let share_a = SharedVec64(share_a_vals);
        let share_b = SharedVec64(share_b_vals);

        let handle = std::thread::spawn(move || {
            server_b_forward_shared(
                &model_b, &share_b, 0, &mut kv_b, &mut dn_b,
                &mut triples_b, &mut trans_b,
            )
        });

        let logit_share_a = server_a_forward_shared(
            &model_a, &share_a, 0, &mut kv_a, &mut dn_a,
            &mut triples_a, &mut trans_a,
        ).expect("Server A shared forward failed");

        let logit_share_b = handle.join().unwrap().expect("Server B shared forward failed");

        // Reconstruct logits
        assert_eq!(logit_share_a.len(), config.vocab_size);
        assert_eq!(logit_share_b.len(), config.vocab_size);

        let logits: Vec<f32> = (0..config.vocab_size)
            .map(|i| from_fixed64(logit_share_a.0[i].wrapping_add(logit_share_b.0[i])))
            .collect();

        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
        }

        eprintln!("Shared forward logits (first 8): {:?}", &logits[..8.min(logits.len())]);
    }

    /// Test reveal-based forward: client creates shares, both servers reveal+plaintext,
    /// client reconstructs logits from shares. Verifies match against plaintext.
    #[test]
    fn test_shared_forward_reveal_matches_plaintext() {
        use klearu_mpc::fixed_point::{to_fixed64, from_fixed64};

        let config = tiny_config();
        let model_a = init_model(&config);
        let model_b = init_model(&config);
        let mut model_plain = init_model(&config);

        let triple_seed = 77777u64;

        let mut kv_a = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut kv_b = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );

        let mut dn_a = create_deltanet_states(&model_a);
        let mut dn_b = create_deltanet_states(&model_b);

        let (mut trans_a, mut trans_b) = tcp_transport_pair().expect("tcp_transport_pair failed");

        // Get plaintext logits
        let token_id = 3u32;
        model_plain.reset_kv_caches();
        let plaintext_logits = model_plain.forward_decode(token_id, 0);

        // Create Q32.32 shares from embedding row
        let hidden_size = config.hidden_size;
        let embedding_row: Vec<f32> = (0..hidden_size)
            .map(|j| model_a.embedding.weights.get_weights(token_id as usize)[j])
            .collect();

        let mut share_a_vals = Vec::with_capacity(hidden_size);
        let mut share_b_vals = Vec::with_capacity(hidden_size);
        let mut rng = rand::thread_rng();
        for &v in &embedding_row {
            let q32 = to_fixed64(v);
            let rb: u64 = rng.gen();
            share_a_vals.push(q32.wrapping_sub(rb));
            share_b_vals.push(rb);
        }
        let share_a = SharedVec64(share_a_vals);
        let share_b = SharedVec64(share_b_vals);

        let handle = std::thread::spawn(move || {
            server_forward_shared_reveal(
                1, &model_b, &share_b, 0, &mut kv_b, &mut dn_b,
                triple_seed, &mut trans_b,
            )
        });

        let logit_share_a = server_forward_shared_reveal(
            0, &model_a, &share_a, 0, &mut kv_a, &mut dn_a,
            triple_seed, &mut trans_a,
        ).expect("Party 0 reveal forward failed");

        let logit_share_b = handle.join().unwrap().expect("Party 1 reveal forward failed");

        // Reconstruct logits from shares
        assert_eq!(logit_share_a.len(), config.vocab_size);
        assert_eq!(logit_share_b.len(), config.vocab_size);

        let logits: Vec<f32> = (0..config.vocab_size)
            .map(|i| from_fixed64(logit_share_a.0[i].wrapping_add(logit_share_b.0[i])))
            .collect();

        let mut max_diff = 0.0f32;
        let mut plaintext_argmax = 0;
        let mut reveal_argmax = 0;
        let mut plaintext_max = f32::NEG_INFINITY;
        let mut reveal_max = f32::NEG_INFINITY;

        for i in 0..config.vocab_size {
            assert!(logits[i].is_finite(), "logit[{i}] is not finite: {}", logits[i]);
            let diff = (logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if plaintext_logits[i] > plaintext_max {
                plaintext_max = plaintext_logits[i];
                plaintext_argmax = i;
            }
            if logits[i] > reveal_max {
                reveal_max = logits[i];
                reveal_argmax = i;
            }
        }

        eprintln!("Reveal forward: max_diff={max_diff}, plain_argmax={plaintext_argmax}, reveal_argmax={reveal_argmax}");
        eprintln!("Reveal logits (first 8): {:?}", &logits[..8.min(logits.len())]);
        eprintln!("Plain logits  (first 8): {:?}", &plaintext_logits[..8.min(plaintext_logits.len())]);

        // Should match closely (only Q32.32 roundtrip error on embedding+logits)
        assert!(max_diff < 0.01, "Reveal forward diverges from plaintext: max_diff={max_diff}");
        assert_eq!(plaintext_argmax, reveal_argmax, "Argmax mismatch");
    }

    /// Test batched prefill: process multiple tokens at once, verify it matches
    /// sequential per-token reveal forward.
    #[test]
    fn test_batched_prefill_matches_sequential() {
        use klearu_mpc::fixed_point::{to_fixed64, from_fixed64};

        let config = tiny_config();
        let model_a_batch = init_model(&config);
        let model_b_batch = init_model(&config);
        let model_a_seq = init_model(&config);
        let model_b_seq = init_model(&config);

        let triple_seed = 88888u64;
        let tokens = [1u32, 5, 3];
        let hidden_size = config.hidden_size;

        // Create embedding shares for each token
        let mut rng = rand::thread_rng();
        let mut shares_a = Vec::new();
        let mut shares_b = Vec::new();
        for &token_id in &tokens {
            let embedding_row: Vec<f32> = (0..hidden_size)
                .map(|j| model_a_batch.embedding.weights.get_weights(token_id as usize)[j])
                .collect();
            let mut sa = Vec::with_capacity(hidden_size);
            let mut sb = Vec::with_capacity(hidden_size);
            for &v in &embedding_row {
                let q32 = to_fixed64(v);
                let rb: u64 = rng.gen();
                sa.push(q32.wrapping_sub(rb));
                sb.push(rb);
            }
            shares_a.push(SharedVec64(sa));
            shares_b.push(SharedVec64(sb));
        }

        let shares_b_batch = shares_b.clone();
        let shares_a_seq = shares_a.clone();
        let shares_b_seq = shares_b;

        // --- Sequential per-token reveal ---
        let (mut trans_a_seq, mut trans_b_seq) = tcp_transport_pair().expect("tcp pair");
        let mut kv_a_seq = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut kv_b_seq = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut dn_a_seq = create_deltanet_states(&model_a_seq);
        let mut dn_b_seq = create_deltanet_states(&model_b_seq);

        let handle_seq = std::thread::spawn(move || {
            let mut last = None;
            for (pos, share) in shares_b_seq.into_iter().enumerate() {
                last = Some(server_forward_shared_reveal(
                    1, &model_b_seq, &share, pos, &mut kv_b_seq, &mut dn_b_seq,
                    triple_seed, &mut trans_b_seq,
                ).unwrap());
            }
            last.unwrap()
        });

        let mut seq_last_a = None;
        for (pos, share) in shares_a_seq.into_iter().enumerate() {
            seq_last_a = Some(server_forward_shared_reveal(
                0, &model_a_seq, &share, pos, &mut kv_a_seq, &mut dn_a_seq,
                triple_seed, &mut trans_a_seq,
            ).unwrap());
        }
        let seq_share_a = seq_last_a.unwrap();
        let seq_share_b = handle_seq.join().unwrap();

        let seq_logits: Vec<f32> = (0..config.vocab_size)
            .map(|i| from_fixed64(seq_share_a.0[i].wrapping_add(seq_share_b.0[i])))
            .collect();

        // --- Batched prefill ---
        let (mut trans_a_bat, mut trans_b_bat) = tcp_transport_pair().expect("tcp pair");
        let mut kv_a_bat = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut kv_b_bat = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut dn_a_bat = create_deltanet_states(&model_a_batch);
        let mut dn_b_bat = create_deltanet_states(&model_b_batch);

        let handle_bat = std::thread::spawn(move || {
            server_forward_shared_reveal_prefill(
                1, &model_b_batch, &shares_b_batch, &mut kv_b_bat, &mut dn_b_bat,
                triple_seed, &mut trans_b_bat,
            ).unwrap()
        });

        let bat_share_a = server_forward_shared_reveal_prefill(
            0, &model_a_batch, &shares_a, &mut kv_a_bat, &mut dn_a_bat,
            triple_seed, &mut trans_a_bat,
        ).unwrap();
        let bat_share_b = handle_bat.join().unwrap();

        let bat_logits: Vec<f32> = (0..config.vocab_size)
            .map(|i| from_fixed64(bat_share_a.0[i].wrapping_add(bat_share_b.0[i])))
            .collect();

        // Compare
        let mut max_diff = 0.0f32;
        for i in 0..config.vocab_size {
            assert!(bat_logits[i].is_finite(), "bat_logit[{i}] NaN");
            let diff = (bat_logits[i] - seq_logits[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }

        eprintln!("Batched vs sequential: max_diff={max_diff}");
        eprintln!("Seq logits (first 8): {:?}", &seq_logits[..8.min(seq_logits.len())]);
        eprintln!("Bat logits (first 8): {:?}", &bat_logits[..8.min(bat_logits.len())]);

        // The sequential version computes lm_head for all tokens but only keeps the last one's logit mask.
        // For position 2 (last token), the mask_seed should be the same.
        // Logits should be very close (both are reveal-based, same model, same shares).
        assert!(max_diff < 0.001, "Batched prefill diverges from sequential: max_diff={max_diff}");

        // Argmax should match
        let seq_argmax = seq_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let bat_argmax = bat_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(seq_argmax, bat_argmax, "Argmax mismatch: seq={seq_argmax}, bat={bat_argmax}");
    }

    /// Test Q16.16 reveal forward matches plaintext (same as test_shared_forward_reveal_matches_plaintext
    /// but using the optimized Q16.16 path with pre-allocated buffers).
    #[test]
    fn test_q16_reveal_matches_plaintext() {
        use klearu_mpc::fixed_point::to_fixed;

        let config = tiny_config();
        let model_a = init_model(&config);
        let model_b = init_model(&config);
        let mut model_plain = init_model(&config);

        let triple_seed = 77777u64;
        let hidden_size = config.hidden_size;

        let (mut trans_a, mut trans_b) = tcp_transport_pair().expect("tcp pair");

        // Get plaintext logits
        let token_id = 3u32;
        model_plain.reset_kv_caches();
        let plaintext_logits = model_plain.forward_decode(token_id, 0);

        // Create Q16.16 embedding shares
        let embedding_row: Vec<f32> = (0..hidden_size)
            .map(|j| model_a.embedding.weights.get_weights(token_id as usize)[j])
            .collect();

        let mut share_a_q16 = Vec::with_capacity(hidden_size);
        let mut share_b_q16 = Vec::with_capacity(hidden_size);
        let mut rng = rand::thread_rng();
        for &v in &embedding_row {
            let q16 = to_fixed(v);
            let rb: u32 = rng.gen();
            share_a_q16.push(q16.wrapping_sub(rb));
            share_b_q16.push(rb);
        }

        let mut kv_a = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut kv_b = KvCacheStore::new(
            config.num_layers, config.num_kv_heads, config.max_seq_len, config.head_dim,
        );
        let mut dn_a = create_deltanet_states(&model_a);
        let mut dn_b = create_deltanet_states(&model_b);

        let mut buffers_a = DecodeBuffers::new(&model_a);
        let mut buffers_b = DecodeBuffers::new(&model_b);

        let handle = std::thread::spawn(move || {
            server_forward_shared_reveal_q16(
                1, &model_b, &share_b_q16, 0, &mut kv_b, &mut dn_b,
                triple_seed, &mut trans_b, &mut buffers_b,
            ).expect("Party 1 Q16 forward failed");
            buffers_b.logit_shares_u32[..config.vocab_size].to_vec()
        });

        server_forward_shared_reveal_q16(
            0, &model_a, &share_a_q16, 0, &mut kv_a, &mut dn_a,
            triple_seed, &mut trans_a, &mut buffers_a,
        ).expect("Party 0 Q16 forward failed");

        let shares_b = handle.join().unwrap();

        // Reconstruct Q16.16 logits
        let logits: Vec<f32> = (0..config.vocab_size)
            .map(|i| {
                let sum = buffers_a.logit_shares_u32[i].wrapping_add(shares_b[i]);
                (sum as i32) as f32 / 65536.0
            })
            .collect();

        let mut max_diff = 0.0f32;
        let mut plaintext_argmax = 0;
        let mut q16_argmax = 0;
        let mut plaintext_max = f32::NEG_INFINITY;
        let mut q16_max = f32::NEG_INFINITY;

        for i in 0..config.vocab_size {
            assert!(logits[i].is_finite(), "logit[{i}] is not finite");
            let diff = (logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if plaintext_logits[i] > plaintext_max {
                plaintext_max = plaintext_logits[i];
                plaintext_argmax = i;
            }
            if logits[i] > q16_max {
                q16_max = logits[i];
                q16_argmax = i;
            }
        }

        eprintln!("Q16 reveal: max_diff={max_diff}, plain_argmax={plaintext_argmax}, q16_argmax={q16_argmax}");
        eprintln!("Q16 logits (first 8): {:?}", &logits[..8.min(logits.len())]);
        eprintln!("Plain logits  (first 8): {:?}", &plaintext_logits[..8.min(plaintext_logits.len())]);

        // Q16.16 has slightly more error than Q32.32 but should still be close
        assert!(max_diff < 0.02, "Q16 reveal diverges from plaintext: max_diff={max_diff}");
        assert_eq!(plaintext_argmax, q16_argmax, "Argmax mismatch");
    }
}
