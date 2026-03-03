//! Private (2PC) evaluation of a full transformer block.
//!
//! Composes private attention and MLP with shared RMSNorm
//! and residual connections to evaluate one transformer layer under
//! secret sharing.
//!
//! Flow: attn_norm → attention → residual → mlp_norm → MLP → residual
//!
//! GatedDeltaNet layers (lower security): reveal-and-compute on normed hidden state.
//! GatedDeltaNet layers (secure): MPC projections, only reveal low-dim outputs.

use klearu_llm::model::block::{AttentionLayer, TransformerBlock};
use klearu_llm::model::gated_deltanet::DeltaNetState;
use klearu_llm::model::kv_cache::KvCache;
use klearu_llm::model::rope::RotaryEmbedding;
use klearu_mpc::beaver::{TripleGenerator, TripleGenerator128};
use klearu_mpc::fixed_point::{from_fixed, to_fixed};
use klearu_mpc::normalization::{rmsnorm_shared, rmsnorm_shared_64};
use klearu_mpc::transport::Transport;
use klearu_mpc::{SharedVec, SharedVec64};
use std::io;

use crate::private_attention::{private_attention_forward, private_attention_forward_secure, private_deltanet_forward_secure};
use crate::private_mlp::{private_dense_mlp_forward, private_dense_mlp_forward_secure, private_sparse_mlp_forward};

/// Compute effective RmsNorm weights, accounting for one_plus_weight variant.
pub(crate) fn effective_norm_weights(norm: &klearu_llm::model::rms_norm::RmsNorm) -> Vec<f32> {
    if norm.is_one_plus_weight() {
        norm.weight.iter().map(|&w| 1.0 + w).collect()
    } else {
        norm.weight.clone()
    }
}

/// Reveal f32 shares: exchange f32 bit representations, reconstruct plaintext.
fn reveal_f32_shares(
    my_f32: &[f32],
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let bits: Vec<u32> = my_f32.iter().map(|&v| v.to_bits()).collect();
    transport.send_u32_slice(&bits)?;
    let other = transport.recv_u32_slice(bits.len())?;
    Ok(my_f32
        .iter()
        .zip(other.iter())
        .map(|(&my, &ob)| my + f32::from_bits(ob))
        .collect())
}

/// Evaluate one private transformer block (Q16.16).
pub fn private_block_forward(
    party: u8,
    block: &TransformerBlock,
    hidden_share: &mut SharedVec,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    deltanet_state: Option<&mut DeltaNetState>,
    triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<()> {
    // 1. Pre-attention RMSNorm
    let mut normed = hidden_share.clone();
    let attn_weights = effective_norm_weights(&block.attn_norm);
    rmsnorm_shared(
        party,
        &mut normed,
        &attn_weights,
        block.attn_norm.eps(),
        transport,
    )?;

    // 2. Attention
    let attn_out = match &block.attention {
        AttentionLayer::Standard(attn) => {
            private_attention_forward(
                party, attn, &normed, position, rope, kv_cache, transport,
            )?
        }
        AttentionLayer::GatedDeltaNet(dn) => {
            // Reveal-and-compute: reveal normed hidden state, run DeltaNet in plaintext
            let normed_f32: Vec<f32> = normed.0.iter().map(|&v| from_fixed(v)).collect();
            let normed_plain = reveal_f32_shares(&normed_f32, transport)?;

            let state = deltanet_state
                .expect("DeltaNet state required for GatedDeltaNet layer");
            let attn_out_f32 = dn.forward_decode(&normed_plain, state);

            // Re-share: party 0 = full value, party 1 = zeros
            if party == 0 {
                SharedVec(attn_out_f32.iter().map(|&v| to_fixed(v)).collect())
            } else {
                SharedVec(vec![0u32; attn_out_f32.len()])
            }
        }
    };

    // 3. Residual
    *hidden_share = hidden_share.add(&attn_out);

    // 4. Pre-MLP RMSNorm
    let mut normed_mlp = hidden_share.clone();
    let mlp_weights = effective_norm_weights(&block.mlp_norm);
    rmsnorm_shared(
        party,
        &mut normed_mlp,
        &mlp_weights,
        block.mlp_norm.eps(),
        transport,
    )?;

    // 5. Private MLP forward
    let mlp_out = private_dense_mlp_forward(
        party,
        &block.mlp,
        &normed_mlp,
        triples,
        transport,
    )?;

    // 6. Residual
    *hidden_share = hidden_share.add(&mlp_out);

    Ok(())
}

/// Sparse variant: uses a subset of MLP neurons.
pub fn private_block_forward_sparse(
    party: u8,
    block: &TransformerBlock,
    hidden_share: &mut SharedVec,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    deltanet_state: Option<&mut DeltaNetState>,
    neuron_indices: &[usize],
    triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<()> {
    // 1. Pre-attention RMSNorm
    let mut normed = hidden_share.clone();
    let attn_weights = effective_norm_weights(&block.attn_norm);
    rmsnorm_shared(
        party,
        &mut normed,
        &attn_weights,
        block.attn_norm.eps(),
        transport,
    )?;

    // 2. Attention
    let attn_out = match &block.attention {
        AttentionLayer::Standard(attn) => {
            private_attention_forward(
                party, attn, &normed, position, rope, kv_cache, transport,
            )?
        }
        AttentionLayer::GatedDeltaNet(dn) => {
            let normed_f32: Vec<f32> = normed.0.iter().map(|&v| from_fixed(v)).collect();
            let normed_plain = reveal_f32_shares(&normed_f32, transport)?;

            let state = deltanet_state
                .expect("DeltaNet state required for GatedDeltaNet layer");
            let attn_out_f32 = dn.forward_decode(&normed_plain, state);

            if party == 0 {
                SharedVec(attn_out_f32.iter().map(|&v| to_fixed(v)).collect())
            } else {
                SharedVec(vec![0u32; attn_out_f32.len()])
            }
        }
    };

    // 3. Residual
    *hidden_share = hidden_share.add(&attn_out);

    // 4. Pre-MLP RMSNorm
    let mut normed_mlp = hidden_share.clone();
    let mlp_weights = effective_norm_weights(&block.mlp_norm);
    rmsnorm_shared(
        party,
        &mut normed_mlp,
        &mlp_weights,
        block.mlp_norm.eps(),
        transport,
    )?;

    // 5. Private sparse MLP forward
    let mlp_out = private_sparse_mlp_forward(
        party,
        &block.mlp,
        &normed_mlp,
        neuron_indices,
        triples,
        transport,
    )?;

    // 6. Residual
    *hidden_share = hidden_share.add(&mlp_out);

    Ok(())
}

/// Secure (Q32.32) evaluation of one transformer block.
///
/// Uses privacy-preserving RMSNorm (Beaver squaring, only reveals sum(x²))
/// and Q32.32 attention/MLP. GatedDeltaNet layers use MPC projections —
/// only low-dimensional outputs (QKV, gates, Z) are revealed, not the
/// full hidden state.
pub fn private_block_forward_secure(
    party: u8,
    block: &TransformerBlock,
    hidden_share: &mut SharedVec64,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    deltanet_state: Option<&mut DeltaNetState>,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    // 1. Pre-attention RMSNorm (privacy-preserving)
    let mut normed = hidden_share.clone();
    let attn_weights = effective_norm_weights(&block.attn_norm);
    rmsnorm_shared_64(
        party,
        &mut normed,
        &attn_weights,
        block.attn_norm.eps(),
        triples,
        transport,
    )?;

    // 2. Attention
    let attn_out = match &block.attention {
        AttentionLayer::Standard(attn) => {
            private_attention_forward_secure(
                party, attn, &normed, position, rope, kv_cache, transport,
            )?
        }
        AttentionLayer::GatedDeltaNet(dn) => {
            let state = deltanet_state
                .expect("DeltaNet state required for GatedDeltaNet layer");
            private_deltanet_forward_secure(party, dn, &normed, state, transport)?
        }
    };

    // 3. Residual
    *hidden_share = hidden_share.add(&attn_out);

    // 4. Pre-MLP RMSNorm (privacy-preserving)
    let mut normed_mlp = hidden_share.clone();
    let mlp_weights = effective_norm_weights(&block.mlp_norm);
    rmsnorm_shared_64(
        party,
        &mut normed_mlp,
        &mlp_weights,
        block.mlp_norm.eps(),
        triples,
        transport,
    )?;

    // 5. Secure MLP forward
    let mlp_out = private_dense_mlp_forward_secure(
        party,
        &block.mlp,
        &normed_mlp,
        transport,
    )?;

    // 6. Residual
    *hidden_share = hidden_share.add(&mlp_out);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_llm::config::LlmConfig;
    use klearu_llm::model::block::TransformerBlock;
    use klearu_llm::model::kv_cache::KvCache;
    use klearu_llm::model::rope::RotaryEmbedding;
    use klearu_mpc::beaver::dummy_triple_pair;
    use klearu_mpc::fixed_point::{from_fixed, to_fixed};
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

    #[test]
    fn test_private_block_runs_without_error() {
        let config = tiny_config();
        let mut block = TransformerBlock::new(&config);

        // Set norm weights to 1.0 so normalization is meaningful
        for w in block.attn_norm.weight.iter_mut() {
            *w = 1.0;
        }
        for w in block.mlp_norm.weight.iter_mut() {
            *w = 1.0;
        }

        let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_theta);

        let input = vec![0.1f32, 0.2, -0.1, 0.05, 0.3, -0.2, 0.15, 0.0];
        let x_fixed: Vec<u32> = input.iter().map(|&v| to_fixed(v)).collect();

        let (mut gen0, mut gen1) = dummy_triple_pair(10000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec(x_fixed);
        let mut share1 = SharedVec(vec![0u32; config.hidden_size]);

        let mut block1 = TransformerBlock::new(&config);
        for w in block1.attn_norm.weight.iter_mut() {
            *w = 1.0;
        }
        for w in block1.mlp_norm.weight.iter_mut() {
            *w = 1.0;
        }

        let rope_clone = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_theta);
        let mut kv0 = KvCache::new(config.num_kv_heads, config.max_seq_len, config.head_dim);
        let mut kv1 = KvCache::new(config.num_kv_heads, config.max_seq_len, config.head_dim);

        let handle = std::thread::spawn(move || {
            private_block_forward(
                1, &block1, &mut share1, 0, &rope_clone, &mut kv1, None, &mut gen1, &mut trans_b,
            ).unwrap();
            share1
        });

        private_block_forward(
            0, &block, &mut share0, 0, &rope, &mut kv0, None, &mut gen0, &mut trans_a,
        ).unwrap();

        let share1 = handle.join().unwrap();

        // Reconstruct and verify finite
        for i in 0..config.hidden_size {
            let result = from_fixed(share0.0[i].wrapping_add(share1.0[i]));
            assert!(result.is_finite(), "block output[{}] is not finite: {}", i, result);
        }
    }
}
