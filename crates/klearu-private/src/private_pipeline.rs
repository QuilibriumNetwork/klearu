//! Private chat pipeline: client-side generation and server-side serving
//! over a 2PC transport.
//!
//! The client holds the secret token IDs; the server never learns them.
//! Both parties hold the same public model weights and run synchronized
//! MPC forward passes.

use klearu_llm::generate::sampler::{SamplerConfig, sample};
use klearu_llm::Model;
use klearu_mpc::beaver::{TripleGenerator, TripleGenerator128};
use klearu_mpc::transport::Transport;
use rand::Rng;
use std::io;

use crate::private_model::{
    private_model_forward, private_model_forward_secure, private_model_forward_sparse,
    shared_embedding_lookup, shared_embedding_lookup_64,
};

/// Security level for private inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityLevel {
    /// Fast mode: reveals embedding, runs plaintext forward. No input privacy.
    /// Communication: ~4.6 KB/token. Triples: 0.
    Lower,
    /// Secure mode: Q32.32 shares, per-layer MPC, only reveals norms/Q/gate.
    /// Communication: ~2 MB/token. Triples: ~34K/token.
    High,
}

/// Configuration for private generation.
pub struct PrivateConfig {
    pub max_new_tokens: usize,
    pub sampler: SamplerConfig,
    pub eos_token_id: Option<u32>,
    /// Use sparse MLP with fixed first-k neurons.
    pub sparse: bool,
    /// Fraction of intermediate neurons to use when sparse (0.0–1.0).
    pub neuron_sparsity: f32,
    /// Security level: Lower (reveal embedding) or High (Q32.32 per-layer MPC).
    pub security: SecurityLevel,
}

/// Metadata exchanged at the start of a private session.
struct SessionMeta {
    num_tokens: u32,
    max_new: u32,
    sparse: bool,
    neuron_sparsity: f32,
    security: SecurityLevel,
}

impl SessionMeta {
    fn send(&self, transport: &mut impl Transport) -> io::Result<()> {
        transport.send_u32(self.num_tokens)?;
        transport.send_u32(self.max_new)?;
        transport.send_u32(if self.sparse { 1 } else { 0 })?;
        transport.send_u32(self.neuron_sparsity.to_bits())?;
        transport.send_u32(match self.security {
            SecurityLevel::Lower => 0,
            SecurityLevel::High => 1,
        })
    }

    fn recv(transport: &mut impl Transport) -> io::Result<Self> {
        let num_tokens = transport.recv_u32()?;
        let max_new = transport.recv_u32()?;
        let sparse_flag = transport.recv_u32()?;
        let sparsity_bits = transport.recv_u32()?;
        let security_flag = transport.recv_u32()?;
        Ok(Self {
            num_tokens,
            max_new,
            sparse: sparse_flag != 0,
            neuron_sparsity: f32::from_bits(sparsity_bits),
            security: if security_flag == 0 { SecurityLevel::Lower } else { SecurityLevel::High },
        })
    }
}

/// Client-side private generation (lower security mode).
///
/// Runs the MPC protocol with the server, streams tokens via `on_token`.
/// Returns all generated token IDs.
pub fn generate_private(
    model: &mut Model,
    prompt_tokens: &[u32],
    config: &PrivateConfig,
    triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
    rng: &mut impl Rng,
    mut on_token: impl FnMut(&str, u32) -> bool,
    tokenizer: &klearu_llm::tokenizer::Tokenizer,
) -> io::Result<Vec<u32>> {
    let party: u8 = 0;

    // Send metadata
    let meta = SessionMeta {
        num_tokens: prompt_tokens.len() as u32,
        max_new: config.max_new_tokens as u32,
        sparse: config.sparse,
        neuron_sparsity: config.neuron_sparsity,
        security: config.security,
    };
    meta.send(transport)?;

    model.reset_kv_caches();

    let neuron_indices: Vec<usize> = if config.sparse {
        let k = (model.config.intermediate_size as f32 * config.neuron_sparsity) as usize;
        (0..k).collect()
    } else {
        Vec::new()
    };

    // Prefill: all tokens except the last (discard logits)
    for i in 0..prompt_tokens.len().saturating_sub(1) {
        let input_share = shared_embedding_lookup(party, model, prompt_tokens[i]);
        if config.sparse {
            let _ = private_model_forward_sparse(
                party, model, &input_share, i, &neuron_indices, triples, transport,
            )?;
        } else {
            let _ = private_model_forward(party, model, &input_share, i, triples, transport)?;
        }
    }

    // Last prefill token: keep logits
    let last_idx = prompt_tokens.len().saturating_sub(1);
    let last_token = prompt_tokens[last_idx];
    let input_share = shared_embedding_lookup(party, model, last_token);
    let mut logits = if config.sparse {
        private_model_forward_sparse(
            party, model, &input_share, last_idx, &neuron_indices, triples, transport,
        )?
    } else {
        private_model_forward(party, model, &input_share, last_idx, triples, transport)?
    };

    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
    let mut generated = Vec::new();

    // Decode loop
    for step in 0..config.max_new_tokens {
        let next_token = sample(&mut logits, &config.sampler, &all_tokens, rng);

        // Check EOS
        if config.eos_token_id == Some(next_token) {
            // Signal stop to server
            transport.send_u32(0)?; // 0 = stop
            break;
        }

        all_tokens.push(next_token);
        generated.push(next_token);

        // Stream token text
        if let Ok(text) = tokenizer.decode(&[next_token]) {
            if !on_token(&text, next_token) {
                transport.send_u32(0)?;
                break;
            }
        }

        // Signal continue to server
        transport.send_u32(1)?; // 1 = continue

        // Next decode step
        let pos = prompt_tokens.len() + step;
        let input_share = shared_embedding_lookup(party, model, next_token);
        logits = if config.sparse {
            private_model_forward_sparse(
                party, model, &input_share, pos, &neuron_indices, triples, transport,
            )?
        } else {
            private_model_forward(party, model, &input_share, pos, triples, transport)?
        };
    }

    Ok(generated)
}

/// Server-side private serving.
///
/// Runs the MPC protocol with the client. The server never learns token IDs.
pub fn serve_private(
    model: &mut Model,
    triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<()> {
    let party: u8 = 1;

    // Receive metadata
    let meta = SessionMeta::recv(transport)?;
    let num_tokens = meta.num_tokens as usize;
    let max_new = meta.max_new as usize;

    model.reset_kv_caches();

    let neuron_indices: Vec<usize> = if meta.sparse {
        let k = (model.config.intermediate_size as f32 * meta.neuron_sparsity) as usize;
        (0..k).collect()
    } else {
        Vec::new()
    };

    // Prefill: server uses zero shares (doesn't know tokens)
    for i in 0..num_tokens.saturating_sub(1) {
        let input_share = shared_embedding_lookup(party, model, 0);
        if meta.sparse {
            let _ = private_model_forward_sparse(
                party, model, &input_share, i, &neuron_indices, triples, transport,
            )?;
        } else {
            let _ = private_model_forward(party, model, &input_share, i, triples, transport)?;
        }
    }

    // Last prefill token
    let last_idx = num_tokens.saturating_sub(1);
    let input_share = shared_embedding_lookup(party, model, 0);
    let _logits = if meta.sparse {
        private_model_forward_sparse(
            party, model, &input_share, last_idx, &neuron_indices, triples, transport,
        )?
    } else {
        private_model_forward(party, model, &input_share, last_idx, triples, transport)?
    };

    // Decode loop
    for step in 0..max_new {
        // Wait for continue/stop from client
        let signal = transport.recv_u32()?;
        if signal == 0 {
            break;
        }

        let pos = num_tokens + step;
        let input_share = shared_embedding_lookup(party, model, 0);
        let _logits = if meta.sparse {
            private_model_forward_sparse(
                party, model, &input_share, pos, &neuron_indices, triples, transport,
            )?
        } else {
            private_model_forward(party, model, &input_share, pos, triples, transport)?
        };
    }

    Ok(())
}

/// Client-side private generation (high security / Q32.32 mode).
pub fn generate_private_secure(
    model: &mut Model,
    prompt_tokens: &[u32],
    config: &PrivateConfig,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
    rng: &mut impl Rng,
    mut on_token: impl FnMut(&str, u32) -> bool,
    tokenizer: &klearu_llm::tokenizer::Tokenizer,
) -> io::Result<Vec<u32>> {
    let party: u8 = 0;

    let meta = SessionMeta {
        num_tokens: prompt_tokens.len() as u32,
        max_new: config.max_new_tokens as u32,
        sparse: config.sparse,
        neuron_sparsity: config.neuron_sparsity,
        security: config.security,
    };
    meta.send(transport)?;

    model.reset_kv_caches();

    // Prefill: all tokens except the last
    for i in 0..prompt_tokens.len().saturating_sub(1) {
        let input_share = shared_embedding_lookup_64(party, model, prompt_tokens[i]);
        let _ = private_model_forward_secure(party, model, &input_share, i, triples, transport)?;
    }

    // Last prefill token: keep logits
    let last_idx = prompt_tokens.len().saturating_sub(1);
    let last_token = prompt_tokens[last_idx];
    let input_share = shared_embedding_lookup_64(party, model, last_token);
    let mut logits = private_model_forward_secure(party, model, &input_share, last_idx, triples, transport)?;

    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
    let mut generated = Vec::new();

    // Decode loop
    for step in 0..config.max_new_tokens {
        let next_token = sample(&mut logits, &config.sampler, &all_tokens, rng);

        if config.eos_token_id == Some(next_token) {
            transport.send_u32(0)?;
            break;
        }

        all_tokens.push(next_token);
        generated.push(next_token);

        if let Ok(text) = tokenizer.decode(&[next_token]) {
            if !on_token(&text, next_token) {
                transport.send_u32(0)?;
                break;
            }
        }

        transport.send_u32(1)?;

        let pos = prompt_tokens.len() + step;
        let input_share = shared_embedding_lookup_64(party, model, next_token);
        logits = private_model_forward_secure(party, model, &input_share, pos, triples, transport)?;
    }

    Ok(generated)
}

/// Server-side private serving (high security / Q32.32 mode).
pub fn serve_private_secure(
    model: &mut Model,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    let party: u8 = 1;

    let meta = SessionMeta::recv(transport)?;
    let num_tokens = meta.num_tokens as usize;
    let max_new = meta.max_new as usize;

    model.reset_kv_caches();

    // Prefill
    for i in 0..num_tokens.saturating_sub(1) {
        let input_share = shared_embedding_lookup_64(party, model, 0);
        let _ = private_model_forward_secure(party, model, &input_share, i, triples, transport)?;
    }

    // Last prefill token
    let last_idx = num_tokens.saturating_sub(1);
    let input_share = shared_embedding_lookup_64(party, model, 0);
    let _ = private_model_forward_secure(party, model, &input_share, last_idx, triples, transport)?;

    // Decode loop
    for step in 0..max_new {
        let signal = transport.recv_u32()?;
        if signal == 0 {
            break;
        }

        let pos = num_tokens + step;
        let input_share = shared_embedding_lookup_64(party, model, 0);
        let _ = private_model_forward_secure(party, model, &input_share, pos, triples, transport)?;
    }

    Ok(())
}

// Re-export detect_eos_token from klearu-llm.
pub use klearu_llm::generate::pipeline::detect_eos_token;

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_llm::config::LlmConfig;
    use klearu_mpc::beaver::dummy_triple_pair;
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
            max_seq_len: 32,
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
    fn test_private_pipeline_one_token() {
        let config = tiny_config();
        let mut model0 = Model::new(config.clone());
        set_norm_weights(&mut model0);
        let mut model1 = Model::new(config.clone());
        set_norm_weights(&mut model1);

        let (mut gen0, mut gen1) = dummy_triple_pair(500_000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let prompt_tokens = vec![1u32, 2, 3];

        let priv_config = PrivateConfig {
            max_new_tokens: 1,
            sampler: SamplerConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
            },
            eos_token_id: None,
            sparse: false,
            neuron_sparsity: 0.5,
            security: SecurityLevel::Lower,
        };

        // Server thread
        let handle = std::thread::spawn(move || {
            serve_private(&mut model1, &mut gen1, &mut trans_b)
        });

        // Client: use a dummy tokenizer-like decode via on_token
        let mut generated_tokens = Vec::new();
        // We can't easily construct a real Tokenizer without files,
        // so we test the protocol flow by checking the server completes.

        // Build a minimal mock: since we can't construct a Tokenizer without a file,
        // we call the lower-level protocol directly.
        // For this test we just verify the server thread doesn't deadlock/error.

        // Client side: manual protocol
        let party: u8 = 0;
        let meta = SessionMeta {
            num_tokens: prompt_tokens.len() as u32,
            max_new: 1,
            sparse: false,
            neuron_sparsity: 0.5,
            security: SecurityLevel::Lower,
        };
        meta.send(&mut trans_a).unwrap();

        model0.reset_kv_caches();

        // Prefill all but last
        for i in 0..prompt_tokens.len() - 1 {
            let input_share = shared_embedding_lookup(party, &model0, prompt_tokens[i]);
            let _ = private_model_forward(party, &mut model0, &input_share, i, &mut gen0, &mut trans_a).unwrap();
        }

        // Last prefill
        let last_idx = prompt_tokens.len() - 1;
        let input_share = shared_embedding_lookup(party, &model0, prompt_tokens[last_idx]);
        let mut logits = private_model_forward(party, &mut model0, &input_share, last_idx, &mut gen0, &mut trans_a).unwrap();

        let next_token = sample(&mut logits, &priv_config.sampler, &prompt_tokens, &mut rand::thread_rng());
        generated_tokens.push(next_token);

        // Signal stop
        trans_a.send_u32(0).unwrap();

        let server_result = handle.join().unwrap();
        assert!(server_result.is_ok(), "server failed: {:?}", server_result);
        assert_eq!(generated_tokens.len(), 1);
    }
}
