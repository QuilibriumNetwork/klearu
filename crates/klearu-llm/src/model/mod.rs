pub mod attention;
pub mod block;
pub mod embedding;
pub mod gated_deltanet;
pub mod kv_cache;
pub mod linear;
pub mod mlp;
pub mod rms_norm;
pub mod rope;

use crate::config::{LayerType, LlmConfig};
use block::{AttentionLayer, TransformerBlock};
use embedding::Embedding;
use gated_deltanet::DeltaNetStateStore;
use kv_cache::KvCacheStore;
use rms_norm::RmsNorm;
use rope::RotaryEmbedding;

/// Full transformer model supporting LLaMA-family and Qwen3.5 hybrid architectures.
pub struct Model {
    pub config: LlmConfig,
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: RmsNorm,
    pub lm_head: Option<linear::Linear>,
    pub rope: RotaryEmbedding,
    pub kv_caches: KvCacheStore,
    pub deltanet_states: DeltaNetStateStore,
}

impl Model {
    /// Create a zeroed model from config (weights must be loaded separately).
    pub fn new(config: LlmConfig) -> Self {
        let rope = if config.is_qwen35() {
            let rotary_dim = config.rotary_dim();
            RotaryEmbedding::new_partial(
                config.head_dim,
                rotary_dim,
                // Limit precomputed cache to something reasonable
                config.max_seq_len.min(8192),
                config.rope_theta,
            )
        } else {
            RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_theta)
        };

        let embedding = Embedding::new(config.vocab_size, config.hidden_size);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerBlock::new_for_layer(&config, i));
        }

        let final_norm = if config.is_qwen35() {
            RmsNorm::new_one_plus(config.hidden_size, config.rms_norm_eps)
        } else {
            RmsNorm::new(config.hidden_size, config.rms_norm_eps)
        };

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(linear::Linear::new(config.hidden_size, config.vocab_size))
        };

        // Count full-attention layers for KV cache allocation
        let num_full_attn_layers = (0..config.num_layers)
            .filter(|&i| config.layer_type(i) == LayerType::FullAttention)
            .count();

        let kv_caches = KvCacheStore::new(
            config.num_layers,
            config.num_kv_heads,
            if config.is_qwen35() {
                config.max_seq_len.min(8192)
            } else {
                config.max_seq_len
            },
            config.head_dim,
        );

        // Create DeltaNet states for linear attention layers
        let mut deltanet_states = DeltaNetStateStore::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            if config.layer_type(i) == LayerType::LinearAttention {
                if let AttentionLayer::GatedDeltaNet(ref dn) = layers[i].attention {
                    deltanet_states.states[i] = Some(dn.create_state());
                }
            }
        }

        let _ = num_full_attn_layers; // used for documentation/clarity

        Self {
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            rope,
            kv_caches,
            deltanet_states,
        }
    }

    /// Reset all KV caches and DeltaNet states (e.g., for a new generation).
    pub fn reset_kv_caches(&mut self) {
        self.kv_caches.clear();
        self.deltanet_states.clear();
    }

    /// Decode a single token, returning logits over the vocabulary.
    ///
    /// `position` is the sequence position for this token (0-indexed).
    pub fn forward_decode(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        // Embed the token
        let mut hidden = vec![0.0f32; hidden_size];
        self.embedding.forward(token_id, &mut hidden);

        // Scratch buffers
        let mut norm_buf = vec![0.0f32; hidden_size];

        for layer_idx in 0..self.config.num_layers {
            // Pre-attention norm
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            // Attention (dispatch based on layer type)
            let attn_out = match &self.layers[layer_idx].attention {
                AttentionLayer::Standard(attn) => {
                    attn.forward(
                        &norm_buf,
                        position,
                        &self.rope,
                        self.kv_caches.layer_mut(layer_idx),
                    )
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    let state = self.deltanet_states.layer_mut(layer_idx)
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
            self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

            // MLP
            let mlp_out = self.layers[layer_idx].mlp.forward(&norm_buf);

            // Residual
            for (h, m) in hidden.iter_mut().zip(mlp_out.iter()) {
                *h += m;
            }
        }

        // Final norm
        self.final_norm.forward(&mut hidden);

        // LM head
        let mut logits = vec![0.0f32; vocab_size];
        match &self.lm_head {
            Some(head) => head.forward(&hidden, &mut logits),
            None => self.embedding.lm_head_forward(&hidden, &mut logits),
        }

        logits
    }

    /// Decode a single token with per-layer diagnostics printed to stderr.
    pub fn forward_decode_debug(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        let mut hidden = vec![0.0f32; hidden_size];
        self.embedding.forward(token_id, &mut hidden);
        let emb_norm: f32 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("  emb norm: {:.4}", emb_norm);
        eprintln!("  emb first 8: {:?}", &hidden[..8]);

        let mut norm_buf = vec![0.0f32; hidden_size];

        for layer_idx in 0..self.config.num_layers {
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            let attn_out = match &self.layers[layer_idx].attention {
                AttentionLayer::Standard(attn) => {
                    attn.forward(
                        &norm_buf,
                        position,
                        &self.rope,
                        self.kv_caches.layer_mut(layer_idx),
                    )
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    let state = self.deltanet_states.layer_mut(layer_idx)
                        .expect("DeltaNet state missing");
                    dn.forward_decode(&norm_buf, state)
                }
            };

            let attn_norm: f32 = attn_out.iter().map(|v| v * v).sum::<f32>().sqrt();
            // Debug: print attention output at layer 3
            if layer_idx == 3 {
                eprintln!("  L3 attn_out first 8: {:?}", &attn_out[..8]);
                eprintln!("  L3 attn_out norm: {:.6}", attn_norm);
            }
            for (h, a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }
            // Debug: print hidden after attention residual at layer 3
            if layer_idx == 3 {
                let post_attn_norm: f32 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
                eprintln!("  L3 post-attn hidden first 4: {:?}  norm: {:.6}", &hidden[..4], post_attn_norm);
            }

            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);
            let mlp_out = self.layers[layer_idx].mlp.forward(&norm_buf);
            let mlp_norm: f32 = mlp_out.iter().map(|v| v * v).sum::<f32>().sqrt();

            for (h, m) in hidden.iter_mut().zip(mlp_out.iter()) {
                *h += m;
            }

            let hidden_norm: f32 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
            let layer_kind = match &self.layers[layer_idx].attention {
                AttentionLayer::Standard(_) => "full",
                AttentionLayer::GatedDeltaNet(_) => "linear",
            };
            eprintln!("  L{:2} ({}): attn={:.4} mlp={:.4} hidden={:.6}  first 4: {:?}",
                layer_idx, layer_kind, attn_norm, mlp_norm, hidden_norm, &hidden[..4]);
        }

        self.final_norm.forward(&mut hidden);
        let mut logits = vec![0.0f32; vocab_size];
        match &self.lm_head {
            Some(head) => head.forward(&hidden, &mut logits),
            None => self.embedding.lm_head_forward(&hidden, &mut logits),
        }
        logits
    }

    /// Prefill multiple tokens (initial prompt processing).
    /// Returns logits for the last token only.
    pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Vec<f32> {
        let mut logits = Vec::new();
        for (pos, &token_id) in token_ids.iter().enumerate() {
            logits = self.forward_decode(token_id, pos);
        }
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> LlmConfig {
        LlmConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            num_layers: 2,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            ..LlmConfig::default()
        }
    }

    #[test]
    fn test_model_creation() {
        let config = tiny_config();
        let model = Model::new(config);
        assert_eq!(model.layers.len(), 2);
        assert!(model.lm_head.is_none()); // tied embeddings
    }

    #[test]
    fn test_model_forward_produces_finite_logits() {
        let config = tiny_config();
        let mut model = Model::new(config.clone());

        // Set embedding weights to small non-zero values
        for i in 0..config.vocab_size {
            let row = model.embedding.weights.get_weights_mut(i);
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((i * 7 + j * 3) % 17) as f32 * 0.01;
            }
        }

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

        let logits = model.forward_decode(1, 0);
        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_prefill_decode_consistency() {
        let config = tiny_config();

        // Run prefill
        let mut model_prefill = Model::new(config.clone());
        // Set some weights
        for layer in &mut model_prefill.layers {
            for w in layer.attn_norm.weight.iter_mut() {
                *w = 1.0;
            }
            for w in layer.mlp_norm.weight.iter_mut() {
                *w = 1.0;
            }
        }
        for w in model_prefill.final_norm.weight.iter_mut() {
            *w = 1.0;
        }
        let _ = model_prefill.forward_prefill(&[0, 1, 2]);

        // Run sequential decode
        let mut model_decode = Model::new(config.clone());
        for layer in &mut model_decode.layers {
            for w in layer.attn_norm.weight.iter_mut() {
                *w = 1.0;
            }
            for w in layer.mlp_norm.weight.iter_mut() {
                *w = 1.0;
            }
        }
        for w in model_decode.final_norm.weight.iter_mut() {
            *w = 1.0;
        }
        let _ = model_decode.forward_decode(0, 0);
        let _ = model_decode.forward_decode(1, 1);
        let _ = model_decode.forward_decode(2, 2);

        // KV caches should have same content
        for layer_idx in 0..config.num_layers {
            let kv_pf = model_prefill.kv_caches.layer(layer_idx);
            let kv_dc = model_decode.kv_caches.layer(layer_idx);
            assert_eq!(kv_pf.current_len(), kv_dc.current_len());
            for h in 0..config.num_kv_heads {
                let k_pf = kv_pf.k_head_positions(h, 3);
                let k_dc = kv_dc.k_head_positions(h, 3);
                for (a, b) in k_pf.iter().zip(k_dc.iter()) {
                    assert!((a - b).abs() < 1e-6, "KV cache mismatch");
                }
            }
        }
    }

    #[test]
    fn test_qwen35_model_creation() {
        let config = LlmConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            intermediate_size: 64,
            num_layers: 4,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            model_type: Some("qwen3_5_text".to_string()),
            layer_types: Some(vec![
                "linear_attention".into(),
                "linear_attention".into(),
                "linear_attention".into(),
                "full_attention".into(),
            ]),
            attn_output_gate: true,
            linear_num_key_heads: 4,
            linear_num_value_heads: 4,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            rope_parameters: Some(crate::config::RopeParameters {
                partial_rotary_factor: 0.25,
                rope_theta: 10000000.0,
                rope_type: None,
                mrope_interleaved: false,
                mrope_section: None,
            }),
            ..LlmConfig::default()
        };

        let model = Model::new(config.clone());
        assert_eq!(model.layers.len(), 4);

        // First 3 layers should be GatedDeltaNet
        for i in 0..3 {
            assert!(matches!(model.layers[i].attention, AttentionLayer::GatedDeltaNet(_)),
                "Layer {i} should be GatedDeltaNet");
            assert!(model.deltanet_states.states[i].is_some(),
                "Layer {i} should have DeltaNet state");
        }

        // Layer 3 should be Standard (gated) attention
        assert!(matches!(model.layers[3].attention, AttentionLayer::Standard(_)),
            "Layer 3 should be Standard attention");
        assert!(model.deltanet_states.states[3].is_none(),
            "Layer 3 should NOT have DeltaNet state");
    }

    #[test]
    fn test_qwen35_forward_produces_finite() {
        let config = LlmConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            num_layers: 2,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            model_type: Some("qwen3_5_text".to_string()),
            layer_types: Some(vec![
                "linear_attention".into(),
                "full_attention".into(),
            ]),
            attn_output_gate: true,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            rope_parameters: Some(crate::config::RopeParameters {
                partial_rotary_factor: 0.5,
                rope_theta: 10000.0,
                rope_type: None,
                mrope_interleaved: false,
                mrope_section: None,
            }),
            ..LlmConfig::default()
        };

        let mut model = Model::new(config.clone());

        // Set embedding to non-zero
        for i in 0..config.vocab_size {
            let row = model.embedding.weights.get_weights_mut(i);
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((i * 7 + j * 3) % 17) as f32 * 0.01;
            }
        }

        let logits = model.forward_decode(1, 0);
        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|x| x.is_finite()), "Logits should be finite");

        // Run a second token to test state evolution
        let logits2 = model.forward_decode(2, 1);
        assert!(logits2.iter().all(|x| x.is_finite()), "Second decode should be finite");
    }
}
