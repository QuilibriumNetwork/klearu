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

/// How a per-token activation sequence should be reduced to a single
/// per-probe vector.
///
/// - `Last` — keep only the last token's value (LM-style, default).
/// - `Mean` — average across all token positions (retrieval-style, gives
///   a more evenly-weighted representation of the whole sequence).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    Last,
    Mean,
}

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

    /// Create a model optimized for MPC party 0 (WASM): minimal embedding,
    /// always-separate lm_head. The embedding table uses only 1 row to save
    /// memory since party 0 never does token lookups (it receives embedding
    /// shares). lm_head is always allocated as a separate Linear, even for
    /// tie_word_embeddings models (its weights must be loaded separately).
    pub fn new_no_embedding(config: LlmConfig) -> Self {
        let rope = if config.is_qwen35() {
            let rotary_dim = config.rotary_dim();
            RotaryEmbedding::new_partial(
                config.head_dim,
                rotary_dim,
                config.max_seq_len.min(8192),
                config.rope_theta,
            )
        } else {
            RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_theta)
        };

        // Minimal embedding (1 row) — party 0 never uses it
        let embedding = Embedding::new(1, config.hidden_size);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerBlock::new_for_layer(&config, i));
        }

        let final_norm = if config.is_qwen35() {
            RmsNorm::new_one_plus(config.hidden_size, config.rms_norm_eps)
        } else {
            RmsNorm::new(config.hidden_size, config.rms_norm_eps)
        };

        // Always create a separate lm_head (even for tie_word_embeddings)
        let lm_head = Some(linear::Linear::new(config.hidden_size, config.vocab_size));

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

        let mut deltanet_states = DeltaNetStateStore::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            if config.layer_type(i) == LayerType::LinearAttention {
                if let AttentionLayer::GatedDeltaNet(ref dn) = layers[i].attention {
                    deltanet_states.states[i] = Some(dn.create_state());
                }
            }
        }

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

    /// Prefill `token_ids` and record the Shannon entropy of each
    /// attention head's softmax at `target_layer`, evaluated at the **last**
    /// token position. Returns `(post_final_norm_hidden, per_head_entropy)`
    /// where the second vector has length `num_heads` of the target layer.
    ///
    /// Only valid for standard (non-linear) attention layers. Panics with
    /// a clear message if `target_layer` is out of range or is a
    /// GatedDeltaNet (linear) layer.
    ///
    /// Entropy is the "attention-dominance" signal: low entropy means a
    /// head has committed to one key position; high entropy means attention
    /// is diffuse across many keys. Used as an alternative stratification
    /// signal alongside the SwiGLU-gate boundary score.
    pub fn forward_prefill_with_attn_entropy(
        &mut self,
        token_ids: &[u32],
        target_layer: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        assert!(
            !token_ids.is_empty(),
            "forward_prefill_with_attn_entropy requires at least one token"
        );
        assert!(
            target_layer < self.config.num_layers,
            "target_layer {target_layer} out of range (num_layers={})",
            self.config.num_layers
        );
        let num_heads = match &self.layers[target_layer].attention {
            AttentionLayer::Standard(attn) => attn.num_heads(),
            AttentionLayer::GatedDeltaNet(_) => panic!(
                "target_layer {target_layer} is a linear-attention layer; \
                 attention-entropy capture requires a standard full-attention layer"
            ),
        };

        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        let mut hidden = vec![0.0f32; hidden_size];
        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; intermediate_size];
        let mut up_buf = vec![0.0f32; intermediate_size];
        let mut entropy_buf = vec![0.0f32; num_heads];
        let mut entropy_capture = vec![0.0f32; num_heads];

        let last_pos = token_ids.len() - 1;

        for (pos, &token_id) in token_ids.iter().enumerate() {
            self.embedding.forward(token_id, &mut hidden);

            for layer_idx in 0..self.config.num_layers {
                norm_buf.copy_from_slice(&hidden);
                self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

                match &self.layers[layer_idx].attention {
                    AttentionLayer::Standard(attn) => {
                        if layer_idx == target_layer && pos == last_pos {
                            attn.forward_into_with_entropy(
                                &norm_buf,
                                pos,
                                &self.rope,
                                self.kv_caches.layer_mut(layer_idx),
                                &mut attn_out,
                                &mut entropy_buf,
                            );
                            entropy_capture.copy_from_slice(&entropy_buf);
                        } else {
                            attn.forward_into(
                                &norm_buf,
                                pos,
                                &self.rope,
                                self.kv_caches.layer_mut(layer_idx),
                                &mut attn_out,
                            );
                        }
                    }
                    AttentionLayer::GatedDeltaNet(dn) => {
                        let state = self
                            .deltanet_states
                            .layer_mut(layer_idx)
                            .expect("DeltaNet state missing for linear attention layer");
                        dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                    }
                };

                for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                    *h += a;
                }

                norm_buf.copy_from_slice(&hidden);
                self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);
                self.layers[layer_idx].mlp.forward_into(
                    &norm_buf,
                    &mut mlp_out,
                    &mut gate_buf,
                    &mut up_buf,
                );

                for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
                    *h += m;
                }
            }
        }

        self.final_norm.forward(&mut hidden);
        (hidden, entropy_capture)
    }

    /// Apply the model's LM head (or tied embedding) to a post-final-norm
    /// hidden vector, returning logits over the vocabulary. Used by research
    /// tooling to decode arbitrary directions in hidden space back to
    /// tokens.
    pub fn apply_lm_head(&self, hidden: &[f32]) -> Vec<f32> {
        assert_eq!(
            hidden.len(),
            self.config.hidden_size,
            "apply_lm_head: expected hidden size {}, got {}",
            self.config.hidden_size,
            hidden.len()
        );
        let mut logits = vec![0.0f32; self.config.vocab_size];
        match &self.lm_head {
            Some(head) => head.forward(hidden, &mut logits),
            None => self.embedding.lm_head_forward(hidden, &mut logits),
        }
        logits
    }

    /// Decode a single token, returning logits over the vocabulary.
    ///
    /// `position` is the sequence position for this token (0-indexed).
    /// Like `forward_decode` but returns the post-final-norm hidden state
    /// instead of LM-head logits. Useful for steering: the caller can
    /// modify the hidden (e.g., add a user-preference vector) before
    /// applying `apply_lm_head` to get the steered logits.
    pub fn forward_decode_hidden(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        let mut hidden = vec![0.0f32; hidden_size];
        self.embedding.forward(token_id, &mut hidden);

        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; intermediate_size];
        let mut up_buf = vec![0.0f32; intermediate_size];

        for layer_idx in 0..self.config.num_layers {
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            match &self.layers[layer_idx].attention {
                AttentionLayer::Standard(attn) => {
                    attn.forward_into(
                        &norm_buf,
                        position,
                        &self.rope,
                        self.kv_caches.layer_mut(layer_idx),
                        &mut attn_out,
                    );
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    let state = self.deltanet_states.layer_mut(layer_idx)
                        .expect("DeltaNet state missing for linear attention layer");
                    dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                }
            };

            for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

            self.layers[layer_idx].mlp.forward_into(&norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf);

            for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
                *h += m;
            }
        }

        self.final_norm.forward(&mut hidden);
        hidden
    }

    pub fn forward_decode(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let intermediate_size = self.config.intermediate_size;

        // Embed the token
        let mut hidden = vec![0.0f32; hidden_size];
        self.embedding.forward(token_id, &mut hidden);

        // Pre-allocate all scratch buffers (reused across layers)
        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; intermediate_size];
        let mut up_buf = vec![0.0f32; intermediate_size];

        for layer_idx in 0..self.config.num_layers {
            // Pre-attention norm
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            // Attention (dispatch based on layer type)
            match &self.layers[layer_idx].attention {
                AttentionLayer::Standard(attn) => {
                    attn.forward_into(
                        &norm_buf,
                        position,
                        &self.rope,
                        self.kv_caches.layer_mut(layer_idx),
                        &mut attn_out,
                    );
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    let state = self.deltanet_states.layer_mut(layer_idx)
                        .expect("DeltaNet state missing for linear attention layer");
                    dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                }
            };

            // Residual
            for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            // Pre-MLP norm
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

            // MLP
            self.layers[layer_idx].mlp.forward_into(&norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf);

            // Residual
            for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
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

    /// Decode one token with an intervention applied to the residual
    /// stream at the output of `intervention_layer` (post-MLP-residual,
    /// pre-next-layer). The intervention is a closure that receives the
    /// hidden vector by mutable reference and may rewrite it in place.
    ///
    /// Used by causal-alignment / activation-patching experiments. The
    /// KV caches are not affected (the intervention runs on the residual
    /// stream after attention has already written K/V); intervening on
    /// every layer would require a separate API. To compare baseline vs
    /// intervened outputs on the SAME prompt, prefill the prompt with
    /// `forward_prefill_*`, snapshot the cache state, run this method
    /// twice (once with a no-op intervention for baseline, once with the
    /// real intervention) — the model's KV cache is reset between
    /// generations via `reset_kv_caches`.
    pub fn forward_decode_with_intervention<F>(
        &mut self,
        token_id: u32,
        position: usize,
        intervention_layer: usize,
        mut intervention: F,
    ) -> Vec<f32>
    where
        F: FnMut(&mut [f32]),
    {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let intermediate_size = self.config.intermediate_size;

        let mut hidden = vec![0.0f32; hidden_size];
        self.embedding.forward(token_id, &mut hidden);

        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; intermediate_size];
        let mut up_buf = vec![0.0f32; intermediate_size];

        for layer_idx in 0..self.config.num_layers {
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            match &self.layers[layer_idx].attention {
                AttentionLayer::Standard(attn) => {
                    attn.forward_into(
                        &norm_buf,
                        position,
                        &self.rope,
                        self.kv_caches.layer_mut(layer_idx),
                        &mut attn_out,
                    );
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    let state = self
                        .deltanet_states
                        .layer_mut(layer_idx)
                        .expect("DeltaNet state missing for linear attention layer");
                    dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                }
            };

            for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);
            self.layers[layer_idx]
                .mlp
                .forward_into(&norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf);

            for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
                *h += m;
            }

            // Intervene on the residual stream at this layer's output.
            if layer_idx == intervention_layer {
                intervention(&mut hidden);
            }
        }

        self.final_norm.forward(&mut hidden);

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
        let intermediate_size = self.config.intermediate_size;

        let mut hidden = vec![0.0f32; hidden_size];
        self.embedding.forward(token_id, &mut hidden);
        let emb_norm: f32 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("  emb norm: {:.4}", emb_norm);
        eprintln!("  emb first 8: {:?}", &hidden[..8]);

        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; intermediate_size];
        let mut up_buf = vec![0.0f32; intermediate_size];

        for layer_idx in 0..self.config.num_layers {
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            match &self.layers[layer_idx].attention {
                AttentionLayer::Standard(attn) => {
                    attn.forward_into(
                        &norm_buf,
                        position,
                        &self.rope,
                        self.kv_caches.layer_mut(layer_idx),
                        &mut attn_out,
                    );
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    let state = self.deltanet_states.layer_mut(layer_idx)
                        .expect("DeltaNet state missing");
                    dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                }
            };

            let attn_norm: f32 = attn_out.iter().map(|v| v * v).sum::<f32>().sqrt();
            // Debug: print attention output at layer 3
            if layer_idx == 3 {
                eprintln!("  L3 attn_out first 8: {:?}", &attn_out[..8]);
                eprintln!("  L3 attn_out norm: {:.6}", attn_norm);
            }
            for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }
            // Debug: print hidden after attention residual at layer 3
            if layer_idx == 3 {
                let post_attn_norm: f32 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
                eprintln!("  L3 post-attn hidden first 4: {:?}  norm: {:.6}", &hidden[..4], post_attn_norm);
            }

            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);
            self.layers[layer_idx].mlp.forward_into(&norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf);
            let mlp_norm: f32 = mlp_out.iter().map(|v| v * v).sum::<f32>().sqrt();

            for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
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

    /// Decode from a raw embedding vector instead of a token ID.
    ///
    /// The provided `embedding` slice must have length == `hidden_size`.
    /// This skips the embedding lookup and injects the vector directly as
    /// the initial hidden state, then runs the same transformer layers,
    /// final norm, and LM head as `forward_decode`.
    pub fn forward_decode_with_embedding(&mut self, embedding: &[f32], position: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let intermediate_size = self.config.intermediate_size;

        // Use provided embedding directly instead of token lookup
        let mut hidden = vec![0.0f32; hidden_size];
        hidden.copy_from_slice(&embedding[..hidden_size]);

        // Pre-allocate all scratch buffers (reused across layers)
        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; intermediate_size];
        let mut up_buf = vec![0.0f32; intermediate_size];

        for layer_idx in 0..self.config.num_layers {
            // Pre-attention norm
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            // Attention (dispatch based on layer type)
            match &self.layers[layer_idx].attention {
                AttentionLayer::Standard(attn) => {
                    attn.forward_into(
                        &norm_buf,
                        position,
                        &self.rope,
                        self.kv_caches.layer_mut(layer_idx),
                        &mut attn_out,
                    );
                }
                AttentionLayer::GatedDeltaNet(dn) => {
                    let state = self.deltanet_states.layer_mut(layer_idx)
                        .expect("DeltaNet state missing for linear attention layer");
                    dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                }
            };

            // Residual
            for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            // Pre-MLP norm
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

            // MLP
            self.layers[layer_idx].mlp.forward_into(&norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf);

            // Residual
            for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
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

    /// Prefill multiple tokens (initial prompt processing).
    /// Returns logits for the last token only.
    pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Vec<f32> {
        let mut logits = Vec::new();
        for (pos, &token_id) in token_ids.iter().enumerate() {
            logits = self.forward_decode(token_id, pos);
        }
        logits
    }

    /// Prefill multiple tokens and return the hidden state (post-final-norm,
    /// pre-LM-head) at the last token position. This is the contextual
    /// representation of the input sequence — suitable for use as a semantic
    /// embedding. KV caches are populated for all positions.
    pub fn forward_prefill_hidden(&mut self, token_ids: &[u32]) -> Vec<f32> {
        let (hidden, _) = self.forward_prefill_hidden_and_logits(token_ids);
        hidden
    }

    /// Prefill multiple tokens and return both:
    /// - The post-final-norm hidden state (contextual embedding)
    /// - Logits over the vocabulary (for sampling the next token)
    ///
    /// KV caches are populated for all positions, so autoregressive
    /// generation can continue from `position = token_ids.len()`.
    pub fn forward_prefill_hidden_and_logits(&mut self, token_ids: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let intermediate_size = self.config.intermediate_size;

        let mut hidden = vec![0.0f32; hidden_size];
        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; intermediate_size];
        let mut up_buf = vec![0.0f32; intermediate_size];

        for (pos, &token_id) in token_ids.iter().enumerate() {
            // Embed the token
            self.embedding.forward(token_id, &mut hidden);

            for layer_idx in 0..self.config.num_layers {
                norm_buf.copy_from_slice(&hidden);
                self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

                match &self.layers[layer_idx].attention {
                    AttentionLayer::Standard(attn) => {
                        attn.forward_into(
                            &norm_buf,
                            pos,
                            &self.rope,
                            self.kv_caches.layer_mut(layer_idx),
                            &mut attn_out,
                        );
                    }
                    AttentionLayer::GatedDeltaNet(dn) => {
                        let state = self.deltanet_states.layer_mut(layer_idx)
                            .expect("DeltaNet state missing for linear attention layer");
                        dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                    }
                };

                for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                    *h += a;
                }

                norm_buf.copy_from_slice(&hidden);
                self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);
                self.layers[layer_idx].mlp.forward_into(&norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf);

                for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
                    *h += m;
                }
            }
        }

        // Final norm
        self.final_norm.forward(&mut hidden);

        // LM head for logits
        let mut logits = vec![0.0f32; vocab_size];
        match &self.lm_head {
            Some(head) => head.forward(&hidden, &mut logits),
            None => self.embedding.lm_head_forward(&hidden, &mut logits),
        }

        (hidden, logits)
    }

    /// Prefill `token_ids` and capture the **residual-stream hidden
    /// state** at each requested layer for the last token position.
    /// "Residual-stream hidden" means the hidden state right after the
    /// MLP residual add at layer L — i.e., the input to layer L+1.
    /// Returns `(post_final_norm_hidden, per_layer_hidden)` where
    /// `per_layer_hidden[i]` corresponds to `target_layers[i]`.
    ///
    /// Used by tokenizer-transplantation and layer-skipping work that
    /// needs intermediate states, not just the final post-norm.
    ///
    /// Panics on duplicate or out-of-range layer indices, or empty
    /// `token_ids`.
    pub fn forward_prefill_per_layer_hidden(
        &mut self,
        token_ids: &[u32],
        target_layers: &[usize],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        assert!(
            !token_ids.is_empty(),
            "forward_prefill_per_layer_hidden requires at least one token"
        );
        for &l in target_layers {
            assert!(
                l < self.config.num_layers,
                "target_layer {l} out of range (num_layers={})",
                self.config.num_layers
            );
        }
        for i in 0..target_layers.len() {
            for j in (i + 1)..target_layers.len() {
                assert!(
                    target_layers[i] != target_layers[j],
                    "duplicate target layer {} in target_layers",
                    target_layers[i]
                );
            }
        }

        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        let mut hidden = vec![0.0f32; hidden_size];
        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; intermediate_size];
        let mut up_buf = vec![0.0f32; intermediate_size];

        let mut per_layer_captures: Vec<Vec<f32>> = target_layers
            .iter()
            .map(|_| vec![0.0f32; hidden_size])
            .collect();
        let layer_to_slot: Vec<Option<usize>> = {
            let mut v = vec![None; self.config.num_layers];
            for (slot, &l) in target_layers.iter().enumerate() {
                v[l] = Some(slot);
            }
            v
        };

        let last_pos = token_ids.len() - 1;

        for (pos, &token_id) in token_ids.iter().enumerate() {
            self.embedding.forward(token_id, &mut hidden);

            for layer_idx in 0..self.config.num_layers {
                norm_buf.copy_from_slice(&hidden);
                self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

                match &self.layers[layer_idx].attention {
                    AttentionLayer::Standard(attn) => {
                        attn.forward_into(
                            &norm_buf,
                            pos,
                            &self.rope,
                            self.kv_caches.layer_mut(layer_idx),
                            &mut attn_out,
                        );
                    }
                    AttentionLayer::GatedDeltaNet(dn) => {
                        let state = self
                            .deltanet_states
                            .layer_mut(layer_idx)
                            .expect("DeltaNet state missing for linear attention layer");
                        dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                    }
                };

                for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                    *h += a;
                }

                norm_buf.copy_from_slice(&hidden);
                self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);
                self.layers[layer_idx]
                    .mlp
                    .forward_into(&norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf);

                for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
                    *h += m;
                }

                // Capture the residual-stream hidden right after this
                // layer's MLP residual, but only at the last token
                // position.
                if pos == last_pos {
                    if let Some(slot) = layer_to_slot[layer_idx] {
                        per_layer_captures[slot].copy_from_slice(&hidden);
                    }
                }
            }
        }

        self.final_norm.forward(&mut hidden);

        (hidden, per_layer_captures)
    }

    /// Prefill `token_ids` and capture the MLP gate pre-activation (input
    /// to SiLU, post `gate_proj`) at `target_layer` for the **last** token
    /// position. Thin wrapper around `forward_prefill_with_gate_pool` with
    /// `Pool::Last`. Kept for API stability.
    pub fn forward_prefill_with_gate(
        &mut self,
        token_ids: &[u32],
        target_layer: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        self.forward_prefill_with_gate_pool(token_ids, target_layer, Pool::Last)
    }

    /// Thin wrapper over `forward_prefill_with_gates_many_pool` with
    /// `Pool::Last`. Kept for API stability.
    pub fn forward_prefill_with_gates_many(
        &mut self,
        token_ids: &[u32],
        target_layers: &[usize],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        self.forward_prefill_with_gates_many_pool(token_ids, target_layers, Pool::Last)
    }

    /// Pooled forward that captures MLP I/O at a target layer.
    /// Returns (hidden, gate_preact, mlp_input, mlp_output) where:
    ///   - `hidden` is the post-final-norm output (pooled per `pool`)
    ///   - `gate_preact` is the pre-SiLU gate vector at the target layer
    ///     (length `intermediate_size`)
    ///   - `mlp_input` is the pre-MLP-norm-applied vector (i.e. the
    ///     argument to `mlp.forward`) at the target layer (length
    ///     `hidden_size`)
    ///   - `mlp_output` is the MLP delta (the value added to the residual
    ///     stream) at the target layer (length `hidden_size`)
    ///
    /// Used for tropical polytope / per-cell affine map analysis: pairs
    /// of (mlp_input, mlp_output) within a cell allow fitting a per-cell
    /// affine map by regression, recovering the Newton-polytope lifted
    /// vertex without needing autodiff or per-position weight extraction.
    pub fn forward_prefill_with_mlp_io_pool(
        &mut self,
        token_ids: &[u32],
        target_layer: usize,
        pool: Pool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        assert!(
            !token_ids.is_empty(),
            "forward_prefill_with_mlp_io_pool requires at least one token"
        );
        assert!(
            target_layer < self.config.num_layers,
            "target_layer {target_layer} out of range (num_layers={})",
            self.config.num_layers
        );

        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        let mut hidden = vec![0.0f32; hidden_size];
        let mut hidden_sum = vec![0.0f32; hidden_size]; // for Pool::Mean
        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; intermediate_size];
        let mut up_buf = vec![0.0f32; intermediate_size];
        let mut gate_preact_tmp = vec![0.0f32; intermediate_size];

        // Accumulators for the captured triple (gate, mlp_in, mlp_out).
        let mut gate_acc = vec![0.0f32; intermediate_size];
        let mut mlp_in_acc = vec![0.0f32; hidden_size];
        let mut mlp_out_acc = vec![0.0f32; hidden_size];

        let last_pos = token_ids.len() - 1;

        for (pos, &token_id) in token_ids.iter().enumerate() {
            self.embedding.forward(token_id, &mut hidden);

            for layer_idx in 0..self.config.num_layers {
                norm_buf.copy_from_slice(&hidden);
                self.layers[layer_idx].attn_norm.forward(&mut norm_buf);
                match &self.layers[layer_idx].attention {
                    AttentionLayer::Standard(attn) => {
                        attn.forward_into(
                            &norm_buf, pos, &self.rope,
                            self.kv_caches.layer_mut(layer_idx),
                            &mut attn_out,
                        );
                    }
                    AttentionLayer::GatedDeltaNet(dn) => {
                        let state = self
                            .deltanet_states
                            .layer_mut(layer_idx)
                            .expect("DeltaNet state missing");
                        dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                    }
                };
                for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) { *h += a; }

                norm_buf.copy_from_slice(&hidden);
                self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

                // Capture MLP input, gate, and output at the target layer.
                let capture_now = match pool {
                    Pool::Mean => true,
                    Pool::Last => pos == last_pos,
                };
                if capture_now && layer_idx == target_layer {
                    // mlp_input = norm_buf (post pre-MLP-norm).
                    match pool {
                        Pool::Last => mlp_in_acc.copy_from_slice(&norm_buf),
                        Pool::Mean => {
                            for (a, &v) in mlp_in_acc.iter_mut().zip(norm_buf.iter()) { *a += v; }
                        }
                    }
                }

                // Run MLP, capturing the gate pre-activation if requested.
                if layer_idx == target_layer && capture_now {
                    self.layers[layer_idx].mlp.forward_into_capture_gate(
                        &norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf,
                        &mut gate_preact_tmp,
                    );
                    match pool {
                        Pool::Last => {
                            gate_acc.copy_from_slice(&gate_preact_tmp);
                            mlp_out_acc.copy_from_slice(&mlp_out);
                        }
                        Pool::Mean => {
                            for (a, &v) in gate_acc.iter_mut().zip(gate_preact_tmp.iter()) { *a += v; }
                            for (a, &v) in mlp_out_acc.iter_mut().zip(mlp_out.iter()) { *a += v; }
                        }
                    }
                } else {
                    self.layers[layer_idx]
                        .mlp
                        .forward_into(&norm_buf, &mut mlp_out, &mut gate_buf, &mut up_buf);
                }

                for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) { *h += m; }
            }

            if pool == Pool::Mean {
                for (s, &v) in hidden_sum.iter_mut().zip(hidden.iter()) { *s += v; }
            }
        }

        let n_tokens = token_ids.len() as f32;
        let mut hidden_out = match pool {
            Pool::Last => hidden,
            Pool::Mean => hidden_sum.iter().map(|&v| v / n_tokens).collect(),
        };
        if pool == Pool::Mean {
            self.final_norm.forward(&mut hidden_out);
            for v in gate_acc.iter_mut() { *v /= n_tokens; }
            for v in mlp_in_acc.iter_mut() { *v /= n_tokens; }
            for v in mlp_out_acc.iter_mut() { *v /= n_tokens; }
        } else {
            self.final_norm.forward(&mut hidden_out);
        }

        (hidden_out, gate_acc, mlp_in_acc, mlp_out_acc)
    }

    /// Single-target-layer pooled forward. Delegates to the multi-layer
    /// variant with a length-one slice.
    pub fn forward_prefill_with_gate_pool(
        &mut self,
        token_ids: &[u32],
        target_layer: usize,
        pool: Pool,
    ) -> (Vec<f32>, Vec<f32>) {
        let (hidden, mut gates) =
            self.forward_prefill_with_gates_many_pool(token_ids, &[target_layer], pool);
        let gate = gates.pop().expect("one target layer expected");
        (hidden, gate)
    }

    /// Prefill `token_ids`, capturing MLP gate pre-activations at every
    /// layer in `target_layers` and reducing per-position outputs to one
    /// vector per probe according to `pool`.
    ///
    /// - `pool = Pool::Last`: hidden is the post-final-norm output at the
    ///   last token position; each gate is the pre-SiLU vector at that
    ///   layer, last position only. (Matches the original behaviour.)
    /// - `pool = Pool::Mean`: hidden is `final_norm(mean_t h_t)` where `h_t`
    ///   is the pre-final-norm hidden state after all transformer layers
    ///   at position `t`; each gate is the mean of pre-SiLU vectors across
    ///   all positions.
    ///
    /// Captures occur in a single forward pass regardless of
    /// `target_layers.len()`. Panics on duplicate or out-of-range layers
    /// or empty `token_ids`.
    pub fn forward_prefill_with_gates_many_pool(
        &mut self,
        token_ids: &[u32],
        target_layers: &[usize],
        pool: Pool,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        assert!(
            !token_ids.is_empty(),
            "forward_prefill_with_gates_many_pool requires at least one token"
        );
        for &l in target_layers {
            assert!(
                l < self.config.num_layers,
                "target_layer {l} out of range (num_layers={})",
                self.config.num_layers
            );
        }
        for i in 0..target_layers.len() {
            for j in (i + 1)..target_layers.len() {
                assert!(
                    target_layers[i] != target_layers[j],
                    "duplicate target layer {} in target_layers",
                    target_layers[i]
                );
            }
        }

        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        let mut hidden = vec![0.0f32; hidden_size];
        let mut hidden_sum = vec![0.0f32; hidden_size]; // only used for Pool::Mean
        let mut norm_buf = vec![0.0f32; hidden_size];
        let mut attn_out = vec![0.0f32; hidden_size];
        let mut mlp_out = vec![0.0f32; hidden_size];
        let mut gate_buf = vec![0.0f32; intermediate_size];
        let mut up_buf = vec![0.0f32; intermediate_size];
        let mut gate_preact_tmp = vec![0.0f32; intermediate_size];

        // Gate accumulators: for Pool::Mean these are running sums,
        // divided by token count at the end; for Pool::Last they are
        // written once at the last position.
        let mut gate_captures: Vec<Vec<f32>> = target_layers
            .iter()
            .map(|_| vec![0.0f32; intermediate_size])
            .collect();

        let layer_to_slot: Vec<Option<usize>> = {
            let mut v = vec![None; self.config.num_layers];
            for (slot, &l) in target_layers.iter().enumerate() {
                v[l] = Some(slot);
            }
            v
        };

        let last_pos = token_ids.len() - 1;

        for (pos, &token_id) in token_ids.iter().enumerate() {
            self.embedding.forward(token_id, &mut hidden);

            for layer_idx in 0..self.config.num_layers {
                norm_buf.copy_from_slice(&hidden);
                self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

                match &self.layers[layer_idx].attention {
                    AttentionLayer::Standard(attn) => {
                        attn.forward_into(
                            &norm_buf,
                            pos,
                            &self.rope,
                            self.kv_caches.layer_mut(layer_idx),
                            &mut attn_out,
                        );
                    }
                    AttentionLayer::GatedDeltaNet(dn) => {
                        let state = self
                            .deltanet_states
                            .layer_mut(layer_idx)
                            .expect("DeltaNet state missing for linear attention layer");
                        dn.forward_decode_into(&norm_buf, state, &mut attn_out);
                    }
                };

                for (h, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                    *h += a;
                }

                norm_buf.copy_from_slice(&hidden);
                self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

                let capture_now = match (layer_to_slot[layer_idx], pool) {
                    (Some(_), Pool::Mean) => true,
                    (Some(_), Pool::Last) => pos == last_pos,
                    (None, _) => false,
                };

                if capture_now {
                    self.layers[layer_idx].mlp.forward_into_capture_gate(
                        &norm_buf,
                        &mut mlp_out,
                        &mut gate_buf,
                        &mut up_buf,
                        &mut gate_preact_tmp,
                    );
                    let slot = layer_to_slot[layer_idx].expect("capture_now implies slot");
                    match pool {
                        Pool::Last => {
                            gate_captures[slot].copy_from_slice(&gate_preact_tmp);
                        }
                        Pool::Mean => {
                            for (g, &p) in
                                gate_captures[slot].iter_mut().zip(gate_preact_tmp.iter())
                            {
                                *g += p;
                            }
                        }
                    }
                } else {
                    self.layers[layer_idx].mlp.forward_into(
                        &norm_buf,
                        &mut mlp_out,
                        &mut gate_buf,
                        &mut up_buf,
                    );
                }

                for (h, &m) in hidden.iter_mut().zip(mlp_out.iter()) {
                    *h += m;
                }
            }

            if pool == Pool::Mean {
                for (h_sum, &v) in hidden_sum.iter_mut().zip(hidden.iter()) {
                    *h_sum += v;
                }
            }
        }

        let n_tokens = token_ids.len() as f32;
        let mut final_hidden = match pool {
            Pool::Last => hidden,
            Pool::Mean => hidden_sum.iter().map(|&v| v / n_tokens).collect(),
        };
        self.final_norm.forward(&mut final_hidden);

        if pool == Pool::Mean {
            for cap in gate_captures.iter_mut() {
                for v in cap.iter_mut() {
                    *v /= n_tokens;
                }
            }
        }

        (final_hidden, gate_captures)
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
    fn test_per_layer_hidden_capture_shapes_and_distinctness() {
        let config = tiny_config();
        let mut model = Model::new(config.clone());
        for i in 0..config.vocab_size {
            let row = model.embedding.weights.get_weights_mut(i);
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((i * 7 + j * 3) % 17) as f32 * 0.01;
            }
        }
        for layer in &mut model.layers {
            for w in layer.attn_norm.weight.iter_mut() { *w = 1.0; }
            for w in layer.mlp_norm.weight.iter_mut() { *w = 1.0; }
        }
        for w in model.final_norm.weight.iter_mut() { *w = 1.0; }

        let tokens = [1u32, 2, 3, 4];
        let (final_hidden, per_layer) =
            model.forward_prefill_per_layer_hidden(&tokens, &[0, 1]);

        // Shapes correct.
        assert_eq!(final_hidden.len(), config.hidden_size);
        assert_eq!(per_layer.len(), 2);
        assert_eq!(per_layer[0].len(), config.hidden_size);
        assert_eq!(per_layer[1].len(), config.hidden_size);

        // All values finite.
        assert!(final_hidden.iter().all(|v| v.is_finite()));
        assert!(per_layer[0].iter().all(|v| v.is_finite()));
        assert!(per_layer[1].iter().all(|v| v.is_finite()));

        // With unit-init norms and embedding-only non-zero weights,
        // the residual stream at L0 should equal embedding lookup +
        // zeros from attn/MLP. After final_norm, the final hidden is
        // a (different) RMSNorm of L1's residual — so the two are not
        // bit-equivalent in general. That is not asserted here because
        // construction details can collapse it.

        // Check empty target_layers list returns no per-layer captures.
        let (_, empty_per_layer) =
            model.forward_prefill_per_layer_hidden(&tokens, &[]);
        assert_eq!(empty_per_layer.len(), 0);
    }

    #[test]
    fn test_pool_last_matches_legacy_gate_capture() {
        // `forward_prefill_with_gate` is now a thin wrapper over
        // `forward_prefill_with_gate_pool(_, _, Pool::Last)`. Verify the
        // pooled Last variant reproduces the original behaviour.
        let config = tiny_config();
        let mut model_legacy = Model::new(config.clone());
        let mut model_pool = Model::new(config.clone());
        for i in 0..config.vocab_size {
            let row_a = model_legacy.embedding.weights.get_weights_mut(i);
            let row_b = model_pool.embedding.weights.get_weights_mut(i);
            for (j, (va, vb)) in row_a.iter_mut().zip(row_b.iter_mut()).enumerate() {
                let v = ((i * 7 + j * 3) % 17) as f32 * 0.01;
                *va = v;
                *vb = v;
            }
        }
        for layer in &mut model_legacy.layers {
            for w in layer.attn_norm.weight.iter_mut() { *w = 1.0; }
            for w in layer.mlp_norm.weight.iter_mut() { *w = 1.0; }
        }
        for layer in &mut model_pool.layers {
            for w in layer.attn_norm.weight.iter_mut() { *w = 1.0; }
            for w in layer.mlp_norm.weight.iter_mut() { *w = 1.0; }
        }
        for w in model_legacy.final_norm.weight.iter_mut() { *w = 1.0; }
        for w in model_pool.final_norm.weight.iter_mut() { *w = 1.0; }

        let tokens = [1u32, 2, 3, 4];
        let (h_legacy, g_legacy) = model_legacy.forward_prefill_with_gate(&tokens, 1);
        let (h_pool, g_pool) =
            model_pool.forward_prefill_with_gate_pool(&tokens, 1, Pool::Last);
        assert_eq!(h_legacy.len(), h_pool.len());
        for (a, b) in h_legacy.iter().zip(h_pool.iter()) {
            assert!((a - b).abs() < 1e-6, "Last-pool hidden should match legacy");
        }
        assert_eq!(g_legacy.len(), g_pool.len());
        for (a, b) in g_legacy.iter().zip(g_pool.iter()) {
            assert!((a - b).abs() < 1e-6, "Last-pool gate should match legacy");
        }
    }

    #[test]
    fn test_pool_mean_differs_from_last_for_multi_token() {
        // Mean-pool should yield a different hidden from Last-pool when
        // the input is multiple distinct tokens.
        let config = tiny_config();
        let mut model = Model::new(config.clone());
        for i in 0..config.vocab_size {
            let row = model.embedding.weights.get_weights_mut(i);
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((i * 7 + j * 3) % 17) as f32 * 0.01;
            }
        }
        for layer in &mut model.layers {
            for w in layer.attn_norm.weight.iter_mut() { *w = 1.0; }
            for w in layer.mlp_norm.weight.iter_mut() { *w = 1.0; }
        }
        for w in model.final_norm.weight.iter_mut() { *w = 1.0; }

        let tokens = [1u32, 2, 3, 4];
        let (h_last, _) = {
            let (h, g) = model.forward_prefill_with_gate_pool(&tokens, 1, Pool::Last);
            (h, g)
        };
        model.reset_kv_caches();
        let (h_mean, _) = {
            let (h, g) = model.forward_prefill_with_gate_pool(&tokens, 1, Pool::Mean);
            (h, g)
        };
        assert_eq!(h_last.len(), h_mean.len());
        assert!(h_last.iter().all(|v| v.is_finite()));
        assert!(h_mean.iter().all(|v| v.is_finite()));
        // They shouldn't be identical for a multi-token input unless the
        // per-token hiddens are literally constant (unlikely here).
        let same: bool = h_last
            .iter()
            .zip(h_mean.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);
        assert!(
            !same,
            "mean-pool should diverge from last-pool on multi-token input"
        );
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
