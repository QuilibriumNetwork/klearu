use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use klearu_llm::generate::pipeline::GenerateConfig;
use klearu_llm::generate::sampler::SamplerConfig;
use klearu_llm::model::block::AttentionLayer;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[derive(Serialize)]
pub struct ModelInfoResponse {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub model_type: Option<String>,
}

pub async fn model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfoResponse> {
    let pipeline = state.pipeline.lock().await;
    let c = &pipeline.model.config;
    Json(ModelInfoResponse {
        vocab_size: c.vocab_size,
        hidden_size: c.hidden_size,
        num_layers: c.num_layers,
        num_heads: c.num_heads,
        num_kv_heads: c.num_kv_heads,
        head_dim: c.head_dim,
        intermediate_size: c.intermediate_size,
        max_seq_len: c.max_seq_len,
        rms_norm_eps: c.rms_norm_eps,
        tie_word_embeddings: c.tie_word_embeddings,
        model_type: c.model_type.clone(),
    })
}

/// Serve a model config file (tokenizer.json, tokenizer_config.json, config.json).
/// These are public model metadata, not user data.
pub async fn model_file(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> impl IntoResponse {
    // Only allow specific safe files
    let allowed = ["tokenizer.json", "tokenizer_config.json", "config.json"];
    if !allowed.contains(&filename.as_str()) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let path = state.model_dir.join(&filename);
    match tokio::fs::read(&path).await {
        Ok(data) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            data,
        )
            .into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Pre-serialize the embedding weight table as contiguous f32 LE bytes.
///
/// Layout: `vocab_size × hidden_size` f32 values, row-major, little-endian.
pub fn serialize_embedding_weights(model: &klearu_llm::model::Model) -> Vec<u8> {
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;

    let mut data = Vec::with_capacity(vocab_size * hidden_size * 4);
    for i in 0..vocab_size {
        let row = model.embedding.weights.get_weights(i);
        for &v in &row[..hidden_size] {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    data
}

/// Pre-serialize model weights (excluding embeddings) as contiguous f32 LE bytes
/// for WASM party 0.
///
/// Embeddings are excluded because:
///   - They are already served separately via /api/weights/embeddings
///   - The forward pass takes embedding shares as input, not raw embeddings
///   - Excluding them saves ~1 GB for large-vocab models (e.g. Qwen3.5 248K vocab)
///
/// Binary format (matching Party0State::load_weights):
///   1. Per layer:
///      a. attn_norm weights: hidden_size f32s
///      b. mlp_norm weights: hidden_size f32s
///      c. Attention projections (Standard or GatedDeltaNet)
///      d. MLP: gate_proj, up_proj, down_proj
///   2. final_norm weights: hidden_size f32s
///   3. lm_head (if !tie_word_embeddings): vocab_size × hidden_size f32s
pub fn serialize_model_weights(model: &klearu_llm::model::Model) -> Vec<u8> {
    let mut data = Vec::new();

    // 1. Per-layer weights
    for layer in &model.layers {
        // attn_norm
        for &v in &layer.attn_norm.weight {
            data.extend_from_slice(&v.to_le_bytes());
        }
        // mlp_norm
        for &v in &layer.mlp_norm.weight {
            data.extend_from_slice(&v.to_le_bytes());
        }

        // Attention projections
        match &layer.attention {
            AttentionLayer::Standard(attn) => {
                serialize_linear(&attn.q_proj, &mut data);
                serialize_linear(&attn.k_proj, &mut data);
                serialize_linear(&attn.v_proj, &mut data);
                serialize_linear(&attn.o_proj, &mut data);
            }
            AttentionLayer::GatedDeltaNet(dn) => {
                serialize_linear(&dn.in_proj_qkv, &mut data);
                serialize_linear(&dn.in_proj_z, &mut data);
                serialize_linear(&dn.in_proj_a, &mut data);
                serialize_linear(&dn.in_proj_b, &mut data);
                serialize_linear(&dn.out_proj, &mut data);
                for &v in &dn.conv_weight {
                    data.extend_from_slice(&v.to_le_bytes());
                }
                for &v in &dn.dt_bias {
                    data.extend_from_slice(&v.to_le_bytes());
                }
                for &v in &dn.a_log {
                    data.extend_from_slice(&v.to_le_bytes());
                }
                for &v in &dn.norm_weight {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
        }

        // MLP projections
        serialize_linear(&layer.mlp.gate_proj, &mut data);
        serialize_linear(&layer.mlp.up_proj, &mut data);
        serialize_linear(&layer.mlp.down_proj, &mut data);
    }

    // 2. final_norm
    for &v in &model.final_norm.weight {
        data.extend_from_slice(&v.to_le_bytes());
    }

    // 3. lm_head (if not tie_word_embeddings)
    if let Some(ref head) = model.lm_head {
        serialize_linear(head, &mut data);
    }

    data
}

fn serialize_linear(linear: &klearu_llm::model::linear::Linear, out: &mut Vec<u8>) {
    for i in 0..linear.out_features() {
        let row = linear.weights.get_weights(i);
        for &v in &row[..linear.in_features()] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

// --- Embedding-seeded generation ---

fn default_emb_max_tokens() -> usize { 20 }
fn default_emb_temperature() -> f32 { 0.4 }
fn default_emb_top_k() -> usize { 10 }
fn default_emb_repetition_penalty() -> f32 { 1.15 }

#[derive(Deserialize)]
pub struct GenerateFromEmbeddingRequest {
    pub embedding: Vec<f32>,
    #[serde(default = "default_emb_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_emb_temperature")]
    pub temperature: f32,
    #[serde(default = "default_emb_top_k")]
    pub top_k: usize,
    #[serde(default = "default_emb_repetition_penalty")]
    pub repetition_penalty: f32,
}

#[derive(Serialize)]
pub struct GenerateFromEmbeddingResponse {
    pub text: String,
    pub tokens: Vec<TokenProb>,
}

pub async fn generate_from_embedding(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateFromEmbeddingRequest>,
) -> Result<Json<GenerateFromEmbeddingResponse>, StatusCode> {
    let mut pipeline = state.pipeline.lock().await;
    let hidden_size = pipeline.model.config.hidden_size;

    if req.embedding.len() != hidden_size {
        tracing::warn!(
            "generate_from_embedding: expected {}D embedding, got {}D",
            hidden_size,
            req.embedding.len()
        );
        return Err(StatusCode::BAD_REQUEST);
    }

    pipeline.model.reset_kv_caches();

    // Forward pass with the full hidden_size-D embedding directly
    let mut logits = pipeline.model.forward_decode_with_embedding(&req.embedding, 0);

    // Extract top-5 tokens from initial logits for the response
    let initial_tokens = {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in probs.iter_mut() {
                *p *= inv;
            }
        }
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let k = 5.min(indices.len());
        indices[..k]
            .iter()
            .enumerate()
            .map(|(rank, &idx)| {
                let token_text = pipeline
                    .tokenizer
                    .decode(&[idx as u32])
                    .unwrap_or_default();
                TokenProb {
                    token: token_text,
                    probability: probs[idx],
                    rank,
                }
            })
            .collect::<Vec<_>>()
    };

    // Sample first token from initial logits
    let mut rng = rand::thread_rng();
    let sampler_config = SamplerConfig {
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: 1.0,
        repetition_penalty: req.repetition_penalty,
    };

    let mut generated_ids: Vec<u32> = Vec::new();
    let first_token = klearu_llm::generate::sampler::sample(&mut logits, &sampler_config, &generated_ids, &mut rng);
    generated_ids.push(first_token);

    // Continue autoregressively
    let max_tokens = req.max_tokens.min(128);
    let mut last_token = first_token;
    for pos in 1..max_tokens {
        let mut logits = pipeline.model.forward_decode(last_token, pos);
        let token = klearu_llm::generate::sampler::sample(&mut logits, &sampler_config, &generated_ids, &mut rng);

        if state.eos_token_ids.contains(&token) {
            break;
        }
        generated_ids.push(token);
        last_token = token;
    }

    let text = pipeline
        .tokenizer
        .decode(&generated_ids)
        .unwrap_or_default();

    Ok(Json(GenerateFromEmbeddingResponse {
        text,
        tokens: initial_tokens,
    }))
}

// --- Token embedding API ---

#[derive(Deserialize)]
pub struct EmbedTokensRequest {
    pub tokens: Vec<String>,
}

#[derive(Serialize)]
pub struct EmbedTokensResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub hidden_size: usize,
}

pub async fn embed_tokens(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedTokensRequest>,
) -> Json<EmbedTokensResponse> {
    let pipeline = state.pipeline.lock().await;
    let hidden_size = pipeline.model.config.hidden_size;
    let mut embeddings = Vec::with_capacity(req.tokens.len());

    for token_str in &req.tokens {
        match pipeline.tokenizer.encode(token_str) {
            Ok(ids) => {
                if ids.is_empty() {
                    embeddings.push(vec![0.0; hidden_size]);
                } else {
                    // Average embeddings of all tokens for multi-token inputs
                    let mut avg = vec![0.0f32; hidden_size];
                    let mut tmp = vec![0.0f32; hidden_size];
                    for &id in &ids {
                        pipeline.model.embedding.forward(id, &mut tmp);
                        for (a, &t) in avg.iter_mut().zip(tmp.iter()) {
                            *a += t;
                        }
                    }
                    let n = ids.len() as f32;
                    for v in avg.iter_mut() {
                        *v /= n;
                    }
                    embeddings.push(avg);
                }
            }
            Err(_) => {
                embeddings.push(vec![0.0; hidden_size]);
            }
        }
    }

    Json(EmbedTokensResponse {
        embeddings,
        hidden_size,
    })
}

// --- Contextual text embedding API ---

#[derive(Deserialize)]
pub struct EmbedTextRequest {
    pub texts: Vec<String>,
}

#[derive(Serialize)]
pub struct EmbedTextResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub hidden_size: usize,
}

/// Embed text strings by running them through the full transformer model.
/// Returns the post-final-norm hidden state at the last token position for
/// each input — a contextual semantic embedding.
pub async fn embed_text(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedTextRequest>,
) -> Json<EmbedTextResponse> {
    let mut pipeline = state.pipeline.lock().await;
    let hidden_size = pipeline.model.config.hidden_size;
    let mut embeddings = Vec::with_capacity(req.texts.len());

    for text in &req.texts {
        match pipeline.tokenizer.encode(text) {
            Ok(ids) => {
                if ids.is_empty() {
                    embeddings.push(vec![0.0; hidden_size]);
                } else {
                    pipeline.model.reset_kv_caches();
                    let hidden = pipeline.model.forward_prefill_hidden(&ids);
                    embeddings.push(hidden);
                }
            }
            Err(_) => {
                embeddings.push(vec![0.0; hidden_size]);
            }
        }
    }

    Json(EmbedTextResponse {
        embeddings,
        hidden_size,
    })
}

// --- Combined embed + generate API ---

#[derive(Deserialize)]
pub struct EmbedAndGenerateRequest {
    pub text: String,
    #[serde(default = "default_emb_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_emb_temperature")]
    pub temperature: f32,
    #[serde(default = "default_emb_top_k")]
    pub top_k: usize,
    #[serde(default = "default_emb_repetition_penalty")]
    pub repetition_penalty: f32,
}

#[derive(Serialize)]
pub struct EmbedAndGenerateResponse {
    pub embedding: Vec<f32>,
    pub text: String,
    pub tokens: Vec<TokenProb>,
}

/// Embed text through the full transformer and generate a continuation.
///
/// Single pass: tokenizes the input, prefills through all layers (populating
/// KV caches), captures the hidden state as the embedding, then samples from
/// the resulting logits and continues autoregressively. The generated text
/// is a natural continuation of the input, maintaining semantic consistency.
pub async fn embed_and_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedAndGenerateRequest>,
) -> Result<Json<EmbedAndGenerateResponse>, StatusCode> {
    let mut pipeline = state.pipeline.lock().await;

    let input_ids = match pipeline.tokenizer.encode(&req.text) {
        Ok(ids) if !ids.is_empty() => ids,
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    pipeline.model.reset_kv_caches();

    // Single prefill: get hidden state (embedding) + logits, KV caches populated
    let (embedding, mut logits) = pipeline.model.forward_prefill_hidden_and_logits(&input_ids);

    // Extract top-5 tokens from the logits
    let initial_tokens = {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in probs.iter_mut() {
                *p *= inv;
            }
        }
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let k = 5.min(indices.len());
        indices[..k]
            .iter()
            .enumerate()
            .map(|(rank, &idx)| {
                let token_text = pipeline
                    .tokenizer
                    .decode(&[idx as u32])
                    .unwrap_or_default();
                TokenProb {
                    token: token_text,
                    probability: probs[idx],
                    rank,
                }
            })
            .collect::<Vec<_>>()
    };

    // Sample first token
    let mut rng = rand::thread_rng();
    let sampler_config = SamplerConfig {
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: 1.0,
        repetition_penalty: req.repetition_penalty,
    };

    // Seed with input tokens so repetition penalty discourages echoing the prompt
    let mut generated_ids: Vec<u32> = input_ids.clone();
    let first_token = klearu_llm::generate::sampler::sample(
        &mut logits, &sampler_config, &generated_ids, &mut rng,
    );
    generated_ids.push(first_token);

    // Continue autoregressively from where prefill left off
    let max_tokens = req.max_tokens.min(128);
    let prefill_len = input_ids.len();
    let mut last_token = first_token;
    for pos in 1..max_tokens {
        let mut logits = pipeline.model.forward_decode(last_token, prefill_len + pos - 1);
        let token = klearu_llm::generate::sampler::sample(
            &mut logits, &sampler_config, &generated_ids, &mut rng,
        );
        if state.eos_token_ids.contains(&token) {
            break;
        }
        generated_ids.push(token);
        last_token = token;
    }

    // Decode only the generated tokens, not the input prefix
    let text = pipeline
        .tokenizer
        .decode(&generated_ids[prefill_len..])
        .unwrap_or_default();

    Ok(Json(EmbedAndGenerateResponse {
        embedding,
        text,
        tokens: initial_tokens,
    }))
}

// --- Text generation API ---

fn default_max_tokens() -> usize { 64 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 1.0 }
fn default_top_k_tokens() -> usize { 5 }

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

#[derive(Serialize)]
pub struct GenerateResponse {
    pub text: String,
}

pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let mut pipeline = state.pipeline.lock().await;
    let mut rng = rand::thread_rng();

    let config = GenerateConfig {
        max_new_tokens: req.max_tokens,
        sampler: SamplerConfig {
            temperature: req.temperature,
            top_k: req.top_k,
            top_p: req.top_p,
            repetition_penalty: 1.0,
        },
        eos_token_ids: state.eos_token_ids.clone(),
    };

    let text = pipeline
        .generate(&req.prompt, &config, &mut rng)
        .unwrap_or_default();

    Json(GenerateResponse { text })
}

#[derive(Deserialize)]
pub struct TopTokensRequest {
    pub prompt: String,
    #[serde(default = "default_top_k_tokens")]
    pub top_k: usize,
}

#[derive(Serialize)]
pub struct TokenProb {
    pub token: String,
    pub probability: f32,
    pub rank: usize,
}

#[derive(Serialize)]
pub struct TopTokensResponse {
    pub tokens: Vec<TokenProb>,
}

pub async fn top_tokens(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TopTokensRequest>,
) -> Json<TopTokensResponse> {
    let mut pipeline = state.pipeline.lock().await;

    let input_ids = match pipeline.tokenizer.encode(&req.prompt) {
        Ok(ids) => ids,
        Err(_) => return Json(TopTokensResponse { tokens: vec![] }),
    };

    pipeline.model.reset_kv_caches();

    if input_ids.len() > 1 {
        let _ = pipeline
            .model
            .forward_prefill(&input_ids[..input_ids.len() - 1]);
    }

    let last_input = *input_ids.last().unwrap_or(&0);
    let position = if input_ids.is_empty() {
        0
    } else {
        input_ids.len() - 1
    };
    let logits = pipeline.model.forward_decode(last_input, position);

    // Softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }

    // Sort indices by probability descending
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let k = req.top_k.min(indices.len());
    let tokens: Vec<TokenProb> = indices[..k]
        .iter()
        .enumerate()
        .map(|(rank, &idx)| {
            let token_text = pipeline
                .tokenizer
                .decode(&[idx as u32])
                .unwrap_or_default();
            TokenProb {
                token: token_text,
                probability: probs[idx],
                rank,
            }
        })
        .collect();

    Json(TopTokensResponse { tokens })
}
