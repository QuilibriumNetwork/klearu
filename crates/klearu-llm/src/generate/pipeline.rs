use std::path::Path;

use rand::Rng;

use crate::error::Result;
use crate::model::Model;
use crate::tokenizer::Tokenizer;

use super::sampler::{SamplerConfig, sample};

/// Detect EOS token ID from a model directory's config files.
///
/// Priority:
/// 1. `tokenizer_config.json` → `eos_token_id` (integer)
/// 2. `tokenizer_config.json` → resolve `eos_token` string via `added_tokens_decoder`
/// 3. `config.json` → top-level `eos_token_id`
/// 4. `config.json` → `text_config.eos_token_id`
/// 5. Fallback to 2 (common LLaMA default)
pub fn detect_eos_token(model_dir: &Path) -> Option<u32> {
    let tok_config_path = model_dir.join("tokenizer_config.json");
    if let Ok(content) = std::fs::read_to_string(&tok_config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            // Try "eos_token_id" integer first
            if let Some(id) = json.get("eos_token_id").and_then(|v| v.as_u64()) {
                return Some(id as u32);
            }

            // Try resolving "eos_token" string via "added_tokens_decoder"
            let eos_token_str = json.get("eos_token").and_then(|v| {
                // Can be a plain string or an object with "content" field
                v.as_str().map(|s| s.to_string())
                    .or_else(|| v.get("content").and_then(|c| c.as_str()).map(|s| s.to_string()))
            });

            if let Some(eos_str) = eos_token_str {
                if let Some(decoder) = json.get("added_tokens_decoder").and_then(|v| v.as_object()) {
                    for (id_str, info) in decoder {
                        if let Some(content) = info.get("content").and_then(|c| c.as_str()) {
                            if content == eos_str {
                                if let Ok(id) = id_str.parse::<u32>() {
                                    return Some(id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Try config.json (supports nested Qwen3.5 format)
    let config_path = model_dir.join("config.json");
    if let Ok(content) = std::fs::read_to_string(config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(id) = json.get("eos_token_id").and_then(|v| v.as_u64()) {
                return Some(id as u32);
            }
            if let Some(tc) = json.get("text_config") {
                if let Some(id) = tc.get("eos_token_id").and_then(|v| v.as_u64()) {
                    return Some(id as u32);
                }
            }
        }
    }

    // Fallback
    Some(2)
}

/// Generation configuration.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub sampler: SamplerConfig,
    /// EOS token ID. Generation stops when this token is produced.
    pub eos_token_id: Option<u32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            sampler: SamplerConfig::default(),
            eos_token_id: Some(2), // common default for LLaMA
        }
    }
}

/// Full generation pipeline: tokenize -> prefill -> decode -> detokenize.
pub struct Pipeline {
    pub model: Model,
    pub tokenizer: Tokenizer,
}

impl Pipeline {
    /// Load a pipeline from a model directory containing config.json, tokenizer.json, and .safetensors.
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let model = crate::weight::load_model(model_dir)?;
        let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json"))?;
        Ok(Self { model, tokenizer })
    }

    /// Create a pipeline from an already-loaded model and tokenizer.
    pub fn new(model: Model, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Generate text from a prompt (non-streaming).
    pub fn generate(
        &mut self,
        prompt: &str,
        config: &GenerateConfig,
        rng: &mut impl Rng,
    ) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt)?;
        let output_ids = self.generate_ids(&input_ids, config, rng);
        let new_tokens = &output_ids[input_ids.len()..];
        self.tokenizer.decode(new_tokens)
    }

    /// Generate text with streaming: calls `on_token` for each new token as it's generated.
    ///
    /// `on_token` receives the token text and returns `true` to continue or `false` to stop.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        config: &GenerateConfig,
        rng: &mut impl Rng,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt)?;
        self.model.reset_kv_caches();

        let mut all_ids: Vec<u32> = input_ids.to_vec();
        let mut generated = Vec::new();

        // Prefill
        if input_ids.len() > 1 {
            let _ = self.model.forward_prefill(&input_ids[..input_ids.len() - 1]);
        }

        let last_input = *input_ids.last().unwrap_or(&0);
        let position = if input_ids.is_empty() { 0 } else { input_ids.len() - 1 };
        let mut logits = self.model.forward_decode(last_input, position);

        for step in 0..config.max_new_tokens {
            let next_token = sample(&mut logits, &config.sampler, &all_ids, rng);

            if config.eos_token_id == Some(next_token) {
                break;
            }

            all_ids.push(next_token);
            generated.push(next_token);

            // Decode this token to text and stream it
            let token_text = self.tokenizer.decode(&[next_token])?;
            if !on_token(&token_text) {
                break;
            }

            let pos = input_ids.len() + step;
            logits = self.model.forward_decode(next_token, pos);
        }

        self.tokenizer.decode(&generated)
    }

    /// Generate token IDs from input token IDs (non-streaming).
    pub fn generate_ids(
        &mut self,
        input_ids: &[u32],
        config: &GenerateConfig,
        rng: &mut impl Rng,
    ) -> Vec<u32> {
        self.model.reset_kv_caches();

        let mut all_ids: Vec<u32> = input_ids.to_vec();

        if input_ids.len() > 1 {
            let _ = self.model.forward_prefill(&input_ids[..input_ids.len() - 1]);
        }

        let last_input = *input_ids.last().unwrap_or(&0);
        let position = if input_ids.is_empty() { 0 } else { input_ids.len() - 1 };
        let mut logits = self.model.forward_decode(last_input, position);

        for step in 0..config.max_new_tokens {
            let next_token = sample(&mut logits, &config.sampler, &all_ids, rng);

            if config.eos_token_id == Some(next_token) {
                break;
            }

            all_ids.push(next_token);

            let pos = input_ids.len() + step;
            logits = self.model.forward_decode(next_token, pos);
        }

        all_ids
    }
}
