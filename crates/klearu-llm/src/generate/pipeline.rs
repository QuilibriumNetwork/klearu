use std::path::Path;

use rand::Rng;

use crate::error::Result;
use crate::model::Model;
use crate::tokenizer::Tokenizer;

use super::sampler::{SamplerConfig, sample};

/// Detect EOS token ID from a model directory's config files.
///
/// Returns the first EOS token from [`detect_eos_tokens`].
/// Use `detect_eos_tokens` to get all EOS token IDs.
pub fn detect_eos_token(model_dir: &Path) -> Option<u32> {
    detect_eos_tokens(model_dir).into_iter().next()
}

/// Detect all EOS token IDs from a model directory's config files.
///
/// Priority:
/// 1. `generation_config.json` → `eos_token_id` (most authoritative for generation)
/// 2. `tokenizer_config.json` → `eos_token_id` (integer or array)
/// 3. `tokenizer_config.json` → resolve `eos_token` string via `added_tokens_decoder`
/// 4. `config.json` → top-level `eos_token_id` (integer or array)
/// 5. `config.json` → `text_config.eos_token_id` (integer or array)
/// 6. Fallback to `[2]`
///
/// Handles both integer and array `eos_token_id` fields.
/// For Qwen models, `generation_config.json` contains both `<|endoftext|>`
/// (151645) and `<|im_end|>` (151643), while other files may only list one.
pub fn detect_eos_tokens(model_dir: &Path) -> Vec<u32> {
    // generation_config.json is checked FIRST because it's the most authoritative
    // for generation behavior. It often has the full array of EOS tokens (e.g.,
    // Qwen: [151645, 151643] for both <|endoftext|> and <|im_end|>), while
    // tokenizer_config.json may only list a single eos_token_id.
    let gen_config_path = model_dir.join("generation_config.json");
    if let Ok(content) = std::fs::read_to_string(gen_config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(ids) = parse_eos_token_id_field(&json, "eos_token_id") {
                return ids;
            }
        }
    }

    let tok_config_path = model_dir.join("tokenizer_config.json");
    if let Ok(content) = std::fs::read_to_string(&tok_config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            // Try "eos_token_id" (integer or array)
            if let Some(ids) = parse_eos_token_id_field(&json, "eos_token_id") {
                return ids;
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
                                    return vec![id];
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
            if let Some(ids) = parse_eos_token_id_field(&json, "eos_token_id") {
                return ids;
            }
            if let Some(tc) = json.get("text_config") {
                if let Some(ids) = parse_eos_token_id_field(tc, "eos_token_id") {
                    return ids;
                }
            }
        }
    }

    // Fallback
    vec![2]
}

/// Parse `eos_token_id` from a JSON value — handles both integer and array.
fn parse_eos_token_id_field(json: &serde_json::Value, field: &str) -> Option<Vec<u32>> {
    let val = json.get(field)?;
    if let Some(id) = val.as_u64() {
        return Some(vec![id as u32]);
    }
    if let Some(arr) = val.as_array() {
        let ids: Vec<u32> = arr.iter().filter_map(|v| v.as_u64().map(|id| id as u32)).collect();
        if !ids.is_empty() {
            return Some(ids);
        }
    }
    None
}

/// Generation configuration.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub sampler: SamplerConfig,
    /// EOS token IDs. Generation stops when any of these tokens is produced.
    pub eos_token_ids: Vec<u32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            sampler: SamplerConfig::default(),
            eos_token_ids: vec![2], // common default for LLaMA
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

            if config.eos_token_ids.contains(&next_token) {
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

    /// Generate text with streaming, applying a per-user steering vector
    /// to the post-final-norm hidden at every step. The vector is added
    /// scaled by `alpha`, and an LM head is run on the steered hidden to
    /// produce the actual logits used for sampling.
    ///
    /// `steering_vec` must be `model.config.hidden_size` floats. Pass a
    /// zero vector or `alpha = 0.0` to disable steering and recover
    /// `generate_streaming` behaviour exactly.
    ///
    /// Also calls `on_response_hidden` once at the end with the post-
    /// final-norm hidden of the *last* token of the response — useful
    /// for feedback-driven Welford updates of the user vector.
    pub fn generate_streaming_steered(
        &mut self,
        prompt: &str,
        config: &GenerateConfig,
        rng: &mut impl Rng,
        steering_vec: &[f32],
        alpha: f32,
        mut on_token: impl FnMut(&str) -> bool,
        mut on_response_hidden: impl FnMut(&[f32]),
    ) -> Result<String> {
        assert_eq!(
            steering_vec.len(),
            self.model.config.hidden_size,
            "steering_vec length {} ≠ hidden_size {}",
            steering_vec.len(),
            self.model.config.hidden_size
        );

        let input_ids = self.tokenizer.encode(prompt)?;
        self.model.reset_kv_caches();

        let mut all_ids: Vec<u32> = input_ids.to_vec();
        let mut generated = Vec::new();

        if input_ids.len() > 1 {
            let _ = self.model.forward_prefill(&input_ids[..input_ids.len() - 1]);
        }

        let last_input = *input_ids.last().unwrap_or(&0);
        let position = if input_ids.is_empty() { 0 } else { input_ids.len() - 1 };

        // First step: get hidden, steer, LM-head.
        let mut hidden = self.model.forward_decode_hidden(last_input, position);
        let mut last_hidden = hidden.clone();
        if alpha != 0.0 {
            for (h, &v) in hidden.iter_mut().zip(steering_vec.iter()) {
                *h += alpha * v;
            }
        }
        let mut logits = self.model.apply_lm_head(&hidden);

        for step in 0..config.max_new_tokens {
            let next_token = sample(&mut logits, &config.sampler, &all_ids, rng);

            if config.eos_token_ids.contains(&next_token) {
                break;
            }

            all_ids.push(next_token);
            generated.push(next_token);

            let token_text = self.tokenizer.decode(&[next_token])?;
            if !on_token(&token_text) {
                break;
            }

            let pos = input_ids.len() + step;
            hidden = self.model.forward_decode_hidden(next_token, pos);
            last_hidden = hidden.clone();
            if alpha != 0.0 {
                for (h, &v) in hidden.iter_mut().zip(steering_vec.iter()) {
                    *h += alpha * v;
                }
            }
            logits = self.model.apply_lm_head(&hidden);
        }

        on_response_hidden(&last_hidden);
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

            if config.eos_token_ids.contains(&next_token) {
                break;
            }

            all_ids.push(next_token);

            let pos = input_ids.len() + step;
            logits = self.model.forward_decode(next_token, pos);
        }

        all_ids
    }
}
