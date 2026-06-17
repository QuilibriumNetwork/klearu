//! LLM engine: wraps `klearu_llm::generate::Pipeline` plus a chat template.
//!
//! Generation is cooperative-cancellable: callers pass an `AtomicBool`
//! that's checked between tokens. This avoids needing a custom kill switch
//! in `Pipeline` itself.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use klearu_llm::generate::chat_template::{ChatMessage, ChatTemplate, Role};
use klearu_llm::generate::pipeline::{detect_eos_tokens, GenerateConfig, Pipeline};
use klearu_llm::generate::sampler::SamplerConfig;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::error::{DesktopError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmLoadStatus {
    pub loaded: bool,
    pub model_dir: Option<String>,
    pub hidden_size: Option<usize>,
    pub num_layers: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmSamplerInput {
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub max_new_tokens: Option<usize>,
    pub seed: Option<u64>,
}

impl LlmSamplerInput {
    pub fn to_config(&self, eos: Vec<u32>) -> GenerateConfig {
        GenerateConfig {
            max_new_tokens: self.max_new_tokens.unwrap_or(512),
            sampler: SamplerConfig {
                temperature: self.temperature.unwrap_or(0.7),
                top_k: self.top_k.unwrap_or(40),
                top_p: self.top_p.unwrap_or(0.95),
                repetition_penalty: self.repetition_penalty.unwrap_or(1.05),
            },
            eos_token_ids: eos,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WireRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMessage {
    pub role: WireRole,
    pub content: String,
}

impl WireMessage {
    fn to_chat(self) -> ChatMessage {
        let role = match self.role {
            WireRole::System => Role::System,
            WireRole::User => Role::User,
            WireRole::Assistant => Role::Assistant,
        };
        ChatMessage { role, content: self.content }
    }
}

struct Loaded {
    pipeline: Pipeline,
    template: ChatTemplate,
    eos: Vec<u32>,
    model_dir: PathBuf,
    hidden_size: usize,
    num_layers: usize,
}

#[derive(Default)]
pub struct LlmEngine {
    loaded: Option<Loaded>,
    /// Set by `abort()` to interrupt in-flight generation. Cleared at the
    /// start of every `stream_reply` call.
    abort: Arc<AtomicBool>,
}

impl LlmEngine {
    pub fn new() -> Self { Self::default() }

    pub fn status(&self) -> LlmLoadStatus {
        match &self.loaded {
            Some(l) => LlmLoadStatus {
                loaded: true,
                model_dir: Some(l.model_dir.to_string_lossy().into_owned()),
                hidden_size: Some(l.hidden_size),
                num_layers: Some(l.num_layers),
            },
            None => LlmLoadStatus {
                loaded: false,
                model_dir: None,
                hidden_size: None,
                num_layers: None,
            },
        }
    }

    pub fn load(&mut self, model_dir: &Path) -> Result<LlmLoadStatus> {
        tracing::info!(path = %model_dir.display(), "loading LLM");
        let pipeline = Pipeline::from_dir(model_dir)
            .map_err(|e| DesktopError::engine(format!("load pipeline: {e}")))?;
        let template = ChatTemplate::detect(model_dir)
            .map_err(|e| DesktopError::engine(format!("detect chat template: {e}")))?;
        let eos = detect_eos_tokens(model_dir);
        let hidden_size = pipeline.model.config.hidden_size;
        let num_layers = pipeline.model.config.num_layers;
        let loaded = Loaded {
            pipeline,
            template,
            eos,
            model_dir: model_dir.to_path_buf(),
            hidden_size,
            num_layers,
        };
        self.loaded = Some(loaded);
        Ok(self.status())
    }

    pub fn unload(&mut self) {
        self.loaded = None;
    }

    pub fn abort_handle(&self) -> Arc<AtomicBool> { Arc::clone(&self.abort) }

    pub fn hidden_size(&self) -> Option<usize> {
        self.loaded.as_ref().map(|l| l.hidden_size)
    }

    /// Stream-generate a reply for `messages`. Always routes through
    /// `generate_streaming_steered` so we can capture the post-final-norm
    /// hidden of the response's last token; pass `&[]` and `alpha=0` when
    /// no steering vector is available (no-op vs the unsteered path).
    ///
    /// `on_response_hidden` is invoked once when generation ends with the
    /// last-token hidden — used by the feedback loop to do Welford updates.
    pub fn stream_reply<F, H>(
        &mut self,
        messages: Vec<WireMessage>,
        sampler: LlmSamplerInput,
        steering_vec: &[f32],
        alpha: f32,
        mut on_token: F,
        mut on_response_hidden: H,
    ) -> Result<String>
    where
        F: FnMut(&str),
        H: FnMut(&[f32]),
    {
        let Some(loaded) = self.loaded.as_mut() else {
            return Err(DesktopError::NotLoaded("llm"));
        };
        self.abort.store(false, Ordering::SeqCst);
        let abort = Arc::clone(&self.abort);

        let chat: Vec<ChatMessage> = messages.into_iter().map(WireMessage::to_chat).collect();
        let prompt = loaded.template.apply(&chat);
        let config = sampler.to_config(loaded.eos.clone());
        let seed = sampler.seed.unwrap_or_else(|| rand::random());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // generate_streaming_steered asserts the vec length equals
        // hidden_size, so synthesize a zero vec when none was supplied.
        let zero;
        let vec_ref: &[f32] = if steering_vec.len() == loaded.hidden_size {
            steering_vec
        } else {
            zero = vec![0.0f32; loaded.hidden_size];
            &zero[..]
        };
        let effective_alpha = if steering_vec.len() == loaded.hidden_size { alpha } else { 0.0 };

        loaded
            .pipeline
            .generate_streaming_steered(
                &prompt,
                &config,
                &mut rng,
                vec_ref,
                effective_alpha,
                |tok| {
                    on_token(tok);
                    !abort.load(Ordering::SeqCst)
                },
                |hidden| on_response_hidden(hidden),
            )
            .map_err(|e| DesktopError::engine(format!("generate: {e}")))
    }
}
