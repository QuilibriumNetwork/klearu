use std::path::Path;

use rand::Rng;

use crate::error::Result;
use crate::sparse::predictor_store::PredictorStore;
use crate::sparse::SparseModel;
use crate::tokenizer::Tokenizer;

use super::pipeline::GenerateConfig;
use super::sampler::sample;

/// Generation pipeline that uses sparse inference for decode steps.
///
/// Prefill is done densely (all heads/neurons), while each decode step
/// uses the sparsity predictors to select only the most important
/// heads and neurons.
pub struct SparsePipeline {
    pub model: SparseModel,
    pub tokenizer: Tokenizer,
}

impl SparsePipeline {
    /// Load a sparse pipeline from a model directory.
    ///
    /// If `model_dir/predictors/` exists, trained predictors are loaded.
    /// Otherwise, the pipeline falls back to selecting the first-k heads/neurons.
    pub fn from_dir(
        model_dir: &Path,
        head_sparsity: f32,
        neuron_sparsity: f32,
    ) -> Result<Self> {
        let model = crate::weight::load_model(model_dir)?;
        let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json"))?;

        let num_layers = model.config.num_layers;
        let predictors_dir = model_dir.join("predictors");

        let store = if predictors_dir.is_dir() {
            PredictorStore::load(&predictors_dir, num_layers)
                .unwrap_or_else(|_| PredictorStore::new(num_layers))
        } else {
            PredictorStore::new(num_layers)
        };

        let sparse_model = SparseModel::new(model, store, head_sparsity, neuron_sparsity);
        Ok(Self {
            model: sparse_model,
            tokenizer,
        })
    }

    /// Create a pipeline from an already-loaded sparse model and tokenizer.
    pub fn new(model: SparseModel, tokenizer: Tokenizer) -> Self {
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

    /// Generate text with streaming: calls `on_token` for each new token.
    ///
    /// Prefill is dense; decode steps use sparse inference.
    /// `on_token` receives the token text and returns `true` to continue or `false` to stop.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        config: &GenerateConfig,
        rng: &mut impl Rng,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> Result<String> {
        let input_ids = self.tokenizer.encode(prompt)?;
        self.model.model.reset_kv_caches();

        let mut all_ids: Vec<u32> = input_ids.to_vec();
        let mut generated = Vec::new();

        // Dense prefill
        if input_ids.len() > 1 {
            let _ = self.model.model.forward_prefill(&input_ids[..input_ids.len() - 1]);
        }

        // First decode step (could also be sparse, but use dense for the last prefill token)
        let last_input = *input_ids.last().unwrap_or(&0);
        let position = if input_ids.is_empty() { 0 } else { input_ids.len() - 1 };
        let mut logits = self.model.forward_decode_sparse(last_input, position);

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
            logits = self.model.forward_decode_sparse(next_token, pos);
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
        self.model.model.reset_kv_caches();

        let mut all_ids: Vec<u32> = input_ids.to_vec();

        // Dense prefill
        if input_ids.len() > 1 {
            let _ = self.model.model.forward_prefill(&input_ids[..input_ids.len() - 1]);
        }

        let last_input = *input_ids.last().unwrap_or(&0);
        let position = if input_ids.is_empty() { 0 } else { input_ids.len() - 1 };
        let mut logits = self.model.forward_decode_sparse(last_input, position);

        for step in 0..config.max_new_tokens {
            let next_token = sample(&mut logits, &config.sampler, &all_ids, rng);

            if config.eos_token_ids.contains(&next_token) {
                break;
            }

            all_ids.push(next_token);

            let pos = input_ids.len() + step;
            logits = self.model.forward_decode_sparse(next_token, pos);
        }

        all_ids
    }
}
