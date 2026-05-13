//! Autoregressive sampling: text prompt → image.
//!
//! Pipeline:
//!   1. Caller pre-tokenizes the prompt via any BPE tokenizer (we don't
//!      bake one in here — keeps the dependency surface clean and lets
//!      the user pick).
//!   2. Build the initial sequence `[BOS, text_tokens..., SEP_IMAGE]`.
//!   3. For each of `image_grid_h × image_grid_w` image-token positions:
//!      a. Forward through the transformer.
//!      b. Slice logits at the last position (over image-vocab).
//!      c. Apply temperature + top-k / top-p; sample.
//!      d. Map the sampled image codeword to its unified-vocab id and
//!         append to the sequence.
//!   4. Decode the generated 256-token grid via the image tokenizer.
//!
//! This module is **inference-only** — no gradient flow. Optimizing
//! the inner forward calls (KV cache across positions, batched
//! per-step sampling) is Phase 2+ work; the baseline here recomputes
//! the full sequence each step (O(n²·d) per step, O(n³·d) total — fine
//! for 256-token sequences).

use crate::error::{ImageGenError, Result};
use crate::model::{ImageTransformer, ImageTransformerConfig};
use crate::tokenizer::ImageTokenizer;

/// Sampling controls.
#[derive(Debug, Clone)]
pub struct SampleConfig {
    /// Softmax temperature. < 1.0 = sharper / less random; > 1.0 = flatter.
    pub temperature: f32,
    /// Restrict sampling to the top-K most-likely tokens. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) sampling: keep the smallest set of tokens whose
    /// cumulative probability ≥ top_p. 0.0 = disabled. Applied AFTER top_k.
    pub top_p: f32,
    /// PRNG seed (deterministic).
    pub seed: u64,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 64, top_p: 0.95, seed: 42 }
    }
}

/// Build the prefix `[BOS, text_tokens..., SEP_IMAGE]` from raw text BPE
/// token ids. Returns an error if `text_tokens` exceeds the model's
/// `max_text_len`.
pub fn build_prefix(
    cfg: &ImageTransformerConfig,
    text_tokens: &[u32],
) -> Result<Vec<u32>> {
    if text_tokens.len() > cfg.max_text_len {
        return Err(ImageGenError::ShapeMismatch {
            expected: format!("text_tokens.len() ≤ max_text_len = {}", cfg.max_text_len),
            got: format!("{}", text_tokens.len()),
        });
    }
    // Sanity: all text ids must fall in the text-BPE range.
    for (i, &tid) in text_tokens.iter().enumerate() {
        if (tid as usize) >= cfg.vocab_text {
            return Err(ImageGenError::ShapeMismatch {
                expected: format!("text id < vocab_text = {}", cfg.vocab_text),
                got: format!("id={tid} at position {i}"),
            });
        }
    }
    let mut seq = Vec::with_capacity(1 + text_tokens.len() + 1);
    seq.push(cfg.bos_token);
    seq.extend_from_slice(text_tokens);
    seq.push(cfg.sep_image_token);
    Ok(seq)
}

/// Sample image tokens autoregressively. Returns the `image_grid_h ×
/// image_grid_w` codewords (already in the [0, vocab_image) range —
/// ready to feed to `tokenizer.decode`).
pub fn sample_image_tokens(
    model: &ImageTransformer,
    text_tokens: &[u32],
    config: &SampleConfig,
) -> Result<Vec<u32>> {
    let mcfg = &model.config;
    let n_image = mcfg.image_grid_h * mcfg.image_grid_w;
    let mut seq = build_prefix(mcfg, text_tokens)?;

    let mut rng = SplitMix64::new(config.seed);
    let mut tokens = Vec::with_capacity(n_image);

    for _step in 0..n_image {
        // Forward through the full current sequence.
        let logits = model.forward(&seq)?;
        // Slice the last position's logits — that's the model's
        // prediction for the NEXT image token.
        let last = seq.len() - 1;
        let v = mcfg.vocab_image;
        let row = &logits[last * v..(last + 1) * v];

        // Apply temperature + top-k + top-p, then sample.
        let codeword = sample_from_logits(row, config, &mut rng);
        tokens.push(codeword);

        // Append the sampled codeword to the sequence in unified-vocab space.
        let unified_id = model.image_token_to_id(codeword);
        seq.push(unified_id);
    }
    Ok(tokens)
}

/// One-shot wrapper: text → image. Returns `[3, H, W]` f32 in [-1, 1],
/// matching `klearu_diffusion::image_io::save_png`'s expected format.
pub fn sample_image<T: ImageTokenizer>(
    model: &ImageTransformer,
    tokenizer: &T,
    text_tokens: &[u32],
    config: &SampleConfig,
) -> Result<Vec<f32>> {
    let tokens = sample_image_tokens(model, text_tokens, config)?;
    tokenizer.decode(&tokens)
}

/// Categorical sample from a logits row using the SampleConfig's
/// temperature, top-k, top-p, and PRNG state.
fn sample_from_logits(
    logits: &[f32],
    config: &SampleConfig,
    rng: &mut SplitMix64,
) -> u32 {
    let v = logits.len();
    let temp = config.temperature.max(1e-6);

    // Build a (logit, index) list scaled by 1/temperature.
    let mut scored: Vec<(f32, u32)> = logits.iter().enumerate()
        .map(|(i, &l)| (l / temp, i as u32))
        .collect();

    // Top-k filter (descending sort and truncate).
    if config.top_k > 0 && config.top_k < v {
        // Partial sort: nth_element-style.
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(config.top_k);
    } else {
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Softmax over the (possibly truncated) set.
    let max = scored.first().map(|x| x.0).unwrap_or(0.0);
    let mut probs: Vec<f32> = scored.iter().map(|(s, _)| (s - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() { *p /= sum; }
    }

    // Top-p (nucleus) filter on top of top-k. Walk the sorted list,
    // accumulate probability mass, cut at the boundary.
    if config.top_p > 0.0 && config.top_p < 1.0 {
        let mut cum = 0.0_f32;
        let mut keep = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= config.top_p { keep = i + 1; break; }
        }
        probs.truncate(keep);
        scored.truncate(keep);
        // Renormalise.
        let sum2: f32 = probs.iter().sum();
        if sum2 > 0.0 {
            for p in probs.iter_mut() { *p /= sum2; }
        }
    }

    // Inverse CDF sample.
    let u = rng.next_unit();
    let mut acc = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u <= acc { return scored[i].1; }
    }
    // Fallback (shouldn't reach: probs sum to 1).
    scored.last().map(|&(_, i)| i).unwrap_or(0)
}

/// SplitMix64 — small deterministic PRNG. Cheap, no dependencies.
struct SplitMix64 { state: u64 }
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f32 {
        // [0, 1) uniform.
        let x = self.next_u64() >> 11;            // 53-bit
        (x as f64 / (1u64 << 53) as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ImageTransformerConfig;

    fn tiny_cfg() -> ImageTransformerConfig {
        ImageTransformerConfig {
            vocab_text: 100,
            vocab_image: 64,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            mlp_intermediate: 64,
            max_text_len: 8,
            image_grid_h: 4,
            image_grid_w: 4,
            bos_token: 100,
            sep_image_token: 101,
            eos_token: 102,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    #[test]
    fn build_prefix_layout() {
        let cfg = tiny_cfg();
        let text = vec![10_u32, 20, 30];
        let prefix = build_prefix(&cfg, &text).expect("prefix");
        assert_eq!(prefix.len(), 5);
        assert_eq!(prefix[0], cfg.bos_token);
        assert_eq!(prefix[1..4], [10, 20, 30]);
        assert_eq!(prefix[4], cfg.sep_image_token);
    }

    #[test]
    fn build_prefix_rejects_bad_text_id() {
        let cfg = tiny_cfg();
        let text = vec![cfg.vocab_text as u32 + 5]; // out of text range
        assert!(build_prefix(&cfg, &text).is_err());
    }

    #[test]
    fn build_prefix_rejects_over_long_text() {
        let cfg = tiny_cfg();
        let text = vec![5_u32; cfg.max_text_len + 1];
        assert!(build_prefix(&cfg, &text).is_err());
    }

    #[test]
    fn sample_image_tokens_runs_with_zero_weights() {
        // Zero-weight model → uniform-ish logits → tokens span the full vocab.
        let cfg = tiny_cfg();
        let model = ImageTransformer::from_config(cfg.clone());
        let text = vec![5_u32, 6, 7];
        let sc = SampleConfig { temperature: 1.0, top_k: 0, top_p: 0.0, seed: 42 };
        let tokens = sample_image_tokens(&model, &text, &sc).expect("sample");
        assert_eq!(tokens.len(), cfg.image_grid_h * cfg.image_grid_w);
        for &t in &tokens {
            assert!(t < cfg.vocab_image as u32, "codeword {t} out of range");
        }
    }

    #[test]
    fn top_k_constrains_output() {
        let cfg = tiny_cfg();
        let model = ImageTransformer::from_config(cfg.clone());
        let text = vec![5_u32];
        // Top-k=1 with deterministic seed: every step picks the SAME token
        // (whatever the zero-weight model decides is argmax). With zero
        // weights argmax is just the first index hit by tie-breaking.
        let sc = SampleConfig { temperature: 1.0, top_k: 1, top_p: 0.0, seed: 42 };
        let tokens = sample_image_tokens(&model, &text, &sc).expect("sample");
        let first = tokens[0];
        // With top-k=1 and identical logits across positions in the
        // zero-weight model, all sampled tokens equal first.
        for &t in &tokens { assert_eq!(t, first); }
    }
}
