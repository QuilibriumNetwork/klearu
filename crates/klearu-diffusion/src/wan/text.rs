//! Prompt -> umT5 hidden states, the conditioning the Wan DiT cross-attends to.
//!
//! The checkpoint ships `tokenizer/tokenizer.json` (a Unigram fast tokenizer)
//! alongside `spiece.model`, so the `tokenizers` crate loads it directly and no
//! SentencePiece protobuf parsing is needed.

use crate::error::{DiffusionError, Result};
use crate::text_encoder::t5::{T5Config, T5Encoder};
use crate::weight::ComponentTensors;

/// Wan's prompt context length. Note this is **226**, not the 512 the encoder
/// could take — the pipeline pads and truncates to exactly this.
pub const MAX_PROMPT_TOKENS: usize = 226;

/// Normalise a prompt the way the reference's `prompt_clean` does: collapse
/// runs of whitespace and trim.
///
/// The reference also runs `ftfy.fix_text` and unescapes HTML entities twice
/// when ftfy is installed; both are no-ops for well-formed prompt text, so
/// only the whitespace pass is reproduced here.
pub fn prompt_clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Tokeniser plus umT5 encoder.
pub struct WanTextEncoder {
    pub tokenizer: tokenizers::Tokenizer,
    pub encoder: T5Encoder,
}

impl WanTextEncoder {
    /// `tokenizer_json` is `tokenizer/tokenizer.json`; `comp` holds the
    /// `text_encoder/` shards.
    pub fn load(tokenizer_json: &std::path::Path, comp: &ComponentTensors) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_json)
            .map_err(|e| DiffusionError::Layout(format!("load tokenizer.json: {e}")))?;
        let mut encoder = T5Encoder::from_config(T5Config::umt5_xxl());
        encoder.load_from(comp)?;
        Ok(Self { tokenizer, encoder })
    }

    /// Token ids for a prompt, truncated to `MAX_PROMPT_TOKENS`, along with
    /// the number of real (non-padding) tokens.
    ///
    /// Returns ids padded to the full length: the encoder runs over every
    /// position, and the padded ones are zeroed afterwards.
    pub fn tokenize(&self, prompt: &str) -> Result<(Vec<u32>, usize)> {
        let cleaned = prompt_clean(prompt);
        let enc = self
            .tokenizer
            .encode(cleaned, true)
            .map_err(|e| DiffusionError::Layout(format!("tokenize: {e}")))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(MAX_PROMPT_TOKENS);
        let real = ids.len();
        // `<pad>` is id 0 in this tokenizer.
        ids.resize(MAX_PROMPT_TOKENS, 0);
        Ok((ids, real))
    }

    /// Encode a prompt to `[MAX_PROMPT_TOKENS, d_model]`.
    ///
    /// The reference runs the encoder over the whole padded sequence with an
    /// attention mask, then **truncates to the real token count and re-pads
    /// with zeros** — so padded positions are exactly zero, not whatever the
    /// encoder produced for them. Reproduced here.
    pub fn encode(&self, prompt: &str) -> Result<Vec<f32>> {
        let (ids, real) = self.tokenize(prompt)?;
        let mask: Vec<bool> = (0..ids.len()).map(|i| i < real).collect();
        let d = self.encoder.config.d_model;
        let mut states = self.encoder.forward(&ids, Some(&mask));
        for v in states[real * d..].iter_mut() {
            *v = 0.0;
        }
        Ok(states)
    }
}
