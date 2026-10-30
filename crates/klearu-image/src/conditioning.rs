//! Portable text conditioning: checkpoints carry the complete tokenizer.
use crate::{
    error::{ImageGenError, Result},
    model::ImageTransformerConfig,
};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Conditioning {
    /// Old checkpoint: its tokenizer identity is unknown.
    #[default]
    Legacy,
    Unconditional,
    Tokenizer(String),
}

fn err(e: impl std::fmt::Display) -> ImageGenError {
    ImageGenError::Config(e.to_string())
}

/// Canonical JSON also makes identity comparisons independent of map order.
pub fn identity(tok: &Tokenizer) -> Result<Conditioning> {
    let value: serde_json::Value =
        serde_json::from_str(&tok.to_string(false).map_err(err)?).map_err(err)?;
    Ok(Conditioning::Tokenizer(
        serde_json::to_string(&value).map_err(err)?,
    ))
}

pub fn vocab_size(tok: &Tokenizer) -> usize {
    tok.get_vocab(true)
        .values()
        .copied()
        .max()
        .map_or(0, |v| v as usize + 1)
}

pub fn validate(tok: &Tokenizer, cfg: &ImageTransformerConfig) -> Result<()> {
    if vocab_size(tok) > cfg.vocab_text {
        return Err(err(format!(
            "tokenizer needs {} IDs but model vocab_text is {}; IDs must not be dropped",
            vocab_size(tok),
            cfg.vocab_text
        )));
    }
    Ok(())
}

pub fn encode(tok: &Tokenizer, cfg: &ImageTransformerConfig, text: &str) -> Result<Vec<u32>> {
    validate(tok, cfg)?;
    let enc = tok.encode(text, false).map_err(err)?;
    let mut ids = enc.get_ids().to_vec();
    if ids.iter().any(|&id| id as usize >= cfg.vocab_text) {
        return Err(err("encoded text ID exceeds model vocabulary"));
    }
    ids.truncate(cfg.max_text_len);
    Ok(ids)
}

/// Use an embedded tokenizer by default. An override must match exactly.
pub fn resolve(
    conditioning: &Conditioning,
    supplied: Option<Tokenizer>,
    unconditional: bool,
) -> Result<Option<Tokenizer>> {
    if unconditional {
        if supplied.is_some() {
            return Err(err("--unconditional conflicts with --bpe"));
        }
        if matches!(conditioning, Conditioning::Tokenizer(_)) {
            return Err(err("checkpoint was trained conditionally"));
        }
        return Ok(None);
    }
    match conditioning {
        Conditioning::Unconditional => Err(err("use --unconditional for this checkpoint")),
        Conditioning::Legacy => supplied
            .map(Some)
            .ok_or_else(|| err("legacy checkpoint requires --bpe or explicit --unconditional")),
        Conditioning::Tokenizer(json) => {
            if let Some(tok) = supplied {
                if &identity(&tok)? != conditioning {
                    return Err(err("--bpe does not match the checkpoint tokenizer"));
                }
                Ok(Some(tok))
            } else {
                Ok(Some(Tokenizer::from_bytes(json.as_bytes()).map_err(err)?))
            }
        }
    }
}
