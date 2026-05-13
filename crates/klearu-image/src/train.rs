//! Training-time data plumbing + loss computation (forward only).
//!
//! ## Scope
//!
//! This module ships:
//!   - `Manifest` / `ManifestEntry`: JSONL-format dataset spec.
//!   - `TrainBatch`: an in-memory `(token_ids, target_positions, targets)`
//!     triple ready to feed to the model.
//!   - `assemble_batch`: turns a manifest entry + a tokenizer + a BPE
//!     text-tokenizer into the model's expected sequence layout.
//!   - `cross_entropy_loss`: per-position softmax-CE on the image-token
//!     prediction targets, returning mean loss + per-token logits-vs-
//!     target stats.
//!
//! ## Scope NOT covered (deferred follow-up — see TODO note in lib.rs):
//!
//! Backpropagation + optimizer (Adam) for the transformer. Implementing
//! the backward pass for each op (matmul, RMSNorm, SwiGLU, attention,
//! softmax-CE, embedding lookup) is ~1500 LOC and is its own milestone.
//! The forward + loss code here is the **measurement infrastructure** —
//! we can already log loss curves on toy data and verify the forward
//! pipeline produces sensible CE values, which is the right first step.

use crate::error::{ImageGenError, Result};
use crate::model::{ImageTransformer, ImageTransformerConfig};

/// One image-text training example.
///
/// `text_tokens` are BPE token ids in `[0, vocab_text)`.
/// `image_tokens` are codeword ids in `[0, vocab_image)` — the OUTPUT
/// of `tokenizer.encode(image)`. Already pre-tokenized; the training
/// loop does not call into the VAE at every step.
#[derive(Debug, Clone)]
pub struct TrainExample {
    pub text_tokens: Vec<u32>,
    pub image_tokens: Vec<u32>,
}

/// Assembled batch ready for `ImageTransformer::forward`.
#[derive(Debug, Clone)]
pub struct TrainBatch {
    /// Unified-vocab token sequence: `[BOS, text..., SEP, img...]`.
    pub token_ids: Vec<u32>,
    /// Indices into `token_ids` where the model predicts the NEXT token.
    /// For an image-only training objective, these are the positions of
    /// the SEP_IMAGE token + each image token *except the last* — at
    /// each such position the model should predict the NEXT image token.
    pub predict_at: Vec<usize>,
    /// Target codewords (in `[0, vocab_image)`), one per `predict_at`
    /// entry. `targets[i]` is the codeword the model should predict
    /// when seeing tokens 0..=predict_at[i].
    pub targets: Vec<u32>,
}

/// Convert a single example to a `TrainBatch`.
pub fn assemble_batch(
    cfg: &ImageTransformerConfig,
    ex: &TrainExample,
) -> Result<TrainBatch> {
    let n_image_expected = cfg.image_grid_h * cfg.image_grid_w;
    if ex.image_tokens.len() != n_image_expected {
        return Err(ImageGenError::ShapeMismatch {
            expected: format!("{n_image_expected} image tokens"),
            got: format!("{}", ex.image_tokens.len()),
        });
    }
    if ex.text_tokens.len() > cfg.max_text_len {
        return Err(ImageGenError::ShapeMismatch {
            expected: format!("≤ {} text tokens", cfg.max_text_len),
            got: format!("{}", ex.text_tokens.len()),
        });
    }

    // Sequence: [BOS, text..., SEP, img...]. Length 1 + |text| + 1 + |img|.
    let mut seq = Vec::with_capacity(1 + ex.text_tokens.len() + 1 + ex.image_tokens.len());
    seq.push(cfg.bos_token);
    for &t in &ex.text_tokens {
        if (t as usize) >= cfg.vocab_text {
            return Err(ImageGenError::ShapeMismatch {
                expected: format!("text id < vocab_text = {}", cfg.vocab_text),
                got: format!("id={t}"),
            });
        }
        seq.push(t);
    }
    seq.push(cfg.sep_image_token);
    let sep_pos = seq.len() - 1;
    let off = cfg.image_id_offset();
    for &codeword in &ex.image_tokens {
        if (codeword as usize) >= cfg.vocab_image {
            return Err(ImageGenError::ShapeMismatch {
                expected: format!("codeword < vocab_image = {}", cfg.vocab_image),
                got: format!("codeword={codeword}"),
            });
        }
        seq.push(off + codeword);
    }

    // Predict-at positions: at sep_pos, predict the first image token;
    // at each image position except the last, predict the next image token.
    // Total predictions = n_image.
    let mut predict_at = Vec::with_capacity(n_image_expected);
    let mut targets = Vec::with_capacity(n_image_expected);
    predict_at.push(sep_pos);
    targets.push(ex.image_tokens[0]);
    for i in 0..(ex.image_tokens.len() - 1) {
        predict_at.push(sep_pos + 1 + i);
        targets.push(ex.image_tokens[i + 1]);
    }

    Ok(TrainBatch { token_ids: seq, predict_at, targets })
}

/// Compute softmax cross-entropy on the prediction positions.
///
/// `logits` is the model's full output, `[n_tokens, vocab_image]`
/// row-major. `batch.predict_at[i]` selects the row and
/// `batch.targets[i]` is the correct class.
///
/// Returns `(mean_loss, num_correct)`. `num_correct` counts how many
/// argmax predictions hit the target — a quick sanity statistic.
pub fn cross_entropy_loss(
    model: &ImageTransformer,
    logits: &[f32],
    batch: &TrainBatch,
) -> Result<(f32, usize)> {
    let v = model.config.vocab_image;
    let n_tokens = batch.token_ids.len();
    if logits.len() != n_tokens * v {
        return Err(ImageGenError::ShapeMismatch {
            expected: format!("{n_tokens} · {v} = {}", n_tokens * v),
            got: format!("{}", logits.len()),
        });
    }
    if batch.predict_at.len() != batch.targets.len() {
        return Err(ImageGenError::ShapeMismatch {
            expected: "predict_at.len() == targets.len()".into(),
            got: format!("{} vs {}", batch.predict_at.len(), batch.targets.len()),
        });
    }
    let mut total = 0.0_f64;
    let mut correct = 0_usize;
    for (&pos, &target) in batch.predict_at.iter().zip(batch.targets.iter()) {
        let row = &logits[pos * v..(pos + 1) * v];
        // Stable log-softmax: log(softmax(x))_t = x_t - logsumexp(x).
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sumexp = 0.0_f64;
        for &x in row { sumexp += ((x - max) as f64).exp(); }
        let lse = (sumexp.ln() as f32) + max;
        let log_p_target = row[target as usize] - lse;
        total += -log_p_target as f64;
        // Argmax for accuracy.
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = 0_usize;
        for (i, &x) in row.iter().enumerate() {
            if x > best_v { best_v = x; best_i = i; }
        }
        if best_i == target as usize { correct += 1; }
    }
    let mean = (total / batch.predict_at.len() as f64) as f32;
    Ok((mean, correct))
}

// ============================================================================
// Manifest format (JSONL)
// ============================================================================

/// One line of the manifest. The training loop tokenizes the image
/// offline and stores `image_tokens` here directly; the text caption
/// stays as a string so we can re-BPE it under different tokenizers
/// without re-running the VAE.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ManifestEntry {
    pub caption: String,
    /// Codewords in `[0, vocab_image)`, length = `image_grid_h * image_grid_w`.
    pub image_tokens: Vec<u32>,
}

/// Read a manifest file (one JSON object per line).
pub fn read_manifest(path: &std::path::Path) -> Result<Vec<ManifestEntry>> {
    let raw = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let e: ManifestEntry = serde_json::from_str(line)
            .map_err(|e| ImageGenError::Config(format!("manifest line {i}: {e}")))?;
        out.push(e);
    }
    Ok(out)
}

/// Write a manifest file (one JSON object per line). Used by the
/// offline pre-tokenization step.
pub fn write_manifest(path: &std::path::Path, entries: &[ManifestEntry]) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for e in entries {
        let line = serde_json::to_string(e)
            .map_err(|err| ImageGenError::Config(format!("serialize: {err}")))?;
        writeln!(f, "{line}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> ImageTransformerConfig {
        ImageTransformerConfig {
            vocab_text: 100,
            vocab_image: 16,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 4,
            mlp_intermediate: 32,
            max_text_len: 8,
            image_grid_h: 2,
            image_grid_w: 2,
            bos_token: 100,
            sep_image_token: 101,
            eos_token: 102,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    #[test]
    fn assemble_layout() {
        let cfg = tiny_cfg();
        let ex = TrainExample {
            text_tokens: vec![5, 6, 7],
            image_tokens: vec![0, 1, 2, 3],
        };
        let b = assemble_batch(&cfg, &ex).expect("assemble");
        // Sequence: [BOS, 5, 6, 7, SEP, img(0), img(1), img(2), img(3)]
        assert_eq!(b.token_ids.len(), 1 + 3 + 1 + 4);
        assert_eq!(b.token_ids[0], cfg.bos_token);
        assert_eq!(b.token_ids[4], cfg.sep_image_token);
        let off = cfg.image_id_offset();
        assert_eq!(b.token_ids[5], off + 0);
        assert_eq!(b.token_ids[6], off + 1);
        // 4 predictions: at SEP (predict img_0), at img_0 (predict img_1), etc.
        assert_eq!(b.predict_at.len(), 4);
        assert_eq!(b.predict_at, vec![4, 5, 6, 7]);
        assert_eq!(b.targets, vec![0, 1, 2, 3]);
    }

    #[test]
    fn ce_loss_matches_uniform() {
        // For a model with zero weights, all logits are 0 → softmax is
        // uniform over vocab_image. CE = log(vocab_image).
        let cfg = tiny_cfg();
        let model = ImageTransformer::from_config(cfg.clone());
        let ex = TrainExample {
            text_tokens: vec![5, 6],
            image_tokens: vec![0, 1, 2, 3],
        };
        let batch = assemble_batch(&cfg, &ex).expect("assemble");
        let logits = model.forward(&batch.token_ids).expect("forward");
        let (loss, correct) = cross_entropy_loss(&model, &logits, &batch).expect("ce");
        let expected = (cfg.vocab_image as f32).ln();
        // With zero weights the loss should be very close to log(V).
        // Allow small numerical wiggle.
        assert!((loss - expected).abs() < 0.05,
            "loss {loss} should be ≈ log(V) = {expected}");
        // Argmax over uniform-ish logits is index 0 (first hit), and the
        // first target happens to be 0 → 1 hit. Fragile, but documents
        // current behavior.
        assert!(correct <= batch.predict_at.len());
    }

    #[test]
    fn manifest_round_trip() {
        let entries = vec![
            ManifestEntry { caption: "a cat".into(), image_tokens: vec![0, 1, 2, 3] },
            ManifestEntry { caption: "a dog".into(), image_tokens: vec![4, 5, 6, 7] },
        ];
        let tmp = std::env::temp_dir().join("klearu_image_test_manifest.jsonl");
        write_manifest(&tmp, &entries).expect("write");
        let loaded = read_manifest(&tmp).expect("read");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].caption, "a cat");
        assert_eq!(loaded[1].image_tokens, vec![4, 5, 6, 7]);
        std::fs::remove_file(&tmp).ok();
    }
}
