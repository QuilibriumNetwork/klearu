//! Thin wrapper around the HuggingFace `tokenizers` crate, configured
//! for SD's CLIP-format tokenizers. Same crate underlies klearu-llm's
//! tokenizer; here we set CLIP-specific behaviour (pad to 77 tokens,
//! BOS/EOS markers, etc.).
//!
//! SD checkpoints ship two tokenizers:
//!   - tokenizer/tokenizer.json (CLIP-L; ID 49406 BOS, 49407 EOS, 49407 pad)
//!   - tokenizer_2/tokenizer.json (CLIP-G, SDXL only; same vocab, different model)

use std::path::Path;

use crate::error::{DiffusionError, Result};

pub const CLIP_BOS_ID: u32 = 49406;
pub const CLIP_EOS_ID: u32 = 49407;
pub const CLIP_MAX_LENGTH: usize = 77;

pub struct CLIPTokenizer {
    inner: tokenizers::Tokenizer,
    pub max_length: usize,
}

impl CLIPTokenizer {
    /// Load a CLIP tokenizer. Accepts either:
    ///   - a `tokenizer.json` (modern HF merged format), or
    ///   - a directory containing `vocab.json` + `merges.txt` (legacy
    ///     split format used by SDXL's `tokenizer/` and `tokenizer_2/`
    ///     subdirs from stabilityai/stable-diffusion-xl-base-1.0).
    /// In the legacy case we construct a CLIP-compatible BPE tokenizer
    /// programmatically (Whitespace pre-tokenizer + BPE model + BOS/EOS
    /// post-processor).
    pub fn from_file(path: &Path) -> Result<Self> {
        // If `path` is a directory: look for tokenizer.json first; if missing,
        // try the legacy split layout.
        if path.is_dir() {
            let merged = path.join("tokenizer.json");
            if merged.exists() {
                return Self::from_merged(&merged);
            }
            return Self::from_legacy_split(path);
        }
        // File path: if it ends with .json and that file is a vocab.json,
        // assume legacy and look for merges.txt next to it.
        if path.file_name().and_then(|s| s.to_str()) == Some("vocab.json") {
            let parent = path.parent().unwrap_or(Path::new("."));
            return Self::from_legacy_split(parent);
        }
        Self::from_merged(path)
    }

    fn from_merged(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| DiffusionError::Unsupported(format!("tokenizer load: {e}")))?;
        Ok(Self { inner, max_length: CLIP_MAX_LENGTH })
    }

    /// Build a CLIP-format tokenizer from `<dir>/vocab.json` + `<dir>/merges.txt`.
    fn from_legacy_split(dir: &Path) -> Result<Self> {
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
        use tokenizers::processors::byte_level::ByteLevel as ByteLevelPost;
        use tokenizers::Tokenizer as HfTokenizer;

        let vocab = dir.join("vocab.json");
        let merges = dir.join("merges.txt");
        if !vocab.exists() || !merges.exists() {
            return Err(DiffusionError::Unsupported(format!(
                "legacy tokenizer: expected {} and {}",
                vocab.display(), merges.display()
            )));
        }
        let bpe = BPE::from_file(
            vocab.to_str().unwrap(),
            merges.to_str().unwrap(),
        )
        .unk_token("<|endoftext|>".into())
        .end_of_word_suffix("</w>".into())
        .build()
        .map_err(|e| DiffusionError::Unsupported(format!("BPE build: {e}")))?;

        let mut tok: HfTokenizer = HfTokenizer::new(bpe);
        // CLIP uses byte-level BPE (same family as GPT-2). Apply ByteLevel
        // pre-tokenization without an added prefix-space (CLIP-specific).
        tok.with_pre_tokenizer(Some(ByteLevelPre::new(false, true, true)));
        tok.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::new(false, true, true)));
        // Add BOS/EOS as post-processing template.
        // Single sequence: <|startoftext|> A <|endoftext|>
        let post = tokenizers::processors::template::TemplateProcessing::builder()
            .try_single("<|startoftext|>:0 $A:0 <|endoftext|>:0")
            .map_err(|e| DiffusionError::Unsupported(format!("template: {e}")))?
            .special_tokens(vec![
                ("<|startoftext|>", CLIP_BOS_ID),
                ("<|endoftext|>", CLIP_EOS_ID),
            ])
            .build()
            .map_err(|e| DiffusionError::Unsupported(format!("post build: {e}")))?;
        tok.with_post_processor(Some(post));

        // Add the special tokens so encode() knows their IDs.
        tok.add_special_tokens(&[
            tokenizers::AddedToken::from("<|startoftext|>", true),
            tokenizers::AddedToken::from("<|endoftext|>", true),
        ]);

        Ok(Self { inner: tok, max_length: CLIP_MAX_LENGTH })
    }

    /// Encode prompt to token IDs, padded with `pad_token` to `max_length`,
    /// BOS-prefixed, EOS-terminated. Returns exactly `max_length` ids.
    ///
    /// **SDXL pad token convention** (per ComfyUI `sdxl_clip.py`):
    ///   - CLIP-L: `pad_token = 49407` (EOS)
    ///   - CLIP-G: `pad_token = 0`  ← this matters; using EOS feeds CLIP-G
    ///     out-of-distribution input at pad positions and produces garbage
    ///     hidden states that swamp cross-attention.
    ///
    /// SD 1.5 uses pad=49407 for CLIP-L (only encoder).
    pub fn encode_padded_with(&self, text: &str, pad_token: u32) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| DiffusionError::Unsupported(format!("encode: {e}")))?;
        let mut ids = Vec::with_capacity(self.max_length);
        ids.push(CLIP_BOS_ID);
        for &id in encoding.get_ids() {
            if ids.len() >= self.max_length - 1 { break; } // leave room for EOS
            ids.push(id);
        }
        ids.push(CLIP_EOS_ID);
        while ids.len() < self.max_length {
            ids.push(pad_token);
        }
        Ok(ids)
    }

    /// Backwards-compatible: pads with EOS (correct for SD 1.5 CLIP-L and
    /// SDXL CLIP-L). For SDXL CLIP-G, use `encode_padded_with(text, 0)`.
    pub fn encode_padded(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_padded_with(text, CLIP_EOS_ID)
    }

    /// Encode without padding (for diagnostics).
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| DiffusionError::Unsupported(format!("encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
