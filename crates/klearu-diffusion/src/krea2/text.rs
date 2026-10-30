//! Prompt encoding for Krea 2: the Qwen-Image chat template run through
//! Qwen3-VL-4B's language model, tapping 12 intermediate hidden states.
//!
//! The reference tokenizes `[prefix | prompt | PAD... | suffix]` to a fixed
//! length, masks the padding, and counts positions over real tokens only
//! (`cumsum(mask) - 1`), so the suffix's rotary phase continues straight on
//! from the prompt. That makes the padding invisible: dropping it, as done
//! here, gives the same hidden states for every real token, and the
//! transformer masks padded text tokens as attention keys anyway. The 34
//! system-prefix tokens are encoded (they condition everything after them)
//! and then discarded.
//!
//! The decoder layers are HiDream's (`hidream::gpu::GpuDecoderLayer`) — the
//! same Qwen3-VL architecture at a different size. Text-only positions are
//! equal on all three mRoPE axes, which collapses mRoPE to plain RoPE.

use crate::{DiffusionError, Result};

/// System prompt and user-turn opener, prepended to every prompt.
pub const PROMPT_PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
/// Closes the user turn and opens the assistant's.
pub const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// Tokens in [`PROMPT_PREFIX`], dropped from the encoder output.
pub const PREFIX_TOKENS: usize = 34;
/// Tokens in [`PROMPT_SUFFIX`].
pub const SUFFIX_TOKENS: usize = 5;
/// Text tokens the transformer sees at most (prompt + suffix).
pub const MAX_SEQUENCE_LENGTH: usize = 512;

/// Tokenize one prompt into the full encoder input: prefix, prompt (truncated
/// as the reference truncates it), suffix. No padding.
pub fn tokenize_prompt(tok: &tokenizers::Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    let enc = |s: &str| -> Result<Vec<u32>> {
        tok.encode(s, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| DiffusionError::Unsupported(format!("krea2 tokenizer: {e}")))
    };
    let mut ids = enc(&format!("{PROMPT_PREFIX}{prompt}"))?;
    let prefix = enc(PROMPT_PREFIX)?;
    if prefix.len() != PREFIX_TOKENS || ids[..PREFIX_TOKENS] != prefix[..] {
        return Err(DiffusionError::Unsupported(format!(
            "krea2 tokenizer splits the system prefix into {} tokens (expected {PREFIX_TOKENS}); \
             is this the Qwen2 tokenizer?",
            prefix.len()
        )));
    }
    // The reference pads/truncates prefix+prompt to this many tokens, then
    // appends the suffix: 512 text tokens survive the prefix drop.
    ids.truncate(MAX_SEQUENCE_LENGTH + PREFIX_TOKENS - SUFFIX_TOKENS);
    let suffix = enc(PROMPT_SUFFIX)?;
    if suffix.len() != SUFFIX_TOKENS {
        return Err(DiffusionError::Unsupported(format!(
            "krea2 tokenizer splits the suffix into {} tokens (expected {SUFFIX_TOKENS})",
            suffix.len()
        )));
    }
    ids.extend(suffix);
    Ok(ids)
}

#[cfg(feature = "metal")]
pub use gpu_encoder::Krea2TextEncoder;

#[cfg(feature = "metal")]
mod gpu_encoder {
    use std::path::Path;

    use super::*;
    use crate::hidream::config::{RopeScaling, TextConfig};
    use crate::hidream::gpu::GpuDecoderLayer;
    use crate::hidream::model::build_position_ids_t2i;
    use crate::hidream::mrope;
    use crate::krea2::config::TEXT_SELECT_LAYERS;
    use crate::metal_backend::{upload_f32_as_f16_buffer, GpuTensor};
    use crate::weight::loader::ComponentTensors;

    /// Qwen3-VL-4B language model on the GPU, truncated after the deepest tap.
    pub struct Krea2TextEncoder {
        pub tokenizer: tokenizers::Tokenizer,
        pub cfg: TextConfig,
        layers: Vec<GpuDecoderLayer>,
        /// Kept open for the embedding lookup, which borrows rows straight out
        /// of the mmap instead of holding a 778 MB copy of the table.
        comp: ComponentTensors,
        embed_name: String,
    }

    impl Krea2TextEncoder {
        /// `encoder_dir` holds `config.json` + safetensors for a `Qwen3VLModel`
        /// (Krea's `text_encoder/`, or a Qwen3-VL-4B-Instruct download).
        pub fn load(encoder_dir: &Path, tokenizer_json: &Path) -> Result<Self> {
            let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_json)
                .map_err(|e| DiffusionError::Unsupported(format!("krea2 tokenizer: {e}")))?;
            let cfg_text = std::fs::read_to_string(encoder_dir.join("config.json"))
                .map_err(|e| DiffusionError::Layout(format!("text_encoder/config.json: {e}")))?;
            let cfg = parse_text_config(&cfg_text)?;
            let comp = ComponentTensors::open_dir(encoder_dir)?;
            let lm = ["language_model.", "model.language_model.", "model."]
                .into_iter()
                .find(|p| comp.has(&format!("{p}embed_tokens.weight")))
                .ok_or_else(|| {
                    DiffusionError::Layout(format!(
                        "{} holds no Qwen3-VL language model (no embed_tokens.weight)",
                        encoder_dir.display()
                    ))
                })?;
            let depth = *TEXT_SELECT_LAYERS.iter().max().unwrap();
            if depth > cfg.num_hidden_layers {
                return Err(DiffusionError::Layout(format!(
                    "text encoder has {} layers; Krea 2 taps layer {depth}",
                    cfg.num_hidden_layers
                )));
            }
            let layers = (0..depth)
                .map(|i| GpuDecoderLayer::load(&comp, &format!("{lm}layers.{i}"), &cfg))
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                tokenizer,
                cfg,
                layers,
                embed_name: format!("{lm}embed_tokens.weight"),
                comp,
            })
        }

        pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
            tokenize_prompt(&self.tokenizer, prompt)
        }

        /// Encode a prompt to its taps, `[n, 12, hidden]` with the system
        /// prefix dropped. Returns `(taps, n)`.
        pub fn encode(&self, prompt: &str) -> Result<(Vec<f32>, usize)> {
            let ids = self.tokenize(prompt)?;
            Ok(self.encode_ids(&ids))
        }

        /// Encode pre-tokenized ids (prefix included). Returns `(taps, n)` for
        /// the `n = ids.len() - PREFIX_TOKENS` tokens after the prefix.
        pub fn encode_ids(&self, ids: &[u32]) -> (Vec<f32>, usize) {
            let hidden = self.cfg.hidden_size;
            let seq = ids.len();
            assert!(seq > PREFIX_TOKENS, "token sequence shorter than the system prefix");

            // Qwen3-VL's massive activations exceed f16 in the MLP outputs.
            crate::metal_backend::set_use_f32_sgemm(true);
            let barrier = crate::metal_backend::set_mps_barrier(true);

            let table = self.comp.get_bf16(&self.embed_name).expect("embed table vanished");
            let mut states = vec![0.0_f32; seq * hidden];
            for (row, &id) in states.chunks_mut(hidden).zip(ids) {
                let src = &table[id as usize * hidden * 2..][..hidden * 2];
                for (v, b) in row.iter_mut().zip(src.chunks_exact(2)) {
                    *v = half::bf16::from_le_bytes([b[0], b[1]]).to_f32();
                }
            }
            drop(table);

            // Positions 0..seq on every axis: plain 1-D RoPE.
            let pos = build_position_ids_t2i(seq, 0, 0);
            let rope = mrope::build_tables(
                &pos, seq, self.cfg.head_dim(), self.cfg.rope_theta, &self.cfg.mrope_section(),
            );
            let cos = upload_f32_as_f16_buffer(&rope.cos);
            let sin = upload_f32_as_f16_buffer(&rope.sin);

            let n_layers = TEXT_SELECT_LAYERS.len();
            let n = seq - PREFIX_TOKENS;
            let mut taps = vec![0.0_f32; n * n_layers * hidden];
            let mut hg = GpuTensor::upload_f32_as_f16(vec![seq, hidden], &states);
            for (i, layer) in self.layers.iter().enumerate() {
                causal_layer_forward(layer, &mut hg, seq, &cos, &sin);
                // hidden_states[i + 1] is the stream after layer i.
                if let Some(slot) = TEXT_SELECT_LAYERS.iter().position(|&s| s == i + 1) {
                    let h = hg.download_to_f32();
                    for t in 0..n {
                        let src = &h[(PREFIX_TOKENS + t) * hidden..][..hidden];
                        taps[(t * n_layers + slot) * hidden..][..hidden].copy_from_slice(src);
                    }
                }
            }
            crate::metal_backend::set_use_f32_sgemm(false);
            crate::metal_backend::set_mps_barrier(barrier);
            (taps, n)
        }
    }

    /// One Qwen3-VL decoder layer with fully causal attention on the GPU.
    ///
    /// HiDream's `GpuDecoderLayer::forward` serves a hybrid mask — causal text
    /// rows recomputed on the CPU after a bidirectional SDPA — which is exact
    /// here too (`txt_len = seq`), but its scalar per-row loop is O(n²) per
    /// layer: ~1 s at 62 tokens, tens of seconds at the 546-token maximum.
    /// This reuses that layer's weights and runs the attention through the
    /// causal flash kernel instead.
    fn causal_layer_forward(
        layer: &GpuDecoderLayer, hidden: &mut GpuTensor, n: usize, cos: &metal::Buffer, sin: &metal::Buffer,
    ) {
        use crate::metal_backend::{
            eadd_f16_gpu, flash_attention_causal_metal, gqa_repeat_f16_gpu, permute_hl_to_lh_f16_gpu,
            permute_lh_to_hl_f16_gpu, rms_norm_f16_gpu, rope_apply_f16_gpu,
        };
        let a = &layer.self_attn;
        let (nh, nkv, dh) = (a.num_heads, a.num_kv_heads, a.head_dim);

        let mut normed = hidden.clone_data();
        rms_norm_f16_gpu(&mut normed, &layer.input_ln_buf, n, layer.hidden, layer.rms_eps);
        let mut q = a.q_proj.forward(&normed, n);
        let mut k = a.k_proj.forward(&normed, n);
        let v = a.v_proj.forward(&normed, n);
        rms_norm_f16_gpu(&mut q, &a.q_norm_buf, n * nh, dh, a.rms_eps);
        rms_norm_f16_gpu(&mut k, &a.k_norm_buf, n * nkv, dh, a.rms_eps);
        rope_apply_f16_gpu(&mut q, cos, sin, n, nh, dh);
        rope_apply_f16_gpu(&mut k, cos, sin, n, nkv, dh);

        let qh = permute_lh_to_hl_f16_gpu(&q, 1, n, nh, dh);
        let kh0 = permute_lh_to_hl_f16_gpu(&k, 1, n, nkv, dh);
        let vh0 = permute_lh_to_hl_f16_gpu(&v, 1, n, nkv, dh);
        let mut kh = GpuTensor::new_f16(vec![nh * n * dh]);
        let mut vh = GpuTensor::new_f16(vec![nh * n * dh]);
        gqa_repeat_f16_gpu(&kh0, &mut kh, nh / nkv, n * dh);
        gqa_repeat_f16_gpu(&vh0, &mut vh, nh / nkv, n * dh);
        // The causal kernel takes host f32 slices; at prompt lengths the
        // round trip is a few MB per layer.
        let (qf, kf, vf) = (qh.download_to_f32(), kh.download_to_f32(), vh.download_to_f32());
        let mut of = vec![0.0_f32; nh * n * dh];
        flash_attention_causal_metal(&qf, &kf, &vf, &mut of, 1, nh, n, n, dh, 1.0 / (dh as f32).sqrt());
        let oh = GpuTensor::upload_f32_as_f16(vec![nh * n * dh], &of);
        let ctx = permute_hl_to_lh_f16_gpu(&oh, 1, n, nh, dh);
        let attn = a.o_proj.forward(&ctx, n);
        eadd_f16_gpu(hidden, &attn);

        let mut normed2 = hidden.clone_data();
        rms_norm_f16_gpu(&mut normed2, &layer.post_ln_buf, n, layer.hidden, layer.rms_eps);
        let mlp = layer.mlp.forward(&normed2, n);
        eadd_f16_gpu(hidden, &mlp);
    }

    /// Parse `text_config` from a Qwen3-VL `config.json`, accepting both the
    /// transformers 4 layout (`rope_theta` + `rope_scaling`) and the
    /// transformers 5 one (`rope_parameters`), which Krea's checkpoint uses.
    pub fn parse_text_config(text: &str) -> Result<TextConfig> {
        let v: serde_json::Value = serde_json::from_str(text)?;
        let tc = v.get("text_config").cloned().unwrap_or(v);
        let mut cfg: TextConfig = serde_json::from_value(tc.clone())?;
        if let Some(rp) = tc.get("rope_parameters") {
            if let Some(theta) = rp.get("rope_theta").and_then(|t| t.as_f64()) {
                cfg.rope_theta = theta as f32;
            }
            let section = rp.get("mrope_section").and_then(|s| {
                s.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
            });
            if section.is_some() {
                cfg.rope_scaling = Some(RopeScaling {
                    mrope_section: section,
                    rope_type: rp.get("rope_type").and_then(|t| t.as_str()).map(String::from),
                });
            }
        }
        Ok(cfg)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_transformers5_rope_parameters() {
            let text = r#"{"model_type":"qwen3_vl","text_config":{
                "hidden_size":2560,"num_hidden_layers":36,"num_attention_heads":32,
                "num_key_value_heads":8,"head_dim":128,"intermediate_size":9728,
                "vocab_size":151936,"rms_norm_eps":1e-6,
                "rope_parameters":{"mrope_interleaved":true,"mrope_section":[24,20,20],
                    "rope_theta":5000000,"rope_type":"default"}}}"#;
            let cfg = parse_text_config(text).unwrap();
            assert_eq!(cfg.rope_theta, 5_000_000.0);
            assert_eq!(cfg.mrope_section(), vec![24, 20, 20]);
            assert_eq!(cfg.kv_groups(), 4);
        }
    }
}
