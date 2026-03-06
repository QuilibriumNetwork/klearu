/// VLM (Vision-Language Model) bridge for Qwen3.5 VL.
///
/// Encodes images using the vision encoder and injects the resulting
/// token embeddings into the LLM's input embedding sequence, replacing
/// image placeholder tokens.

use klearu_vision::model::qwen_vision::QwenVisionEncoder;

/// Bridge between vision encoder and LLM.
pub struct VlmBridge {
    pub vision_encoder: QwenVisionEncoder,
    /// Token ID for `<image>` placeholder in the tokenizer (248056).
    pub image_token_id: u32,
    /// Token ID for `<|vision_start|>` (248053).
    pub vision_start_token_id: u32,
    /// Token ID for `<|vision_end|>` (248054).
    pub vision_end_token_id: u32,
}

/// A single image to be injected into the token sequence.
pub struct VlmImage {
    /// CHW f32 tensor, normalized.
    pub data: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

impl VlmBridge {
    pub fn new(
        vision_encoder: QwenVisionEncoder,
        image_token_id: u32,
        vision_start_token_id: u32,
        vision_end_token_id: u32,
    ) -> Self {
        Self {
            vision_encoder,
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
        }
    }

    /// Encode a single image through the vision encoder.
    ///
    /// Returns: `[num_vision_tokens, out_hidden_size]` flattened.
    pub fn encode_image(&self, image: &VlmImage) -> Vec<f32> {
        self.vision_encoder.forward(&image.data, image.height, image.width)
    }

    /// Inject vision token embeddings into the text embedding sequence.
    ///
    /// Given:
    /// - `token_ids`: the full prompt token IDs (including image placeholder tokens)
    /// - `text_embeddings`: `[seq_len, hidden_size]` embeddings from the LLM's embedding table
    /// - `images`: the images to encode (one per contiguous run of image_token_ids)
    ///
    /// Returns merged embeddings with image tokens replaced by vision encoder outputs.
    ///
    /// The vision encoder output dimension must match the LLM hidden size
    /// (i.e., `out_hidden_size == hidden_size`).
    pub fn inject_vision_tokens(
        &self,
        token_ids: &[u32],
        text_embeddings: &[f32],
        images: &[VlmImage],
        hidden_size: usize,
    ) -> Vec<f32> {
        let seq_len = token_ids.len();
        debug_assert_eq!(text_embeddings.len(), seq_len * hidden_size);

        let mut result = text_embeddings.to_vec();

        // Find contiguous runs of image_token_id
        let mut image_idx = 0;
        let mut t = 0;
        while t < seq_len {
            if token_ids[t] == self.image_token_id {
                // Count the run
                let run_start = t;
                while t < seq_len && token_ids[t] == self.image_token_id {
                    t += 1;
                }
                let run_len = t - run_start;

                if image_idx < images.len() {
                    let vision_tokens = self.encode_image(&images[image_idx]);
                    let out_dim = self.vision_encoder.config.out_hidden_size;
                    let num_vision_tokens = vision_tokens.len() / out_dim;

                    // Replace up to run_len positions with vision tokens
                    let copy_len = run_len.min(num_vision_tokens);
                    for vt in 0..copy_len {
                        let dst_start = (run_start + vt) * hidden_size;
                        let src_start = vt * out_dim;
                        let copy_dim = hidden_size.min(out_dim);
                        result[dst_start..dst_start + copy_dim]
                            .copy_from_slice(&vision_tokens[src_start..src_start + copy_dim]);
                        // Zero any remaining dimensions if hidden_size > out_dim
                        for d in copy_dim..hidden_size {
                            result[dst_start + d] = 0.0;
                        }
                    }

                    image_idx += 1;
                }
            } else {
                t += 1;
            }
        }

        result
    }
}

/// Extract vision-related token IDs from a Qwen3.5 config.json.
pub struct VlmTokenIds {
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl VlmTokenIds {
    /// Parse from config.json string.
    pub fn from_config_json(json: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        Some(Self {
            image_token_id: v["image_token_id"].as_u64()? as u32,
            vision_start_token_id: v["vision_start_token_id"].as_u64()? as u32,
            vision_end_token_id: v["vision_end_token_id"].as_u64()? as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_vision::config::QwenVisionConfig;

    fn tiny_encoder() -> QwenVisionEncoder {
        let config = QwenVisionConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 2,
            intermediate_size: 16,
            patch_size: 8,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            out_hidden_size: 8,
            num_position_embeddings: 16,
            layer_norm_eps: 1e-5,
        };
        QwenVisionEncoder::new(config)
    }

    #[test]
    fn test_vlm_bridge_encode() {
        let encoder = tiny_encoder();
        let bridge = VlmBridge::new(encoder, 100, 101, 102);

        let image = VlmImage {
            data: vec![0.1f32; 3 * 16 * 16],
            height: 16,
            width: 16,
        };
        let tokens = bridge.encode_image(&image);
        // patch_size=8, 16/8=2 grid, merge_size=2 → 1 merged token × 8 dim
        assert_eq!(tokens.len(), 1 * 8);
        for &v in &tokens {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_vlm_bridge_inject() {
        let encoder = tiny_encoder();
        let hidden_size = 8;
        let bridge = VlmBridge::new(encoder, 100, 101, 102);

        // Sequence: [normal, image, normal]
        let token_ids = vec![1, 100, 2];
        let text_embeddings = vec![0.5f32; 3 * hidden_size];

        let image = VlmImage {
            data: vec![0.1f32; 3 * 16 * 16],
            height: 16,
            width: 16,
        };

        let merged = bridge.inject_vision_tokens(&token_ids, &text_embeddings, &[image], hidden_size);
        assert_eq!(merged.len(), 3 * hidden_size);

        // Token 0 and 2 should be unchanged
        assert_eq!(&merged[0..hidden_size], &[0.5f32; 8][..]);
        assert_eq!(&merged[2 * hidden_size..3 * hidden_size], &[0.5f32; 8][..]);
    }

    #[test]
    fn test_vlm_token_ids() {
        let json = r#"{"image_token_id": 248056, "vision_start_token_id": 248053, "vision_end_token_id": 248054}"#;
        let ids = VlmTokenIds::from_config_json(json).unwrap();
        assert_eq!(ids.image_token_id, 248056);
        assert_eq!(ids.vision_start_token_id, 248053);
        assert_eq!(ids.vision_end_token_id, 248054);
    }
}
