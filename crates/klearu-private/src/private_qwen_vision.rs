//! Private (2PC) evaluation of Qwen3.5 Vision Encoder.
//!
//! Two modes:
//! - **Lower security** (`private_qwen_vision_forward`): reveal image, plaintext forward.
//! - **Higher security** (`private_qwen_vision_forward_secure`): Q32.32 shares,
//!   Beaver-based attention (no Q/K reveal), polynomial GELU (no intermediate reveal).

#[cfg(feature = "vision")]
use klearu_vision::model::qwen_vision::QwenVisionEncoder;
use klearu_mpc::beaver::{TripleGenerator, TripleGenerator128};
use klearu_mpc::fixed_point::{from_fixed, to_fixed64};
use klearu_mpc::sharing::{SharedVec, SharedVec64};
use klearu_mpc::transport::Transport;
use std::io;

use crate::private_vision_helpers::*;

/// Run private Qwen Vision inference (lower security).
#[cfg(feature = "vision")]
pub fn private_qwen_vision_forward(
    _party: u8,
    model: &QwenVisionEncoder,
    image_share: &SharedVec,
    _triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
    h: usize,
    w: usize,
) -> io::Result<Vec<f32>> {
    let in_channels = model.config.in_channels * model.config.temporal_patch_size;
    let total_pixels = in_channels * h * w;

    assert_eq!(image_share.len(), total_pixels);

    transport.send_u32_slice(&image_share.0)?;
    let other = transport.recv_u32_slice(total_pixels)?;

    let mut image = vec![0.0f32; total_pixels];
    for i in 0..total_pixels {
        image[i] = from_fixed(image_share.0[i].wrapping_add(other[i]));
    }

    Ok(model.forward(&image, h, w))
}

/// Run private Qwen Vision inference (higher security).
///
/// Returns merged token embeddings `[num_merged_tokens, out_hidden_size]` as Q32.32 shares.
#[cfg(feature = "vision")]
pub fn private_qwen_vision_forward_secure(
    party: u8,
    model: &QwenVisionEncoder,
    image_share: &SharedVec64,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
    h: usize,
    w: usize,
) -> io::Result<Vec<f32>> {
    let in_channels = model.config.in_channels * model.config.temporal_patch_size;
    let hidden_size = model.config.hidden_size;
    let eps = model.config.layer_norm_eps;
    let total_pixels = in_channels * h * w;

    assert_eq!(image_share.len(), total_pixels);

    // === Patch embedding: Conv2d ===
    let pe = &model.patch_embed;
    let (patch_out, grid_h, grid_w) = crate::private_conv2d::shared_conv2d_forward_64(
        party,
        &pe.weight,
        pe.bias.as_deref(),
        pe.out_channels,
        pe.in_channels,
        pe.kernel_h,
        pe.kernel_w,
        pe.stride_h,
        pe.stride_w,
        pe.padding_h,
        pe.padding_w,
        pe.groups,
        &image_share.0,
        h,
        w,
    );

    let num_patches = grid_h * grid_w;

    // Reshape [hidden_size, grid_h, grid_w] → [num_patches, hidden_size]
    let mut tokens = channel_first_to_tokens_64(&patch_out.0, hidden_size, grid_h, grid_w);

    // Add position embeddings (party 0)
    if party == 0 {
        let pos_len = num_patches.min(model.pos_embed.len() / hidden_size);
        for i in 0..pos_len * hidden_size {
            tokens[i] = tokens[i].wrapping_add(to_fixed64(model.pos_embed[i]));
        }
    }

    // === ViT blocks ===
    for block in &model.blocks {
        // Pre-norm → attention → residual
        let mut normed = tokens.clone();
        for t in 0..num_patches {
            let mut tok = SharedVec64(normed[t * hidden_size..(t + 1) * hidden_size].to_vec());
            klearu_mpc::normalization::layernorm_shared_64(
                party,
                &mut tok,
                &block.norm1.weight,
                &block.norm1.bias,
                eps,
                triples,
                transport,
            )?;
            normed[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(&tok.0);
        }

        let attn_out = klearu_mpc::attention_mpc::self_attention_secure_64(
            party,
            &normed,
            num_patches,
            hidden_size,
            model.config.num_heads,
            block.attn.qkv.weights.as_raw_slice(),
            &block.attn.qkv.bias,
            block.attn.proj.weights.as_raw_slice(),
            &block.attn.proj.bias,
            &mut |n| triples.generate(n),
            transport,
        )?;

        for i in 0..tokens.len() {
            tokens[i] = tokens[i].wrapping_add(attn_out[i]);
        }

        // Pre-norm → MLP → residual
        for t in 0..num_patches {
            let mut tok = SharedVec64(tokens[t * hidden_size..(t + 1) * hidden_size].to_vec());
            klearu_mpc::normalization::layernorm_shared_64(
                party,
                &mut tok,
                &block.norm2.weight,
                &block.norm2.bias,
                eps,
                triples,
                transport,
            )?;

            let mlp_out = mlp_secure_no_reveal(
                party,
                &tok,
                block.mlp_fc1.weights.as_raw_slice(),
                &block.mlp_fc1.bias,
                hidden_size,
                block.mlp_fc1.out_features(),
                block.mlp_fc2.weights.as_raw_slice(),
                &block.mlp_fc2.bias,
                hidden_size,
                triples,
                transport,
            )?;

            for d in 0..hidden_size {
                tokens[t * hidden_size + d] =
                    tokens[t * hidden_size + d].wrapping_add(mlp_out.0[d]);
            }
        }
    }

    // === PatchMerger: LN → fc1 → GELU → fc2 ===
    let merger = &model.merger;
    let merge_size = merger.merge_size;
    let out_hidden = merger.fc2.out_features();
    let merge_h = grid_h / merge_size;
    let merge_w = grid_w / merge_size;
    let num_merged = merge_h * merge_w;
    let tokens_per_group = merge_size * merge_size;
    let merged_in_dim = hidden_size * tokens_per_group;

    let mut merged_tokens = vec![0u64; num_merged * out_hidden];

    for my in 0..merge_h {
        for mx in 0..merge_w {
            // Gather token group
            let mut group = Vec::with_capacity(merged_in_dim);
            for sy in 0..merge_size {
                for sx in 0..merge_size {
                    let gy = my * merge_size + sy;
                    let gx = mx * merge_size + sx;
                    let gi = gy * grid_w + gx;
                    group.extend_from_slice(&tokens[gi * hidden_size..(gi + 1) * hidden_size]);
                }
            }

            // LayerNorm on the full concatenated group (dim = merged_in_dim)
            let mut group_share = SharedVec64(group);
            klearu_mpc::normalization::layernorm_shared_64(
                party,
                &mut group_share,
                &merger.norm.weight,
                &merger.norm.bias,
                eps,
                triples,
                transport,
            )?;

            // fc1 → GELU → fc2
            let merged_tok = mlp_secure_no_reveal(
                party,
                &group_share,
                merger.fc1.weights.as_raw_slice(),
                &merger.fc1.bias,
                merged_in_dim,
                merger.fc1.out_features(),
                merger.fc2.weights.as_raw_slice(),
                &merger.fc2.bias,
                out_hidden,
                triples,
                transport,
            )?;

            let mi = my * merge_w + mx;
            merged_tokens[mi * out_hidden..(mi + 1) * out_hidden].copy_from_slice(&merged_tok.0);
        }
    }

    // Reveal merged token embeddings
    let result_share = SharedVec64(merged_tokens);
    reveal_logits(&result_share, transport)
}

#[cfg(all(test, feature = "vision"))]
mod tests {
    use super::*;
    use klearu_vision::config::QwenVisionConfig;
    use klearu_mpc::beaver::dummy_triple_pair;
    use klearu_mpc::transport::memory_transport_pair;

    fn tiny_qwen_config() -> QwenVisionConfig {
        QwenVisionConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 2,
            intermediate_size: 16,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            out_hidden_size: 8,
            num_position_embeddings: 256,
            layer_norm_eps: 1e-6,
        }
    }

    #[test]
    fn test_private_qwen_vision_forward_runs() {
        let config = tiny_qwen_config();
        let model0 = QwenVisionEncoder::new(config.clone());
        let model1 = QwenVisionEncoder::new(config.clone());

        // Input size must be multiple of patch_size * spatial_merge_size = 14*2 = 28
        let h = 28;
        let w = 28;
        let in_c = config.in_channels * config.temporal_patch_size;
        let image = vec![0.1f32; in_c * h * w];

        let share0 = shared_image(0, &image);
        let share1 = shared_image(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_qwen_vision_forward(1, &model1, &share1, &mut gen1, &mut trans_b, h, w)
        });

        let result0 =
            private_qwen_vision_forward(0, &model0, &share0, &mut gen0, &mut trans_a, h, w)
                .unwrap();
        let result1 = handle.join().unwrap().unwrap();

        assert_eq!(result0.len(), result1.len());
        for i in 0..result0.len() {
            assert_eq!(result0[i], result1[i], "token[{i}] mismatch");
        }
    }

    #[test]
    fn test_private_qwen_vision_secure_runs() {
        use klearu_mpc::beaver::dummy_triple_pair_128;

        let config = tiny_qwen_config();
        let model0 = QwenVisionEncoder::new(config.clone());
        let model1 = QwenVisionEncoder::new(config.clone());

        let h = 28;
        let w = 28;
        let in_c = config.in_channels * config.temporal_patch_size;
        let image = vec![0.1f32; in_c * h * w];

        let share0 = shared_image_64(0, &image);
        let share1 = shared_image_64(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(100000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_qwen_vision_forward_secure(1, &model1, &share1, &mut gen1, &mut trans_b, h, w)
        });

        let result0 =
            private_qwen_vision_forward_secure(0, &model0, &share0, &mut gen0, &mut trans_a, h, w)
                .unwrap();
        let result1 = handle.join().unwrap().unwrap();

        assert_eq!(result0.len(), result1.len());
        for i in 0..result0.len() {
            assert!(result0[i].is_finite(), "token[{i}] not finite");
        }
    }
}
