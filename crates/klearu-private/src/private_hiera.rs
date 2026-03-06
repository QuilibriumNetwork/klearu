//! Private (2PC) evaluation of Hiera (Hierarchical ViT).
//!
//! Two modes:
//! - **Lower security** (`private_hiera_forward`): reveal image, plaintext forward.
//! - **Higher security** (`private_hiera_forward_secure`): Q32.32, Beaver-based attention,
//!   polynomial GELU. Token merging is purely local concat + linear projection.

#[cfg(feature = "vision")]
use klearu_vision::model::hiera::HieraModel;
use klearu_mpc::beaver::{TripleGenerator, TripleGenerator128};
use klearu_mpc::fixed_point::{from_fixed, to_fixed64};
use klearu_mpc::sharing::{SharedVec, SharedVec64};
use klearu_mpc::transport::Transport;
use std::io;

use crate::private_vision_helpers::*;

/// Run private Hiera inference (lower security).
#[cfg(feature = "vision")]
pub fn private_hiera_forward(
    _party: u8,
    model: &HieraModel,
    image_share: &SharedVec,
    _triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let image_size = model.config.image_size;
    let total_pixels = model.config.in_channels * image_size * image_size;
    assert_eq!(image_share.len(), total_pixels);

    transport.send_u32_slice(&image_share.0)?;
    let other = transport.recv_u32_slice(total_pixels)?;

    let mut image = vec![0.0f32; total_pixels];
    for i in 0..total_pixels {
        image[i] = from_fixed(image_share.0[i].wrapping_add(other[i]));
    }

    Ok(model.forward(&image))
}

/// Run private Hiera inference (higher security).
#[cfg(feature = "vision")]
pub fn private_hiera_forward_secure(
    party: u8,
    model: &HieraModel,
    image_share: &SharedVec64,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let image_size = model.config.image_size;
    let in_channels = model.config.in_channels;
    let total_pixels = in_channels * image_size * image_size;
    assert_eq!(image_share.len(), total_pixels);

    // === Patch embedding: Conv2d ===
    let (patch_out, grid_h, grid_w) = crate::private_conv2d::shared_conv2d_forward_64(
        party,
        &model.patch_embed.weight,
        model.patch_embed.bias.as_deref(),
        model.patch_embed.out_channels,
        model.patch_embed.in_channels,
        model.patch_embed.kernel_h,
        model.patch_embed.kernel_w,
        model.patch_embed.stride_h,
        model.patch_embed.stride_w,
        model.patch_embed.padding_h,
        model.patch_embed.padding_w,
        model.patch_embed.groups,
        &image_share.0,
        image_size,
        image_size,
    );

    let dim0 = model.config.embed_dims[0];
    let _num_patches = grid_h * grid_w;

    // [C, H, W] → [N, C]
    let mut tokens = channel_first_to_tokens_64(&patch_out.0, dim0, grid_h, grid_w);

    let mut h = grid_h;
    let mut w = grid_w;
    let mut dim = dim0;

    // === Per stage ===
    for (s, stage_blocks) in model.stages.iter().enumerate() {
        // Token merge (between stages, not before first)
        if let Some(ref merge) = model.merges[s] {
            let new_h = h / 2;
            let new_w = w / 2;
            let num_out = new_h * new_w;
            let merged_dim = 4 * dim;
            let out_dim = merge.proj.out_features();
            let empty_triples = vec![];

            let mut new_tokens = vec![0u64; num_out * out_dim];

            for y in 0..new_h {
                for x in 0..new_w {
                    let out_idx = y * new_w + x;
                    let offsets = [
                        (2 * y) * w + (2 * x),
                        (2 * y) * w + (2 * x + 1),
                        (2 * y + 1) * w + (2 * x),
                        (2 * y + 1) * w + (2 * x + 1),
                    ];

                    let mut merged = vec![0u64; merged_dim];
                    for (i, &off) in offsets.iter().enumerate() {
                        merged[i * dim..(i + 1) * dim]
                            .copy_from_slice(&tokens[off * dim..(off + 1) * dim]);
                    }

                    let merged_share = SharedVec64(merged);
                    let mut proj_out = klearu_mpc::linear::shared_linear_forward_64(
                        party,
                        merge.proj.weights.as_raw_slice(),
                        merged_dim,
                        out_dim,
                        &merged_share,
                        &empty_triples,
                        transport,
                    )?;
                    if party == 0 {
                        for d in 0..out_dim {
                            proj_out.0[d] =
                                proj_out.0[d].wrapping_add(to_fixed64(merge.proj.bias[d]));
                        }
                    }
                    new_tokens[out_idx * out_dim..(out_idx + 1) * out_dim]
                        .copy_from_slice(&proj_out.0);
                }
            }

            tokens = new_tokens;
            h = new_h;
            w = new_w;
            dim = model.config.embed_dims[s];
        }

        let num_tokens = h * w;

        // Hiera blocks
        for block in stage_blocks {
            // Pre-norm → attention → residual
            let mut normed = tokens.clone();
            for t in 0..num_tokens {
                let mut tok = SharedVec64(normed[t * dim..(t + 1) * dim].to_vec());
                klearu_mpc::normalization::layernorm_shared_64(
                    party,
                    &mut tok,
                    &block.norm1.weight,
                    &block.norm1.bias,
                    block.norm1.eps,
                    triples,
                    transport,
                )?;
                normed[t * dim..(t + 1) * dim].copy_from_slice(&tok.0);
            }

            let attn_out = klearu_mpc::attention_mpc::self_attention_secure_64(
                party,
                &normed,
                num_tokens,
                dim,
                block.attn.num_heads,
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
            for t in 0..num_tokens {
                let mut tok = SharedVec64(tokens[t * dim..(t + 1) * dim].to_vec());
                klearu_mpc::normalization::layernorm_shared_64(
                    party,
                    &mut tok,
                    &block.norm2.weight,
                    &block.norm2.bias,
                    block.norm2.eps,
                    triples,
                    transport,
                )?;

                let mlp_out = mlp_secure_no_reveal(
                    party,
                    &tok,
                    block.mlp_fc1.weights.as_raw_slice(),
                    &block.mlp_fc1.bias,
                    dim,
                    block.mlp_fc1.out_features(),
                    block.mlp_fc2.weights.as_raw_slice(),
                    &block.mlp_fc2.bias,
                    dim,
                    triples,
                    transport,
                )?;

                for d in 0..dim {
                    tokens[t * dim + d] = tokens[t * dim + d].wrapping_add(mlp_out.0[d]);
                }
            }
        }
    }

    // === Final norm + pool + head ===
    let final_dim = model.config.embed_dims[3];
    let num_tokens = h * w;

    for t in 0..num_tokens {
        let mut tok = SharedVec64(tokens[t * final_dim..(t + 1) * final_dim].to_vec());
        klearu_mpc::normalization::layernorm_shared_64(
            party,
            &mut tok,
            &model.final_norm.weight,
            &model.final_norm.bias,
            model.final_norm.eps,
            triples,
            transport,
        )?;
        tokens[t * final_dim..(t + 1) * final_dim].copy_from_slice(&tok.0);
    }

    let pooled = global_avg_pool_64(&tokens, num_tokens, final_dim);

    let empty_triples = vec![];
    let pooled_share = SharedVec64(pooled);
    let mut logits = klearu_mpc::linear::shared_linear_forward_64(
        party,
        model.head.weights.as_raw_slice(),
        final_dim,
        model.head.out_features(),
        &pooled_share,
        &empty_triples,
        transport,
    )?;
    if party == 0 {
        for i in 0..model.head.out_features() {
            logits.0[i] = logits.0[i].wrapping_add(to_fixed64(model.head.bias[i]));
        }
    }

    reveal_logits(&logits, transport)
}

#[cfg(all(test, feature = "vision"))]
mod tests {
    use super::*;
    use klearu_vision::config::HieraConfig;
    use klearu_mpc::beaver::dummy_triple_pair;
    use klearu_mpc::transport::memory_transport_pair;

    fn tiny_hiera_config() -> HieraConfig {
        HieraConfig {
            image_size: 32,
            in_channels: 3,
            patch_size: 4,
            embed_dims: [8, 16, 32, 64],
            num_heads: [1, 2, 4, 8],
            depths: [1, 1, 1, 1],
            mask_unit_size: 8,
            mlp_ratio: 2.0,
            num_classes: 5,
        }
    }

    #[test]
    fn test_private_hiera_forward_runs() {
        let config = tiny_hiera_config();
        let model0 = HieraModel::new(config.clone());
        let model1 = HieraModel::new(config.clone());

        let image = vec![0.1f32; 3 * 32 * 32];

        let share0 = shared_image(0, &image);
        let share1 = shared_image(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_hiera_forward(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let result0 =
            private_hiera_forward(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let result1 = handle.join().unwrap().unwrap();

        assert_eq!(result0.len(), config.num_classes);
        for i in 0..config.num_classes {
            assert_eq!(result0[i], result1[i], "logit[{i}] mismatch");
        }
    }

    #[test]
    fn test_private_hiera_secure_runs() {
        use klearu_mpc::beaver::dummy_triple_pair_128;

        let config = tiny_hiera_config();
        let model0 = HieraModel::new(config.clone());
        let model1 = HieraModel::new(config.clone());

        let image = vec![0.1f32; 3 * 32 * 32];

        let share0 = shared_image_64(0, &image);
        let share1 = shared_image_64(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(100000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_hiera_forward_secure(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let logits0 =
            private_hiera_forward_secure(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let logits1 = handle.join().unwrap().unwrap();

        assert_eq!(logits0.len(), config.num_classes);
        for i in 0..config.num_classes {
            assert!(logits0[i].is_finite(), "logit[{i}] not finite");
            assert!(logits1[i].is_finite());
        }
    }
}
