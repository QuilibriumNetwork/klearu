//! Private (2PC) evaluation of Swin Transformer.
//!
//! Two modes:
//! - **Lower security** (`private_swin_forward`): reveal image, plaintext forward.
//! - **Higher security** (`private_swin_forward_secure`): Q32.32, Beaver-based attention,
//!   polynomial GELU. Shifted window is handled by local cyclic index remapping on shares.

#[cfg(feature = "vision")]
use klearu_vision::model::swin::SwinModel;
use klearu_mpc::beaver::{TripleGenerator, TripleGenerator128};
use klearu_mpc::fixed_point::{from_fixed, to_fixed64};
use klearu_mpc::sharing::{SharedVec, SharedVec64};
use klearu_mpc::transport::Transport;
use std::io;

use crate::private_vision_helpers::*;

/// Run private Swin inference (lower security).
#[cfg(feature = "vision")]
pub fn private_swin_forward(
    _party: u8,
    model: &SwinModel,
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

/// Run private Swin inference (higher security).
#[cfg(feature = "vision")]
pub fn private_swin_forward_secure(
    party: u8,
    model: &SwinModel,
    image_share: &SharedVec64,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let image_size = model.config.image_size;
    let in_channels = model.config.in_channels;
    let eps = model.config.layer_norm_eps;
    let total_pixels = in_channels * image_size * image_size;

    assert_eq!(image_share.len(), total_pixels);

    // === Patch embedding: Conv2d + LayerNorm ===
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
    let num_patches = grid_h * grid_w;

    // [C, H, W] → [N, C]
    let mut tokens = channel_first_to_tokens_64(&patch_out.0, dim0, grid_h, grid_w);

    // Patch norm
    for t in 0..num_patches {
        let mut tok = SharedVec64(tokens[t * dim0..(t + 1) * dim0].to_vec());
        klearu_mpc::normalization::layernorm_shared_64(
            party,
            &mut tok,
            &model.patch_norm.weight,
            &model.patch_norm.bias,
            eps,
            triples,
            transport,
        )?;
        tokens[t * dim0..(t + 1) * dim0].copy_from_slice(&tok.0);
    }

    let mut h = grid_h;
    let mut w = grid_w;
    let mut dim = dim0;

    // === Per stage ===
    for (s, stage) in model.stages.iter().enumerate() {
        // PatchMerge downsample
        if let Some(ref pm) = stage.downsample {
            let new_h = h / 2;
            let new_w = w / 2;
            let merged_dim = 4 * dim;
            let out_dim = 2 * dim;
            let num_out = new_h * new_w;

            // Gather 2×2 patches
            let mut merged = vec![0u64; num_out * merged_dim];
            for y in 0..new_h {
                for x in 0..new_w {
                    let out_idx = y * new_w + x;
                    let offsets = [
                        (2 * y) * w + (2 * x),
                        (2 * y) * w + (2 * x + 1),
                        (2 * y + 1) * w + (2 * x),
                        (2 * y + 1) * w + (2 * x + 1),
                    ];
                    for (i, &off) in offsets.iter().enumerate() {
                        merged[out_idx * merged_dim + i * dim..out_idx * merged_dim + (i + 1) * dim]
                            .copy_from_slice(&tokens[off * dim..(off + 1) * dim]);
                    }
                }
            }

            // LayerNorm → Linear reduction
            let empty_triples = vec![];
            let mut new_tokens = vec![0u64; num_out * out_dim];
            for t in 0..num_out {
                let mut tok = SharedVec64(merged[t * merged_dim..(t + 1) * merged_dim].to_vec());
                klearu_mpc::normalization::layernorm_shared_64(
                    party,
                    &mut tok,
                    &pm.norm.weight,
                    &pm.norm.bias,
                    eps,
                    triples,
                    transport,
                )?;
                let reduced = klearu_mpc::linear::shared_linear_forward_64(
                    party,
                    pm.reduction.weights.as_raw_slice(),
                    merged_dim,
                    out_dim,
                    &tok,
                    &empty_triples,
                    transport,
                )?;
                new_tokens[t * out_dim..(t + 1) * out_dim].copy_from_slice(&reduced.0);
                if party == 0 {
                    for d in 0..out_dim {
                        new_tokens[t * out_dim + d] =
                            new_tokens[t * out_dim + d].wrapping_add(to_fixed64(pm.reduction.bias[d]));
                    }
                }
            }

            tokens = new_tokens;
            h = new_h;
            w = new_w;
            dim = model.config.embed_dims[s];
        }

        // Swin blocks
        let ws = model.config.window_size;
        let num_tokens = h * w;

        for block in &stage.blocks {
            // Pre-norm → attention → residual
            let mut normed = tokens.clone();
            for t in 0..num_tokens {
                let mut tok = SharedVec64(normed[t * dim..(t + 1) * dim].to_vec());
                klearu_mpc::normalization::layernorm_shared_64(
                    party,
                    &mut tok,
                    &block.norm1.weight,
                    &block.norm1.bias,
                    eps,
                    triples,
                    transport,
                )?;
                normed[t * dim..(t + 1) * dim].copy_from_slice(&tok.0);
            }

            // Handle shifted window: cyclic shift is purely local index remapping
            let shift = block.shift_size;
            let shifted = if shift > 0 {
                // Cyclic shift: move tokens by (shift, shift) positions
                let mut shifted = vec![0u64; num_tokens * dim];
                for y in 0..h {
                    for x in 0..w {
                        let src = y * w + x;
                        let dst_y = (y + h - shift) % h;
                        let dst_x = (x + w - shift) % w;
                        let dst = dst_y * w + dst_x;
                        shifted[dst * dim..(dst + 1) * dim]
                            .copy_from_slice(&normed[src * dim..(src + 1) * dim]);
                    }
                }
                shifted
            } else {
                normed
            };

            // Window attention (secure, per window)
            let num_win_h = h / ws;
            let num_win_w = w / ws;
            let win_tokens = ws * ws;
            let mut attn_all = vec![0u64; num_tokens * dim];

            for wy in 0..num_win_h {
                for wx in 0..num_win_w {
                    let mut window = vec![0u64; win_tokens * dim];
                    for sy in 0..ws {
                        for sx in 0..ws {
                            let gy = wy * ws + sy;
                            let gx = wx * ws + sx;
                            let gi = gy * w + gx;
                            let li = sy * ws + sx;
                            window[li * dim..(li + 1) * dim]
                                .copy_from_slice(&shifted[gi * dim..(gi + 1) * dim]);
                        }
                    }

                    let win_out = klearu_mpc::attention_mpc::self_attention_secure_64(
                        party,
                        &window,
                        win_tokens,
                        dim,
                        block.attn.num_heads,
                        block.attn.qkv.weights.as_raw_slice(),
                        &block.attn.qkv.bias,
                        block.attn.proj.weights.as_raw_slice(),
                        &block.attn.proj.bias,
                        &mut |n| triples.generate(n),
                        transport,
                    )?;

                    for sy in 0..ws {
                        for sx in 0..ws {
                            let gy = wy * ws + sy;
                            let gx = wx * ws + sx;
                            let gi = gy * w + gx;
                            let li = sy * ws + sx;
                            attn_all[gi * dim..(gi + 1) * dim]
                                .copy_from_slice(&win_out[li * dim..(li + 1) * dim]);
                        }
                    }
                }
            }

            // Reverse cyclic shift
            let unshifted = if shift > 0 {
                let mut unshifted = vec![0u64; num_tokens * dim];
                for y in 0..h {
                    for x in 0..w {
                        let src_y = (y + h - shift) % h;
                        let src_x = (x + w - shift) % w;
                        let src = src_y * w + src_x;
                        let dst = y * w + x;
                        unshifted[dst * dim..(dst + 1) * dim]
                            .copy_from_slice(&attn_all[src * dim..(src + 1) * dim]);
                    }
                }
                unshifted
            } else {
                attn_all
            };

            // Residual
            for i in 0..tokens.len() {
                tokens[i] = tokens[i].wrapping_add(unshifted[i]);
            }

            // Pre-norm → MLP → residual
            for t in 0..num_tokens {
                let mut tok = SharedVec64(tokens[t * dim..(t + 1) * dim].to_vec());
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
            eps,
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
    use klearu_vision::config::SwinConfig;
    use klearu_mpc::beaver::dummy_triple_pair;
    use klearu_mpc::transport::memory_transport_pair;

    fn tiny_swin_config() -> SwinConfig {
        SwinConfig {
            image_size: 56,
            in_channels: 3,
            patch_size: 4,
            embed_dims: [8, 16, 32, 64],
            num_heads: [2, 2, 4, 8],
            depths: [1, 1, 1, 1],
            window_size: 7,
            mlp_ratio: 2.0,
            num_classes: 5,
            layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn test_private_swin_forward_runs() {
        let config = tiny_swin_config();
        let model0 = SwinModel::new(config.clone());
        let model1 = SwinModel::new(config.clone());

        let image = vec![0.1f32; 3 * 56 * 56];

        let share0 = shared_image(0, &image);
        let share1 = shared_image(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_swin_forward(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let result0 = private_swin_forward(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let result1 = handle.join().unwrap().unwrap();

        assert_eq!(result0.len(), config.num_classes);
        for i in 0..config.num_classes {
            assert_eq!(result0[i], result1[i], "logit[{i}] mismatch");
        }
    }

    #[test]
    fn test_private_swin_secure_runs() {
        use klearu_mpc::beaver::dummy_triple_pair_128;

        let config = tiny_swin_config();
        let model0 = SwinModel::new(config.clone());
        let model1 = SwinModel::new(config.clone());

        let image = vec![0.1f32; 3 * 56 * 56];

        let share0 = shared_image_64(0, &image);
        let share1 = shared_image_64(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(200000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_swin_forward_secure(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let logits0 =
            private_swin_forward_secure(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let logits1 = handle.join().unwrap().unwrap();

        assert_eq!(logits0.len(), config.num_classes);
        for i in 0..config.num_classes {
            assert!(logits0[i].is_finite(), "logit[{i}] not finite");
            assert!(logits1[i].is_finite());
        }
    }
}
