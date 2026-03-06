//! Private (2PC) evaluation of ConvNeXt (V1 and V2).
//!
//! Two modes:
//! - **Lower security** (`private_convnext_forward`): reveal image, plaintext forward.
//! - **Higher security** (`private_convnext_forward_secure`): Q32.32, polynomial GELU,
//!   Beaver-based GRN (V2). No attention in ConvNeXt — only Conv2d, LayerNorm, MLP.

#[cfg(feature = "vision")]
use klearu_vision::model::convnext::ConvNextModel;
use klearu_mpc::beaver::{TripleGenerator, TripleGenerator128};
use klearu_mpc::fixed_point::{from_fixed, to_fixed64};
use klearu_mpc::sharing::{SharedVec, SharedVec64};
use klearu_mpc::transport::Transport;
use std::io;

use crate::private_vision_helpers::*;

/// Run private ConvNeXt inference (lower security).
#[cfg(feature = "vision")]
pub fn private_convnext_forward(
    _party: u8,
    model: &ConvNextModel,
    image_share: &SharedVec,
    _triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let image_size = 224; // Standard ConvNeXt input
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

/// Run private ConvNeXt inference (higher security).
#[cfg(feature = "vision")]
pub fn private_convnext_forward_secure(
    party: u8,
    model: &ConvNextModel,
    image_share: &SharedVec64,
    image_size: usize,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let in_channels = model.config.in_channels;
    let eps = model.config.layer_norm_eps;
    let total_pixels = in_channels * image_size * image_size;
    assert_eq!(image_share.len(), total_pixels);

    // === Stem: Conv2d 4×4 + LayerNorm2d ===
    let (stem_out, mut h, mut w) = crate::private_conv2d::shared_conv2d_forward_64(
        party,
        &model.stem_conv.weight,
        model.stem_conv.bias.as_deref(),
        model.stem_conv.out_channels,
        model.stem_conv.in_channels,
        model.stem_conv.kernel_h,
        model.stem_conv.kernel_w,
        model.stem_conv.stride_h,
        model.stem_conv.stride_w,
        model.stem_conv.padding_h,
        model.stem_conv.padding_w,
        model.stem_conv.groups,
        &image_share.0,
        image_size,
        image_size,
    );

    let mut dim = model.config.dims[0];
    let mut features = stem_out;

    // Stem LayerNorm2d
    layernorm_2d_shared_64(
        party,
        &mut features.0,
        dim,
        h,
        w,
        &model.stem_norm.weight,
        &model.stem_norm.bias,
        eps,
        triples,
        transport,
    )?;

    // === Per stage ===
    for s in 0..4 {
        // Downsample between stages
        if let Some(ref ds) = model.downsamples[s] {
            // LayerNorm2d on input
            layernorm_2d_shared_64(
                party,
                &mut features.0,
                dim,
                h,
                w,
                &ds.norm.weight,
                &ds.norm.bias,
                eps,
                triples,
                transport,
            )?;

            // Conv2d 2×2 stride 2
            let (ds_out, new_h, new_w) = crate::private_conv2d::shared_conv2d_forward_64(
                party,
                &ds.conv.weight,
                ds.conv.bias.as_deref(),
                ds.conv.out_channels,
                ds.conv.in_channels,
                ds.conv.kernel_h,
                ds.conv.kernel_w,
                ds.conv.stride_h,
                ds.conv.stride_w,
                ds.conv.padding_h,
                ds.conv.padding_w,
                ds.conv.groups,
                &features.0,
                h,
                w,
            );
            features = ds_out;
            h = new_h;
            w = new_w;
            dim = model.config.dims[s];
        }

        let spatial = h * w;

        // ConvNeXt blocks
        for block in &model.stages[s] {
            // Depthwise conv (groups = dim, local)
            let (conv_out, _, _) = crate::private_conv2d::shared_conv2d_forward_64(
                party,
                &block.dwconv.weight,
                block.dwconv.bias.as_deref(),
                block.dwconv.out_channels,
                block.dwconv.in_channels,
                block.dwconv.kernel_h,
                block.dwconv.kernel_w,
                block.dwconv.stride_h,
                block.dwconv.stride_w,
                block.dwconv.padding_h,
                block.dwconv.padding_w,
                block.dwconv.groups,
                &features.0,
                h,
                w,
            );

            // Per spatial position: LN → fc1 → GELU → (optional GRN) → fc2 → (optional LS) → residual
            let mlp_hidden = block.fc1.out_features();

            for p in 0..spatial {
                // Gather channel vector at spatial position p
                let mut token = Vec::with_capacity(dim);
                for c in 0..dim {
                    token.push(conv_out.0[c * spatial + p]);
                }

                // LayerNorm
                let mut tok_share = SharedVec64(token);
                klearu_mpc::normalization::layernorm_shared_64(
                    party,
                    &mut tok_share,
                    &block.norm.weight,
                    &block.norm.bias,
                    eps,
                    triples,
                    transport,
                )?;

                // MLP: fc1 → GELU_poly → (optional GRN) → fc2
                let empty_triples = vec![];
                let mut fc1_out = klearu_mpc::linear::shared_linear_forward_64(
                    party,
                    block.fc1.weights.as_raw_slice(),
                    dim,
                    mlp_hidden,
                    &tok_share,
                    &empty_triples,
                    transport,
                )?;
                if party == 0 {
                    for i in 0..mlp_hidden {
                        fc1_out.0[i] = fc1_out.0[i].wrapping_add(to_fixed64(block.fc1.bias[i]));
                    }
                }

                // GELU polynomial (no reveal)
                let gelu_triples = triples.generate(2 * mlp_hidden);
                let mut gelu_out = klearu_mpc::activation::gelu_approx_shared_64(
                    party,
                    &fc1_out,
                    &gelu_triples,
                    transport,
                )?;

                // Optional GRN (ConvNeXt V2)
                if let Some(ref grn) = block.grn {
                    grn_shared_64(
                        party,
                        &mut gelu_out.0,
                        mlp_hidden,
                        &grn.gamma,
                        &grn.beta,
                        triples,
                        transport,
                    )?;
                }

                // fc2
                let mut fc2_out = klearu_mpc::linear::shared_linear_forward_64(
                    party,
                    block.fc2.weights.as_raw_slice(),
                    mlp_hidden,
                    dim,
                    &gelu_out,
                    &empty_triples,
                    transport,
                )?;
                if party == 0 {
                    for i in 0..dim {
                        fc2_out.0[i] = fc2_out.0[i].wrapping_add(to_fixed64(block.fc2.bias[i]));
                    }
                }

                // Optional LayerScale (ConvNeXt V1)
                if let Some(ref ls) = block.layer_scale {
                    layer_scale_shared_64(&mut fc2_out.0, &ls.gamma);
                }

                // Residual: add to features (scatter back to [C, H, W])
                for c in 0..dim {
                    features.0[c * spatial + p] =
                        features.0[c * spatial + p].wrapping_add(fc2_out.0[c]);
                }
            }
        }
    }

    // === Final norm + global avg pool + head ===
    let final_dim = model.config.dims[3];
    let spatial = h * w;

    // LayerNorm2d (channel-wise)
    layernorm_2d_shared_64(
        party,
        &mut features.0,
        final_dim,
        h,
        w,
        &model.final_norm.weight,
        &model.final_norm.bias,
        eps,
        triples,
        transport,
    )?;

    // Global average pooling over spatial dims
    let mut pooled = vec![0u64; final_dim];
    for c in 0..final_dim {
        let mut sum = 0i128;
        for p in 0..spatial {
            sum += features.0[c * spatial + p] as i64 as i128;
        }
        let inv_n = 1.0 / spatial as f64;
        let inv_n_q32 = (inv_n * klearu_mpc::fixed_point::SCALE_64).round() as i64;
        pooled[c] = (((sum) * (inv_n_q32 as i128)) >> klearu_mpc::fixed_point::FRAC_BITS_64) as i64 as u64;
    }

    // Head
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
    use klearu_vision::config::ConvNextConfig;
    use klearu_mpc::beaver::dummy_triple_pair;
    use klearu_mpc::transport::memory_transport_pair;

    fn tiny_convnext_config() -> ConvNextConfig {
        ConvNextConfig {
            in_channels: 3,
            dims: [8, 16, 32, 64],
            depths: [1, 1, 1, 1],
            num_classes: 5,
            layer_scale_init: 1e-6,
            layer_norm_eps: 1e-6,
            is_v2: false,
        }
    }

    #[test]
    fn test_private_convnext_forward_runs() {
        let config = tiny_convnext_config();
        let model0 = ConvNextModel::new(config.clone());
        let model1 = ConvNextModel::new(config.clone());

        let image = vec![0.1f32; 3 * 224 * 224];

        let share0 = shared_image(0, &image);
        let share1 = shared_image(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_convnext_forward(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let result0 =
            private_convnext_forward(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let result1 = handle.join().unwrap().unwrap();

        assert_eq!(result0.len(), config.num_classes);
        for i in 0..config.num_classes {
            assert_eq!(result0[i], result1[i], "logit[{i}] mismatch");
        }
    }

    #[test]
    fn test_private_convnext_secure_runs() {
        use klearu_mpc::beaver::dummy_triple_pair_128;

        let config = tiny_convnext_config();
        let model0 = ConvNextModel::new(config.clone());
        let model1 = ConvNextModel::new(config.clone());

        // Use small image for speed
        let image_size = 32;
        let image = vec![0.1f32; 3 * image_size * image_size];

        let share0 = shared_image_64(0, &image);
        let share1 = shared_image_64(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(500000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_convnext_forward_secure(1, &model1, &share1, image_size, &mut gen1, &mut trans_b)
        });

        let logits0 = private_convnext_forward_secure(
            0, &model0, &share0, image_size, &mut gen0, &mut trans_a,
        )
        .unwrap();
        let logits1 = handle.join().unwrap().unwrap();

        assert_eq!(logits0.len(), config.num_classes);
        for i in 0..config.num_classes {
            assert!(logits0[i].is_finite(), "logit[{i}] not finite");
            assert!(logits1[i].is_finite());
        }
    }
}
