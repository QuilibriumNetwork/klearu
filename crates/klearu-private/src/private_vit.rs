//! Private (2PC) evaluation of ViT (Vision Transformer).
//!
//! Two modes:
//! - **Lower security** (`private_vit_forward`): reveal the image share once,
//!   then both parties run identical plaintext forward pass.
//! - **Higher security** (`private_vit_forward_secure`): maintain Q32.32 shares
//!   throughout. Q and K vectors are never revealed individually — only their
//!   pairwise dot products (attention pattern) leak via `self_attention_secure_64`.
//!   GELU intermediates use polynomial approximation (no reveal).

#[cfg(feature = "vision")]
use klearu_vision::model::vit::ViTModel;
use klearu_mpc::beaver::TripleGenerator128;
use klearu_mpc::fixed_point::{from_fixed, to_fixed64};
use klearu_mpc::transport::Transport;
use klearu_mpc::sharing::SharedVec64;
use klearu_mpc::beaver::TripleGenerator;
use klearu_mpc::SharedVec;
use std::io;

use crate::private_vision_helpers::*;

/// Run private ViT inference (lower security mode).
///
/// Image is revealed, then both parties run identical plaintext forward.
#[cfg(feature = "vision")]
pub fn private_vit_forward(
    _party: u8,
    model: &ViTModel,
    image_share: &SharedVec,
    _triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let image_size = model.config.image_size;
    let in_channels = model.config.in_channels;
    let total_pixels = in_channels * image_size * image_size;

    assert_eq!(image_share.len(), total_pixels);

    transport.send_u32_slice(&image_share.0)?;
    let other = transport.recv_u32_slice(total_pixels)?;

    let mut image = vec![0.0f32; total_pixels];
    for i in 0..total_pixels {
        image[i] = from_fixed(image_share.0[i].wrapping_add(other[i]));
    }

    Ok(model.forward(&image))
}

/// Run private ViT inference (higher security mode).
///
/// Maintains Q32.32 shares throughout. Never reveals Q, K, or GELU intermediates.
/// Leaks: scalar LayerNorm statistics, attention pattern (Q·K^T scores).
#[cfg(feature = "vision")]
pub fn private_vit_forward_secure(
    party: u8,
    model: &ViTModel,
    image_share: &SharedVec64,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let image_size = model.config.image_size;
    let in_channels = model.config.in_channels;
    let embed_dim = model.config.embed_dim;
    let eps = model.config.layer_norm_eps;
    let total_pixels = in_channels * image_size * image_size;

    assert_eq!(image_share.len(), total_pixels);

    // === Patch embedding: Conv2d ===
    let patch = &model.patch_embed;
    let (tokens_share, _ph, _pw) = crate::private_conv2d::shared_conv2d_forward_64(
        party,
        &patch.proj.weight,
        patch.proj.bias.as_deref(),
        patch.proj.out_channels,
        patch.proj.in_channels,
        patch.proj.kernel_h,
        patch.proj.kernel_w,
        patch.proj.stride_h,
        patch.proj.stride_w,
        patch.proj.padding_h,
        patch.proj.padding_w,
        patch.proj.groups,
        &image_share.0,
        image_size,
        image_size,
    );

    let num_patches = patch.num_patches;

    // Reshape from [embed_dim, ph, pw] to [num_patches, embed_dim]
    let patch_tokens = channel_first_to_tokens_64(&tokens_share.0, embed_dim, _ph, _pw);

    // Prepend CLS token + add position embeddings
    let seq_len = num_patches + 1;
    let mut tokens = vec![0u64; seq_len * embed_dim];

    // CLS token (party 0 adds, party 1 zeros)
    if party == 0 {
        for d in 0..embed_dim {
            tokens[d] = to_fixed64(model.cls_token[d]);
        }
    }
    // Patch tokens
    tokens[embed_dim..].copy_from_slice(&patch_tokens);

    // Position embeddings (party 0 adds)
    if party == 0 {
        for i in 0..seq_len * embed_dim {
            tokens[i] = tokens[i].wrapping_add(to_fixed64(model.pos_embed[i]));
        }
    }

    // === Transformer blocks ===
    for block in &model.blocks {
        // Pre-norm → attention → residual
        let mut normed = tokens.clone();
        for t in 0..seq_len {
            let mut tok = SharedVec64(normed[t * embed_dim..(t + 1) * embed_dim].to_vec());
            klearu_mpc::normalization::layernorm_shared_64(
                party,
                &mut tok,
                &block.norm1.weight,
                &block.norm1.bias,
                eps,
                triples,
                transport,
            )?;
            normed[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&tok.0);
        }

        // Self-attention (secure, Beaver-based scores)
        let attn_out = klearu_mpc::attention_mpc::self_attention_secure_64(
            party,
            &normed,
            seq_len,
            embed_dim,
            model.config.num_heads,
            block.attn.qkv.weights.as_raw_slice(),
            &block.attn.qkv.bias,
            block.attn.proj.weights.as_raw_slice(),
            &block.attn.proj.bias,
            &mut |n| triples.generate(n),
            transport,
        )?;

        // Residual
        for i in 0..tokens.len() {
            tokens[i] = tokens[i].wrapping_add(attn_out[i]);
        }

        // Pre-norm → MLP → residual
        for t in 0..seq_len {
            let mut tok = SharedVec64(tokens[t * embed_dim..(t + 1) * embed_dim].to_vec());
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
                embed_dim,
                block.mlp_fc1.out_features(),
                block.mlp_fc2.weights.as_raw_slice(),
                &block.mlp_fc2.bias,
                embed_dim,
                triples,
                transport,
            )?;

            for d in 0..embed_dim {
                tokens[t * embed_dim + d] = tokens[t * embed_dim + d].wrapping_add(mlp_out.0[d]);
            }
        }
    }

    // === Final norm ===
    for t in 0..seq_len {
        let mut tok = SharedVec64(tokens[t * embed_dim..(t + 1) * embed_dim].to_vec());
        klearu_mpc::normalization::layernorm_shared_64(
            party,
            &mut tok,
            &model.norm.weight,
            &model.norm.bias,
            eps,
            triples,
            transport,
        )?;
        tokens[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&tok.0);
    }

    // === Pooling ===
    use klearu_vision::model::vit::PoolType;
    let pooled = match model.pool_type {
        PoolType::Cls => {
            // CLS token is first token
            tokens[..embed_dim].to_vec()
        }
        PoolType::Mean => {
            // Average pool over patch tokens (exclude CLS)
            global_avg_pool_64(&tokens[embed_dim..], num_patches, embed_dim)
        }
    };

    // === Classification head ===
    let empty_triples = vec![];
    let pooled_share = SharedVec64(pooled);
    let mut logits = klearu_mpc::linear::shared_linear_forward_64(
        party,
        model.head.weights.as_raw_slice(),
        embed_dim,
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
    use klearu_vision::config::ViTConfig;
    use klearu_mpc::beaver::dummy_triple_pair;
    use klearu_mpc::transport::memory_transport_pair;

    fn tiny_vit_config() -> ViTConfig {
        ViTConfig {
            image_size: 32,
            in_channels: 3,
            patch_size: 8,
            embed_dim: 8,
            num_heads: 2,
            num_layers: 1,
            mlp_ratio: 2.0,
            num_classes: 5,
            layer_norm_eps: 1e-5,
            pool_type: klearu_vision::model::vit::PoolType::Cls,
        }
    }

    #[test]
    fn test_private_vit_forward_runs() {
        let config = tiny_vit_config();
        let model0 = ViTModel::new(config.clone());
        let model1 = ViTModel::new(config.clone());

        let image = vec![0.1f32; 3 * 32 * 32];

        let share0 = shared_image(0, &image);
        let share1 = shared_image(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_vit_forward(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let result0 = private_vit_forward(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let result1 = handle.join().unwrap().unwrap();

        assert_eq!(result0.len(), config.num_classes);
        assert_eq!(result1.len(), config.num_classes);
        for i in 0..config.num_classes {
            assert_eq!(result0[i], result1[i], "logit[{i}] mismatch");
        }
    }

    #[test]
    fn test_private_vit_matches_plaintext() {
        let config = tiny_vit_config();
        let model_plain = ViTModel::new(config.clone());
        let model0 = ViTModel::new(config.clone());
        let model1 = ViTModel::new(config.clone());

        let image = vec![0.1f32; 3 * 32 * 32];
        let plaintext_logits = model_plain.forward(&image);

        let share0 = shared_image(0, &image);
        let share1 = shared_image(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_vit_forward(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let mpc_logits = private_vit_forward(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let _ = handle.join().unwrap().unwrap();

        let mut max_diff = 0.0f32;
        for i in 0..config.num_classes {
            let diff = (mpc_logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 5.0,
            "ViT MPC vs plaintext: max_diff={max_diff}"
        );
    }

    #[test]
    fn test_private_vit_forward_secure_runs() {
        use klearu_mpc::beaver::dummy_triple_pair_128;

        let config = tiny_vit_config();
        let model0 = ViTModel::new(config.clone());
        let model1 = ViTModel::new(config.clone());

        let image = vec![0.1f32; 3 * 32 * 32];

        let share0 = shared_image_64(0, &image);
        let share1 = shared_image_64(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(50000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_vit_forward_secure(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let logits0 = private_vit_forward_secure(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let logits1 = handle.join().unwrap().unwrap();

        assert_eq!(logits0.len(), config.num_classes);
        for i in 0..config.num_classes {
            assert!(logits0[i].is_finite(), "logit[{i}] not finite");
            assert!(logits1[i].is_finite(), "logit[{i}] not finite");
        }
    }
}
