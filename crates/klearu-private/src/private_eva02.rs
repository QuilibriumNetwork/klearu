//! Private (2PC) evaluation of EVA-02 Vision Transformer.
//!
//! Two modes:
//! - **Lower security** (`private_eva02_forward`): reveal image, plaintext forward.
//! - **Higher security** (`private_eva02_forward_secure`): Q32.32 shares,
//!   Beaver-based attention, polynomial SwiGLU, sub-layer norm.
//!   RoPE is applied as public rotation on Q/K shares (purely local).

#[cfg(feature = "vision")]
use klearu_vision::model::eva02::EVA02Model;
use klearu_mpc::beaver::{TripleGenerator, TripleGenerator128};
use klearu_mpc::fixed_point::{from_fixed, to_fixed64};
use klearu_mpc::sharing::{SharedVec, SharedVec64};
use klearu_mpc::transport::Transport;
use std::io;

use crate::private_vision_helpers::*;

/// Apply 2D RoPE rotation to Q and K shares in Q32.32.
///
/// RoPE is a public rotation matrix (cos/sin are deterministic from position),
/// so `q_rot[2i] = cos·q[2i] - sin·q[2i+1]` is local (public × share).
#[cfg(feature = "vision")]
fn rope_on_shares_64(
    q_shares: &mut [u64],
    k_shares: &mut [u64],
    num_tokens: usize,
    head_dim: usize,
    grid_h: usize,
    grid_w: usize,
    theta: f32,
) {
    let num_pairs = head_dim / 2;
    let half_pairs = num_pairs / 2;

    for t in 0..num_tokens {
        let row = t / grid_w;
        let col = t % grid_w;
        if row >= grid_h || col >= grid_w {
            continue;
        }

        for p in 0..num_pairs {
            let (pos, freq_idx, freq_base) = if p < half_pairs {
                (row, p, half_pairs)
            } else {
                (col, p - half_pairs, half_pairs)
            };

            let freq = 1.0 / theta.powf(2.0 * freq_idx as f32 / freq_base as f32);
            let angle = pos as f32 * freq;
            let c = angle.cos() as f64;
            let s = angle.sin() as f64;

            let d0 = p * 2;
            let d1 = p * 2 + 1;
            let qi = t * head_dim;

            // Q rotation: local (public cos/sin × share)
            let q0 = q_shares[qi + d0] as i64 as f64;
            let q1 = q_shares[qi + d1] as i64 as f64;
            q_shares[qi + d0] = (c * q0 - s * q1).round() as i64 as u64;
            q_shares[qi + d1] = (s * q0 + c * q1).round() as i64 as u64;

            // K rotation
            let k0 = k_shares[qi + d0] as i64 as f64;
            let k1 = k_shares[qi + d1] as i64 as f64;
            k_shares[qi + d0] = (c * k0 - s * k1).round() as i64 as u64;
            k_shares[qi + d1] = (s * k0 + c * k1).round() as i64 as u64;
        }
    }
}

/// Run private EVA-02 inference (lower security).
#[cfg(feature = "vision")]
pub fn private_eva02_forward(
    _party: u8,
    model: &EVA02Model,
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

/// Run private EVA-02 inference (higher security).
///
/// SwiGLU uses polynomial SiLU + Beaver multiply (no reveal).
/// RoPE is applied as public rotation on Q/K shares (local, no communication).
/// Sub-layer norm reveals only scalar statistics.
#[cfg(feature = "vision")]
pub fn private_eva02_forward_secure(
    party: u8,
    model: &EVA02Model,
    image_share: &SharedVec64,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let image_size = model.config.image_size;
    let in_channels = model.config.in_channels;
    let embed_dim = model.config.embed_dim;
    let num_heads = model.config.num_heads;
    let head_dim = embed_dim / num_heads;
    let eps = model.config.layer_norm_eps;
    let total_pixels = in_channels * image_size * image_size;

    assert_eq!(image_share.len(), total_pixels);

    // === Patch embedding ===
    let patch = &model.patch_embed;
    let (patch_out, grid_h, grid_w) = crate::private_conv2d::shared_conv2d_forward_64(
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
    let patch_tokens = channel_first_to_tokens_64(&patch_out.0, embed_dim, grid_h, grid_w);

    // Prepend CLS + position embeddings
    use klearu_vision::model::vit::PoolType;
    let has_cls = model.config.pool_type == PoolType::Cls;
    let seq_len = if has_cls { num_patches + 1 } else { num_patches };
    let mut tokens = vec![0u64; seq_len * embed_dim];

    if has_cls {
        if party == 0 {
            for d in 0..embed_dim {
                tokens[d] = to_fixed64(model.cls_token[d]);
            }
        }
        tokens[embed_dim..].copy_from_slice(&patch_tokens);
    } else {
        tokens.copy_from_slice(&patch_tokens);
    }

    if party == 0 {
        for i in 0..seq_len * embed_dim {
            tokens[i] = tokens[i].wrapping_add(to_fixed64(model.pos_embed[i]));
        }
    }

    let has_rope = model.rope.is_some();
    let rope_theta = 10000.0f32;

    // === Transformer blocks ===
    for block in &model.blocks {
        // Pre-norm → attention + sub-norm → residual
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

        // QKV projection (local)
        let qkv_dim = embed_dim * 3;
        let empty_triples = vec![];
        let mut qkv_buf = vec![0u64; seq_len * qkv_dim];
        for t in 0..seq_len {
            let tok = SharedVec64(normed[t * embed_dim..(t + 1) * embed_dim].to_vec());
            let qkv_tok = klearu_mpc::linear::shared_linear_forward_64(
                party,
                block.attn.qkv.weights.as_raw_slice(),
                embed_dim,
                qkv_dim,
                &tok,
                &empty_triples,
                transport,
            )?;
            qkv_buf[t * qkv_dim..(t + 1) * qkv_dim].copy_from_slice(&qkv_tok.0);
            if party == 0 {
                for d in 0..qkv_dim {
                    qkv_buf[t * qkv_dim + d] =
                        qkv_buf[t * qkv_dim + d].wrapping_add(to_fixed64(block.attn.qkv.bias[d]));
                }
            }
        }

        // Apply RoPE to Q and K shares per head (local rotation)
        if has_rope {
            for head in 0..num_heads {
                let ho = head * head_dim;

                // Extract per-head Q and K for patch tokens (skip CLS for RoPE)
                let rope_start = if has_cls { 1 } else { 0 };
                let rope_tokens = seq_len - rope_start;

                let mut q_head = vec![0u64; rope_tokens * head_dim];
                let mut k_head = vec![0u64; rope_tokens * head_dim];
                for t in 0..rope_tokens {
                    let ti = rope_start + t;
                    for d in 0..head_dim {
                        q_head[t * head_dim + d] = qkv_buf[ti * qkv_dim + ho + d];
                        k_head[t * head_dim + d] = qkv_buf[ti * qkv_dim + embed_dim + ho + d];
                    }
                }

                rope_on_shares_64(
                    &mut q_head,
                    &mut k_head,
                    rope_tokens,
                    head_dim,
                    grid_h,
                    grid_w,
                    rope_theta,
                );

                // Write back
                for t in 0..rope_tokens {
                    let ti = rope_start + t;
                    for d in 0..head_dim {
                        qkv_buf[ti * qkv_dim + ho + d] = q_head[t * head_dim + d];
                        qkv_buf[ti * qkv_dim + embed_dim + ho + d] = k_head[t * head_dim + d];
                    }
                }
            }
        }

        // Per-head Beaver attention scores + softmax + weighted V
        let scale = (head_dim as f32).powf(-0.5);
        let mut attn_output = vec![0u64; seq_len * embed_dim];

        for head in 0..num_heads {
            let ho = head * head_dim;

            let mut q_head = vec![0u64; seq_len * head_dim];
            let mut k_head = vec![0u64; seq_len * head_dim];
            let mut v_head = vec![0u64; seq_len * head_dim];
            for t in 0..seq_len {
                for d in 0..head_dim {
                    q_head[t * head_dim + d] = qkv_buf[t * qkv_dim + ho + d];
                    k_head[t * head_dim + d] = qkv_buf[t * qkv_dim + embed_dim + ho + d];
                    v_head[t * head_dim + d] = qkv_buf[t * qkv_dim + 2 * embed_dim + ho + d];
                }
            }

            let score_triples = triples.generate(seq_len * seq_len * head_dim);
            let score_shares = klearu_mpc::attention_mpc::beaver_attention_scores_64(
                party,
                &q_head,
                &k_head,
                seq_len,
                head_dim,
                &score_triples,
                transport,
            )?;

            let head_out = klearu_mpc::attention_mpc::softmax_weighted_v_64(
                party,
                &score_shares,
                &v_head,
                seq_len,
                head_dim,
                scale,
                transport,
            )?;

            for t in 0..seq_len {
                for d in 0..head_dim {
                    attn_output[t * embed_dim + ho + d] = head_out[t * head_dim + d];
                }
            }
        }

        // Output projection (local)
        let mut proj_output = vec![0u64; seq_len * embed_dim];
        for t in 0..seq_len {
            let tok = SharedVec64(attn_output[t * embed_dim..(t + 1) * embed_dim].to_vec());
            let proj_out = klearu_mpc::linear::shared_linear_forward_64(
                party,
                block.attn.proj.weights.as_raw_slice(),
                embed_dim,
                embed_dim,
                &tok,
                &empty_triples,
                transport,
            )?;
            for d in 0..embed_dim {
                let mut val = proj_out.0[d];
                if party == 0 {
                    val = val.wrapping_add(to_fixed64(block.attn.proj.bias[d]));
                }
                proj_output[t * embed_dim + d] = val;
            }
        }

        // Sub-layer norm
        for t in 0..seq_len {
            let mut tok = SharedVec64(proj_output[t * embed_dim..(t + 1) * embed_dim].to_vec());
            klearu_mpc::normalization::layernorm_shared_64(
                party,
                &mut tok,
                &block.sub_norm.weight,
                &block.sub_norm.bias,
                eps,
                triples,
                transport,
            )?;
            proj_output[t * embed_dim..(t + 1) * embed_dim].copy_from_slice(&tok.0);
        }

        // Residual
        for i in 0..tokens.len() {
            tokens[i] = tokens[i].wrapping_add(proj_output[i]);
        }

        // Pre-norm → SwiGLU MLP → residual
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

            let mlp_out = swiglu_mlp_secure(
                party,
                &tok,
                block.mlp.gate_proj.weights.as_raw_slice(),
                &block.mlp.gate_proj.bias,
                block.mlp.up_proj.weights.as_raw_slice(),
                &block.mlp.up_proj.bias,
                block.mlp.down_proj.weights.as_raw_slice(),
                &block.mlp.down_proj.bias,
                embed_dim,
                block.mlp.gate_proj.out_features(),
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
    let pooled = if has_cls {
        tokens[..embed_dim].to_vec()
    } else {
        global_avg_pool_64(&tokens, seq_len, embed_dim)
    };

    // === Head ===
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

    fn tiny_eva02_config() -> ViTConfig {
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
    fn test_private_eva02_forward_runs() {
        let config = tiny_eva02_config();
        let model0 = EVA02Model::new(config.clone());
        let model1 = EVA02Model::new(config.clone());

        let image = vec![0.1f32; 3 * 32 * 32];

        let share0 = shared_image(0, &image);
        let share1 = shared_image(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_eva02_forward(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let result0 =
            private_eva02_forward(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let result1 = handle.join().unwrap().unwrap();

        assert_eq!(result0.len(), config.num_classes);
        for i in 0..config.num_classes {
            assert_eq!(result0[i], result1[i], "logit[{i}] mismatch");
        }
    }

    #[test]
    fn test_private_eva02_secure_runs() {
        use klearu_mpc::beaver::dummy_triple_pair_128;

        let config = tiny_eva02_config();
        let model0 = EVA02Model::new(config.clone());
        let model1 = EVA02Model::new(config.clone());

        let image = vec![0.1f32; 3 * 32 * 32];

        let share0 = shared_image_64(0, &image);
        let share1 = shared_image_64(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(100000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_eva02_forward_secure(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let logits0 =
            private_eva02_forward_secure(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let logits1 = handle.join().unwrap().unwrap();

        assert_eq!(logits0.len(), config.num_classes);
        for i in 0..config.num_classes {
            assert!(logits0[i].is_finite(), "logit[{i}] not finite");
            assert!(logits1[i].is_finite());
        }
    }
}
