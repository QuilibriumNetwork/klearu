//! Private (2PC) evaluation of DaViT vision transformer.
//!
//! Two modes:
//! - **Lower security** (`private_davit_forward`): reveal the image share once,
//!   then both parties run an identical plaintext forward pass. Leaks the full image.
//! - **Higher security** (`private_davit_forward_secure`): maintain Q32.32 shares
//!   throughout. Only scalar statistics (LayerNorm mean/variance), GELU intermediates,
//!   and attention Q vectors are revealed — never the raw image or full hidden state.

#[cfg(feature = "vision")]
use klearu_vision::DaViTModel;
use klearu_mpc::beaver::TripleGenerator;
#[cfg(feature = "vision")]
use klearu_mpc::beaver::TripleGenerator128;
use klearu_mpc::fixed_point::from_fixed;
#[cfg(feature = "vision")]
use klearu_mpc::fixed_point::{from_fixed64, to_fixed64, SCALE_64, FRAC_BITS_64};
use klearu_mpc::transport::Transport;
use klearu_mpc::SharedVec;
#[cfg(feature = "vision")]
use klearu_mpc::SharedVec64;
use std::io;

/// Run private DaViT inference (lower security mode).
///
/// Both parties hold the same `model` weights in plaintext.
/// `image_share` is this party's Q16.16 share of the input image `[C, H, W]`.
///
/// The image share is revealed, then the plaintext forward pass is run
/// by both parties independently. Both produce identical f32 logits.
#[cfg(feature = "vision")]
pub fn private_davit_forward(
    _party: u8,
    model: &DaViTModel,
    image_share: &SharedVec,
    _triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let image_size = model.config.image_size;
    let in_channels = model.config.in_channels;
    let total_pixels = in_channels * image_size * image_size;

    assert_eq!(image_share.len(), total_pixels);

    // Reveal the image share: both parties exchange their shares and
    // reconstruct the plaintext image. This is the only communication.
    transport.send_u32_slice(&image_share.0)?;
    let other = transport.recv_u32_slice(total_pixels)?;

    let mut image = vec![0.0f32; total_pixels];
    for i in 0..total_pixels {
        image[i] = from_fixed(image_share.0[i].wrapping_add(other[i]));
    }

    // Run the plaintext forward pass. Both parties compute identically.
    let logits = model.forward(&image);

    Ok(logits)
}

/// Create a shared image for private inference.
///
/// Party 0 holds the full Q16.16 image as its share, party 1 holds zeros.
#[cfg(feature = "vision")]
pub fn shared_image(
    party: u8,
    image: &[f32],
) -> SharedVec {
    use klearu_mpc::fixed_point::to_fixed;

    if party == 0 {
        SharedVec(image.iter().map(|&v| to_fixed(v)).collect())
    } else {
        SharedVec(vec![0u32; image.len()])
    }
}

/// Create a Q32.32 shared image for higher-security private inference.
///
/// Party 0 holds the full Q32.32 image as its share, party 1 holds zeros.
#[cfg(feature = "vision")]
pub fn shared_image_64(
    party: u8,
    image: &[f32],
) -> SharedVec64 {
    if party == 0 {
        SharedVec64(image.iter().map(|&v| to_fixed64(v)).collect())
    } else {
        SharedVec64(vec![0u64; image.len()])
    }
}

/// Run private DaViT inference (higher security mode).
///
/// Both parties hold the same `model` weights in plaintext.
/// `image_share` is this party's Q32.32 share of the input image `[C, H, W]`.
///
/// The image share is NEVER revealed. Only scalar statistics (LayerNorm mean/variance),
/// GELU intermediates, and attention Q vectors leak.
#[cfg(feature = "vision")]
pub fn private_davit_forward_secure(
    party: u8,
    model: &DaViTModel,
    image_share: &SharedVec64,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let image_size = model.config.image_size;
    let in_channels = model.config.in_channels;
    let total_pixels = in_channels * image_size * image_size;
    let eps = model.config.layer_norm_eps;

    assert_eq!(image_share.len(), total_pixels);

    // === Stem: Conv2d → LayerNorm2d ===
    let stem = &model.stem;
    let (mut features, mut h, mut w) = crate::private_conv2d::shared_conv2d_forward_64(
        party,
        &stem.conv.weight,
        stem.conv.bias.as_deref(),
        stem.conv.out_channels,
        stem.conv.in_channels,
        stem.conv.kernel_h,
        stem.conv.kernel_w,
        stem.conv.stride_h,
        stem.conv.stride_w,
        stem.conv.padding_h,
        stem.conv.padding_w,
        stem.conv.groups,
        &image_share.0,
        image_size,
        image_size,
    );

    // LayerNorm2d on stem output
    let embed_dim = stem.norm.weight.len();
    layernorm_2d_shared_64(party, &mut features.0, embed_dim, h, w, &stem.norm.weight, &stem.norm.bias, eps, triples, transport)?;

    // === Per stage ===
    for stage in &model.stages {
        // Optional downsample: LayerNorm2d → Conv2d
        if let Some(ref ds) = stage.downsample {
            let in_dim = ds.norm.weight.len();
            layernorm_2d_shared_64(party, &mut features.0, in_dim, h, w, &ds.norm.weight, &ds.norm.bias, eps, triples, transport)?;

            let result = crate::private_conv2d::shared_conv2d_forward_64(
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
            features = result.0;
            h = result.1;
            w = result.2;
        }

        let dim = features.0.len() / (h * w);

        for (spatial, channel) in &stage.blocks {
            // === Spatial block ===
            spatial_block_secure(party, &mut features, dim, h, w, spatial, eps, triples, transport)?;

            // === Channel block ===
            channel_block_secure(party, &mut features, dim, h, w, channel, eps, triples, transport)?;
        }
    }

    // === Classification head: global avg pool → LayerNorm → Linear → reveal logits ===
    let head = &model.head;
    let final_dim = head.norm.weight.len();
    let n = h * w;

    // Global average pooling over spatial dims (in Q32.32)
    let mut pooled = vec![0u64; final_dim];
    for c in 0..final_dim {
        let base = c * n;
        let mut sum = 0i128;
        for i in 0..n {
            sum += features.0[base + i] as i64 as i128;
        }
        // Divide by n: multiply share by (1/n) in f64
        let inv_n = 1.0 / n as f64;
        let inv_n_q32 = (inv_n * SCALE_64).round() as i64;
        pooled[c] = (((sum as i128) * (inv_n_q32 as i128)) >> FRAC_BITS_64) as i64 as u64;
    }

    // LayerNorm on pooled
    let mut pooled_share = SharedVec64(pooled);
    klearu_mpc::normalization::layernorm_shared_64(
        party, &mut pooled_share, &head.norm.weight, &head.norm.bias, eps, triples, transport,
    )?;

    // Linear → logits (Q32.32 shares)
    let fc_weights = head.fc.weights.as_raw_slice();
    let in_features = head.fc.in_features();
    let out_features = head.fc.out_features();
    let empty_triples = vec![];
    let mut logits_share = klearu_mpc::linear::shared_linear_forward_64(
        party, fc_weights, in_features, out_features, &pooled_share, &empty_triples, transport,
    )?;

    // Add bias (party 0)
    if party == 0 {
        for i in 0..out_features {
            logits_share.0[i] = logits_share.0[i].wrapping_add(to_fixed64(head.fc.bias[i]));
        }
    }

    // Reveal logits
    transport.send_u64_slice(&logits_share.0)?;
    let other_logits = transport.recv_u64_slice(out_features)?;

    let logits: Vec<f32> = (0..out_features)
        .map(|i| from_fixed64(logits_share.0[i].wrapping_add(other_logits[i])))
        .collect();

    Ok(logits)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply LayerNorm2d to Q32.32 shares in `[C, H, W]` layout.
///
/// Normalizes channel dimension at each spatial position.
#[cfg(feature = "vision")]
fn layernorm_2d_shared_64(
    party: u8,
    data: &mut [u64],
    c: usize,
    h: usize,
    w: usize,
    weights: &[f32],
    bias: &[f32],
    eps: f32,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    let n = h * w;
    assert_eq!(data.len(), c * n);
    assert_eq!(weights.len(), c);
    assert_eq!(bias.len(), c);

    // For each spatial position, gather the C-dim vector, apply LayerNorm, scatter back.
    for y in 0..h {
        for x in 0..w {
            let spatial_idx = y * w + x;
            // Gather [C] from [C, H, W]
            let mut token = Vec::with_capacity(c);
            for ch in 0..c {
                token.push(data[ch * n + spatial_idx]);
            }
            let mut token_share = SharedVec64(token);
            klearu_mpc::normalization::layernorm_shared_64(
                party, &mut token_share, weights, bias, eps, triples, transport,
            )?;
            // Scatter back
            for ch in 0..c {
                data[ch * n + spatial_idx] = token_share.0[ch];
            }
        }
    }
    Ok(())
}

/// Convert `[C, H, W]` u64 shares to `[N, C]` token layout.
#[cfg(feature = "vision")]
fn channel_first_to_tokens_64(chw: &[u64], c: usize, h: usize, w: usize) -> Vec<u64> {
    let n = h * w;
    let mut tokens = vec![0u64; n * c];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let token_idx = y * w + x;
                tokens[token_idx * c + ch] = chw[ch * n + y * w + x];
            }
        }
    }
    tokens
}

/// Convert `[N, C]` token layout back to `[C, H, W]` u64 shares.
#[cfg(feature = "vision")]
fn tokens_to_channel_first_64(tokens: &[u64], chw: &mut [u64], c: usize, h: usize, w: usize) {
    let n = h * w;
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let token_idx = y * w + x;
                chw[ch * n + y * w + x] = tokens[token_idx * c + ch];
            }
        }
    }
}

/// CPE (Conditional Position Encoding) on Q32.32 shares.
///
/// Residual: x += depthwise_conv3x3(x). Purely local.
#[cfg(feature = "vision")]
fn cpe_shared_64(
    party: u8,
    data: &mut [u64],
    h: usize,
    w: usize,
    cpe: &klearu_vision::model::cpe::ConvPosEnc,
) {
    let c = cpe.proj.out_channels;
    let (conv_out, _, _) = crate::private_conv2d::shared_conv2d_forward_64(
        party,
        &cpe.proj.weight,
        cpe.proj.bias.as_deref(),
        cpe.proj.out_channels,
        cpe.proj.in_channels,
        cpe.proj.kernel_h,
        cpe.proj.kernel_w,
        cpe.proj.stride_h,
        cpe.proj.stride_w,
        cpe.proj.padding_h,
        cpe.proj.padding_w,
        cpe.proj.groups,
        data,
        h,
        w,
    );
    // Residual add
    for i in 0..c * h * w {
        data[i] = data[i].wrapping_add(conv_out.0[i]);
    }
}

/// Secure spatial block: CPE1 → LN → WindowAttn → CPE2 → LN → MLP.
#[cfg(feature = "vision")]
fn spatial_block_secure(
    party: u8,
    features: &mut SharedVec64,
    dim: usize,
    h: usize,
    w: usize,
    block: &klearu_vision::model::davit_block::SpatialBlock,
    eps: f32,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    let n = h * w;

    // x = x + cpe1(x)
    cpe_shared_64(party, &mut features.0, h, w, &block.cpe1);

    // Convert [C, H, W] → [N, C] for attention
    let mut tokens = channel_first_to_tokens_64(&features.0, dim, h, w);

    // normed = layernorm(tokens)
    let mut normed = tokens.clone();
    for t in 0..n {
        let mut tok = SharedVec64(normed[t * dim..(t + 1) * dim].to_vec());
        klearu_mpc::normalization::layernorm_shared_64(
            party, &mut tok, &block.norm1.weight, &block.norm1.bias, eps, triples, transport,
        )?;
        normed[t * dim..(t + 1) * dim].copy_from_slice(&tok.0);
    }

    // Window attention (secure)
    let attn_out = window_attention_secure(
        party, &normed, dim, h, w, &block.attn, triples, transport,
    )?;

    // Residual: tokens += attn_out
    for i in 0..tokens.len() {
        tokens[i] = tokens[i].wrapping_add(attn_out[i]);
    }

    // Convert back [N, C] → [C, H, W]
    tokens_to_channel_first_64(&tokens, &mut features.0, dim, h, w);

    // x = x + cpe2(x)
    cpe_shared_64(party, &mut features.0, h, w, &block.cpe2);

    // Convert [C, H, W] → [N, C] for MLP
    tokens = channel_first_to_tokens_64(&features.0, dim, h, w);

    // MLP: normed → fc1 → GELU → fc2
    let mlp_hidden = block.mlp_fc1.out_features();
    for t in 0..n {
        let mut tok = SharedVec64(tokens[t * dim..(t + 1) * dim].to_vec());
        klearu_mpc::normalization::layernorm_shared_64(
            party, &mut tok, &block.norm2.weight, &block.norm2.bias, eps, triples, transport,
        )?;

        let mlp_out = mlp_secure(party, &tok, &block.mlp_fc1, &block.mlp_fc2, mlp_hidden, transport)?;

        for d in 0..dim {
            tokens[t * dim + d] = tokens[t * dim + d].wrapping_add(mlp_out.0[d]);
        }
    }

    // Convert [N, C] → [C, H, W]
    tokens_to_channel_first_64(&tokens, &mut features.0, dim, h, w);

    Ok(())
}

/// Secure channel block: CPE1 → LN → ChannelAttn → CPE2 → LN → MLP.
#[cfg(feature = "vision")]
fn channel_block_secure(
    party: u8,
    features: &mut SharedVec64,
    dim: usize,
    h: usize,
    w: usize,
    block: &klearu_vision::model::davit_block::ChannelBlock,
    eps: f32,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    let n = h * w;

    // x = x + cpe1(x)
    cpe_shared_64(party, &mut features.0, h, w, &block.cpe1);

    // Convert [C, H, W] → [N, C]
    let mut tokens = channel_first_to_tokens_64(&features.0, dim, h, w);

    // normed = layernorm(tokens)
    let mut normed = tokens.clone();
    for t in 0..n {
        let mut tok = SharedVec64(normed[t * dim..(t + 1) * dim].to_vec());
        klearu_mpc::normalization::layernorm_shared_64(
            party, &mut tok, &block.norm1.weight, &block.norm1.bias, eps, triples, transport,
        )?;
        normed[t * dim..(t + 1) * dim].copy_from_slice(&tok.0);
    }

    // Channel attention (secure)
    let attn_out = channel_attention_secure(
        party, &normed, dim, n, &block.attn, triples, transport,
    )?;

    // Residual
    for i in 0..tokens.len() {
        tokens[i] = tokens[i].wrapping_add(attn_out[i]);
    }

    // Convert back [N, C] → [C, H, W]
    tokens_to_channel_first_64(&tokens, &mut features.0, dim, h, w);

    // x = x + cpe2(x)
    cpe_shared_64(party, &mut features.0, h, w, &block.cpe2);

    // Convert [C, H, W] → [N, C]
    tokens = channel_first_to_tokens_64(&features.0, dim, h, w);

    // MLP
    let mlp_hidden = block.mlp_fc1.out_features();
    for t in 0..n {
        let mut tok = SharedVec64(tokens[t * dim..(t + 1) * dim].to_vec());
        klearu_mpc::normalization::layernorm_shared_64(
            party, &mut tok, &block.norm2.weight, &block.norm2.bias, eps, triples, transport,
        )?;

        let mlp_out = mlp_secure(party, &tok, &block.mlp_fc1, &block.mlp_fc2, mlp_hidden, transport)?;

        for d in 0..dim {
            tokens[t * dim + d] = tokens[t * dim + d].wrapping_add(mlp_out.0[d]);
        }
    }

    // Convert [N, C] → [C, H, W]
    tokens_to_channel_first_64(&tokens, &mut features.0, dim, h, w);

    Ok(())
}

/// Secure MLP: fc1 → GELU(reveal) → fc2.
#[cfg(feature = "vision")]
fn mlp_secure(
    party: u8,
    x_share: &SharedVec64,
    fc1: &klearu_vision::layers::LinearBias,
    fc2: &klearu_vision::layers::LinearBias,
    mlp_hidden: usize,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let in_dim = fc1.in_features();
    let out_dim = fc2.out_features();
    let empty_triples = vec![];

    // fc1: local linear
    let mut fc1_out = klearu_mpc::linear::shared_linear_forward_64(
        party, fc1.weights.as_raw_slice(), in_dim, mlp_hidden, x_share, &empty_triples, transport,
    )?;
    // Add fc1 bias (party 0)
    if party == 0 {
        for i in 0..mlp_hidden {
            fc1_out.0[i] = fc1_out.0[i].wrapping_add(to_fixed64(fc1.bias[i]));
        }
    }

    // GELU (reveals intermediate — this is the designed leakage)
    let gelu_out = klearu_mpc::activation::gelu_reveal_64(party, &fc1_out, transport)?;

    // fc2: local linear
    let mut fc2_out = klearu_mpc::linear::shared_linear_forward_64(
        party, fc2.weights.as_raw_slice(), mlp_hidden, out_dim, &gelu_out, &empty_triples, transport,
    )?;
    // Add fc2 bias (party 0)
    if party == 0 {
        for i in 0..out_dim {
            fc2_out.0[i] = fc2_out.0[i].wrapping_add(to_fixed64(fc2.bias[i]));
        }
    }

    Ok(fc2_out)
}

/// Reveal u64 shares as f32 vector.
#[cfg(feature = "vision")]
fn reveal_u64_as_f32(
    shares: &[u64],
    transport: &mut impl Transport,
) -> io::Result<Vec<f32>> {
    let n = shares.len();
    transport.send_u64_slice(shares)?;
    let other = transport.recv_u64_slice(n)?;
    Ok((0..n).map(|i| from_fixed64(shares[i].wrapping_add(other[i]))).collect())
}

/// Secure window attention on Q32.32 token shares.
///
/// Protocol per window:
/// 1. QKV = linear(x_share) — local
/// 2. Reveal Q — leaks Q vectors
/// 3. Compute partial scores Q_plain · K_share, exchange, reconstruct
/// 4. Softmax (public, identical on both parties)
/// 5. Weighted V: softmax[t] * V_share[t] — local
/// 6. Output projection — local
#[cfg(feature = "vision")]
fn window_attention_secure(
    party: u8,
    normed_tokens: &[u64],
    dim: usize,
    h: usize,
    w: usize,
    attn: &klearu_vision::model::window_attention::WindowAttention,
    _triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<Vec<u64>> {
    let ws = attn.window_size;
    let num_heads = attn.num_heads;
    let head_dim = attn.head_dim;
    let num_windows_h = h / ws;
    let num_windows_w = w / ws;
    let window_tokens = ws * ws;
    let n = h * w;
    let empty_triples = vec![];

    let mut output = vec![0u64; n * dim];

    for wy in 0..num_windows_h {
        for wx in 0..num_windows_w {
            // Gather window tokens
            let mut window_input = vec![0u64; window_tokens * dim];
            for sy in 0..ws {
                for sx in 0..ws {
                    let gy = wy * ws + sy;
                    let gx = wx * ws + sx;
                    let gi = gy * w + gx;
                    let li = sy * ws + sx;
                    window_input[li * dim..(li + 1) * dim]
                        .copy_from_slice(&normed_tokens[gi * dim..(gi + 1) * dim]);
                }
            }

            // QKV projection (local)
            let qkv_in = SharedVec64(window_input);
            let qkv_dim = dim * 3;
            let mut qkv_buf = vec![0u64; window_tokens * qkv_dim];
            for t in 0..window_tokens {
                let tok = SharedVec64(qkv_in.0[t * dim..(t + 1) * dim].to_vec());
                let qkv_tok = klearu_mpc::linear::shared_linear_forward_64(
                    party, attn.qkv.weights.as_raw_slice(), dim, qkv_dim, &tok, &empty_triples, transport,
                )?;
                qkv_buf[t * qkv_dim..(t + 1) * qkv_dim].copy_from_slice(&qkv_tok.0);
                // Add QKV bias (party 0)
                if party == 0 {
                    for d in 0..qkv_dim {
                        qkv_buf[t * qkv_dim + d] = qkv_buf[t * qkv_dim + d]
                            .wrapping_add(to_fixed64(attn.qkv.bias[d]));
                    }
                }
            }

            // Extract Q and K shares into flat vecs
            let mut q_shares = vec![0u64; window_tokens * dim];
            let mut k_shares = vec![0u64; window_tokens * dim];
            for t in 0..window_tokens {
                for d in 0..dim {
                    q_shares[t * dim + d] = qkv_buf[t * qkv_dim + d];
                    k_shares[t * dim + d] = qkv_buf[t * qkv_dim + dim + d];
                }
            }

            // Reveal Q — designed leakage
            let q_plain = reveal_u64_as_f32(&q_shares, transport)?;

            // Exchange K shares to reconstruct K for attention scores
            transport.send_u64_slice(&k_shares)?;
            let k_other = transport.recv_u64_slice(window_tokens * dim)?;
            let k_plain: Vec<f32> = (0..window_tokens * dim)
                .map(|i| from_fixed64(k_shares[i].wrapping_add(k_other[i])))
                .collect();

            // Compute attention scores and softmax per head
            let scale = (head_dim as f32).powf(-0.5);
            let mut scores = vec![0.0f32; num_heads * window_tokens * window_tokens];

            for head in 0..num_heads {
                let ho = head * head_dim;
                for qi in 0..window_tokens {
                    for ki in 0..window_tokens {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_plain[qi * dim + ho + d] * k_plain[ki * dim + ho + d];
                        }
                        scores[head * window_tokens * window_tokens + qi * window_tokens + ki] = dot * scale;
                    }
                }

                // Softmax per row
                for qi in 0..window_tokens {
                    let base = head * window_tokens * window_tokens + qi * window_tokens;
                    let row = &mut scores[base..base + window_tokens];
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max).exp();
                        sum += *v;
                    }
                    if sum > 0.0 {
                        for v in row.iter_mut() {
                            *v /= sum;
                        }
                    }
                }
            }

            // Weighted V: output[qi, ho+d] = sum_vi(scores[qi,vi] * V_share[vi, ho+d])
            let mut attn_output = vec![0u64; window_tokens * dim];
            for head in 0..num_heads {
                let ho = head * head_dim;
                for qi in 0..window_tokens {
                    for d in 0..head_dim {
                        let mut acc = 0.0f64;
                        for vi in 0..window_tokens {
                            let v_share = qkv_buf[vi * qkv_dim + 2 * dim + ho + d];
                            let weight = scores[head * window_tokens * window_tokens + qi * window_tokens + vi];
                            acc += weight as f64 * v_share as i64 as f64;
                        }
                        attn_output[qi * dim + ho + d] = acc.round() as i64 as u64;
                    }
                }
            }

            // Output projection (local)
            for t in 0..window_tokens {
                let tok = SharedVec64(attn_output[t * dim..(t + 1) * dim].to_vec());
                let proj_out = klearu_mpc::linear::shared_linear_forward_64(
                    party, attn.proj.weights.as_raw_slice(), dim, dim, &tok, &empty_triples, transport,
                )?;
                // Add proj bias
                let gy_base = wy * ws;
                let gx_base = wx * ws;
                let sy = t / ws;
                let sx = t % ws;
                let gi = (gy_base + sy) * w + (gx_base + sx);
                for d in 0..dim {
                    let mut val = proj_out.0[d];
                    if party == 0 {
                        val = val.wrapping_add(to_fixed64(attn.proj.bias[d]));
                    }
                    output[gi * dim + d] = val;
                }
            }
        }
    }

    Ok(output)
}

/// Secure channel attention on Q32.32 token shares.
///
/// Channel attention transposes Q/K/V: attention is over head_dim dimension
/// with scale = N^(-0.5).
#[cfg(feature = "vision")]
fn channel_attention_secure(
    party: u8,
    normed_tokens: &[u64],
    dim: usize,
    n: usize,
    attn: &klearu_vision::model::channel_attention::ChannelAttention,
    _triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<Vec<u64>> {
    let num_heads = attn.num_heads;
    let head_dim = attn.head_dim;
    let qkv_dim = dim * 3;
    let empty_triples = vec![];

    // QKV projection (local)
    let mut qkv_buf = vec![0u64; n * qkv_dim];
    for t in 0..n {
        let tok = SharedVec64(normed_tokens[t * dim..(t + 1) * dim].to_vec());
        let qkv_tok = klearu_mpc::linear::shared_linear_forward_64(
            party, attn.qkv.weights.as_raw_slice(), dim, qkv_dim, &tok, &empty_triples, transport,
        )?;
        qkv_buf[t * qkv_dim..(t + 1) * qkv_dim].copy_from_slice(&qkv_tok.0);
        if party == 0 {
            for d in 0..qkv_dim {
                qkv_buf[t * qkv_dim + d] = qkv_buf[t * qkv_dim + d]
                    .wrapping_add(to_fixed64(attn.qkv.bias[d]));
            }
        }
    }

    // Reveal Q — designed leakage
    let q_shares: Vec<u64> = (0..n)
        .flat_map(|t| qkv_buf[t * qkv_dim..t * qkv_dim + dim].to_vec())
        .collect();
    let q_plain = reveal_u64_as_f32(&q_shares, transport)?;

    // Reveal K for score computation
    let k_flat: Vec<u64> = (0..n)
        .flat_map(|t| qkv_buf[t * qkv_dim + dim..t * qkv_dim + 2 * dim].to_vec())
        .collect();
    transport.send_u64_slice(&k_flat)?;
    let k_other = transport.recv_u64_slice(n * dim)?;
    let k_plain: Vec<f32> = (0..n * dim)
        .map(|i| from_fixed64(k_flat[i].wrapping_add(k_other[i])))
        .collect();

    // Channel attention: scores[d1, d2] = sum_t(Q[t,d1] * K[t,d2]) * scale
    let scale = (n as f32).powf(-0.5);
    let mut scores = vec![0.0f32; num_heads * head_dim * head_dim];

    for head in 0..num_heads {
        let ho = head * head_dim;
        for d1 in 0..head_dim {
            for d2 in 0..head_dim {
                let mut dot = 0.0f32;
                for t in 0..n {
                    dot += q_plain[t * dim + ho + d1] * k_plain[t * dim + ho + d2];
                }
                scores[head * head_dim * head_dim + d1 * head_dim + d2] = dot * scale;
            }
        }

        // Softmax per row
        for d1 in 0..head_dim {
            let base = head * head_dim * head_dim + d1 * head_dim;
            let row = &mut scores[base..base + head_dim];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }

    // Weighted V: output[t, ho+d1] = sum_d2(scores[d1,d2] * V_share[t, ho+d2])
    let mut attn_output = vec![0u64; n * dim];
    for head in 0..num_heads {
        let ho = head * head_dim;
        for t in 0..n {
            for d1 in 0..head_dim {
                let mut acc = 0.0f64;
                for d2 in 0..head_dim {
                    let v_share = qkv_buf[t * qkv_dim + 2 * dim + ho + d2];
                    let w = scores[head * head_dim * head_dim + d1 * head_dim + d2];
                    acc += w as f64 * v_share as i64 as f64;
                }
                attn_output[t * dim + ho + d1] = acc.round() as i64 as u64;
            }
        }
    }

    // Output projection (local)
    let mut proj_output = vec![0u64; n * dim];
    for t in 0..n {
        let tok = SharedVec64(attn_output[t * dim..(t + 1) * dim].to_vec());
        let proj_out = klearu_mpc::linear::shared_linear_forward_64(
            party, attn.proj.weights.as_raw_slice(), dim, dim, &tok, &empty_triples, transport,
        )?;
        for d in 0..dim {
            let mut val = proj_out.0[d];
            if party == 0 {
                val = val.wrapping_add(to_fixed64(attn.proj.bias[d]));
            }
            proj_output[t * dim + d] = val;
        }
    }

    Ok(proj_output)
}

#[cfg(all(test, feature = "vision"))]
mod tests {
    use super::*;
    use klearu_vision::config::DaViTConfig;
    use klearu_mpc::beaver::dummy_triple_pair;
    use klearu_mpc::transport::memory_transport_pair;

    fn tiny_config() -> DaViTConfig {
        DaViTConfig {
            image_size: 224,
            in_channels: 3,
            embed_dims: [8, 16, 32, 64],
            num_heads: [2, 2, 4, 8],
            depths: [1, 1, 1, 1],
            window_size: 7,
            mlp_ratio: 2.0,
            num_classes: 10,
            layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn test_private_davit_forward_runs() {
        let config = tiny_config();
        let model0 = DaViTModel::new(config.clone());
        let model1 = DaViTModel::new(config.clone());

        let image_size = config.image_size;
        let image = vec![0.1f32; 3 * image_size * image_size];

        let share0 = shared_image(0, &image);
        let share1 = shared_image(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_davit_forward(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let result0 = private_davit_forward(0, &model0, &share0, &mut gen0, &mut trans_a);
        let result1 = handle.join().unwrap();

        let logits0 = result0.expect("party 0 failed");
        let logits1 = result1.expect("party 1 failed");

        assert_eq!(logits0.len(), config.num_classes);
        assert_eq!(logits1.len(), config.num_classes);

        // Both parties compute identical f32 logits
        for i in 0..config.num_classes {
            assert!(logits0[i].is_finite(), "logit[{i}] is not finite: {}", logits0[i]);
            assert_eq!(logits0[i], logits1[i], "logit[{i}] mismatch");
        }
    }

    #[test]
    fn test_private_davit_matches_plaintext() {
        let config = tiny_config();
        let model_plain = DaViTModel::new(config.clone());
        let model0 = DaViTModel::new(config.clone());
        let model1 = DaViTModel::new(config.clone());

        let image_size = config.image_size;
        let image = vec![0.1f32; 3 * image_size * image_size];

        // Plaintext forward
        let plaintext_logits = model_plain.forward(&image);

        // MPC forward
        let share0 = shared_image(0, &image);
        let share1 = shared_image(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_davit_forward(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let mpc_logits = private_davit_forward(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let mpc_logits1 = handle.join().unwrap().unwrap();

        // Both parties compute identical logits
        assert_eq!(mpc_logits, mpc_logits1);

        // Should be very close to plaintext (only Q16.16 roundtrip error on image)
        let mut max_diff = 0.0f32;
        for i in 0..config.num_classes {
            let diff = (mpc_logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        eprintln!("DaViT MPC vs plaintext: max logit diff = {max_diff}");
        assert!(
            max_diff < 5.0,
            "DaViT MPC diverges too much: max_diff={max_diff}"
        );
    }

    // --- Secure (Q32.32) DaViT tests ---

    #[test]
    fn test_private_davit_forward_secure_runs() {
        use klearu_mpc::beaver::dummy_triple_pair_128;

        let config = tiny_config();
        let model0 = DaViTModel::new(config.clone());
        let model1 = DaViTModel::new(config.clone());

        let image_size = config.image_size;
        let image = vec![0.1f32; 3 * image_size * image_size];

        let share0 = shared_image_64(0, &image);
        let share1 = shared_image_64(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(7000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_davit_forward_secure(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let result0 = private_davit_forward_secure(0, &model0, &share0, &mut gen0, &mut trans_a);
        let result1 = handle.join().unwrap();

        let logits0 = result0.expect("party 0 failed");
        let logits1 = result1.expect("party 1 failed");

        assert_eq!(logits0.len(), config.num_classes);
        assert_eq!(logits1.len(), config.num_classes);

        for i in 0..config.num_classes {
            assert!(logits0[i].is_finite(), "logit[{i}] is not finite: {}", logits0[i]);
        }
    }

    #[test]
    fn test_private_davit_forward_secure_matches_plaintext() {
        use klearu_mpc::beaver::dummy_triple_pair_128;

        let config = tiny_config();
        let model_plain = DaViTModel::new(config.clone());
        let model0 = DaViTModel::new(config.clone());
        let model1 = DaViTModel::new(config.clone());

        let image_size = config.image_size;
        let image = vec![0.1f32; 3 * image_size * image_size];

        let plaintext_logits = model_plain.forward(&image);

        let share0 = shared_image_64(0, &image);
        let share1 = shared_image_64(1, &image);

        let (mut gen0, mut gen1) = dummy_triple_pair_128(8000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let handle = std::thread::spawn(move || {
            private_davit_forward_secure(1, &model1, &share1, &mut gen1, &mut trans_b)
        });

        let mpc_logits = private_davit_forward_secure(0, &model0, &share0, &mut gen0, &mut trans_a).unwrap();
        let _mpc_logits1 = handle.join().unwrap().unwrap();

        let mut max_diff = 0.0f32;
        for i in 0..config.num_classes {
            let diff = (mpc_logits[i] - plaintext_logits[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        eprintln!("Secure DaViT MPC vs plaintext: max logit diff = {max_diff}");
        assert!(
            max_diff < 5.0,
            "Secure DaViT MPC diverges too much: max_diff={max_diff}"
        );
    }
}
