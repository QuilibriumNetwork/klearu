//! Semantics tests for the Wan 3D causal VAE primitives.
//!
//! These pin behaviour that is easy to get subtly wrong when porting from the
//! reference: which side the temporal padding goes on, the width-major
//! patchify channel order, and — most importantly — that chunked inference
//! with the feature cache reproduces a whole-clip run exactly.

use super::*;

/// Deterministic ramp fill, distinct per seed.
fn fill(v: &mut Vol, seed: f32) {
    for (i, x) in v.data.iter_mut().enumerate() {
        *x = ((i as f32 * 0.7 + seed).sin()) * 0.9;
    }
}

fn conv_3x3x3(in_c: usize, out_c: usize, seed: f32) -> CausalConv3d {
    let mut c = CausalConv3d::new(in_c, out_c, (3, 3, 3), (1, 1, 1), (1, 1, 1), true);
    for (i, w) in c.weight.iter_mut().enumerate() {
        *w = ((i as f32 * 0.31 + seed).cos()) * 0.2;
    }
    if let Some(b) = c.bias.as_mut() {
        for (i, x) in b.iter_mut().enumerate() {
            *x = (i as f32) * 0.05;
        }
    }
    c
}

fn max_diff(a: &Vol, b: &Vol) -> f32 {
    assert_eq!((a.c, a.t, a.h, a.w), (b.c, b.t, b.h, b.w), "shape mismatch");
    a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

// -- CausalConv3d ---------------------------------------------------------

#[test]
fn causal_conv_preserves_temporal_length() {
    // kernel 3 / pad 1 pads two frames on the past side and none on the
    // future side, so T is unchanged.
    let conv = conv_3x3x3(2, 3, 0.1);
    let mut x = Vol::zeros(2, 5, 4, 4);
    fill(&mut x, 1.0);
    let y = conv.forward(&x, None);
    assert_eq!((y.c, y.t, y.h, y.w), (3, 5, 4, 4));
}

#[test]
fn causal_conv_output_cannot_see_the_future() {
    // Perturbing frame `k` must leave every output frame < k untouched. This
    // is the property the whole chunked-decode scheme rests on.
    let conv = conv_3x3x3(2, 2, 0.7);
    let mut x = Vol::zeros(2, 6, 3, 3);
    fill(&mut x, 2.0);
    let base = conv.forward(&x, None);

    for k in 1..x.t {
        let mut perturbed = x.clone();
        for c in 0..x.c {
            for h in 0..x.h {
                for w in 0..x.w {
                    let i = perturbed.idx(c, k, h, w);
                    perturbed.data[i] += 10.0;
                }
            }
        }
        let got = conv.forward(&perturbed, None);
        for t in 0..k {
            for c in 0..got.c {
                for h in 0..got.h {
                    for w in 0..got.w {
                        assert!(
                            (got.at(c, t, h, w) - base.at(c, t, h, w)).abs() < 1e-6,
                            "frame {k} leaked into output frame {t}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn causal_conv_chunked_matches_whole_clip() {
    // The reason the cache exists: decoding [0..3) then [3..7) with the
    // trailing-frame cache must equal decoding [0..7) in one go.
    let conv = conv_3x3x3(2, 3, 0.4);
    let mut x = Vol::zeros(2, 7, 4, 5);
    fill(&mut x, 3.0);
    let whole = conv.forward(&x, None);

    let split = 3;
    let mut x1 = Vol::zeros(2, split, x.h, x.w);
    let mut x2 = Vol::zeros(2, x.t - split, x.h, x.w);
    for c in 0..x.c {
        for t in 0..x.t {
            for h in 0..x.h {
                for w in 0..x.w {
                    let v = x.at(c, t, h, w);
                    if t < split {
                        let i = x1.idx(c, t, h, w);
                        x1.data[i] = v;
                    } else {
                        let i = x2.idx(c, t - split, h, w);
                        x2.data[i] = v;
                    }
                }
            }
        }
    }

    let y1 = conv.forward(&x1, None);
    let cache = x1.tail_frames(CACHE_T);
    let y2 = conv.forward(&x2, Some(&cache));
    assert_eq!(y1.t, split);
    assert_eq!(y2.t, x.t - split);

    let joined = Vol::concat_time(&y1, &y2);
    let d = max_diff(&joined, &whole);
    assert!(d < 1e-5, "chunked decode diverged from whole-clip decode: max_diff={d}");
}

#[test]
fn causal_conv_identity_kernel_reproduces_input() {
    // A delta at the last temporal tap and the spatial centre is the identity,
    // which also confirms the past-side padding does not shift the output.
    let mut conv = CausalConv3d::new(1, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), false);
    // weight index (a*kh + b)*kw + d, with a = temporal tap.
    conv.weight[(2 * 3 + 1) * 3 + 1] = 1.0;
    let mut x = Vol::zeros(1, 4, 3, 3);
    fill(&mut x, 5.0);
    let y = conv.forward(&x, None);
    assert!(max_diff(&y, &x) < 1e-6, "delta kernel should be the identity");
}

#[test]
fn causal_conv_without_temporal_pad_takes_no_cache() {
    // pad_t = 0 means no past padding to replace, so the cache is ignored and
    // the temporal length shrinks by kt-1.
    let conv = CausalConv3d::new(1, 1, (3, 1, 1), (1, 1, 1), (0, 0, 0), false);
    assert_eq!(conv.cache_frames(), 0);
    let mut x = Vol::zeros(1, 5, 2, 2);
    fill(&mut x, 6.0);
    let y = conv.forward(&x, None);
    assert_eq!(y.t, 3);
}

// -- RmsNormChannel -------------------------------------------------------

#[test]
fn rms_norm_matches_reference_formula() {
    // out[c] = x[c] / ||x||_2 * sqrt(C) * gamma[c] + bias[c]
    let x = Vol::new(vec![3.0, 4.0], 2, 1, 1, 1);
    let mut n = RmsNormChannel::new(2, true);
    n.gamma = vec![2.0, 0.5];
    n.bias = Some(vec![1.0, -1.0]);
    let y = n.forward(&x);
    let norm = 5.0_f32;
    let s = (2.0_f32).sqrt();
    assert!((y.data[0] - (3.0 / norm * s * 2.0 + 1.0)).abs() < 1e-6);
    assert!((y.data[1] - (4.0 / norm * s * 0.5 - 1.0)).abs() < 1e-6);
}

#[test]
fn rms_norm_gives_unit_rms_across_channels() {
    let mut x = Vol::zeros(8, 2, 3, 3);
    fill(&mut x, 7.0);
    let y = RmsNormChannel::new(8, false).forward(&x);
    let plane = y.plane();
    for p in 0..plane {
        let mut sum = 0.0_f32;
        for c in 0..y.c {
            let v = y.data[c * plane + p];
            sum += v * v;
        }
        let rms = (sum / y.c as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-5, "rms={rms} at position {p}");
    }
}

#[test]
fn rms_norm_all_zero_input_does_not_produce_nan() {
    // F.normalize clamps the norm at 1e-12 rather than adding eps under the
    // root, so an all-zero position must come out zero, not NaN.
    let x = Vol::zeros(4, 1, 1, 1);
    let y = RmsNormChannel::new(4, false).forward(&x);
    assert!(y.data.iter().all(|v| v.is_finite() && *v == 0.0), "{:?}", y.data);
}

// -- AvgDown3D / DupUp3D --------------------------------------------------

#[test]
fn avg_down_pools_time() {
    let x = Vol::new(vec![1.0, 2.0, 3.0, 4.0], 1, 4, 1, 1);
    let y = avg_down_3d(&x, 1, 2, 1);
    assert_eq!((y.c, y.t, y.h, y.w), (1, 2, 1, 1));
    assert_eq!(y.data, vec![1.5, 3.5]);
}

#[test]
fn avg_down_pads_time_on_the_past_side() {
    // T=3 with factor_t=2 pads one frame in *front*, so the first output
    // averages {0, x0} and the second averages {x1, x2}. Padding on the future
    // side would give {x0,x1},{x2,0} instead — the difference is causality.
    let x = Vol::new(vec![1.0, 2.0, 3.0], 1, 3, 1, 1);
    let y = avg_down_3d(&x, 1, 2, 1);
    assert_eq!(y.t, 2);
    assert_eq!(y.data, vec![0.5, 2.5]);
}

#[test]
fn avg_down_averages_channel_groups() {
    // factor 1, 2 channels -> 1: a pure channel-group mean.
    let x = Vol::new(vec![1.0, 3.0], 2, 1, 1, 1);
    let y = avg_down_3d(&x, 1, 1, 1);
    assert_eq!(y.data, vec![2.0]);
}

#[test]
fn avg_down_pools_space() {
    let x = Vol::new(vec![1.0, 2.0, 3.0, 4.0], 1, 1, 2, 2);
    let y = avg_down_3d(&x, 1, 1, 2);
    assert_eq!((y.t, y.h, y.w), (1, 1, 1));
    assert_eq!(y.data, vec![2.5]);
}

#[test]
fn dup_up_unfolds_channels_into_time() {
    // 2 channels -> 1 channel at factor_t=2: repeats=1, so the two channels
    // interleave into consecutive frames.
    let x = Vol::new(vec![10.0, 11.0, 20.0, 21.0], 2, 2, 1, 1);
    let y = dup_up_3d(&x, 1, 2, 1, false);
    assert_eq!((y.c, y.t), (1, 4));
    assert_eq!(y.data, vec![10.0, 20.0, 11.0, 21.0]);
}

#[test]
fn dup_up_first_chunk_drops_leading_frames() {
    let x = Vol::new(vec![10.0, 11.0, 20.0, 21.0], 2, 2, 1, 1);
    let y = dup_up_3d(&x, 1, 2, 1, true);
    assert_eq!(y.t, 3, "first chunk drops factor_t-1 leading frames");
    assert_eq!(y.data, vec![20.0, 11.0, 21.0]);
}

#[test]
fn dup_up_duplicates_into_space() {
    let x = Vol::new(vec![5.0], 1, 1, 1, 1);
    let y = dup_up_3d(&x, 1, 1, 2, false);
    assert_eq!((y.t, y.h, y.w), (1, 2, 2));
    assert_eq!(y.data, vec![5.0; 4]);
}

// -- patchify -------------------------------------------------------------

#[test]
fn patchify_channel_order_is_width_major() {
    // The reference permutes (0,1,6,4,2,3,5), putting the width-within-patch
    // index *outside* the height-within-patch index: c*p^2 + iw*p + ih.
    // A 2x2 tile [[0,1],[2,3]] therefore lands as channels [0, 2, 1, 3].
    let x = Vol::new(vec![0.0, 1.0, 2.0, 3.0], 1, 1, 2, 2);
    let y = patchify(&x, 2);
    assert_eq!((y.c, y.t, y.h, y.w), (4, 1, 1, 1));
    assert_eq!(y.data, vec![0.0, 2.0, 1.0, 3.0]);
}

#[test]
fn patchify_unpatchify_round_trips() {
    let mut x = Vol::zeros(3, 2, 8, 6);
    fill(&mut x, 9.0);
    let round = unpatchify(&patchify(&x, 2), 2);
    assert_eq!(round, x);
}

#[test]
fn patchify_is_identity_at_patch_size_one() {
    let mut x = Vol::zeros(2, 2, 2, 2);
    fill(&mut x, 4.0);
    assert_eq!(patchify(&x, 1), x);
    assert_eq!(unpatchify(&x, 1), x);
}

// -- config ---------------------------------------------------------------

#[test]
fn parses_wan22_ti2v_5b_vae_config() {
    // Field-for-field from Wan2.2-TI2V-5B-Diffusers/vae/config.json.
    let json = r#"{
        "_class_name": "AutoencoderKLWan",
        "attn_scales": [],
        "base_dim": 160,
        "clip_output": false,
        "decoder_base_dim": 256,
        "dim_mult": [1, 2, 4, 4],
        "dropout": 0.0,
        "in_channels": 12,
        "is_residual": true,
        "latents_mean": [0.0, 1.0],
        "latents_std": [1.0, 2.0],
        "num_res_blocks": 2,
        "out_channels": 12,
        "patch_size": 2,
        "scale_factor_spatial": 16,
        "scale_factor_temporal": 4,
        "temperal_downsample": [false, true, true],
        "z_dim": 48
    }"#;
    let cfg = WanVaeConfig::from_json(json).unwrap();
    assert_eq!(cfg.z_dim, 48);
    assert_eq!(cfg.base_dim, 160);
    assert_eq!(cfg.decoder_dim(), 256);
    assert!(cfg.is_residual, "Wan 2.2 uses the residual down/up blocks");
    assert!(cfg.attn_scales.is_empty(), "this VAE has no attention anywhere");
    assert_eq!(cfg.patch_size, 2);
    // The two numbers the token budget depends on.
    assert_eq!(cfg.spatial_compression(), 16);
    assert_eq!(cfg.temporal_compression(), 4);
    assert_eq!(cfg.stage_dims(cfg.base_dim), vec![160, 320, 640, 640]);
    assert_eq!(cfg.stage_dims(cfg.decoder_dim()), vec![256, 512, 1024, 1024]);
}

#[test]
fn decoder_dim_defaults_to_base_dim() {
    let json = r#"{
        "base_dim": 96, "z_dim": 16, "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2, "temperal_downsample": [false, true, true],
        "in_channels": 3
    }"#;
    let cfg = WanVaeConfig::from_json(json).unwrap();
    assert_eq!(cfg.decoder_dim(), 96);
    assert_eq!(cfg.patch_size, 1, "Wan 2.1 has no pixel patchify");
    assert!(!cfg.is_residual);
    assert_eq!(cfg.spatial_compression(), 8);
}

// -- encoder / decoder assembly -------------------------------------------

use super::blocks::FeatCache;
use super::model::WanVae;

/// A miniature Wan 2.2-shaped config: same structure and same schedule of
/// spatial/temporal resampling, small enough to run on CPU in a test.
fn tiny_cfg() -> WanVaeConfig {
    WanVaeConfig::from_json(
        r#"{
            "base_dim": 4, "decoder_base_dim": 4, "z_dim": 4,
            "dim_mult": [1, 2, 4, 4], "num_res_blocks": 1,
            "attn_scales": [], "temperal_downsample": [false, true, true],
            "in_channels": 12, "out_channels": 12, "patch_size": 2,
            "is_residual": true
        }"#,
    )
    .unwrap()
}

#[test]
fn decoder_frame_arithmetic_matches_a_five_second_clip() {
    // The property the whole chunked design exists to produce: the first
    // latent frame yields one output frame, every later one yields four.
    let vae = WanVae::new(tiny_cfg());
    assert_eq!(vae.decoded_frame_count(1), 1);
    assert_eq!(vae.decoded_frame_count(2), 5);
    // 5s at 24fps is 121 frames from 31 latent frames — the real target.
    assert_eq!(vae.decoded_frame_count(31), 121);
    assert_eq!(vae.encoded_frame_count(121), 31);
}

#[test]
fn decoder_produces_expected_shape() {
    let cfg = tiny_cfg();
    let vae = WanVae::new(cfg);
    let z = Vol::zeros(4, 3, 2, 2);
    let out = vae.decode(&z);
    // 3 latent frames -> 1 + 4*2 = 9 frames; 2 -> 2*8 (three up blocks) = 16,
    // then unpatchify by 2 -> 32. Channels 12 -> 3 after unpatchify.
    assert_eq!((out.c, out.t, out.h, out.w), (3, 9, 32, 32));
    assert_eq!(out.h / z.h, vae.cfg.spatial_compression());
}

#[test]
fn encoder_produces_expected_shape() {
    let vae = WanVae::new(tiny_cfg());
    // 9 pixel frames -> 1 + ceil(8/4) = 3 latent frames.
    let pixels = Vol::zeros(3, 9, 32, 32);
    let enc = vae.encode(&pixels);
    assert_eq!(enc.c, vae.cfg.z_dim * 2, "encoder emits mean and log-variance");
    assert_eq!(enc.t, 3);
    assert_eq!((enc.h, enc.w), (2, 2));
}

#[test]
fn decoder_cache_walk_length_matches_declared_slot_count() {
    // Every chunk must walk exactly the same number of cache slots, and it
    // must equal `convs()`. A mismatch would silently mis-address the cache
    // and corrupt temporal context — the kind of bug that shows up as
    // flickering frames rather than a crash.
    let vae = WanVae::new(tiny_cfg());
    let declared = vae.decoder.convs();
    let mut cache = FeatCache::new(declared);
    let z = Vol::zeros(4, 1, 2, 2);

    cache.rewind();
    vae.decoder.forward(&z, &mut cache, true);
    assert_eq!(cache.idx, declared, "first chunk walked a different slot count");

    cache.rewind();
    vae.decoder.forward(&z, &mut cache, false);
    assert_eq!(cache.idx, declared, "later chunk walked a different slot count");
}

#[test]
fn encoder_cache_walk_length_matches_declared_slot_count() {
    let vae = WanVae::new(tiny_cfg());
    let declared = vae.encoder.convs();
    let mut cache = FeatCache::new(declared);

    cache.rewind();
    vae.encoder.forward(&Vol::zeros(12, 1, 16, 16), &mut cache);
    assert_eq!(cache.idx, declared);

    cache.rewind();
    vae.encoder.forward(&Vol::zeros(12, 4, 16, 16), &mut cache);
    assert_eq!(cache.idx, declared);
}

#[test]
fn decode_is_deterministic_and_finite() {
    let vae = WanVae::new(tiny_cfg());
    let mut z = Vol::zeros(4, 3, 2, 2);
    fill(&mut z, 11.0);
    let a = vae.decode(&z);
    let b = vae.decode(&z);
    assert_eq!(a, b, "decode must not depend on leftover cache state");
    assert!(a.data.iter().all(|v| v.is_finite()));
    assert!(
        a.data.iter().all(|v| (-1.0..=1.0).contains(v)),
        "decode clamps its output to [-1, 1]"
    );
}

#[test]
fn attention_block_is_a_softmax_average_of_values() {
    // With zero qkv weights every score is equal, so the softmax is uniform
    // and attention degenerates to the mean of V — which here is the qkv bias,
    // constant at every position. With `proj` set to the identity the block
    // must return exactly x + that constant, confirming the softmax
    // normalises rather than leaking a factor of H*W.
    use super::blocks::AttentionBlock;
    let dim = 2;
    let mut a = AttentionBlock::new(dim);
    if let Some(b) = a.to_qkv.bias.as_mut() {
        b[2 * dim] = 3.0;      // value channel 0
        b[2 * dim + 1] = -1.0; // value channel 1
    }
    // proj weight is [out, in, 1, 1, 1]; the diagonal is the identity.
    for c in 0..dim {
        a.proj.weight[c * dim + c] = 1.0;
    }
    let mut x = Vol::zeros(2, 1, 2, 2);
    fill(&mut x, 2.0);
    let y = a.forward(&x);
    for h in 0..2 {
        for w in 0..2 {
            assert!((y.at(0, 0, h, w) - (x.at(0, 0, h, w) + 3.0)).abs() < 1e-5);
            assert!((y.at(1, 0, h, w) - (x.at(1, 0, h, w) - 1.0)).abs() < 1e-5);
        }
    }
}

// -- weight loading -------------------------------------------------------

/// The real Wan2.2-TI2V-5B config, verbatim from `vae/config.json`.
fn wan22_cfg() -> WanVaeConfig {
    WanVaeConfig::from_json(
        r#"{
            "base_dim": 160, "decoder_base_dim": 256, "z_dim": 48,
            "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2,
            "attn_scales": [], "temperal_downsample": [false, true, true],
            "in_channels": 12, "out_channels": 12, "patch_size": 2,
            "is_residual": true, "clip_output": false
        }"#,
    )
    .unwrap()
}

/// The strongest structural check available without downloading weights: the
/// model must ask for exactly the tensors the real checkpoint contains.
///
/// `Wan2.2-TI2V-5B-Diffusers/vae/diffusion_pytorch_model.safetensors` holds
/// **196** tensors (read from its safetensors header). Any miscount — a
/// missing shortcut, an extra time_conv, the wrong number of resnets per
/// stage — moves this number.
#[test]
fn tensor_names_match_the_real_checkpoint_inventory() {
    let vae = WanVae::new(wan22_cfg());
    let names = vae.tensor_names();
    assert_eq!(names.len(), 196, "checkpoint has 196 tensors");

    let unique: std::collections::HashSet<_> = names.iter().collect();
    assert_eq!(unique.len(), names.len(), "duplicate tensor name requested");

    let has = |n: &str| names.iter().any(|x| x == n);

    // Spot-check the names and structure read out of the real header.
    assert!(has("decoder.conv_in.weight"));
    assert!(has("decoder.norm_out.gamma"), "RMS norm scale is `gamma`, not `weight`");
    assert!(has("quant_conv.weight") && has("post_quant_conv.weight"));

    // Spatial convs sit at index 1 of the nn.Sequential, behind the
    // upsample / zero-pad at index 0.
    assert!(has("decoder.up_blocks.0.upsampler.resample.1.weight"));
    assert!(!has("decoder.up_blocks.0.upsampler.resample.0.weight"));

    // temperal_downsample [false, true, true] reversed -> up blocks 0 and 1
    // are the temporal ones, and block 3 does not upsample at all.
    assert!(has("decoder.up_blocks.0.upsampler.time_conv.weight"));
    assert!(has("decoder.up_blocks.1.upsampler.time_conv.weight"));
    assert!(!has("decoder.up_blocks.2.upsampler.time_conv.weight"));
    assert!(!has("decoder.up_blocks.3.upsampler.resample.1.weight"));

    // Encoder mirrors it: down blocks 1 and 2 are temporal, block 3 has no
    // downsampler.
    assert!(!has("encoder.down_blocks.0.downsampler.time_conv.weight"));
    assert!(has("encoder.down_blocks.1.downsampler.time_conv.weight"));
    assert!(has("encoder.down_blocks.2.downsampler.time_conv.weight"));
    assert!(!has("encoder.down_blocks.3.downsampler.resample.1.weight"));

    // Shortcut convs appear exactly where the channel count changes:
    // decoder 1024->512->256, encoder 160->320->640.
    assert!(has("decoder.up_blocks.2.resnets.0.conv_shortcut.weight"));
    assert!(has("decoder.up_blocks.3.resnets.0.conv_shortcut.weight"));
    assert!(!has("decoder.up_blocks.0.resnets.0.conv_shortcut.weight"));
    assert!(has("encoder.down_blocks.1.resnets.0.conv_shortcut.weight"));
    assert!(has("encoder.down_blocks.2.resnets.0.conv_shortcut.weight"));
    assert!(!has("encoder.down_blocks.0.resnets.0.conv_shortcut.weight"));

    // AvgDown3D / DupUp3D are parameter-free.
    assert!(!names.iter().any(|n| n.contains("avg_shortcut")));

    // Decoder stages carry num_res_blocks + 1; encoder stages carry
    // num_res_blocks.
    assert!(has("decoder.up_blocks.0.resnets.2.conv1.weight"));
    assert!(!has("decoder.up_blocks.0.resnets.3.conv1.weight"));
    assert!(has("encoder.down_blocks.0.resnets.1.conv1.weight"));
    assert!(!has("encoder.down_blocks.0.resnets.2.conv1.weight"));
}

/// Every conv the model declares must have the element count the checkpoint
/// stores. Shapes here are read from the real safetensors header; note the
/// per-frame convs are stored 4D `[out, in, kh, kw]` while we hold them as
/// `[out, in, 1, kh, kw]` — same elements, different rank.
#[test]
fn declared_conv_shapes_match_the_checkpoint() {
    let vae = WanVae::new(wan22_cfg());
    let d = &vae.decoder;
    let e = &vae.encoder;

    assert_eq!(d.conv_in.weight.len(), 1024 * 48 * 27, "decoder.conv_in [1024,48,3,3,3]");
    assert_eq!(d.conv_out.weight.len(), 12 * 256 * 27, "decoder.conv_out [12,256,3,3,3]");
    assert_eq!(e.conv_in.weight.len(), 160 * 12 * 27, "encoder.conv_in [160,12,3,3,3]");
    assert_eq!(e.conv_out.weight.len(), 96 * 640 * 27, "encoder.conv_out [96,640,3,3,3]");

    // Attention widths: 1024 in the decoder mid block, 640 in the encoder's.
    assert_eq!(d.mid_block.attentions[0].to_qkv.weight.len(), 3072 * 1024);
    assert_eq!(e.mid_block.attentions[0].to_qkv.weight.len(), 1920 * 640);

    // time_conv doubles channels on the way up, keeps them on the way down.
    let up0 = d.up_blocks[0].upsampler.as_ref().unwrap();
    assert_eq!(up0.time_conv.as_ref().unwrap().weight.len(), 2048 * 1024 * 3);
    let down1 = e.down_blocks[1].downsampler.as_ref().unwrap();
    assert_eq!(down1.time_conv.as_ref().unwrap().weight.len(), 320 * 320 * 3);

    // Spatial resample convs are 2D: [out, in, 3, 3].
    assert_eq!(up0.conv.weight.len(), 1024 * 1024 * 9);
    assert_eq!(down1.conv.weight.len(), 320 * 320 * 9);

    // Shortcuts are 1x1x1.
    let sc = d.up_blocks[2].resnets[0].shortcut.as_ref().unwrap();
    assert_eq!(sc.weight.len(), 512 * 1024);
}

#[test]
fn missing_tensors_are_reported_rather_than_panicking() {
    use crate::weight::loader::ComponentTensors;
    let empty = ComponentTensors {
        mmaps: Vec::new(),
        owned: Vec::new(),
        tensors: std::collections::HashMap::new(),
    };
    let vae = WanVae::new(tiny_cfg());
    assert_eq!(vae.missing_tensors(&empty).len(), vae.tensor_names().len());
    // `WanVae` deliberately has no Debug impl — deriving one would format
    // every weight — so match rather than unwrap_err.
    match WanVae::load(tiny_cfg(), &empty) {
        Ok(_) => panic!("loading from an empty checkpoint should fail"),
        Err(e) => assert!(format!("{e}").contains("missing"), "{e}"),
    }
}

/// Same cross-wiring check as the DiT: fill every tensor with a constant
/// derived from its own name, load, then verify each field holds *its* value.
/// A size check cannot catch a swap between two same-shaped tensors — norm1
/// with norm2, conv1 with conv2, encoder with decoder — but this does.
#[test]
fn vae_loads_every_tensor_into_the_right_place() {
    use crate::weight::loader::ComponentTensors;

    fn tag(name: &str) -> f32 {
        let mut h: u32 = 2_166_136_261;
        for b in name.bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(16_777_619);
        }
        (h % 100_000) as f32 / 100_000.0 + 0.1
    }

    let cfg = tiny_cfg();
    let blank = WanVae::new(cfg.clone());
    let mut comp = ComponentTensors {
        mmaps: Vec::new(),
        owned: Vec::new(),
        tensors: std::collections::HashMap::new(),
    };
    for (name, n) in blank.tensor_specs() {
        let bytes: Vec<u8> = std::iter::repeat_n(tag(&name).to_le_bytes(), n).flatten().collect();
        comp.append_synth_tensor(&name, safetensors::Dtype::F32, vec![n], &bytes).unwrap();
    }

    let vae = WanVae::load(cfg, &comp).unwrap();
    let all = |v: &[f32], name: &str| {
        assert!(!v.is_empty(), "{name} loaded empty");
        assert!(
            v.iter().all(|x| (*x - tag(name)).abs() < 1e-6),
            "{name} did not receive its own values (cross-wired?)"
        );
    };

    all(&vae.decoder.conv_in.weight, "decoder.conv_in.weight");
    all(&vae.encoder.conv_in.weight, "encoder.conv_in.weight");
    all(&vae.quant_conv.weight, "quant_conv.weight");
    all(&vae.post_quant_conv.weight, "post_quant_conv.weight");
    all(&vae.decoder.norm_out.gamma, "decoder.norm_out.gamma");

    // norm1/norm2 and conv1/conv2 within a resnet are the same shape, so only
    // a value check distinguishes them.
    let r = &vae.decoder.mid_block.resnets[0];
    all(&r.norm1.gamma, "decoder.mid_block.resnets.0.norm1.gamma");
    all(&r.norm2.gamma, "decoder.mid_block.resnets.0.norm2.gamma");
    all(&r.conv1.weight, "decoder.mid_block.resnets.0.conv1.weight");
    all(&r.conv2.weight, "decoder.mid_block.resnets.0.conv2.weight");

    let a = &vae.decoder.mid_block.attentions[0];
    all(&a.to_qkv.weight, "decoder.mid_block.attentions.0.to_qkv.weight");
    all(&a.proj.weight, "decoder.mid_block.attentions.0.proj.weight");

    // The spatial conv sits at resample.1, and the temporal conv is separate.
    let up0 = vae.decoder.up_blocks[0].upsampler.as_ref().unwrap();
    all(&up0.conv.weight, "decoder.up_blocks.0.upsampler.resample.1.weight");
    all(&up0.time_conv.as_ref().unwrap().weight, "decoder.up_blocks.0.upsampler.time_conv.weight");

    // And it decodes.
    let out = vae.decode(&Vol::zeros(cfg_z(&vae), 2, 2, 2));
    assert!(out.data.iter().all(|v| v.is_finite()));
}

fn cfg_z(v: &WanVae) -> usize {
    v.cfg.z_dim
}

#[test]
fn vae_reports_mis_sized_tensors_before_loading_any() {
    use crate::weight::loader::ComponentTensors;
    let cfg = tiny_cfg();
    let blank = WanVae::new(cfg.clone());
    let mut comp = ComponentTensors {
        mmaps: Vec::new(),
        owned: Vec::new(),
        tensors: std::collections::HashMap::new(),
    };
    for (name, n) in blank.tensor_specs() {
        comp.append_synth_tensor(&name, safetensors::Dtype::F32, vec![n], &vec![0u8; n * 4])
            .unwrap();
    }
    let name = "decoder.conv_in.weight";
    comp.tensors.remove(name);
    comp.append_synth_tensor(name, safetensors::Dtype::F32, vec![5], &[0u8; 20]).unwrap();
    assert_eq!(blank.mismatched_tensors(&comp), vec![name.to_string()]);
    match WanVae::load(cfg, &comp) {
        Ok(_) => panic!("mis-sized tensor should not load"),
        Err(e) => assert!(format!("{e}").contains("unexpected size"), "{e}"),
    }
}

/// Compare the causal VAE decoder against a diffusers reference.
///
/// This is the piece with the most opportunity for silent error: causal
/// padding on the past side only, the per-conv feature cache, the first-chunk
/// skips in `upsample3d` and `DupUp3D`, the width-major patchify. All of it
/// produces correctly-shaped output when wrong.
///
///   /tmp/klearu_ref_venv/bin/python scripts/dump_wan_reference.py \
///       --out /tmp/dit.safetensors --out-vae /tmp/vae/vae_reference.safetensors
///   KLEARU_VAE_REF=/tmp/vae/vae_reference.safetensors \
///   cargo test -p klearu-diffusion --lib vae_reference -- --ignored --nocapture
#[test]
#[ignore]
fn vae_reference_matches_diffusers() {
    use crate::weight::loader::ComponentTensors;

    let path = std::env::var("KLEARU_VAE_REF")
        .unwrap_or_else(|_| "vae_reference.safetensors".to_string());
    assert!(
        std::path::Path::new(&path).exists(),
        "reference not found at {path}; run scripts/dump_wan_reference.py --out-vae first"
    );
    let dir = std::path::Path::new(&path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let comp = ComponentTensors::open_dir(&dir).expect("open reference");

    let cfg = tiny_cfg();
    let vae = WanVae::load(cfg.clone(), &comp).expect("load reference weights");

    let z = comp.get_f32("in.latent").unwrap();
    let (t, h, w) = (3usize, 2usize, 2usize);
    assert_eq!(z.len(), cfg.z_dim * t * h * w);

    let got = vae.decode(&Vol::new(z, cfg.z_dim, t, h, w));
    let want = comp.get_f32("out.video").unwrap();

    // 3 latent frames -> 1 + 4*2 = 9 frames; 2 latent px -> 2*16 = 32.
    assert_eq!((got.c, got.t, got.h, got.w), (3, 9, 32, 32));
    assert_eq!(got.data.len(), want.len(), "output length mismatch");

    let worst = got.data.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let dot: f64 = got.data.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = got.data.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    println!(
        "vae decode              max_abs_diff={worst:.3e}  cos_sim={:.9}",
        dot / (na * nb + 1e-12)
    );

    // Per-frame breakdown: a cache or first-chunk error shows up as one frame
    // diverging while the rest match, which a single max would obscure.
    let plane = got.h * got.w;
    for f in 0..got.t {
        let mut fw = 0.0_f32;
        for c in 0..got.c {
            for i in 0..plane {
                let idx = (c * got.t + f) * plane + i;
                fw = fw.max((got.data[idx] - want[idx]).abs());
            }
        }
        println!("  frame {f}: max_abs_diff={fw:.3e}");
    }

    assert!(worst < 2e-4, "VAE decode diverged from diffusers: {worst}");
}

// -- GPU path -------------------------------------------------------------

/// The channel norm reduces across the *outer* axis, unlike the row-wise
/// `rms_norm_f16_gpu`. It also has to reproduce `F.normalize`'s clamped norm
/// so an all-zero position stays zero instead of going NaN.
#[cfg(feature = "metal")]
#[test]
fn gpu_channel_norm_matches_cpu() {
    use super::gpu::{download_vol, upload_vol, GpuRmsNormChannel};

    let (c, t, h, w) = (8usize, 2usize, 3usize, 3usize);
    let mut x = Vol::zeros(c, t, h, w);
    fill(&mut x, 13.0);
    let mut n = RmsNormChannel::new(c, false);
    fill_slice(&mut n.gamma, 14.0);

    let want = n.forward(&x);
    let gn = GpuRmsNormChannel::from_cpu(&n);
    let mut g = upload_vol(&x);
    gn.forward(&mut g, t, h, w);
    let got = download_vol(&g, c, t, h, w);

    let worst = got.data.iter().zip(&want.data).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(worst < 5e-2, "channel norm diverged: {worst}");

    // All-zero column must survive rather than NaN.
    let zero = Vol::zeros(c, 1, 1, 1);
    let mut gz = upload_vol(&zero);
    gn.forward(&mut gz, 1, 1, 1);
    let out = download_vol(&gz, c, 1, 1, 1);
    assert!(out.data.iter().all(|v| v.is_finite()), "{:?}", out.data);
}

fn fill_slice(v: &mut [f32], seed: f32) {
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i as f32 * 0.7 + seed).sin()) * 0.9;
    }
}

/// The causal conv3d decomposition — kt batched 2D convs, each folded in at
/// its own frame offset — must reproduce the CPU convolution exactly,
/// including the past-side-only padding.
#[cfg(feature = "metal")]
#[test]
fn gpu_causal_conv3d_matches_cpu() {
    use super::gpu::{download_vol, upload_vol, GpuCausalConv3d};

    for (in_c, out_c, t, h, w) in [(2usize, 3usize, 4usize, 5usize, 5usize), (4, 2, 1, 3, 3)] {
        let mut conv = CausalConv3d::new(in_c, out_c, (3, 3, 3), (1, 1, 1), (1, 1, 1), true);
        fill_slice(&mut conv.weight, 15.0);
        if let Some(b) = conv.bias.as_mut() {
            fill_slice(b, 16.0);
        }
        let mut x = Vol::zeros(in_c, t, h, w);
        fill(&mut x, 17.0);

        let want = conv.forward(&x, None);
        let g = GpuCausalConv3d::from_cpu(&conv);
        let (out, t_out, h_out, w_out) = g.forward(&upload_vol(&x), t, h, w);
        assert_eq!((t_out, h_out, w_out), (want.t, want.h, want.w));
        let got = download_vol(&out, out_c, t_out, h_out, w_out);

        let worst =
            got.data.iter().zip(&want.data).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        let dot: f64 = got.data.iter().zip(&want.data).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = got.data.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = want.data.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb + 1e-12);
        println!("conv3d [{in_c}->{out_c}, t={t}] max_abs_diff={worst:.3e} cos_sim={cos:.6}");
        assert!(cos > 0.999, "conv3d cos_sim {cos} too low for {in_c}->{out_c}");
    }
}

/// Causality must survive the decomposition: a later input frame cannot reach
/// an earlier output frame. A sign error in the tap offset would break this
/// while leaving every shape intact.
#[cfg(feature = "metal")]
#[test]
fn gpu_causal_conv3d_stays_causal() {
    use super::gpu::{download_vol, upload_vol, GpuCausalConv3d};

    let (in_c, out_c, t, h, w) = (2usize, 2usize, 5usize, 3usize, 3usize);
    let mut conv = CausalConv3d::new(in_c, out_c, (3, 3, 3), (1, 1, 1), (1, 1, 1), false);
    fill_slice(&mut conv.weight, 18.0);
    let g = GpuCausalConv3d::from_cpu(&conv);

    let mut x = Vol::zeros(in_c, t, h, w);
    fill(&mut x, 19.0);
    let (base, ..) = g.forward(&upload_vol(&x), t, h, w);
    let base = download_vol(&base, out_c, t, h, w);

    for k in 1..t {
        let mut perturbed = x.clone();
        for c in 0..in_c {
            for hh in 0..h {
                for ww in 0..w {
                    let i = perturbed.idx(c, k, hh, ww);
                    perturbed.data[i] += 10.0;
                }
            }
        }
        let (got, ..) = g.forward(&upload_vol(&perturbed), t, h, w);
        let got = download_vol(&got, out_c, t, h, w);
        for frame in 0..k {
            for c in 0..out_c {
                for hh in 0..h {
                    for ww in 0..w {
                        let d = (got.at(c, frame, hh, ww) - base.at(c, frame, hh, ww)).abs();
                        assert!(d < 1e-2, "frame {k} leaked into output frame {frame}: {d}");
                    }
                }
            }
        }
    }
}

/// Not a correctness check — confirms the GPU conv3d is actually worth having.
/// Run with:
///   cargo test -p klearu-diffusion --features metal --lib --release \
///     -- --ignored --nocapture conv3d_speed
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn gpu_conv3d_speed() {
    use super::gpu::{upload_vol, GpuCausalConv3d};
    use std::time::Instant;

    println!("\n{:>18} {:>12} {:>12} {:>9}", "shape", "cpu (ms)", "gpu (ms)", "speedup");
    for (c, t, h, w) in [(32usize, 4usize, 32usize, 32usize), (64, 4, 64, 64)] {
        let mut conv = CausalConv3d::new(c, c, (3, 3, 3), (1, 1, 1), (1, 1, 1), true);
        fill_slice(&mut conv.weight, 21.0);
        if let Some(b) = conv.bias.as_mut() {
            fill_slice(b, 22.0);
        }
        let mut x = Vol::zeros(c, t, h, w);
        fill(&mut x, 23.0);

        let g = GpuCausalConv3d::from_cpu(&conv);
        let gx = upload_vol(&x);
        // Warm the MPSGraph cache (keyed on shape) out of the timed region.
        let _ = g.forward(&gx, t, h, w);
        crate::metal_backend::flush();

        let t0 = Instant::now();
        let _ = conv.forward(&x, None);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let _ = g.forward(&gx, t, h, w);
        crate::metal_backend::flush();
        let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;

        println!(
            "{:>18} {cpu_ms:>12.1} {gpu_ms:>12.1} {:>8.1}x",
            format!("{c}c {t}x{h}x{w}"),
            cpu_ms / gpu_ms
        );
    }
}

/// The GPU conv must reproduce a whole-clip run when fed chunk-by-chunk with
/// the trailing-frame cache — the property the entire decoder rests on, now on
/// the GPU side too.
#[cfg(feature = "metal")]
#[test]
fn gpu_causal_conv3d_chunked_matches_whole_clip() {
    use super::gpu::{download_vol, upload_vol, GpuCausalConv3d};

    let (in_c, out_c, t, h, w) = (3usize, 2usize, 7usize, 4usize, 4usize);
    let mut conv = CausalConv3d::new(in_c, out_c, (3, 3, 3), (1, 1, 1), (1, 1, 1), true);
    fill_slice(&mut conv.weight, 24.0);
    if let Some(b) = conv.bias.as_mut() {
        fill_slice(b, 25.0);
    }
    let mut x = Vol::zeros(in_c, t, h, w);
    fill(&mut x, 26.0);

    let g = GpuCausalConv3d::from_cpu(&conv);
    let (whole, ..) = g.forward(&upload_vol(&x), t, h, w);
    let whole = download_vol(&whole, out_c, t, h, w);

    // Split [0..3) then [3..7), passing the first chunk's trailing frames.
    let split = 3usize;
    let mut x1 = Vol::zeros(in_c, split, h, w);
    let mut x2 = Vol::zeros(in_c, t - split, h, w);
    for c in 0..in_c {
        for ti in 0..t {
            for hh in 0..h {
                for ww in 0..w {
                    let v = x.at(c, ti, hh, ww);
                    if ti < split {
                        let i = x1.idx(c, ti, hh, ww);
                        x1.data[i] = v;
                    } else {
                        let i = x2.idx(c, ti - split, hh, ww);
                        x2.data[i] = v;
                    }
                }
            }
        }
    }

    let (y1, ..) = g.forward(&upload_vol(&x1), split, h, w);
    let y1 = download_vol(&y1, out_c, split, h, w);
    let cache = x1.tail_frames(CACHE_T);
    let (y2, t2, ..) = g.forward_cached(
        &upload_vol(&x2), t - split, h, w, Some(&upload_vol(&cache)));
    assert_eq!(t2, t - split, "cached chunk should keep its frame count");
    let y2 = download_vol(&y2, out_c, t - split, h, w);

    let joined = Vol::concat_time(&y1, &y2);
    let worst =
        joined.data.iter().zip(&whole.data).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(worst < 5e-2, "GPU chunked decode diverged from whole-clip: {worst}");
}

/// The GPU residual block against the CPU one, cacheless and cached.
#[cfg(feature = "metal")]
#[test]
fn gpu_residual_block_matches_cpu() {
    use super::blocks::ResidualBlock;
    use super::gpu::{download_vol, upload_vol, GpuResidualBlock};

    for (in_c, out_c) in [(4usize, 4usize), (4, 6)] {
        let mut b = ResidualBlock::new(in_c, out_c);
        fill_slice(&mut b.norm1.gamma, 27.0);
        fill_slice(&mut b.norm2.gamma, 28.0);
        fill_slice(&mut b.conv1.weight, 29.0);
        fill_slice(&mut b.conv2.weight, 30.0);
        if let Some(bi) = b.conv1.bias.as_mut() {
            fill_slice(bi, 31.0);
        }
        if let Some(bi) = b.conv2.bias.as_mut() {
            fill_slice(bi, 32.0);
        }
        if let Some(sc) = b.shortcut.as_mut() {
            fill_slice(&mut sc.weight, 33.0);
            if let Some(bi) = sc.bias.as_mut() {
                fill_slice(bi, 34.0);
            }
        }

        let (t, h, w) = (3usize, 4usize, 4usize);
        let mut x = Vol::zeros(in_c, t, h, w);
        fill(&mut x, 35.0);

        let want = b.forward(&x, None);
        let g = GpuResidualBlock::from_cpu(&b);
        let got = g.forward(&upload_vol(&x), t, h, w, None);
        let got = download_vol(&got, out_c, t, h, w);

        assert_eq!((got.c, got.t, got.h, got.w), (want.c, want.t, want.h, want.w));
        let dot: f64 = got.data.iter().zip(&want.data).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = got.data.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = want.data.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb + 1e-12);
        let worst =
            got.data.iter().zip(&want.data).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        println!("resblock {in_c}->{out_c}: max_abs_diff={worst:.3e} cos_sim={cos:.6}");
        assert!(cos > 0.999, "resblock {in_c}->{out_c} cos_sim {cos} too low");
    }
}

/// `DupUp3D` on the GPU, including the first-chunk frame drop that makes the
/// first latent frame expand to one output frame instead of `factor_t`.
#[cfg(feature = "metal")]
#[test]
fn gpu_dup_up_matches_cpu() {
    use super::gpu::{download_vol, dup_up_gpu, upload_vol};

    for (in_c, out_c, ft, fs, first) in [
        (2usize, 1usize, 2usize, 1usize, false),
        (2, 1, 2, 1, true),
        (4, 2, 2, 2, false),
        (4, 2, 2, 2, true),
        (2, 2, 1, 2, false),
    ] {
        let (t, h, w) = (3usize, 2usize, 2usize);
        let mut x = Vol::zeros(in_c, t, h, w);
        fill(&mut x, 41.0);

        let want = dup_up_3d(&x, out_c, ft, fs, first);
        let got = dup_up_gpu(&upload_vol(&x), in_c, out_c, t, h, w, ft, fs, first);
        let got = download_vol(&got, want.c, want.t, want.h, want.w);
        assert_eq!((got.c, got.t, got.h, got.w), (want.c, want.t, want.h, want.w));
        let worst =
            got.data.iter().zip(&want.data).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(
            worst < 5e-2,
            "dup_up {in_c}->{out_c} ft={ft} fs={fs} first={first}: {worst}"
        );
    }
}

/// GPU upsampling resample against CPU, both modes, and — for `upsample3d` —
/// both the first chunk (temporal conv skipped) and a later one.
#[cfg(feature = "metal")]
#[test]
fn gpu_resample_matches_cpu() {
    use super::blocks::{Resample, ResampleMode};
    use super::gpu::{download_vol, upload_vol, GpuResample};

    for mode in [ResampleMode::Upsample2d, ResampleMode::Upsample3d] {
        let dim = 4usize;
        let mut r = Resample::new(dim, mode, Some(dim));
        fill_slice(&mut r.conv.weight, 42.0);
        if let Some(b) = r.conv.bias.as_mut() {
            fill_slice(b, 43.0);
        }
        if let Some(tc) = r.time_conv.as_mut() {
            fill_slice(&mut tc.weight, 44.0);
            if let Some(b) = tc.bias.as_mut() {
                fill_slice(b, 45.0);
            }
        }

        let (t, h, w) = (2usize, 3usize, 3usize);
        let mut x = Vol::zeros(dim, t, h, w);
        fill(&mut x, 46.0);
        let g = GpuResample::from_cpu(&r);

        // First chunk: upsample3d skips its temporal conv.
        // NOTE: the cache walks slots sequentially, so a driver must rewind
        // before each chunk — without it the second call reads a fresh slot
        // and silently takes the first-chunk path again.
        let mut cache = FeatCache::new(4);
        cache.rewind();
        let want = r.forward(&x, Some(&mut cache));
        let mut gcache = super::gpu::GpuFeatCache::new(4);
        gcache.rewind();
        let (got, ot, oh, ow) = g.forward(&upload_vol(&x), t, h, w, &mut gcache);
        let got = download_vol(&got, want.c, ot, oh, ow);
        assert_eq!(
            (got.c, got.t, got.h, got.w),
            (want.c, want.t, want.h, want.w),
            "{mode:?} first chunk shape"
        );
        let worst =
            got.data.iter().zip(&want.data).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(worst < 5e-2, "{mode:?} first chunk: {worst}");

        // Second chunk: now the temporal conv runs and doubles the frames.
        let mut x2 = Vol::zeros(dim, t, h, w);
        fill(&mut x2, 47.0);
        cache.rewind();
        let want2 = r.forward(&x2, Some(&mut cache));
        gcache.rewind();
        let (got2, ot2, oh2, ow2) = g.forward(&upload_vol(&x2), t, h, w, &mut gcache);
        let got2 = download_vol(&got2, want2.c, ot2, oh2, ow2);
        assert_eq!(
            (got2.c, got2.t, got2.h, got2.w),
            (want2.c, want2.t, want2.h, want2.w),
            "{mode:?} second chunk shape"
        );
        let dot: f64 = got2.data.iter().zip(&want2.data).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = got2.data.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = want2.data.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb + 1e-12);
        println!("resample {mode:?} chunk2: cos_sim={cos:.6}");
        assert!(cos > 0.999, "{mode:?} second chunk cos_sim {cos}");
    }
}

/// The VAE's single-head spatial attention on the GPU. Uses explicit matmuls
/// rather than the flash kernel, because head_dim here is the full channel
/// count and falls outside its supported dims.
#[cfg(feature = "metal")]
#[test]
fn gpu_attention_block_matches_cpu() {
    use super::blocks::AttentionBlock;
    use super::gpu::{download_vol, upload_vol, GpuAttentionBlock};

    let dim = 6usize;
    let mut a = AttentionBlock::new(dim);
    fill_slice(&mut a.norm.gamma, 51.0);
    fill_slice(&mut a.to_qkv.weight, 52.0);
    fill_slice(&mut a.proj.weight, 53.0);
    if let Some(b) = a.to_qkv.bias.as_mut() {
        fill_slice(b, 54.0);
    }
    if let Some(b) = a.proj.bias.as_mut() {
        fill_slice(b, 55.0);
    }

    let (t, h, w) = (2usize, 3usize, 4usize);
    let mut x = Vol::zeros(dim, t, h, w);
    fill(&mut x, 56.0);

    let want = a.forward(&x);
    let g = GpuAttentionBlock::from_cpu(&a);
    let got = download_vol(&g.forward(&upload_vol(&x), t, h, w), dim, t, h, w);

    assert_eq!((got.c, got.t, got.h, got.w), (want.c, want.t, want.h, want.w));
    let dot: f64 = got.data.iter().zip(&want.data).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = got.data.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.data.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb + 1e-12);
    let worst = got.data.iter().zip(&want.data).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    println!("attention block: max_abs_diff={worst:.3e} cos_sim={cos:.6}");
    assert!(cos > 0.999, "attention cos_sim {cos} too low");

    // Attention must actually mix across spatial positions: perturbing one
    // position has to move the others, or the softmax/matmul wiring is wrong.
    //
    // Measured on the CPU (f32) rather than the GPU: at a far position the
    // effect arrives only through the softmax weights, which is second order
    // and lands below bf16's ~3 significant digits — the GPU legitimately
    // rounds it to exactly 0. The GPU is separately pinned to the CPU above.
    let mut perturbed = x.clone();
    for c in 0..dim {
        let i = perturbed.idx(c, 0, 0, 0);
        perturbed.data[i] += 5.0;
    }
    let moved_cpu = a.forward(&perturbed);
    let far = (moved_cpu.at(0, 0, h - 1, w - 1) - want.at(0, 0, h - 1, w - 1)).abs();
    assert!(far > 1e-6, "attention did not mix across positions: {far}");
    println!("  cross-position sensitivity (cpu f32): {far:.3e}");
}

/// The mid block: resnet -> attention -> resnet, all on GPU.
#[cfg(feature = "metal")]
#[test]
fn gpu_mid_block_matches_cpu() {
    use super::blocks::MidBlock;
    use super::gpu::{download_vol, upload_vol, GpuMidBlock};

    let dim = 6usize;
    let mut m = MidBlock::new(dim, 1);
    for (i, r) in m.resnets.iter_mut().enumerate() {
        let s = 60.0 + i as f32;
        fill_slice(&mut r.norm1.gamma, s);
        fill_slice(&mut r.norm2.gamma, s + 0.3);
        fill_slice(&mut r.conv1.weight, s + 0.6);
        fill_slice(&mut r.conv2.weight, s + 0.9);
        if let Some(b) = r.conv1.bias.as_mut() {
            fill_slice(b, s + 1.2);
        }
        if let Some(b) = r.conv2.bias.as_mut() {
            fill_slice(b, s + 1.5);
        }
    }
    for a in m.attentions.iter_mut() {
        fill_slice(&mut a.norm.gamma, 70.0);
        fill_slice(&mut a.to_qkv.weight, 71.0);
        fill_slice(&mut a.proj.weight, 72.0);
    }

    let (t, h, w) = (2usize, 3usize, 3usize);
    let mut x = Vol::zeros(dim, t, h, w);
    fill(&mut x, 73.0);

    let want = m.forward(&x, None);
    let g = GpuMidBlock::from_cpu(&m);
    let got = download_vol(&g.forward(&upload_vol(&x), t, h, w), dim, t, h, w);

    let dot: f64 = got.data.iter().zip(&want.data).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = got.data.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.data.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb + 1e-12);
    println!("mid block: cos_sim={cos:.6}");
    assert!(cos > 0.999, "mid block cos_sim {cos} too low");
}

/// The whole GPU decoder against the CPU one — chunked, one latent frame at a
/// time, exactly as the real decode runs.
///
/// This is the composition test for the GPU path: it exercises the feature
/// cache across chunks, the first-chunk skips in `upsample3d` and `DupUp3D`,
/// the frame arithmetic, and every block type at once. The CPU decoder it is
/// checked against is itself pinned to diffusers by
/// `vae_reference_matches_diffusers`.
#[cfg(feature = "metal")]
#[test]
fn gpu_decoder_matches_cpu() {
    use super::gpu::GpuDecoder3d;
    use super::model::Decoder3d;

    let cfg = tiny_cfg();
    let mut d = Decoder3d::new(&cfg);

    // Give every parameter a distinct value; zeros would hide index errors.
    let mut seed = 80.0_f32;
    let mut next = || {
        seed += 0.37;
        seed
    };
    fill_slice(&mut d.conv_in.weight, next());
    if let Some(b) = d.conv_in.bias.as_mut() {
        fill_slice(b, next());
    }
    fill_slice(&mut d.norm_out.gamma, next());
    fill_slice(&mut d.conv_out.weight, next());
    if let Some(b) = d.conv_out.bias.as_mut() {
        fill_slice(b, next());
    }
    let fill_res = |r: &mut super::blocks::ResidualBlock, n: &mut dyn FnMut() -> f32| {
        fill_slice(&mut r.norm1.gamma, n());
        fill_slice(&mut r.norm2.gamma, n());
        fill_slice(&mut r.conv1.weight, n());
        fill_slice(&mut r.conv2.weight, n());
        if let Some(b) = r.conv1.bias.as_mut() {
            fill_slice(b, n());
        }
        if let Some(b) = r.conv2.bias.as_mut() {
            fill_slice(b, n());
        }
        if let Some(sc) = r.shortcut.as_mut() {
            fill_slice(&mut sc.weight, n());
            if let Some(b) = sc.bias.as_mut() {
                fill_slice(b, n());
            }
        }
    };
    for r in d.mid_block.resnets.iter_mut() {
        fill_res(r, &mut next);
    }
    for a in d.mid_block.attentions.iter_mut() {
        fill_slice(&mut a.norm.gamma, next());
        fill_slice(&mut a.to_qkv.weight, next());
        fill_slice(&mut a.proj.weight, next());
    }
    for b in d.up_blocks.iter_mut() {
        for r in b.resnets.iter_mut() {
            fill_res(r, &mut next);
        }
        if let Some(u) = b.upsampler.as_mut() {
            fill_slice(&mut u.conv.weight, next());
            if let Some(bi) = u.conv.bias.as_mut() {
                fill_slice(bi, next());
            }
            if let Some(tc) = u.time_conv.as_mut() {
                fill_slice(&mut tc.weight, next());
                if let Some(bi) = tc.bias.as_mut() {
                    fill_slice(bi, next());
                }
            }
        }
    }

    let (t, h, w) = (3usize, 2usize, 2usize);
    let mut z = Vol::zeros(cfg.z_dim, t, h, w);
    fill(&mut z, 99.0);

    // CPU: chunked, one latent frame at a time.
    let mut cache = FeatCache::new(d.convs());
    let mut want: Option<Vol> = None;
    for ti in 0..t {
        cache.rewind();
        let mut frame = Vol::zeros(z.c, 1, z.h, z.w);
        for c in 0..z.c {
            for hh in 0..z.h {
                for ww in 0..z.w {
                    let v = z.at(c, ti, hh, ww);
                    let i = frame.idx(c, 0, hh, ww);
                    frame.data[i] = v;
                }
            }
        }
        let dec = d.forward(&frame, &mut cache, ti == 0);
        want = Some(match want {
            None => dec,
            Some(prev) => Vol::concat_time(&prev, &dec),
        });
    }
    let want = want.unwrap();

    let g = GpuDecoder3d::from_cpu(&d);
    let got = g.decode_all(&z);

    // 3 latent frames -> 1 + 4*2 = 9; 2 latent px -> 2*8 = 16 (pre-unpatchify).
    assert_eq!(
        (got.c, got.t, got.h, got.w),
        (want.c, want.t, want.h, want.w),
        "GPU decoder shape disagrees with CPU"
    );
    assert_eq!(want.t, 9, "frame arithmetic: 1 + 4*(T-1)");

    let dot: f64 = got.data.iter().zip(&want.data).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = got.data.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.data.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb + 1e-12);
    println!("gpu decoder: cos_sim={cos:.6}  frames={}", got.t);

    // Per-frame, so a cache or first-chunk error shows as one bad frame.
    let plane = got.h * got.w;
    for f in 0..got.t {
        let mut fd = 0.0_f64;
        let mut fa = 0.0_f64;
        let mut fb = 0.0_f64;
        for c in 0..got.c {
            for i in 0..plane {
                let k = (c * got.t + f) * plane + i;
                fd += got.data[k] as f64 * want.data[k] as f64;
                fa += (got.data[k] as f64).powi(2);
                fb += (want.data[k] as f64).powi(2);
            }
        }
        let fc = fd / (fa.sqrt() * fb.sqrt() + 1e-12);
        println!("  frame {f}: cos_sim={fc:.6}");
        assert!(fc > 0.99, "frame {f} diverged: cos_sim={fc}");
    }
    // 0.9985 is bf16 storage accumulating over ~25 sequential stages
    // (conv_in -> mid -> 4 up blocks -> conv_out), each rounding to ~0.4% with
    // 7 mantissa bits. Individual blocks score 0.99999; the loss is depth, not
    // a defect in any one of them. Enabling the f32 sgemm scratch moved this
    // only 0.998457 -> 0.998514, so the matmuls are not the bottleneck —
    // the intermediate *storage* is. If this ever proves visible on real
    // weights, the fix is f32 storage for the late, high-resolution stages.
    assert!(cos > 0.995, "gpu decoder cos_sim {cos} too low");
}

/// Load the **real** Wan 2.2 VAE and decode with it.
///
/// Everything else in this file uses miniature models; this is the only test
/// that touches the shipped 2.8 GB checkpoint. It validates what small models
/// cannot: that the tensor names and shapes we derived from the safetensors
/// header actually match the file, that the dtype decodes, and that a decode
/// with trained weights produces plausible pixels rather than saturation.
///
///   KLEARU_WAN_VAE=models/wan22-ti2v-5b/vae \
///   cargo test -p klearu-diffusion --features metal --lib --release \

/// The VAE *encoder*, against diffusers.
///
/// The decoder has been compared for a while; the encoder only ever had shape
/// tests. Start/end frame conditioning feeds an encoded image straight into the
/// latent, so a wrong encoder does not fail loudly — the video generates and
/// quietly ignores or corrupts the reference frame. Worth pinning down before

/// How long encoding a conditioning frame actually costs.
///
/// Only the *decoder* is GPU-accelerated (`GpuDecoder3d`); the encoder runs on
/// the CPU, and start/end frame conditioning has to encode one image before
/// denoising can begin. If that is minutes at native resolution it is a real

/// The GPU encoder must agree with the CPU encoder, and be far faster.
///
/// The CPU side is the reference here because it is itself pinned to diffusers
/// by `encoder_matches_diffusers` — so agreement chains all the way back to the

/// Isolate the one genuinely new piece: a stride-2 conv with end-only padding.
///
/// `gpu_encoder_matches_cpu` failing tells you the stack is wrong, not where.
/// This is the smallest thing that could be wrong — everything else in the

/// A wide 3x3x3 conv at small spatial sizes, CPU vs GPU.
///
/// The encoder's last block runs 640->640 at 4x4, and that is exactly where the
/// GPU stack diverged while every earlier block (8x8 and up) was exact. This
/// pins whether the conv itself is size-dependent.
#[cfg(feature = "metal")]
#[test]
fn gpu_conv_matches_cpu_at_small_spatial_sizes() {
    use crate::vae3d::gpu::GpuCausalConv3d;

    for (c, hw) in [(64usize, 4usize), (64, 8), (64, 2), (640, 4)] {
        let mut conv = CausalConv3d::new(c, c, (3, 3, 3), (1, 1, 1), (1, 1, 1), true);
        for (i, v) in conv.weight.iter_mut().enumerate() {
            *v = (((i * 37) % 61) as f32 - 30.0) * 0.004;
        }
        if let Some(b) = conv.bias.as_mut() {
            for (i, v) in b.iter_mut().enumerate() {
                *v = ((i % 7) as f32 - 3.0) * 0.02;
            }
        }
        let x: Vec<f32> =
            (0..c * hw * hw).map(|i| (((i * 13) % 47) as f32 - 23.0) * 0.03).collect();

        let want = conv.forward(&Vol::new(x.clone(), c, 1, hw, hw), None);
        let g = GpuCausalConv3d::from_cpu(&conv);
        let gx = crate::vae3d::gpu::upload_vol(&Vol::new(x, c, 1, hw, hw));
        let (out, ..) = g.forward(&gx, 1, hw, hw);
        let got = out.download_to_f32();

        let dot: f64 = got.iter().zip(&want.data).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = want.data.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb + 1e-12);
        println!("c={c:<4} {hw}x{hw}: cos_sim={cos:.8}");
        assert!(cos > 0.999, "conv diverges at c={c} {hw}x{hw}: cos_sim {cos}");
    }
}

/// encoder is machinery the decoder already exercises.
#[cfg(feature = "metal")]
#[test]
fn gpu_pad_end_conv_matches_cpu() {
    use crate::metal_backend::GpuTensor;
    use crate::vae3d::gpu::GpuCausalConv3d;

    let (in_c, out_c, h, w) = (3usize, 5usize, 8usize, 10usize);
    let mut conv = CausalConv3d::conv2d_pad_end(in_c, out_c, 3, 2);
    // Deterministic non-symmetric weights: a symmetric kernel would hide a
    // transposed or mirrored padding.
    for (i, v) in conv.weight.iter_mut().enumerate() {
        *v = ((i % 17) as f32 - 8.0) * 0.05;
    }
    if let Some(b) = conv.bias.as_mut() {
        for (i, v) in b.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 0.2;
        }
    }

    let x: Vec<f32> = (0..in_c * h * w).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
    let want = conv.forward(&Vol::new(x.clone(), in_c, 1, h, w), None);

    let g = GpuCausalConv3d::from_cpu(&conv);
    // Must match the VAE stack's storage — see `upload_vol`.
    let gx = crate::vae3d::gpu::upload_vol(&Vol::new(x.clone(), in_c, 1, h, w));
    let (out, t, oh, ow) = g.forward(&gx, 1, h, w);
    let got = out.download_to_f32();

    assert_eq!((t, oh, ow), (want.t, want.h, want.w), "output shape");
    assert_eq!(got.len(), want.data.len());
    let worst = got.iter().zip(&want.data).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let scale = want.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max).max(1e-6);
    println!("pad-end conv: max_abs_diff={worst:.4e} (scale {scale:.3})");
    assert!(worst / scale < 0.02, "pad-end conv diverges: {worst:.4e} vs scale {scale:.3}");
}


/// Where does the GPU encoder diverge from the CPU one?
///
/// `gpu_encoder_matches_cpu` says the stack is wrong; this says which stage.
/// Walk both implementations in lockstep and print agreement after each, so

/// The exact conv that diverges, on its own.
///
/// Synthetic weights of the same shape pass, so this uses the real ones and
/// prints their statistics — the remaining difference between the passing
/// synthetic case and the failing real one.
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn real_encoder_conv_matches_cpu() {
    use crate::vae3d::gpu::GpuCausalConv3d;
    use crate::weight::loader::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_VAE") else {
        eprintln!("set KLEARU_WAN_VAE; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    let vae = WanVae::load(cfg, &ComponentTensors::open_dir(&dir).expect("open"))
        .expect("load real weights");

    for (bi, b) in vae.encoder.down_blocks.iter().enumerate() {
        for (ri, r) in b.resnets.iter().enumerate() {
            for (name, conv) in [("conv1", &r.conv1), ("conv2", &r.conv2)] {
                let c = conv.in_c;
                let hw = 4usize;
                let x: Vec<f32> = (0..c * hw * hw)
                    .map(|i| (((i * 13) % 47) as f32 - 23.0) * 0.03)
                    .collect();
                let want = conv.forward(&Vol::new(x.clone(), c, 1, hw, hw), None);
                let g = GpuCausalConv3d::from_cpu(conv);
                let gx = crate::vae3d::gpu::upload_vol(&Vol::new(x, c, 1, hw, hw));
                let (out, ..) = g.forward(&gx, 1, hw, hw);
                let got = out.download_to_f32();
                let dot: f64 =
                    got.iter().zip(&want.data).map(|(a, b)| *a as f64 * *b as f64).sum();
                let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let nb: f64 =
                    want.data.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
                let cos = dot / (na * nb + 1e-12);
                let wmax = conv.weight.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                let wmin = conv.weight.iter().map(|v| v.abs()).fold(f32::MAX, f32::min);
                if cos < 0.999 {
                    println!(
                        "down[{bi}].resnet[{ri}].{name} in={} out={} k={:?} \
                         cos_sim={cos:.6} |w|max={wmax:.4} |w|min={wmin:.3e}",
                        conv.in_c, conv.out_c, conv.kernel
                    );
                }
            }
        }
    }
    println!("(only diverging convs listed)");
}

/// the first row that drops is the culprit rather than a guess.
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn bisect_gpu_encoder_stages() {
    use super::gpu::{upload_vol, GpuCausalConv3d, GpuFeatCache, GpuResidualDownBlock, GpuSlot};
    use crate::weight::loader::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_VAE") else {
        eprintln!("set KLEARU_WAN_VAE; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    let vae = WanVae::load(cfg.clone(), &ComponentTensors::open_dir(&dir).expect("open"))
        .expect("load real weights");
    let e = &vae.encoder;

    // Small: this walks the CPU encoder, which is the slow side.
    let (h, w) = (64usize, 64usize);
    let px: Vec<f32> = (0..3 * h * w)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.06)
        .collect();
    let x0 = patchify(&Vol::new(px, 3, 1, h, w), cfg.patch_size);

    let report = |label: &str, got: &[f32], want: &[f32]| {
        if got.len() != want.len() {
            println!("{label:22} LENGTH {} vs {}", got.len(), want.len());
            return;
        }
        let dot: f64 = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        println!("{label:22} cos_sim={:.8} max_abs={worst:.3e}", dot / (na * nb + 1e-12));
    };

    let mut ccache = FeatCache::new(e.convs());
    let mut gcache = GpuFeatCache::new(e.convs());

    let mut cy = cached_conv(&e.conv_in, &x0, Some(&mut ccache));
    let gx = upload_vol(&x0);
    let gconv_in = GpuCausalConv3d::from_cpu(&e.conv_in);
    let (mut gy, mut gt, mut gh, mut gw) =
        gconv_in.forward_slot(&gx, x0.t, x0.h, x0.w, &mut gcache);
    report("conv_in", &gy.download_to_f32(), &cy.data);

    // Walk each block's resnets and downsampler separately: a whole-block
    // comparison says the block is wrong, not which part of it.
    let mut in_c = e.conv_in.out_c;
    for (i, d) in e.down_blocks.iter().enumerate() {
        let gd = GpuResidualDownBlock::from_cpu(d);
        // For the block that diverges, step its parts so the culprit is named.
        if i == e.down_blocks.len() - 1 {
            println!("   [last block] resnets={} downsampler={:?} factors t={} s={}",
                     d.resnets.len(),
                     d.downsampler.as_ref().map(|r| r.mode),
                     d.factor_t, d.factor_s);
            let mut c2 = FeatCache::new(e.convs());
            let mut g2 = GpuFeatCache::new(e.convs());
            let mut cs = cy.clone();
            let (mut gs, mut st, mut sh, mut sw) = (gy.clone_data(), gt, gh, gw);
            // Step inside the first resnet: norm/conv/norm/conv/add.
            {
                let cr = &d.resnets[0];
                let grr = &gd.resnets[0];
                // `RmsNormChannel::forward` returns a new Vol; it does not
                // mutate. Discarding it compares an unnormalised CPU tensor
                // against a normalised GPU one and blames the GPU.
                let mut ca = cr.norm1.forward(&cs);
                crate::vae3d::silu(&mut ca.data);
                let mut ga = gs.clone_data();
                grr.norm1.forward(&mut ga, st, sh, sw);
                super::gpu::silu_gpu(&mut ga);
                report("    norm1+silu", &ga.download_to_f32(), &ca.data);

                let cb = cached_conv(&cr.conv1, &ca, Some(&mut c2));
                let (gb, bt, bh, bw) = grr.conv1.forward_slot(&ga, st, sh, sw, &mut g2);
                report("    conv1", &gb.download_to_f32(), &cb.data);
                // Decisive: feed the CPU's own input to the GPU conv. If this
                // matches, the conv is fine and the GPU input was not; if it
                // does not, the conv is at fault with these weights.
                {
                    let mut g3 = GpuFeatCache::new(e.convs());
                    let gca = upload_vol(&ca);
                    let (gb2, ..) = grr.conv1.forward_slot(&gca, st, sh, sw, &mut g3);
                    report("    conv1(cpu in)", &gb2.download_to_f32(), &cb.data);
                    let nonfinite = gb.download_to_f32().iter().filter(|v| !v.is_finite()).count();
                    let gmax = gb.download_to_f32().iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                    let cmax = cb.data.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                    println!("    conv1 |gpu|max={gmax:.3} |cpu|max={cmax:.3} nonfinite={nonfinite}");
                }

                let mut cc = cr.norm2.forward(&cb);
                crate::vae3d::silu(&mut cc.data);
                let mut gc = gb.clone_data();
                grr.norm2.forward(&mut gc, bt, bh, bw);
                super::gpu::silu_gpu(&mut gc);
                report("    norm2+silu", &gc.download_to_f32(), &cc.data);

                let cd2 = cached_conv(&cr.conv2, &cc, Some(&mut c2));
                let (gd2, ..) = grr.conv2.forward_slot(&gc, bt, bh, bw, &mut g2);
                report("    conv2", &gd2.download_to_f32(), &cd2.data);
            }
            let mut c2 = FeatCache::new(e.convs());
            let mut g2 = GpuFeatCache::new(e.convs());
            for (r, (cr, grr)) in d.resnets.iter().zip(&gd.resnets).enumerate() {
                cs = cr.forward(&cs, Some(&mut c2));
                let (o2, t2, h2, w2) = grr.forward_slot(&gs, st, sh, sw, &mut g2);
                gs = o2; st = t2; sh = h2; sw = w2;
                report(&format!("  last.resnet[{r}]"), &gs.download_to_f32(), &cs.data);
            }
            let sc = crate::vae3d::avg_down_3d(&cy, d.out_dim, d.factor_t, d.factor_s);
            let (gsc, ..) = crate::metal_backend::avg_down_3d_f16_gpu(
                &gy, cy.c, gt, gh, gw, d.out_dim, d.factor_t, d.factor_s);
            report("  last.shortcut", &gsc.download_to_f32(), &sc.data);
        }
        cy = d.forward(&cy, Some(&mut ccache));
        let (o, ot, oh, ow) = gd.forward(&gy, in_c, gt, gh, gw, &mut gcache);
        gy = o;
        gt = ot;
        gh = oh;
        gw = ow;
        in_c = d.out_dim;
        report(&format!("down_block[{i}]"), &gy.download_to_f32(), &cy.data);
        println!("   shapes cpu [{}, {}, {}, {}] gpu [_, {gt}, {gh}, {gw}]",
                 cy.c, cy.t, cy.h, cy.w);
    }

    // Mid block, then the norm/silu/conv_out tail — the rest of the encoder.
    let mut cm = e.mid_block.forward(&cy, Some(&mut ccache));
    let genc = super::gpu::GpuEncoder3d::from_cpu(e, &vae.quant_conv);
    let mut gm = genc.mid_block.resnets[0].forward_slot(&gy, gt, gh, gw, &mut gcache).0;
    for (attn, resnet) in genc.mid_block.attentions.iter().zip(&genc.mid_block.resnets[1..]) {
        gm = attn.forward(&gm, gt, gh, gw);
        gm = resnet.forward_slot(&gm, gt, gh, gw, &mut gcache).0;
    }
    report("mid_block", &gm.download_to_f32(), &cm.data);

    let mut cm = e.norm_out.forward(&cm);
    crate::vae3d::silu(&mut cm.data);
    let ctail = cached_conv(&e.conv_out, &cm, Some(&mut ccache));
    genc.norm_out.forward(&mut gm, gt, gh, gw);
    super::gpu::silu_gpu(&mut gm);
    let (gtail, ..) = genc.conv_out.forward_slot(&gm, gt, gh, gw, &mut gcache);
    report("conv_out", &gtail.download_to_f32(), &ctail.data);

}


/// Multi-chunk GPU encode must match the CPU, not just the single-frame case.
///
/// A 5-frame clip is two causal chunks, and the second one is where the parts
/// the single-frame path never touches come in: `Downsample3d`'s stride-2
/// temporal convolution and the cache carried across chunks. This is what makes
/// end-frame conditioning affordable — the CPU takes 744 s for this clip.
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn gpu_encoder_matches_cpu_multi_chunk() {
    use crate::weight::loader::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_VAE") else {
        eprintln!("set KLEARU_WAN_VAE; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    let vae = WanVae::load(cfg.clone(), &ComponentTensors::open_dir(&dir).expect("open"))
        .expect("load real weights");

    // Small spatial size: the CPU leg is the slow one and the point here is the
    // temporal path, not resolution.
    let (h, w) = (64usize, 64usize);
    let frames = 5usize;
    let mut px = vec![0.0_f32; 3 * frames * h * w];
    for c in 0..3 {
        for f in 0..frames {
            for y in 0..h {
                for x in 0..w {
                    // Move the bright square per frame so the temporal
                    // convolutions have something to actually see.
                    let inside = (10 + f * 6..30 + f * 6).contains(&y) && (20..44).contains(&x);
                    px[((c * frames + f) * h + y) * w + x] = if inside {
                        0.85
                    } else {
                        (x as f32 / w as f32) * 1.2 - 0.6 + (c as f32) * 0.1
                    };
                }
            }
        }
    }
    let vol = Vol::new(px, 3, frames, h, w);

    let t0 = std::time::Instant::now();
    let want = vae.encode(&vol);
    let cpu_s = t0.elapsed().as_secs_f32();
    let t1 = std::time::Instant::now();
    let got = vae.encode_gpu(&vol).expect("gpu multi-chunk encode");
    let gpu_s = t1.elapsed().as_secs_f32();
    println!("cpu {cpu_s:.1}s -> gpu {gpu_s:.1}s ({:.0}x)", cpu_s / gpu_s.max(1e-6));

    assert_eq!(
        (got.c, got.t, got.h, got.w),
        (want.c, want.t, want.h, want.w),
        "shape differs"
    );
    let half = got.data.len() / 2;
    let cos = |a: &[f32], b: &[f32]| -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb + 1e-12)
    };
    let mean_only = cos(&got.data[..half], &want.data[..half]);
    println!("all: cos_sim={:.8}  mean half: cos_sim={mean_only:.8}",
             cos(&got.data, &want.data));
    // Compare the second latent frame on its own: that is the one the end-frame
    // slot takes, and the one the first chunk cannot exercise.
    if got.t >= 2 {
        let fp = got.h * got.w;
        let z = cfg.z_dim;
        let grab = |v: &Vol| -> Vec<f32> {
            (0..z).flat_map(|c| {
                let b = (c * v.t + 1) * fp;
                v.data[b..b + fp].to_vec()
            }).collect()
        };
        println!("frame 1 only: cos_sim={:.8}", cos(&grab(&got), &grab(&want)));
        assert!(cos(&grab(&got), &grab(&want)) > 0.999, "second latent frame diverges");
    }
    assert!(mean_only > 0.999, "gpu multi-chunk encode diverges: {mean_only}");
}

/// reference implementation.
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn gpu_encoder_matches_cpu() {
    use super::gpu::GpuEncoder3d;
    use crate::metal_backend::GpuTensor;
    use crate::weight::loader::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_VAE") else {
        eprintln!("set KLEARU_WAN_VAE; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    let vae = WanVae::load(cfg.clone(), &ComponentTensors::open_dir(&dir).expect("open"))
        .expect("load real weights");
    let gpu = GpuEncoder3d::from_cpu(&vae.encoder, &vae.quant_conv);

    // Structured input, not noise: a VAE destroys random pixels, so noise would
    // agree even if the two paths disagreed on real images.
    let (h, w) = (192usize, 320usize);
    let mut px = vec![0.0_f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let inside = (40..120).contains(&y) && (60..200).contains(&x);
            for c in 0..3 {
                px[(c * h + y) * w + x] = if inside {
                    0.8
                } else {
                    (x as f32 / w as f32) * 1.2 - 0.6 + (c as f32) * 0.15
                };
            }
        }
    }

    // Both encoders take patchified input; `WanVae::encode` patchifies
    // internally, so drive the GPU stack through the same front door.
    let t0 = std::time::Instant::now();
    let want = vae.encode(&Vol::new(px.clone(), 3, 1, h, w));
    let cpu_s = t0.elapsed().as_secs_f32();

    let patched = patchify(&Vol::new(px, 3, 1, h, w), cfg.patch_size);
    let t1 = std::time::Instant::now();
    let x = crate::vae3d::gpu::upload_vol(&patched);
    let (out, ot, oh, ow) = gpu.forward(&x, patched.t, patched.h, patched.w);
    let got = out.download_to_f32();
    let gpu_s = t1.elapsed().as_secs_f32();

    println!("cpu {cpu_s:.1}s -> gpu {gpu_s:.1}s  ({:.0}x)", cpu_s / gpu_s.max(1e-6));
    println!("gpu [{}, {ot}, {oh}, {ow}] vs cpu [{}, {}, {}, {}]",
             out.elements() / (ot * oh * ow), want.c, want.t, want.h, want.w);
    assert_eq!(got.len(), want.data.len(), "moment count differs");

    let half = got.len() / 2;
    let cos = |a: &[f32], b: &[f32]| -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb + 1e-12)
    };
    let all = cos(&got, &want.data);
    // The mean half is what conditioning uses; logvar is large and negative and
    // would otherwise dominate the whole-tensor figure.
    let mean_only = cos(&got[..half], &want.data[..half]);
    let worst = got[..half]
        .iter()
        .zip(&want.data[..half])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("all: cos_sim={all:.8}  mean half: cos_sim={mean_only:.8} max_abs={worst:.3e}");

    assert!(mean_only > 0.999, "gpu encoder mean diverges from cpu: {mean_only}");
    assert!(all > 0.999, "gpu encoder diverges from cpu: {all}");
    assert!(gpu_s < cpu_s, "gpu encoder ({gpu_s:.1}s) should beat cpu ({cpu_s:.1}s)");
}


/// Is a single-frame encode good enough for the *end* slot?
///
/// The end image is currently encoded as a five-frame clip, because latent
/// frames after the first stand for four pixel frames each. That is faithful
/// but costs two causal chunks on the CPU — 758 s at 320x192, and ~3.25 hours
/// extrapolated to native 1280x704, which is not a usable feature.
///
/// A single-frame encode is seconds on the GPU. This measures what that
/// approximation actually costs in the latent, so the choice is made on a
/// number rather than on the fact that one path is faster.
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn end_slot_single_frame_encode_vs_clip() {
    use crate::weight::loader::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_VAE") else {
        eprintln!("set KLEARU_WAN_VAE; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    let vae = WanVae::load(cfg.clone(), &ComponentTensors::open_dir(&dir).expect("open"))
        .expect("load real weights");

    let (h, w) = (192usize, 320usize);
    let mut px = vec![0.0_f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let inside = (40..120).contains(&y) && (60..200).contains(&x);
            for c in 0..3 {
                px[(c * h + y) * w + x] = if inside {
                    0.8
                } else {
                    (x as f32 / w as f32) * 1.2 - 0.6 + (c as f32) * 0.15
                };
            }
        }
    }
    let img = Vol::new(px, 3, 1, h, w);

    // What the pipeline does today: repeat to five frames, take latent frame 1.
    let t0 = std::time::Instant::now();
    let mut clip_data = Vec::with_capacity(3 * 5 * h * w);
    for c in 0..3 {
        for _ in 0..5 {
            clip_data.extend_from_slice(&img.data[c * h * w..(c + 1) * h * w]);
        }
    }
    let clip = vae.encode(&Vol::new(clip_data, 3, 5, h, w));
    let clip_s = t0.elapsed().as_secs_f32();

    let t1 = std::time::Instant::now();
    let single = vae.encode_frame_gpu(&img).expect("gpu single-frame encode");
    let single_s = t1.elapsed().as_secs_f32();
    println!("clip(5f, cpu) {clip_s:.1}s   single(1f, gpu) {single_s:.1}s");

    // Compare the mean halves at the frame each strategy would place.
    let z = cfg.z_dim;
    let fp = clip.h * clip.w;
    let grab = |v: &Vol, frame: usize| -> Vec<f32> {
        (0..z)
            .flat_map(|c| {
                let base = (c * v.t + frame) * fp;
                v.data[base..base + fp].to_vec()
            })
            .collect()
    };
    let a = grab(&clip, 1);
    let b = grab(&single, 0);
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
    println!("end-slot latent: cos_sim={:.6}", dot / (na * nb + 1e-12));
    println!("  clip  mean/std = {:.4} / {:.4}",
             a.iter().sum::<f32>() / a.len() as f32,
             (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt());
    println!("  single mean/std = {:.4} / {:.4}",
             b.iter().sum::<f32>() / b.len() as f32,
             (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt());
}

/// cost to state up front, not a detail.
#[test]
#[ignore]
fn encoder_timing_by_resolution() {
    use crate::weight::loader::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_VAE") else {
        eprintln!("set KLEARU_WAN_VAE; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    let t0 = std::time::Instant::now();
    let vae = WanVae::load(cfg, &ComponentTensors::open_dir(&dir).expect("open"))
        .expect("load real weights");
    println!("vae load: {:.1}s", t0.elapsed().as_secs_f32());

    for (h, w, frames) in [(192usize, 320usize, 1usize), (352, 640, 1), (704, 1280, 1), (192, 320, 5)] {
        let px = Vol::new(vec![0.1_f32; 3 * frames * h * w], 3, frames, h, w);
        let t = std::time::Instant::now();
        let out = vae.encode(&px);
        println!(
            "encode {w}x{h} x{frames} frames -> [{}, {}, {}, {}] in {:.1}s",
            out.c, out.t, out.h, out.w,
            t.elapsed().as_secs_f32()
        );
    }
}

/// anything depends on it.
#[test]
#[ignore]
fn encoder_matches_diffusers() {
    use crate::weight::loader::ComponentTensors;

    let (Ok(dir), Ok(ref_path)) =
        (std::env::var("KLEARU_WAN_VAE"), std::env::var("KLEARU_VAE_ENCODE_REF"))
    else {
        eprintln!("set KLEARU_WAN_VAE and KLEARU_VAE_ENCODE_REF; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let ref_dir = std::path::Path::new(&ref_path).parent().unwrap().to_path_buf();
    let r = ComponentTensors::open_dir(&ref_dir).expect("open reference");

    let shape = r.get_f32("shape").expect("shape");
    let (t, h, w) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let px = r.get_f32("in.pixels").expect("in.pixels");
    let want = r.get_f32("out.moments").expect("out.moments");

    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    let comp = ComponentTensors::open_dir(&dir).expect("open checkpoint");
    let vae = WanVae::load(cfg.clone(), &comp).expect("load real weights");

    let got = vae.encode(&Vol::new(px, 3, t, h, w));
    println!("klearu [{}, {}, {}, {}]", got.c, got.t, got.h, got.w);
    assert_eq!(got.data.len(), want.len(), "moment count differs");

    let worst = got
        .data
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let dot: f64 = got.data.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = got.data.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb + 1e-12);
    println!("vs diffusers: cos_sim={cos:.8} max_abs_diff={worst:.4e}");

    // The mean half is what conditioning actually uses, so check it on its own:
    // logvar is large and negative, and it would otherwise dominate cos_sim and
    // hide an error in the half that matters.
    let half = got.data.len() / 2;
    let dotm: f64 =
        got.data[..half].iter().zip(&want[..half]).map(|(a, b)| *a as f64 * *b as f64).sum();
    let nam: f64 = got.data[..half].iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nbm: f64 = want[..half].iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cos_mean = dotm / (nam * nbm + 1e-12);
    let worst_mean = got.data[..half]
        .iter()
        .zip(&want[..half])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("mean half:    cos_sim={cos_mean:.8} max_abs_diff={worst_mean:.4e}");

    assert!(cos_mean > 0.9999, "encoder mean diverges from diffusers: {cos_mean}");
    assert!(cos > 0.9999, "encoder diverges from diffusers: {cos}");
}

///     -- --ignored --nocapture real_wan_vae
#[test]
#[ignore]
fn real_wan_vae_loads_and_decodes() {
    use crate::weight::loader::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_VAE") else {
        eprintln!("set KLEARU_WAN_VAE to the vae/ directory; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");

    println!(
        "config: z_dim={} base={} dec_base={} patch={} residual={} spatial={}x temporal={}x",
        cfg.z_dim,
        cfg.base_dim,
        cfg.decoder_dim(),
        cfg.patch_size,
        cfg.is_residual,
        cfg.spatial_compression(),
        cfg.temporal_compression(),
    );

    let t0 = std::time::Instant::now();
    let comp = ComponentTensors::open_dir(&dir).expect("open checkpoint");
    println!("checkpoint holds {} tensors", comp.tensors.len());

    let blank = WanVae::new(cfg.clone());
    let specs = blank.tensor_specs();
    println!("model requests {} tensors", specs.len());
    let missing = blank.missing_tensors(&comp);
    assert!(missing.is_empty(), "missing {}: {:?}", missing.len(), &missing[..missing.len().min(8)]);
    let bad = blank.mismatched_tensors(&comp);
    assert!(bad.is_empty(), "mis-sized {}: {:?}", bad.len(), &bad[..bad.len().min(8)]);

    let vae = WanVae::load(cfg.clone(), &comp).expect("load real weights");
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // A small latent: 2 frames -> 5 output frames at 16x spatial.
    // Overridable, because 2x2 is not a real test of the spatial path: a 3x3
    // convolution on a 2x2 input is almost all padding, and the decoder runs at
    // 44x80 in a native-resolution generation. `KLEARU_VAE_HW=8,12` etc.
    let (t, h, w) = {
        let hw = std::env::var("KLEARU_VAE_HW").unwrap_or_else(|_| "2,2".into());
        let mut it = hw.split(',').map(|v| v.trim().parse::<usize>().unwrap());
        let (hh, ww) = (it.next().unwrap(), it.next().unwrap());
        let tt: usize = std::env::var("KLEARU_VAE_T")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        (tt, hh, ww)
    };
    println!("vae gpu-vs-cpu on real weights at latent ({t}, {h}, {w})");
    let mut z = Vol::zeros(cfg.z_dim, t, h, w);
    // Latents are roughly unit-normal in normalised space.
    let mut s = 12345u32;
    for v in z.data.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *v = ((s >> 8) as f32 / 8_388_608.0) - 1.0;
    }

    let t1 = std::time::Instant::now();
    let out = vae.decode(&z);
    println!(
        "decode {:?} -> [{}, {}, {}, {}] in {:.1}s",
        (cfg.z_dim, t, h, w), out.c, out.t, out.h, out.w,
        t1.elapsed().as_secs_f32()
    );

    assert_eq!(out.c, 3, "RGB after unpatchify");
    assert_eq!(out.t, 1 + (t - 1) * cfg.temporal_compression());
    assert_eq!(out.h, h * cfg.spatial_compression());

    assert!(out.data.iter().all(|v| v.is_finite()), "decode produced non-finite pixels");
    let min = out.data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = out.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = out.data.iter().sum::<f32>() / out.data.len() as f32;
    let var = out.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / out.data.len() as f32;
    println!("pixels: min={min:.4} max={max:.4} mean={mean:.4} std={:.4}", var.sqrt());

    // Trained weights should give a spread of values, not a saturated or flat
    // image — either would mean the weights were loaded but misapplied.
    assert!(var.sqrt() > 1e-3, "output is nearly constant (std={})", var.sqrt());
    let saturated =
        out.data.iter().filter(|v| v.abs() >= 0.999).count() as f32 / out.data.len() as f32;
    println!("saturated fraction: {saturated:.3}");
    assert!(saturated < 0.9, "almost everything clamped: {saturated}");

    // The GPU decoder on the same real weights: this is the comparison that
    // matters, since every earlier GPU check used miniature models whose
    // channel counts (4-6) are nothing like the real 256-1024.
    #[cfg(feature = "metal")]
    {
        use super::gpu::GpuDecoder3d;
        let t2 = std::time::Instant::now();
        let gpu = GpuDecoder3d::from_cpu(&vae.decoder);
        let upload = t2.elapsed().as_secs_f32();

        let t3 = std::time::Instant::now();
        let zq = vae.post_quant_conv.forward(&z, None);
        let raw = gpu.decode_all(&zq);
        let mut gpu_out = unpatchify(&raw, cfg.patch_size);
        for v in gpu_out.data.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        let gpu_secs = t3.elapsed().as_secs_f32();
        println!("gpu: weights uploaded in {upload:.1}s, decode in {gpu_secs:.1}s");

        assert_eq!(
            (gpu_out.c, gpu_out.t, gpu_out.h, gpu_out.w),
            (out.c, out.t, out.h, out.w)
        );
        let dot: f64 =
            gpu_out.data.iter().zip(&out.data).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = gpu_out.data.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = out.data.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb + 1e-12);
        println!("gpu vs cpu on real weights: cos_sim={cos:.6}");
        assert!(gpu_out.data.iter().all(|v| v.is_finite()), "gpu decode produced non-finite");
        // 0.99 is far too loose to catch a real divergence in an image
        // decoder; anything genuinely correct here sits at 0.9999+.
        assert!(cos > 0.999, "gpu decode diverged on real weights: {cos}");
    }
}

/// The decoder must be translation-invariant, checked at full native size.
///
/// Every op in the decoder is position-independent — convolutions, nearest
/// upsampling, channel RMSNorm, and attention (whose queries and keys are all
/// identical when the input is). So a latent that is constant across height and
/// width must decode to an image that is constant across its interior. Any
/// indexing, stride, tiling or padding bug in the GPU path breaks that, and it
/// breaks visibly.
///
/// This exists because the real-weights GPU-vs-CPU check runs at a 2x2 latent,
/// where a 3x3 convolution is almost entirely padding, while generation runs at
/// 44x80. The scalar CPU decoder is too slow to serve as ground truth at that
/// size (45+ minutes for an 8x12 latent), and this needs no reference at all.
///
///   KLEARU_WAN_VAE=models/wan22-ti2v-5b/vae cargo test -p klearu-diffusion \
///     --features metal --lib --release -- --ignored --nocapture \
///     decoder_is_translation_invariant
#[test]
#[ignore]
fn decoder_is_translation_invariant_at_native_size() {
    use super::gpu::GpuDecoder3d;
    use crate::weight::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_VAE") else {
        eprintln!("set KLEARU_WAN_VAE to the vae/ directory; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("vae config"),
    )
    .expect("parse vae config");
    let comp = ComponentTensors::open_dir(&dir).expect("vae shards");
    let vae = WanVae::load(cfg.clone(), &comp).expect("load real weights");
    let gpu = GpuDecoder3d::from_cpu(&vae.decoder);

    // Native spatial size; two latent frames is enough to exercise the causal
    // temporal path without decoding a full clip.
    let (t, h, w) = (2usize, 44usize, 80usize);
    let mut z = Vol::zeros(cfg.z_dim, t, h, w);
    for c in 0..cfg.z_dim {
        for ti in 0..t {
            // Distinct per (channel, frame) so the test is not degenerate, but
            // constant across space, which is the invariant under test.
            let val = 0.35 * ((c * 7 + ti * 13) % 11) as f32 - 1.5;
            for y in 0..h {
                for x in 0..w {
                    let i = z.idx(c, ti, y, x);
                    z.data[i] = val;
                }
            }
        }
    }

    let out = gpu.decode_all(&z);
    println!("decoded [{}, {}, {}, {}]", out.c, out.t, out.h, out.w);

    // Ignore a margin wider than the decoder's receptive field, where padding
    // legitimately makes the output differ.
    let margin = 12 * cfg.spatial_compression();
    let mut worst = 0.0f32;
    let (mut wc, mut wt) = (0usize, 0usize);
    for c in 0..out.c {
        for ti in 0..out.t {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for y in margin..out.h - margin {
                for x in margin..out.w - margin {
                    let v = out.at(c, ti, y, x);
                    assert!(v.is_finite(), "non-finite pixel at c={c} t={ti} y={y} x={x}");
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            if hi - lo > worst {
                worst = hi - lo;
                wc = c;
                wt = ti;
            }
        }
    }
    println!(
        "max interior spread: {worst:.3e} (channel {wc}, frame {wt}), margin {margin}px"
    );
    // bf16 storage carries ~3 decimal digits, so exact equality is not the bar;
    // a genuine indexing or tiling bug produces spreads orders of magnitude
    // larger than rounding.
    assert!(worst < 1e-2, "decoder is not translation-invariant: interior spread {worst:.3e}");
}

/// The GPU decoder against diffusers on real weights at a realistic size.
///
/// This is the check the 2x2 real-weights test cannot be: at 22x40 the spatial
/// convolutions actually do work, where at 2x2 a 3x3 kernel is nearly all
/// padding. klearu's CPU decoder is unusable as ground truth here (scalar Rust,
/// 45+ minutes on an 8x12 latent), so the reference comes from Python.
///
///   /tmp/klearu_ref_venv/bin/python scripts/dump_wan_vae_reference.py \
///       --vae models/wan22-ti2v-5b/vae --out /tmp/wanvae/vae_ref.safetensors
///   KLEARU_WAN_VAE=models/wan22-ti2v-5b/vae \
///   KLEARU_VAE_DIFFUSERS_REF=/tmp/wanvae/vae_ref.safetensors \
///   cargo test -p klearu-diffusion --features metal --lib --release \
///     -- --ignored --nocapture gpu_decoder_matches_diffusers
#[test]
#[ignore]
fn gpu_decoder_matches_diffusers_at_real_size() {
    use super::gpu::GpuDecoder3d;
    use crate::weight::ComponentTensors;

    let (Ok(dir), Ok(ref_path)) = (
        std::env::var("KLEARU_WAN_VAE"),
        std::env::var("KLEARU_VAE_DIFFUSERS_REF"),
    ) else {
        eprintln!("set KLEARU_WAN_VAE and KLEARU_VAE_DIFFUSERS_REF; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("vae config"),
    )
    .expect("parse vae config");
    let comp = ComponentTensors::open_dir(&dir).expect("vae shards");
    let vae = WanVae::load(cfg.clone(), &comp).expect("load real weights");
    let gpu = GpuDecoder3d::from_cpu(&vae.decoder);

    // `open_dir` scans a directory of safetensors, so point it at the dump's
    // parent — the reference file is the only thing in /tmp/wanvae.
    let ref_dir = std::path::Path::new(&ref_path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let rf = ComponentTensors::open_dir(&ref_dir).expect("open reference");
    let latent = rf.get_f32("in.latent").expect("in.latent");
    let want = rf.get_f32("out.video").expect("out.video");

    // Shape comes from the dump; override to match a smaller reference.
    let (t, h, w) = {
        let hw = std::env::var("KLEARU_VAE_HW").unwrap_or_else(|_| "22,40".into());
        let mut it = hw.split(',').map(|v| v.trim().parse::<usize>().unwrap());
        let (hh, ww) = (it.next().unwrap(), it.next().unwrap());
        let tt: usize = std::env::var("KLEARU_VAE_T")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        (tt, hh, ww)
    };
    assert_eq!(latent.len(), cfg.z_dim * t * h * w, "reference latent shape moved");

    let zq = vae
        .post_quant_conv
        .forward(&Vol::new(latent, cfg.z_dim, t, h, w), None);
    let raw = gpu.decode_all(&zq);
    let mut got = unpatchify(&raw, cfg.patch_size);
    // `WanVae::decode` clamps to [-1,1] and the generation pipeline does too,
    // so the GPU side must clamp before the comparison — otherwise this
    // measures a missing clamp rather than a decoder difference.
    if std::env::var_os("KLEARU_VAE_NO_CLAMP").is_none() {
        for v in got.data.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
    }
    println!("klearu [{}, {}, {}, {}]", got.c, got.t, got.h, got.w);
    assert_eq!(got.data.len(), want.len(), "output element count differs");

    let (mut dot, mut na, mut nb, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (x, y) in want.iter().zip(got.data.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        worst = worst.max((x - y).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    let wm = want.iter().sum::<f32>() / want.len() as f32;
    let gm = got.data.iter().sum::<f32>() / got.data.len() as f32;
    println!(
        "vs diffusers: cos_sim={cos:.8} max_abs_diff={worst:.4e}           mean {wm:.5} vs {gm:.5}"
    );
    // `KLEARU_VAE_ALSO_CPU=1` runs klearu's CPU decoder on the same input.
    // That bisects the divergence: if the CPU path matches diffusers and the
    // GPU one does not, the fault is in the Metal kernels rather than in this
    // crate's understanding of the architecture. It is off by default because
    // the scalar decoder is impractically slow above ~6x8.
    if std::env::var_os("KLEARU_VAE_ALSO_CPU").is_some() {
        // `WanVae::decode` applies post_quant_conv itself, so it takes the raw
        // latent rather than `zq`.
        // `WanVae::decode` applies post_quant_conv AND unpatchify itself, so
        // it takes the raw latent and its output needs no further unpacking.
        let cpu = vae.decode(&Vol::new(
            rf.get_f32("in.latent").expect("in.latent"),
            cfg.z_dim, t, h, w,
        ));
        let (mut d2, mut n2a, mut n2b, mut w2) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
        for (x, y) in want.iter().zip(cpu.data.iter()) {
            d2 += (*x as f64) * (*y as f64);
            n2a += (*x as f64) * (*x as f64);
            n2b += (*y as f64) * (*y as f64);
            w2 = w2.max((x - y).abs());
        }
        println!(
            "cpu vs diffusers: cos_sim={:.8} max_abs_diff={w2:.4e}",
            d2 / (n2a.sqrt() * n2b.sqrt() + 1e-12)
        );
        // And GPU against CPU on the *same* input, so all three comparisons
        // share one latent. The pre-existing GPU-vs-CPU test uses a uniform
        // [-1,1] latent and scores 0.99998, which cannot be reconciled with a
        // GPU-vs-diffusers score of 0.9985 unless the input distribution
        // matters — this line settles that rather than leaving it inferred.
        let (mut d3, mut n3a, mut n3b, mut w3) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
        for (x, y) in cpu.data.iter().zip(got.data.iter()) {
            d3 += (*x as f64) * (*y as f64);
            n3a += (*x as f64) * (*x as f64);
            n3b += (*y as f64) * (*y as f64);
            w3 = w3.max((x - y).abs());
        }
        println!(
            "gpu vs cpu (same input): cos_sim={:.8} max_abs_diff={w3:.4e}",
            d3 / (n3a.sqrt() * n3b.sqrt() + 1e-12)
        );
    }
    assert!(cos > 0.999, "GPU decoder diverges from diffusers: cos_sim {cos}");
}

/// Bisect the GPU-vs-CPU decoder divergence to a single stage.
///
/// `gpu_decoder_matches_diffusers_at_real_size` shows the GPU decoder differs
/// from both the CPU decoder and diffusers (cos_sim 0.9985, max_abs 0.85 on an
/// output whose std is 0.50) while the CPU path is exact. This walks the two
/// decoders in lockstep and reports agreement after every stage, so the first
/// one to drop identifies the op — the same approach that took the DiT NaN down
/// to `attn.q_matmul`.
///
/// Uses a randn input at the real latent scale: the divergence is
/// magnitude-dependent and a uniform [-1,1] latent understates it.
///
///   KLEARU_WAN_VAE=models/wan22-ti2v-5b/vae cargo test -p klearu-diffusion \
///     --features metal --lib --release -- --ignored --nocapture bisect_decoder
#[test]
#[ignore]
fn bisect_decoder_stage_divergence() {
    use super::blocks::cached_conv;
    use super::gpu::{download_vol, upload_vol, GpuDecoder3d, GpuFeatCache};
    use super::blocks::FeatCache;
    use crate::metal_backend::GpuTensor;
    use crate::weight::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_VAE") else {
        eprintln!("set KLEARU_WAN_VAE to the vae/ directory; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("vae config"),
    )
    .expect("parse vae config");
    let comp = ComponentTensors::open_dir(&dir).expect("vae shards");
    let vae = WanVae::load(cfg.clone(), &comp).expect("load real weights");
    let gpu = GpuDecoder3d::from_cpu(&vae.decoder);

    // Use the SAME latent as the diffusers reference. Driving this bisection
    // off a locally generated one made its per-stage numbers incomparable with
    // the end-to-end result, which is how a clean-looking 0.99998 here sat next
    // to a 0.99849 there and looked like the tail was at fault.
    let (t, h, w) = (1usize, 4usize, 6usize);
    let ref_path = std::env::var("KLEARU_VAE_DIFFUSERS_REF")
        .expect("set KLEARU_VAE_DIFFUSERS_REF to a --frames 1 dump");
    let ref_dir = std::path::Path::new(&ref_path)
        .parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let rf = ComponentTensors::open_dir(&ref_dir).expect("open reference");
    let z = Vol::new(rf.get_f32("in.latent").expect("in.latent"), cfg.z_dim, t, h, w);
    assert_eq!(z.data.len(), cfg.z_dim * t * h * w, "reference is not a 1-frame 4x6 dump");
    let zq = vae.post_quant_conv.forward(&z, None);

    let mut c_cpu = FeatCache::new(vae.decoder.convs());
    let mut c_gpu = GpuFeatCache::new(gpu.convs);
    crate::metal_backend::set_use_f32_sgemm(true);

    let report = |label: &str, cpu: &Vol, g: &GpuTensor, gt: usize, gh: usize, gw: usize| {
        let got = download_vol(g, g.elements() / (gt * gh * gw), gt, gh, gw);
        assert_eq!(got.data.len(), cpu.data.len(), "{label}: shape differs");
        let (mut dot, mut na, mut nb, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
        for (x, y) in cpu.data.iter().zip(got.data.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
            worst = worst.max((x - y).abs());
        }
        let cpu_absmax = cpu.data.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        println!(
            "{label:<22} cos_sim={:.8}  max_abs_diff={worst:.4e}  cpu_absmax={cpu_absmax:.3e}",
            dot / (na.sqrt() * nb.sqrt() + 1e-12)
        );
    };

    // conv_in
    let cpu_x = cached_conv(&vae.decoder.conv_in, &zq, Some(&mut c_cpu));
    let (mut gx, mut gt, mut gh, mut gw) =
        gpu.conv_in.forward_slot(&upload_vol(&zq), t, h, w, &mut c_gpu);
    report("conv_in", &cpu_x, &gx, gt, gh, gw);

    // mid block
    let cpu_x = vae.decoder.mid_block.forward(&cpu_x, Some(&mut c_cpu));
    let mut gy = gpu.mid_block.resnets[0].forward_slot(&gx, gt, gh, gw, &mut c_gpu).0;
    for (attn, resnet) in gpu.mid_block.attentions.iter().zip(&gpu.mid_block.resnets[1..]) {
        gy = attn.forward(&gy, gt, gh, gw);
        gy = resnet.forward_slot(&gy, gt, gh, gw, &mut c_gpu).0;
    }
    gx = gy;
    report("mid_block", &cpu_x, &gx, gt, gh, gw);

    // up blocks
    let mut cpu_x = cpu_x;
    for (i, (cup, gup)) in vae.decoder.up_blocks.iter().zip(&gpu.up_blocks).enumerate() {
        cpu_x = cup.forward(&cpu_x, Some(&mut c_cpu), true);
        let (o, ot, oh, ow) = gup.forward(&gx, gt, gh, gw, &mut c_gpu, true);
        gx = o;
        gt = ot;
        gh = oh;
        gw = ow;
        report(&format!("up_block[{i}]"), &cpu_x, &gx, gt, gh, gw);
    }

    // The tail. up_block[3] still agrees to 0.99996 while the full decode is
    // 0.99849, so most of the error arrives after this point.
    let mut cpu_n = vae.decoder.norm_out.forward(&cpu_x);
    let mut gn = gx.clone_data();
    gpu.norm_out.forward(&mut gn, gt, gh, gw);
    report("norm_out", &cpu_n, &gn, gt, gh, gw);

    super::silu(&mut cpu_n.data);
    crate::metal_backend::silu_f16_gpu(&mut gn);
    report("norm_out+silu", &cpu_n, &gn, gt, gh, gw);

    let cpu_o = cached_conv(&vae.decoder.conv_out, &cpu_n, Some(&mut c_cpu));
    let (go, ot, oh, ow) = gpu.conv_out.forward_slot(&gn, gt, gh, gw, &mut c_gpu);
    report("conv_out", &cpu_o, &go, ot, oh, ow);

    crate::metal_backend::set_use_f32_sgemm(false);
}

// -- Wan 2.1 / Qwen-Image (plain stages) ------------------------------------

/// The Qwen-Image VAE config Krea 2 ships, verbatim from
/// `Krea-2-Turbo/vae/config.json`.
const QWEN_IMAGE_VAE_CONFIG: &str = r#"{
  "_class_name": "AutoencoderKLQwenImage",
  "_diffusers_version": "0.39.0.dev0",
  "_name_or_path": "Qwen/Qwen-Image",
  "attn_scales": [],
  "base_dim": 96,
  "dim_mult": [1, 2, 4, 4],
  "dropout": 0.0,
  "input_channels": 3,
  "latents_mean": [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
  ],
  "latents_std": [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916
  ],
  "num_res_blocks": 2,
  "temperal_downsample": [false, true, true],
  "z_dim": 16
}"#;

fn qwen_image_cfg() -> WanVaeConfig {
    WanVaeConfig::from_json(QWEN_IMAGE_VAE_CONFIG).unwrap()
}

#[test]
fn parses_qwen_image_vae_config() {
    let cfg = qwen_image_cfg();
    // `input_channels`, not `in_channels`: without the alias this is a parse
    // error, and with a silent default of 0 the encoder would take no input.
    assert_eq!(cfg.in_channels, 3);
    assert_eq!(cfg.z_dim, 16);
    assert_eq!(cfg.base_dim, 96);
    assert_eq!(cfg.decoder_dim(), 96, "no decoder_base_dim: decoder matches the encoder");
    assert!(!cfg.is_residual, "Qwen-Image is the Wan 2.1 (plain) layout");
    assert_eq!(super::blocks::StageKind::of(&cfg), super::blocks::StageKind::Plain);
    assert_eq!(cfg.patch_size, 1, "no pixel patchify");
    assert_eq!(cfg.spatial_compression(), 8);
    assert_eq!(cfg.temporal_compression(), 4);
    assert_eq!(cfg.latents_mean.len(), 16);
    assert_eq!(cfg.latents_std.len(), 16);
    assert_eq!(cfg.latents_std[15], 1.916);
}

/// A miniature Wan 2.1-shaped config: the Qwen-Image structure at width 4.
fn tiny21_cfg() -> WanVaeConfig {
    WanVaeConfig::from_json(
        r#"{
            "base_dim": 4, "z_dim": 4, "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 1, "attn_scales": [],
            "temperal_downsample": [false, true, true], "input_channels": 3
        }"#,
    )
    .unwrap()
}

/// [`tiny21_cfg`] with attention in the second encoder stage (scale 0.5) — a
/// layout no shipped checkpoint uses, but the reference supports, and the one
/// that would expose an off-by-one in the flat encoder numbering.
fn tiny21_attn_cfg() -> WanVaeConfig {
    let mut cfg = tiny21_cfg();
    cfg.attn_scales = vec![0.5];
    cfg
}

/// Name hash -> distinct deterministic value, as in
/// `vae_loads_every_tensor_into_the_right_place`.
fn name_tag(name: &str) -> f32 {
    let mut h: u32 = 2_166_136_261;
    for b in name.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(16_777_619);
    }
    (h % 100_000) as f32 / 100_000.0 + 0.1
}

/// Build `cfg` with every tensor filled with a distinct deterministic ramp,
/// **through the loader** — so a test using it also exercises the names.
fn randomized_vae(cfg: &WanVaeConfig) -> WanVae {
    use crate::weight::loader::ComponentTensors;
    let blank = WanVae::new(cfg.clone());
    let mut comp = ComponentTensors {
        mmaps: Vec::new(),
        owned: Vec::new(),
        tensors: std::collections::HashMap::new(),
    };
    for (name, n) in blank.tensor_specs() {
        let mut v = vec![0.0_f32; n];
        fill_slice(&mut v, name_tag(&name) * 100.0);
        // Keep conv weights modest so activations stay in a range where the
        // GPU's half storage is not the thing being measured.
        if name.ends_with(".weight") {
            for x in v.iter_mut() {
                *x *= 0.3;
            }
        }
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        comp.append_synth_tensor(&name, safetensors::Dtype::F32, vec![n], &bytes).unwrap();
    }
    WanVae::load(cfg.clone(), &comp).expect("load randomized VAE")
}

fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cos_sim length mismatch");
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb + 1e-12)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn wan21_decoder_produces_expected_shape() {
    let vae = WanVae::new(tiny21_cfg());
    // 3 latent frames -> 1 + 4*2 = 9; 2x2 -> 16x16 (three halvings, no patchify).
    let out = vae.decode(&Vol::zeros(4, 3, 2, 2));
    assert_eq!((out.c, out.t, out.h, out.w), (3, 9, 16, 16));
    // The image case: one latent frame is one picture.
    let out = vae.decode(&Vol::zeros(4, 1, 2, 3));
    assert_eq!((out.c, out.t, out.h, out.w), (3, 1, 16, 24));
}

#[test]
fn wan21_stage_widths_follow_the_halving_upsamplers() {
    // The Qwen-Image decoder: dims [384, 384, 384, 192, 96], every stage after
    // the first entering at half width because the previous upsampler halved
    // it. Wan 2.2 would have the second stage take 384, not 192.
    let d = super::model::Decoder3d::new(&qwen_image_cfg());
    let io: Vec<(usize, usize)> =
        d.up_blocks.iter().map(|b| (b.resnets[0].conv1.in_c, b.out_dim)).collect();
    assert_eq!(io, vec![(384, 384), (192, 384), (192, 192), (96, 96)]);
    let ups: Vec<Option<(usize, usize)>> = d
        .up_blocks
        .iter()
        .map(|b| b.upsampler.as_ref().map(|u| (u.conv.in_c, u.conv.out_c)))
        .collect();
    assert_eq!(ups, vec![Some((384, 192)), Some((384, 192)), Some((192, 96)), None]);
    assert!(d.up_blocks.iter().all(|b| !b.has_shortcut), "no DupUp3D in Wan 2.1");
    assert_eq!(d.conv_out.in_c, 96);
    assert_eq!(d.conv_out.out_c, 3);

    let e = super::model::Encoder3d::new(&qwen_image_cfg());
    assert_eq!(e.conv_in.in_c, 3);
    assert_eq!(e.conv_out.out_c, 32, "mean and log-variance");
    let modes: Vec<Option<ResampleMode>> =
        e.down_blocks.iter().map(|b| b.downsampler.as_ref().map(|r| r.mode)).collect();
    assert_eq!(
        modes,
        vec![
            Some(ResampleMode::Downsample2d),
            Some(ResampleMode::Downsample3d),
            Some(ResampleMode::Downsample3d),
            None
        ]
    );
}

#[test]
fn wan21_encoder_produces_expected_shape() {
    let vae = WanVae::new(tiny21_cfg());
    let enc = vae.encode(&Vol::zeros(3, 9, 16, 16));
    assert_eq!((enc.c, enc.t, enc.h, enc.w), (8, 3, 2, 2));
    let enc = vae.encode(&Vol::zeros(3, 1, 16, 24));
    assert_eq!((enc.c, enc.t, enc.h, enc.w), (8, 1, 2, 3));
}

#[test]
fn wan21_cache_walk_lengths_match_declared_slot_counts() {
    for cfg in [tiny21_cfg(), tiny21_attn_cfg()] {
        let vae = WanVae::new(cfg);
        let declared = vae.decoder.convs();
        let mut cache = FeatCache::new(declared);
        for first in [true, false] {
            cache.rewind();
            vae.decoder.forward(&Vol::zeros(4, 1, 2, 2), &mut cache, first);
            assert_eq!(cache.idx, declared, "decoder walk (first_chunk={first})");
        }
        let declared = vae.encoder.convs();
        let mut cache = FeatCache::new(declared);
        for t in [1, 4] {
            cache.rewind();
            vae.encoder.forward(&Vol::zeros(3, t, 16, 16), &mut cache);
            assert_eq!(cache.idx, declared, "encoder walk (chunk of {t})");
        }
    }
}

#[test]
fn wan21_decode_is_deterministic_finite_and_clamped() {
    let vae = randomized_vae(&tiny21_cfg());
    let mut z = Vol::zeros(4, 3, 2, 2);
    fill(&mut z, 5.0);
    let a = vae.decode(&z);
    let b = vae.decode(&z);
    assert_eq!(a, b, "decode must not depend on leftover cache state");
    assert!(a.data.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
    // Randomized weights must actually move the output, or every parity test
    // built on `randomized_vae` would compare zeros with zeros.
    let spread = a.data.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(spread > 1e-3, "decode output is degenerate (max |v| = {spread})");
}

/// The strongest structural check without weights: the model asks for
/// exactly the 194 tensors `Krea-2-Turbo/vae/diffusion_pytorch_model.safetensors`
/// holds, under the names read from its header.
#[test]
fn wan21_tensor_names_match_the_qwen_image_checkpoint() {
    let vae = WanVae::new(qwen_image_cfg());
    let specs = vae.tensor_specs();
    assert_eq!(specs.len(), 194, "checkpoint has 194 tensors");
    let unique: std::collections::HashSet<_> = specs.iter().map(|(n, _)| n).collect();
    assert_eq!(unique.len(), specs.len(), "duplicate tensor name requested");
    let size = |n: &str| specs.iter().find(|(x, _)| x == n).map(|(_, s)| *s);
    let has = |n: &str| size(n).is_some();

    // The encoder's down_blocks is flat: resnets at 0,1 / 3,4 / 6,7 / 9,10,
    // resamples at 2, 5, 8 — and only 5 and 8 are temporal.
    for i in [0, 1, 3, 4, 6, 7, 9, 10] {
        assert!(has(&format!("encoder.down_blocks.{i}.conv1.weight")), "resnet at {i}");
    }
    for i in [2, 5, 8] {
        assert!(has(&format!("encoder.down_blocks.{i}.resample.1.weight")), "resample at {i}");
    }
    assert!(!has("encoder.down_blocks.2.time_conv.weight"), "stage 0 is downsample2d");
    assert_eq!(size("encoder.down_blocks.5.time_conv.weight"), Some(192 * 192 * 3));
    assert_eq!(size("encoder.down_blocks.8.time_conv.weight"), Some(384 * 384 * 3));
    assert_eq!(size("encoder.down_blocks.2.resample.1.weight"), Some(96 * 96 * 9));
    assert!(!has("encoder.down_blocks.11.conv1.weight"));
    assert!(!names_contain(&specs, "encoder.down_blocks.0.resnets"), "not nested");
    // Shortcuts exactly where the width changes: 96->192 at 3, 192->384 at 6.
    assert_eq!(size("encoder.down_blocks.3.conv_shortcut.weight"), Some(192 * 96));
    assert_eq!(size("encoder.down_blocks.6.conv_shortcut.weight"), Some(384 * 192));
    assert!(!has("encoder.down_blocks.0.conv_shortcut.weight"));
    assert_eq!(size("encoder.conv_out.weight"), Some(32 * 384 * 27));

    // Decoder upsamplers live in a one-element list, and halve channels.
    assert_eq!(size("decoder.up_blocks.0.upsamplers.0.resample.1.weight"), Some(192 * 384 * 9));
    assert_eq!(size("decoder.up_blocks.0.upsamplers.0.time_conv.weight"), Some(768 * 384 * 3));
    assert_eq!(size("decoder.up_blocks.1.upsamplers.0.time_conv.weight"), Some(768 * 384 * 3));
    assert_eq!(size("decoder.up_blocks.2.upsamplers.0.resample.1.weight"), Some(96 * 192 * 9));
    assert!(!has("decoder.up_blocks.2.upsamplers.0.time_conv.weight"));
    assert!(!names_contain(&specs, "decoder.up_blocks.3.upsamplers"));
    assert!(!names_contain(&specs, ".upsampler."), "Wan 2.2 naming leaked in");
    // The halved input shows up as a 192 -> 384 first resnet in stage 1.
    assert_eq!(size("decoder.up_blocks.1.resnets.0.conv1.weight"), Some(384 * 192 * 27));
    assert_eq!(size("decoder.up_blocks.1.resnets.0.conv_shortcut.weight"), Some(384 * 192));
    assert!(!has("decoder.up_blocks.2.resnets.0.conv_shortcut.weight"), "192 -> 192");
    assert!(has("decoder.up_blocks.0.resnets.2.conv1.weight"), "num_res_blocks + 1");
    assert_eq!(size("decoder.conv_out.weight"), Some(3 * 96 * 27));
    assert_eq!(size("decoder.mid_block.attentions.0.to_qkv.weight"), Some(1152 * 384));
}

fn names_contain(specs: &[(String, usize)], needle: &str) -> bool {
    specs.iter().any(|(n, _)| n.contains(needle))
}

/// Flat numbering with attention interleaved, and the `upsamplers.0` names,
/// loaded into the right fields. The name-tag fill makes a cross-wired load
/// visible even between same-shaped tensors.
#[test]
fn wan21_loads_every_tensor_into_the_right_place() {
    use crate::weight::loader::ComponentTensors;
    let cfg = tiny21_attn_cfg();
    let blank = WanVae::new(cfg.clone());
    let names = blank.tensor_names();
    // tiny21_attn: stage 0 = [resnet 0, resample 1]; stage 1 (scale 0.5) =
    // [resnet 2, attention 3, resample 4]; stage 2 = [resnet 5, resample 6];
    // stage 3 = [resnet 7].
    for (i, what) in [(0, "conv1"), (1, "resample.1"), (2, "conv1"), (3, "to_qkv"),
                      (4, "resample.1"), (5, "conv1"), (6, "resample.1"), (7, "conv1")] {
        let n = format!("encoder.down_blocks.{i}.{what}.weight");
        assert!(names.contains(&n), "missing {n}");
    }
    assert!(!names.iter().any(|n| n.starts_with("encoder.down_blocks.8.")));

    let mut comp = ComponentTensors {
        mmaps: Vec::new(),
        owned: Vec::new(),
        tensors: std::collections::HashMap::new(),
    };
    for (name, n) in blank.tensor_specs() {
        let bytes: Vec<u8> =
            std::iter::repeat_n(name_tag(&name).to_le_bytes(), n).flatten().collect();
        comp.append_synth_tensor(&name, safetensors::Dtype::F32, vec![n], &bytes).unwrap();
    }
    let vae = WanVae::load(cfg, &comp).unwrap();
    let all = |v: &[f32], name: &str| {
        assert!(!v.is_empty(), "{name} loaded empty");
        assert!(
            v.iter().all(|x| (*x - name_tag(name)).abs() < 1e-6),
            "{name} did not receive its own values (cross-wired?)"
        );
    };
    let e = &vae.encoder;
    all(&e.down_blocks[0].resnets[0].conv1.weight, "encoder.down_blocks.0.conv1.weight");
    all(&e.down_blocks[0].downsampler.as_ref().unwrap().conv.weight,
        "encoder.down_blocks.1.resample.1.weight");
    all(&e.down_blocks[1].resnets[0].norm2.gamma, "encoder.down_blocks.2.norm2.gamma");
    all(&e.down_blocks[1].attentions[0].to_qkv.weight, "encoder.down_blocks.3.to_qkv.weight");
    all(&e.down_blocks[1].attentions[0].norm.gamma, "encoder.down_blocks.3.norm.gamma");
    let ds1 = e.down_blocks[1].downsampler.as_ref().unwrap();
    all(&ds1.conv.weight, "encoder.down_blocks.4.resample.1.weight");
    all(&ds1.time_conv.as_ref().unwrap().weight, "encoder.down_blocks.4.time_conv.weight");
    all(&e.down_blocks[3].resnets[0].conv2.weight, "encoder.down_blocks.7.conv2.weight");

    let up0 = vae.decoder.up_blocks[0].upsampler.as_ref().unwrap();
    all(&up0.conv.weight, "decoder.up_blocks.0.upsamplers.0.resample.1.weight");
    all(&up0.time_conv.as_ref().unwrap().weight, "decoder.up_blocks.0.upsamplers.0.time_conv.weight");
    let up2 = vae.decoder.up_blocks[2].upsampler.as_ref().unwrap();
    assert_eq!(up2.mode, ResampleMode::Upsample2d);
    all(&up2.conv.bias.as_ref().unwrap()[..], "decoder.up_blocks.2.upsamplers.0.resample.1.bias");
    all(&vae.decoder.up_blocks[1].resnets[0].shortcut.as_ref().unwrap().weight,
        "decoder.up_blocks.1.resnets.0.conv_shortcut.weight");

    // And both directions run.
    assert!(vae.decode(&Vol::zeros(4, 2, 2, 2)).data.iter().all(|v| v.is_finite()));
    assert!(vae.encode(&Vol::zeros(3, 1, 16, 16)).data.iter().all(|v| v.is_finite()));
}

#[test]
fn wan21_encoder_attention_runs_and_matters() {
    // The attention-bearing encoder must differ from the plain one on the same
    // shared weights — otherwise the interleaved attention is not running.
    let with = randomized_vae(&tiny21_attn_cfg());
    let mut px = Vol::zeros(3, 1, 16, 16);
    fill(&mut px, 2.0);
    let a = with.encode(&px);
    assert_eq!((a.c, a.t, a.h, a.w), (8, 1, 2, 2));
    assert!(a.data.iter().all(|v| v.is_finite()));
    let mut without = randomized_vae(&tiny21_attn_cfg());
    for b in without.encoder.down_blocks.iter_mut() {
        b.attentions.clear();
    }
    assert!(max_diff(&a, &without.encode(&px)) > 1e-4, "attention had no effect");
}

// -- Wan 2.1 on the GPU -----------------------------------------------------

/// `GpuImageVae` against the CPU VAE at `T = 1`, decode and encode, on a
/// non-square size so an h/w swap cannot hide.
#[cfg(feature = "metal")]
#[test]
fn wan21_gpu_image_vae_matches_cpu() {
    use super::gpu::GpuImageVae;
    let vae = randomized_vae(&tiny21_attn_cfg());
    let gpu = GpuImageVae::from_vae(&vae);
    assert_eq!(gpu.spatial_compression(), 8);

    let (h, w) = (4usize, 6usize);
    let mut z = Vol::zeros(4, 1, h, w);
    fill(&mut z, 7.0);
    let want = vae.decode(&z);
    let got = gpu.decode(&z.data, h, w);
    assert_eq!(got.len(), 3 * 8 * h * 8 * w);
    let (c, m) = (cos_sim(&got, &want.data), max_abs(&got, &want.data));
    println!("wan21 gpu decode vs cpu: cos_sim={c:.7} max_abs={m:.3e}");
    assert!(c > 0.9999, "GPU image decode diverges from CPU: {c}");

    // The one-shot helper is the same computation.
    let once = vae.decode_image_gpu(&z.data, h, w);
    assert_eq!(once, got, "decode_image_gpu differs from GpuImageVae::decode");

    let (ph, pw) = (32usize, 48usize);
    let mut px = Vol::zeros(3, 1, ph, pw);
    fill(&mut px, 3.0);
    let moments = vae.encode(&px);
    let want_mean = &moments.data[..4 * (ph / 8) * (pw / 8)];
    let got = gpu.encode(&px.data, ph, pw);
    assert_eq!(got.len(), want_mean.len());
    let (c, m) = (cos_sim(&got, want_mean), max_abs(&got, want_mean));
    println!("wan21 gpu encode vs cpu: cos_sim={c:.7} max_abs={m:.3e}");
    assert!(c > 0.9999, "GPU image encode diverges from CPU: {c}");
    assert_eq!(vae.encode_image_gpu(&px.data, ph, pw), got);
}

/// Multi-frame on the GPU for the plain layout: `decode_all` exercises the
/// halving `upsample3d`'s temporal conv (skipped on the first chunk, run on
/// the rest), and `encode_gpu` the stride-2 `downsample3d` across chunks.
#[cfg(feature = "metal")]
#[test]
fn wan21_gpu_video_paths_match_cpu() {
    use super::gpu::GpuDecoder3d;
    let vae = randomized_vae(&tiny21_cfg());

    let mut z = Vol::zeros(4, 3, 2, 3);
    fill(&mut z, 9.0);
    let want = vae.decode(&z);
    let zq = vae.post_quant_conv.forward(&z, None);
    let mut got = GpuDecoder3d::from_cpu(&vae.decoder).decode_all(&zq);
    for v in got.data.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
    assert_eq!((got.c, got.t, got.h, got.w), (3, 9, 16, 24));
    let c = cos_sim(&got.data, &want.data);
    println!("wan21 gpu decode_all vs cpu (9 frames): cos_sim={c:.7}");
    // Per frame: a cache or first-chunk error shows as one bad frame, which
    // the overall number would average away.
    let plane = got.h * got.w;
    for f in 0..got.t {
        let pick = |v: &Vol| -> Vec<f32> {
            (0..v.c).flat_map(|ch| {
                let o = (ch * v.t + f) * plane;
                v.data[o..o + plane].to_vec()
            }).collect()
        };
        let fc = cos_sim(&pick(&got), &pick(&want));
        println!("  frame {f}: cos_sim={fc:.7}");
        assert!(fc > 0.999, "frame {f} diverged: {fc}");
    }
    // Lower than the single-frame bound: the randomized weights drive large
    // activations through the half-precision temporal convs of later chunks.
    assert!(c > 0.9995, "GPU multi-frame decode diverges: {c}");

    let mut px = Vol::zeros(3, 5, 16, 24);
    fill(&mut px, 4.0);
    let want = vae.encode(&px);
    let got = vae.encode_gpu(&px).expect("gpu encode");
    assert_eq!((got.c, got.t, got.h, got.w), (want.c, want.t, want.h, want.w));
    let c = cos_sim(&got.data, &want.data);
    println!("wan21 gpu encode_gpu vs cpu (5 frames): cos_sim={c:.7}");
    assert!(c > 0.9999, "GPU multi-chunk encode diverges: {c}");
}

/// The single-frame cache must change memory, not results: bit-identical to a
/// retaining cache on one chunk, for both directions.
#[cfg(feature = "metal")]
#[test]
fn single_frame_cache_matches_retaining_cache_exactly() {
    use super::gpu::{upload_vol, GpuDecoder3d, GpuEncoder3d, GpuFeatCache};
    let vae = randomized_vae(&tiny21_attn_cfg());

    let dec = GpuDecoder3d::from_cpu(&vae.decoder);
    let mut z = Vol::zeros(4, 1, 3, 2);
    fill(&mut z, 1.5);
    let run = |mut cache: GpuFeatCache| {
        cache.rewind();
        dec.forward(&upload_vol(&z), 1, 3, 2, &mut cache, true).0.download_to_f32()
    };
    let a = run(GpuFeatCache::new(dec.convs));
    let b = run(GpuFeatCache::single_frame(dec.convs));
    assert_eq!(a, b, "single_frame decode differs from a retaining cache");

    let enc = GpuEncoder3d::from_cpu(&vae.encoder, &vae.quant_conv);
    let mut px = Vol::zeros(3, 1, 16, 24);
    fill(&mut px, 2.5);
    let run = |mut cache: GpuFeatCache| {
        cache.rewind();
        enc.forward_chunk(&upload_vol(&px), 1, 16, 24, &mut cache).0.download_to_f32()
    };
    let a = run(GpuFeatCache::new(enc.convs));
    let b = run(GpuFeatCache::single_frame(enc.convs));
    assert_eq!(a, b, "single_frame encode differs from a retaining cache");
}

#[cfg(feature = "metal")]
#[test]
#[should_panic(expected = "exactly one chunk")]
fn single_frame_cache_refuses_a_second_chunk() {
    use super::gpu::{upload_vol, GpuDecoder3d, GpuFeatCache};
    let vae = randomized_vae(&tiny21_cfg());
    let dec = GpuDecoder3d::from_cpu(&vae.decoder);
    let z = Vol::zeros(4, 1, 2, 2);
    let mut cache = GpuFeatCache::single_frame(dec.convs);
    cache.rewind();
    dec.forward(&upload_vol(&z), 1, 2, 2, &mut cache, true);
    cache.rewind();
}

/// The row-blocked mid-block attention against the CPU block, at a size the
/// CPU can check: 4x5 = 20 positions in blocks of 3 rows, so the last block is
/// ragged and overlaps its predecessor.
#[cfg(feature = "metal")]
#[test]
fn gpu_attention_row_blocks_match_cpu() {
    use super::blocks::AttentionBlock;
    use super::gpu::{download_vol, upload_vol, GpuAttentionBlock};
    let d = 8;
    let mut a = AttentionBlock::new(d);
    fill_slice(&mut a.norm.gamma, 1.0);
    fill_slice(&mut a.to_qkv.weight, 2.0);
    fill_slice(a.to_qkv.bias.as_mut().unwrap(), 3.0);
    fill_slice(&mut a.proj.weight, 4.0);
    fill_slice(a.proj.bias.as_mut().unwrap(), 5.0);
    let (t, h, w) = (2usize, 4usize, 5usize);
    let mut x = Vol::zeros(d, t, h, w);
    fill(&mut x, 6.0);
    let want = a.forward(&x);

    let g = GpuAttentionBlock::from_cpu(&a);
    crate::metal_backend::set_use_f32_sgemm(true);
    let whole = download_vol(&g.forward(&upload_vol(&x), t, h, w), d, t, h, w);
    let blocked = download_vol(&g.forward_blocked(&upload_vol(&x), t, h, w, 3 * h * w), d, t, h, w);
    crate::metal_backend::set_use_f32_sgemm(false);

    let (cw, cb) = (cos_sim(&whole.data, &want.data), cos_sim(&blocked.data, &want.data));
    println!("attention vs cpu: whole cos_sim={cw:.7}  blocked cos_sim={cb:.7}");
    assert!(cw > 0.9999 && cb > 0.9999);
    assert!(
        max_diff(&whole, &blocked) < 1e-2,
        "row blocking changed the result: {}",
        max_diff(&whole, &blocked)
    );
}

/// The real Qwen-Image VAE (Krea 2's) against diffusers.
///
/// Loads the shipped checkpoint, checks its inventory against the model, then
/// compares the GPU image decoder and encoder against a reference dump at
/// 512x768 (which also takes the mid-block attention through its row-blocked
/// path), and the scalar CPU decoder at a 4x6 latent.
///
///   $VENV/bin/python -I scripts/dump_qwenimage_vae_reference.py \
///       --vae <Krea-2-Turbo>/vae --out <dir>/qwen_vae_ref.safetensors
///   KLEARU_QWEN_VAE=<Krea-2-Turbo>/vae \
///   KLEARU_QWEN_VAE_REF=<dir>/qwen_vae_ref.safetensors \
///   cargo test -p klearu-diffusion --release --features metal --lib \
///       -- --ignored --nocapture qwen_image_vae_matches_diffusers
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn qwen_image_vae_matches_diffusers() {
    use super::gpu::GpuImageVae;
    use crate::weight::loader::ComponentTensors;

    let (Ok(dir), Ok(ref_path)) =
        (std::env::var("KLEARU_QWEN_VAE"), std::env::var("KLEARU_QWEN_VAE_REF"))
    else {
        eprintln!("set KLEARU_QWEN_VAE and KLEARU_QWEN_VAE_REF; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    assert!(!cfg.is_residual);
    let comp = ComponentTensors::open_dir(&dir).expect("open checkpoint");
    let t0 = std::time::Instant::now();
    let vae = WanVae::load(cfg.clone(), &comp).expect("load real weights");
    println!("load: {:.2}s", t0.elapsed().as_secs_f32());
    // Inventory both ways: `load` proves nothing is missing; this proves the
    // checkpoint holds nothing the model ignores.
    let ours: std::collections::HashSet<String> = vae.tensor_names().into_iter().collect();
    let unused: Vec<&String> = comp.tensors.keys().filter(|n| !ours.contains(*n)).collect();
    assert!(unused.is_empty(), "checkpoint tensors the model never reads: {unused:?}");
    assert_eq!(ours.len(), comp.tensors.len());
    drop(comp);

    let ref_dir = std::path::Path::new(&ref_path).parent().unwrap().to_path_buf();
    let r = ComponentTensors::open_dir(&ref_dir).expect("open reference");
    let shape = r.get_f32("shape").expect("shape");
    let (h, w) = (shape[0] as usize, shape[1] as usize);
    let s = cfg.spatial_compression();

    let report = |what: &str, got: &[f32], want: &[f32]| -> f64 {
        assert_eq!(got.len(), want.len(), "{what}: length differs");
        let c = cos_sim(got, want);
        let m = max_abs(got, want);
        let mean_abs = want.iter().map(|v| v.abs() as f64).sum::<f64>() / want.len() as f64;
        println!("{what:<28} cos_sim={c:.8} max_abs_diff={m:.4e} (ref mean |v| {mean_abs:.4})");
        c
    };

    // CPU decoder, small: pins the CPU path itself to diffusers.
    let zs = r.get_f32("in.latent_small").expect("in.latent_small");
    let t1 = std::time::Instant::now();
    let cpu = vae.decode(&Vol::new(zs.clone(), cfg.z_dim, 1, 4, 6));
    println!("cpu decode 4x6: {:.2}s", t1.elapsed().as_secs_f32());
    let want_small = r.get_f32("out.image_small").expect("out.image_small");
    let c_cpu = report("cpu decode 32x48", &cpu.data, &want_small);

    let gpu = GpuImageVae::from_vae(&vae);
    let gpu_video = super::gpu::GpuDecoder3d::from_cpu(&vae.decoder);
    let (pq_weight, pq_bias) =
        (vae.post_quant_conv.weight.clone(), vae.post_quant_conv.bias.clone().unwrap());
    drop(vae);
    let got_small = gpu.decode(&zs, 4, 6);
    let c_gpu_small = report("gpu decode 32x48", &got_small, &want_small);

    let z = r.get_f32("in.latent").expect("in.latent");
    let want = r.get_f32("out.image").expect("out.image");
    // First call pays graph builds and weight uploads; time the second too.
    let t2 = std::time::Instant::now();
    let got = gpu.decode(&z, h, w);
    let first = t2.elapsed().as_secs_f32();
    let t3 = std::time::Instant::now();
    let again = gpu.decode(&z, h, w);
    println!(
        "gpu decode {}x{}: first {first:.2}s, again {:.2}s",
        w * s,
        h * s,
        t3.elapsed().as_secs_f32()
    );
    assert_eq!(got, again, "GPU decode is not deterministic");
    let c_dec = report(&format!("gpu decode {}x{}", h * s, w * s), &got, &want);

    // Three latent frames through `decode_all`: the plain layout's temporal
    // path (upsample3d's time conv via the feature cache) on trained weights.
    let zv = r.get_f32("in.latent_video").expect("in.latent_video");
    let zv = Vol::new(zv, cfg.z_dim, 3, 8, 12);
    let mut cpu_pq = super::CausalConv3d::new(
        cfg.z_dim, cfg.z_dim, (1, 1, 1), (1, 1, 1), (0, 0, 0), true,
    );
    cpu_pq.weight = pq_weight.clone();
    cpu_pq.bias = Some(pq_bias.clone());
    let mut video = gpu_video.decode_all(&cpu_pq.forward(&zv, None));
    for v in video.data.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
    assert_eq!((video.c, video.t, video.h, video.w), (3, 9, 8 * s, 12 * s));
    let want_video = r.get_f32("out.video").expect("out.video");
    let c_vid = report("gpu decode_all 9x64x96", &video.data, &want_video);
    let plane = video.h * video.w;
    for f in 0..video.t {
        let pick = |v: &[f32]| -> Vec<f32> {
            (0..3).flat_map(|ch| {
                let o = (ch * video.t + f) * plane;
                v[o..o + plane].to_vec()
            }).collect()
        };
        println!("  frame {f}: cos_sim={:.8}", cos_sim(&pick(&video.data), &pick(&want_video)));
    }

    let img = r.get_f32("in.image").expect("in.image");
    let want_mean = r.get_f32("out.mean").expect("out.mean");
    let t4 = std::time::Instant::now();
    let mean = gpu.encode(&img, h * s, w * s);
    println!("gpu encode: {:.2}s", t4.elapsed().as_secs_f32());
    let c_enc = report("gpu encode mean", &mean, &want_mean);

    assert!(c_cpu > 0.99999, "CPU decoder diverges from diffusers: {c_cpu}");
    assert!(c_gpu_small > 0.9999, "GPU decoder diverges at 32x48: {c_gpu_small}");
    assert!(c_dec > 0.9999, "GPU decoder diverges from diffusers: {c_dec}");
    assert!(c_enc > 0.9999, "GPU encoder diverges from diffusers: {c_enc}");
    assert!(c_vid > 0.9999, "GPU multi-frame decoder diverges from diffusers: {c_vid}");
}

/// Time and peak GPU allocation of `GpuImageVae` on the real Qwen-Image VAE,
/// per image size.
///
/// The peak is sampled from the device's `currentAllocatedSize` on a side
/// thread every 2 ms, so it is a lower bound — but a close one, since the big
/// tensors live for whole convolutions. `KLEARU_QWEN_VAE_SIZES` takes latent
/// sizes, e.g. `64x96,128x128` (the default is 1024x1024, Krea's default).
///
///   KLEARU_QWEN_VAE=<Krea-2-Turbo>/vae cargo test -p klearu-diffusion \
///       --release --features metal --lib -- --ignored --nocapture qwen_image_vae_gpu_cost
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn qwen_image_vae_gpu_cost_by_resolution() {
    use super::gpu::GpuImageVae;
    use crate::weight::loader::ComponentTensors;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    let Ok(dir) = std::env::var("KLEARU_QWEN_VAE") else {
        eprintln!("set KLEARU_QWEN_VAE; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    let vae = WanVae::load(cfg.clone(), &ComponentTensors::open_dir(&dir).expect("open"))
        .expect("load real weights");
    let device = metal::Device::system_default().expect("metal device");
    let gb = |b: u64| b as f64 / 1e9;
    let base0 = device.current_allocated_size();
    let gpu = GpuImageVae::from_vae(&vae);
    drop(vae);

    let peak = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let (peak, stop) = (peak.clone(), stop.clone());
        std::thread::spawn(move || {
            let d = metal::Device::system_default().unwrap();
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(d.current_allocated_size(), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        })
    };

    let sizes = std::env::var("KLEARU_QWEN_VAE_SIZES").unwrap_or_else(|_| "128x128".into());
    let s = cfg.spatial_compression();
    for spec in sizes.split(',') {
        let mut it = spec.trim().split('x').map(|v| v.parse::<usize>().unwrap());
        let (h, w) = (it.next().unwrap(), it.next().unwrap());
        let z: Vec<f32> = (0..cfg.z_dim * h * w)
            .map(|i| ((i as f32 * 0.37).sin() * 2.0) + cfg.latents_mean[i / (h * w)])
            .collect();
        for pass in ["first", "again"] {
            let base = device.current_allocated_size();
            peak.store(base, Ordering::Relaxed);
            let t = std::time::Instant::now();
            let img = gpu.decode(&z, h, w);
            let dt = t.elapsed().as_secs_f32();
            assert!(img.iter().all(|v| v.is_finite()));
            println!(
                "decode {}x{} ({pass}): {dt:.2}s  peak +{:.2} GB over {:.2} GB resident",
                w * s, h * s,
                gb(peak.load(Ordering::Relaxed).saturating_sub(base)),
                gb(base)
            );
        }
        let base = device.current_allocated_size();
        peak.store(base, Ordering::Relaxed);
        let rgb: Vec<f32> = (0..3 * h * s * w * s).map(|i| (i as f32 * 0.013).sin() * 0.8).collect();
        let t = std::time::Instant::now();
        let lat = gpu.encode(&rgb, h * s, w * s);
        println!(
            "encode {}x{}: {:.2}s  peak +{:.2} GB",
            w * s, h * s,
            t.elapsed().as_secs_f32(),
            gb(peak.load(Ordering::Relaxed).saturating_sub(base))
        );
        assert_eq!(lat.len(), cfg.z_dim * h * w);
    }
    stop.store(true, Ordering::Relaxed);
    sampler.join().unwrap();
    println!(
        "resident after: {:.2} GB (weights uploaded + cached graphs; {:.2} GB before building)",
        gb(device.current_allocated_size()),
        gb(base0)
    );
}

/// A runner holds the image VAE across calls, possibly behind a lock or on a
/// worker thread, so it must be movable between threads. (The backend's caches
/// are thread-local, so *using* it from several threads at once is still not
/// something to do; see `GpuImageVae`.)
#[cfg(feature = "metal")]
#[test]
fn gpu_image_vae_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<super::gpu::GpuImageVae>();
}
