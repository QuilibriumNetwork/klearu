//! Structural tests for the Wan DiT, validated against the real checkpoint's
//! inventory rather than against our own assumptions.

use super::*;

/// Verbatim from `Wan2.2-TI2V-5B-Diffusers/transformer/config.json`.
fn cfg() -> WanDitConfig {
    WanDitConfig::from_json(
        r#"{
            "num_layers": 30,
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "in_channels": 48,
            "out_channels": 48,
            "ffn_dim": 14336,
            "text_dim": 4096,
            "freq_dim": 256,
            "patch_size": [1, 2, 2],
            "eps": 1e-06,
            "qk_norm": "rms_norm_across_heads",
            "cross_attn_norm": true,
            "rope_max_seq_len": 1024,
            "added_kv_proj_dim": null,
            "image_dim": null,
            "pos_embed_seq_len": null
        }"#,
    )
    .unwrap()
}

#[test]
fn config_derives_the_expected_widths() {
    let c = cfg();
    assert_eq!(c.inner_dim(), 3072);
    assert_eq!(c.patch(), (1, 2, 2));
    // Six modulation vectors: shift/scale/gate for attention and for the FFN.
    assert_eq!(c.time_proj_dim(), 6 * 3072);
}

#[test]
fn rope_axis_split_is_uneven_and_time_biased() {
    // head_dim 128: h = w = 2 * (128/6) = 42, t takes the remaining 44.
    // Time is *wider* than either spatial axis, and 128 does not divide by 3.
    let (t, h, w) = cfg().rope_dims();
    assert_eq!((t, h, w), (44, 42, 42));
    assert_eq!(t + h + w, 128);
    assert!(t > h, "time axis should get the remainder");
    for d in [t, h, w] {
        assert_eq!(d % 2, 0, "each axis must be even to rotate in pairs");
    }
}

#[test]
fn rope_at_origin_is_the_identity() {
    let c = cfg();
    let tables = RopeTables::new(&c, 10_000.0);
    let (cos, sin) = tables.token(0, 0, 0);
    assert!(cos.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    assert!(sin.iter().all(|v| v.abs() < 1e-6));

    let mut x: Vec<f32> = (0..c.attention_head_dim).map(|i| i as f32 * 0.01 - 0.3).collect();
    let before = x.clone();
    apply_rope(&mut x, &cos, &sin);
    for (a, b) in x.iter().zip(&before) {
        assert!((a - b).abs() < 1e-6, "position 0 must not rotate");
    }
}

#[test]
fn rope_preserves_the_norm_of_each_pair() {
    // A rotation changes direction, never magnitude.
    let c = cfg();
    let tables = RopeTables::new(&c, 10_000.0);
    let (cos, sin) = tables.token(7, 3, 11);
    let mut x: Vec<f32> = (0..c.attention_head_dim).map(|i| ((i as f32) * 0.37).sin()).collect();
    let before = x.clone();
    apply_rope(&mut x, &cos, &sin);
    for i in 0..x.len() / 2 {
        let n0 = before[2 * i].hypot(before[2 * i + 1]);
        let n1 = x[2 * i].hypot(x[2 * i + 1]);
        assert!((n0 - n1).abs() < 1e-5, "pair {i}: {n0} -> {n1}");
    }
    assert!(x.iter().zip(&before).any(|(a, b)| (a - b).abs() > 1e-3), "should actually rotate");
}

#[test]
fn rope_axes_use_independent_positions_and_frequencies() {
    let c = cfg();
    let tables = RopeTables::new(&c, 10_000.0);
    let (dt, dh, _) = c.rope_dims();

    // Moving along h must leave the time slice untouched, and vice versa.
    let (cos_a, _) = tables.token(5, 0, 0);
    let (cos_b, _) = tables.token(5, 9, 0);
    assert_eq!(cos_a[..dt], cos_b[..dt], "changing h perturbed the t slice");
    assert_ne!(cos_a[dt..dt + dh], cos_b[dt..dt + dh], "h slice should change with h");

    // Each axis derives frequencies from its own width, so the first rotating
    // frequency differs between the 44-wide time axis and the 42-wide h axis.
    let (cos_t1, _) = tables.token(1, 0, 0);
    let (cos_h1, _) = tables.token(0, 1, 0);
    let t_second = cos_t1[2];
    let h_second = cos_h1[dt + 2];
    assert!(
        (t_second - h_second).abs() > 1e-6,
        "axes must not share a frequency schedule: {t_second} vs {h_second}"
    );
}

#[test]
fn tensor_inventory_matches_the_real_checkpoint() {
    // `transformer/diffusion_pytorch_model.safetensors.index.json` maps
    // exactly 825 tensors over five shards. Any structural slip moves it.
    let c = cfg();
    let names = tensor_names(&c);
    assert_eq!(names.len(), 825, "checkpoint index lists 825 tensors");

    let unique: std::collections::HashSet<_> = names.iter().collect();
    assert_eq!(unique.len(), names.len(), "duplicate tensor name");

    let has = |n: &str| names.iter().any(|x| x == n);

    // to_out is a ModuleList; the linear is index 0, dropout is index 1.
    assert!(has("blocks.0.attn1.to_out.0.weight"));
    assert!(!has("blocks.0.attn1.to_out.1.weight"));

    // FeedForward's net is [GELU(proj), Dropout, Linear] — index 1 is absent.
    assert!(has("blocks.0.ffn.net.0.proj.weight"));
    assert!(has("blocks.0.ffn.net.2.weight"));
    assert!(!has("blocks.0.ffn.net.1.weight"));

    // norm1 and norm3 have no affine parameters; only norm2 does.
    assert!(has("blocks.0.norm2.weight"));
    assert!(!has("blocks.0.norm1.weight"));
    assert!(!has("blocks.0.norm3.weight"));

    // scale_shift_table is a bare Parameter, per block and once at top level.
    assert!(has("blocks.0.scale_shift_table"));
    assert!(has("scale_shift_table"));
    assert!(!has("scale_shift_table.weight"));

    // qk norms exist and are single-tensor (no bias).
    assert!(has("blocks.29.attn1.norm_q.weight"));
    assert!(!has("blocks.29.attn1.norm_q.bias"));

    // added_kv_proj_dim is null for TI2V-5B: no I2V image-conditioning path.
    assert!(!names.iter().any(|n| n.contains("add_k_proj")));

    assert!(!has("blocks.30.scale_shift_table"), "only 30 blocks, 0..=29");
}

#[test]
fn declared_tensor_sizes_match_the_checkpoint_shapes() {
    let c = cfg();
    let n = |s: &str| tensor_numel(&c, s).unwrap_or_else(|| panic!("no size for {s}"));

    // Shapes below are from the real checkpoint's config and module tree.
    // Conv3d patch embed over a 1x2x2 patch = 4 voxels.
    assert_eq!(n("patch_embedding.weight"), 3072 * 48 * 4);
    assert_eq!(n("blocks.0.attn1.to_q.weight"), 3072 * 3072);
    // "rms_norm_across_heads": one scale over the whole inner_dim, not per head.
    assert_eq!(n("blocks.0.attn1.norm_q.weight"), 3072);
    assert_ne!(n("blocks.0.attn1.norm_q.weight"), c.attention_head_dim);
    assert_eq!(n("blocks.0.ffn.net.0.proj.weight"), 14336 * 3072);
    assert_eq!(n("blocks.0.ffn.net.2.weight"), 3072 * 14336);
    assert_eq!(n("blocks.0.scale_shift_table"), 6 * 3072);
    assert_eq!(n("condition_embedder.time_proj.weight"), 6 * 3072 * 3072);
    assert_eq!(n("condition_embedder.text_embedder.linear_1.weight"), 3072 * 4096);
    assert_eq!(n("scale_shift_table"), 2 * 3072);
    assert_eq!(n("proj_out.weight"), 4 * 48 * 3072);

    // Every declared name must have a known size.
    for name in tensor_names(&c) {
        assert!(tensor_numel(&c, &name).is_some(), "no size rule for {name}");
    }
}

#[test]
fn parameter_count_is_about_five_billion() {
    // The checkpoint index reports total_size 19,999,150,848 bytes at F32,
    // i.e. ~5.0B parameters. Summing our declared tensors must land there.
    let c = cfg();
    let total: usize = tensor_names(&c).iter().map(|n| tensor_numel(&c, n).unwrap()).sum();
    let bytes = total * 4;
    let reported = 19_999_150_848_usize;
    let diff = bytes.abs_diff(reported) as f64 / reported as f64;
    assert!(
        diff < 0.001,
        "parameter total {total} ({bytes} bytes at f32) vs checkpoint {reported}: {:.3}% off",
        diff * 100.0
    );
}

// -- forward pass ---------------------------------------------------------

/// Small but structurally faithful: same block layout, same patch size, same
/// uneven RoPE split (head_dim 12 -> t/h/w = 4/4/4).
fn tiny() -> WanDitConfig {
    WanDitConfig::from_json(
        r#"{
            "num_layers": 2, "num_attention_heads": 2, "attention_head_dim": 12,
            "in_channels": 4, "out_channels": 4, "ffn_dim": 16,
            "text_dim": 8, "freq_dim": 8, "patch_size": [1, 2, 2],
            "eps": 1e-06, "qk_norm": "rms_norm_across_heads",
            "cross_attn_norm": true, "rope_max_seq_len": 64
        }"#,
    )
    .unwrap()
}

fn fill(v: &mut [f32], seed: f32) {
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i as f32 * 0.53 + seed).sin()) * 0.25;
    }
}

fn fill_linear(l: &mut crate::layers::Linear, seed: f32) {
    fill(&mut l.weight, seed);
    if let Some(b) = l.bias.as_mut() {
        fill(b, seed + 1.7);
    }
}

/// Give every parameter a deterministic non-zero value, so the tests exercise
/// real arithmetic rather than a model that outputs zeros.
fn randomize(m: &mut WanDit) {
    fill_linear(&mut m.patch_embedding, 0.1);
    fill_linear(&mut m.cond.time_linear_1, 0.2);
    fill_linear(&mut m.cond.time_linear_2, 0.3);
    fill_linear(&mut m.cond.time_proj, 0.4);
    fill_linear(&mut m.cond.text_linear_1, 0.5);
    fill_linear(&mut m.cond.text_linear_2, 0.6);
    for (i, b) in m.blocks.iter_mut().enumerate() {
        let s = i as f32 * 0.9;
        fill(&mut b.norm2.gamma, s + 0.7);
        fill(&mut b.norm2.beta, s + 0.8);
        for a in [&mut b.attn1, &mut b.attn2] {
            fill_linear(&mut a.to_q, s + 1.0);
            fill_linear(&mut a.to_k, s + 1.1);
            fill_linear(&mut a.to_v, s + 1.2);
            fill_linear(&mut a.to_out, s + 1.3);
            fill(&mut a.norm_q.weight, s + 1.4);
            fill(&mut a.norm_k.weight, s + 1.5);
        }
        fill_linear(&mut b.ffn_proj, s + 1.6);
        fill_linear(&mut b.ffn_out, s + 1.7);
        fill(&mut b.scale_shift_table, s + 1.8);
    }
    fill(&mut m.scale_shift_table, 2.0);
    fill_linear(&mut m.proj_out, 2.1);
}

fn tiny_inputs(c: &WanDitConfig) -> (Vec<f32>, Vec<f32>, usize) {
    let (t, h, w) = (2usize, 4usize, 4usize);
    let mut z = vec![0.0; c.in_channels * t * h * w];
    fill(&mut z, 3.0);
    let n_ctx = 3;
    let mut enc = vec![0.0; n_ctx * c.text_dim];
    fill(&mut enc, 4.0);
    (z, enc, n_ctx)
}

#[test]
fn forward_produces_the_input_shape() {
    let c = tiny();
    let mut m = WanDit::new(c.clone());
    randomize(&mut m);
    let (z, enc, n_ctx) = tiny_inputs(&c);
    let out = m.forward(&z, (2, 4, 4), &[0.5], &enc, n_ctx);
    assert_eq!(out.len(), c.out_channels * 2 * 4 * 4);
    assert!(out.iter().all(|v| v.is_finite()), "forward produced non-finite values");
    assert!(out.iter().any(|v| v.abs() > 1e-6), "forward produced all zeros");
}

#[test]
fn patchify_folds_channels_first_and_unpatchify_reads_channels_last() {
    // The two are deliberately *not* mirror images. patch_embedding is a
    // Conv3d with weight [dim, in_c, p_t, p_h, p_w], so its input is
    // channel-major within a patch. proj_out's output is reshaped
    // (p_t, p_h, p_w, out_c), i.e. channels innermost. Getting either
    // backwards scrambles patches without changing any shape.
    let c = tiny();
    let m = WanDit::new(c.clone());
    let (t, h, w) = (1usize, 2usize, 2usize);

    // One patch, 4 channels: value = channel index.
    let mut z = vec![0.0; 4 * t * h * w];
    for ci in 0..4 {
        for i in 0..(t * h * w) {
            z[ci * t * h * w + i] = ci as f32;
        }
    }
    let p = m.patchify(&z, 4, t, h, w);
    assert_eq!(p.len(), 16);
    // Channel-major: four copies of channel 0, then four of channel 1, ...
    assert_eq!(&p[0..4], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&p[4..8], &[1.0, 1.0, 1.0, 1.0]);

    // Channels innermost on the way out: [c0,c1,c2,c3] per voxel.
    let tokens: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let u = m.unpatchify(&tokens, t, h, w);
    assert_eq!(u.len(), 16);
    // Voxel (h=0,w=0) holds tokens[0..4] spread across the four channels.
    for ci in 0..4 {
        assert_eq!(u[ci * 4], ci as f32);
    }
}

#[test]
fn per_token_timesteps_reduce_to_per_sample_when_uniform() {
    // The TI2V path and the Wan2.1 path must agree when every token gets the
    // same timestep. If they disagree, one of the two modulation branches is
    // indexing its table wrongly — and only the per-token one is exercised by
    // real image-to-video.
    let c = tiny();
    let mut m = WanDit::new(c.clone());
    randomize(&mut m);
    let (z, enc, n_ctx) = tiny_inputs(&c);

    let seq = 2 * (4 / 2) * (4 / 2);
    let a = m.forward(&z, (2, 4, 4), &[0.37], &enc, n_ctx);
    let b = m.forward(&z, (2, 4, 4), &vec![0.37_f32; seq], &enc, n_ctx);

    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
    assert!(worst < 1e-5, "per-token and per-sample disagree at uniform t: {worst}");
}

#[test]
fn per_token_timesteps_actually_change_the_output() {
    // Holding some tokens at t=0 while denoising the rest is the whole point
    // of the per-token path; it must not be silently ignored.
    let c = tiny();
    let mut m = WanDit::new(c.clone());
    randomize(&mut m);
    let (z, enc, n_ctx) = tiny_inputs(&c);
    let seq = 2 * 2 * 2;

    let uniform = m.forward(&z, (2, 4, 4), &vec![0.37_f32; seq], &enc, n_ctx);
    // First frame's tokens pinned at 0, as an image-conditioned run would.
    let mut mixed = vec![0.37_f32; seq];
    for t in mixed.iter_mut().take(seq / 2) {
        *t = 0.0;
    }
    let got = m.forward(&z, (2, 4, 4), &mixed, &enc, n_ctx);

    let worst = uniform.iter().zip(&got).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
    assert!(worst > 1e-4, "per-token timesteps had no effect: {worst}");
    assert!(got.iter().all(|v| v.is_finite()));
}

#[test]
fn rope_makes_token_position_matter() {
    // Swapping two tokens' latent content must change the result: without
    // position embeddings self-attention is permutation-equivariant and the
    // output would just permute with it.
    let c = tiny();
    let mut m = WanDit::new(c.clone());
    randomize(&mut m);
    let (z, enc, n_ctx) = tiny_inputs(&c);
    let base = m.forward(&z, (2, 4, 4), &[0.5], &enc, n_ctx);

    // Swap the two temporal halves of the latent.
    let plane = 4 * 4;
    let mut swapped = z.clone();
    for ci in 0..c.in_channels {
        for i in 0..plane {
            let a = (ci * 2) * plane + i;
            let b = (ci * 2 + 1) * plane + i;
            swapped.swap(a, b);
        }
    }
    let got = m.forward(&swapped, (2, 4, 4), &[0.5], &enc, n_ctx);
    let worst = base.iter().zip(&got).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
    assert!(worst > 1e-4, "position appears not to affect the output: {worst}");
}

#[test]
fn text_conditioning_reaches_the_output_through_cross_attention() {
    // Two parts, because a small non-zero difference alone could be noise:
    // changing the text states must move the output, AND severing the
    // cross-attention output projection must stop it moving. The second half
    // is what proves the sensitivity travels through attn2 specifically.
    let c = tiny();
    let (z, enc_a, n_ctx) = tiny_inputs(&c);
    let mut enc_b = enc_a.clone();
    fill(&mut enc_b, 9.0);
    let worst = |m: &WanDit| {
        let a = m.forward(&z, (2, 4, 4), &[0.5], &enc_a, n_ctx);
        let b = m.forward(&z, (2, 4, 4), &[0.5], &enc_b, n_ctx);
        a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    };

    let mut m = WanDit::new(c.clone());
    randomize(&mut m);
    let wired = worst(&m);
    assert!(wired > 1e-5, "text states do not reach the output: {wired}");

    // Sever cross-attention: its output projection now contributes nothing.
    for b in m.blocks.iter_mut() {
        b.attn2.to_out.weight.iter_mut().for_each(|v| *v = 0.0);
        if let Some(bias) = b.attn2.to_out.bias.as_mut() {
            bias.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    let severed = worst(&m);
    assert_eq!(severed, 0.0, "text still reaches the output with attn2 severed: {severed}");
}

#[test]
fn forward_is_deterministic() {
    let c = tiny();
    let mut m = WanDit::new(c.clone());
    randomize(&mut m);
    let (z, enc, n_ctx) = tiny_inputs(&c);
    let a = m.forward(&z, (2, 4, 4), &[0.5], &enc, n_ctx);
    let b = m.forward(&z, (2, 4, 4), &[0.5], &enc, n_ctx);
    assert_eq!(a, b);
}

// -- weight loading -------------------------------------------------------

use crate::weight::loader::ComponentTensors;

/// Build a synthetic checkpoint where every tensor is filled with a value
/// derived from its own name, so cross-wiring is detectable: if the loader

/// Conditioned frames must come out of the loop bit-identical to the reference.
///
/// This is the property the whole feature rests on. `expand_timesteps` tells
/// the model those tokens are already clean; if the scheduler is allowed to
/// move them anyway, the model is being lied to and the output's first frame is
/// not the image the user supplied. The bug is invisible without this test —

/// Round-trip a real image through conditioning and back out of the VAE.
///
/// This is the one that catches a wrong normalisation direction or a
/// misplaced frame — both of which pass every shape and mask test above while
/// putting something that is not the user's image into the held slot. It walks
/// the full chain: encode -> take the distribution mean -> normalise -> place at
/// a latent frame -> denormalise -> decode, and asks whether pixel frame 0 is

/// Our conditioning latent must equal the one diffusers would hand the
/// transformer for the same image.
///
/// The encoder is already proven exact, so what is left is the two steps after
/// it: taking the distribution *mean* rather than a sample, and normalising in
/// the right direction. An inverted normalisation passes every shape and mask
/// test in this file while putting something that is not the user's image into
/// the held slot — and the video still generates, so nothing complains.
///
/// Compared against the reference latent rather than by decoding: klearu's CPU
/// VAE decode is far too slow to sit in a test (the GPU decoder exists for
/// exactly that reason), and this is the stricter check anyway.
#[test]
#[ignore]
fn conditioning_latent_matches_diffusers() {
    use super::pipeline::WanConditioning;
    use crate::vae3d::{Vol, WanVae, WanVaeConfig};
    use crate::weight::ComponentTensors;

    let (Ok(dir), Ok(ref_path)) =
        (std::env::var("KLEARU_WAN_VAE"), std::env::var("KLEARU_COND_REF"))
    else {
        eprintln!("set KLEARU_WAN_VAE and KLEARU_COND_REF; skipping");
        return;
    };
    let ref_dir = std::path::Path::new(&ref_path).parent().unwrap().to_path_buf();
    let r = ComponentTensors::open_dir(&ref_dir).expect("open reference");
    let shape = r.get_f32("shape").expect("shape");
    let (pf, ph, pw) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    assert_eq!(pf, 1, "conditioning reference must be a single frame");
    let px = r.get_f32("in.pixels").expect("in.pixels");
    let want = r.get_f32("out.normalized").expect("out.normalized");

    let dir = std::path::PathBuf::from(dir);
    let cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    let comp = ComponentTensors::open_dir(&dir).expect("open checkpoint");
    let vae = WanVae::load(cfg.clone(), &comp).expect("load real weights");

    let sc = cfg.spatial_compression();
    let (lt, lh, lw) = (3usize, ph / sc, pw / sc);
    let img = Vol::new(px, 3, 1, ph, pw);
    let cond = WanConditioning::from_images(&vae, (lt, lh, lw), Some(&img), None);

    // Exactly frame 0 is held.
    let fp = lh * lw;
    assert!(cond.mask[..fp].iter().all(|m| *m == 0.0), "frame 0 must be held");
    assert!(cond.mask[fp..].iter().all(|m| *m == 1.0), "only frame 0 may be held");

    // Pull frame 0 back out of the [z_dim, T, H, W] layout.
    let plane = lt * fp;
    let mut got = vec![0.0_f32; cfg.z_dim * fp];
    for c in 0..cfg.z_dim {
        got[c * fp..(c + 1) * fp]
            .copy_from_slice(&cond.latents[c * plane..c * plane + fp]);
    }
    assert_eq!(got.len(), want.len(), "conditioning latent size");

    let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let dot: f64 = got.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb + 1e-12);
    println!("conditioning latent: cos_sim={cos:.8} max_abs_diff={worst:.4e}");
    assert!(cos > 0.9999, "conditioning latent diverges from diffusers: {cos}");

    // Unconditioned frames must be left at zero, not filled with the image.
    let rest: f32 =
        (0..cfg.z_dim).map(|c| cond.latents[c * plane + fp..(c + 1) * plane]
            .iter().map(|v| v.abs()).fold(0.0, f32::max)).fold(0.0, f32::max);
    assert_eq!(rest, 0.0, "frames that are not held must stay zero");
}

/// the video still generates and still looks influenced by the image.
#[test]
fn conditioned_frames_survive_the_scheduler() {
    use super::pipeline::{apply_conditioning, WanConditioning};

    let (t, h, w) = (3usize, 4usize, 4usize);
    let z = 6usize;
    let plane = t * h * w;
    let fp = h * w;

    let mut mask = vec![1.0_f32; plane];
    for m in mask[..fp].iter_mut() {
        *m = 0.0; // first frame held
    }
    for m in mask[(t - 1) * fp..].iter_mut() {
        *m = 0.0; // last frame held
    }
    let cond = WanConditioning {
        latents: (0..z * plane).map(|i| (i as f32) * 0.01 - 1.0).collect(),
        mask,
    };

    // Whatever the scheduler produced this step.
    let mut latents: Vec<f32> = (0..z * plane).map(|i| ((i * 7) % 13) as f32).collect();
    let before = latents.clone();
    apply_conditioning(&mut latents, &cond, z);

    for c in 0..z {
        for i in 0..plane {
            let k = c * plane + i;
            if cond.mask[i] == 0.0 {
                assert_eq!(latents[k], cond.latents[k], "held position {i} channel {c}");
            } else {
                assert_eq!(latents[k], before[k], "free position {i} must be untouched");
            }
        }
    }
}

/// Holding a frame must not disturb the frames around it.
#[test]
fn conditioning_only_touches_masked_frames() {
    use super::pipeline::{apply_conditioning, WanConditioning};

    let (t, h, w) = (4usize, 2usize, 2usize);
    let z = 3usize;
    let plane = t * h * w;
    let fp = h * w;
    let mut mask = vec![1.0_f32; plane];
    for m in mask[fp..2 * fp].iter_mut() {
        *m = 0.0; // only frame 1
    }
    let cond = WanConditioning { latents: vec![9.0; z * plane], mask };
    let mut latents = vec![1.0_f32; z * plane];
    apply_conditioning(&mut latents, &cond, z);

    for c in 0..z {
        for f in 0..t {
            for p in 0..fp {
                let v = latents[c * plane + f * fp + p];
                let want = if f == 1 { 9.0 } else { 1.0 };
                assert_eq!(v, want, "channel {c} frame {f} pos {p}");
            }
        }
    }
}

/// A held frame's tokens must get timestep 0 while the rest denoise.
///
/// The mask and the pinned latents are two halves of one contract; this checks
/// the half that reaches the model, at both ends at once.
#[test]
fn start_and_end_conditioning_zero_both_ends_of_the_timesteps() {
    let dims = (4usize, 4usize, 4usize);
    let fp = dims.1 * dims.2;
    let mut mask = vec![1.0_f32; dims.0 * fp];
    for m in mask[..fp].iter_mut() {
        *m = 0.0;
    }
    for m in mask[3 * fp..].iter_mut() {
        *m = 0.0;
    }
    let ts = expand_timesteps(&mask, dims, (1, 2, 2), 700.0);
    let per_frame = ts.len() / dims.0;
    assert!(ts[..per_frame].iter().all(|t| *t == 0.0), "start frame held");
    assert!(ts[3 * per_frame..].iter().all(|t| *t == 0.0), "end frame held");
    assert!(
        ts[per_frame..3 * per_frame].iter().all(|t| (*t - 700.0).abs() < 1e-6),
        "middle frames must still denoise"
    );
}


/// The CFG band must cover every step by default and drop the right ones when
/// narrowed — this is what decides how many forwards a generation costs.
#[test]
fn cfg_interval_selects_the_expected_steps() {
    use super::pipeline::WanGenConfig;
    let base = WanGenConfig { num_steps: 10, flow_shift: 5.0, guidance_scale: 5.0,
                              cfg_interval: (0.0, 1.0) };
    // Default: guidance on every step, so two forwards each.
    assert!((0..10).all(|i| base.cfg_applies_at(i, 10)));
    assert_eq!(base.forward_count(), 20);

    // A middle band: steps 2..8 of 10.
    let mid = WanGenConfig { cfg_interval: (0.2, 0.8), ..base.clone() };
    let on: Vec<usize> = (0..10).filter(|i| mid.cfg_applies_at(*i, 10)).collect();
    assert_eq!(on, vec![2, 3, 4, 5, 6, 7], "band should be half-open on the right");
    assert_eq!(mid.forward_count(), 16);

    // Guidance off entirely costs one forward a step regardless of the band.
    let none = WanGenConfig { guidance_scale: 1.0, ..base.clone() };
    assert_eq!(none.forward_count(), 10);

    // An empty band degenerates to no guidance rather than panicking.
    let empty = WanGenConfig { cfg_interval: (0.5, 0.5), ..base.clone() };
    assert!(!(0..10).any(|i| empty.cfg_applies_at(i, 10)));
    assert_eq!(empty.forward_count(), 10);
}

/// puts `to_k` into `to_q`, the constant found there is the wrong one.
fn synthetic_checkpoint(c: &WanDitConfig) -> ComponentTensors {
    fn tag(name: &str) -> f32 {
        // Small distinct value per name; stable and collision-free enough.
        let mut h: u32 = 2_166_136_261;
        for b in name.bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(16_777_619);
        }
        (h % 100_000) as f32 / 100_000.0 + 0.1
    }
    let mut comp = ComponentTensors {
        mmaps: Vec::new(),
        owned: Vec::new(),
        tensors: std::collections::HashMap::new(),
    };
    for name in tensor_names(c) {
        let n = tensor_numel(c, &name).unwrap();
        let bytes: Vec<u8> = std::iter::repeat_n(tag(&name).to_le_bytes(), n).flatten().collect();
        comp.append_synth_tensor(&name, safetensors::Dtype::F32, vec![n], &bytes).unwrap();
    }
    comp
}

#[test]
fn loads_every_tensor_into_the_right_place() {
    let c = tiny();
    let comp = synthetic_checkpoint(&c);
    let m = WanDit::load(c.clone(), &comp).unwrap();

    // Same tagging function as the checkpoint builder.
    let tag = |name: &str| -> f32 {
        let mut h: u32 = 2_166_136_261;
        for b in name.bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(16_777_619);
        }
        (h % 100_000) as f32 / 100_000.0 + 0.1
    };
    let all = |v: &[f32], name: &str| {
        assert!(!v.is_empty(), "{name} loaded empty");
        assert!(
            v.iter().all(|x| (*x - tag(name)).abs() < 1e-6),
            "{name} did not receive its own values (cross-wired?)"
        );
    };

    all(&m.patch_embedding.weight, "patch_embedding.weight");
    all(&m.proj_out.weight, "proj_out.weight");
    all(&m.scale_shift_table, "scale_shift_table");
    all(&m.cond.time_proj.weight, "condition_embedder.time_proj.weight");
    all(&m.cond.text_linear_1.weight, "condition_embedder.text_embedder.linear_1.weight");

    for (i, b) in m.blocks.iter().enumerate() {
        all(&b.scale_shift_table, &format!("blocks.{i}.scale_shift_table"));
        // q/k/v must not be swapped with one another.
        all(&b.attn1.to_q.weight, &format!("blocks.{i}.attn1.to_q.weight"));
        all(&b.attn1.to_k.weight, &format!("blocks.{i}.attn1.to_k.weight"));
        all(&b.attn1.to_v.weight, &format!("blocks.{i}.attn1.to_v.weight"));
        all(&b.attn1.to_out.weight, &format!("blocks.{i}.attn1.to_out.0.weight"));
        all(&b.attn1.norm_q.weight, &format!("blocks.{i}.attn1.norm_q.weight"));
        // self- and cross-attention must not be swapped.
        all(&b.attn2.to_q.weight, &format!("blocks.{i}.attn2.to_q.weight"));
        // norm2 is the only norm with weights; gamma and beta must not swap.
        all(&b.norm2.gamma, &format!("blocks.{i}.norm2.weight"));
        all(&b.norm2.beta, &format!("blocks.{i}.norm2.bias"));
        all(&b.ffn_proj.weight, &format!("blocks.{i}.ffn.net.0.proj.weight"));
        all(&b.ffn_out.weight, &format!("blocks.{i}.ffn.net.2.weight"));
    }

    // norm1 and norm3 have no checkpoint tensors and must stay identity.
    for b in &m.blocks {
        assert!(b.norm1.gamma.iter().all(|v| *v == 1.0), "norm1 must have no affine");
        assert!(b.norm1.beta.iter().all(|v| *v == 0.0));
        assert!(b.norm3.gamma.iter().all(|v| *v == 1.0), "norm3 must have no affine");
    }
    assert!(m.norm_out.gamma.iter().all(|v| *v == 1.0), "norm_out must have no affine");
}

#[test]
fn a_loaded_model_runs() {
    let c = tiny();
    let comp = synthetic_checkpoint(&c);
    let m = WanDit::load(c.clone(), &comp).unwrap();
    let (z, enc, n_ctx) = tiny_inputs(&c);
    let out = m.forward(&z, (2, 4, 4), &[0.5], &enc, n_ctx);
    assert_eq!(out.len(), c.out_channels * 2 * 4 * 4);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn missing_and_mis_sized_tensors_are_reported_up_front() {
    let c = tiny();

    let empty = ComponentTensors {
        mmaps: Vec::new(),
        owned: Vec::new(),
        tensors: std::collections::HashMap::new(),
    };
    match WanDit::load(c.clone(), &empty) {
        Ok(_) => panic!("empty checkpoint should not load"),
        Err(e) => assert!(format!("{e}").contains("missing"), "{e}"),
    }

    // A checkpoint with every name present but one tensor the wrong size
    // must be rejected before any weights are copied, not partway through.
    let mut comp = synthetic_checkpoint(&c);
    let name = "blocks.0.attn1.to_q.weight";
    comp.tensors.remove(name);
    comp.append_synth_tensor(name, safetensors::Dtype::F32, vec![7], &[0u8; 28]).unwrap();
    let m = WanDit::new(c.clone());
    assert_eq!(m.mismatched_tensors(&comp), vec![name.to_string()]);
    match WanDit::load(c, &comp) {
        Ok(_) => panic!("mis-sized tensor should not load"),
        Err(e) => assert!(format!("{e}").contains("unexpected size"), "{e}"),
    }
}

// -- pipeline -------------------------------------------------------------

use super::pipeline::{
    apply_cfg, denormalize_latents, expand_timesteps, normalize_latents, wan_sigmas,
    WanGenConfig, WanPipeline,
};

#[test]
fn wan_sigma_schedule_matches_the_reference_shape() {
    let s = wan_sigmas(8, 5.0);
    assert_eq!(s.len(), 9, "steps + 1, with a terminal zero");
    // linspace starts at 1.0 and the shift maps 1.0 -> 1.0 for any shift, but
    // the first sigma must land *just below* 1: UniPC's
    // lambda(s) = ln(1-s) - ln(s) is -inf at exactly 1.0, which NaNs the first
    // two solver updates. diffusers nudges it by 1e-6 for the same reason.
    assert!(s[0] < 1.0, "first sigma must be strictly below 1, got {}", s[0]);
    assert!((s[0] - 1.0).abs() < 1e-5, "but only just: {}", s[0]);
    assert!((1.0_f32 - s[0]).ln().is_finite(), "ln(1 - sigma_0) must be finite");
    assert_eq!(*s.last().unwrap(), 0.0, "final_sigmas_type is zero");
    for w in s.windows(2) {
        assert!(w[0] > w[1], "sigmas must decrease monotonically: {w:?}");
    }
    // The shift pushes mass toward high noise: with shift > 1 every interior
    // sigma sits above the unshifted linspace value.
    let unshifted = wan_sigmas(8, 1.0);
    for i in 1..8 {
        assert!(s[i] > unshifted[i], "shift 5 should raise sigma {i}");
    }
}

#[test]
fn timestep_expansion_is_uniform_for_text_to_video() {
    // An all-ones mask gives every token the same timestep — the Wan2.1
    // behaviour, reached through the per-token path.
    let dims = (2usize, 4usize, 4usize);
    let mask = vec![1.0_f32; dims.0 * dims.1 * dims.2];
    let ts = expand_timesteps(&mask, dims, (1, 2, 2), 700.0);
    assert_eq!(ts.len(), 2 * 2 * 2, "one entry per token");
    assert!(ts.iter().all(|t| (*t - 700.0).abs() < 1e-6));
}

#[test]
fn timestep_expansion_zeroes_conditioned_frames_for_image_to_video() {
    // Frame 0 held as conditioning: its tokens get timestep 0 while the rest
    // denoise at t. This is what the per-token modulation exists for.
    let dims = (2usize, 4usize, 4usize);
    let mut mask = vec![1.0_f32; dims.0 * dims.1 * dims.2];
    for v in mask[..dims.1 * dims.2].iter_mut() {
        *v = 0.0;
    }
    let ts = expand_timesteps(&mask, dims, (1, 2, 2), 700.0);
    assert_eq!(ts.len(), 8);
    assert!(ts[..4].iter().all(|t| *t == 0.0), "frame 0 tokens must be held at 0");
    assert!(ts[4..].iter().all(|t| (*t - 700.0).abs() < 1e-6));
}

#[test]
fn latent_normalisation_round_trips() {
    let c = 3;
    let mean = [0.5_f32, -1.0, 2.0];
    let std = [2.0_f32, 0.5, 1.5];
    let orig: Vec<f32> = (0..12).map(|i| i as f32 * 0.3 - 1.0).collect();

    let mut z = orig.clone();
    normalize_latents(&mut z, c, &mean, &std);
    assert_ne!(z, orig);
    denormalize_latents(&mut z, c, &mean, &std);
    for (a, b) in z.iter().zip(&orig) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }

    // Denormalise is per channel, so channel 0's scale must not leak.
    let mut z = vec![1.0_f32; 12];
    denormalize_latents(&mut z, c, &mean, &std);
    assert!((z[0] - (1.0 * 2.0 + 0.5)).abs() < 1e-6);
    assert!((z[4] - (1.0 * 0.5 - 1.0)).abs() < 1e-6);
    assert!((z[8] - (1.0 * 1.5 + 2.0)).abs() < 1e-6);
}

#[test]
fn cfg_interpolates_between_the_two_predictions() {
    let uncond = [1.0_f32, 2.0, 3.0];
    let base = [2.0_f32, 4.0, 6.0];

    let mut c = base;
    apply_cfg(&mut c, &uncond, 1.0);
    assert_eq!(c, base, "scale 1 returns the conditional prediction");

    let mut c = base;
    apply_cfg(&mut c, &uncond, 0.0);
    assert_eq!(c, uncond, "scale 0 returns the unconditional prediction");

    let mut c = base;
    apply_cfg(&mut c, &uncond, 5.0);
    assert_eq!(c, [6.0, 12.0, 18.0], "uncond + 5 * (cond - uncond)");
}

/// End to end on miniature models: noise -> DiT denoising loop -> VAE decode.
/// Proves the three components actually compose — matching channel counts,
/// latent layout, frame arithmetic and token ordering — which no single
/// component test can show.
#[test]
fn tiny_pipeline_runs_end_to_end() {
    use crate::vae3d::{WanVae, WanVaeConfig};

    let dit_cfg = tiny(); // in/out channels 4, patch [1,2,2]
    let mut dit = WanDit::new(dit_cfg.clone());
    randomize(&mut dit);

    let vae_cfg = WanVaeConfig::from_json(
        r#"{
            "base_dim": 4, "decoder_base_dim": 4, "z_dim": 4,
            "dim_mult": [1, 2, 4, 4], "num_res_blocks": 1,
            "attn_scales": [], "temperal_downsample": [false, true, true],
            "in_channels": 12, "out_channels": 12, "patch_size": 2,
            "is_residual": true
        }"#,
    )
    .unwrap();
    assert_eq!(
        vae_cfg.z_dim, dit_cfg.in_channels,
        "the DiT works in the VAE's latent space"
    );
    let vae = WanVae::new(vae_cfg);
    let pipe = WanPipeline::new(dit, vae);

    let dims = (2usize, 4usize, 4usize);
    let n = 4 * dims.0 * dims.1 * dims.2;
    let mut latents = vec![0.0_f32; n];
    fill(&mut latents, 5.0);

    let n_ctx = 3;
    let mut prompt = vec![0.0_f32; n_ctx * dit_cfg.text_dim];
    fill(&mut prompt, 6.0);
    let mut negative = vec![0.0_f32; n_ctx * dit_cfg.text_dim];
    fill(&mut negative, 7.0);

    let cfg = WanGenConfig { num_steps: 3, flow_shift: 5.0, guidance_scale: 5.0 , cfg_interval: (0.0, 1.0) };
    let mask = vec![1.0_f32; dims.0 * dims.1 * dims.2];

    let mut seen = 0usize;
    let out_latents = pipe.denoise(
        latents.clone(),
        dims,
        &prompt,
        Some(&negative),
        n_ctx,
        &mask,
        None,
        &cfg,
        |i: usize, total: usize, latents: &[f32]| {
            assert_eq!(total, 3);
            assert_eq!(latents.len(), n, "callback should see the stepped latents");
            seen = i;
        },
    );
    assert_eq!(seen, 3, "callback should fire once per step");
    assert_eq!(out_latents.len(), n);
    assert!(out_latents.iter().all(|v| v.is_finite()));
    assert!(
        out_latents.iter().zip(&latents).any(|(a, b)| (a - b).abs() > 1e-6),
        "denoising should move the latents"
    );

    let video = pipe.decode(&out_latents, dims);
    // 2 latent frames -> 1 + 4 = 5 output frames; 4 latent px -> 4*16 = 64.
    assert_eq!((video.c, video.t, video.h, video.w), (3, 5, 64, 64));
    assert!(video.data.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
}

#[test]
fn guidance_scale_of_one_skips_the_negative_pass() {
    // A cheap but real saving: at scale <= 1 the unconditional forward is
    // pure waste, and half the compute per step.
    let dit_cfg = tiny();
    let mut dit = WanDit::new(dit_cfg.clone());
    randomize(&mut dit);
    let vae = crate::vae3d::WanVae::new(
        crate::vae3d::WanVaeConfig::from_json(
            r#"{"base_dim":4,"decoder_base_dim":4,"z_dim":4,"dim_mult":[1,2,4,4],
                "num_res_blocks":1,"attn_scales":[],
                "temperal_downsample":[false,true,true],"in_channels":12,
                "out_channels":12,"patch_size":2,"is_residual":true}"#,
        )
        .unwrap(),
    );
    let pipe = WanPipeline::new(dit, vae);

    let dims = (2usize, 4usize, 4usize);
    let mut latents = vec![0.0_f32; 4 * dims.0 * dims.1 * dims.2];
    fill(&mut latents, 5.0);
    let n_ctx = 3;
    let mut prompt = vec![0.0_f32; n_ctx * dit_cfg.text_dim];
    fill(&mut prompt, 6.0);
    let mut negative = vec![0.0_f32; n_ctx * dit_cfg.text_dim];
    fill(&mut negative, 7.0);
    let mask = vec![1.0_f32; dims.0 * dims.1 * dims.2];

    let cfg = WanGenConfig { num_steps: 2, flow_shift: 5.0, guidance_scale: 1.0 , cfg_interval: (0.0, 1.0) };
    let with_neg =
        pipe.denoise(latents.clone(), dims, &prompt, Some(&negative), n_ctx, &mask, None, &cfg, |_, _, _| {});
    let without =
        pipe.denoise(latents, dims, &prompt, None, n_ctx, &mask, None, &cfg, |_, _, _| {});
    assert_eq!(with_neg, without, "scale 1 must ignore the negative prompt");
}

// -- numerical reference comparison ---------------------------------------

/// Compare the DiT against a diffusers reference, weight for weight.
///
/// The real checkpoint is 20 GB, so instead we build a *miniature* model with
/// the same structure in both implementations, give it identical weights, and
/// require identical outputs. Every code path the full model uses — 3D RoPE,
/// per-token modulation, cross-attention, the gated FFN, patchify/unpatchify —
/// is exercised; only the widths shrink.
///
/// Generate the reference first:
///   /tmp/klearu_ref_venv/bin/python scripts/dump_wan_reference.py \
///       --out wan_reference.safetensors
/// then:
///   KLEARU_WAN_REF=wan_reference.safetensors \
///   cargo test -p klearu-diffusion --lib wan_reference -- --ignored --nocapture
#[test]
#[ignore]
fn wan_reference_matches_diffusers() {
    use crate::weight::loader::ComponentTensors;

    let path = std::env::var("KLEARU_WAN_REF")
        .unwrap_or_else(|_| "wan_reference.safetensors".to_string());
    let dir = std::path::Path::new(&path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    assert!(
        std::path::Path::new(&path).exists(),
        "reference not found at {path}; run scripts/dump_wan_reference.py first"
    );

    // ComponentTensors::open_dir reads every .safetensors in a directory, so
    // point it at the file's parent.
    let comp = ComponentTensors::open_dir(&dir).expect("open reference");

    let c = tiny();
    let m = WanDit::load(c.clone(), &comp).expect("load reference weights");

    let latent = comp.get_f32("in.latent").unwrap();
    let enc = comp.get_f32("in.encoder_hidden_states").unwrap();
    let n_ctx = enc.len() / c.text_dim;
    let dims = (2usize, 4usize, 4usize);

    let report = |label: &str, got: &[f32], want: &[f32]| -> f32 {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        let dot: f64 = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb + 1e-12);
        println!("{label:24} max_abs_diff={worst:.3e}  cos_sim={cos:.9}");
        worst
    };

    // Per-sample timestep (Wan 2.1 / 14B modulation path).
    let t_scalar = comp.get_f32("in.timestep_scalar").unwrap();
    let got = m.forward(&latent, dims, &t_scalar, &enc, n_ctx);
    let want = comp.get_f32("out.per_sample").unwrap();
    let d1 = report("per_sample", &got, &want);

    // Per-token timestep (Wan 2.2 TI2V path), frame 0 held at t=0.
    let ts = comp.get_f32("in.timestep_per_token").unwrap();
    let got = m.forward(&latent, dims, &ts, &enc, n_ctx);
    let want = comp.get_f32("out.per_token").unwrap();
    let d2 = report("per_token", &got, &want);

    // Uniform per-token, which the reference itself says should match
    // per-sample to ~5e-7.
    let seq = 2 * 2 * 2;
    let got = m.forward(&latent, dims, &vec![700.0_f32; seq], &enc, n_ctx);
    let want = comp.get_f32("out.per_token_uniform").unwrap();
    let d3 = report("per_token_uniform", &got, &want);

    let tol = 2e-4;
    assert!(d1 < tol, "per_sample diverged: {d1}");
    assert!(d2 < tol, "per_token diverged: {d2}");
    assert!(d3 < tol, "per_token_uniform diverged: {d3}");
}

// -- GPU path -------------------------------------------------------------

/// The GPU DiT must agree with the CPU one, which is itself pinned against
/// diffusers to ~5e-7 by `wan_reference_matches_diffusers`. So this is a real
/// guarantee, not two implementations agreeing on the same mistake.
///
/// Tolerance is set by bf16 storage (~3 decimal digits), not by the algorithm.
#[cfg(feature = "metal")]
#[test]
fn gpu_dit_matches_cpu() {
    use super::gpu::GpuWanDit;

    let c = tiny();
    let mut cpu = WanDit::new(c.clone());
    randomize(&mut cpu);
    let gpu = GpuWanDit::from_cpu(&cpu);

    let (z, enc, n_ctx) = tiny_inputs(&c);
    let dims = (2usize, 4usize, 4usize);
    let seq = 2 * 2 * 2;

    let compare = |label: &str, ts: &[f32]| {
        let want = cpu.forward(&z, dims, ts, &enc, n_ctx);
        let got = gpu.forward(&z, dims, ts, &enc, n_ctx);
        assert_eq!(got.len(), want.len());
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        let dot: f64 = got.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb + 1e-12);
        println!("{label:20} max_abs_diff={worst:.4e}  cos_sim={cos:.6}");
        assert!(got.iter().all(|v| v.is_finite()), "{label}: non-finite output");
        assert!(cos > 0.999, "{label}: cos_sim {cos} too low");
        worst
    };

    // Per-sample and per-token modulation take different kernel paths.
    compare("per_sample", &[700.0]);
    compare("per_token_uniform", &vec![700.0_f32; seq]);
    let mut mixed = vec![700.0_f32; seq];
    for t in mixed.iter_mut().take(seq / 2) {
        *t = 0.0;
    }
    compare("per_token_mixed", &mixed);
}

/// The GPU path must show the same per-token sensitivity as the CPU one —
/// a wrong modulation row stride would make every token identical while
/// still producing finite, plausible output.
#[cfg(feature = "metal")]
#[test]
fn gpu_dit_respects_per_token_timesteps() {
    use super::gpu::GpuWanDit;

    let c = tiny();
    let mut cpu = WanDit::new(c.clone());
    randomize(&mut cpu);
    let gpu = GpuWanDit::from_cpu(&cpu);
    let (z, enc, n_ctx) = tiny_inputs(&c);
    let dims = (2usize, 4usize, 4usize);
    let seq = 8;

    let uniform = gpu.forward(&z, dims, &vec![700.0_f32; seq], &enc, n_ctx);
    let mut mixed = vec![700.0_f32; seq];
    for t in mixed.iter_mut().take(seq / 2) {
        *t = 0.0;
    }
    let got = gpu.forward(&z, dims, &mixed, &enc, n_ctx);
    let worst = uniform.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(worst > 1e-3, "GPU ignored per-token timesteps: {worst}");
}

/// The full GPU pipeline against the full CPU pipeline: noise -> denoising
/// loop -> denormalise -> chunked VAE decode -> pixels.
///
/// The closing composition test. The CPU pipeline it is measured against runs
/// on a DiT and VAE that are each pinned to diffusers, so this transitively
/// ties the GPU output back to the reference implementation.
#[cfg(feature = "metal")]
#[test]
fn gpu_pipeline_matches_cpu_end_to_end() {
    use super::pipeline::GpuWanPipeline;
    use crate::vae3d::{WanVae, WanVaeConfig};

    let dit_cfg = tiny();
    let mut dit = WanDit::new(dit_cfg.clone());
    randomize(&mut dit);

    let vae_cfg = WanVaeConfig::from_json(
        r#"{"base_dim":4,"decoder_base_dim":4,"z_dim":4,"dim_mult":[1,2,4,4],
            "num_res_blocks":1,"attn_scales":[],
            "temperal_downsample":[false,true,true],"in_channels":12,
            "out_channels":12,"patch_size":2,"is_residual":true}"#,
    )
    .unwrap();

    // Fill EVERY VAE parameter. Leaving any conv at zero annihilates the
    // signal — with `post_quant_conv` zeroed, both paths decode all-zero
    // video and compare "equal" while testing nothing.
    let mut vae = WanVae::new(vae_cfg);
    let mut seed = 200.0_f32;
    let fill_v = |v: &mut [f32], s: &mut f32| {
        *s += 0.41;
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f32) * 0.19 + *s).sin() * 0.25;
        }
    };
    fill_v(&mut vae.post_quant_conv.weight, &mut seed);
    if let Some(b) = vae.post_quant_conv.bias.as_mut() {
        fill_v(b, &mut seed);
    }
    fill_v(&mut vae.decoder.conv_in.weight, &mut seed);
    if let Some(b) = vae.decoder.conv_in.bias.as_mut() {
        fill_v(b, &mut seed);
    }
    fill_v(&mut vae.decoder.norm_out.gamma, &mut seed);
    fill_v(&mut vae.decoder.conv_out.weight, &mut seed);
    if let Some(b) = vae.decoder.conv_out.bias.as_mut() {
        fill_v(b, &mut seed);
    }
    {
        let fill_res = |r: &mut crate::vae3d::blocks::ResidualBlock, s: &mut f32| {
            fill_v(&mut r.norm1.gamma, s);
            fill_v(&mut r.norm2.gamma, s);
            fill_v(&mut r.conv1.weight, s);
            fill_v(&mut r.conv2.weight, s);
            if let Some(b) = r.conv1.bias.as_mut() {
                fill_v(b, s);
            }
            if let Some(b) = r.conv2.bias.as_mut() {
                fill_v(b, s);
            }
            if let Some(sc) = r.shortcut.as_mut() {
                fill_v(&mut sc.weight, s);
                if let Some(b) = sc.bias.as_mut() {
                    fill_v(b, s);
                }
            }
        };
        for r in vae.decoder.mid_block.resnets.iter_mut() {
            fill_res(r, &mut seed);
        }
        for a in vae.decoder.mid_block.attentions.iter_mut() {
            fill_v(&mut a.norm.gamma, &mut seed);
            fill_v(&mut a.to_qkv.weight, &mut seed);
            fill_v(&mut a.proj.weight, &mut seed);
        }
        for b in vae.decoder.up_blocks.iter_mut() {
            for r in b.resnets.iter_mut() {
                fill_res(r, &mut seed);
            }
            if let Some(u) = b.upsampler.as_mut() {
                fill_v(&mut u.conv.weight, &mut seed);
                if let Some(bi) = u.conv.bias.as_mut() {
                    fill_v(bi, &mut seed);
                }
                if let Some(tc) = u.time_conv.as_mut() {
                    fill_v(&mut tc.weight, &mut seed);
                    if let Some(bi) = tc.bias.as_mut() {
                        fill_v(bi, &mut seed);
                    }
                }
            }
        }
    }

    let dims = (2usize, 4usize, 4usize);
    let n = dit_cfg.in_channels * dims.0 * dims.1 * dims.2;
    let mut latents = vec![0.0_f32; n];
    fill(&mut latents, 101.0);
    let n_ctx = 3;
    let mut prompt = vec![0.0_f32; n_ctx * dit_cfg.text_dim];
    fill(&mut prompt, 102.0);
    let mask = vec![1.0_f32; dims.0 * dims.1 * dims.2];
    let cfg = WanGenConfig { num_steps: 2, flow_shift: 5.0, guidance_scale: 1.0 , cfg_interval: (0.0, 1.0) };

    let cpu = WanPipeline::new(dit, vae);
    let cpu_latents =
        cpu.denoise(latents.clone(), dims, &prompt, None, n_ctx, &mask, None, &cfg, |_, _, _| {});
    let cpu_video = cpu.decode(&cpu_latents, dims);

    let gpu = GpuWanPipeline::new(cpu.dit, cpu.vae);
    let gpu_latents =
        gpu.denoise(latents, dims, &prompt, None, n_ctx, &mask, None, &cfg, |_, _, _| {});
    let gpu_video = gpu.decode(&gpu_latents, dims);

    let cos = |a: &[f32], b: &[f32]| -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb + 1e-12)
    };

    let lat_cos = cos(&gpu_latents, &cpu_latents);
    println!("pipeline latents: cos_sim={lat_cos:.6}");
    assert!(lat_cos > 0.999, "denoised latents diverged: {lat_cos}");

    assert_eq!(
        (gpu_video.c, gpu_video.t, gpu_video.h, gpu_video.w),
        (cpu_video.c, cpu_video.t, cpu_video.h, cpu_video.w)
    );
    // 2 latent frames -> 5 output frames; 4 latent px -> 64 after unpatchify.
    assert_eq!((cpu_video.c, cpu_video.t, cpu_video.h, cpu_video.w), (3, 5, 64, 64));

    let vid_cos = cos(&gpu_video.data, &cpu_video.data);
    println!("pipeline video:   cos_sim={vid_cos:.6}");
    assert!(vid_cos > 0.99, "decoded video diverged: {vid_cos}");
    assert!(gpu_video.data.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
}

/// Load the **real** 20 GB Wan DiT and run a denoising step with it.
///
/// The only test that touches the shipped transformer. Validates what the
/// miniature model cannot: that the 825-tensor inventory derived from the
/// shard index matches the actual files, that a 5B-parameter load fits and
/// works, and that CPU and GPU agree at the real widths (3072 / 14336 /
/// head_dim 128, where the GPU routes to the head-dim-split attention kernel).
///
///   KLEARU_WAN_DIT=/abs/path/to/transformer \
///   cargo test -p klearu-diffusion --features metal --lib --release \
///     -- --ignored --nocapture real_wan_dit
#[test]
#[ignore]
fn real_wan_dit_loads_and_runs() {
    use crate::weight::loader::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_DIT") else {
        eprintln!("set KLEARU_WAN_DIT to the transformer/ directory; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanDitConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    println!(
        "config: {} layers, dim {} ({}x{}), ffn {}, in/out {}/{}, patch {:?}",
        cfg.num_layers,
        cfg.inner_dim(),
        cfg.num_attention_heads,
        cfg.attention_head_dim,
        cfg.ffn_dim,
        cfg.in_channels,
        cfg.out_channels,
        cfg.patch(),
    );

    // Inventory check first — cheap, and it fails with names rather than
    // after minutes of loading.
    let t0 = std::time::Instant::now();
    let comp = ComponentTensors::open_dir(&dir).expect("open checkpoint");
    println!("checkpoint holds {} tensors (mapped in {:.1}s)",
             comp.tensors.len(), t0.elapsed().as_secs_f32());
    let blank = WanDit::new(cfg.clone());
    let names = tensor_names(&cfg);
    println!("model requests {} tensors", names.len());
    let missing = blank.missing_tensors(&comp);
    assert!(missing.is_empty(), "missing {}: {:?}", missing.len(), &missing[..missing.len().min(8)]);
    let bad = blank.mismatched_tensors(&comp);
    assert!(bad.is_empty(), "mis-sized {}: {:?}", bad.len(), &bad[..bad.len().min(8)]);
    println!("inventory matches");

    let t1 = std::time::Instant::now();
    let dit = WanDit::load(cfg.clone(), &comp).expect("load real weights");
    println!("loaded 5B params in {:.1}s", t1.elapsed().as_secs_f32());

    // A small latent so the CPU reference stays tractable: 2x4x4 with patch
    // [1,2,2] gives 8 tokens.
    let (t, h, w) = (2usize, 4usize, 4usize);
    let seq = t * (h / 2) * (w / 2);
    let mut z = vec![0.0_f32; cfg.in_channels * t * h * w];
    let mut s = 777u32;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        ((s >> 8) as f32 / 8_388_608.0) - 1.0
    };
    for v in z.iter_mut() {
        *v = rnd();
    }
    let n_ctx = 8usize;
    let mut enc = vec![0.0_f32; n_ctx * cfg.text_dim];
    for v in enc.iter_mut() {
        *v = rnd() * 0.5;
    }
    // Per-token timesteps, first frame held at 0 — the TI2V path.
    let mut ts = vec![700.0_f32; seq];
    for v in ts.iter_mut().take(seq / 2) {
        *v = 0.0;
    }

    let t2 = std::time::Instant::now();
    let cpu_out = dit.forward(&z, (t, h, w), &ts, &enc, n_ctx);
    let cpu_secs = t2.elapsed().as_secs_f32();
    println!("cpu forward ({seq} tokens): {cpu_secs:.1}s");
    assert_eq!(cpu_out.len(), cfg.out_channels * t * h * w);
    assert!(cpu_out.iter().all(|v| v.is_finite()), "cpu forward produced non-finite values");

    let mean = cpu_out.iter().sum::<f32>() / cpu_out.len() as f32;
    let var = cpu_out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cpu_out.len() as f32;
    println!("velocity prediction: mean={mean:.4} std={:.4}", var.sqrt());
    // A flow-matching velocity on unit-scale noise should be O(1), not tiny
    // or exploding — either would mean the weights loaded but misapplied.
    assert!(var.sqrt() > 1e-3 && var.sqrt() < 100.0, "implausible velocity std {}", var.sqrt());


    #[cfg(feature = "metal")]
    {
        use super::gpu::GpuWanDit;
        let t3 = std::time::Instant::now();
        let gpu = GpuWanDit::from_cpu(&dit);
        println!("gpu weights uploaded in {:.1}s", t3.elapsed().as_secs_f32());

        let t4 = std::time::Instant::now();
        let gpu_out = gpu.forward(&z, (t, h, w), &ts, &enc, n_ctx);
        let gpu_secs = t4.elapsed().as_secs_f32();
        println!("gpu forward: {gpu_secs:.1}s  ({:.1}x)", cpu_secs / gpu_secs);

        assert!(gpu_out.iter().all(|v| v.is_finite()), "gpu forward produced non-finite values");
        let dot: f64 = gpu_out.iter().zip(&cpu_out).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = gpu_out.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = cpu_out.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb + 1e-12);
        println!("gpu vs cpu on real weights: cos_sim={cos:.6}");
        assert!(cos > 0.99, "gpu diverged on real weights: {cos}");
    }
}

// -- prompt encoding ------------------------------------------------------

#[test]
fn prompt_clean_collapses_whitespace() {
    use super::text::prompt_clean;
    assert_eq!(prompt_clean("  a   cat \n on  a\tmat "), "a cat on a mat");
    assert_eq!(prompt_clean("already clean"), "already clean");
    assert_eq!(prompt_clean(""), "");
}

/// Tokenize a real prompt with the shipped tokenizer and encode it with the
/// real umT5. The last piece between the port and actual text-to-video.
///
///   KLEARU_WAN_ROOT=/abs/path/to/models/wan22-ti2v-5b \
///   cargo test -p klearu-diffusion --lib --release \
/// Our token ids must equal the reference tokenizer's, byte for byte.
///
/// `umt5_matches_transformers` proves the encoder is exact, but it is fed
/// fixed ids — it cannot see a tokenizer that turns the prompt into
/// different-but-valid tokens. The encoder would then faithfully encode the
/// wrong words, and the only symptom is a generation that looks fine and
/// ignores what you asked for. The Chinese negative prompt is the case that
/// matters: it conditions *every* generation.
#[test]
#[ignore]
fn tokenizer_matches_transformers() {
    use super::text::WanTextEncoder;
    use crate::weight::ComponentTensors;

    let (Ok(root), Ok(ref_path)) =
        (std::env::var("KLEARU_WAN_ROOT"), std::env::var("KLEARU_TOK_REF"))
    else {
        eprintln!("set KLEARU_WAN_ROOT and KLEARU_TOK_REF; skipping");
        return;
    };
    let root = std::path::PathBuf::from(root);
    let ref_dir = std::path::Path::new(&ref_path).parent().unwrap().to_path_buf();
    let r = ComponentTensors::open_dir(&ref_dir).expect("open reference");
    let prompts: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{ref_path}.prompts.json")).expect("prompts json"),
    )
    .expect("parse prompts");

    let comp =
        ComponentTensors::open_dir(&root.join("text_encoder")).expect("open text_encoder");
    let te = WanTextEncoder::load(&root.join("tokenizer/tokenizer.json"), &comp)
        .expect("load tokenizer");

    let mut failures = Vec::new();
    for name in ["neg", "pos", "mixed"] {
        let want: Vec<u32> =
            r.get_f32(&format!("ids.{name}")).unwrap().iter().map(|v| *v as u32).collect();
        let text = prompts[name].as_str().expect("prompt text");
        let (ids, real) = te.tokenize(text).expect("tokenize");
        let got = &ids[..real];
        let first_diff = got
            .iter()
            .zip(&want)
            .position(|(a, b)| a != b)
            .unwrap_or(got.len().min(want.len()));
        let ok = got.len() == want.len() && got == want.as_slice();
        println!(
            "{name:6} ours={:3} ref={:3} {}",
            got.len(),
            want.len(),
            if ok {
                "match".to_string()
            } else {
                format!(
                    "DIFFER at {first_diff}: ours={:?} ref={:?}",
                    &got[first_diff..(first_diff + 6).min(got.len())],
                    &want[first_diff..(first_diff + 6).min(want.len())]
                )
            }
        );
        if !ok {
            failures.push(name);
        }
    }
    assert!(failures.is_empty(), "tokenizer differs from reference on: {failures:?}");
}

///     -- --ignored --nocapture real_prompt_encoding
#[test]
#[ignore]
fn real_prompt_encoding() {
    use super::text::{WanTextEncoder, MAX_PROMPT_TOKENS};
    use crate::weight::ComponentTensors;

    let Ok(root) = std::env::var("KLEARU_WAN_ROOT") else {
        eprintln!("set KLEARU_WAN_ROOT to the checkpoint root; skipping");
        return;
    };
    let root = std::path::PathBuf::from(root);

    let t0 = std::time::Instant::now();
    let comp = ComponentTensors::open_dir(&root.join("text_encoder")).expect("open text_encoder");
    let te = WanTextEncoder::load(&root.join("tokenizer/tokenizer.json"), &comp)
        .expect("load tokenizer + umT5");
    println!("loaded tokenizer + umT5 in {:.1}s", t0.elapsed().as_secs_f32());

    let prompt = "A cat wearing sunglasses drives a red convertible along a coastal road";
    let (ids, real) = te.tokenize(prompt).expect("tokenize");
    println!("prompt: {prompt:?}");
    println!("  {real} real tokens, padded to {}", ids.len());
    println!("  first ids: {:?}", &ids[..real.min(12)]);

    assert_eq!(ids.len(), MAX_PROMPT_TOKENS);
    assert!(real > 5, "prompt should produce more than a handful of tokens");
    assert!(real < MAX_PROMPT_TOKENS, "short prompt should not fill the context");
    // umT5 appends </s> = 1 as its final special token.
    assert_eq!(ids[real - 1], 1, "expected </s> to close the sequence");
    assert!(ids[real..].iter().all(|i| *i == 0), "padding should be <pad> = 0");
    let vocab = te.encoder.config.vocab_size as u32;
    assert!(ids.iter().all(|i| *i < vocab), "token id outside the embedding table");

    let t1 = std::time::Instant::now();
    let states = te.encode(prompt).expect("encode");
    println!("encoded in {:.1}s", t1.elapsed().as_secs_f32());

    let d = te.encoder.config.d_model;
    assert_eq!(states.len(), MAX_PROMPT_TOKENS * d);
    assert!(states.iter().all(|v| v.is_finite()));

    // Real positions carry signal; padded ones are exactly zero.
    let real_part = &states[..real * d];
    let pad_part = &states[real * d..];
    let mean = real_part.iter().sum::<f32>() / real_part.len() as f32;
    let var = real_part.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / real_part.len() as f32;
    println!("real positions:   mean={mean:.4} std={:.4}", var.sqrt());
    assert!(var.sqrt() > 1e-3, "real positions carry no signal");
    assert!(pad_part.iter().all(|v| *v == 0.0), "padded positions must be zeroed");

    // A different prompt must give different conditioning.
    let other = te.encode("An empty parking lot at night in the rain").expect("encode");
    let diff = states[..real * d]
        .iter()
        .zip(&other[..real * d])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("max |delta| vs a different prompt: {diff:.4}");
    assert!(diff > 1e-2, "conditioning does not depend on the prompt");
}

/// A real end-to-end text-to-video generation with the shipped checkpoints.
///
/// Uses **sequential residency**: umT5 (11 GB -> 22 GB as f32) is loaded,
/// used, and dropped before the DiT (20 GB) is loaded. Holding both would
/// need ~52 GB.
///
/// Resolution is deliberately small — 16x16 latent -> 256x256 pixels, 4 latent
/// frames -> 13 output frames — so this finishes in minutes rather than hours.
///
///   KLEARU_WAN_ROOT=/abs/path/to/models/wan22-ti2v-5b \
///   KLEARU_WAN_OUT=/tmp/wan_frame0.ppm \
///   cargo test -p klearu-diffusion --features metal --lib --release \
///     -- --ignored --nocapture real_text_to_video
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn real_text_to_video() {
    use super::gpu::GpuWanDit;
    use super::text::WanTextEncoder;
    use crate::vae3d::{gpu::GpuDecoder3d, unpatchify, Vol, WanVae, WanVaeConfig};
    use crate::weight::ComponentTensors;

    let Ok(root) = std::env::var("KLEARU_WAN_ROOT") else {
        eprintln!("set KLEARU_WAN_ROOT to the checkpoint root; skipping");
        return;
    };
    let root = std::path::PathBuf::from(root);
    let total = std::time::Instant::now();

    // Re-enter at the decode with latents checkpointed by a previous run.
    // Denoising 480p is ~15 minutes and the decode after it is the part that
    // gets OOM-killed, so without this every decode experiment costs a full
    // denoise.
    if let Ok(path) = std::env::var("KLEARU_WAN_LATENTS_IN") {
        let raw = std::fs::read(&path).expect("read latents");
        let rd = |i: usize| u32::from_le_bytes(raw[i..i + 4].try_into().unwrap()) as usize;
        let dims = (rd(0), rd(4), rd(8));
        let latents: Vec<f32> = raw[12..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        println!("re-entering at decode with latents from {path}, dims {dims:?}");
        decode_and_dump(&root, latents, dims, total);
        return;
    }

    // A detailed prompt, in the register Wan's own examples use: subject,
    // action, setting, camera, light. The previous one-liner was a smoke-test
    // fixture and left the model to invent everything it did not say, which is
    // how you get flat, generic output.
    let default_prompt = "A ginger tabby cat wearing dark aviator sunglasses \
        drives a vintage red convertible along a winding coastal highway. \
        Wind ruffles its fur as the car speeds past. Turquoise ocean and \
        rocky cliffs stretch to the horizon on the right. Warm golden \
        late-afternoon sunlight, long shadows, cinematic tracking shot from \
        the side, shallow depth of field, photorealistic, 35mm film.";
    let prompt = std::env::var("KLEARU_WAN_PROMPT")
        .unwrap_or_else(|_| default_prompt.to_string());
    let prompt = prompt.as_str();
    // The model's own default, not an invented one — see DEFAULT_NEGATIVE_PROMPT.
    let negative = super::pipeline::DEFAULT_NEGATIVE_PROMPT;
    println!("      prompt: {prompt}");

    // 1. Text conditioning, then drop the encoder before loading the DiT.
    let (cond, uncond, n_ctx) = {
        let t = std::time::Instant::now();
        let comp = ComponentTensors::open_dir(&root.join("text_encoder")).expect("text_encoder");
        let te = WanTextEncoder::load(&root.join("tokenizer/tokenizer.json"), &comp)
            .expect("load text encoder");
        let c = te.encode(prompt).expect("encode prompt");
        let u = te.encode(negative).expect("encode negative");
        let n = super::text::MAX_PROMPT_TOKENS;
        println!("[1/4] text conditioning in {:.1}s", t.elapsed().as_secs_f32());
        (c, u, n)
    };

    // 2. The DiT, on the GPU.
    let t = std::time::Instant::now();
    let dit_dir = root.join("transformer");
    let dit_cfg = WanDitConfig::from_json(
        &std::fs::read_to_string(dit_dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let dit_comp = ComponentTensors::open_dir(&dit_dir).expect("transformer");
    // Direct bf16 load: never materialises the 20 GB f32 model. The f32 route
    // peaked at 45 GB here (right after umT5's 22 GB was freed) on a 69 GB
    // machine with **no swap**, and that is where the intermittent all-NaN
    // forwards appeared — never in the isolated test, which loads only the DiT.
    let gpu_dit = GpuWanDit::load(dit_cfg.clone(), &dit_comp).expect("load DiT");
    println!("[2/4] DiT loaded direct to GPU bf16 in {:.1}s", t.elapsed().as_secs_f32());

    // 3. Denoise. Latent dims from the environment so resolution can be
    //    raised without a recompile — Wan 2.2 needs ~480p to produce anything
    //    coherent (the reference pipeline also fails at 256x256), so the
    //    default here is a smoke test, not a quality target.
    let dims = (
        std::env::var("KLEARU_WAN_T").ok().and_then(|v| v.parse().ok()).unwrap_or(4usize),
        std::env::var("KLEARU_WAN_H").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize),
        std::env::var("KLEARU_WAN_W").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize),
    );
    let seq = dims.0 * (dims.1 / 2) * (dims.2 / 2);
    println!("      latent {:?}, {seq} tokens", dims);
    // Gaussian noise, via Box-Muller. Diffusion models are trained on normal
    // noise; a uniform draw has the right variance but no tails, which starts
    // the trajectory off-distribution.
    let mut latents = vec![0.0_f32; dit_cfg.in_channels * dims.0 * dims.1 * dims.2];
    let mut s = 20260831u32;
    let mut uniform = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        // Open interval, so ln() below never sees 0.
        ((s >> 8) as f32 + 0.5) / 16_777_216.0
    };
    for pair in latents.chunks_mut(2) {
        let (u1, u2) = (uniform(), uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        pair[0] = r * theta.cos();
        if pair.len() > 1 {
            pair[1] = r * theta.sin();
        }
    }
    let nmean = latents.iter().sum::<f32>() / latents.len() as f32;
    let nvar =
        latents.iter().map(|v| (v - nmean).powi(2)).sum::<f32>() / latents.len() as f32;
    println!("      initial noise: mean={nmean:.4} std={:.4}", nvar.sqrt());

    let cfg = WanGenConfig {
        // 50 is the reference default for TI2V-5B; 20 was my own shortcut.
        num_steps: std::env::var("KLEARU_WAN_STEPS")
            .ok().and_then(|v| v.parse().ok())
            .unwrap_or(super::pipeline::REFERENCE_STEPS),
        // From the checkpoint, not from a resolution rule of thumb: Wan 2.2
        // TI2V-5B ships flow_shift 5.0.
        flow_shift: super::pipeline::flow_shift_from_checkpoint(&root)
            .expect("flow_shift from scheduler_config.json"),
        guidance_scale: std::env::var("KLEARU_WAN_GUIDANCE")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(5.0),
        // `KLEARU_WAN_CFG_INTERVAL=0.1,0.9` narrows the band guidance runs
        // over, trading a second forward on the outer steps for whatever
        // quality those steps' guidance was buying.
        cfg_interval: std::env::var("KLEARU_WAN_CFG_INTERVAL")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once(',')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .unwrap_or((0.0, 1.0)),
    };
    println!(
        "      sampling: steps={} flow_shift={} guidance={} cfg_interval={:?} \
         forwards={}",
        cfg.num_steps, cfg.flow_shift, cfg.guidance_scale, cfg.cfg_interval,
        cfg.forward_count()
    );
    let mask = vec![1.0_f32; dims.0 * dims.1 * dims.2];
    let sigmas = wan_sigmas(cfg.num_steps, cfg.flow_shift);
    let mut sched = crate::hidream::scheduler::UniPCScheduler::from_sigmas(sigmas);
    let timesteps = sched.timesteps.clone();
    let patch = dit_cfg.patch();
    let t = std::time::Instant::now();
    {
        use crate::hidream::scheduler::DenoiseScheduler;
        // Built once per guidance branch rather than once per forward: the text
        // projection and RoPE tables do not depend on the timestep.
        let ctx_cond = gpu_dit.prepare(dims, &cond, n_ctx);
        let ctx_uncond = gpu_dit.prepare(dims, &uncond, n_ctx);
        for (i, &ts_val) in timesteps.iter().enumerate() {
            let step = std::time::Instant::now();
            let ts = expand_timesteps(&mask, dims, patch, ts_val);
            // `KLEARU_WAN_NO_HOIST=1` rebuilds the context every forward, as
            // the code did before `prepare` was hoisted out of this loop. The
            // hoisted context holds pooled GPU buffers across all 100 forwards,
            // which is exactly the kind of lifetime the buffer pool can break.
            // Guidance runs only inside the configured band, so the second
            // forward — half the cost of a step — is skipped outside it.
            let use_cfg =
                cfg.guidance_scale > 1.0 && cfg.cfg_applies_at(i, timesteps.len());
            let no_hoist = std::env::var_os("KLEARU_WAN_NO_HOIST").is_some();
            let mut pred = if no_hoist {
                gpu_dit.forward(&latents, dims, &ts, &cond, n_ctx)
            } else {
                gpu_dit.forward_prepared(&latents, dims, &ts, &ctx_cond)
            };
            let un = if use_cfg {
                Some(if no_hoist {
                    gpu_dit.forward(&latents, dims, &ts, &uncond, n_ctx)
                } else {
                    gpu_dit.forward_prepared(&latents, dims, &ts, &ctx_uncond)
                })
            } else {
                None
            };
            if i == 0 {
                let bad = |v: &[f32]| v.iter().filter(|x| !x.is_finite()).count();
                println!(
                    "      branch check: cond bad={} uncond bad={} (ts uniq={})",
                    bad(&pred),
                    un.as_ref().map(|u| bad(u)).unwrap_or(0),
                    {
                        let mut u: Vec<u32> = ts.iter().map(|t| t.to_bits()).collect();
                        u.sort_unstable();
                        u.dedup();
                        u.len()
                    }
                );
            }
            if let Some(un) = &un {
                apply_cfg(&mut pred, un, cfg.guidance_scale);
            }
            // Per-step statistics: a stepper that diverges shows as growing
            // std long before it reaches NaN, and knowing whether the
            // conditional or unconditional branch goes first localises it.
            let stat = |v: &[f32]| -> (f32, usize) {
                let m = v.iter().sum::<f32>() / v.len() as f32;
                let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
                (sd, v.iter().filter(|x| !x.is_finite()).count())
            };
            let (psd, pbad) = stat(&pred);
            latents = sched.step(i, &pred, &latents, None, 1.0, 0.0);
            let (lsd, lbad) = stat(&latents);
            println!(
                "      step {:>2}/{} t={ts_val:>6.1} {:.1}s  pred_std={psd:.3} lat_std={lsd:.3} bad={pbad}/{lbad}",
                i + 1,
                timesteps.len(),
                step.elapsed().as_secs_f32()
            );
            if lbad > 0 {
                panic!("latents went non-finite at step {}", i + 1);
            }
        }
    }
    println!("[3/4] denoised in {:.1}s", t.elapsed().as_secs_f32());
    assert!(latents.iter().all(|v| v.is_finite()), "denoising produced non-finite latents");
    // Denoised latents live in normalised space and should stay order-1. A
    // large drift here means the schedule or CFG scale pushed them off
    // distribution, which the VAE then renders as oversaturated smear.
    let lmean = latents.iter().sum::<f32>() / latents.len() as f32;
    let lvar = latents.iter().map(|v| (v - lmean).powi(2)).sum::<f32>() / latents.len() as f32;
    println!("      denoised latents: mean={lmean:.4} std={:.4}", lvar.sqrt());

    // Checkpoint the latents before decoding. Denoising is ~15 minutes at
    // 480p; the decode after it is where this run has been OOM-killed
    // (SIGKILL/137), and losing the whole denoise to each decode attempt makes
    // iterating on the decode impossible. With this, `KLEARU_WAN_LATENTS_IN`
    // re-enters at step 4.
    if let Ok(path) = std::env::var("KLEARU_WAN_LATENTS_OUT") {
        let mut buf = Vec::with_capacity(latents.len() * 4 + 12);
        for d in [dims.0, dims.1, dims.2] {
            buf.extend_from_slice(&(d as u32).to_le_bytes());
        }
        for v in latents.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, buf).expect("write latents");
        println!("      checkpointed latents to {path}");
    }

    // Release the DiT's ~10 GB of GPU weights AND the pooled scratch buffers
    // before the decoder allocates. `drop` alone leaves the pool holding
    // several 291 MB f32 sgemm scratch buffers sized for the transformer,
    // which the decoder cannot reuse — they are simply dead weight at exactly
    // the moment memory is tightest.
    drop(gpu_dit);
    crate::metal_backend::drain_pool();

    decode_and_dump(&root, latents, dims, total);
}

/// Steps 4+ of `real_text_to_video`: VAE-decode the latents, check the video,
/// and dump every frame. Split out so a checkpointed-latents run can re-enter
/// here without repeating the ~15-minute denoise.
fn decode_and_dump(
    root: &std::path::Path,
    latents: Vec<f32>,
    dims: (usize, usize, usize),
    total: std::time::Instant,
) {
    use crate::vae3d::{gpu::GpuDecoder3d, unpatchify, Vol, WanVae, WanVaeConfig};
    use crate::weight::ComponentTensors;

    // 4. Decode.
    let t = std::time::Instant::now();
    let vae_dir = root.join("vae");
    let vae_cfg =
        WanVaeConfig::from_json(&std::fs::read_to_string(vae_dir.join("config.json")).unwrap())
            .unwrap();
    let vae_comp = ComponentTensors::open_dir(&vae_dir).expect("vae");
    let vae = WanVae::load(vae_cfg.clone(), &vae_comp).expect("load VAE");
    let gpu_vae = GpuDecoder3d::from_cpu(&vae.decoder);

    let mut z = latents.clone();
    if vae_cfg.latents_mean.len() == vae_cfg.z_dim {
        denormalize_latents(
            &mut z,
            vae_cfg.z_dim,
            &vae_cfg.latents_mean.clone(),
            &vae_cfg.latents_std.clone(),
        );
    }
    let zq = vae
        .post_quant_conv
        .forward(&Vol::new(z, vae_cfg.z_dim, dims.0, dims.1, dims.2), None);
    let raw = gpu_vae.decode_all(&zq);
    let mut video = unpatchify(&raw, vae_cfg.patch_size);
    for v in video.data.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
    println!("[4/4] decoded in {:.1}s", t.elapsed().as_secs_f32());

    println!(
        "video: [{}, {}, {}, {}]  total {:.1}s",
        video.c, video.t, video.h, video.w,
        total.elapsed().as_secs_f32()
    );
    assert_eq!(video.c, 3);
    assert_eq!(video.t, 1 + (dims.0 - 1) * vae_cfg.temporal_compression());
    assert_eq!(video.h, dims.1 * vae_cfg.spatial_compression());
    assert!(video.data.iter().all(|v| v.is_finite()));

    let mean = video.data.iter().sum::<f32>() / video.data.len() as f32;
    let var = video.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / video.data.len() as f32;
    let sat =
        video.data.iter().filter(|v| v.abs() >= 0.999).count() as f32 / video.data.len() as f32;
    println!("pixels: mean={mean:.4} std={:.4} saturated={sat:.3}", var.sqrt());
    assert!(var.sqrt() > 0.02, "output is nearly flat (std={})", var.sqrt());
    assert!(sat < 0.5, "output is mostly clamped ({sat})");

    // Frames must differ from one another — a static clip would mean the
    // temporal path collapsed.
    let plane = video.h * video.w;
    let frame_delta = (0..video.c)
        .flat_map(|c| (0..plane).map(move |i| (c, i)))
        .map(|(c, i)| {
            let a = video.data[(c * video.t) * plane + i];
            let b = video.data[(c * video.t + video.t - 1) * plane + i];
            (a - b).abs()
        })
        .fold(0.0_f32, f32::max);
    println!("max |frame0 - frameN|: {frame_delta:.4}");
    // Only meaningful with more than one frame: a single-frame decode compares
    // frame 0 against itself. Decoding one latent frame is how you get pixels
    // out at resolutions where the full sequence will not fit in memory.
    assert!(
        video.t == 1 || frame_delta > 0.01,
        "all frames identical; temporal path collapsed"
    );

    // Dump EVERY frame as a PPM (no image-crate dependency needed), so the
    // result is an actual watchable sequence rather than a single still.
    // `KLEARU_WAN_OUT` is a directory; frames land in it as `frame_000.ppm`
    // ... ready for `ffmpeg -i frame_%03d.ppm out.mp4`.
    if let Ok(dir) = std::env::var("KLEARU_WAN_OUT") {
        std::fs::create_dir_all(&dir).expect("create output dir");
        for f in 0..video.t {
            let mut buf = format!("P6\n{} {}\n255\n", video.w, video.h).into_bytes();
            for y in 0..video.h {
                for x in 0..video.w {
                    for c in 0..3 {
                        let v = video.data[(c * video.t + f) * plane + y * video.w + x];
                        buf.push((((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0) as u8);
                    }
                }
            }
            let path = format!("{dir}/frame_{f:03}.ppm");
            std::fs::write(&path, buf).unwrap_or_else(|e| panic!("write {path}: {e}"));
        }
        println!("wrote {} frames ({}x{}) to {dir}/frame_NNN.ppm",
                 video.t, video.w, video.h);
    }
}


/// Compare klearu against diffusers using the **real 5B weights**.
///
/// `wan_reference_matches_diffusers` validates the architecture on a miniature
/// model. This validates the shipped checkpoint at its real widths, which is
/// what distinguishes an implementation bug from a usage problem (resolution,
/// noise distribution, schedule) when a generation looks wrong.
///
///   /tmp/klearu_ref_venv/bin/python scripts/dump_wan_real_forward.py \
///       --model models/wan22-ti2v-5b/transformer --out <dir>/real_forward.safetensors
///   KLEARU_WAN_DIT=/abs/transformer KLEARU_WAN_REAL_REF=<dir>/real_forward.safetensors \
///   cargo test -p klearu-diffusion --features metal --lib --release \

/// The whole denoising loop, against diffusers, step by step.
///
/// Every component matches the reference exactly, which leaves the parts no
/// component test can see: CFG, the sigma schedule, per-token timestep
/// expansion, and the order the scheduler is stepped in. A bug there yields a
/// video that is coherent and wrong, so it survives every check that looks at
/// one piece at a time. This drives `WanPipeline::denoise` itself rather than
/// reimplementing the loop — a reimplementation would validate the test, not
/// the shipped path.
///
/// The reference uses random prompt embeddings, not real ones: umT5 is already
/// proven exact, and skipping it keeps the dump near 5 GB instead of ~34 GB.
#[test]
#[ignore]
fn denoise_loop_matches_diffusers() {
    use super::pipeline::{GpuWanPipeline, WanGenConfig, WanPipeline};
    use super::WanDit;
    use crate::vae3d::{WanVae, WanVaeConfig};
    use crate::weight::ComponentTensors;
    use crate::wan::WanDitConfig;

    let (Ok(root), Ok(ref_path)) =
        (std::env::var("KLEARU_WAN_ROOT"), std::env::var("KLEARU_DENOISE_REF"))
    else {
        eprintln!("set KLEARU_WAN_ROOT and KLEARU_DENOISE_REF; skipping");
        return;
    };
    let root = std::path::PathBuf::from(root);
    let ref_dir = std::path::Path::new(&ref_path).parent().unwrap().to_path_buf();
    let r = ComponentTensors::open_dir(&ref_dir).expect("open reference");

    let shape = r.get_f32("shape").expect("shape");
    let (t, h, w) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let conf = r.get_f32("config").expect("config");
    let cfg = WanGenConfig {
        num_steps: conf[0] as usize,
        flow_shift: conf[1],
        guidance_scale: conf[2],
        // The reference applies guidance on every step, so this comparison
        // must too.
        cfg_interval: (0.0, 1.0),
    };
    println!(
        "t={t} h={h} w={w} steps={} shift={} guidance={}",
        cfg.num_steps, cfg.flow_shift, cfg.guidance_scale
    );

    let latents = r.get_f32("in.latent").expect("in.latent");
    let prompt = r.get_f32("in.prompt").expect("in.prompt");
    let negative = r.get_f32("in.negative").expect("in.negative");

    // The real pipeline, loaded the way generation loads it. The VAE is not
    // exercised here — this compares latents — but `GpuWanPipeline` owns it, and
    // a stub would be a second thing to keep true.
    let dit_dir = root.join("transformer");
    let dit_cfg = WanDitConfig::from_json(
        &std::fs::read_to_string(dit_dir.join("config.json")).expect("read dit config"),
    )
    .expect("parse dit config");
    let dit_comp = ComponentTensors::open_dir(&dit_dir).expect("open transformer");
    let vae_dir = root.join("vae");
    let vae_cfg = WanVaeConfig::from_json(
        &std::fs::read_to_string(vae_dir.join("config.json")).expect("read vae config"),
    )
    .expect("parse vae config");
    let vae = WanVae::load(vae_cfg, &ComponentTensors::open_dir(&vae_dir).expect("open vae"))
        .expect("load vae");
    let n_ctx = prompt.len() / dit_cfg.text_dim;

    // Our schedule must agree before the trajectory can mean anything.
    let ours = super::pipeline::wan_sigmas(cfg.num_steps, cfg.flow_shift);
    let ref_sigmas = r.get_f32("sigmas").expect("sigmas");
    let sig_diff = ours
        .iter()
        .zip(ref_sigmas.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("sigma max_abs_diff = {sig_diff:.3e}");
    assert!(sig_diff < 1e-5, "sigma schedule differs: {sig_diff}");

    // Text-to-video: every token denoises on the same timestep.
    let mask = vec![1.0_f32; t * h * w];

    let mut worst_cos = 1.0_f64;
    let mut worst_step = 0usize;
    let mut step_err = Vec::new();
    // `KLEARU_DENOISE_CPU=1` runs the same loop on the exact CPU DiT. That is
    // what separates the two things this test could be measuring: our loop
    // being wrong, versus GPU bf16/half storage accumulating. The CPU DiT
    // matches diffusers to cos_sim 1.00000000, so any error the CPU run shows
    // is composition and nothing else. It is slow — one forward is ~50 s at
    // T=4 — so use a short reference.
    let on_cpu = std::env::var_os("KLEARU_DENOISE_CPU").is_some();
    let mut record = |i: usize, _total: usize, got: &[f32]| {
            let want = r.get_f32(&format!("step.{}", i - 1)).expect("step tensor");
            assert_eq!(got.len(), want.len(), "step {i}: length");
            let dot: f64 = got.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
            let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (na * nb + 1e-12);
            let worst =
                got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            println!("  step {i:2} cos_sim={cos:.8} max_abs_diff={worst:.3e}");
            step_err.push(cos);
            if cos < worst_cos {
                worst_cos = cos;
                worst_step = i;
            }
    };
    let out = if on_cpu {
        println!("running the CPU pipeline (exact DiT)");
        let dit = WanDit::load(dit_cfg, &dit_comp).expect("load cpu DiT");
        let pipe = WanPipeline::new(dit, vae);
        pipe.denoise(
            latents.clone(),
            (t, h, w),
            &prompt,
            Some(&negative),
            n_ctx,
            &mask,
            None,
            &cfg,
            &mut record,
        )
    } else {
        let pipe = GpuWanPipeline::load(dit_cfg, &dit_comp, vae).expect("load pipeline");
        pipe.denoise(
            latents.clone(),
            (t, h, w),
            &prompt,
            Some(&negative),
            n_ctx,
            &mask,
            None,
            &cfg,
            &mut record,
        )
    };

    let want = r.get_f32("out.latent").expect("out.latent");
    assert_eq!(out.len(), want.len(), "final latent length");
    println!("worst step {worst_step}: cos_sim {worst_cos:.8}");

    // The bar is set for GPU storage precision (bf16/half), which the DiT
    // comparison puts at ~1e-2 max_abs on a single forward. A composition bug —
    // an inverted CFG term, a shifted sigma index, a per-sample timestep — moves
    // cos_sim by orders of magnitude more than that, and diverges progressively.
    assert!(
        worst_cos > 0.999,
        "denoise loop diverges from diffusers at step {worst_step}: cos_sim {worst_cos}"
    );
    // Divergence must not grow: a loop that is merely imprecise stays flat,
    // one that is wrong compounds.
    let first = step_err.first().copied().unwrap_or(1.0);
    let last = step_err.last().copied().unwrap_or(1.0);
    println!("cos_sim first={first:.8} last={last:.8}");
    assert!(
        last > first - 1e-3,
        "error compounds across steps ({first:.8} -> {last:.8}), which is what a \
         composition bug looks like"
    );
}

///     -- --ignored --nocapture real_dit_matches_diffusers
#[test]
#[ignore]
fn real_dit_matches_diffusers() {
    use crate::weight::loader::ComponentTensors;

    let (Ok(dit_dir), Ok(ref_path)) =
        (std::env::var("KLEARU_WAN_DIT"), std::env::var("KLEARU_WAN_REAL_REF"))
    else {
        eprintln!("set KLEARU_WAN_DIT and KLEARU_WAN_REAL_REF; skipping");
        return;
    };
    let dit_dir = std::path::PathBuf::from(dit_dir);
    let ref_dir = std::path::Path::new(&ref_path).parent().unwrap().to_path_buf();

    let cfg = WanDitConfig::from_json(
        &std::fs::read_to_string(dit_dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let dit = WanDit::load(cfg.clone(), &ComponentTensors::open_dir(&dit_dir).unwrap())
        .expect("load real DiT");

    let r = ComponentTensors::open_dir(&ref_dir).expect("open reference");
    let latent = r.get_f32("in.latent").unwrap();
    let enc = r.get_f32("in.encoder_hidden_states").unwrap();
    let ts = r.get_f32("in.timestep").unwrap();
    let want = r.get_f32("out.velocity").unwrap();

    let n_ctx = enc.len() / cfg.text_dim;
    // Shape comes from the dump, so a reference at the real 13-latent frame
    // count can be compared without editing this test.
    let (t, h, w) = match r.get_f32("shape") {
        Ok(v) => (v[0] as usize, v[1] as usize, v[2] as usize),
        Err(_) => (4usize, 16usize, 16usize),
    };
    println!("latent shape t={t} h={h} w={w}");
    assert_eq!(latent.len(), cfg.in_channels * t * h * w);
    assert_eq!(ts.len(), t * (h / 2) * (w / 2));

    let report = |label: &str, got: &[f32]| {
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        let dot: f64 = got.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        println!("{label:12} max_abs_diff={worst:.4e}  cos_sim={:.8}", dot / (na * nb + 1e-12));
        dot / (na * nb + 1e-12)
    };

    let t0 = std::time::Instant::now();
    let cpu = dit.forward(&latent, (t, h, w), &ts, &enc, n_ctx);
    println!("cpu forward in {:.1}s", t0.elapsed().as_secs_f32());
    let cpu_cos = report("cpu", &cpu);

    #[cfg(feature = "metal")]
    let gpu_cos = {
        use super::gpu::GpuWanDit;
        let g = GpuWanDit::from_cpu(&dit);
        let out = g.forward(&latent, (t, h, w), &ts, &enc, n_ctx);
        report("gpu", &out)
    };
    #[cfg(not(feature = "metal"))]
    let gpu_cos = 1.0;

    assert!(cpu_cos > 0.9999, "CPU diverged from diffusers on real weights: {cpu_cos}");
    assert!(gpu_cos > 0.99, "GPU diverged from diffusers on real weights: {gpu_cos}");
}

/// Compare the UniPC stepper against diffusers over a full trajectory.
///
/// The DiT is verified against diffusers on real weights, so when a generation
/// still looks wrong the remaining surface is the scheduler, CFG and latent
/// handling. This isolates the scheduler: identical sigmas, identical
/// pseudo-velocities, compare every intermediate sample.
///
///   /tmp/klearu_ref_venv/bin/python scripts/dump_wan_scheduler.py --out <dir>/sched.safetensors
///   KLEARU_SCHED_REF=<dir>/sched.safetensors \
///   cargo test -p klearu-diffusion --lib -- --ignored --nocapture scheduler_matches_diffusers
#[test]
#[ignore]
fn scheduler_matches_diffusers() {
    use crate::hidream::scheduler::{DenoiseScheduler, UniPCScheduler};
    use crate::weight::loader::ComponentTensors;

    let Ok(path) = std::env::var("KLEARU_SCHED_REF") else {
        eprintln!("set KLEARU_SCHED_REF; skipping");
        return;
    };
    let dir = std::path::Path::new(&path).parent().unwrap().to_path_buf();
    let r = ComponentTensors::open_dir(&dir).expect("open reference");

    let ref_sigmas = r.get_f32("sigmas").unwrap();
    let ref_ts = r.get_f32("timesteps").unwrap();
    let mut sample = r.get_f32("in.sample").unwrap();
    let traj = r.get_f32("out.trajectory").unwrap();
    let steps = ref_ts.len();
    let n = sample.len();

    // Our schedule must match theirs before the stepping can be compared. The
    // shift comes from the dump: the checkpoint's flow_shift is 5.0, and a
    // hardcoded 3.0 here reports a 1.3e-1 sigma mismatch that looks like a
    // scheduler bug but is only the two sides disagreeing about the config.
    let shift = r.get_f32("shift").map(|v| v[0]).unwrap_or(3.0);
    let ours = wan_sigmas(steps, shift);
    assert_eq!(ours.len(), ref_sigmas.len(), "sigma count");
    let sigma_diff =
        ours.iter().zip(&ref_sigmas).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    println!("sigma schedule max_abs_diff = {sigma_diff:.3e}");
    assert!(sigma_diff < 1e-5, "sigma schedule differs: {sigma_diff}");

    let mut sched = UniPCScheduler::from_sigmas(ours);
    let mut worst_overall = 0.0_f32;
    // Replay the reference's exact velocity sequence rather than reproducing
    // torch's RNG. These are shaped like a real model's output (std ~0.8),
    // which is what exercises the stepper's *stability* — bounded synthetic
    // velocities can match exactly and still hide divergence.
    let vels = r.get_f32("in.velocities").unwrap();
    for i in 0..steps {
        let model_out = &vels[i * n..(i + 1) * n];
        sample = sched.step(i, model_out, &sample, None, 1.0, 0.0);
        let sd = {
            let m = sample.iter().sum::<f32>() / n as f32;
            (sample.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n as f32).sqrt()
        };
        let want_sd = {
            let w = &traj[i * n..(i + 1) * n];
            let m = w.iter().sum::<f32>() / n as f32;
            (w.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n as f32).sqrt()
        };
        if i < 3 || i + 1 == steps {
            println!("  step {:>2}: std ours={sd:.4} ref={want_sd:.4}", i + 1);
        }
        let want = &traj[i * n..(i + 1) * n];
        let worst =
            sample.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        if i < 3 || i + 1 == steps || worst > 1e-3 {
            println!("  step {:>2}: max_abs_diff={worst:.3e}", i + 1);
        }
        worst_overall = worst_overall.max(worst);
    }
    println!("trajectory max_abs_diff = {worst_overall:.3e}");
    assert!(worst_overall < 1e-3, "UniPC stepping diverged from diffusers: {worst_overall}");
}

/// A checkpoint whose tensors carry *varied* values.
///
/// `synthetic_checkpoint` fills each tensor with one constant, which is what
/// makes it catch cross-wiring — but a constant weight matrix makes every
/// output channel identical, which degenerates the norms and produces NaN.
/// Fine for "did this tensor land here", useless for "does this model run".
fn synthetic_checkpoint_varied(c: &WanDitConfig) -> ComponentTensors {
    let mut comp = ComponentTensors {
        mmaps: Vec::new(),
        owned: Vec::new(),
        tensors: std::collections::HashMap::new(),
    };
    for (k, name) in tensor_names(c).into_iter().enumerate() {
        let n = tensor_numel(c, &name).unwrap();
        let mut bytes = Vec::with_capacity(n * 4);
        for i in 0..n {
            let v = ((i as f32) * 0.37 + k as f32 * 1.1).sin() * 0.2;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        comp.append_synth_tensor(&name, safetensors::Dtype::F32, vec![n], &bytes).unwrap();
    }
    comp
}

/// `GpuWanDit::load` must produce exactly what `from_cpu` does — it is the
/// same weights by a different route, so any difference is a loading bug.
#[cfg(feature = "metal")]
#[test]
fn gpu_dit_direct_load_matches_from_cpu() {
    use super::gpu::GpuWanDit;

    let c = tiny();
    let comp = synthetic_checkpoint_varied(&c);

    let cpu = WanDit::load(c.clone(), &comp).expect("cpu load");
    let via_cpu = GpuWanDit::from_cpu(&cpu);
    let direct = GpuWanDit::load(c.clone(), &comp).expect("direct load");

    let (z, enc, n_ctx) = tiny_inputs(&c);
    let dims = (2usize, 4usize, 4usize);
    let seq = 8;
    let mut ts = vec![700.0_f32; seq];
    for t in ts.iter_mut().take(seq / 2) {
        *t = 0.0;
    }

    let a = via_cpu.forward(&z, dims, &ts, &enc, n_ctx);
    let b = direct.forward(&z, dims, &ts, &enc, n_ctx);
    assert_eq!(a.len(), b.len());
    assert!(a.iter().all(|v| v.is_finite()), "reference path produced non-finite output");
    // Not exact any more, and legitimately so. Both paths used to land in bf16
    // buffers, so they agreed bit for bit. With half storage `from_cpu` goes
    // f32 -> half and keeps 10 mantissa bits, while the direct loader goes
    // f32 -> bf16 -> half and keeps 8 — and THIS checkpoint is synthetic f32.
    // Against the real bf16 checkpoint the two are still identical, because
    // bf16 -> half is lossless.
    let worst = a.iter().zip(b.iter()).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    let absmax = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        worst <= 0.02 * absmax.max(1.0),
        "direct load diverges from the f32 round trip: max_abs_diff {worst:.3e}          against absmax {absmax:.3e}"
    );
}

/// The direct loader must still reject a bad checkpoint up front, before
/// uploading 20 GB.
#[cfg(feature = "metal")]
#[test]
fn gpu_dit_direct_load_validates_first() {
    use super::gpu::GpuWanDit;

    let c = tiny();
    let empty = crate::weight::loader::ComponentTensors {
        mmaps: Vec::new(),
        owned: Vec::new(),
        tensors: std::collections::HashMap::new(),
    };
    match GpuWanDit::load(c.clone(), &empty) {
        Ok(_) => panic!("empty checkpoint should not load"),
        Err(e) => assert!(format!("{e}").contains("missing"), "{e}"),
    }

    let mut comp = synthetic_checkpoint(&c);
    let name = "blocks.0.attn1.to_q.weight";
    comp.tensors.remove(name);
    comp.append_synth_tensor(name, safetensors::Dtype::F32, vec![7], &[0u8; 28]).unwrap();
    match GpuWanDit::load(c, &comp) {
        Ok(_) => panic!("mis-sized tensor should not load"),
        Err(e) => assert!(format!("{e}").contains("unexpected size"), "{e}"),
    }
}

/// Load the real 5B DiT straight to GPU bf16 and run a step, so peak host
/// memory can be measured against the f32 path.
///
///   KLEARU_WAN_DIT=/abs/transformer /usr/bin/time -l \
///   cargo test -p klearu-diffusion --features metal --lib --release \
///     -- --ignored --nocapture real_wan_dit_direct
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn real_wan_dit_direct() {
    use super::gpu::GpuWanDit;
    use crate::weight::loader::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_DIT") else {
        eprintln!("set KLEARU_WAN_DIT; skipping");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = WanDitConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();

    let t0 = std::time::Instant::now();
    let comp = ComponentTensors::open_dir(&dir).expect("open checkpoint");
    let gpu = GpuWanDit::load(cfg.clone(), &comp).expect("direct load");
    println!("direct-loaded 5B params to GPU bf16 in {:.1}s", t0.elapsed().as_secs_f32());

    // Dims from the environment: a single forward at generation scale is the
    // cheap way to reproduce anything that only appears at long sequences.
    let t: usize = std::env::var("KLEARU_WAN_T").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let h: usize = std::env::var("KLEARU_WAN_H").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let w: usize = std::env::var("KLEARU_WAN_W").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let seq = t * (h / 2) * (w / 2);
    println!("latent ({t}, {h}, {w}) -> {seq} tokens");
    let mut z = vec![0.0_f32; cfg.in_channels * t * h * w];
    let mut s = 999u32;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        ((s >> 8) as f32 / 8_388_608.0) - 1.0
    };
    for v in z.iter_mut() {
        *v = rnd();
    }
    let n_ctx = 8usize;
    let mut enc = vec![0.0_f32; n_ctx * cfg.text_dim];
    for v in enc.iter_mut() {
        *v = rnd() * 0.5;
    }
    // `KLEARU_WAN_SGEMM_AB=1`: run the same forward down both matmul-scratch
    // paths and compare. The f32 path is already validated against diffusers
    // at cos_sim 1.00000000, so agreement here validates the f16 path
    // transitively — without reloading the 20 GB fp32 reference model, which
    // is what made the earlier reference runs a memory hazard.
    if std::env::var_os("KLEARU_WAN_SGEMM_AB").is_some() {
        let ts = vec![999.999_f32; seq];
        std::env::remove_var("KLEARU_WAN_F16_SGEMM");
        std::env::remove_var("KLEARU_WAN_BF16_GEMM");
        let t0 = std::time::Instant::now();
        let a = gpu.forward(&z, (t, h, w), &ts, &enc, n_ctx);
        let ta = t0.elapsed().as_secs_f32();
        // Which fast path to compare against: the native bf16 MMA kernel by
        // default, the MPS f16 scratch path with KLEARU_WAN_AB_F16=1.
        if std::env::var_os("KLEARU_WAN_AB_F16").is_some() {
            std::env::set_var("KLEARU_WAN_F16_SGEMM", "1");
        } else {
            std::env::set_var("KLEARU_WAN_BF16_GEMM", "1");
        }
        let t0 = std::time::Instant::now();
        let b = gpu.forward(&z, (t, h, w), &ts, &enc, n_ctx);
        let tb = t0.elapsed().as_secs_f32();

        let (mut dot, mut na, mut nb, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
            worst = worst.max((x - y).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        let bad = b.iter().filter(|v| !v.is_finite()).count();
        std::env::remove_var("KLEARU_WAN_F16_SGEMM");
        std::env::remove_var("KLEARU_WAN_BF16_GEMM");
        println!("  mps f32: {ta:.2}s   fast path: {tb:.2}s   speedup {:.2}x", ta / tb);
        println!("  cos_sim={cos:.8}  max_abs_diff={worst:.3e}  non-finite={bad}");
        // The f32 path must be sound; the f16 path is the thing under
        // measurement, and it is *expected* to fail intermittently — asserting
        // on it would just make this harness flaky instead of informative.
        assert!(a.iter().all(|v| v.is_finite()), "f32 scratch produced non-finite output");
        if bad > 0 {
            println!("  fast path NON-FINITE this run ({bad} values)");
        } else if cos < 0.9995 {
            println!("  fast path diverged further than usual: cos_sim {cos}");
        }
        return;
    }

    // Repeat the *same* forward many times. The failure is intermittent
    // (~1 in 13 whole runs), so a rate estimate needs many samples — and
    // amortising the 16 s model load over N forwards is the difference
    // between a 15-minute experiment and an hour of process launches.
    let reps: usize = std::env::var("KLEARU_WAN_REPS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if reps > 0 {
        let ts = vec![999.999_f32; seq];
        let mut fails = 0usize;
        for r in 0..reps {
            let out = gpu.forward(&z, (t, h, w), &ts, &enc, n_ctx);
            let bad = out.iter().filter(|v| !v.is_finite()).count();
            if bad > 0 {
                fails += 1;
                println!("  rep {r}: {bad}/{} non-finite", out.len());
            }
            // Statistics, not just finiteness. A forward can be entirely finite
            // and still be noise: the 30-step 704x1280 generation had a healthy
            // latent std and 0.2% clipping while producing static.
            let mn = out.iter().sum::<f32>() / out.len() as f32;
            let sd = (out.iter().map(|v| (v - mn).powi(2)).sum::<f32>()
                / out.len() as f32).sqrt();
            println!("  rep {r}: mean={mn:.5} std={sd:.5}");
        }
        println!("FAILURES: {fails}/{reps}");
        return;
    }

    let mut any_bad = false;
    for tv in [999.999_f32, 982.8, 964.3, 944.5, 923.2, 900.1, 786.0, 700.0, 500.0, 138.7] {
        let ts = vec![tv; seq];
        let t1 = std::time::Instant::now();
        let out = gpu.forward(&z, (t, h, w), &ts, &enc, n_ctx);
        let bad = out.iter().filter(|v| !v.is_finite()).count();
        let finite: Vec<f32> = out.iter().copied().filter(|v| v.is_finite()).collect();
        let mean = if finite.is_empty() { 0.0 } else { finite.iter().sum::<f32>() / finite.len() as f32 };
        let sd = if finite.is_empty() {
            0.0
        } else {
            (finite.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / finite.len() as f32).sqrt()
        };
        println!(
            "  t={tv:>8.3}  bad={bad:>7}/{}  std={sd:.4}  ({:.0}s)",
            out.len(),
            t1.elapsed().as_secs_f32()
        );
        any_bad |= bad > 0;
    }
    assert!(!any_bad, "some timestep produced non-finite output at {seq} tokens");
}

/// GPU DiT against the CPU DiT on real weights, in two passes.
///
/// `gpu_dit_matches_cpu` runs at 8 tokens on a synthetic config against a loose
/// `cos > 0.999`, and `real_dit_matches_diffusers` only exercises the CPU path —
/// so the GPU transformer has never been checked against a trustworthy
/// reference on real weights. That is the same gap that hid a large bf16
/// storage error in the VAE decoder, and the DiT has ~3x more sequential stores
/// and runs 100 times per generation.
///
/// Two passes because both models at once is ~30 GB (CPU f32 20 GB + GPU bf16
/// 10 GB) and this machine does not reliably have that:
///
///   KLEARU_WAN_DIT=<transformer/> KLEARU_DIT_SAVE=/tmp/dit_cpu.bin \
///     ... -- --ignored --nocapture dit_gpu_vs_cpu_real_weights
///   KLEARU_WAN_DIT=<transformer/> KLEARU_DIT_CMP=/tmp/dit_cpu.bin \
///     ... -- --ignored --nocapture dit_gpu_vs_cpu_real_weights
///
/// Token count is deliberately small: bf16 rounding is per-op, so the error
/// shows at any sequence length, and the scalar CPU forward is the slow part.
#[test]
#[ignore]
fn dit_gpu_vs_cpu_real_weights() {
    use super::gpu::GpuWanDit;
    use crate::weight::ComponentTensors;

    let Ok(dir) = std::env::var("KLEARU_WAN_DIT") else {
        eprintln!("set KLEARU_WAN_DIT; skipping");
        return;
    };
    let save = std::env::var("KLEARU_DIT_SAVE").ok();
    let cmp = std::env::var("KLEARU_DIT_CMP").ok();
    if save.is_none() && cmp.is_none() {
        eprintln!("set KLEARU_DIT_SAVE or KLEARU_DIT_CMP; skipping");
        return;
    }

    let dir = std::path::PathBuf::from(dir);
    let cfg = WanDitConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("transformer config"),
    )
    .expect("parse config");

    let (t, h, w) = (2usize, 16usize, 24usize);
    let seq = t * (h / 2) * (w / 2);
    let n_ctx = 16usize;
    let mut s = 4242u32;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        ((s >> 8) as f32 / 8_388_608.0) - 1.0
    };
    let z: Vec<f32> = (0..cfg.in_channels * t * h * w).map(|_| rnd()).collect();
    let enc: Vec<f32> = (0..n_ctx * cfg.text_dim).map(|_| rnd() * 0.5).collect();
    let ts = vec![700.0_f32; seq];
    println!("latent ({t}, {h}, {w}) -> {seq} tokens");

    let comp = ComponentTensors::open_dir(&dir).expect("transformer shards");

    if let Some(path) = save {
        let dit = WanDit::load(cfg, &comp).expect("load CPU DiT");
        let out = dit.forward(&z, (t, h, w), &ts, &enc, n_ctx);
        let mut buf = Vec::with_capacity(out.len() * 4);
        for v in &out {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, buf).expect("write cpu output");
        println!("cpu forward saved to {path} ({} values)", out.len());
        return;
    }

    let path = cmp.unwrap();
    let raw = std::fs::read(&path).expect("read cpu output");
    let want: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let gpu = GpuWanDit::load(cfg, &comp).expect("load GPU DiT");
    let got = gpu.forward(&z, (t, h, w), &ts, &enc, n_ctx);
    assert_eq!(got.len(), want.len(), "output length differs from the saved CPU run");

    let (mut dot, mut na, mut nb, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (x, y) in want.iter().zip(got.iter()) {
        assert!(y.is_finite(), "gpu produced non-finite output");
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        worst = worst.max((x - y).abs());
    }
    let absmax = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    println!(
        "gpu vs cpu DiT: cos_sim={:.8} max_abs_diff={worst:.4e} cpu_absmax={absmax:.3e}",
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    );
}
