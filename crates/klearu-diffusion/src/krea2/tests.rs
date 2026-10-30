//! Krea 2 tests. The GPU ones need `--release --features metal`.

use std::collections::HashMap;

use safetensors::Dtype;

use super::config::Krea2Config;
use super::model::{pack_latents, tensor_specs, unpack_latents, Krea2Transformer};
use crate::weight::loader::ComponentTensors;

/// Deterministic uniform values in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

/// A synthetic checkpoint with the real layout: matmul weights bf16 (so CPU
/// and GPU see bit-identical values), norm scales and modulation tables f32
/// like the released files.
fn synth_checkpoint(cfg: &Krea2Config, seed: u64) -> ComponentTensors {
    let mut rng = Lcg(seed);
    let mut comp = ComponentTensors { mmaps: Vec::new(), owned: Vec::new(), tensors: HashMap::new() };
    for (name, shape) in tensor_specs(cfg) {
        let n: usize = shape.iter().product();
        let is_norm = name.ends_with("norm.weight") || name.contains("norm1") || name.contains("norm2")
            || name.ends_with("norm_q.weight") || name.ends_with("norm_k.weight");
        if is_norm || name.contains("scale_shift_table") || name.contains("projector") {
            let scale = if name.contains("projector") { 0.5 } else { 0.2 };
            let v = rng.vec(n, scale);
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            comp.append_synth_tensor(&name, Dtype::F32, shape, &bytes).unwrap();
        } else {
            let fan_in = *shape.last().unwrap() as f32;
            let scale = if name.ends_with(".bias") { 0.05 } else { 1.0 / fan_in.sqrt() };
            let v = rng.vec(n, scale);
            let bytes: Vec<u8> =
                v.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect();
            comp.append_synth_tensor(&name, Dtype::BF16, shape, &bytes).unwrap();
        }
    }
    comp
}

fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64) * (*x as f64);
        bb += (*y as f64) * (*y as f64);
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

/// `dump_krea2_reference.py --klearu-noise` reproduces the runner's starting
/// noise in numpy; these are its first values for seed 11.
#[test]
fn starting_noise_matches_the_script_reimplementation() {
    let x = crate::flux_runner::sample_gaussian(&[16, 96, 96], 11);
    let want = [-0.632732f32, 2.149202, -0.978847, 0.425348, -0.182513, -0.916198];
    for (a, b) in x.iter().zip(want) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}

#[test]
fn pack_unpack_round_trips() {
    let (c, h, w, p) = (16, 6, 8, 2);
    let z: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
    let tokens = pack_latents(&z, c, h, w, p);
    // Token 1 is grid cell (0, 1): channel 0's top-left value is z[0, 0, 2].
    assert_eq!(tokens[64], z[2]);
    // Feature 1 of token 0 is (c=0, py=0, px=1).
    assert_eq!(tokens[1], z[1]);
    // Feature 2 of token 0 is (c=0, py=1, px=0).
    assert_eq!(tokens[2], z[w]);
    assert_eq!(unpack_latents(&tokens, c, h, w, p), z);
}

#[test]
fn synthetic_checkpoint_passes_inventory() {
    let cfg = Krea2Config::tiny();
    let comp = synth_checkpoint(&cfg, 1);
    assert!(super::model::check_inventory(&cfg, &comp).is_empty());
    Krea2Transformer::load(cfg, &comp).unwrap();
}

#[test]
fn cpu_forward_is_finite_and_text_conditioned() {
    let cfg = Krea2Config::tiny();
    let comp = synth_checkpoint(&cfg, 2);
    let m = Krea2Transformer::load(cfg.clone(), &comp).unwrap();
    let mut rng = Lcg(9);
    let n_txt = 5;
    let taps = rng.vec(n_txt * cfg.num_text_layers * cfg.text_hidden_dim, 1.0);
    let taps2 = rng.vec(n_txt * cfg.num_text_layers * cfg.text_hidden_dim, 1.0);
    let (gh, gw) = (3, 4);
    let img = rng.vec(gh * gw * cfg.in_channels, 1.0);
    let a = m.forward(&img, gh, gw, &m.text.forward(&taps, n_txt), 0.6);
    let b = m.forward(&img, gh, gw, &m.text.forward(&taps2, n_txt), 0.6);
    assert!(a.iter().all(|v| v.is_finite()));
    assert!(cos_sim(&a, &b) < 0.9999, "output ignores the prompt");
}

#[cfg(feature = "metal")]
mod gpu {
    use super::*;
    use crate::krea2::gpu::GpuKrea2Transformer;

    #[test]
    fn gpu_matches_cpu_on_tiny_config() {
        let cfg = Krea2Config::tiny();
        let comp = synth_checkpoint(&cfg, 3);
        let cpu = Krea2Transformer::load(cfg.clone(), &comp).unwrap();
        let gpu = GpuKrea2Transformer::load(cfg.clone(), &comp, |_, _| {}).unwrap();

        let mut rng = Lcg(11);
        let n_txt = 7;
        let taps = rng.vec(n_txt * cfg.num_text_layers * cfg.text_hidden_dim, 2.0);
        let cpu_txt = cpu.text.forward(&taps, n_txt);
        let gpu_txt = gpu.encode_text(&taps, n_txt);
        let c_txt = cos_sim(&cpu_txt, &gpu_txt);
        println!("text head cos={c_txt:.6}");
        assert!(c_txt > 0.999, "text head cos {c_txt}");

        let (gh, gw) = (6, 5);
        let img = rng.vec(gh * gw * cfg.in_channels, 1.0);
        for t in [0.95_f32, 0.4] {
            let want = cpu.forward(&img, gh, gw, &cpu_txt, t);
            // Same text tokens on both sides, so this isolates the blocks.
            let ctx = gpu.prepare(&cpu_txt, (gh, gw), n_txt);
            let got = gpu.forward(&img, t, &ctx);
            let c = cos_sim(&want, &got);
            println!("t={t} forward cos={c:.6}");
            assert!(c > 0.999, "forward cos {c} at t={t}");

            // Padding the text out to a longer branch must not change the
            // image tokens: padded rows are not keys.
            let padded = gpu.prepare(&cpu_txt, (gh, gw), n_txt + 9);
            let got_p = gpu.forward(&img, t, &padded);
            let cp = cos_sim(&got, &got_p);
            println!("t={t} padded-vs-unpadded cos={cp:.8}");
            assert!(cp > 0.99999, "padding leaked into attention: cos {cp}");
        }
    }

    /// The register kernel with a padded K/V stride must give the padding rows
    /// exactly zero weight — even when they hold large values.
    #[test]
    fn masked_flash_attention_ignores_padding_rows() {
        use crate::metal_backend::{flash_attention_masked_f16_gpu, GpuTensor};
        let (h, d) = (3usize, 128usize);
        let mut rng = Lcg(21);
        for (l_q, valid, stride) in [(40usize, 37usize, 40usize), (1031, 1027, 1032), (64, 64, 64)] {
            let q = rng.vec(h * l_q * d, 1.0);
            let mut k = rng.vec(h * stride * d, 1.0);
            let mut v = rng.vec(h * stride * d, 1.0);
            // Poison the padding: if any of it leaks, the result moves a lot.
            for hh in 0..h {
                for r in valid..stride {
                    for c in 0..d {
                        k[(hh * stride + r) * d + c] = 8.0;
                        v[(hh * stride + r) * d + c] = 1000.0;
                    }
                }
            }
            let scale = 1.0 / (d as f32).sqrt();
            let mut want = vec![0.0f32; h * l_q * d];
            for hh in 0..h {
                for i in 0..l_q {
                    let qi = &q[(hh * l_q + i) * d..][..d];
                    let s: Vec<f32> = (0..valid)
                        .map(|j| qi.iter().zip(&k[(hh * stride + j) * d..][..d]).map(|(a, b)| a * b).sum::<f32>() * scale)
                        .collect();
                    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
                    let z: f32 = e.iter().sum();
                    for (j, w) in e.iter().enumerate() {
                        for c in 0..d {
                            want[(hh * l_q + i) * d + c] += w / z * v[(hh * stride + j) * d + c];
                        }
                    }
                }
            }
            let up = |x: &[f32]| GpuTensor::upload_f32_as_f16(vec![x.len()], x);
            let mut o = GpuTensor::new_f16(vec![h * l_q * d]);
            assert!(flash_attention_masked_f16_gpu(&up(&q), &up(&k), &up(&v), &mut o, 1, h, l_q, stride, valid, d, scale));
            let got = o.download_to_f32();
            let c = cos_sim(&got, &want);
            let max = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!("l_q={l_q} valid={valid} stride={stride}: cos={c:.6} max|Δ|={max:.4}");
            assert!(c > 0.9999 && max < 0.05, "masked attention leaked padding: cos {c}, max {max}");
        }
    }

    /// Above 4096 rows `GpuLinear` splits the matmul into equal chunks; check
    /// every row lands, including a ragged split, against the CPU.
    #[test]
    fn gpu_linear_chunks_tall_inputs_exactly() {
        use crate::krea2::gpu::GpuLinear;
        let cfg = Krea2Config::tiny();
        let comp = synth_checkpoint(&cfg, 6);
        let d = cfg.hidden();
        let cpu = Krea2Transformer::load(cfg.clone(), &comp).unwrap();
        let gl = GpuLinear::load(&comp, "img_in", cfg.in_channels, d, true).unwrap();
        let mut rng = Lcg(8);
        for m in [4096usize, 4124, 9001] {
            let x = rng.vec(m * cfg.in_channels, 1.0);
            let mut want = vec![0.0f32; m * d];
            cpu.img_in.forward_batch(&x, &mut want);
            let got = gl
                .forward(&crate::metal_backend::GpuTensor::upload_f32_as_f16(vec![m, cfg.in_channels], &x), m)
                .download_to_f32();
            // Per-row check: a skipped or overwritten chunk shows up as a bad row.
            let worst = (0..m)
                .map(|r| cos_sim(&got[r * d..(r + 1) * d], &want[r * d..(r + 1) * d]))
                .fold(1.0f64, f64::min);
            println!("m={m}: worst row cos={worst:.6}");
            assert!(worst > 0.999, "m={m}: worst row cos {worst}");
        }
    }

    /// Large attention calls split into head groups; one head per dispatch
    /// must give the same answer as one dispatch for all heads.
    #[test]
    fn head_split_attention_matches_single_dispatch() {
        let mut cfg = Krea2Config::tiny();
        cfg.text_hidden_dim = 512;
        cfg.text_num_attention_heads = 4; // head_dim 128: register kernel
        cfg.text_num_key_value_heads = 4;
        let comp = synth_checkpoint(&cfg, 12);
        let gpu = GpuKrea2Transformer::load(cfg.clone(), &comp, |_, _| {}).unwrap();
        let mut rng = Lcg(13);
        let n_txt = 40;
        let taps = rng.vec(n_txt * cfg.num_text_layers * cfg.text_hidden_dim, 1.0);
        let whole = gpu.encode_text(&taps, n_txt);
        // A fresh thread reads the env override afresh.
        let split = std::thread::spawn(move || {
            std::env::set_var("KLEARU_KREA2_ATTN_FLOPS", "1");
            gpu.encode_text(&taps, n_txt)
        })
        .join()
        .unwrap();
        std::env::remove_var("KLEARU_KREA2_ATTN_FLOPS");
        let c = cos_sim(&whole, &split);
        println!("head-split vs single dispatch cos={c:.8}");
        assert!(c > 0.99999, "head-split attention differs: cos {c}");
    }

    /// The 12-row layerwise attention runs as hundreds of tiny batched
    /// sequences; make sure that shape is handled at the real head dim.
    #[test]
    fn gpu_text_head_handles_real_head_dim() {
        let mut cfg = Krea2Config::tiny();
        cfg.text_hidden_dim = 256;
        cfg.text_num_attention_heads = 2; // head_dim 128, as in the real model
        cfg.text_num_key_value_heads = 2;
        cfg.num_text_layers = 12;
        let comp = synth_checkpoint(&cfg, 4);
        let cpu = Krea2Transformer::load(cfg.clone(), &comp).unwrap();
        let gpu = GpuKrea2Transformer::load(cfg.clone(), &comp, |_, _| {}).unwrap();
        let mut rng = Lcg(5);
        let n_txt = 37;
        let taps = rng.vec(n_txt * cfg.num_text_layers * cfg.text_hidden_dim, 3.0);
        let c = cos_sim(&cpu.text.forward(&taps, n_txt), &gpu.encode_text(&taps, n_txt));
        println!("text head (dh=128, 12 taps) cos={c:.6}");
        assert!(c > 0.999, "text head cos {c}");
    }
}

/// Checks against tensors dumped by `scripts/dump_krea2_reference.py`.
///
/// Gated on two env vars: `KLEARU_KREA2_REF` (the dump's `--out` directory)
/// and, for anything that loads weights, `KLEARU_KREA2_DIR` (a Krea-2-Turbo
/// snapshot). The transformer test holds ~26 GB on the GPU — run it alone.
mod reference {
    use super::*;
    use std::path::PathBuf;

    /// Must match `PROMPT` in the dump script.
    pub const PROMPT: &str =
        "a red fox sitting in fresh snow at golden hour, photorealistic, a wooden sign reads 'KREA'";

    pub fn ref_dir() -> Option<PathBuf> {
        std::env::var_os("KLEARU_KREA2_REF").map(PathBuf::from)
    }

    pub fn model_dir() -> Option<PathBuf> {
        std::env::var_os("KLEARU_KREA2_DIR").map(PathBuf::from)
    }

    /// Raw little-endian i32s (`get_f32` does not decode integer tensors).
    pub fn get_i32(comp: &ComponentTensors, name: &str) -> Vec<i32> {
        let r = comp.tensors.get(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(r.dtype, Dtype::I32, "{name} is not i32");
        comp.mmaps[r.mmap_index][r.offset..r.end]
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// The real-token rows of padded `[512, 12, 2560]` taps, in order.
    pub fn real_rows(taps: &[f32], mask: &[i32], row: usize) -> (Vec<f32>, usize) {
        let mut out = Vec::new();
        for (i, &m) in mask.iter().enumerate() {
            if m != 0 {
                out.extend_from_slice(&taps[i * row..(i + 1) * row]);
            }
        }
        let n = out.len() / row;
        (out, n)
    }

    #[test]
    fn scheduler_sigmas_match_diffusers() {
        use crate::scheduler::flow_match::{FlowMatchConfig, FlowMatchScheduler};
        use crate::scheduler::Scheduler;
        let Some(dir) = ref_dir() else {
            eprintln!("SKIP (set KLEARU_KREA2_REF)");
            return;
        };
        let r = ComponentTensors::open_file(&dir.join("krea2_sched_ref.safetensors")).unwrap();
        let cases = [
            ("sched.turbo8", FlowMatchConfig::krea2_turbo(), 8, 4096),
            ("sched.raw28_4096", FlowMatchConfig::krea2_raw(), 28, 4096),
        ];
        for (name, cfg, steps, seq) in cases {
            let mut s = FlowMatchScheduler::new(cfg);
            s.set_image_seq_len(seq);
            s.set_timesteps(steps);
            let want = r.get_f32(name).unwrap();
            let max = s.sigmas().iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            println!("{name}: max |Δσ| = {max:.2e}");
            assert_eq!(s.sigmas().len(), want.len());
            assert!(max < 1e-5, "{name} sigmas differ by {max}");
        }
    }

    #[test]
    fn tokenizer_matches_reference() {
        let (Some(dir), Some(model)) = (ref_dir(), model_dir()) else {
            eprintln!("SKIP (set KLEARU_KREA2_REF and KLEARU_KREA2_DIR)");
            return;
        };
        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer/tokenizer.json")).unwrap();
        let r = ComponentTensors::open_file(&dir.join("krea2_text_ref.safetensors")).unwrap();
        let want: Vec<u32> = get_i32(&r, "cond.ids").iter().map(|&v| v as u32).collect();
        let got = crate::krea2::text::tokenize_prompt(&tok, PROMPT).unwrap();
        assert_eq!(got, want);
        // The empty negative prompt: prefix + suffix only, 5 real tokens.
        let neg = crate::krea2::text::tokenize_prompt(&tok, "").unwrap();
        let mask = get_i32(&r, "neg.mask");
        assert_eq!(neg.len() - crate::krea2::text::PREFIX_TOKENS, mask.iter().filter(|&&m| m != 0).count());
    }
}

#[cfg(feature = "metal")]
mod reference_gpu {
    use super::reference::*;
    use super::*;
    use crate::krea2::config::TEXT_SELECT_LAYERS;
    use crate::krea2::gpu::GpuKrea2Transformer;
    use crate::krea2::text::Krea2TextEncoder;

    /// `KLEARU_KREA2_PROMPT` + `KLEARU_KREA2_TEXT_REF` (a file stem) check
    /// another prompt, dumped with the script's `--prompt` / `--text-name`.
    #[test]
    fn text_encoder_matches_reference() {
        let (Some(dir), Some(model)) = (ref_dir(), model_dir()) else {
            eprintln!("SKIP (set KLEARU_KREA2_REF and KLEARU_KREA2_DIR)");
            return;
        };
        let prompt_override = std::env::var("KLEARU_KREA2_PROMPT").ok();
        let cond_prompt = prompt_override.as_deref().unwrap_or(PROMPT);
        let stem = std::env::var("KLEARU_KREA2_TEXT_REF").unwrap_or_else(|_| "krea2_text_ref".into());
        let r = ComponentTensors::open_file(&dir.join(format!("{stem}.safetensors"))).unwrap();
        let t0 = std::time::Instant::now();
        let enc = Krea2TextEncoder::load(&model.join("text_encoder"), &model.join("tokenizer/tokenizer.json"))
            .unwrap();
        println!("text encoder loaded in {:.1}s", t0.elapsed().as_secs_f64());
        let hidden = enc.cfg.hidden_size;
        let nl = TEXT_SELECT_LAYERS.len();

        for name in ["cond", "neg"] {
            let prompt = if name == "cond" { cond_prompt } else { "" };
            let t0 = std::time::Instant::now();
            let (taps, n) = enc.encode(prompt).unwrap();
            // `KLEARU_KREA2_DUMP_TAPS=<dir>`: save klearu's taps (raw f32
            // `[n, 12, hidden]`) so the reference pipeline can run on them.
            if let Some(d) = std::env::var_os("KLEARU_KREA2_DUMP_TAPS") {
                let raw: Vec<u8> = taps.iter().flat_map(|a| a.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&d).join(format!("klearu_{name}_taps.f32")), raw).unwrap();
            }
            println!("{name}: encoded {n} tokens in {:.2}s", t0.elapsed().as_secs_f64());
            let (want, wn) = real_rows(
                &r.get_f32(&format!("{name}.taps")).unwrap(),
                &get_i32(&r, &format!("{name}.mask")),
                nl * hidden,
            );
            assert_eq!(n, wn, "{name}: token count");
            let mut worst = 1.0f64;
            for slot in 0..nl {
                let pick = |v: &[f32]| -> Vec<f32> {
                    (0..n).flat_map(|t| v[(t * nl + slot) * hidden..][..hidden].to_vec()).collect()
                };
                let c = cos_sim(&pick(&taps), &pick(&want));
                println!("  {name} tap {slot} (hidden_states[{}]) cos={c:.6}", TEXT_SELECT_LAYERS[slot]);
                worst = worst.min(c);
            }
            assert!(worst > 0.999, "{name}: worst tap cos {worst}");
        }

        // A maximum-length prompt: encoding cost should stay a few seconds.
        let long = "a very detailed description of a sunlit alpine meadow ".repeat(60);
        let t0 = std::time::Instant::now();
        let (_, n_long) = enc.encode(&long).unwrap();
        println!("long prompt: {n_long} tokens in {:.2}s", t0.elapsed().as_secs_f64());
        assert_eq!(n_long, crate::krea2::text::MAX_SEQUENCE_LENGTH);

        // The text head on these taps, GPU against the f32 CPU reference: the
        // transformer test covers it only at the fox prompt's token count.
        {
            use crate::krea2::gpu::GpuTextHead;
            use crate::krea2::model::TextEncoderHead;
            let cfg = Krea2Config::default();
            let comp = ComponentTensors::open_dir(&model.join("transformer")).unwrap();
            let (taps, n) = enc.encode(cond_prompt).unwrap();
            let head = GpuTextHead::load(&comp, &cfg).unwrap();
            let gpu = head.forward(&taps, n);
            let cpu = TextEncoderHead::load(&comp, &cfg).unwrap().forward(&taps, n);
            let c = cos_sim(&gpu, &cpu);
            println!("text head on {n} tokens: GPU vs CPU cos={c:.6}");

            // Repeatability: the transformer forward is bitwise deterministic,
            // so any run-to-run difference here is this path's own.
            let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            let (mut enc_differ, mut head_differ, mut worst) = (0, 0, 1.0f64);
            for _ in 0..5 {
                let (t2, _) = enc.encode(cond_prompt).unwrap();
                if !same(&t2, &taps) {
                    enc_differ += 1;
                    worst = worst.min(cos_sim(&t2, &taps));
                }
                let h2 = head.forward(&taps, n);
                if !same(&h2, &gpu) {
                    head_differ += 1;
                    worst = worst.min(cos_sim(&h2, &gpu));
                }
            }
            println!("repeats: encoder {enc_differ}/5 and text head {head_differ}/5 differ bitwise (worst cos {worst:.6})");
            assert!(c > 0.999, "text head GPU vs CPU cos {c}");
        }

        // Off-by-one guard: tap 0 must match hidden_states[2] better than its
        // neighbours.
        let (taps, n) = enc.encode(cond_prompt).unwrap();
        let tap0: Vec<f32> = (0..n).flat_map(|t| taps[(t * nl) * hidden..][..hidden].to_vec()).collect();
        let sims: Vec<f64> = (1..=3)
            .map(|i| cos_sim(&tap0, &r.get_f32(&format!("cond.hidden_{i}")).unwrap()))
            .collect();
        println!("tap 0 vs hidden_states[1,2,3]: {sims:?}");
        assert!(sims[1] > sims[0] && sims[1] > sims[2], "tap 0 is not hidden_states[2]");
    }

    /// Forward timing at the 1024² token count (64x64 grid + a prompt).
    /// `KLEARU_KREA2_PROFILE=1` to run; add `KLEARU_WAN_PROF=1` for the
    /// per-phase split (which serialises the GPU, so totals inflate).
    #[test]
    fn profile_forward() {
        let Some(model) = model_dir() else {
            eprintln!("SKIP (set KLEARU_KREA2_DIR)");
            return;
        };
        if std::env::var_os("KLEARU_KREA2_PROFILE").is_none() {
            eprintln!("SKIP (set KLEARU_KREA2_PROFILE=1)");
            return;
        }
        let side: usize = std::env::var("KLEARU_KREA2_GRID").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let cfg = Krea2Config::from_json(
            &std::fs::read_to_string(model.join("transformer/config.json")).unwrap(),
        )
        .unwrap();
        let comp = ComponentTensors::open_dir(&model.join("transformer")).unwrap();
        let gpu = GpuKrea2Transformer::load(cfg.clone(), &comp, |_, _| {}).unwrap();
        let mut rng = Lcg(3);
        let n_txt = 28;
        let txt = rng.vec(n_txt * cfg.hidden(), 1.0);
        let ctx = gpu.prepare(&txt, (side, side), n_txt);
        let img = rng.vec(side * side * cfg.in_channels, 1.0);
        for i in 0..3 {
            let t0 = std::time::Instant::now();
            let v = gpu.forward(&img, 0.7, &ctx);
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            println!("grid {side}x{side} forward {i}: {:.2}s (non-finite {bad})", t0.elapsed().as_secs_f64());
        }
    }

    /// Both CFG branches at a generation-sized grid, as the runner drives
    /// them. The reference only covers a 16x16 grid with the prompt padded by
    /// 3; this adds the empty negative prompt (5 tokens) padded to the
    /// prompt's length, which masks whole key tiles. Checks: the padded
    /// unconditional pass equals the minimally padded one, repeats are
    /// deterministic, and no tokens stand out in velocity norm.
    /// `KLEARU_KREA2_CFG_DIAG=1` to run; `KLEARU_KREA2_GRID` (default 48).
    #[test]
    fn cfg_branches_at_generation_scale() {
        use crate::krea2::config::LATENT_CHANNELS;
        let (Some(dir), Some(model)) = (ref_dir(), model_dir()) else {
            eprintln!("SKIP (set KLEARU_KREA2_REF and KLEARU_KREA2_DIR)");
            return;
        };
        if std::env::var_os("KLEARU_KREA2_CFG_DIAG").is_none() {
            eprintln!("SKIP (set KLEARU_KREA2_CFG_DIAG=1)");
            return;
        }
        let side: usize = std::env::var("KLEARU_KREA2_GRID").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
        let tr = ComponentTensors::open_file(&dir.join("krea2_text_ref.safetensors")).unwrap();
        let cfg = Krea2Config::from_json(
            &std::fs::read_to_string(model.join("transformer/config.json")).unwrap(),
        )
        .unwrap();
        let mut comp = ComponentTensors::open_dir(&model.join("transformer")).unwrap();
        crate::krea2::names::canonicalize(&mut comp);
        let gpu = GpuKrea2Transformer::load(cfg.clone(), &comp, |_, _| {}).unwrap();
        drop(comp);

        let row = cfg.num_text_layers * cfg.text_hidden_dim;
        let text = |name: &str| {
            let (taps, n) =
                real_rows(&tr.get_f32(&format!("{name}.taps")).unwrap(), &get_i32(&tr, &format!("{name}.mask")), row);
            (gpu.encode_text(&taps, n), n)
        };
        let (txt_c, n_c) = text("cond");
        let (txt_u, n_u) = text("neg");
        println!("grid {side}x{side}: cond {n_c} tokens, neg {n_u} tokens");

        let lat = 2 * side;
        let x = pack_latents(
            &crate::flux_runner::sample_gaussian(&[LATENT_CHANNELS, lat, lat], 11),
            LATENT_CHANNELS, lat, lat, cfg.patch(),
        );
        let grid = (side, side);
        let ctx_c = gpu.prepare(&txt_c, grid, n_c);
        let ctx_u_min = gpu.prepare(&txt_u, grid, n_u);
        let ctx_u_cfg = gpu.prepare(&txt_u, grid, n_c);
        println!(
            "rows: cond {} (pad {}), neg minimal {} (pad {}), neg as CFG {} (pad {})",
            ctx_c.rows(), ctx_c.n_pad, ctx_u_min.rows(), ctx_u_min.n_pad, ctx_u_cfg.rows(), ctx_u_cfg.n_pad
        );

        let d_in = cfg.in_channels;
        let token_norms = |v: &[f32]| -> Vec<f32> {
            v.chunks_exact(d_in).map(|t| t.iter().map(|a| a * a).sum::<f32>().sqrt()).collect()
        };
        // Map of tokens that are non-finite (N) or > 4x the median norm (#).
        let outliers = |what: &str, v: &[f32]| -> usize {
            let norms = token_norms(v);
            let mut sorted: Vec<f32> = norms.iter().copied().filter(|a| a.is_finite()).collect();
            sorted.sort_by(f32::total_cmp);
            let med = sorted.get(sorted.len() / 2).copied().unwrap_or(f32::NAN);
            let is_bad = |a: f32| !a.is_finite() || a > 4.0 * med;
            let n_nan = norms.iter().filter(|a| !a.is_finite()).count();
            let n_bad = norms.iter().filter(|&&a| is_bad(a)).count();
            println!(
                "  {what}: median token norm {med:.3}, max {:.3}, {n_nan} non-finite tokens, {} > 4x median",
                sorted.last().copied().unwrap_or(f32::NAN), n_bad - n_nan
            );
            if n_bad > 0 {
                for r in 0..side {
                    let line: String = (0..side)
                        .map(|c| {
                            let a = norms[r * side + c];
                            if !a.is_finite() { 'N' } else if is_bad(a) { '#' } else { '.' }
                        })
                        .collect();
                    println!("    {r:2} {line}");
                }
            }
            n_bad
        };
        let report = |a: &str, va: &[f32], b: &str, vb: &[f32]| {
            let max = va.iter().zip(vb).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
            println!("  {a} vs {b}: cos={:.6} max|Δ|={max:.4}", cos_sim(va, vb));
            cos_sim(va, vb)
        };

        // Same input, repeated: any difference between runs is nondeterminism.
        let repeats: usize = std::env::var("KLEARU_KREA2_REPEATS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if repeats > 0 {
            let first = gpu.forward(&x, 1.0, &ctx_c);
            outliers("cond t=1 run 0", &first);
            let mut differ = 0;
            for i in 1..repeats {
                let v = gpu.forward(&x, 1.0, &ctx_c);
                let same = v.iter().zip(&first).all(|(a, b)| a.to_bits() == b.to_bits());
                if !same {
                    differ += 1;
                    report(&format!("cond t=1 run {i}"), &v, "run 0", &first);
                    outliers(&format!("cond t=1 run {i}"), &v);
                }
            }
            println!("repeats: {differ}/{} differ bitwise from run 0", repeats - 1);
            crate::metal_backend::gpu_memory_report("after repeats");
        }

        let mut worst = 1.0f64;
        let mut hot = 0;
        for t in [1.0f32, 0.5] {
            println!("t={t}");
            let v_c = gpu.forward(&x, t, &ctx_c);
            let v_u_cfg = gpu.forward(&x, t, &ctx_u_cfg);
            let v_c2 = gpu.forward(&x, t, &ctx_c);
            let v_u_min = gpu.forward(&x, t, &ctx_u_min);
            worst = worst.min(report("cond", &v_c, "cond repeat", &v_c2));
            worst = worst.min(report("neg as CFG", &v_u_cfg, "neg minimal", &v_u_min));
            report("cond", &v_c, "neg", &v_u_cfg);
            hot += outliers("cond", &v_c);
            hot += outliers("neg as CFG", &v_u_cfg);
            let guided: Vec<f32> = v_c.iter().zip(&v_u_cfg).map(|(a, b)| a + 4.5 * (a - b)).collect();
            outliers("guided (g=4.5)", &guided);
        }
        assert!(worst > 0.9999, "padding or repeat changed the velocity: cos {worst}");
        assert_eq!(hot, 0, "tokens with outlying velocity norm");
    }

    /// A whole Raw generation against `Krea2Pipeline` itself
    /// (`dump_krea2_reference.py --parts traj`): the same initial latent and
    /// diffusers' text embeddings, 768x768, 28 steps, guidance 4.5, with the
    /// runner's sampling loop step for step. Diffusers runs in bf16 and keeps
    /// its latents and timesteps in bf16; this keeps both in f32, so
    /// agreement decays slowly over the trajectory rather than staying exact.
    /// With `KLEARU_KREA2_VAE` naming a `vae/` directory the final latent is
    /// also decoded, to `krea2_traj_klearu.png` beside the reference's image.
    #[test]
    fn raw_trajectory_matches_reference() {
        use crate::krea2::config::LATENT_CHANNELS;
        use crate::scheduler::{FlowMatchConfig, FlowMatchScheduler, Scheduler};
        let (Some(dir), Some(model)) = (ref_dir(), model_dir()) else {
            eprintln!("SKIP (set KLEARU_KREA2_REF and KLEARU_KREA2_DIR)");
            return;
        };
        // `KLEARU_KREA2_TRAJ` / `KLEARU_KREA2_TEXT_REF` select another
        // prompt's trajectory (the script's `--traj-name` / `--text-name`).
        let stem = std::env::var("KLEARU_KREA2_TRAJ").unwrap_or_else(|_| "krea2_traj".into());
        let text_stem = std::env::var("KLEARU_KREA2_TEXT_REF").unwrap_or_else(|_| "krea2_text_ref".into());
        let traj_path = dir.join(format!("{stem}_ref.safetensors"));
        if !traj_path.is_file() {
            eprintln!("SKIP (no {}; run the dump script's traj part)", traj_path.display());
            return;
        }
        const SIDE: usize = 768;
        const STEPS: usize = 28;
        const GUIDANCE: f32 = 4.5;
        let tj = ComponentTensors::open_file(&traj_path).unwrap();
        let tr = ComponentTensors::open_file(&dir.join(format!("{text_stem}.safetensors"))).unwrap();
        let cfg = Krea2Config::from_json(
            &std::fs::read_to_string(model.join("transformer/config.json")).unwrap(),
        )
        .unwrap();
        let mut comp = ComponentTensors::open_dir(&model.join("transformer")).unwrap();
        crate::krea2::names::canonicalize(&mut comp);
        let gpu = GpuKrea2Transformer::load(cfg.clone(), &comp, |_, _| {}).unwrap();
        drop(comp);

        let row = cfg.num_text_layers * cfg.text_hidden_dim;
        let text = |name: &str| {
            let (taps, n) =
                real_rows(&tr.get_f32(&format!("{name}.taps")).unwrap(), &get_i32(&tr, &format!("{name}.mask")), row);
            (gpu.encode_text(&taps, n), n)
        };
        // `KLEARU_KREA2_TEXT_MODEL` (a snapshot with text_encoder/ and
        // tokenizer/) + `KLEARU_KREA2_PROMPT`: condition on klearu's own text
        // encoder instead of the reference taps, exactly as the runner does.
        let ((txt_c, n_c), (txt_u, n_u)) = match std::env::var_os("KLEARU_KREA2_TEXT_MODEL") {
            Some(tm) => {
                let tm = std::path::PathBuf::from(tm);
                let te = crate::krea2::text::Krea2TextEncoder::load(
                    &tm.join("text_encoder"), &tm.join("tokenizer/tokenizer.json"),
                )
                .unwrap();
                let prompt = std::env::var("KLEARU_KREA2_PROMPT").unwrap_or_else(|_| PROMPT.to_string());
                let own = |p: &str| {
                    let (taps, n) = te.encode(p).unwrap();
                    (gpu.encode_text(&taps, n), n)
                };
                let (c, u) = (own(&prompt), own(""));
                let (rc, ru) = (text("cond"), text("neg"));
                println!("own text vs reference: cond cos={:.6}, uncond cos={:.6}", cos_sim(&c.0, &rc.0), cos_sim(&u.0, &ru.0));
                (c, u)
            }
            None => (text("cond"), text("neg")),
        };
        let (p, lat) = (cfg.patch(), SIDE / 8);
        let grid = (lat / p, lat / p);
        let n_max = n_c.max(n_u);
        let ctx_c = gpu.prepare(&txt_c, grid, n_max);
        let ctx_u = gpu.prepare(&txt_u, grid, n_max);

        let mut sched = FlowMatchScheduler::new(FlowMatchConfig::krea2_raw());
        sched.set_image_seq_len(grid.0 * grid.1);
        sched.set_timesteps(STEPS);
        let sigmas = sched.sigmas().to_vec();
        let want_sigmas = tj.get_f32("traj.sigmas").unwrap();
        let ds = sigmas.iter().zip(&want_sigmas).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("sigmas: {} vs {}, max|Δ|={ds:.2e}", sigmas.len(), want_sigmas.len());
        assert!(ds < 1e-5);

        let guided = |x: &[f32], s: f32| -> Vec<f32> {
            let mut v = gpu.forward(x, s, &ctx_c);
            let vu = gpu.forward(x, s, &ctx_u);
            v.iter_mut().zip(&vu).for_each(|(a, b)| *a += GUIDANCE * (*a - b));
            v
        };
        let bf16_round = |v: &[f32]| -> Vec<f32> { v.iter().map(|&a| half::bf16::from_f32(a).to_f32()).collect() };
        // One-step agreement at a few points: our guided velocity from the
        // reference's own latent, against the velocity its next latent
        // implies. That removes trajectory drift; what remains is per-step
        // error plus the rounding of the reference's bf16 latents, which is
        // largest relative to the step early on.
        const PROBES: [usize; 8] = [0, 4, 8, 12, 16, 20, 24, 27];
        let mut worst_step = 1.0f64;
        for &i in &PROBES {
            let (s, s_next) = (sigmas[i], sigmas[i + 1]);
            // Diffusers casts the initial latent to bf16 before step 1.
            let x_in = if i == 0 {
                bf16_round(&tj.get_f32("traj.noise").unwrap())
            } else {
                tj.get_f32(&format!("traj.x{}", i - 1)).unwrap()
            };
            let x_out = tj.get_f32(&format!("traj.x{i}")).unwrap();
            let implied: Vec<f32> = x_out.iter().zip(&x_in).map(|(o, a)| (o - a) / (s_next - s)).collect();
            let c = cos_sim(&guided(&x_in, s), &implied);
            println!("one step {:2} from the reference latent: guided velocity cos={c:.6}", i + 1);
            worst_step = worst_step.min(c);
        }

        // The runner's loop (`Krea2Runner::generate`), from the reference's noise.
        let mut x = tj.get_f32("traj.noise").unwrap();
        let mut last = 1.0;
        for i in 0..STEPS {
            let (s, s_next) = (sigmas[i], sigmas[i + 1]);
            let mut v = gpu.forward(&x, s, &ctx_c);
            let vu = gpu.forward(&x, s, &ctx_u);
            for (what, out, ctx) in [("cond", &v, &ctx_c), ("uncond", &vu, &ctx_u)] {
                let bad_tok = out.chunks_exact(cfg.in_channels).filter(|t| t.iter().any(|a| !a.is_finite())).count();
                if bad_tok > 0 {
                    let again = gpu.forward(&x, s, ctx);
                    let bad_again =
                        again.chunks_exact(cfg.in_channels).filter(|t| t.iter().any(|a| !a.is_finite())).count();
                    println!(
                        "step {}: {what} has {bad_tok}/{} non-finite tokens; an immediate rerun has {bad_again}",
                        i + 1, grid.0 * grid.1
                    );
                }
            }
            v.iter_mut().zip(&vu).for_each(|(a, b)| *a += GUIDANCE * (*a - b));
            x.iter_mut().zip(&v).for_each(|(a, b)| *a += (s_next - s) * b);
            assert!(x.iter().all(|a| a.is_finite()), "non-finite latent after step {}", i + 1);
            let want = tj.get_f32(&format!("traj.x{i}")).unwrap();
            last = cos_sim(&x, &want);
            println!("step {:2} σ {s:.4} -> {s_next:.4}: latent cos={last:.6}", i + 1);
        }

        if let Some(vae_dir) = std::env::var_os("KLEARU_KREA2_VAE").map(std::path::PathBuf::from) {
            use crate::vae3d::{GpuImageVae, WanVae, WanVaeConfig};
            let vcfg = WanVaeConfig::from_json(&std::fs::read_to_string(vae_dir.join("config.json")).unwrap()).unwrap();
            let (mean, std) = (vcfg.latents_mean.clone(), vcfg.latents_std.clone());
            let vae = GpuImageVae::from_vae(&WanVae::load(vcfg, &ComponentTensors::open_dir(&vae_dir).unwrap()).unwrap());
            let mut z = unpack_latents(&x, LATENT_CHANNELS, lat, lat, p);
            for (ch, plane) in z.chunks_mut(lat * lat).enumerate() {
                plane.iter_mut().for_each(|v| *v = *v * std[ch] + mean[ch]);
            }
            // Fenced, as the runner decodes.
            let barrier = crate::metal_backend::set_mps_barrier(true);
            let img = vae.decode(&z, lat, lat);
            crate::metal_backend::set_mps_barrier(barrier);
            let out = dir.join(format!("{stem}_klearu.png"));
            match crate::image_io::save_png(&img, SIDE, SIDE, &out) {
                Ok(()) => println!("wrote {}", out.display()),
                Err(e) => println!("no PNG ({e})"),
            }
        }
        // The final latent too, raw little-endian f32 `[tokens, 64]`, for
        // decoding with the reference's own VAE.
        let raw: Vec<u8> = x.iter().flat_map(|a| a.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("{stem}_klearu.f32")), raw).unwrap();
        assert!(worst_step > 0.99, "one-step guided velocity cos {worst_step}");
        assert!(last > 0.9, "final latent cos {last}");
    }

    /// The VAE decode of the reference trajectory's final latent, three times:
    /// repeats must be bitwise identical (a clean decode is deterministic),
    /// and the result must match diffusers' own decode of that latent
    /// (`krea2_traj_diffusers.png`). `KLEARU_KREA2_VAE` names the `vae/` dir.
    #[test]
    #[cfg(feature = "image-io")]
    fn vae_decode_matches_reference_and_repeats() {
        use crate::krea2::config::LATENT_CHANNELS;
        use crate::vae3d::{GpuImageVae, WanVae, WanVaeConfig};
        let (Some(dir), Some(vae_dir)) = (ref_dir(), std::env::var_os("KLEARU_KREA2_VAE").map(std::path::PathBuf::from)) else {
            eprintln!("SKIP (set KLEARU_KREA2_REF and KLEARU_KREA2_VAE)");
            return;
        };
        let traj_path = dir.join("krea2_traj_ref.safetensors");
        // `KLEARU_KREA2_WANT_PNG`: compare against another diffusers decode.
        let want_png = std::env::var_os("KLEARU_KREA2_WANT_PNG")
            .map_or_else(|| dir.join("krea2_traj_diffusers.png"), std::path::PathBuf::from);
        if !traj_path.is_file() || !want_png.is_file() {
            eprintln!("SKIP (run the dump script's traj part)");
            return;
        }
        const SIDE: usize = 768;
        let lat = SIDE / 8;
        let tj = ComponentTensors::open_file(&traj_path).unwrap();
        // `KLEARU_KREA2_LATENT_F32`: decode a raw f32 packed latent instead
        // (e.g. `raw_trajectory_matches_reference`'s `<stem>_klearu.f32`).
        let x = match std::env::var_os("KLEARU_KREA2_LATENT_F32") {
            Some(p) => std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            None => tj.get_f32("traj.x27").unwrap(),
        };
        let vcfg = WanVaeConfig::from_json(&std::fs::read_to_string(vae_dir.join("config.json")).unwrap()).unwrap();
        let (mean, std) = (vcfg.latents_mean.clone(), vcfg.latents_std.clone());
        let vae = GpuImageVae::from_vae(&WanVae::load(vcfg, &ComponentTensors::open_dir(&vae_dir).unwrap()).unwrap());
        let mut z = unpack_latents(&x, LATENT_CHANNELS, lat, lat, 2);
        for (ch, plane) in z.chunks_mut(lat * lat).enumerate() {
            plane.iter_mut().for_each(|v| *v = *v * std[ch] + mean[ch]);
        }
        // What the runner has resident when it decodes:
        // `KLEARU_KREA2_VAE_AFTER_DIT=1` loads the transformer (from
        // `KLEARU_KREA2_DIR`) first; `KLEARU_KREA2_VAE_BALLAST_GB=<n>` instead
        // holds n GB of plain GPU buffers.
        let _dit = std::env::var_os("KLEARU_KREA2_VAE_AFTER_DIT").and(model_dir()).map(|m| {
            let cfg = Krea2Config::from_json(&std::fs::read_to_string(m.join("transformer/config.json")).unwrap()).unwrap();
            let mut comp = ComponentTensors::open_dir(&m.join("transformer")).unwrap();
            crate::krea2::names::canonicalize(&mut comp);
            GpuKrea2Transformer::load(cfg, &comp, |_, _| {}).unwrap()
        });
        // `KLEARU_KREA2_VAE_BALLAST_MB` sets each ballast buffer's size (1024).
        let gb: usize = std::env::var("KLEARU_KREA2_VAE_BALLAST_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mb: usize = std::env::var("KLEARU_KREA2_VAE_BALLAST_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
        let _ballast: Vec<crate::metal_backend::GpuTensor> = (0..gb * 1024 / mb)
            .map(|_| {
                let t = crate::metal_backend::GpuTensor::new_f16(vec![mb << 19]);
                // `KLEARU_KREA2_VAE_BALLAST_FILL=<byte>`: fill from the CPU with
                // that byte instead of zeros (255 reads as NaN in f16 or bf16),
                // so a kernel reading past its own buffer shows up.
                match std::env::var("KLEARU_KREA2_VAE_BALLAST_FILL").ok().and_then(|v| v.parse::<u8>().ok()) {
                    Some(b) => unsafe { std::ptr::write_bytes(t.buffer.contents() as *mut u8, b, mb << 20) },
                    None => crate::metal_backend::zero_buffer_gpu(&t.buffer, mb << 20),
                }
                t
            })
            .collect();
        // `KLEARU_KREA2_VAE_DROP_DIT=1`: load the transformer, then free it.
        let _dit = if std::env::var_os("KLEARU_KREA2_VAE_DROP_DIT").is_some() { drop(_dit); None } else { _dit };
        crate::metal_backend::gpu_memory_report("before decode");
        // As the runner decodes; `KLEARU_KREA2_VAE_BARRIER=0` decodes unfenced.
        let fenced = std::env::var("KLEARU_KREA2_VAE_BARRIER").map_or(true, |v| v != "0");
        let barrier = crate::metal_backend::set_mps_barrier(fenced);
        let decodes: Vec<Vec<f32>> = (0..3).map(|_| vae.decode(&z, lat, lat)).collect();
        crate::metal_backend::set_mps_barrier(barrier);
        let black = |d: &[f32]| (0..SIDE * SIDE).filter(|&p| (0..3).all(|c| d[c * SIDE * SIDE + p] <= -1.0)).count() as f64
            / (SIDE * SIDE) as f64;
        println!("fenced={fenced}: exact-black fraction per decode {:?}", decodes.iter().map(|d| (black(d) * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        // `clamp` passes NaN through and the PNG writer turns it into black,
        // so a non-finite decode looks like a black region, not an error.
        for (i, d) in decodes.iter().enumerate() {
            let bad: Vec<usize> = (0..SIDE * SIDE).filter(|&p| (0..3).any(|c| !d[c * SIDE * SIDE + p].is_finite())).collect();
            if let (Some(first), Some(last)) = (bad.first(), bad.last()) {
                println!("decode {i}: {} non-finite pixels, image rows {}..={}", bad.len(), first / SIDE, last / SIDE);
            }
        }

        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits());
        let differ = decodes[1..].iter().filter(|d| !same(d, &decodes[0])).count();
        let want: Vec<f32> = image::open(&want_png).unwrap().to_rgb8().pixels()
            .flat_map(|p| p.0)
            .map(|v| v as f32 / 127.5 - 1.0)
            .collect();
        // decode() is planar [3, H, W]; the PNG is interleaved.
        for (i, d) in decodes.iter().enumerate() {
            let inter: Vec<f32> = (0..SIDE * SIDE)
                .flat_map(|p| (0..3).map(move |c| d[c * SIDE * SIDE + p].clamp(-1.0, 1.0)))
                .collect();
            let mad = inter.iter().zip(&want).map(|(a, b)| (a - b).abs()).sum::<f32>() / want.len() as f32;
            println!("decode {i}: cos vs diffusers {:.6}, mean |Δ| {:.4} (in [-1,1])", cos_sim(&inter, &want), mad);
        }
        println!("decode repeats: {differ}/2 differ bitwise from the first");
        crate::image_io::save_png(&decodes[0], SIDE, SIDE, &dir.join("krea2_traj_ref_klearu_vae.png")).ok();
        assert_eq!(differ, 0, "VAE decode is not repeatable");
    }

    /// The runner's own text conditioning, in the state `krea2_generate`
    /// leaves the process in (transformer and VAE resident, text encoder
    /// loaded on demand), against the GPU text head applied to diffusers'
    /// taps for the same prompt. `KLEARU_KREA2_PROMPT` + `KLEARU_KREA2_TEXT_REF`
    /// pick the prompt; `KLEARU_KREA2_RUNNER_TEXT=1` to run (loads ~27 GB).
    #[test]
    fn runner_text_conditioning_matches_reference() {
        use crate::krea2::gpu::GpuTextHead;
        use crate::krea2::runner::{Krea2GenOpts, Krea2LoadSpec, Krea2Runner};
        let (Some(dir), Some(model)) = (ref_dir(), model_dir()) else {
            eprintln!("SKIP (set KLEARU_KREA2_REF and KLEARU_KREA2_DIR)");
            return;
        };
        if std::env::var_os("KLEARU_KREA2_RUNNER_TEXT").is_none() {
            eprintln!("SKIP (set KLEARU_KREA2_RUNNER_TEXT=1)");
            return;
        }
        let prompt = std::env::var("KLEARU_KREA2_PROMPT").unwrap_or_else(|_| PROMPT.to_string());
        let stem = std::env::var("KLEARU_KREA2_TEXT_REF").unwrap_or_else(|_| "krea2_text_ref".into());
        let tr = ComponentTensors::open_file(&dir.join(format!("{stem}.safetensors"))).unwrap();

        let spec = Krea2LoadSpec { checkpoint: model.clone(), offload_text_encoder: true, ..Default::default() };
        let runner = Krea2Runner::load(&spec).unwrap();
        let opts = Krea2GenOpts { prompt, ..Default::default() };
        let runs: Vec<_> = (0..3).map(|_| runner.text_tokens(&opts, true).unwrap()).collect();

        let cfg = Krea2Config::default();
        let comp = ComponentTensors::open_dir(&model.join("transformer")).unwrap();
        let head = GpuTextHead::load(&comp, &cfg).unwrap();
        let row = cfg.num_text_layers * cfg.text_hidden_dim;
        let want = |name: &str| {
            let (taps, n) =
                real_rows(&tr.get_f32(&format!("{name}.taps")).unwrap(), &get_i32(&tr, &format!("{name}.mask")), row);
            head.forward(&taps, n)
        };
        let (want_c, want_u) = (want("cond"), want("neg"));
        let mut worst = 1.0f64;
        for (i, (c, n, u)) in runs.iter().enumerate() {
            let u = u.as_ref().unwrap();
            let (cc, cu) = (cos_sim(c, &want_c), cos_sim(u, &want_u));
            println!("run {i}: {n} prompt tokens, cond cos={cc:.6}, uncond cos={cu:.6}");
            worst = worst.min(cc).min(cu);
        }
        // Per token: a whole-tensor cosine is dominated by the largest-norm
        // tokens and can hide one that is badly off.
        let d = cfg.hidden();
        let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>().sqrt();
        for (what, got, want) in [("cond", &runs[0].0, &want_c), ("uncond", runs[0].2.as_ref().unwrap(), &want_u)] {
            let per: Vec<(usize, f64, f32, f32)> = got
                .chunks_exact(d)
                .zip(want.chunks_exact(d))
                .enumerate()
                .map(|(t, (g, w))| (t, cos_sim(g, w), norm(g), norm(w)))
                .collect();
            let (t, c, ng, nw) = per.iter().copied().fold((0, 1.0, 0.0, 0.0), |a, e| if e.1 < a.1 { e } else { a });
            println!("  {what}: worst token {t} cos={c:.6} (|got| {ng:.1}, |want| {nw:.1}); norms {:?}",
                per.iter().map(|e| e.3.round() as i64).collect::<Vec<_>>());
            worst = worst.min(c);
        }
        assert!(worst > 0.999, "runner text conditioning cos {worst}");
    }

    /// Catch an intermittently non-finite forward in the act. Some processes
    /// produce NaN forwards for long stretches and others none at all, so
    /// this does what the runner does up to the first few forwards — text
    /// tokens for both branches (checked for finiteness and repeatability),
    /// then CFG forwards at t=1 — and on a non-finite result reruns that
    /// forward with magnitude probes on, naming the first block and site to go
    /// non-finite. `KLEARU_KREA2_CATCH=<forwards>` to run (each ~10 s at 768²).
    #[test]
    fn catch_non_finite_forward() {
        use crate::krea2::config::LATENT_CHANNELS;
        use crate::krea2::gpu::mag;
        let (Some(dir), Some(model)) = (ref_dir(), model_dir()) else {
            eprintln!("SKIP (set KLEARU_KREA2_REF and KLEARU_KREA2_DIR)");
            return;
        };
        let Some(n_fwd) = std::env::var("KLEARU_KREA2_CATCH").ok().and_then(|v| v.parse::<usize>().ok()) else {
            eprintln!("SKIP (set KLEARU_KREA2_CATCH=<forwards>)");
            return;
        };
        let tr = ComponentTensors::open_file(&dir.join("krea2_text_ref.safetensors")).unwrap();
        let cfg = Krea2Config::from_json(
            &std::fs::read_to_string(model.join("transformer/config.json")).unwrap(),
        )
        .unwrap();
        let mut comp = ComponentTensors::open_dir(&model.join("transformer")).unwrap();
        crate::krea2::names::canonicalize(&mut comp);
        let gpu = GpuKrea2Transformer::load(cfg.clone(), &comp, |_, _| {}).unwrap();
        drop(comp);

        let row = cfg.num_text_layers * cfg.text_hidden_dim;
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let text = |name: &str| {
            let (taps, n) =
                real_rows(&tr.get_f32(&format!("{name}.taps")).unwrap(), &get_i32(&tr, &format!("{name}.mask")), row);
            let a = gpu.encode_text(&taps, n);
            let b = gpu.encode_text(&taps, n);
            let bad = a.iter().filter(|v| !v.is_finite()).count();
            println!(
                "{name}: {n} text tokens, {bad} non-finite values, repeat {}",
                if same(&a, &b) { "bitwise identical".to_string() } else { format!("DIFFERS (cos {:.6})", cos_sim(&a, &b)) }
            );
            (a, n)
        };
        let ((txt_c, n_c), (txt_u, n_u)) = (text("cond"), text("neg"));
        let grid = (48, 48);
        let n_max = n_c.max(n_u);
        let ctx_c = gpu.prepare(&txt_c, grid, n_max);
        let ctx_u = gpu.prepare(&txt_u, grid, n_max);
        let x = pack_latents(
            &crate::flux_runner::sample_gaussian(&[LATENT_CHANNELS, 96, 96], 11),
            LATENT_CHANNELS, 96, 96, cfg.patch(),
        );
        let mut bad_forwards = 0;
        // A clean forward is bitwise deterministic, so a repeat that differs
        // is corruption even when every value is finite.
        let mut first: [Option<Vec<f32>>; 2] = [None, None];
        let mut differing = 0;
        for i in 0..n_fwd {
            let (what, ctx) = if i % 2 == 0 { ("cond", &ctx_c) } else { ("uncond", &ctx_u) };
            let v = gpu.forward(&x, 1.0, ctx);
            match &first[i % 2] {
                None => first[i % 2] = Some(v.clone()),
                Some(f) => {
                    let rows: Vec<usize> = (0..v.len() / cfg.in_channels)
                        .filter(|&r| {
                            let span = r * cfg.in_channels..(r + 1) * cfg.in_channels;
                            v[span.clone()].iter().zip(&f[span]).any(|(a, b)| a.to_bits() != b.to_bits())
                        })
                        .collect();
                    if !rows.is_empty() {
                        differing += 1;
                        println!(
                            "forward {i} ({what}): {} token rows differ from the first, rows {}..={}",
                            rows.len(), rows[0], rows[rows.len() - 1]
                        );
                    }
                }
            }
            let bad = v.iter().filter(|a| !a.is_finite()).count();
            if bad > 0 {
                bad_forwards += 1;
                println!("forward {i} ({what}): {bad}/{} non-finite; rerunning with probes", v.len());
                mag::set_on(true);
                let again = gpu.forward(&x, 1.0, ctx);
                mag::set_on(false);
                println!("  probed rerun: {} non-finite", again.iter().filter(|a| !a.is_finite()).count());
            }
        }
        println!("catch: {bad_forwards}/{n_fwd} forwards non-finite, {differing}/{} repeats differ", n_fwd.saturating_sub(2));
        crate::metal_backend::gpu_memory_report("end");
        assert_eq!(bad_forwards, 0);
        assert_eq!(differing, 0);
    }

    /// The runner's own decode of a saved latent, before and after a short
    /// generation in the same process: the CLI's image came out with black
    /// regions although the same latent decodes cleanly in a fresh process.
    /// `KLEARU_KREA2_RUNNER_DECODE=1` with `KLEARU_KREA2_LATENT_F32` (768²,
    /// packed) and `KLEARU_KREA2_WANT_PNG` (diffusers' decode of it);
    /// `KLEARU_KREA2_STEPS` sets the generation's length (default 2).
    #[test]
    #[cfg(feature = "image-io")]
    fn runner_decode_after_generate() {
        use crate::krea2::runner::{Krea2GenOpts, Krea2LoadSpec, Krea2Runner};
        let Some(model) = model_dir() else {
            eprintln!("SKIP (set KLEARU_KREA2_DIR)");
            return;
        };
        let (Some(_), Some(latent), Some(want_png)) = (
            std::env::var_os("KLEARU_KREA2_RUNNER_DECODE"),
            std::env::var_os("KLEARU_KREA2_LATENT_F32"),
            std::env::var_os("KLEARU_KREA2_WANT_PNG"),
        ) else {
            eprintln!("SKIP (set KLEARU_KREA2_RUNNER_DECODE, KLEARU_KREA2_LATENT_F32, KLEARU_KREA2_WANT_PNG)");
            return;
        };
        const SIDE: usize = 768;
        let x: Vec<f32> = std::fs::read(latent).unwrap().chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let want: Vec<f32> = image::open(want_png).unwrap().to_rgb8().pixels()
            .flat_map(|p| p.0).map(|v| v as f32 / 127.5 - 1.0).collect();
        let check = |what: &str, d: &[f32]| {
            let inter: Vec<f32> = (0..SIDE * SIDE)
                .flat_map(|p| (0..3).map(move |c| d[c * SIDE * SIDE + p]))
                .collect();
            let bad: Vec<usize> = (0..SIDE * SIDE).filter(|&p| (0..3).any(|c| !inter[p * 3 + c].is_finite())).collect();
            let clean: Vec<f32> = inter.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
            let rows = bad.first().map_or(String::new(), |f| format!(", image rows {}..={}", f / SIDE, bad[bad.len() - 1] / SIDE));
            println!("{what}: cos vs diffusers {:.6}, {} non-finite pixels{rows}", cos_sim(&clean, &want), bad.len());
        };
        let spec = Krea2LoadSpec { checkpoint: model, offload_text_encoder: true, ..Default::default() };
        let runner = Krea2Runner::load(&spec).unwrap();
        let lat = SIDE / 8;
        check("decode before generating", &runner.decode_packed(&x, lat, lat));
        let steps = std::env::var("KLEARU_KREA2_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        let opts = Krea2GenOpts { prompt: PROMPT.to_string(), width: SIDE, height: SIDE, seed: 11, steps, ..Default::default() };
        let img = runner.generate(&opts, |_, _| {}, &std::sync::atomic::AtomicBool::new(false)).unwrap();
        println!("generated {steps} steps: {} non-finite pixels", img.chw.iter().filter(|v| !v.is_finite()).count());
        check("decode after generating", &runner.decode_packed(&x, lat, lat));
    }

    /// Repeat the text head on real weights and count non-finite runs.
    /// `KLEARU_KREA2_STRESS=<n>` to run.
    #[test]
    fn stress_text_head() {
        use crate::krea2::gpu::GpuTextHead;
        let (Some(dir), Some(model)) = (ref_dir(), model_dir()) else {
            eprintln!("SKIP (set KLEARU_KREA2_REF and KLEARU_KREA2_DIR)");
            return;
        };
        let Some(n_runs) = std::env::var("KLEARU_KREA2_STRESS").ok().and_then(|v| v.parse::<usize>().ok()) else {
            eprintln!("SKIP (set KLEARU_KREA2_STRESS=<runs>)");
            return;
        };
        let tr = ComponentTensors::open_file(&dir.join("krea2_text_ref.safetensors")).unwrap();
        let cfg = Krea2Config::default();
        let comp = ComponentTensors::open_dir(&model.join("transformer")).unwrap();
        let head = GpuTextHead::load(&comp, &cfg).unwrap();
        let row = cfg.num_text_layers * cfg.text_hidden_dim;
        let (taps, n) = real_rows(&tr.get_f32("cond.taps").unwrap(), &get_i32(&tr, "cond.mask"), row);
        let first = head.forward(&taps, n);
        let mut bad_runs = 0;
        for i in 0..n_runs {
            let out = head.forward(&taps, n);
            let bad = out.iter().filter(|x| !x.is_finite()).count();
            if bad > 0 || cos_sim(&out, &first) < 0.99999 {
                bad_runs += 1;
                println!("run {i}: non-finite {bad}, cos vs first {:.6}", cos_sim(&out, &first));
            }
        }
        println!("text head stress: {bad_runs}/{n_runs} bad runs");
        assert_eq!(bad_runs, 0);
    }

    #[test]
    fn transformer_matches_reference() {
        let (Some(dir), Some(model)) = (ref_dir(), model_dir()) else {
            eprintln!("SKIP (set KLEARU_KREA2_REF and KLEARU_KREA2_DIR)");
            return;
        };
        let tag = model.file_name().unwrap().to_string_lossy().to_string();
        let tr = ComponentTensors::open_file(&dir.join("krea2_text_ref.safetensors")).unwrap();
        let dr = ComponentTensors::open_file(&dir.join(format!("krea2_dit_ref_{tag}.safetensors"))).unwrap();
        let cfg = Krea2Config::from_json(
            &std::fs::read_to_string(model.join("transformer/config.json")).unwrap(),
        )
        .unwrap();
        let mut comp = ComponentTensors::open_dir(&model.join("transformer")).unwrap();
        crate::krea2::names::canonicalize(&mut comp);
        let t0 = std::time::Instant::now();
        let gpu = GpuKrea2Transformer::load(cfg.clone(), &comp, |i, n| {
            if i % 7 == 0 || i == n {
                println!("  block {i}/{n}");
            }
        })
        .unwrap();
        println!("transformer loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let row = cfg.num_text_layers * cfg.text_hidden_dim;
        let (taps, n) = real_rows(&tr.get_f32("cond.taps").unwrap(), &get_i32(&tr, "cond.mask"), row);
        let t0 = std::time::Instant::now();
        let txt = gpu.encode_text(&taps, n);
        println!("text head {:.2}s", t0.elapsed().as_secs_f64());
        let c = cos_sim(&txt, &dr.get_f32("dit.txt").unwrap());
        println!("txt_in output cos={c:.6}");
        assert!(c > 0.999, "text head cos {c}");

        let packed = dr.get_f32("dit.packed").unwrap();
        let grid = (16, 16);
        // Pad as a CFG branch would, to exercise the padded path on real weights.
        let ctx = gpu.prepare(&txt, grid, n + 3);
        for i in 0..2 {
            let t = dr.get_f32(&format!("dit.t{i}")).unwrap()[0];
            let t0 = std::time::Instant::now();
            let v = gpu.forward(&packed, t, &ctx);
            let want = dr.get_f32(&format!("dit.v{i}")).unwrap();
            let c = cos_sim(&v, &want);
            let max = v.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!("t={t}: forward {:.2}s cos={c:.6} max|Δ|={max:.4}", t0.elapsed().as_secs_f64());
            assert!(v.iter().all(|x| x.is_finite()));
            // Not 0.999: diffusers' own bf16 forward scores 0.99879 against
            // this fp32-arithmetic reference (`reference_blocks` prints it),
            // and every block in isolation agrees to >= 0.9998 — the gap is
            // bf16 rounding compounding over 28 blocks, matched not exceeded.
            assert!(c > 0.998, "velocity cos {c} at t={t}");
        }
    }
}

#[cfg(feature = "metal")]
mod reference_blocks {
    use super::reference::*;
    use super::*;
    use crate::krea2::gpu::{GpuBlock, GpuLinear, GpuRope};
    use crate::krea2::model::{image_positions, Rope};
    use crate::metal_backend::{upload_f32_as_f16_buffer, GpuTensor};

    /// One block at a time from the reference's own inputs. Tells drift
    /// (each block near-exact, error compounding over 28) apart from a defect
    /// (one block clearly off).
    #[test]
    fn blocks_match_reference_in_isolation() {
        let (Some(dir), Some(model)) = (ref_dir(), model_dir()) else {
            eprintln!("SKIP (set KLEARU_KREA2_REF and KLEARU_KREA2_DIR)");
            return;
        };
        let tag = model.file_name().unwrap().to_string_lossy().to_string();
        let tr = ComponentTensors::open_file(&dir.join("krea2_text_ref.safetensors")).unwrap();
        let dr = ComponentTensors::open_file(&dir.join(format!("krea2_dit_ref_{tag}.safetensors"))).unwrap();
        let cfg = Krea2Config::from_json(
            &std::fs::read_to_string(model.join("transformer/config.json")).unwrap(),
        )
        .unwrap();
        let mut comp = ComponentTensors::open_dir(&model.join("transformer")).unwrap();
        crate::krea2::names::canonicalize(&mut comp);
        let d = cfg.hidden();

        let v32 = dr.get_f32("dit.v0").unwrap();
        let v16 = dr.get_f32("dit.v0_bf16").unwrap();
        println!("reference bf16 vs fp32-compute velocity cos={:.6}", cos_sim(&v16, &v32));

        let mask = get_i32(&tr, "cond.mask");
        let real: Vec<usize> = mask.iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
        let n_pad_txt = mask.len();
        let n_img = 256;
        // Reference rows [txt(512) | img(256)] -> ours [img | real txt].
        let reorder = |v: &[f32]| -> Vec<f32> {
            let mut o = Vec::with_capacity((n_img + real.len()) * d);
            o.extend_from_slice(&v[n_pad_txt * d..(n_pad_txt + n_img) * d]);
            for &r in &real {
                o.extend_from_slice(&v[r * d..(r + 1) * d]);
            }
            o
        };

        // Block 0's input image rows are img_in(packed).
        let img_in = GpuLinear::load(&comp, "img_in", cfg.in_channels, d, true).unwrap();
        let packed = dr.get_f32("dit.packed").unwrap();
        let x_img = img_in
            .forward(&GpuTensor::upload_f32_as_f16(vec![n_img, cfg.in_channels], &packed), n_img)
            .download_to_f32();
        let in0 = reorder(&dr.get_f32("blk.in0").unwrap());
        println!("img_in cos={:.6}", cos_sim(&x_img, &in0[..n_img * d]));

        let temb_mod = dr.get_f32("blk.temb_mod").unwrap();
        let mod_buf = upload_f32_as_f16_buffer(&temb_mod);
        let rows = n_img + real.len();
        let mut pos = image_positions(16, 16);
        pos.extend(std::iter::repeat_n([0.0; 3], real.len()));
        let rope = GpuRope::from_cpu(&Rope::new(&pos, cfg.axes_dims_rope, cfg.rope_theta));

        for i in [0usize, 1, 14, 27] {
            let blk = GpuBlock::load(&comp, &format!("transformer_blocks.{i}"), &cfg).unwrap();
            let x_in = reorder(&dr.get_f32(&format!("blk.in{i}")).unwrap());
            let want = reorder(&dr.get_f32(&format!("blk.out{i}")).unwrap());
            let mut x = GpuTensor::upload_f32_as_f16(vec![rows, d], &x_in);
            blk.forward(&mut x, rows, rows, &mod_buf, &rope);
            let got = x.download_to_f32();
            let delta = |o: &[f32]| -> Vec<f32> { o.iter().zip(&x_in).map(|(a, b)| a - b).collect() };
            let (gi, wi) = (&got[..n_img * d], &want[..n_img * d]);
            let (gd, wd) = (delta(&got), delta(&want));
            let peak = x_in.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            println!(
                "block {i:2}: out cos={:.6} (img {:.6})  update cos={:.6} (img {:.6}, txt {:.6})  max|x_in|={peak:.1}",
                cos_sim(&got, &want), cos_sim(gi, wi),
                cos_sim(&gd, &wd), cos_sim(&gd[..n_img * d], &wd[..n_img * d]),
                cos_sim(&gd[n_img * d..], &wd[n_img * d..]),
            );
        }
    }
}
