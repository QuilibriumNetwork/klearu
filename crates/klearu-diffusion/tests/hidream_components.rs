//! Component-level validation of the HiDream-O1 Qwen3-VL decoder against a
//! reference PyTorch activation dump.
//!
//! Run `python scripts/dump_hidream_components.py` first to produce
//! `experiments/hidream_component_ref.safetensors`. If that file is absent the
//! test skips (so CI without the venv stays green). The tiny config below MUST
//! match the Python script.

use klearu_diffusion::hidream::config::TextConfig;
use klearu_diffusion::hidream::layers::{
    BottleneckPatchEmbed, FinalLayer, Linear, RmsNorm, TimestepEmbedder,
};
use klearu_diffusion::hidream::model::{Attention, DecoderLayer, Mlp};
use klearu_diffusion::hidream::mrope;
use safetensors::SafeTensors;
use std::path::PathBuf;

// ---- tiny config (keep in sync with scripts/dump_hidream_components.py) ----
const HIDDEN: usize = 16;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const INTERMEDIATE: usize = 32;
const ROPE_THETA: f32 = 10000.0;
const RMS_EPS: f32 = 1e-6;
const SEQ: usize = 6;
const PATCH_DIM: usize = 12;
const BOTTLENECK: usize = HIDDEN / 4;
const FREQ: usize = 8;
const NPATCH: usize = 4;
const MROPE_SECTION: [usize; 3] = [1, 1, 0];

fn tiny_text_cfg() -> TextConfig {
    TextConfig {
        hidden_size: HIDDEN,
        num_hidden_layers: 2,
        num_attention_heads: HEADS,
        num_key_value_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        intermediate_size: INTERMEDIATE,
        vocab_size: 256,
        rms_norm_eps: RMS_EPS,
        rope_theta: ROPE_THETA,
        attention_bias: false,
        rope_scaling: Some(klearu_diffusion::hidream::config::RopeScaling {
            mrope_section: Some(MROPE_SECTION.to_vec()),
            rope_type: None,
        }),
        max_position_embeddings: 4096,
    }
}

struct Ref {
    bytes: Vec<u8>,
}

impl Ref {
    fn st(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.bytes).expect("deserialize ref")
    }
}

fn f32_of(st: &SafeTensors, name: &str) -> Vec<f32> {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("missing tensor {name}"));
    let d = t.data();
    d.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn i64_of(st: &SafeTensors, name: &str) -> Vec<i64> {
    let t = st.tensor(name).unwrap();
    t.data()
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn cmp(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut dot = 0.0_f64;
    let mut ng = 0.0_f64;
    let mut nw = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        dot += g as f64 * w as f64;
        ng += g as f64 * g as f64;
        nw += w as f64 * w as f64;
        max_abs = max_abs.max((g as f64 - w as f64).abs());
    }
    let cos = dot / (ng.sqrt() * nw.sqrt() + 1e-12);
    println!("  {name:14} cos={cos:.6} max_abs={max_abs:.2e}");
    assert!(cos > 0.9999, "{name}: cosine {cos} too low");
    assert!(max_abs < 1e-3, "{name}: max_abs {max_abs} too high");
}

fn set_linear(l: &mut Linear, st: &SafeTensors, prefix: &str) {
    l.weight = f32_of(st, &format!("{prefix}.weight"));
    if l.bias.is_some() {
        l.bias = Some(f32_of(st, &format!("{prefix}.bias")));
    }
}

fn set_attn(a: &mut Attention, st: &SafeTensors, prefix: &str) {
    set_linear(&mut a.q_proj, st, &format!("{prefix}.q_proj"));
    set_linear(&mut a.k_proj, st, &format!("{prefix}.k_proj"));
    set_linear(&mut a.v_proj, st, &format!("{prefix}.v_proj"));
    set_linear(&mut a.o_proj, st, &format!("{prefix}.o_proj"));
    a.q_norm.weight = f32_of(st, &format!("{prefix}.q_norm.weight"));
    a.k_norm.weight = f32_of(st, &format!("{prefix}.k_norm.weight"));
}

fn ref_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/hidream_component_ref.safetensors")
}

#[test]
fn validate_components_against_reference() {
    let path = ref_path();
    if !path.exists() {
        eprintln!(
            "SKIP: {} not found — run `python scripts/dump_hidream_components.py`",
            path.display()
        );
        return;
    }
    let r = Ref { bytes: std::fs::read(&path).unwrap() };
    let st = r.st();
    let cfg = tiny_text_cfg();

    let x = f32_of(&st, "x"); // [SEQ, HIDDEN]
    let pos_ids = i64_of(&st, "pos_ids"); // [3, SEQ]

    // ---- mRoPE ----
    let tables = mrope::build_tables(&pos_ids, SEQ, HEAD_DIM, ROPE_THETA, &MROPE_SECTION);
    cmp("rope.cos", &tables.cos, &f32_of(&st, "rope.cos"));
    cmp("rope.sin", &tables.sin, &f32_of(&st, "rope.sin"));

    // ---- RMSNorm ----
    let mut rms = RmsNorm::new(HIDDEN, RMS_EPS);
    rms.weight = f32_of(&st, "rmsnorm.weight");
    let mut rms_out = x.clone();
    rms.apply(&mut rms_out, SEQ);
    cmp("rmsnorm", &rms_out, &f32_of(&st, "rmsnorm.out"));

    // ---- MLP (SwiGLU) ----
    let mut mlp = Mlp::new(HIDDEN, INTERMEDIATE);
    set_linear(&mut mlp.gate_proj, &st, "mlp.gate_proj");
    set_linear(&mut mlp.up_proj, &st, "mlp.up_proj");
    set_linear(&mut mlp.down_proj, &st, "mlp.down_proj");
    cmp("mlp", &mlp.forward(&x, SEQ), &f32_of(&st, "mlp.out"));

    // ---- BottleneckPatchEmbed ----
    let mut xemb = BottleneckPatchEmbed::new(PATCH_DIM, BOTTLENECK, HIDDEN);
    set_linear(&mut xemb.proj1, &st, "xemb.proj1");
    set_linear(&mut xemb.proj2, &st, "xemb.proj2");
    let xin = f32_of(&st, "xemb.in");
    cmp("xemb", &xemb.forward(&xin, NPATCH), &f32_of(&st, "xemb.out"));

    // ---- TimestepEmbedder ----
    let mut temb = TimestepEmbedder::new(FREQ, HIDDEN);
    set_linear(&mut temb.lin1, &st, "temb.mlp.0");
    set_linear(&mut temb.lin2, &st, "temb.mlp.2");
    let t = f32_of(&st, "temb.t")[0];
    cmp("temb", &temb.forward(t), &f32_of(&st, "temb.out"));

    // ---- FinalLayer ----
    let mut final_layer = FinalLayer::new(HIDDEN, PATCH_DIM);
    set_linear(&mut final_layer.linear, &st, "final.linear");
    cmp("final", &final_layer.forward(&x, SEQ), &f32_of(&st, "final.out"));

    // ---- Attention (hybrid mask) ----
    let gen_mask_i = i64_of(&st, "attn.gen_mask");
    let gen_mask: Vec<bool> = gen_mask_i.iter().map(|&v| v != 0).collect();
    let mut attn = Attention::new(&cfg);
    set_attn(&mut attn, &st, "attn");
    let attn_out = attn.forward(&x, SEQ, &tables, &gen_mask);
    cmp("attn", &attn_out, &f32_of(&st, "attn.out"));

    // ---- Full 2-layer decoder stack ----
    let mut states = x.clone();
    for i in 0..2 {
        let mut layer = DecoderLayer::new(&cfg);
        layer.input_layernorm.weight = f32_of(&st, &format!("layer{i}.input_layernorm.weight"));
        layer.post_attention_layernorm.weight =
            f32_of(&st, &format!("layer{i}.post_attention_layernorm.weight"));
        set_attn(&mut layer.self_attn, &st, &format!("layer{i}.self_attn"));
        set_linear(&mut layer.mlp.gate_proj, &st, &format!("layer{i}.mlp.gate_proj"));
        set_linear(&mut layer.mlp.up_proj, &st, &format!("layer{i}.mlp.up_proj"));
        set_linear(&mut layer.mlp.down_proj, &st, &format!("layer{i}.mlp.down_proj"));
        layer.forward(&mut states, SEQ, &tables, &gen_mask);
    }
    cmp("stack", &states, &f32_of(&st, "stack.out"));
}

#[test]
fn validate_ref_forward_against_reference() {
    use klearu_diffusion::hidream::config::HiDreamConfig;
    use klearu_diffusion::hidream::model::{DecoderLayer, HiDreamModel};

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/hidream_ref_forward.safetensors");
    if !path.exists() {
        eprintln!("SKIP: run `python scripts/dump_hidream_ref_forward.py` first");
        return;
    }
    let r = Ref { bytes: std::fs::read(&path).unwrap() };
    let st = r.st();

    // tiny config — keep in sync with scripts/dump_hidream_ref_forward.py.
    const IMG: u32 = 4;
    const TMS: u32 = 7;
    const LAYERS: usize = 2;
    const TXT_LEN: usize = 8;
    const GEN_LEN: usize = 8;
    const TGT_LEN: usize = 4;

    let cfg = HiDreamConfig {
        image_token_id: IMG,
        video_token_id: 30,
        vision_start_token_id: 3,
        text_config: tiny_text_cfg(),
    };
    let mut model = HiDreamModel::new(cfg);

    // ---- wire weights ----
    model.embed_tokens = f32_of(&st, "embed_tokens");
    model.norm.weight = f32_of(&st, "norm.weight");
    assert_eq!(model.layers.len(), LAYERS);
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let l: &mut DecoderLayer = layer;
        l.input_layernorm.weight = f32_of(&st, &format!("layer{i}.input_layernorm.weight"));
        l.post_attention_layernorm.weight =
            f32_of(&st, &format!("layer{i}.post_attention_layernorm.weight"));
        set_attn(&mut l.self_attn, &st, &format!("layer{i}.self_attn"));
        set_linear(&mut l.mlp.gate_proj, &st, &format!("layer{i}.mlp.gate_proj"));
        set_linear(&mut l.mlp.up_proj, &st, &format!("layer{i}.mlp.up_proj"));
        set_linear(&mut l.mlp.down_proj, &st, &format!("layer{i}.mlp.down_proj"));
    }
    set_linear(&mut model.x_embedder.proj1, &st, "x_embedder.proj1");
    set_linear(&mut model.x_embedder.proj2, &st, "x_embedder.proj2");
    set_linear(&mut model.t_embedder1.lin1, &st, "t_embedder1.mlp.0");
    set_linear(&mut model.t_embedder1.lin2, &st, "t_embedder1.mlp.2");
    set_linear(&mut model.final_layer2.linear, &st, "final_layer2.linear");

    // ---- inputs ----
    let text_ids: Vec<u32> = i64_of(&st, "input_ids").iter().map(|&v| v as u32).collect();
    assert_eq!(text_ids.len(), TXT_LEN);
    let positions = i64_of(&st, "position_ids"); // [3, seq] row-major
    let vinputs = f32_of(&st, "vinputs"); // [GEN_LEN, PATCH_DIM]
    let vision_embeds = f32_of(&st, "vision_embeds"); // [num_vlm, hidden]
    let deepstack: Vec<Vec<f32>> =
        (0..LAYERS).map(|i| f32_of(&st, &format!("deepstack.{i}"))).collect();

    let out = model.forward_ref(
        &text_ids, &vision_embeds, &deepstack, &vinputs, GEN_LEN, TGT_LEN, &positions, 0.37, IMG,
        TMS,
    );
    cmp("ref_forward.x_pred", &out, &f32_of(&st, "x_pred_target"));
}

#[test]
fn validate_vlm_image_against_reference() {
    use klearu_diffusion::hidream::vlm_image::{
        patchify_merge_block, smart_resize, VLM_MAX_PIXELS, VLM_MERGE, VLM_MIN_PIXELS, VLM_PATCH,
        VLM_TEMPORAL,
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/hidream_vlm_image_ref.safetensors");
    if !path.exists() {
        eprintln!("SKIP: run `python scripts/dump_hidream_vlm_image.py` first");
        return;
    }
    let r = Ref { bytes: std::fs::read(&path).unwrap() };
    let st = r.st();

    // ---- patchify (exact, no resampling fuzz) ----
    let img = f32_of(&st, "image_chw"); // [3, 64, 96]
    let (px, grid) =
        patchify_merge_block(&img, 3, 64, 96, VLM_PATCH, VLM_TEMPORAL, VLM_MERGE);
    assert_eq!(grid, (1, 4, 6));
    cmp("vlm.pixel_values", &px, &f32_of(&st, "pixel_values"));

    // ---- smart_resize dims (exact) ----
    let sr = i64_of(&st, "smart_resize"); // [n, 4] rows of (h, w, rh, rw)
    let factor = VLM_PATCH * VLM_MERGE;
    for row in sr.chunks_exact(4) {
        let (h, w, rh, rw) = (row[0] as usize, row[1] as usize, row[2], row[3]);
        let (gh, gw) = smart_resize(h, w, factor, VLM_MIN_PIXELS, VLM_MAX_PIXELS);
        assert_eq!((gh as i64, gw as i64), (rh, rw), "smart_resize({h},{w})");
    }
}

#[test]
fn validate_vision_encoder_against_reference() {
    use klearu_diffusion::hidream::vision::{VisionConfig, VisionEncoder};
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/hidream_vision_ref.safetensors");
    if !path.exists() {
        eprintln!("SKIP: run `python scripts/dump_hidream_vision.py` first");
        return;
    }
    let bytes = std::fs::read(&path).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();

    // Tiny config — keep in sync with scripts/dump_hidream_vision.py.
    let cfg = VisionConfig {
        depth: 2,
        hidden_size: 16,
        num_heads: 2,
        intermediate_size: 32,
        patch_size: 4,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        in_channels: 3,
        out_hidden_size: 24,
        num_position_embeddings: 36,
        deepstack_visual_indexes: vec![0, 1],
    };
    let mut enc = VisionEncoder::new(cfg);
    // The dump stores weights under `w.<state_dict_key>`; with prefix "" the
    // loader produces names like ".patch_embed..." so strip the leading dot.
    let getter = |name: &str| -> klearu_diffusion::Result<Vec<f32>> {
        let key = name.strip_prefix('.').unwrap_or(name);
        Ok(f32_of(&st, &format!("w.{key}")))
    };
    enc.load_with(&getter, "").unwrap();

    let pixel_values = f32_of(&st, "pixel_values");
    let (image_embeds, deepstack) = enc.forward(&pixel_values, (1, 4, 4));

    cmp("vision.image_embeds", &image_embeds, &f32_of(&st, "image_embeds"));
    cmp("vision.deepstack0", &deepstack[0], &f32_of(&st, "deepstack.0"));
    cmp("vision.deepstack1", &deepstack[1], &f32_of(&st, "deepstack.1"));
}

#[test]
fn inspect_real_weights() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/HiDream-O1-Image-Dev");
    if !dir.exists() {
        eprintln!("SKIP: model dir not present");
        return;
    }
    use klearu_diffusion::weight::loader::ComponentTensors;
    let comp = ComponentTensors::open_dir(&dir).unwrap();
    let emb = comp.get_f32("model.language_model.embed_tokens.weight").unwrap();
    let (mn, mx) = emb.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    println!("embed_tokens: len={} min={mn} max={mx}", emb.len());
    let q = comp.get_f32("model.language_model.layers.0.self_attn.q_proj.weight").unwrap();
    println!("q_proj[0]={} q_proj[1]={} (expect -0.05810547, 0.0012435913)", q[0], q[1]);
    println!("q_proj len={} (expect {})", q.len(), 4096 * 4096);
}
