//! Checkpoint save/load for [`ImageTransformer`].
//!
//! Format: standard `.safetensors`. One tensor per param buffer. The
//! safetensors metadata includes a serialised [`ImageTransformerConfig`]
//! JSON, so the loader can reconstruct the model architecture from the
//! file alone (no side-channel config required).
//!
//! Optimizer state (AdamW's `m` and `v` buffers) lives in a separate
//! `.optim.safetensors` companion file when [`save_with_optimizer`] is
//! used. This keeps the model file portable (loadable for inference
//! without optimizer overhead) while still supporting resume.
//!
//! [`ImageTransformer`]: crate::model::ImageTransformer
//! [`ImageTransformerConfig`]: crate::model::ImageTransformerConfig

use std::collections::HashMap;
use std::path::Path;

use safetensors::{SafeTensors, tensor::TensorView, Dtype};

use crate::error::{ImageGenError, Result};
use crate::model::{ImageTransformer, ImageTransformerConfig};
use crate::optim::AdamW;

// ---- Config serialization (manual to avoid pulling serde into model) ----

#[derive(serde::Serialize, serde::Deserialize)]
struct ConfigJson {
    vocab_text: usize,
    vocab_image: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    mlp_intermediate: usize,
    max_text_len: usize,
    image_grid_h: usize,
    image_grid_w: usize,
    bos_token: u32,
    sep_image_token: u32,
    eos_token: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
}

impl From<&ImageTransformerConfig> for ConfigJson {
    fn from(c: &ImageTransformerConfig) -> Self {
        Self {
            vocab_text: c.vocab_text,
            vocab_image: c.vocab_image,
            hidden_size: c.hidden_size,
            num_layers: c.num_layers,
            num_heads: c.num_heads,
            mlp_intermediate: c.mlp_intermediate,
            max_text_len: c.max_text_len,
            image_grid_h: c.image_grid_h,
            image_grid_w: c.image_grid_w,
            bos_token: c.bos_token,
            sep_image_token: c.sep_image_token,
            eos_token: c.eos_token,
            rms_norm_eps: c.rms_norm_eps,
            rope_theta: c.rope_theta,
        }
    }
}

impl From<ConfigJson> for ImageTransformerConfig {
    fn from(c: ConfigJson) -> Self {
        Self {
            vocab_text: c.vocab_text,
            vocab_image: c.vocab_image,
            hidden_size: c.hidden_size,
            num_layers: c.num_layers,
            num_heads: c.num_heads,
            mlp_intermediate: c.mlp_intermediate,
            max_text_len: c.max_text_len,
            image_grid_h: c.image_grid_h,
            image_grid_w: c.image_grid_w,
            bos_token: c.bos_token,
            sep_image_token: c.sep_image_token,
            eos_token: c.eos_token,
            rms_norm_eps: c.rms_norm_eps,
            rope_theta: c.rope_theta,
        }
    }
}

/// Helper: turn an `[f32]` slice into bytes for safetensors. Safetensors
/// owns the byte data via the metadata header so we wrap the borrowed
/// bytes in a `Vec<u8>`.
fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v { out.extend_from_slice(&x.to_le_bytes()); }
    out
}

fn bytes_to_f32(bytes: &[u8], dtype: Dtype, n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    match dtype {
        Dtype::F32 => {
            if bytes.len() != n * 4 {
                return Err(ImageGenError::ShapeMismatch {
                    expected: format!("{} f32 bytes", n * 4),
                    got: format!("{}", bytes.len()),
                });
            }
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        other => return Err(ImageGenError::Unsupported(format!("dtype {other:?}"))),
    }
    Ok(out)
}

/// Save just the model weights (no optimizer state) to `path`.
pub fn save_model(model: &ImageTransformer, path: &Path) -> Result<()> {
    let mut storage: HashMap<String, Vec<u8>> = HashMap::new();
    storage.insert("embed".into(), f32_to_bytes(&model.embed));
    storage.insert("pos_embed".into(), f32_to_bytes(&model.pos_embed));
    storage.insert("final_norm.gamma".into(), f32_to_bytes(&model.final_norm.gamma));
    storage.insert("lm_head.weight".into(), f32_to_bytes(&model.lm_head.weight));
    for (i, b) in model.blocks.iter().enumerate() {
        let p = format!("blocks.{i}.");
        storage.insert(format!("{p}norm_attn.gamma"), f32_to_bytes(&b.norm_attn.gamma));
        storage.insert(format!("{p}q_proj.weight"), f32_to_bytes(&b.q_proj.weight));
        storage.insert(format!("{p}k_proj.weight"), f32_to_bytes(&b.k_proj.weight));
        storage.insert(format!("{p}v_proj.weight"), f32_to_bytes(&b.v_proj.weight));
        storage.insert(format!("{p}o_proj.weight"), f32_to_bytes(&b.o_proj.weight));
        storage.insert(format!("{p}norm_mlp.gamma"), f32_to_bytes(&b.norm_mlp.gamma));
        storage.insert(format!("{p}mlp_gate.weight"), f32_to_bytes(&b.mlp_gate.weight));
        storage.insert(format!("{p}mlp_up.weight"), f32_to_bytes(&b.mlp_up.weight));
        storage.insert(format!("{p}mlp_down.weight"), f32_to_bytes(&b.mlp_down.weight));
    }
    write_safetensors(&model.config, storage, path)
}

/// Save model + optimizer state to `path` and `path.with_extension("optim.safetensors")`.
pub fn save_with_optimizer(
    model: &ImageTransformer,
    optimizer: &AdamW,
    path: &Path,
) -> Result<()> {
    save_model(model, path)?;
    let optim_path = path.with_extension("optim.safetensors");
    let mut storage: HashMap<String, Vec<u8>> = HashMap::new();
    let dump = |s: &mut HashMap<String, Vec<u8>>, name: String, m: &[f32], v: &[f32]| {
        s.insert(format!("{name}.m"), f32_to_bytes(m));
        s.insert(format!("{name}.v"), f32_to_bytes(v));
    };
    dump(&mut storage, "embed".into(), &optimizer.m.embed, &optimizer.v.embed);
    dump(&mut storage, "pos_embed".into(), &optimizer.m.pos_embed, &optimizer.v.pos_embed);
    dump(&mut storage, "final_norm.gamma".into(),
         &optimizer.m.final_norm_gamma, &optimizer.v.final_norm_gamma);
    dump(&mut storage, "lm_head.weight".into(),
         &optimizer.m.lm_head_w, &optimizer.v.lm_head_w);
    for (i, (mb, vb)) in optimizer.m.blocks.iter().zip(optimizer.v.blocks.iter()).enumerate() {
        let p = format!("blocks.{i}.");
        dump(&mut storage, format!("{p}norm_attn.gamma"), &mb.norm_attn_gamma, &vb.norm_attn_gamma);
        dump(&mut storage, format!("{p}q_proj.weight"), &mb.q_proj_w, &vb.q_proj_w);
        dump(&mut storage, format!("{p}k_proj.weight"), &mb.k_proj_w, &vb.k_proj_w);
        dump(&mut storage, format!("{p}v_proj.weight"), &mb.v_proj_w, &vb.v_proj_w);
        dump(&mut storage, format!("{p}o_proj.weight"), &mb.o_proj_w, &vb.o_proj_w);
        dump(&mut storage, format!("{p}norm_mlp.gamma"), &mb.norm_mlp_gamma, &vb.norm_mlp_gamma);
        dump(&mut storage, format!("{p}mlp_gate.weight"), &mb.mlp_gate_w, &vb.mlp_gate_w);
        dump(&mut storage, format!("{p}mlp_up.weight"), &mb.mlp_up_w, &vb.mlp_up_w);
        dump(&mut storage, format!("{p}mlp_down.weight"), &mb.mlp_down_w, &vb.mlp_down_w);
    }
    // Optimizer step counter goes in metadata.
    let metadata: HashMap<String, String> = vec![
        ("optimizer_step".to_string(), optimizer.step.to_string()),
        ("lr".to_string(), optimizer.config.lr.to_string()),
        ("beta1".to_string(), optimizer.config.beta1.to_string()),
        ("beta2".to_string(), optimizer.config.beta2.to_string()),
        ("eps".to_string(), optimizer.config.eps.to_string()),
        ("weight_decay".to_string(), optimizer.config.weight_decay.to_string()),
    ].into_iter().collect();
    write_safetensors_with_metadata(storage, &optim_path, Some(metadata))?;
    Ok(())
}

/// Load model weights from `path`. Reconstructs the architecture from
/// the file's metadata (`klearu_image_config` key holds the JSON).
pub fn load_model(path: &Path) -> Result<ImageTransformer> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    // Recover config from metadata.
    let (_, raw_meta) = SafeTensors::read_metadata(&bytes)?;
    let metadata = raw_meta.metadata().clone().unwrap_or_default();
    let cfg_json = metadata.get("klearu_image_config")
        .ok_or_else(|| ImageGenError::MissingTensor("metadata.klearu_image_config".into()))?;
    let cfg: ConfigJson = serde_json::from_str(cfg_json)
        .map_err(|e| ImageGenError::Config(format!("config parse: {e}")))?;
    let mut model = ImageTransformer::from_config(cfg.into());
    fill_model_from_st(&mut model, &st)?;
    Ok(model)
}

/// Load model + optimizer. Returns the trained-up AdamW state.
pub fn load_with_optimizer(path: &Path) -> Result<(ImageTransformer, AdamW)> {
    let model = load_model(path)?;
    let optim_path = path.with_extension("optim.safetensors");
    let bytes = std::fs::read(&optim_path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let (_, raw_meta) = SafeTensors::read_metadata(&bytes)?;
    let metadata = raw_meta.metadata().clone().unwrap_or_default();
    let opt_cfg = crate::optim::AdamWConfig {
        lr: metadata.get("lr").and_then(|s| s.parse().ok()).unwrap_or(3e-4),
        beta1: metadata.get("beta1").and_then(|s| s.parse().ok()).unwrap_or(0.9),
        beta2: metadata.get("beta2").and_then(|s| s.parse().ok()).unwrap_or(0.95),
        eps: metadata.get("eps").and_then(|s| s.parse().ok()).unwrap_or(1e-8),
        weight_decay: metadata.get("weight_decay").and_then(|s| s.parse().ok()).unwrap_or(0.1),
    };
    let mut opt = AdamW::new(&model, opt_cfg);
    opt.step = metadata.get("optimizer_step").and_then(|s| s.parse().ok()).unwrap_or(0);
    fill_optimizer_from_st(&mut opt, &st)?;
    Ok((model, opt))
}

fn fill_model_from_st(model: &mut ImageTransformer, st: &SafeTensors) -> Result<()> {
    fill_vec(st, "embed", &mut model.embed)?;
    fill_vec(st, "pos_embed", &mut model.pos_embed)?;
    fill_vec(st, "final_norm.gamma", &mut model.final_norm.gamma)?;
    fill_vec(st, "lm_head.weight", &mut model.lm_head.weight)?;
    for (i, b) in model.blocks.iter_mut().enumerate() {
        let p = format!("blocks.{i}.");
        fill_vec(st, &format!("{p}norm_attn.gamma"), &mut b.norm_attn.gamma)?;
        fill_vec(st, &format!("{p}q_proj.weight"), &mut b.q_proj.weight)?;
        fill_vec(st, &format!("{p}k_proj.weight"), &mut b.k_proj.weight)?;
        fill_vec(st, &format!("{p}v_proj.weight"), &mut b.v_proj.weight)?;
        fill_vec(st, &format!("{p}o_proj.weight"), &mut b.o_proj.weight)?;
        fill_vec(st, &format!("{p}norm_mlp.gamma"), &mut b.norm_mlp.gamma)?;
        fill_vec(st, &format!("{p}mlp_gate.weight"), &mut b.mlp_gate.weight)?;
        fill_vec(st, &format!("{p}mlp_up.weight"), &mut b.mlp_up.weight)?;
        fill_vec(st, &format!("{p}mlp_down.weight"), &mut b.mlp_down.weight)?;
    }
    Ok(())
}

fn fill_optimizer_from_st(opt: &mut AdamW, st: &SafeTensors) -> Result<()> {
    fill_pair(st, "embed", &mut opt.m.embed, &mut opt.v.embed)?;
    fill_pair(st, "pos_embed", &mut opt.m.pos_embed, &mut opt.v.pos_embed)?;
    fill_pair(st, "final_norm.gamma", &mut opt.m.final_norm_gamma, &mut opt.v.final_norm_gamma)?;
    fill_pair(st, "lm_head.weight", &mut opt.m.lm_head_w, &mut opt.v.lm_head_w)?;
    let nb = opt.m.blocks.len();
    for i in 0..nb {
        let p = format!("blocks.{i}.");
        let (mb, vb) = (&mut opt.m.blocks[i], &mut opt.v.blocks[i]);
        fill_pair(st, &format!("{p}norm_attn.gamma"), &mut mb.norm_attn_gamma, &mut vb.norm_attn_gamma)?;
        fill_pair(st, &format!("{p}q_proj.weight"), &mut mb.q_proj_w, &mut vb.q_proj_w)?;
        fill_pair(st, &format!("{p}k_proj.weight"), &mut mb.k_proj_w, &mut vb.k_proj_w)?;
        fill_pair(st, &format!("{p}v_proj.weight"), &mut mb.v_proj_w, &mut vb.v_proj_w)?;
        fill_pair(st, &format!("{p}o_proj.weight"), &mut mb.o_proj_w, &mut vb.o_proj_w)?;
        fill_pair(st, &format!("{p}norm_mlp.gamma"), &mut mb.norm_mlp_gamma, &mut vb.norm_mlp_gamma)?;
        fill_pair(st, &format!("{p}mlp_gate.weight"), &mut mb.mlp_gate_w, &mut vb.mlp_gate_w)?;
        fill_pair(st, &format!("{p}mlp_up.weight"), &mut mb.mlp_up_w, &mut vb.mlp_up_w)?;
        fill_pair(st, &format!("{p}mlp_down.weight"), &mut mb.mlp_down_w, &mut vb.mlp_down_w)?;
    }
    Ok(())
}

fn fill_vec(st: &SafeTensors, name: &str, dst: &mut Vec<f32>) -> Result<()> {
    let t = st.tensor(name)
        .map_err(|_| ImageGenError::MissingTensor(name.to_string()))?;
    let parsed = bytes_to_f32(t.data(), t.dtype(), dst.len())?;
    dst.copy_from_slice(&parsed);
    Ok(())
}

fn fill_pair(st: &SafeTensors, base: &str, m: &mut Vec<f32>, v: &mut Vec<f32>) -> Result<()> {
    fill_vec(st, &format!("{base}.m"), m)?;
    fill_vec(st, &format!("{base}.v"), v)?;
    Ok(())
}

fn write_safetensors(
    cfg: &ImageTransformerConfig,
    storage: HashMap<String, Vec<u8>>,
    path: &Path,
) -> Result<()> {
    let cfg_json = serde_json::to_string(&ConfigJson::from(cfg))
        .map_err(|e| ImageGenError::Config(format!("config serialize: {e}")))?;
    let mut metadata: HashMap<String, String> = HashMap::new();
    metadata.insert("klearu_image_config".to_string(), cfg_json);
    metadata.insert("klearu_image_version".to_string(), "0.1.0".to_string());
    write_safetensors_with_metadata(storage, path, Some(metadata))
}

fn write_safetensors_with_metadata(
    storage: HashMap<String, Vec<u8>>,
    path: &Path,
    metadata: Option<HashMap<String, String>>,
) -> Result<()> {
    // Build TensorView entries. Each view describes a 1-D tensor (we
    // flatten to vectors at write-time — the model's shape information
    // is implicit in the config).
    let mut views: Vec<(String, TensorView)> = Vec::with_capacity(storage.len());
    for (name, bytes) in storage.iter() {
        let n = bytes.len() / 4;
        let view = TensorView::new(Dtype::F32, vec![n], bytes.as_slice())
            .map_err(|e| ImageGenError::Unsupported(format!("tensor view: {e}")))?;
        views.push((name.clone(), view));
    }
    safetensors::serialize_to_file(views, &metadata, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grad::Gradients;
    use crate::optim::AdamWConfig;

    fn tiny_cfg() -> ImageTransformerConfig {
        ImageTransformerConfig {
            vocab_text: 100, vocab_image: 16,
            hidden_size: 16, num_layers: 1, num_heads: 4,
            mlp_intermediate: 32, max_text_len: 4,
            image_grid_h: 2, image_grid_w: 2,
            bos_token: 100, sep_image_token: 101, eos_token: 102,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
        }
    }

    #[test]
    fn model_round_trip() {
        let cfg = tiny_cfg();
        let mut model = ImageTransformer::from_config(cfg.clone());
        // Put recognizable values in.
        for (i, w) in model.embed.iter_mut().enumerate() { *w = i as f32 * 0.01; }
        for (i, w) in model.lm_head.weight.iter_mut().enumerate() { *w = -(i as f32) * 0.02; }
        for b in model.blocks.iter_mut() {
            for (i, w) in b.q_proj.weight.iter_mut().enumerate() { *w = i as f32 * 0.001; }
        }
        let tmp = std::env::temp_dir().join("klearu_image_ckpt_test.safetensors");
        save_model(&model, &tmp).expect("save");
        let loaded = load_model(&tmp).expect("load");
        // Verify identical.
        assert_eq!(loaded.embed, model.embed);
        assert_eq!(loaded.lm_head.weight, model.lm_head.weight);
        assert_eq!(loaded.blocks[0].q_proj.weight, model.blocks[0].q_proj.weight);
        // Config preserved.
        assert_eq!(loaded.config.hidden_size, cfg.hidden_size);
        assert_eq!(loaded.config.num_layers, cfg.num_layers);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn optimizer_round_trip() {
        let cfg = tiny_cfg();
        let mut model = ImageTransformer::from_config(cfg.clone());
        for w in model.embed.iter_mut() { *w = 0.1; }
        let mut opt = AdamW::new(&model, AdamWConfig::default());
        // Take one step to populate m and v.
        let mut grad = Gradients::zeros_for(&model);
        for g in grad.embed.iter_mut() { *g = 0.05; }
        opt.step(&mut model, &grad);

        let tmp = std::env::temp_dir().join("klearu_image_opt_test.safetensors");
        save_with_optimizer(&model, &opt, &tmp).expect("save");
        let (m2, o2) = load_with_optimizer(&tmp).expect("load");
        assert_eq!(m2.embed[0], model.embed[0], "model weights round-trip");
        assert!((o2.m.embed[0] - opt.m.embed[0]).abs() < 1e-7,
            "optimizer m round-trip: {} vs {}", o2.m.embed[0], opt.m.embed[0]);
        assert_eq!(o2.step, opt.step, "step counter round-trip");
        std::fs::remove_file(&tmp).ok();
        std::fs::remove_file(tmp.with_extension("optim.safetensors")).ok();
    }
}
