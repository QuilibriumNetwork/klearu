/// Weight loader for DINOv2 models.
///
/// DINOv2 is a standard ViT with optional register tokens.
/// The loader extends `ViTModel` by setting `pool_type = Mean`
/// and handling register tokens if present.

use std::path::Path;

use crate::config::ViTConfig;
use crate::error::{VisionError, Result};
use crate::model::vit::{ViTModel, PoolType};

use super::loader::{SafeTensorsFile, load_2d_into_store, load_1d_into_vec};

/// Load a DINOv2 model from a timm model directory.
///
/// DINOv2 uses CLS token but pools with Mean over patch tokens (ignoring CLS).
/// Register tokens (if present) are loaded but discarded during pooling.
pub fn load_dinov2_model(model_dir: &Path) -> Result<ViTModel> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let mut config = ViTConfig::from_timm_config(&config_str)?;
    config.pool_type = PoolType::Cls;

    tracing::info!(
        "Loading DINOv2: embed_dim={}, depth={}, patch_size={}",
        config.embed_dim, config.num_layers, config.patch_size,
    );

    let mut model = ViTModel::new(config);

    let mut st_files = Vec::new();
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            st_files.push(path);
        }
    }

    if st_files.is_empty() {
        return Err(VisionError::WeightLoad("No .safetensors files found".into()));
    }
    st_files.sort();

    for st_path in &st_files {
        let st = SafeTensorsFile::open(st_path)?;
        for (name, info) in &st.tensors {
            let data = st.tensor_to_f32(name)?;
            load_dinov2_weight(&mut model, name, &data, &info.shape)?;
        }
    }

    Ok(model)
}

fn load_dinov2_weight(
    model: &mut ViTModel,
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    match name {
        "patch_embed.proj.weight" => {
            let expected = model.patch_embed.proj.weight.len();
            let got: usize = shape.iter().product();
            if got != expected {
                return Err(VisionError::ShapeMismatch {
                    expected: format!("[{expected}]"), got: format!("[{got}]"),
                });
            }
            model.patch_embed.proj.weight.copy_from_slice(data);
        }
        "patch_embed.proj.bias" => {
            load_1d_into_vec(model.patch_embed.proj.bias.as_mut().unwrap(), data, shape)?;
        }
        "cls_token" => {
            let embed_dim = model.config.embed_dim;
            if data.len() >= embed_dim {
                model.cls_token.copy_from_slice(&data[data.len() - embed_dim..]);
            }
        }
        "pos_embed" => {
            let expected = model.pos_embed.len();
            if data.len() >= expected {
                model.pos_embed.copy_from_slice(&data[data.len() - expected..]);
            }
        }
        "register_tokens" => {
            // DINOv2 register tokens — stored but not used in standard forward
            tracing::debug!("Skipping register_tokens (shape {:?})", shape);
        }
        "norm.weight" | "fc_norm.weight" => {
            load_1d_into_vec(&mut model.norm.weight, data, shape)?;
        }
        "norm.bias" | "fc_norm.bias" => {
            load_1d_into_vec(&mut model.norm.bias, data, shape)?;
        }
        "head.weight" | "head.fc.weight" => {
            load_2d_into_store(&mut model.head.weights, data, shape)?;
        }
        "head.bias" | "head.fc.bias" => {
            load_1d_into_vec(&mut model.head.bias, data, shape)?;
        }
        _ => {
            if let Some(rest) = name.strip_prefix("blocks.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                        let suffix = &rest[dot_pos + 1..];
                        if idx < model.blocks.len() {
                            load_block_weight(&mut model.blocks[idx], suffix, data, shape)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn load_block_weight(
    block: &mut crate::model::vit::ViTBlock,
    suffix: &str,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    match suffix {
        "norm1.weight" | "ls1.gamma" => load_1d_into_vec(&mut block.norm1.weight, data, shape),
        "norm1.bias" => load_1d_into_vec(&mut block.norm1.bias, data, shape),
        "norm2.weight" | "ls2.gamma" => load_1d_into_vec(&mut block.norm2.weight, data, shape),
        "norm2.bias" => load_1d_into_vec(&mut block.norm2.bias, data, shape),
        "attn.qkv.weight" => load_2d_into_store(&mut block.attn.qkv.weights, data, shape),
        "attn.qkv.bias" => load_1d_into_vec(&mut block.attn.qkv.bias, data, shape),
        "attn.proj.weight" => load_2d_into_store(&mut block.attn.proj.weights, data, shape),
        "attn.proj.bias" => load_1d_into_vec(&mut block.attn.proj.bias, data, shape),
        "mlp.fc1.weight" => load_2d_into_store(&mut block.mlp_fc1.weights, data, shape),
        "mlp.fc1.bias" => load_1d_into_vec(&mut block.mlp_fc1.bias, data, shape),
        "mlp.fc2.weight" => load_2d_into_store(&mut block.mlp_fc2.weights, data, shape),
        "mlp.fc2.bias" => load_1d_into_vec(&mut block.mlp_fc2.bias, data, shape),
        _ => Ok(()),
    }
}
