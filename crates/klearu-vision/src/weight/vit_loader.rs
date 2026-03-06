use std::path::Path;

use crate::config::ViTConfig;
use crate::error::{VisionError, Result};
use crate::model::vit::ViTModel;

use super::loader::{SafeTensorsFile, load_2d_into_store, load_1d_into_vec};

/// Identifies where a timm ViT weight tensor should be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ViTWeightTarget {
    PatchEmbedProjWeight,
    PatchEmbedProjBias,
    ClsToken,
    PosEmbed,
    BlockNorm1Weight(usize),
    BlockNorm1Bias(usize),
    BlockAttnQkvWeight(usize),
    BlockAttnQkvBias(usize),
    BlockAttnProjWeight(usize),
    BlockAttnProjBias(usize),
    BlockNorm2Weight(usize),
    BlockNorm2Bias(usize),
    BlockMlpFc1Weight(usize),
    BlockMlpFc1Bias(usize),
    BlockMlpFc2Weight(usize),
    BlockMlpFc2Bias(usize),
    NormWeight,
    NormBias,
    HeadWeight,
    HeadBias,
}

/// Parse a timm ViT weight name.
///
/// Supported patterns:
/// ```text
/// patch_embed.proj.weight / .bias
/// cls_token
/// pos_embed
/// blocks.{i}.norm1.weight / .bias
/// blocks.{i}.attn.qkv.weight / .bias
/// blocks.{i}.attn.proj.weight / .bias
/// blocks.{i}.norm2.weight / .bias
/// blocks.{i}.mlp.fc1.weight / .bias
/// blocks.{i}.mlp.fc2.weight / .bias
/// norm.weight / .bias
/// head.weight / .bias
/// head.fc.weight / .bias
/// ```
fn parse_vit_weight_name(name: &str) -> Option<ViTWeightTarget> {
    match name {
        "patch_embed.proj.weight" => return Some(ViTWeightTarget::PatchEmbedProjWeight),
        "patch_embed.proj.bias" => return Some(ViTWeightTarget::PatchEmbedProjBias),
        "cls_token" => return Some(ViTWeightTarget::ClsToken),
        "pos_embed" => return Some(ViTWeightTarget::PosEmbed),
        "norm.weight" | "fc_norm.weight" => return Some(ViTWeightTarget::NormWeight),
        "norm.bias" | "fc_norm.bias" => return Some(ViTWeightTarget::NormBias),
        "head.weight" | "head.fc.weight" => return Some(ViTWeightTarget::HeadWeight),
        "head.bias" | "head.fc.bias" => return Some(ViTWeightTarget::HeadBias),
        _ => {}
    }

    // blocks.{i}.*
    let rest = name.strip_prefix("blocks.")?;
    let dot_pos = rest.find('.')?;
    let block_idx: usize = rest[..dot_pos].parse().ok()?;
    let suffix = &rest[dot_pos + 1..];

    match suffix {
        "norm1.weight" => Some(ViTWeightTarget::BlockNorm1Weight(block_idx)),
        "norm1.bias" => Some(ViTWeightTarget::BlockNorm1Bias(block_idx)),
        "attn.qkv.weight" => Some(ViTWeightTarget::BlockAttnQkvWeight(block_idx)),
        "attn.qkv.bias" => Some(ViTWeightTarget::BlockAttnQkvBias(block_idx)),
        "attn.proj.weight" => Some(ViTWeightTarget::BlockAttnProjWeight(block_idx)),
        "attn.proj.bias" => Some(ViTWeightTarget::BlockAttnProjBias(block_idx)),
        "norm2.weight" | "ls2.gamma" => Some(ViTWeightTarget::BlockNorm2Weight(block_idx)),
        "norm2.bias" => Some(ViTWeightTarget::BlockNorm2Bias(block_idx)),
        "mlp.fc1.weight" => Some(ViTWeightTarget::BlockMlpFc1Weight(block_idx)),
        "mlp.fc1.bias" => Some(ViTWeightTarget::BlockMlpFc1Bias(block_idx)),
        "mlp.fc2.weight" => Some(ViTWeightTarget::BlockMlpFc2Weight(block_idx)),
        "mlp.fc2.bias" => Some(ViTWeightTarget::BlockMlpFc2Bias(block_idx)),
        _ => None,
    }
}

/// Load a ViT model from a timm model directory.
///
/// Expects `config.json` and one or more `.safetensors` files.
pub fn load_vit_model(model_dir: &Path) -> Result<ViTModel> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config = ViTConfig::from_timm_config(&config_str)?;

    tracing::info!(
        "Loading ViT: embed_dim={}, depth={}, num_heads={}, patch_size={}, classes={}",
        config.embed_dim, config.num_layers, config.num_heads, config.patch_size, config.num_classes,
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
        return Err(VisionError::WeightLoad(
            "No .safetensors files found in model directory".into(),
        ));
    }

    st_files.sort();

    for st_path in &st_files {
        tracing::info!("Loading weights from {:?}", st_path);
        let st = SafeTensorsFile::open(st_path)?;

        for (name, info) in &st.tensors {
            let target = match parse_vit_weight_name(name) {
                Some(t) => t,
                None => {
                    tracing::debug!("Skipping unknown weight: {name}");
                    continue;
                }
            };

            let data = st.tensor_to_f32(name)?;
            load_vit_weight(&mut model, &target, &data, &info.shape)?;
        }
    }

    Ok(model)
}

fn load_vit_weight(
    model: &mut ViTModel,
    target: &ViTWeightTarget,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    match target {
        ViTWeightTarget::PatchEmbedProjWeight => {
            // Conv weight: [out_channels, in_channels, kH, kW]
            let expected = model.patch_embed.proj.weight.len();
            let got: usize = shape.iter().product();
            if got != expected {
                return Err(VisionError::ShapeMismatch {
                    expected: format!("[{expected} total]"),
                    got: format!("[{got} total] (shape {shape:?})"),
                });
            }
            model.patch_embed.proj.weight.copy_from_slice(data);
        }
        ViTWeightTarget::PatchEmbedProjBias => {
            load_1d_into_vec(model.patch_embed.proj.bias.as_mut().unwrap(), data, shape)?;
        }
        ViTWeightTarget::ClsToken => {
            // cls_token shape: [1, 1, embed_dim] or [1, embed_dim]
            let embed_dim = model.config.embed_dim;
            if data.len() < embed_dim {
                return Err(VisionError::ShapeMismatch {
                    expected: format!("[..., {embed_dim}]"),
                    got: format!("{} total", data.len()),
                });
            }
            // Take last embed_dim values (handles [1,1,D] or [1,D])
            model.cls_token.copy_from_slice(&data[data.len() - embed_dim..]);
        }
        ViTWeightTarget::PosEmbed => {
            // pos_embed shape: [1, seq_len, embed_dim] or [seq_len, embed_dim]
            let expected = model.pos_embed.len();
            if data.len() < expected {
                return Err(VisionError::ShapeMismatch {
                    expected: format!("[{expected} total]"),
                    got: format!("{} total", data.len()),
                });
            }
            // Take last `expected` values (handles leading batch dim)
            model.pos_embed.copy_from_slice(&data[data.len() - expected..]);
        }
        ViTWeightTarget::BlockNorm1Weight(i) => {
            load_1d_into_vec(&mut model.blocks[*i].norm1.weight, data, shape)?;
        }
        ViTWeightTarget::BlockNorm1Bias(i) => {
            load_1d_into_vec(&mut model.blocks[*i].norm1.bias, data, shape)?;
        }
        ViTWeightTarget::BlockAttnQkvWeight(i) => {
            load_2d_into_store(&mut model.blocks[*i].attn.qkv.weights, data, shape)?;
        }
        ViTWeightTarget::BlockAttnQkvBias(i) => {
            load_1d_into_vec(&mut model.blocks[*i].attn.qkv.bias, data, shape)?;
        }
        ViTWeightTarget::BlockAttnProjWeight(i) => {
            load_2d_into_store(&mut model.blocks[*i].attn.proj.weights, data, shape)?;
        }
        ViTWeightTarget::BlockAttnProjBias(i) => {
            load_1d_into_vec(&mut model.blocks[*i].attn.proj.bias, data, shape)?;
        }
        ViTWeightTarget::BlockNorm2Weight(i) => {
            load_1d_into_vec(&mut model.blocks[*i].norm2.weight, data, shape)?;
        }
        ViTWeightTarget::BlockNorm2Bias(i) => {
            load_1d_into_vec(&mut model.blocks[*i].norm2.bias, data, shape)?;
        }
        ViTWeightTarget::BlockMlpFc1Weight(i) => {
            load_2d_into_store(&mut model.blocks[*i].mlp_fc1.weights, data, shape)?;
        }
        ViTWeightTarget::BlockMlpFc1Bias(i) => {
            load_1d_into_vec(&mut model.blocks[*i].mlp_fc1.bias, data, shape)?;
        }
        ViTWeightTarget::BlockMlpFc2Weight(i) => {
            load_2d_into_store(&mut model.blocks[*i].mlp_fc2.weights, data, shape)?;
        }
        ViTWeightTarget::BlockMlpFc2Bias(i) => {
            load_1d_into_vec(&mut model.blocks[*i].mlp_fc2.bias, data, shape)?;
        }
        ViTWeightTarget::NormWeight => {
            load_1d_into_vec(&mut model.norm.weight, data, shape)?;
        }
        ViTWeightTarget::NormBias => {
            load_1d_into_vec(&mut model.norm.bias, data, shape)?;
        }
        ViTWeightTarget::HeadWeight => {
            load_2d_into_store(&mut model.head.weights, data, shape)?;
        }
        ViTWeightTarget::HeadBias => {
            load_1d_into_vec(&mut model.head.bias, data, shape)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vit_weight_names() {
        assert_eq!(
            parse_vit_weight_name("patch_embed.proj.weight"),
            Some(ViTWeightTarget::PatchEmbedProjWeight)
        );
        assert_eq!(
            parse_vit_weight_name("cls_token"),
            Some(ViTWeightTarget::ClsToken)
        );
        assert_eq!(
            parse_vit_weight_name("pos_embed"),
            Some(ViTWeightTarget::PosEmbed)
        );
        assert_eq!(
            parse_vit_weight_name("blocks.0.norm1.weight"),
            Some(ViTWeightTarget::BlockNorm1Weight(0))
        );
        assert_eq!(
            parse_vit_weight_name("blocks.11.attn.qkv.weight"),
            Some(ViTWeightTarget::BlockAttnQkvWeight(11))
        );
        assert_eq!(
            parse_vit_weight_name("blocks.5.mlp.fc1.bias"),
            Some(ViTWeightTarget::BlockMlpFc1Bias(5))
        );
        assert_eq!(
            parse_vit_weight_name("norm.weight"),
            Some(ViTWeightTarget::NormWeight)
        );
        assert_eq!(
            parse_vit_weight_name("head.weight"),
            Some(ViTWeightTarget::HeadWeight)
        );
        assert_eq!(
            parse_vit_weight_name("head.fc.weight"),
            Some(ViTWeightTarget::HeadWeight)
        );
    }

    #[test]
    fn test_parse_unknown_vit_weight() {
        assert_eq!(parse_vit_weight_name("something.random"), None);
        assert_eq!(parse_vit_weight_name("blocks.0.unknown.weight"), None);
    }
}
