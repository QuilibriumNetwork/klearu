/// Weight loader for Qwen3.5 Vision Encoder.
///
/// Loads `model.visual.*` weights from Qwen3.5 VLM safetensors files.
/// Handles Conv3d→Conv2d reshape for patch embedding.

use std::path::Path;

use crate::config::QwenVisionConfig;
use crate::error::{VisionError, Result};
use crate::model::qwen_vision::QwenVisionEncoder;
use super::loader::{SafeTensorsFile, load_2d_into_store, load_1d_into_vec};

/// Load a Qwen3.5 vision encoder from safetensors files.
///
/// `safetensors_paths`: paths to the safetensors files containing vision weights.
/// `prefix`: weight name prefix, e.g. `"model.visual."`.
pub fn load_qwen_vision_encoder(
    safetensors_paths: &[&Path],
    prefix: &str,
    config: QwenVisionConfig,
) -> Result<QwenVisionEncoder> {
    let mut encoder = QwenVisionEncoder::new(config.clone());

    for st_path in safetensors_paths {
        let st = SafeTensorsFile::open(st_path)?;

        for (name, info) in &st.tensors {
            let suffix = match name.strip_prefix(prefix) {
                Some(s) => s,
                None => continue,
            };

            let data = st.tensor_to_f32(name)?;
            load_qwen_weight(&mut encoder, suffix, &data, &info.shape, &config)?;
        }
    }

    Ok(encoder)
}

/// Load a Qwen3.5 vision encoder from a model directory.
///
/// Reads config.json for vision_config and loads weights from safetensors.
pub fn load_qwen_vision_from_dir(model_dir: &Path) -> Result<QwenVisionEncoder> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config = QwenVisionConfig::from_hf_config(&config_str)?;

    tracing::info!(
        "Loading Qwen Vision: depth={}, hidden={}, out_hidden={}",
        config.depth, config.hidden_size, config.out_hidden_size,
    );

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

    let paths: Vec<&Path> = st_files.iter().map(|p| p.as_path()).collect();
    load_qwen_vision_encoder(&paths, "model.visual.", config)
}

fn load_qwen_weight(
    encoder: &mut QwenVisionEncoder,
    suffix: &str,
    data: &[f32],
    shape: &[usize],
    _config: &QwenVisionConfig,
) -> Result<()> {
    match suffix {
        // Patch embedding
        "patch_embed.proj.weight" => {
            // Conv3d weight: [out_ch, in_ch, temporal, kH, kW]
            // Reshape to Conv2d: [out_ch, in_ch*temporal, kH, kW]
            let expected = encoder.patch_embed.weight.len();
            let got: usize = shape.iter().product();
            if got != expected {
                return Err(VisionError::ShapeMismatch {
                    expected: format!("[{expected} total]"),
                    got: format!("[{got} total] (shape {shape:?})"),
                });
            }
            // The reshaping is a no-op on the flat data: Conv3d
            // [out, in, T, H, W] is treated as Conv2d [out, in*T, H, W].
            encoder.patch_embed.weight.copy_from_slice(data);
        }
        "patch_embed.proj.bias" => {
            load_1d_into_vec(encoder.patch_embed.bias.as_mut().unwrap(), data, shape)?;
        }

        // Position embedding
        "pos_embed.weight" => {
            // Shape: [num_position_embeddings, hidden_size]
            let expected = encoder.pos_embed.len();
            if data.len() != expected {
                return Err(VisionError::ShapeMismatch {
                    expected: format!("[{expected} total]"),
                    got: format!("[{} total]", data.len()),
                });
            }
            encoder.pos_embed.copy_from_slice(data);
        }

        // Merger
        "merger.norm.weight" => {
            load_1d_into_vec(&mut encoder.merger.norm.weight, data, shape)?;
        }
        "merger.norm.bias" => {
            load_1d_into_vec(&mut encoder.merger.norm.bias, data, shape)?;
        }
        "merger.linear_fc1.weight" => {
            load_2d_into_store(&mut encoder.merger.fc1.weights, data, shape)?;
        }
        "merger.linear_fc1.bias" => {
            load_1d_into_vec(&mut encoder.merger.fc1.bias, data, shape)?;
        }
        "merger.linear_fc2.weight" => {
            load_2d_into_store(&mut encoder.merger.fc2.weights, data, shape)?;
        }
        "merger.linear_fc2.bias" => {
            load_1d_into_vec(&mut encoder.merger.fc2.bias, data, shape)?;
        }

        _ => {
            // Transformer blocks: blocks.{i}.{component}
            if let Some(rest) = suffix.strip_prefix("blocks.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(block_idx) = rest[..dot_pos].parse::<usize>() {
                        let component = &rest[dot_pos + 1..];
                        if block_idx < encoder.blocks.len() {
                            load_block_weight(
                                &mut encoder.blocks[block_idx],
                                component,
                                data,
                                shape,
                            )?;
                        }
                    }
                }
            } else {
                tracing::debug!("Skipping unknown Qwen vision weight: {suffix}");
            }
        }
    }
    Ok(())
}

fn load_block_weight(
    block: &mut crate::model::vit::ViTBlock,
    component: &str,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    match component {
        "norm1.weight" => load_1d_into_vec(&mut block.norm1.weight, data, shape),
        "norm1.bias" => load_1d_into_vec(&mut block.norm1.bias, data, shape),
        "norm2.weight" => load_1d_into_vec(&mut block.norm2.weight, data, shape),
        "norm2.bias" => load_1d_into_vec(&mut block.norm2.bias, data, shape),

        "attn.qkv.weight" => load_2d_into_store(&mut block.attn.qkv.weights, data, shape),
        "attn.qkv.bias" => load_1d_into_vec(&mut block.attn.qkv.bias, data, shape),
        "attn.proj.weight" => load_2d_into_store(&mut block.attn.proj.weights, data, shape),
        "attn.proj.bias" => load_1d_into_vec(&mut block.attn.proj.bias, data, shape),

        // Qwen uses linear_fc1/linear_fc2 naming for MLP
        "mlp.linear_fc1.weight" | "mlp.fc1.weight" => {
            load_2d_into_store(&mut block.mlp_fc1.weights, data, shape)
        }
        "mlp.linear_fc1.bias" | "mlp.fc1.bias" => {
            load_1d_into_vec(&mut block.mlp_fc1.bias, data, shape)
        }
        "mlp.linear_fc2.weight" | "mlp.fc2.weight" => {
            load_2d_into_store(&mut block.mlp_fc2.weights, data, shape)
        }
        "mlp.linear_fc2.bias" | "mlp.fc2.bias" => {
            load_1d_into_vec(&mut block.mlp_fc2.bias, data, shape)
        }

        _ => {
            tracing::debug!("Skipping unknown block component: {component}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_parse_block_suffix() {
        // Just verify the suffix parsing logic works
        let suffix = "blocks.5.attn.qkv.weight";
        let rest = suffix.strip_prefix("blocks.").unwrap();
        let dot_pos = rest.find('.').unwrap();
        let idx: usize = rest[..dot_pos].parse().unwrap();
        let component = &rest[dot_pos + 1..];
        assert_eq!(idx, 5);
        assert_eq!(component, "attn.qkv.weight");
    }
}
