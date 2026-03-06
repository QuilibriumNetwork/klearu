/// Weight loader for Swin Transformer models.
///
/// Maps timm Swin weight names to `SwinModel`.

use std::path::Path;

use crate::config::SwinConfig;
use crate::error::{VisionError, Result};
use crate::model::swin::SwinModel;

use super::loader::{SafeTensorsFile, load_2d_into_store, load_1d_into_vec};

/// Load a Swin model from a timm model directory.
pub fn load_swin_model(model_dir: &Path, config: SwinConfig) -> Result<SwinModel> {
    tracing::info!(
        "Loading Swin: embed_dims={:?}, depths={:?}, window_size={}",
        config.embed_dims, config.depths, config.window_size,
    );

    let mut model = SwinModel::new(config);

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
            load_swin_weight(&mut model, name, &data, &info.shape)?;
        }
    }

    Ok(model)
}

fn load_swin_weight(
    model: &mut SwinModel,
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    match name {
        "patch_embed.proj.weight" => {
            model.patch_embed.weight.copy_from_slice(data);
        }
        "patch_embed.proj.bias" => {
            load_1d_into_vec(model.patch_embed.bias.as_mut().unwrap(), data, shape)?;
        }
        "patch_embed.norm.weight" => {
            load_1d_into_vec(&mut model.patch_norm.weight, data, shape)?;
        }
        "patch_embed.norm.bias" => {
            load_1d_into_vec(&mut model.patch_norm.bias, data, shape)?;
        }
        "norm.weight" => {
            load_1d_into_vec(&mut model.final_norm.weight, data, shape)?;
        }
        "norm.bias" => {
            load_1d_into_vec(&mut model.final_norm.bias, data, shape)?;
        }
        "head.weight" | "head.fc.weight" => {
            load_2d_into_store(&mut model.head.weights, data, shape)?;
        }
        "head.bias" | "head.fc.bias" => {
            load_1d_into_vec(&mut model.head.bias, data, shape)?;
        }
        _ => {
            // Stage/block weights
            // Pattern: layers.{s}.blocks.{b}.{component}
            // Pattern: layers.{s}.downsample.{component}
            tracing::debug!("Skipping unhandled Swin weight: {name}");
        }
    }
    Ok(())
}
