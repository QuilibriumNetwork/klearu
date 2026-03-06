/// Weight loader for ConvNeXt models.
///
/// Maps timm ConvNeXt weight names to `ConvNextModel`.

use std::path::Path;

use crate::config::ConvNextConfig;
use crate::error::{VisionError, Result};
use crate::model::convnext::ConvNextModel;

use super::loader::{SafeTensorsFile, load_2d_into_store, load_1d_into_vec};

/// Load a ConvNeXt model from a timm model directory.
pub fn load_convnext_model(model_dir: &Path, config: ConvNextConfig) -> Result<ConvNextModel> {
    tracing::info!(
        "Loading ConvNeXt: dims={:?}, depths={:?}, v2={}",
        config.dims, config.depths, config.is_v2,
    );

    let mut model = ConvNextModel::new(config);

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
            load_convnext_weight(&mut model, name, &data, &info.shape)?;
        }
    }

    Ok(model)
}

fn load_convnext_weight(
    model: &mut ConvNextModel,
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    match name {
        "stem.0.weight" => {
            model.stem_conv.weight.copy_from_slice(data);
        }
        "stem.0.bias" => {
            load_1d_into_vec(model.stem_conv.bias.as_mut().unwrap(), data, shape)?;
        }
        "stem.1.weight" => {
            load_1d_into_vec(&mut model.stem_norm.weight, data, shape)?;
        }
        "stem.1.bias" => {
            load_1d_into_vec(&mut model.stem_norm.bias, data, shape)?;
        }
        "head.fc.weight" | "head.weight" => {
            load_2d_into_store(&mut model.head.weights, data, shape)?;
        }
        "head.fc.bias" | "head.bias" => {
            load_1d_into_vec(&mut model.head.bias, data, shape)?;
        }
        "norm.weight" | "head.norm.weight" => {
            load_1d_into_vec(&mut model.final_norm.weight, data, shape)?;
        }
        "norm.bias" | "head.norm.bias" => {
            load_1d_into_vec(&mut model.final_norm.bias, data, shape)?;
        }
        _ => {
            // Downsample and stage block weights
            // Pattern: stages.{s}.downsample.{0=norm,1=conv}.{weight,bias}
            // Pattern: stages.{s}.blocks.{b}.{component}
            tracing::debug!("Skipping unhandled ConvNeXt weight: {name}");
        }
    }
    Ok(())
}
