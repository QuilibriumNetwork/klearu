/// Weight loader for Hiera (SAM2) models.
///
/// Maps Hiera weight names to `HieraModel`.

use std::path::Path;

use crate::config::HieraConfig;
use crate::error::{VisionError, Result};
use crate::model::hiera::HieraModel;

use super::loader::{SafeTensorsFile, load_2d_into_store, load_1d_into_vec};

/// Load a Hiera model from a timm model directory.
pub fn load_hiera_model(model_dir: &Path, config: HieraConfig) -> Result<HieraModel> {
    tracing::info!(
        "Loading Hiera: embed_dims={:?}, depths={:?}",
        config.embed_dims, config.depths,
    );

    let mut model = HieraModel::new(config);

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
            load_hiera_weight(&mut model, name, &data, &info.shape)?;
        }
    }

    Ok(model)
}

fn load_hiera_weight(
    model: &mut HieraModel,
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
            tracing::debug!("Skipping unhandled Hiera weight: {name}");
        }
    }
    Ok(())
}
