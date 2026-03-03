use std::path::Path;

use crate::config::LlmConfig;
use crate::error::{LlmError, Result};
use crate::model::Model;
use crate::model::block::AttentionLayer;

use super::name_map::{parse_weight_name, WeightTarget};
use super::safetensors::SafeTensorsFile;

/// Load a model from a HuggingFace model directory.
///
/// Expects `config.json` and one or more `.safetensors` files in the directory.
pub fn load_model(model_dir: &Path) -> Result<Model> {
    // Load config
    let config_path = model_dir.join("config.json");
    let config = LlmConfig::from_file(&config_path)?;

    tracing::info!(
        "Loading model: {} layers, hidden_size={}, vocab_size={}{}",
        config.num_layers,
        config.hidden_size,
        config.vocab_size,
        if config.is_qwen35() { " (Qwen3.5)" } else { "" }
    );

    let mut model = Model::new(config);

    // Find all .safetensors files
    let mut st_files = Vec::new();
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            st_files.push(path);
        }
    }

    if st_files.is_empty() {
        return Err(LlmError::WeightLoad(
            "No .safetensors files found in model directory".into(),
        ));
    }

    st_files.sort();

    // Load each safetensors file
    for st_path in &st_files {
        tracing::info!("Loading weights from {:?}", st_path);
        let st = SafeTensorsFile::open(st_path)?;

        for (name, info) in st.tensors() {
            let target = match parse_weight_name(name) {
                Some(t) => t,
                None => {
                    tracing::debug!("Skipping unknown weight: {name}");
                    continue;
                }
            };

            let data = st.tensor_to_f32(name)?;
            load_weight_into_model(&mut model, &target, &data, &info.shape)?;
        }
    }

    Ok(model)
}

fn load_weight_into_model(
    model: &mut Model,
    target: &WeightTarget,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    match target {
        WeightTarget::Embedding => {
            load_2d_into_store(&mut model.embedding.weights, data, shape)?;
        }
        WeightTarget::LmHead => {
            if let Some(ref mut head) = model.lm_head {
                load_2d_into_store(&mut head.weights, data, shape)?;
            }
            // If tie_word_embeddings, lm_head is None; skip silently.
        }
        WeightTarget::FinalNorm => {
            load_1d_into_vec(&mut model.final_norm.weight, data, shape)?;
        }
        WeightTarget::LayerAttnNorm(i) => {
            load_1d_into_vec(&mut model.layers[*i].attn_norm.weight, data, shape)?;
        }
        WeightTarget::LayerMlpNorm(i) => {
            load_1d_into_vec(&mut model.layers[*i].mlp_norm.weight, data, shape)?;
        }

        // Standard attention projections
        WeightTarget::LayerQProj(i) => {
            let attn = get_standard_attn_mut(&mut model.layers[*i].attention, *i)?;
            load_2d_into_store(&mut attn.q_proj.weights, data, shape)?;
        }
        WeightTarget::LayerKProj(i) => {
            let attn = get_standard_attn_mut(&mut model.layers[*i].attention, *i)?;
            load_2d_into_store(&mut attn.k_proj.weights, data, shape)?;
        }
        WeightTarget::LayerVProj(i) => {
            let attn = get_standard_attn_mut(&mut model.layers[*i].attention, *i)?;
            load_2d_into_store(&mut attn.v_proj.weights, data, shape)?;
        }
        WeightTarget::LayerOProj(i) => {
            let attn = get_standard_attn_mut(&mut model.layers[*i].attention, *i)?;
            load_2d_into_store(&mut attn.o_proj.weights, data, shape)?;
        }

        // Q/K RMSNorm (Qwen3.5 gated full attention)
        WeightTarget::LayerQNorm(i) => {
            let attn = get_standard_attn_mut(&mut model.layers[*i].attention, *i)?;
            if let Some(ref mut w) = attn.q_norm_weight {
                load_1d_into_vec(w, data, shape)?;
            }
        }
        WeightTarget::LayerKNorm(i) => {
            let attn = get_standard_attn_mut(&mut model.layers[*i].attention, *i)?;
            if let Some(ref mut w) = attn.k_norm_weight {
                load_1d_into_vec(w, data, shape)?;
            }
        }

        // MLP projections
        WeightTarget::LayerGateProj(i) => {
            load_2d_into_store(&mut model.layers[*i].mlp.gate_proj.weights, data, shape)?;
        }
        WeightTarget::LayerUpProj(i) => {
            load_2d_into_store(&mut model.layers[*i].mlp.up_proj.weights, data, shape)?;
        }
        WeightTarget::LayerDownProj(i) => {
            load_2d_into_store(&mut model.layers[*i].mlp.down_proj.weights, data, shape)?;
        }

        // GatedDeltaNet weights
        WeightTarget::LayerDeltaNetQKV(i) => {
            let dn = get_deltanet_mut(&mut model.layers[*i].attention, *i)?;
            load_2d_into_store(&mut dn.in_proj_qkv.weights, data, shape)?;
        }
        WeightTarget::LayerDeltaNetZ(i) => {
            let dn = get_deltanet_mut(&mut model.layers[*i].attention, *i)?;
            load_2d_into_store(&mut dn.in_proj_z.weights, data, shape)?;
        }
        WeightTarget::LayerDeltaNetA(i) => {
            let dn = get_deltanet_mut(&mut model.layers[*i].attention, *i)?;
            load_2d_into_store(&mut dn.in_proj_a.weights, data, shape)?;
        }
        WeightTarget::LayerDeltaNetB(i) => {
            let dn = get_deltanet_mut(&mut model.layers[*i].attention, *i)?;
            load_2d_into_store(&mut dn.in_proj_b.weights, data, shape)?;
        }
        WeightTarget::LayerDeltaNetConv(i) => {
            let dn = get_deltanet_mut(&mut model.layers[*i].attention, *i)?;
            // Conv weight comes as [conv_dim, 1, kernel_size] -> flatten to [conv_dim × kernel_size]
            let flat_data = data;
            let expected_len = dn.conv_dim * dn.kernel_size;
            if flat_data.len() != expected_len {
                return Err(LlmError::ShapeMismatch {
                    expected: format!("[{} total elements]", expected_len),
                    got: format!("[{} total elements] (shape {:?})", flat_data.len(), shape),
                });
            }
            dn.conv_weight.copy_from_slice(flat_data);
        }
        WeightTarget::LayerDeltaNetDtBias(i) => {
            let dn = get_deltanet_mut(&mut model.layers[*i].attention, *i)?;
            load_1d_into_vec(&mut dn.dt_bias, data, shape)?;
        }
        WeightTarget::LayerDeltaNetALog(i) => {
            let dn = get_deltanet_mut(&mut model.layers[*i].attention, *i)?;
            load_1d_into_vec(&mut dn.a_log, data, shape)?;
        }
        WeightTarget::LayerDeltaNetNorm(i) => {
            let dn = get_deltanet_mut(&mut model.layers[*i].attention, *i)?;
            load_1d_into_vec(&mut dn.norm_weight, data, shape)?;
        }
        WeightTarget::LayerDeltaNetOutProj(i) => {
            let dn = get_deltanet_mut(&mut model.layers[*i].attention, *i)?;
            load_2d_into_store(&mut dn.out_proj.weights, data, shape)?;
        }
    }
    Ok(())
}

fn get_standard_attn_mut(
    layer: &mut AttentionLayer,
    layer_idx: usize,
) -> Result<&mut crate::model::attention::Attention> {
    match layer {
        AttentionLayer::Standard(ref mut attn) => Ok(attn),
        AttentionLayer::GatedDeltaNet(_) => Err(LlmError::WeightLoad(format!(
            "Expected Standard attention at layer {layer_idx}, got GatedDeltaNet"
        ))),
    }
}

fn get_deltanet_mut(
    layer: &mut AttentionLayer,
    layer_idx: usize,
) -> Result<&mut crate::model::gated_deltanet::GatedDeltaNet> {
    match layer {
        AttentionLayer::GatedDeltaNet(ref mut dn) => Ok(dn),
        AttentionLayer::Standard(_) => Err(LlmError::WeightLoad(format!(
            "Expected GatedDeltaNet at layer {layer_idx}, got Standard attention"
        ))),
    }
}

fn load_2d_into_store(
    store: &mut klearu_accel::memory::ContiguousWeightStore,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    if shape.len() != 2 {
        return Err(LlmError::ShapeMismatch {
            expected: "2D".into(),
            got: format!("{}D", shape.len()),
        });
    }
    let (rows, cols) = (shape[0], shape[1]);
    if rows != store.num_neurons() || cols != store.neuron_dim() {
        return Err(LlmError::ShapeMismatch {
            expected: format!("[{}, {}]", store.num_neurons(), store.neuron_dim()),
            got: format!("[{rows}, {cols}]"),
        });
    }
    for row in 0..rows {
        let src = &data[row * cols..(row + 1) * cols];
        store.set_weights(row, src);
    }
    Ok(())
}

fn load_1d_into_vec(vec: &mut [f32], data: &[f32], shape: &[usize]) -> Result<()> {
    if shape.len() != 1 {
        return Err(LlmError::ShapeMismatch {
            expected: "1D".into(),
            got: format!("{}D", shape.len()),
        });
    }
    if shape[0] != vec.len() {
        return Err(LlmError::ShapeMismatch {
            expected: format!("[{}]", vec.len()),
            got: format!("[{}]", shape[0]),
        });
    }
    vec.copy_from_slice(data);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_2d_into_store() {
        let mut store = klearu_accel::memory::ContiguousWeightStore::new(2, 3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        load_2d_into_store(&mut store, &data, &[2, 3]).unwrap();

        assert_eq!(&store.get_weights(0)[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&store.get_weights(1)[..3], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_load_2d_shape_mismatch() {
        let mut store = klearu_accel::memory::ContiguousWeightStore::new(2, 3);
        let data = vec![1.0; 8];
        assert!(load_2d_into_store(&mut store, &data, &[2, 4]).is_err());
    }

    #[test]
    fn test_load_1d_into_vec() {
        let mut v = vec![0.0; 3];
        let data = vec![1.0, 2.0, 3.0];
        load_1d_into_vec(&mut v, &data, &[3]).unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_load_1d_shape_mismatch() {
        let mut v = vec![0.0; 3];
        let data = vec![1.0, 2.0];
        assert!(load_1d_into_vec(&mut v, &data, &[2]).is_err());
    }
}
