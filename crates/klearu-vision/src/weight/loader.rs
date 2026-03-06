use std::collections::HashMap;
use std::path::Path;

use crate::config::DaViTConfig;
use crate::error::{VisionError, Result};
use crate::model::DaViTModel;
use crate::model::davit_block::{SpatialBlock, ChannelBlock};

use super::name_map::{parse_davit_weight_name, DaViTWeightTarget, BlockType};

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug)]
pub(crate) struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data_offset: usize,
    pub data_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dtype {
    F32,
    F16,
    BF16,
}

/// A memory-mapped safetensors file.
pub(crate) struct SafeTensorsFile {
    _mmap: memmap2::Mmap,
    data_start: usize,
    pub tensors: HashMap<String, TensorInfo>,
    raw: *const u8,
    raw_len: usize,
}

unsafe impl Send for SafeTensorsFile {}
unsafe impl Sync for SafeTensorsFile {}

impl SafeTensorsFile {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(VisionError::SafeTensors("File too small".into()));
        }

        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        if 8 + header_len > mmap.len() {
            return Err(VisionError::SafeTensors("Invalid header length".into()));
        }

        let header_bytes = &mmap[8..8 + header_len];
        let header: serde_json::Value =
            serde_json::from_slice(header_bytes).map_err(|e| VisionError::SafeTensors(e.to_string()))?;

        let data_start = 8 + header_len;
        let mut tensors = HashMap::new();

        let obj = header
            .as_object()
            .ok_or_else(|| VisionError::SafeTensors("Header is not an object".into()))?;

        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype_str = info["dtype"]
                .as_str()
                .ok_or_else(|| VisionError::SafeTensors(format!("Missing dtype for {name}")))?;

            let dtype = match dtype_str {
                "F32" => Dtype::F32,
                "F16" => Dtype::F16,
                "BF16" => Dtype::BF16,
                other => return Err(VisionError::UnsupportedDtype(other.to_string())),
            };

            let shape: Vec<usize> = info["shape"]
                .as_array()
                .ok_or_else(|| VisionError::SafeTensors(format!("Missing shape for {name}")))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();

            let offsets = info["data_offsets"]
                .as_array()
                .ok_or_else(|| VisionError::SafeTensors(format!("Missing data_offsets for {name}")))?;

            let start = offsets[0].as_u64().unwrap_or(0) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;

            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    data_offset: start,
                    data_len: end - start,
                },
            );
        }

        let raw = mmap.as_ptr();
        let raw_len = mmap.len();

        Ok(Self {
            _mmap: mmap,
            data_start,
            tensors,
            raw,
            raw_len,
        })
    }

    pub(crate) fn tensor_to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| VisionError::MissingWeight(name.to_string()))?;
        let start = self.data_start + info.data_offset;
        let end = start + info.data_len;
        if end > self.raw_len {
            return Err(VisionError::SafeTensors(format!(
                "Tensor {name} data out of bounds"
            )));
        }
        let data = unsafe { std::slice::from_raw_parts(self.raw.add(start), info.data_len) };

        match info.dtype {
            Dtype::F32 => {
                Ok(data.chunks_exact(4).map(|c| {
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                }).collect())
            }
            Dtype::BF16 => {
                Ok(data.chunks_exact(2).map(|c| {
                    half::bf16::from_le_bytes([c[0], c[1]]).to_f32()
                }).collect())
            }
            Dtype::F16 => {
                Ok(data.chunks_exact(2).map(|c| {
                    half::f16::from_le_bytes([c[0], c[1]]).to_f32()
                }).collect())
            }
        }
    }
}

/// Load a DaViT model from a timm model directory.
///
/// Expects a `config.json` and one or more `.safetensors` files in the directory.
pub fn load_davit_model(model_dir: &Path) -> Result<DaViTModel> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config = DaViTConfig::from_timm_config(&config_str)?;

    tracing::info!(
        "Loading DaViT: embed_dims={:?}, depths={:?}, num_classes={}",
        config.embed_dims,
        config.depths,
        config.num_classes,
    );

    let mut model = DaViTModel::new(config);

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
            let target = match parse_davit_weight_name(name) {
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
    model: &mut DaViTModel,
    target: &DaViTWeightTarget,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    match target {
        // Stem
        DaViTWeightTarget::StemConvWeight => {
            load_conv_weight(&mut model.stem.conv, data, shape)?;
        }
        DaViTWeightTarget::StemConvBias => {
            load_1d_into_vec(model.stem.conv.bias.as_mut().unwrap(), data, shape)?;
        }
        DaViTWeightTarget::StemNormWeight => {
            load_1d_into_vec(&mut model.stem.norm.weight, data, shape)?;
        }
        DaViTWeightTarget::StemNormBias => {
            load_1d_into_vec(&mut model.stem.norm.bias, data, shape)?;
        }

        // Downsample
        DaViTWeightTarget::StageDownsampleNormWeight(s) => {
            let ds = model.stages[*s].downsample.as_mut()
                .ok_or_else(|| VisionError::WeightLoad(format!("Stage {s} has no downsample")))?;
            load_1d_into_vec(&mut ds.norm.weight, data, shape)?;
        }
        DaViTWeightTarget::StageDownsampleNormBias(s) => {
            let ds = model.stages[*s].downsample.as_mut()
                .ok_or_else(|| VisionError::WeightLoad(format!("Stage {s} has no downsample")))?;
            load_1d_into_vec(&mut ds.norm.bias, data, shape)?;
        }
        DaViTWeightTarget::StageDownsampleConvWeight(s) => {
            let ds = model.stages[*s].downsample.as_mut()
                .ok_or_else(|| VisionError::WeightLoad(format!("Stage {s} has no downsample")))?;
            load_conv_weight(&mut ds.conv, data, shape)?;
        }
        DaViTWeightTarget::StageDownsampleConvBias(s) => {
            let ds = model.stages[*s].downsample.as_mut()
                .ok_or_else(|| VisionError::WeightLoad(format!("Stage {s} has no downsample")))?;
            load_1d_into_vec(ds.conv.bias.as_mut().unwrap(), data, shape)?;
        }

        // Block components
        DaViTWeightTarget::BlockCpe1Weight(s, b, bt) => {
            let (spatial, channel) = &mut model.stages[*s].blocks[*b];
            let conv = match bt {
                BlockType::Spatial => &mut spatial.cpe1.proj,
                BlockType::Channel => &mut channel.cpe1.proj,
            };
            load_conv_weight(conv, data, shape)?;
        }
        DaViTWeightTarget::BlockCpe1Bias(s, b, bt) => {
            let (spatial, channel) = &mut model.stages[*s].blocks[*b];
            let conv = match bt {
                BlockType::Spatial => &mut spatial.cpe1.proj,
                BlockType::Channel => &mut channel.cpe1.proj,
            };
            load_1d_into_vec(conv.bias.as_mut().unwrap(), data, shape)?;
        }
        DaViTWeightTarget::BlockNorm1Weight(s, b, bt) => {
            let norm = get_norm1_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_1d_into_vec(&mut norm.weight, data, shape)?;
        }
        DaViTWeightTarget::BlockNorm1Bias(s, b, bt) => {
            let norm = get_norm1_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_1d_into_vec(&mut norm.bias, data, shape)?;
        }
        DaViTWeightTarget::BlockAttnQkvWeight(s, b, bt) => {
            let qkv = get_attn_qkv_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_2d_into_store(&mut qkv.weights, data, shape)?;
        }
        DaViTWeightTarget::BlockAttnQkvBias(s, b, bt) => {
            let qkv = get_attn_qkv_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_1d_into_vec(&mut qkv.bias, data, shape)?;
        }
        DaViTWeightTarget::BlockAttnProjWeight(s, b, bt) => {
            let proj = get_attn_proj_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_2d_into_store(&mut proj.weights, data, shape)?;
        }
        DaViTWeightTarget::BlockAttnProjBias(s, b, bt) => {
            let proj = get_attn_proj_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_1d_into_vec(&mut proj.bias, data, shape)?;
        }
        DaViTWeightTarget::BlockCpe2Weight(s, b, bt) => {
            let (spatial, channel) = &mut model.stages[*s].blocks[*b];
            let conv = match bt {
                BlockType::Spatial => &mut spatial.cpe2.proj,
                BlockType::Channel => &mut channel.cpe2.proj,
            };
            load_conv_weight(conv, data, shape)?;
        }
        DaViTWeightTarget::BlockCpe2Bias(s, b, bt) => {
            let (spatial, channel) = &mut model.stages[*s].blocks[*b];
            let conv = match bt {
                BlockType::Spatial => &mut spatial.cpe2.proj,
                BlockType::Channel => &mut channel.cpe2.proj,
            };
            load_1d_into_vec(conv.bias.as_mut().unwrap(), data, shape)?;
        }
        DaViTWeightTarget::BlockNorm2Weight(s, b, bt) => {
            let norm = get_norm2_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_1d_into_vec(&mut norm.weight, data, shape)?;
        }
        DaViTWeightTarget::BlockNorm2Bias(s, b, bt) => {
            let norm = get_norm2_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_1d_into_vec(&mut norm.bias, data, shape)?;
        }
        DaViTWeightTarget::BlockMlpFc1Weight(s, b, bt) => {
            let fc = get_mlp_fc1_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_2d_into_store(&mut fc.weights, data, shape)?;
        }
        DaViTWeightTarget::BlockMlpFc1Bias(s, b, bt) => {
            let fc = get_mlp_fc1_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_1d_into_vec(&mut fc.bias, data, shape)?;
        }
        DaViTWeightTarget::BlockMlpFc2Weight(s, b, bt) => {
            let fc = get_mlp_fc2_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_2d_into_store(&mut fc.weights, data, shape)?;
        }
        DaViTWeightTarget::BlockMlpFc2Bias(s, b, bt) => {
            let fc = get_mlp_fc2_mut(&mut model.stages[*s].blocks[*b], *bt);
            load_1d_into_vec(&mut fc.bias, data, shape)?;
        }

        // Head
        DaViTWeightTarget::HeadNormWeight => {
            load_1d_into_vec(&mut model.head.norm.weight, data, shape)?;
        }
        DaViTWeightTarget::HeadNormBias => {
            load_1d_into_vec(&mut model.head.norm.bias, data, shape)?;
        }
        DaViTWeightTarget::HeadFcWeight => {
            load_2d_into_store(&mut model.head.fc.weights, data, shape)?;
        }
        DaViTWeightTarget::HeadFcBias => {
            load_1d_into_vec(&mut model.head.fc.bias, data, shape)?;
        }
    }
    Ok(())
}

// Helper accessors to avoid borrow checker issues with tuple destructuring

fn get_norm1_mut<'a>(
    pair: &'a mut (SpatialBlock, ChannelBlock),
    bt: BlockType,
) -> &'a mut crate::layers::LayerNorm {
    match bt {
        BlockType::Spatial => &mut pair.0.norm1,
        BlockType::Channel => &mut pair.1.norm1,
    }
}

fn get_norm2_mut<'a>(
    pair: &'a mut (SpatialBlock, ChannelBlock),
    bt: BlockType,
) -> &'a mut crate::layers::LayerNorm {
    match bt {
        BlockType::Spatial => &mut pair.0.norm2,
        BlockType::Channel => &mut pair.1.norm2,
    }
}

fn get_attn_qkv_mut<'a>(
    pair: &'a mut (SpatialBlock, ChannelBlock),
    bt: BlockType,
) -> &'a mut crate::layers::LinearBias {
    match bt {
        BlockType::Spatial => &mut pair.0.attn.qkv,
        BlockType::Channel => &mut pair.1.attn.qkv,
    }
}

fn get_attn_proj_mut<'a>(
    pair: &'a mut (SpatialBlock, ChannelBlock),
    bt: BlockType,
) -> &'a mut crate::layers::LinearBias {
    match bt {
        BlockType::Spatial => &mut pair.0.attn.proj,
        BlockType::Channel => &mut pair.1.attn.proj,
    }
}

fn get_mlp_fc1_mut<'a>(
    pair: &'a mut (SpatialBlock, ChannelBlock),
    bt: BlockType,
) -> &'a mut crate::layers::LinearBias {
    match bt {
        BlockType::Spatial => &mut pair.0.mlp_fc1,
        BlockType::Channel => &mut pair.1.mlp_fc1,
    }
}

fn get_mlp_fc2_mut<'a>(
    pair: &'a mut (SpatialBlock, ChannelBlock),
    bt: BlockType,
) -> &'a mut crate::layers::LinearBias {
    match bt {
        BlockType::Spatial => &mut pair.0.mlp_fc2,
        BlockType::Channel => &mut pair.1.mlp_fc2,
    }
}

fn load_conv_weight(
    conv: &mut crate::layers::Conv2d,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    // Conv weight: [out_channels, in_channels/groups, kH, kW]
    let expected_len = conv.weight.len();
    let data_len: usize = shape.iter().product();
    if data_len != expected_len {
        return Err(VisionError::ShapeMismatch {
            expected: format!("[{} total]", expected_len),
            got: format!("[{} total] (shape {:?})", data_len, shape),
        });
    }
    conv.weight.copy_from_slice(data);
    Ok(())
}

pub(crate) fn load_2d_into_store(
    store: &mut klearu_accel::memory::ContiguousWeightStore,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    if shape.len() != 2 {
        return Err(VisionError::ShapeMismatch {
            expected: "2D".into(),
            got: format!("{}D", shape.len()),
        });
    }
    let (rows, cols) = (shape[0], shape[1]);
    if rows != store.num_neurons() || cols != store.neuron_dim() {
        return Err(VisionError::ShapeMismatch {
            expected: format!("[{}, {}]", store.num_neurons(), store.neuron_dim()),
            got: format!("[{rows}, {cols}]"),
        });
    }
    for row in 0..rows {
        store.set_weights(row, &data[row * cols..(row + 1) * cols]);
    }
    Ok(())
}

pub(crate) fn load_1d_into_vec(vec: &mut [f32], data: &[f32], shape: &[usize]) -> Result<()> {
    if shape.len() != 1 {
        return Err(VisionError::ShapeMismatch {
            expected: "1D".into(),
            got: format!("{}D", shape.len()),
        });
    }
    if shape[0] != vec.len() {
        return Err(VisionError::ShapeMismatch {
            expected: format!("[{}]", vec.len()),
            got: format!("[{}]", shape[0]),
        });
    }
    vec.copy_from_slice(data);
    Ok(())
}
