//! Helpers for loading individual layer weights from a `ComponentTensors`
//! into the typed module layers (Linear, Conv2d, GroupNorm, LayerNorm).
//!
//! HuggingFace Diffusers stores weights under hierarchical names like:
//!   `text_model.encoder.layers.0.self_attn.q_proj.weight` — Linear weight
//!   `text_model.encoder.layers.0.self_attn.q_proj.bias`   — Linear bias
//!   `down_blocks.0.resnets.0.norm1.weight`                — GroupNorm γ
//!   `down_blocks.0.resnets.0.norm1.bias`                  — GroupNorm β
//!   `down_blocks.0.resnets.0.conv1.weight`                — Conv2d kernel
//!   `down_blocks.0.resnets.0.conv1.bias`                  — Conv2d bias
//!
//! Each helper takes a `prefix` (the module path without trailing dot)
//! and looks for `<prefix>.weight` / `<prefix>.bias` in the component.

use crate::error::Result;
use crate::layers::{Conv2d, GroupNorm, LayerNorm, Linear};
use crate::weight::ComponentTensors;

pub fn load_linear(comp: &ComponentTensors, prefix: &str, target: &mut Linear) -> Result<()> {
    let w = comp.get_f32(&format!("{prefix}.weight"))?;
    if w.len() != target.weight.len() {
        return Err(crate::error::DiffusionError::ShapeMismatch {
            expected: format!("{}={} ({}×{})", prefix, target.weight.len(),
                target.out_features, target.in_features),
            got: format!("{}", w.len()),
        });
    }
    target.weight = w;
    if target.bias.is_some() {
        let b = comp.get_f32(&format!("{prefix}.bias"))?;
        target.bias = Some(b);
    }
    Ok(())
}

pub fn load_conv2d(comp: &ComponentTensors, prefix: &str, target: &mut Conv2d) -> Result<()> {
    let w = comp.get_f32(&format!("{prefix}.weight"))?;
    if w.len() != target.weight.len() {
        return Err(crate::error::DiffusionError::ShapeMismatch {
            expected: format!("{}={} ({}×{}×{}×{})", prefix, target.weight.len(),
                target.out_channels, target.in_channels, target.kernel_h, target.kernel_w),
            got: format!("{}", w.len()),
        });
    }
    target.weight = w;
    if target.bias.is_some() {
        let b = comp.get_f32(&format!("{prefix}.bias"))?;
        target.bias = Some(b);
    }
    Ok(())
}

pub fn load_group_norm(comp: &ComponentTensors, prefix: &str, target: &mut GroupNorm) -> Result<()> {
    target.gamma = comp.get_f32(&format!("{prefix}.weight"))?;
    target.beta = comp.get_f32(&format!("{prefix}.bias"))?;
    Ok(())
}

pub fn load_layer_norm(comp: &ComponentTensors, prefix: &str, target: &mut LayerNorm) -> Result<()> {
    target.gamma = comp.get_f32(&format!("{prefix}.weight"))?;
    target.beta = comp.get_f32(&format!("{prefix}.bias"))?;
    Ok(())
}

pub fn load_embedding(comp: &ComponentTensors, name: &str, target: &mut Vec<f32>) -> Result<()> {
    *target = comp.get_f32(name)?;
    Ok(())
}
