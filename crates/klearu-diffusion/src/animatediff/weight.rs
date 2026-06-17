//! Weight loader for AnimateDiff motion adapters.
//!
//! Reads a single-file `.safetensors` motion adapter (e.g.
//! `mm_sdxl_v10_beta.safetensors`) and populates the temporal
//! transformers in a [`MotionModules`] struct.
//!
//! Key format in the safetensors:
//! ```text
//! down_blocks.{block}.motion_modules.{layer}.temporal_transformer.norm.weight
//! down_blocks.{block}.motion_modules.{layer}.temporal_transformer.norm.bias
//! down_blocks.{block}.motion_modules.{layer}.temporal_transformer.proj_in.weight
//! down_blocks.{block}.motion_modules.{layer}.temporal_transformer.proj_in.bias
//! down_blocks.{block}.motion_modules.{layer}.temporal_transformer.transformer_blocks.{tb}.attn1.to_q.weight
//! ... (to_k, to_v, to_out.0.weight, to_out.0.bias)
//! ... (norm1, norm2, ff.net.0.proj.weight/bias, ff.net.2.weight/bias)
//! down_blocks.{block}.motion_modules.{layer}.temporal_transformer.proj_out.weight
//! down_blocks.{block}.motion_modules.{layer}.temporal_transformer.proj_out.bias
//! mid_block.motion_modules.{layer}.temporal_transformer.{...}
//! up_blocks.{block}.motion_modules.{layer}.temporal_transformer.{...}
//! ```

use std::path::Path;

use crate::error::{DiffusionError, Result};
use crate::weight::SingleFileLoader;

use super::motion_module::MotionModules;
use super::temporal::{TemporalTransformer, TemporalTransformerBlock};

/// Load a motion adapter from a safetensors file into an existing
/// [`MotionModules`] skeleton (created via `sdxl_default` or
/// `sd15_default`).
pub fn load_motion_adapter(
    modules: &mut MotionModules,
    path: &Path,
) -> Result<()> {
    let data = std::fs::read(path)
        .map_err(|e| DiffusionError::Unsupported(format!("read {}: {e}", path.display())))?;
    let tensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| DiffusionError::Unsupported(format!("safetensors: {e}")))?;

    let mut loaded = 0usize;
    let mut skipped = 0usize;

    for (name, view) in tensors.tensors() {
        // Parse block location from key name.
        if let Some(rest) = name.strip_prefix("down_blocks.") {
            if let Some((block_idx, module_idx, param_path)) = parse_motion_key(rest) {
                if block_idx < modules.down.len() && module_idx < modules.down[block_idx].len() {
                    load_temporal_param(
                        &mut modules.down[block_idx][module_idx],
                        &param_path,
                        &view,
                    )?;
                    loaded += 1;
                } else { skipped += 1; }
            }
        } else if let Some(rest) = name.strip_prefix("mid_block.") {
            if let Some((_block_idx, module_idx, param_path)) = parse_motion_key(rest) {
                if module_idx < modules.mid.len() {
                    load_temporal_param(
                        &mut modules.mid[module_idx],
                        &param_path,
                        &view,
                    )?;
                    loaded += 1;
                } else { skipped += 1; }
            }
        } else if let Some(rest) = name.strip_prefix("up_blocks.") {
            if let Some((block_idx, module_idx, param_path)) = parse_motion_key(rest) {
                if block_idx < modules.up.len() && module_idx < modules.up[block_idx].len() {
                    load_temporal_param(
                        &mut modules.up[block_idx][module_idx],
                        &param_path,
                        &view,
                    )?;
                    loaded += 1;
                } else { skipped += 1; }
            }
        } else {
            skipped += 1;
        }
    }

    eprintln!(
        "[animatediff] loaded {loaded} params, skipped {skipped} from {}",
        path.display()
    );
    Ok(())
}

/// Parse `"0.motion_modules.1.norm.weight"` into
/// `(block=0, module=1, "norm.weight")`.
///
/// Also handles `"motion_modules.0.norm.weight"` (mid_block format —
/// no block index, returns block=0).
fn parse_motion_key(s: &str) -> Option<(usize, usize, String)> {
    // Try "{block}.motion_modules.{module}.{param}" first.
    if let Some(dot1) = s.find('.') {
        if let Ok(block_idx) = s[..dot1].parse::<usize>() {
            let after_block = &s[dot1 + 1..];
            if let Some(rest) = after_block.strip_prefix("motion_modules.") {
                let dot2 = rest.find('.')?;
                let module_idx: usize = rest[..dot2].parse().ok()?;
                let param_path = &rest[dot2 + 1..];
                // Strip optional "temporal_transformer." prefix (some
                // adapters include it, the SDXL beta does not).
                let param_path = param_path
                    .strip_prefix("temporal_transformer.")
                    .unwrap_or(param_path);
                return Some((block_idx, module_idx, param_path.to_string()));
            }
        }
    }
    // Mid-block format: "motion_modules.{module}.{param}" (no block index).
    if let Some(rest) = s.strip_prefix("motion_modules.") {
        let dot = rest.find('.')?;
        let module_idx: usize = rest[..dot].parse().ok()?;
        let param_path = &rest[dot + 1..];
        let param_path = param_path
            .strip_prefix("temporal_transformer.")
            .unwrap_or(param_path);
        return Some((0, module_idx, param_path.to_string()));
    }
    None
}

fn load_temporal_param(
    tt: &mut TemporalTransformer,
    param_path: &str,
    view: &safetensors::tensor::TensorView,
) -> Result<()> {
    let floats = view_to_f32(view)?;

    // Dispatch on param path.
    if param_path == "norm.weight" {
        tt.norm.gamma = floats;
    } else if param_path == "norm.bias" {
        tt.norm.beta = floats;
    } else if param_path == "proj_in.weight" {
        tt.proj_in.weight = floats;
    } else if param_path == "proj_in.bias" {
        tt.proj_in.bias = Some(floats);
    } else if param_path == "proj_out.weight" {
        tt.proj_out.weight = floats;
    } else if param_path == "proj_out.bias" {
        tt.proj_out.bias = Some(floats);
    } else if let Some(rest) = param_path.strip_prefix("transformer_blocks.") {
        // "0.attn1.to_q.weight", "0.norm1.weight", "0.ff.net.0.proj.weight", etc.
        let dot = rest.find('.').unwrap_or(rest.len());
        let tb_idx: usize = rest[..dot].parse().unwrap_or(0);
        let tb_param = &rest[dot + 1..];
        if tb_idx < tt.blocks.len() {
            load_block_param(&mut tt.blocks[tb_idx], tb_param, floats)?;
        }
    }
    // Ignore unknown paths silently — the adapter may have extra metadata.
    Ok(())
}

fn load_block_param(
    block: &mut TemporalTransformerBlock,
    path: &str,
    floats: Vec<f32>,
) -> Result<()> {
    match path {
        "norm1.weight" => block.norm1.gamma = floats,
        "norm1.bias" => block.norm1.beta = floats,
        "norm2.weight" => block.norm2.gamma = floats,
        "norm2.bias" => block.norm2.beta = floats,
        "norm3.weight" => block.norm3.gamma = floats,
        "norm3.bias" => block.norm3.beta = floats,
        "attn1.to_q.weight" => block.attn1.to_q.weight = floats,
        "attn1.to_k.weight" => block.attn1.to_k.weight = floats,
        "attn1.to_v.weight" => block.attn1.to_v.weight = floats,
        "attn1.to_out.0.weight" | "attn1.to_out.weight" => block.attn1.to_out.weight = floats,
        "attn1.to_out.0.bias" | "attn1.to_out.bias" => block.attn1.to_out.bias = Some(floats),
        "attn2.to_q.weight" => block.attn2.to_q.weight = floats,
        "attn2.to_k.weight" => block.attn2.to_k.weight = floats,
        "attn2.to_v.weight" => block.attn2.to_v.weight = floats,
        "attn2.to_out.0.weight" | "attn2.to_out.weight" => block.attn2.to_out.weight = floats,
        "attn2.to_out.0.bias" | "attn2.to_out.bias" => block.attn2.to_out.bias = Some(floats),
        "ff.net.0.proj.weight" => block.ff_net_0.weight = floats,
        "ff.net.0.proj.bias" => block.ff_net_0.bias = Some(floats),
        "ff.net.2.weight" => block.ff_net_2.weight = floats,
        "ff.net.2.bias" => block.ff_net_2.bias = Some(floats),
        "pos_embed.pe" => {
            // Shape is [1, max_frames, inner_dim]; flatten to [max_frames * inner_dim].
            let inner = block.inner_dim;
            if floats.len() >= inner {
                block.pos_embed_max_frames = floats.len() / inner;
                block.pos_embed = Some(floats);
            }
        }
        _ => {}
    }
    Ok(())
}

fn view_to_f32(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>> {
    use safetensors::Dtype;
    let data = view.data();
    match view.dtype() {
        Dtype::F32 => {
            let n = data.len() / 4;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 4;
                out.push(f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]));
            }
            Ok(out)
        }
        Dtype::F16 => {
            let n = data.len() / 2;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 2;
                let h = half::f16::from_le_bytes([data[off], data[off+1]]);
                out.push(h.to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let n = data.len() / 2;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 2;
                let h = half::bf16::from_le_bytes([data[off], data[off+1]]);
                out.push(h.to_f32());
            }
            Ok(out)
        }
        other => Err(DiffusionError::Unsupported(format!(
            "unsupported dtype {other:?} in motion adapter"
        ))),
    }
}
