//! Load Wan DiT weights from a Diffusers `transformer/` checkpoint.
//!
//! Every layout here matches PyTorch's `[out, in]` linear convention with no
//! transposition, including two that look like they should need one:
//!
//!   * `patch_embedding` is a `Conv3d` stored `[dim, in_c, p_t, p_h, p_w]`.
//!     Because its kernel equals its stride, it is a linear map over each
//!     patch, and the flattened weight is exactly `[dim, in_c * p_t*p_h*p_w]`
//!     — the same bytes, reinterpreted. The patch vector must be gathered
//!     channel-major to match (see `WanDit::patchify`).
//!   * `scale_shift_table` is a bare `nn.Parameter` of shape `[1, 6, dim]`
//!     (or `[1, 2, dim]` at model level), which flattens to `6 * dim`.

use crate::weight::loader::ComponentTensors;
use crate::{DiffusionError, Result};

use super::model::{WanAttention, WanBlock, WanDit};
use super::names::{tensor_names, tensor_numel};

fn load_vec(comp: &ComponentTensors, name: &str, dst: &mut Vec<f32>) -> Result<()> {
    let v = comp.get_f32(name)?;
    if v.len() != dst.len() {
        return Err(DiffusionError::ShapeMismatch {
            expected: format!("{} elements in {name}", dst.len()),
            got: format!("{}", v.len()),
        });
    }
    *dst = v;
    Ok(())
}

fn load_linear(l: &mut crate::layers::Linear, comp: &ComponentTensors, p: &str) -> Result<()> {
    load_vec(comp, &format!("{p}.weight"), &mut l.weight)?;
    if let Some(b) = l.bias.as_mut() {
        load_vec(comp, &format!("{p}.bias"), b)?;
    }
    Ok(())
}

impl WanAttention {
    pub fn load(&mut self, comp: &ComponentTensors, p: &str) -> Result<()> {
        load_vec(comp, &format!("{p}.norm_q.weight"), &mut self.norm_q.weight)?;
        load_vec(comp, &format!("{p}.norm_k.weight"), &mut self.norm_k.weight)?;
        load_linear(&mut self.to_q, comp, &format!("{p}.to_q"))?;
        load_linear(&mut self.to_k, comp, &format!("{p}.to_k"))?;
        load_linear(&mut self.to_v, comp, &format!("{p}.to_v"))?;
        // `to_out` is an nn.ModuleList: index 0 is the linear, index 1 dropout.
        load_linear(&mut self.to_out, comp, &format!("{p}.to_out.0"))
    }
}

impl WanBlock {
    pub fn load(&mut self, comp: &ComponentTensors, p: &str, cross_attn_norm: bool) -> Result<()> {
        load_vec(comp, &format!("{p}.scale_shift_table"), &mut self.scale_shift_table)?;
        self.attn1.load(comp, &format!("{p}.attn1"))?;
        self.attn2.load(comp, &format!("{p}.attn2"))?;
        if cross_attn_norm {
            load_vec(comp, &format!("{p}.norm2.weight"), &mut self.norm2.gamma)?;
            load_vec(comp, &format!("{p}.norm2.bias"), &mut self.norm2.beta)?;
        }
        // net.0 is GELU(proj), net.1 dropout, net.2 the output linear.
        load_linear(&mut self.ffn_proj, comp, &format!("{p}.ffn.net.0.proj"))?;
        load_linear(&mut self.ffn_out, comp, &format!("{p}.ffn.net.2"))
    }
}

/// Load the timestep/text embedders. Split out so the GPU path can load just
/// these — a few hundred MB — without materialising the 20 GB of blocks.
pub fn load_conditioning(
    c: &mut super::model::Conditioning,
    comp: &ComponentTensors,
) -> Result<()> {
    load_linear(&mut c.time_linear_1, comp, "condition_embedder.time_embedder.linear_1")?;
    load_linear(&mut c.time_linear_2, comp, "condition_embedder.time_embedder.linear_2")?;
    load_linear(&mut c.time_proj, comp, "condition_embedder.time_proj")?;
    load_linear(&mut c.text_linear_1, comp, "condition_embedder.text_embedder.linear_1")?;
    load_linear(&mut c.text_linear_2, comp, "condition_embedder.text_embedder.linear_2")
}

impl WanDit {
    /// Names present in the model but absent from `comp`.
    pub fn missing_tensors(&self, comp: &ComponentTensors) -> Vec<String> {
        tensor_names(&self.cfg).into_iter().filter(|n| !comp.has(n)).collect()
    }

    /// Names in the checkpoint whose element count differs from what the
    /// config implies — a shape mismatch that would otherwise only surface
    /// partway through loading.
    pub fn mismatched_tensors(&self, comp: &ComponentTensors) -> Vec<String> {
        tensor_names(&self.cfg)
            .into_iter()
            .filter(|n| match (comp.numel(n), tensor_numel(&self.cfg, n)) {
                (Some(a), Some(b)) => a != b,
                _ => false,
            })
            .collect()
    }

    /// Build from config and load every weight.
    pub fn load(cfg: super::WanDitConfig, comp: &ComponentTensors) -> Result<Self> {
        let mut m = WanDit::new(cfg);

        let missing = m.missing_tensors(comp);
        if !missing.is_empty() {
            return Err(DiffusionError::Layout(format!(
                "Wan DiT checkpoint is missing {} tensor(s), first few: {:?}",
                missing.len(),
                &missing[..missing.len().min(5)]
            )));
        }
        let bad = m.mismatched_tensors(comp);
        if !bad.is_empty() {
            return Err(DiffusionError::Layout(format!(
                "Wan DiT checkpoint has {} tensor(s) of unexpected size, first few: {:?}",
                bad.len(),
                &bad[..bad.len().min(5)]
            )));
        }

        load_linear(&mut m.patch_embedding, comp, "patch_embedding")?;
        load_conditioning(&mut m.cond, comp)?;

        let cross = m.cfg.cross_attn_norm;
        for (i, b) in m.blocks.iter_mut().enumerate() {
            b.load(comp, &format!("blocks.{i}"), cross)?;
        }

        let mut sst = std::mem::take(&mut m.scale_shift_table);
        load_vec(comp, "scale_shift_table", &mut sst)?;
        m.scale_shift_table = sst;
        load_linear(&mut m.proj_out, comp, "proj_out")?;
        Ok(m)
    }
}
