//! Load Wan VAE weights from a Diffusers `vae/` safetensors checkpoint.
//!
//! Naming follows the reference module tree, with two wrinkles:
//!
//!   * The per-frame spatial convolutions live inside an `nn.Sequential`
//!     alongside an upsample or a zero-pad, so they are addressed as
//!     `<resampler>.resample.1` — index 1, not 0.
//!   * Those are `nn.Conv2d`, stored `[out, in, kh, kw]`, while we hold them as
//!     `CausalConv3d` with `[out, in, 1, kh, kw]`. Same element count, same
//!     order, so the flat data loads unchanged; only the declared rank differs.
//!
//! `AvgDown3D` and `DupUp3D` carry no parameters, so the residual shortcuts
//! contribute no tensors.
//!
//! The Wan 2.1 / Qwen-Image layout ([`StageKind::Plain`]) differs in two
//! places, both read from the Qwen-Image safetensors header:
//!
//!   * The encoder's `down_blocks` is one **flat** `nn.ModuleList` — every
//!     resnet, attention and resample of every stage, numbered in forward
//!     order — so `[1,2,4,4]` with two resnets per stage is indices 0..=10
//!     with resamples at 2, 5 and 8. Wan 2.2 nests a module per stage.
//!   * A decoder stage's upsampler sits in a one-element list:
//!     `up_blocks.{i}.upsamplers.0`, not `up_blocks.{i}.upsampler`.

use crate::weight::loader::ComponentTensors;
use crate::{DiffusionError, Result};

use super::blocks::{
    AttentionBlock, MidBlock, ResidualBlock, ResidualDownBlock, ResidualUpBlock, StageKind,
};
use super::model::{Decoder3d, Encoder3d, WanVae};
use super::{CausalConv3d, RmsNormChannel};

/// Copy `{prefix}.weight` (and `.bias` when the layer has one) into `conv`.
fn load_conv(conv: &mut CausalConv3d, comp: &ComponentTensors, prefix: &str) -> Result<()> {
    let name = format!("{prefix}.weight");
    let w = comp.get_f32(&name)?;
    if w.len() != conv.weight.len() {
        return Err(DiffusionError::ShapeMismatch {
            expected: format!("{} elements in {name}", conv.weight.len()),
            got: format!("{}", w.len()),
        });
    }
    conv.weight = w;
    if let Some(b) = conv.bias.as_mut() {
        let name = format!("{prefix}.bias");
        let v = comp.get_f32(&name)?;
        if v.len() != b.len() {
            return Err(DiffusionError::ShapeMismatch {
                expected: format!("{} elements in {name}", b.len()),
                got: format!("{}", v.len()),
            });
        }
        *b = v;
    }
    Ok(())
}

/// Wan's RMS norm stores its scale as `gamma`, not `weight`.
fn load_norm(norm: &mut RmsNormChannel, comp: &ComponentTensors, prefix: &str) -> Result<()> {
    let name = format!("{prefix}.gamma");
    let g = comp.get_f32(&name)?;
    if g.len() != norm.gamma.len() {
        return Err(DiffusionError::ShapeMismatch {
            expected: format!("{} elements in {name}", norm.gamma.len()),
            got: format!("{}", g.len()),
        });
    }
    norm.gamma = g;
    if let Some(b) = norm.bias.as_mut() {
        *b = comp.get_f32(&format!("{prefix}.bias"))?;
    }
    Ok(())
}

/// One expected tensor: its name and element count.
pub type TensorSpec = (String, usize);

/// Specs a conv contributes, for [`WanVae::tensor_specs`].
fn conv_names(out: &mut Vec<TensorSpec>, conv: &CausalConv3d, prefix: &str) {
    out.push((format!("{prefix}.weight"), conv.weight.len()));
    if let Some(b) = &conv.bias {
        out.push((format!("{prefix}.bias"), b.len()));
    }
}

fn norm_names(out: &mut Vec<TensorSpec>, norm: &RmsNormChannel, prefix: &str) {
    out.push((format!("{prefix}.gamma"), norm.gamma.len()));
    if let Some(b) = &norm.bias {
        out.push((format!("{prefix}.bias"), b.len()));
    }
}

impl ResidualBlock {
    pub fn load(&mut self, comp: &ComponentTensors, p: &str) -> Result<()> {
        load_norm(&mut self.norm1, comp, &format!("{p}.norm1"))?;
        load_conv(&mut self.conv1, comp, &format!("{p}.conv1"))?;
        load_norm(&mut self.norm2, comp, &format!("{p}.norm2"))?;
        load_conv(&mut self.conv2, comp, &format!("{p}.conv2"))?;
        if let Some(sc) = self.shortcut.as_mut() {
            load_conv(sc, comp, &format!("{p}.conv_shortcut"))?;
        }
        Ok(())
    }

    pub fn names(&self, out: &mut Vec<TensorSpec>, p: &str) {
        norm_names(out, &self.norm1, &format!("{p}.norm1"));
        conv_names(out, &self.conv1, &format!("{p}.conv1"));
        norm_names(out, &self.norm2, &format!("{p}.norm2"));
        conv_names(out, &self.conv2, &format!("{p}.conv2"));
        if let Some(sc) = &self.shortcut {
            conv_names(out, sc, &format!("{p}.conv_shortcut"));
        }
    }
}

impl AttentionBlock {
    pub fn load(&mut self, comp: &ComponentTensors, p: &str) -> Result<()> {
        load_norm(&mut self.norm, comp, &format!("{p}.norm"))?;
        load_conv(&mut self.to_qkv, comp, &format!("{p}.to_qkv"))?;
        load_conv(&mut self.proj, comp, &format!("{p}.proj"))
    }

    pub fn names(&self, out: &mut Vec<TensorSpec>, p: &str) {
        norm_names(out, &self.norm, &format!("{p}.norm"));
        conv_names(out, &self.to_qkv, &format!("{p}.to_qkv"));
        conv_names(out, &self.proj, &format!("{p}.proj"));
    }
}

impl MidBlock {
    pub fn load(&mut self, comp: &ComponentTensors, p: &str) -> Result<()> {
        for (i, r) in self.resnets.iter_mut().enumerate() {
            r.load(comp, &format!("{p}.resnets.{i}"))?;
        }
        for (i, a) in self.attentions.iter_mut().enumerate() {
            a.load(comp, &format!("{p}.attentions.{i}"))?;
        }
        Ok(())
    }

    pub fn names(&self, out: &mut Vec<TensorSpec>, p: &str) {
        for (i, r) in self.resnets.iter().enumerate() {
            r.names(out, &format!("{p}.resnets.{i}"));
        }
        for (i, a) in self.attentions.iter().enumerate() {
            a.names(out, &format!("{p}.attentions.{i}"));
        }
    }
}

impl super::blocks::Resample {
    pub fn load(&mut self, comp: &ComponentTensors, p: &str) -> Result<()> {
        // Index 1 of the nn.Sequential; index 0 is the upsample or zero-pad.
        load_conv(&mut self.conv, comp, &format!("{p}.resample.1"))?;
        if let Some(tc) = self.time_conv.as_mut() {
            load_conv(tc, comp, &format!("{p}.time_conv"))?;
        }
        Ok(())
    }

    pub fn names(&self, out: &mut Vec<TensorSpec>, p: &str) {
        conv_names(out, &self.conv, &format!("{p}.resample.1"));
        if let Some(tc) = &self.time_conv {
            conv_names(out, tc, &format!("{p}.time_conv"));
        }
    }
}

/// Checkpoint prefixes of one encoder stage's layers.
struct DownPaths {
    resnets: Vec<String>,
    attentions: Vec<String>,
    downsampler: Option<String>,
}

impl ResidualDownBlock {
    /// Where this stage's layers live under `p` (the encoder's `down_blocks`).
    ///
    /// `stage` is this stage's position; `flat` is the running index into the
    /// plain layout's flat list, advanced past this stage's layers. The
    /// residual layout ignores `flat`, the plain one ignores `stage`.
    fn paths(&self, p: &str, stage: usize, flat: &mut usize) -> DownPaths {
        match self.kind {
            StageKind::Residual => DownPaths {
                resnets: (0..self.resnets.len())
                    .map(|j| format!("{p}.{stage}.resnets.{j}"))
                    .collect(),
                attentions: Vec::new(),
                downsampler: self.downsampler.as_ref().map(|_| format!("{p}.{stage}.downsampler")),
            },
            StageKind::Plain => {
                let mut next = || {
                    let i = *flat;
                    *flat += 1;
                    format!("{p}.{i}")
                };
                let mut resnets = Vec::new();
                let mut attentions = Vec::new();
                for j in 0..self.resnets.len() {
                    resnets.push(next());
                    if j < self.attentions.len() {
                        attentions.push(next());
                    }
                }
                let downsampler = self.downsampler.as_ref().map(|_| next());
                DownPaths { resnets, attentions, downsampler }
            }
        }
    }

    pub fn load(
        &mut self,
        comp: &ComponentTensors,
        p: &str,
        stage: usize,
        flat: &mut usize,
    ) -> Result<()> {
        let paths = self.paths(p, stage, flat);
        for (r, rp) in self.resnets.iter_mut().zip(&paths.resnets) {
            r.load(comp, rp)?;
        }
        for (a, ap) in self.attentions.iter_mut().zip(&paths.attentions) {
            a.load(comp, ap)?;
        }
        if let (Some(d), Some(dp)) = (self.downsampler.as_mut(), &paths.downsampler) {
            d.load(comp, dp)?;
        }
        Ok(())
    }

    pub fn names(&self, out: &mut Vec<TensorSpec>, p: &str, stage: usize, flat: &mut usize) {
        let paths = self.paths(p, stage, flat);
        for (r, rp) in self.resnets.iter().zip(&paths.resnets) {
            r.names(out, rp);
        }
        for (a, ap) in self.attentions.iter().zip(&paths.attentions) {
            a.names(out, ap);
        }
        if let (Some(d), Some(dp)) = (&self.downsampler, &paths.downsampler) {
            d.names(out, dp);
        }
    }
}

impl ResidualUpBlock {
    /// `upsampler` (Wan 2.2) or `upsamplers.0` (Wan 2.1 / Qwen-Image).
    fn upsampler_path(&self, p: &str) -> String {
        match self.kind {
            StageKind::Residual => format!("{p}.upsampler"),
            StageKind::Plain => format!("{p}.upsamplers.0"),
        }
    }

    pub fn load(&mut self, comp: &ComponentTensors, p: &str) -> Result<()> {
        for (i, r) in self.resnets.iter_mut().enumerate() {
            r.load(comp, &format!("{p}.resnets.{i}"))?;
        }
        let up = self.upsampler_path(p);
        if let Some(u) = self.upsampler.as_mut() {
            u.load(comp, &up)?;
        }
        Ok(())
    }

    pub fn names(&self, out: &mut Vec<TensorSpec>, p: &str) {
        for (i, r) in self.resnets.iter().enumerate() {
            r.names(out, &format!("{p}.resnets.{i}"));
        }
        if let Some(u) = &self.upsampler {
            u.names(out, &self.upsampler_path(p));
        }
    }
}

impl Decoder3d {
    pub fn load(&mut self, comp: &ComponentTensors, p: &str) -> Result<()> {
        load_conv(&mut self.conv_in, comp, &format!("{p}.conv_in"))?;
        self.mid_block.load(comp, &format!("{p}.mid_block"))?;
        for (i, b) in self.up_blocks.iter_mut().enumerate() {
            b.load(comp, &format!("{p}.up_blocks.{i}"))?;
        }
        load_norm(&mut self.norm_out, comp, &format!("{p}.norm_out"))?;
        load_conv(&mut self.conv_out, comp, &format!("{p}.conv_out"))
    }

    pub fn names(&self, out: &mut Vec<TensorSpec>, p: &str) {
        conv_names(out, &self.conv_in, &format!("{p}.conv_in"));
        self.mid_block.names(out, &format!("{p}.mid_block"));
        for (i, b) in self.up_blocks.iter().enumerate() {
            b.names(out, &format!("{p}.up_blocks.{i}"));
        }
        norm_names(out, &self.norm_out, &format!("{p}.norm_out"));
        conv_names(out, &self.conv_out, &format!("{p}.conv_out"));
    }
}

impl Encoder3d {
    pub fn load(&mut self, comp: &ComponentTensors, p: &str) -> Result<()> {
        load_conv(&mut self.conv_in, comp, &format!("{p}.conv_in"))?;
        let mut flat = 0;
        for (i, b) in self.down_blocks.iter_mut().enumerate() {
            b.load(comp, &format!("{p}.down_blocks"), i, &mut flat)?;
        }
        self.mid_block.load(comp, &format!("{p}.mid_block"))?;
        load_norm(&mut self.norm_out, comp, &format!("{p}.norm_out"))?;
        load_conv(&mut self.conv_out, comp, &format!("{p}.conv_out"))
    }

    pub fn names(&self, out: &mut Vec<TensorSpec>, p: &str) {
        conv_names(out, &self.conv_in, &format!("{p}.conv_in"));
        let mut flat = 0;
        for (i, b) in self.down_blocks.iter().enumerate() {
            b.names(out, &format!("{p}.down_blocks"), i, &mut flat);
        }
        self.mid_block.names(out, &format!("{p}.mid_block"));
        norm_names(out, &self.norm_out, &format!("{p}.norm_out"));
        conv_names(out, &self.conv_out, &format!("{p}.conv_out"));
    }
}

impl WanVae {
    /// Every tensor this model will ask for, with its expected element count,
    /// in load order.
    ///
    /// Used to validate a checkpoint before touching it, and to assert the
    /// model we build has exactly the shape the real one does.
    pub fn tensor_specs(&self) -> Vec<TensorSpec> {
        let mut out = Vec::new();
        self.encoder.names(&mut out, "encoder");
        self.decoder.names(&mut out, "decoder");
        conv_names(&mut out, &self.quant_conv, "quant_conv");
        conv_names(&mut out, &self.post_quant_conv, "post_quant_conv");
        out
    }

    /// Just the names, in load order.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensor_specs().into_iter().map(|(n, _)| n).collect()
    }

    /// Names present in the model but missing from `comp` — empty when the
    /// checkpoint satisfies the model.
    pub fn missing_tensors(&self, comp: &ComponentTensors) -> Vec<String> {
        self.tensor_names().into_iter().filter(|n| !comp.has(n)).collect()
    }

    /// Names whose element count in `comp` differs from what the config
    /// implies, so a shape error is reported before any weight is copied.
    pub fn mismatched_tensors(&self, comp: &ComponentTensors) -> Vec<String> {
        self.tensor_specs()
            .into_iter()
            .filter(|(n, want)| comp.numel(n).is_some_and(|got| got != *want))
            .map(|(n, _)| n)
            .collect()
    }

    /// Build from config and load every weight.
    pub fn load(cfg: super::WanVaeConfig, comp: &ComponentTensors) -> Result<Self> {
        let mut vae = WanVae::new(cfg);
        let missing = vae.missing_tensors(comp);
        if !missing.is_empty() {
            return Err(DiffusionError::Layout(format!(
                "Wan VAE checkpoint is missing {} tensor(s), first few: {:?}",
                missing.len(),
                &missing[..missing.len().min(5)]
            )));
        }
        let bad = vae.mismatched_tensors(comp);
        if !bad.is_empty() {
            return Err(DiffusionError::Layout(format!(
                "Wan VAE checkpoint has {} tensor(s) of unexpected size, first few: {:?}",
                bad.len(),
                &bad[..bad.len().min(5)]
            )));
        }
        vae.encoder.load(comp, "encoder")?;
        vae.decoder.load(comp, "decoder")?;
        load_conv(&mut vae.quant_conv, comp, "quant_conv")?;
        load_conv(&mut vae.post_quant_conv, comp, "post_quant_conv")?;
        Ok(vae)
    }
}
