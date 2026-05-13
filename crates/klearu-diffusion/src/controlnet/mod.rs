//! ControlNet (Zhang & Agrawala, 2023): an auxiliary network that injects
//! per-block residuals into a frozen UNet's skip connections, conditioning
//! generation on a control image (canny edges, depth, masked-image-for-
//! inpaint, …).
//!
//! Architecture mirrors the *encoder* half of the UNet:
//!   - `controlnet_cond_embedding`: small CNN that downsamples the control
//!     image (typically 3-channel RGB) by 8× to latent resolution, ending
//!     in `block_out_channels[0]` channels (320 for SD/SDXL).
//!   - `conv_in`: same as UNet — projects the noisy latent.
//!   - `time_embedding` (+ `text_time_embedding` for SDXL): same.
//!   - `down_blocks` + `mid_block`: identical to UNet's encoder
//!     (literally the same struct types, separate weights).
//!   - **ControlNet-specific tail**: zero-initialized 1×1 convs that
//!     project each skip from `down_blocks` (and the mid-block output)
//!     into a residual added to the corresponding UNet skip during the
//!     UNet's up-pass. `controlnet_down_blocks: Vec<Conv2d>` — one per
//!     skip in the order they're collected. `controlnet_mid_block: Conv2d`.
//!
//! Forward returns `(down_residuals, mid_residual)` matching the layout
//! the UNet expects (one residual per skip plus one for mid). The UNet's
//! `forward*` overloads accept these as optional slices and add them to
//! the corresponding skip / mid output before the up-pass consumes them.

use crate::config::UNetConfig;
use crate::error::{DiffusionError, Result};
use crate::layers::{Conv2d, Downsample, silu_inplace};
use crate::unet::blocks::{DownBlock2D, MidBlock};
use crate::unet::resnet_block::ResnetBlock2D;
use crate::unet::text_time_embedding::TextTimeEmbedding;
use crate::unet::time_embedding::TimeEmbedding;
use crate::unet::transformer_2d::Transformer2DModel;
use crate::unet::{SDXLAdditionalConditioning, expand_attention_head_dim};
use crate::weight::{ComponentTensors, load_conv2d, load_linear};

/// Configuration for a ControlNet. Most fields mirror `UNetConfig` (the
/// ControlNet's encoder must match the UNet's encoder block-for-block to
/// produce residuals at compatible shapes). The extra fields are for the
/// `controlnet_cond_embedding` tail.
#[derive(Debug, Clone)]
pub struct ControlNetConfig {
    pub unet: UNetConfig,
    /// Number of channels in the control image input. 3 for RGB
    /// (edges/depth/etc.); for some inpaint-specific ControlNets this may
    /// be 4 (RGB + mask) — the cond embedding's first conv adapts.
    pub conditioning_channels: usize,
    /// Channel progression of the cond embedding's intermediate convs.
    /// Diffusers default: [16, 32, 96, 256]. Three stride-2 downsamples
    /// happen at indices 1, 2, 3 (yielding 8× downscale to latent res).
    pub conditioning_embedding_out_channels: Vec<usize>,
}

impl ControlNetConfig {
    pub fn from_unet(unet: UNetConfig) -> Self {
        Self {
            unet,
            conditioning_channels: 3,
            conditioning_embedding_out_channels: vec![16, 32, 96, 256],
        }
    }
}

/// The mini-CNN that turns a `[B, conditioning_channels, H, W]` control
/// image into a `[B, block_out_channels[0], H/8, W/8]` feature map ready
/// to add to `conv_in`'s output.
///
/// Layout (Diffusers convention):
///   conv_in     : conditioning_channels → 16, 3×3 stride 1
///   blocks[0]   : 16 → 16, 3×3 stride 1
///   blocks[1]   : 16 → 32, 3×3 stride 2     ← downsample
///   blocks[2]   : 32 → 32, 3×3 stride 1
///   blocks[3]   : 32 → 96, 3×3 stride 2     ← downsample
///   blocks[4]   : 96 → 96, 3×3 stride 1
///   blocks[5]   : 96 → 256, 3×3 stride 2    ← downsample
///   blocks[6]   : 256 → 256, 3×3 stride 1
///   conv_out    : 256 → block_out_channels[0], 3×3 stride 1
///   (SiLU between every conv pair; conv_out is the final layer.)
pub struct ControlNetCondEmbedding {
    pub conv_in: Conv2d,
    pub blocks: Vec<Conv2d>,
    pub conv_out: Conv2d,
}

impl ControlNetCondEmbedding {
    pub fn new(
        in_channels: usize,
        out_channels: &[usize],   // [16, 32, 96, 256] typically
        block_out_channels_0: usize,
    ) -> Self {
        // First conv: in_channels → out_channels[0].
        let conv_in = Conv2d::new(in_channels, out_channels[0], 3, 1, 1, true);

        // Pair the channel sequence: alternate same-channel + downsample.
        // Build the explicit `blocks` Vec from the channel list.
        let mut blocks: Vec<Conv2d> = Vec::new();
        // For each pair of adjacent entries in out_channels:
        //   in_pair = out_channels[i],     out_pair = out_channels[i]    (stride 1)
        //   in_pair = out_channels[i],     out_pair = out_channels[i+1]  (stride 2)
        for i in 0..out_channels.len() {
            // Same-channel conv (stride 1).
            blocks.push(Conv2d::new(out_channels[i], out_channels[i], 3, 1, 1, true));
            // Downsample-or-final-same conv.
            if i + 1 < out_channels.len() {
                blocks.push(Conv2d::new(out_channels[i], out_channels[i + 1], 3, 2, 1, true));
            }
        }

        let last_oc = *out_channels.last().expect("out_channels non-empty");
        let conv_out = Conv2d::new(last_oc, block_out_channels_0, 3, 1, 1, true);

        Self { conv_in, blocks, conv_out }
    }

    /// Forward CPU. Input `[N, conditioning_channels, H, W]` (typically
    /// H=W=image side, i.e., 8× larger than latent). Output
    /// `[N, block_out_channels[0], H/8, W/8]`.
    pub fn forward(&self, image: &[f32], n: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let mut x = Vec::new();
        let (mut h_cur, mut w_cur) = self.conv_in.forward(image, n, h, w, &mut x);
        silu_inplace(&mut x);
        for blk in &self.blocks {
            let mut next = Vec::new();
            let (hn, wn) = blk.forward(&x, n, h_cur, w_cur, &mut next);
            x = next; h_cur = hn; w_cur = wn;
            silu_inplace(&mut x);
        }
        let mut out = Vec::new();
        let (hn, wn) = self.conv_out.forward(&x, n, h_cur, w_cur, &mut out);
        // No SiLU after conv_out; this is the final activation that gets
        // added to UNet's conv_in output.
        (out, hn, wn)
    }

    /// GPU forward. Same shape contract as `forward`; runs through
    /// Conv2d::forward_gpu and an in-place SiLU GPU kernel.
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        image: &crate::metal_backend::GpuTensor,
        n: usize, h: usize, w: usize,
    ) -> (crate::metal_backend::GpuTensor, usize, usize) {
        use crate::metal_backend::silu_f16_gpu;
        let (mut x, mut h_cur, mut w_cur) = self.conv_in.forward_gpu(image, n, h, w);
        silu_f16_gpu(&mut x);
        for blk in &self.blocks {
            let (next, hn, wn) = blk.forward_gpu(&x, n, h_cur, w_cur);
            x = next; h_cur = hn; w_cur = wn;
            silu_f16_gpu(&mut x);
        }
        let (out, hn, wn) = self.conv_out.forward_gpu(&x, n, h_cur, w_cur);
        (out, hn, wn)
    }

    /// Load weights from a diffusers-format component. Keys live under
    /// `controlnet_cond_embedding.*`.
    pub fn load_from(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        load_conv2d(comp, &format!("{prefix}.conv_in"), &mut self.conv_in)?;
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            load_conv2d(comp, &format!("{prefix}.blocks.{i}"), blk)?;
        }
        load_conv2d(comp, &format!("{prefix}.conv_out"), &mut self.conv_out)?;
        Ok(())
    }
}

/// The full ControlNet model.
pub struct ControlNet2DModel {
    pub config: ControlNetConfig,
    /// Mirrors UNet's conv_in. Loaded from `conv_in.*`.
    pub conv_in: Conv2d,
    /// `time_embedding.linear_1/linear_2` MLP (same as UNet).
    pub time_embedding: TimeEmbedding,
    /// SDXL only: pooled-text + size-id embedding (same as UNet).
    pub text_time_embedding: Option<TextTimeEmbedding>,
    pub down_blocks: Vec<DownBlock2D>,
    pub mid_block: MidBlock,
    /// Cond image → latent-resolution feature map.
    pub cond_embedding: ControlNetCondEmbedding,
    /// 1×1 zero-initialised projectors, one per skip from conv_in /
    /// down_blocks. Order matches the order the UNet collects skips.
    pub controlnet_down_blocks: Vec<Conv2d>,
    /// 1×1 zero-initialised projector for mid_block output.
    pub controlnet_mid_block: Conv2d,
}

impl ControlNet2DModel {
    pub fn from_config(config: ControlNetConfig) -> Self {
        let unet = &config.unet;
        let groups = unet.norm_num_groups;
        let eps = unet.norm_eps;
        let block_channels = unet.block_out_channels.clone();
        let layers_per_block = unet.layers_per_block;
        let cross_attn_dim = unet.cross_attention_dim;
        let n_blocks = block_channels.len();
        let time_embed_dim = block_channels[0] * 4;
        let head_dim_per_block = expand_attention_head_dim(&unet.attention_head_dim, n_blocks);
        let transformer_layers = unet.transformer_layers_expanded();

        let conv_in = Conv2d::new(unet.in_channels, block_channels[0], 3, 1, 1, true);
        let time_embedding = TimeEmbedding::new(block_channels[0], time_embed_dim);
        let text_time_embedding = if unet.is_sdxl() {
            Some(TextTimeEmbedding::new(
                unet.projection_class_embeddings_input_dim.unwrap_or(1280)
                    .saturating_sub(6 * unet.addition_time_embed_dim.unwrap_or(256)),
                unet.addition_time_embed_dim.unwrap_or(256),
                time_embed_dim,
            ))
        } else { None };

        // Build down_blocks identical to UNet's encoder (mirror the
        // logic in UNet2DConditionModel::from_config).
        let mut down_blocks = Vec::with_capacity(n_blocks);
        let mut prev_channels = block_channels[0];
        for i in 0..n_blocks {
            let out_c = block_channels[i];
            let is_cross_attn = unet.down_block_types.get(i)
                .map(|t| t.contains("CrossAttn"))
                .unwrap_or(true);
            let mut resnets = Vec::with_capacity(layers_per_block);
            let mut attns = Vec::with_capacity(layers_per_block);
            for j in 0..layers_per_block {
                let in_c = if j == 0 { prev_channels } else { out_c };
                resnets.push(ResnetBlock2D::new(in_c, out_c, Some(time_embed_dim), groups, eps));
                if is_cross_attn {
                    attns.push(Transformer2DModel::new(
                        out_c,
                        head_dim_per_block[i].max(1).min(out_c),
                        out_c / head_dim_per_block[i].max(1).max(1),
                        cross_attn_dim,
                        transformer_layers[i],
                        unet.use_linear_projection,
                        groups,
                    ));
                }
            }
            let downsample = if i + 1 < n_blocks {
                Some(Downsample::new(out_c))
            } else { None };
            down_blocks.push(DownBlock2D {
                resnets,
                attentions: if is_cross_attn { Some(attns) } else { None },
                downsample,
                channels_out: out_c,
            });
            prev_channels = out_c;
        }

        // Mid block — same as UNet's mid block.
        let mid_c = *block_channels.last().unwrap();
        let mid_layers = transformer_layers.last().copied().unwrap_or(1);
        let mid_head_dim = *head_dim_per_block.last().unwrap_or(&8);
        let mid_block = MidBlock {
            resnet_1: ResnetBlock2D::new(mid_c, mid_c, Some(time_embed_dim), groups, eps),
            attn: Transformer2DModel::new(
                mid_c,
                mid_head_dim,
                mid_c / mid_head_dim.max(1),
                cross_attn_dim,
                mid_layers,
                unet.use_linear_projection,
                groups,
            ),
            resnet_2: ResnetBlock2D::new(mid_c, mid_c, Some(time_embed_dim), groups, eps),
        };

        // Cond embedding (3 → 16 → … → 320).
        let cond_embedding = ControlNetCondEmbedding::new(
            config.conditioning_channels,
            &config.conditioning_embedding_out_channels,
            block_channels[0],
        );

        // controlnet_down_blocks: one 1×1 conv per skip. The skip layout
        // mirrors UNet's all_skips construction (see `unet/mod.rs::forward`):
        //
        //   skip 0: conv_in output                       (block_channels[0])
        //   for each down_block i:
        //     skip: resnet_0 output                      (block_channels[i])
        //     skip: resnet_1 output                      (block_channels[i])
        //     if i has downsample:
        //       skip: downsample output                  (block_channels[i])
        //
        // For SD 1.5 (4 blocks, last has no downsample): 1 + 3+3+3+2 = 12.
        // For SDXL (3 blocks, last has no downsample):   1 + 3+3+2     = 9.
        let mut controlnet_down_blocks: Vec<Conv2d> = Vec::new();
        // The conv_in skip — channels = block_channels[0].
        controlnet_down_blocks.push(Conv2d::new(block_channels[0], block_channels[0], 1, 1, 0, true));
        for i in 0..n_blocks {
            let c = block_channels[i];
            // Two resnet skips per block.
            for _ in 0..layers_per_block {
                controlnet_down_blocks.push(Conv2d::new(c, c, 1, 1, 0, true));
            }
            // Downsample skip (all but last block).
            if i + 1 < n_blocks {
                controlnet_down_blocks.push(Conv2d::new(c, c, 1, 1, 0, true));
            }
        }
        let controlnet_mid_block = Conv2d::new(mid_c, mid_c, 1, 1, 0, true);

        Self {
            config,
            conv_in,
            time_embedding,
            text_time_embedding,
            down_blocks,
            mid_block,
            cond_embedding,
            controlnet_down_blocks,
            controlnet_mid_block,
        }
    }

    /// Total number of down-residuals this ControlNet produces (matches
    /// UNet's `all_skips.len()` after the down-pass).
    pub fn num_down_residuals(&self) -> usize {
        self.controlnet_down_blocks.len()
    }

    /// Load weights from a diffusers-format ControlNet safetensors file.
    /// Keys live under root (no prefix). The mid-block is at `mid_block.*`,
    /// down_blocks at `down_blocks.<i>.*`, etc. — same naming as UNet.
    pub fn load_from(&mut self, comp: &ComponentTensors) -> Result<()> {
        load_conv2d(comp, "conv_in", &mut self.conv_in)?;
        self.time_embedding.load_from(comp, "time_embedding")?;
        if let Some(tte) = &mut self.text_time_embedding {
            tte.load_from(comp, "add_embedding")?;
        }

        // Down blocks: walk fields manually (matching UNet::load_from).
        for (i, db) in self.down_blocks.iter_mut().enumerate() {
            for (j, r) in db.resnets.iter_mut().enumerate() {
                r.load_from(comp, &format!("down_blocks.{i}.resnets.{j}"))?;
            }
            if let Some(attns) = &mut db.attentions {
                for (j, t) in attns.iter_mut().enumerate() {
                    t.load_from(comp, &format!("down_blocks.{i}.attentions.{j}"))?;
                }
            }
            if let Some(d) = &mut db.downsample {
                load_conv2d(comp, &format!("down_blocks.{i}.downsamplers.0.conv"), &mut d.conv)?;
            }
        }
        // Mid block: resnets[0,1] + attentions[0].
        self.mid_block.resnet_1.load_from(comp, "mid_block.resnets.0")?;
        self.mid_block.attn.load_from(comp, "mid_block.attentions.0")?;
        self.mid_block.resnet_2.load_from(comp, "mid_block.resnets.1")?;

        self.cond_embedding.load_from(comp, "controlnet_cond_embedding")?;

        for (i, conv) in self.controlnet_down_blocks.iter_mut().enumerate() {
            load_conv2d(comp, &format!("controlnet_down_blocks.{i}"), conv)?;
        }
        load_conv2d(comp, "controlnet_mid_block", &mut self.controlnet_mid_block)?;
        let _ = load_linear; // suppress unused-import warning when not directly invoked
        Ok(())
    }

    /// CPU forward. Returns `(down_residuals, mid_residual)` where
    /// `down_residuals.len() == num_down_residuals()`.
    ///
    /// `image` is the control image at full resolution (8× the latent),
    /// `[N, conditioning_channels, H_img, W_img]`. Layout matches
    /// klearu's standard CHW convention.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        latent: &[f32],
        timestep: f32,
        text_embedding: &[f32],
        image: &[f32],
        image_h: usize,
        image_w: usize,
        // SDXL only — pass `None` for SD 1.5.
        sdxl_pooled: Option<&[f32]>,
        sdxl_additional: Option<SDXLAdditionalConditioning>,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let cfg = &self.config.unet;
        let h = cfg.sample_size;
        let w = cfg.sample_size;
        let n = (latent.len() / (cfg.in_channels * h * w)).max(1);
        let text_seq = text_embedding.len() / (n * cfg.cross_attention_dim);

        // Time embedding (+ SDXL add_embedding).
        let mut time_emb = self.time_embedding.forward(timestep);
        if n > 1 {
            let single = time_emb.clone();
            time_emb.clear();
            for _ in 0..n { time_emb.extend_from_slice(&single); }
        }
        if let (Some(tte), Some(pooled), Some(addl)) = (
            self.text_time_embedding.as_ref(), sdxl_pooled, sdxl_additional,
        ) {
            let tids = addl.as_floats();
            let pooled_dim = pooled.len() / n.max(1);
            let mut add = Vec::with_capacity(n * tte.time_embed_dim);
            for ni in 0..n {
                let pooled_b = &pooled[ni * pooled_dim..(ni + 1) * pooled_dim];
                add.extend(tte.forward(pooled_b, &tids));
            }
            for (a, b) in time_emb.iter_mut().zip(add.iter()) { *a += b; }
        }

        // conv_in + cond_embedding addition.
        let mut x = Vec::new();
        let (mut h_cur, mut w_cur) = self.conv_in.forward(latent, n, h, w, &mut x);
        let (cond, ch, cw) = self.cond_embedding.forward(image, n, image_h, image_w);
        if ch != h_cur || cw != w_cur || cond.len() != x.len() {
            return Err(DiffusionError::ShapeMismatch {
                expected: format!("cond embedding to match conv_in output: \
                    h_cur={h_cur} w_cur={w_cur} len={}", x.len()),
                got: format!("ch={ch} cw={cw} len={}", cond.len()),
            });
        }
        for (a, b) in x.iter_mut().zip(cond.iter()) { *a += b; }

        // Collect skips like the UNet does.
        let mut all_skips: Vec<Vec<f32>> = Vec::new();
        all_skips.push(x.clone());
        for db in &self.down_blocks {
            let (out, skips, _skip_chans, hn, wn) = db.forward(
                &x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);
            x = out;
            all_skips.extend(skips);
            h_cur = hn; w_cur = wn;
        }

        // Mid block.
        let mid = self.mid_block.forward(
            &x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);

        // Project each skip and the mid output through their 1×1 convs.
        if all_skips.len() != self.controlnet_down_blocks.len() {
            return Err(DiffusionError::Unsupported(format!(
                "skip count mismatch: collected {} skips but ControlNet has {} projectors",
                all_skips.len(), self.controlnet_down_blocks.len()
            )));
        }
        let mut down_residuals: Vec<Vec<f32>> = Vec::with_capacity(all_skips.len());
        // The skips were collected at varying resolutions; we need to track
        // h/w per skip to feed the conv. Re-derive from skip length.
        for (i, skip) in all_skips.into_iter().enumerate() {
            let conv = &self.controlnet_down_blocks[i];
            // skip channels = conv.in_channels; spatial = skip.len() / (n * c).
            let c = conv.in_channels;
            let spatial = skip.len() / (n * c).max(1);
            let side = (spatial as f32).sqrt() as usize;
            // Skips are square (sample_size powers-of-two pyramid).
            let (sh, sw) = (side, side);
            let mut out = Vec::new();
            conv.forward(&skip, n, sh, sw, &mut out);
            down_residuals.push(out);
        }
        let mut mid_residual = Vec::new();
        self.controlnet_mid_block.forward(&mid, n, h_cur, w_cur, &mut mid_residual);

        Ok((down_residuals, mid_residual))
    }

    /// GPU forward. Returns the residuals as `GpuTensor`s ready to be fed
    /// to UNet's GPU forward path.
    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_gpu(
        &self,
        latent: &[f32],
        timestep: f32,
        text_embedding: &[f32],
        image: &[f32],
        image_h: usize,
        image_w: usize,
        sdxl_pooled: Option<&[f32]>,
        sdxl_additional: Option<SDXLAdditionalConditioning>,
    ) -> Result<(Vec<crate::metal_backend::GpuTensor>, crate::metal_backend::GpuTensor)> {
        use crate::metal_backend::*;
        let cfg = &self.config.unet;
        let h = cfg.sample_size;
        let w = cfg.sample_size;
        let n = (latent.len() / (cfg.in_channels * h * w)).max(1);
        let text_seq = text_embedding.len() / (n * cfg.cross_attention_dim);

        // Time embedding + SDXL add_embedding (on CPU; small + cached).
        let mut time_emb = self.time_embedding.forward(timestep);
        if n > 1 {
            let single = time_emb.clone();
            time_emb.clear();
            for _ in 0..n { time_emb.extend_from_slice(&single); }
        }
        if let (Some(tte), Some(pooled), Some(addl)) = (
            self.text_time_embedding.as_ref(), sdxl_pooled, sdxl_additional,
        ) {
            let tids = addl.as_floats();
            let pooled_dim = pooled.len() / n.max(1);
            let mut add = Vec::with_capacity(n * tte.time_embed_dim);
            for ni in 0..n {
                let pooled_b = &pooled[ni * pooled_dim..(ni + 1) * pooled_dim];
                add.extend(tte.forward(pooled_b, &tids));
            }
            for (a, b) in time_emb.iter_mut().zip(add.iter()) { *a += b; }
        }

        // Open a single command batch for the whole ControlNet pass.
        begin_batch();

        // Upload latent and run conv_in.
        let lat_gpu = GpuTensor::upload_f32_as_f16(
            vec![n, cfg.in_channels, h, w], latent);
        let (mut x, mut h_cur, mut w_cur) = self.conv_in.forward_gpu(&lat_gpu, n, h, w);
        drop(lat_gpu);

        // Upload control image, run cond_embedding, add to x.
        let img_gpu = GpuTensor::upload_f32_as_f16(
            vec![n, self.config.conditioning_channels, image_h, image_w], image);
        let (cond, ch, cw) = self.cond_embedding.forward_gpu(
            &img_gpu, n, image_h, image_w);
        drop(img_gpu);
        if ch != h_cur || cw != w_cur {
            return Err(DiffusionError::ShapeMismatch {
                expected: format!("cond embedding spatial to match conv_in: {h_cur}x{w_cur}"),
                got: format!("{ch}x{cw}"),
            });
        }
        eadd_f16_gpu(&mut x, &cond);
        drop(cond);

        // Collect skips just like UNet does.
        let mut all_skips: Vec<GpuTensor> = Vec::new();
        all_skips.push(x.clone_data());
        for db in &self.down_blocks {
            let (out, skips, _skip_chans, hn, wn) = db.forward_gpu(
                &x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);
            x = out;
            all_skips.extend(skips);
            h_cur = hn; w_cur = wn;
        }

        // Mid block.
        let mid = self.mid_block.forward_gpu(
            &x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);

        if all_skips.len() != self.controlnet_down_blocks.len() {
            return Err(DiffusionError::Unsupported(format!(
                "skip count mismatch: collected {} skips but ControlNet has {} projectors",
                all_skips.len(), self.controlnet_down_blocks.len()
            )));
        }

        // Project each skip + mid through 1×1 convs.
        let mut down_residuals: Vec<GpuTensor> = Vec::with_capacity(all_skips.len());
        for (i, skip) in all_skips.into_iter().enumerate() {
            let conv = &self.controlnet_down_blocks[i];
            let c = conv.in_channels;
            // Skip shape is [n, c, sh, sw].
            let sh = skip.shape[2];
            let sw = skip.shape[3];
            let _ = c;
            let (out, _, _) = conv.forward_gpu(&skip, n, sh, sw);
            down_residuals.push(out);
        }
        let (mid_residual, _, _) = self.controlnet_mid_block.forward_gpu(
            &mid, n, h_cur, w_cur);

        Ok((down_residuals, mid_residual))
    }
}
