//! UNet2DConditionModel — the denoising network at the heart of SD.
//!
//! Implements the conditional 2D UNet used by SD 1.5 and SDXL:
//!   - 2D convolutions (the dominant compute)
//!   - GroupNorm (32 groups, ε=1e-5)
//!   - Self-attention within the spatial latent grid
//!   - Cross-attention between latent and text embedding
//!   - Transformer 2D blocks (LayerNorm + FF + attn)
//!   - Time embedding (sinusoidal → MLP)
//!   - Down / Mid / Up sample blocks with residual connections
//!   - Skip connections from down to up blocks
//!
//! ## SDXL additions
//!
//!   - cross_attention_dim = 2048 (CLIP-L 768 + CLIP-G 1280 concat).
//!   - addition_embed_type = "text_time": an MLP that consumes the
//!     pooled CLIP-G embedding (1280-d) plus a sinusoidal encoding of
//!     6 size/crop conditioning floats (256-d each, 1536 total),
//!     concatenated to give 2816-d, projected to 1280-d, added to the
//!     time embedding.
//!   - transformer_layers_per_block is per-block (e.g., [1, 2, 10]):
//!     deeper transformer stacks at lower-resolution levels.
//!   - block_out_channels = [320, 640, 1280] (SD 1.5 has four blocks).

pub mod blocks;
pub mod resnet_block;
pub mod text_time_embedding;
pub mod time_embedding;
pub mod transformer_2d;
pub use blocks::{DownBlock2D, MidBlock, UpBlock2D};
pub use resnet_block::ResnetBlock2D;
pub use text_time_embedding::TextTimeEmbedding;
pub use time_embedding::{TimeEmbedding, sinusoidal_embedding};
pub use transformer_2d::{BasicTransformerBlock, FeedForwardGeGLU, Transformer2DModel};

use crate::config::{AttentionHeadDim, UNetConfig};
use crate::error::{DiffusionError, Result};
use crate::layers::{Conv2d, GroupNorm, silu_inplace};
use crate::weight::{ComponentTensors, load_conv2d, load_group_norm};

pub struct UNet2DConditionModel {
    pub config: UNetConfig,
    pub conv_in: Conv2d,
    pub time_embedding: TimeEmbedding,
    pub text_time_embedding: Option<TextTimeEmbedding>, // SDXL only
    pub down_blocks: Vec<DownBlock2D>,
    pub mid_block: MidBlock,
    pub up_blocks: Vec<UpBlock2D>,
    pub conv_norm_out: GroupNorm,
    pub conv_out: Conv2d,
}

/// SDXL conditioning ids: original / target / crop dimensions.
/// On SD 1.5 this is unused; on SDXL it feeds the addition_embed.
#[derive(Debug, Clone, Copy)]
pub struct SDXLAdditionalConditioning {
    pub original_size: (u32, u32),
    pub crops_coords_top_left: (u32, u32),
    pub target_size: (u32, u32),
}

impl SDXLAdditionalConditioning {
    /// Default for 1024×1024 generation with no cropping.
    pub fn default_1024() -> Self {
        Self {
            original_size: (1024, 1024),
            crops_coords_top_left: (0, 0),
            target_size: (1024, 1024),
        }
    }

    /// Returns the 6 conditioning floats in the order SDXL expects:
    /// [orig_h, orig_w, crop_top, crop_left, target_h, target_w].
    pub fn as_floats(&self) -> [f32; 6] {
        [
            self.original_size.0 as f32,
            self.original_size.1 as f32,
            self.crops_coords_top_left.0 as f32,
            self.crops_coords_top_left.1 as f32,
            self.target_size.0 as f32,
            self.target_size.1 as f32,
        ]
    }
}

impl UNet2DConditionModel {
    /// Construct from config; weights are NOT loaded yet (separate step).
    pub fn from_config(config: UNetConfig) -> Self {
        let groups = config.norm_num_groups;
        let eps = config.norm_eps;
        let block_channels = config.block_out_channels.clone();
        let layers_per_block = config.layers_per_block;
        let cross_attn_dim = config.cross_attention_dim;
        let n_blocks = block_channels.len();
        let time_embed_dim = block_channels[0] * 4; // SD: 320 → 1280
        let head_dim_per_block = expand_attention_head_dim(&config.attention_head_dim, n_blocks);
        let transformer_layers = config.transformer_layers_expanded();

        let conv_in = Conv2d::new(config.in_channels, block_channels[0], 3, 1, 1, true);
        let time_embedding = TimeEmbedding::new(block_channels[0], time_embed_dim);
        let text_time_embedding = if config.is_sdxl() {
            // pooled CLIP-G is 1280, time_id_dim=256, time_embed_dim=1280
            Some(TextTimeEmbedding::new(
                config.projection_class_embeddings_input_dim.unwrap_or(1280)
                    .saturating_sub(6 * config.addition_time_embed_dim.unwrap_or(256)),
                config.addition_time_embed_dim.unwrap_or(256),
                time_embed_dim,
            ))
        } else { None };

        // Build down blocks.
        let mut down_blocks = Vec::with_capacity(n_blocks);
        let mut prev_channels = block_channels[0];
        for i in 0..n_blocks {
            let out_c = block_channels[i];
            let is_cross_attn = config.down_block_types.get(i)
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
                        config.use_linear_projection,
                        groups,
                    ));
                }
            }
            // SDXL has no downsample on the last block; SD 1.5 has downsample on first 3.
            let downsample = if i + 1 < n_blocks {
                Some(crate::layers::Downsample::new(out_c))
            } else { None };
            down_blocks.push(DownBlock2D {
                resnets,
                attentions: if is_cross_attn { Some(attns) } else { None },
                downsample,
                channels_out: out_c,
            });
            prev_channels = out_c;
        }

        // Mid block.
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
                config.use_linear_projection,
                groups,
            ),
            resnet_2: ResnetBlock2D::new(mid_c, mid_c, Some(time_embed_dim), groups, eps),
        };

        // Up blocks (mirror of down). Each has layers_per_block + 1 resnets.
        let mut up_blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let block_idx = n_blocks - 1 - i;
            let out_c = block_channels[block_idx];
            let prev_out_c = if i == 0 { mid_c } else { block_channels[n_blocks - i] };
            let is_cross_attn = config.up_block_types.get(i)
                .map(|t| t.contains("CrossAttn"))
                .unwrap_or(true);
            let mut resnets = Vec::with_capacity(layers_per_block + 1);
            let mut attns = Vec::with_capacity(layers_per_block + 1);
            let mut skip_channels = Vec::with_capacity(layers_per_block + 1);
            for j in 0..(layers_per_block + 1) {
                // Skip channels come from the matching down block's outputs.
                let skip_idx_block = block_idx;
                let skip_c = if j == 0 && skip_idx_block + 1 < n_blocks {
                    // From the downsample output of block (skip_idx_block + 1) = block_channels[skip_idx_block]
                    block_channels[skip_idx_block]
                } else if j < layers_per_block {
                    block_channels[skip_idx_block]
                } else {
                    // Last skip is from one block higher (when present).
                    if skip_idx_block > 0 { block_channels[skip_idx_block - 1] }
                    else { block_channels[0] }
                };
                skip_channels.push(skip_c);
                let in_c = if j == 0 { prev_out_c + skip_c } else { out_c + skip_c };
                resnets.push(ResnetBlock2D::new(in_c, out_c, Some(time_embed_dim), groups, eps));
                if is_cross_attn {
                    attns.push(Transformer2DModel::new(
                        out_c,
                        head_dim_per_block[block_idx].max(1).min(out_c),
                        out_c / head_dim_per_block[block_idx].max(1),
                        cross_attn_dim,
                        transformer_layers[block_idx],
                        config.use_linear_projection,
                        groups,
                    ));
                }
            }
            let upsample = if i + 1 < n_blocks {
                Some(crate::layers::Upsample::new(out_c))
            } else { None };
            up_blocks.push(UpBlock2D {
                resnets,
                attentions: if is_cross_attn { Some(attns) } else { None },
                upsample,
                channels_out: out_c,
                skip_channels,
            });
        }

        let bottom_c = block_channels[0];
        Self {
            conv_in,
            time_embedding,
            text_time_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out: GroupNorm::new(groups, bottom_c, eps),
            conv_out: Conv2d::new(bottom_c, config.out_channels, 3, 1, 1, true),
            config,
        }
    }

    /// Forward pass for SD 1.5 / 2.x.
    ///
    /// Supports batched inference: `latent` of shape `[N, in_C, H, W]` flat
    /// and `text_embedding` of shape `[N, text_seq, cross_attn_dim]` flat.
    /// Used for batched CFG (N=2 — one each for unconditional and
    /// conditional pass). `N` is derived from `latent.len()`.
    pub fn forward(
        &self,
        latent: &[f32],
        timestep: f32,
        text_embedding: &[f32],
    ) -> Result<Vec<f32>> {
        let h = self.config.sample_size;
        let w = self.config.sample_size;
        let n = (latent.len() / (self.config.in_channels * h * w)).max(1);
        let text_seq = text_embedding.len() / (n * self.config.cross_attention_dim);

        // time_emb is computed once for the (single) timestep, then replicated
        // across batch — both batch elements use the same t.
        let mut time_emb = self.time_embedding.forward(timestep);
        if n > 1 {
            let single = time_emb.clone();
            time_emb.clear();
            for _ in 0..n { time_emb.extend_from_slice(&single); }
        }

        // conv_in
        let mut x = Vec::new();
        let (mut h_cur, mut w_cur) = self.conv_in.forward(latent, n, h, w, &mut x);

        // Down path: collect skips. The first skip is conv_in's output
        // (Diffusers convention) — the up-path's last resnet consumes it.
        let mut all_skips: Vec<Vec<f32>> = Vec::new();
        let mut all_skip_channels: Vec<usize> = Vec::new();
        let conv_in_channels = x.len() / (n * h_cur * w_cur).max(1);
        all_skip_channels.push(conv_in_channels);
        all_skips.push(x.clone());
        for db in &self.down_blocks {
            let (out, skips, skip_chans, hn, wn) = db.forward(&x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);
            x = out;
            for (s, cc) in skips.into_iter().zip(skip_chans) {
                all_skip_channels.push(cc);
                all_skips.push(s);
            }
            h_cur = hn; w_cur = wn;
        }

        // Mid.
        x = self.mid_block.forward(&x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);

        // Up path.
        for ub in &self.up_blocks {
            let (out, hn, wn) = ub.forward(&x, n, h_cur, w_cur, &mut all_skips, &mut all_skip_channels, &time_emb, text_embedding, text_seq);
            x = out; h_cur = hn; w_cur = wn;
        }

        // conv_norm_out + SiLU + conv_out
        self.conv_norm_out.forward_inplace(&mut x, n, h_cur, w_cur);
        silu_inplace(&mut x);
        let mut out = Vec::new();
        self.conv_out.forward(&x, n, h_cur, w_cur, &mut out);
        Ok(out)
    }

    /// Forward pass for SDXL: takes the additional conditioning (pooled
    /// CLIP-G embedding + size/crop ids) on top of the standard inputs.
    ///
    /// Inputs:
    ///   - `latent`: [batch, in_channels, 128, 128] for default 1024×1024
    ///   - `timestep`: scalar
    ///   - `text_embedding`: [batch, 77, 2048] (CLIP-L+G concat)
    ///   - `pooled_text_embedding`: [batch, 1280] (CLIP-G pooled)
    ///   - `additional_cond`: SDXLAdditionalConditioning (size/crop ids)
    pub fn forward_sdxl(
        &self,
        latent: &[f32],
        timestep: f32,
        text_embedding: &[f32],
        pooled_text_embedding: &[f32],
        additional_cond: SDXLAdditionalConditioning,
    ) -> Result<Vec<f32>> {
        if !self.config.is_sdxl() {
            return Err(DiffusionError::Unsupported(
                "forward_sdxl called on a non-SDXL UNet config".into(),
            ));
        }
        let h = self.config.sample_size;
        let w = self.config.sample_size;
        let n = (latent.len() / (self.config.in_channels * h * w)).max(1);
        let text_seq = text_embedding.len() / (n * self.config.cross_attention_dim);

        // Time embedding + SDXL TextTimeEmbedding addition. The TTE is per-batch
        // (different pooled embedding per batch element); time_emb is per-timestep
        // (single, replicated across batch). We expand both to [n × hidden] then
        // add elementwise.
        let mut time_emb = self.time_embedding.forward(timestep);
        if n > 1 {
            let single = time_emb.clone();
            time_emb.clear();
            for _ in 0..n { time_emb.extend_from_slice(&single); }
        }
        if let Some(tte) = &self.text_time_embedding {
            let tids = additional_cond.as_floats();
            let pooled_dim = pooled_text_embedding.len() / n.max(1);
            let mut add = Vec::with_capacity(n * tte.time_embed_dim);
            for ni in 0..n {
                let pooled_b = &pooled_text_embedding[ni * pooled_dim..(ni + 1) * pooled_dim];
                add.extend(tte.forward(pooled_b, &tids));
            }
            for (a, b) in time_emb.iter_mut().zip(add.iter()) { *a += b; }
        }

        let mut x = Vec::new();
        let (mut h_cur, mut w_cur) = self.conv_in.forward(latent, n, h, w, &mut x);

        let mut all_skips: Vec<Vec<f32>> = Vec::new();
        let mut all_skip_channels: Vec<usize> = Vec::new();
        // Push conv_in's output as the first skip (Diffusers convention).
        let conv_in_channels = x.len() / (n * h_cur * w_cur).max(1);
        all_skip_channels.push(conv_in_channels);
        all_skips.push(x.clone());
        for db in &self.down_blocks {
            let (out, skips, skip_chans, hn, wn) = db.forward(&x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);
            x = out;
            for (s, cc) in skips.into_iter().zip(skip_chans) {
                all_skip_channels.push(cc);
                all_skips.push(s);
            }
            h_cur = hn; w_cur = wn;
        }

        x = self.mid_block.forward(&x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);

        for ub in &self.up_blocks {
            let (out, hn, wn) = ub.forward(&x, n, h_cur, w_cur, &mut all_skips, &mut all_skip_channels, &time_emb, text_embedding, text_seq);
            x = out; h_cur = hn; w_cur = wn;
        }

        self.conv_norm_out.forward_inplace(&mut x, n, h_cur, w_cur);
        silu_inplace(&mut x);
        let mut out = Vec::new();
        self.conv_out.forward(&x, n, h_cur, w_cur, &mut out);
        Ok(out)
    }

    /// GPU-resident SD 1.5 / 2.x forward. Single CPU→GPU upload at the top,
    /// single GPU→CPU download at the bottom. Everything in between flows
    /// through fp16 GpuTensors. ResnetBlock chains stay GPU-resident; each
    /// Transformer2DModel call still bridges through CPU (until full GPU
    /// attention lands).
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        latent: &[f32],
        timestep: f32,
        text_embedding: &[f32],
    ) -> Result<Vec<f32>> {
        use crate::metal_backend::*;
        // Open a batched command buffer for the entire UNet step. Every
        // `*_f16_gpu` and MPS sgemm appends to this single cmd buffer
        // instead of creating its own — saves ~600+ new_command_buffer +
        // commit pairs per step. The `download_to_f32()` at the end
        // implicitly closes the batch via `flush()`.
        begin_batch();
        let h = self.config.sample_size;
        let w = self.config.sample_size;
        let n = (latent.len() / (self.config.in_channels * h * w)).max(1);
        let text_seq = text_embedding.len() / (n * self.config.cross_attention_dim);

        let mut time_emb = self.time_embedding.forward(timestep);
        if n > 1 {
            let single = time_emb.clone();
            time_emb.clear();
            for _ in 0..n { time_emb.extend_from_slice(&single); }
        }

        // Upload latent: f32 → fp16 GPU buffer.
        let x_in = GpuTensor::upload_f32_as_f16(
            vec![n, self.config.in_channels, h, w], latent);

        let trace = std::env::var_os("KLEARU_UNET_TRACE").is_some();
        let trace_stat = |name: &str, x: &GpuTensor| {
            if trace {
                let f32 = x.download_to_f32();
                let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
                let mut sum_abs = 0.0f64; let mut nan = 0usize; let mut inf = 0usize;
                for &v in &f32 {
                    if v.is_nan() { nan += 1; }
                    else if v.is_infinite() { inf += 1; }
                    else { mn = mn.min(v); mx = mx.max(v); sum_abs += v.abs() as f64; }
                }
                eprintln!("    [unet] {name:<30} shape={:?}, min={mn:.3}, max={mx:.3}, mean_abs={:.4}, NaN={nan}, Inf={inf}",
                          x.shape, sum_abs / f32.len() as f64);
            }
        };

        trace_stat("after upload (latent in)", &x_in);

        // conv_in
        let (mut x, mut h_cur, mut w_cur) = self.conv_in.forward_gpu(&x_in, n, h, w);
        drop(x_in);
        trace_stat("after conv_in", &x);

        // Down path. First skip is conv_in's output (Diffusers convention).
        let mut all_skips: Vec<GpuTensor> = Vec::new();
        let mut all_skip_channels: Vec<usize> = Vec::new();
        let conv_in_channels = x.shape[1];
        all_skip_channels.push(conv_in_channels);
        all_skips.push(x.clone_data());
        for (db_idx, db) in self.down_blocks.iter().enumerate() {
            let (out, skips, skip_chans, hn, wn) =
                db.forward_gpu(&x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);
            x = out;
            for (s, cc) in skips.into_iter().zip(skip_chans) {
                all_skip_channels.push(cc);
                all_skips.push(s);
            }
            h_cur = hn; w_cur = wn;
            trace_stat(&format!("after down_block[{db_idx}]"), &x);
        }

        // Mid.
        x = self.mid_block.forward_gpu(&x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);
        trace_stat("after mid_block", &x);

        // Up path.
        for (ub_idx, ub) in self.up_blocks.iter().enumerate() {
            let (out, hn, wn) = ub.forward_gpu(
                &x, n, h_cur, w_cur, &mut all_skips, &mut all_skip_channels,
                &time_emb, text_embedding, text_seq);
            x = out; h_cur = hn; w_cur = wn;
            trace_stat(&format!("after up_block[{ub_idx}]"), &x);
        }

        // conv_norm_out + SiLU + conv_out, GPU-resident.
        let no_g = weight_f16_buffer(&self.conv_norm_out.gamma);
        let no_b = weight_f16_buffer(&self.conv_norm_out.beta);
        let c_now = x.shape[1];
        if std::env::var_os("KLEARU_DISABLE_FUSED_NORMS").is_some() {
            groupnorm_f16_gpu(&mut x, &no_g, &no_b,
                              n, c_now, h_cur, w_cur,
                              self.conv_norm_out.num_groups, self.conv_norm_out.eps);
            silu_f16_gpu(&mut x);
        } else {
            groupnorm_silu_f16_gpu_inplace(&mut x, &no_g, &no_b,
                              n, c_now, h_cur, w_cur,
                              self.conv_norm_out.num_groups, self.conv_norm_out.eps);
        }
        let (out_gpu, _, _) = self.conv_out.forward_gpu(&x, n, h_cur, w_cur);

        // Download fp16 → f32 — one boundary download per UNet step.
        Ok(out_gpu.download_to_f32())
    }

    /// GPU-resident SDXL forward. Same structure as `forward_sdxl` but with
    /// fp16 GpuTensors throughout. The TextTimeEmbedding addition (per-batch
    /// pooled-CLIP-G + size/crop-id sinusoidals → linear → SiLU → linear)
    /// stays on CPU — it's a small fixed-cost computation that's recomputed
    /// per step regardless. The big UNet body chains on GPU.
    #[cfg(feature = "metal")]
    pub fn forward_sdxl_gpu(
        &self,
        latent: &[f32],
        timestep: f32,
        text_embedding: &[f32],
        pooled_text_embedding: &[f32],
        additional_cond: SDXLAdditionalConditioning,
    ) -> Result<Vec<f32>> {
        use crate::metal_backend::*;
        begin_batch();
        if !self.config.is_sdxl() {
            return Err(DiffusionError::Unsupported(
                "forward_sdxl_gpu called on a non-SDXL UNet config".into(),
            ));
        }
        let h = self.config.sample_size;
        let w = self.config.sample_size;
        let n = (latent.len() / (self.config.in_channels * h * w)).max(1);
        let text_seq = text_embedding.len() / (n * self.config.cross_attention_dim);

        // KLEARU_TIMING=1 prints per-section GPU timings. Inserts a
        // `flush()` between sections so the timings are exclusive — total
        // wall time grows because cross-section batching is disabled.
        let timing = std::env::var_os("KLEARU_TIMING").is_some();
        let tick = |label: &str, t0: std::time::Instant| {
            if timing {
                flush();
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("    [t] {:<24} {:>8.2} ms", label, ms);
            }
        };

        // Time embedding + SDXL TextTimeEmbedding addition (CPU — small).
        let mut time_emb = self.time_embedding.forward(timestep);
        if n > 1 {
            let single = time_emb.clone();
            time_emb.clear();
            for _ in 0..n { time_emb.extend_from_slice(&single); }
        }
        if let Some(tte) = &self.text_time_embedding {
            let tids = additional_cond.as_floats();
            let pooled_dim = pooled_text_embedding.len() / n.max(1);
            let mut add = Vec::with_capacity(n * tte.time_embed_dim);
            for ni in 0..n {
                let pooled_b = &pooled_text_embedding[ni * pooled_dim..(ni + 1) * pooled_dim];
                add.extend(tte.forward(pooled_b, &tids));
            }
            for (a, b) in time_emb.iter_mut().zip(add.iter()) { *a += b; }
        }

        // Upload latent.
        let t = std::time::Instant::now();
        let x_in = GpuTensor::upload_f32_as_f16(
            vec![n, self.config.in_channels, h, w], latent);
        let (mut x, mut h_cur, mut w_cur) = self.conv_in.forward_gpu(&x_in, n, h, w);
        drop(x_in);
        tick("conv_in", t);

        // Down path with conv_in skip.
        let mut all_skips: Vec<GpuTensor> = Vec::new();
        let mut all_skip_channels: Vec<usize> = Vec::new();
        let conv_in_channels = x.shape[1];
        all_skip_channels.push(conv_in_channels);
        all_skips.push(x.clone_data());
        for (i, db) in self.down_blocks.iter().enumerate() {
            let t = std::time::Instant::now();
            let (out, skips, skip_chans, hn, wn) =
                db.forward_gpu(&x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);
            x = out;
            for (s, cc) in skips.into_iter().zip(skip_chans) {
                all_skip_channels.push(cc);
                all_skips.push(s);
            }
            h_cur = hn; w_cur = wn;
            tick(&format!("down_block[{i}] @{h_cur}×{w_cur}"), t);
        }

        let t = std::time::Instant::now();
        x = self.mid_block.forward_gpu(&x, n, h_cur, w_cur, &time_emb, text_embedding, text_seq);
        tick("mid_block", t);

        for (i, ub) in self.up_blocks.iter().enumerate() {
            let t = std::time::Instant::now();
            let (out, hn, wn) = ub.forward_gpu(
                &x, n, h_cur, w_cur, &mut all_skips, &mut all_skip_channels,
                &time_emb, text_embedding, text_seq);
            x = out; h_cur = hn; w_cur = wn;
            tick(&format!("up_block[{i}]   @{h_cur}×{w_cur}"), t);
        }

        // Final norm + activation + conv.
        let t = std::time::Instant::now();
        let no_g = weight_f16_buffer(&self.conv_norm_out.gamma);
        let no_b = weight_f16_buffer(&self.conv_norm_out.beta);
        let c_now = x.shape[1];
        if std::env::var_os("KLEARU_DISABLE_FUSED_NORMS").is_some() {
            groupnorm_f16_gpu(&mut x, &no_g, &no_b,
                              n, c_now, h_cur, w_cur,
                              self.conv_norm_out.num_groups, self.conv_norm_out.eps);
            silu_f16_gpu(&mut x);
        } else {
            groupnorm_silu_f16_gpu_inplace(&mut x, &no_g, &no_b,
                              n, c_now, h_cur, w_cur,
                              self.conv_norm_out.num_groups, self.conv_norm_out.eps);
        }
        let (out_gpu, _, _) = self.conv_out.forward_gpu(&x, n, h_cur, w_cur);
        tick("conv_out + norm", t);

        Ok(out_gpu.download_to_f32())
    }
}

impl UNet2DConditionModel {
    /// Precompute cross-attention K, V projections from the (constant
    /// across timesteps) text embedding. Call once before the diffusion
    /// loop. Eliminates ~30 cross-attn × 25 timesteps = 750 redundant K/V
    /// projection matmuls per generation.
    ///
    /// `text_seq` is the sequence length of `text_embedding` (typically 77
    /// for CLIP-L; SDXL: same 77 with 2048-d concatenated channels).
    pub fn precompute_text_kv(&self, text_embedding: &[f32], text_seq: usize) {
        // Derive batch from buffer: text_embedding is [N, text_seq, cross_attn_dim].
        let n = (text_embedding.len() / (text_seq * self.config.cross_attention_dim)).max(1);
        let visit = |t2d: &Transformer2DModel| {
            for tb in &t2d.blocks {
                tb.cross_attn.precompute_kv(text_embedding, n, text_seq);
            }
        };
        for db in &self.down_blocks {
            if let Some(attns) = &db.attentions {
                for t2d in attns { visit(t2d); }
            }
        }
        visit(&self.mid_block.attn);
        for ub in &self.up_blocks {
            if let Some(attns) = &ub.attentions {
                for t2d in attns { visit(t2d); }
            }
        }
    }

    /// GPU-resident cross-attention K/V precompute. Computes K, V on CPU in
    /// **fp32** (Accelerate) for SDXL fidelity — CLIP-G text embeddings have
    /// extreme attention-sink values (±85) that lose precision in fp16
    /// matmul accumulators. Then uploads K, V to GPU as fp16 for the per-
    /// step attention dispatch.
    ///
    /// Saves ~3500 redundant K/V matmuls per SDXL inference (50 forwards ×
    /// 70 cross-attn layers × 2 projections). The one-time fp32 matmul costs
    /// ~50-100ms total — negligible vs. the 25-step UNet body.
    #[cfg(feature = "metal")]
    pub fn precompute_text_kv_gpu(&self, text_embedding: &[f32], text_seq: usize) {
        let n = (text_embedding.len() / (text_seq * self.config.cross_attention_dim)).max(1);
        let visit = |t2d: &Transformer2DModel| {
            for tb in &t2d.blocks {
                tb.cross_attn.precompute_kv_gpu_from_f32(text_embedding, n, text_seq);
            }
        };
        for db in &self.down_blocks {
            if let Some(attns) = &db.attentions {
                for t2d in attns { visit(t2d); }
            }
        }
        visit(&self.mid_block.attn);
        for ub in &self.up_blocks {
            if let Some(attns) = &ub.attentions {
                for t2d in attns { visit(t2d); }
            }
        }
    }

    /// Drop all precomputed cross-attention K/V caches (call between
    /// generations if the text changes).
    pub fn clear_text_kv(&self) {
        let visit = |t2d: &Transformer2DModel| {
            for tb in &t2d.blocks {
                tb.cross_attn.clear_kv_cache();
            }
        };
        for db in &self.down_blocks {
            if let Some(attns) = &db.attentions {
                for t2d in attns { visit(t2d); }
            }
        }
        visit(&self.mid_block.attn);
        for ub in &self.up_blocks {
            if let Some(attns) = &ub.attentions {
                for t2d in attns { visit(t2d); }
            }
        }
    }

    /// Load all UNet weights from the `unet/` component.
    pub fn load_from(&mut self, comp: &ComponentTensors) -> Result<()> {
        load_conv2d(comp, "conv_in", &mut self.conv_in)?;
        self.time_embedding.load_from(comp, "time_embedding")?;
        if let Some(tte) = &mut self.text_time_embedding {
            tte.load_from(comp, "add_embedding")?;
        }
        // Down blocks: each has resnets, optional attentions, optional downsample.
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
        // Mid block: resnets[0,1] + attentions[0]
        self.mid_block.resnet_1.load_from(comp, "mid_block.resnets.0")?;
        self.mid_block.attn.load_from(comp, "mid_block.attentions.0")?;
        self.mid_block.resnet_2.load_from(comp, "mid_block.resnets.1")?;
        // Up blocks
        for (i, ub) in self.up_blocks.iter_mut().enumerate() {
            for (j, r) in ub.resnets.iter_mut().enumerate() {
                r.load_from(comp, &format!("up_blocks.{i}.resnets.{j}"))?;
            }
            if let Some(attns) = &mut ub.attentions {
                for (j, t) in attns.iter_mut().enumerate() {
                    t.load_from(comp, &format!("up_blocks.{i}.attentions.{j}"))?;
                }
            }
            if let Some(up) = &mut ub.upsample {
                load_conv2d(comp, &format!("up_blocks.{i}.upsamplers.0.conv"), &mut up.conv)?;
            }
        }
        load_group_norm(comp, "conv_norm_out", &mut self.conv_norm_out)?;
        load_conv2d(comp, "conv_out", &mut self.conv_out)?;
        Ok(())
    }
}

fn expand_attention_head_dim(d: &AttentionHeadDim, n_blocks: usize) -> Vec<usize> {
    match d {
        AttentionHeadDim::Single(n) => vec![*n; n_blocks],
        AttentionHeadDim::PerBlock(v) => v.clone(),
    }
}
