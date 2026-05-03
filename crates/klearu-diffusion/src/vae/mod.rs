//! AutoencoderKL — VAE decoder/encoder for SD.
//!
//! Same architecture for SD 1.5 and SDXL (the VAE is shared). Decoder
//! turns 4-channel latent [B, 4, H, W] into RGB image [B, 3, 8H, 8W].
//!
//! Topology (decoder):
//!   - post_quant_conv: Conv 1×1 (latent_channels → latent_channels)
//!   - decoder.conv_in: Conv 3×3 (latent_channels → block_out_channels[-1])
//!   - decoder.mid_block: ResnetBlock + AttentionBlock + ResnetBlock
//!   - decoder.up_blocks: 4× UpBlock2D (3 ResnetBlocks + Upsample)
//!   - decoder.conv_norm_out: GroupNorm
//!   - SiLU
//!   - decoder.conv_out: Conv 3×3 (block_out_channels[0] → 3)
//!
//! Encoder is the mirror with stride-2 conv downsamples instead of
//! upsamples; produces a `DiagonalGaussianDistribution` (mean + logvar)
//! sampled once.

use crate::config::VAEConfig;
use crate::error::Result;
use crate::layers::{Conv2d, GroupNorm, Upsample, Downsample, silu_inplace};
use crate::unet::resnet_block::ResnetBlock2D;
use crate::weight::{ComponentTensors, load_conv2d, load_group_norm, load_linear};

/// Helper: load a VAEAttentionBlock at HF Diffusers prefix.
/// Names: `<prefix>.group_norm.{weight,bias}`,
///        `<prefix>.{to_q,to_k,to_v}.{weight,bias}`,
///        `<prefix>.to_out.0.{weight,bias}` (PyTorch nn.Sequential index 0 = the linear).
pub fn load_attention_block(
    comp: &ComponentTensors,
    prefix: &str,
    target: &mut VAEAttentionBlock,
) -> Result<()> {
    load_group_norm(comp, &format!("{prefix}.group_norm"), &mut target.group_norm)?;
    load_linear(comp, &format!("{prefix}.to_q"), &mut target.to_q)?;
    load_linear(comp, &format!("{prefix}.to_k"), &mut target.to_k)?;
    load_linear(comp, &format!("{prefix}.to_v"), &mut target.to_v)?;
    load_linear(comp, &format!("{prefix}.to_out.0"), &mut target.to_out)?;
    Ok(())
}

/// Self-attention block used in the VAE mid-block (no time conditioning,
/// no cross-attention, just spatial self-attention).
pub struct VAEAttentionBlock {
    pub group_norm: GroupNorm,
    pub to_q: crate::layers::Linear,
    pub to_k: crate::layers::Linear,
    pub to_v: crate::layers::Linear,
    pub to_out: crate::layers::Linear,
    pub channels: usize,
}

impl VAEAttentionBlock {
    pub fn new(channels: usize, groups: usize, eps: f32) -> Self {
        Self {
            group_norm: GroupNorm::new(groups, channels, eps),
            to_q: crate::layers::Linear::new(channels, channels, true),
            to_k: crate::layers::Linear::new(channels, channels, true),
            to_v: crate::layers::Linear::new(channels, channels, true),
            to_out: crate::layers::Linear::new(channels, channels, true),
            channels,
        }
    }

    /// Forward [N, C, H, W] → [N, C, H, W] residual.
    /// Single-head attention over flattened spatial positions.
    pub fn forward(&self, input: &[f32], n: usize, h: usize, w: usize) -> Vec<f32> {
        let c = self.channels;
        let hw = h * w;
        // norm
        let mut x = input.to_vec();
        self.group_norm.forward_inplace(&mut x, n, h, w);
        // [N,C,H,W] → [N, HW, C]
        let mut seq = vec![0.0f32; n * hw * c];
        for ni in 0..n {
            for ci in 0..c {
                for j in 0..hw {
                    seq[ni * hw * c + j * c + ci] = x[ni * c * hw + ci * hw + j];
                }
            }
        }
        // q, k, v all have same shape [n*hw*c]
        let mut q = vec![0.0f32; n * hw * c];
        let mut k = vec![0.0f32; n * hw * c];
        let mut v = vec![0.0f32; n * hw * c];
        self.to_q.forward_batch(&seq, &mut q);
        self.to_k.forward_batch(&seq, &mut k);
        self.to_v.forward_batch(&seq, &mut v);
        // Single-head attention.
        let scale = 1.0 / (c as f32).sqrt();
        let mut out_seq = vec![0.0f32; n * hw * c];
        for ni in 0..n {
            for lq in 0..hw {
                let q_off = ni * hw * c + lq * c;
                let mut scores = vec![0.0f32; hw];
                for lk in 0..hw {
                    let k_off = ni * hw * c + lk * c;
                    let mut s = 0.0f32;
                    for i in 0..c { s += q[q_off + i] * k[k_off + i]; }
                    scores[lk] = s * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
                if sum > 0.0 { let inv = 1.0 / sum; for s in scores.iter_mut() { *s *= inv; } }
                let out_off = ni * hw * c + lq * c;
                for lk in 0..hw {
                    let v_off = ni * hw * c + lk * c;
                    let w = scores[lk];
                    if w == 0.0 { continue; }
                    for i in 0..c { out_seq[out_off + i] += w * v[v_off + i]; }
                }
            }
        }
        // proj_out
        let mut proj = vec![0.0f32; n * hw * c];
        self.to_out.forward_batch(&out_seq, &mut proj);
        // [N,HW,C] → [N,C,H,W]
        let mut nchw = vec![0.0f32; n * c * hw];
        for ni in 0..n {
            for ci in 0..c {
                for j in 0..hw {
                    nchw[ni * c * hw + ci * hw + j] = proj[ni * hw * c + j * c + ci];
                }
            }
        }
        // residual
        for (a, b) in nchw.iter_mut().zip(input.iter()) { *a += b; }
        nchw
    }

    /// GPU-resident forward. Same shape as `forward` but every op chains on
    /// GPU buffers. Single-head attention via `flash_attention_f16_gpu`
    /// (treating channels as head_dim, num_heads=1).
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        input: &crate::metal_backend::GpuTensor,
        n: usize, h: usize, w: usize,
    ) -> crate::metal_backend::GpuTensor {
        use crate::metal_backend::*;
        let c = self.channels;
        let hw = h * w;

        // GroupNorm input (out-of-place — keep `input` for residual).
        let g = weight_f16_buffer(&self.group_norm.gamma);
        let b = weight_f16_buffer(&self.group_norm.beta);
        let mut x = input.clone_data();
        groupnorm_f16_gpu(&mut x, &g, &b, n, c, h, w,
                          self.group_norm.num_groups, self.group_norm.eps);

        // NCHW → NHWC for sequence-style ops.
        let seq = nchw_to_nhwc_f16_gpu(&x, n, c, hw);
        drop(x);

        // Q, K, V via Linear (c → c).
        let q = self.to_q.forward_gpu(&seq);
        let k = self.to_k.forward_gpu(&seq);
        let v = self.to_v.forward_gpu(&seq);
        drop(seq);

        // Single-head attention. flash_attention_f16_gpu expects [n, h, l_q,
        // d]; with h=1 the layout is identical to the [n, l_q, d] we have.
        let mut o = GpuTensor::new_f16(vec![n, 1, hw, c]);
        let scale = 1.0f32 / (c as f32).sqrt();
        flash_attention_f16_gpu(&q, &k, &v, &mut o, n, 1, hw, hw, c, scale);
        drop(q); drop(k); drop(v);

        // proj_out (Linear c → c).
        let o = o.reshape(vec![n * hw, c]);
        let proj = self.to_out.forward_gpu(&o).reshape(vec![n, hw, c]);
        drop(o);

        // NHWC → NCHW.
        let mut out = nhwc_to_nchw_f16_gpu(&proj, n, c, hw).reshape(vec![n, c, h, w]);
        drop(proj);

        // residual + input.
        eadd_f16_gpu(&mut out, input);
        out
    }
}

/// VAE decoder.
pub struct VAEDecoder {
    pub conv_in: Conv2d,                       // latent → top-channel-block
    pub mid_resnet_1: ResnetBlock2D,
    pub mid_attn: VAEAttentionBlock,
    pub mid_resnet_2: ResnetBlock2D,
    pub up_blocks: Vec<UpBlock2DVae>,
    pub conv_norm_out: GroupNorm,
    pub conv_out: Conv2d,                      // → 3 channels
    pub block_out_channels: Vec<usize>,
}

pub struct UpBlock2DVae {
    pub resnets: Vec<ResnetBlock2D>,
    pub upsample: Option<Upsample>, // None on the last block
    pub channels: usize,
}

impl VAEDecoder {
    pub fn from_config(cfg: &VAEConfig) -> Self {
        let groups = cfg.norm_num_groups;
        let eps = 1e-6;
        let top_c = *cfg.block_out_channels.last().unwrap();

        // mid block
        let mid_resnet_1 = ResnetBlock2D::new(top_c, top_c, None, groups, eps);
        let mid_attn = VAEAttentionBlock::new(top_c, groups, eps);
        let mid_resnet_2 = ResnetBlock2D::new(top_c, top_c, None, groups, eps);

        // up blocks: walk block_out_channels in reverse.
        let n_blocks = cfg.block_out_channels.len();
        let mut up_blocks: Vec<UpBlock2DVae> = Vec::with_capacity(n_blocks);
        let mut prev_out = top_c;
        let layers_per_block = cfg.layers_per_block + 1; // SD VAE has +1 layer in decoder
        for i in 0..n_blocks {
            let block_idx = n_blocks - 1 - i;
            let out_c = cfg.block_out_channels[block_idx];
            let mut resnets = Vec::with_capacity(layers_per_block);
            for j in 0..layers_per_block {
                let in_c = if j == 0 { prev_out } else { out_c };
                resnets.push(ResnetBlock2D::new(in_c, out_c, None, groups, eps));
            }
            // No upsample on the last (highest-resolution) block.
            let upsample = if i < n_blocks - 1 { Some(Upsample::new(out_c)) } else { None };
            up_blocks.push(UpBlock2DVae { resnets, upsample, channels: out_c });
            prev_out = out_c;
        }

        let bottom_c = cfg.block_out_channels[0];
        Self {
            conv_in: Conv2d::new(cfg.latent_channels, top_c, 3, 1, 1, true),
            mid_resnet_1,
            mid_attn,
            mid_resnet_2,
            up_blocks,
            conv_norm_out: GroupNorm::new(groups, bottom_c, eps),
            conv_out: Conv2d::new(bottom_c, cfg.out_channels, 3, 1, 1, true),
            block_out_channels: cfg.block_out_channels.clone(),
        }
    }

    /// Decode latent [N, latent_C, H, W] to image [N, 3, H*8, W*8].
    pub fn forward(&self, latent: &[f32], n: usize, h: usize, w: usize) -> Vec<f32> {
        let trace = std::env::var_os("KLEARU_VAE_TRACE").is_some();
        let stat = |name: &str, x: &[f32]| {
            if trace {
                let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
                let mut sum_abs = 0.0f64; let mut nan = 0usize; let mut inf = 0usize;
                for &v in x {
                    if v.is_nan() { nan += 1; }
                    else if v.is_infinite() { inf += 1; }
                    else { mn = mn.min(v); mx = mx.max(v); sum_abs += v.abs() as f64; }
                }
                eprintln!("    [vae] {name:<24} len={}, min={mn:.4}, max={mx:.4}, mean_abs={:.5}, NaN={nan}, Inf={inf}",
                          x.len(), sum_abs / x.len() as f64);
            }
        };

        // conv_in
        let mut x = Vec::new();
        let (mut h, mut w) = self.conv_in.forward(latent, n, h, w, &mut x);
        stat("after conv_in", &x);

        // Mid block (no time conditioning).
        x = self.mid_resnet_1.forward(&x, n, h, w, &[]);
        stat("after mid_resnet_1", &x);
        x = self.mid_attn.forward(&x, n, h, w);
        stat("after mid_attn", &x);
        x = self.mid_resnet_2.forward(&x, n, h, w, &[]);
        stat("after mid_resnet_2", &x);

        // Up blocks
        for (ub_idx, ub) in self.up_blocks.iter().enumerate() {
            for (r_idx, r) in ub.resnets.iter().enumerate() {
                x = r.forward(&x, n, h, w, &[]);
                stat(&format!("up{ub_idx}.resnet{r_idx}"), &x);
            }
            if let Some(up) = &ub.upsample {
                let (next, h_next, w_next) = up.forward(&x, n, h, w);
                x = next; h = h_next; w = w_next;
                stat(&format!("up{ub_idx}.upsample"), &x);
            }
        }

        // out norm + silu + out conv
        self.conv_norm_out.forward_inplace(&mut x, n, h, w);
        stat("after conv_norm_out", &x);
        silu_inplace(&mut x);
        let mut out = Vec::new();
        self.conv_out.forward(&x, n, h, w, &mut out);
        stat("after conv_out", &out);
        out
    }

    /// GPU-resident decode forward. Mirror of `forward` but every op
    /// (Conv2d, ResnetBlock2D, attention, Upsample, GroupNorm, SiLU) chains
    /// on GPU via fp16 GpuTensors. Input is f32, output is f32 (boundary
    /// upload + download on either side).
    #[cfg(feature = "metal")]
    pub fn forward_gpu(&self, latent: &[f32], n: usize, h: usize, w: usize) -> Vec<f32> {
        use crate::metal_backend::*;
        // Batch all VAE dispatches into a single command buffer (closed by
        // download_to_f32 at end). Saves ~hundred new_command_buffer/commit
        // pairs per decode.
        begin_batch();
        // VAE intermediate activations grow large (SDXL VAE decoder can
        // reach ~23K) — beyond fp16 max of 65504 once a Linear/Conv output
        // gets cast back to f16 storage in the fast sgemm path. Force the
        // f32-precision sgemm wrapper for the duration of decode.
        set_use_f32_sgemm(true);

        // Upload latent → fp16 GPU.
        let lc = self.conv_in.in_channels;
        let x_in = GpuTensor::upload_f32_as_f16(vec![n, lc, h, w], latent);

        // conv_in.
        let (mut x, mut h_cur, mut w_cur) = self.conv_in.forward_gpu(&x_in, n, h, w);
        drop(x_in);

        // Mid block: resnet → attn → resnet (no time conditioning).
        x = self.mid_resnet_1.forward_gpu(&x, n, h_cur, w_cur, &[]);
        x = self.mid_attn.forward_gpu(&x, n, h_cur, w_cur);
        x = self.mid_resnet_2.forward_gpu(&x, n, h_cur, w_cur, &[]);

        // Up blocks.
        for ub in &self.up_blocks {
            for r in &ub.resnets {
                x = r.forward_gpu(&x, n, h_cur, w_cur, &[]);
            }
            if let Some(up) = &ub.upsample {
                let (next, hn, wn) = up.forward_gpu(&x, n, h_cur, w_cur);
                x = next; h_cur = hn; w_cur = wn;
            }
        }

        // conv_norm_out + SiLU (fused) + conv_out.
        let no_g = weight_f16_buffer(&self.conv_norm_out.gamma);
        let no_b = weight_f16_buffer(&self.conv_norm_out.beta);
        let c_now = x.shape[1];
        let disable_fused = std::env::var_os("KLEARU_DISABLE_FUSED_NORMS").is_some();
        if disable_fused {
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

        // One boundary download. Reset the f32-sgemm flag so subsequent
        // (e.g., next REPL iteration) UNet matmuls go back to the fast f16
        // path. download_to_f32 implicitly closes the open command batch
        // via flush().
        let result = out_gpu.download_to_f32();
        set_use_f32_sgemm(false);
        result
    }
}

/// VAE encoder. Mirror of decoder: in_conv (3 → bottom_c) → 4× DownBlock
/// (2 ResnetBlocks + Downsample) → mid_block (Resnet + Attn + Resnet) →
/// out_norm → SiLU → out_conv → quant_conv (latent_channels*2 = mean+logvar).
pub struct VAEEncoder {
    pub conv_in: Conv2d,
    pub down_blocks: Vec<DownBlock2DVae>,
    pub mid_resnet_1: ResnetBlock2D,
    pub mid_attn: VAEAttentionBlock,
    pub mid_resnet_2: ResnetBlock2D,
    pub conv_norm_out: GroupNorm,
    pub conv_out: Conv2d,                      // → 2 * latent_channels (mean + logvar)
    pub quant_conv: Conv2d,                    // 1×1 conv before split
    pub block_out_channels: Vec<usize>,
    pub latent_channels: usize,
}

pub struct DownBlock2DVae {
    pub resnets: Vec<ResnetBlock2D>,
    pub downsample: Option<Downsample>,
    pub channels: usize,
}

impl VAEEncoder {
    pub fn from_config(cfg: &VAEConfig) -> Self {
        let groups = cfg.norm_num_groups;
        let eps = 1e-6;
        let bottom_c = cfg.block_out_channels[0];

        let n_blocks = cfg.block_out_channels.len();
        let mut down_blocks: Vec<DownBlock2DVae> = Vec::with_capacity(n_blocks);
        let mut prev_out = bottom_c;
        for i in 0..n_blocks {
            let out_c = cfg.block_out_channels[i];
            let mut resnets = Vec::with_capacity(cfg.layers_per_block);
            for j in 0..cfg.layers_per_block {
                let in_c = if j == 0 { prev_out } else { out_c };
                resnets.push(ResnetBlock2D::new(in_c, out_c, None, groups, eps));
            }
            let downsample = if i + 1 < n_blocks { Some(Downsample::new(out_c)) } else { None };
            down_blocks.push(DownBlock2DVae { resnets, downsample, channels: out_c });
            prev_out = out_c;
        }

        let top_c = *cfg.block_out_channels.last().unwrap();
        Self {
            conv_in: Conv2d::new(cfg.in_channels, bottom_c, 3, 1, 1, true),
            down_blocks,
            mid_resnet_1: ResnetBlock2D::new(top_c, top_c, None, groups, eps),
            mid_attn: VAEAttentionBlock::new(top_c, groups, eps),
            mid_resnet_2: ResnetBlock2D::new(top_c, top_c, None, groups, eps),
            conv_norm_out: GroupNorm::new(groups, top_c, eps),
            conv_out: Conv2d::new(top_c, 2 * cfg.latent_channels, 3, 1, 1, true),
            quant_conv: Conv2d::new(2 * cfg.latent_channels, 2 * cfg.latent_channels, 1, 1, 0, true),
            block_out_channels: cfg.block_out_channels.clone(),
            latent_channels: cfg.latent_channels,
        }
    }

    /// Encode image [N, 3, H, W] → mean + logvar, each [N, latent_channels, H/8, W/8].
    /// Returns (mean, logvar) flat vecs.
    pub fn forward(&self, image: &[f32], n: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::new();
        let (mut h_cur, mut w_cur) = self.conv_in.forward(image, n, h, w, &mut x);
        for db in &self.down_blocks {
            for r in &db.resnets {
                x = r.forward(&x, n, h_cur, w_cur, &[]);
            }
            if let Some(d) = &db.downsample {
                let (next, hn, wn) = d.forward(&x, n, h_cur, w_cur);
                x = next; h_cur = hn; w_cur = wn;
            }
        }
        x = self.mid_resnet_1.forward(&x, n, h_cur, w_cur, &[]);
        x = self.mid_attn.forward(&x, n, h_cur, w_cur);
        x = self.mid_resnet_2.forward(&x, n, h_cur, w_cur, &[]);
        self.conv_norm_out.forward_inplace(&mut x, n, h_cur, w_cur);
        silu_inplace(&mut x);
        let mut conv_out = Vec::new();
        self.conv_out.forward(&x, n, h_cur, w_cur, &mut conv_out);
        let mut quant = Vec::new();
        self.quant_conv.forward(&conv_out, n, h_cur, w_cur, &mut quant);
        // Split into mean and logvar along channel axis.
        let total_c = 2 * self.latent_channels;
        let lc = self.latent_channels;
        let hw = h_cur * w_cur;
        let mut mean = vec![0.0f32; n * lc * hw];
        let mut logvar = vec![0.0f32; n * lc * hw];
        for ni in 0..n {
            for ci in 0..lc {
                let m_off = ni * lc * hw + ci * hw;
                let v_off = ni * lc * hw + ci * hw;
                let q_m = ni * total_c * hw + ci * hw;
                let q_v = ni * total_c * hw + (ci + lc) * hw;
                mean[m_off..m_off + hw].copy_from_slice(&quant[q_m..q_m + hw]);
                logvar[v_off..v_off + hw].copy_from_slice(&quant[q_v..q_v + hw]);
            }
        }
        (mean, logvar)
    }
}

pub struct AutoencoderKL {
    pub config: VAEConfig,
    pub decoder: VAEDecoder,
    pub encoder: VAEEncoder,
    /// 1×1 conv applied to the latent BEFORE the decoder. Diffusers' VAE
    /// decode is `decoder(post_quant_conv(latent))` — without this, the
    /// decoder receives the latent in the wrong "space" and produces an
    /// image roughly anti-correlated with the correct one (cos≈-0.4).
    pub post_quant_conv: Conv2d,
}

impl AutoencoderKL {
    pub fn from_config(config: VAEConfig) -> Self {
        let decoder = VAEDecoder::from_config(&config);
        let encoder = VAEEncoder::from_config(&config);
        let post_quant_conv = Conv2d::new(
            config.latent_channels, config.latent_channels, 1, 1, 0, true,
        );
        Self { config, decoder, encoder, post_quant_conv }
    }

    /// Load all VAE weights (encoder + decoder + quant convs) from a
    /// component (the `vae/` directory's safetensors files).
    pub fn load_from(&mut self, comp: &ComponentTensors) -> Result<()> {
        // --- Decoder ---
        load_conv2d(comp, "decoder.conv_in", &mut self.decoder.conv_in)?;
        // mid_block: resnets[0,1] + attentions[0]
        self.decoder.mid_resnet_1.load_from(comp, "decoder.mid_block.resnets.0")?;
        load_attention_block(comp, "decoder.mid_block.attentions.0", &mut self.decoder.mid_attn)?;
        self.decoder.mid_resnet_2.load_from(comp, "decoder.mid_block.resnets.1")?;
        // up_blocks (HF convention iterates them as 0..N from highest to lowest resolution
        // in the up path; our struct stores them in the same order we forward, which is
        // top → bottom of channel pyramid — matching HF's "up_blocks.0..N").
        for (i, ub) in self.decoder.up_blocks.iter_mut().enumerate() {
            for (j, r) in ub.resnets.iter_mut().enumerate() {
                r.load_from(comp, &format!("decoder.up_blocks.{i}.resnets.{j}"))?;
            }
            if let Some(up) = &mut ub.upsample {
                load_conv2d(comp, &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                    &mut up.conv)?;
            }
        }
        load_group_norm(comp, "decoder.conv_norm_out", &mut self.decoder.conv_norm_out)?;
        load_conv2d(comp, "decoder.conv_out", &mut self.decoder.conv_out)?;

        // --- Encoder ---
        load_conv2d(comp, "encoder.conv_in", &mut self.encoder.conv_in)?;
        for (i, db) in self.encoder.down_blocks.iter_mut().enumerate() {
            for (j, r) in db.resnets.iter_mut().enumerate() {
                r.load_from(comp, &format!("encoder.down_blocks.{i}.resnets.{j}"))?;
            }
            if let Some(d) = &mut db.downsample {
                load_conv2d(comp, &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                    &mut d.conv)?;
            }
        }
        self.encoder.mid_resnet_1.load_from(comp, "encoder.mid_block.resnets.0")?;
        load_attention_block(comp, "encoder.mid_block.attentions.0", &mut self.encoder.mid_attn)?;
        self.encoder.mid_resnet_2.load_from(comp, "encoder.mid_block.resnets.1")?;
        load_group_norm(comp, "encoder.conv_norm_out", &mut self.encoder.conv_norm_out)?;
        load_conv2d(comp, "encoder.conv_out", &mut self.encoder.conv_out)?;
        // quant_conv (encoder side): named just `quant_conv` at the top level.
        load_conv2d(comp, "quant_conv", &mut self.encoder.quant_conv)?;
        // post_quant_conv: applied to the latent before decode. 1×1 conv.
        load_conv2d(comp, "post_quant_conv", &mut self.post_quant_conv)?;
        Ok(())
    }

    /// Decode latent [B, latent_channels, H, W] → image [B, 3, 8H, 8W] in [-1, 1].
    /// SD's VAE pre-divides the latent by `scaling_factor` before decode.
    pub fn decode(&self, latent: &[f32]) -> Result<Vec<f32>> {
        // We accept already-scaled latents; caller is responsible for scaling.
        // Infer batch / spatial from total length.
        let lc = self.config.latent_channels;
        let total = latent.len();
        // Assume square latent and batch size 1 unless we have info — here we
        // enforce caller passes through `decode_with_dims`.
        if !total.is_multiple_of(lc) {
            return Err(crate::error::DiffusionError::ShapeMismatch {
                expected: format!("multiple of latent_channels {lc}"),
                got: format!("{total}"),
            });
        }
        let spatial = total / lc;
        let side = (spatial as f32).sqrt() as usize;
        if side * side != spatial {
            return Err(crate::error::DiffusionError::ShapeMismatch {
                expected: "square latent or use decode_with_dims".into(),
                got: format!("spatial = {spatial}, not square"),
            });
        }
        Ok(self.decode_with_dims(latent, 1, side, side))
    }

    /// Decode with explicit [N, latent_C, H, W] dimensions (preferred).
    pub fn decode_with_dims(&self, latent: &[f32], n: usize, h: usize, w: usize) -> Vec<f32> {
        let mut z = Vec::new();
        self.post_quant_conv.forward(latent, n, h, w, &mut z);
        self.decoder.forward(&z, n, h, w)
    }

    /// GPU-resident decode. Applies `post_quant_conv` then `decoder.forward_gpu`,
    /// keeping intermediate activations on GPU (fp16) — only the final image
    /// downloads to CPU. ~10× faster than the CPU decode for SD 1.5 (which
    /// runs 12 ResnetBlocks at 64→128→256→512 spatial through Accelerate
    /// sgemm conv2d sequentially).
    #[cfg(feature = "metal")]
    pub fn decode_with_dims_gpu(&self, latent: &[f32], n: usize, h: usize, w: usize) -> Vec<f32> {
        use crate::metal_backend::*;
        // post_quant_conv on GPU. Upload latent, run 1×1 conv, then hand the
        // GPU buffer straight to the decoder.
        let lc = self.config.latent_channels;
        let x_in = GpuTensor::upload_f32_as_f16(vec![n, lc, h, w], latent);
        let (z_gpu, _, _) = self.post_quant_conv.forward_gpu(&x_in, n, h, w);
        drop(x_in);
        // Download z to feed decoder.forward_gpu (which re-uploads inside).
        // Cheap for an 8KB latent. A future optim could plumb GpuTensor through.
        let z_cpu = z_gpu.download_to_f32();
        drop(z_gpu);
        self.decoder.forward_gpu(&z_cpu, n, h, w)
    }

    /// Encode image [N, 3, H, W] → latent mean (deterministic; we ignore
    /// logvar and use the mode of the distribution). Output [N, latent_C, H/8, W/8].
    pub fn encode(&self, image: &[f32], n: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        let (mean, _logvar) = self.encoder.forward(image, n, h, w);
        Ok(mean)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VAEConfig;

    fn tiny_vae_config() -> VAEConfig {
        VAEConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            act_fn: "silu".into(),
            norm_num_groups: 4,
            sample_size: 8,
            scaling_factor: 0.18215,
        }
    }

    #[test]
    fn vae_decode_correct_shape() {
        let vae = AutoencoderKL::from_config(tiny_vae_config());
        // n_blocks=2 puts an upsample on n_blocks-1 = block 0 only:
        // latent [1, 4, 2, 2] → image [1, 3, 4, 4].
        let latent = vec![0.0f32; 1 * 4 * 2 * 2];
        let img = vae.decode_with_dims(&latent, 1, 2, 2);
        // n_blocks=2 → 1 upsample → 2x → 4x4 output
        assert_eq!(img.len(), 1 * 3 * 4 * 4);
    }
}
