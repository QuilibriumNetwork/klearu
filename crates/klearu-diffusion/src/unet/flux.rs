//! Flux MMDiT (Multi-Modal Diffusion Transformer) backbone.
//!
//! Despite living under `unet/`, this is *not* a U-Net — it is a transformer
//! that processes patches of the latent jointly with text tokens. The module
//! placement matches the existing pipeline plumbing (the SD pipeline calls
//! `unet.forward(...)`); the `unet/` namespace just collects "the backbone
//! that turns conditioning + noisy latent into a velocity prediction".
//!
//! Architecture (BFL flux1-dev / flux1-schnell):
//!
//!   img patchify  ┐                                                      ┐
//!                 ├──── img_in ───→ [N, L_img, H]  ─┐                    │
//!   T5 tokens ────┴──── txt_in ───→ [N, L_txt, H]  ─┤  19 double_blocks  │
//!                                                  │  (img + txt streams) │
//!                       time_in                    │                    │
//!   t, y, g  ─────────→ vector_in ─→ [N, H]     ─→ │       38 single    │
//!                       guidance_in (dev only)     │       blocks       │
//!                                                  │  (concatenated)    │
//!                                                  └──── final_layer ──→ unpatchify
//!                                                       [N, L_img, p²·C_out]
//!
//! - Hidden size H = 3072, num_heads = 24, head_dim = 128.
//! - patch_size = 2 over the 16-channel latent ⇒ each patch is 16·2² = 64
//!   features. img_in projects 64 → 3072.
//! - LayerNorms are no-affine; modulation parameters come from the AdaLN
//!   path (`norm1.linear`, `norm1_context.linear`, etc. in diffusers terms;
//!   `img_mod.lin`, `txt_mod.lin`, `modulation.lin` in BFL terms).
//! - 2D RoPE applied to image tokens only (txt tokens get no positional
//!   encoding — text order matters less than image patch geometry).
//! - QK-RMSNorm on every attention block (separate scales for Q and K).
//!
//! Conditioning vector formation:
//!   vec = time_embedder(silu(timestep_sinusoid))
//!       + vector_embedder(silu(pooled_clip_l))
//!       + (dev only) guidance_embedder(silu(guidance_sinusoid))
//! Each block's modulation projector reads `silu(vec)`.
//!
//! Forward returns *velocity* prediction (the model is trained on
//! flow-matching, so output is dx/dσ at the input σ). The flow_match
//! scheduler consumes this directly.

use crate::error::Result;
use crate::weight::ComponentTensors;
use std::f32::consts::PI;

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone)]
pub struct FluxConfig {
    pub in_channels: usize,        // 16 (latent channels)
    pub out_channels: usize,       // 16
    pub hidden_size: usize,        // 3072
    pub num_heads: usize,          // 24
    pub head_dim: usize,           // 128 (= hidden_size / num_heads)
    pub mlp_ratio: usize,          // 4 (so MLP intermediate = 4*hidden_size)
    pub num_double_blocks: usize,  // 19
    pub num_single_blocks: usize,  // 38
    pub patch_size: usize,         // 2
    pub time_emb_dim: usize,       // 256 (sinusoidal frequency dim before MLP)
    pub pooled_clip_dim: usize,    // 768
    pub t5_dim: usize,             // 4096
    pub guidance_embeds: bool,     // true for flux1-dev, false for schnell
    /// Theta for the 2D RoPE. BFL uses 10000 broken across the (txt, h, w)
    /// axes per a custom `axes_dim` partition.
    pub rope_theta: f32,
    /// RoPE axes partition: (axes_for_first, axes_for_h, axes_for_w). BFL
    /// uses (16, 56, 56) summing to 128 = head_dim. The "first" axis is
    /// for txt-token IDs (which we always set to 0 in the input) so img
    /// patches end up freely encoded by the (h, w) halves.
    pub axes_dim: (usize, usize, usize),
}

impl FluxConfig {
    pub fn flux_dev() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            hidden_size: 3072,
            num_heads: 24,
            head_dim: 128,
            mlp_ratio: 4,
            num_double_blocks: 19,
            num_single_blocks: 38,
            patch_size: 2,
            time_emb_dim: 256,
            pooled_clip_dim: 768,
            t5_dim: 4096,
            guidance_embeds: true,
            rope_theta: 10_000.0,
            axes_dim: (16, 56, 56),
        }
    }
    pub fn flux_schnell() -> Self {
        let mut c = Self::flux_dev();
        c.guidance_embeds = false;
        c
    }
}

// ============================================================================
// Primitives
// ============================================================================

pub struct Linear {
    pub weight: Vec<f32>, // [out, in]
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(inf: usize, outf: usize, bias: bool) -> Self {
        Self {
            weight: vec![0.0; inf * outf],
            bias: if bias { Some(vec![0.0; outf]) } else { None },
            in_features: inf,
            out_features: outf,
        }
    }

    pub fn forward(&self, x: &[f32], n_tokens: usize, out: &mut [f32]) {
        let inf = self.in_features;
        let outf = self.out_features;
        debug_assert_eq!(x.len(), n_tokens * inf);
        debug_assert_eq!(out.len(), n_tokens * outf);
        // out[n × outf] = x[n × inf] · weight^T[inf × outf]
        // weight is [outf, inf] in row-major, exactly what sgemm_a_btrans expects.
        crate::blas::sgemm_a_btrans(
            n_tokens, outf, inf,
            x, &self.weight, out,
        );
        if let Some(b) = &self.bias {
            use rayon::prelude::*;
            out.par_chunks_mut(outf).for_each(|row| {
                for (r, bv) in row.iter_mut().zip(b.iter()) { *r += bv; }
            });
        }
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        self.weight = comp.get_f32(&format!("{prefix}.weight"))?;
        if self.bias.is_some() {
            self.bias = Some(comp.get_f32(&format!("{prefix}.bias"))?);
        }
        Ok(())
    }
}

/// Load an RMSNorm scale tensor, accepting either the BFL-native `.scale`
/// suffix or the diffusers/HF `.weight` suffix at `<base>`. Some Flux
/// re-distributions (HF / Comfy translations) flip the convention; this
/// keeps the loader insensitive to which one shipped.
fn load_norm_scale(comp: &ComponentTensors, base: &str) -> Result<Vec<f32>> {
    if let Ok(v) = comp.get_f32(&format!("{base}.scale")) {
        return Ok(v);
    }
    comp.get_f32(&format!("{base}.weight"))
}

/// RMSNorm with a learnable scale (`scale` field) and no bias. Used for
/// QK-norm on every Flux attention block.
pub struct RmsNorm {
    pub scale: Vec<f32>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { scale: vec![1.0; dim], eps }
    }

    pub fn apply_per_head(&self, x: &mut [f32], n_tokens: usize, num_heads: usize) {
        // x: [n_tokens, num_heads, head_dim]; norm over head_dim with shared
        // scale [head_dim]. (Flux's QK-norm operates per-head.)
        let dh = self.scale.len();
        debug_assert_eq!(x.len(), n_tokens * num_heads * dh);
        for t in 0..n_tokens {
            for h in 0..num_heads {
                let off = (t * num_heads + h) * dh;
                let row = &mut x[off..off + dh];
                let mut s = 0.0_f64;
                for &v in row.iter() { s += (v as f64) * (v as f64); }
                let inv = ((s / dh as f64) + self.eps as f64).sqrt().recip() as f32;
                for (xi, &g) in row.iter_mut().zip(self.scale.iter()) {
                    *xi = *xi * inv * g;
                }
            }
        }
    }
}

/// Plain LayerNorm (no affine — Flux uses `affine=False` and adds AdaLN
/// modulation externally).
fn layer_norm_no_affine(x: &mut [f32], n_tokens: usize, d: usize, eps: f32) {
    for t in 0..n_tokens {
        let row = &mut x[t * d..(t + 1) * d];
        // Mean.
        let mut mean = 0.0_f64;
        for &v in row.iter() { mean += v as f64; }
        mean /= d as f64;
        // Var.
        let mut var = 0.0_f64;
        for &v in row.iter() {
            let d_v = v as f64 - mean;
            var += d_v * d_v;
        }
        var /= d as f64;
        let inv = (var + eps as f64).sqrt().recip() as f32;
        let mean_f32 = mean as f32;
        for v in row.iter_mut() {
            *v = (*v - mean_f32) * inv;
        }
    }
}

/// Apply (1 + scale) · x + shift, in place, broadcasting scale/shift
/// across the sequence dimension. `scale`, `shift`: [hidden_size].
fn modulate(x: &mut [f32], n_tokens: usize, scale: &[f32], shift: &[f32]) {
    let d = scale.len();
    debug_assert_eq!(shift.len(), d);
    debug_assert_eq!(x.len(), n_tokens * d);
    for t in 0..n_tokens {
        let row = &mut x[t * d..(t + 1) * d];
        for ((v, &s), &b) in row.iter_mut().zip(scale.iter()).zip(shift.iter()) {
            *v = *v * (1.0 + s) + b;
        }
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() { *v = silu(*v); }
}

fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
}

fn erf_approx(x: f32) -> f32 {
    let a1 = 0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 = 1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 = 1.061405429_f32;
    let p = 0.3275911_f32;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    sign * y
}

fn add_inplace(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (a, b) in dst.iter_mut().zip(src.iter()) { *a += b; }
}

/// out = x * scalar_per_feature, broadcasting scalar across n_tokens.
/// Used to apply `gate` after a residual block computes its update.
fn scale_per_token(x: &mut [f32], n_tokens: usize, gate: &[f32]) {
    let d = gate.len();
    for t in 0..n_tokens {
        let row = &mut x[t * d..(t + 1) * d];
        for (v, &g) in row.iter_mut().zip(gate.iter()) { *v *= g; }
    }
}

// ============================================================================
// Time / vector embedders
// ============================================================================

/// Sinusoidal time embedding (DiT/Flux variant). Produces a `dim`-dim
/// vector for one scalar input. Same formula as DDPM, with `max_period`
/// defaulting to 10000.
pub fn timestep_embedding(t: f32, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0_f32; dim];
    // freqs[i] = exp(-ln(max_period) * i / half)  for i in 0..half
    let log_mp = max_period.ln();
    for i in 0..half {
        let freq = (-log_mp * (i as f32) / (half as f32)).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    if dim % 2 == 1 {
        // odd dim — pad with a final zero. Flux uses even dims so this never fires.
        out.push(0.0);
    }
    out
}

/// MLP embedder used for time / pooled-vector / guidance inputs.
/// `silu(in_layer(x)) → out_layer`.
pub struct MLPEmbedder {
    pub in_layer: Linear,
    pub out_layer: Linear,
}

impl MLPEmbedder {
    pub fn new(in_dim: usize, hidden: usize) -> Self {
        Self {
            in_layer: Linear::new(in_dim, hidden, true),
            out_layer: Linear::new(hidden, hidden, true),
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0_f32; self.in_layer.out_features];
        self.in_layer.forward(x, 1, &mut h);
        silu_inplace(&mut h);
        let mut out = vec![0.0_f32; self.out_layer.out_features];
        self.out_layer.forward(&h, 1, &mut out);
        out
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        self.in_layer.load(comp, &format!("{prefix}.in_layer"))?;
        self.out_layer.load(comp, &format!("{prefix}.out_layer"))?;
        Ok(())
    }
}

// ============================================================================
// 2D RoPE for image tokens
// ============================================================================

/// Build the RoPE frequency table for the (txt, h, w) axis partition.
/// Returns a per-token [num_heads_unused, head_dim/2] cosine/sine pair.
///
/// Output shape: `(cos, sin)` each of length `num_tokens * head_dim`,
/// laid out as [num_tokens, head_dim] interleaved (cos and sin pairs match
/// alternating positions in the rotated representation).
///
/// `seq_ids[i]` = (txt_id, h_id, w_id) for token i. txt tokens have h=0,
/// w=0; img tokens have txt_id=0 and (h_id, w_id) running over the patch
/// grid.
pub fn build_rope_2d(
    seq_ids: &[(u32, u32, u32)],
    head_dim: usize,
    axes_dim: (usize, usize, usize),
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = seq_ids.len();
    // Default rotation is identity (cos=1, sin=0). Slots not covered by
    // any axis (e.g., when an axis_dim is odd, the trailing slot is left
    // unrotated) keep the identity.
    let mut cos = vec![1.0_f32; n * head_dim];
    let mut sin = vec![0.0_f32; n * head_dim];
    let (a0, a1, a2) = axes_dim;
    debug_assert_eq!(a0 + a1 + a2, head_dim);
    // Each axis contributes `axis_size / 2` complex frequencies.
    // Frequency formula: freqs[k] = theta^(-2k/axis_size) for k in 0..axis_size/2.
    // Then for position p, angle = p * freqs[k].
    let axis_freqs = |axis_size: usize| -> Vec<f32> {
        let half = axis_size / 2;
        (0..half)
            .map(|k| theta.powf(-((2 * k) as f32) / axis_size as f32))
            .collect()
    };
    let f0 = axis_freqs(a0);
    let f1 = axis_freqs(a1);
    let f2 = axis_freqs(a2);

    for (i, &(p0, p1, p2)) in seq_ids.iter().enumerate() {
        let off = i * head_dim;
        // Axis 0: occupies head_dim positions [0..a0]. Pairs (k, k+a0/2).
        for k in 0..(a0 / 2) {
            let angle = p0 as f32 * f0[k];
            cos[off + 2 * k] = angle.cos();
            cos[off + 2 * k + 1] = angle.cos();
            sin[off + 2 * k] = angle.sin();
            sin[off + 2 * k + 1] = angle.sin();
        }
        // Axis 1: positions [a0..a0+a1].
        for k in 0..(a1 / 2) {
            let angle = p1 as f32 * f1[k];
            cos[off + a0 + 2 * k] = angle.cos();
            cos[off + a0 + 2 * k + 1] = angle.cos();
            sin[off + a0 + 2 * k] = angle.sin();
            sin[off + a0 + 2 * k + 1] = angle.sin();
        }
        // Axis 2: positions [a0+a1..head_dim].
        for k in 0..(a2 / 2) {
            let angle = p2 as f32 * f2[k];
            cos[off + a0 + a1 + 2 * k] = angle.cos();
            cos[off + a0 + a1 + 2 * k + 1] = angle.cos();
            sin[off + a0 + a1 + 2 * k] = angle.sin();
            sin[off + a0 + a1 + 2 * k + 1] = angle.sin();
        }
    }
    (cos, sin)
}

/// Apply RoPE to a [n_tokens, num_heads, head_dim] tensor in place.
/// `cos`, `sin` are [n_tokens, head_dim].
pub fn apply_rope(
    x: &mut [f32],
    n_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    debug_assert_eq!(x.len(), n_tokens * num_heads * head_dim);
    debug_assert_eq!(cos.len(), n_tokens * head_dim);
    for t in 0..n_tokens {
        let cos_row = &cos[t * head_dim..(t + 1) * head_dim];
        let sin_row = &sin[t * head_dim..(t + 1) * head_dim];
        for h in 0..num_heads {
            let off = (t * num_heads + h) * head_dim;
            let row = &mut x[off..off + head_dim];
            // Pairs (2k, 2k+1) — rotate (a, b) → (a·cos - b·sin, a·sin + b·cos).
            for k in 0..(head_dim / 2) {
                let a = row[2 * k];
                let b = row[2 * k + 1];
                let c = cos_row[2 * k];
                let s = sin_row[2 * k];
                row[2 * k] = a * c - b * s;
                row[2 * k + 1] = a * s + b * c;
            }
        }
    }
}

// ============================================================================
// Modulation (AdaLN)
// ============================================================================

/// Output of a modulation projector — six broadcast vectors per double
/// block (for the {attn, mlp} × {scale, shift, gate} combinations).
pub struct ModulationOut {
    pub shift_msa: Vec<f32>, // [hidden_size]
    pub scale_msa: Vec<f32>,
    pub gate_msa: Vec<f32>,
    pub shift_mlp: Vec<f32>,
    pub scale_mlp: Vec<f32>,
    pub gate_mlp: Vec<f32>,
}

/// `Linear(hidden, n_chunks * hidden)` projector that splits its output
/// into N broadcast vectors of dimension `hidden`.
pub struct Modulation {
    pub linear: Linear,
    pub n_chunks: usize,
    pub hidden: usize,
}

impl Modulation {
    pub fn new(hidden: usize, n_chunks: usize) -> Self {
        Self {
            linear: Linear::new(hidden, n_chunks * hidden, true),
            n_chunks,
            hidden,
        }
    }

    /// Project conditioning vec (post-silu) into n_chunks · hidden chunks.
    /// Returns the chunked vectors as a Vec<Vec<f32>>.
    pub fn forward(&self, vec_silu: &[f32]) -> Vec<Vec<f32>> {
        let mut out = vec![0.0_f32; self.n_chunks * self.hidden];
        self.linear.forward(vec_silu, 1, &mut out);
        out.chunks(self.hidden)
            .map(|c| c.to_vec())
            .collect()
    }
}

// ============================================================================
// Double block (separate img + txt streams with joint attention)
// ============================================================================

pub struct DoubleBlock {
    // img stream
    pub img_mod: Modulation,                 // 6 chunks
    pub img_attn_qkv: Linear,                // hidden → 3*hidden
    pub img_attn_norm_q: RmsNorm,            // [head_dim]
    pub img_attn_norm_k: RmsNorm,
    pub img_attn_proj: Linear,               // hidden → hidden
    pub img_mlp_0: Linear,                   // hidden → 4*hidden
    pub img_mlp_2: Linear,                   // 4*hidden → hidden
    // txt stream
    pub txt_mod: Modulation,
    pub txt_attn_qkv: Linear,
    pub txt_attn_norm_q: RmsNorm,
    pub txt_attn_norm_k: RmsNorm,
    pub txt_attn_proj: Linear,
    pub txt_mlp_0: Linear,
    pub txt_mlp_2: Linear,
    pub hidden: usize,
    pub mlp_h: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl DoubleBlock {
    pub fn new(cfg: &FluxConfig) -> Self {
        let h = cfg.hidden_size;
        let mlp_hidden = h * cfg.mlp_ratio;
        Self {
            img_mod: Modulation::new(h, 6),
            img_attn_qkv: Linear::new(h, 3 * h, true),
            img_attn_norm_q: RmsNorm::new(cfg.head_dim, 1e-6),
            img_attn_norm_k: RmsNorm::new(cfg.head_dim, 1e-6),
            img_attn_proj: Linear::new(h, h, true),
            img_mlp_0: Linear::new(h, mlp_hidden, true),
            img_mlp_2: Linear::new(mlp_hidden, h, true),
            txt_mod: Modulation::new(h, 6),
            txt_attn_qkv: Linear::new(h, 3 * h, true),
            txt_attn_norm_q: RmsNorm::new(cfg.head_dim, 1e-6),
            txt_attn_norm_k: RmsNorm::new(cfg.head_dim, 1e-6),
            txt_attn_proj: Linear::new(h, h, true),
            txt_mlp_0: Linear::new(h, mlp_hidden, true),
            txt_mlp_2: Linear::new(mlp_hidden, h, true),
            hidden: h,
            mlp_h: mlp_hidden,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        }
    }

    pub fn load(&mut self, comp: &ComponentTensors, idx: usize) -> Result<()> {
        let p = format!("double_blocks.{idx}");
        // img
        self.img_mod.linear.load(comp, &format!("{p}.img_mod.lin"))?;
        self.img_attn_qkv.load(comp, &format!("{p}.img_attn.qkv"))?;
        self.img_attn_norm_q.scale = load_norm_scale(comp, &format!("{p}.img_attn.norm.query_norm"))?;
        self.img_attn_norm_k.scale = load_norm_scale(comp, &format!("{p}.img_attn.norm.key_norm"))?;
        self.img_attn_proj.load(comp, &format!("{p}.img_attn.proj"))?;
        self.img_mlp_0.load(comp, &format!("{p}.img_mlp.0"))?;
        self.img_mlp_2.load(comp, &format!("{p}.img_mlp.2"))?;
        // txt
        self.txt_mod.linear.load(comp, &format!("{p}.txt_mod.lin"))?;
        self.txt_attn_qkv.load(comp, &format!("{p}.txt_attn.qkv"))?;
        self.txt_attn_norm_q.scale = load_norm_scale(comp, &format!("{p}.txt_attn.norm.query_norm"))?;
        self.txt_attn_norm_k.scale = load_norm_scale(comp, &format!("{p}.txt_attn.norm.key_norm"))?;
        self.txt_attn_proj.load(comp, &format!("{p}.txt_attn.proj"))?;
        self.txt_mlp_0.load(comp, &format!("{p}.txt_mlp.0"))?;
        self.txt_mlp_2.load(comp, &format!("{p}.txt_mlp.2"))?;
        Ok(())
    }

    /// Forward update for one double block. `img` and `txt` are residual
    /// streams; both are mutated in place. `cos_img` / `sin_img` are RoPE
    /// rows for the image tokens only.
    pub fn forward(
        &self,
        img: &mut [f32], l_img: usize,
        txt: &mut [f32], l_txt: usize,
        vec_silu: &[f32],
        cos: &[f32], sin: &[f32],   // [(l_txt + l_img), head_dim]
    ) {
        let h = self.hidden;
        let nh = self.num_heads;
        let dh = self.head_dim;

        // ===== Modulation =====
        let img_mod = self.img_mod.forward(vec_silu);
        let (i_shift_msa, i_scale_msa, i_gate_msa, i_shift_mlp, i_scale_mlp, i_gate_mlp) =
            (&img_mod[0], &img_mod[1], &img_mod[2], &img_mod[3], &img_mod[4], &img_mod[5]);
        let txt_mod = self.txt_mod.forward(vec_silu);
        let (t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp) =
            (&txt_mod[0], &txt_mod[1], &txt_mod[2], &txt_mod[3], &txt_mod[4], &txt_mod[5]);

        // ===== Attention sub-block =====
        // Pre-attention: LN(no-affine) → modulate.
        let mut img_norm = img.to_vec();
        layer_norm_no_affine(&mut img_norm, l_img, h, 1e-6);
        modulate(&mut img_norm, l_img, i_scale_msa, i_shift_msa);

        let mut txt_norm = txt.to_vec();
        layer_norm_no_affine(&mut txt_norm, l_txt, h, 1e-6);
        modulate(&mut txt_norm, l_txt, t_scale_msa, t_shift_msa);

        // QKV projections.
        let mut img_qkv = vec![0.0_f32; l_img * 3 * h];
        let mut txt_qkv = vec![0.0_f32; l_txt * 3 * h];
        self.img_attn_qkv.forward(&img_norm, l_img, &mut img_qkv);
        self.txt_attn_qkv.forward(&txt_norm, l_txt, &mut txt_qkv);

        // Split q,k,v out of [n, 3*h]. Layout: [n, 3, num_heads, head_dim]
        // — interleaved as Q, K, V along axis 1.
        let split = |qkv: &[f32], n: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            // qkv[t, c, hi, di] where c ∈ {0,1,2}.
            let mut q = vec![0.0_f32; n * h];
            let mut k = vec![0.0_f32; n * h];
            let mut v = vec![0.0_f32; n * h];
            for t in 0..n {
                for hi in 0..nh {
                    for di in 0..dh {
                        let qkv_off = t * 3 * h + 0 * h + hi * dh + di;
                        let off = (t * nh + hi) * dh + di;
                        q[off] = qkv[qkv_off];
                        k[off] = qkv[t * 3 * h + 1 * h + hi * dh + di];
                        v[off] = qkv[t * 3 * h + 2 * h + hi * dh + di];
                    }
                }
            }
            (q, k, v)
        };
        let (mut img_q, mut img_k, img_v) = split(&img_qkv, l_img);
        let (mut txt_q, mut txt_k, txt_v) = split(&txt_qkv, l_txt);
        drop(img_qkv); drop(txt_qkv);

        // QK-norm (per head).
        self.img_attn_norm_q.apply_per_head(&mut img_q, l_img, nh);
        self.img_attn_norm_k.apply_per_head(&mut img_k, l_img, nh);
        self.txt_attn_norm_q.apply_per_head(&mut txt_q, l_txt, nh);
        self.txt_attn_norm_k.apply_per_head(&mut txt_k, l_txt, nh);

        // Concatenate [txt, img] for joint attention. RoPE applies to the
        // entire concatenated sequence — txt token positions are (0,0,0)
        // so they get identity rotation; img positions encode the patch
        // grid.
        let l_joint = l_txt + l_img;
        let mut q = vec![0.0_f32; l_joint * h];
        let mut k = vec![0.0_f32; l_joint * h];
        let mut v = vec![0.0_f32; l_joint * h];
        q[..l_txt * h].copy_from_slice(&txt_q);
        q[l_txt * h..].copy_from_slice(&img_q);
        k[..l_txt * h].copy_from_slice(&txt_k);
        k[l_txt * h..].copy_from_slice(&img_k);
        v[..l_txt * h].copy_from_slice(&txt_v);
        v[l_txt * h..].copy_from_slice(&img_v);

        // Apply RoPE to Q and K.
        apply_rope(&mut q, l_joint, nh, dh, cos, sin);
        apply_rope(&mut k, l_joint, nh, dh, cos, sin);

        // Joint attention. No mask (all tokens attend to all).
        let attn_out = scaled_dot_product_attention(&q, &k, &v, l_joint, nh, dh);
        drop(q); drop(k); drop(v);

        // Split attention output back into (txt, img) along axis 0.
        let txt_attn_out = attn_out[..l_txt * h].to_vec();
        let img_attn_out = attn_out[l_txt * h..].to_vec();

        // proj + gate + residual.
        let mut img_proj = vec![0.0_f32; l_img * h];
        let mut txt_proj = vec![0.0_f32; l_txt * h];
        self.img_attn_proj.forward(&img_attn_out, l_img, &mut img_proj);
        self.txt_attn_proj.forward(&txt_attn_out, l_txt, &mut txt_proj);
        scale_per_token(&mut img_proj, l_img, i_gate_msa);
        scale_per_token(&mut txt_proj, l_txt, t_gate_msa);
        add_inplace(img, &img_proj);
        add_inplace(txt, &txt_proj);

        // ===== MLP sub-block =====
        // Pre-MLP: LN(no-affine) → modulate.
        let mut img_norm2 = img.to_vec();
        layer_norm_no_affine(&mut img_norm2, l_img, h, 1e-6);
        modulate(&mut img_norm2, l_img, i_scale_mlp, i_shift_mlp);
        let mut txt_norm2 = txt.to_vec();
        layer_norm_no_affine(&mut txt_norm2, l_txt, h, 1e-6);
        modulate(&mut txt_norm2, l_txt, t_scale_mlp, t_shift_mlp);

        // GELU MLP.
        let mlp_h = self.mlp_h;
        let mut img_mlp_h = vec![0.0_f32; l_img * mlp_h];
        let mut txt_mlp_h = vec![0.0_f32; l_txt * mlp_h];
        self.img_mlp_0.forward(&img_norm2, l_img, &mut img_mlp_h);
        self.txt_mlp_0.forward(&txt_norm2, l_txt, &mut txt_mlp_h);
        for v in img_mlp_h.iter_mut() { *v = gelu_exact(*v); }
        for v in txt_mlp_h.iter_mut() { *v = gelu_exact(*v); }
        let mut img_mlp_out = vec![0.0_f32; l_img * h];
        let mut txt_mlp_out = vec![0.0_f32; l_txt * h];
        self.img_mlp_2.forward(&img_mlp_h, l_img, &mut img_mlp_out);
        self.txt_mlp_2.forward(&txt_mlp_h, l_txt, &mut txt_mlp_out);
        scale_per_token(&mut img_mlp_out, l_img, i_gate_mlp);
        scale_per_token(&mut txt_mlp_out, l_txt, t_gate_mlp);
        add_inplace(img, &img_mlp_out);
        add_inplace(txt, &txt_mlp_out);
    }
}

// ============================================================================
// Attention primitive
// ============================================================================

/// Standard scaled-dot-product attention without mask.
/// q, k, v: [n_tokens, num_heads, head_dim].
/// Output: [n_tokens, num_heads, head_dim] flattened.
fn scaled_dot_product_attention(
    q: &[f32], k: &[f32], v: &[f32],
    n_tokens: usize, num_heads: usize, head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; n_tokens * num_heads * head_dim];
    for h in 0..num_heads {
        for qi in 0..n_tokens {
            let q_off = (qi * num_heads + h) * head_dim;
            let mut scores = vec![0.0_f32; n_tokens];
            for ki in 0..n_tokens {
                let k_off = (ki * num_heads + h) * head_dim;
                let mut s = 0.0_f32;
                for di in 0..head_dim { s += q[q_off + di] * k[k_off + di]; }
                scores[ki] = s * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
            if sum > 0.0 { let inv = 1.0 / sum; for s in scores.iter_mut() { *s *= inv; } }
            let out_off = (qi * num_heads + h) * head_dim;
            for ki in 0..n_tokens {
                let v_off = (ki * num_heads + h) * head_dim;
                let w = scores[ki];
                if w == 0.0 { continue; }
                for di in 0..head_dim { out[out_off + di] += w * v[v_off + di]; }
            }
        }
    }
    out
}

// ============================================================================
// Single block (concatenated stream)
// ============================================================================

pub struct SingleBlock {
    pub modulation: Modulation,              // 3 chunks
    /// Linear(h, 3h + mlp_h) — fused [Q | K | V | MLP_in]. Output is split
    /// 4-way at use.
    pub linear1: Linear,
    pub norm_q: RmsNorm,
    pub norm_k: RmsNorm,
    /// Linear(h + mlp_h, h) — fused [out_proj_input | mlp_in_postgelu] →
    /// the residual update.
    pub linear2: Linear,
    pub hidden: usize,
    pub mlp_h: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl SingleBlock {
    pub fn new(cfg: &FluxConfig) -> Self {
        let h = cfg.hidden_size;
        let mlp_h = h * cfg.mlp_ratio;
        Self {
            modulation: Modulation::new(h, 3),
            linear1: Linear::new(h, 3 * h + mlp_h, true),
            norm_q: RmsNorm::new(cfg.head_dim, 1e-6),
            norm_k: RmsNorm::new(cfg.head_dim, 1e-6),
            linear2: Linear::new(h + mlp_h, h, true),
            hidden: h,
            mlp_h,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        }
    }

    pub fn load(&mut self, comp: &ComponentTensors, idx: usize) -> Result<()> {
        let p = format!("single_blocks.{idx}");
        self.modulation.linear.load(comp, &format!("{p}.modulation.lin"))?;
        self.linear1.load(comp, &format!("{p}.linear1"))?;
        self.linear2.load(comp, &format!("{p}.linear2"))?;
        self.norm_q.scale = load_norm_scale(comp, &format!("{p}.norm.query_norm"))?;
        self.norm_k.scale = load_norm_scale(comp, &format!("{p}.norm.key_norm"))?;
        Ok(())
    }

    /// Forward update for one single block over the concatenated [txt, img]
    /// sequence.
    pub fn forward(
        &self,
        x: &mut [f32], l_total: usize,
        vec_silu: &[f32],
        cos: &[f32], sin: &[f32],
    ) {
        let h = self.hidden;
        let mlp_h = self.mlp_h;
        let nh = self.num_heads;
        let dh = self.head_dim;

        let mod_out = self.modulation.forward(vec_silu);
        let (shift, scale, gate) = (&mod_out[0], &mod_out[1], &mod_out[2]);

        // Pre-block: LN(no-affine) → modulate.
        let mut x_norm = x.to_vec();
        layer_norm_no_affine(&mut x_norm, l_total, h, 1e-6);
        modulate(&mut x_norm, l_total, scale, shift);

        // linear1: produces [n, 3h + mlp_h]. Split: q (h), k (h), v (h),
        // mlp_in (mlp_h).
        let mut fused = vec![0.0_f32; l_total * (3 * h + mlp_h)];
        self.linear1.forward(&x_norm, l_total, &mut fused);
        let mut q = vec![0.0_f32; l_total * h];
        let mut k = vec![0.0_f32; l_total * h];
        let mut v = vec![0.0_f32; l_total * h];
        let mut mlp_in = vec![0.0_f32; l_total * mlp_h];
        for t in 0..l_total {
            let off = t * (3 * h + mlp_h);
            q[t * h..(t + 1) * h].copy_from_slice(&fused[off..off + h]);
            k[t * h..(t + 1) * h].copy_from_slice(&fused[off + h..off + 2 * h]);
            v[t * h..(t + 1) * h].copy_from_slice(&fused[off + 2 * h..off + 3 * h]);
            mlp_in[t * mlp_h..(t + 1) * mlp_h]
                .copy_from_slice(&fused[off + 3 * h..off + 3 * h + mlp_h]);
        }
        drop(fused);

        // QK-norm.
        self.norm_q.apply_per_head(&mut q, l_total, nh);
        self.norm_k.apply_per_head(&mut k, l_total, nh);

        // RoPE.
        apply_rope(&mut q, l_total, nh, dh, cos, sin);
        apply_rope(&mut k, l_total, nh, dh, cos, sin);

        // Attention.
        let attn_out = scaled_dot_product_attention(&q, &k, &v, l_total, nh, dh);
        drop(q); drop(k); drop(v);

        // GELU on MLP path.
        let mut mlp_act = mlp_in;
        for vv in mlp_act.iter_mut() { *vv = gelu_exact(*vv); }

        // linear2: input is [attn_out | mlp_act] concat along feature axis.
        let mut combined = vec![0.0_f32; l_total * (h + mlp_h)];
        for t in 0..l_total {
            let off = t * (h + mlp_h);
            combined[off..off + h].copy_from_slice(&attn_out[t * h..(t + 1) * h]);
            combined[off + h..off + h + mlp_h]
                .copy_from_slice(&mlp_act[t * mlp_h..(t + 1) * mlp_h]);
        }
        let mut update = vec![0.0_f32; l_total * h];
        self.linear2.forward(&combined, l_total, &mut update);
        scale_per_token(&mut update, l_total, gate);
        add_inplace(x, &update);
    }
}

// ============================================================================
// Final layer
// ============================================================================

pub struct FinalLayer {
    pub ada_ln_modulation: Linear, // 2 chunks: shift, scale
    pub linear: Linear,            // hidden → patch_size² * out_channels
    pub patch_size: usize,
    pub out_channels: usize,
}

impl FinalLayer {
    pub fn new(cfg: &FluxConfig) -> Self {
        let p2c = cfg.patch_size * cfg.patch_size * cfg.out_channels;
        Self {
            ada_ln_modulation: Linear::new(cfg.hidden_size, 2 * cfg.hidden_size, true),
            linear: Linear::new(cfg.hidden_size, p2c, true),
            patch_size: cfg.patch_size,
            out_channels: cfg.out_channels,
        }
    }

    pub fn load(&mut self, comp: &ComponentTensors) -> Result<()> {
        // BFL: `final_layer.adaLN_modulation.1.{weight,bias}` (the `.1`
        // is a `nn.Sequential` index — index 0 is `nn.SiLU`, index 1 is the
        // Linear). We bake the silu into the call site so we only load
        // the linear here.
        self.ada_ln_modulation.load(comp, "final_layer.adaLN_modulation.1")?;
        self.linear.load(comp, "final_layer.linear")?;
        Ok(())
    }

    /// Forward returns the unpatchified prediction for one image's worth of
    /// tokens. `img`: [l_img, hidden]. Output: [l_img, p² * out_channels]
    /// before unpatchify.
    pub fn forward(&self, img: &[f32], l_img: usize, vec_silu: &[f32]) -> Vec<f32> {
        let h = self.ada_ln_modulation.in_features;
        // Modulate: project conditioning → [shift, scale].
        let mut mod_out = vec![0.0_f32; 2 * h];
        self.ada_ln_modulation.forward(vec_silu, 1, &mut mod_out);
        let shift = &mod_out[..h];
        let scale = &mod_out[h..];

        let mut x = img.to_vec();
        layer_norm_no_affine(&mut x, l_img, h, 1e-6);
        modulate(&mut x, l_img, scale, shift);

        let p2c = self.linear.out_features;
        let mut out = vec![0.0_f32; l_img * p2c];
        self.linear.forward(&x, l_img, &mut out);
        out
    }
}

// ============================================================================
// Top-level transformer
// ============================================================================

pub struct FluxTransformer {
    pub cfg: FluxConfig,
    pub img_in: Linear,
    pub txt_in: Linear,
    pub time_in: MLPEmbedder,
    pub vector_in: MLPEmbedder,
    pub guidance_in: Option<MLPEmbedder>,
    pub double_blocks: Vec<DoubleBlock>,
    pub single_blocks: Vec<SingleBlock>,
    pub final_layer: FinalLayer,
}

impl FluxTransformer {
    pub fn from_config(cfg: FluxConfig) -> Self {
        let img_in_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size;
        let img_in = Linear::new(img_in_dim, cfg.hidden_size, true);
        let txt_in = Linear::new(cfg.t5_dim, cfg.hidden_size, true);
        let time_in = MLPEmbedder::new(cfg.time_emb_dim, cfg.hidden_size);
        let vector_in = MLPEmbedder::new(cfg.pooled_clip_dim, cfg.hidden_size);
        let guidance_in = if cfg.guidance_embeds {
            Some(MLPEmbedder::new(cfg.time_emb_dim, cfg.hidden_size))
        } else {
            None
        };
        let double_blocks: Vec<_> = (0..cfg.num_double_blocks).map(|_| DoubleBlock::new(&cfg)).collect();
        let single_blocks: Vec<_> = (0..cfg.num_single_blocks).map(|_| SingleBlock::new(&cfg)).collect();
        let final_layer = FinalLayer::new(&cfg);
        Self {
            cfg, img_in, txt_in, time_in, vector_in, guidance_in,
            double_blocks, single_blocks, final_layer,
        }
    }

    pub fn load_from(&mut self, comp: &ComponentTensors) -> Result<()> {
        let t0 = std::time::Instant::now();
        eprintln!("[flux]   prelude (img_in / txt_in / time_in / vector_in / guidance_in)…");
        self.img_in.load(comp, "img_in")?;
        self.txt_in.load(comp, "txt_in")?;
        self.time_in.load(comp, "time_in")?;
        self.vector_in.load(comp, "vector_in")?;
        if let Some(g) = self.guidance_in.as_mut() {
            g.load(comp, "guidance_in")?;
        }
        let n_double = self.double_blocks.len();
        for (i, blk) in self.double_blocks.iter_mut().enumerate() {
            blk.load(comp, i)?;
            eprintln!("[flux]   double block {}/{n_double}  ({:.1}s elapsed)",
                i + 1, t0.elapsed().as_secs_f32());
        }
        let n_single = self.single_blocks.len();
        for (i, blk) in self.single_blocks.iter_mut().enumerate() {
            blk.load(comp, i)?;
            // Single blocks are smaller; print every 4th to avoid spam.
            if (i + 1) % 4 == 0 || i + 1 == n_single {
                eprintln!("[flux]   single block {}/{n_single}  ({:.1}s elapsed)",
                    i + 1, t0.elapsed().as_secs_f32());
            }
        }
        eprintln!("[flux]   final layer…");
        self.final_layer.load(comp)?;
        eprintln!("[flux]   transformer loaded in {:.1}s", t0.elapsed().as_secs_f32());
        Ok(())
    }

    /// One forward pass.
    ///
    /// Inputs:
    ///   - `latent`: [in_channels, H, W] noisy latent (already at the
    ///     model's σ — caller does not pre-scale by anything).
    ///   - `t5_embed`: [L_txt, t5_dim] T5-XXL output (no padding stripped;
    ///     pad tokens carry zero attention contribution because we don't
    ///     mask in attention here — Flux relies on T5's last-token-zero
    ///     behavior at pad positions, plus learned attention-suppression).
    ///   - `pooled_clip`: [pooled_clip_dim] pooled CLIP-L output.
    ///   - `timestep`: σ × 1000 (Flux's model-input convention).
    ///   - `guidance`: scalar guidance value for flux1-dev (e.g., 3.5).
    ///     Pass 0.0 for schnell (ignored).
    ///
    /// Output: [out_channels, H, W] velocity prediction.
    pub fn forward(
        &self,
        latent: &[f32],
        latent_h: usize,
        latent_w: usize,
        t5_embed: &[f32],
        l_txt: usize,
        pooled_clip: &[f32],
        timestep: f32,
        guidance: f32,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let p = cfg.patch_size;
        let h_p = latent_h / p;
        let w_p = latent_w / p;
        let l_img = h_p * w_p;
        let in_c = cfg.in_channels;

        // ===== Patchify =====
        // [in_c, H, W] → [l_img, p²·in_c]. Each patch flattens p×p×in_c.
        let patch_dim = in_c * p * p;
        let mut img_tokens = vec![0.0_f32; l_img * patch_dim];
        for hi in 0..h_p {
            for wi in 0..w_p {
                let token_idx = hi * w_p + wi;
                let token_off = token_idx * patch_dim;
                let mut k = 0;
                for c in 0..in_c {
                    for dy in 0..p {
                        for dx in 0..p {
                            let lh = hi * p + dy;
                            let lw = wi * p + dx;
                            let lat_idx = c * latent_h * latent_w + lh * latent_w + lw;
                            img_tokens[token_off + k] = latent[lat_idx];
                            k += 1;
                        }
                    }
                }
            }
        }

        // img_in: project patches → [l_img, hidden]
        let mut img = vec![0.0_f32; l_img * h];
        self.img_in.forward(&img_tokens, l_img, &mut img);
        drop(img_tokens);

        // txt_in: project T5 → [l_txt, hidden]
        let mut txt = vec![0.0_f32; l_txt * h];
        self.txt_in.forward(t5_embed, l_txt, &mut txt);

        // ===== Conditioning vector =====
        // vec = silu(time_in(t)) + silu(vector_in(y)) (+ silu(guidance_in(g)))
        // [Note] silu happens INSIDE the MLPEmbedder (between in_layer and
        // out_layer), so the output is already post-silu-internal. The
        // ADDITIONAL silu in the formula is applied AFTER the embedders,
        // before being fed to per-block modulators. We bake that silu
        // into the modulation projector inputs below.
        let t_emb = timestep_embedding(timestep, cfg.time_emb_dim, 10_000.0);
        let mut vec_cond = self.time_in.forward(&t_emb);
        let pooled_emb = self.vector_in.forward(pooled_clip);
        for (a, b) in vec_cond.iter_mut().zip(pooled_emb.iter()) { *a += b; }
        if let Some(gi) = &self.guidance_in {
            let g_emb = timestep_embedding(guidance * 1000.0, cfg.time_emb_dim, 10_000.0);
            let g_cond = gi.forward(&g_emb);
            for (a, b) in vec_cond.iter_mut().zip(g_cond.iter()) { *a += b; }
        }
        // The per-block modulator inputs are silu(vec).
        let mut vec_silu = vec_cond.clone();
        silu_inplace(&mut vec_silu);

        // ===== RoPE position IDs =====
        // txt tokens: (0, 0, 0). img tokens: (0, hi, wi).
        let mut seq_ids: Vec<(u32, u32, u32)> = Vec::with_capacity(l_txt + l_img);
        for _ in 0..l_txt { seq_ids.push((0, 0, 0)); }
        for hi in 0..h_p {
            for wi in 0..w_p {
                seq_ids.push((0, hi as u32, wi as u32));
            }
        }
        let (cos, sin) = build_rope_2d(&seq_ids, cfg.head_dim, cfg.axes_dim, cfg.rope_theta);

        // ===== Double blocks =====
        let progress = std::env::var("KLEARU_QUIET").is_err();
        let fwd_t0 = std::time::Instant::now();
        let n_d = self.double_blocks.len();
        for (i, blk) in self.double_blocks.iter().enumerate() {
            blk.forward(&mut img, l_img, &mut txt, l_txt, &vec_silu, &cos, &sin);
            if progress {
                eprintln!("[flux]     transformer.forward: double {}/{n_d}  ({:.1}s)",
                    i + 1, fwd_t0.elapsed().as_secs_f32());
            }
        }

        // ===== Concatenate streams for single blocks =====
        let l_total = l_txt + l_img;
        let mut joint = vec![0.0_f32; l_total * h];
        joint[..l_txt * h].copy_from_slice(&txt);
        joint[l_txt * h..].copy_from_slice(&img);
        drop(img); drop(txt);

        let n_s = self.single_blocks.len();
        for (i, blk) in self.single_blocks.iter().enumerate() {
            blk.forward(&mut joint, l_total, &vec_silu, &cos, &sin);
            if progress && ((i + 1) % 4 == 0 || i + 1 == n_s) {
                eprintln!("[flux]     transformer.forward: single {}/{n_s}  ({:.1}s)",
                    i + 1, fwd_t0.elapsed().as_secs_f32());
            }
        }

        // ===== Final layer (img-only) =====
        // Drop txt prefix; project to [l_img, p²·out_c].
        let img_final = joint[l_txt * h..].to_vec();
        drop(joint);
        let img_out = self.final_layer.forward(&img_final, l_img, &vec_silu);

        // ===== Unpatchify =====
        // [l_img, p²·out_c] → [out_c, H, W]
        let out_c = cfg.out_channels;
        let mut out = vec![0.0_f32; out_c * latent_h * latent_w];
        for hi in 0..h_p {
            for wi in 0..w_p {
                let token_idx = hi * w_p + wi;
                let mut k = 0;
                for c in 0..out_c {
                    for dy in 0..p {
                        for dx in 0..p {
                            let lh = hi * p + dy;
                            let lw = wi * p + dx;
                            let lat_idx = c * latent_h * latent_w + lh * latent_w + lw;
                            out[lat_idx] = img_out[token_idx * (out_c * p * p) + k];
                            k += 1;
                        }
                    }
                }
            }
        }
        out
    }
}

// Suppress unused-import lint while these helpers wait for a faster
// matmul backend.
#[allow(dead_code)]
fn _pi() -> f32 { PI }

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> FluxConfig {
        FluxConfig {
            in_channels: 4,
            out_channels: 4,
            hidden_size: 32,
            num_heads: 4,
            head_dim: 8,
            mlp_ratio: 2,
            num_double_blocks: 1,
            num_single_blocks: 1,
            patch_size: 2,
            time_emb_dim: 8,
            pooled_clip_dim: 8,
            t5_dim: 16,
            guidance_embeds: false,
            rope_theta: 100.0,
            axes_dim: (2, 3, 3),
        }
    }

    #[test]
    fn rope_identity_at_origin() {
        let (cos, sin) = build_rope_2d(&[(0, 0, 0)], 8, (2, 3, 3), 100.0);
        // p=0 ⇒ all angles zero ⇒ cos = 1, sin = 0.
        for &c in &cos { assert!((c - 1.0).abs() < 1e-6); }
        for &s in &sin { assert!(s.abs() < 1e-6); }
    }

    #[test]
    fn timestep_embedding_finite() {
        let e = timestep_embedding(500.0, 16, 10_000.0);
        assert_eq!(e.len(), 16);
        assert!(e.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_runs_with_zero_weights() {
        let cfg = tiny_cfg();
        let mut model = FluxTransformer::from_config(cfg.clone());
        // Zero-init weights (default). Set RmsNorm scales to 1.0 to avoid
        // divide-by-zero ⇒ 0/0 paths; LayerNorm doesn't depend on weights.
        for blk in model.double_blocks.iter_mut() {
            for v in blk.img_attn_norm_q.scale.iter_mut() { *v = 1.0; }
            for v in blk.img_attn_norm_k.scale.iter_mut() { *v = 1.0; }
            for v in blk.txt_attn_norm_q.scale.iter_mut() { *v = 1.0; }
            for v in blk.txt_attn_norm_k.scale.iter_mut() { *v = 1.0; }
        }
        for blk in model.single_blocks.iter_mut() {
            for v in blk.norm_q.scale.iter_mut() { *v = 1.0; }
            for v in blk.norm_k.scale.iter_mut() { *v = 1.0; }
        }

        let h = 4; let w = 4;
        let latent = vec![0.0_f32; cfg.in_channels * h * w];
        let l_txt = 3;
        let t5 = vec![0.0_f32; l_txt * cfg.t5_dim];
        let pooled = vec![0.0_f32; cfg.pooled_clip_dim];
        let out = model.forward(&latent, h, w, &t5, l_txt, &pooled, 500.0, 0.0);
        assert_eq!(out.len(), cfg.out_channels * h * w);
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
