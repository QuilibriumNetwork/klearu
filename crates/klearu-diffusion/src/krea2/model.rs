//! CPU reference for the Krea 2 transformer (`Krea2Transformer2DModel`).
//!
//! This is the correctness oracle for [`super::gpu`], written to read like the
//! reference rather than to be fast: f32 throughout, the sequence laid out
//! text-then-image exactly as diffusers does, full attention with no padding.
//! Loading the real 12.9B model here would need ~52 GB of f32, so outside the
//! tiny-config parity tests it is only ever loaded piecemeal.
//!
//! The structure, for orientation:
//!
//! ```text
//! taps [L, 12, 2560] ─ text fusion (2 blocks across the 12 taps, Linear 12→1,
//!                      2 blocks across tokens) ─ txt_in ─┐
//! latent ─ patchify ─ img_in ───────────────────────────┴─ concat [txt, img]
//!   ─ 28 × single-stream block (shared time modulation + per-block table)
//!   ─ final layer on the image tokens ─ unpatchify
//! ```

use rayon::prelude::*;

use super::config::Krea2Config;
use crate::layers::Linear;
use crate::weight::loader::ComponentTensors;
use crate::{DiffusionError, Result};

/// Fetch a tensor and check its element count.
pub(crate) fn get_checked(comp: &ComponentTensors, name: &str, numel: usize) -> Result<Vec<f32>> {
    let v = comp.get_f32(name)?;
    if v.len() != numel {
        return Err(DiffusionError::ShapeMismatch {
            expected: format!("{numel} elements in {name}"),
            got: format!("{}", v.len()),
        });
    }
    Ok(v)
}

fn load_linear(
    comp: &ComponentTensors, p: &str, in_f: usize, out_f: usize, bias: bool,
) -> Result<Linear> {
    Ok(Linear {
        weight: get_checked(comp, &format!("{p}.weight"), in_f * out_f)?,
        bias: if bias { Some(get_checked(comp, &format!("{p}.bias"), out_f)?) } else { None },
        in_features: in_f,
        out_features: out_f,
    })
}

fn linear(l: &Linear, x: &[f32]) -> Vec<f32> {
    let n = x.len() / l.in_features;
    let mut y = vec![0.0_f32; n * l.out_features];
    l.forward_batch(x, &mut y);
    y
}

/// tanh-approximate GELU, scalar so tests do not depend on the Metal path.
fn gelu_tanh(x: &mut [f32]) {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    for v in x.iter_mut() {
        let u = *v;
        *v = 0.5 * u * (1.0 + (c * (u + 0.044715 * u * u * u)).tanh());
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// RMSNorm with a zero-centred scale: the checkpoint stores `w` and the
/// multiplier is `1 + w`. Held here with the 1 already added.
#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn load(comp: &ComponentTensors, name: &str, dim: usize, eps: f32) -> Result<Self> {
        let w = get_checked(comp, name, dim)?;
        Ok(Self { weight: w.iter().map(|v| 1.0 + v).collect(), eps })
    }

    /// Normalise each `weight.len()`-wide row of `x` in place.
    pub fn forward(&self, x: &mut [f32]) {
        let d = self.weight.len();
        x.par_chunks_mut(d).for_each(|row| {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + self.eps).sqrt();
            for (v, w) in row.iter_mut().zip(&self.weight) {
                *v *= inv * w;
            }
        });
    }
}

/// Per-token rotary tables, `[n, head_dim]`, each value repeated across the
/// pair it rotates (diffusers' `repeat_interleave_real=True`).
pub struct Rope {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub head_dim: usize,
}

impl Rope {
    /// Build from `(t, h, w)` positions. Each axis owns a contiguous slice of
    /// the head dimension (`axes`), rotating pairs at frequencies
    /// `theta^(-2j / axis_dim)`. Computed in f64 like the reference.
    pub fn new(positions: &[[f64; 3]], axes: [usize; 3], theta: f64) -> Self {
        let head_dim: usize = axes.iter().sum();
        let mut cos = Vec::with_capacity(positions.len() * head_dim);
        let mut sin = Vec::with_capacity(positions.len() * head_dim);
        for p in positions {
            for (axis, &dim) in axes.iter().enumerate() {
                for j in 0..dim / 2 {
                    let omega = 1.0 / theta.powf(2.0 * j as f64 / dim as f64);
                    let a = p[axis] * omega;
                    let (s, c) = a.sin_cos();
                    cos.extend_from_slice(&[c as f32, c as f32]);
                    sin.extend_from_slice(&[s as f32, s as f32]);
                }
            }
        }
        Self { cos, sin, head_dim }
    }

    /// Rotate `x` laid out `[n, heads, head_dim]` in place: for each pair,
    /// `(a, b) -> (a cos - b sin, a sin + b cos)`.
    pub fn apply(&self, x: &mut [f32], heads: usize) {
        let dh = self.head_dim;
        x.par_chunks_mut(heads * dh).enumerate().for_each(|(t, row)| {
            let c = &self.cos[t * dh..(t + 1) * dh];
            let s = &self.sin[t * dh..(t + 1) * dh];
            for head in row.chunks_mut(dh) {
                for k in (0..dh).step_by(2) {
                    let (a, b) = (head[k], head[k + 1]);
                    head[k] = a * c[k] - b * s[k];
                    head[k + 1] = a * s[k] + b * c[k];
                }
            }
        });
    }
}

/// Image-token positions on the latent grid: `(0, row, col)`, raster order.
pub fn image_positions(grid_h: usize, grid_w: usize) -> Vec<[f64; 3]> {
    (0..grid_h * grid_w)
        .map(|i| [0.0, (i / grid_w) as f64, (i % grid_w) as f64])
        .collect()
}

/// Gated grouped-query attention with per-head QK RMSNorm.
pub struct Attention {
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_gate: Linear,
    pub to_out: Linear,
    pub norm_q: RmsNorm,
    pub norm_k: RmsNorm,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    pub fn load(
        comp: &ComponentTensors, p: &str, dim: usize, heads: usize, kv_heads: usize, eps: f32,
    ) -> Result<Self> {
        let dh = dim / heads;
        Ok(Self {
            to_q: load_linear(comp, &format!("{p}.to_q"), dim, heads * dh, false)?,
            to_k: load_linear(comp, &format!("{p}.to_k"), dim, kv_heads * dh, false)?,
            to_v: load_linear(comp, &format!("{p}.to_v"), dim, kv_heads * dh, false)?,
            to_gate: load_linear(comp, &format!("{p}.to_gate"), dim, dim, false)?,
            to_out: load_linear(comp, &format!("{p}.to_out.0"), dim, dim, false)?,
            norm_q: RmsNorm::load(comp, &format!("{p}.norm_q.weight"), dh, eps)?,
            norm_k: RmsNorm::load(comp, &format!("{p}.norm_k.weight"), dh, eps)?,
            heads,
            kv_heads,
            head_dim: dh,
        })
    }

    /// `x` is `groups` independent sequences of `seq` tokens, `[groups * seq,
    /// dim]`; tokens attend only within their group. `rope` (if any) is
    /// indexed by position within the group.
    pub fn forward(&self, x: &[f32], groups: usize, seq: usize, rope: Option<&Rope>) -> Vec<f32> {
        let (h, kvh, dh) = (self.heads, self.kv_heads, self.head_dim);
        let mut q = linear(&self.to_q, x);
        let mut k = linear(&self.to_k, x);
        let v = linear(&self.to_v, x);
        let gate = linear(&self.to_gate, x);
        self.norm_q.forward(&mut q);
        self.norm_k.forward(&mut k);
        if let Some(r) = rope {
            for g in 0..groups {
                r.apply(&mut q[g * seq * h * dh..(g + 1) * seq * h * dh], h);
                r.apply(&mut k[g * seq * kvh * dh..(g + 1) * seq * kvh * dh], kvh);
            }
        }
        let mut ctx = sdpa(&q, &k, &v, groups, seq, h, kvh, dh);
        for (c, g) in ctx.iter_mut().zip(&gate) {
            *c *= sigmoid(*g);
        }
        linear(&self.to_out, &ctx)
    }
}

/// Unmasked scaled dot-product attention. `q` is `[groups, seq, h, dh]`,
/// `k`/`v` are `[groups, seq, kvh, dh]`; query head `i` reads kv head
/// `i / (h / kvh)`. Returns `[groups, seq, h * dh]`.
#[allow(clippy::too_many_arguments)]
fn sdpa(
    q: &[f32], k: &[f32], v: &[f32], groups: usize, seq: usize, h: usize, kvh: usize, dh: usize,
) -> Vec<f32> {
    let rep = h / kvh;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut out = vec![0.0_f32; groups * seq * h * dh];
    out.par_chunks_mut(h * dh).enumerate().for_each(|(row, o)| {
        let g = row / seq;
        let qi = row % seq;
        let mut scores = vec![0.0_f32; seq];
        for head in 0..h {
            let kv = head / rep;
            let qv = &q[((g * seq + qi) * h + head) * dh..][..dh];
            let mut max = f32::NEG_INFINITY;
            for (kj, s) in scores.iter_mut().enumerate() {
                let kr = &k[((g * seq + kj) * kvh + kv) * dh..][..dh];
                *s = qv.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() * scale;
                max = max.max(*s);
            }
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let oh = &mut o[head * dh..(head + 1) * dh];
            for (kj, s) in scores.iter().enumerate() {
                let w = s / sum;
                let vr = &v[((g * seq + kj) * kvh + kv) * dh..][..dh];
                for (a, b) in oh.iter_mut().zip(vr) {
                    *a += w * b;
                }
            }
        }
    });
    out
}

/// `down(silu(gate(x)) * up(x))`.
pub struct SwiGlu {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl SwiGlu {
    pub fn load(comp: &ComponentTensors, p: &str, dim: usize, inter: usize) -> Result<Self> {
        Ok(Self {
            gate: load_linear(comp, &format!("{p}.gate"), dim, inter, false)?,
            up: load_linear(comp, &format!("{p}.up"), dim, inter, false)?,
            down: load_linear(comp, &format!("{p}.down"), inter, dim, false)?,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut g = linear(&self.gate, x);
        let u = linear(&self.up, x);
        for (a, b) in g.iter_mut().zip(&u) {
            *a = silu(*a) * b;
        }
        linear(&self.down, &g)
    }
}

/// Pre-norm block with no modulation and no rotary embedding, used by the
/// text fusion stage.
pub struct TextFusionBlock {
    pub norm1: RmsNorm,
    pub norm2: RmsNorm,
    pub attn: Attention,
    pub ff: SwiGlu,
}

impl TextFusionBlock {
    pub fn load(comp: &ComponentTensors, p: &str, cfg: &Krea2Config) -> Result<Self> {
        let d = cfg.text_hidden_dim;
        Ok(Self {
            norm1: RmsNorm::load(comp, &format!("{p}.norm1.weight"), d, cfg.norm_eps)?,
            norm2: RmsNorm::load(comp, &format!("{p}.norm2.weight"), d, cfg.norm_eps)?,
            attn: Attention::load(
                comp, &format!("{p}.attn"), d, cfg.text_num_attention_heads,
                cfg.text_num_key_value_heads, cfg.norm_eps,
            )?,
            ff: SwiGlu::load(comp, &format!("{p}.ff"), d, cfg.text_intermediate_size)?,
        })
    }

    pub fn forward(&self, x: &mut [f32], groups: usize, seq: usize) {
        let mut h = x.to_vec();
        self.norm1.forward(&mut h);
        let a = self.attn.forward(&h, groups, seq, None);
        x.iter_mut().zip(&a).for_each(|(x, a)| *x += a);
        let mut h = x.to_vec();
        self.norm2.forward(&mut h);
        let f = self.ff.forward(&h);
        x.iter_mut().zip(&f).for_each(|(x, f)| *x += f);
    }
}

/// Text conditioning: fuse the tapped encoder layers, refine across tokens,
/// and project to the transformer width.
pub struct TextEncoderHead {
    pub layerwise: Vec<TextFusionBlock>,
    /// `Linear(num_text_layers -> 1)` weights, collapsing the tap axis.
    pub projector: Vec<f32>,
    pub refiner: Vec<TextFusionBlock>,
    pub txt_norm: RmsNorm,
    pub txt_l1: Linear,
    pub txt_l2: Linear,
    pub num_layers: usize,
    pub text_dim: usize,
}

impl TextEncoderHead {
    pub fn load(comp: &ComponentTensors, cfg: &Krea2Config) -> Result<Self> {
        let td = cfg.text_hidden_dim;
        let d = cfg.hidden();
        let blocks = |kind: &str, n: usize| -> Result<Vec<TextFusionBlock>> {
            (0..n)
                .map(|i| TextFusionBlock::load(comp, &format!("text_fusion.{kind}.{i}"), cfg))
                .collect()
        };
        Ok(Self {
            layerwise: blocks("layerwise_blocks", cfg.num_layerwise_text_blocks)?,
            projector: get_checked(comp, "text_fusion.projector.weight", cfg.num_text_layers)?,
            refiner: blocks("refiner_blocks", cfg.num_refiner_text_blocks)?,
            txt_norm: RmsNorm::load(comp, "txt_in.norm.weight", td, cfg.norm_eps)?,
            txt_l1: load_linear(comp, "txt_in.linear_1", td, d, true)?,
            txt_l2: load_linear(comp, "txt_in.linear_2", d, d, true)?,
            num_layers: cfg.num_text_layers,
            text_dim: td,
        })
    }

    /// `taps` is `[n_tokens, num_layers, text_dim]` (real tokens only — no
    /// padding, which the reference masks out anyway). Returns the text
    /// tokens at transformer width, `[n_tokens, hidden]`.
    pub fn forward(&self, taps: &[f32], n_tokens: usize) -> Vec<f32> {
        let (nl, td) = (self.num_layers, self.text_dim);
        assert_eq!(taps.len(), n_tokens * nl * td, "taps must be [n_tokens, layers, dim]");
        let mut x = taps.to_vec();
        for b in &self.layerwise {
            b.forward(&mut x, n_tokens, nl);
        }
        let mut fused = vec![0.0_f32; n_tokens * td];
        for (t, out) in fused.chunks_mut(td).enumerate() {
            for (l, w) in self.projector.iter().enumerate() {
                let src = &x[(t * nl + l) * td..][..td];
                out.iter_mut().zip(src).for_each(|(o, s)| *o += w * s);
            }
        }
        for b in &self.refiner {
            b.forward(&mut fused, 1, n_tokens);
        }
        self.txt_norm.forward(&mut fused);
        let mut h = linear(&self.txt_l1, &fused);
        gelu_tanh(&mut h);
        linear(&self.txt_l2, &h)
    }
}

/// Timestep embedding: `temb` for the final layer and the shared six-way
/// modulation `temb_mod` for the blocks. Tiny, so it always runs on the CPU.
#[derive(Clone)]
pub struct TimeEmbed {
    pub l1: Linear,
    pub l2: Linear,
    pub mod_proj: Linear,
    pub embed_dim: usize,
}

impl TimeEmbed {
    pub fn load(comp: &ComponentTensors, cfg: &Krea2Config) -> Result<Self> {
        let d = cfg.hidden();
        Ok(Self {
            l1: load_linear(comp, "time_embed.linear_1", cfg.timestep_embed_dim, d, true)?,
            l2: load_linear(comp, "time_embed.linear_2", d, d, true)?,
            mod_proj: load_linear(comp, "time_mod_proj", d, 6 * d, true)?,
            embed_dim: cfg.timestep_embed_dim,
        })
    }

    /// `t` is the flow time in `[0, 1]` (1 = pure noise). Returns
    /// `(temb [d], temb_mod [6d])`.
    pub fn forward(&self, t: f32) -> (Vec<f32>, Vec<f32>) {
        let half = self.embed_dim / 2;
        let mut emb = vec![0.0_f32; self.embed_dim];
        for i in 0..half {
            let freq = (-(1e4_f32.ln()) * i as f32 / half as f32).exp();
            let a = t * 1e3 * freq;
            emb[i] = a.cos();
            emb[half + i] = a.sin();
        }
        let mut h = vec![0.0_f32; self.l1.out_features];
        self.l1.forward(&emb, &mut h);
        gelu_tanh(&mut h);
        let mut temb = vec![0.0_f32; self.l2.out_features];
        self.l2.forward(&h, &mut temb);
        let mut g = temb.clone();
        gelu_tanh(&mut g);
        let mut temb_mod = vec![0.0_f32; self.mod_proj.out_features];
        self.mod_proj.forward(&g, &mut temb_mod);
        (temb, temb_mod)
    }
}

/// One single-stream block. The six modulation vectors are the shared
/// `temb_mod` plus this block's learned `scale_shift_table`, in the order
/// prescale, preshift, pregate, postscale, postshift, postgate — scale
/// *before* shift, the reverse of Flux.
pub struct Block {
    pub table: Vec<f32>,
    pub norm1: RmsNorm,
    pub norm2: RmsNorm,
    pub attn: Attention,
    pub ff: SwiGlu,
}

impl Block {
    pub fn load(comp: &ComponentTensors, p: &str, cfg: &Krea2Config) -> Result<Self> {
        let d = cfg.hidden();
        Ok(Self {
            table: get_checked(comp, &format!("{p}.scale_shift_table"), 6 * d)?,
            norm1: RmsNorm::load(comp, &format!("{p}.norm1.weight"), d, cfg.norm_eps)?,
            norm2: RmsNorm::load(comp, &format!("{p}.norm2.weight"), d, cfg.norm_eps)?,
            attn: Attention::load(
                comp, &format!("{p}.attn"), d, cfg.num_attention_heads,
                cfg.num_key_value_heads, cfg.norm_eps,
            )?,
            ff: SwiGlu::load(comp, &format!("{p}.ff"), d, cfg.intermediate_size)?,
        })
    }

    pub fn forward(&self, x: &mut [f32], temb_mod: &[f32], rope: &Rope) {
        let d = self.norm1.weight.len();
        let m = |i: usize| -> Vec<f32> {
            (0..d).map(|j| temb_mod[i * d + j] + self.table[i * d + j]).collect()
        };
        let (prescale, preshift, pregate) = (m(0), m(1), m(2));
        let (postscale, postshift, postgate) = (m(3), m(4), m(5));
        let seq = x.len() / d;

        let mut h = x.to_vec();
        self.norm1.forward(&mut h);
        modulate(&mut h, &prescale, &preshift);
        let a = self.attn.forward(&h, 1, seq, Some(rope));
        gated_add(x, &a, &pregate);

        let mut h = x.to_vec();
        self.norm2.forward(&mut h);
        modulate(&mut h, &postscale, &postshift);
        let f = self.ff.forward(&h);
        gated_add(x, &f, &postgate);
    }
}

fn modulate(x: &mut [f32], scale: &[f32], shift: &[f32]) {
    let d = scale.len();
    for row in x.chunks_mut(d) {
        for ((v, s), b) in row.iter_mut().zip(scale).zip(shift) {
            *v = *v * (1.0 + s) + b;
        }
    }
}

fn gated_add(x: &mut [f32], y: &[f32], gate: &[f32]) {
    let d = gate.len();
    for (xr, yr) in x.chunks_mut(d).zip(y.chunks(d)) {
        for ((a, b), g) in xr.iter_mut().zip(yr).zip(gate) {
            *a += g * b;
        }
    }
}

/// The whole transformer, f32 on the CPU.
pub struct Krea2Transformer {
    pub cfg: Krea2Config,
    pub img_in: Linear,
    pub time: TimeEmbed,
    pub text: TextEncoderHead,
    pub blocks: Vec<Block>,
    pub final_table: Vec<f32>,
    pub final_norm: RmsNorm,
    pub final_linear: Linear,
}

impl Krea2Transformer {
    /// Load every tensor as f32. Only sensible for small configs — the real
    /// model needs ~52 GB this way; the GPU loader streams instead.
    pub fn load(cfg: Krea2Config, comp: &ComponentTensors) -> Result<Self> {
        let d = cfg.hidden();
        Ok(Self {
            img_in: load_linear(comp, "img_in", cfg.in_channels, d, true)?,
            time: TimeEmbed::load(comp, &cfg)?,
            text: TextEncoderHead::load(comp, &cfg)?,
            blocks: (0..cfg.num_layers)
                .map(|i| Block::load(comp, &format!("transformer_blocks.{i}"), &cfg))
                .collect::<Result<_>>()?,
            final_table: get_checked(comp, "final_layer.scale_shift_table", 2 * d)?,
            final_norm: RmsNorm::load(comp, "final_layer.norm.weight", d, cfg.norm_eps)?,
            final_linear: load_linear(comp, "final_layer.linear", d, cfg.in_channels, true)?,
            cfg,
        })
    }

    /// Velocity for packed image tokens `img` `[grid_h * grid_w, in_channels]`
    /// at flow time `t`, given text tokens already through
    /// [`TextEncoderHead::forward`] (`txt` `[n_txt, hidden]`).
    pub fn forward(&self, img: &[f32], grid_h: usize, grid_w: usize, txt: &[f32], t: f32) -> Vec<f32> {
        let d = self.cfg.hidden();
        let n_img = grid_h * grid_w;
        let n_txt = txt.len() / d;
        assert_eq!(img.len(), n_img * self.cfg.in_channels);

        let (temb, temb_mod) = self.time.forward(t);
        let img_tok = linear(&self.img_in, img);
        let mut x = Vec::with_capacity((n_txt + n_img) * d);
        x.extend_from_slice(txt);
        x.extend_from_slice(&img_tok);

        let mut pos = vec![[0.0; 3]; n_txt];
        pos.extend(image_positions(grid_h, grid_w));
        let rope = Rope::new(&pos, self.cfg.axes_dims_rope, self.cfg.rope_theta);

        for b in &self.blocks {
            b.forward(&mut x, &temb_mod, &rope);
        }

        let mut h = x[n_txt * d..].to_vec();
        self.final_norm.forward(&mut h);
        let scale: Vec<f32> = (0..d).map(|j| temb[j] + self.final_table[j]).collect();
        let shift: Vec<f32> = (0..d).map(|j| temb[j] + self.final_table[d + j]).collect();
        modulate(&mut h, &scale, &shift);
        linear(&self.final_linear, &h)
    }
}

/// Pack a latent `[16, H, W]` into tokens `[(H/p)(W/p), 16 p p]`, raster
/// order, features ordered `(c, py, px)` — diffusers' `_pack_latents`.
pub fn pack_latents(z: &[f32], c: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (gh, gw) = (h / p, w / p);
    let mut out = vec![0.0_f32; z.len()];
    let feat = c * p * p;
    for gy in 0..gh {
        for gx in 0..gw {
            let tok = (gy * gw + gx) * feat;
            for ch in 0..c {
                for py in 0..p {
                    for px in 0..p {
                        out[tok + (ch * p + py) * p + px] = z[(ch * h + gy * p + py) * w + gx * p + px];
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`pack_latents`].
pub fn unpack_latents(tokens: &[f32], c: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let (gh, gw) = (h / p, w / p);
    let mut out = vec![0.0_f32; tokens.len()];
    let feat = c * p * p;
    for gy in 0..gh {
        for gx in 0..gw {
            let tok = (gy * gw + gx) * feat;
            for ch in 0..c {
                for py in 0..p {
                    for px in 0..p {
                        out[(ch * h + gy * p + py) * w + gx * p + px] = tokens[tok + (ch * p + py) * p + px];
                    }
                }
            }
        }
    }
    out
}

/// Every tensor the transformer expects, with its element count. Used to
/// validate a checkpoint before uploading anything, and by tests to build
/// synthetic checkpoints.
pub fn tensor_specs(cfg: &Krea2Config) -> Vec<(String, Vec<usize>)> {
    let d = cfg.hidden();
    let td = cfg.text_hidden_dim;
    let mut v: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |n: String, s: Vec<usize>| v.push((n, s));
    push("img_in.weight".into(), vec![d, cfg.in_channels]);
    push("img_in.bias".into(), vec![d]);
    push("time_embed.linear_1.weight".into(), vec![d, cfg.timestep_embed_dim]);
    push("time_embed.linear_1.bias".into(), vec![d]);
    push("time_embed.linear_2.weight".into(), vec![d, d]);
    push("time_embed.linear_2.bias".into(), vec![d]);
    push("time_mod_proj.weight".into(), vec![6 * d, d]);
    push("time_mod_proj.bias".into(), vec![6 * d]);
    let attn = |push: &mut dyn FnMut(String, Vec<usize>), p: &str, dim: usize, h: usize, kvh: usize| {
        let dh = dim / h;
        push(format!("{p}.to_q.weight"), vec![h * dh, dim]);
        push(format!("{p}.to_k.weight"), vec![kvh * dh, dim]);
        push(format!("{p}.to_v.weight"), vec![kvh * dh, dim]);
        push(format!("{p}.to_gate.weight"), vec![dim, dim]);
        push(format!("{p}.to_out.0.weight"), vec![dim, dim]);
        push(format!("{p}.norm_q.weight"), vec![dh]);
        push(format!("{p}.norm_k.weight"), vec![dh]);
    };
    let ff = |push: &mut dyn FnMut(String, Vec<usize>), p: &str, dim: usize, inter: usize| {
        push(format!("{p}.gate.weight"), vec![inter, dim]);
        push(format!("{p}.up.weight"), vec![inter, dim]);
        push(format!("{p}.down.weight"), vec![dim, inter]);
    };
    for (kind, n) in [
        ("layerwise_blocks", cfg.num_layerwise_text_blocks),
        ("refiner_blocks", cfg.num_refiner_text_blocks),
    ] {
        for i in 0..n {
            let p = format!("text_fusion.{kind}.{i}");
            push(format!("{p}.norm1.weight"), vec![td]);
            push(format!("{p}.norm2.weight"), vec![td]);
            attn(&mut push, &format!("{p}.attn"), td, cfg.text_num_attention_heads, cfg.text_num_key_value_heads);
            ff(&mut push, &format!("{p}.ff"), td, cfg.text_intermediate_size);
        }
    }
    push("text_fusion.projector.weight".into(), vec![1, cfg.num_text_layers]);
    push("txt_in.norm.weight".into(), vec![td]);
    push("txt_in.linear_1.weight".into(), vec![d, td]);
    push("txt_in.linear_1.bias".into(), vec![d]);
    push("txt_in.linear_2.weight".into(), vec![d, d]);
    push("txt_in.linear_2.bias".into(), vec![d]);
    for i in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{i}");
        push(format!("{p}.scale_shift_table"), vec![6, d]);
        push(format!("{p}.norm1.weight"), vec![d]);
        push(format!("{p}.norm2.weight"), vec![d]);
        attn(&mut push, &format!("{p}.attn"), d, cfg.num_attention_heads, cfg.num_key_value_heads);
        ff(&mut push, &format!("{p}.ff"), d, cfg.intermediate_size);
    }
    push("final_layer.scale_shift_table".into(), vec![2, d]);
    push("final_layer.norm.weight".into(), vec![d]);
    push("final_layer.linear.weight".into(), vec![cfg.in_channels, d]);
    push("final_layer.linear.bias".into(), vec![cfg.in_channels]);
    v
}

/// Names that are missing or have the wrong element count, as messages.
pub fn check_inventory(cfg: &Krea2Config, comp: &ComponentTensors) -> Vec<String> {
    tensor_specs(cfg)
        .into_iter()
        .filter_map(|(name, shape)| {
            let want: usize = shape.iter().product();
            match comp.numel(&name) {
                None => Some(format!("missing {name}")),
                Some(n) if n != want => Some(format!("{name}: {n} elements, expected {want}")),
                _ => None,
            }
        })
        .collect()
}
