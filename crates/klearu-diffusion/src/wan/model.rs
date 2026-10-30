//! Wan DiT blocks and forward pass (CPU reference).

use crate::layers::{silu_inplace, LayerNorm, Linear};
use crate::unet::time_embedding::sinusoidal_embedding;

use super::config::WanDitConfig;
use super::rope::{apply_rope, RopeTables};

/// RoPE tables plus the `(f, y, x)` grid position of each query token.
/// Present for self-attention, absent for cross-attention.
pub type RopeArg<'a> = (&'a RopeTables, &'a [(usize, usize, usize)]);

/// RMS norm over the whole vector, with a learned scale.
///
/// Wan's `qk_norm: "rms_norm_across_heads"` means exactly this: normalise the
/// full `inner_dim` **before** the tensor is split into heads, so every head
/// shares one normalisation. Contrast Qwen3-VL, which norms each head's
/// `head_dim` separately.
#[derive(Clone, Debug)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { weight: vec![1.0; dim], eps }
    }

    /// Normalise each `dim`-long row of `x` in place.
    pub fn forward_inplace(&self, x: &mut [f32]) {
        let d = self.weight.len();
        for row in x.chunks_mut(d) {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + self.eps).sqrt();
            for (v, w) in row.iter_mut().zip(&self.weight) {
                *v = *v * inv * w;
            }
        }
    }
}

/// `gelu-approximate`: the tanh formulation, not the erf one.
fn gelu_tanh_inplace(x: &mut [f32]) {
    const C: f32 = 0.797_885; // sqrt(2/pi)
    for v in x.iter_mut() {
        let t = *v;
        *v = 0.5 * t * (1.0 + (C * (t + 0.044_715 * t * t * t)).tanh());
    }
}

/// Softmax over a row, in place.
fn softmax(row: &mut [f32]) {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Self- or cross-attention. `norm_q`/`norm_k` span the full inner dim.
pub struct WanAttention {
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_out: Linear,
    pub norm_q: RmsNorm,
    pub norm_k: RmsNorm,
    pub heads: usize,
    pub head_dim: usize,
}

impl WanAttention {
    pub fn new(dim: usize, kv_dim: usize, heads: usize, head_dim: usize, eps: f32) -> Self {
        let inner = heads * head_dim;
        Self {
            to_q: Linear::new(dim, inner, true),
            to_k: Linear::new(kv_dim, inner, true),
            to_v: Linear::new(kv_dim, inner, true),
            to_out: Linear::new(inner, dim, true),
            norm_q: RmsNorm::new(inner, eps),
            norm_k: RmsNorm::new(inner, eps),
            heads,
            head_dim,
        }
    }

    /// `x`: `[n_q, dim]`. `kv`: `[n_kv, kv_dim]` — the same tensor as `x` for
    /// self-attention. `rope` is applied only when given (self-attention);
    /// cross-attention has no positional structure to rotate.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        n_q: usize,
        kv: &[f32],
        n_kv: usize,
        rope: Option<RopeArg<'_>>,
        out: &mut [f32],
    ) {
        let inner = self.heads * self.head_dim;
        let hd = self.head_dim;

        let mut q = vec![0.0; n_q * inner];
        let mut k = vec![0.0; n_kv * inner];
        let mut v = vec![0.0; n_kv * inner];
        self.to_q.forward_batch(x, &mut q);
        self.to_k.forward_batch(kv, &mut k);
        self.to_v.forward_batch(kv, &mut v);

        // Norm before the head split — this is what "across_heads" means.
        self.norm_q.forward_inplace(&mut q);
        self.norm_k.forward_inplace(&mut k);

        if let Some((tables, grid)) = rope {
            debug_assert_eq!(grid.len(), n_q);
            debug_assert_eq!(n_q, n_kv, "rope only applies to self-attention");
            for (t, &(f, y, xx)) in grid.iter().enumerate() {
                let (cos, sin) = tables.token(f, y, xx);
                for h in 0..self.heads {
                    let a = t * inner + h * hd;
                    apply_rope(&mut q[a..a + hd], &cos, &sin);
                    apply_rope(&mut k[a..a + hd], &cos, &sin);
                }
            }
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let mut ctx = vec![0.0; n_q * inner];
        let mut scores = vec![0.0; n_kv];
        for h in 0..self.heads {
            for i in 0..n_q {
                let qa = i * inner + h * hd;
                for (j, s) in scores.iter_mut().enumerate() {
                    let ka = j * inner + h * hd;
                    let mut acc = 0.0;
                    for d in 0..hd {
                        acc += q[qa + d] * k[ka + d];
                    }
                    *s = acc * scale;
                }
                softmax(&mut scores);
                let oa = i * inner + h * hd;
                for (j, &p) in scores.iter().enumerate() {
                    let va = j * inner + h * hd;
                    for d in 0..hd {
                        ctx[oa + d] += p * v[va + d];
                    }
                }
            }
        }
        self.to_out.forward_batch(&ctx, out);
    }
}

/// Per-block AdaLN modulation.
///
/// The two variants are not interchangeable, and picking the wrong one is
/// silent: Wan 2.1 and the 14B models emit one `[6, dim]` per sample, while
/// TI2V-5B emits one **per token**, which is how it holds some frames at
/// timestep 0 for image-to-video while denoising the rest.
pub enum Modulation<'a> {
    /// `[6 * dim]`
    PerSample(&'a [f32]),
    /// `[seq * 6 * dim]`
    PerToken(&'a [f32]),
}

impl Modulation<'_> {
    /// Modulation vector `which` (0..6) for token `t`, added to the block's
    /// `scale_shift_table`.
    fn get(&self, table: &[f32], which: usize, t: usize, dim: usize, out: &mut [f32]) {
        let base = which * dim;
        match self {
            Modulation::PerSample(v) => {
                for k in 0..dim {
                    out[k] = table[base + k] + v[base + k];
                }
            }
            Modulation::PerToken(v) => {
                let off = (t * 6 + which) * dim;
                for k in 0..dim {
                    out[k] = table[base + k] + v[off + k];
                }
            }
        }
    }
}

/// One transformer block.
pub struct WanBlock {
    /// No affine parameters — the scale and shift come from the timestep.
    pub norm1: LayerNorm,
    pub attn1: WanAttention,
    /// Affine, because `cross_attn_norm` is true. The only norm in the block
    /// with weights in the checkpoint.
    pub norm2: LayerNorm,
    pub attn2: WanAttention,
    pub norm3: LayerNorm,
    pub ffn_proj: Linear,
    pub ffn_out: Linear,
    /// `[6, dim]`, added to the timestep modulation.
    pub scale_shift_table: Vec<f32>,
    pub dim: usize,
}

impl WanBlock {
    pub fn new(cfg: &WanDitConfig) -> Self {
        let d = cfg.inner_dim();
        let (h, hd) = (cfg.num_attention_heads, cfg.attention_head_dim);
        Self {
            norm1: LayerNorm::new(d, cfg.eps),
            attn1: WanAttention::new(d, d, h, hd, cfg.eps),
            norm2: LayerNorm::new(d, cfg.eps),
            attn2: WanAttention::new(d, d, h, hd, cfg.eps),
            norm3: LayerNorm::new(d, cfg.eps),
            ffn_proj: Linear::new(d, cfg.ffn_dim, true),
            ffn_out: Linear::new(cfg.ffn_dim, d, true),
            scale_shift_table: vec![0.0; 6 * d],
            dim: d,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &mut [f32],
        seq: usize,
        enc: &[f32],
        n_ctx: usize,
        temb: &Modulation,
        rope: &RopeTables,
        grid: &[(usize, usize, usize)],
    ) {
        let d = self.dim;
        let mut m = vec![0.0; d];

        // 1. Self-attention, modulated by (shift, scale) and gated.
        let mut h = x.to_vec();
        self.norm1.forward_inplace(&mut h);
        for t in 0..seq {
            self.temb_apply(temb, 1, t, &mut m); // scale_msa
            let row = &mut h[t * d..(t + 1) * d];
            for (k, v) in row.iter_mut().enumerate() {
                *v *= 1.0 + m[k];
            }
            self.temb_apply(temb, 0, t, &mut m); // shift_msa
            for (k, v) in row.iter_mut().enumerate() {
                *v += m[k];
            }
        }
        let mut attn = vec![0.0; seq * d];
        self.attn1.forward(&h, seq, &h, seq, Some((rope, grid)), &mut attn);
        for t in 0..seq {
            self.temb_apply(temb, 2, t, &mut m); // gate_msa
            for k in 0..d {
                x[t * d + k] += attn[t * d + k] * m[k];
            }
        }

        // 2. Cross-attention to the text states — ungated, unmodulated.
        let mut h = x.to_vec();
        self.norm2.forward_inplace(&mut h);
        let mut attn = vec![0.0; seq * d];
        self.attn2.forward(&h, seq, enc, n_ctx, None, &mut attn);
        for (o, a) in x.iter_mut().zip(&attn) {
            *o += *a;
        }

        // 3. Feed-forward, modulated by (c_shift, c_scale) and gated.
        let mut h = x.to_vec();
        self.norm3.forward_inplace(&mut h);
        for t in 0..seq {
            self.temb_apply(temb, 4, t, &mut m); // c_scale_msa
            let row = &mut h[t * d..(t + 1) * d];
            for (k, v) in row.iter_mut().enumerate() {
                *v *= 1.0 + m[k];
            }
            self.temb_apply(temb, 3, t, &mut m); // c_shift_msa
            for (k, v) in row.iter_mut().enumerate() {
                *v += m[k];
            }
        }
        let mut ff = vec![0.0; seq * self.ffn_proj.out_features];
        self.ffn_proj.forward_batch(&h, &mut ff);
        gelu_tanh_inplace(&mut ff);
        let mut ff_out = vec![0.0; seq * d];
        self.ffn_out.forward_batch(&ff, &mut ff_out);
        for t in 0..seq {
            self.temb_apply(temb, 5, t, &mut m); // c_gate_msa
            for k in 0..d {
                x[t * d + k] += ff_out[t * d + k] * m[k];
            }
        }
    }

    fn temb_apply(&self, temb: &Modulation, which: usize, t: usize, out: &mut [f32]) {
        temb.get(&self.scale_shift_table, which, t, self.dim, out);
    }
}

/// The timestep and text embedders.
///
/// Split out because they are *small* — about 350 MB in f32 against the
/// blocks' 20 GB — and run on the CPU even in the GPU path. Keeping them in
/// their own cloneable struct lets `GpuWanDit` own a copy, so the full f32
/// `WanDit` can be dropped once its weights are on the GPU.
#[derive(Clone)]
pub struct Conditioning {
    pub time_linear_1: Linear,
    pub time_linear_2: Linear,
    pub time_proj: Linear,
    pub text_linear_1: Linear,
    pub text_linear_2: Linear,
    pub freq_dim: usize,
}

impl Conditioning {
    pub fn new(cfg: &WanDitConfig) -> Self {
        let d = cfg.inner_dim();
        Self {
            time_linear_1: Linear::new(cfg.freq_dim, d, true),
            time_linear_2: Linear::new(d, d, true),
            time_proj: Linear::new(d, cfg.time_proj_dim(), true),
            text_linear_1: Linear::new(cfg.text_dim, d, true),
            text_linear_2: Linear::new(d, d, true),
            freq_dim: cfg.freq_dim,
        }
    }

    /// Project text-encoder hidden states into the model width, GELU-tanh
    /// between the two layers.
    pub fn embed_text(&self, enc: &[f32], n_ctx: usize) -> Vec<f32> {
        let d = self.time_linear_2.out_features;
        let mut h = vec![0.0; n_ctx * d];
        self.text_linear_1.forward_batch(enc, &mut h);
        gelu_tanh_inplace(&mut h);
        let mut out = vec![0.0; n_ctx * d];
        self.text_linear_2.forward_batch(&h, &mut out);
        out
    }

    /// Timestep -> `(temb [d], timestep_proj [6*d])`.
    pub fn embed_timestep(&self, t: f32) -> (Vec<f32>, Vec<f32>) {
        let d = self.time_linear_2.out_features;
        // Timesteps(flip_sin_to_cos=true, downscale_freq_shift=0): cos first.
        let sin = sinusoidal_embedding(t, self.freq_dim, 10_000.0);
        let mut h = vec![0.0; d];
        self.time_linear_1.forward(&sin, &mut h);
        silu_inplace(&mut h);
        let mut temb = vec![0.0; d];
        self.time_linear_2.forward(&h, &mut temb);

        let mut act = temb.clone();
        silu_inplace(&mut act);
        let mut proj = vec![0.0; self.time_proj.out_features];
        self.time_proj.forward(&act, &mut proj);
        (temb, proj)
    }
}

/// Token grid `(f, y, x)` in sequence order — width fastest, matching the
/// reference's `flatten(2).transpose(1, 2)` over the Conv3d output.
pub fn token_grid(ppf: usize, pph: usize, ppw: usize) -> Vec<(usize, usize, usize)> {
    let mut g = Vec::with_capacity(ppf * pph * ppw);
    for f in 0..ppf {
        for y in 0..pph {
            for x in 0..ppw {
                g.push((f, y, x));
            }
        }
    }
    g
}

/// Fold the latent into patch tokens: `[seq, in_c * p_t * p_h * p_w]`,
/// channel-major within a patch to match the Conv3d weight layout
/// `[dim, in_c, p_t, p_h, p_w]`.
pub fn patchify(
    cfg: &WanDitConfig, z: &[f32], c: usize, t: usize, h: usize, w: usize,
) -> Vec<f32> {
    let (pt, ph, pw) = cfg.patch();
    let (ppf, pph, ppw) = (t / pt, h / ph, w / pw);
    let vol = pt * ph * pw;
    let mut out = vec![0.0; ppf * pph * ppw * c * vol];
    let mut i = 0;
    for f in 0..ppf {
        for y in 0..pph {
            for x in 0..ppw {
                for ci in 0..c {
                    for a in 0..pt {
                        for b in 0..ph {
                            for d in 0..pw {
                                let src = ((ci * t + f * pt + a) * h + y * ph + b) * w
                                    + x * pw + d;
                                out[i] = z[src];
                                i += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`patchify`] over `proj_out`'s output, whose per-token layout is
/// `(p_t, p_h, p_w, out_c)` — channels **innermost**, unlike the patch
/// embedding's channel-major input.
pub fn unpatchify(cfg: &WanDitConfig, tokens: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
    let (pt, ph, pw) = cfg.patch();
    let c = cfg.out_channels;
    let (ppf, pph, ppw) = (t / pt, h / ph, w / pw);
    let per = pt * ph * pw * c;
    let mut out = vec![0.0; c * t * h * w];
    for f in 0..ppf {
        for y in 0..pph {
            for x in 0..ppw {
                let tok = ((f * pph + y) * ppw + x) * per;
                for a in 0..pt {
                    for b in 0..ph {
                        for d in 0..pw {
                            for ci in 0..c {
                                let src = tok + ((a * ph + b) * pw + d) * c + ci;
                                let dst = ((ci * t + f * pt + a) * h + y * ph + b) * w
                                    + x * pw + d;
                                out[dst] = tokens[src];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// The full Wan DiT.
pub struct WanDit {
    pub cfg: WanDitConfig,
    /// Conv3d with kernel == stride == patch, which is a linear map over each
    /// patch: `[dim, in_c * p_t * p_h * p_w]`.
    pub patch_embedding: Linear,
    /// The small timestep/text embedders, held separately so the GPU path can
    /// own a copy and let the rest of the f32 model be dropped.
    pub cond: Conditioning,
    pub blocks: Vec<WanBlock>,
    pub norm_out: LayerNorm,
    /// `[2, dim]` — the final shift and scale.
    pub scale_shift_table: Vec<f32>,
    pub proj_out: Linear,
    pub rope: RopeTables,
}

impl WanDit {
    pub fn new(cfg: WanDitConfig) -> Self {
        let d = cfg.inner_dim();
        let (pt, ph, pw) = cfg.patch();
        let patch_vol = pt * ph * pw;
        let rope = RopeTables::new(&cfg, 10_000.0);
        let blocks = (0..cfg.num_layers).map(|_| WanBlock::new(&cfg)).collect();
        Self {
            patch_embedding: Linear::new(cfg.in_channels * patch_vol, d, true),
            cond: Conditioning::new(&cfg),
            blocks,
            norm_out: LayerNorm::new(d, cfg.eps),
            scale_shift_table: vec![0.0; 2 * d],
            proj_out: Linear::new(d, patch_vol * cfg.out_channels, true),
            cfg,
            rope,
        }
    }

    /// Config and conditioning only — **no blocks allocated**.
    ///
    /// `WanDit::new` allocates every block up front, which for the 5B model is
    /// ~20 GB of zeroed f32 before a single weight is read. The GPU loader
    /// only needs somewhere to hang `missing_tensors`/`mismatched_tensors`,
    /// both of which read the config rather than the blocks.
    pub fn new_shallow(cfg: WanDitConfig) -> Self {
        let d = cfg.inner_dim();
        let (pt, ph, pw) = cfg.patch();
        let patch_vol = pt * ph * pw;
        let rope = RopeTables::new(&cfg, 10_000.0);
        Self {
            patch_embedding: Linear::new(cfg.in_channels * patch_vol, d, true),
            cond: Conditioning::new(&cfg),
            blocks: Vec::new(),
            norm_out: LayerNorm::new(d, cfg.eps),
            scale_shift_table: vec![0.0; 2 * d],
            proj_out: Linear::new(d, patch_vol * cfg.out_channels, true),
            cfg,
            rope,
        }
    }

    /// See [`token_grid`].
    pub fn token_grid(&self, ppf: usize, pph: usize, ppw: usize) -> Vec<(usize, usize, usize)> {
        token_grid(ppf, pph, ppw)
    }

    /// See [`Conditioning::embed_text`].
    pub fn embed_text(&self, enc: &[f32], n_ctx: usize) -> Vec<f32> {
        self.cond.embed_text(enc, n_ctx)
    }

    /// See [`Conditioning::embed_timestep`].
    pub fn embed_timestep(&self, t: f32) -> (Vec<f32>, Vec<f32>) {
        self.cond.embed_timestep(t)
    }

    /// See [`patchify`].
    pub fn patchify(&self, z: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
        patchify(&self.cfg, z, c, t, h, w)
    }

    /// See [`unpatchify`].
    pub fn unpatchify(&self, tokens: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        unpatchify(&self.cfg, tokens, t, h, w)
    }

    /// One denoising step.
    ///
    /// `z` is `[in_c, T, H, W]`. `timesteps` is either a single value (one
    /// modulation for the whole sample) or one per token — the TI2V path,
    /// which must have exactly `seq` entries. `enc` is `[n_ctx, text_dim]`.
    pub fn forward(
        &self,
        z: &[f32],
        (t_len, h_len, w_len): (usize, usize, usize),
        timesteps: &[f32],
        enc: &[f32],
        n_ctx: usize,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.inner_dim();
        let (pt, ph, pw) = cfg.patch();
        let (ppf, pph, ppw) = (t_len / pt, h_len / ph, w_len / pw);
        let seq = ppf * pph * ppw;
        assert_eq!(z.len(), cfg.in_channels * t_len * h_len * w_len, "latent size mismatch");

        let patches = self.patchify(z, cfg.in_channels, t_len, h_len, w_len);
        let mut x = vec![0.0; seq * d];
        self.patch_embedding.forward_batch(&patches, &mut x);

        let enc = self.embed_text(enc, n_ctx);
        let grid = self.token_grid(ppf, pph, ppw);

        // Per-token conditioning when a timestep is supplied for every token.
        let per_token = timesteps.len() > 1;
        assert!(
            !per_token || timesteps.len() == seq,
            "per-token timesteps must have one entry per token ({} vs {seq})",
            timesteps.len()
        );

        let mut temb_final = vec![0.0; if per_token { seq * d } else { d }];
        let mut proj = vec![0.0; if per_token { seq * 6 * d } else { 6 * d }];
        for (i, &t) in timesteps.iter().enumerate() {
            let (te, pr) = self.embed_timestep(t);
            if per_token {
                temb_final[i * d..(i + 1) * d].copy_from_slice(&te);
                proj[i * 6 * d..(i + 1) * 6 * d].copy_from_slice(&pr);
            } else {
                temb_final.copy_from_slice(&te);
                proj.copy_from_slice(&pr);
            }
        }
        let modulation = if per_token {
            Modulation::PerToken(&proj)
        } else {
            Modulation::PerSample(&proj)
        };

        for block in &self.blocks {
            block.forward(&mut x, seq, &enc, n_ctx, &modulation, &self.rope, &grid);
        }

        // Final norm with its own two-vector modulation.
        self.norm_out.forward_inplace(&mut x);
        for t in 0..seq {
            // Both shift and scale add the *same* temb row; the two halves of
            // scale_shift_table are what separates them.
            let te = if per_token { &temb_final[t * d..(t + 1) * d] } else { &temb_final[..] };
            for k in 0..d {
                let shift = self.scale_shift_table[k] + te[k];
                let scale = self.scale_shift_table[d + k] + te[k];
                let v = &mut x[t * d + k];
                *v = *v * (1.0 + scale) + shift;
            }
        }

        let mut tokens = vec![0.0; seq * pt * ph * pw * cfg.out_channels];
        self.proj_out.forward_batch(&x, &mut tokens);
        self.unpatchify(&tokens, t_len, h_len, w_len)
    }
}
