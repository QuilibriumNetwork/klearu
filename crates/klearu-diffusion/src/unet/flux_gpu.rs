//! Metal GPU implementation of the Flux MMDiT transformer.
//!
//! Mirrors the CPU [`super::flux::FluxTransformer`] math exactly, but runs the
//! 19 double + 38 single blocks GPU-resident (the CPU path is ~12B params of
//! pure-CPU matmuls — impractically slow). Reuses the shared Metal primitives
//! (`sgemm`, MPSGraph SDPA, RMSNorm, LayerNorm, SiLU, residual add, permutes)
//! plus the four Flux-specific kernels (`gelu_erf`, interleaved-pair RoPE,
//! `modulate`, `gate_mul`) and `slice_cols` / `blit_copy` helpers.
//!
//! Gated on `feature = "metal"`. Build the GPU transformer with
//! [`GpuFluxTransformer::from_cpu`] (after the CPU loader has read the weights).

#![cfg(feature = "metal")]

use super::flux::{
    build_rope_2d, timestep_embedding, DoubleBlock, FinalLayer, FluxConfig, FluxTransformer,
    Linear, MLPEmbedder, SingleBlock,
};
use crate::metal_backend::{
    bias_add_f16_gpu, blit_copy_f16, eadd_f16_gpu, gate_mul_f16_gpu, gelu_erf_f16_gpu,
    layer_norm_f16_gpu, modulate_f16_gpu, mpsgraph_sdpa_bf16_gpu, permute_hl_to_lh_f16_gpu,
    permute_lh_to_hl_f16_gpu, rms_norm_f16_gpu, rope_pairs_f16_gpu, set_use_f32_sgemm,
    sgemm_f16_a_btrans_buf, slice_cols_f16_gpu, upload_f32_as_f16_buffer, GpuTensor,
};
use metal::Buffer;

const EPS: f32 = 1e-6;

/// GPU linear layer: bf16 weight `[out, in]` + optional bias.
struct GpuLinear {
    weight: Buffer,
    bias: Option<Buffer>,
    in_f: usize,
    out_f: usize,
}

impl GpuLinear {
    fn from_cpu(l: &Linear) -> Self {
        Self {
            weight: upload_f32_as_f16_buffer(&l.weight),
            bias: l.bias.as_ref().map(|b| upload_f32_as_f16_buffer(b)),
            in_f: l.in_features,
            out_f: l.out_features,
        }
    }

    /// `x [m, in]` → `[m, out]`.
    fn forward(&self, x: &GpuTensor, m: usize) -> GpuTensor {
        let mut out = GpuTensor::new_f16(vec![m, self.out_f]);
        sgemm_f16_a_btrans_buf(&out.buffer, &x.buffer, &self.weight, m, self.out_f, self.in_f);
        if let Some(b) = &self.bias {
            bias_add_f16_gpu(&mut out, b, 1, self.out_f);
        }
        out
    }
}

/// Plain LayerNorm (no affine) on `x [n, d]`, in place — Flux modulates with
/// AdaLN externally so the norm itself has γ=1, β=0.
fn layer_norm_no_affine(x: &mut GpuTensor, _n: usize, d: usize, gamma: &Buffer, zeros: &Buffer) {
    layer_norm_f16_gpu(x, gamma, zeros, d, EPS);
}

/// Joint scaled-dot-product attention over `q,k,v [n, nh·dh]` (no mask).
/// Returns `[n, nh·dh]`.
fn joint_attention(q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, n: usize, nh: usize, dh: usize) -> GpuTensor {
    let scale = 1.0_f32 / (dh as f32).sqrt();
    let q_hl = permute_lh_to_hl_f16_gpu(q, 1, n, nh, dh); // [nh, n, dh]
    let k_hl = permute_lh_to_hl_f16_gpu(k, 1, n, nh, dh);
    let v_hl = permute_lh_to_hl_f16_gpu(v, 1, n, nh, dh);
    let mut o_hl = GpuTensor::new_f16(vec![nh * n * dh]);
    mpsgraph_sdpa_bf16_gpu(&q_hl, &k_hl, &v_hl, &mut o_hl, 1, nh, n, n, dh, scale);
    permute_hl_to_lh_f16_gpu(&o_hl, 1, n, nh, dh) // [n, nh*dh]
}

/// One stream's QKV → q/k/v split + QK-norm + (the q/k are RoPE'd after the
/// joint concat by the caller). Returns (q, k, v) each `[n, h]`.
#[allow(clippy::too_many_arguments)]
fn qkv_split_norm(
    qkv: &GpuTensor, n: usize, h: usize, nh: usize, dh: usize,
    norm_q: &Buffer, norm_k: &Buffer,
) -> (GpuTensor, GpuTensor, GpuTensor) {
    let mut q = slice_cols_f16_gpu(qkv, n, 3 * h, 0, h);
    let mut k = slice_cols_f16_gpu(qkv, n, 3 * h, h, h);
    let v = slice_cols_f16_gpu(qkv, n, 3 * h, 2 * h, h);
    rms_norm_f16_gpu(&mut q, norm_q, n * nh, dh, EPS);
    rms_norm_f16_gpu(&mut k, norm_k, n * nh, dh, EPS);
    (q, k, v)
}

/// Concatenate `[txt; img]` along the token axis (contiguous in `[n, h]`).
fn concat_tokens(txt: &GpuTensor, l_txt: usize, img: &GpuTensor, l_img: usize, h: usize) -> GpuTensor {
    let out = GpuTensor::new_f16(vec![l_txt + l_img, h]);
    blit_copy_f16(&txt.buffer, 0, &out.buffer, 0, l_txt * h);
    blit_copy_f16(&img.buffer, 0, &out.buffer, l_txt * h, l_img * h);
    out
}

// ============================================================================
// Double block
// ============================================================================

struct GpuDoubleBlock {
    img_mod: GpuLinear,
    img_qkv: GpuLinear,
    img_nq: Buffer,
    img_nk: Buffer,
    img_proj: GpuLinear,
    img_mlp0: GpuLinear,
    img_mlp2: GpuLinear,
    txt_mod: GpuLinear,
    txt_qkv: GpuLinear,
    txt_nq: Buffer,
    txt_nk: Buffer,
    txt_proj: GpuLinear,
    txt_mlp0: GpuLinear,
    txt_mlp2: GpuLinear,
    h: usize,
    nh: usize,
    dh: usize,
}

impl GpuDoubleBlock {
    fn from_cpu(b: &DoubleBlock) -> Self {
        Self {
            img_mod: GpuLinear::from_cpu(&b.img_mod.linear),
            img_qkv: GpuLinear::from_cpu(&b.img_attn_qkv),
            img_nq: upload_f32_as_f16_buffer(&b.img_attn_norm_q.scale),
            img_nk: upload_f32_as_f16_buffer(&b.img_attn_norm_k.scale),
            img_proj: GpuLinear::from_cpu(&b.img_attn_proj),
            img_mlp0: GpuLinear::from_cpu(&b.img_mlp_0),
            img_mlp2: GpuLinear::from_cpu(&b.img_mlp_2),
            txt_mod: GpuLinear::from_cpu(&b.txt_mod.linear),
            txt_qkv: GpuLinear::from_cpu(&b.txt_attn_qkv),
            txt_nq: upload_f32_as_f16_buffer(&b.txt_attn_norm_q.scale),
            txt_nk: upload_f32_as_f16_buffer(&b.txt_attn_norm_k.scale),
            txt_proj: GpuLinear::from_cpu(&b.txt_attn_proj),
            txt_mlp0: GpuLinear::from_cpu(&b.txt_mlp_0),
            txt_mlp2: GpuLinear::from_cpu(&b.txt_mlp_2),
            h: b.hidden,
            nh: b.num_heads,
            dh: b.head_dim,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &mut GpuTensor, l_img: usize,
        txt: &mut GpuTensor, l_txt: usize,
        vec_silu: &GpuTensor,
        cos: &Buffer, sin: &Buffer,
        gamma: &Buffer, zeros: &Buffer,
    ) {
        let (h, nh, dh) = (self.h, self.nh, self.dh);
        // Modulation chunks: 0 shift_msa, h scale_msa, 2h gate_msa,
        //                    3h shift_mlp, 4h scale_mlp, 5h gate_mlp.
        let im = self.img_mod.forward(vec_silu, 1);
        let tm = self.txt_mod.forward(vec_silu, 1);

        // ---- attention sub-block ----
        let mut in_norm = img.clone_data();
        layer_norm_no_affine(&mut in_norm, l_img, h, gamma, zeros);
        modulate_f16_gpu(&mut in_norm, &im.buffer, h, &im.buffer, 0, l_img, h);
        let mut tn_norm = txt.clone_data();
        layer_norm_no_affine(&mut tn_norm, l_txt, h, gamma, zeros);
        modulate_f16_gpu(&mut tn_norm, &tm.buffer, h, &tm.buffer, 0, l_txt, h);

        let img_qkv = self.img_qkv.forward(&in_norm, l_img);
        let txt_qkv = self.txt_qkv.forward(&tn_norm, l_txt);
        let (iq, ik, iv) = qkv_split_norm(&img_qkv, l_img, h, nh, dh, &self.img_nq, &self.img_nk);
        let (tq, tk, tv) = qkv_split_norm(&txt_qkv, l_txt, h, nh, dh, &self.txt_nq, &self.txt_nk);

        // Joint [txt, img] for q/k/v, RoPE on q/k, attention.
        let l_j = l_txt + l_img;
        let mut q = concat_tokens(&tq, l_txt, &iq, l_img, h);
        let mut k = concat_tokens(&tk, l_txt, &ik, l_img, h);
        let vv = concat_tokens(&tv, l_txt, &iv, l_img, h);
        rope_pairs_f16_gpu(&mut q, cos, sin, l_j, nh, dh);
        rope_pairs_f16_gpu(&mut k, cos, sin, l_j, nh, dh);
        let attn = joint_attention(&q, &k, &vv, l_j, nh, dh);

        // Split attn → (txt, img), proj, gate, residual.
        let mut txt_attn = GpuTensor::new_f16(vec![l_txt, h]);
        blit_copy_f16(&attn.buffer, 0, &txt_attn.buffer, 0, l_txt * h);
        let mut img_attn = GpuTensor::new_f16(vec![l_img, h]);
        blit_copy_f16(&attn.buffer, l_txt * h, &img_attn.buffer, 0, l_img * h);
        let _ = (&mut txt_attn, &mut img_attn);

        let mut img_p = self.img_proj.forward(&img_attn, l_img);
        gate_mul_f16_gpu(&mut img_p, &im.buffer, 2 * h, l_img, h);
        eadd_f16_gpu(img, &img_p);
        let mut txt_p = self.txt_proj.forward(&txt_attn, l_txt);
        gate_mul_f16_gpu(&mut txt_p, &tm.buffer, 2 * h, l_txt, h);
        eadd_f16_gpu(txt, &txt_p);

        // ---- MLP sub-block ----
        let mut in2 = img.clone_data();
        layer_norm_no_affine(&mut in2, l_img, h, gamma, zeros);
        modulate_f16_gpu(&mut in2, &im.buffer, 4 * h, &im.buffer, 3 * h, l_img, h);
        let mut ih = self.img_mlp0.forward(&in2, l_img);
        gelu_erf_f16_gpu(&mut ih);
        let mut io = self.img_mlp2.forward(&ih, l_img);
        gate_mul_f16_gpu(&mut io, &im.buffer, 5 * h, l_img, h);
        eadd_f16_gpu(img, &io);

        let mut tn2 = txt.clone_data();
        layer_norm_no_affine(&mut tn2, l_txt, h, gamma, zeros);
        modulate_f16_gpu(&mut tn2, &tm.buffer, 4 * h, &tm.buffer, 3 * h, l_txt, h);
        let mut th = self.txt_mlp0.forward(&tn2, l_txt);
        gelu_erf_f16_gpu(&mut th);
        let mut to = self.txt_mlp2.forward(&th, l_txt);
        gate_mul_f16_gpu(&mut to, &tm.buffer, 5 * h, l_txt, h);
        eadd_f16_gpu(txt, &to);
    }
}

// ============================================================================
// Single block
// ============================================================================

struct GpuSingleBlock {
    modulation: GpuLinear,
    linear1: GpuLinear,
    linear2: GpuLinear,
    nq: Buffer,
    nk: Buffer,
    h: usize,
    mlp_h: usize,
    nh: usize,
    dh: usize,
}

impl GpuSingleBlock {
    fn from_cpu(b: &SingleBlock) -> Self {
        Self {
            modulation: GpuLinear::from_cpu(&b.modulation.linear),
            linear1: GpuLinear::from_cpu(&b.linear1),
            linear2: GpuLinear::from_cpu(&b.linear2),
            nq: upload_f32_as_f16_buffer(&b.norm_q.scale),
            nk: upload_f32_as_f16_buffer(&b.norm_k.scale),
            h: b.hidden,
            mlp_h: b.mlp_h,
            nh: b.num_heads,
            dh: b.head_dim,
        }
    }

    fn forward(
        &self,
        x: &mut GpuTensor, l: usize,
        vec_silu: &GpuTensor,
        cos: &Buffer, sin: &Buffer,
        gamma: &Buffer, zeros: &Buffer,
    ) {
        let (h, mlp_h, nh, dh) = (self.h, self.mlp_h, self.nh, self.dh);
        // chunks: 0 shift, h scale, 2h gate.
        let m = self.modulation.forward(vec_silu, 1);

        let mut xn = x.clone_data();
        layer_norm_no_affine(&mut xn, l, h, gamma, zeros);
        modulate_f16_gpu(&mut xn, &m.buffer, h, &m.buffer, 0, l, h);

        // linear1 → [l, 3h + mlp_h]; split q/k/v (each h) + mlp_in (mlp_h).
        let fused = self.linear1.forward(&xn, l);
        let stride = 3 * h + mlp_h;
        let mut q = slice_cols_f16_gpu(&fused, l, stride, 0, h);
        let mut k = slice_cols_f16_gpu(&fused, l, stride, h, h);
        let v = slice_cols_f16_gpu(&fused, l, stride, 2 * h, h);
        let mut mlp_in = slice_cols_f16_gpu(&fused, l, stride, 3 * h, mlp_h);

        rms_norm_f16_gpu(&mut q, &self.nq, l * nh, dh, EPS);
        rms_norm_f16_gpu(&mut k, &self.nk, l * nh, dh, EPS);
        rope_pairs_f16_gpu(&mut q, cos, sin, l, nh, dh);
        rope_pairs_f16_gpu(&mut k, cos, sin, l, nh, dh);
        let attn = joint_attention(&q, &k, &v, l, nh, dh); // [l, h]

        gelu_erf_f16_gpu(&mut mlp_in); // [l, mlp_h]

        // linear2 input = [attn | mlp_act] concat along feature axis → [l, h+mlp_h].
        let combined = GpuTensor::new_f16(vec![l, h + mlp_h]);
        for t in 0..l {
            blit_copy_f16(&attn.buffer, t * h, &combined.buffer, t * (h + mlp_h), h);
            blit_copy_f16(&mlp_in.buffer, t * mlp_h, &combined.buffer, t * (h + mlp_h) + h, mlp_h);
        }
        let mut update = self.linear2.forward(&combined, l);
        gate_mul_f16_gpu(&mut update, &m.buffer, 2 * h, l, h);
        eadd_f16_gpu(x, &update);
    }
}

// ============================================================================
// Final layer
// ============================================================================

struct GpuFinalLayer {
    ada_ln: GpuLinear, // h → 2h
    linear: GpuLinear, // h → p²·out_c
    h: usize,
}

impl GpuFinalLayer {
    fn from_cpu(f: &FinalLayer) -> Self {
        Self {
            ada_ln: GpuLinear::from_cpu(&f.ada_ln_modulation),
            linear: GpuLinear::from_cpu(&f.linear),
            h: f.ada_ln_modulation.in_features,
        }
    }

    fn forward(&self, img: &GpuTensor, l_img: usize, vec_silu: &GpuTensor, gamma: &Buffer, zeros: &Buffer) -> GpuTensor {
        let h = self.h;
        let m = self.ada_ln.forward(vec_silu, 1); // [1, 2h]: shift@0, scale@h
        let mut x = img.clone_data();
        layer_norm_no_affine(&mut x, l_img, h, gamma, zeros);
        modulate_f16_gpu(&mut x, &m.buffer, h, &m.buffer, 0, l_img, h);
        self.linear.forward(&x, l_img)
    }
}

// ============================================================================
// Top-level GPU transformer
// ============================================================================

pub struct GpuFluxTransformer {
    cfg: FluxConfig,
    img_in: GpuLinear,
    txt_in: GpuLinear,
    time_in: MLPEmbedder,
    vector_in: MLPEmbedder,
    guidance_in: Option<MLPEmbedder>,
    double_blocks: Vec<GpuDoubleBlock>,
    single_blocks: Vec<GpuSingleBlock>,
    final_layer: GpuFinalLayer,
    ln_gamma: Buffer, // ones[hidden]
    ln_zeros: Buffer, // zeros[hidden]
}

impl GpuFluxTransformer {
    /// Build from a (CPU-loaded) [`FluxTransformer`], uploading all weights to
    /// GPU bf16 buffers. The small time/vector/guidance embedders stay on CPU
    /// (one tiny matmul per forward).
    pub fn from_cpu(m: &FluxTransformer) -> Self {
        let h = m.cfg.hidden_size;
        Self {
            cfg: m.cfg.clone(),
            img_in: GpuLinear::from_cpu(&m.img_in),
            txt_in: GpuLinear::from_cpu(&m.txt_in),
            time_in: m.time_in.clone(),
            vector_in: m.vector_in.clone(),
            guidance_in: m.guidance_in.clone(),
            double_blocks: m.double_blocks.iter().map(GpuDoubleBlock::from_cpu).collect(),
            single_blocks: m.single_blocks.iter().map(GpuSingleBlock::from_cpu).collect(),
            final_layer: GpuFinalLayer::from_cpu(&m.final_layer),
            ln_gamma: upload_f32_as_f16_buffer(&vec![1.0_f32; h]),
            ln_zeros: upload_f32_as_f16_buffer(&vec![0.0_f32; h]),
        }
    }

    /// Load weights directly to GPU from a checkpoint component, block by
    /// block, dropping each CPU block's f32 after converting it to bf16. Peak
    /// RAM is one block (~1.4 GB), not the whole 12B-param model in f32 — so
    /// this is the production path (`from_cpu` would need the full f32 model).
    pub fn load(cfg: FluxConfig, comp: &crate::weight::loader::ComponentTensors) -> crate::Result<Self> {
        let h = cfg.hidden_size;
        let load_lin = |prefix: &str, in_f: usize, out_f: usize, bias: bool| -> crate::Result<GpuLinear> {
            let mut l = Linear::new(in_f, out_f, bias);
            l.load(comp, prefix)?;
            Ok(GpuLinear::from_cpu(&l))
        };
        let img_in = load_lin("img_in", cfg.in_channels * cfg.patch_size * cfg.patch_size, h, true)?;
        let txt_in = load_lin("txt_in", cfg.t5_dim, h, true)?;
        let mut time_in = MLPEmbedder::new(cfg.time_emb_dim, h);
        time_in.load(comp, "time_in")?;
        let mut vector_in = MLPEmbedder::new(cfg.pooled_clip_dim, h);
        vector_in.load(comp, "vector_in")?;
        let guidance_in = if cfg.guidance_embeds {
            let mut g = MLPEmbedder::new(cfg.time_emb_dim, h);
            g.load(comp, "guidance_in")?;
            Some(g)
        } else {
            None
        };
        let mut double_blocks = Vec::with_capacity(cfg.num_double_blocks);
        for i in 0..cfg.num_double_blocks {
            let mut b = DoubleBlock::new(&cfg);
            b.load(comp, i)?;
            double_blocks.push(GpuDoubleBlock::from_cpu(&b)); // b dropped here
        }
        let mut single_blocks = Vec::with_capacity(cfg.num_single_blocks);
        for i in 0..cfg.num_single_blocks {
            let mut b = SingleBlock::new(&cfg);
            b.load(comp, i)?;
            single_blocks.push(GpuSingleBlock::from_cpu(&b));
        }
        let mut fl = FinalLayer::new(&cfg);
        fl.load(comp)?;
        Ok(Self {
            cfg: cfg.clone(),
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            double_blocks,
            single_blocks,
            final_layer: GpuFinalLayer::from_cpu(&fl),
            ln_gamma: upload_f32_as_f16_buffer(&vec![1.0_f32; h]),
            ln_zeros: upload_f32_as_f16_buffer(&vec![0.0_f32; h]),
        })
    }

    /// One forward pass — same signature/semantics as the CPU
    /// [`FluxTransformer::forward`]. Returns `[out_c, H, W]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        latent: &[f32], latent_h: usize, latent_w: usize,
        t5_embed: &[f32], l_txt: usize,
        pooled_clip: &[f32],
        timestep: f32, guidance: f32,
    ) -> Vec<f32> {
        // Flux activations are large; force the f32-accumulate sgemm scratch.
        set_use_f32_sgemm(true);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let p = cfg.patch_size;
        let (h_p, w_p) = (latent_h / p, latent_w / p);
        let l_img = h_p * w_p;
        let in_c = cfg.in_channels;
        let patch_dim = in_c * p * p;

        // ---- patchify (CPU) → img_in (GPU) ----
        let mut img_tokens = vec![0.0_f32; l_img * patch_dim];
        for hi in 0..h_p {
            for wi in 0..w_p {
                let toff = (hi * w_p + wi) * patch_dim;
                let mut kk = 0;
                for c in 0..in_c {
                    for dy in 0..p {
                        for dx in 0..p {
                            let li = c * latent_h * latent_w + (hi * p + dy) * latent_w + (wi * p + dx);
                            img_tokens[toff + kk] = latent[li];
                            kk += 1;
                        }
                    }
                }
            }
        }
        let img_tok_gpu = GpuTensor::upload_f32_as_f16(vec![l_img, patch_dim], &img_tokens);
        let mut img = self.img_in.forward(&img_tok_gpu, l_img);
        let t5_gpu = GpuTensor::upload_f32_as_f16(vec![l_txt, cfg.t5_dim], t5_embed);
        let mut txt = self.txt_in.forward(&t5_gpu, l_txt);

        // ---- conditioning vec (CPU) ----
        let t_emb = timestep_embedding(timestep, cfg.time_emb_dim, 10_000.0);
        let mut vec_cond = self.time_in.forward(&t_emb);
        let pooled_emb = self.vector_in.forward(pooled_clip);
        for (a, b) in vec_cond.iter_mut().zip(pooled_emb.iter()) { *a += b; }
        if let Some(gi) = &self.guidance_in {
            let g_emb = timestep_embedding(guidance * 1000.0, cfg.time_emb_dim, 10_000.0);
            let g = gi.forward(&g_emb);
            for (a, b) in vec_cond.iter_mut().zip(g.iter()) { *a += b; }
        }
        for v in vec_cond.iter_mut() { *v /= 1.0 + (-*v).exp(); } // silu
        let vec_silu = GpuTensor::upload_f32_as_f16(vec![1, h], &vec_cond);

        // ---- RoPE tables (CPU build → GPU) ----
        let mut seq_ids: Vec<(u32, u32, u32)> = Vec::with_capacity(l_txt + l_img);
        for _ in 0..l_txt { seq_ids.push((0, 0, 0)); }
        for hi in 0..h_p {
            for wi in 0..w_p { seq_ids.push((0, hi as u32, wi as u32)); }
        }
        let (cos, sin) = build_rope_2d(&seq_ids, cfg.head_dim, cfg.axes_dim, cfg.rope_theta);
        let cos_buf = upload_f32_as_f16_buffer(&cos);
        let sin_buf = upload_f32_as_f16_buffer(&sin);

        // ---- double blocks ----
        for blk in &self.double_blocks {
            blk.forward(&mut img, l_img, &mut txt, l_txt, &vec_silu, &cos_buf, &sin_buf, &self.ln_gamma, &self.ln_zeros);
        }

        // ---- concat → single blocks ----
        let l_total = l_txt + l_img;
        let mut joint = concat_tokens(&txt, l_txt, &img, l_img, h);
        for blk in &self.single_blocks {
            blk.forward(&mut joint, l_total, &vec_silu, &cos_buf, &sin_buf, &self.ln_gamma, &self.ln_zeros);
        }

        // ---- final layer (img tokens only) ----
        let mut img_final = GpuTensor::new_f16(vec![l_img, h]);
        blit_copy_f16(&joint.buffer, l_txt * h, &img_final.buffer, 0, l_img * h);
        let _ = &mut img_final;
        let img_out = self.final_layer.forward(&img_final, l_img, &vec_silu, &self.ln_gamma, &self.ln_zeros).download_to_f32();

        // ---- unpatchify (CPU) ----
        let out_c = cfg.out_channels;
        let p2c = out_c * p * p;
        let mut out = vec![0.0_f32; out_c * latent_h * latent_w];
        for hi in 0..h_p {
            for wi in 0..w_p {
                let tidx = hi * w_p + wi;
                let mut kk = 0;
                for c in 0..out_c {
                    for dy in 0..p {
                        for dx in 0..p {
                            let li = c * latent_h * latent_w + (hi * p + dy) * latent_w + (wi * p + dx);
                            out[li] = img_out[tidx * p2c + kk];
                            kk += 1;
                        }
                    }
                }
            }
        }
        out
    }
}

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
            num_double_blocks: 2,
            num_single_blocks: 2,
            patch_size: 2,
            time_emb_dim: 8,
            pooled_clip_dim: 8,
            t5_dim: 16,
            guidance_embeds: false,
            rope_theta: 100.0,
            axes_dim: (2, 3, 3),
        }
    }

    fn fill(weight: &mut [f32], seed: f32) {
        for (i, w) in weight.iter_mut().enumerate() {
            *w = ((i as f32 * seed).sin()) * 0.05;
        }
    }

    fn randomize(m: &mut FluxTransformer) {
        let mut s = 0.11_f32;
        let mut next = || { s += 0.013; s };
        let mut do_lin = |l: &mut Linear| {
            fill(&mut l.weight, next());
            if let Some(b) = l.bias.as_mut() { fill(b, next()); }
        };
        do_lin(&mut m.img_in);
        do_lin(&mut m.txt_in);
        for e in [&mut m.time_in, &mut m.vector_in] {
            do_lin(&mut e.in_layer);
            do_lin(&mut e.out_layer);
        }
        for blk in m.double_blocks.iter_mut() {
            do_lin(&mut blk.img_mod.linear); do_lin(&mut blk.img_attn_qkv);
            do_lin(&mut blk.img_attn_proj); do_lin(&mut blk.img_mlp_0); do_lin(&mut blk.img_mlp_2);
            do_lin(&mut blk.txt_mod.linear); do_lin(&mut blk.txt_attn_qkv);
            do_lin(&mut blk.txt_attn_proj); do_lin(&mut blk.txt_mlp_0); do_lin(&mut blk.txt_mlp_2);
            for nq in [&mut blk.img_attn_norm_q.scale, &mut blk.img_attn_norm_k.scale,
                       &mut blk.txt_attn_norm_q.scale, &mut blk.txt_attn_norm_k.scale] {
                for v in nq.iter_mut() { *v = 1.0; }
            }
        }
        for blk in m.single_blocks.iter_mut() {
            do_lin(&mut blk.modulation.linear); do_lin(&mut blk.linear1); do_lin(&mut blk.linear2);
            for nq in [&mut blk.norm_q.scale, &mut blk.norm_k.scale] {
                for v in nq.iter_mut() { *v = 1.0; }
            }
        }
        fill(&mut m.final_layer.ada_ln_modulation.weight, next());
        if let Some(b) = m.final_layer.ada_ln_modulation.bias.as_mut() { fill(b, next()); }
        fill(&mut m.final_layer.linear.weight, next());
        if let Some(b) = m.final_layer.linear.bias.as_mut() { fill(b, next()); }
    }

    /// Loads the real flux_dev transformer to GPU and runs one forward at a
    /// modest resolution — validates the production `load()` (12B params → GPU
    /// bf16, prefix correctness) and the forward at real scale (finite, no OOM).
    /// Gated on `KLEARU_FLUX_REAL=1` + the checkpoint being present. Run with
    /// `--release`.
    #[test]
    fn real_flux_load_and_forward() {
        if std::env::var("KLEARU_FLUX_REAL").is_err() {
            eprintln!("SKIP real_flux_load_and_forward (set KLEARU_FLUX_REAL=1)");
            return;
        }
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/flux_dev.safetensors");
        if !path.exists() {
            eprintln!("SKIP: {} not present", path.display());
            return;
        }
        use crate::weight::flux_checkpoint::transformer_component_from_loader;
        use crate::weight::SingleFileLoader;
        let loader = SingleFileLoader::open(&path).expect("open flux bundle");
        let comp = transformer_component_from_loader(&loader).expect("transformer comp");
        let cfg = FluxConfig::flux_dev();
        let t0 = std::time::Instant::now();
        let gpu = GpuFluxTransformer::load(cfg.clone(), &comp).expect("load to GPU");
        eprintln!("[flux-real] loaded 12B transformer to GPU in {:.1}s", t0.elapsed().as_secs_f32());
        drop(comp);
        drop(loader);

        // 512² → latent 64×64; l_txt = 512 (real T5 length).
        let (lh, lw) = (64usize, 64usize);
        let l_txt = 512usize;
        let latent: Vec<f32> = (0..cfg.in_channels * lh * lw).map(|i| (i as f32 * 0.001).sin()).collect();
        let t5: Vec<f32> = (0..l_txt * cfg.t5_dim).map(|i| (i as f32 * 0.0007).cos() * 0.1).collect();
        let pooled: Vec<f32> = (0..cfg.pooled_clip_dim).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
        let f0 = std::time::Instant::now();
        let out = gpu.forward(&latent, lh, lw, &t5, l_txt, &pooled, 0.8, 3.5);
        let nan = out.iter().filter(|v| !v.is_finite()).count();
        eprintln!("[flux-real] one forward in {:.1}s, out.len={}, nonfinite={nan}", f0.elapsed().as_secs_f32(), out.len());
        assert_eq!(out.len(), cfg.out_channels * lh * lw);
        assert_eq!(nan, 0, "real Flux GPU forward produced NaN/Inf");
    }

    #[test]
    fn gpu_flux_matches_cpu() {
        let cfg = tiny_cfg();
        let mut cpu = FluxTransformer::from_config(cfg.clone());
        randomize(&mut cpu);
        let gpu = GpuFluxTransformer::from_cpu(&cpu);

        let (lh, lw) = (4usize, 4usize);
        let l_txt = 3usize;
        let latent: Vec<f32> = (0..cfg.in_channels * lh * lw).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let t5: Vec<f32> = (0..l_txt * cfg.t5_dim).map(|i| (i as f32 * 0.07).cos() * 0.2).collect();
        let pooled: Vec<f32> = (0..cfg.pooled_clip_dim).map(|i| (i as f32 * 0.1).sin() * 0.2).collect();

        let cpu_out = cpu.forward(&latent, lh, lw, &t5, l_txt, &pooled, 0.5, 1.0);
        let gpu_out = gpu.forward(&latent, lh, lw, &t5, l_txt, &pooled, 0.5, 1.0);
        assert_eq!(cpu_out.len(), gpu_out.len());
        let mut dot = 0.0f64; let mut nc = 0.0f64; let mut ng = 0.0f64;
        for (c, g) in cpu_out.iter().zip(gpu_out.iter()) {
            dot += *c as f64 * *g as f64; nc += (*c as f64).powi(2); ng += (*g as f64).powi(2);
        }
        let cos = dot / (nc.sqrt() * ng.sqrt() + 1e-12);
        println!("gpu_flux cos={cos:.6}");
        assert!(cos > 0.999, "gpu/cpu Flux transformer cosine too low: {cos}");
    }
}
