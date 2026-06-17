//! Metal GPU path for the HiDream-O1 Qwen3-VL decoder.
//!
//! The CPU modules in [`super::model`] are the correctness oracle; this module
//! is the performance path used for real 8B inference. Weights upload once as
//! persistent bf16 buffers; the hot matmuls run via `sgemm_f16_*` under the
//! **f32-scratch sgemm path** (`set_use_f32_sgemm`) and the gen-token attention
//! via `flash_attention_f16_gpu`.
//!
//! CRITICAL precision note: Qwen3-VL has "massive activations" exceeding the
//! IEEE-f16 max of 65504. The default matmul casts bf16→f16 (output capped at
//! ±65504) → Inf/NaN; we force the f32 sgemm scratch path. Likewise the MPSGraph
//! SDPA casts to f16 and overflows — so attention uses `flash_attention_f16_gpu`
//! instead, which reads bf16 (range 3e38) and accumulates the online softmax in
//! f32. Gen tokens (`<tms>`+image) attend bidirectionally (no mask); the small
//! causal text prefix stays on the CPU.
//!
//! Gated behind `feature = "metal"`.
#![cfg(feature = "metal")]

use crate::hidream::config::TextConfig;
use crate::hidream::layers::Linear;
use crate::hidream::model::{Attention, DecoderLayer, Mlp};
use crate::hidream::mrope;
use crate::metal_backend::{
    bias_add_f16_gpu, eadd_f16_gpu, emul_f16_gpu, gqa_repeat_f16_gpu, mpsgraph_sdpa_bf16_gpu,
    permute_hl_to_lh_f16_gpu, permute_lh_to_hl_f16_gpu, rms_norm_f16_gpu, rope_apply_f16_gpu,
    sgemm_f16_a_btrans_buf, silu_f16_gpu, upload_f32_as_f16_buffer, GpuTensor,
};
use crate::weight::loader::ComponentTensors;
use crate::Result;
use metal::Buffer;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Profiling: when `HIDREAM_PROF` is set, print per-section timings for the
/// first few layer invocations of a run.
static PROF_COUNT: AtomicUsize = AtomicUsize::new(0);
fn prof_on() -> bool {
    std::env::var("HIDREAM_PROF").is_ok() && PROF_COUNT.load(Ordering::Relaxed) < 6
}

/// Read the first `count` bf16 elements of a GPU tensor into f32 (shared
/// memory; flushes pending GPU work first). Used for the small text-prefix
/// causal fixup without downloading the whole tensor.
fn download_prefix(t: &GpuTensor, count: usize) -> Vec<f32> {
    use half::bf16;
    use half::slice::HalfFloatSliceExt;
    crate::metal_backend::flush();
    let mut out = vec![0.0_f32; count];
    unsafe {
        let p = t.buffer.contents() as *const bf16;
        let s = std::slice::from_raw_parts(p, count);
        s.convert_to_f32_slice(&mut out);
    }
    out
}

/// Overwrite the first `data.len()` bf16 elements of a GPU tensor from f32.
fn write_prefix(t: &GpuTensor, data: &[f32]) {
    use half::bf16;
    use half::slice::HalfFloatSliceExt;
    crate::metal_backend::flush();
    unsafe {
        let p = t.buffer.contents() as *mut bf16;
        let s = std::slice::from_raw_parts_mut(p, data.len());
        s.convert_from_f32_slice(data);
    }
}

/// A linear layer whose weight is converted to a **persistent bf16 GPU buffer**
/// once at construction (the f32 source is dropped). This keeps the resident
/// footprint at bf16 (~16 GB for the 8B model) instead of holding 32 GB of f32
/// masters. Weight is `[out_features, in_features]` row-major.
pub struct GpuLinear {
    pub weight_buf: Buffer,
    pub bias_buf: Option<Buffer>,
    pub in_features: usize,
    pub out_features: usize,
}

impl GpuLinear {
    pub fn from_f32(weight: &[f32], bias: Option<&[f32]>, in_f: usize, out_f: usize) -> Self {
        debug_assert_eq!(weight.len(), in_f * out_f);
        Self {
            weight_buf: upload_f32_as_f16_buffer(weight),
            bias_buf: bias.map(upload_f32_as_f16_buffer),
            in_features: in_f,
            out_features: out_f,
        }
    }

    pub fn from_cpu(l: &super::layers::Linear) -> Self {
        Self::from_f32(&l.weight, l.bias.as_deref(), l.in_features, l.out_features)
    }

    /// Load `{prefix}.weight` (+ optional `{prefix}.bias`) straight into bf16
    /// GPU buffers; the transient f32 is freed before the next weight loads.
    pub fn load(comp: &ComponentTensors, prefix: &str, in_f: usize, out_f: usize, bias: bool) -> Result<Self> {
        let w = comp.get_f32(&format!("{prefix}.weight"))?;
        let b = if bias { Some(comp.get_f32(&format!("{prefix}.bias"))?) } else { None };
        Ok(Self::from_f32(&w, b.as_deref(), in_f, out_f))
    }

    /// `out[m, out_f] = x[m, in_f] · weightᵀ (+ bias)`, all GPU-resident.
    pub fn forward(&self, x: &GpuTensor, m: usize) -> GpuTensor {
        debug_assert_eq!(x.elements(), m * self.in_features);
        let mut out = GpuTensor::new_f16(vec![m, self.out_features]);
        sgemm_f16_a_btrans_buf(&out.buffer, &x.buffer, &self.weight_buf, m, self.out_features, self.in_features);
        if let Some(b) = &self.bias_buf {
            bias_add_f16_gpu(&mut out, b, 1, self.out_features);
        }
        out
    }
}

/// GPU grouped-query attention — fully GPU-resident. q/k/v projections,
/// per-head q/k RMSNorm, RoPE, head-major permute + GQA repeat, and the SDPA
/// all run on-GPU with no intermediate downloads. Only the small causal
/// text-prefix correction touches the CPU (the bidirectional SDPA over all
/// tokens is correct for the gen tokens; the few text rows are then overwritten
/// with their causal result).
pub struct GpuAttention {
    pub q_proj: GpuLinear,
    pub k_proj: GpuLinear,
    pub v_proj: GpuLinear,
    pub o_proj: GpuLinear,
    pub q_norm_buf: Buffer,
    pub k_norm_buf: Buffer,
    pub rms_eps: f32,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GpuAttention {
    pub fn from_cpu(a: &Attention) -> Self {
        Self {
            q_proj: GpuLinear::from_cpu(&a.q_proj),
            k_proj: GpuLinear::from_cpu(&a.k_proj),
            v_proj: GpuLinear::from_cpu(&a.v_proj),
            o_proj: GpuLinear::from_cpu(&a.o_proj),
            q_norm_buf: upload_f32_as_f16_buffer(&a.q_norm.weight),
            k_norm_buf: upload_f32_as_f16_buffer(&a.k_norm.weight),
            rms_eps: a.q_norm.eps,
            num_heads: a.num_heads,
            num_kv_heads: a.num_kv_heads,
            head_dim: a.head_dim,
        }
    }

    pub fn load(comp: &ComponentTensors, prefix: &str, cfg: &TextConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let dh = cfg.head_dim();
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let bias = cfg.attention_bias;
        let qn = comp.get_f32(&format!("{prefix}.q_norm.weight"))?;
        let kn = comp.get_f32(&format!("{prefix}.k_norm.weight"))?;
        Ok(Self {
            q_proj: GpuLinear::load(comp, &format!("{prefix}.q_proj"), h, nh * dh, bias)?,
            k_proj: GpuLinear::load(comp, &format!("{prefix}.k_proj"), h, nkv * dh, bias)?,
            v_proj: GpuLinear::load(comp, &format!("{prefix}.v_proj"), h, nkv * dh, bias)?,
            o_proj: GpuLinear::load(comp, &format!("{prefix}.o_proj"), nh * dh, h, bias)?,
            q_norm_buf: upload_f32_as_f16_buffer(&qn),
            k_norm_buf: upload_f32_as_f16_buffer(&kn),
            rms_eps: cfg.rms_norm_eps,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: dh,
        })
    }

    /// `normed`: GPU tensor [n, hidden] (already input-layernorm'd). `cos`/`sin`
    /// are the RoPE tables as GPU buffers [n, head_dim]. `txt_len` is the text
    /// prefix length (the last text token is the bidirectional `<tms>`; rows
    /// `0..txt_len-1` are the causal text rows fixed up on CPU). Returns the
    /// attention output [n, hidden] as a GPU tensor.
    pub fn forward(
        &self,
        normed: &GpuTensor,
        n: usize,
        cos: &Buffer,
        sin: &Buffer,
        txt_len: usize,
    ) -> GpuTensor {
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let dh = self.head_dim;
        let n_rep = nh / nkv;
        let scale = 1.0_f32 / (dh as f32).sqrt();
        let prof = prof_on();
        let t = std::time::Instant::now();
        macro_rules! lap {
            ($label:expr) => {
                if prof {
                    eprintln!("    [prof] {:<14} {:>7.1}ms", $label, t.elapsed().as_secs_f64() * 1000.0);
                }
            };
        }

        // 1. Projections (GPU-resident).
        let mut q = self.q_proj.forward(normed, n); // [n, nh*dh]
        let mut k = self.k_proj.forward(normed, n); // [n, nkv*dh]
        let v = self.v_proj.forward(normed, n); // [n, nkv*dh]
        // 2. Per-head q/k RMSNorm + RoPE on GPU.
        rms_norm_f16_gpu(&mut q, &self.q_norm_buf, n * nh, dh, self.rms_eps);
        rms_norm_f16_gpu(&mut k, &self.k_norm_buf, n * nkv, dh, self.rms_eps);
        rope_apply_f16_gpu(&mut q, cos, sin, n, nh, dh);
        rope_apply_f16_gpu(&mut k, cos, sin, n, nkv, dh);
        lap!("qkv+norm+rope");

        // 3. Head-major permute + GQA repeat, then full (unmasked) SDPA.
        let q_hl = permute_lh_to_hl_f16_gpu(&q, 1, n, nh, dh); // [nh, n, dh]
        let k_hl0 = permute_lh_to_hl_f16_gpu(&k, 1, n, nkv, dh); // [nkv, n, dh]
        let v_hl0 = permute_lh_to_hl_f16_gpu(&v, 1, n, nkv, dh);
        let mut k_hl = GpuTensor::new_f16(vec![nh * n * dh]);
        let mut v_hl = GpuTensor::new_f16(vec![nh * n * dh]);
        gqa_repeat_f16_gpu(&k_hl0, &mut k_hl, n_rep, n * dh);
        gqa_repeat_f16_gpu(&v_hl0, &mut v_hl, n_rep, n * dh);
        let mut o_hl = GpuTensor::new_f16(vec![nh * n * dh]);
        mpsgraph_sdpa_bf16_gpu(&q_hl, &k_hl, &v_hl, &mut o_hl, 1, nh, n, n, dh, scale);
        let out = permute_hl_to_lh_f16_gpu(&o_hl, 1, n, nh, dh); // [n, nh*dh]
        lap!("attn");

        // 4. Text-causal fixup (rows 0..txt_len-1; the `<tms>` row txt_len-1 is
        //    bidirectional, handled correctly by the full SDPA above).
        let n_text = txt_len.saturating_sub(1);
        if n_text > 0 {
            let qf = download_prefix(&q, n_text * nh * dh);
            let kf = download_prefix(&k, n_text * nkv * dh);
            let vf = download_prefix(&v, n_text * nkv * dh);
            let mut o_text = vec![0.0_f32; n_text * nh * dh];
            for qi in 0..n_text {
                let last = qi + 1;
                for h in 0..nh {
                    let kv_h = h / n_rep;
                    let q_off = (qi * nh + h) * dh;
                    let mut scores = vec![0.0_f32; last];
                    let mut max = f32::NEG_INFINITY;
                    for (ki, sc) in scores.iter_mut().enumerate() {
                        let k_off = (ki * nkv + kv_h) * dh;
                        let mut s = 0.0_f32;
                        for d in 0..dh {
                            s += qf[q_off + d] * kf[k_off + d];
                        }
                        s *= scale;
                        *sc = s;
                        if s > max {
                            max = s;
                        }
                    }
                    let mut sum = 0.0_f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    let o_off = (qi * nh + h) * dh;
                    for (ki, &w) in scores.iter().enumerate() {
                        let w = w * inv;
                        let v_off = (ki * nkv + kv_h) * dh;
                        for d in 0..dh {
                            o_text[o_off + d] += w * vf[v_off + d];
                        }
                    }
                }
            }
            write_prefix(&out, &o_text);
        }
        lap!("text_attn");

        // Re-open the batch (the text-fixup download/write flushed it) so o_proj
        // + the MLP cluster coalesce until the next layer's SDPA flush.
        crate::metal_backend::begin_batch();
        // 5. Output projection (GPU).
        let result = self.o_proj.forward(&out, n);
        lap!("o_proj");
        if prof {
            PROF_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}

/// GPU SwiGLU MLP. The three matmuls run on the GPU; the silu·gate elementwise
/// step stays on the CPU (no GPU elementwise-mul kernel).
pub struct GpuMlp {
    pub gate_proj: GpuLinear,
    pub up_proj: GpuLinear,
    pub down_proj: GpuLinear,
    pub intermediate: usize,
}

impl GpuMlp {
    pub fn from_cpu(m: &Mlp) -> Self {
        Self {
            gate_proj: GpuLinear::from_cpu(&m.gate_proj),
            up_proj: GpuLinear::from_cpu(&m.up_proj),
            down_proj: GpuLinear::from_cpu(&m.down_proj),
            intermediate: m.gate_proj.out_features,
        }
    }

    pub fn load(comp: &ComponentTensors, prefix: &str, hidden: usize, inter: usize) -> Result<Self> {
        Ok(Self {
            gate_proj: GpuLinear::load(comp, &format!("{prefix}.gate_proj"), hidden, inter, false)?,
            up_proj: GpuLinear::load(comp, &format!("{prefix}.up_proj"), hidden, inter, false)?,
            down_proj: GpuLinear::load(comp, &format!("{prefix}.down_proj"), inter, hidden, false)?,
            intermediate: inter,
        })
    }

    /// SwiGLU, fully GPU-resident: `down(silu(gate(x)) * up(x))`.
    pub fn forward(&self, x: &GpuTensor, n: usize) -> GpuTensor {
        let mut gate = self.gate_proj.forward(x, n); // [n, inter]
        let up = self.up_proj.forward(x, n);
        silu_f16_gpu(&mut gate);
        emul_f16_gpu(&mut gate, &up);
        self.down_proj.forward(&gate, n) // [n, hidden]
    }
}

/// GPU decoder layer — fully GPU-resident (RMSNorm, attention, MLP, residuals
/// all on-GPU; no per-op downloads).
pub struct GpuDecoderLayer {
    pub input_ln_buf: Buffer,
    pub post_ln_buf: Buffer,
    pub rms_eps: f32,
    pub hidden: usize,
    pub self_attn: GpuAttention,
    pub mlp: GpuMlp,
}

impl GpuDecoderLayer {
    pub fn from_cpu(l: &DecoderLayer) -> Self {
        Self {
            input_ln_buf: upload_f32_as_f16_buffer(&l.input_layernorm.weight),
            post_ln_buf: upload_f32_as_f16_buffer(&l.post_attention_layernorm.weight),
            rms_eps: l.input_layernorm.eps,
            hidden: l.input_layernorm.weight.len(),
            self_attn: GpuAttention::from_cpu(&l.self_attn),
            mlp: GpuMlp::from_cpu(&l.mlp),
        }
    }

    pub fn load(comp: &ComponentTensors, prefix: &str, cfg: &TextConfig) -> Result<Self> {
        let iln = comp.get_f32(&format!("{prefix}.input_layernorm.weight"))?;
        let pln = comp.get_f32(&format!("{prefix}.post_attention_layernorm.weight"))?;
        Ok(Self {
            input_ln_buf: upload_f32_as_f16_buffer(&iln),
            post_ln_buf: upload_f32_as_f16_buffer(&pln),
            rms_eps: cfg.rms_norm_eps,
            hidden: cfg.hidden_size,
            self_attn: GpuAttention::load(comp, &format!("{prefix}.self_attn"), cfg)?,
            mlp: GpuMlp::load(comp, &format!("{prefix}.mlp"), cfg.hidden_size, cfg.intermediate_size)?,
        })
    }

    /// `hidden`: GPU tensor [n, hidden], updated in place (residual stream).
    pub fn forward(&self, hidden: &mut GpuTensor, n: usize, cos: &Buffer, sin: &Buffer, txt_len: usize) {
        // Batch the elementwise/norm GPU ops into shared command buffers. (The
        // matmuls and SDPA manage their own scratch/commits; this coalesces the
        // many small kernel dispatches between sync points.)
        crate::metal_backend::begin_batch();
        // Attention sub-block.
        let mut normed = hidden.clone_data();
        rms_norm_f16_gpu(&mut normed, &self.input_ln_buf, n, self.hidden, self.rms_eps);
        let attn = self.self_attn.forward(&normed, n, cos, sin, txt_len);
        eadd_f16_gpu(hidden, &attn);
        // MLP sub-block.
        let mut normed2 = hidden.clone_data();
        rms_norm_f16_gpu(&mut normed2, &self.post_ln_buf, n, self.hidden, self.rms_eps);
        let mlp = self.mlp.forward(&normed2, n);
        eadd_f16_gpu(hidden, &mlp);
        // No explicit flush: queue ordering keeps the residual correct for the
        // next layer (only CPU reads/writes need a barrier). The batch stays
        // open and is committed at the next SDPA's internal flush.
    }
}

/// GPU bottleneck patch embed (`x_embedder`): two GPU matmuls.
pub struct GpuPatchEmbed {
    pub proj1: GpuLinear,
    pub proj2: GpuLinear,
}

impl GpuPatchEmbed {
    pub fn from_cpu(p: &super::layers::BottleneckPatchEmbed) -> Self {
        Self { proj1: GpuLinear::from_cpu(&p.proj1), proj2: GpuLinear::from_cpu(&p.proj2) }
    }

    pub fn load(comp: &ComponentTensors, prefix: &str, patch_dim: usize, bottleneck: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            proj1: GpuLinear::load(comp, &format!("{prefix}.proj1"), patch_dim, bottleneck, false)?,
            proj2: GpuLinear::load(comp, &format!("{prefix}.proj2"), bottleneck, hidden, true)?,
        })
    }

    /// `z`: [n, PATCH_DIM] → [n, hidden] (CPU in/out, matmuls on GPU).
    pub fn forward(&self, z: &[f32], n: usize) -> Vec<f32> {
        let zg = GpuTensor::upload_f32_as_f16(vec![n, self.proj1.in_features], z);
        let mid = self.proj1.forward(&zg, n);
        self.proj2.forward(&mid, n).download_to_f32()
    }
}

/// Full GPU HiDream-O1 model: the 36 decoder layers + patch embed + final
/// unembed run on the GPU; token embedding lookup, the timestep embedder, and
/// the final RMSNorm (all negligible) stay on the CPU.
pub struct GpuHiDreamModel {
    pub cfg: super::config::HiDreamConfig,
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<GpuDecoderLayer>,
    pub norm_buf: Buffer,
    pub rms_eps: f32,
    pub x_embedder: GpuPatchEmbed,
    pub t_embedder1: super::layers::TimestepEmbedder,
    pub final_layer2: GpuLinear,
}

impl GpuHiDreamModel {
    pub fn from_cpu(m: &super::model::HiDreamModel) -> Self {
        Self {
            cfg: m.cfg.clone(),
            embed_tokens: m.embed_tokens.clone(),
            layers: m.layers.iter().map(GpuDecoderLayer::from_cpu).collect(),
            norm_buf: upload_f32_as_f16_buffer(&m.norm.weight),
            rms_eps: m.norm.eps,
            x_embedder: GpuPatchEmbed::from_cpu(&m.x_embedder),
            t_embedder1: super::layers::TimestepEmbedder {
                lin1: clone_linear(&m.t_embedder1.lin1),
                lin2: clone_linear(&m.t_embedder1.lin2),
                freq_size: m.t_embedder1.freq_size,
            },
            final_layer2: GpuLinear::from_cpu(&m.final_layer2.linear),
        }
    }

    /// Load the full 8B model directly into GPU bf16 buffers from a HiDream-O1
    /// HuggingFace directory (sharded safetensors). The token embedding,
    /// timestep MLP, and final norm stay on the CPU (negligible).
    pub fn load(cfg: super::config::HiDreamConfig, comp: &ComponentTensors) -> Result<Self> {
        use super::config::{FREQ_EMBED_SIZE, PATCH_DIM};
        let tc = &cfg.text_config;
        let hidden = tc.hidden_size;
        let lm = "model.language_model";

        let mut layers = Vec::with_capacity(tc.num_hidden_layers);
        for i in 0..tc.num_hidden_layers {
            layers.push(GpuDecoderLayer::load(comp, &format!("{lm}.layers.{i}"), tc)?);
        }
        let norm_w = comp.get_f32(&format!("{lm}.norm.weight"))?;

        // CPU-side small modules: token embedding + timestep MLP.
        let embed_tokens = comp.get_f32(&format!("{lm}.embed_tokens.weight"))?;
        let mut t_embedder1 = super::layers::TimestepEmbedder::new(FREQ_EMBED_SIZE, hidden);
        t_embedder1.lin1.load(comp, "model.t_embedder1.mlp.0")?;
        t_embedder1.lin2.load(comp, "model.t_embedder1.mlp.2")?;

        let bottleneck = hidden / 4;
        let x_embedder = GpuPatchEmbed::load(comp, "model.x_embedder", PATCH_DIM, bottleneck, hidden)?;
        let final_layer2 = GpuLinear::load(comp, "model.final_layer2.linear", hidden, PATCH_DIM, true)?;

        Ok(Self {
            norm_buf: upload_f32_as_f16_buffer(&norm_w),
            rms_eps: tc.rms_norm_eps,
            cfg,
            embed_tokens,
            layers,
            x_embedder,
            t_embedder1,
            final_layer2,
        })
    }

    fn hidden(&self) -> usize {
        self.cfg.text_config.hidden_size
    }

    /// GPU text-to-image forward. Returns predicted x₀ patches [image_len, PATCH_DIM].
    /// Mirrors [`super::model::HiDreamModel::forward_t2i`] exactly.
    pub fn forward_t2i(
        &self,
        text_ids: &[u32],
        z: &[f32],
        timestep: f32,
        h_patches: usize,
        w_patches: usize,
    ) -> Vec<f32> {
        use super::config::{PATCH_DIM, TMS_TOKEN_ID};
        // Qwen3-VL's massive activations exceed f16's ±65504 in multiple matmuls
        // (verified: the f16 path NaNs at ≥1024²), so force the f32 sgemm scratch
        // path. (Attention uses MPSGraph SDPA, whose chain is f32 internally, so
        // its f16 input cast is safe since q/k/v stay well under 65504.)
        crate::metal_backend::set_use_f32_sgemm(true);
        let hidden = self.hidden();
        let txt_len = text_ids.len();
        let image_len = h_patches * w_patches;
        let seq = txt_len + image_len;

        // Assemble the initial sequence on CPU (text embeds + timestep-tms +
        // patch-embedded image), then upload once. After this the whole layer
        // stack runs GPU-resident with no per-op downloads.
        let mut states = vec![0.0_f32; seq * hidden];
        for (i, &id) in text_ids.iter().enumerate() {
            let off = id as usize * hidden;
            states[i * hidden..(i + 1) * hidden].copy_from_slice(&self.embed_tokens[off..off + hidden]);
        }
        let t_emb = self.t_embedder1.forward(timestep);
        for (i, &id) in text_ids.iter().enumerate() {
            if id == TMS_TOKEN_ID {
                states[i * hidden..(i + 1) * hidden].copy_from_slice(&t_emb);
            }
        }
        let v_emb = self.x_embedder.forward(z, image_len);
        states[txt_len * hidden..].copy_from_slice(&v_emb);

        // RoPE tables → GPU buffers (built once per forward).
        let position_ids = super::model::build_position_ids_t2i(txt_len, h_patches, w_patches);
        let tc = &self.cfg.text_config;
        let rope = mrope::build_tables(&position_ids, seq, tc.head_dim(), tc.rope_theta, &tc.mrope_section());
        let cos_buf = upload_f32_as_f16_buffer(&rope.cos);
        let sin_buf = upload_f32_as_f16_buffer(&rope.sin);

        let mut hg = GpuTensor::upload_f32_as_f16(vec![seq, hidden], &states);
        for layer in &self.layers {
            layer.forward(&mut hg, seq, &cos_buf, &sin_buf, txt_len);
        }
        rms_norm_f16_gpu(&mut hg, &self.norm_buf, seq, hidden, self.rms_eps);

        // Unembed at the image positions (contiguous suffix) → predicted x₀.
        let full = hg.download_to_f32();
        let img_states = &full[txt_len * hidden..];
        let ig = GpuTensor::upload_f32_as_f16(vec![image_len, hidden], img_states);
        let out = self.final_layer2.forward(&ig, image_len).download_to_f32();
        debug_assert_eq!(out.len(), image_len * PATCH_DIM);
        out
    }

    /// Reference-editing forward (GPU-resident). Mirrors [`super::model::HiDreamModel::forward_ref`]
    /// — the CPU oracle validated at `cos=1.0` — on the GPU layer stack. The
    /// caller supplies the vision encoder's precomputed `vision_embeds` +
    /// `deepstack`; this scatters them at the image-token placeholders, runs the
    /// decoder with the same `txt_len`-based hybrid mask as t2i (VLM-image tokens
    /// are causal text, `<tms>` + target + ref are bidirectional), injects the
    /// DeepStack features after the first 3 layers, and unembeds at the target
    /// positions. Returns x₀ for the target `[tgt_len, PATCH_DIM]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_ref(
        &self,
        text_ids: &[u32],
        vision_embeds: &[f32],
        deepstack: &[Vec<f32>],
        vinputs: &[f32],
        gen_len: usize,
        tgt_len: usize,
        positions: &[i64],
        timestep: f32,
        image_token_id: u32,
        tms_token_id: u32,
    ) -> Vec<f32> {
        use super::config::PATCH_DIM;
        crate::metal_backend::set_use_f32_sgemm(true);
        let hidden = self.hidden();
        let txt_len = text_ids.len();
        let seq = txt_len + gen_len;

        // Assemble the initial sequence on CPU: text embeds, vision embeds at the
        // image-token placeholders, the timestep embedding at <tms>, and the
        // patch-embedded (target ++ reference) vinputs.
        let mut states = vec![0.0_f32; seq * hidden];
        let t_emb = self.t_embedder1.forward(timestep);
        let mut vlm_positions = Vec::new();
        let mut vlm_idx = 0usize;
        for (i, &id) in text_ids.iter().enumerate() {
            let dst = &mut states[i * hidden..(i + 1) * hidden];
            if id == image_token_id {
                dst.copy_from_slice(&vision_embeds[vlm_idx * hidden..(vlm_idx + 1) * hidden]);
                vlm_positions.push(i);
                vlm_idx += 1;
            } else if id == tms_token_id {
                dst.copy_from_slice(&t_emb);
            } else {
                let off = id as usize * hidden;
                dst.copy_from_slice(&self.embed_tokens[off..off + hidden]);
            }
        }
        let v_emb = self.x_embedder.forward(vinputs, gen_len);
        states[txt_len * hidden..].copy_from_slice(&v_emb);

        let tc = &self.cfg.text_config;
        let rope = mrope::build_tables(positions, seq, tc.head_dim(), tc.rope_theta, &tc.mrope_section());
        let cos_buf = upload_f32_as_f16_buffer(&rope.cos);
        let sin_buf = upload_f32_as_f16_buffer(&rope.sin);

        let mut hg = GpuTensor::upload_f32_as_f16(vec![seq, hidden], &states);
        for (li, layer) in self.layers.iter().enumerate() {
            layer.forward(&mut hg, seq, &cos_buf, &sin_buf, txt_len);
            // DeepStack: add the intermediate visual features at the VLM image
            // positions after the first few decoder layers. Only 3 injections,
            // so a download/add/re-upload is cheap enough (kernel TODO).
            if li < deepstack.len() {
                let mut h = hg.download_to_f32();
                let ds = &deepstack[li];
                for (j, &p) in vlm_positions.iter().enumerate() {
                    let s = &mut h[p * hidden..(p + 1) * hidden];
                    for (x, d) in s.iter_mut().zip(&ds[j * hidden..(j + 1) * hidden]) {
                        *x += d;
                    }
                }
                hg = GpuTensor::upload_f32_as_f16(vec![seq, hidden], &h);
            }
        }
        rms_norm_f16_gpu(&mut hg, &self.norm_buf, seq, hidden, self.rms_eps);

        // Unembed at the target positions (first tgt_len of the gen region).
        let full = hg.download_to_f32();
        let tgt = &full[txt_len * hidden..(txt_len + tgt_len) * hidden];
        let ig = GpuTensor::upload_f32_as_f16(vec![tgt_len, hidden], tgt);
        let out = self.final_layer2.forward(&ig, tgt_len).download_to_f32();
        debug_assert_eq!(out.len(), tgt_len * PATCH_DIM);
        out
    }
}

fn clone_linear(l: &Linear) -> Linear {
    Linear {
        weight: l.weight.clone(),
        bias: l.bias.clone(),
        in_features: l.in_features,
        out_features: l.out_features,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hidream::config::{RopeScaling, TextConfig};
    use crate::hidream::layers::Linear;
    use crate::hidream::model::{Attention, DecoderLayer, Mlp};
    use crate::metal_backend::{upload_f32_as_f16_buffer, GpuTensor};

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    fn tiny_cfg() -> TextConfig {
        // head_dim 16 (closer to the real 128 → more averaging) so the bf16
        // SDPA vs f32-CPU gap reflects precision, not a tiny-dim artifact.
        TextConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 152000, // must cover TMS_TOKEN_ID (151673)
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            attention_bias: false,
            rope_scaling: Some(RopeScaling { mrope_section: Some(vec![4, 2, 2]), rope_type: None }),
            max_position_embeddings: 4096,
        }
    }

    fn fill(l: &mut Linear, seed: f32) {
        for (i, w) in l.weight.iter_mut().enumerate() {
            *w = ((i as f32 * seed).sin()) * 0.08;
        }
        if let Some(b) = l.bias.as_mut() {
            for (i, bv) in b.iter_mut().enumerate() {
                *bv = (i as f32 * 0.01).cos() * 0.05;
            }
        }
    }

    fn rope_for(cfg: &TextConfig, n: usize) -> mrope::RopeTables {
        // Simple sequential positions on all 3 axes.
        let mut pos = vec![0i64; 3 * n];
        for i in 0..n {
            pos[i] = i as i64;
            pos[n + i] = i as i64;
            pos[2 * n + i] = i as i64;
        }
        mrope::build_tables(&pos, n, cfg.head_dim(), cfg.rope_theta, &cfg.mrope_section())
    }

    // GPU tests share the (thread-local) Metal backend + MPSGraph cache and
    // MPSGraph construction is not safe across threads, so all GPU checks run
    // sequentially inside one #[test].
    #[test]
    fn gpu_modules_match_cpu() {
        check_gpu_kernels();
        check_gpu_linear();
        check_gpu_mlp();
        check_gpu_attention();
        check_gpu_decoder_layer();
        check_gpu_forward_t2i();
        check_gpu_forward_ref();
    }

    fn check_gpu_forward_ref() {
        use crate::hidream::config::{HiDreamConfig, PATCH_DIM, TMS_TOKEN_ID};
        use crate::hidream::model::HiDreamModel;
        const IMG: u32 = 151655;
        let cfg = HiDreamConfig {
            image_token_id: IMG,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            text_config: tiny_cfg(),
        };
        let mut cpu = HiDreamModel::new(cfg);
        for (i, w) in cpu.embed_tokens.iter_mut().enumerate() {
            *w = ((i % 13) as f32 - 6.0) * 0.01;
        }
        for layer in cpu.layers.iter_mut() {
            fill(&mut layer.self_attn.q_proj, 0.11);
            fill(&mut layer.self_attn.k_proj, 0.22);
            fill(&mut layer.self_attn.v_proj, 0.33);
            fill(&mut layer.self_attn.o_proj, 0.44);
            fill(&mut layer.mlp.gate_proj, 0.5);
            fill(&mut layer.mlp.up_proj, 0.6);
            fill(&mut layer.mlp.down_proj, 0.7);
        }
        fill(&mut cpu.x_embedder.proj1, 0.8);
        fill(&mut cpu.x_embedder.proj2, 0.9);
        fill(&mut cpu.final_layer2.linear, 0.15);
        fill(&mut cpu.t_embedder1.lin1, 0.25);
        fill(&mut cpu.t_embedder1.lin2, 0.35);
        let gpu = GpuHiDreamModel::from_cpu(&cpu);

        let hidden = tiny_cfg().hidden_size;
        // text: [tok, <img>, <img>, tok, <tms>] → 2 VLM placeholders.
        let text_ids = vec![5u32, IMG, IMG, 9u32, TMS_TOKEN_ID];
        let (tgt_len, ref_len) = (4usize, 2usize);
        let gen_len = tgt_len + ref_len;
        let seq = text_ids.len() + gen_len;
        let num_vlm = 2usize;
        let vision_embeds: Vec<f32> =
            (0..num_vlm * hidden).map(|i| (i as f32 * 0.03).sin() * 0.1).collect();
        let deepstack: Vec<Vec<f32>> = (0..2)
            .map(|l| (0..num_vlm * hidden).map(|i| ((i + l) as f32 * 0.05).cos() * 0.1).collect())
            .collect();
        let vinputs: Vec<f32> =
            (0..gen_len * PATCH_DIM).map(|i| (i as f32 * 0.01).sin() * 0.2).collect();
        // arbitrary-but-consistent positions (both paths take them as input).
        let mut positions = vec![0i64; 3 * seq];
        for ax in 0..3 {
            for i in 0..seq {
                positions[ax * seq + i] = i as i64;
            }
        }

        let cpu_out = cpu.forward_ref(
            &text_ids, &vision_embeds, &deepstack, &vinputs, gen_len, tgt_len, &positions, 0.5, IMG,
            TMS_TOKEN_ID,
        );
        let gpu_out = gpu.forward_ref(
            &text_ids, &vision_embeds, &deepstack, &vinputs, gen_len, tgt_len, &positions, 0.5, IMG,
            TMS_TOKEN_ID,
        );
        let cos = cosine(&cpu_out, &gpu_out);
        println!("gpu_forward_ref cos={cos:.6}");
        assert!(cos > 0.99, "gpu/cpu forward_ref cosine too low: {cos}");
    }

    fn check_gpu_kernels() {
        use crate::hidream::layers::RmsNorm;
        use crate::metal_backend::{
            emul_f16_gpu, gqa_repeat_f16_gpu, rms_norm_f16_gpu, rope_apply_f16_gpu,
            upload_f32_as_f16_buffer,
        };

        // --- rms_norm (per-token) ---
        let (n_rows, dim) = (5usize, 16usize);
        let x: Vec<f32> = (0..n_rows * dim).map(|i| (i as f32 * 0.1).sin() * 2.0).collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 * 0.05)).collect();
        let mut cpu = RmsNorm { weight: w.clone(), eps: 1e-6 };
        let mut cpu_out = x.clone();
        cpu.apply(&mut cpu_out, n_rows);
        let mut xg = GpuTensor::upload_f32_as_f16(vec![n_rows * dim], &x);
        let wbuf = upload_f32_as_f16_buffer(&w);
        rms_norm_f16_gpu(&mut xg, &wbuf, n_rows, dim, 1e-6);
        let cos1 = cosine(&xg.download_to_f32(), &cpu_out);
        println!("gpu rms_norm cos={cos1:.6}");
        assert!(cos1 > 0.999, "rms_norm {cos1}");
        cpu.weight = w; // silence unused-mut

        // --- rope ---
        let (nn, nh, dh) = (4usize, 2usize, 8usize);
        let mut pos = vec![0i64; 3 * nn];
        for i in 0..nn {
            pos[i] = i as i64;
            pos[nn + i] = i as i64;
            pos[2 * nn + i] = i as i64;
        }
        let tables = mrope::build_tables(&pos, nn, dh, 10000.0, &[2, 1, 1]);
        let q: Vec<f32> = (0..nn * nh * dh).map(|i| (i as f32 * 0.07).cos()).collect();
        let mut q_cpu = q.clone();
        mrope::apply(&mut q_cpu, &tables, nn, nh);
        let mut qg = GpuTensor::upload_f32_as_f16(vec![nn * nh * dh], &q);
        let cosb = upload_f32_as_f16_buffer(&tables.cos);
        let sinb = upload_f32_as_f16_buffer(&tables.sin);
        rope_apply_f16_gpu(&mut qg, &cosb, &sinb, nn, nh, dh);
        let cos2 = cosine(&qg.download_to_f32(), &q_cpu);
        println!("gpu rope cos={cos2:.6}");
        assert!(cos2 > 0.999, "rope {cos2}");

        // --- emul ---
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.2).sin()).collect();
        let b: Vec<f32> = (0..32).map(|i| (i as f32 * 0.3).cos()).collect();
        let mut ag = GpuTensor::upload_f32_as_f16(vec![32], &a);
        let bg = GpuTensor::upload_f32_as_f16(vec![32], &b);
        emul_f16_gpu(&mut ag, &bg);
        let prod: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
        let cos3 = cosine(&ag.download_to_f32(), &prod);
        println!("gpu emul cos={cos3:.6}");
        assert!(cos3 > 0.999, "emul {cos3}");

        // --- gqa_repeat ---
        let (nkv, nrep, block) = (2usize, 3usize, 6usize);
        let nh2 = nkv * nrep;
        let inp: Vec<f32> = (0..nkv * block).map(|i| i as f32).collect();
        let inpg = GpuTensor::upload_f32_as_f16(vec![nkv * block], &inp);
        let mut outg = GpuTensor::new_f16(vec![nh2 * block]);
        gqa_repeat_f16_gpu(&inpg, &mut outg, nrep, block);
        let mut expect = vec![0.0_f32; nh2 * block];
        for h in 0..nh2 {
            let kv = h / nrep;
            expect[h * block..(h + 1) * block].copy_from_slice(&inp[kv * block..(kv + 1) * block]);
        }
        let cos4 = cosine(&outg.download_to_f32(), &expect);
        println!("gpu gqa_repeat cos={cos4:.6}");
        assert!(cos4 > 0.999, "gqa_repeat {cos4}");
    }

    fn check_gpu_forward_t2i() {
        use crate::hidream::config::{HiDreamConfig, PATCH_DIM, TMS_TOKEN_ID};
        use crate::hidream::model::HiDreamModel;
        let cfg = HiDreamConfig {
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            text_config: tiny_cfg(),
        };
        let mut cpu = HiDreamModel::new(cfg);
        for (i, w) in cpu.embed_tokens.iter_mut().enumerate() {
            *w = ((i % 13) as f32 - 6.0) * 0.01;
        }
        for layer in cpu.layers.iter_mut() {
            fill(&mut layer.self_attn.q_proj, 0.11);
            fill(&mut layer.self_attn.k_proj, 0.22);
            fill(&mut layer.self_attn.v_proj, 0.33);
            fill(&mut layer.self_attn.o_proj, 0.44);
            fill(&mut layer.mlp.gate_proj, 0.5);
            fill(&mut layer.mlp.up_proj, 0.6);
            fill(&mut layer.mlp.down_proj, 0.7);
        }
        fill(&mut cpu.x_embedder.proj1, 0.8);
        fill(&mut cpu.x_embedder.proj2, 0.9);
        fill(&mut cpu.final_layer2.linear, 0.15);
        fill(&mut cpu.t_embedder1.lin1, 0.25);
        fill(&mut cpu.t_embedder1.lin2, 0.35);

        let gpu = GpuHiDreamModel::from_cpu(&cpu);
        let (hp, wp) = (3usize, 2usize);
        let image_len = hp * wp;
        let text_ids = vec![5u32, 9u32, TMS_TOKEN_ID];
        let z: Vec<f32> = (0..image_len * PATCH_DIM).map(|i| (i as f32 * 0.01).sin() * 0.2).collect();

        let cpu_out = cpu.forward_t2i(&text_ids, &z, 0.5, hp, wp);
        let gpu_out = gpu.forward_t2i(&text_ids, &z, 0.5, hp, wp);
        let cos = cosine(&cpu_out, &gpu_out);
        println!("gpu_forward_t2i cos={cos:.6}");
        assert!(cos > 0.99, "gpu/cpu forward_t2i cosine too low: {cos}");
    }

    fn check_gpu_attention() {
        let cfg = tiny_cfg();
        let n = 6;
        let mut cpu = Attention::new(&cfg);
        fill(&mut cpu.q_proj, 0.11);
        fill(&mut cpu.k_proj, 0.22);
        fill(&mut cpu.v_proj, 0.33);
        fill(&mut cpu.o_proj, 0.44);
        for (i, w) in cpu.q_norm.weight.iter_mut().enumerate() {
            *w = 1.0 + (i as f32 * 0.05);
        }
        for (i, w) in cpu.k_norm.weight.iter_mut().enumerate() {
            *w = 1.0 - (i as f32 * 0.03);
        }
        let gpu = GpuAttention::from_cpu(&cpu);
        let rope = rope_for(&cfg, n);
        // txt_len=3 → text-causal rows [0,1], <tms> row 2, gen rows [2..].
        // Matches CPU gen_mask `i >= 2`.
        let txt_len = 3usize;
        let gen_mask: Vec<bool> = (0..n).map(|i| i >= 2).collect();
        let x: Vec<f32> = (0..n * cfg.hidden_size).map(|i| (i as f32 * 0.017).sin() * 0.5).collect();

        let cpu_out = cpu.forward(&x, n, &rope, &gen_mask);
        let cos_buf = upload_f32_as_f16_buffer(&rope.cos);
        let sin_buf = upload_f32_as_f16_buffer(&rope.sin);
        let xg = GpuTensor::upload_f32_as_f16(vec![n * cfg.hidden_size], &x);
        let gpu_out = gpu.forward(&xg, n, &cos_buf, &sin_buf, txt_len).download_to_f32();
        let cos = cosine(&cpu_out, &gpu_out);
        println!("gpu_attention cos={cos:.6}");
        assert!(cos > 0.99, "gpu/cpu attention cosine too low: {cos}");
    }

    fn check_gpu_mlp() {
        let n = 5;
        let (hidden, inter) = (32usize, 64usize);
        let mut cpu = Mlp::new(hidden, inter);
        fill(&mut cpu.gate_proj, 0.5);
        fill(&mut cpu.up_proj, 0.6);
        fill(&mut cpu.down_proj, 0.7);
        let gpu = GpuMlp::from_cpu(&cpu);
        let x: Vec<f32> = (0..n * hidden).map(|i| (i as f32 * 0.02).cos() * 0.4).collect();
        let xg = GpuTensor::upload_f32_as_f16(vec![n * hidden], &x);
        let gpu_out = gpu.forward(&xg, n).download_to_f32();
        let cos = cosine(&cpu.forward(&x, n), &gpu_out);
        println!("gpu_mlp cos={cos:.6}");
        assert!(cos > 0.998, "gpu/cpu mlp cosine too low: {cos}");
    }

    fn check_gpu_decoder_layer() {
        let cfg = tiny_cfg();
        let n = 6;
        let mut cpu = DecoderLayer::new(&cfg);
        fill(&mut cpu.self_attn.q_proj, 0.11);
        fill(&mut cpu.self_attn.k_proj, 0.22);
        fill(&mut cpu.self_attn.v_proj, 0.33);
        fill(&mut cpu.self_attn.o_proj, 0.44);
        fill(&mut cpu.mlp.gate_proj, 0.5);
        fill(&mut cpu.mlp.up_proj, 0.6);
        fill(&mut cpu.mlp.down_proj, 0.7);
        for (i, w) in cpu.input_layernorm.weight.iter_mut().enumerate() {
            *w = 1.0 + (i as f32 * 0.01);
        }
        for (i, w) in cpu.post_attention_layernorm.weight.iter_mut().enumerate() {
            *w = 1.0 - (i as f32 * 0.01);
        }
        let gpu = GpuDecoderLayer::from_cpu(&cpu);
        let rope = rope_for(&cfg, n);
        let txt_len = 3usize;
        let gen_mask: Vec<bool> = (0..n).map(|i| i >= 2).collect();
        let x: Vec<f32> = (0..n * cfg.hidden_size).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();

        let mut cpu_h = x.clone();
        cpu.forward(&mut cpu_h, n, &rope, &gen_mask);
        let cos_buf = upload_f32_as_f16_buffer(&rope.cos);
        let sin_buf = upload_f32_as_f16_buffer(&rope.sin);
        let mut hg = GpuTensor::upload_f32_as_f16(vec![n * cfg.hidden_size], &x);
        gpu.forward(&mut hg, n, &cos_buf, &sin_buf, txt_len);
        let cos = cosine(&cpu_h, &hg.download_to_f32());
        println!("gpu_decoder_layer cos={cos:.6}");
        assert!(cos > 0.99, "gpu/cpu decoder layer cosine too low: {cos}");
    }

    fn check_gpu_linear() {
        let (m, inf, outf) = (8usize, 64usize, 48usize);
        let mut cpu = Linear::new(inf, outf, true);
        for (i, w) in cpu.weight.iter_mut().enumerate() {
            *w = ((i as f32 * 0.013).sin()) * 0.1;
        }
        for (i, b) in cpu.bias.as_mut().unwrap().iter_mut().enumerate() {
            *b = (i as f32 * 0.01) - 0.2;
        }
        let x: Vec<f32> = (0..m * inf).map(|i| (i as f32 * 0.007).cos() * 0.5).collect();

        let cpu_out = cpu.forward_vec(&x, m);

        let gpu = GpuLinear::from_cpu(&cpu);
        let xg = GpuTensor::upload_f32_as_f16(vec![m, inf], &x);
        let gpu_out = gpu.forward(&xg, m).download_to_f32();

        let cos = cosine(&cpu_out, &gpu_out);
        println!("gpu_linear cos={cos:.6}");
        assert!(cos > 0.999, "gpu/cpu linear cosine too low: {cos}");
    }
}
