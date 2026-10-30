//! Metal GPU path for the Krea 2 transformer.
//!
//! Mirrors [`super::model`], the f32 CPU oracle, with one deliberate change of
//! layout: the joint sequence is **image first**, `[image | text | padding]`,
//! where diffusers uses `[text | image]`. Nothing in a block depends on token
//! order — attention has no causal mask and rotary phases come from explicit
//! positions (text tokens all sit at the origin) — so the reorder is exact,
//! and it buys two things:
//!
//! * The final layer reads the image rows as a contiguous prefix.
//! * Padding rows go at the end, where limiting attention's key length to the
//!   real rows masks them exactly. That lets both CFG branches run at one
//!   sequence length even though their prompts differ, so the second branch
//!   reuses the first branch's pooled buffers instead of allocating its own.
//!
//! Storage is bf16 (`GpuDtype::F16` — see `feedback_metal_bf16_constraints`).
//! Norm scales stay f32 on the GPU: they are stored zero-centred and the
//! effective `1 + w` does not survive bf16.
//!
//! Gated by `feature = "metal"` at the module declaration.

use metal::Buffer;

use super::config::Krea2Config;
use super::model::{check_inventory, get_checked, image_positions, Rope, TimeEmbed};
use crate::metal_backend::{
    bias_add_f16_gpu, blit_copy_f16, eadd_f16_gpu, emul_f16_gpu, final_modulate_f16_gpu,
    flash_attention_best_f16_gpu, flash_attention_masked_f16_gpu, gate_rows_f16_gpu, gelu_f16_gpu, gqa_repeat_f16_gpu,
    modulate_rows_f16_gpu, permute_hl_to_lh_f16_gpu, permute_lh_to_hl_f16_gpu,
    rms_norm_wf32_f16_gpu, rope_pairs_f16_gpu, sgemm_f16_buf, sgemm_f16_buf_with_offsets,
    sigmoid_gate_f16_gpu,
    silu_f16_gpu, transpose_outer_f16_gpu, upload_bf16_bytes, upload_f32_as_f16_buffer,
    upload_f32_buffer, zero_buffer_gpu, zero_range_gpu, GpuDtype, GpuTensor, ModLayout,
};
use crate::wan::gpu::prof;
use crate::weight::loader::ComponentTensors;
use crate::{DiffusionError, Result};

/// Diagnostic magnitude probes: `KLEARU_KREA2_MAG=1` records the largest
/// |value| at each matmul boundary in every block and prints, per forward,
/// each site's peak against fp16's 65504 — the limit of the MPS f16 matmul
/// path these tensors pass through. Each probe downloads its tensor, so a
/// probed forward is several times slower.
pub mod mag {
    use std::cell::{Cell, RefCell};

    use crate::metal_backend::GpuTensor;

    thread_local! {
        static ON: Cell<bool> = Cell::new(std::env::var_os("KLEARU_KREA2_MAG").is_some());
        static BLOCK: Cell<usize> = const { Cell::new(0) };
        /// (site, block, max |finite value|, non-finite count)
        static SEEN: RefCell<Vec<(&'static str, usize, f32, usize)>> = const { RefCell::new(Vec::new()) };
    }

    pub fn on() -> bool {
        ON.with(|v| v.get())
    }

    /// Switch probing on or off at runtime, e.g. to rerun one failing forward.
    pub fn set_on(v: bool) {
        ON.with(|c| c.set(v));
    }

    pub fn set_block(i: usize) {
        BLOCK.with(|b| b.set(i));
    }

    /// Drop what earlier passes (e.g. the text head's) recorded.
    pub fn reset() {
        SEEN.with(|s| s.borrow_mut().clear());
    }

    /// Record the first `n` elements of `t` under `site`.
    pub fn probe(site: &'static str, t: &GpuTensor, n: usize) {
        if !on() {
            return;
        }
        let v = t.download_to_f32();
        let (mut max, mut bad) = (0.0f32, 0usize);
        for &a in &v[..n] {
            if a.is_finite() { max = max.max(a.abs()) } else { bad += 1 }
        }
        let block = BLOCK.with(|b| b.get());
        SEEN.with(|s| s.borrow_mut().push((site, block, max, bad)));
    }

    /// Print each site's peak (and the first non-finite site), then reset.
    pub fn report() {
        if !on() {
            return;
        }
        let seen = SEEN.with(|s| std::mem::take(&mut *s.borrow_mut()));
        let mut sites: Vec<&'static str> = Vec::new();
        for (site, ..) in &seen {
            if !sites.contains(site) {
                sites.push(site);
            }
        }
        println!("[krea2 mag] per-site peak |value| (fp16 max 65504):");
        for site in sites {
            let (blk, max) = seen
                .iter()
                .filter(|e| e.0 == site)
                .map(|e| (e.1, e.2))
                .fold((0, 0.0f32), |acc, e| if e.1 > acc.1 { e } else { acc });
            println!("  {site:10} {max:12.1} (block {blk})");
        }
        if let Some((site, blk, _, bad)) = seen.iter().find(|e| e.3 > 0) {
            println!("  first non-finite: {site} in block {blk} ({bad} values)");
        }
    }
}

/// Linear layer with its weight resident as bf16, **pre-transposed** to
/// `[in, out]`. MPS's B-transposed matmul runs tall-K shapes several times
/// slower than the untransposed one (measured 4.6x on Wan's FFN), and the
/// weights never change, so the transpose is paid once at load.
pub struct GpuLinear {
    pub weight: Buffer,
    pub bias: Option<Buffer>,
    pub in_f: usize,
    pub out_f: usize,
}

impl GpuLinear {
    /// Load `{p}.weight` (+ `{p}.bias`) straight from the checkpoint. A bf16
    /// tensor goes mmap -> MTLBuffer with no f32 intermediate.
    pub fn load(comp: &ComponentTensors, p: &str, in_f: usize, out_f: usize, bias: bool) -> Result<Self> {
        let name = format!("{p}.weight");
        expect_numel(comp, &name, in_f * out_f)?;
        let raw = comp.get_bf16(&name)?;
        let src = GpuTensor {
            buffer: upload_bf16_bytes(&raw),
            shape: vec![out_f, in_f],
            dtype: GpuDtype::F16,
        };
        drop(raw);
        let weight = transpose_outer_f16_gpu(&src, out_f, in_f, 1).buffer;
        let bias = if bias {
            let name = format!("{p}.bias");
            expect_numel(comp, &name, out_f)?;
            Some(upload_bf16_bytes(&comp.get_bf16(&name)?))
        } else {
            None
        };
        Ok(Self { weight, bias, in_f, out_f })
    }

    /// `y[m, out] = x[:m] · W (+ b)`. Reads only the first `m` rows of `x`,
    /// so a prefix of a longer tensor can be passed as-is.
    ///
    /// Tall inputs are split into **equal** chunks of at most
    /// [`MATMUL_MAX_ROWS`]. The shared wrapper's own chunking keeps every call
    /// exactly 4096 tall and bottom-aligns the last one, recomputing the
    /// overlap — which at 1024² (4124 rows: image plus prompt) computes 8192
    /// rows, nearly double the work.
    pub fn forward(&self, x: &GpuTensor, m: usize) -> GpuTensor {
        debug_assert!(x.elements() >= m * self.in_f);
        let mut out = GpuTensor::new_f16(vec![m, self.out_f]);
        if m <= MATMUL_MAX_ROWS {
            sgemm_f16_buf(&out.buffer, &x.buffer, &self.weight, m, self.out_f, self.in_f);
        } else {
            let chunks = m.div_ceil(MATMUL_MAX_ROWS);
            let rows = m.div_ceil(chunks);
            for c in 0..chunks {
                let s0 = c * rows;
                let r = rows.min(m - s0);
                sgemm_f16_buf_with_offsets(
                    &out.buffer, s0 * self.out_f, &x.buffer, s0 * self.in_f, &self.weight, 0,
                    r, self.out_f, self.in_f,
                );
            }
        }
        if let Some(b) = &self.bias {
            bias_add_f16_gpu(&mut out, b, 1, self.out_f);
        }
        out
    }
}

/// Attention work per GPU dispatch, in FLOPs: ~0.5 s at the register
/// kernel's ~2.75 TF/s, comfortably inside the GPU watchdog's patience.
/// `KLEARU_KREA2_ATTN_FLOPS` overrides it (read once per thread — `getenv`
/// takes a process-wide lock).
fn attn_flops_per_call() -> f64 {
    thread_local! {
        static V: f64 = std::env::var("KLEARU_KREA2_ATTN_FLOPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.5e12);
    }
    V.with(|v| *v)
}

/// Tallest matmul handed to MPS in one call — the backend's duration guard
/// against the GPU watchdog (`metal_backend::MPS_MAX_ROWS`).
const MATMUL_MAX_ROWS: usize = 4096;

fn expect_numel(comp: &ComponentTensors, name: &str, want: usize) -> Result<()> {
    match comp.numel(name) {
        None => Err(DiffusionError::MissingTensor(name.to_string())),
        Some(n) if n != want => Err(DiffusionError::ShapeMismatch {
            expected: format!("{want} elements in {name}"),
            got: format!("{n}"),
        }),
        _ => Ok(()),
    }
}

/// Zero-centred RMSNorm with its `1 + w` multiplier held in f32.
pub struct GpuRmsNorm {
    pub weight: Buffer,
    pub dim: usize,
    pub eps: f32,
}

impl GpuRmsNorm {
    pub fn load(comp: &ComponentTensors, name: &str, dim: usize, eps: f32) -> Result<Self> {
        let w: Vec<f32> = get_checked(comp, name, dim)?.iter().map(|v| 1.0 + v).collect();
        Ok(Self { weight: upload_f32_buffer(&w), dim, eps })
    }

    pub fn forward(&self, x: &mut GpuTensor, rows: usize) {
        rms_norm_wf32_f16_gpu(x, &self.weight, rows, self.dim, self.eps);
    }
}

/// Rotary tables for one sequence, `[rows, head_dim]` bf16.
pub struct GpuRope {
    pub cos: Buffer,
    pub sin: Buffer,
}

impl GpuRope {
    pub fn from_cpu(r: &Rope) -> Self {
        Self { cos: upload_f32_as_f16_buffer(&r.cos), sin: upload_f32_as_f16_buffer(&r.sin) }
    }
}

/// Gated grouped-query attention.
pub struct GpuAttention {
    pub to_q: GpuLinear,
    pub to_k: GpuLinear,
    pub to_v: GpuLinear,
    pub to_gate: GpuLinear,
    pub to_out: GpuLinear,
    pub norm_q: GpuRmsNorm,
    pub norm_k: GpuRmsNorm,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl GpuAttention {
    pub fn load(
        comp: &ComponentTensors, p: &str, dim: usize, heads: usize, kv_heads: usize, eps: f32,
    ) -> Result<Self> {
        let dh = dim / heads;
        Ok(Self {
            to_q: GpuLinear::load(comp, &format!("{p}.to_q"), dim, heads * dh, false)?,
            to_k: GpuLinear::load(comp, &format!("{p}.to_k"), dim, kv_heads * dh, false)?,
            to_v: GpuLinear::load(comp, &format!("{p}.to_v"), dim, kv_heads * dh, false)?,
            to_gate: GpuLinear::load(comp, &format!("{p}.to_gate"), dim, dim, false)?,
            to_out: GpuLinear::load(comp, &format!("{p}.to_out.0"), dim, dim, false)?,
            norm_q: GpuRmsNorm::load(comp, &format!("{p}.norm_q.weight"), dh, eps)?,
            norm_k: GpuRmsNorm::load(comp, &format!("{p}.norm_k.weight"), dh, eps)?,
            heads,
            kv_heads,
            head_dim: dh,
        })
    }

    /// `x` holds `groups` independent sequences of `seq` rows each. Within a
    /// sequence every row is a query, but only the first `seq_kv` rows are
    /// keys — padding rows past that are invisible to everyone. `seq_kv` <
    /// `seq` requires `groups == 1` (keys must be a prefix of the buffer).
    pub fn forward(
        &self, x: &GpuTensor, groups: usize, seq: usize, seq_kv: usize, rope: Option<&GpuRope>,
    ) -> GpuTensor {
        assert!(seq_kv <= seq && (seq_kv == seq || groups == 1));
        let (h, kvh, dh) = (self.heads, self.kv_heads, self.head_dim);
        let rows = groups * seq;

        prof::phase("attn.qkvg_matmul");
        mag::probe("attn.in", x, rows * self.to_q.in_f);
        // Keys over every row, not just the first `seq_kv`: the padded rows
        // get zero weight from the masked kernel below, and a stride that is
        // a multiple of 8 is what lets the fastest kernel run at all.
        let mut q = self.to_q.forward(x, rows);
        let mut k = self.to_k.forward(x, rows);
        let v = self.to_v.forward(x, rows);
        let gate = self.to_gate.forward(x, rows);
        for (site, t) in [("attn.q", &q), ("attn.k", &k), ("attn.v", &v), ("attn.gate", &gate)] {
            mag::probe(site, t, t.elements());
        }

        prof::phase("attn.qknorm_rope");
        self.norm_q.forward(&mut q, rows * h);
        self.norm_k.forward(&mut k, rows * kvh);
        if let Some(r) = rope {
            rope_pairs_f16_gpu(&mut q, &r.cos, &r.sin, rows, h, dh);
            rope_pairs_f16_gpu(&mut k, &r.cos, &r.sin, rows, kvh, dh);
        }
        if seq_kv < seq {
            // Padding rows carry no signal, and nothing bounds what they grow
            // into over 28 blocks — f16 overflow included. The masked kernel
            // gives them P == 0, but 0 x inf is NaN in the P·V product, so
            // zero their keys and values outright: then they are exactly inert.
            let (off, len) = (seq_kv * kvh * dh * 2, (seq - seq_kv) * kvh * dh * 2);
            zero_range_gpu(&k.buffer, off, len);
            zero_range_gpu(&v.buffer, off, len);
        }

        prof::phase("attn.permute_gqa");
        let qh = permute_lh_to_hl_f16_gpu(&q, groups, seq, h, dh);
        let kv_heads = |k: &GpuTensor, v: &GpuTensor, l: usize| -> (GpuTensor, GpuTensor) {
            let kh0 = permute_lh_to_hl_f16_gpu(k, groups, l, kvh, dh);
            let vh0 = permute_lh_to_hl_f16_gpu(v, groups, l, kvh, dh);
            if h == kvh {
                return (kh0, vh0);
            }
            // Query head i reads kv head i / (h / kvh) — repeat_interleave.
            let mut kh = GpuTensor::new_f16(vec![groups * h * l * dh]);
            let mut vh = GpuTensor::new_f16(vec![groups * h * l * dh]);
            gqa_repeat_f16_gpu(&kh0, &mut kh, h / kvh, l * dh);
            gqa_repeat_f16_gpu(&vh0, &mut vh, h / kvh, l * dh);
            (kh, vh)
        };
        let (kh, vh) = kv_heads(&k, &v, seq);
        let mut oh = GpuTensor::new_f16(vec![groups * h * seq * dh]);
        let scale = 1.0 / (dh as f32).sqrt();
        prof::phase("attn.flash");
        // Flash, not MPSGraph SDPA: MPSGraph was ~40% faster on attention
        // alone at 1024² but only ~5% on the whole forward, and its O(L²)
        // scores (~55 GB at 2048²) do not fit this machine.
        let attend = |qh: &GpuTensor, kh: &GpuTensor, vh: &GpuTensor, oh: &mut GpuTensor, nh: usize| {
            if seq_kv == seq {
                flash_attention_best_f16_gpu(qh, kh, vh, oh, groups, nh, seq, seq, dh, scale);
                true
            } else {
                flash_attention_masked_f16_gpu(qh, kh, vh, oh, groups, nh, seq, seq, seq_kv, dh, scale)
            }
        };
        // Heads per dispatch. Each dispatch is its own command buffer, and one
        // over all 48 heads at 2048² is ~6.6 TFLOP — seconds on the GPU,
        // which is the regime where macOS kills command buffers. At 1024² the
        // whole thing fits one call.
        let per_head = 4.0 * seq as f64 * seq as f64 * dh as f64;
        let hpc = ((attn_flops_per_call() / per_head) as usize).clamp(1, h);
        let ok = if groups == 1 && hpc < h {
            let mut all = true;
            for h0 in (0..h).step_by(hpc) {
                let hn = hpc.min(h - h0);
                let slice = |t: &GpuTensor| {
                    let out = GpuTensor::new_f16(vec![hn * seq * dh]);
                    blit_copy_f16(&t.buffer, h0 * seq * dh, &out.buffer, 0, hn * seq * dh);
                    out
                };
                let (qs, ks, vs) = (slice(&qh), slice(&kh), slice(&vh));
                let mut os = GpuTensor::new_f16(vec![hn * seq * dh]);
                all &= attend(&qs, &ks, &vs, &mut os, hn);
                blit_copy_f16(&os.buffer, 0, &oh.buffer, h0 * seq * dh, hn * seq * dh);
            }
            all
        } else {
            attend(&qh, &kh, &vh, &mut oh, h)
        };
        if !ok {
            // No register kernel for this shape (small test configs, or a
            // stride that is not a multiple of 8): attend over just the real
            // keys with the general kernel.
            let (kh, vh) = kv_heads(&k, &v, seq_kv);
            flash_attention_best_f16_gpu(&qh, &kh, &vh, &mut oh, groups, h, seq, seq_kv, dh, scale);
        }

        prof::phase("attn.permute_gate");
        let mut ctx = permute_hl_to_lh_f16_gpu(&oh, groups, seq, h, dh);
        ctx.shape = vec![rows, h * dh];
        sigmoid_gate_f16_gpu(&mut ctx, &gate);
        mag::probe("attn.ctx", &ctx, ctx.elements());
        prof::phase("attn.out_matmul");
        let out = self.to_out.forward(&ctx, rows);
        mag::probe("attn.out", &out, out.elements());
        out
    }
}

/// `down(silu(gate(x)) * up(x))`.
pub struct GpuSwiGlu {
    pub gate: GpuLinear,
    pub up: GpuLinear,
    pub down: GpuLinear,
}

impl GpuSwiGlu {
    pub fn load(comp: &ComponentTensors, p: &str, dim: usize, inter: usize) -> Result<Self> {
        Ok(Self {
            gate: GpuLinear::load(comp, &format!("{p}.gate"), dim, inter, false)?,
            up: GpuLinear::load(comp, &format!("{p}.up"), dim, inter, false)?,
            down: GpuLinear::load(comp, &format!("{p}.down"), inter, dim, false)?,
        })
    }

    pub fn forward(&self, x: &GpuTensor, rows: usize) -> GpuTensor {
        prof::phase("ffn.gate_up_matmul");
        mag::probe("ff.in", x, rows * self.gate.in_f);
        let mut g = self.gate.forward(x, rows);
        let u = self.up.forward(x, rows);
        mag::probe("ff.gate", &g, g.elements());
        mag::probe("ff.up", &u, u.elements());
        prof::phase("ffn.silu_mul");
        silu_f16_gpu(&mut g);
        emul_f16_gpu(&mut g, &u);
        mag::probe("ff.mid", &g, g.elements());
        prof::phase("ffn.down_matmul");
        let out = self.down.forward(&g, rows);
        mag::probe("ff.out", &out, out.elements());
        out
    }
}

/// Text fusion block: pre-norm, no modulation, no rotary embedding.
pub struct GpuTextFusionBlock {
    pub norm1: GpuRmsNorm,
    pub norm2: GpuRmsNorm,
    pub attn: GpuAttention,
    pub ff: GpuSwiGlu,
}

impl GpuTextFusionBlock {
    pub fn load(comp: &ComponentTensors, p: &str, cfg: &Krea2Config) -> Result<Self> {
        let d = cfg.text_hidden_dim;
        Ok(Self {
            norm1: GpuRmsNorm::load(comp, &format!("{p}.norm1.weight"), d, cfg.norm_eps)?,
            norm2: GpuRmsNorm::load(comp, &format!("{p}.norm2.weight"), d, cfg.norm_eps)?,
            attn: GpuAttention::load(
                comp, &format!("{p}.attn"), d, cfg.text_num_attention_heads,
                cfg.text_num_key_value_heads, cfg.norm_eps,
            )?,
            ff: GpuSwiGlu::load(comp, &format!("{p}.ff"), d, cfg.text_intermediate_size)?,
        })
    }

    pub fn forward(&self, x: &mut GpuTensor, groups: usize, seq: usize) {
        let rows = groups * seq;
        let mut h = x.clone_data();
        self.norm1.forward(&mut h, rows);
        let a = self.attn.forward(&h, groups, seq, seq, None);
        eadd_f16_gpu(x, &a);
        let mut h = x.clone_data();
        self.norm2.forward(&mut h, rows);
        let f = self.ff.forward(&h, rows);
        eadd_f16_gpu(x, &f);
    }
}

/// One single-stream block.
pub struct GpuBlock {
    /// `[6, d]`: prescale, preshift, pregate, postscale, postshift, postgate.
    pub table: Buffer,
    pub norm1: GpuRmsNorm,
    pub norm2: GpuRmsNorm,
    pub attn: GpuAttention,
    pub ff: GpuSwiGlu,
    pub dim: usize,
}

impl GpuBlock {
    pub fn load(comp: &ComponentTensors, p: &str, cfg: &Krea2Config) -> Result<Self> {
        let d = cfg.hidden();
        let name = format!("{p}.scale_shift_table");
        expect_numel(comp, &name, 6 * d)?;
        Ok(Self {
            table: upload_bf16_bytes(&comp.get_bf16(&name)?),
            norm1: GpuRmsNorm::load(comp, &format!("{p}.norm1.weight"), d, cfg.norm_eps)?,
            norm2: GpuRmsNorm::load(comp, &format!("{p}.norm2.weight"), d, cfg.norm_eps)?,
            attn: GpuAttention::load(
                comp, &format!("{p}.attn"), d, cfg.num_attention_heads,
                cfg.num_key_value_heads, cfg.norm_eps,
            )?,
            ff: GpuSwiGlu::load(comp, &format!("{p}.ff"), d, cfg.intermediate_size)?,
            dim: d,
        })
    }

    /// `x` is the residual stream `[rows, d]`; the first `n_kv` rows are real
    /// tokens and the rest padding.
    pub fn forward(&self, x: &mut GpuTensor, rows: usize, n_kv: usize, temb_mod: &Buffer, rope: &GpuRope) {
        let d = self.dim;
        let lay = ModLayout::PerSample;
        prof::phase("block.norm_mod");
        let mut h = x.clone_data();
        self.norm1.forward(&mut h, rows);
        modulate_rows_f16_gpu(&mut h, &self.table, temb_mod, rows, d, lay, 0, 1);
        let mut a = self.attn.forward(&h, 1, rows, n_kv, Some(rope));
        prof::phase("block.gate_add");
        gate_rows_f16_gpu(&mut a, &self.table, temb_mod, rows, d, lay, 2);
        eadd_f16_gpu(x, &a);

        prof::phase("block.norm_mod");
        let mut h = x.clone_data();
        self.norm2.forward(&mut h, rows);
        modulate_rows_f16_gpu(&mut h, &self.table, temb_mod, rows, d, lay, 3, 4);
        let mut f = self.ff.forward(&h, rows);
        prof::phase("block.gate_add");
        gate_rows_f16_gpu(&mut f, &self.table, temb_mod, rows, d, lay, 5);
        eadd_f16_gpu(x, &f);
    }
}

/// Text fusion + `txt_in` on the GPU. Runs once per prompt.
pub struct GpuTextHead {
    pub layerwise: Vec<GpuTextFusionBlock>,
    pub projector: Vec<f32>,
    pub refiner: Vec<GpuTextFusionBlock>,
    pub txt_norm: GpuRmsNorm,
    pub txt_l1: GpuLinear,
    pub txt_l2: GpuLinear,
    pub num_layers: usize,
    pub text_dim: usize,
    pub hidden: usize,
}

impl GpuTextHead {
    pub fn load(comp: &ComponentTensors, cfg: &Krea2Config) -> Result<Self> {
        let td = cfg.text_hidden_dim;
        let d = cfg.hidden();
        let blocks = |kind: &str, n: usize| -> Result<Vec<GpuTextFusionBlock>> {
            (0..n)
                .map(|i| GpuTextFusionBlock::load(comp, &format!("text_fusion.{kind}.{i}"), cfg))
                .collect()
        };
        Ok(Self {
            layerwise: blocks("layerwise_blocks", cfg.num_layerwise_text_blocks)?,
            projector: get_checked(comp, "text_fusion.projector.weight", cfg.num_text_layers)?,
            refiner: blocks("refiner_blocks", cfg.num_refiner_text_blocks)?,
            txt_norm: GpuRmsNorm::load(comp, "txt_in.norm.weight", td, cfg.norm_eps)?,
            txt_l1: GpuLinear::load(comp, "txt_in.linear_1", td, d, true)?,
            txt_l2: GpuLinear::load(comp, "txt_in.linear_2", d, d, true)?,
            num_layers: cfg.num_text_layers,
            text_dim: td,
            hidden: d,
        })
    }

    /// `taps` `[n_tokens, num_layers, text_dim]` -> text tokens `[n_tokens,
    /// hidden]`, matching [`super::model::TextEncoderHead::forward`].
    pub fn forward(&self, taps: &[f32], n_tokens: usize) -> Vec<f32> {
        let (nl, td) = (self.num_layers, self.text_dim);
        assert_eq!(taps.len(), n_tokens * nl * td, "taps must be [n_tokens, layers, dim]");
        // The taps include raw residual streams from deep in Qwen3-VL, whose
        // "massive activations" exceed f16's 65504; keep the matmul scratch
        // f32. This runs once per prompt, so the cost is irrelevant.
        crate::metal_backend::set_use_f32_sgemm(true);
        let barrier = crate::metal_backend::set_mps_barrier(true);

        let mut x = GpuTensor::upload_f32_as_f16(vec![n_tokens * nl, td], taps);
        for b in &self.layerwise {
            b.forward(&mut x, n_tokens, nl);
        }
        // Collapse the tap axis. A 12-wide weighted sum per element; cheaper
        // to do here than to write a kernel for.
        let xs = x.download_to_f32();
        let mut fused = vec![0.0_f32; n_tokens * td];
        for (t, out) in fused.chunks_mut(td).enumerate() {
            for (l, w) in self.projector.iter().enumerate() {
                let src = &xs[(t * nl + l) * td..][..td];
                out.iter_mut().zip(src).for_each(|(o, s)| *o += w * s);
            }
        }
        let mut y = GpuTensor::upload_f32_as_f16(vec![n_tokens, td], &fused);
        for b in &self.refiner {
            b.forward(&mut y, 1, n_tokens);
        }
        self.txt_norm.forward(&mut y, n_tokens);
        let mut h = self.txt_l1.forward(&y, n_tokens);
        gelu_f16_gpu(&mut h);
        let out = self.txt_l2.forward(&h, n_tokens).download_to_f32();
        crate::metal_backend::set_use_f32_sgemm(false);
        crate::metal_backend::set_mps_barrier(barrier);
        out
    }
}

/// The timestep-independent part of a forward: text tokens on the GPU and the
/// rotary tables for the whole padded sequence. Build once per prompt (and
/// per CFG branch) with [`GpuKrea2Transformer::prepare`].
pub struct Krea2Context {
    pub txt: GpuTensor,
    pub n_txt: usize,
    /// Rows past `n_img + n_txt`, there only to equalise sequence lengths.
    pub n_pad: usize,
    pub grid: (usize, usize),
    pub rope: GpuRope,
}

impl Krea2Context {
    pub fn rows(&self) -> usize {
        self.grid.0 * self.grid.1 + self.n_txt + self.n_pad
    }
}

/// The full transformer on the GPU.
pub struct GpuKrea2Transformer {
    pub cfg: Krea2Config,
    pub img_in: GpuLinear,
    /// Timestep MLPs stay on the CPU: one row per step against tens of
    /// thousands of tokens through the blocks.
    pub time: TimeEmbed,
    pub text: GpuTextHead,
    pub blocks: Vec<GpuBlock>,
    /// Stored as `[shift row, scale row]` so `final_modulate_f16_gpu` (which
    /// reads shift first, Wan's order) applies Krea's scale-first table.
    pub final_table: Buffer,
    pub final_norm: GpuRmsNorm,
    pub final_linear: GpuLinear,
}

impl GpuKrea2Transformer {
    /// Load straight from a checkpoint into GPU bf16, one tensor at a time.
    /// Peak host memory is one tensor plus the ~1.1 GB of f32 timestep MLPs —
    /// against ~52 GB for an f32 [`super::model::Krea2Transformer`].
    ///
    /// `on_block(i, n)` is called after each transformer block uploads.
    pub fn load(
        cfg: Krea2Config, comp: &ComponentTensors, mut on_block: impl FnMut(usize, usize),
    ) -> Result<Self> {
        // Validate everything before uploading anything, so a bad checkpoint
        // fails in milliseconds rather than 25 GB in.
        let problems = check_inventory(&cfg, comp);
        if !problems.is_empty() {
            return Err(DiffusionError::Layout(format!(
                "Krea 2 transformer checkpoint has {} problem(s), first few: {:?}",
                problems.len(),
                &problems[..problems.len().min(5)]
            )));
        }
        let d = cfg.hidden();
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(GpuBlock::load(comp, &format!("transformer_blocks.{i}"), &cfg)?);
            on_block(i + 1, cfg.num_layers);
        }
        let ft = get_checked(comp, "final_layer.scale_shift_table", 2 * d)?;
        let swapped: Vec<f32> = ft[d..].iter().chain(&ft[..d]).copied().collect();
        Ok(Self {
            img_in: GpuLinear::load(comp, "img_in", cfg.in_channels, d, true)?,
            time: TimeEmbed::load(comp, &cfg)?,
            text: GpuTextHead::load(comp, &cfg)?,
            blocks,
            final_table: upload_f32_as_f16_buffer(&swapped),
            final_norm: GpuRmsNorm::load(comp, "final_layer.norm.weight", d, cfg.norm_eps)?,
            final_linear: GpuLinear::load(comp, "final_layer.linear", d, cfg.in_channels, true)?,
            cfg,
        })
    }

    /// Text conditioning for one prompt: taps `[n, layers, text_dim]` ->
    /// text tokens `[n, hidden]`.
    pub fn encode_text(&self, taps: &[f32], n_tokens: usize) -> Vec<f32> {
        self.text.forward(taps, n_tokens)
    }

    /// Upload one branch's text tokens and build its rotary tables, padded to
    /// `pad_txt_to` text rows (pass the longer branch's length to make both
    /// CFG branches the same size, or `n_txt` for no padding).
    pub fn prepare(&self, txt: &[f32], grid: (usize, usize), pad_txt_to: usize) -> Krea2Context {
        let d = self.cfg.hidden();
        let n_txt = txt.len() / d;
        assert!(pad_txt_to >= n_txt);
        // Round the whole sequence up to a multiple of 8 rows: the fastest
        // attention kernel needs that stride (see `GpuAttention::forward`).
        // Rounding the *padded* length keeps CFG branches equal.
        let n_img = grid.0 * grid.1;
        let n_pad = (n_img + pad_txt_to).next_multiple_of(8) - n_img - n_txt;
        let mut pos = image_positions(grid.0, grid.1);
        pos.extend(std::iter::repeat_n([0.0; 3], n_txt + n_pad));
        let rope = Rope::new(&pos, self.cfg.axes_dims_rope, self.cfg.rope_theta);
        Krea2Context {
            txt: GpuTensor::upload_f32_as_f16(vec![n_txt, d], txt),
            n_txt,
            n_pad,
            grid,
            rope: GpuRope::from_cpu(&rope),
        }
    }

    /// Velocity for packed image tokens `img` `[gh * gw, in_channels]` at flow
    /// time `t` (1 = noise). Matches [`super::model::Krea2Transformer::forward`].
    pub fn forward(&self, img: &[f32], t: f32, ctx: &Krea2Context) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.hidden();
        let n_img = ctx.grid.0 * ctx.grid.1;
        let n_kv = n_img + ctx.n_txt;
        let rows = ctx.rows();
        assert_eq!(img.len(), n_img * cfg.in_channels, "image tokens do not match the context grid");

        // Diagnostic: `KLEARU_KREA2_F32_SGEMM=1` runs the MPS matmuls through
        // the f32 scratch path instead of f16.
        let f32_sgemm = std::env::var_os("KLEARU_KREA2_F32_SGEMM").is_some();
        crate::metal_backend::set_use_f32_sgemm(f32_sgemm);
        // Unfenced, a chain of matmuls at the 6144x16384 FFN shape can still
        // garble a block of rows (see `set_mps_barrier`); fenced, it never has.
        let barrier = crate::metal_backend::set_mps_barrier(true);

        let (temb, temb_mod) = self.time.forward(t);
        let temb_buf = upload_f32_as_f16_buffer(&temb);
        let mod_buf = upload_f32_as_f16_buffer(&temb_mod);

        let img_g = GpuTensor::upload_f32_as_f16(vec![n_img, cfg.in_channels], img);
        let img_tok = self.img_in.forward(&img_g, n_img);
        let mut x = GpuTensor::new_f16(vec![rows, d]);
        if ctx.n_pad > 0 {
            // Padding rows are never keys and never read back, but they are
            // normalised like every other row; zeros keep them finite.
            zero_buffer_gpu(&x.buffer, rows * d * 2);
        }
        blit_copy_f16(&img_tok.buffer, 0, &x.buffer, 0, n_img * d);
        blit_copy_f16(&ctx.txt.buffer, 0, &x.buffer, n_img * d, ctx.n_txt * d);

        prof::start();
        mag::reset();
        for (i, b) in self.blocks.iter().enumerate() {
            mag::set_block(i);
            b.forward(&mut x, rows, n_kv, &mod_buf, &ctx.rope);
            mag::probe("residual", &x, x.elements());
        }
        mag::report();

        prof::phase("tail");
        self.final_norm.forward(&mut x, rows);
        final_modulate_f16_gpu(&mut x, &self.final_table, &temb_buf, rows, d, ModLayout::PerSample);
        let out = self.final_linear.forward(&x, n_img).download_to_f32();
        prof::phase("tail");
        prof::report();
        crate::metal_backend::set_use_f32_sgemm(false);
        crate::metal_backend::set_mps_barrier(barrier);
        out
    }
}
