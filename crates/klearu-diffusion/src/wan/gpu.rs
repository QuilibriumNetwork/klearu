//! Metal GPU path for the Wan DiT.
//!
//! Mirrors [`super::model`], which is the correctness oracle — and, unusually,
//! an oracle that is itself validated: `wan_reference_matches_diffusers` pins
//! the CPU path against diffusers to ~5e-7, so agreeing with it here is a real
//! guarantee rather than two implementations agreeing on the same mistake.
//!
//! Weights upload once as persistent bf16 buffers (see
//! `feedback_metal_bf16_constraints`: `GpuDtype::F16` denotes bf16 storage).
//! Attention runs through `flash_attention_best_f16_gpu`, which at Wan's
//! head_dim of 128 selects the head-dim-split tiled kernel.
//!
//! Gated by `feature = "metal"` at the module declaration in `wan/mod.rs`.

use metal::Buffer;

use crate::layers::Linear;
use crate::weight::loader::ComponentTensors;
use crate::metal_backend::{
    bias_add_f16_gpu, eadd_f16_gpu, final_modulate_f16_gpu, flash_attention_best_f16_gpu,
    gate_rows_f16_gpu, gelu_f16_gpu, layer_norm_f16_gpu, modulate_rows_f16_gpu,
    permute_hl_to_lh_f16_gpu, permute_lh_to_hl_f16_gpu, rms_norm_f16_gpu, rope_pairs_f16_gpu,
    sgemm_f16_a_btrans_buf, sgemm_f16_buf, transpose_outer_f16_gpu, upload_bf16_bytes, GpuDtype, upload_f32_as_f16_buffer, GpuTensor, ModLayout,
};

use super::config::WanDitConfig;
use super::model::{
    patchify, token_grid, unpatchify, Conditioning, WanAttention, WanBlock, WanDit,
};
use super::rope::RopeTables;
use crate::Result;

/// Linear layer with its weight resident on the GPU as bf16.
pub struct GpuLinear {
    pub weight: Buffer,
    pub bias: Option<Buffer>,
    pub in_f: usize,
    pub out_f: usize,
    /// Call-site name, used only by the `KLEARU_WAN_MAG` magnitude probe.
    pub tag: &'static str,
}

impl GpuLinear {
    fn from_cpu(l: &Linear, tag: &'static str) -> Self {
        Self {
            weight: transpose_weight(&l.weight, l.out_features, l.in_features),
            bias: l.bias.as_ref().map(|b| upload_dit_weight(b)),
            in_f: l.in_features,
            out_f: l.out_features,
            tag,
        }
    }

    /// Load straight from the checkpoint into GPU bf16, with no f32
    /// intermediate. For a bf16-native tensor this is mmap -> memcpy -> GPU;
    /// the `Linear` + `from_cpu` route materialises the full f32 weight first,
    /// which for the 5B DiT is ~20 GB of host memory that is then thrown away.
    fn load(
        comp: &ComponentTensors, p: &str, in_f: usize, out_f: usize, tag: &'static str,
    ) -> Result<Self> {
        let w = comp.get_bf16(&format!("{p}.weight"))?;
        let weight = transpose_weight_bf16(&w, out_f, in_f);
        drop(w);
        let bias = {
            let b = comp.get_bf16(&format!("{p}.bias"))?;
            Some(upload_dit_bf16_bytes(&b))
        };
        Ok(Self { weight, bias, in_f, out_f, tag })
    }

    fn forward(&self, x: &GpuTensor, m: usize) -> GpuTensor {
        let mut out = new_dit_tensor(vec![m, self.out_f]);
        // Untransposed matmul against a pre-transposed weight.
        //
        // MPS's B-transposed path is drastically slower for tall-K shapes:
        // measured at `ffn_out`'s 11440x3072x14336, B-transposed runs at
        // 1.52 TF/s and untransposed at 6.98 TF/s — 4.6x, on identical FLOPs.
        // Weights never change, so the transpose is paid once at load.
        if dit_transpose_weights() {
            sgemm_f16_buf(&out.buffer, &x.buffer, &self.weight, m, self.out_f, self.in_f);
        } else {
            sgemm_f16_a_btrans_buf(
                &out.buffer, &x.buffer, &self.weight, m, self.out_f, self.in_f);
        }
        if let Some(b) = &self.bias {
            bias_add_f16_gpu(&mut out, b, 1, self.out_f);
        }
        prof::mag(self.tag, x, &out);
        out
    }
}

/// Upload a `[out, in]` weight transposed to `[in, out]`, on the GPU.
///
/// The transpose runs once per weight at load and lets every `Linear` use the
/// untransposed matmul, which MPS handles far better (see `GpuLinear::forward`).
fn transpose_weight(w: &[f32], out_f: usize, in_f: usize) -> Buffer {
    debug_assert_eq!(w.len(), out_f * in_f);
    let src = upload_dit(vec![out_f, in_f], w);
    if !dit_transpose_weights() {
        return src.buffer;
    }
    transpose_outer_f16_gpu(&src, out_f, in_f, 1).buffer
}

/// Same, from the checkpoint's raw bf16 bytes.
fn transpose_weight_bf16(bytes: &[u8], out_f: usize, in_f: usize) -> Buffer {
    debug_assert_eq!(bytes.len(), out_f * in_f * 2);
    let buf = upload_dit_bf16_bytes(bytes);
    if !dit_transpose_weights() {
        return buf;
    }
    let src = GpuTensor {
        buffer: buf,
        shape: vec![out_f, in_f],
        dtype: if dit_uses_half() { GpuDtype::Half } else { GpuDtype::F16 },
    };
    transpose_outer_f16_gpu(&src, out_f, in_f, 1).buffer
}

/// Storage type for DiT activations.
///
/// IEEE f16 by default (10 mantissa bits vs bfloat's 8, same 2 bytes).
/// Measured against the CPU DiT on real weights, the bfloat path scored
/// cos_sim 0.99943 with a 5% peak relative error — and unlike the VAE, that
/// error lands 100 times per generation with each step feeding the next.
/// `KLEARU_WAN_BF16=1` reverts for A/B.
/// `KLEARU_WAN_NO_TRANSPOSE=1` keeps `Linear` weights in PyTorch's `[out, in]`
/// layout and uses the B-transposed matmul, i.e. the pre-optimisation path.
/// Diagnostic only: it exists to A/B the transpose at generation scale, which
/// is the only place the two can disagree.
pub fn dit_transpose_weights() -> bool {
    thread_local! {
        static T: bool = std::env::var_os("KLEARU_WAN_NO_TRANSPOSE").is_none();
    }
    T.with(|v| *v)
}

pub fn dit_uses_half() -> bool {
    // Read ONCE. `std::env::var_os` calls `getenv`, which takes a process-wide
    // lock; this is consulted on every tensor allocation, and calling it in
    // that hot path collapsed a forward into `_os_unfair_lock_lock_slow` —
    // 50 steps in 30 minutes became zero steps in 15. The rest of this backend
    // caches env flags in thread_locals for exactly this reason.
    thread_local! {
        static HALF: bool = std::env::var_os("KLEARU_WAN_BF16").is_none();
    }
    HALF.with(|v| *v)
}

fn new_dit_tensor(shape: Vec<usize>) -> GpuTensor {
    if dit_uses_half() { GpuTensor::new_half(shape) } else { GpuTensor::new_f16(shape) }
}

fn upload_dit(shape: Vec<usize>, v: &[f32]) -> GpuTensor {
    if dit_uses_half() {
        GpuTensor::upload_f32_as_half(shape, v)
    } else {
        GpuTensor::upload_f32_as_f16(shape, v)
    }
}

/// Checkpoint weights arrive as raw bf16 bytes; convert when the DiT is
/// running half storage. Skipping this leaves bfloat weights being read as
/// half, which is silent garbage (cos_sim 0.379 when I first missed it).
fn upload_dit_bf16_bytes(b: &[u8]) -> Buffer {
    if dit_uses_half() {
        crate::metal_backend::upload_bf16_bytes_as_half(b)
    } else {
        upload_bf16_bytes(b)
    }
}

fn upload_dit_weight(v: &[f32]) -> Buffer {
    if dit_uses_half() {
        crate::metal_backend::upload_f32_as_half_buffer(v)
    } else {
        upload_f32_as_f16_buffer(v)
    }
}

/// Cos/sin tables for one denoising call, flattened to `[seq, head_dim]`.
pub struct GpuRope {
    pub cos: Buffer,
    pub sin: Buffer,
}

impl GpuRope {
    /// Build from the token grid. `rope_pairs_f16` reads cos and sin at the
    /// same even index, which is equivalent to the reference's
    /// `cos[..0::2]` / `sin[..1::2]` because the tables repeat each value
    /// across the pair it rotates.
    pub fn build(tables: &RopeTables, grid: &[(usize, usize, usize)]) -> Self {
        let d = tables.head_dim;
        let mut cos = Vec::with_capacity(grid.len() * d);
        let mut sin = Vec::with_capacity(grid.len() * d);
        for &(f, y, x) in grid {
            let (c, s) = tables.token(f, y, x);
            cos.extend_from_slice(&c);
            sin.extend_from_slice(&s);
        }
        Self {
            cos: upload_dit_weight(&cos),
            sin: upload_dit_weight(&sin),
        }
    }
}

/// Self- or cross-attention on the GPU.
pub struct GpuWanAttention {
    pub to_q: GpuLinear,
    pub to_k: GpuLinear,
    pub to_v: GpuLinear,
    pub to_out: GpuLinear,
    pub norm_q: Buffer,
    pub norm_k: Buffer,
    pub heads: usize,
    pub head_dim: usize,
    pub eps: f32,
}

impl GpuWanAttention {
    fn from_cpu(a: &WanAttention, eps: f32) -> Self {
        Self {
            to_q: GpuLinear::from_cpu(&a.to_q, "to_q"),
            to_k: GpuLinear::from_cpu(&a.to_k, "to_k"),
            to_v: GpuLinear::from_cpu(&a.to_v, "to_v"),
            to_out: GpuLinear::from_cpu(&a.to_out, "to_out"),
            norm_q: upload_dit_weight(&a.norm_q.weight),
            norm_k: upload_dit_weight(&a.norm_k.weight),
            heads: a.heads,
            head_dim: a.head_dim,
            eps,
        }
    }

    fn load(comp: &ComponentTensors, p: &str, cfg: &WanDitConfig) -> Result<Self> {
        let d = cfg.inner_dim();
        let norm = |n: &str| -> Result<Buffer> {
            let v = comp.get_bf16(&format!("{p}.{n}.weight"))?;
            Ok(upload_dit_bf16_bytes(&v))
        };
        Ok(Self {
            to_q: GpuLinear::load(comp, &format!("{p}.to_q"), d, d, "to_q")?,
            to_k: GpuLinear::load(comp, &format!("{p}.to_k"), d, d, "to_k")?,
            to_v: GpuLinear::load(comp, &format!("{p}.to_v"), d, d, "to_v")?,
            to_out: GpuLinear::load(comp, &format!("{p}.to_out.0"), d, d, "to_out")?,
            norm_q: norm("norm_q")?,
            norm_k: norm("norm_k")?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            eps: cfg.eps,
        })
    }

    fn forward(
        &self,
        x: &GpuTensor,
        n_q: usize,
        kv: &GpuTensor,
        n_kv: usize,
        rope: Option<&GpuRope>,
    ) -> GpuTensor {
        let inner = self.heads * self.head_dim;
        let (h, dh) = (self.heads, self.head_dim);

        prof::phase("attn.qkv_matmul");
        let mut q = self.to_q.forward(x, n_q);
        let mut k = self.to_k.forward(kv, n_kv);
        let v = self.to_v.forward(kv, n_kv);

        // "rms_norm_across_heads": one norm over the whole inner_dim, applied
        // before the head split — so `dim` here is `inner`, not `head_dim`.
        prof::stage("attn.q_matmul", &q);
        prof::stage("attn.k_matmul", &k);
        prof::stage("attn.v_matmul", &v);
        prof::phase("attn.qk_norm");
        rms_norm_f16_gpu(&mut q, &self.norm_q, n_q, inner, self.eps);
        rms_norm_f16_gpu(&mut k, &self.norm_k, n_kv, inner, self.eps);

        prof::phase("attn.rope");
        if let Some(r) = rope {
            rope_pairs_f16_gpu(&mut q, &r.cos, &r.sin, n_q, h, dh);
            rope_pairs_f16_gpu(&mut k, &r.cos, &r.sin, n_kv, h, dh);
        }

        // Flash attention wants head-major [h, l, d].
        prof::stage("attn.after_rope_q", &q);
        prof::stage("attn.after_rope_k", &k);
        prof::phase("attn.permute");
        let qh = permute_lh_to_hl_f16_gpu(&q, 1, n_q, h, dh);
        let kh = permute_lh_to_hl_f16_gpu(&k, 1, n_kv, h, dh);
        let vh = permute_lh_to_hl_f16_gpu(&v, 1, n_kv, h, dh);
        let mut oh = new_dit_tensor(vec![h * n_q * dh]);
        let scale = 1.0 / (dh as f32).sqrt();
        prof::phase("attn.flash");
        // Flash attention's simdgroup kernels are bfloat-only, so convert
        // around them. That is one rounding per block rather than one per op.
        if qh.dtype.is_half() {
            let qb = crate::metal_backend::tensor_half_to_bf16(&qh);
            let kb = crate::metal_backend::tensor_half_to_bf16(&kh);
            let vb = crate::metal_backend::tensor_half_to_bf16(&vh);
            let mut ob = GpuTensor::new_f16(vec![h * n_q * dh]);
            flash_attention_best_f16_gpu(&qb, &kb, &vb, &mut ob, 1, h, n_q, n_kv, dh, scale);
            oh = crate::metal_backend::tensor_bf16_to_half(&ob);
        } else {
            flash_attention_best_f16_gpu(&qh, &kh, &vh, &mut oh, 1, h, n_q, n_kv, dh, scale);
        }

        prof::stage("attn.qh", &qh);
        prof::stage("attn.kh", &kh);
        prof::stage("attn.vh", &vh);
        prof::stage("attn.flash_out", &oh);
        prof::phase("attn.permute");
        let ctx = permute_hl_to_lh_f16_gpu(&oh, 1, n_q, h, dh);
        prof::phase("attn.out_matmul");
        let r = self.to_out.forward(&ctx, n_q);
        prof::phase("attn.tail");
        r
    }
}

/// One transformer block on the GPU.
pub struct GpuWanBlock {
    pub attn1: GpuWanAttention,
    pub attn2: GpuWanAttention,
    /// norm2's affine parameters; norm1 and norm3 have none.
    pub norm2_gamma: Buffer,
    pub norm2_beta: Buffer,
    pub ffn_proj: GpuLinear,
    pub ffn_out: GpuLinear,
    pub scale_shift_table: Buffer,
    pub dim: usize,
    pub eps: f32,
}

impl GpuWanBlock {
    fn from_cpu(b: &WanBlock, eps: f32) -> Self {
        Self {
            attn1: GpuWanAttention::from_cpu(&b.attn1, eps),
            attn2: GpuWanAttention::from_cpu(&b.attn2, eps),
            norm2_gamma: upload_dit_weight(&b.norm2.gamma),
            norm2_beta: upload_dit_weight(&b.norm2.beta),
            ffn_proj: GpuLinear::from_cpu(&b.ffn_proj, "ffn_proj"),
            ffn_out: GpuLinear::from_cpu(&b.ffn_out, "ffn_out"),
            scale_shift_table: upload_dit_weight(&b.scale_shift_table),
            dim: b.dim,
            eps,
        }
    }

    fn load(comp: &ComponentTensors, p: &str, cfg: &WanDitConfig) -> Result<Self> {
        let d = cfg.inner_dim();
        let raw = |n: &str| -> Result<Buffer> {
            let v = comp.get_bf16(&format!("{p}.{n}"))?;
            Ok(upload_dit_bf16_bytes(&v))
        };
        Ok(Self {
            attn1: GpuWanAttention::load(comp, &format!("{p}.attn1"), cfg)?,
            attn2: GpuWanAttention::load(comp, &format!("{p}.attn2"), cfg)?,
            norm2_gamma: raw("norm2.weight")?,
            norm2_beta: raw("norm2.bias")?,
            ffn_proj: GpuLinear::load(comp, &format!("{p}.ffn.net.0.proj"), d, cfg.ffn_dim, "ffn_proj")?,
            ffn_out: GpuLinear::load(comp, &format!("{p}.ffn.net.2"), cfg.ffn_dim, d, "ffn_out")?,
            scale_shift_table: raw("scale_shift_table")?,
            dim: d,
            eps: cfg.eps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &mut GpuTensor,
        seq: usize,
        enc: &GpuTensor,
        n_ctx: usize,
        temb: &Buffer,
        layout: ModLayout,
        rope: &GpuRope,
        ones: &Buffer,
        zeros: &Buffer,
    ) {
        let d = self.dim;

        // 1. Self-attention: norm (no affine) -> modulate -> attend -> gate.
        prof::phase("block.norm_mod");
        prof::stage("block.input_x", x);
        let mut h = layer_norm_out(x, ones, zeros, d, self.eps);
        prof::stage("block.after_ln", &h);
        modulate_rows_f16_gpu(&mut h, &self.scale_shift_table, temb, seq, d, layout, 1, 0);
        prof::stage("block.after_modulate", &h);
        let mut attn = self.attn1.forward(&h, seq, &h, seq, Some(rope));
        prof::phase("block.gate_add");
        gate_rows_f16_gpu(&mut attn, &self.scale_shift_table, temb, seq, d, layout, 2);
        eadd_f16_gpu(x, &attn);

        // 2. Cross-attention: norm (affine) -> attend -> add. No modulation.
        prof::phase("block.norm_mod");
        let h = layer_norm_out(x, &self.norm2_gamma, &self.norm2_beta, d, self.eps);
        let attn = self.attn2.forward(&h, seq, enc, n_ctx, None);
        prof::phase("block.gate_add");
        eadd_f16_gpu(x, &attn);

        // 3. Feed-forward: norm (no affine) -> modulate -> GELU MLP -> gate.
        prof::phase("block.norm_mod");
        let mut h = layer_norm_out(x, ones, zeros, d, self.eps);
        modulate_rows_f16_gpu(&mut h, &self.scale_shift_table, temb, seq, d, layout, 4, 3);
        prof::phase("ffn.proj_matmul");
        let mut ff = self.ffn_proj.forward(&h, seq);
        // `gelu_f16` is the tanh formulation with an overflow clamp, which is
        // what `gelu-approximate` means — not the erf variant.
        prof::phase("ffn.gelu");
        gelu_f16_gpu(&mut ff);
        prof::phase("ffn.out_matmul");
        let mut ff = self.ffn_out.forward(&ff, seq);
        prof::phase("block.gate_add");
        gate_rows_f16_gpu(&mut ff, &self.scale_shift_table, temb, seq, d, layout, 5);
        eadd_f16_gpu(x, &ff);
    }
}

/// Env-gated phase profiler (`KLEARU_WAN_PROF=1`).
///
/// Each `phase` call flushes the GPU queue and charges the elapsed wall time
/// to a named bucket. The flushes serialize what would otherwise overlap, so
/// the *total* is inflated; the point is the relative split, which is what
/// tells you which phase to attack.
pub(crate) mod prof {
    use std::cell::RefCell;

    thread_local! {
        static ON: bool = std::env::var("KLEARU_WAN_PROF").is_ok();
        static ACC: RefCell<Vec<(&'static str, f64)>> = const { RefCell::new(Vec::new()) };
        /// The currently *open* phase: the name of the work that follows the
        /// last `phase()` call, and when it started.
        static OPEN: RefCell<Option<(&'static str, std::time::Instant)>>
            = const { RefCell::new(None) };
    }

    pub fn on() -> bool { ON.with(|v| *v) }

    /// Close whatever phase is open — charging its elapsed time to *its own*
    /// name, not to `name` — and open `name` for the code that follows.
    ///
    /// Getting this backwards is easy and silently shifts every attribution by
    /// one call site, which reads as a plausible profile of the wrong thing.
    pub fn phase(name: &'static str) {
        if !on() { return; }
        crate::metal_backend::flush();
        let now = std::time::Instant::now();
        OPEN.with(|l| {
            if let Some((prev_name, prev_t)) = l.borrow_mut().replace((name, now)) {
                let dt = now.duration_since(prev_t).as_secs_f64();
                ACC.with(|a| {
                    let mut a = a.borrow_mut();
                    match a.iter_mut().find(|(n, _)| *n == prev_name) {
                        Some(e) => e.1 += dt,
                        None => a.push((prev_name, dt)),
                    }
                });
            }
        });
    }

    /// Start a fresh measurement window, discarding any open phase.
    pub fn start() {
        if !on() { return; }
        crate::metal_backend::flush();
        OPEN.with(|l| *l.borrow_mut() = None);
    }

    thread_local! {
        static MAG_ON: bool = std::env::var("KLEARU_WAN_MAG").is_ok();
        /// tag -> (max |input|, max |output|)
        static MAG: RefCell<Vec<(&'static str, f32, f32)>> = const { RefCell::new(Vec::new()) };
    }

    /// Record the peak input and output magnitude of one matmul.
    ///
    /// This is what decides whether a matmul can use the fast f16 sgemm
    /// scratch: MPS accumulates in fp32 internally, so only the **output**
    /// cast is limited to 65504. A matmul whose result stays well under that
    /// does not need the 2.6x-slower f32 path.
    pub fn mag(tag: &'static str, x: &crate::metal_backend::GpuTensor,
               out: &crate::metal_backend::GpuTensor) {
        if !MAG_ON.with(|v| *v) { return; }
        let peak = |t: &crate::metal_backend::GpuTensor| {
            t.download_to_f32().iter().fold(0.0_f32, |a, v| {
                if v.is_finite() { a.max(v.abs()) } else { f32::INFINITY }
            })
        };
        let (pi, po) = (peak(x), peak(out));
        MAG.with(|m| {
            let mut m = m.borrow_mut();
            match m.iter_mut().find(|(n, _, _)| *n == tag) {
                Some(e) => { e.1 = e.1.max(pi); e.2 = e.2.max(po); }
                None => m.push((tag, pi, po)),
            }
        });
    }

    fn mag_report() {
        if !MAG_ON.with(|v| *v) { return; }
        MAG.with(|m| {
            let mut m = m.borrow_mut();
            println!("      --- matmul peak magnitudes (f16 output cast clips at 65504) ---");
            for (n, i, o) in m.iter() {
                let verdict = if *o > 60000.0 { "NEEDS f32" } else { "f16 ok" };
                println!("      {n:<14} max|in|={i:11.1}  max|out|={o:11.1}   {verdict}");
            }
            m.clear();
        });
    }


    /// `KLEARU_WAN_STAGE=1`: report the first stage inside a block whose output
    /// goes non-finite. Block-level checking says "block 0 broke"; this says
    /// which of the dozen ops in it did, which is the difference between
    /// reading one kernel and reading all of them.
    pub fn stage(name: &'static str, t: &crate::metal_backend::GpuTensor) {
        thread_local! {
            static ON: bool = std::env::var("KLEARU_WAN_STAGE").is_ok();
            static DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        if !ON.with(|v| *v) || DONE.with(|d| d.get()) { return; }
        let v = t.download_to_f32();
        let bad = v.iter().filter(|x| !x.is_finite()).count();
        let mx = v.iter().filter(|x| x.is_finite()).fold(0.0f32, |a, x| a.max(x.abs()));
        if bad > 0 {
            println!("[stage] {name}: {bad}/{} NON-FINITE (max|finite|={mx:.3e})", v.len());
            DONE.with(|d| d.set(true));
        } else {
            println!("[stage] {name}: ok  max={mx:.3e}  n={}", v.len());
        }
    }

    /// Print the accumulated split and reset.
    pub fn report() {
        mag_report();
        if !on() { return; }
        crate::metal_backend::flush();
        ACC.with(|a| {
            let mut a = a.borrow_mut();
            let total: f64 = a.iter().map(|(_, t)| t).sum();
            a.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
            println!("      --- phase profile (total {total:.2}s) ---");
            for (n, t) in a.iter() {
                println!("      {n:<22} {t:7.3}s  {:5.1}%", 100.0 * t / total);
            }
            a.clear();
        });
        OPEN.with(|l| *l.borrow_mut() = None);
    }
}

/// LayerNorm into a fresh tensor, leaving the source untouched.
fn layer_norm_out(
    src: &GpuTensor,
    gamma: &Buffer,
    beta: &Buffer,
    d: usize,
    eps: f32,
) -> GpuTensor {
    let mut out = src.clone_data();
    layer_norm_f16_gpu(&mut out, gamma, beta, d, eps);
    out
}

/// The part of a forward that depends only on the prompt and the latent grid,
/// not on the timestep.
///
/// Both were being rebuilt on every forward even though they are identical
/// across all of them: `embed_text` is ~10 GFLOP of scalar CPU matmul over the
/// 226-token context, and the RoPE tables are ~1.5M floats per trig table built
/// on the host and uploaded. A 50-step CFG generation calls forward 100 times
/// and needs exactly two of these (one per guidance branch).
pub struct GpuWanContext {
    pub enc: GpuTensor,
    pub n_ctx: usize,
    pub rope: GpuRope,
    /// Grid this was built for; `forward_prepared` checks it rather than
    /// trusting the caller to pair them correctly.
    pub grid: (usize, usize, usize),
}

/// GPU Wan DiT. Build with [`GpuWanDit::from_cpu`].
pub struct GpuWanDit {
    pub cfg: WanDitConfig,
    pub patch_embedding: GpuLinear,
    pub blocks: Vec<GpuWanBlock>,
    pub proj_out: GpuLinear,
    pub scale_shift_table: Buffer,
    /// The small timestep/text embedders, kept on the host. A few hundred MB
    /// against the blocks' 20 GB, and they run once per step over a few
    /// hundred rows — owning a copy is what lets the f32 `WanDit` be dropped.
    pub cond: Conditioning,
    pub rope: RopeTables,
    /// Identity affine parameters for the three non-affine LayerNorms.
    ones: Buffer,
    zeros: Buffer,
}

impl GpuWanDit {
    pub fn from_cpu(m: &WanDit) -> Self {
        let d = m.cfg.inner_dim();
        Self {
            patch_embedding: GpuLinear::from_cpu(&m.patch_embedding, "patch_embed"),
            blocks: m.blocks.iter().map(|b| GpuWanBlock::from_cpu(b, m.cfg.eps)).collect(),
            proj_out: GpuLinear::from_cpu(&m.proj_out, "proj_out"),
            scale_shift_table: upload_dit_weight(&m.scale_shift_table),
            cond: m.cond.clone(),
            rope: m.rope.clone(),
            ones: upload_dit_weight(&vec![1.0_f32; d]),
            zeros: upload_dit_weight(&vec![0.0_f32; d]),
            cfg: m.cfg.clone(),
        }
    }

    /// Load straight from a checkpoint into GPU bf16.
    ///
    /// Peak host memory is one tensor at a time plus the ~350 MB of
    /// conditioning weights, instead of the ~20 GB `WanDit::load` +
    /// `from_cpu` needs (a full f32 model, then a bf16 GPU copy on top of it).
    pub fn load(cfg: WanDitConfig, comp: &ComponentTensors) -> Result<Self> {
        // Validate the whole inventory before uploading anything, so a bad
        // checkpoint fails immediately rather than 20 GB in.
        let probe = WanDit::new_shallow(cfg.clone());
        let missing = probe.missing_tensors(comp);
        if !missing.is_empty() {
            return Err(crate::DiffusionError::Layout(format!(
                "Wan DiT checkpoint is missing {} tensor(s), first few: {:?}",
                missing.len(),
                &missing[..missing.len().min(5)]
            )));
        }
        let bad = probe.mismatched_tensors(comp);
        if !bad.is_empty() {
            return Err(crate::DiffusionError::Layout(format!(
                "Wan DiT checkpoint has {} tensor(s) of unexpected size, first few: {:?}",
                bad.len(),
                &bad[..bad.len().min(5)]
            )));
        }
        drop(probe);

        let d = cfg.inner_dim();
        let (pt, ph, pw) = cfg.patch();
        let patch_vol = pt * ph * pw;

        let mut cond = Conditioning::new(&cfg);
        super::weight::load_conditioning(&mut cond, comp)?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(GpuWanBlock::load(comp, &format!("blocks.{i}"), &cfg)?);
        }

        let sst = comp.get_bf16("scale_shift_table")?;
        Ok(Self {
            patch_embedding: GpuLinear::load(
                comp, "patch_embedding", cfg.in_channels * patch_vol, d, "patch_embed")?,
            blocks,
            proj_out: GpuLinear::load(
                comp, "proj_out", d, patch_vol * cfg.out_channels, "proj_out")?,
            scale_shift_table: upload_dit_bf16_bytes(&sst),
            cond,
            rope: RopeTables::new(&cfg, 10_000.0),
            ones: upload_dit_weight(&vec![1.0_f32; d]),
            zeros: upload_dit_weight(&vec![0.0_f32; d]),
            cfg,
        })
    }

    /// One denoising step, matching [`WanDit::forward`].
    ///
    /// The conditioning MLPs (timestep and text embedders) stay on the CPU:
    /// they run once per step over at most a few hundred rows, against 30
    /// blocks over tens of thousands of tokens.
    /// Build the timestep-independent part of a forward once.
    pub fn prepare(
        &self,
        dims: (usize, usize, usize),
        enc: &[f32],
        n_ctx: usize,
    ) -> GpuWanContext {
        let (pt, ph, pw) = self.cfg.patch();
        let (ppf, pph, ppw) = (dims.0 / pt, dims.1 / ph, dims.2 / pw);
        let enc_proj = self.cond.embed_text(enc, n_ctx);
        let enc_gpu = upload_dit(vec![enc_proj.len()], &enc_proj);
        let grid = token_grid(ppf, pph, ppw);
        GpuWanContext {
            enc: enc_gpu,
            n_ctx,
            rope: GpuRope::build(&self.rope, &grid),
            grid: (ppf, pph, ppw),
        }
    }

    /// Convenience wrapper that builds the context and throws it away. Prefer
    /// `prepare` + `forward_prepared` in a denoising loop.
    pub fn forward(
        &self,
        z: &[f32],
        dims: (usize, usize, usize),
        timesteps: &[f32],
        enc: &[f32],
        n_ctx: usize,
    ) -> Vec<f32> {
        let ctx = self.prepare(dims, enc, n_ctx);
        self.forward_prepared(z, dims, timesteps, &ctx)
    }

    pub fn forward_prepared(
        &self,
        z: &[f32],
        dims: (usize, usize, usize),
        timesteps: &[f32],
        ctx: &GpuWanContext,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.inner_dim();
        let (pt, ph, pw) = cfg.patch();
        let (t_len, h_len, w_len) = dims;
        let (ppf, pph, ppw) = (t_len / pt, h_len / ph, w_len / pw);
        let seq = ppf * pph * ppw;

        // f32 sgemm scratch. Two separate reasons, neither of them the one
        // originally written here (overflow):
        //
        //   1. The f16 path is intermittently non-finite. Same binary, same
        //      input, `KLEARU_WAN_SGEMM_AB=1` at 5070 tokens: one run gives
        //      cos_sim 0.99980 against f32, the next gives 973440/973440
        //      non-finite. The f32 path has never failed this way — 0 in 100
        //      forwards in the same harness.
        //   2. Even when it succeeds it is only ~1.3-1.5x faster, for
        //      cos_sim 0.99980 / max_abs_diff 6.6e-2. The f32 path matches
        //      diffusers to 1.5e-5. That exactness is worth more than 1.4x.
        //
        // It is NOT overflow: `KLEARU_WAN_MAG=1` measures peak magnitudes in
        // and out of every matmul, and across 4 denoising steps on real
        // weights the largest value anywhere is ~154 (`ffn_out`) against the
        // f16 output cast's 65504 limit. Nor is bf16 -> f16 lossy at these
        // magnitudes — f16 has the *wider* mantissa. Whatever is wrong lives
        // inside MPS's f16 sgemm.
        //
        // `KLEARU_WAN_F16_SGEMM=1` opts in anyway, for measurement.
        // `KLEARU_WAN_BF16_GEMM=1` routes the B-transposed matmuls (every
        // Linear in the model) through the native bf16 simdgroup-MMA kernel,
        // which bypasses the MPS scratch cast entirely.
        let bf16_gemm = std::env::var_os("KLEARU_WAN_BF16_GEMM").is_some();
        crate::metal_backend::set_use_bf16_gemm(bf16_gemm);
        crate::metal_backend::set_sgemm_half_io(dit_uses_half());
        // f16 MPS compute, not f32. `KLEARU_WAN_F32_SGEMM=1` forces the older
        // path back.
        //
        // The f32 default existed because an f16 scratch clips at 65504, which
        // really does bite elsewhere — the SDXL VAE decoder reaches ~23K. This
        // model is nowhere near it. Peak matmul magnitudes at native
        // resolution, from `KLEARU_WAN_MAG=1`:
        //
        //     ffn_out 203.6   to_out 79.3   ffn_proj 57.4
        //     to_q 55.8       to_k 50.6     everything else < 25
        //
        // A 322x margin on the largest. And it costs nothing measurable: the
        // DiT against diffusers scores cos_sim 0.99998536 in f16 against
        // 0.99998611 in f32, because accuracy here is bounded by bf16 *storage*
        // rather than by the compute precision. The matmul phases drop 24% and
        // the forward 44.6s -> 39.9s.
        crate::metal_backend::set_use_f32_sgemm(
            !bf16_gemm && std::env::var_os("KLEARU_WAN_F32_SGEMM").is_some(),
        );

        // NOTE: wrapping this forward in `begin_batch()` was tried and is
        // WRONG here. It produced all-NaN output and no speedup. The
        // f32-scratch matmul path acquires pooled scratch buffers and releases
        // them immediately; with a batch open nothing has executed yet, so the
        // next matmul re-acquires the same buffers and overwrites them before
        // the GPU reads them. Batching is only safe for paths that do not
        // recycle scratch through BUF_POOL mid-sequence.
        let patches = patchify(cfg, z, cfg.in_channels, t_len, h_len, w_len);
        let pin = upload_dit(vec![patches.len()], &patches);
        let mut x = self.patch_embedding.forward(&pin, seq);

        assert_eq!(
            ctx.grid, (ppf, pph, ppw),
            "context was prepared for a different latent grid"
        );
        let enc_gpu = &ctx.enc;
        let n_ctx = ctx.n_ctx;
        let rope = &ctx.rope;

        let per_token = timesteps.len() > 1;
        let layout = if per_token { ModLayout::PerToken } else { ModLayout::PerSample };

        // Deduplicate before embedding. `timesteps` has one entry per token —
        // 1560 at 480p — but the timestep MLP is 256->3072->3072->18432
        // (~67M MACs) and runs on the CPU. Evaluating it per token was ~209
        // GFLOP of scalar work per forward and dominated everything: the
        // matmuls are only ~3.7 s of a 101 s forward.
        //
        // Text-to-video uses a single value for every token; image-to-video
        // uses two (0 for the conditioned frames, t for the rest). So this
        // collapses 1560 evaluations to one or two.
        let mut uniq: Vec<f32> = Vec::new();
        let mut idx_of: Vec<usize> = Vec::with_capacity(timesteps.len());
        for &t in timesteps {
            match uniq.iter().position(|u| u.to_bits() == t.to_bits()) {
                Some(i) => idx_of.push(i),
                None => {
                    uniq.push(t);
                    idx_of.push(uniq.len() - 1);
                }
            }
        }
        let embedded: Vec<(Vec<f32>, Vec<f32>)> =
            uniq.iter().map(|&t| self.cond.embed_timestep(t)).collect();

        let mut temb_flat = vec![0.0_f32; if per_token { seq * d } else { d }];
        let mut proj_flat = vec![0.0_f32; if per_token { seq * 6 * d } else { 6 * d }];
        if per_token {
            for (i, &u) in idx_of.iter().enumerate() {
                let (te, pr) = &embedded[u];
                temb_flat[i * d..(i + 1) * d].copy_from_slice(te);
                proj_flat[i * 6 * d..(i + 1) * 6 * d].copy_from_slice(pr);
            }
        } else {
            let (te, pr) = &embedded[0];
            temb_flat.copy_from_slice(te);
            proj_flat.copy_from_slice(pr);
        }

        let temb_buf = upload_dit_weight(&temb_flat);
        let proj_buf = upload_dit_weight(&proj_flat);

        // KLEARU_WAN_NAN_CHECK=1 downloads the residual stream after each
        // block and reports the first one to go non-finite. Slow (a full
        // readback per block) but it turns "the forward returned NaN" into
        // "block N did", which is the difference between auditing 30 blocks
        // and looking at one.
        let nan_check = std::env::var_os("KLEARU_WAN_NAN_CHECK").is_some();
        if nan_check {
            crate::metal_backend::gpu_memory_report("before blocks");
        }
        prof::start();
        for (bi, b) in self.blocks.iter().enumerate() {
            b.forward(
                &mut x, seq, enc_gpu, n_ctx, &proj_buf, layout, rope, &self.ones, &self.zeros,
            );
            if nan_check {
                let h = x.download_to_f32();
                let bad = h.iter().filter(|v| !v.is_finite()).count();
                let m = h.iter().filter(|v| v.is_finite()).fold(0.0_f32, |a, v| a.max(v.abs()));
                if bad > 0 {
                    eprintln!("[wan] block {bi}: {bad}/{} non-finite, max|finite|={m:.3e}",
                              h.len());
                    panic!("residual stream went non-finite at block {bi}");
                }
                if bi == 0 {
                    crate::metal_backend::gpu_memory_report("after block 0");
                }
                if bi % 6 == 0 || bi + 1 == self.blocks.len() {
                    eprintln!("[wan] block {bi}: ok, max|x|={m:.3e}");
                }
            }
        }

        prof::phase("tail");
        layer_norm_f16_gpu(&mut x, &self.ones, &self.zeros, d, cfg.eps);
        final_modulate_f16_gpu(&mut x, &self.scale_shift_table, &temb_buf, seq, d, layout);
        let tokens = self.proj_out.forward(&x, seq);
        // `download_to_f32` flushes, closing the batch.
        let out = unpatchify(cfg, &tokens.download_to_f32(), t_len, h_len, w_len);
        prof::phase("tail");
        prof::report();
        crate::metal_backend::set_use_f32_sgemm(false);
        crate::metal_backend::set_use_bf16_gemm(false);
        crate::metal_backend::set_sgemm_half_io(false);
        out
    }
}
