//! Multi-head attention. Used by the SD UNet's Transformer2DModel
//! (self + cross), the VAE mid-block (self only), and CLIP text models
//! (self with causal mask).
//!
//! Operates on a flattened sequence: query [N×L_q×D_q], key/value
//! [N×L_kv×D_kv]. For self-attention, kv == q. For 2D self-attention
//! in the UNet, the latent grid is reshaped to [N, H*W, C] and back.

use std::cell::RefCell;

use rayon::prelude::*;

use crate::layers::Linear;

pub struct Attention {
    pub to_q: Linear, // [D_q → D_inner]
    pub to_k: Linear, // [D_kv → D_inner]
    pub to_v: Linear, // [D_kv → D_inner]
    pub to_out: Linear, // [D_inner → D_q]
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    /// CPU-side precomputed (K, V) projections. Used by `forward` (CPU path).
    pub cached_kv: RefCell<Option<(Vec<f32>, Vec<f32>)>>,
    /// GPU-side precomputed (K, V) projections, fp16. Used by `forward_gpu`.
    /// Holds shape `(n, l_kv)` alongside the buffers for size validation.
    /// Eliminates ~3500 redundant K/V projections per SDXL inference run
    /// (50 forwards × 70 cross-attn layers × 2 = wasted matmuls without cache).
    #[cfg(feature = "metal")]
    pub cached_kv_gpu: RefCell<Option<CachedKvGpu>>,
    /// Lazily-built concatenated `[W_q | W_k | W_v]` weight `[3·inner, query_dim]`,
    /// row-major, used by `forward_gpu_self`. Replaces three separate
    /// projections with one wider sgemm — fewer dispatches, better cache
    /// reuse on the input. Built on first call when query_dim == kv_dim.
    pub qkv_concat_weight: RefCell<Option<Vec<f32>>>,
}

#[cfg(feature = "metal")]
pub struct CachedKvGpu {
    pub k: metal::Buffer,
    pub v: metal::Buffer,
    pub n: usize,
    pub l_kv: usize,
}

#[cfg(feature = "metal")]
unsafe impl Send for CachedKvGpu {}
#[cfg(feature = "metal")]
unsafe impl Sync for CachedKvGpu {}

impl Attention {
    pub fn new(
        query_dim: usize,
        kv_dim: usize,
        num_heads: usize,
        head_dim: usize,
        out_bias: bool,
    ) -> Self {
        Self::new_with_qkv_bias(query_dim, kv_dim, num_heads, head_dim, out_bias, false)
    }

    /// CLIP / GPT-style attention has bias on q_proj/k_proj/v_proj. SD's UNet
    /// attention does not. Pick the right constructor for your use case.
    /// `qkv_bias=true` is required to correctly load OpenAI CLIP weights —
    /// without it, `load_linear` silently drops `q_proj.bias` etc.
    pub fn new_with_qkv_bias(
        query_dim: usize,
        kv_dim: usize,
        num_heads: usize,
        head_dim: usize,
        out_bias: bool,
        qkv_bias: bool,
    ) -> Self {
        let inner = num_heads * head_dim;
        Self {
            to_q: Linear::new(query_dim, inner, qkv_bias),
            to_k: Linear::new(kv_dim, inner, qkv_bias),
            to_v: Linear::new(kv_dim, inner, qkv_bias),
            to_out: Linear::new(inner, query_dim, out_bias),
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            cached_kv: RefCell::new(None),
            #[cfg(feature = "metal")]
            cached_kv_gpu: RefCell::new(None),
            qkv_concat_weight: RefCell::new(None),
        }
    }

    /// Precompute K = kv·Wk and V = kv·Wv from a fixed kv_in (typically a
    /// text embedding). Subsequent `forward` calls will use these cached
    /// projections. Call once before the diffusion loop; clear with
    /// `clear_kv_cache()` when done with that text.
    pub fn precompute_kv(&self, kv_in: &[f32], n: usize, l_kv: usize) {
        let inner = self.num_heads * self.head_dim;
        let mut k = vec![0.0f32; n * l_kv * inner];
        let mut v = vec![0.0f32; n * l_kv * inner];
        self.to_k.forward_batch(kv_in, &mut k);
        self.to_v.forward_batch(kv_in, &mut v);
        *self.cached_kv.borrow_mut() = Some((k, v));
    }

    /// Drop any precomputed K/V cache (CPU and GPU).
    pub fn clear_kv_cache(&self) {
        *self.cached_kv.borrow_mut() = None;
        #[cfg(feature = "metal")]
        {
            *self.cached_kv_gpu.borrow_mut() = None;
        }
    }

    /// GPU-resident K/V precompute, with **f32 precision for the projection
    /// matmul**. SDXL CLIP-G text embeddings have extreme attention-sink
    /// values (~±85 at BOS/EOS positions vs mean_abs ~0.24 elsewhere). Doing
    /// `kv_in · Wᵀ` over 2048 input dims with fp16 accumulators loses the
    /// per-channel signal — sums of ~2048 products spanning [0.01, 4] in
    /// magnitude exceed fp16's ~1024 representable integers, so most of the
    /// prompt's actual contribution gets quantized away. We do the matmul on
    /// CPU in fp32 (Accelerate sgemm), then upload the result to GPU as fp16
    /// for the per-step attention dispatch.
    ///
    /// One-time cost per generation: ~2 × ~70 cross-attn × 154 × 1280 × 2048
    /// ≈ 56G fp32 FLOPs ≈ ~50ms on AMX. Negligible vs. the 25-step UNet body.
    #[cfg(feature = "metal")]
    pub fn precompute_kv_gpu_from_f32(&self, kv_in_f32: &[f32], n: usize, l_kv: usize) {
        use crate::metal_backend::*;
        let inner = self.num_heads * self.head_dim;
        let mut k_f32 = vec![0.0f32; n * l_kv * inner];
        let mut v_f32 = vec![0.0f32; n * l_kv * inner];
        // CPU fp32 forward — uses Accelerate (or matrixmultiply) sgemm.
        self.to_k.forward_batch(kv_in_f32, &mut k_f32);
        self.to_v.forward_batch(kv_in_f32, &mut v_f32);
        // Upload as fp16 for the GPU attention dispatch.
        let k_gpu = GpuTensor::upload_f32_as_f16(vec![n * l_kv, inner], &k_f32);
        let v_gpu = GpuTensor::upload_f32_as_f16(vec![n * l_kv, inner], &v_f32);
        *self.cached_kv_gpu.borrow_mut() = Some(CachedKvGpu {
            k: k_gpu.buffer,
            v: v_gpu.buffer,
            n,
            l_kv,
        });
    }

    /// Legacy fp16-throughout path. Keep for benchmarking; the `_from_f32`
    /// variant above is correct for SDXL's CLIP-G activations.
    #[cfg(feature = "metal")]
    pub fn precompute_kv_gpu(&self, kv_in: &crate::metal_backend::GpuTensor, n: usize, l_kv: usize) {
        use crate::metal_backend::*;
        debug_assert_eq!(kv_in.dtype, GpuDtype::F16);
        let k = self.to_k.forward_gpu(kv_in);
        let v = self.to_v.forward_gpu(kv_in);
        *self.cached_kv_gpu.borrow_mut() = Some(CachedKvGpu {
            k: k.buffer,
            v: v.buffer,
            n,
            l_kv,
        });
    }

    /// Forward: q [n*l_q*query_dim], kv [n*l_kv*kv_dim]; outputs into `out` of size [n*l_q*query_dim].
    /// For self-attention, pass the same buffer twice.
    pub fn forward(
        &self,
        q_in: &[f32],
        kv_in: &[f32],
        n: usize,
        l_q: usize,
        l_kv: usize,
        causal: bool,
        out: &mut [f32],
    ) {
        let inner = self.num_heads * self.head_dim;
        let h = self.num_heads;
        let d = self.head_dim;

        // Project Q. Use cached K, V if precomputed; else compute Q/K/V in parallel.
        let mut q = vec![0.0f32; n * l_q * inner];
        let has_cache = self.cached_kv.borrow().is_some();
        let mut k_owned: Vec<f32> = Vec::new();
        let mut v_owned: Vec<f32> = Vec::new();
        if has_cache {
            self.to_q.forward_batch(q_in, &mut q);
        } else {
            // No cache — three independent matmuls. Run in parallel via rayon::scope.
            // (When `accelerate` feature is on, sgemm parallelises internally; the
            // outer rayon may add only marginal speedup or none, but doesn't hurt.)
            k_owned = vec![0.0f32; n * l_kv * inner];
            v_owned = vec![0.0f32; n * l_kv * inner];
            let to_q = &self.to_q;
            let to_k = &self.to_k;
            let to_v = &self.to_v;
            let q_slot = &mut q;
            let k_slot = &mut k_owned;
            let v_slot = &mut v_owned;
            rayon::scope(|s| {
                s.spawn(|_| to_q.forward_batch(q_in, q_slot));
                s.spawn(|_| to_k.forward_batch(kv_in, k_slot));
                s.spawn(|_| to_v.forward_batch(kv_in, v_slot));
            });
        }
        let cache_ref = self.cached_kv.borrow();
        let (k, v): (&[f32], &[f32]) = match &*cache_ref {
            Some((ck, cv)) => (ck.as_slice(), cv.as_slice()),
            None => (k_owned.as_slice(), v_owned.as_slice()),
        };

        let mut head_out = vec![0.0f32; n * l_q * inner];
        let scale = self.scale;

        // Two paths:
        //
        //   * Batched-softmax path: materialise the full scores tensor
        //     `[N, H, L_q, L_kv]`, dispatch one Metal softmax over all rows,
        //     then accumulate V·scores into `head_out`. Big win for cross-attn
        //     (L_kv=77, modest L_q) and low-resolution self-attn — exactly the
        //     cases where the materialised buffer fits in a few hundred MB.
        //
        //   * Streaming per-(n,h,lq) path: original implementation. Used when
        //     the scores tensor would be too large to materialise (SD1.5
        //     self-attn at 64×64 → 128M scores = 512MB), or when causal
        //     masking is needed (CLIP — the row-dependent mask doesn't map
        //     cleanly onto the batched softmax kernel without an extra
        //     uniform; CLIP attention is small enough that CPU is faster
        //     than Metal dispatch anyway).
        //
        // Three paths:
        //   1. Causal → streaming (we don't materialise the upper-triangle
        //      mask cleanly; CLIP attention is small anyway).
        //   2. Materialise-all → one big scores buffer; only when total fits.
        //   3. Per-pair materialise → loop over (n, h); each pair gets its
        //      own scores scratch (l_q×l_kv = 64MB max for SD 1.5 self-attn
        //      at 64×64). This is the fast path for the 16 big self-attn
        //      calls per UNet step that previously fell into the slow
        //      streaming loop with per-row Vec allocations.
        let scores_total = n * h * l_q * l_kv;
        let scores_per_pair = l_q * l_kv;
        let materialise = !causal && scores_total <= (1 << 25);
        let per_pair_materialise = !causal && !materialise && scores_per_pair <= (1 << 25);

        if per_pair_materialise {
            // Per-(n,h) materialise. Each thread allocates its own l_q*l_kv
            // scratch buffer (so 16 × 64MB peak for SD 1.5 64×64 self-attn —
            // fits comfortably). Three sgemm calls per pair: QK^T, then ·V.
            let q_perm = permute_lh_to_hl(&q, n, l_q, h, d);
            let k_perm = permute_lh_to_hl(k, n, l_kv, h, d);
            let v_perm = permute_lh_to_hl(v, n, l_kv, h, d);
            let mut q_scaled = q_perm.clone();
            for vv in q_scaled.iter_mut() { *vv *= scale; }
            let mut out_perm = vec![0.0f32; n * h * l_q * d];

            out_perm.par_chunks_mut(l_q * d).enumerate().for_each(|(nh, out_chunk)| {
                let q_off = nh * l_q * d;
                let k_off = nh * l_kv * d;
                let v_off = nh * l_kv * d;
                let mut scores = vec![0.0f32; scores_per_pair];
                crate::blas::sgemm_a_btrans(
                    l_q, l_kv, d,
                    &q_scaled[q_off..q_off + l_q * d],
                    &k_perm[k_off..k_off + l_kv * d],
                    &mut scores,
                );
                for row in scores.chunks_mut(l_kv) {
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for s in row.iter_mut() {
                        *s = if (*s - max).is_finite() { (*s - max).exp() } else { 0.0 };
                        sum += *s;
                    }
                    if sum > 0.0 { let inv = 1.0 / sum; for s in row.iter_mut() { *s *= inv; } }
                }
                crate::blas::sgemm_row_major(
                    l_q, d, l_kv,
                    1.0,
                    &scores, l_kv,
                    &v_perm[v_off..v_off + l_kv * d], d,
                    0.0,
                    out_chunk, d,
                );
            });
            permute_hl_to_lh_into(&out_perm, &mut head_out, n, l_q, h, d);
        } else if materialise {
            // Q, K, V come in as [N, L, H*D] (interleaved heads). Permute to
            // [N, H, L, D] so each (n, h) head is contiguous.
            let q_perm = permute_lh_to_hl(&q, n, l_q, h, d);
            let k_perm = permute_lh_to_hl(k, n, l_kv, h, d);
            let v_perm = permute_lh_to_hl(v, n, l_kv, h, d);

            let mut out_perm = vec![0.0f32; n * h * l_q * d];

            // Flash attention path: fused QK^T + softmax + ·V on GPU with
            // online softmax — no score-matrix materialisation.
            // Eliminates the 320MB+ intermediate that the sgemm path needs
            // for SDXL self-attn at 32×32 (1·20·1024·1024·4 bytes = 320MB).
            #[cfg(feature = "metal")]
            let used_flash = if n * h * l_q >= 256 {
                crate::metal_backend::flash_attention_metal(
                    &q_perm, &k_perm, &v_perm, &mut out_perm,
                    n, h, l_q, l_kv, d, scale,
                );
                true
            } else {
                false
            };
            #[cfg(not(feature = "metal"))]
            let used_flash = false;

            if !used_flash {
                // CPU fallback: sgemm-based materialised path. Q · Kᵀ →
                // softmax → · V using three stages with full score-matrix
                // materialisation. Each stage uses Accelerate sgemm.
                let mut q_scaled = q_perm.clone();
                for v in q_scaled.iter_mut() { *v *= scale; }

                let mut scores_full = vec![0.0f32; scores_total];
                for nh in 0..(n * h) {
                    let q_off = nh * l_q * d;
                    let k_off = nh * l_kv * d;
                    let s_off = nh * l_q * l_kv;
                    crate::blas::sgemm_a_btrans(
                        l_q, l_kv, d,
                        &q_scaled[q_off..q_off + l_q * d],
                        &k_perm[k_off..k_off + l_kv * d],
                        &mut scores_full[s_off..s_off + l_q * l_kv],
                    );
                }
                softmax_rows_dispatch(&mut scores_full, l_kv);
                for nh in 0..(n * h) {
                    let s_off = nh * l_q * l_kv;
                    let v_off = nh * l_kv * d;
                    let o_off = nh * l_q * d;
                    crate::blas::sgemm_row_major(
                        l_q, d, l_kv,
                        1.0,
                        &scores_full[s_off..s_off + l_q * l_kv], l_kv,
                        &v_perm[v_off..v_off + l_kv * d], d,
                        0.0,
                        &mut out_perm[o_off..o_off + l_q * d], d,
                    );
                }
            }

            // Permute back [N, H, L_q, D] → head_out [N, L_q, H*D].
            permute_hl_to_lh_into(&out_perm, &mut head_out, n, l_q, h, d);
        } else {
            // ---- Streaming path (original) ----
            let chunks: Vec<(usize, usize, Vec<(usize, Vec<f32>)>)> = (0..n * h)
                .into_par_iter()
                .map(|nh| {
                    let ni = nh / h;
                    let hi = nh % h;
                    let mut local: Vec<(usize, Vec<f32>)> = Vec::with_capacity(l_q);
                    let mut scores = vec![0.0f32; l_kv];
                    for lq in 0..l_q {
                        let q_off = ni * l_q * inner + lq * inner + hi * d;
                        let q_h = &q[q_off..q_off + d];

                        for lk in 0..l_kv {
                            let k_off = ni * l_kv * inner + lk * inner + hi * d;
                            let k_h = &k[k_off..k_off + d];
                            let mut s = 0.0f32;
                            for i in 0..d { s += q_h[i] * k_h[i]; }
                            scores[lk] = s * scale;
                        }
                        if causal {
                            for lk in (lq + 1)..l_kv { scores[lk] = f32::NEG_INFINITY; }
                        }
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0f32;
                        for s in scores.iter_mut() {
                            *s = if (*s - max).is_finite() { (*s - max).exp() } else { 0.0 };
                            sum += *s;
                        }
                        if sum > 0.0 {
                            let inv = 1.0 / sum;
                            for s in scores.iter_mut() { *s *= inv; }
                        }
                        let mut out_d = vec![0.0f32; d];
                        for lk in 0..l_kv {
                            let v_off = ni * l_kv * inner + lk * inner + hi * d;
                            let w = scores[lk];
                            if w == 0.0 { continue; }
                            for i in 0..d { out_d[i] += w * v[v_off + i]; }
                        }
                        let out_off = ni * l_q * inner + lq * inner + hi * d;
                        local.push((out_off, out_d));
                    }
                    (ni, hi, local)
                })
                .collect();
            for (_, _, locals) in chunks {
                for (out_off, out_d) in locals {
                    head_out[out_off..out_off + d].copy_from_slice(&out_d);
                }
            }
        }

        // Output projection.
        self.to_out.forward_batch(&head_out, out);
    }

    /// GPU-resident attention. `q_in` and `kv_in` are fp16 GpuTensors with
    /// flat layout `[N, L, query_dim/kv_dim]`; output is `[N, L_q, query_dim]`.
    ///
    /// Internally:
    ///   1. Project Q, K, V via Linear::forward_gpu (cached f16 weights).
    ///   2. Permute to `[N, H, L, D]` (head-contiguous).
    ///   3. fp16 flash attention (online softmax, no score-matrix materialisation).
    ///   4. Permute back to `[N, L_q, inner]`.
    ///   5. Output projection via Linear::forward_gpu.
    ///
    /// `causal=true` is not yet supported — CLIP attention (the only causal
    /// use case in SD) stays on the CPU path. Cross-attention KV is recomputed
    /// each call for now (no GPU cache yet).
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &self,
        q_in: &crate::metal_backend::GpuTensor,
        kv_in: &crate::metal_backend::GpuTensor,
        n: usize, l_q: usize, l_kv: usize,
        causal: bool,
    ) -> crate::metal_backend::GpuTensor {
        use crate::metal_backend::*;
        debug_assert!(!causal, "Attention::forward_gpu does not yet support causal mask");

        let h = self.num_heads;
        let d = self.head_dim;
        let inner = h * d;

        // Q always recomputed from the (per-step changing) query input.
        let q = self.to_q.forward_gpu(q_in);   // [N*L_q, inner]
        let q_perm = permute_lh_to_hl_f16_gpu(&q, n, l_q, h, d);
        drop(q);

        // K, V come from the cache when available (cross-attention with
        // text embedding stable across timesteps), else freshly computed.
        let cache_ref = self.cached_kv_gpu.borrow();
        let (k_perm, v_perm) = match &*cache_ref {
            Some(cached) if cached.n == n && cached.l_kv == l_kv => {
                // Wrap cached buffers as GpuTensors. Buffer.clone() bumps the
                // ObjC retain count; the wrappers drop normally and the cache
                // retains its refcount = 1 reference.
                let k_t = GpuTensor {
                    buffer: cached.k.clone(),
                    shape: vec![n * l_kv, inner],
                    dtype: GpuDtype::F16,
                };
                let v_t = GpuTensor {
                    buffer: cached.v.clone(),
                    shape: vec![n * l_kv, inner],
                    dtype: GpuDtype::F16,
                };
                let k_perm = permute_lh_to_hl_f16_gpu(&k_t, n, l_kv, h, d);
                let v_perm = permute_lh_to_hl_f16_gpu(&v_t, n, l_kv, h, d);
                drop(k_t); drop(v_t);
                (k_perm, v_perm)
            }
            _ => {
                let k = self.to_k.forward_gpu(kv_in);
                let v = self.to_v.forward_gpu(kv_in);
                let k_perm = permute_lh_to_hl_f16_gpu(&k, n, l_kv, h, d);
                let v_perm = permute_lh_to_hl_f16_gpu(&v, n, l_kv, h, d);
                drop(k); drop(v);
                (k_perm, v_perm)
            }
        };
        drop(cache_ref);

        // Fused attention. Default uses Apple's MPSGraph SDPA
        // (graph-compiled, picks tile sizes per shape). Set
        // KLEARU_DISABLE_MPSGRAPH_SDPA=1 to route through the hand-rolled
        // flash-attention kernel instead (slower but more numerically stable).
        let mut o_perm = GpuTensor::new_f16(vec![n, h, l_q, d]);
        if std::env::var_os("KLEARU_DISABLE_MPSGRAPH_SDPA").is_some() {
            flash_attention_f16_gpu(
                &q_perm, &k_perm, &v_perm, &mut o_perm,
                n, h, l_q, l_kv, d, self.scale,
            );
        } else {
            mpsgraph_sdpa_bf16_gpu(
                &q_perm, &k_perm, &v_perm, &mut o_perm,
                n, h, l_q, l_kv, d, self.scale,
            );
        }
        drop(q_perm); drop(k_perm); drop(v_perm);

        // Permute back and output-project.
        let o = permute_hl_to_lh_f16_gpu(&o_perm, n, l_q, h, d);
        drop(o_perm);
        self.to_out.forward_gpu(&o)
    }

    /// QKV-fused self-attention forward. Replaces three separate Q/K/V
    /// projections (4 dispatches each: bf16↔f32 casts + sgemm) with one
    /// wider matmul into a [N·L, 3·inner] buffer, then strided permute
    /// to break out Q/K/V slices without copying. Saves ~2/3 of the
    /// projection dispatches per self-attn block (~70 blocks × 25 steps
    /// in SDXL).
    ///
    /// Requires `query_dim == kv_dim` (self-attention). Caller passes
    /// the input once. Bias is fused via concatenated bias vector.
    #[cfg(feature = "metal")]
    pub fn forward_gpu_self(
        &self,
        x_in: &crate::metal_backend::GpuTensor,
        n: usize, l: usize,
    ) -> crate::metal_backend::GpuTensor {
        use crate::metal_backend::*;
        debug_assert_eq!(self.to_q.in_features, self.to_k.in_features,
            "forward_gpu_self requires query_dim == kv_dim");
        debug_assert_eq!(self.to_q.in_features, self.to_v.in_features,
            "forward_gpu_self requires query_dim == kv_dim");
        debug_assert_eq!(self.to_q.out_features, self.to_k.out_features);
        debug_assert_eq!(self.to_q.out_features, self.to_v.out_features);

        let h = self.num_heads;
        let d = self.head_dim;
        let inner = h * d;
        let qd = self.to_q.in_features;
        let n_rows = n * l;

        // Lazy-build [W_q | W_k | W_v] concatenated weight, [3·inner, qd].
        // PyTorch convention: weight is [out, in], so concat along dim 0.
        {
            let mut cw = self.qkv_concat_weight.borrow_mut();
            if cw.is_none() {
                let mut concat = Vec::with_capacity(3 * inner * qd);
                concat.extend_from_slice(&self.to_q.weight);
                concat.extend_from_slice(&self.to_k.weight);
                concat.extend_from_slice(&self.to_v.weight);
                *cw = Some(concat);
            }
        }
        let cw = self.qkv_concat_weight.borrow();
        let concat_w = cw.as_ref().unwrap();

        // Single matmul: x_in [n·l, qd] · W_qkv^T = [n·l, 3·inner].
        let qkv = GpuTensor::new_f16(vec![n_rows, 3 * inner]);
        let w_buf = weight_f16_buffer(concat_w);
        sgemm_f16_a_btrans_buf(&qkv.buffer, &x_in.buffer, &w_buf,
                               n_rows, 3 * inner, qd);
        // Bias: if any of q/k/v has bias, build a concatenated bias and
        // add. (Common case in SD UNet: all three have no bias, so we
        // skip entirely.)
        if self.to_q.bias.is_some() || self.to_k.bias.is_some() || self.to_v.bias.is_some() {
            let z = vec![0.0f32; inner];
            let bq = self.to_q.bias.as_deref().unwrap_or(&z);
            let bk = self.to_k.bias.as_deref().unwrap_or(&z);
            let bv = self.to_v.bias.as_deref().unwrap_or(&z);
            let mut bcat = Vec::with_capacity(3 * inner);
            bcat.extend_from_slice(bq);
            bcat.extend_from_slice(bk);
            bcat.extend_from_slice(bv);
            let bb = weight_f16_buffer(&bcat);
            let mut qkv_mut = qkv;
            bias_add_f16_gpu(&mut qkv_mut, &bb, 1, 3 * inner);
            return self.finish_self_attn(qkv_mut, n, l, h, d, inner);
        }
        self.finish_self_attn(qkv, n, l, h, d, inner)
    }

    /// Shared tail of `forward_gpu_self`: split QKV via strided permute,
    /// SDPA, permute back, output project.
    #[cfg(feature = "metal")]
    fn finish_self_attn(
        &self,
        qkv: crate::metal_backend::GpuTensor,
        n: usize, l: usize, h: usize, d: usize, inner: usize,
    ) -> crate::metal_backend::GpuTensor {
        use crate::metal_backend::*;
        // Strided permute: read the [n, l, 3·inner] qkv buffer at three
        // f16-element offsets (0, inner, 2·inner) with stride 3·inner,
        // writing each as [n, h, l, d].
        let inner_stride = 3 * inner;
        let elem_bytes = 2; // f16
        let q_perm = permute_lh_to_hl_f16_gpu_strided(
            &qkv.buffer, 0, n, l, h, d, inner_stride);
        let k_perm = permute_lh_to_hl_f16_gpu_strided(
            &qkv.buffer, inner * elem_bytes, n, l, h, d, inner_stride);
        let v_perm = permute_lh_to_hl_f16_gpu_strided(
            &qkv.buffer, 2 * inner * elem_bytes, n, l, h, d, inner_stride);
        drop(qkv);

        let mut o_perm = GpuTensor::new_f16(vec![n, h, l, d]);
        if std::env::var_os("KLEARU_DISABLE_MPSGRAPH_SDPA").is_some() {
            flash_attention_f16_gpu(
                &q_perm, &k_perm, &v_perm, &mut o_perm,
                n, h, l, l, d, self.scale,
            );
        } else {
            mpsgraph_sdpa_bf16_gpu(
                &q_perm, &k_perm, &v_perm, &mut o_perm,
                n, h, l, l, d, self.scale,
            );
        }
        drop(q_perm); drop(k_perm); drop(v_perm);

        let o = permute_hl_to_lh_f16_gpu(&o_perm, n, l, h, d);
        drop(o_perm);
        self.to_out.forward_gpu(&o)
    }
}

/// Permute [N, L, H*D] → [N, H, L, D] (move head-dim from interleaved fastest
/// inner stride to its own axis). Returns a fresh vec of equal length.
#[inline]
fn permute_lh_to_hl(src: &[f32], n: usize, l: usize, h: usize, d: usize) -> Vec<f32> {
    let inner = h * d;
    let mut out = vec![0.0f32; n * h * l * d];
    // out[n, h, l, d] = src[n, l, h*d + d_i]
    // Parallelise over n*h since each (ni, hi) writes a disjoint contiguous block.
    use rayon::prelude::*;
    out.par_chunks_mut(l * d).enumerate().for_each(|(nh, dst)| {
        let ni = nh / h;
        let hi = nh % h;
        let src_n_base = ni * l * inner;
        let h_base = hi * d;
        for li in 0..l {
            let src_off = src_n_base + li * inner + h_base;
            let dst_off = li * d;
            dst[dst_off..dst_off + d].copy_from_slice(&src[src_off..src_off + d]);
        }
    });
    out
}

/// Permute [N, H, L, D] → [N, L, H*D] in-place over the destination.
#[inline]
fn permute_hl_to_lh_into(src: &[f32], dst: &mut [f32], n: usize, l: usize, h: usize, d: usize) {
    let inner = h * d;
    use rayon::prelude::*;
    dst.par_chunks_mut(l * inner).enumerate().for_each(|(ni, dst_n)| {
        for hi in 0..h {
            let src_base = (ni * h + hi) * l * d;
            let h_base = hi * d;
            for li in 0..l {
                let src_off = src_base + li * d;
                let dst_off = li * inner + h_base;
                dst_n[dst_off..dst_off + d].copy_from_slice(&src[src_off..src_off + d]);
            }
        }
    });
}

/// Row-wise softmax over a flat buffer. Dispatches to Metal when the
/// `metal` feature is on; falls back to a parallel CPU softmax otherwise.
#[inline]
fn softmax_rows_dispatch(scores: &mut [f32], width: usize) {
    #[cfg(feature = "metal")]
    if scores.len() >= 1 << 14 {
        crate::metal_backend::softmax_metal(scores, width, false);
        return;
    }
    scores.par_chunks_mut(width).for_each(|row| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in row.iter_mut() {
            *s = if (*s - max).is_finite() { (*s - max).exp() } else { 0.0 };
            sum += *s;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for s in row.iter_mut() { *s *= inv; }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_runs_and_preserves_shape() {
        // Self-attention with 2 heads, head_dim=4 (inner=8), query_dim=8.
        let a = Attention::new(8, 8, 2, 4, false);
        let q = vec![0.1f32; 1 * 5 * 8]; // n=1, l=5, dim=8
        let mut out = vec![0.0f32; 1 * 5 * 8];
        a.forward(&q, &q, 1, 5, 5, false, &mut out);
        // With zero weights, output is all zeros.
        for v in &out { assert!(v.abs() < 1e-6); }
    }

    /// Hand-rolled reference attention (single head, no batching) — independent
    /// of the materialised vs streaming path inside `Attention::forward`. Used
    /// to verify that the refactored materialised path produces correct numbers,
    /// not just correct shapes.
    fn ref_attention_single_head(
        q: &[f32], k: &[f32], v: &[f32],
        l_q: usize, l_kv: usize, d: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0.0f32; l_q * d];
        for lq in 0..l_q {
            let mut scores = vec![0.0f32; l_kv];
            for lk in 0..l_kv {
                let mut s = 0.0f32;
                for i in 0..d { s += q[lq*d+i] * k[lk*d+i]; }
                scores[lk] = s * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - m).exp(); sum += *s; }
            for s in scores.iter_mut() { *s /= sum; }
            for lk in 0..l_kv {
                for i in 0..d { out[lq*d+i] += scores[lk] * v[lk*d+i]; }
            }
        }
        out
    }

    #[test]
    fn attention_matches_reference_identity_proj() {
        // num_heads=1, head_dim=4 → inner=4, query_dim=4. With identity
        // to_q/to_k/to_v/to_out (each weight = identity matrix) the attention
        // module should produce exactly `softmax(QK^T/√d)·V`.
        let mut a = Attention::new(4, 4, 1, 4, false);
        // Linear stores weights as [out_features, in_features] row-major.
        // Identity: w[i,i]=1, else 0.
        let id4: Vec<f32> = (0..16).map(|k| if k/4 == k%4 { 1.0 } else { 0.0 }).collect();
        a.to_q.weight.copy_from_slice(&id4);
        a.to_k.weight.copy_from_slice(&id4);
        a.to_v.weight.copy_from_slice(&id4);
        a.to_out.weight.copy_from_slice(&id4);

        let q: Vec<f32> = vec![
            0.5, -0.1,  0.3, 0.2,
            0.1,  0.4, -0.2, 0.6,
            -0.3, 0.2,  0.5, 0.1,
        ];
        let kv: Vec<f32> = vec![
            0.2, -0.3, 0.4, 0.1,
            0.5,  0.1, 0.0, 0.3,
            -0.1, 0.6, 0.2, 0.4,
        ];
        let mut out = vec![0.0f32; 3 * 4];
        // Hits the materialised path (3*1*3*3 = 27, well under 2^25 cap).
        a.forward(&q, &kv, 1, 3, 3, false, &mut out);

        let expected = ref_attention_single_head(&q, &kv, &kv, 3, 3, 4);
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5,
                "got {got}, want {want}, diff {}", (got - want).abs());
        }
    }

    /// Exercises flash attention on Metal (when feature is on) by using
    /// dims large enough to trip the n*h*l_q >= 256 threshold. Compares
    /// against the per-row reference within fp32 tolerance.
    #[test]
    fn attention_flash_path_matches_reference() {
        // num_heads=4, head_dim=8, query_dim=32. l_q=l_kv=64 → n*h*l_q=256.
        let mut a = Attention::new(32, 32, 4, 8, false);
        for (i, w) in a.to_q.weight.iter_mut().enumerate() { *w = ((i % 7) as f32 - 3.0) * 0.1; }
        for (i, w) in a.to_k.weight.iter_mut().enumerate() { *w = ((i % 5) as f32 - 2.0) * 0.1; }
        for (i, w) in a.to_v.weight.iter_mut().enumerate() { *w = ((i % 11) as f32 - 5.0) * 0.05; }
        for i in 0..32 { a.to_out.weight[i*32 + i] = 1.0; }

        let l = 64usize;
        let q: Vec<f32> = (0..l*32).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let mut out = vec![0.0f32; l * 32];
        a.forward(&q, &q, 1, l, l, false, &mut out);

        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
        let mag: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(mag > 1e-3, "output appears all-zero (mag={mag})");
    }

    /// Direct flash-attention kernel correctness test: feeds permuted Q/K/V
    /// through `flash_attention_metal` and the materialised sgemm path, and
    /// requires bit-equivalent (within fp32 tolerance) outputs. Catches any
    /// online-softmax accumulation bug.
    #[cfg(feature = "metal")]
    #[test]
    fn flash_attention_kernel_matches_sgemm_path() {
        // Dims: N=1, H=2, L_q=L_kv=128, D=32. n*h*l_q = 256 ≥ threshold.
        let n = 1usize;
        let h = 2usize;
        let l = 128usize;
        let d = 32usize;
        let scale = 1.0 / (d as f32).sqrt();

        // Deterministic random Q, K, V in [N, H, L, D] permuted layout.
        let total = n * h * l * d;
        let q: Vec<f32> = (0..total).map(|i| ((i * 37) % 71) as f32 / 71.0 - 0.5).collect();
        let k: Vec<f32> = (0..total).map(|i| ((i * 53) % 89) as f32 / 89.0 - 0.5).collect();
        let v: Vec<f32> = (0..total).map(|i| ((i * 41) % 53) as f32 / 53.0 - 0.5).collect();

        // Path 1: flash attention.
        let mut o_flash = vec![0.0f32; total];
        crate::metal_backend::flash_attention_metal(
            &q, &k, &v, &mut o_flash,
            n, h, l, l, d, scale,
        );

        // Path 2: materialised sgemm reference. Q · Kᵀ → softmax → · V.
        let mut o_ref = vec![0.0f32; total];
        for nh in 0..(n * h) {
            let q_off = nh * l * d;
            let k_off = nh * l * d;
            let v_off = nh * l * d;
            let o_off = nh * l * d;
            // scores_lq_lk = (q · k^T) * scale per row.
            let mut scores = vec![0.0f32; l * l];
            for lq in 0..l {
                for lk in 0..l {
                    let mut s = 0.0f32;
                    for di in 0..d {
                        s += q[q_off + lq*d + di] * k[k_off + lk*d + di];
                    }
                    scores[lq*l + lk] = s * scale;
                }
            }
            // Row-wise softmax.
            for lq in 0..l {
                let row = &mut scores[lq*l..(lq+1)*l];
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in row.iter_mut() { *s = (*s - m).exp(); sum += *s; }
                for s in row.iter_mut() { *s /= sum; }
            }
            // out = scores · V.
            for lq in 0..l {
                for di in 0..d {
                    let mut s = 0.0f32;
                    for lk in 0..l {
                        s += scores[lq*l + lk] * v[v_off + lk*d + di];
                    }
                    o_ref[o_off + lq*d + di] = s;
                }
            }
        }

        // Compare element-wise. fp32 tolerance must accommodate sum ordering.
        let mut max_diff = 0.0f32;
        for (got, want) in o_flash.iter().zip(o_ref.iter()) {
            let diff = (got - want).abs();
            if diff > max_diff { max_diff = diff; }
        }
        assert!(max_diff < 1e-4,
            "flash attention diverges from sgemm reference: max_diff={max_diff}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn attention_forward_gpu_matches_cpu() {
        // Deterministic non-trivial weights, self-attn, small dims.
        let h = 4; let d = 8; let inner = h * d;
        let mut a = Attention::new(inner, inner, h, d, false);
        for (i, w) in a.to_q.weight.iter_mut().enumerate() {
            *w = ((i * 31) % 17) as f32 / 17.0 - 0.5;
        }
        for (i, w) in a.to_k.weight.iter_mut().enumerate() {
            *w = ((i * 23) % 19) as f32 / 19.0 - 0.5;
        }
        for (i, w) in a.to_v.weight.iter_mut().enumerate() {
            *w = ((i * 29) % 13) as f32 / 13.0 - 0.5;
        }
        for (i, w) in a.to_out.weight.iter_mut().enumerate() {
            *w = ((i * 37) % 11) as f32 / 11.0 - 0.5;
        }

        let n = 1; let l = 64;
        let x: Vec<f32> = (0..n*l*inner).map(|i| ((i * 7) % 13) as f32 / 13.0 - 0.5).collect();

        // CPU reference (self-attn, non-causal).
        let mut out_cpu = vec![0.0f32; n * l * inner];
        a.forward(&x, &x, n, l, l, false, &mut out_cpu);

        // GPU.
        let x_gpu = crate::metal_backend::GpuTensor::upload_f32_as_f16(
            vec![n, l, inner], &x);
        let out_gpu_t = a.forward_gpu(&x_gpu, &x_gpu, n, l, l, false);
        let out_gpu = out_gpu_t.download_to_f32();

        let mut max_diff = 0.0f32;
        for (g, c) in out_gpu.iter().zip(out_cpu.iter()) {
            let dd = (g - c).abs();
            if dd > max_diff { max_diff = dd; }
        }
        // Many fp16 ops chained (3 projs + flash + perm + out_proj); allow
        // headroom for accumulated quantisation error.
        assert!(max_diff < 0.3,
            "Attention::forward_gpu diverges from CPU: max_diff={max_diff}");
    }

    #[test]
    fn attention_causal_uses_streaming_path() {
        // causal=true forces streaming path. Verify causal mask is applied:
        // first query token cannot attend to anything beyond position 0.
        let mut a = Attention::new(2, 2, 1, 2, false);
        let id2: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        a.to_q.weight.copy_from_slice(&id2);
        a.to_k.weight.copy_from_slice(&id2);
        a.to_v.weight.copy_from_slice(&id2);
        a.to_out.weight.copy_from_slice(&id2);

        // Two tokens; if causal works, token 0's output equals v[0]. Token 1
        // is a softmax-weighted mix of v[0] and v[1].
        let q = vec![1.0, 0.0,  0.0, 1.0];
        let v0 = vec![3.0, 5.0];
        let v1 = vec![7.0, 11.0];
        let kv: Vec<f32> = [&v0[..], &v1[..]].concat();
        let mut out = vec![0.0f32; 2 * 2];
        a.forward(&q, &kv, 1, 2, 2, true, &mut out);

        // Token 0 must equal v[0] exactly (only one unmasked position).
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 5.0).abs() < 1e-5);
    }
}
