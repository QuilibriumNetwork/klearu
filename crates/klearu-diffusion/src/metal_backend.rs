//! Metal GPU compute backend (macOS, gated by `metal` feature).
//!
//! Routes sgemm through Apple's Metal Performance Shaders (MPS) framework
//! via `MPSMatrixMultiplication`. Includes a thread-local sized-buffer
//! pool to amortise allocation across thousands of per-step sgemm calls.
//! Also exposes a set of small Metal compute kernels for dominant
//! non-matmul ops (GroupNorm, SiLU, GELU, quick_gelu).
//!
//! ## Capabilities
//!
//! - `sgemm_metal` — Apple-tuned matmul via MPS. For SDXL-sized matrices
//!   this is typically 5-15× faster than a naive Metal kernel and
//!   competitive with Accelerate.framework's AMX path on M-series GPUs.
//! - Buffer pooling: per-thread `HashMap<bytes, Vec<MTLBuffer>>` so the
//!   thousands of sgemm calls per generation amortise to near-zero
//!   allocation.
//! - `groupnorm_metal`, `silu_metal`, `gelu_metal`, `quick_gelu_metal` —
//!   GPU kernels for the most common non-matmul ops in SD.
//! - `GpuTensor` plus a set of primitive Metal ops support partially
//!   GPU-resident execution paths.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

use metal::{Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library};
use objc2::msg_send;
use objc2::runtime::{AnyClass, AnyObject};

// ---------------------------------------------------------------------------
// MPSGraph: Apple's high-level graph compute path. We use it for Conv2d
// because Apple's optimised conv kernels are dramatically faster than our
// im2col → MPSMatrixMultiplication path (no 47MB-per-call im2col scratch
// buffer, fused conv kernel uses Apple's Winograd / direct conv as
// appropriate).
//
// Graphs are cached per (weight pointer, input shape, conv params) so the
// JIT compile cost (~tens-hundreds ms) is paid once per layer per shape,
// not per call. The Rust side caches *MPSGraph instance + placeholder
// tensor pointers; per-call work is just `MPSGraphTensorData` wrapping
// the input/output buffers, then `encodeToCommandBuffer:`.
// ---------------------------------------------------------------------------

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct ConvGraphKey {
    n: usize, in_c: usize, h_in: usize, w_in: usize,
    out_c: usize, kh: usize, kw: usize,
    stride: usize, pad: usize,
    /// f16 or f32 compute. Part of the key because the two produce different
    /// graphs and must not share a cache entry.
    f32_compute: bool,
}

struct CachedConvGraph {
    graph: *mut AnyObject,           // MPSGraph
    input_tensor: *mut AnyObject,    // MPSGraphTensor (placeholder)
    weight_tensor: *mut AnyObject,   // MPSGraphTensor (placeholder)
    output_tensor: *mut AnyObject,   // MPSGraphTensor
    h_out: usize, w_out: usize,
    /// Generation this entry was last used in, for `conv_graph_end_generation`.
    gen: u64,
}

// Send/Sync: the cached objc objects are never sent across threads — the
// thread_local enforces single-thread access. We mark them Sync via raw
// pointer wrapper so the thread_local with! works.
unsafe impl Send for CachedConvGraph {}
unsafe impl Sync for CachedConvGraph {}

thread_local! {
    static CONV_GRAPH_CACHE: std::cell::RefCell<HashMap<ConvGraphKey, CachedConvGraph>>
        = std::cell::RefCell::new(HashMap::new());
    static CONV_GRAPH_GEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Close the current conv-graph generation and release graphs that were not
/// used in it.
///
/// Each cached MPSGraph holds its own intermediate storage, which at decoder
/// resolutions is large. The video decoder walks frames whose causal cache
/// makes the conv batch dimension differ for frame 0, frame 1, and the steady
/// state, so a plain cache keeps three full sets of graphs alive: 294 of them
/// at 98 per regime, and at 704x1280 that measured +7.4, +15.4 and +15.3 GB
/// over the first three frames — enough to be killed just before the cache
/// would have converged.
///
/// Calling this per frame bounds the cache to one frame's working set. The
/// steady state re-uses its own graphs, so nothing is rebuilt after the
/// transient frames.
pub fn conv_graph_end_generation() {
    let g = CONV_GRAPH_GEN.with(|c| {
        let next = c.get() + 1;
        c.set(next);
        next
    });
    CONV_GRAPH_CACHE.with_borrow_mut(|cache| {
        cache.retain(|_, v| {
            // Keep anything used in the generation that just ended.
            if v.gen + 1 >= g {
                return true;
            }
            unsafe {
                let _: () = msg_send![v.graph, release];
            }
            false
        });
    });
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct SdpaGraphKey {
    n: usize, h: usize, l_q: usize, l_kv: usize, d: usize,
    // Scale is encoded as bits because f32 isn't Eq/Hash. Different scales
    // would build distinct graphs; in practice scale = 1/sqrt(d) is fixed
    // per (head_dim) — so this is just paranoia.
    scale_bits: u32,
}

struct CachedSdpaGraph {
    graph: *mut AnyObject,           // MPSGraph
    q_tensor: *mut AnyObject,        // placeholder [N,H,L_q,D]
    k_tensor: *mut AnyObject,        // placeholder [N,H,L_kv,D]
    v_tensor: *mut AnyObject,        // placeholder [N,H,L_kv,D]
    o_tensor: *mut AnyObject,        // SDPA output [N,H,L_q,D]
}
unsafe impl Send for CachedSdpaGraph {}
unsafe impl Sync for CachedSdpaGraph {}

thread_local! {
    static SDPA_GRAPH_CACHE: std::cell::RefCell<HashMap<SdpaGraphKey, CachedSdpaGraph>>
        = std::cell::RefCell::new(HashMap::new());
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct GegluGraphKey {
    n_rows: usize, in_dim: usize, hidden: usize, out_dim: usize,
    // Weight pointers identify the specific block (each block has its
    // own weights). Without these, two blocks of the same shape would
    // share a graph and produce wrong results.
    proj_in_ptr: usize, proj_out_ptr: usize,
    has_in_bias: bool, has_out_bias: bool,
}

struct CachedGegluGraph {
    graph: *mut AnyObject,
    x_tensor: *mut AnyObject,
    proj_in_w_tensor: *mut AnyObject,
    proj_in_b_tensor: *mut AnyObject, // null when has_in_bias=false
    proj_out_w_tensor: *mut AnyObject,
    proj_out_b_tensor: *mut AnyObject, // null when has_out_bias=false
    o_tensor: *mut AnyObject,
}
unsafe impl Send for CachedGegluGraph {}
unsafe impl Sync for CachedGegluGraph {}

thread_local! {
    static GEGLU_GRAPH_CACHE: std::cell::RefCell<HashMap<GegluGraphKey, CachedGegluGraph>>
        = std::cell::RefCell::new(HashMap::new());
}

// MPSGraph layout enum values. From <MetalPerformanceShadersGraph/MPSGraphConvolutionOps.h>:
//   MPSGraphTensorNamedDataLayoutNCHW    = 0
//   MPSGraphTensorNamedDataLayoutNHWC    = 1
//   MPSGraphTensorNamedDataLayoutOIHW    = 2  (output-input-height-width — weights)
//   MPSGraphTensorNamedDataLayoutHWIO    = 3
//   MPSGraphTensorNamedDataLayoutCHW     = 4  (NOT a weight layout — input only)
//   MPSGraphPaddingStyleExplicit         = 0
// NSUInteger in the ObjC signatures (type code 'Q'). Declaring these as i64
// passes the right bits but the wrong type, which objc2's debug-build
// signature check rejects — and that panic cannot unwind out of the
// surrounding exception-catch, so it aborts the process. Release builds were
// unaffected, which is why this survived.
const MPSGRAPH_LAYOUT_NCHW: u64 = 0;
const MPSGRAPH_LAYOUT_OIHW: u64 = 2;
const MPSGRAPH_PADDING_EXPLICIT: u64 = 0;

// MPS data type constants. From <MetalPerformanceShaders/MPSCore.h>:
//   MPSDataTypeFloatBit  = 0x10000000
//   MPSDataTypeFloat32   = 0x10000000 | 32 = 0x10000020
//   MPSDataTypeFloat16   = 0x10000000 | 16 = 0x10000010
//
// Storage uses f16 (8-bit exponent, fp32 dynamic range) to avoid the
// ±65504 ceiling of fp16, which SD's deep activations regularly exceed.
// MPSMatrixMultiplication does NOT support f16 inputs — the matmul path
// casts f16 → fp32 → MPS-fp32 sgemm → fp32 → f16. This costs ~2× per
// sgemm but is the only clean way to combine f16's range with MPS's
// hand-tuned matmul. (`sgemm_metal_f16` / `_a_btrans` keep their `_f16`
// names but the underlying storage is f16; see kernel definitions.)
const MPS_DATA_TYPE_FLOAT32: u32 = 0x1000_0020;
#[allow(dead_code)]
const MPS_DATA_TYPE_FLOAT16: u32 = 0x1000_0010;
// BFloat16 = MPSAlternateEncodingBit (0x02000000) | FloatBit (0x10000000) | 16
// Supported by MPSGraph since macOS 14 / iOS 17.
const MPS_DATA_TYPE_BFLOAT16: u32 = 0x1200_0010;

const KERNELS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// SiLU(x) = x * sigmoid(x)
kernel void silu(
    device float* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float v = x[gid];
    x[gid] = v / (1.0 + exp(-v));
}

// quick_gelu(x) = x * sigmoid(1.702 * x)
kernel void quick_gelu(
    device float* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float v = x[gid];
    x[gid] = v / (1.0 + exp(-1.702 * v));
}

// In-place a += b. Used for residual connections.
kernel void elementwise_add(
    device float*       a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    a[gid] += b[gid];
}

// In-place a += bias[c] per channel; n × c × hw layout.
kernel void broadcast_add_channel(
    device float*       x        [[buffer(0)]],
    device const float* bias     [[buffer(1)]],
    constant uint&      c        [[buffer(2)]],
    constant uint&      hw       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx_in_chw = gid % (c * hw);
    uint ci = idx_in_chw / hw;
    x[gid] += bias[ci];
}

// LayerNorm per row. Each threadgroup processes one row of `dim` elements.
// gid.x = position within row, gid.y = row index.
kernel void layer_norm(
    device float*       x       [[buffer(0)]],
    constant float*     gamma   [[buffer(1)]],
    constant float*     beta    [[buffer(2)]],
    constant uint&      dim     [[buffer(3)]],
    constant float&     eps     [[buffer(4)]],
    threadgroup float*  scratch [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tpg     [[threads_per_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]]
) {
    uint row = gid;
    uint row_off = row * dim;

    // Strided sum + sum-of-squares.
    float ls = 0.0;
    float lss = 0.0;
    for (uint i = tid; i < dim; i += tpg) {
        float v = x[row_off + i];
        ls += v;
        lss += v * v;
    }
    scratch[tid] = ls;
    scratch[tpg + tid] = lss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
            scratch[tpg + tid] += scratch[tpg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(dim);
    float var  = scratch[tpg] / float(dim) - mean * mean;
    if (var < 0.0) var = 0.0;
    float inv_std = 1.0 / sqrt(var + eps);

    for (uint i = tid; i < dim; i += tpg) {
        float v = x[row_off + i];
        x[row_off + i] = (v - mean) * inv_std * gamma[i] + beta[i];
    }
}

// Row-wise softmax with optional causal mask. One threadgroup per row.
// Each row has `width` elements. With `causal=1`, position i in the row
// is set to -inf when i > row_index (used for causal self-attention).
//
// gid.x = row index
kernel void softmax_rowwise(
    device float*       x       [[buffer(0)]],
    constant uint&      width   [[buffer(1)]],
    constant uint&      causal  [[buffer(2)]],
    threadgroup float*  scratch [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tpg     [[threads_per_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]]
) {
    uint row = gid;
    uint row_off = row * width;

    // Apply causal mask first (it changes the max).
    if (causal != 0u) {
        for (uint i = tid; i < width; i += tpg) {
            if (i > row) {
                x[row_off + i] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 1: find row max for numerical stability.
    float local_max = -INFINITY;
    for (uint i = tid; i < width; i += tpg) {
        local_max = max(local_max, x[row_off + i]);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];

    // Pass 2: compute exp(x - max) and accumulate sum.
    float local_sum = 0.0;
    for (uint i = tid; i < width; i += tpg) {
        float v = x[row_off + i];
        float e = (isfinite(v - row_max)) ? exp(v - row_max) : 0.0;
        x[row_off + i] = e;
        local_sum += e;
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = scratch[0];

    // Pass 3: normalise.
    float inv_sum = (row_sum > 0.0) ? (1.0 / row_sum) : 0.0;
    for (uint i = tid; i < width; i += tpg) {
        x[row_off + i] *= inv_sum;
    }
}

// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Metal's `tanh` is implemented as (exp(2x)-1)/(exp(2x)+1); for arguments
// outside ~|x|>44 both exponentials overflow to +Inf and the result is NaN.
// The cubic blowup means a perfectly finite v≈16 produces tanh-arg ≈161 →
// NaN. Clamp the argument before calling tanh: it's already saturated at ±1
// for |arg|≥10, so the clamp is mathematically lossless and removes the
// overflow path.
kernel void gelu(
    device float* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float v = x[gid];
    float c = 0.7978845608028654; // sqrt(2/pi)
    float arg = clamp(c * (v + 0.044715 * v * v * v), -10.0f, 10.0f);
    x[gid] = 0.5 * v * (1.0 + tanh(arg));
}

// GroupNorm: x has [N, C, H, W], we operate on (N × num_groups) groups.
// Each threadgroup processes one (batch, group) — group_size threads each.
// Two-pass: stage 1 computes mean/variance via threadgroup reduction;
// stage 2 normalises and applies scale/shift. Combined here as a single
// kernel using simdgroup reductions.
//
// gid.x = position within the group (0..group_size)
// gid.y = group index (0..N*num_groups)
//
// This kernel takes the per-group reduction strategy. For our case (group
// sizes typically 1024-65536), we launch one threadgroup per group, with
// 256 threads doing strided read + reduction.
kernel void groupnorm(
    device float*       x       [[buffer(0)]],
    constant float*     gamma   [[buffer(1)]],
    constant float*     beta    [[buffer(2)]],
    constant uint&      group_size  [[buffer(3)]], // = (C/G) * H * W
    constant uint&      cg          [[buffer(4)]], // C / num_groups
    constant uint&      hw          [[buffer(5)]], // H * W
    constant uint&      num_groups  [[buffer(6)]],
    constant float&     eps         [[buffer(7)]],
    threadgroup float*  scratch [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tpg     [[threads_per_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]]
) {
    uint group_idx = gid; // (n*num_groups + g)
    uint group_offset = group_idx * group_size;

    // Stage 1: compute sum and sum-of-squares across the group, strided.
    float local_sum = 0.0;
    float local_sum_sq = 0.0;
    for (uint i = tid; i < group_size; i += tpg) {
        float v = x[group_offset + i];
        local_sum += v;
        local_sum_sq += v * v;
    }
    scratch[tid]     = local_sum;
    scratch[tpg + tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction.
    for (uint stride = tpg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
            scratch[tpg + tid] += scratch[tpg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(group_size);
    float var  = scratch[tpg] / float(group_size) - mean * mean;
    if (var < 0.0) var = 0.0;
    float inv_std = 1.0 / sqrt(var + eps);

    // Stage 2: normalise + per-channel scale/shift.
    uint g = group_idx % num_groups;
    for (uint i = tid; i < group_size; i += tpg) {
        uint ci_in_group = i / hw;
        uint channel_idx = g * cg + ci_in_group;
        float v = x[group_offset + i];
        x[group_offset + i] = (v - mean) * inv_std * gamma[channel_idx] + beta[channel_idx];
    }
}

// ============================================================================
// fp16 variants of the kernels above. Used by the GPU-resident pipeline where
// activations live on the GPU as fp16 and never round-trip to CPU between
// layers. Each kernel is a direct float→bfloat translation of its f32 sibling.
// Reductions and accumulators are kept in `float` for numerical stability
// (standard mixed-precision practice).
// ============================================================================

kernel void silu_f16(
    device bfloat* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float v = float(x[gid]);
    x[gid] = bfloat(v / (1.0 + exp(-v)));
}

kernel void gelu_f16(
    device bfloat* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float v = float(x[gid]);
    float c = 0.7978845608028654;
    // Clamp tanh-arg to avoid Metal's tanh overflow → NaN at large |v|.
    // See the f32 `gelu` kernel for the rationale.
    float arg = clamp(c * (v + 0.044715 * v * v * v), -10.0f, 10.0f);
    x[gid] = bfloat(0.5 * v * (1.0 + tanh(arg)));
}

kernel void quick_gelu_f16(
    device bfloat* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float v = float(x[gid]);
    x[gid] = bfloat(v / (1.0 + exp(-1.702 * v)));
}

kernel void elementwise_add_f16(
    device bfloat*       a [[buffer(0)]],
    device bfloat*     b [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    a[gid] = bfloat(float(a[gid]) + float(b[gid]));
}

// In-place element-wise multiply: a[i] *= b[i] (used for SwiGLU silu(g)*up).
kernel void elementwise_mul_f16(
    device bfloat*       a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    a[gid] = bfloat(float(a[gid]) * float(b[gid]));
}

// Gated attention output: a[i] *= sigmoid(g[i]). Krea 2 multiplies every
// attention output by a learned per-channel sigmoid gate before `to_out`.
kernel void sigmoid_gate_f16(
    device bfloat*       a [[buffer(0)]],
    device const bfloat* g [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float gv = float(g[gid]);
    a[gid] = bfloat(float(a[gid]) / (1.0f + exp(-gv)));
}

// RMSNorm over the last `dim` of each row, scaled by `weight[dim]`. One thread
// per row; fp32 accumulation. Handles per-token (dim=hidden) and per-head
// q/k-norm (dim=head_dim, n_rows = n*heads). In place.
kernel void rms_norm_f16(
    device bfloat*       x      [[buffer(0)]],   // [n_rows, dim]
    device const bfloat* weight [[buffer(1)]],   // [dim]
    constant uint&       dim    [[buffer(2)]],
    constant float&      eps    [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    uint base = row * dim;
    float ss = 0.0;
    for (uint i = 0; i < dim; ++i) { float v = float(x[base + i]); ss += v * v; }
    float inv = rsqrt(ss / float(dim) + eps);
    for (uint i = 0; i < dim; ++i) {
        x[base + i] = bfloat(float(x[base + i]) * inv * float(weight[i]));
    }
}


// RMSNorm with an **f32** weight, one threadgroup per row. Krea 2 stores its
// norm scales zero-centred in f32 (multiplier `1 + w`); rounding `1 + w` to
// bfloat would bias every channel by up to 0.4%, systematically, so the
// weight stays f32 while the activations keep their storage type.
kernel void rms_norm_wf32_f16(
    device bfloat*      x      [[buffer(0)]],   // [n_rows, dim]
    device const float* weight [[buffer(1)]],   // [dim]
    constant uint&      dim    [[buffer(2)]],
    constant float&     eps    [[buffer(3)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    uint base = row * dim;
    float ss = 0.0f;
    for (uint i = tid; i < dim; i += tpg) { float v = float(x[base + i]); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint nsg = (tpg + 31) / 32;
        float t = lane < nsg ? partial[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) partial[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(partial[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += tpg) {
        x[base + i] = bfloat(float(x[base + i]) * inv * weight[i]);
    }
}

// Apply rotary position embedding to x[n, num_heads, head_dim] using cos/sin
// tables [n, head_dim] (rotate_half convention). One thread per (token, head).
kernel void rope_apply_f16(
    device bfloat*       x   [[buffer(0)]],   // [n, num_heads, head_dim]
    device const bfloat* cosb [[buffer(1)]],  // [n, head_dim]
    device const bfloat* sinb [[buffer(2)]],  // [n, head_dim]
    constant uint&       num_heads [[buffer(3)]],
    constant uint&       head_dim  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]      // in [0, n*num_heads)
) {
    uint t = gid / num_heads;
    uint hd2 = head_dim / 2;
    uint base = gid * head_dim;
    uint cbase = t * head_dim;
    for (uint i = 0; i < hd2; ++i) {
        float a = float(x[base + i]);
        float b = float(x[base + hd2 + i]);
        float ci = float(cosb[cbase + i]);
        float si = float(sinb[cbase + i]);
        float ch = float(cosb[cbase + hd2 + i]);
        float sh = float(sinb[cbase + hd2 + i]);
        x[base + i]       = bfloat(a * ci - b * si);
        x[base + hd2 + i] = bfloat(b * ch + a * sh);
    }
}

// Grouped-query repeat: in[nkv, block] -> out[nh, block], out[h] = in[h/nrep].
// `block` = n*head_dim. One thread per output element.
kernel void gqa_repeat_f16(
    device const bfloat* inp [[buffer(0)]],   // [nkv, block]
    device bfloat*       out [[buffer(1)]],   // [nh,  block]
    constant uint&       nrep  [[buffer(2)]],
    constant uint&       block [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint h = gid / block;
    uint rest = gid % block;
    uint kv_h = h / nrep;
    out[gid] = inp[kv_h * block + rest];
}

// Scaled element-wise add: a[i] += scale * b[i]. Used by the ControlNet
// residual injection so a non-1.0 controlnet_weight applies on the Metal
// residence path. With scale=1.0 it's equivalent to elementwise_add_f16
// at the cost of one extra fp32 multiply per thread.
kernel void elementwise_add_scaled_f16(
    device bfloat*       a     [[buffer(0)]],
    device bfloat*     b     [[buffer(1)]],
    constant float&    scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    a[gid] = bfloat(float(a[gid]) + scale * float(b[gid]));
}

kernel void layer_norm_f16(
    device bfloat*       x       [[buffer(0)]],
    device bfloat*     gamma   [[buffer(1)]],
    device bfloat*     beta    [[buffer(2)]],
    constant uint&     dim     [[buffer(3)]],
    constant float&    eps     [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint   tid [[thread_position_in_threadgroup]],
    uint   tpg [[threads_per_threadgroup]],
    uint   gid [[threadgroup_position_in_grid]]
) {
    uint row = gid;
    uint row_off = row * dim;
    float ls = 0.0;
    float lss = 0.0;
    for (uint i = tid; i < dim; i += tpg) {
        float v = float(x[row_off + i]);
        ls += v;
        lss += v * v;
    }
    scratch[tid] = ls;
    scratch[tpg + tid] = lss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
            scratch[tpg + tid] += scratch[tpg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(dim);
    float var  = scratch[tpg] / float(dim) - mean * mean;
    if (var < 0.0) var = 0.0;
    float inv_std = 1.0 / sqrt(var + eps);
    for (uint i = tid; i < dim; i += tpg) {
        float v = float(x[row_off + i]);
        x[row_off + i] = bfloat((v - mean) * inv_std * float(gamma[i]) + float(beta[i]));
    }
}

// Out-of-place LayerNorm. Reads from `src`, writes to `dst`. Saves the
// `clone_data + in-place LN` two-step pattern in transformer blocks where
// the input is needed both for LN and as a residual.
//
// `src` uses `device` (not `constant`) because the constant address space
// has a per-binding size limit (~64KB on Apple Silicon). SDXL transformer
// LN inputs reach ~10MB, far exceeding that — using `constant` would
// silently produce wrong results. `gamma`/`beta` are tiny (per-channel)
// so `constant` is fine and gives a small cache-locality win.
kernel void layer_norm_f16_out(
    device   bfloat*     src     [[buffer(0)]],
    device   bfloat*     dst     [[buffer(1)]],
    device bfloat*     gamma   [[buffer(2)]],
    device bfloat*     beta    [[buffer(3)]],
    constant uint&     dim     [[buffer(4)]],
    constant float&    eps     [[buffer(5)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint   tid [[thread_position_in_threadgroup]],
    uint   tpg [[threads_per_threadgroup]],
    uint   gid [[threadgroup_position_in_grid]]
) {
    uint row = gid;
    uint row_off = row * dim;
    float ls = 0.0;
    float lss = 0.0;
    for (uint i = tid; i < dim; i += tpg) {
        float v = float(src[row_off + i]);
        ls += v;
        lss += v * v;
    }
    scratch[tid] = ls;
    scratch[tpg + tid] = lss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
            scratch[tpg + tid] += scratch[tpg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(dim);
    float var  = scratch[tpg] / float(dim) - mean * mean;
    if (var < 0.0) var = 0.0;
    float inv_std = 1.0 / sqrt(var + eps);
    for (uint i = tid; i < dim; i += tpg) {
        float v = float(src[row_off + i]);
        dst[row_off + i] = bfloat((v - mean) * inv_std * float(gamma[i]) + float(beta[i]));
    }
}

kernel void groupnorm_f16(
    device bfloat*       x          [[buffer(0)]],
    device bfloat*     gamma      [[buffer(1)]],
    device bfloat*     beta       [[buffer(2)]],
    constant uint&     group_size [[buffer(3)]],
    constant uint&     cg         [[buffer(4)]],
    constant uint&     hw         [[buffer(5)]],
    constant uint&     num_groups [[buffer(6)]],
    constant float&    eps        [[buffer(7)]],
    threadgroup float* scratch    [[threadgroup(0)]],
    uint   tid [[thread_position_in_threadgroup]],
    uint   tpg [[threads_per_threadgroup]],
    uint   gid [[threadgroup_position_in_grid]]
) {
    uint group_idx = gid;
    uint group_offset = group_idx * group_size;
    float local_sum = 0.0;
    float local_sum_sq = 0.0;
    for (uint i = tid; i < group_size; i += tpg) {
        float v = float(x[group_offset + i]);
        local_sum += v;
        local_sum_sq += v * v;
    }
    scratch[tid] = local_sum;
    scratch[tpg + tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
            scratch[tpg + tid] += scratch[tpg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(group_size);
    float var  = scratch[tpg] / float(group_size) - mean * mean;
    if (var < 0.0) var = 0.0;
    float inv_std = 1.0 / sqrt(var + eps);
    uint g = group_idx % num_groups;
    for (uint i = tid; i < group_size; i += tpg) {
        uint ci_in_group = i / hw;
        uint channel_idx = g * cg + ci_in_group;
        float v = float(x[group_offset + i]);
        x[group_offset + i] = bfloat((v - mean) * inv_std * float(gamma[channel_idx]) + float(beta[channel_idx]));
    }
}

// Out-of-place fused GroupNorm + SiLU. `src` and `dst` must be distinct
// buffers — for in-place, use `groupnorm_silu_f16_inplace` instead.
//
// `src` uses `device` (not `constant`) because SDXL resnet inputs at
// stage 1 are ~10MB (320ch × 128×128 fp16), far over the ~64KB constant
// address space limit. `gamma`/`beta` are small enough for `constant`.
kernel void groupnorm_silu_f16_out(
    device   bfloat*     src        [[buffer(0)]],
    device   bfloat*     dst        [[buffer(1)]],
    device bfloat*     gamma      [[buffer(2)]],
    device bfloat*     beta       [[buffer(3)]],
    constant uint&     group_size [[buffer(4)]],
    constant uint&     cg         [[buffer(5)]],
    constant uint&     hw         [[buffer(6)]],
    constant uint&     num_groups [[buffer(7)]],
    constant float&    eps        [[buffer(8)]],
    threadgroup float* scratch    [[threadgroup(0)]],
    uint   tid [[thread_position_in_threadgroup]],
    uint   tpg [[threads_per_threadgroup]],
    uint   gid [[threadgroup_position_in_grid]]
) {
    uint group_idx = gid;
    uint group_offset = group_idx * group_size;
    float local_sum = 0.0;
    float local_sum_sq = 0.0;
    for (uint i = tid; i < group_size; i += tpg) {
        float v = float(src[group_offset + i]);
        local_sum += v;
        local_sum_sq += v * v;
    }
    scratch[tid] = local_sum;
    scratch[tpg + tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
            scratch[tpg + tid] += scratch[tpg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(group_size);
    float var  = scratch[tpg] / float(group_size) - mean * mean;
    if (var < 0.0) var = 0.0;
    float inv_std = 1.0 / sqrt(var + eps);
    uint g = group_idx % num_groups;
    for (uint i = tid; i < group_size; i += tpg) {
        uint ci_in_group = i / hw;
        uint channel_idx = g * cg + ci_in_group;
        float v = float(src[group_offset + i]);
        float n = (v - mean) * inv_std * float(gamma[channel_idx]) + float(beta[channel_idx]);
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        float s = n / (1.0 + exp(-n));
        dst[group_offset + i] = bfloat(s);
    }
}

// In-place fused GroupNorm + SiLU. Single buffer; reads in the reduction
// pass complete (via threadgroup barrier) before any writes happen, and
// in the apply pass each thread reads-then-writes its own disjoint indices.
kernel void groupnorm_silu_f16_inplace(
    device   bfloat*     x          [[buffer(0)]],
    device bfloat*     gamma      [[buffer(1)]],
    device bfloat*     beta       [[buffer(2)]],
    constant uint&     group_size [[buffer(3)]],
    constant uint&     cg         [[buffer(4)]],
    constant uint&     hw         [[buffer(5)]],
    constant uint&     num_groups [[buffer(6)]],
    constant float&    eps        [[buffer(7)]],
    threadgroup float* scratch    [[threadgroup(0)]],
    uint   tid [[thread_position_in_threadgroup]],
    uint   tpg [[threads_per_threadgroup]],
    uint   gid [[threadgroup_position_in_grid]]
) {
    uint group_idx = gid;
    uint group_offset = group_idx * group_size;
    float local_sum = 0.0;
    float local_sum_sq = 0.0;
    for (uint i = tid; i < group_size; i += tpg) {
        float v = float(x[group_offset + i]);
        local_sum += v;
        local_sum_sq += v * v;
    }
    scratch[tid] = local_sum;
    scratch[tpg + tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
            scratch[tpg + tid] += scratch[tpg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(group_size);
    float var  = scratch[tpg] / float(group_size) - mean * mean;
    if (var < 0.0) var = 0.0;
    float inv_std = 1.0 / sqrt(var + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint g = group_idx % num_groups;
    for (uint i = tid; i < group_size; i += tpg) {
        uint ci_in_group = i / hw;
        uint channel_idx = g * cg + ci_in_group;
        float v = float(x[group_offset + i]);
        float n = (v - mean) * inv_std * float(gamma[channel_idx]) + float(beta[channel_idx]);
        float s = n / (1.0 + exp(-n));
        x[group_offset + i] = bfloat(s);
    }
}

// im2col for fp16 input → fp16 output. One thread per (ki, hw_out_idx)
// element of the col matrix. Pads with 0 outside input bounds.
//
// Input  [C_in, H_in, W_in], col [kk, HW_out] where kk = C_in * kH * kW
// and HW_out = H_out * W_out.
//
// gid.x = hw_out_idx (0..H_out*W_out)
// gid.y = ki        (0..C_in*kH*kW)
kernel void im2col_f16(
    device bfloat* input    [[buffer(0)]],
    device bfloat*   col      [[buffer(1)]],
    constant uint& c_in     [[buffer(2)]],
    constant uint& h_in     [[buffer(3)]],
    constant uint& w_in     [[buffer(4)]],
    constant uint& kh       [[buffer(5)]],
    constant uint& kw       [[buffer(6)]],
    constant uint& stride   [[buffer(7)]],
    constant uint& pad      [[buffer(8)]],
    constant uint& h_out    [[buffer(9)]],
    constant uint& w_out    [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint hw_out_idx = gid.x;
    uint ki         = gid.y;
    uint hw_out = h_out * w_out;
    uint kk = c_in * kh * kw;
    if (hw_out_idx >= hw_out || ki >= kk) return;

    uint y_out = hw_out_idx / w_out;
    uint x_out = hw_out_idx % w_out;
    uint c     = ki / (kh * kw);
    uint k_pos = ki % (kh * kw);
    uint k_y   = k_pos / kw;
    uint k_x   = k_pos % kw;

    int y_in = (int)(y_out * stride) + (int)k_y - (int)pad;
    int x_in = (int)(x_out * stride) + (int)k_x - (int)pad;

    bfloat v;
    if (y_in < 0 || y_in >= (int)h_in || x_in < 0 || x_in >= (int)w_in) {
        v = bfloat(0.0);
    } else {
        v = input[c * h_in * w_in + (uint)y_in * w_in + (uint)x_in];
    }
    col[ki * hw_out + hw_out_idx] = v;
}

// Bias-add with strided broadcast. `bias_idx = (gid / period) % bias_len`.
//   - Linear bias [out]:    period=1,  bias_len=out_features
//   - Conv bias [C]:        period=HW, bias_len=C
kernel void bias_add_strided_f16(
    device bfloat*       x          [[buffer(0)]],
    device bfloat*     bias       [[buffer(1)]],
    constant uint&     period     [[buffer(2)]],
    constant uint&     bias_len   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint bias_idx = (gid / period) % bias_len;
    x[gid] = bfloat(float(x[gid]) + float(bias[bias_idx]));
}

// Fused GeGLU split: input [n, 2*hidden], output gated [n, hidden].
// out[i, j] = in[i, j] * gelu(in[i, hidden + j]). One thread per output element.
kernel void geglu_split_f16(
    device bfloat*     in_buf [[buffer(0)]],
    device bfloat*       out    [[buffer(1)]],
    constant uint&     hidden [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint two_hidden = hidden * 2u;
    uint ni = gid / hidden;
    uint hi = gid % hidden;
    float a = float(in_buf[ni * two_hidden + hi]);
    float b = float(in_buf[ni * two_hidden + hidden + hi]);
    float c = 0.7978845608028654;  // sqrt(2/pi)
    // Clamp tanh-arg — Metal's tanh overflows to NaN for |arg|>~44 because
    // it computes (exp(2·arg)-1)/(exp(2·arg)+1). The cubic blowup makes
    // |arg|>>44 reachable for finite b≈16. tanh saturates by |arg|=10.
    float arg = clamp(c * (b + 0.044715 * b * b * b), -10.0f, 10.0f);
    float gelu_b = 0.5 * b * (1.0 + tanh(arg));
    out[gid] = bfloat(a * gelu_b);
}

// NCHW → NHWC permute: src [N, C, HW] → dst [N, HW, C]. One thread per element.
kernel void nchw_to_nhwc_f16(
    device bfloat*     src [[buffer(0)]],
    device bfloat*       dst [[buffer(1)]],
    constant uint&     c   [[buffer(2)]],
    constant uint&     hw  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint chw = c * hw;
    uint ni = gid / chw;
    uint pos = gid % chw;
    // dst layout [N, HW, C]: dst[ni, j, ci] at ni*HW*C + j*C + ci → flat = gid
    uint j  = pos / c;
    uint ci = pos % c;
    uint src_idx = ni * c * hw + ci * hw + j;
    dst[gid] = src[src_idx];
}

// NHWC → NCHW (inverse).
kernel void nhwc_to_nchw_f16(
    device bfloat*     src [[buffer(0)]],
    device bfloat*       dst [[buffer(1)]],
    constant uint&     c   [[buffer(2)]],
    constant uint&     hw  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint chw = c * hw;
    uint ni = gid / chw;
    uint pos = gid % chw;
    // dst layout [N, C, HW]: dst[ni, ci, j] at ni*C*HW + ci*HW + j → flat = gid
    uint ci = pos / hw;
    uint j  = pos % hw;
    uint src_idx = ni * hw * c + j * c + ci;
    dst[gid] = src[src_idx];
}

// fp16 flash attention. Same algorithm as `flash_attention` (online softmax,
// no score-matrix materialisation) but I/O is bfloat-precision. Internal
// accumulators stay fp32 for stability — only the buffer reads/writes cross
// the bfloat boundary.
kernel void flash_attention_f16(
    const device bfloat* Q       [[buffer(0)]],
    const device bfloat* K       [[buffer(1)]],
    const device bfloat* V       [[buffer(2)]],
    device bfloat*       O       [[buffer(3)]],
    constant uint&     L_q     [[buffer(4)]],
    constant uint&     L_kv    [[buffer(5)]],
    constant uint&     D       [[buffer(6)]],
    constant float&    scale   [[buffer(7)]],
    threadgroup float* smem    [[threadgroup(0)]],
    uint               tid     [[thread_position_in_threadgroup]],
    uint               tpg     [[threads_per_threadgroup]],
    uint               gid     [[threadgroup_position_in_grid]]
) {
    uint nh = gid / L_q;
    uint lq = gid % L_q;
    uint q_base = (nh * L_q + lq) * D;
    uint kv_nh_base = nh * L_kv * D;

    threadgroup float* k_row  = smem;
    threadgroup float* v_row  = smem + tpg;
    threadgroup float* reduce = smem + 2u * tpg;

    float q_d = (tid < D) ? float(Q[q_base + tid]) * scale : 0.0;
    float o_d = 0.0;

    threadgroup float m_tg;
    threadgroup float l_tg;
    if (tid == 0) {
        m_tg = -INFINITY;
        l_tg = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint lk = 0; lk < L_kv; ++lk) {
        if (tid < D) {
            k_row[tid] = float(K[kv_nh_base + lk * D + tid]);
            v_row[tid] = float(V[kv_nh_base + lk * D + tid]);
        } else {
            k_row[tid] = 0.0;
            v_row[tid] = 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        reduce[tid] = q_d * k_row[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpg / 2u; stride > 0u; stride /= 2u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float s = reduce[0];

        float m_old = m_tg;
        float m_new = max(m_old, s);
        float factor = exp(m_old - m_new);
        float p      = exp(s - m_new);

        if (tid < D) {
            o_d = o_d * factor + p * v_row[tid];
        }
        // Every thread read m_tg above and thread 0 overwrites it below. With
        // tpg up to 128 (four simdgroups) those do not run in lockstep, so
        // without this barrier thread 0 can clobber m_tg before another
        // simdgroup has read it — a rare, data-dependent wrong answer.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            l_tg = l_tg * factor + p;
            m_tg = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < D) {
        O[q_base + tid] = bfloat(o_d / l_tg);
    }
}

// ---------------------------------------------------------------------------
// Tiled flash attention — bf16 storage, f32 math, simdgroup MMA.
//
// Replaces the one-threadgroup-per-query kernel above, whose cost model falls
// apart at video sequence lengths: it re-reads all of K and V once per query
// (L_q x L_kv x D x 2 bytes of traffic) and runs a log2(D)-step tree reduction
// with a barrier at every step for every single (query, kv) pair.
//
// This version:
//   * tiles queries FA_BM at a time, so each K/V tile is loaded once and
//     amortised over 32 queries instead of being re-read per query;
//   * computes QK^T and PV with simdgroup_matrix MMA rather than scalar
//     reductions, removing the barrier storm entirely (4 barriers per KV
//     tile, not ~9 per KV element);
//   * keeps the online softmax statistics in f32, matching the numerics of
//     the scalar kernel — which stays as the correctness oracle.
//
// Layout matches the scalar kernel: Q/K/V/O are [N, H, L, D], head-contiguous.
//
// Threadgroup: 128 threads = FA_SG simdgroups; simdgroup `sg` owns query rows
// [8*sg, 8*sg+8) of the block. D is a template parameter so the fragment
// arrays stay in registers and every loop over the head dim unrolls — a
// runtime D would force `q_frag[dk]` into a dynamically indexed array and
// spill it to device memory.
//
// Threadgroup memory (floats), total 32*D + 864:
//   [0,        32*D)  `tile` — Q staging, then aliased as K|V tiles, then O
//                     staging on the way out. Q is read into registers and a
//                     barrier taken before the region is reused.
//   [32*D,     +512)  `ss`   — the FA_BM x FA_BN score tile
//   [.,        +96 )  per-row online-softmax state: m, l, rescale
//   [.,        +256)  `rs`   — per-simdgroup 8x8 diagonal scratch
// At D=128 that is 19,840 bytes, inside the 32 KB threadgroup limit.
// ---------------------------------------------------------------------------

// Tile sizes. BN=16 was measured and is worse — 1.31 TF/s against 1.58 at
// BN=32 — so this kernel is limited by tile reuse, not occupancy, even though
// at D=128 it uses 27.4 KB of a 32 KB threadgroup budget (1.17 threadgroups
// per core). Growing BM or BN further needs the K|V staging gone first: it
// shares a 16 KB region with the f32 output staging, and BN=64 alone would
// need 32 KB for K|V.
#define FA_BM 32u   // queries per threadgroup
#define FA_BN 32u   // kv positions per tile (tiled kernel)
// The split kernel reads K/V straight from device memory, so its traversal
// block is not bounded by a staging buffer — only by the score tile, which is
// 2*FA_BM*FA_BNS floats. A wider block amortises the softmax and its barriers
// over more work, which is why Metal FlashAttention runs 80-128 here against
// the 32 a staged kernel can afford.
#define FA_BNS 64u  // kv positions per tile (split kernel)
#define FA_SG 4u    // simdgroups per threadgroup (FA_BM / 8)
#define FA_TPG 128u // threads per threadgroup (FA_SG * 32)

typedef simdgroup_matrix<bfloat, 8, 8> simdgroup_bfloat8x8;

template<uint D>
void flash_attention_tiled_impl(
    const device bfloat* Q,
    const device bfloat* K,
    const device bfloat* V,
    device bfloat*       O,
    uint L_q, uint L_kv, float scale,
    threadgroup float* smem,
    uint tid, uint sgid, uint gid)
{
    constexpr uint NDK = D / 8u;   // 8-column fragments spanning the head dim

    const uint n_qb = (L_q + FA_BM - 1u) / FA_BM;
    const uint nh   = gid / n_qb;
    const uint qb   = gid % n_qb;
    const uint q0   = qb * FA_BM;

    // Threadgroup layout. Region A is addressed as bfloat while it holds the
    // Q / K / V tiles and as float when it stages the output on the way out;
    // sized by the larger of those (FA_BM*D floats == 2*FA_BN*D bfloats).
    threadgroup bfloat* tile = (threadgroup bfloat*)smem;   // Q tile, then K|V
    threadgroup bfloat* ks   = tile;                        // FA_BN*D bfloat
    threadgroup bfloat* vs   = tile + FA_BN * D;            // FA_BN*D bfloat
    threadgroup float*  ostg = smem;                        // FA_BM*D float
    threadgroup float*  ss   = smem + FA_BM * D;            // FA_BM*FA_BN float
    threadgroup bfloat* ps   = (threadgroup bfloat*)(ss + FA_BM * FA_BN);
    threadgroup float*  mrow = ss + FA_BM * FA_BN + (FA_BM * FA_BN) / 2u;
    threadgroup float*  lrow = mrow + FA_BM;
    threadgroup float*  rrow = lrow + FA_BM;
    threadgroup float*  rs   = rrow + FA_BM;                // FA_SG*64 float

    const uint q_nh_base  = nh * L_q  * D;
    const uint kv_nh_base = nh * L_kv * D;
    const uint r0 = sgid * 8u;

    // Q staging is a straight bf16 copy — no dtype conversion, and no scaling
    // either: the 1/sqrt(d) factor is folded into the softmax, which already
    // touches every score.
    for (uint i = tid; i < FA_BM * D; i += FA_TPG) {
        uint r = i / D, c = i % D;
        uint gq = q0 + r;
        tile[i] = (gq < L_q) ? Q[q_nh_base + gq * D + c] : bfloat(0.0f);
    }
    if (tid < FA_BM) { mrow[tid] = -INFINITY; lrow[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inputs are bf16 fragments; accumulators stay f32. bf16 carries f32's
    // exponent range, so unlike an IEEE-half MMA this cannot overflow on
    // large activations — the reason the whole stack stores bf16.
    simdgroup_bfloat8x8 q_frag[NDK];
    simdgroup_float8x8  o_frag[NDK];
    for (uint dk = 0; dk < NDK; ++dk) {
        simdgroup_load(q_frag[dk], tile + r0 * D + dk * 8u, D);
        o_frag[dk] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // Q now lives in registers; the staging region is free for K/V.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv0 = 0; kv0 < L_kv; kv0 += FA_BN) {
        for (uint i = tid; i < FA_BN * D; i += FA_TPG) {
            uint r = i / D, c = i % D;
            uint gk = kv0 + r;
            bool ok = gk < L_kv;
            ks[i] = ok ? K[kv_nh_base + gk * D + c] : bfloat(0.0f);
            vs[i] = ok ? V[kv_nh_base + gk * D + c] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q @ K^T. The (dk, j) tile of K^T is the (j, dk) tile of K read
        // back transposed, which simdgroup_load does natively.
        for (uint j = 0; j < FA_BN / 8u; ++j) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint dk = 0; dk < NDK; ++dk) {
                simdgroup_bfloat8x8 kf;
                simdgroup_load(kf, ks + (j * 8u) * D + dk * 8u, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, q_frag[dk], kf, acc);
            }
            simdgroup_store(acc, ss + r0 * FA_BN + j * 8u, FA_BN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax over this tile's scores, one thread per query row.
        // Emits P as bf16 for the PV matmul while keeping the running
        // statistics in f32.
        // Same all-threads softmax as the split kernel: FA_TPG / FA_BM lanes
        // per row rather than one thread per row with the rest idle.
        {
            const uint per_row = FA_TPG / FA_BM;      // 4 at 128 threads
            const uint row  = tid / per_row;
            const uint lane = tid % per_row;
            const uint span = FA_BN / per_row;
            const uint c0   = lane * span;
            threadgroup float*  rowp  = ss + row * FA_BN;
            threadgroup bfloat* prowp = ps + row * FA_BN;

            float m_old = mrow[row];
            float m_loc = m_old;
            for (uint j = c0; j < c0 + span; ++j) {
                if (kv0 + j < L_kv) m_loc = max(m_loc, rowp[j] * scale);
            }
            m_loc = max(m_loc, simd_shuffle_xor(m_loc, 1u));
            m_loc = max(m_loc, simd_shuffle_xor(m_loc, 2u));
            const float m_new = m_loc;

            // exp(-inf - -inf) is NaN; on the first tile the running max is
            // still -inf and contributes nothing, so pin the factor to 0.
            const float factor = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
            float lsum = 0.0f;
            for (uint j = c0; j < c0 + span; ++j) {
                float p = (kv0 + j < L_kv) ? exp(rowp[j] * scale - m_new) : 0.0f;
                prowp[j] = bfloat(p);
                lsum += p;
            }
            lsum += simd_shuffle_xor(lsum, 1u);
            lsum += simd_shuffle_xor(lsum, 2u);

            if (lane == 0u) {
                mrow[row] = m_new;
                lrow[row] = lrow[row] * factor + lsum;
                rrow[row] = factor;
            }
        }
        for (uint i = tid; i < FA_SG * 64u; i += FA_TPG) rs[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The running output needs a *per-row* rescale, which a simdgroup
        // matrix cannot express elementwise — but diag(factor) @ O does
        // exactly that, and rides the same MMA path.
        if (tid < FA_BM) {
            uint sg = tid / 8u, r = tid % 8u;
            rs[sg * 64u + r * 8u + r] = rrow[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 rf;
        simdgroup_load(rf, rs + sgid * 64u, 8);
        for (uint dk = 0; dk < NDK; ++dk) {
            simdgroup_float8x8 tmp;
            simdgroup_multiply(tmp, rf, o_frag[dk]);
            o_frag[dk] = tmp;
        }
        // NOTE: hoisting `pf` out of the dk loop (it depends only on j) was
        // tried to cut 64 redundant threadgroup loads per tile per simdgroup.
        // It measured ~11% *slower*: at D=128 the 16 q_frag + 16 o_frag
        // fragments already sit near the register limit, and four more spill.
        // This kernel is register bound, not operand-load bound.
        for (uint dk = 0; dk < NDK; ++dk) {
            for (uint j = 0; j < FA_BN / 8u; ++j) {
                simdgroup_bfloat8x8 pf, vf;
                simdgroup_load(pf, ps + r0 * FA_BN + j * 8u, FA_BN);
                simdgroup_load(vf, vs + (j * 8u) * D + dk * 8u, D);
                simdgroup_multiply_accumulate(o_frag[dk], pf, vf, o_frag[dk]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalise by the softmax denominator, same diagonal-matrix trick.
    for (uint i = tid; i < FA_SG * 64u; i += FA_TPG) rs[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < FA_BM) {
        uint sg = tid / 8u, r = tid % 8u;
        float l = lrow[tid];
        rs[sg * 64u + r * 8u + r] = (l > 0.0f) ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 lf;
    simdgroup_load(lf, rs + sgid * 64u, 8);
    for (uint dk = 0; dk < NDK; ++dk) {
        simdgroup_float8x8 tmp;
        simdgroup_multiply(tmp, lf, o_frag[dk]);
        simdgroup_store(tmp, ostg + r0 * D + dk * 8u, D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < FA_BM * D; i += FA_TPG) {
        uint r = i / D, c = i % D;
        uint gq = q0 + r;
        if (gq < L_q) O[q_nh_base + gq * D + c] = bfloat(ostg[i]);
    }
}

#define FA_TILED_KERNEL(NAME, DVAL)                                  \
kernel void NAME(                                                    \
    const device bfloat* Q   [[buffer(0)]],                          \
    const device bfloat* K   [[buffer(1)]],                          \
    const device bfloat* V   [[buffer(2)]],                          \
    device bfloat*       O   [[buffer(3)]],                          \
    constant uint&    L_q    [[buffer(4)]],                          \
    constant uint&    L_kv   [[buffer(5)]],                          \
    constant float&   scale  [[buffer(6)]],                          \
    threadgroup float* smem  [[threadgroup(0)]],                     \
    uint tid  [[thread_position_in_threadgroup]],                    \
    uint sgid [[simdgroup_index_in_threadgroup]],                    \
    uint gid  [[threadgroup_position_in_grid]]                       \
) {                                                                  \
    flash_attention_tiled_impl<DVAL>(                                \
        Q, K, V, O, L_q, L_kv, scale, smem, tid, sgid, gid);         \
}

// ---------------------------------------------------------------------------
// Head-dim-split variant, for large D.
//
// The kernel above keeps 2*(D/8) simdgroup fragments live per simdgroup —
// q_frag plus the o_frag accumulator. At D=128 that is 32 fragments, past the
// spill boundary: sweeping (h, d) at constant h*d shows d=64 (16 fragments)
// and d=32 (8) both plateau near 1180 GF/s while d=128 collapses to ~708.
//
// This variant halves the fragment count by splitting the head dim across
// simdgroup *pairs*: 8 simdgroups of 256 threads, where simdgroup `sg` owns
// row block `sg/2` and head-dim half `sg%2`, giving (D/16) fragments each for
// q_frag and o_frag — 8+8 at D=128.
//
// Only the score matmul needs combining: each half produces a partial
// Q·K^T over its own columns, summed in threadgroup memory. The PV
// accumulation needs no reduction at all, because each simdgroup already owns
// a disjoint set of output columns.
// ---------------------------------------------------------------------------

#define FA_SG2  8u    // simdgroups per threadgroup (FA_BM/8 row blocks x 2 halves)
#define FA_TPG2 256u  // threads per threadgroup (FA_SG2 * 32)

template<uint D>
void flash_attention_split_impl(
    const device bfloat* Q,
    const device bfloat* K,
    const device bfloat* V,
    device bfloat*       O,
    uint L_q, uint L_kv, uint kv_rows, float scale,
    threadgroup float* smem,
    uint tid, uint sgid, uint gid)
{
    constexpr uint DH  = D / 2u;    // head-dim columns owned by one simdgroup
    constexpr uint NDH = DH / 8u;   // fragments spanning that half

    const uint n_qb = (L_q + FA_BM - 1u) / FA_BM;
    const uint nh   = gid / n_qb;
    const uint qb   = gid % n_qb;
    const uint q0   = qb * FA_BM;

    const uint rb = sgid / 2u;      // row block owned by this simdgroup
    const uint dh = sgid % 2u;      // head-dim half owned by this simdgroup
    const uint r0 = rb * 8u;
    const uint c0 = dh * DH;

    // Region A holds the Q tile (FA_BM*D bfloat) and later an 8 x D f32
    // output slab; sized by the larger, which is the Q tile.
    threadgroup bfloat* tile = (threadgroup bfloat*)smem;   // Q tile
    threadgroup float*  ostg = smem;                        // 8*D float slab
    threadgroup float*  ss   = smem + (FA_BM * D) / 2u;     // 2*FA_BM*FA_BNS float
    threadgroup bfloat* ps   = (threadgroup bfloat*)(ss + 2u * FA_BM * FA_BNS);
    threadgroup float*  mrow = ss + 2u * FA_BM * FA_BNS + (FA_BM * FA_BNS) / 2u;
    threadgroup float*  lrow = mrow + FA_BM;
    threadgroup float*  rrow = lrow + FA_BM;
    // The diagonal-matrix scratch this kernel used to rescale O is gone: the
    // rescale is now a per-lane scalar multiply.

    // The row of an 8x8 fragment this lane owns, per MLX's steel GEMM mapping
    // (verified by `simdgroup_fragment_layout_matches_mlx`). A lane holds one
    // row and two adjacent columns, which is what makes a per-row rescale a
    // plain multiply instead of a diagonal-matrix MMA.
    const uint frag_lane = tid % 32u;
    const uint frag_qid  = frag_lane / 4u;
    const uint frag_row  = (frag_qid & 4u) + ((frag_lane / 2u) % 4u);

    const uint q_nh_base  = nh * L_q  * D;
    const uint kv_nh_base = nh * L_kv * D;

    for (uint i = tid; i < FA_BM * D; i += FA_TPG2) {
        uint r = i / D, c = i % D;
        uint gq = q0 + r;
        tile[i] = (gq < L_q) ? Q[q_nh_base + gq * D + c] : bfloat(0.0f);
    }
    if (tid < FA_BM) { mrow[tid] = -INFINITY; lrow[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_bfloat8x8 q_frag[NDH];
    simdgroup_float8x8  o_frag[NDH];
    for (uint dk = 0; dk < NDH; ++dk) {
        simdgroup_load(q_frag[dk], tile + r0 * D + c0 + dk * 8u, D);
        o_frag[dk] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // K and V are read straight from device memory as simdgroup fragments —
    // no threadgroup staging. The staged version copied FA_BNS*D elements
    // device->threadgroup per tile and then simdgroup_load'd them back out,
    // paying a full round trip plus a barrier for data each fragment reads
    // once. llama.cpp's Metal flash attention does the same thing (direct
    // simdgroup_load from global, no async copy), and the GPU's own cache
    // handles the reuse across simdgroups.
    //
    // Safety: a fragment that starts past L_kv is clamped back onto the last
    // full 8 rows of this head. Its columns are masked to P == 0 by the
    // softmax below, so the values are multiplied away and the clamp only
    // exists to keep the read inside the buffer.
    //
    // This requires L_kv % 8 == 0, which the dispatcher enforces. Clamping a
    // fragment that *straddles* L_kv would slide its live rows down and
    // silently corrupt them — the throughput benchmark cannot see that, and
    // `flash_attention_tiled_matches_scalar_and_cpu` caught it at (33, 17).
    const uint kv_max_base = L_kv - 8u;
    for (uint kv0 = 0; kv0 < L_kv; kv0 += FA_BNS) {
        // Partial S over this simdgroup's head-dim half, into its own buffer.
        for (uint j = 0; j < FA_BNS / 8u; ++j) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            const uint kb = min(kv0 + j * 8u, kv_max_base);
            for (uint dk = 0; dk < NDH; ++dk) {
                simdgroup_bfloat8x8 kf;
                simdgroup_load(kf, K + kv_nh_base + kb * D + c0 + dk * 8u,
                               D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, q_frag[dk], kf, acc);
            }
            simdgroup_store(acc, ss + dh * (FA_BM * FA_BNS) + r0 * FA_BNS + j * 8u, FA_BNS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Combine the two halves, and clear the diagonal scratch while every
        // thread is already here — that lets the softmax write its own
        // rescale slot directly and saves a barrier.
        for (uint i = tid; i < FA_BM * FA_BNS; i += FA_TPG2) {
            ss[i] += ss[FA_BM * FA_BNS + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax across ALL threads, 8 per query row.
        //
        // This used to run under `if (tid < FA_BM)`: 32 of 256 threads, each
        // walking two serial 32-iteration loops with `exp()`, while the other
        // 224 waited at the barrier — every KV tile, ~358 times per query
        // block. That idle 87.5% is what pinned this kernel near 1.05 TF/s
        // regardless of head dim.
        //
        // FA_BM rows x 8 lanes = 256 = FA_TPG2, and the 8 lanes of a row are
        // consecutive `tid`s, so they always sit inside one simdgroup and the
        // row reductions are shuffles rather than threadgroup traffic.
        {
            const uint row  = tid / 8u;
            const uint lane = tid % 8u;
            const uint span = FA_BNS / 8u;
            const uint c0   = lane * span;
            threadgroup float*  rowp  = ss + row * FA_BNS;
            threadgroup bfloat* prowp = ps + row * FA_BNS;

            float m_old = mrow[row];
            float m_loc = m_old;
            for (uint j = c0; j < c0 + span; ++j) {
                if (kv0 + j < L_kv) m_loc = max(m_loc, rowp[j] * scale);
            }
            m_loc = max(m_loc, simd_shuffle_xor(m_loc, 1u));
            m_loc = max(m_loc, simd_shuffle_xor(m_loc, 2u));
            m_loc = max(m_loc, simd_shuffle_xor(m_loc, 4u));
            const float m_new = m_loc;

            // exp(-inf - -inf) is NaN; on the first tile the running max is
            // still -inf and contributes nothing, so pin the factor to 0.
            const float factor = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
            float lsum = 0.0f;
            for (uint j = c0; j < c0 + span; ++j) {
                float p = (kv0 + j < L_kv) ? exp(rowp[j] * scale - m_new) : 0.0f;
                prowp[j] = bfloat(p);
                lsum += p;
            }
            lsum += simd_shuffle_xor(lsum, 1u);
            lsum += simd_shuffle_xor(lsum, 2u);
            lsum += simd_shuffle_xor(lsum, 4u);

            if (lane == 0u) {
                mrow[row] = m_new;
                lrow[row] = lrow[row] * factor + lsum;
                rrow[row] = factor;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale the running output by this row's softmax correction.
        //
        // This built a diagonal matrix in threadgroup memory and ran
        // diag(factor) @ O as NDH extra MMAs per tile — 8 on top of the 128
        // doing real work, plus zeroing and reloading the scratch every tile.
        // A lane owns exactly one fragment row, so it is a scalar multiply.
        const float row_factor = rrow[r0 + frag_row];
        for (uint dk = 0; dk < NDH; ++dk) {
            o_frag[dk].thread_elements()[0] *= row_factor;
            o_frag[dk].thread_elements()[1] *= row_factor;
        }
        // `pf` is reloaded for every (dk, j) rather than hoisted out of the dk
        // loop, even though it depends only on j — 64 threadgroup loads a tile
        // where 8 would do.
        //
        // That looks wasteful and is not. Hoisting it measured 2.05 -> 1.55
        // TF/s here (three runs, no overlap). The tiled kernel carries the same
        // note from an earlier attempt at NDK=16; this was re-tested after the
        // head-dim split halved the live fragment count to NDH=8, and one more
        // fragment held across the inner loop still spills. The kernel is
        // register bound, so a redundant threadgroup load is cheaper than the
        // spill that removing it causes.
        for (uint dk = 0; dk < NDH; ++dk) {
            for (uint j = 0; j < FA_BNS / 8u; ++j) {
                simdgroup_bfloat8x8 pf, vf;
                const uint vb = min(kv0 + j * 8u, kv_max_base);
                simdgroup_load(pf, ps + r0 * FA_BNS + j * 8u, FA_BNS);
                simdgroup_load(vf, V + kv_nh_base + vb * D + c0 + dk * 8u, D);
                simdgroup_multiply_accumulate(o_frag[dk], pf, vf, o_frag[dk]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final 1/l normalisation, in the same per-lane scalar form.
    const float l_row = lrow[r0 + frag_row];
    const float inv_l = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    for (uint dk = 0; dk < NDH; ++dk) {
        o_frag[dk].thread_elements()[0] *= inv_l;
        o_frag[dk].thread_elements()[1] *= inv_l;
    }

    // Write out one row block at a time through an 8 x D staging slab rather
    // than a full FA_BM x D one. Staging the whole tile in f32 cost 16 KB of
    // threadgroup memory for the kernel's entire lifetime — memory that is
    // idle until the last instant, and that capped occupancy at 1.17
    // threadgroups per core. The slab is 4 KB and lives in the region the Q
    // tile already owns.
    //
    // The two simdgroups of a row block hold disjoint head-dim halves, so they
    // write the slab concurrently; only the four row blocks serialise, once
    // per threadgroup rather than once per KV tile.
    for (uint b = 0; b < FA_BM / 8u; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rb == b) {
            for (uint dk = 0; dk < NDH; ++dk) {
                simdgroup_store(o_frag[dk], ostg + c0 + dk * 8u, D);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8u * D; i += FA_TPG2) {
            uint r = i / D, c = i % D;
            uint gq = q0 + b * 8u + r;
            if (gq < L_q) O[q_nh_base + gq * D + c] = bfloat(ostg[i]);
        }
    }
}

// Register-resident flash attention — CORRECT AND SLOWER. Keep it flag-gated.
//
// Measured on M1 Max at 11440 tokens, D=128:
//
//     split (default)          2.07 TF/s
//     reg, FA_BMR=32           0.52 TF/s
//     reg, FA_BMR=64           0.65 TF/s
//
// The idea was sound and is what Metal FlashAttention and llama.cpp describe:
// the split kernel rounds S and P through threadgroup memory — two simdgroups
// each compute half of S over their head-dim half, so combining them needs
// shared memory and P is re-read as fragments, 64 threadgroup loads and three
// barriers a tile. Here each simdgroup owns 8 query rows across the *whole*
// head dim, so S is its own business: accumulated in registers, softmaxed in
// registers, handed straight to the PV matmul. No sharing, one barrier, no P
// traffic.
//
// It loses anyway, and the reason is the trade it cannot avoid: owning the full
// head dim means `o_frag[D/8]` — 16 fragments at D=128 — stays live across the
// entire traversal. That spills, and spilling to device memory costs more than
// the threadgroup round trip it was meant to remove. The split kernel's design
// is the better bargain at this head dim: share S through memory and halve the
// live fragment count.
//
// MFA escapes this with a third blocking dimension along D. At D=128 that is
// precisely what the split kernel already does — it just shares S rather than
// recomputing it per half. So the way past 2.07 TF/s is not this rewrite.
//
// The softmax reduction rides the fragment layout. A lane owns one row and two
// adjacent columns (MLX's mapping, checked by
// `simdgroup_fragment_layout_matches_mlx`), and a row's eight columns sit in
// lanes {L, L+1, L+8, L+9} — so `simd_shuffle_xor` by 1 and by 8 reduces a row
// with no memory at all.
//
// Register budget is the constraint that shapes the rest. `o_frag[D/8]` must
// stay live across the whole traversal (16 fragments at D=128); Q is left in
// threadgroup memory and re-loaded per head-dim step rather than held as
// another 16.
#define FA_BMR 32u   // queries per threadgroup
// KV positions per tile. Swept at D=128, 11440 tokens with FA_NREG=6:
//
//     8 -> 1.86   16 -> 2.79   24 -> 2.39   32 -> 2.60   48 -> 2.46   64 -> 2.71
//
// Notably *against* Metal FlashAttention's 80-128 guidance — this kernel keeps
// its scores in registers, so a wider tile buys no reuse and costs `FA_BNR / 8`
// live score fragments. The optimum is where that pressure and the per-tile
// softmax overhead cross, and it had to be measured.
#define FA_BNR 16u   // kv positions per tile
#define FA_TPGR 128u // 4 simdgroups, one 8-row block each
// Output fragments kept in registers; the rest live in a threadgroup arena
// this kernel manages itself (see the spill note in the body). Swept at D=128,
// 11440 tokens on M1 Max — the curve is flat across 4-6 and falls away on both
// sides, which is the whole point of choosing the split by hand:
//
//     1 -> 1.96   2 -> 1.98   3 -> 1.91   4 -> 2.67
//     5 -> 2.71   6 -> 2.75   8 -> 2.41  12 -> 2.10   16 -> spills, 0.65
#define FA_NREG 6u

template<uint D>
void flash_attention_reg_impl(
    const device bfloat* Q,
    const device bfloat* K,
    const device bfloat* V,
    device bfloat*       O,
    uint L_q, uint L_kv, uint L_valid, float scale,
    threadgroup float* smem,
    uint tid, uint sgid, uint gid)
{
    // L_kv is the per-head row stride of K/V; only the first L_valid rows are
    // keys. Equal for ordinary callers. A caller with a ragged key count pads
    // the stride to a multiple of 8 and passes the true count here, which the
    // P == 0 column mask below then enforces exactly.
    constexpr uint NDK = D / 8u;        // head-dim fragments
    constexpr uint NS  = FA_BNR / 8u;   // score fragments per KV tile

    const uint n_qb = (L_q + FA_BMR - 1u) / FA_BMR;
    const uint nh   = gid / n_qb;
    const uint qb   = gid % n_qb;
    const uint q0   = qb * FA_BMR;

    // Region A: the Q tile as bfloat, then an 8 x D f32 slab on the way out.
    threadgroup bfloat* qtile = (threadgroup bfloat*)smem;
    threadgroup float*  ostg  = smem;

    const uint q_nh_base  = nh * L_q  * D;
    const uint kv_nh_base = nh * L_kv * D;
    const uint r0 = sgid * 8u;          // this simdgroup's first query row

    // This lane's (row, first column) inside an 8x8 fragment.
    const uint lane = tid % 32u;
    const uint qid  = lane / 4u;
    const uint frow = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fcol = (qid & 2u) * 2u + (lane % 2u) * 2u;

    for (uint i = tid; i < FA_BMR * D; i += FA_TPGR) {
        uint r = i / D, c = i % D;
        uint gq = q0 + r;
        qtile[i] = (gq < L_q) ? Q[q_nh_base + gq * D + c] : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hand-managed spilling.
    //
    // Owning the full head dim needs NDK=16 accumulator fragments live across
    // the traversal, and the compiler's answer to that is to spill them to
    // device memory — which is what made the first version of this kernel 3x
    // slower than the split one. So only FA_NREG stay in registers and the rest
    // go to a threadgroup arena, explicitly: loaded once per head-dim step,
    // accumulated over the whole KV tile, stored once. Threadgroup memory is
    // far closer than the stack the compiler would have used, and the traffic
    // is 8 loads and 8 stores a tile instead of whatever spill code it emits.
    simdgroup_float8x8 o_frag[FA_NREG];
    for (uint dk = 0; dk < FA_NREG; ++dk) {
        o_frag[dk] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // Arena: (NDK - FA_NREG) fragments per simdgroup, 64 floats each.
    threadgroup float* arena =
        smem + (FA_BMR * D) / 2u + sgid * (NDK - FA_NREG) * 64u;
    for (uint i = tid % 32u; i < (NDK - FA_NREG) * 64u; i += 32u) {
        arena[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Running softmax statistics for this lane's row. Every lane of a row holds
    // the same values once the shuffles below have run.
    float m_run = -INFINITY;
    float l_run = 0.0f;

    // Fragments starting past L_kv are clamped onto the last full 8 rows; their
    // columns are masked to P == 0, so the values are multiplied away. Needs
    // L_kv % 8 == 0, which the dispatcher enforces.
    const uint kv_max_base = L_kv - 8u;

    for (uint kv0 = 0; kv0 < L_valid; kv0 += FA_BNR) {
        simdgroup_float8x8 s_frag[NS];
        for (uint j = 0; j < NS; ++j) {
            s_frag[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        // dk outermost so Q is fetched once per head-dim step, not once per
        // (step, score fragment).
        for (uint dk = 0; dk < NDK; ++dk) {
            simdgroup_bfloat8x8 qf;
            simdgroup_load(qf, qtile + r0 * D + dk * 8u, D);
            for (uint j = 0; j < NS; ++j) {
                const uint kb = min(kv0 + j * 8u, kv_max_base);
                simdgroup_bfloat8x8 kf;
                simdgroup_load(kf, K + kv_nh_base + kb * D + dk * 8u,
                               D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s_frag[j], qf, kf, s_frag[j]);
            }
        }

        // Row max, in registers.
        float m_loc = m_run;
        for (uint j = 0; j < NS; ++j) {
            const uint gc = kv0 + j * 8u + fcol;
            const float e0 = s_frag[j].thread_elements()[0] * scale;
            const float e1 = s_frag[j].thread_elements()[1] * scale;
            if (gc + 0u < L_valid) m_loc = max(m_loc, e0);
            if (gc + 1u < L_valid) m_loc = max(m_loc, e1);
        }
        m_loc = max(m_loc, simd_shuffle_xor(m_loc, 1u));
        m_loc = max(m_loc, simd_shuffle_xor(m_loc, 8u));
        const float m_new = m_loc;

        // exp(-inf - -inf) is NaN; on the first tile the running max carries
        // nothing, so pin the correction to 0.
        const float factor = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);

        // P, held as two floats per lane per fragment rather than as bfloat
        // fragments — that keeps only `s_frag` and `o_frag` live here.
        float pv[NS][2];
        float lsum = 0.0f;
        for (uint j = 0; j < NS; ++j) {
            const uint gc = kv0 + j * 8u + fcol;
            const float e0 = s_frag[j].thread_elements()[0] * scale;
            const float e1 = s_frag[j].thread_elements()[1] * scale;
            pv[j][0] = (gc + 0u < L_valid) ? exp(e0 - m_new) : 0.0f;
            pv[j][1] = (gc + 1u < L_valid) ? exp(e1 - m_new) : 0.0f;
            lsum += pv[j][0] + pv[j][1];
        }
        lsum += simd_shuffle_xor(lsum, 1u);
        lsum += simd_shuffle_xor(lsum, 8u);

        m_run = m_new;
        l_run = l_run * factor + lsum;

        // Head-dim step outermost so an arena fragment is touched once per
        // tile rather than once per (tile, score fragment).
        for (uint dk = 0; dk < NDK; ++dk) {
            const bool resident = dk < FA_NREG;
            simdgroup_float8x8 acc;
            if (resident) {
                acc = o_frag[dk];
            } else {
                // The store at the end of the previous tile has to be visible
                // to this load. Both are simdgroup-scoped and the arena is
                // private to this simdgroup, so a simdgroup_barrier orders them
                // without any cross-simdgroup synchronisation.
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(acc, arena + (dk - FA_NREG) * 64u, 8);
            }
            acc.thread_elements()[0] *= factor;
            acc.thread_elements()[1] *= factor;
            for (uint j = 0; j < NS; ++j) {
                simdgroup_bfloat8x8 pf;
                pf.thread_elements()[0] = bfloat(pv[j][0]);
                pf.thread_elements()[1] = bfloat(pv[j][1]);
                const uint vb = min(kv0 + j * 8u, kv_max_base);
                simdgroup_bfloat8x8 vf;
                simdgroup_load(vf, V + kv_nh_base + vb * D + dk * 8u, D);
                simdgroup_multiply_accumulate(acc, pf, vf, acc);
            }
            if (resident) {
                o_frag[dk] = acc;
            } else {
                simdgroup_store(acc, arena + (dk - FA_NREG) * 64u, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    const float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    for (uint dk = 0; dk < FA_NREG; ++dk) {
        o_frag[dk].thread_elements()[0] *= inv_l;
        o_frag[dk].thread_elements()[1] *= inv_l;
    }

    // One 8-row block at a time through the slab that region A already owns.
    // The arena sits past region A, so writing the slab does not disturb it.
    for (uint b = 0; b < FA_BMR / 8u; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == b) {
            for (uint dk = 0; dk < NDK; ++dk) {
                if (dk < FA_NREG) {
                    simdgroup_store(o_frag[dk], ostg + dk * 8u, D);
                } else {
                    simdgroup_float8x8 acc;
                    simdgroup_load(acc, arena + (dk - FA_NREG) * 64u, 8);
                    acc.thread_elements()[0] *= inv_l;
                    acc.thread_elements()[1] *= inv_l;
                    simdgroup_store(acc, ostg + dk * 8u, D);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8u * D; i += FA_TPGR) {
            uint r = i / D, c = i % D;
            uint gq = q0 + b * 8u + r;
            if (gq < L_q) O[q_nh_base + gq * D + c] = bfloat(ostg[i]);
        }
    }
}

#define FA_REG_KERNEL(NAME, DVAL)                                    \
kernel void NAME(                                                    \
    const device bfloat* Q   [[buffer(0)]],                          \
    const device bfloat* K   [[buffer(1)]],                          \
    const device bfloat* V   [[buffer(2)]],                          \
    device bfloat*       O   [[buffer(3)]],                          \
    constant uint&    L_q    [[buffer(4)]],                          \
    constant uint&    L_kv   [[buffer(5)]],                          \
    constant float&   scale  [[buffer(6)]],                          \
    constant uint&    L_valid [[buffer(8)]],                         \
    threadgroup float* smem  [[threadgroup(0)]],                     \
    uint tid  [[thread_position_in_threadgroup]],                    \
    uint sgid [[simdgroup_index_in_threadgroup]],                    \
    uint gid  [[threadgroup_position_in_grid]]                       \
) {                                                                  \
    flash_attention_reg_impl<DVAL>(                                  \
        Q, K, V, O, L_q, L_kv, L_valid, scale, smem, tid, sgid, gid);\
}

FA_REG_KERNEL(flash_attention_reg_d96,  96)
FA_REG_KERNEL(flash_attention_reg_d128, 128)

#define FA_SPLIT_KERNEL(NAME, DVAL)                                  \
kernel void NAME(                                                    \
    const device bfloat* Q   [[buffer(0)]],                          \
    const device bfloat* K   [[buffer(1)]],                          \
    const device bfloat* V   [[buffer(2)]],                          \
    device bfloat*       O   [[buffer(3)]],                          \
    constant uint&    L_q    [[buffer(4)]],                          \
    constant uint&    L_kv   [[buffer(5)]],                          \
    constant float&   scale  [[buffer(6)]],                          \
    constant uint&    kv_rows [[buffer(7)]],                         \
    threadgroup float* smem  [[threadgroup(0)]],                     \
    uint tid  [[thread_position_in_threadgroup]],                    \
    uint sgid [[simdgroup_index_in_threadgroup]],                    \
    uint gid  [[threadgroup_position_in_grid]]                       \
) {                                                                  \
    flash_attention_split_impl<DVAL>(                                \
        Q, K, V, O, L_q, L_kv, kv_rows, scale, smem, tid, sgid, gid);\
}

FA_SPLIT_KERNEL(flash_attention_split_d96,  96)
FA_SPLIT_KERNEL(flash_attention_split_d128, 128)

FA_TILED_KERNEL(flash_attention_tiled_d16,  16)
FA_TILED_KERNEL(flash_attention_tiled_d32,  32)
FA_TILED_KERNEL(flash_attention_tiled_d64,  64)
FA_TILED_KERNEL(flash_attention_tiled_d96,  96)
FA_TILED_KERNEL(flash_attention_tiled_d128, 128)


// Permute fp16 [N, L, H*D] → [N, H, L, D]. One thread per output element.
// Used inside `Attention::forward_gpu` to transition from per-token-interleaved
// heads to head-contiguous layout that flash attention requires.
kernel void permute_lh_to_hl_f16(
    device bfloat*     src    [[buffer(0)]],
    device bfloat*       dst    [[buffer(1)]],
    constant uint&     n      [[buffer(2)]],
    constant uint&     l      [[buffer(3)]],
    constant uint&     h      [[buffer(4)]],
    constant uint&     d      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // dst layout: [N, H, L, D]
    uint inner = h * d;
    uint hld = h * l * d;
    uint ld = l * d;
    uint ni = gid / hld;
    uint pos = gid % hld;
    uint hi = pos / ld;
    uint pos2 = pos % ld;
    uint li = pos2 / d;
    uint di = pos2 % d;
    // src layout: [N, L, H*D] → src[ni, li, hi*d + di]
    uint src_idx = ni * l * inner + li * inner + hi * d + di;
    dst[gid] = src[src_idx];
}

// Strided variant of `permute_lh_to_hl_f16`. Same output layout, but reads
// the source as if its inner stride were `inner_stride` (allowing the
// caller to alias one third of a fused [N, L, 3·H·D] QKV-projection buffer
// without a copy). Set buffer offset on the encoder to select Q/K/V slice.
kernel void permute_lh_to_hl_f16_strided(
    device bfloat*     src           [[buffer(0)]],
    device bfloat*     dst           [[buffer(1)]],
    constant uint&     n             [[buffer(2)]],
    constant uint&     l             [[buffer(3)]],
    constant uint&     h             [[buffer(4)]],
    constant uint&     d             [[buffer(5)]],
    constant uint&     inner_stride  [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint hld = h * l * d;
    uint ld = l * d;
    uint ni = gid / hld;
    uint pos = gid % hld;
    uint hi = pos / ld;
    uint pos2 = pos % ld;
    uint li = pos2 / d;
    uint di = pos2 % d;
    // src layout: [N, L, inner_stride] where the slice we want is the first
    // h*d columns. Caller has set buffer offset to skip into the desired
    // [Q | K | V] third of the row.
    uint src_idx = ni * l * inner_stride + li * inner_stride + hi * d + di;
    dst[gid] = src[src_idx];
}

// Inverse: fp16 [N, H, L, D] → [N, L, H*D].
kernel void permute_hl_to_lh_f16(
    device bfloat*     src    [[buffer(0)]],
    device bfloat*       dst    [[buffer(1)]],
    constant uint&     n      [[buffer(2)]],
    constant uint&     l      [[buffer(3)]],
    constant uint&     h      [[buffer(4)]],
    constant uint&     d      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // dst layout: [N, L, H*D]
    uint inner = h * d;
    uint l_inner = l * inner;
    uint ni = gid / l_inner;
    uint pos = gid % l_inner;
    uint li = pos / inner;
    uint pos2 = pos % inner;
    uint hi = pos2 / d;
    uint di = pos2 % d;
    // src layout: [N, H, L, D] → src[ni, hi, li, di]
    uint src_idx = ni * h * l * d + hi * l * d + li * d + di;
    dst[gid] = src[src_idx];
}

// Channel-wise concat: out[N, Ca+Cb, HW] = a[N, Ca, HW] | b[N, Cb, HW].
// One thread per output element. Used in UNet up-path skip-connection merge.
kernel void cat_channels_f16(
    device bfloat*     a   [[buffer(0)]],
    device bfloat*     b   [[buffer(1)]],
    device bfloat*       out [[buffer(2)]],
    constant uint&     ca  [[buffer(3)]],
    constant uint&     cb  [[buffer(4)]],
    constant uint&     hw  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_c = ca + cb;
    uint c_hw = total_c * hw;
    uint ni = gid / c_hw;
    uint pos = gid % c_hw;
    uint ci = pos / hw;
    uint hi = pos % hw;
    if (ci < ca) {
        out[gid] = a[ni * ca * hw + ci * hw + hi];
    } else {
        out[gid] = b[ni * cb * hw + (ci - ca) * hw + hi];
    }
}

// Nearest-neighbor 2× upsample for fp16 [N, C, H, W] → [N, C, 2H, 2W].
// One thread per output element.
// Zero-pad the *end* of each H/W plane by one row and one column.
//
// The VAE encoder's downsample convs are ZeroPad2d((0,1,0,1)) + a stride-2
// kernel-3 conv with no symmetric padding. MPSGraph's conv takes symmetric
// padding only, so the asymmetry is materialised here and the conv then runs
// with pad 0.
// `avg_down_3d`: the encoder down block's shortcut. Folds a (factor_t,
// factor_s, factor_s) neighbourhood plus channels into one averaged output
// channel, mirroring `vae3d::avg_down_3d`. Frames are padded at the *front*,
// and padded positions contribute zero to the sum but still count in the
// divisor — matching the CPU, which skips them and divides by group_size.
kernel void avg_down_3d_f16(
    device bfloat*     input   [[buffer(0)]],
    device bfloat*     output  [[buffer(1)]],
    constant uint&     in_c    [[buffer(2)]],
    constant uint&     in_t    [[buffer(3)]],
    constant uint&     in_h    [[buffer(4)]],
    constant uint&     in_w    [[buffer(5)]],
    constant uint&     out_c   [[buffer(6)]],
    constant uint&     ft      [[buffer(7)]],
    constant uint&     fs      [[buffer(8)]],
    constant uint&     pad_t   [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint factor = ft * fs * fs;
    uint group_size = in_c * factor / out_c;
    uint oh = in_h / fs;
    uint ow = in_w / fs;
    uint ot = (in_t + pad_t) / ft;
    uint total = out_c * ot * oh * ow;
    if (gid >= total) { return; }

    uint w  = gid % ow;
    uint h  = (gid / ow) % oh;
    uint t  = (gid / (ow * oh)) % ot;
    uint oc = gid / (ow * oh * ot);

    float acc = 0.0f;
    for (uint g = 0u; g < group_size; ++g) {
        uint flat = oc * group_size + g;
        uint c    = flat / factor;
        uint rem  = flat % factor;
        uint it   = rem / (fs * fs);
        uint ih   = (rem / fs) % fs;
        uint iw   = rem % fs;
        int ts = (int)(t * ft + it) - (int)pad_t;
        if (ts < 0) { continue; }
        uint si = ((c * in_t + (uint)ts) * in_h + (h * fs + ih)) * in_w + (w * fs + iw);
        acc += (float)input[si];
    }
    output[gid] = (bfloat)(acc / (float)group_size);
}


// Confirm the simdgroup fragment layout against MLX's documented mapping.
//
// MLX's steel GEMM (mlx/backend/metal/kernels/steel/gemm/mma.h) states that for
// an 8x8 fragment each lane owns one row and two consecutive columns:
//
//     qid = lane / 4;  fm = (qid & 4) + ((lane / 2) % 4);
//                      fn = (qid & 2) * 2 + (lane % 2) * 2;
//
// If that holds, a per-row rescale of the attention accumulator is a plain
// multiply on each lane's two elements — no diagonal-matrix MMA, no
// threadgroup round trip. Worth confirming on the hardware rather than trusting
// a header: a wrong mapping corrupts output silently.
kernel void simdgroup_layout_probe(
    device const float* input  [[buffer(0)]],   // 64 floats, value = row*8+col
    device float*       output [[buffer(1)]],   // 32 lanes x 2 elements
    uint tid [[thread_position_in_threadgroup]]
) {
    simdgroup_float8x8 m;
    simdgroup_load(m, input, 8);
    output[tid * 2u + 0u] = m.thread_elements()[0];
    output[tid * 2u + 1u] = m.thread_elements()[1];
}

kernel void pad_end_1_f16(
    device bfloat*     input  [[buffer(0)]],
    device bfloat*     output [[buffer(1)]],
    constant uint&     c      [[buffer(2)]],
    constant uint&     h      [[buffer(3)]],
    constant uint&     w      [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint w_out = w + 1u;
    uint h_out = h + 1u;
    uint hw_out = h_out * w_out;
    uint ci = gid / hw_out;
    uint pos = gid % hw_out;
    uint hi = pos / w_out;
    uint wi = pos % w_out;
    if (ci >= c) { return; }
    if (hi >= h || wi >= w) { output[gid] = (bfloat)0.0f; return; }
    output[gid] = input[ci * h * w + hi * w + wi];
}

kernel void nearest_upsample_2x_f16(
    device bfloat*     input  [[buffer(0)]],
    device bfloat*       output [[buffer(1)]],
    constant uint&     c      [[buffer(2)]],
    constant uint&     h      [[buffer(3)]],
    constant uint&     w      [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint w_up  = w * 2u;
    uint h_up  = h * 2u;
    uint hw_up = h_up * w_up;
    uint chw_up = c * hw_up;
    uint ni = gid / chw_up;
    uint pos_in_n = gid % chw_up;
    uint ci = pos_in_n / hw_up;
    uint pos_in_c = pos_in_n % hw_up;
    uint hi_up = pos_in_c / w_up;
    uint wi_up = pos_in_c % w_up;
    uint hi_in = hi_up / 2u;
    uint wi_in = wi_up / 2u;
    uint in_idx = ni * c * h * w + ci * h * w + hi_in * w + wi_in;
    output[gid] = input[in_idx];
}

// Per-batch per-channel bias add: x[n, c, hw] += bias[n, c]. Used by
// ResnetBlock to inject the time-embedding projection into the conv output.
kernel void bias_add_nc_to_nchw_f16(
    device bfloat*       x      [[buffer(0)]],
    device bfloat*     bias   [[buffer(1)]],
    constant uint&     c      [[buffer(2)]],
    constant uint&     hw     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint chw = c * hw;
    uint ni = gid / chw;
    uint pos_in_n = gid % chw;
    uint ci = pos_in_n / hw;
    uint bias_idx = ni * c + ci;
    x[gid] = bfloat(float(x[gid]) + float(bias[bias_idx]));
}

// Convert f32 → f16 GPU buffer. **Use `device` for src** — `constant`
// has a size limit (~64KB on Apple Silicon) and these are dispatched on
// multi-MB buffers in the f16-via-fp32 sgemm wrapper.
kernel void cast_f32_to_f16(
    device float* src [[buffer(0)]],
    device bfloat* dst    [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = bfloat(src[gid]);
}

// Convert f16 → f32 GPU buffer.
kernel void cast_f16_to_f32(
    device bfloat* src [[buffer(0)]],
    device float* dst  [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = float(src[gid]);
}

// Cast bf16 → IEEE half. Used by the sgemm wrapper when MPS f16 (faster)
// is safe — the temporary scratch buffer is half-precision, but storage
// stays bf16 between layers so we keep the f32 dynamic range across the
// network.
kernel void copy_f16(
    device half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = src[gid];
}

kernel void cast_half_to_f32(
    device half*  src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = float(src[gid]);
}

kernel void cast_f32_to_half(
    device float* src [[buffer(0)]],
    device half*  dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = half(src[gid]);
}

kernel void cast_bf16_to_half(
    device bfloat* src [[buffer(0)]],
    device half*   dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = half(float(src[gid]));
}

// Cast IEEE half → bf16 (after MPS f16 sgemm, before storing back to the
// network's bf16 activation cache).
kernel void cast_half_to_bf16(
    device half*   src [[buffer(0)]],
    device bfloat* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = bfloat(float(src[gid]));
}

// Flash attention: fused QK^T + softmax + ·V with online softmax.
// Inputs Q, K, V each [N, H, L, D] in head-contiguous layout. Output O same.
// One threadgroup per (n, h, l_q) row — gid = nh * L_q + lq. Threadgroup
// size = next_pow2(D); first D threads hold q[d]/o[d], remainder zero out.
// Threadgroup memory: 3 * tpg floats (k_row | v_row | reduce_scratch).
//
// Memory footprint per row is O(D), not O(L_kv). We never materialise the
// score matrix — for SDXL self-attn at 32×32 with H=20 that's a 320MB
// intermediate eliminated.
kernel void flash_attention(
    const device float* Q       [[buffer(0)]],
    const device float* K       [[buffer(1)]],
    const device float* V       [[buffer(2)]],
    device float*       O       [[buffer(3)]],
    constant uint&      L_q     [[buffer(4)]],
    constant uint&      L_kv    [[buffer(5)]],
    constant uint&      D       [[buffer(6)]],
    constant float&     scale   [[buffer(7)]],
    threadgroup float*  smem    [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tpg     [[threads_per_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]]
) {
    uint nh = gid / L_q;
    uint lq = gid % L_q;
    uint q_base = (nh * L_q + lq) * D;
    uint kv_nh_base = nh * L_kv * D;

    threadgroup float* k_row  = smem;
    threadgroup float* v_row  = smem + tpg;
    threadgroup float* reduce = smem + 2u * tpg;

    // Pre-scale q so we don't multiply per-K-row inside the loop.
    float q_d = (tid < D) ? Q[q_base + tid] * scale : 0.0;
    float o_d = 0.0;

    // Online softmax statistics shared across the threadgroup.
    threadgroup float m_tg;
    threadgroup float l_tg;
    if (tid == 0) {
        m_tg = -INFINITY;
        l_tg = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint lk = 0; lk < L_kv; ++lk) {
        // Load one K, V row cooperatively. Threads tid >= D zero-pad.
        if (tid < D) {
            k_row[tid] = K[kv_nh_base + lk * D + tid];
            v_row[tid] = V[kv_nh_base + lk * D + tid];
        } else {
            k_row[tid] = 0.0;
            v_row[tid] = 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // s = q · k_row (q_d already includes scale; padded threads contribute 0).
        reduce[tid] = q_d * k_row[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpg / 2u; stride > 0u; stride /= 2u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float s = reduce[0];

        // Online softmax update — every thread computes the same scalars.
        // First iteration: m_tg = -INFINITY, m_new = s, factor = exp(-INF) = 0
        // — so the prior accumulation in o_d (initialised to 0) and l_tg
        // gets correctly dropped. No special-case needed.
        float m_old = m_tg;
        float m_new = max(m_old, s);
        float factor = exp(m_old - m_new);
        float p      = exp(s - m_new);

        if (tid < D) {
            o_d = o_d * factor + p * v_row[tid];
        }
        // Every thread read m_tg above and thread 0 overwrites it below. With
        // tpg up to 128 (four simdgroups) those do not run in lockstep, so
        // without this barrier thread 0 can clobber m_tg before another
        // simdgroup has read it — a rare, data-dependent wrong answer.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            l_tg = l_tg * factor + p;
            m_tg = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalisation. The barrier above ensures all threads see the
    // last l_tg written by tid 0.
    if (tid < D) {
        O[q_base + tid] = o_d / l_tg;
    }
}

// Causal flash attention (f32). Same as `flash_attention` plus a causal
// mask: when lk > lq, that K/V row contributes -INFINITY to s, so its p =
// exp(s − m_new) = 0 and the iteration is a no-op (factor = 1, m_new
// unchanged). The L_kv loop still runs to keep barriers synchronised
// across the threadgroup; the masked iterations are cheap, and the
// real-world gain is having a single GPU kernel handle the per-query
// SDPA in decoder-only transformers (klearu-image).
kernel void flash_attention_causal(
    const device float* Q       [[buffer(0)]],
    const device float* K       [[buffer(1)]],
    const device float* V       [[buffer(2)]],
    device float*       O       [[buffer(3)]],
    constant uint&      L_q     [[buffer(4)]],
    constant uint&      L_kv    [[buffer(5)]],
    constant uint&      D       [[buffer(6)]],
    constant float&     scale   [[buffer(7)]],
    threadgroup float*  smem    [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tpg     [[threads_per_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]]
) {
    uint nh = gid / L_q;
    uint lq = gid % L_q;
    uint q_base = (nh * L_q + lq) * D;
    uint kv_nh_base = nh * L_kv * D;

    threadgroup float* k_row  = smem;
    threadgroup float* v_row  = smem + tpg;
    threadgroup float* reduce = smem + 2u * tpg;

    float q_d = (tid < D) ? Q[q_base + tid] * scale : 0.0;
    float o_d = 0.0;

    threadgroup float m_tg;
    threadgroup float l_tg;
    if (tid == 0) {
        m_tg = -INFINITY;
        l_tg = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint lk = 0; lk < L_kv; ++lk) {
        if (tid < D) {
            k_row[tid] = K[kv_nh_base + lk * D + tid];
            v_row[tid] = V[kv_nh_base + lk * D + tid];
        } else {
            k_row[tid] = 0.0;
            v_row[tid] = 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        reduce[tid] = q_d * k_row[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpg / 2u; stride > 0u; stride /= 2u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float s = (lk > lq) ? -INFINITY : reduce[0];

        float m_old = m_tg;
        float m_new = max(m_old, s);
        float factor = exp(m_old - m_new);
        float p      = exp(s - m_new);

        if (tid < D) {
            o_d = o_d * factor + p * v_row[tid];
        }
        // Every thread read m_tg above and thread 0 overwrites it below. With
        // tpg up to 128 (four simdgroups) those do not run in lockstep, so
        // without this barrier thread 0 can clobber m_tg before another
        // simdgroup has read it — a rare, data-dependent wrong answer.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            l_tg = l_tg * factor + p;
            m_tg = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < D) {
        // l_tg may be 0 if every K position was masked (lq == 0 path is
        // fine because lk=0 satisfies lk <= lq). Guard against division
        // by zero defensively.
        O[q_base + tid] = (l_tg > 0.0) ? (o_d / l_tg) : 0.0;
    }
}

// RMSNorm (f32). y_i = gamma_i · x_i / sqrt(mean(x²) + eps).
// One threadgroup per row. Matches `layer_norm` shape conventions:
//   x:     [N, dim] flat, row-major. In place.
//   gamma: [dim]
kernel void rms_norm(
    device float*       x       [[buffer(0)]],
    constant float*     gamma   [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    constant float&     eps     [[buffer(3)]],
    threadgroup float*  scratch [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tpg     [[threads_per_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]]
) {
    uint row = gid;
    uint row_off = row * dim;

    float lss = 0.0;
    for (uint i = tid; i < dim; i += tpg) {
        float v = x[row_off + i];
        lss += v * v;
    }
    scratch[tid] = lss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = 1.0 / sqrt(scratch[0] / float(dim) + eps);

    for (uint i = tid; i < dim; i += tpg) {
        x[row_off + i] = x[row_off + i] * inv_rms * gamma[i];
    }
}

// Temporal reshape: [B*F, C, H, W] (NCHW) -> [B*H*W, F, C].
// One thread per output element.
kernel void temporal_reshape_to_f16(
    device bfloat*     src     [[buffer(0)]],
    device bfloat*     dst     [[buffer(1)]],
    constant uint&     batch   [[buffer(2)]],
    constant uint&     frames  [[buffer(3)]],
    constant uint&     c       [[buffer(4)]],
    constant uint&     h       [[buffer(5)]],
    constant uint&     w       [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint hw = h * w;
    uint fc = frames * c;
    uint bhw_idx = gid / fc;
    uint pos_fc = gid % fc;
    uint fi = pos_fc / c;
    uint ci = pos_fc % c;
    uint bi = bhw_idx / hw;
    uint pos_hw = bhw_idx % hw;
    uint hi = pos_hw / w;
    uint wi = pos_hw % w;
    uint src_n = bi * frames + fi;
    dst[gid] = src[src_n * c * hw + ci * hw + hi * w + wi];
}

// Inverse temporal reshape: [B*H*W, F, C] -> [B*F, C, H, W].
kernel void temporal_reshape_from_f16(
    device bfloat*     src     [[buffer(0)]],
    device bfloat*     dst     [[buffer(1)]],
    constant uint&     batch   [[buffer(2)]],
    constant uint&     frames  [[buffer(3)]],
    constant uint&     c       [[buffer(4)]],
    constant uint&     h       [[buffer(5)]],
    constant uint&     w       [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint hw = h * w;
    uint chw = c * hw;
    uint bf_idx = gid / chw;
    uint pos_chw = gid % chw;
    uint ci = pos_chw / hw;
    uint pos_hw = pos_chw % hw;
    uint hi = pos_hw / w;
    uint wi = pos_hw % w;
    uint bi = bf_idx / frames;
    uint fi = bf_idx % frames;
    uint bhw_idx = bi * hw + hi * w + wi;
    uint fc = frames * c;
    dst[gid] = src[bhw_idx * fc + fi * c + ci];
}

// Add positional embedding: x[n_seq * f * d] += pe[f * d], broadcast across n_seq.
kernel void pos_embed_add_f16(
    device bfloat*       x       [[buffer(0)]],
    device bfloat*       pe      [[buffer(1)]],
    constant uint&       frames  [[buffer(2)]],
    constant uint&       max_f   [[buffer(3)]],
    constant uint&       d       [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint fd = frames * d;
    uint fi = (gid % fd) / d;
    uint di = gid % d;
    if (fi < max_f) {
        x[gid] = bfloat(float(x[gid]) + float(pe[fi * d + di]));
    }
}

// ---- Flux MMDiT kernels ---------------------------------------------------

// Exact (erf-based) GELU: 0.5*v*(1+erf(v/sqrt2)). Flux uses the exact GELU,
// not the tanh approximation in `gelu_f16`. erf via Abramowitz-Stegun 7.1.26
// (matches the CPU `erf_approx` coefficients exactly for GPU/CPU parity).
kernel void gelu_erf_f16(
    device bfloat* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float v = float(x[gid]);
    float xin = v * 0.7071067811865476f; // v / sqrt(2)
    float sign = xin < 0.0f ? -1.0f : 1.0f;
    float xa = fabs(xin);
    float t = 1.0f / (1.0f + 0.3275911f * xa);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                       - 0.284496736f) * t + 0.254829592f) * t * exp(-xa * xa);
    float erf = sign * y;
    x[gid] = bfloat(0.5f * v * (1.0f + erf));
}

// Interleaved-pair RoPE (Flux convention): rotates pairs (2k, 2k+1) using
// cos/sin indexed at 2k (build_rope_2d duplicates cos[2k]==cos[2k+1]). This
// differs from `rope_apply_f16` which uses the cat-half (NeoX) pairing.
kernel void rope_pairs_f16(
    device bfloat*       x    [[buffer(0)]],  // [n, num_heads, head_dim]
    device const bfloat* cosb [[buffer(1)]],  // [n, head_dim]
    device const bfloat* sinb [[buffer(2)]],
    constant uint&       num_heads [[buffer(3)]],
    constant uint&       head_dim  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]       // in [0, n*num_heads)
) {
    uint t = gid / num_heads;
    uint hd2 = head_dim / 2;
    uint base = gid * head_dim;
    uint cbase = t * head_dim;
    for (uint k = 0; k < hd2; ++k) {
        float a = float(x[base + 2 * k]);
        float b = float(x[base + 2 * k + 1]);
        float c = float(cosb[cbase + 2 * k]);
        float s = float(sinb[cbase + 2 * k]);
        x[base + 2 * k]     = bfloat(a * c - b * s);
        x[base + 2 * k + 1] = bfloat(a * s + b * c);
    }
}

// AdaLN modulate: x[t,di] = x[t,di] * (1 + scale[di]) + shift[di], broadcasting
// scale/shift [d] across the n token rows.
kernel void modulate_f16(
    device bfloat*       x     [[buffer(0)]],  // [n, d]
    device const bfloat* scale [[buffer(1)]],  // [d]
    device const bfloat* shift [[buffer(2)]],  // [d]
    constant uint&       d     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]        // in [0, n*d)
) {
    uint di = gid % d;
    float v = float(x[gid]);
    x[gid] = bfloat(v * (1.0f + float(scale[di])) + float(shift[di]));
}

// Column slice: out[t, j] = in[t, off + j], i.e. extract a contiguous
// `blk`-wide column block from a row-major `[n, stride]` tensor. Used to split
// fused QKV(/MLP) projections into separate q/k/v tensors.
kernel void slice_cols_f16(
    device const bfloat* inp [[buffer(0)]],  // [n, stride]
    device bfloat*       out [[buffer(1)]],  // [n, blk]
    constant uint&       stride [[buffer(2)]],
    constant uint&       off    [[buffer(3)]],
    constant uint&       blk    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]       // [0, n*blk)
) {
    uint t = gid / blk;
    uint j = gid % blk;
    out[gid] = inp[t * stride + off + j];
}

// Gate: x[t,di] *= gate[di], broadcasting gate [d] across n token rows.
kernel void gate_mul_f16(
    device bfloat*       x    [[buffer(0)]],  // [n, d]
    device const bfloat* gate [[buffer(1)]],  // [d]
    constant uint&       d    [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint di = gid % d;
    x[gid] = bfloat(float(x[gid]) * float(gate[di]));
}

// ---- Wan AdaLN modulation ------------------------------------------------
//
// `modulate_f16` / `gate_mul_f16` above broadcast one [d] vector over every
// row. Wan 2.2 TI2V modulates **per token**, so scale/shift/gate vary by row.
// A `temb_stride` of 0 collapses these back to the broadcast case, which is
// the Wan 2.1 / 14B path — one kernel serves both.
//
// The effective modulation is always the block's `scale_shift_table` plus the
// timestep projection, so both are passed and summed here rather than
// materialising the sum (which at 30k tokens x 6 x 3072 would be large).

// x[t, i] = x[t, i] * (1 + scale) + shift
kernel void modulate_rows_f16(
    device bfloat*       x         [[buffer(0)]],  // [n, d]
    device const bfloat* table     [[buffer(1)]],  // [6, d]
    device const bfloat* temb      [[buffer(2)]],  // [n, 6, d] or [6, d]
    constant uint&       d         [[buffer(3)]],
    constant uint&       temb_str  [[buffer(4)]],  // 6*d per token, 0 to broadcast
    constant uint&       scale_i   [[buffer(5)]],
    constant uint&       shift_i   [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint ti = gid / d;
    uint di = gid % d;
    uint base = ti * temb_str;
    float scale = float(table[scale_i * d + di]) + float(temb[base + scale_i * d + di]);
    float shift = float(table[shift_i * d + di]) + float(temb[base + shift_i * d + di]);
    x[gid] = bfloat(float(x[gid]) * (1.0f + scale) + shift);
}

// x[t, i] *= gate
kernel void gate_rows_f16(
    device bfloat*       x         [[buffer(0)]],
    device const bfloat* table     [[buffer(1)]],
    device const bfloat* temb      [[buffer(2)]],
    constant uint&       d         [[buffer(3)]],
    constant uint&       temb_str  [[buffer(4)]],
    constant uint&       gate_i    [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint ti = gid / d;
    uint di = gid % d;
    float g = float(table[gate_i * d + di])
            + float(temb[ti * temb_str + gate_i * d + di]);
    x[gid] = bfloat(float(x[gid]) * g);
}

// Wan VAE channel norm: normalise across the CHANNEL axis at each spatial
// position. Layout is [C, P] with P = T*H*W, so channels are the *outer*
// dimension and the reduction is strided by P — unlike `rms_norm_f16`, which
// reduces over a contiguous row.
//
// Semantics follow `F.normalize(x, dim=1) * sqrt(C) * gamma`: the L2 norm is
// **clamped** at 1e-12 rather than eps being added under the root, so an
// all-zero position comes out zero rather than NaN.
kernel void rms_norm_channels_f16(
    device bfloat*       x     [[buffer(0)]],  // [C, P]
    device const bfloat* gamma [[buffer(1)]],  // [C]
    constant uint&       c     [[buffer(2)]],
    constant uint&       p     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]        // in [0, P)
) {
    float sum = 0.0f;
    for (uint i = 0; i < c; ++i) {
        float v = float(x[i * p + gid]);
        sum += v * v;
    }
    float inv = 1.0f / max(sqrt(sum), 1e-12f);
    float scale = sqrt(float(c));
    for (uint i = 0; i < c; ++i) {
        x[i * p + gid] = bfloat(float(x[i * p + gid]) * inv * scale * float(gamma[i]));
    }
}

// Row softmax with the attention scale folded in, for the VAE's single-head
// spatial attention. One thread per row; the row length is the number of
// spatial positions, ~1000 at the mid block's resolution.
kernel void softmax_rows_scaled_f16(
    device bfloat*    x     [[buffer(0)]],
    constant uint&    width [[buffer(1)]],
    constant float&   scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = gid * width;
    float m = -INFINITY;
    for (uint j = 0; j < width; ++j) {
        m = max(m, float(x[base + j]) * scale);
    }
    float sum = 0.0f;
    for (uint j = 0; j < width; ++j) {
        float e = exp(float(x[base + j]) * scale - m);
        x[base + j] = bfloat(e);
        sum += e;
    }
    float inv = 1.0f / sum;
    for (uint j = 0; j < width; ++j) {
        x[base + j] = bfloat(float(x[base + j]) * inv);
    }
}

// Pull one frame's channel slice out of a [total_c, t, p] volume, transposed
// to [p, c] — the layout the attention matmuls want. `c0` selects q, k or v
// from a fused qkv tensor.
kernel void extract_frame_pc_f16(
    device const bfloat* src   [[buffer(0)]],
    device bfloat*       dst   [[buffer(1)]],
    constant uint&       c0    [[buffer(2)]],
    constant uint&       c     [[buffer(3)]],
    constant uint&       t_idx [[buffer(4)]],
    constant uint&       t     [[buffer(5)]],
    constant uint&       p     [[buffer(6)]],
    uint gid [[thread_position_in_grid]]   // over p*c
) {
    uint ci = gid % c;
    uint pi = gid / c;
    dst[gid] = src[((c0 + ci) * t + t_idx) * p + pi];
}

// Inverse: scatter a [p, c] frame back into a [c, t, p] volume.
kernel void insert_frame_pc_f16(
    device const bfloat* src   [[buffer(0)]],
    device bfloat*       dst   [[buffer(1)]],
    constant uint&       c     [[buffer(2)]],
    constant uint&       t_idx [[buffer(3)]],
    constant uint&       t     [[buffer(4)]],
    constant uint&       p     [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint ci = gid % c;
    uint pi = gid / c;
    dst[(ci * t + t_idx) * p + pi] = src[gid];
}

// Wan VAE `DupUp3D`: duplicate channels, then unfold them into time and space.
// `drop` removes the leading `factor_t - 1` frames on the first chunk, which is
// what makes the first latent frame expand to one output frame rather than
// `factor_t`.
kernel void dup_up_3d_f16(
    device const bfloat* src   [[buffer(0)]],   // [in_c, t, h, w]
    device bfloat*       dst   [[buffer(1)]],   // [out_c, t*ft - drop, h*fs, w*fs]
    constant uint&       in_c  [[buffer(2)]],
    constant uint&       out_c [[buffer(3)]],
    constant uint&       t     [[buffer(4)]],
    constant uint&       h     [[buffer(5)]],
    constant uint&       w     [[buffer(6)]],
    constant uint&       ft    [[buffer(7)]],
    constant uint&       fs    [[buffer(8)]],
    constant uint&       drop  [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint ow = w * fs;
    uint oh = h * fs;
    uint ot = t * ft - drop;
    uint wi = gid % ow;
    uint hi = (gid / ow) % oh;
    uint ti = (gid / (ow * oh)) % ot;
    uint oc = gid / (ow * oh * ot);

    uint gt = ti + drop;
    uint tt = gt / ft, it = gt % ft;
    uint hh = hi / fs, ih = hi % fs;
    uint ww = wi / fs, iw = wi % fs;

    uint factor = ft * fs * fs;
    uint repeats = out_c * factor / in_c;
    uint flat = ((oc * ft + it) * fs + ih) * fs + iw;
    uint src_c = flat / repeats;
    dst[gid] = src[(src_c * t + tt) * (h * w) + hh * w + ww];
}

// `upsample3d`'s temporal expansion: the time conv doubles the channel count,
// and the two halves interleave into consecutive frames.
// src [2c, t, p] -> dst [c, 2t, p], dst[ci, 2*tt + k] = src[k*c + ci, tt].
kernel void interleave_time_f16(
    device const bfloat* src [[buffer(0)]],
    device bfloat*       dst [[buffer(1)]],
    constant uint&       c   [[buffer(2)]],
    constant uint&       t   [[buffer(3)]],
    constant uint&       p   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pi = gid % p;
    uint ti = (gid / p) % (2u * t);
    uint ci = gid / (p * 2u * t);
    uint tt = ti / 2u;
    uint k = ti % 2u;
    dst[gid] = src[((k * c + ci) * t + tt) * p + pi];
}

// Swap the two outer axes of a [A, B, P] volume -> [B, A, P]. Used to view a
// [C, T, H*W] video volume as [T, C, H*W] for batched 2D convolution, and back.
kernel void transpose_outer_f16(
    device const bfloat* src [[buffer(0)]],
    device bfloat*       dst [[buffer(1)]],
    constant uint&       a   [[buffer(2)]],
    constant uint&       b   [[buffer(3)]],
    constant uint&       p   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]   // in [0, a*b*p)
) {
    uint pi = gid % p;
    uint bi = (gid / p) % b;
    uint ai = gid / (p * b);
    dst[(bi * a + ai) * p + pi] = src[gid];
}

// out[dst_off + i] += src[src_off + i], over `len` elements. Used to fold each
// temporal tap of a causal conv3d into the accumulated output at its own
// frame offset.
kernel void add_shifted_f16(
    device bfloat*       dst     [[buffer(0)]],
    device const bfloat* src     [[buffer(1)]],
    constant uint&       dst_off [[buffer(2)]],
    constant uint&       src_off [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[dst_off + gid] = bfloat(float(dst[dst_off + gid]) + float(src[src_off + gid]));
}

// Final output norm modulation. Unlike the per-block case, shift and scale add
// the *same* temb row; the two halves of the [2, d] table separate them.
kernel void final_modulate_f16(
    device bfloat*       x         [[buffer(0)]],  // [n, d]
    device const bfloat* table     [[buffer(1)]],  // [2, d]
    device const bfloat* temb      [[buffer(2)]],  // [n, d] or [d]
    constant uint&       d         [[buffer(3)]],
    constant uint&       temb_str  [[buffer(4)]],  // d per token, 0 to broadcast
    uint gid [[thread_position_in_grid]]
) {
    uint ti = gid / d;
    uint di = gid % d;
    float t = float(temb[ti * temb_str + di]);
    float shift = float(table[di]) + t;
    float scale = float(table[d + di]) + t;
    x[gid] = bfloat(float(x[gid]) * (1.0f + scale) + shift);
}


// ---------- IEEE f16 twins of the VAE decoder kernels --------------------
//
// Identical bodies, `half` storage instead of `bfloat`. These exist because the
// GPU VAE decoder sat at cos_sim 0.9985 against diffusers while the CPU decoder
// was exact, and a per-stage bisection showed the error was distributed rather
// than from any one op: ~0.2% per store (bf16 has 8 mantissa bits) over ~100
// sequential stores. half has 10 mantissa bits, so ~4x less rounding for the
// same 2 bytes, and decoder activations peak near 30 against half's 65504 limit
// so the range is not a concern.
//
// Every kernel here uses its storage type only to load and store, computing in
// float throughout, which is what makes the substitution safe and mechanical.

kernel void silu_h16(
    device half* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float v = float(x[gid]);
    x[gid] = half(v / (1.0 + exp(-v)));
}
kernel void elementwise_mul_h16(
    device half*       a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    a[gid] = half(float(a[gid]) * float(b[gid]));
}
kernel void rms_norm_wf32_h16(
    device half*        x      [[buffer(0)]],   // [n_rows, dim]
    device const float* weight [[buffer(1)]],   // [dim]
    constant uint&      dim    [[buffer(2)]],
    constant float&     eps    [[buffer(3)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    uint base = row * dim;
    float ss = 0.0f;
    for (uint i = tid; i < dim; i += tpg) { float v = float(x[base + i]); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint nsg = (tpg + 31) / 32;
        float t = lane < nsg ? partial[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) partial[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(partial[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += tpg) {
        x[base + i] = half(float(x[base + i]) * inv * weight[i]);
    }
}
kernel void sigmoid_gate_h16(
    device half*       a [[buffer(0)]],
    device const half* g [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float gv = float(g[gid]);
    a[gid] = half(float(a[gid]) / (1.0f + exp(-gv)));
}
kernel void elementwise_add_h16(
    device half*       a [[buffer(0)]],
    device half*     b [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    a[gid] = half(float(a[gid]) + float(b[gid]));
}
kernel void bias_add_strided_h16(
    device half*       x          [[buffer(0)]],
    device half*     bias       [[buffer(1)]],
    constant uint&     period     [[buffer(2)]],
    constant uint&     bias_len   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint bias_idx = (gid / period) % bias_len;
    x[gid] = half(float(x[gid]) + float(bias[bias_idx]));
}
kernel void bias_add_nc_to_nchw_h16(
    device half*       x      [[buffer(0)]],
    device half*     bias   [[buffer(1)]],
    constant uint&     c      [[buffer(2)]],
    constant uint&     hw     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint chw = c * hw;
    uint ni = gid / chw;
    uint pos_in_n = gid % chw;
    uint ci = pos_in_n / hw;
    uint bias_idx = ni * c + ci;
    x[gid] = half(float(x[gid]) + float(bias[bias_idx]));
}
// Zero-pad the *end* of each H/W plane by one row and one column.
//
// The VAE encoder's downsample convs are ZeroPad2d((0,1,0,1)) + a stride-2
// kernel-3 conv with no symmetric padding. MPSGraph's conv takes symmetric
// padding only, so the asymmetry is materialised here and the conv then runs
// with pad 0.
// `avg_down_3d`: the encoder down block's shortcut. Folds a (factor_t,
// factor_s, factor_s) neighbourhood plus channels into one averaged output
// channel, mirroring `vae3d::avg_down_3d`. Frames are padded at the *front*,
// and padded positions contribute zero to the sum but still count in the
// divisor — matching the CPU, which skips them and divides by group_size.
kernel void avg_down_3d_h16(
    device half*     input   [[buffer(0)]],
    device half*     output  [[buffer(1)]],
    constant uint&     in_c    [[buffer(2)]],
    constant uint&     in_t    [[buffer(3)]],
    constant uint&     in_h    [[buffer(4)]],
    constant uint&     in_w    [[buffer(5)]],
    constant uint&     out_c   [[buffer(6)]],
    constant uint&     ft      [[buffer(7)]],
    constant uint&     fs      [[buffer(8)]],
    constant uint&     pad_t   [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint factor = ft * fs * fs;
    uint group_size = in_c * factor / out_c;
    uint oh = in_h / fs;
    uint ow = in_w / fs;
    uint ot = (in_t + pad_t) / ft;
    uint total = out_c * ot * oh * ow;
    if (gid >= total) { return; }

    uint w  = gid % ow;
    uint h  = (gid / ow) % oh;
    uint t  = (gid / (ow * oh)) % ot;
    uint oc = gid / (ow * oh * ot);

    float acc = 0.0f;
    for (uint g = 0u; g < group_size; ++g) {
        uint flat = oc * group_size + g;
        uint c    = flat / factor;
        uint rem  = flat % factor;
        uint it   = rem / (fs * fs);
        uint ih   = (rem / fs) % fs;
        uint iw   = rem % fs;
        int ts = (int)(t * ft + it) - (int)pad_t;
        if (ts < 0) { continue; }
        uint si = ((c * in_t + (uint)ts) * in_h + (h * fs + ih)) * in_w + (w * fs + iw);
        acc += (float)input[si];
    }
    output[gid] = (half)(acc / (float)group_size);
}

kernel void pad_end_1_h16(
    device half*     input  [[buffer(0)]],
    device half*     output [[buffer(1)]],
    constant uint&     c      [[buffer(2)]],
    constant uint&     h      [[buffer(3)]],
    constant uint&     w      [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint w_out = w + 1u;
    uint h_out = h + 1u;
    uint hw_out = h_out * w_out;
    uint ci = gid / hw_out;
    uint pos = gid % hw_out;
    uint hi = pos / w_out;
    uint wi = pos % w_out;
    if (ci >= c) { return; }
    if (hi >= h || wi >= w) { output[gid] = (half)0.0f; return; }
    output[gid] = input[ci * h * w + hi * w + wi];
}

kernel void nearest_upsample_2x_h16(
    device half*     input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     c      [[buffer(2)]],
    constant uint&     h      [[buffer(3)]],
    constant uint&     w      [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint w_up  = w * 2u;
    uint h_up  = h * 2u;
    uint hw_up = h_up * w_up;
    uint chw_up = c * hw_up;
    uint ni = gid / chw_up;
    uint pos_in_n = gid % chw_up;
    uint ci = pos_in_n / hw_up;
    uint pos_in_c = pos_in_n % hw_up;
    uint hi_up = pos_in_c / w_up;
    uint wi_up = pos_in_c % w_up;
    uint hi_in = hi_up / 2u;
    uint wi_in = wi_up / 2u;
    uint in_idx = ni * c * h * w + ci * h * w + hi_in * w + wi_in;
    output[gid] = input[in_idx];
}
kernel void rms_norm_channels_h16(
    device half*       x     [[buffer(0)]],  // [C, P]
    device const half* gamma [[buffer(1)]],  // [C]
    constant uint&       c     [[buffer(2)]],
    constant uint&       p     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]        // in [0, P)
) {
    float sum = 0.0f;
    for (uint i = 0; i < c; ++i) {
        float v = float(x[i * p + gid]);
        sum += v * v;
    }
    float inv = 1.0f / max(sqrt(sum), 1e-12f);
    float scale = sqrt(float(c));
    for (uint i = 0; i < c; ++i) {
        x[i * p + gid] = half(float(x[i * p + gid]) * inv * scale * float(gamma[i]));
    }
}
kernel void softmax_rows_scaled_h16(
    device half*    x     [[buffer(0)]],
    constant uint&    width [[buffer(1)]],
    constant float&   scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = gid * width;
    float m = -INFINITY;
    for (uint j = 0; j < width; ++j) {
        m = max(m, float(x[base + j]) * scale);
    }
    float sum = 0.0f;
    for (uint j = 0; j < width; ++j) {
        float e = exp(float(x[base + j]) * scale - m);
        x[base + j] = half(e);
        sum += e;
    }
    float inv = 1.0f / sum;
    for (uint j = 0; j < width; ++j) {
        x[base + j] = half(float(x[base + j]) * inv);
    }
}
kernel void extract_frame_pc_h16(
    device const half* src   [[buffer(0)]],
    device half*       dst   [[buffer(1)]],
    constant uint&       c0    [[buffer(2)]],
    constant uint&       c     [[buffer(3)]],
    constant uint&       t_idx [[buffer(4)]],
    constant uint&       t     [[buffer(5)]],
    constant uint&       p     [[buffer(6)]],
    uint gid [[thread_position_in_grid]]   // over p*c
) {
    uint ci = gid % c;
    uint pi = gid / c;
    dst[gid] = src[((c0 + ci) * t + t_idx) * p + pi];
}
kernel void insert_frame_pc_h16(
    device const half* src   [[buffer(0)]],
    device half*       dst   [[buffer(1)]],
    constant uint&       c     [[buffer(2)]],
    constant uint&       t_idx [[buffer(3)]],
    constant uint&       t     [[buffer(4)]],
    constant uint&       p     [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint ci = gid % c;
    uint pi = gid / c;
    dst[(ci * t + t_idx) * p + pi] = src[gid];
}
kernel void dup_up_3d_h16(
    device const half* src   [[buffer(0)]],   // [in_c, t, h, w]
    device half*       dst   [[buffer(1)]],   // [out_c, t*ft - drop, h*fs, w*fs]
    constant uint&       in_c  [[buffer(2)]],
    constant uint&       out_c [[buffer(3)]],
    constant uint&       t     [[buffer(4)]],
    constant uint&       h     [[buffer(5)]],
    constant uint&       w     [[buffer(6)]],
    constant uint&       ft    [[buffer(7)]],
    constant uint&       fs    [[buffer(8)]],
    constant uint&       drop  [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint ow = w * fs;
    uint oh = h * fs;
    uint ot = t * ft - drop;
    uint wi = gid % ow;
    uint hi = (gid / ow) % oh;
    uint ti = (gid / (ow * oh)) % ot;
    uint oc = gid / (ow * oh * ot);

    uint gt = ti + drop;
    uint tt = gt / ft, it = gt % ft;
    uint hh = hi / fs, ih = hi % fs;
    uint ww = wi / fs, iw = wi % fs;

    uint factor = ft * fs * fs;
    uint repeats = out_c * factor / in_c;
    uint flat = ((oc * ft + it) * fs + ih) * fs + iw;
    uint src_c = flat / repeats;
    dst[gid] = src[(src_c * t + tt) * (h * w) + hh * w + ww];
}
kernel void interleave_time_h16(
    device const half* src [[buffer(0)]],
    device half*       dst [[buffer(1)]],
    constant uint&       c   [[buffer(2)]],
    constant uint&       t   [[buffer(3)]],
    constant uint&       p   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pi = gid % p;
    uint ti = (gid / p) % (2u * t);
    uint ci = gid / (p * 2u * t);
    uint tt = ti / 2u;
    uint k = ti % 2u;
    dst[gid] = src[((k * c + ci) * t + tt) * p + pi];
}
kernel void transpose_outer_h16(
    device const half* src [[buffer(0)]],
    device half*       dst [[buffer(1)]],
    constant uint&       a   [[buffer(2)]],
    constant uint&       b   [[buffer(3)]],
    constant uint&       p   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]   // in [0, a*b*p)
) {
    uint pi = gid % p;
    uint bi = (gid / p) % b;
    uint ai = gid / (p * b);
    dst[(bi * a + ai) * p + pi] = src[gid];
}
kernel void add_shifted_h16(
    device half*       dst     [[buffer(0)]],
    device const half* src     [[buffer(1)]],
    constant uint&       dst_off [[buffer(2)]],
    constant uint&       src_off [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[dst_off + gid] = half(float(dst[dst_off + gid]) + float(src[src_off + gid]));
}

kernel void layer_norm_h16(
    device half*       x       [[buffer(0)]],
    device half*     gamma   [[buffer(1)]],
    device half*     beta    [[buffer(2)]],
    constant uint&     dim     [[buffer(3)]],
    constant float&    eps     [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint   tid [[thread_position_in_threadgroup]],
    uint   tpg [[threads_per_threadgroup]],
    uint   gid [[threadgroup_position_in_grid]]
) {
    uint row = gid;
    uint row_off = row * dim;
    float ls = 0.0;
    float lss = 0.0;
    for (uint i = tid; i < dim; i += tpg) {
        float v = float(x[row_off + i]);
        ls += v;
        lss += v * v;
    }
    scratch[tid] = ls;
    scratch[tpg + tid] = lss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
            scratch[tpg + tid] += scratch[tpg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(dim);
    float var  = scratch[tpg] / float(dim) - mean * mean;
    if (var < 0.0) var = 0.0;
    float inv_std = 1.0 / sqrt(var + eps);
    for (uint i = tid; i < dim; i += tpg) {
        float v = float(x[row_off + i]);
        x[row_off + i] = half((v - mean) * inv_std * float(gamma[i]) + float(beta[i]));
    }
}
kernel void rms_norm_h16(
    device half*       x      [[buffer(0)]],   // [n_rows, dim]
    device const half* weight [[buffer(1)]],   // [dim]
    constant uint&       dim    [[buffer(2)]],
    constant float&      eps    [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    uint base = row * dim;
    float ss = 0.0;
    for (uint i = 0; i < dim; ++i) { float v = float(x[base + i]); ss += v * v; }
    float inv = rsqrt(ss / float(dim) + eps);
    for (uint i = 0; i < dim; ++i) {
        x[base + i] = half(float(x[base + i]) * inv * float(weight[i]));
    }
}
kernel void rope_pairs_h16(
    device half*       x    [[buffer(0)]],  // [n, num_heads, head_dim]
    device const half* cosb [[buffer(1)]],  // [n, head_dim]
    device const half* sinb [[buffer(2)]],
    constant uint&       num_heads [[buffer(3)]],
    constant uint&       head_dim  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]       // in [0, n*num_heads)
) {
    uint t = gid / num_heads;
    uint hd2 = head_dim / 2;
    uint base = gid * head_dim;
    uint cbase = t * head_dim;
    for (uint k = 0; k < hd2; ++k) {
        float a = float(x[base + 2 * k]);
        float b = float(x[base + 2 * k + 1]);
        float c = float(cosb[cbase + 2 * k]);
        float s = float(sinb[cbase + 2 * k]);
        x[base + 2 * k]     = half(a * c - b * s);
        x[base + 2 * k + 1] = half(a * s + b * c);
    }
}
kernel void modulate_rows_h16(
    device half*       x         [[buffer(0)]],  // [n, d]
    device const half* table     [[buffer(1)]],  // [6, d]
    device const half* temb      [[buffer(2)]],  // [n, 6, d] or [6, d]
    constant uint&       d         [[buffer(3)]],
    constant uint&       temb_str  [[buffer(4)]],  // 6*d per token, 0 to broadcast
    constant uint&       scale_i   [[buffer(5)]],
    constant uint&       shift_i   [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint ti = gid / d;
    uint di = gid % d;
    uint base = ti * temb_str;
    float scale = float(table[scale_i * d + di]) + float(temb[base + scale_i * d + di]);
    float shift = float(table[shift_i * d + di]) + float(temb[base + shift_i * d + di]);
    x[gid] = half(float(x[gid]) * (1.0f + scale) + shift);
}
kernel void gate_rows_h16(
    device half*       x         [[buffer(0)]],
    device const half* table     [[buffer(1)]],
    device const half* temb      [[buffer(2)]],
    constant uint&       d         [[buffer(3)]],
    constant uint&       temb_str  [[buffer(4)]],
    constant uint&       gate_i    [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint ti = gid / d;
    uint di = gid % d;
    float g = float(table[gate_i * d + di])
            + float(temb[ti * temb_str + gate_i * d + di]);
    x[gid] = half(float(x[gid]) * g);
}
kernel void final_modulate_h16(
    device half*       x         [[buffer(0)]],  // [n, d]
    device const half* table     [[buffer(1)]],  // [2, d]
    device const half* temb      [[buffer(2)]],  // [n, d] or [d]
    constant uint&       d         [[buffer(3)]],
    constant uint&       temb_str  [[buffer(4)]],  // d per token, 0 to broadcast
    uint gid [[thread_position_in_grid]]
) {
    uint ti = gid / d;
    uint di = gid % d;
    float t = float(temb[ti * temb_str + di]);
    float shift = float(table[di]) + t;
    float scale = float(table[d + di]) + t;
    x[gid] = half(float(x[gid]) * (1.0f + scale) + shift);
}
kernel void gelu_h16(
    device half* x [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float v = float(x[gid]);
    float c = 0.7978845608028654;
    // Clamp tanh-arg to avoid Metal's tanh overflow → NaN at large |v|.
    // See the f32 `gelu` kernel for the rationale.
    float arg = clamp(c * (v + 0.044715 * v * v * v), -10.0f, 10.0f);
    x[gid] = half(0.5 * v * (1.0 + tanh(arg)));
}
kernel void permute_lh_to_hl_h16(
    device half*     src    [[buffer(0)]],
    device half*       dst    [[buffer(1)]],
    constant uint&     n      [[buffer(2)]],
    constant uint&     l      [[buffer(3)]],
    constant uint&     h      [[buffer(4)]],
    constant uint&     d      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // dst layout: [N, H, L, D]
    uint inner = h * d;
    uint hld = h * l * d;
    uint ld = l * d;
    uint ni = gid / hld;
    uint pos = gid % hld;
    uint hi = pos / ld;
    uint pos2 = pos % ld;
    uint li = pos2 / d;
    uint di = pos2 % d;
    // src layout: [N, L, H*D] → src[ni, li, hi*d + di]
    uint src_idx = ni * l * inner + li * inner + hi * d + di;
    dst[gid] = src[src_idx];
}
kernel void permute_hl_to_lh_h16(
    device half*     src    [[buffer(0)]],
    device half*       dst    [[buffer(1)]],
    constant uint&     n      [[buffer(2)]],
    constant uint&     l      [[buffer(3)]],
    constant uint&     h      [[buffer(4)]],
    constant uint&     d      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // dst layout: [N, L, H*D]
    uint inner = h * d;
    uint l_inner = l * inner;
    uint ni = gid / l_inner;
    uint pos = gid % l_inner;
    uint li = pos / inner;
    uint pos2 = pos % inner;
    uint hi = pos2 / d;
    uint di = pos2 % d;
    // src layout: [N, H, L, D] → src[ni, hi, li, di]
    uint src_idx = ni * h * l * d + hi * l * d + li * d + di;
    dst[gid] = src[src_idx];
}

// ---------- Native bf16 GEMM (simdgroup MMA) ------------------------------
//
// C[M,N] = A[M,K] * B[N,K]^T, bfloat storage throughout, float accumulation.
// This is the `Linear` shape: PyTorch stores weights [out, in], so the second
// operand is used transposed.
//
// Why this exists rather than MPS: MPSMatrixMultiplication has no bf16 mode,
// so everything went through a cast to fp32 (safe, but capped at ~2.2 TFLOP/s
// on M1 Max) or to IEEE fp16 (faster, but it loses precision in accumulation
// and clips its output at 65504). bfloat has fp32's exponent range, so an MMA
// kernel with bfloat inputs and a float accumulator has neither problem.
//
// 128 threads = 4 simdgroups in a 2x2 arrangement over a 64x64 output tile;
// each simdgroup owns a 32x32 quadrant as 4x4 float8x8 accumulators (32
// floats per thread, which fits without spilling).

#define GM 64u   // output rows per threadgroup
#define GN 64u   // output cols per threadgroup
#define GK 32u   // K-slice staged per iteration
#define GT 128u  // threads per threadgroup

kernel void sgemm_bf16_btrans(
    const device bfloat* A          [[buffer(0)]],
    const device bfloat* B          [[buffer(1)]],
    device bfloat*       C          [[buffer(2)]],
    constant uint&       M          [[buffer(3)]],
    constant uint&       N          [[buffer(4)]],
    constant uint&       K          [[buffer(5)]],
    threadgroup float*   sh         [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    // The staging area is used for two different things at two different
    // times: bfloat A/B tiles during the K loop (8 KB), then a float C tile
    // for the bounds-checked write-out (16 KB). They alias; the barrier
    // before the switch is what makes that safe.
    threadgroup bfloat* As = (threadgroup bfloat*)sh;
    threadgroup bfloat* Bs = As + GM * GK;   // staged TRANSPOSED, as [GK][GN]

    const uint m0 = tgid.y * GM;
    const uint n0 = tgid.x * GN;
    const uint sm = (sgid / 2u) * 32u;
    const uint sn = (sgid % 2u) * 32u;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += GK) {
        // Out-of-range elements are staged as zero rather than skipped, so a
        // ragged edge tile contributes nothing to the dot product and needs no
        // special case in the MMA loop below.
        for (uint e = tid; e < GM * GK; e += GT) {
            uint r = e / GK, c = e % GK;
            uint gr = m0 + r, gc = k0 + c;
            As[e] = (gr < M && gc < K) ? A[gr * K + gc] : bfloat(0.0f);
        }
        // Transpose B on the way INTO threadgroup memory, not on the way out.
        // The MMA wants B as [k][n]; staging it that way costs one scalar
        // scatter per element per tile, whereas a transposing `simdgroup_load`
        // pays for the transpose 4 times per 8-deep K step, every step.
        for (uint e = tid; e < GN * GK; e += GT) {
            uint r = e / GK, c = e % GK;
            uint gr = n0 + r, gc = k0 + c;
            Bs[c * GN + r] = (gr < N && gc < K) ? B[gr * K + gc] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < GK; kk += 8u) {
            simdgroup_bfloat8x8 af[4], bf[4];
            for (uint i = 0; i < 4; ++i)
                simdgroup_load(af[i], As + (sm + i * 8u) * GK + kk, GK);
            for (uint j = 0; j < 4; ++j)
                simdgroup_load(bf[j], Bs + kk * GN + (sn + j * 8u), GN);
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // simdgroup_store has no bounds checking, so stage the tile and let the
    // threads write it out with explicit range checks. M is routinely not a
    // multiple of 64 (Wan runs 5070 tokens), so this edge is the common case,
    // not a rare one.
    // Write out half the tile at a time. The accumulators are float and
    // `simdgroup_store` will not narrow, so C must stage as float — but a full
    // 64x64 float tile is 16 KB, which would double this kernel's threadgroup
    // allocation and halve how many threadgroups a core can keep resident. A
    // 32x64 half-tile is 8 KB, the same as the A/B tiles it aliases.
    //
    // The split falls out for free: simdgroups 0,1 own rows 0-31 and 2,3 own
    // rows 32-63, so each pass is exactly one row-half of the simdgroups.
    threadgroup float* Cs = sh;
    // `half` is a Metal type name, hence `hpart`.
    for (uint hpart = 0; hpart < 2u; ++hpart) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sm / 32u == hpart) {
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j)
                    simdgroup_store(acc[i][j], Cs + (i * 8u) * GN + (sn + j * 8u), GN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 32u * GN; e += GT) {
            uint r = e / GN, c = e % GN;
            uint gr = m0 + hpart * 32u + r, gc = n0 + c;
            if (gr < M && gc < N) C[gr * N + gc] = bfloat(Cs[e]);
        }
    }
}
"#;

struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    silu_pipe: ComputePipelineState,
    quick_gelu_pipe: ComputePipelineState,
    gelu_pipe: ComputePipelineState,
    groupnorm_pipe: ComputePipelineState,
    layernorm_pipe: ComputePipelineState,
    softmax_pipe: ComputePipelineState,
    eadd_pipe: ComputePipelineState,
    bcast_add_pipe: ComputePipelineState,
    flash_attn_pipe: ComputePipelineState,
    flash_attn_causal_pipe: ComputePipelineState,
    rms_norm_pipe: ComputePipelineState,
    // fp16 GPU-residence kernels.
    silu_f16_pipe: ComputePipelineState,
    silu_h16_pipe: ComputePipelineState,
    gelu_f16_pipe: ComputePipelineState,
    gelu_f16_pipe_h: ComputePipelineState,
    quick_gelu_f16_pipe: ComputePipelineState,
    eadd_f16_pipe: ComputePipelineState,
    eadd_f16_h_pipe: ComputePipelineState,
    emul_f16_pipe: ComputePipelineState,
    emul_h16_pipe: ComputePipelineState,
    sigmoid_gate_f16_pipe: ComputePipelineState,
    sigmoid_gate_h16_pipe: ComputePipelineState,
    rms_norm_f16_pipe: ComputePipelineState,
    rms_norm_f16_pipe_h: ComputePipelineState,
    rms_norm_wf32_f16_pipe: ComputePipelineState,
    rms_norm_wf32_h16_pipe: ComputePipelineState,
    rope_apply_f16_pipe: ComputePipelineState,
    gqa_repeat_f16_pipe: ComputePipelineState,
    eadd_scaled_f16_pipe: ComputePipelineState,
    layernorm_f16_pipe: ComputePipelineState,
    layernorm_f16_pipe_h: ComputePipelineState,
    layernorm_f16_out_pipe: ComputePipelineState,
    groupnorm_f16_pipe: ComputePipelineState,
    groupnorm_silu_f16_out_pipe: ComputePipelineState,
    groupnorm_silu_f16_inplace_pipe: ComputePipelineState,
    cast_f32_to_f16_pipe: ComputePipelineState,
    cast_f16_to_f32_pipe: ComputePipelineState,
    cast_bf16_to_half_pipe: ComputePipelineState,
    cast_half_to_f32_pipe: ComputePipelineState,
    copy_f16_pipe: ComputePipelineState,
    cast_f32_to_half_pipe: ComputePipelineState,
    cast_half_to_bf16_pipe: ComputePipelineState,
    bias_add_f16_pipe: ComputePipelineState,
    bias_add_strided_h_pipe: ComputePipelineState,
    im2col_f16_pipe: ComputePipelineState,
    bias_add_nc_pipe: ComputePipelineState,
    bias_add_nc_h_pipe: ComputePipelineState,
    upsample_2x_pipe: ComputePipelineState,
    pad_end_1_pipe: ComputePipelineState,
    simdgroup_layout_probe_pipe: ComputePipelineState,
    avg_down_3d_pipe: ComputePipelineState,
    upsample_2x_h_pipe: ComputePipelineState,
    pad_end_1_h_pipe: ComputePipelineState,
    avg_down_3d_h_pipe: ComputePipelineState,
    cat_channels_pipe: ComputePipelineState,
    flash_attn_f16_pipe: ComputePipelineState,
    modulate_rows_f16_pipe: ComputePipelineState,
    modulate_rows_f16_pipe_h: ComputePipelineState,
    rms_norm_channels_f16_pipe: ComputePipelineState,
    rms_norm_channels_h16_pipe: ComputePipelineState,
    transpose_outer_f16_pipe: ComputePipelineState,
    transpose_outer_h16_pipe: ComputePipelineState,
    dup_up_3d_f16_pipe: ComputePipelineState,
    dup_up_3d_h16_pipe: ComputePipelineState,
    softmax_rows_scaled_f16_pipe: ComputePipelineState,
    softmax_rows_scaled_h16_pipe: ComputePipelineState,
    extract_frame_pc_f16_pipe: ComputePipelineState,
    extract_frame_pc_h16_pipe: ComputePipelineState,
    insert_frame_pc_f16_pipe: ComputePipelineState,
    insert_frame_pc_h16_pipe: ComputePipelineState,
    interleave_time_f16_pipe: ComputePipelineState,
    interleave_time_h16_pipe: ComputePipelineState,
    add_shifted_f16_pipe: ComputePipelineState,
    add_shifted_h16_pipe: ComputePipelineState,
    gate_rows_f16_pipe: ComputePipelineState,
    gate_rows_f16_pipe_h: ComputePipelineState,
    final_modulate_f16_pipe: ComputePipelineState,
    final_modulate_f16_pipe_h: ComputePipelineState,
    /// Tiled flash attention, one pipeline per supported head dim (D is a
    /// template parameter in MSL so the fragment arrays stay in registers).
    sgemm_bf16_btrans_pipe: ComputePipelineState,
    flash_attn_tiled_pipes: [(usize, ComputePipelineState); 5],
    /// Head-dim-split variant, used in preference to the above where it
    /// exists: at large D the plain kernel spills its simdgroup fragments.
    flash_attn_split_pipes: [(usize, ComputePipelineState); 2],
    /// Register-resident variant: S and P never reach threadgroup memory.
    flash_attn_reg_pipes: [(usize, ComputePipelineState); 2],
    permute_lh_to_hl_pipe: ComputePipelineState,
    permute_lh_to_hl_pipe_h: ComputePipelineState,
    permute_lh_to_hl_strided_pipe: ComputePipelineState,
    permute_hl_to_lh_pipe: ComputePipelineState,
    permute_hl_to_lh_pipe_h: ComputePipelineState,
    geglu_split_pipe: ComputePipelineState,
    nchw_to_nhwc_pipe: ComputePipelineState,
    nhwc_to_nchw_pipe: ComputePipelineState,
    temporal_reshape_to_pipe: ComputePipelineState,
    temporal_reshape_from_pipe: ComputePipelineState,
    pos_embed_add_pipe: ComputePipelineState,
    // Flux MMDiT kernels.
    gelu_erf_f16_pipe: ComputePipelineState,
    rope_pairs_f16_pipe: ComputePipelineState,
    rope_pairs_f16_pipe_h: ComputePipelineState,
    modulate_f16_pipe: ComputePipelineState,
    gate_mul_f16_pipe: ComputePipelineState,
    slice_cols_f16_pipe: ComputePipelineState,
}

unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

static METAL: OnceLock<MetalBackend> = OnceLock::new();

fn get_metal() -> &'static MetalBackend {
    METAL.get_or_init(|| {
        let device = Device::system_default()
            .expect("no system default Metal device");
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        let library: Library = device.new_library_with_source(KERNELS_MSL, &opts)
            .expect("compile MSL kernels");

        let make_pipe = |name: &str| -> ComputePipelineState {
            let func = library.get_function(name, None)
                .unwrap_or_else(|_| panic!("kernel function {name} not found"));
            device.new_compute_pipeline_state_with_function(&func)
                .unwrap_or_else(|_| panic!("compute pipeline for {name}"))
        };
        MetalBackend {
            silu_pipe: make_pipe("silu"),
            quick_gelu_pipe: make_pipe("quick_gelu"),
            gelu_pipe: make_pipe("gelu"),
            groupnorm_pipe: make_pipe("groupnorm"),
            layernorm_pipe: make_pipe("layer_norm"),
            softmax_pipe: make_pipe("softmax_rowwise"),
            eadd_pipe: make_pipe("elementwise_add"),
            bcast_add_pipe: make_pipe("broadcast_add_channel"),
            flash_attn_pipe: make_pipe("flash_attention"),
            flash_attn_causal_pipe: make_pipe("flash_attention_causal"),
            rms_norm_pipe: make_pipe("rms_norm"),
            silu_f16_pipe: make_pipe("silu_f16"),
            silu_h16_pipe: make_pipe("silu_h16"),
            gelu_f16_pipe: make_pipe("gelu_f16"),
            gelu_f16_pipe_h: make_pipe("gelu_h16"),
            quick_gelu_f16_pipe: make_pipe("quick_gelu_f16"),
            eadd_f16_pipe: make_pipe("elementwise_add_f16"),
            eadd_f16_h_pipe: make_pipe("elementwise_add_h16"),
            emul_f16_pipe: make_pipe("elementwise_mul_f16"),
            emul_h16_pipe: make_pipe("elementwise_mul_h16"),
            sigmoid_gate_f16_pipe: make_pipe("sigmoid_gate_f16"),
            sigmoid_gate_h16_pipe: make_pipe("sigmoid_gate_h16"),
            rms_norm_f16_pipe: make_pipe("rms_norm_f16"),
            rms_norm_f16_pipe_h: make_pipe("rms_norm_h16"),
            rms_norm_wf32_f16_pipe: make_pipe("rms_norm_wf32_f16"),
            rms_norm_wf32_h16_pipe: make_pipe("rms_norm_wf32_h16"),
            rope_apply_f16_pipe: make_pipe("rope_apply_f16"),
            gqa_repeat_f16_pipe: make_pipe("gqa_repeat_f16"),
            eadd_scaled_f16_pipe: make_pipe("elementwise_add_scaled_f16"),
            layernorm_f16_pipe: make_pipe("layer_norm_f16"),
            layernorm_f16_pipe_h: make_pipe("layer_norm_h16"),
            layernorm_f16_out_pipe: make_pipe("layer_norm_f16_out"),
            groupnorm_f16_pipe: make_pipe("groupnorm_f16"),
            groupnorm_silu_f16_out_pipe: make_pipe("groupnorm_silu_f16_out"),
            groupnorm_silu_f16_inplace_pipe: make_pipe("groupnorm_silu_f16_inplace"),
            cast_f32_to_f16_pipe: make_pipe("cast_f32_to_f16"),
            cast_f16_to_f32_pipe: make_pipe("cast_f16_to_f32"),
            cast_bf16_to_half_pipe: make_pipe("cast_bf16_to_half"),
            cast_half_to_f32_pipe: make_pipe("cast_half_to_f32"),
            copy_f16_pipe: make_pipe("copy_f16"),
            cast_f32_to_half_pipe: make_pipe("cast_f32_to_half"),
            cast_half_to_bf16_pipe: make_pipe("cast_half_to_bf16"),
            bias_add_f16_pipe: make_pipe("bias_add_strided_f16"),
            bias_add_strided_h_pipe: make_pipe("bias_add_strided_h16"),
            im2col_f16_pipe: make_pipe("im2col_f16"),
            bias_add_nc_pipe: make_pipe("bias_add_nc_to_nchw_f16"),
            bias_add_nc_h_pipe: make_pipe("bias_add_nc_to_nchw_h16"),
            upsample_2x_pipe: make_pipe("nearest_upsample_2x_f16"),
            pad_end_1_pipe: make_pipe("pad_end_1_f16"),
            simdgroup_layout_probe_pipe: make_pipe("simdgroup_layout_probe"),
            avg_down_3d_pipe: make_pipe("avg_down_3d_f16"),
            upsample_2x_h_pipe: make_pipe("nearest_upsample_2x_h16"),
            pad_end_1_h_pipe: make_pipe("pad_end_1_h16"),
            avg_down_3d_h_pipe: make_pipe("avg_down_3d_h16"),
            cat_channels_pipe: make_pipe("cat_channels_f16"),
            flash_attn_f16_pipe: make_pipe("flash_attention_f16"),
            modulate_rows_f16_pipe: make_pipe("modulate_rows_f16"),
            modulate_rows_f16_pipe_h: make_pipe("modulate_rows_h16"),
            rms_norm_channels_f16_pipe: make_pipe("rms_norm_channels_f16"),
            rms_norm_channels_h16_pipe: make_pipe("rms_norm_channels_h16"),
            transpose_outer_f16_pipe: make_pipe("transpose_outer_f16"),
            transpose_outer_h16_pipe: make_pipe("transpose_outer_h16"),
            dup_up_3d_f16_pipe: make_pipe("dup_up_3d_f16"),
            dup_up_3d_h16_pipe: make_pipe("dup_up_3d_h16"),
            softmax_rows_scaled_f16_pipe: make_pipe("softmax_rows_scaled_f16"),
            softmax_rows_scaled_h16_pipe: make_pipe("softmax_rows_scaled_h16"),
            extract_frame_pc_f16_pipe: make_pipe("extract_frame_pc_f16"),
            extract_frame_pc_h16_pipe: make_pipe("extract_frame_pc_h16"),
            insert_frame_pc_f16_pipe: make_pipe("insert_frame_pc_f16"),
            insert_frame_pc_h16_pipe: make_pipe("insert_frame_pc_h16"),
            interleave_time_f16_pipe: make_pipe("interleave_time_f16"),
            interleave_time_h16_pipe: make_pipe("interleave_time_h16"),
            add_shifted_f16_pipe: make_pipe("add_shifted_f16"),
            add_shifted_h16_pipe: make_pipe("add_shifted_h16"),
            gate_rows_f16_pipe: make_pipe("gate_rows_f16"),
            gate_rows_f16_pipe_h: make_pipe("gate_rows_h16"),
            final_modulate_f16_pipe: make_pipe("final_modulate_f16"),
            final_modulate_f16_pipe_h: make_pipe("final_modulate_h16"),
            sgemm_bf16_btrans_pipe: make_pipe("sgemm_bf16_btrans"),
            flash_attn_tiled_pipes: [
                (16,  make_pipe("flash_attention_tiled_d16")),
                (32,  make_pipe("flash_attention_tiled_d32")),
                (64,  make_pipe("flash_attention_tiled_d64")),
                (96,  make_pipe("flash_attention_tiled_d96")),
                (128, make_pipe("flash_attention_tiled_d128")),
            ],
            flash_attn_split_pipes: [
                (96,  make_pipe("flash_attention_split_d96")),
                (128, make_pipe("flash_attention_split_d128")),
            ],
            flash_attn_reg_pipes: [
                (96,  make_pipe("flash_attention_reg_d96")),
                (128, make_pipe("flash_attention_reg_d128")),
            ],
            permute_lh_to_hl_pipe: make_pipe("permute_lh_to_hl_f16"),
            permute_lh_to_hl_pipe_h: make_pipe("permute_lh_to_hl_h16"),
            permute_lh_to_hl_strided_pipe: make_pipe("permute_lh_to_hl_f16_strided"),
            permute_hl_to_lh_pipe: make_pipe("permute_hl_to_lh_f16"),
            permute_hl_to_lh_pipe_h: make_pipe("permute_hl_to_lh_h16"),
            geglu_split_pipe: make_pipe("geglu_split_f16"),
            nchw_to_nhwc_pipe: make_pipe("nchw_to_nhwc_f16"),
            nhwc_to_nchw_pipe: make_pipe("nhwc_to_nchw_f16"),
            temporal_reshape_to_pipe: make_pipe("temporal_reshape_to_f16"),
            temporal_reshape_from_pipe: make_pipe("temporal_reshape_from_f16"),
            pos_embed_add_pipe: make_pipe("pos_embed_add_f16"),
            gelu_erf_f16_pipe: make_pipe("gelu_erf_f16"),
            rope_pairs_f16_pipe: make_pipe("rope_pairs_f16"),
            rope_pairs_f16_pipe_h: make_pipe("rope_pairs_h16"),
            modulate_f16_pipe: make_pipe("modulate_f16"),
            gate_mul_f16_pipe: make_pipe("gate_mul_f16"),
            slice_cols_f16_pipe: make_pipe("slice_cols_f16"),
            device,
            queue,
        }
    })
}

// ---------- Buffer pool ----------------------------------------------------

thread_local! {
    /// Pooled buffers, each tagged with the commit sequence in effect when it
    /// was released. A buffer must not be handed out again until the GPU work
    /// that was still reading it has finished.
    static BUF_POOL: RefCell<HashMap<usize, Vec<(Option<metal::CommandBuffer>, Buffer)>>> =
        RefCell::new(HashMap::new());
    /// The most recently committed command buffer, so a released scratch
    /// buffer can be held until the work that touched it has finished.
    static LAST_CMD: RefCell<Option<metal::CommandBuffer>> = const { RefCell::new(None) };
}

/// Record a committed command buffer as the one a subsequent `release_buffer`
/// should wait on.
///
/// Without this the pool recycled scratch the moment it was *encoded*, so a
/// later acquire could zero-fill a buffer that an earlier, still-running matmul
/// was reading — and that reader then produced a row of zeros. It only bites
/// when command buffers actually overlap, which is why it looked like a rare
/// per-process flake for most of this project and then reproduced 8 times in 9
/// under GPU contention.
fn note_commit(cmd: &metal::CommandBufferRef) {
    LAST_CMD.with(|c| *c.borrow_mut() = Some(cmd.to_owned()));
}

/// True once the GPU has finished with this command buffer (or there is none).
fn cmd_done(c: &Option<metal::CommandBuffer>) -> bool {
    match c {
        None => true,
        Some(cb) => matches!(
            cb.status(),
            metal::MTLCommandBufferStatus::Completed | metal::MTLCommandBufferStatus::Error
        ),
    }
}

/// Acquire a `[bytes]`-sized MTLBuffer (Shared storage). Pulled from a
/// thread-local pool when one of the same size is available, else allocated.
///
/// Recycled buffers are zeroed via a blit `fill_buffer` before being handed
/// out. Pool buffers can hold leftover Inf/NaN from prior ops; if a
/// downstream consumer doesn't fully overwrite the buffer (MPS matmul with
/// beta=0 reading C, partial-coverage dispatches, etc.), the stale data
/// poisons the next op via `0 * Inf = NaN`. The blit is queued on the
/// current command buffer so Metal hazard tracking serializes it after any
/// prior in-flight op that might still be using the buffer.
fn acquire_buffer(bytes: usize) -> Buffer {
    let recycled = BUF_POOL.with_borrow_mut(|pool| {
        if let Some(bufs) = pool.get_mut(&bytes) {
            // Only take a buffer whose last user has finished. Anything still
            // in flight stays put rather than being handed to a second reader;
            // the fresh allocation below is the safe fallback.
            if let Some(i) = bufs.iter().position(|(cmd, _)| cmd_done(cmd)) {
                return Some(bufs.swap_remove(i).1);
            }
        }
        None
    });
    if let Some(buf) = recycled {
        with_cmd(|cmd| {
            let blit = cmd.new_blit_command_encoder();
            blit.fill_buffer(&buf, metal::NSRange::new(0, bytes as u64), 0);
            blit.end_encoding();
        });
        return buf;
    }
    // NOTE: fresh allocations are deliberately NOT zeroed here. Zeroing them
    // as well was tried against the intermittent non-finite Wan forward and
    // made it strictly worse (9/9 runs non-finite, against 3/6 without), so
    // whatever that failure is, reading uninitialised buffer memory is not it.
    let backend = get_metal();
    let buf = backend.device.new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared);
    PEAK_ALLOCATED.fetch_max(backend.device.current_allocated_size(), std::sync::atomic::Ordering::Relaxed);
    buf
}

/// Whether `buf` straddles a 4 GiB boundary in the GPU's address space.
fn crosses_4gib(buf: &Buffer) -> bool {
    let start = buf.gpu_address();
    let end = start + buf.length().max(1) - 1;
    start >> 32 != end >> 32
}

/// `acquire_buffer` for an MPS matmul operand: never one that straddles a
/// 4 GiB boundary of GPU address space.
///
/// MPSMatrixMultiplication computes garbage for the whole 256-row tile in
/// which an operand crosses such a boundary — measured on the Krea 2 VAE's
/// attention, where the scores buffer crossed at row 2588 and rows 2368-2623
/// came back NaN, identically in every query block and every repeat. Where a
/// buffer lands depends on everything allocated before it, which is why this
/// looked like a race: the same forward failed in one process and not the
/// next, and anything that changed pool reuse (CPU waits around each matmul)
/// changed how often it hit. Large scratch is the exposed case — a 192 MB
/// operand crosses about one time in twenty.
///
/// A crossing buffer is held while the next is fetched, so the retry lands
/// elsewhere, then pooled again for non-MPS use.
fn acquire_mps_buffer(bytes: usize) -> Buffer {
    if bytes as u64 >= 1 << 32 {
        // Cannot fit between two boundaries; the caller must split the matmul.
        panic!("acquire_mps_buffer: a {bytes}-byte MPS operand necessarily crosses a 4 GiB boundary");
    }
    let mut held = Vec::new();
    let buf = loop {
        let b = acquire_buffer(bytes);
        if !crosses_4gib(&b) {
            break b;
        }
        held.push(b);
        assert!(held.len() < 64, "acquire_mps_buffer: no {bytes}-byte buffer clear of a 4 GiB boundary");
    };
    for b in held {
        release_buffer(b);
    }
    buf
}

/// Diagnostic, `KLEARU_MPS_ADDR=1`: each MPS operand that ends within 64 MB
/// of a 4 GiB boundary of GPU address space, or starts within 64 MB after one.
fn mps_addr_report(ops: &[(&str, &Buffer)]) {
    // `KLEARU_MPS_ADDR=all` prints every operand.
    thread_local! { static ON: Option<bool> = std::env::var("KLEARU_MPS_ADDR").ok().map(|v| v == "all"); }
    let Some(all) = ON.with(|v| *v) else { return };
    const MARGIN: u64 = 64 << 20;
    for (what, b) in ops {
        let (start, end) = (b.gpu_address(), b.gpu_address() + b.length());
        let to_next = ((end >> 32) + 1) * (1 << 32) - end;
        let from_prev = start - ((start >> 32) << 32);
        if all || crosses_4gib(b) || to_next < MARGIN || from_prev < MARGIN {
            println!(
                "[mps addr] {what} {start:#x}..{end:#x} ({} MB): ends {:.2} MB before a boundary, starts {:.2} MB after one{}",
                b.length() >> 20, to_next as f64 / 1048576.0, from_prev as f64 / 1048576.0,
                if crosses_4gib(b) { " — CROSSES" } else { "" }
            );
        }
    }
}

/// High-water mark of the device's allocated size, sampled at each fresh pool
/// allocation. Reported by `gpu_memory_report`: the size at a quiet point says
/// nothing about how far in-flight work pushed it.
static PEAK_ALLOCATED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// ---------- Async dispatch + lazy flush -----------------------------------
//
// Each `*_gpu` / `*_buf` dispatcher commits its command buffer to the
// shared queue WITHOUT calling `wait_until_completed`. Metal serializes
// kernels on the same queue automatically, and Metal's hazard tracker
// inserts the right inter-cmd dependencies for shared buffers — so
// successive kernels see correct ordering without an explicit CPU wait.
//
// The wait is moved to CPU-access boundaries: `download_to_f32`, the
// `upload_*` paths that CPU-write into a freshly-acquired buffer, and
// `convert_or_get_cached_f16` on cache miss. We track the most recent
// committed cmd and `flush()` waits on it (which transitively guarantees
// every prior cmd on the queue is also complete).
//
// Per SDXL inference: ~30K kernel dispatches → ~30K → ~50 waits. At ~1ms
// per wait, that's ~30s saved per inference on the GPU residence path.

thread_local! {
    /// Counter of cmds committed without an explicit wait. When > 0,
    /// `flush()` submits an empty barrier cmd and waits on it — Metal
    /// serializes ops on the queue, so a single trailing wait drains
    /// all prior in-flight work.
    /// Every command buffer committed since the last `flush()`. `flush` waits
    /// on each one individually: command buffers on a single queue are
    /// *scheduled* in order but may *execute* concurrently, so an empty barrier
    /// buffer committed last can complete while a heavy matmul is still
    /// writing. That produced output whose trailing rows were still zero.
    static PENDING_CMDS: std::cell::RefCell<Vec<metal::CommandBuffer>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// Shared command buffer for the current "batch". When set, every
    /// `*_f16_gpu` kernel dispatch and MPS sgemm appends a new encoder to
    /// this buffer instead of creating its own — saving the
    /// `new_command_buffer + commit` overhead (~50-200μs CPU each) per
    /// kernel call. With ~600+ kernels per SDXL UNet step, this collapses
    /// to a single command-buffer creation+commit per `flush()`.
    static CURRENT_CMD: std::cell::RefCell<Option<metal::CommandBuffer>>
        = const { std::cell::RefCell::new(None) };

    /// Encoders appended to the open batch since it was created. A batch that
    /// grows without bound eventually holds the GPU long enough to be killed
    /// by the watchdog, so `with_cmd` rolls it over at `MAX_BATCH_ENCODERS`.
    static BATCH_ENCODERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// When true, the MPS sgemm wrapper uses f32 scratch + MPS f32 sgemm
    /// (the slower-but-safer path). Default is false (fast f16 scratch
    /// path). Set true around code paths whose intermediate matmul
    /// outputs can exceed fp16 max (≈65504) — notably SDXL VAE decoder,
    /// where some activations grow to ~23K and MPS-f16-output overflow
    /// produces Inf → NaN in the final image.
    static USE_F32_SGEMM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FORCE_FLASH_ATTN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Route `sgemm_f16_a_btrans_buf` through the native bf16 MMA kernel
    /// instead of MPS. Opt-in per code path while it earns trust.
    static USE_BF16_GEMM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Fence every thread's MPS matmuls (older spelling of
    /// `KLEARU_MPS_FENCE=on`).
    static MM_FLUSH: bool = std::env::var("KLEARU_MM_FLUSH").is_ok();

    /// Per-thread opt-in to fencing MPS matmuls (see `set_mps_barrier`).
    static MPS_BARRIER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Process-wide override of the MPS fence for A/B runs:
    /// `KLEARU_MPS_FENCE=off` or `on`, whatever the thread asked for.
    static MPS_FENCE_ENV: Option<bool> = match std::env::var("KLEARU_MPS_FENCE").as_deref() {
        Ok("off") => Some(false),
        Ok("on") => Some(true),
        _ => None,
    };

    /// Route the MPS matmul scratch through IEEE f16 storage instead of
    /// bfloat16. Set while a half-storage model (currently the VAE decoder,
    /// and the DiT when enabled) is running.
    static SGEMM_HALF_IO: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Run MPSGraph convolutions in f32 rather than f16.
    ///
    /// The input cast is not what costs precision — bf16's 8 mantissa bits fit
    /// inside f16's 10 — but a 3x3 convolution over up to 1024 channels sums
    /// ~9216 products, and doing that in f16 loses far more than the storage
    /// format does. The VAE decoder is the path where it shows.
    static USE_F32_CONV: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Tell the MPS matmul wrapper that its bf16 arguments are actually IEEE f16.
pub fn set_sgemm_half_io(v: bool) {
    SGEMM_HALF_IO.with(|c| c.set(v));
}

/// Run MPSGraph convolutions in f32. Costs ~2x conv memory and time.
pub fn set_use_f32_conv(v: bool) {
    USE_F32_CONV.with(|c| c.set(v));
}

/// Enable the native bf16 simdgroup GEMM for B-transposed matmuls.
pub fn set_use_bf16_gemm(v: bool) {
    USE_BF16_GEMM.with(|c| c.set(v));
}

/// Switch the MPS sgemm wrapper to the f32-precision path. Call before
/// VAE decode (or any other matmul-heavy code with large intermediate
/// magnitudes) and reset to false afterward.
pub fn set_use_f32_sgemm(v: bool) {
    USE_F32_SGEMM.with(|c| c.set(v));
}

/// Fence every MPS matmul on both sides with a CPU wait: before it, for the
/// bf16 -> f16/f32 scratch casts and everything else in flight, and after it,
/// before anything reads its output. Returns the previous setting.
///
/// The failure behind Krea 2's all-NaN forwards and the VAE's black bands
/// turned out to be address placement, not ordering: MPSMatrixMultiplication
/// garbles the tile in which an operand crosses a 4 GiB boundary of GPU
/// address space, fenced or not, and `acquire_mps_buffer` now keeps every
/// operand clear of one (`gemm_bench::mps_matmul_across_4gib`). With that,
/// Krea's transformer runs clean unfenced (4 of 4 processes).
///
/// The fence stays on for Krea because one unfenced failure remains:
/// `gemm_bench::mps_unfenced_chain_locate` still garbles a 64-480-row block,
/// fixed per process, in a dependent chain at Krea's FFN shape (6144 x
/// 16384), with no operand near a 4 GiB boundary — but only unfenced. Fenced,
/// it has never failed (every reference run, and ~20 Krea processes), and it
/// costs Krea nothing measurable. SD 1.5, SDXL, Flux, HiDream-O1 and Wan 2.2
/// shapes stay clean unfenced in the same chain, and the fence would cost
/// them up to 2x per matmul (`mps_fence_cost`), so it is opt-in per thread.
///
/// Ruled out along the way, on the Krea transformer: GPU-side ordering (an
/// MTLEvent chain serialising every command buffer), MPS object lifetime,
/// MPS-internal scratch (it allocates none), and working-set overcommit.
///
/// `KLEARU_MPS_FENCE=on|off` overrides this for every thread.
pub fn set_mps_barrier(v: bool) -> bool {
    MPS_BARRIER.with(|c| c.replace(v))
}

/// Whether this thread's MPS matmuls are fenced.
fn mps_fenced() -> bool {
    MPS_FENCE_ENV.with(|v| *v)
        .unwrap_or_else(|| MM_FLUSH.with(|v| *v) || MPS_BARRIER.with(|c| c.get()))
}

/// Run one MPS matmul — its input casts, the matmul and its output cast — with
/// a CPU wait on each side of the matmul when this thread asks for one (see
/// `set_mps_barrier`).
fn fence_mps(casts: impl FnOnce(), matmul: impl FnOnce(), out_cast: impl FnOnce()) {
    let fenced = mps_fenced();
    // `flush` commits the caller's open batch; reopen it, so the fence does
    // not also change how the rest of the caller's work is batched.
    let fence = || {
        if fenced {
            let in_batch = CURRENT_CMD.with(|c| c.borrow().is_some());
            flush();
            if in_batch {
                begin_batch();
            }
        }
    };
    casts();
    fence();
    matmul();
    fence();
    out_cast();
}

/// Force all SDPA to use flash attention (bypass MPSGraph). MPSGraph's
/// SDPA calls `flush()` internally to create its own command buffer,
/// which disrupts the caller's open batch and causes NaN in the
/// per-block video forward path. Flash attention dispatches via
/// `with_cmd` and stays within the caller's batch.
pub fn set_force_flash_attn(v: bool) {
    FORCE_FLASH_ATTN.with(|c| c.set(v));
}

pub fn force_flash_attn() -> bool {
    FORCE_FLASH_ATTN.with(|c| c.get())
}

/// Open a "batch" — subsequent kernel dispatches and MPS sgemms append to
/// a single shared command buffer, committed at the next `flush()`.
/// Idempotent if a batch is already open.
/// Counter for labeling command buffers (diagnostic).
pub static CMD_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Encoders per batched command buffer before it is committed and a fresh one
/// opened. Purely a duration guard — see `MMA_MAX_ROWS` for why.
const MAX_BATCH_ENCODERS: usize = 96;

pub fn begin_batch() {
    CURRENT_CMD.with(|c| {
        let mut bw = c.borrow_mut();
        if bw.is_none() {
            let backend = get_metal();
            let cmd = backend.queue.new_command_buffer().to_owned();
            let id = CMD_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            cmd.set_label(&format!("klearu-batch-{id}"));
            *bw = Some(cmd);
            BATCH_ENCODERS.with(|b| b.set(0));
        }
    });
}

/// Run `f` with a `&CommandBufferRef`. If a batch is open, the shared
/// buffer is used (caller does NOT commit). If not, a fresh buffer is
/// created and committed automatically. This is the single entry point
/// for every kernel dispatcher in this module.
#[inline]
fn with_cmd<F>(f: F)
where
    F: FnOnce(&metal::CommandBufferRef),
{
    CURRENT_CMD.with(|c| {
        let bw = c.borrow();
        if let Some(cmd) = bw.as_ref() {
            // Batch is open — share. Caller must NOT commit.
            f(cmd);
            drop(bw);
            let n = BATCH_ENCODERS.with(|b| { let n = b.get() + 1; b.set(n); n });
            if n >= MAX_BATCH_ENCODERS {
                // Roll over rather than let one command buffer grow long enough
                // for the watchdog to kill it. The queue is serial, so the work
                // still runs in encode order; only the commit boundary moves.
                if let Some(cmd) = c.borrow_mut().take() {
                    note_commit(&cmd);
                    cmd.commit();
                    PENDING_CMDS.with(|p| p.borrow_mut().push(cmd));
                }
                BATCH_ENCODERS.with(|b| b.set(0));
                begin_batch();
            }
        } else {
            drop(bw);
            // No batch. Create + commit our own.
            let backend = get_metal();
            let cmd = backend.queue.new_command_buffer();
            f(cmd);
            note_commit(cmd);
            cmd.commit();
            PENDING_CMDS.with(|p| p.borrow_mut().push(cmd.to_owned()));
        }
    });
}

#[inline]
#[allow(dead_code)] // legacy callers; use with_cmd in new code
fn commit_async(cmd: &metal::CommandBufferRef) {
    // When a batch is open, the caller's cmd IS the shared cmd — don't
    // commit it; flush() will handle that. Detect by reference equality
    // (CommandBufferRef pointer).
    let in_batch = CURRENT_CMD.with(|c| c.borrow().is_some());
    if in_batch { return; }
    note_commit(cmd);
    cmd.commit();
    PENDING_CMDS.with(|c| c.borrow_mut().push(cmd.to_owned()));
}

/// Wait for all in-flight GPU work to complete. Cheap if no async cmds are
/// pending. Call before any CPU read or fresh CPU write to a buffer that
/// may have been touched by an async-committed kernel. If a batch is open,
/// commits and clears it first.
pub fn flush() {
    CURRENT_CMD.with(|c| {
        if let Some(cmd) = c.borrow_mut().take() {
            note_commit(&cmd);
            cmd.commit();
            PENDING_CMDS.with(|p| p.borrow_mut().push(cmd));
        }
    });
    // Wait on each committed buffer. An empty barrier buffer is not enough:
    // it can complete before work committed ahead of it has finished.
    let pending: Vec<metal::CommandBuffer> =
        PENDING_CMDS.with(|c| std::mem::take(&mut *c.borrow_mut()));
    for cmd in pending {
        cmd.wait_until_completed();
        // A command buffer that faults or is killed by the GPU watchdog leaves
        // its output *partially written* and reports nothing unless asked, so
        // silence here would look exactly like a miscomputing kernel.
        if cmd.status() == metal::MTLCommandBufferStatus::Error {
            use metal::foreign_types::ForeignType;
            let raw = cmd.as_ptr() as *mut objc2::runtime::AnyObject;
            let err: *mut objc2::runtime::AnyObject = unsafe { objc2::msg_send![raw, error] };
            let desc = if err.is_null() {
                "(no description)".to_string()
            } else {
                unsafe {
                    let s: *mut objc2::runtime::AnyObject =
                        objc2::msg_send![err, localizedDescription];
                    let c: *const std::os::raw::c_char = objc2::msg_send![s, UTF8String];
                    std::ffi::CStr::from_ptr(c).to_string_lossy().into_owned()
                }
            };
            panic!(
                "Metal command buffer failed: {desc}\n\
                 If this is kIOGPUCommandBufferCallbackErrorImpactingInteractivity, \
                 macOS killed the work for holding the GPU too long; the output \
                 buffer is partially written, so continuing would silently produce \
                 zeros. It fires when something else is competing for the GPU — \
                 another generation process, or heavy display work. Run one job at \
                 a time (scripts/wan_run.sh enforces this) and retry."
            );
        }
    }
}

/// Start a GPU trace capture to a `.gputrace` file. Requires
/// `METAL_CAPTURE_ENABLED=1` in the environment. The trace can be
/// opened in Xcode → Open Developer Tool → Metal System Trace, or
/// just double-click the `.gputrace` bundle.
pub fn gpu_capture_start(path: &str) {
    unsafe {
        let mgr_cls = AnyClass::get("MTLCaptureManager").expect("MTLCaptureManager");
        let mgr: *mut AnyObject = msg_send![mgr_cls, sharedCaptureManager];
        let desc_cls = AnyClass::get("MTLCaptureDescriptor").expect("MTLCaptureDescriptor");
        let desc: *mut AnyObject = msg_send![desc_cls, alloc];
        let desc: *mut AnyObject = msg_send![desc, init];
        let backend = get_metal();
        use metal::foreign_types::ForeignType;
        let dev_ptr: *mut AnyObject = ForeignType::as_ptr(&backend.device) as *mut AnyObject;
        let _: () = msg_send![desc, setCaptureObject: dev_ptr];
        // destination = 2 = MTLCaptureDestinationGPUTraceDocument
        let _: () = msg_send![desc, setDestination: 2usize];
        let nsstring_cls = AnyClass::get("NSString").unwrap();
        let path_ns: *mut AnyObject = msg_send![nsstring_cls,
            stringWithUTF8String: std::ffi::CString::new(path).unwrap().as_ptr()];
        let url_cls = AnyClass::get("NSURL").unwrap();
        let url: *mut AnyObject = msg_send![url_cls, fileURLWithPath: path_ns];
        let _: () = msg_send![desc, setOutputURL: url];
        let mut err: *mut AnyObject = std::ptr::null_mut();
        let ok: bool = msg_send![mgr, startCaptureWithDescriptor: desc error: &mut err];
        if !ok {
            if !err.is_null() {
                let reason: *mut AnyObject = msg_send![err, localizedDescription];
                let utf8: *const u8 = msg_send![reason, UTF8String];
                if !utf8.is_null() {
                    let s = std::ffi::CStr::from_ptr(utf8 as *const _).to_string_lossy();
                    eprintln!("[GPU CAPTURE] failed to start: {s}");
                }
            } else {
                eprintln!("[GPU CAPTURE] failed to start (no error)");
            }
        } else {
            eprintln!("[GPU CAPTURE] started → {path}");
        }
    }
}

/// Stop an active GPU trace capture.
pub fn gpu_capture_stop() {
    unsafe {
        let mgr_cls = AnyClass::get("MTLCaptureManager").expect("MTLCaptureManager");
        let mgr: *mut AnyObject = msg_send![mgr_cls, sharedCaptureManager];
        let _: () = msg_send![mgr, stopCapture];
        eprintln!("[GPU CAPTURE] stopped");
    }
}

/// Drain the buffer pool, releasing all cached Metal buffers.
/// Call between per-frame GPU dispatches in the video forward to
/// prevent cross-frame buffer reuse within the pool.
pub fn drain_pool() {
    BUF_POOL.with_borrow_mut(|pool| pool.clear());
}

thread_local! {
    /// `KLEARU_NO_BUF_POOL=1` disables buffer recycling entirely: every
    /// `release_buffer` drops its buffer and every `acquire_buffer` allocates
    /// fresh. Diagnostic only — it trades allocation churn for the guarantee
    /// that no buffer is ever handed out while a still-running kernel reads it.
    static NO_POOL: bool = std::env::var("KLEARU_NO_BUF_POOL").is_ok();
}

/// GPU allocation state: what this process has allocated on the device, and
/// what the device says it can hold. MPS silently returns NaN rather than
/// failing when the working set is overcommitted, so this is how you tell an
/// overcommit from a numerical bug.
pub fn gpu_memory_report(tag: &str) {
    let d = &get_metal().device;
    let alloc = d.current_allocated_size() as f64 / 1e9;
    let rec = d.recommended_max_working_set_size() as f64 / 1e9;
    // Cache sizes alongside the total: a cache that grows per frame is a
    // keying bug (an unstable cache key), and it looks identical to a plain
    // leak if you only watch the byte count.
    let graphs = CONV_GRAPH_CACHE.with_borrow(|c| c.len());
    let wcache = MPSGRAPH_WEIGHT_F16_CACHE.with_borrow(|c| c.len());
    let pool = BUF_POOL.with_borrow(|p| p.values().map(|v| v.len()).sum::<usize>());
    let peak = PEAK_ALLOCATED.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
    println!(
        "[gpu] {tag}: allocated {alloc:.2} GB (peak {peak:.2}) / max {rec:.2} GB           conv_graphs={graphs} weight_cache={wcache} pooled_bufs={pool}"
    );
}

/// Blit-zero `bytes` of `buf` on the current command buffer.
fn zero_buffer(buf: &Buffer, bytes: usize) {
    with_cmd(|cmd| {
        let blit = cmd.new_blit_command_encoder();
        blit.fill_buffer(buf, metal::NSRange::new(0, bytes as u64), 0);
        blit.end_encoding();
    });
}

/// Return a buffer to the pool for reuse.
fn release_buffer(buf: Buffer) {
    if NO_POOL.with(|v| *v) {
        return;
    }
    let bytes = buf.length() as usize;
    let cmd = LAST_CMD.with(|c| c.borrow().clone());
    BUF_POOL.with_borrow_mut(|pool| {
        pool.entry(bytes).or_default().push((cmd, buf));
    });
}

// ---------- f16 weight cache ----------------------------------------------
//
// Per-call f32→f16 conversion is bandwidth-bound (~2-5MB per side, ~1-3ms
// per call at M1 RAM bandwidth). For SDXL UNet that's ~100 big matmuls × 50
// CFG forwards × 2 sides × 1ms = ~10s of pure conversion work per inference.
//
// Most of those calls re-convert the SAME WEIGHTS — Linear/Conv weights
// don't change across timesteps. Cache the f16 buffer keyed by (ptr, len)
// + a content fingerprint so we re-convert only when the source changes
// (i.e., activations get cache misses, weights get hits).
//
// Activation buffers may collide with cached entries when memory is reused
// — the fingerprint catches that and we re-convert. Weights are stable.

struct CachedF16 { fingerprint: u64, buffer: Buffer }

thread_local! {
    static F16_CACHE: RefCell<HashMap<(usize, usize), CachedF16>> = RefCell::new(HashMap::new());
}

// SDXL UNet has ~1700 weight tensors (50 resnets × ~8 + 70 transformer blocks ×
// ~18). The cache must hold ALL of them to avoid steady-state thrashing —
// each miss does an `acquire_buffer + flush + CPU-write` which forces a sync
// point. Even the largest UNet variant fits well under this cap.
//
// The cache only holds Metal buffer *handles*, not weight memory itself
// (those are referenced via the underlying MTLBuffer's retain count), so
// raising the cap costs O(entries) HashMap memory — negligible.
const F16_CACHE_MAX: usize = 4096;

#[inline]
fn fingerprint_f32(slice: &[f32]) -> u64 {
    if slice.is_empty() { return 0; }
    let n = slice.len();
    let take = n.min(8);
    let mut h: u64 = n as u64;
    // First few + last few elements — collision-resistant enough for f32 weights.
    for i in 0..take {
        h = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(slice[i].to_bits() as u64);
    }
    for i in 0..take {
        h = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(slice[n - 1 - i].to_bits() as u64);
    }
    h
}

/// Return a Metal buffer holding the f16-converted contents of `slice`.
/// On cache hit (same `(ptr, len)` and matching fingerprint), returns a
/// retain-bumped clone of the cached buffer — zero conversion work. On
/// miss, converts and inserts. The returned buffer must be released by
/// the caller (via `release_buffer`) when done — same lifecycle as
/// `acquire_buffer` returns.
fn convert_or_get_cached_f16(slice: &[f32]) -> Buffer {
    use half::bf16;
    use half::slice::HalfFloatSliceExt;
    let key = (slice.as_ptr() as usize, slice.len());
    let fp = fingerprint_f32(slice);

    F16_CACHE.with_borrow_mut(|cache| {
        if let Some(cached) = cache.get(&key) {
            if cached.fingerprint == fp {
                // Diagnostic: when KLEARU_DEBUG_WEIGHT_CACHE is set, scan the
                // cached buffer's contents for NaN/Inf before returning. If
                // a previously-cached weight is now corrupted, something
                // outside the cache wrote into its memory — that's the bug.
                if std::env::var_os("KLEARU_DEBUG_WEIGHT_CACHE").is_some() {
                    use half::bf16;
                    let n = slice.len();
                    let mut nan = 0_usize;
                    let mut inf = 0_usize;
                    unsafe {
                        let p: *const bf16 = cached.buffer.contents() as *const bf16;
                        for i in 0..n {
                            let v = (*p.add(i)).to_f32();
                            if v.is_nan() { nan += 1; }
                            else if v.is_infinite() { inf += 1; }
                        }
                    }
                    if nan > 0 || inf > 0 {
                        eprintln!("[weight-cache-corrupt] key=({:#x}, {n}), NaN={nan}, Inf={inf} \
                            — cached f16 weight has been overwritten between iterations. \
                            First/last 8 elements still match fingerprint, but mid-buffer is bad.",
                            key.0);
                    }
                }
                // Hit. Clone bumps Objective-C retain; caller drops their copy
                // after use, leaving the cache entry alive at refcount 1.
                return cached.buffer.clone();
            }
            // Stale entry. We MUST NOT release this Buffer to BUF_POOL:
            // callers' previous `clone()` of this Buffer may still be queued
            // for GPU consumption (sgemm reading it from a yet-to-commit
            // batch). If the buffer is recycled to the pool and a future
            // `acquire_buffer` pops it for an `upload_f32_as_f16` write,
            // the new write races against the in-flight sgemm read — Metal
            // hazard tracking only covers committed cmd buffers, so a partial
            // overwrite slips through and silently corrupts the weight,
            // producing NaN in some output channels (sporadic across batch
            // iterations; observed in VAE encode after a few REPL turns).
            // Just drop the cached Buffer; Objective-C retain count handles
            // teardown when the last caller clone goes away. Worst case is
            // a small steady-state allocation of fresh GPU memory; far
            // cheaper than a corrupted run.
            cache.remove(&key);
        }

        // Bound cache size to prevent unbounded growth from activation pointers.
        // Same reasoning as above — drop, don't recycle.
        if cache.len() >= F16_CACHE_MAX {
            if let Some(evict_key) = cache.keys().next().copied() {
                cache.remove(&evict_key);
            }
        }

        // Convert into a fresh buffer. Flush any in-flight GPU work that may
        // still be reading the buffer we're about to acquire from the pool.
        let bytes = slice.len() * 2;
        let buf = acquire_buffer(bytes);
        flush();
        unsafe {
            let dst: *mut bf16 = buf.contents() as *mut bf16;
            let dst_slice: &mut [bf16] = std::slice::from_raw_parts_mut(dst, slice.len());
            dst_slice.convert_from_f32_slice(slice);
        }
        cache.insert(key, CachedF16 { fingerprint: fp, buffer: buf.clone() });
        buf
    })
}

// ---------- MPS sgemm ------------------------------------------------------

/// SGEMM via MPS. C[m,n] = A[m,k] · B[k,n], row-major, alpha=1, beta=0, no transpose.
///
/// Apple's MPSMatrixMultiplication is hand-tuned by Apple. Unified-memory
/// MTLBuffers allow CPU/GPU sharing without a PCIe transfer.
pub fn sgemm_metal(
    m: usize, n: usize, k: usize,
    a: &[f32], b: &[f32], c: &mut [f32],
) {
    let backend = get_metal();
    let bytes = std::mem::size_of::<f32>();

    let a_bytes = a.len() * bytes;
    let b_bytes = b.len() * bytes;
    let c_bytes = c.len() * bytes;
    let buf_a = acquire_mps_buffer(a_bytes);
    let buf_b = acquire_mps_buffer(b_bytes);
    let buf_c = acquire_mps_buffer(c_bytes);

    // Copy A and B to GPU buffers (Shared = same pages on M-series).
    // CRITICAL: flush before CPU write — pool may return a buffer still in
    // flight from a prior async-committed kernel; CPU-writing into it would
    // race with the GPU's read of the prior contents.
    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(a.as_ptr(), buf_a.contents() as *mut f32, a.len());
        std::ptr::copy_nonoverlapping(b.as_ptr(), buf_b.contents() as *mut f32, b.len());
    }

    unsafe { dispatch_mps_matmul(backend, &buf_a, &buf_b, &buf_c, m, n, k,
                                 MPS_DATA_TYPE_FLOAT32, std::mem::size_of::<f32>()) };

    // Read result back. Flush async cmd queue first.
    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(buf_c.contents() as *const f32, c.as_mut_ptr(), c.len());
    }

    release_buffer(buf_a);
    release_buffer(buf_b);
    release_buffer(buf_c);
}

/// SGEMM via MPS in fp16. C[m,n] = A[m,k] · B[k,n], inputs/output as f32.
///
/// Internally converts A, B from f32 → f16 into shared MTLBuffers, runs
/// MPSMatrixMultiplication with `MPSDataTypeFloat16`, converts the f16
/// result back to f32. M1 GPU does fp16 matmul at ~5 TFLOPS vs ~1 TFLOPS
/// AMX fp32 — for large matmuls this is a meaningful win even with
/// per-call conversion overhead. fp16 has ~3.5 decimal digits of mantissa
/// precision; SD/SDXL are robust to this in feature-path matmuls (the
/// quality loss is below sampling noise).
///
/// f16 dynamic range is ±65504, so very large activations could
/// overflow. SD's GroupNorm + activation pipeline keeps values bounded
/// in roughly ±10 range, so this is safe in practice.
/// fp16 MPS sgemm with C = A · Bᵀ. A is [m, k], B is [n, k] (logically [k, n]
/// after transpose). Same fp16 conversion + accuracy story as `sgemm_metal_f16`.
/// Used for `Linear::forward_batch` where weight is stored as `[out, in]`.
pub fn sgemm_metal_f16_a_btrans(
    m: usize, n: usize, k: usize,
    a: &[f32], b: &[f32], c: &mut [f32],
) {
    use half::bf16;
    use half::slice::HalfFloatSliceExt;
    let backend = get_metal();

    let c_bytes_f16 = c.len() * 2;
    // Both A and B go through the f16 cache. Weights (B is typical for
    // Linear) hit the cache; activations (A) typically miss and reconvert.
    let buf_a = convert_or_get_cached_f16(a);
    let buf_b = convert_or_get_cached_f16(b);
    let buf_c = acquire_buffer(c_bytes_f16);

    // Storage is f16; MPS doesn't support f16 sgemm so we cast to fp32.
    unsafe {
        dispatch_mps_matmul_bf16_via_f32(
            backend, &buf_a, &buf_b, &buf_c,
            m, n, k,
            /*transpose_left=*/false, /*transpose_right=*/true,
            /*b_rows=*/n, /*b_cols=*/k,
        )
    };

    flush();
    unsafe {
        let src_c: *const bf16 = buf_c.contents() as *const bf16;
        let slice_c: &[bf16] = std::slice::from_raw_parts(src_c, c.len());
        slice_c.convert_to_f32_slice(c);
    }

    // buf_a and buf_b are shared with the F16_CACHE — drop naturally so the
    // ObjC retain count goes back to 1 (cache only). Releasing into BUF_POOL
    // would let `acquire_buffer` hand the same MTLBuffer to another caller
    // who'd overwrite the cached fp16 data.
    drop(buf_a);
    drop(buf_b);
    release_buffer(buf_c);
}

pub fn sgemm_metal_f16(
    m: usize, n: usize, k: usize,
    a: &[f32], b: &[f32], c: &mut [f32],
) {
    use half::bf16;
    use half::slice::HalfFloatSliceExt;
    let backend = get_metal();

    let c_bytes_f16 = c.len() * 2;
    let buf_a = convert_or_get_cached_f16(a);
    let buf_b = convert_or_get_cached_f16(b);
    let buf_c = acquire_buffer(c_bytes_f16);

    unsafe { dispatch_mps_matmul_bf16_via_f32(backend, &buf_a, &buf_b, &buf_c,
                                               m, n, k, false, false, k, n) };

    // f16 → f32 (SIMD-vectorized). Flush async queue first.
    flush();
    unsafe {
        let src_c: *const bf16 = buf_c.contents() as *const bf16;
        let slice_c: &[bf16] = std::slice::from_raw_parts(src_c, c.len());
        slice_c.convert_to_f32_slice(c);
    }

    drop(buf_a);
    drop(buf_b);
    release_buffer(buf_c);
}

unsafe fn dispatch_mps_matmul(
    backend: &MetalBackend,
    a: &Buffer, b: &Buffer, c: &Buffer,
    m: usize, n: usize, k: usize,
    dtype: u32,
    bytes_per_elem: usize,
) {
    dispatch_mps_matmul_with_transpose(
        backend, a, b, c, m, n, k, dtype, bytes_per_elem,
        false, false, k, n,
    )
}

/// GPU f16 → f32 cast via the `cast_f16_to_f32` kernel. (`f16` in the
/// kernel name is historical; the source storage is f16.) Both buffers
/// are GPU-resident.
fn cast_half_to_f32_buf(src: &Buffer, dst: &Buffer, n_elements: usize) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.cast_half_to_f32_pipe);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid  = MTLSize::new(n_elements as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}
fn cast_f32_to_half_buf(src: &Buffer, dst: &Buffer, n_elements: usize) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.cast_f32_to_half_pipe);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid  = MTLSize::new(n_elements as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}
fn cast_bf16_to_f32_buf(src: &Buffer, dst: &Buffer, n_elements: usize) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.cast_f16_to_f32_pipe);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid  = MTLSize::new(n_elements as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// GPU bf16 → IEEE half cast. Used by the MPS f16 sgemm path (faster
/// than the legacy f32 cast wrapper).
fn cast_bf16_to_half_buf(src: &Buffer, dst: &Buffer, n_elements: usize) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.cast_bf16_to_half_pipe);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid  = MTLSize::new(n_elements as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// GPU IEEE half → bf16 cast.
fn cast_half_to_bf16_buf(src: &Buffer, dst: &Buffer, n_elements: usize) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.cast_half_to_bf16_pipe);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid  = MTLSize::new(n_elements as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// GPU f32 → f16 cast.
fn cast_f32_to_bf16_buf(src: &Buffer, dst: &Buffer, n_elements: usize) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.cast_f32_to_f16_pipe);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid  = MTLSize::new(n_elements as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// MPS matmul wrapper: f16 inputs → fp32 staging → MPS f32 sgemm → f16
/// output. Used everywhere that storage is f16. MPSMatrixMultiplication
/// supports only fp16/fp32/int dtypes (not f16), so we cast through fp32.
/// Cost: 3 cast dispatches + 3 fp32 buffer allocations per sgemm + ~2× the
/// matmul throughput vs fp16. Required for correctness — fp16 overflows on
/// SDXL deep-stage activations and on SD VAE upper decoder activations.
unsafe fn dispatch_mps_matmul_bf16_via_f32(
    backend: &MetalBackend,
    a_bf16: &Buffer, b_bf16: &Buffer, c_bf16: &Buffer,
    m: usize, n: usize, k: usize,
    transpose_left: bool, transpose_right: bool,
    b_rows: usize, b_cols: usize,
) {
    // Two precision paths share storage (always bf16 between layers):
    //   * Fast f16 scratch — UNet matmul. MPS f16 sgemm, ~2× faster on
    //     Apple Silicon, half the memory. MPS accumulates in fp32 so
    //     transient sums don't overflow; output cast to f16 IS limited
    //     to ±65504 but UNet activation magnitudes stay well under.
    //   * Safe f32 scratch — VAE decoder & anywhere intermediate
    //     activations can exceed fp16 max (~23K in SDXL VAE, would
    //     overflow on the f16 output cast). Caller flips USE_F32_SGEMM.
    let use_f32 = USE_F32_SGEMM.with(|c| c.get());
    let half_io = SGEMM_HALF_IO.with(|c| c.get());
    let a_elems = m * k;
    let b_elems = b_rows * b_cols;
    let c_elems = m * n;

    // acquire_buffer zero-initializes recycled buffers, so c_f32/c_f16 are
    // safe inputs to MPS matmul's beta=0 path by construction.
    //
    // Explicitly zeroing C here as well was tried against the deterministic
    // large-sequence NaN and changed nothing (9360 tokens stayed 100%
    // non-finite), so a dirty C is not that bug.
    if use_f32 {
        let a_f32 = acquire_mps_buffer(a_elems * 4);
        let b_f32 = acquire_mps_buffer(b_elems * 4);
        let c_f32 = acquire_mps_buffer(c_elems * 4);
        mps_addr_report(&[("A", &a_f32), ("B", &b_f32), ("C", &c_f32), ("a_src", a_bf16), ("b_src", b_bf16), ("c_dst", c_bf16)]);
        // Large-K MPS matmuls corrupt rows unless the CPU waits around them;
        // `fence_mps` does when this thread asks for it (`set_mps_barrier`).
        fence_mps(
            || {
                // Storage may be bfloat16 or IEEE f16; composing half->bf16->f32
                // would round away the very mantissa bits half exists to keep.
                if half_io {
                    cast_half_to_f32_buf(a_bf16, &a_f32, a_elems);
                    cast_half_to_f32_buf(b_bf16, &b_f32, b_elems);
                } else {
                    cast_bf16_to_f32_buf(a_bf16, &a_f32, a_elems);
                    cast_bf16_to_f32_buf(b_bf16, &b_f32, b_elems);
                }
            },
            || dispatch_mps_matmul_with_transpose(
                backend, &a_f32, &b_f32, &c_f32,
                m, n, k, MPS_DATA_TYPE_FLOAT32, 4,
                transpose_left, transpose_right, b_rows, b_cols,
            ),
            || if half_io {
                cast_f32_to_half_buf(&c_f32, c_bf16, c_elems);
            } else {
                cast_f32_to_bf16_buf(&c_f32, c_bf16, c_elems);
            },
        );
        release_buffer(a_f32);
        release_buffer(b_f32);
        release_buffer(c_f32);
    } else {
        let a_f16 = acquire_mps_buffer(a_elems * 2);
        let b_f16 = acquire_mps_buffer(b_elems * 2);
        let c_f16 = acquire_mps_buffer(c_elems * 2);
        mps_addr_report(&[("A", &a_f16), ("B", &b_f16), ("C", &c_f16), ("a_src", a_bf16), ("b_src", b_bf16), ("c_dst", c_bf16)]);
        // Fenced as in the f32 branch above.
        fence_mps(
            || if half_io {
                // Already IEEE f16 — copy rather than convert.
                blit_copy_f16(a_bf16, 0, &a_f16, 0, a_elems);
                blit_copy_f16(b_bf16, 0, &b_f16, 0, b_elems);
            } else {
                cast_bf16_to_half_buf(a_bf16, &a_f16, a_elems);
                cast_bf16_to_half_buf(b_bf16, &b_f16, b_elems);
            },
            || dispatch_mps_matmul_with_transpose(
                backend, &a_f16, &b_f16, &c_f16,
                m, n, k, MPS_DATA_TYPE_FLOAT16, 2,
                transpose_left, transpose_right, b_rows, b_cols,
            ),
            || if half_io {
                blit_copy_f16(&c_f16, 0, c_bf16, 0, c_elems);
            } else {
                cast_half_to_bf16_buf(&c_f16, c_bf16, c_elems);
            },
        );
        release_buffer(a_f16);
        release_buffer(b_f16);
        release_buffer(c_f16);
    }
}

/// Same as above but with explicit byte offsets into each buffer. Used by
/// `sgemm_f16_buf_with_offsets` for the per-batch im2col→conv loop.
unsafe fn dispatch_mps_matmul_bf16_via_f32_with_offsets(
    backend: &MetalBackend,
    a_bf16: &Buffer, a_off_bytes_bf16: usize,
    b_bf16: &Buffer, b_off_bytes_bf16: usize,
    c_bf16: &Buffer, c_off_bytes_bf16: usize,
    m: usize, n: usize, k: usize,
    transpose_left: bool, transpose_right: bool,
    b_rows: usize, b_cols: usize,
) {
    use metal::MTLSize;
    let use_f32 = USE_F32_SGEMM.with(|c| c.get());
    // Storage may be bfloat16 or IEEE f16. This variant previously assumed
    // bfloat unconditionally, so when the chunked path (m > MPS_MAX_ROWS) took
    // over for a half-storage model it reinterpreted half bits as bfloat — a
    // DiT that was correct at 192 tokens produced noise at 11440.
    let half_io = SGEMM_HALF_IO.with(|c| c.get());
    let a_elems = m * k;
    let b_elems = b_rows * b_cols;
    let c_elems = m * n;
    let bytes = if use_f32 { 4 } else { 2 };

    // K-alignment workaround: MPSMatrixMultiplication produces NaN in
    // trailing output rows when K is small and not aligned to the
    // simdgroup width (32). Specifically observed: m=128, n=1M, k=27
    // → output rows 24..27 are NaN (4M elements). The trailing 5 K
    // lanes within a 32-wide simdgroup reduce apparently aren't
    // initialised, contaminating the dot-product sum.
    //
    // Workaround: pad K to a multiple of K_ALIGN with zeros. Zero
    // contributes nothing to the dot product (`0 * a_ki = 0`), so the
    // mathematical result is unchanged. We allocate larger A and B
    // buffers, copy the original data into the leading K positions,
    // and rely on `acquire_buffer`'s pool zero-init to leave the
    // trailing K positions as zero.
    //
    // Only activates when transpose flags are false (the conv2d /
    // im2col path); the linear/attention paths use larger K (≥128)
    // which are already aligned and unaffected.
    const K_ALIGN: usize = 32;
    let needs_pad = k % K_ALIGN != 0 && !transpose_left && !transpose_right;
    let k_padded = if needs_pad {
        k.div_ceil(K_ALIGN) * K_ALIGN
    } else { k };
    let a_alloc_elems = m * k_padded;
    let b_alloc_elems = if needs_pad { k_padded * b_cols } else { b_elems };

    // acquire_buffer zero-initializes recycled buffers, so c_buf (matmul
    // destination, beta=0 path) and the trailing K columns/rows of a_buf
    // and b_buf are safe by construction.
    let a_buf = acquire_mps_buffer(a_alloc_elems * bytes);
    let b_buf = acquire_mps_buffer(b_alloc_elems * bytes);
    let c_buf = acquire_mps_buffer(c_elems * bytes);
    mps_addr_report(&[("A", &a_buf), ("B", &b_buf), ("C", &c_buf), ("a_src", a_bf16), ("b_src", b_bf16), ("c_dst", c_bf16)]);

    let dispatch_cast = |pipe: &ComputePipelineState, src: &Buffer, src_off: usize,
                         dst: &Buffer, dst_off: usize, n_elem: usize| {
        with_cmd(|cmd| {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(src), src_off as u64);
            enc.set_buffer(1, Some(dst), dst_off as u64);
            let group = MTLSize::new(256, 1, 1);
            let grid  = MTLSize::new(n_elem as u64, 1, 1);
            enc.dispatch_threads(grid, group);
            enc.end_encoding();
        });
    };

    // Cast inputs into the (possibly padded) buffers. Layouts for the
    // non-transposed path:
    //   A is [m, k_padded] row-major. Original A is [m, k]; the trailing
    //     (k_padded - k) columns per row are zero from pool init. To get
    //     this layout we need a row-by-row cast (each row's data lives at
    //     a different stride in the destination).
    //   B is [k_padded, b_cols] row-major. Original B is [k, b_cols] which
    //     is contiguous from position 0. Padded rows (k..k_padded) at
    //     positions [k * b_cols, k_padded * b_cols) stay zero.
    let cast_a_padded = |pipe: &ComputePipelineState| {
        if !needs_pad {
            dispatch_cast(pipe, a_bf16, a_off_bytes_bf16, &a_buf, 0, a_elems);
        } else {
            // Per-row cast — m small dispatches. For the conv2d use case
            // m ∈ {128, 256, 512}, well within reasonable dispatch overhead.
            for mi in 0..m {
                let src_off = a_off_bytes_bf16 + mi * k * 2;
                let dst_off = mi * k_padded * bytes;
                dispatch_cast(pipe, a_bf16, src_off, &a_buf, dst_off, k);
            }
        }
    };
    let cast_b_padded = |pipe: &ComputePipelineState| {
        // Single contiguous cast either way; padded rows at the end stay zero.
        dispatch_cast(pipe, b_bf16, b_off_bytes_bf16, &b_buf, 0, b_elems);
    };

    if use_f32 {
        let to_f32 = if half_io {
            &backend.cast_half_to_f32_pipe
        } else {
            &backend.cast_f16_to_f32_pipe
        };
        fence_mps(
            || {
                cast_a_padded(to_f32);
                cast_b_padded(to_f32);
            },
            || dispatch_mps_matmul_with_offsets(
                backend, &a_buf, 0, &b_buf, 0, &c_buf, 0,
                m, n, k_padded, MPS_DATA_TYPE_FLOAT32, 4,
                transpose_left, transpose_right,
                if needs_pad { k_padded } else { b_rows }, b_cols,
            ),
            || {
                // Diagnostic: snapshot f32 inputs / output of MPS matmul. If c_buf
                // already has NaN here, MPS itself is producing it (independent of
                // the f32→f16 cast or bias-add downstream).
                if std::env::var_os("KLEARU_DEBUG_MPS_MATMUL").is_some() {
                    flush();
                    // Window-scan a buffer at start / mid / end so we can localise
                    // NaN to a specific row range (each row = `n` elements, so each
                    // 1M-element window covers 1M/n rows when c is row-major [m, n]).
                    let win = 1 << 20; // 1M elements per window.
                    let scan_full = |label: &str, buf: &Buffer, total: usize| {
                        let n_total_nan = {
                            // Full scan (slow but definitive).
                            let mut out = vec![0.0_f32; total];
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    buf.contents() as *const f32, out.as_mut_ptr(), total);
                            }
                            let mut nan = 0_usize; let mut inf = 0_usize;
                            let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
                            let mut sa = 0.0_f64;
                            let mut first_nan_idx: Option<usize> = None;
                            for (i, &x) in out.iter().enumerate() {
                                if x.is_nan() {
                                    nan += 1;
                                    if first_nan_idx.is_none() { first_nan_idx = Some(i); }
                                } else if x.is_infinite() { inf += 1; }
                                else { mn = mn.min(x); mx = mx.max(x); sa += x.abs() as f64; }
                            }
                            eprintln!("[mps-matmul] {label:<24} m={m} n={n} k={k} total={total} \
                                NaN={nan} Inf={inf} min={mn:.3} max={mx:.3} mean_abs={:.5} \
                                first_nan_idx={first_nan_idx:?} (row={:?})",
                                sa / total as f64,
                                first_nan_idx.map(|i| i / n));
                            nan
                        };
                        let _ = win;
                        n_total_nan
                    };
                    // a_f32 is small, scan in full.
                    scan_full("a_f32 (post-cast)", &a_buf, a_elems);
                    // b_f32 and c_f32 can be huge — full scan is slow but accurate.
                    // Set KLEARU_MPS_MATMUL_QUICK=1 to limit to 1M-element windows.
                    let quick = std::env::var_os("KLEARU_MPS_MATMUL_QUICK").is_some();
                    if quick {
                        let mut out = vec![0.0_f32; c_elems.min(win)];
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                c_buf.contents() as *const f32, out.as_mut_ptr(),
                                out.len());
                        }
                        let nan = out.iter().filter(|x| x.is_nan()).count();
                        eprintln!("[mps-matmul] c_f32 first 1M  m={m} n={n} k={k} NaN={nan}");
                        // End window: last 1M elements.
                        let tail_off = c_elems.saturating_sub(win);
                        let tail_len = c_elems - tail_off;
                        let mut tail = vec![0.0_f32; tail_len];
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                (c_buf.contents() as *const f32).add(tail_off),
                                tail.as_mut_ptr(), tail_len);
                        }
                        let nan_tail = tail.iter().filter(|x| x.is_nan()).count();
                        eprintln!("[mps-matmul] c_f32 last 1M   m={m} n={n} k={k} NaN={nan_tail} \
                            (tail starts at row {})", tail_off / n);
                    } else {
                        scan_full("c_f32 (post-matmul)", &c_buf, c_elems);
                    }
                }
                let from_f32 = if half_io {
                    &backend.cast_f32_to_half_pipe
                } else {
                    &backend.cast_f32_to_f16_pipe
                };
                dispatch_cast(from_f32, &c_buf, 0, c_bf16, c_off_bytes_bf16, c_elems);
            },
        );
    } else {
        // half storage is already IEEE f16, so the cast is a straight copy.
        let to_half = if half_io {
            &backend.copy_f16_pipe
        } else {
            &backend.cast_bf16_to_half_pipe
        };
        fence_mps(
            || {
                cast_a_padded(to_half);
                cast_b_padded(to_half);
            },
            || dispatch_mps_matmul_with_offsets(
                backend, &a_buf, 0, &b_buf, 0, &c_buf, 0,
                m, n, k_padded, MPS_DATA_TYPE_FLOAT16, 2,
                transpose_left, transpose_right,
                if needs_pad { k_padded } else { b_rows }, b_cols,
            ),
            || {
                let from_half = if half_io {
                    &backend.copy_f16_pipe
                } else {
                    &backend.cast_half_to_bf16_pipe
                };
                dispatch_cast(from_half, &c_buf, 0, c_bf16, c_off_bytes_bf16, c_elems);
            },
        );
    }

    release_buffer(a_buf);
    release_buffer(b_buf);
    release_buffer(c_buf);
}

/// MPS matmul with explicit byte offsets into each buffer. Used for batched
/// conv where per-batch slices live at non-zero offsets within larger buffers.
unsafe fn dispatch_mps_matmul_with_offsets(
    backend: &MetalBackend,
    a: &Buffer, a_off: usize,
    b: &Buffer, b_off: usize,
    c: &Buffer, c_off: usize,
    m: usize, n: usize, k: usize,
    dtype: u32,
    bytes_per_elem: usize,
    transpose_left: bool,
    transpose_right: bool,
    b_rows: usize,
    b_cols: usize,
) {
    let row_bytes = bytes_per_elem;
    let descriptor_cls = AnyClass::get("MPSMatrixDescriptor")
        .expect("MPSMatrixDescriptor not registered — link MetalPerformanceShaders.framework");
    let matrix_cls = AnyClass::get("MPSMatrix").expect("MPSMatrix not registered");
    let mul_cls = AnyClass::get("MPSMatrixMultiplication")
        .expect("MPSMatrixMultiplication not registered");

    let a_desc: *mut AnyObject = msg_send![descriptor_cls,
        matrixDescriptorWithRows: m as usize
        columns: k as usize
        rowBytes: (k * row_bytes) as usize
        dataType: dtype];
    let b_desc: *mut AnyObject = msg_send![descriptor_cls,
        matrixDescriptorWithRows: b_rows as usize
        columns: b_cols as usize
        rowBytes: (b_cols * row_bytes) as usize
        dataType: dtype];
    let c_desc: *mut AnyObject = msg_send![descriptor_cls,
        matrixDescriptorWithRows: m as usize
        columns: n as usize
        rowBytes: (n * row_bytes) as usize
        dataType: dtype];

    use metal::foreign_types::ForeignType;
    let a_buf_ptr: *mut AnyObject = ForeignType::as_ptr(a) as *mut AnyObject;
    let b_buf_ptr: *mut AnyObject = ForeignType::as_ptr(b) as *mut AnyObject;
    let c_buf_ptr: *mut AnyObject = ForeignType::as_ptr(c) as *mut AnyObject;

    let a_mat: *mut AnyObject = msg_send![matrix_cls, alloc];
    let a_mat: *mut AnyObject = msg_send![a_mat,
        initWithBuffer: a_buf_ptr offset: a_off as usize descriptor: a_desc];
    let b_mat: *mut AnyObject = msg_send![matrix_cls, alloc];
    let b_mat: *mut AnyObject = msg_send![b_mat,
        initWithBuffer: b_buf_ptr offset: b_off as usize descriptor: b_desc];
    let c_mat: *mut AnyObject = msg_send![matrix_cls, alloc];
    let c_mat: *mut AnyObject = msg_send![c_mat,
        initWithBuffer: c_buf_ptr offset: c_off as usize descriptor: c_desc];

    let device_ptr: *mut AnyObject = ForeignType::as_ptr(&backend.device) as *mut AnyObject;
    let mul: *mut AnyObject = msg_send![mul_cls, alloc];
    let mul: *mut AnyObject = msg_send![mul,
        initWithDevice: device_ptr
        transposeLeft: transpose_left
        transposeRight: transpose_right
        resultRows: m as usize
        resultColumns: n as usize
        interiorColumns: k as usize
        alpha: 1.0_f64
        beta: 0.0_f64
    ];

    with_cmd(|cmd| {
        use metal::foreign_types::ForeignTypeRef;
        let cmd_ptr: *mut AnyObject = ForeignTypeRef::as_ptr(&*cmd) as *mut AnyObject;
        let _: () = msg_send![mul,
            encodeToCommandBuffer: cmd_ptr
            leftMatrix: a_mat
            rightMatrix: b_mat
            resultMatrix: c_mat
        ];
    });

    let _: () = msg_send![mul, release];
    let _: () = msg_send![c_mat, release];
    let _: () = msg_send![b_mat, release];
    let _: () = msg_send![a_mat, release];
}

/// Generalised MPS matmul with transpose flags. `b_rows`/`b_cols` describe
/// the *physical* layout of `b` in memory; MPS applies the transpose when
/// reading. For `transpose_right=true`, `b` is physically [n, k] and is
/// treated as logical [k, n] post-transpose.
unsafe fn dispatch_mps_matmul_with_transpose(
    backend: &MetalBackend,
    a: &Buffer, b: &Buffer, c: &Buffer,
    m: usize, n: usize, k: usize,
    dtype: u32,
    bytes_per_elem: usize,
    transpose_left: bool,
    transpose_right: bool,
    b_rows: usize,
    b_cols: usize,
) {
    let row_bytes = bytes_per_elem;

    // Class lookups. `MPSMatrixDescriptor`, `MPSMatrix`, `MPSMatrixMultiplication`
    // are loaded at runtime from MetalPerformanceShaders.framework.
    let descriptor_cls = AnyClass::get("MPSMatrixDescriptor")
        .expect("MPSMatrixDescriptor not registered — link MetalPerformanceShaders.framework");
    let matrix_cls = AnyClass::get("MPSMatrix")
        .expect("MPSMatrix not registered");
    let mul_cls = AnyClass::get("MPSMatrixMultiplication")
        .expect("MPSMatrixMultiplication not registered");

    // --- Descriptors ---
    let a_desc: *mut AnyObject = msg_send![descriptor_cls,
        matrixDescriptorWithRows: m as usize
        columns: k as usize
        rowBytes: (k * row_bytes) as usize
        dataType: dtype
    ];
    let b_desc: *mut AnyObject = msg_send![descriptor_cls,
        matrixDescriptorWithRows: b_rows as usize
        columns: b_cols as usize
        rowBytes: (b_cols * row_bytes) as usize
        dataType: dtype
    ];
    let c_desc: *mut AnyObject = msg_send![descriptor_cls,
        matrixDescriptorWithRows: m as usize
        columns: n as usize
        rowBytes: (n * row_bytes) as usize
        dataType: dtype
    ];

    // --- Wrap MTLBuffers into MPSMatrices ---
    use metal::foreign_types::ForeignType;
    let a_buf_ptr: *mut AnyObject = ForeignType::as_ptr(a) as *mut AnyObject;
    let b_buf_ptr: *mut AnyObject = ForeignType::as_ptr(b) as *mut AnyObject;
    let c_buf_ptr: *mut AnyObject = ForeignType::as_ptr(c) as *mut AnyObject;

    let a_mat: *mut AnyObject = msg_send![matrix_cls, alloc];
    let a_mat: *mut AnyObject = msg_send![a_mat, initWithBuffer: a_buf_ptr descriptor: a_desc];
    let b_mat: *mut AnyObject = msg_send![matrix_cls, alloc];
    let b_mat: *mut AnyObject = msg_send![b_mat, initWithBuffer: b_buf_ptr descriptor: b_desc];
    let c_mat: *mut AnyObject = msg_send![matrix_cls, alloc];
    let c_mat: *mut AnyObject = msg_send![c_mat, initWithBuffer: c_buf_ptr descriptor: c_desc];

    // --- Multiplication operation ---
    let device_ptr: *mut AnyObject = ForeignType::as_ptr(&backend.device) as *mut AnyObject;
    let mul: *mut AnyObject = msg_send![mul_cls, alloc];
    let mul: *mut AnyObject = msg_send![mul,
        initWithDevice: device_ptr
        transposeLeft: transpose_left
        transposeRight: transpose_right
        resultRows: m as usize
        resultColumns: n as usize
        interiorColumns: k as usize
        alpha: 1.0_f64
        beta: 0.0_f64
    ];

    with_cmd(|cmd| {
        use metal::foreign_types::ForeignTypeRef;
        let cmd_ptr: *mut AnyObject = ForeignTypeRef::as_ptr(&*cmd) as *mut AnyObject;
        let _: () = msg_send![mul,
            encodeToCommandBuffer: cmd_ptr
            leftMatrix: a_mat
            rightMatrix: b_mat
            resultMatrix: c_mat
        ];
    });

    // Release the four objc objects we alloc/init'd. (Descriptors are autoreleased.)
    let _: () = msg_send![mul, release];
    let _: () = msg_send![c_mat, release];
    let _: () = msg_send![b_mat, release];
    let _: () = msg_send![a_mat, release];
}

// ---------- Element-wise activations ---------------------------------------

fn dispatch_elementwise(pipe: &ComputePipelineState, x: &mut [f32]) {
    use metal::MTLSize;
    let backend = get_metal();
    let bytes = x.len() * std::mem::size_of::<f32>();
    let buf = acquire_buffer(bytes);
    flush();  // pool may return in-flight buffer; flush before CPU write
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), buf.contents() as *mut f32, x.len());
    }
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&buf), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(x.len() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(buf.contents() as *const f32, x.as_mut_ptr(), x.len());
    }
    release_buffer(buf);
}

pub fn silu_metal(x: &mut [f32]) {
    dispatch_elementwise(&get_metal().silu_pipe, x);
}
pub fn quick_gelu_metal(x: &mut [f32]) {
    dispatch_elementwise(&get_metal().quick_gelu_pipe, x);
}
pub fn gelu_metal(x: &mut [f32]) {
    dispatch_elementwise(&get_metal().gelu_pipe, x);
}

// ---------- GroupNorm ------------------------------------------------------

/// In-place GroupNorm on GPU. `x` has layout [N, C, H, W] flat.
pub fn groupnorm_metal(
    x: &mut [f32],
    gamma: &[f32], beta: &[f32],
    n: usize, c: usize, h: usize, w: usize,
    num_groups: usize, eps: f32,
) {
    use metal::MTLSize;
    let backend = get_metal();
    let cg = c / num_groups;
    let hw = h * w;
    let group_size = cg * hw;

    let bytes_x = x.len() * std::mem::size_of::<f32>();
    let bytes_g = gamma.len() * std::mem::size_of::<f32>();
    let buf_x = acquire_buffer(bytes_x);
    let buf_g = acquire_buffer(bytes_g);
    let buf_b = acquire_buffer(bytes_g);
    flush();  // pool may return in-flight buffer; flush before CPU write
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), buf_x.contents() as *mut f32, x.len());
        std::ptr::copy_nonoverlapping(gamma.as_ptr(), buf_g.contents() as *mut f32, gamma.len());
        std::ptr::copy_nonoverlapping(beta.as_ptr(), buf_b.contents() as *mut f32, beta.len());
    }

    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.groupnorm_pipe);
        enc.set_buffer(0, Some(&buf_x), 0);
        enc.set_buffer(1, Some(&buf_g), 0);
        enc.set_buffer(2, Some(&buf_b), 0);
        let group_size_u = group_size as u32;
        let cg_u = cg as u32;
        let hw_u = hw as u32;
        let ng_u = num_groups as u32;
        enc.set_bytes(3, 4, &group_size_u as *const _ as *const _);
        enc.set_bytes(4, 4, &cg_u as *const _ as *const _);
        enc.set_bytes(5, 4, &hw_u as *const _ as *const _);
        enc.set_bytes(6, 4, &ng_u as *const _ as *const _);
        enc.set_bytes(7, 4, &eps as *const _ as *const _);

        let tpg = 256u64;
        let scratch_bytes = (tpg as usize * 2 * std::mem::size_of::<f32>()) as u64;
        enc.set_threadgroup_memory_length(0, scratch_bytes);

        let groups = (n * num_groups) as u64;
        let group_dim = MTLSize::new(tpg, 1, 1);
        let grid_dim = MTLSize::new(groups, 1, 1);
        enc.dispatch_thread_groups(grid_dim, group_dim);
        enc.end_encoding();
    });

    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(buf_x.contents() as *const f32, x.as_mut_ptr(), x.len());
    }
    release_buffer(buf_x);
    release_buffer(buf_g);
    release_buffer(buf_b);
}

// ---------- LayerNorm ------------------------------------------------------

/// In-place LayerNorm. `x.len()` must be a multiple of `dim`; each chunk
/// of `dim` floats is normalised independently.
pub fn layer_norm_metal(x: &mut [f32], gamma: &[f32], beta: &[f32], dim: usize, eps: f32) {
    use metal::MTLSize;
    let backend = get_metal();
    let n_rows = x.len() / dim;

    let bytes_x = x.len() * std::mem::size_of::<f32>();
    let bytes_g = gamma.len() * std::mem::size_of::<f32>();
    let buf_x = acquire_buffer(bytes_x);
    let buf_g = acquire_buffer(bytes_g);
    let buf_b = acquire_buffer(bytes_g);
    flush();  // pool may return in-flight buffer; flush before CPU write
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), buf_x.contents() as *mut f32, x.len());
        std::ptr::copy_nonoverlapping(gamma.as_ptr(), buf_g.contents() as *mut f32, gamma.len());
        std::ptr::copy_nonoverlapping(beta.as_ptr(), buf_b.contents() as *mut f32, beta.len());
    }

    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.layernorm_pipe);
        enc.set_buffer(0, Some(&buf_x), 0);
        enc.set_buffer(1, Some(&buf_g), 0);
        enc.set_buffer(2, Some(&buf_b), 0);
        let dim_u = dim as u32;
        enc.set_bytes(3, 4, &dim_u as *const _ as *const _);
        enc.set_bytes(4, 4, &eps as *const _ as *const _);
        let tpg = 256u64;
        enc.set_threadgroup_memory_length(0, (tpg as usize * 2 * 4) as u64);
        enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });

    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(buf_x.contents() as *const f32, x.as_mut_ptr(), x.len());
    }
    release_buffer(buf_x);
    release_buffer(buf_g);
    release_buffer(buf_b);
}

// ---------- Softmax --------------------------------------------------------

/// In-place row-wise softmax. `x.len()` must be `n_rows * width`. Each row
/// is treated as a separate softmax. With `causal=true`, position i in the
/// row is masked to -inf when i > row_index.
pub fn softmax_metal(x: &mut [f32], width: usize, causal: bool) {
    use metal::MTLSize;
    let backend = get_metal();
    let n_rows = x.len() / width;

    let bytes_x = x.len() * std::mem::size_of::<f32>();
    let buf_x = acquire_buffer(bytes_x);
    flush();  // pool may return in-flight buffer; flush before CPU write
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), buf_x.contents() as *mut f32, x.len());
    }

    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.softmax_pipe);
        enc.set_buffer(0, Some(&buf_x), 0);
        let width_u = width as u32;
        let causal_u: u32 = if causal { 1 } else { 0 };
        enc.set_bytes(1, 4, &width_u as *const _ as *const _);
        enc.set_bytes(2, 4, &causal_u as *const _ as *const _);
        let tpg = 256u64;
        enc.set_threadgroup_memory_length(0, (tpg as usize * 4) as u64);
        enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });

    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(buf_x.contents() as *const f32, x.as_mut_ptr(), x.len());
    }
    release_buffer(buf_x);
}

// ---------- Elementwise add ------------------------------------------------

/// In-place a[i] += b[i]. Slices must have equal length.
pub fn eadd_metal(a: &mut [f32], b: &[f32]) {
    use metal::MTLSize;
    let backend = get_metal();
    debug_assert_eq!(a.len(), b.len());

    let bytes = a.len() * std::mem::size_of::<f32>();
    let buf_a = acquire_buffer(bytes);
    let buf_b = acquire_buffer(bytes);
    flush();  // pool may return in-flight buffer; flush before CPU write
    unsafe {
        std::ptr::copy_nonoverlapping(a.as_ptr(), buf_a.contents() as *mut f32, a.len());
        std::ptr::copy_nonoverlapping(b.as_ptr(), buf_b.contents() as *mut f32, b.len());
    }
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.eadd_pipe);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.dispatch_threads(MTLSize::new(a.len() as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(buf_a.contents() as *const f32, a.as_mut_ptr(), a.len());
    }
    release_buffer(buf_a);
    release_buffer(buf_b);
}

// ---------- Flash attention -----------------------------------------------

/// Fused attention: O = softmax(Q · Kᵀ · scale) · V via online softmax,
/// no score-matrix materialisation.
///
/// Inputs are head-contiguous: Q, K, V have shape `[N, H, L, D]` flat
/// (i.e. `q[n, h, l, d]` is at offset `((n*H + h)*L + l)*D + d`). Output
/// `o` has the same `[N, H, L_q, D]` layout.
///
/// One threadgroup per output row; threadgroup size is `next_pow2(D)`. For
/// SDXL the attention head_dim is 64 (already a power of two); SD 1.5 uses
/// 40 which rounds up to 64 (24 padding threads contribute zero). The
/// kernel works for any D ≥ 1.
pub fn flash_attention_metal(
    q: &[f32], k: &[f32], v: &[f32], o: &mut [f32],
    n: usize, h: usize, l_q: usize, l_kv: usize, d: usize, scale: f32,
) {
    use metal::MTLSize;
    debug_assert_eq!(q.len(), n * h * l_q * d);
    debug_assert_eq!(k.len(), n * h * l_kv * d);
    debug_assert_eq!(v.len(), n * h * l_kv * d);
    debug_assert_eq!(o.len(), n * h * l_q * d);

    let backend = get_metal();
    let tpg = (d as u64).next_power_of_two().max(1);

    let bytes_q = q.len() * 4;
    let bytes_k = k.len() * 4;
    let bytes_v = v.len() * 4;
    let bytes_o = o.len() * 4;
    let buf_q = acquire_buffer(bytes_q);
    let buf_k = acquire_buffer(bytes_k);
    let buf_v = acquire_buffer(bytes_v);
    let buf_o = acquire_buffer(bytes_o);
    flush();  // pool may return in-flight buffer; flush before CPU write
    unsafe {
        std::ptr::copy_nonoverlapping(q.as_ptr(), buf_q.contents() as *mut f32, q.len());
        std::ptr::copy_nonoverlapping(k.as_ptr(), buf_k.contents() as *mut f32, k.len());
        std::ptr::copy_nonoverlapping(v.as_ptr(), buf_v.contents() as *mut f32, v.len());
    }

    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.flash_attn_pipe);
        enc.set_buffer(0, Some(&buf_q), 0);
        enc.set_buffer(1, Some(&buf_k), 0);
        enc.set_buffer(2, Some(&buf_v), 0);
        enc.set_buffer(3, Some(&buf_o), 0);
        let lq_u = l_q as u32;
        let lkv_u = l_kv as u32;
        let d_u = d as u32;
        enc.set_bytes(4, 4, &lq_u as *const _ as *const _);
        enc.set_bytes(5, 4, &lkv_u as *const _ as *const _);
        enc.set_bytes(6, 4, &d_u as *const _ as *const _);
        enc.set_bytes(7, 4, &scale as *const _ as *const _);

        // Threadgroup memory: k_row (tpg) | v_row (tpg) | reduce (tpg).
        let smem_bytes = (3 * tpg * 4) as u64;
        enc.set_threadgroup_memory_length(0, smem_bytes);

        let groups = (n * h * l_q) as u64;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });

    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(buf_o.contents() as *const f32, o.as_mut_ptr(), o.len());
    }
    release_buffer(buf_q);
    release_buffer(buf_k);
    release_buffer(buf_v);
    release_buffer(buf_o);
}

/// Causal flash attention (f32). Same shapes as `flash_attention_metal`:
///   Q: `[n, h, l_q, d]`, K/V: `[n, h, l_kv, d]`, O: `[n, h, l_q, d]`.
/// Each query at position `lq` only attends to keys at positions
/// `lk ≤ lq` (causal mask). For decoder-only transformers (klearu-image,
/// klearu-llm) this replaces the per-row scoring loop with a single GPU
/// kernel call.
pub fn flash_attention_causal_metal(
    q: &[f32], k: &[f32], v: &[f32], o: &mut [f32],
    n: usize, h: usize, l_q: usize, l_kv: usize, d: usize, scale: f32,
) {
    use metal::MTLSize;
    debug_assert_eq!(q.len(), n * h * l_q * d);
    debug_assert_eq!(k.len(), n * h * l_kv * d);
    debug_assert_eq!(v.len(), n * h * l_kv * d);
    debug_assert_eq!(o.len(), n * h * l_q * d);

    let backend = get_metal();
    let tpg = (d as u64).next_power_of_two().max(1);

    let buf_q = acquire_buffer(q.len() * 4);
    let buf_k = acquire_buffer(k.len() * 4);
    let buf_v = acquire_buffer(v.len() * 4);
    let buf_o = acquire_buffer(o.len() * 4);
    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(q.as_ptr(), buf_q.contents() as *mut f32, q.len());
        std::ptr::copy_nonoverlapping(k.as_ptr(), buf_k.contents() as *mut f32, k.len());
        std::ptr::copy_nonoverlapping(v.as_ptr(), buf_v.contents() as *mut f32, v.len());
    }

    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.flash_attn_causal_pipe);
        enc.set_buffer(0, Some(&buf_q), 0);
        enc.set_buffer(1, Some(&buf_k), 0);
        enc.set_buffer(2, Some(&buf_v), 0);
        enc.set_buffer(3, Some(&buf_o), 0);
        let lq_u = l_q as u32;
        let lkv_u = l_kv as u32;
        let d_u = d as u32;
        enc.set_bytes(4, 4, &lq_u as *const _ as *const _);
        enc.set_bytes(5, 4, &lkv_u as *const _ as *const _);
        enc.set_bytes(6, 4, &d_u as *const _ as *const _);
        enc.set_bytes(7, 4, &scale as *const _ as *const _);

        let smem_bytes = (3 * tpg * 4) as u64;
        enc.set_threadgroup_memory_length(0, smem_bytes);

        let groups = (n * h * l_q) as u64;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });

    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(buf_o.contents() as *const f32, o.as_mut_ptr(), o.len());
    }
    release_buffer(buf_q);
    release_buffer(buf_k);
    release_buffer(buf_v);
    release_buffer(buf_o);
}

/// RMSNorm (f32) in place via Metal.
///   `x`: row-major `[n_rows, dim]`. Each row is normalised by
///        `1 / sqrt(mean(x²) + eps)` and scaled by `gamma`.
///   `gamma`: `[dim]` per-channel scale.
pub fn rms_norm_metal(x: &mut [f32], gamma: &[f32], dim: usize, eps: f32) {
    use metal::MTLSize;
    let backend = get_metal();
    debug_assert_eq!(gamma.len(), dim);
    let n_rows = x.len() / dim;
    debug_assert_eq!(n_rows * dim, x.len());
    let tpg = (dim as u64).next_power_of_two().min(1024).max(1);

    let buf_x = acquire_buffer(x.len() * 4);
    let buf_g = acquire_buffer(gamma.len() * 4);
    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), buf_x.contents() as *mut f32, x.len());
        std::ptr::copy_nonoverlapping(gamma.as_ptr(), buf_g.contents() as *mut f32, gamma.len());
    }

    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.rms_norm_pipe);
        enc.set_buffer(0, Some(&buf_x), 0);
        enc.set_buffer(1, Some(&buf_g), 0);
        let dim_u = dim as u32;
        enc.set_bytes(2, 4, &dim_u as *const _ as *const _);
        enc.set_bytes(3, 4, &eps as *const _ as *const _);
        // scratch buffer: tpg floats for the reduction.
        enc.set_threadgroup_memory_length(0, (tpg * 4) as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(n_rows as u64, 1, 1),
            MTLSize::new(tpg, 1, 1),
        );
        enc.end_encoding();
    });

    flush();
    unsafe {
        std::ptr::copy_nonoverlapping(buf_x.contents() as *const f32, x.as_mut_ptr(), x.len());
    }
    release_buffer(buf_x);
    release_buffer(buf_g);
}

// ---------- GpuTensor (GPU-resident pipeline) -----------------------------

/// Element dtype of a `GpuTensor`. The GPU-resident pipeline stores
/// activations as `F16` (bfloat the bandwidth, 4× the M1 GPU throughput vs
/// fp32). Boundary I/O is f32 — `upload_f32_as_f16` and `download_to_f32`
/// handle the conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// `F16` is a misnomer kept for compatibility: it means **bfloat16**, the
/// default storage across this backend. `Half` is genuine IEEE f16 — same two
/// bytes, but 10 mantissa bits against bfloat's 8, which is what the VAE
/// decoder needs to stop shedding precision at every store.
pub enum GpuDtype { F16, Half, F32 }

impl GpuDtype {
    pub fn bytes(self) -> usize {
        match self { GpuDtype::F16 | GpuDtype::Half => 2, GpuDtype::F32 => 4 }
    }
    /// True for the IEEE f16 storage mode.
    pub fn is_half(self) -> bool { matches!(self, GpuDtype::Half) }
}

/// A tensor whose data lives in an MTLBuffer.
pub struct GpuTensor {
    pub buffer: Buffer,
    pub shape: Vec<usize>,
    pub dtype: GpuDtype,
}

impl GpuTensor {
    pub fn elements(&self) -> usize { self.shape.iter().product() }

    /// Allocate a fresh fp16 buffer of the given shape (uninitialised contents).
    pub fn new_f16(shape: Vec<usize>) -> Self {
        let elements: usize = shape.iter().product();
        let bytes = elements * 2;
        Self { buffer: acquire_buffer(bytes), shape, dtype: GpuDtype::F16 }
    }

    /// Allocate a fresh fp32 buffer.
    pub fn new_f32(shape: Vec<usize>) -> Self {
        let elements: usize = shape.iter().product();
        let bytes = elements * 4;
        Self { buffer: acquire_buffer(bytes), shape, dtype: GpuDtype::F32 }
    }

    /// Upload an f32 slice to GPU as fp16 — the boundary entry point for
    /// the GPU-resident pipeline. Conversion uses SIMD-vectorized
    /// `convert_from_f32_slice` (NEON on M-series).
    pub fn upload_f32_as_f16(shape: Vec<usize>, data: &[f32]) -> Self {
        use half::bf16;
        use half::slice::HalfFloatSliceExt;
        let elements: usize = shape.iter().product();
        debug_assert_eq!(elements, data.len());
        let buffer = acquire_buffer(elements * 2);
        // Pool buffers may still be in flight from prior async cmds; flush
        // before CPU-writing into them.
        flush();
        unsafe {
            let dst: *mut bf16 = buffer.contents() as *mut bf16;
            let dst_slice: &mut [bf16] = std::slice::from_raw_parts_mut(dst, elements);
            dst_slice.convert_from_f32_slice(data);
        }
        Self { buffer, shape, dtype: GpuDtype::F16 }
    }

    /// Upload f32 data as f16 using a FRESH Metal buffer allocation
    /// (bypasses pool). Does NOT call `flush()` — safe to use within an
    /// open `begin_batch()` window without breaking the batch. The
    /// tradeoff: ~1µs allocation per call instead of pool reuse. Use
    /// this when batching multiple frames' uploads + dispatches within
    /// a single command buffer (the video forward path).
    pub fn upload_f32_as_f16_fresh(shape: Vec<usize>, data: &[f32]) -> Self {
        use half::bf16;
        use half::slice::HalfFloatSliceExt;
        let elements: usize = shape.iter().product();
        debug_assert_eq!(elements, data.len());
        let backend = get_metal();
        let buffer = backend.device.new_buffer(
            (elements * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let dst: *mut bf16 = buffer.contents() as *mut bf16;
            let dst_slice: &mut [bf16] = std::slice::from_raw_parts_mut(dst, elements);
            dst_slice.convert_from_f32_slice(data);
        }
        Self { buffer, shape, dtype: GpuDtype::F16 }
    }

    /// Upload via the f16 weight cache. Use for inputs that are **stable
    /// across calls** (same pointer + same contents) like SD's text-embedding
    /// conditioning — uploaded once, reused for every step and every
    /// Transformer2DModel inside the UNet. Cache hit returns a retain-bumped
    /// clone of the cached buffer; no CPU work, no flush, zero conversion.
    pub fn upload_f32_as_f16_cached(shape: Vec<usize>, data: &[f32]) -> Self {
        let elements: usize = shape.iter().product();
        debug_assert_eq!(elements, data.len());
        let buffer = convert_or_get_cached_f16(data);
        Self { buffer, shape, dtype: GpuDtype::F16 }
    }

    /// Download an fp16 GPU tensor to CPU as f32. Flushes async cmd queue
    /// first to ensure prior GPU work has completed.
    pub fn download_to_f32(&self) -> Vec<f32> {
        use half::bf16;
        use half::slice::HalfFloatSliceExt;
        flush();
        let elements = self.elements();
        let mut out = vec![0.0f32; elements];
        match self.dtype {
            GpuDtype::F16 => unsafe {
                let src: *const bf16 = self.buffer.contents() as *const bf16;
                let src_slice: &[bf16] = std::slice::from_raw_parts(src, elements);
                src_slice.convert_to_f32_slice(&mut out);
            },
            GpuDtype::Half => unsafe {
                let src: *const half::f16 = self.buffer.contents() as *const half::f16;
                let src_slice: &[half::f16] = std::slice::from_raw_parts(src, elements);
                src_slice.convert_to_f32_slice(&mut out);
            },
            GpuDtype::F32 => unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buffer.contents() as *const f32, out.as_mut_ptr(), elements);
            },
        }
        out
    }

    /// Allocate with the same storage type as `like`.
    ///
    /// Helpers that allocate their own output must not hard-code bf16, or a
    /// half-storage pipeline silently gets a bf16 buffer that later kernels
    /// read as half. The storage type is a runtime field, so nothing catches
    /// that at compile time.
    pub fn new_like(like: &GpuTensor, shape: Vec<usize>) -> Self {
        match like.dtype {
            GpuDtype::Half => Self::new_half(shape),
            GpuDtype::F32 => Self::new_f32(shape),
            GpuDtype::F16 => Self::new_f16(shape),
        }
    }

    /// Allocate IEEE-f16 storage.
    pub fn new_half(shape: Vec<usize>) -> Self {
        let elements: usize = shape.iter().product();
        Self { buffer: acquire_buffer(elements * 2), shape, dtype: GpuDtype::Half }
    }

    /// Upload f32 data as IEEE f16.
    pub fn upload_f32_as_half(shape: Vec<usize>, data: &[f32]) -> Self {
        use half::slice::HalfFloatSliceExt;
        let elements: usize = shape.iter().product();
        debug_assert_eq!(elements, data.len());
        let buffer = acquire_buffer(elements * 2);
        flush();
        unsafe {
            let dst: *mut half::f16 = buffer.contents() as *mut half::f16;
            let dst_slice: &mut [half::f16] = std::slice::from_raw_parts_mut(dst, elements);
            dst_slice.convert_from_f32_slice(data);
        }
        Self { buffer, shape, dtype: GpuDtype::Half }
    }

    /// Reshape (no data movement) — only the shape metadata changes.
    pub fn reshape(self, new_shape: Vec<usize>) -> Self {
        let new_elems: usize = new_shape.iter().product();
        debug_assert_eq!(new_elems, self.elements(), "reshape must preserve element count");
        Self { buffer: self.buffer, shape: new_shape, dtype: self.dtype }
    }
}

// ---------- Buffer-resident kernel dispatchers ----------------------------
//
// These take an MTLBuffer in/out, no CPU↔GPU copying. Used inside layer-level
// `forward_gpu` methods to chain ops on the GPU. Each dispatch still performs
// `wait_until_completed` for safety; future work could batch multiple ops
// into a single command buffer for further latency reduction.

#[inline]
fn dispatch_elementwise_buf_inplace(pipe: &ComputePipelineState, buf: &Buffer, len: usize) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(buf), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(len as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// In-place SiLU on an fp16 GPU buffer.
pub fn silu_f16_gpu(x: &mut GpuTensor) {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if x.dtype.is_half() { &get_metal().silu_h16_pipe } else { &get_metal().silu_f16_pipe };
    debug_assert!(matches!(x.dtype, GpuDtype::F16 | GpuDtype::Half));
    dispatch_elementwise_buf_inplace(pipe, &x.buffer, x.elements());
}

/// In-place GELU on an fp16 GPU buffer.
pub fn gelu_f16_gpu(x: &mut GpuTensor) {
    // bfloat or IEEE f16 storage; same kernel body either way.
    let pipe = if x.dtype.is_half() { &get_metal().gelu_f16_pipe_h } else { &get_metal().gelu_f16_pipe };
    debug_assert!(matches!(x.dtype, GpuDtype::F16 | GpuDtype::Half));
    dispatch_elementwise_buf_inplace(pipe, &x.buffer, x.elements());
}

/// In-place quick_gelu on an fp16 GPU buffer.
pub fn quick_gelu_f16_gpu(x: &mut GpuTensor) {
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    dispatch_elementwise_buf_inplace(&get_metal().quick_gelu_f16_pipe, &x.buffer, x.elements());
}

/// In-place a += b for fp16 GPU tensors of equal length.
pub fn eadd_f16_gpu(a: &mut GpuTensor, b: &GpuTensor) {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if a.dtype.is_half() { &get_metal().eadd_f16_h_pipe } else { &get_metal().eadd_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(a.dtype, GpuDtype::F16 | GpuDtype::Half));
    debug_assert!(matches!(b.dtype, GpuDtype::F16 | GpuDtype::Half));
    debug_assert_eq!(a.elements(), b.elements());
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&a.buffer), 0);
        enc.set_buffer(1, Some(&b.buffer), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(a.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// In-place element-wise multiply `a[i] *= b[i]` (fp16 tensors, equal length).
pub fn emul_f16_gpu(a: &mut GpuTensor, b: &GpuTensor) {
    use metal::MTLSize;
    debug_assert_eq!(a.elements(), b.elements());
    debug_assert_eq!(a.dtype.is_half(), b.dtype.is_half());
    // bfloat or IEEE f16 storage; same kernel body either way.
    let pipe = if a.dtype.is_half() { &get_metal().emul_h16_pipe } else { &get_metal().emul_f16_pipe };
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&a.buffer), 0);
        enc.set_buffer(1, Some(&b.buffer), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(a.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// `a[i] *= sigmoid(g[i])` — the per-channel output gate of Krea 2's attention.
pub fn sigmoid_gate_f16_gpu(a: &mut GpuTensor, g: &GpuTensor) {
    use metal::MTLSize;
    debug_assert_eq!(a.elements(), g.elements());
    debug_assert_eq!(a.dtype.is_half(), g.dtype.is_half());
    let pipe = if a.dtype.is_half() {
        &get_metal().sigmoid_gate_h16_pipe
    } else {
        &get_metal().sigmoid_gate_f16_pipe
    };
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&a.buffer), 0);
        enc.set_buffer(1, Some(&g.buffer), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(a.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// In-place RMSNorm over the last `dim` of each of `n_rows` rows, scaled by
/// the fp16 `weight` buffer `[dim]`. fp32 internal accumulation.
pub fn rms_norm_f16_gpu(x: &mut GpuTensor, weight: &Buffer, n_rows: usize, dim: usize, eps: f32) {
    // bfloat or IEEE f16 storage; same kernel body either way.
    let pipe = if x.dtype.is_half() { &get_metal().rms_norm_f16_pipe_h } else { &get_metal().rms_norm_f16_pipe };
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(weight), 0);
        let d = dim as u32;
        enc.set_bytes(2, 4, &d as *const _ as *const _);
        enc.set_bytes(3, 4, &eps as *const _ as *const _);
        let group = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n_rows as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// In-place RMSNorm over rows of `dim`, scaled by an **f32** `weight` buffer
/// (see [`upload_f32_buffer`]). Same math as [`rms_norm_f16_gpu`]; the weight
/// keeps full precision, and one threadgroup reduces each row.
pub fn rms_norm_wf32_f16_gpu(x: &mut GpuTensor, weight: &Buffer, n_rows: usize, dim: usize, eps: f32) {
    use metal::MTLSize;
    debug_assert!(matches!(x.dtype, GpuDtype::F16 | GpuDtype::Half));
    debug_assert_eq!(x.elements(), n_rows * dim);
    let pipe = if x.dtype.is_half() {
        &get_metal().rms_norm_wf32_h16_pipe
    } else {
        &get_metal().rms_norm_wf32_f16_pipe
    };
    let tpg = (dim.next_power_of_two()).clamp(32, 256) as u64;
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(weight), 0);
        let d = dim as u32;
        enc.set_bytes(2, 4, &d as *const _ as *const _);
        enc.set_bytes(3, 4, &eps as *const _ as *const _);
        enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });
}

/// In-place RoPE on `x[n, num_heads, head_dim]` using fp16 cos/sin buffers
/// `[n, head_dim]` (rotate_half convention).
pub fn rope_apply_f16_gpu(
    x: &mut GpuTensor, cos: &Buffer, sin: &Buffer, n: usize, num_heads: usize, head_dim: usize,
) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.rope_apply_f16_pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(cos), 0);
        enc.set_buffer(2, Some(sin), 0);
        let nh = num_heads as u32;
        let hd = head_dim as u32;
        enc.set_bytes(3, 4, &nh as *const _ as *const _);
        enc.set_bytes(4, 4, &hd as *const _ as *const _);
        let group = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new((n * num_heads) as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// Exact (erf) GELU in place on an fp16 buffer — Flux's GELU. (`gelu_f16_gpu`
/// is the tanh approximation; this matches the CPU `gelu_exact`.)
pub fn gelu_erf_f16_gpu(x: &mut GpuTensor) {
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    dispatch_elementwise_buf_inplace(&get_metal().gelu_erf_f16_pipe, &x.buffer, x.elements());
}

/// Interleaved-pair RoPE (Flux): rotate pairs `(2k, 2k+1)` of `x [n, nh, dh]`
/// using `cos`/`sin` `[n, dh]`. (`rope_apply_f16_gpu` is the cat-half variant.)
pub fn rope_pairs_f16_gpu(
    x: &mut GpuTensor, cos: &Buffer, sin: &Buffer, n: usize, num_heads: usize, head_dim: usize,
) {
    // bfloat or IEEE f16 storage; same kernel body either way.
    let pipe = if x.dtype.is_half() { &get_metal().rope_pairs_f16_pipe_h } else { &get_metal().rope_pairs_f16_pipe };
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(cos), 0);
        enc.set_buffer(2, Some(sin), 0);
        let nh = num_heads as u32;
        let hd = head_dim as u32;
        enc.set_bytes(3, 4, &nh as *const _ as *const _);
        enc.set_bytes(4, 4, &hd as *const _ as *const _);
        let group = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new((n * num_heads) as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// AdaLN modulate in place: `x[t,di] = x[t,di]·(1 + scale[di]) + shift[di]`,
/// broadcasting `scale`/`shift` `[d]` across `n` token rows. `scale_off` /
/// `shift_off` are element offsets into their buffers (for binding chunks of a
/// fused modulation output without copying). bf16 → 2 bytes/element.
#[allow(clippy::too_many_arguments)]
pub fn modulate_f16_gpu(
    x: &mut GpuTensor, scale: &Buffer, scale_off: usize, shift: &Buffer, shift_off: usize,
    n: usize, d: usize,
) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.modulate_f16_pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(scale), (scale_off * 2) as u64);
        enc.set_buffer(2, Some(shift), (shift_off * 2) as u64);
        let dd = d as u32;
        enc.set_bytes(3, 4, &dd as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((n * d) as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// Extract a contiguous `blk`-wide column block at column `off` from a
/// row-major `[n, stride]` fp16 tensor → `[n, blk]`.
pub fn slice_cols_f16_gpu(inp: &GpuTensor, n: usize, stride: usize, off: usize, blk: usize) -> GpuTensor {
    use metal::MTLSize;
    let backend = get_metal();
    let out = GpuTensor::new_f16(vec![n, blk]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.slice_cols_f16_pipe);
        enc.set_buffer(0, Some(&inp.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        let (s, o, b) = (stride as u32, off as u32, blk as u32);
        enc.set_bytes(2, 4, &s as *const _ as *const _);
        enc.set_bytes(3, 4, &o as *const _ as *const _);
        enc.set_bytes(4, 4, &b as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((n * blk) as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    out
}

/// Blit-copy `len` bf16 elements from `src[src_off..]` to `dst[dst_off..]`
/// (offsets and length in elements). Used to concatenate tensors along the
/// token axis (contiguous in `[n, h]` row-major layout).
pub fn blit_copy_f16(src: &Buffer, src_off: usize, dst: &Buffer, dst_off: usize, len: usize) {
    with_cmd(|cmd| {
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(
            src, (src_off * 2) as u64,
            dst, (dst_off * 2) as u64,
            (len * 2) as u64,
        );
        blit.end_encoding();
    });
}

/// Gate in place: `x[t,di] *= gate[di]`, broadcasting `gate` `[d]` across `n`.
/// `gate_off` is an element offset into `gate` (bf16, 2 bytes/element).
pub fn gate_mul_f16_gpu(x: &mut GpuTensor, gate: &Buffer, gate_off: usize, n: usize, d: usize) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.gate_mul_f16_pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(gate), (gate_off * 2) as u64);
        let dd = d as u32;
        enc.set_bytes(2, 4, &dd as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((n * d) as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// Grouped-query repeat: `inp[nkv, block]` → `out[nh, block]`, `out[h] =
/// inp[h/nrep]`. `block = n*head_dim`.
pub fn gqa_repeat_f16_gpu(inp: &GpuTensor, out: &mut GpuTensor, nrep: usize, block: usize) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.gqa_repeat_f16_pipe);
        enc.set_buffer(0, Some(&inp.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        let r = nrep as u32;
        let b = block as u32;
        enc.set_bytes(2, 4, &r as *const _ as *const _);
        enc.set_bytes(3, 4, &b as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(out.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// In-place `a += scale * b` for fp16 GPU tensors of equal length. Used
/// by the ControlNet residual injection so non-default `controlnet_weight`
/// values apply on the Metal residence path. When `scale == 1.0`, this
/// produces identical output to `eadd_f16_gpu` (single extra mul per
/// thread; negligible vs the cost of any matmul).
pub fn eadd_scaled_f16_gpu(a: &mut GpuTensor, b: &GpuTensor, scale: f32) {
    use metal::MTLSize;
    debug_assert_eq!(a.dtype, GpuDtype::F16);
    debug_assert_eq!(b.dtype, GpuDtype::F16);
    debug_assert_eq!(a.elements(), b.elements());
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.eadd_scaled_f16_pipe);
        enc.set_buffer(0, Some(&a.buffer), 0);
        enc.set_buffer(1, Some(&b.buffer), 0);
        enc.set_bytes(2, 4, &scale as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(a.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// In-place LayerNorm on an fp16 GPU buffer. `gamma`, `beta` are f16
/// GPU buffers (typically pre-uploaded weights from the cache).
pub fn layer_norm_f16_gpu(x: &mut GpuTensor, gamma: &Buffer, beta: &Buffer, dim: usize, eps: f32) {
    // bfloat or IEEE f16 storage; same kernel body either way.
    let pipe = if x.dtype.is_half() { &get_metal().layernorm_f16_pipe_h } else { &get_metal().layernorm_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(x.dtype, GpuDtype::F16 | GpuDtype::Half));
    debug_assert_eq!(x.elements() % dim, 0);
    let backend = get_metal();
    let n_rows = x.elements() / dim;
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(gamma), 0);
        enc.set_buffer(2, Some(beta), 0);
        let dim_u = dim as u32;
        enc.set_bytes(3, 4, &dim_u as *const _ as *const _);
        enc.set_bytes(4, 4, &eps as *const _ as *const _);
        let tpg = 256u64;
        enc.set_threadgroup_memory_length(0, (tpg as usize * 2 * 4) as u64);
        enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });
}

/// Out-of-place LayerNorm: reads from `src`, writes to a freshly-allocated
/// `dst` GpuTensor. Use in transformer blocks to avoid the
/// `clone_data + in-place LN` pattern (saves one blit per LN dispatch).
pub fn layer_norm_f16_gpu_out(
    src: &GpuTensor, gamma: &Buffer, beta: &Buffer, dim: usize, eps: f32,
) -> GpuTensor {
    use metal::MTLSize;
    debug_assert_eq!(src.dtype, GpuDtype::F16);
    debug_assert_eq!(src.elements() % dim, 0);
    let backend = get_metal();
    let n_rows = src.elements() / dim;
    let dst = GpuTensor::new_f16(src.shape.clone());
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.layernorm_f16_out_pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&dst.buffer), 0);
        enc.set_buffer(2, Some(gamma), 0);
        enc.set_buffer(3, Some(beta), 0);
        let dim_u = dim as u32;
        enc.set_bytes(4, 4, &dim_u as *const _ as *const _);
        enc.set_bytes(5, 4, &eps as *const _ as *const _);
        let tpg = 256u64;
        enc.set_threadgroup_memory_length(0, (tpg as usize * 2 * 4) as u64);
        enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });
    dst
}

/// In-place GroupNorm on an fp16 GPU buffer with shape `[N, C, H, W]`.
pub fn groupnorm_f16_gpu(
    x: &mut GpuTensor, gamma: &Buffer, beta: &Buffer,
    n: usize, c: usize, h: usize, w: usize,
    num_groups: usize, eps: f32,
) {
    use metal::MTLSize;
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    let backend = get_metal();
    let cg = c / num_groups;
    let hw = h * w;
    let group_size = cg * hw;
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.groupnorm_f16_pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(gamma), 0);
        enc.set_buffer(2, Some(beta), 0);
        let group_size_u = group_size as u32;
        let cg_u = cg as u32;
        let hw_u = hw as u32;
        let ng_u = num_groups as u32;
        enc.set_bytes(3, 4, &group_size_u as *const _ as *const _);
        enc.set_bytes(4, 4, &cg_u as *const _ as *const _);
        enc.set_bytes(5, 4, &hw_u as *const _ as *const _);
        enc.set_bytes(6, 4, &ng_u as *const _ as *const _);
        enc.set_bytes(7, 4, &eps as *const _ as *const _);
        let tpg = 256u64;
        enc.set_threadgroup_memory_length(0, (tpg as usize * 2 * 4) as u64);
        let groups = (n * num_groups) as u64;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });
}

/// In-place fused GroupNorm + SiLU. Replaces `groupnorm_inplace + silu_inplace`
/// (2 dispatches → 1) on a buffer that doesn't need to be preserved.
/// Uses the dedicated `groupnorm_silu_f16_inplace` kernel (single buffer
/// argument) — never bind the same buffer to two `device` slots, that's
/// MSL undefined behavior.
pub fn groupnorm_silu_f16_gpu_inplace(
    x: &mut GpuTensor, gamma: &Buffer, beta: &Buffer,
    n: usize, c: usize, h: usize, w: usize,
    num_groups: usize, eps: f32,
) {
    use metal::MTLSize;
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    let backend = get_metal();
    let cg = c / num_groups;
    let hw = h * w;
    let group_size = cg * hw;
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.groupnorm_silu_f16_inplace_pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(gamma), 0);
        enc.set_buffer(2, Some(beta), 0);
        let group_size_u = group_size as u32;
        let cg_u = cg as u32;
        let hw_u = hw as u32;
        let ng_u = num_groups as u32;
        enc.set_bytes(3, 4, &group_size_u as *const _ as *const _);
        enc.set_bytes(4, 4, &cg_u as *const _ as *const _);
        enc.set_bytes(5, 4, &hw_u as *const _ as *const _);
        enc.set_bytes(6, 4, &ng_u as *const _ as *const _);
        enc.set_bytes(7, 4, &eps as *const _ as *const _);
        let tpg = 256u64;
        enc.set_threadgroup_memory_length(0, (tpg as usize * 2 * 4) as u64);
        let groups = (n * num_groups) as u64;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });
}

/// Out-of-place fused GroupNorm + SiLU. Reads from `src`, writes to a fresh
/// dst, applies SiLU in the same kernel. Replaces the resnet block's
/// `clone + groupnorm + silu` three-dispatch pattern with one dispatch.
pub fn groupnorm_silu_f16_gpu_out(
    src: &GpuTensor, gamma: &Buffer, beta: &Buffer,
    n: usize, c: usize, h: usize, w: usize,
    num_groups: usize, eps: f32,
) -> GpuTensor {
    use metal::MTLSize;
    debug_assert_eq!(src.dtype, GpuDtype::F16);
    let backend = get_metal();
    let cg = c / num_groups;
    let hw = h * w;
    let group_size = cg * hw;
    let dst = GpuTensor::new_f16(src.shape.clone());
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.groupnorm_silu_f16_out_pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&dst.buffer), 0);
        enc.set_buffer(2, Some(gamma), 0);
        enc.set_buffer(3, Some(beta), 0);
        let group_size_u = group_size as u32;
        let cg_u = cg as u32;
        let hw_u = hw as u32;
        let ng_u = num_groups as u32;
        enc.set_bytes(4, 4, &group_size_u as *const _ as *const _);
        enc.set_bytes(5, 4, &cg_u as *const _ as *const _);
        enc.set_bytes(6, 4, &hw_u as *const _ as *const _);
        enc.set_bytes(7, 4, &ng_u as *const _ as *const _);
        enc.set_bytes(8, 4, &eps as *const _ as *const _);
        let tpg = 256u64;
        enc.set_threadgroup_memory_length(0, (tpg as usize * 2 * 4) as u64);
        let groups = (n * num_groups) as u64;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });
    dst
}

/// f16 sgemm where all three buffers are already on the GPU.
/// Storage f16, compute fp32 (MPS doesn't support f16 sgemm directly).
pub fn sgemm_f16_gpu(c: &mut GpuTensor, a: &GpuTensor, b: &GpuTensor,
                     m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.dtype, GpuDtype::F16);
    debug_assert_eq!(b.dtype, GpuDtype::F16);
    debug_assert_eq!(c.dtype, GpuDtype::F16);
    let backend = get_metal();
    unsafe {
        dispatch_mps_matmul_bf16_via_f32(
            backend, &a.buffer, &b.buffer, &c.buffer,
            m, n, k, false, false, k, n,
        );
    }
}

/// f16 sgemm with B transposed (`C = A · Bᵀ`), all GPU-resident.
pub fn sgemm_f16_a_btrans_gpu(c: &mut GpuTensor, a: &GpuTensor, b: &GpuTensor,
                              m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.dtype, GpuDtype::F16);
    debug_assert_eq!(b.dtype, GpuDtype::F16);
    debug_assert_eq!(c.dtype, GpuDtype::F16);
    let backend = get_metal();
    unsafe {
        dispatch_mps_matmul_bf16_via_f32(
            backend, &a.buffer, &b.buffer, &c.buffer,
            m, n, k, false, true, n, k,
        );
    }
}

/// Fetch (or upload) an f16 GPU buffer view of a CPU f32 weight slice.
/// Returns a `Buffer` whose lifetime is tied to the F16 weight cache —
/// callers should NOT release it back to the pool.
pub fn weight_f16_buffer(slice: &[f32]) -> Buffer {
    convert_or_get_cached_f16(slice)
}

/// In-place bias add: `x[i] += bias[(i / period) % bias_len]`.
/// For Linear output `[n_rows, out]`: `period=1, bias_len=out`.
/// For Conv2d output `[N, C, HW]`: `period=HW, bias_len=C`.
pub fn bias_add_f16_gpu(x: &mut GpuTensor, bias: &Buffer, period: usize, bias_len: usize) {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if x.dtype.is_half() { &get_metal().bias_add_strided_h_pipe } else { &get_metal().bias_add_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(x.dtype, GpuDtype::F16 | GpuDtype::Half));
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(bias), 0);
        let p = period as u32;
        let bl = bias_len as u32;
        enc.set_bytes(2, 4, &p as *const _ as *const _);
        enc.set_bytes(3, 4, &bl as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(x.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// f16 sgemm with `Buffer` arguments directly (skipping `GpuTensor` wrap).
/// Storage is f16, computed via fp32 (MPS doesn't support f16 sgemm).
/// Used by `Linear::forward_gpu` so the cached weight `Buffer` from
/// `weight_f16_buffer` can be passed as B without re-wrapping.
/// `C[m,n] = A[m,k] * B[n,k]^T` with a native bf16 simdgroup-MMA kernel.
///
/// bfloat in, float accumulate, bfloat out — no cast to fp32 (which caps at
/// ~2.2 TFLOP/s on M1 Max) and no cast to IEEE fp16 (which loses accumulation
/// precision and clips at 65504). Nothing is staged through the buffer pool,
/// so this also does one dispatch where the MPS route did four.
///
/// **Correct, and currently 10x too slow to use.** Against MPS f32 on a full
/// Wan forward at 5070 tokens, on an idle machine, 3 runs: cos_sim
/// 1.00000000 and max_abs_diff exactly 0.0 (both sides accumulate in fp32 and
/// round to bf16, so they agree bit-for-bit) — but 267 s against 27 s, about
/// 0.17 TFLOP/s, under 2% of what simdgroup MMA should reach.
///
/// Two rounds of tuning did not move it (274 s -> 267 s): staging B
/// pre-transposed so the inner loop drops its transposing `simdgroup_load`,
/// and halving threadgroup memory to 8 KB by writing C out in half-tiles. The
/// remaining suspect is the K loop's barrier density — GK=32 gives only 4 MMA
/// steps per staging round, so K=3072 pays ~192 threadgroup barriers per
/// output tile. Deepening GK to 64 (16 KB threadgroup) and double-buffering
/// the staging is the next thing to try.
///
/// Left behind `set_use_bf16_gemm` / `KLEARU_WAN_BF16_GEMM=1` rather than
/// removed: the numerics are exactly right, which is the hard part, and it is
/// the only path that escapes the MPS f32 ceiling.
pub fn sgemm_bf16_btrans_gpu(out: &Buffer, a: &Buffer, b: &Buffer,
                             m: usize, n: usize, k: usize) {
    use metal::MTLSize;
    let backend = get_metal();
    let pipe = &backend.sgemm_bf16_btrans_pipe;
    let (gm, gn) = (64usize, 64usize);
    let (nu, ku) = (n as u32, k as u32);
    // Chunk by rows so no single dispatch runs long enough for the GPU
    // watchdog to kill it (see `MMA_MAX_ROWS`).
    let mut start = 0usize;
    while start < m {
        let rows = MMA_MAX_ROWS.min(m - start);
        let mu = rows as u32;
        with_cmd(|cmd| {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(a), (start * k * 2) as u64);
            enc.set_buffer(1, Some(b), 0);
            enc.set_buffer(2, Some(out), (start * n * 2) as u64);
            enc.set_bytes(3, 4, &mu as *const _ as *const _);
            enc.set_bytes(4, 4, &nu as *const _ as *const _);
            enc.set_bytes(5, 4, &ku as *const _ as *const _);
            // A tile (64x32) + B tile (32x64) as bfloat = 8 KB; the C write-out
            // half-tile (32x64 float) aliases them at 8 KB too.
            enc.set_threadgroup_memory_length(0, (gm * gn * 2) as u64);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(gn) as u64, rows.div_ceil(gm) as u64, 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
        });
        start += rows;
    }
}

/// Largest M the native MMA kernel does in one dispatch.
///
/// Not a correctness limit — a duration one. macOS kills any command buffer
/// that holds the GPU long enough to stall the display, with
/// `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`, and a killed
/// buffer leaves its output *partially written* rather than failing loudly.
/// This kernel runs at ~0.17 TF/s, so m=11440 x k=3072 x n=3072 is ~0.6 s on
/// an idle GPU and several seconds when anything else is drawing — right in
/// the kill window. 2048 rows keeps each dispatch around 0.1 s.
const MMA_MAX_ROWS: usize = 2048;

/// Largest M handed to MPS in one call.
///
/// This began as a correctness workaround and is now a duration guard. The
/// symptom that motivated it — `chunked_sgemm_f16_buf_matches_reference`
/// reading cos_sim 0.24 at m=8192 and 1.00000000 on the very next identical
/// call — was never an MPS numerical bug. It was macOS killing the command
/// buffer with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` and
/// leaving the output partially written; `flush` now checks for that instead of
/// letting the zeros through. Keeping M bounded still shortens each buffer,
/// which is worth the one extra MPS call per matmul.
const MPS_MAX_ROWS: usize = 4096;

/// `C[m,n] = A[m,k] * B[n,k]^T` — every `Linear` in the model.
///
/// Chunked at `MPS_MAX_ROWS`; see there for why. The native bf16 MMA kernel
/// (`sgemm_bf16_btrans_gpu`) is available behind `set_use_bf16_gemm` and is
/// bit-exact, but ~10x slower than MPS, so it serves as the independent
/// reference in tests rather than as the production path.
///
/// Two distinct failures were long conflated here, and both are now explained.
/// The *non-finite* outputs at particular token counts (9360 fully NaN, 11440
/// corrupt in its trailing 152 rows) were memory pressure on a box with no
/// swap, not a shape-dependent MPS bug. The *finite but wrong* outputs — whole
/// regions of exact zeros, a different M each run — were command buffers killed
/// by the macOS GPU watchdog. Neither was MPS miscomputing, which is why no
/// rule over M was ever found: there wasn't one to find.
pub fn sgemm_f16_a_btrans_buf(out: &Buffer, a: &Buffer, b: &Buffer,
                              m: usize, n: usize, k: usize) {
    if USE_BF16_GEMM.with(|c| c.get()) {
        sgemm_bf16_btrans_gpu(out, a, b, m, n, k);
        return;
    }
    let backend = get_metal();

    // Above MPS_MAX_ROWS, issue a series of calls that are EACH exactly
    // MPS_MAX_ROWS tall, with the last one aligned to the bottom of the matrix
    // rather than left ragged. Overlapping rows are simply recomputed to the
    // same values, so the overlap is free of consequences.
    //
    // The raggedness is the actual trigger. 11440 as 8192+3248 was clean, but
    // 9360 as 8192+1168 corrupted 152 rows of the small chunk, and equal chunks
    // of 4680 were worse still — while an unchunked 5070 is fine. Holding every
    // call at one known-good height sidesteps having to know which heights are
    // bad, and keeps MPS's speed: the bf16 MMA fallback is ~10x slower, which
    // at native resolution is 676 s per forward against roughly 60 s.
    if m > MPS_MAX_ROWS {
        let rows = MPS_MAX_ROWS;
        let mut start = 0usize;
        loop {
            let last = start + rows >= m;
            let s0 = if last { m - rows } else { start };
            unsafe {
                dispatch_mps_matmul_bf16_via_f32_with_offsets(
                    backend,
                    a, s0 * k * 2,
                    b, 0,
                    out, s0 * n * 2,
                    rows, n, k,
                    false, true, n, k,
                );
            }
            if last { break; }
            start += rows;
        }
        return;
    }
    unsafe {
        dispatch_mps_matmul_bf16_via_f32(
            backend, a, b, out, m, n, k,
            false, true, n, k,
        );
    }
}

/// `C[m,n] = A[m,k] * B[k,n]`, B already in the orientation MPS wants.
///
/// Chunked above `MPS_MAX_ROWS` for the same reason as
/// [`sgemm_f16_a_btrans_buf`]: MPS returns NaN for some large-M shapes. This
/// path needed it the moment the DiT's `Linear`s moved onto it — 11440 rows
/// went non-finite 6 times out of 6 until the chunking was mirrored here.
pub fn sgemm_f16_buf(out: &Buffer, a: &Buffer, b: &Buffer,
                     m: usize, n: usize, k: usize) {
    let backend = get_metal();
    if m > MPS_MAX_ROWS {
        let rows = MPS_MAX_ROWS;
        let mut start = 0usize;
        loop {
            let last = start + rows >= m;
            let s0 = if last { m - rows } else { start };
            unsafe {
                dispatch_mps_matmul_bf16_via_f32_with_offsets(
                    backend,
                    a, s0 * k * 2,
                    b, 0,
                    out, s0 * n * 2,
                    rows, n, k,
                    false, false, k, n,
                );
            }
            if last { break; }
            start += rows;
        }
        return;
    }
    unsafe {
        dispatch_mps_matmul_bf16_via_f32(
            backend, a, b, out, m, n, k,
            false, false, k, n,
        );
    }
}

/// im2col on GPU: input fp16 buffer `[C_in × H_in × W_in]` (single batch)
/// → col fp16 buffer `[C_in*kH*kW × H_out*W_out]`. Used by `Conv2d::forward_gpu`.
pub fn im2col_f16_gpu(
    input_buf: &Buffer, col_buf: &Buffer,
    c_in: usize, h_in: usize, w_in: usize,
    kh: usize, kw: usize, stride: usize, pad: usize,
    h_out: usize, w_out: usize,
) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.im2col_f16_pipe);
        enc.set_buffer(0, Some(input_buf), 0);
        enc.set_buffer(1, Some(col_buf), 0);
        let c_in_u = c_in as u32;
        let h_in_u = h_in as u32;
        let w_in_u = w_in as u32;
        let kh_u = kh as u32;
        let kw_u = kw as u32;
        let stride_u = stride as u32;
        let pad_u = pad as u32;
        let h_out_u = h_out as u32;
        let w_out_u = w_out as u32;
        enc.set_bytes(2, 4, &c_in_u as *const _ as *const _);
        enc.set_bytes(3, 4, &h_in_u as *const _ as *const _);
        enc.set_bytes(4, 4, &w_in_u as *const _ as *const _);
        enc.set_bytes(5, 4, &kh_u as *const _ as *const _);
        enc.set_bytes(6, 4, &kw_u as *const _ as *const _);
        enc.set_bytes(7, 4, &stride_u as *const _ as *const _);
        enc.set_bytes(8, 4, &pad_u as *const _ as *const _);
        enc.set_bytes(9, 4, &h_out_u as *const _ as *const _);
        enc.set_bytes(10, 4, &w_out_u as *const _ as *const _);

        let hw_out = (h_out * w_out) as u64;
        let kk = (c_in * kh * kw) as u64;
        let group = MTLSize::new(16, 16, 1);
        let grid = MTLSize::new(hw_out, kk, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// im2col on GPU for a single batch element identified by `batch_idx`.
/// `input_buf` is the full `[N, C_in, H_in, W_in]` fp16 buffer; we offset
/// into it by `batch_idx * C_in * H_in * W_in` elements (× 2 bytes).
pub fn im2col_per_batch_dispatch(
    input_buf: &Buffer, batch_idx: usize, hw_in_per_batch: usize,
    col_buf: &Buffer,
    c_in: usize, h_in: usize, w_in: usize,
    kh: usize, kw: usize, stride: usize, pad: usize,
    h_out: usize, w_out: usize,
) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.im2col_f16_pipe);

        let input_offset_bytes = (batch_idx * c_in * hw_in_per_batch * 2) as u64;
        enc.set_buffer(0, Some(input_buf), input_offset_bytes);
        enc.set_buffer(1, Some(col_buf), 0);

        let c_in_u = c_in as u32;
        let h_in_u = h_in as u32;
        let w_in_u = w_in as u32;
        let kh_u = kh as u32;
        let kw_u = kw as u32;
        let stride_u = stride as u32;
        let pad_u = pad as u32;
        let h_out_u = h_out as u32;
        let w_out_u = w_out as u32;
        enc.set_bytes(2, 4, &c_in_u as *const _ as *const _);
        enc.set_bytes(3, 4, &h_in_u as *const _ as *const _);
        enc.set_bytes(4, 4, &w_in_u as *const _ as *const _);
        enc.set_bytes(5, 4, &kh_u as *const _ as *const _);
        enc.set_bytes(6, 4, &kw_u as *const _ as *const _);
        enc.set_bytes(7, 4, &stride_u as *const _ as *const _);
        enc.set_bytes(8, 4, &pad_u as *const _ as *const _);
        enc.set_bytes(9, 4, &h_out_u as *const _ as *const _);
        enc.set_bytes(10, 4, &w_out_u as *const _ as *const _);

        let hw_out = (h_out * w_out) as u64;
        let kk = (c_in * kh * kw) as u64;
        let group = MTLSize::new(16, 16, 1);
        let grid = MTLSize::new(hw_out, kk, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// MPSGraph-based 2D convolution. Replaces the per-batch im2col + MPS
/// matmul path with Apple's optimised conv kernel. Storage stays bf16
/// across layers; we cast bf16↔f16 around the MPSGraph call (MPSGraph
/// supports f16 natively but not bf16).
///
/// Graphs are cached per (weight_ptr, input shape, conv params). Per call
/// we wrap input/output buffers as MPSGraphTensorData and encode into the
/// shared command buffer.
///
/// Returns (output_buffer, h_out, w_out). Output is freshly allocated bf16.
/// Hand out a process-unique weight identity.
///
/// Used as the key for uploaded-weight caches. A pointer cannot serve: freed
/// allocations hand their address to the next one, and the cache then serves
/// the wrong weights to a same-shaped convolution.
pub fn next_weight_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Drop the uploaded copies of a weight tensor.
///
/// The flip side of `next_weight_id`: ids are unique per conv object, so a
/// caller that builds and drops convs (the VAE encoder builds one per encode)
/// would otherwise grow the upload caches without bound — hundreds of MB per
/// rebuild, never reclaimed. Called from `GpuCausalConv3d::drop`.
pub fn release_weight_id(id: u64) {
    let k = id as usize;
    MPSGRAPH_WEIGHT_HALF_CACHE.with_borrow_mut(|c| c.retain(|(i, _), _| *i != k));
    MPSGRAPH_WEIGHT_F32_CACHE.with_borrow_mut(|c| c.retain(|(i, _), _| *i != k));
    MPSGRAPH_WEIGHT_F16_CACHE.with_borrow_mut(|c| c.retain(|(i, _), _| *i != k));
}

pub fn mpsgraph_conv2d(
    input_bf16: &Buffer,
    weight_f32: &[f32],
    // Stable identity for `weight_f32`, from `next_weight_id`: constant for
    // the life of a weight tensor, never reused for a different one.
    weight_id: u64,
    n: usize, in_c: usize, h_in: usize, w_in: usize,
    out_c: usize, kh: usize, kw: usize,
    stride: usize, pad: usize,
) -> (Buffer, usize, usize) {
    // Drain per call. This path allocates several autoreleased MPSGraph
    // objects per convolution (the MPSGraphTensorData wrappers, plus anything
    // an evicted graph releases), and each holds a reference to its MTLBuffer.
    // With no pool they accumulate for the life of the process: the video
    // decoder leaked ~0.4 GB per frame at 128x192 and ~15 GB per frame at
    // 704x1280, which is what made full-length native decoding impossible.
    // `mpsgraph_sdpa_bf16_gpu` already wraps its graph work the same way.
    objc2::rc::autoreleasepool(|_| {
        mpsgraph_conv2d_inner(
            input_bf16, weight_f32, weight_id, n, in_c, h_in, w_in, out_c, kh, kw,
            stride, pad, false,
        )
    })
}

/// As [`mpsgraph_conv2d`], but the input and returned buffers hold IEEE f16
/// rather than bfloat16.
///
/// This is both more accurate and cheaper: MPSGraph already runs the
/// convolution in f16, so with half storage the bf16->f16 and f16->bf16 casts
/// around it disappear, and with them two roundings per convolution.
pub fn mpsgraph_conv2d_half(
    input_half: &Buffer,
    weight_f32: &[f32],
    // Stable identity for `weight_f32`, from `next_weight_id`: constant for
    // the life of a weight tensor, never reused for a different one.
    weight_id: u64,
    n: usize, in_c: usize, h_in: usize, w_in: usize,
    out_c: usize, kh: usize, kw: usize,
    stride: usize, pad: usize,
) -> (Buffer, usize, usize) {
    objc2::rc::autoreleasepool(|_| {
        mpsgraph_conv2d_inner(
            input_half, weight_f32, weight_id, n, in_c, h_in, w_in, out_c, kh, kw,
            stride, pad, true,
        )
    })
}

fn mpsgraph_conv2d_inner(
    input_bf16: &Buffer,
    weight_f32: &[f32],
    // Stable identity for `weight_f32`, from `next_weight_id`: constant for
    // the life of a weight tensor, never reused for a different one.
    weight_id: u64,
    n: usize, in_c: usize, h_in: usize, w_in: usize,
    out_c: usize, kh: usize, kw: usize,
    stride: usize, pad: usize,
    half_io: bool,
) -> (Buffer, usize, usize) {
    let h_out = (h_in + 2 * pad - kh) / stride + 1;
    let w_out = (w_in + 2 * pad - kw) / stride + 1;

    // Half storage is already the format MPSGraph wants, so the casts drop out.
    let f32_conv = !half_io && USE_F32_CONV.with(|c| c.get());
    let esz = if f32_conv { 4 } else { 2 };
    // Cast input bf16 → f16 (or f32) scratch.
    let input_elems = n * in_c * h_in * w_in;
    // `KLEARU_CONV_MAG=1` reports the peak magnitude entering the conv. The
    // cast below goes to IEEE f16, which saturates at 65504 while the bf16
    // input carries f32's exponent range — so anything above that becomes Inf
    // here and there is no other sign of it.
    if std::env::var_os("KLEARU_CONV_MAG").is_some() {
        // Read the bf16 buffer directly. Wrapping it in a GpuTensor would need
        // a retained clone and a matching leak to avoid a double free, which is
        // not worth it for a diagnostic.
        flush();
        let mut mx = 0.0f32;
        let mut over = 0usize;
        unsafe {
            let p = input_bf16.contents() as *const half::bf16;
            for i in 0..input_elems {
                let v = f32::from(*p.add(i));
                if v.is_finite() {
                    mx = mx.max(v.abs());
                    if v.abs() > 65504.0 {
                        over += 1;
                    }
                }
            }
        }
        if mx > 1000.0 || over > 0 {
            println!("[conv] in_c={in_c} {h_in}x{w_in}: max|in|={mx:.4e} over_f16_max={over}");
        }
    }
    let input_f16 = if half_io {
        input_bf16.clone()
    } else {
        let b = acquire_buffer(input_elems * esz);
        if f32_conv {
            cast_bf16_to_f32_buf(input_bf16, &b, input_elems);
        } else {
            cast_bf16_to_half_buf(input_bf16, &b, input_elems);
        }
        b
    };

    // Weight: bf16 cache → f16 cache. We use the bf16 weight cache as the
    // canonical storage and cast to f16 once (cached per weight slice).
    let weight_bf16 = weight_f16_buffer(weight_f32);
    let weight_elems = out_c * in_c * kh * kw;
    let weight_f16 = if half_io {
        // Upload from f32 straight to half. Going via the bf16 cache would
        // throw away two mantissa bits before the convolution even starts.
        mpsgraph_weight_half_cached(weight_f32, weight_id)
    } else if f32_conv {
        // Upload the f32 weights directly rather than round-tripping through
        // bf16 — the whole point of this path is to stop discarding mantissa.
        mpsgraph_weight_f32_cached(weight_f32, weight_id)
    } else {
        mpsgraph_weight_f16_cached(&weight_bf16, weight_id, weight_elems)
    };

    // Output buffer.
    let out_elems = n * out_c * h_out * w_out;
    let output_f16 = acquire_buffer(out_elems * esz);

    // Shapes only: weights enter the graph as a placeholder, fed per call.
    let key = ConvGraphKey {
        n, in_c, h_in, w_in, out_c, kh, kw, stride, pad,
        f32_compute: f32_conv,
    };

    unsafe {
        encode_mpsgraph_conv(key, &input_f16, &weight_f16, &output_f16, h_out, w_out);
    }

    if half_io {
        // The graph already wrote IEEE f16; hand it straight back.
        return (output_f16, h_out, w_out);
    }

    // Cast output f16 → bf16.
    let output_bf16 = acquire_buffer(out_elems * 2);
    if f32_conv {
        cast_f32_to_bf16_buf(&output_f16, &output_bf16, out_elems);
    } else {
        cast_half_to_bf16_buf(&output_f16, &output_bf16, out_elems);
    }

    release_buffer(input_f16);
    release_buffer(output_f16);

    (output_bf16, h_out, w_out)
}

thread_local! {
    static MPSGRAPH_WEIGHT_F16_CACHE: std::cell::RefCell<HashMap<(usize, usize), Buffer>>
        = std::cell::RefCell::new(HashMap::new());
}

thread_local! {
    static MPSGRAPH_WEIGHT_F32_CACHE: std::cell::RefCell<HashMap<(usize, usize), Buffer>>
        = std::cell::RefCell::new(HashMap::new());
}

/// Weights uploaded as IEEE f16 for the half convolution path.
fn mpsgraph_weight_half_cached(weight_f32: &[f32], weight_id: u64) -> Buffer {
    use half::slice::HalfFloatSliceExt;
    // Keyed by an explicit id, never by the slice's address.
    //
    // An address is not an identity: a freed weight `Vec` releases it, and the
    // next allocation of the same size can land there. Keying on the pointer
    // then returns a buffer holding *different weights* — the convolution runs
    // happily and produces confident nonsense. Adding the length to the key is
    // not enough, because the collision that bites is precisely a same-shaped
    // conv reusing a freed address. Callers that keep their weights alive for
    // the process (the decoder) never noticed; anything that rebuilds conv
    // objects, like the encoder, hit it immediately.
    let key = (weight_id as usize, weight_f32.len());
    MPSGRAPH_WEIGHT_HALF_CACHE.with_borrow_mut(|cache| {
        if let Some(b) = cache.get(&key) {
            return b.clone();
        }
        let buf = acquire_buffer(weight_f32.len() * 2);
        flush();
        unsafe {
            let dst: *mut half::f16 = buf.contents() as *mut half::f16;
            let dst_slice: &mut [half::f16] =
                std::slice::from_raw_parts_mut(dst, weight_f32.len());
            dst_slice.convert_from_f32_slice(weight_f32);
        }
        let h = buf.clone();
        cache.insert(key, buf);
        h
    })
}

thread_local! {
    static MPSGRAPH_WEIGHT_HALF_CACHE: std::cell::RefCell<HashMap<(usize, usize), Buffer>>
        = std::cell::RefCell::new(HashMap::new());
}

/// Weights uploaded as f32 for the f32 convolution path, cached for the process
/// lifetime and keyed by weight id like its f16 counterpart.
fn mpsgraph_weight_f32_cached(weight_f32: &[f32], weight_id: u64) -> Buffer {
    let key = (weight_id as usize, weight_f32.len());
    MPSGRAPH_WEIGHT_F32_CACHE.with_borrow_mut(|cache| {
        if let Some(b) = cache.get(&key) {
            return b.clone();
        }
        let bytes = weight_f32.len() * 4;
        let buf = acquire_buffer(bytes);
        flush();
        unsafe {
            std::ptr::copy_nonoverlapping(
                weight_f32.as_ptr(),
                buf.contents() as *mut f32,
                weight_f32.len(),
            );
        }
        let h = buf.clone();
        cache.insert(key, buf);
        h
    })
}

/// One-time conversion of a bf16 weight buffer to a f16 buffer that lives
/// for the process lifetime. Keyed by source CPU pointer.
fn mpsgraph_weight_f16_cached(bf16_buf: &Buffer, weight_id: u64, n_elements: usize) -> Buffer {
    // See `mpsgraph_weight_half_cached`: keyed by id, not address.
    let key = (weight_id as usize, n_elements);
    MPSGRAPH_WEIGHT_F16_CACHE.with_borrow_mut(|cache| {
        if let Some(b) = cache.get(&key) {
            return b.clone();
        }
        let f16 = acquire_buffer(n_elements * 2);
        cast_bf16_to_half_buf(bf16_buf, &f16, n_elements);
        // Block here once to ensure cast completes before first use. Subsequent
        // hits never block.
        flush();
        let h = f16.clone();
        cache.insert(key, f16);
        h
    })
}

unsafe fn encode_mpsgraph_conv(
    key: ConvGraphKey,
    input_buf: &Buffer,
    weight_buf: &Buffer,
    output_buf: &Buffer,
    h_out: usize,
    w_out: usize,
) {
    use metal::foreign_types::ForeignType;

    let this_gen = CONV_GRAPH_GEN.with(|c| c.get());
    let cached = CONV_GRAPH_CACHE.with_borrow_mut(|cache| {
        if let Some(c) = cache.get_mut(&key) {
            // Existing cached pointers are valid (objc retain count maintained).
            // Stamp the generation so `conv_graph_end_generation` keeps it.
            c.gen = this_gen;
            (c.graph, c.input_tensor, c.weight_tensor, c.output_tensor)
        } else {
            // Cache miss — build the graph. Catch exceptions so we report
            // the actual NSException reason rather than the opaque "Rust
            // cannot catch foreign exceptions" abort.
            if std::env::var_os("KLEARU_CONV_KEYS").is_some() {
                println!(
                    "[convkey] n={} in_c={} h={} w={} out_c={} k={}x{} s={} p={}",
                    key.n, key.in_c, key.h_in, key.w_in,
                    key.out_c, key.kh, key.kw, key.stride, key.pad
                );
            }
            let built = match objc2::exception::catch(std::panic::AssertUnwindSafe(|| {
                build_conv_graph(key, h_out, w_out)
            })) {
                Ok(g) => g,
                Err(exc) => {
                    if let Some(exc) = exc {
                        let reason: *mut AnyObject = msg_send![exc.as_ref() as *const _ as *mut AnyObject, reason];
                        let utf8: *const std::os::raw::c_char = msg_send![reason, UTF8String];
                        if !utf8.is_null() {
                            let cstr = std::ffi::CStr::from_ptr(utf8);
                            eprintln!("MPSGraph build_conv_graph threw: {}", cstr.to_string_lossy());
                            eprintln!("  shape: n={} in_c={} h_in={} w_in={} out_c={} kh={} kw={} stride={} pad={}",
                                key.n, key.in_c, key.h_in, key.w_in, key.out_c, key.kh, key.kw, key.stride, key.pad);
                        }
                    }
                    panic!("MPSGraph build failed");
                }
            };
            let result = (built.graph, built.input_tensor, built.weight_tensor, built.output_tensor);
            let mut built = built;
            built.gen = this_gen;
            cache.insert(key, built);
            result
        }
    });
    let (graph, input_tensor, weight_tensor, output_tensor) = cached;

    // Wrap input and weight buffers as MPSGraphTensorData. Wrap output too —
    // MPSGraph writes directly into our pre-allocated output buffer.
    let nsnumber_cls = AnyClass::get("NSNumber").unwrap();
    let make_shape = |dims: &[usize]| -> *mut AnyObject {
        let arr_cls = AnyClass::get("NSMutableArray").unwrap();
        let arr: *mut AnyObject = msg_send![arr_cls, arrayWithCapacity: dims.len()];
        for &d in dims {
            let n: *mut AnyObject = msg_send![nsnumber_cls, numberWithUnsignedLongLong: d as u64];
            let _: () = msg_send![arr, addObject: n];
        }
        arr
    };

    let input_shape = make_shape(&[key.n, key.in_c, key.h_in, key.w_in]);
    let weight_shape = make_shape(&[key.out_c, key.in_c, key.kh, key.kw]);
    let output_shape = make_shape(&[key.n, key.out_c, h_out, w_out]);

    let tensor_data_cls = AnyClass::get("MPSGraphTensorData").unwrap();
    let input_buf_ptr: *mut AnyObject = ForeignType::as_ptr(input_buf) as *mut AnyObject;
    let weight_buf_ptr: *mut AnyObject = ForeignType::as_ptr(weight_buf) as *mut AnyObject;
    let output_buf_ptr: *mut AnyObject = ForeignType::as_ptr(output_buf) as *mut AnyObject;

    let dt = if key.f32_compute { MPS_DATA_TYPE_FLOAT32 } else { MPS_DATA_TYPE_FLOAT16 };
    let input_data: *mut AnyObject = msg_send![tensor_data_cls, alloc];
    let input_data: *mut AnyObject = msg_send![input_data,
        initWithMTLBuffer: input_buf_ptr
        shape: input_shape
        dataType: dt];

    let weight_data: *mut AnyObject = msg_send![tensor_data_cls, alloc];
    let weight_data: *mut AnyObject = msg_send![weight_data,
        initWithMTLBuffer: weight_buf_ptr
        shape: weight_shape
        dataType: dt];

    let output_data: *mut AnyObject = msg_send![tensor_data_cls, alloc];
    let output_data: *mut AnyObject = msg_send![output_data,
        initWithMTLBuffer: output_buf_ptr
        shape: output_shape
        dataType: dt];

    // Build feeds dict: { input_tensor: input_data, weight_tensor: weight_data }
    let dict_cls = AnyClass::get("NSMutableDictionary").unwrap();
    let feeds: *mut AnyObject = msg_send![dict_cls, dictionaryWithCapacity: 2usize];
    let _: () = msg_send![feeds, setObject: input_data forKey: input_tensor];
    let _: () = msg_send![feeds, setObject: weight_data forKey: weight_tensor];

    // Results dict: { output_tensor: output_data }
    let results: *mut AnyObject = msg_send![dict_cls, dictionaryWithCapacity: 1usize];
    let _: () = msg_send![results, setObject: output_data forKey: output_tensor];

    // MPSCommandBuffer self-rotates its underlying MTL buffer (kernels can
    // call commit on the inner buffer and the wrapper allocates a fresh
    // one). Sharing the batched CURRENT_CMD does not work because the
    // wrapper's commit collides with `flush()`'s commit. Mint a dedicated
    // MPSCommandBuffer from the queue, encode + commit + wait synchronously,
    // and let the normal batch resume after.
    //
    // Opt-in via KLEARU_MPSGRAPH_CONV. The default sgemm-based conv path
    // is generally faster.
    flush();
    let backend = get_metal();
    let queue_ptr: *mut AnyObject = ForeignType::as_ptr(&backend.queue) as *mut AnyObject;
    let mps_cmd_cls = AnyClass::get("MPSCommandBuffer").unwrap();
    let result = objc2::exception::catch(std::panic::AssertUnwindSafe(|| {
        let mps_cmd: *mut AnyObject = msg_send![mps_cmd_cls,
            commandBufferFromCommandQueue: queue_ptr];
        let _: () = msg_send![graph,
            encodeToCommandBuffer: mps_cmd
            feeds: feeds
            targetOperations: std::ptr::null_mut::<AnyObject>()
            resultsDictionary: results
            executionDescriptor: std::ptr::null_mut::<AnyObject>()];
        // Wait on the *root* MTLCommandBuffer, not the MPSCommandBuffer.
        // MPSCommandBuffer satisfies MTLCommandBuffer by message forwarding,
        // so `class_getInstanceMethod(MPSCommandBuffer, waitUntilCompleted)`
        // is NULL: the send works at runtime but objc2's debug-build
        // signature check rejects it, and because we are inside
        // `objc2::exception::catch` that panic cannot unwind — the process
        // aborts. `rootCommandBuffer` hands back the concrete MTLCommandBuffer,
        // which implements the selector for real. Captured after encoding so
        // any buffer rotation MPSGraph performed is already reflected.
        let root: *mut AnyObject = msg_send![mps_cmd, rootCommandBuffer];
        let _: *mut AnyObject = msg_send![root, retain];
        let _: () = msg_send![mps_cmd, commit];
        let _: () = msg_send![root, waitUntilCompleted];
        let _: () = msg_send![root, release];
    }));
    if let Err(exc) = result {
        if let Some(exc) = exc {
            let reason: *mut AnyObject = msg_send![exc.as_ref() as *const _ as *mut AnyObject, reason];
            let utf8: *const std::os::raw::c_char = msg_send![reason, UTF8String];
            if !utf8.is_null() {
                let cstr = std::ffi::CStr::from_ptr(utf8);
                eprintln!("MPSGraph dispatch threw: {}", cstr.to_string_lossy());
            } else {
                eprintln!("MPSGraph dispatch threw (no reason string)");
            }
        } else {
            eprintln!("MPSGraph dispatch threw (no exception object)");
        }
        panic!("MPSGraph dispatch failed");
    }

    // Release per-call objects (graph + tensors are kept alive in cache).
    let _: () = msg_send![input_data, release];
    let _: () = msg_send![weight_data, release];
    let _: () = msg_send![output_data, release];
}

fn build_conv_graph(key: ConvGraphKey, h_out: usize, w_out: usize) -> CachedConvGraph {
    unsafe {
        let nsnumber_cls = AnyClass::get("NSNumber").unwrap();
        let make_shape = |dims: &[usize]| -> *mut AnyObject {
            let arr_cls = AnyClass::get("NSMutableArray").unwrap();
            let arr: *mut AnyObject = msg_send![arr_cls, arrayWithCapacity: dims.len()];
            for &d in dims {
                let n: *mut AnyObject = msg_send![nsnumber_cls, numberWithUnsignedLongLong: d as u64];
                let _: () = msg_send![arr, addObject: n];
            }
            arr
        };

        let graph_cls = AnyClass::get("MPSGraph")
            .expect("MPSGraph not registered — link MetalPerformanceShadersGraph.framework");
        let graph: *mut AnyObject = msg_send![graph_cls, alloc];
        let graph: *mut AnyObject = msg_send![graph, init];
        // Retain the graph so it survives outside this scope (MPSGraphCache holds it).
        let _: *mut AnyObject = msg_send![graph, retain];

        let input_shape = make_shape(&[key.n, key.in_c, key.h_in, key.w_in]);
        let weight_shape = make_shape(&[key.out_c, key.in_c, key.kh, key.kw]);

        let dt = if key.f32_compute { MPS_DATA_TYPE_FLOAT32 } else { MPS_DATA_TYPE_FLOAT16 };
        let input_tensor: *mut AnyObject = msg_send![graph,
            placeholderWithShape: input_shape
            dataType: dt
            name: std::ptr::null_mut::<AnyObject>()];
        let weight_tensor: *mut AnyObject = msg_send![graph,
            placeholderWithShape: weight_shape
            dataType: dt
            name: std::ptr::null_mut::<AnyObject>()];
        let _: *mut AnyObject = msg_send![input_tensor, retain];
        let _: *mut AnyObject = msg_send![weight_tensor, retain];

        let desc_cls = AnyClass::get("MPSGraphConvolution2DOpDescriptor")
            .expect("MPSGraphConvolution2DOpDescriptor not registered");
        let desc: *mut AnyObject = msg_send![desc_cls,
            descriptorWithStrideInX: key.stride strideInY: key.stride
            dilationRateInX: 1usize dilationRateInY: 1usize
            groups: 1usize
            paddingLeft: key.pad paddingRight: key.pad
            paddingTop: key.pad paddingBottom: key.pad
            paddingStyle: MPSGRAPH_PADDING_EXPLICIT
            dataLayout: MPSGRAPH_LAYOUT_NCHW
            weightsLayout: MPSGRAPH_LAYOUT_OIHW];

        let output_tensor: *mut AnyObject = msg_send![graph,
            convolution2DWithSourceTensor: input_tensor
            weightsTensor: weight_tensor
            descriptor: desc
            name: std::ptr::null_mut::<AnyObject>()];
        let _: *mut AnyObject = msg_send![output_tensor, retain];

        CachedConvGraph {
            graph,
            input_tensor,
            weight_tensor,
            output_tensor,
            h_out, w_out,
            gen: 0,
        }
    }
}

/// MPSGraph scaled-dot-product attention. Opt-in alternative to the
/// hand-rolled `flash_attention_f16_gpu`; the default path uses the
/// hand-rolled kernel for numerical stability. The graph compiler picks
/// tile sizes for the actual shapes and the Q·Kᵀ→softmax→·V chain
/// runs without materialising the full attention matrix.
///
/// MPSGraph placeholders only accept real f16 (bf16 is rejected with
/// `unexpected MPSDataType`), so the wrapper casts bf16 → f16 around the
/// call. Cast cost is small relative to the attention itself
/// (~10MB per Q/K/V at 4096×640).
pub fn mpsgraph_sdpa_bf16_gpu(
    q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, o: &mut GpuTensor,
    n: usize, h: usize, l_q: usize, l_kv: usize, d: usize, scale: f32,
) {
    if force_flash_attn() {
        // Tiled when a pipeline exists for this head dim, scalar otherwise.
        // Both are bf16-in / f32-accumulate, so this is a pure speed change:
        // `flash_attention_tiled_matches_scalar_and_cpu` pins them together.
        flash_attention_best_f16_gpu(q, k, v, o, n, h, l_q, l_kv, d, scale);
        return;
    }
    objc2::rc::autoreleasepool(|_| {
        mpsgraph_sdpa_bf16_gpu_inner(q, k, v, o, n, h, l_q, l_kv, d, scale)
    });
}

fn mpsgraph_sdpa_bf16_gpu_inner(
    q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, o: &mut GpuTensor,
    n: usize, h: usize, l_q: usize, l_kv: usize, d: usize, scale: f32,
) {
    use metal::foreign_types::ForeignType;
    debug_assert_eq!(q.dtype, GpuDtype::F16);
    debug_assert_eq!(k.dtype, GpuDtype::F16);
    debug_assert_eq!(v.dtype, GpuDtype::F16);
    debug_assert_eq!(o.dtype, GpuDtype::F16);

    let q_elems = n * h * l_q * d;
    let kv_elems = n * h * l_kv * d;

    // Manual SDPA decomposition with f32 internals. MPSGraph rejects
    // bf16 placeholders (assertion in getMLIRElementType), so f16
    // placeholders are used with bf16↔f16 cast scratch buffers (same
    // plumbing as the conv path); inside the graph the chain casts to f32
    // around `transpose → matmul → multiply → softmax → matmul`. The
    // compiler keeps the f32 casts because there is no monolithic f16
    // implementation of that chain to substitute — unlike the fused SDPA
    // op which has a hardware-accelerated f16 path.
    let q_f16 = acquire_buffer(q_elems * 2);
    let k_f16 = acquire_buffer(kv_elems * 2);
    let v_f16 = acquire_buffer(kv_elems * 2);
    let o_f16 = acquire_buffer(q_elems * 2);
    cast_bf16_to_half_buf(&q.buffer, &q_f16, q_elems);
    cast_bf16_to_half_buf(&k.buffer, &k_f16, kv_elems);
    cast_bf16_to_half_buf(&v.buffer, &v_f16, kv_elems);

    let key = SdpaGraphKey { n, h, l_q, l_kv, d, scale_bits: scale.to_bits() };

    let cached = SDPA_GRAPH_CACHE.with_borrow_mut(|cache| {
        if let Some(c) = cache.get(&key) {
            (c.graph, c.q_tensor, c.k_tensor, c.v_tensor, c.o_tensor)
        } else {
            let built = match unsafe {
                objc2::exception::catch(std::panic::AssertUnwindSafe(|| {
                    build_sdpa_graph(key, scale)
                }))
            } {
                Ok(g) => g,
                Err(exc) => {
                    if let Some(exc) = exc {
                        let reason: *mut AnyObject = unsafe { msg_send![exc.as_ref() as *const _ as *mut AnyObject, reason] };
                        let utf8: *const std::os::raw::c_char = unsafe { msg_send![reason, UTF8String] };
                        if !utf8.is_null() {
                            let cstr = unsafe { std::ffi::CStr::from_ptr(utf8) };
                            eprintln!("MPSGraph SDPA build threw: {}", cstr.to_string_lossy());
                            eprintln!("  shape: n={n} h={h} l_q={l_q} l_kv={l_kv} d={d}");
                        }
                    }
                    panic!("MPSGraph SDPA build failed");
                }
            };
            let r = (built.graph, built.q_tensor, built.k_tensor, built.v_tensor, built.o_tensor);
            cache.insert(key, built);
            r
        }
    });
    let (graph, q_tensor, k_tensor, v_tensor, o_tensor) = cached;

    unsafe {
        let nsnumber_cls = AnyClass::get("NSNumber").unwrap();
        let make_shape = |dims: &[usize]| -> *mut AnyObject {
            let arr_cls = AnyClass::get("NSMutableArray").unwrap();
            let arr: *mut AnyObject = msg_send![arr_cls, arrayWithCapacity: dims.len()];
            for &dd in dims {
                let nn: *mut AnyObject = msg_send![nsnumber_cls, numberWithUnsignedLongLong: dd as u64];
                let _: () = msg_send![arr, addObject: nn];
            }
            arr
        };
        let q_shape = make_shape(&[n, h, l_q, d]);
        let kv_shape = make_shape(&[n, h, l_kv, d]);
        let o_shape = make_shape(&[n, h, l_q, d]);

        let tensor_data_cls = AnyClass::get("MPSGraphTensorData").unwrap();
        let q_buf_ptr: *mut AnyObject = ForeignType::as_ptr(&q_f16) as *mut AnyObject;
        let k_buf_ptr: *mut AnyObject = ForeignType::as_ptr(&k_f16) as *mut AnyObject;
        let v_buf_ptr: *mut AnyObject = ForeignType::as_ptr(&v_f16) as *mut AnyObject;
        let o_buf_ptr: *mut AnyObject = ForeignType::as_ptr(&o_f16) as *mut AnyObject;

        let q_data: *mut AnyObject = msg_send![tensor_data_cls, alloc];
        let q_data: *mut AnyObject = msg_send![q_data,
            initWithMTLBuffer: q_buf_ptr shape: q_shape dataType: MPS_DATA_TYPE_FLOAT16];
        let k_data: *mut AnyObject = msg_send![tensor_data_cls, alloc];
        let k_data: *mut AnyObject = msg_send![k_data,
            initWithMTLBuffer: k_buf_ptr shape: kv_shape dataType: MPS_DATA_TYPE_FLOAT16];
        let v_data: *mut AnyObject = msg_send![tensor_data_cls, alloc];
        let v_data: *mut AnyObject = msg_send![v_data,
            initWithMTLBuffer: v_buf_ptr shape: kv_shape dataType: MPS_DATA_TYPE_FLOAT16];
        let o_data: *mut AnyObject = msg_send![tensor_data_cls, alloc];
        let o_data: *mut AnyObject = msg_send![o_data,
            initWithMTLBuffer: o_buf_ptr shape: o_shape dataType: MPS_DATA_TYPE_FLOAT16];

        let dict_cls = AnyClass::get("NSMutableDictionary").unwrap();
        let feeds: *mut AnyObject = msg_send![dict_cls, dictionaryWithCapacity: 3usize];
        let _: () = msg_send![feeds, setObject: q_data forKey: q_tensor];
        let _: () = msg_send![feeds, setObject: k_data forKey: k_tensor];
        let _: () = msg_send![feeds, setObject: v_data forKey: v_tensor];

        let results: *mut AnyObject = msg_send![dict_cls, dictionaryWithCapacity: 1usize];
        let _: () = msg_send![results, setObject: o_data forKey: o_tensor];

        flush();
        let backend = get_metal();
        let queue_ptr: *mut AnyObject = ForeignType::as_ptr(&backend.queue) as *mut AnyObject;
        let mps_cmd_cls = AnyClass::get("MPSCommandBuffer").unwrap();
        let result = objc2::exception::catch(std::panic::AssertUnwindSafe(|| {
            let mps_cmd: *mut AnyObject = msg_send![mps_cmd_cls,
                commandBufferFromCommandQueue: queue_ptr];
            let _: () = msg_send![graph,
                encodeToCommandBuffer: mps_cmd
                feeds: feeds
                targetOperations: std::ptr::null_mut::<AnyObject>()
                resultsDictionary: results
                executionDescriptor: std::ptr::null_mut::<AnyObject>()];
            // Wait on the *root* MTLCommandBuffer, not the MPSCommandBuffer.
            // MPSCommandBuffer satisfies MTLCommandBuffer by message forwarding,
            // so `class_getInstanceMethod(MPSCommandBuffer, waitUntilCompleted)`
            // is NULL: the send works at runtime but objc2's debug-build
            // signature check rejects it, and because we are inside
            // `objc2::exception::catch` that panic cannot unwind — the process
            // aborts. `rootCommandBuffer` hands back the concrete MTLCommandBuffer,
            // which implements the selector for real. Captured after encoding so
            // any buffer rotation MPSGraph performed is already reflected.
            let root: *mut AnyObject = msg_send![mps_cmd, rootCommandBuffer];
            let _: *mut AnyObject = msg_send![root, retain];
            let _: () = msg_send![mps_cmd, commit];
            let _: () = msg_send![root, waitUntilCompleted];
            let _: () = msg_send![root, release];
        }));
        if let Err(exc) = result {
            if let Some(exc) = exc {
                let reason: *mut AnyObject = msg_send![exc.as_ref() as *const _ as *mut AnyObject, reason];
                let utf8: *const std::os::raw::c_char = msg_send![reason, UTF8String];
                if !utf8.is_null() {
                    let cstr = std::ffi::CStr::from_ptr(utf8);
                    eprintln!("MPSGraph SDPA dispatch threw: {}", cstr.to_string_lossy());
                } else {
                    eprintln!("MPSGraph SDPA dispatch threw (no reason string)");
                }
            } else {
                eprintln!("MPSGraph SDPA dispatch threw (no exception object)");
            }
            panic!("MPSGraph SDPA dispatch failed");
        }

        let _: () = msg_send![q_data, release];
        let _: () = msg_send![k_data, release];
        let _: () = msg_send![v_data, release];
        let _: () = msg_send![o_data, release];
    }

    // Defensive NaN/Inf guard: sample the f16 output; if any sampled
    // element has all-1 exponent bits, recompute via the hand-rolled
    // flash_attention_f16_gpu kernel.
    let nan_or_inf = unsafe {
        let ptr = o_f16.contents() as *const u16;
        let n_samples = 32.min(q_elems);
        let stride = (q_elems / n_samples).max(1);
        let mut bad = false;
        for i in 0..n_samples {
            let idx = i * stride;
            if idx >= q_elems { break; }
            // f16 NaN or Inf: all 5 exponent bits set (mask 0x7C00).
            let bits = std::ptr::read(ptr.add(idx));
            if bits & 0x7C00 == 0x7C00 {
                bad = true;
                break;
            }
        }
        bad
    };

    if nan_or_inf {
        eprintln!(
            "[mpsgraph-sdpa] NaN/Inf in output, falling back to flash_attention_f16_gpu \
             (n={n} h={h} l_q={l_q} l_kv={l_kv} d={d})"
        );
        flash_attention_f16_gpu(q, k, v, o, n, h, l_q, l_kv, d, scale);
    } else {
        cast_half_to_bf16_buf(&o_f16, &o.buffer, q_elems);
    }

    release_buffer(q_f16);
    release_buffer(k_f16);
    release_buffer(v_f16);
    release_buffer(o_f16);
}

unsafe fn build_sdpa_graph(key: SdpaGraphKey, scale: f32) -> CachedSdpaGraph {
    let nsnumber_cls = AnyClass::get("NSNumber").unwrap();
    let make_shape = |dims: &[usize]| -> *mut AnyObject {
        let arr_cls = AnyClass::get("NSMutableArray").unwrap();
        let arr: *mut AnyObject = msg_send![arr_cls, arrayWithCapacity: dims.len()];
        for &d in dims {
            let n: *mut AnyObject = msg_send![nsnumber_cls, numberWithUnsignedLongLong: d as u64];
            let _: () = msg_send![arr, addObject: n];
        }
        arr
    };

    let graph_cls = AnyClass::get("MPSGraph")
        .expect("MPSGraph not registered — link MetalPerformanceShadersGraph.framework");
    let graph: *mut AnyObject = msg_send![graph_cls, alloc];
    let graph: *mut AnyObject = msg_send![graph, init];
    let _: *mut AnyObject = msg_send![graph, retain];

    let q_shape = make_shape(&[key.n, key.h, key.l_q, key.d]);
    let kv_shape = make_shape(&[key.n, key.h, key.l_kv, key.d]);
    let null = || std::ptr::null_mut::<AnyObject>();

    // f16 placeholders (bf16↔f16 casts happen outside the graph).
    let q_tensor: *mut AnyObject = msg_send![graph,
        placeholderWithShape: q_shape dataType: MPS_DATA_TYPE_FLOAT16 name: null()];
    let k_tensor: *mut AnyObject = msg_send![graph,
        placeholderWithShape: kv_shape dataType: MPS_DATA_TYPE_FLOAT16 name: null()];
    let v_tensor: *mut AnyObject = msg_send![graph,
        placeholderWithShape: kv_shape dataType: MPS_DATA_TYPE_FLOAT16 name: null()];
    let _: *mut AnyObject = msg_send![q_tensor, retain];
    let _: *mut AnyObject = msg_send![k_tensor, retain];
    let _: *mut AnyObject = msg_send![v_tensor, retain];

    // Cast f16 → f32 inside the graph. With manual decomposition (a
    // chain of distinct ops, not a single fused SDPA), the compiler has
    // no monolithic f16 path to substitute, so these casts stick and
    // matmul/softmax run in f32 internals.
    let q_f32: *mut AnyObject = msg_send![graph,
        castTensor: q_tensor toType: MPS_DATA_TYPE_FLOAT32 name: null()];
    let k_f32: *mut AnyObject = msg_send![graph,
        castTensor: k_tensor toType: MPS_DATA_TYPE_FLOAT32 name: null()];
    let v_f32: *mut AnyObject = msg_send![graph,
        castTensor: v_tensor toType: MPS_DATA_TYPE_FLOAT32 name: null()];

    // Decomposed SDPA: transpose K → matmul → scale → softmax → matmul.
    // K transpose [N, H, L_kv, D] → [N, H, D, L_kv] via permutation
    // [0, 1, 3, 2]. matmul on last two dims → scores [N, H, L_q, L_kv].
    let perm = make_shape(&[0, 1, 3, 2]);
    let k_t: *mut AnyObject = msg_send![graph,
        transposeTensor: k_f32 permutation: perm name: null()];

    let scores: *mut AnyObject = msg_send![graph,
        matrixMultiplicationWithPrimaryTensor: q_f32 secondaryTensor: k_t
        name: null()];

    let scale_const: *mut AnyObject = msg_send![graph,
        constantWithScalar: scale as f64 dataType: MPS_DATA_TYPE_FLOAT32];
    let scaled_scores: *mut AnyObject = msg_send![graph,
        multiplicationWithPrimaryTensor: scores secondaryTensor: scale_const
        name: null()];

    let probs: *mut AnyObject = msg_send![graph,
        softMaxWithTensor: scaled_scores axis: 3isize name: null()];

    let o_f32: *mut AnyObject = msg_send![graph,
        matrixMultiplicationWithPrimaryTensor: probs secondaryTensor: v_f32
        name: null()];

    let o_tensor: *mut AnyObject = msg_send![graph,
        castTensor: o_f32 toType: MPS_DATA_TYPE_FLOAT16 name: null()];
    let _: *mut AnyObject = msg_send![o_tensor, retain];

    CachedSdpaGraph { graph, q_tensor, k_tensor, v_tensor, o_tensor }
}

/// MPSGraph-fused GeGLU feed-forward: replaces the three separate dispatches
/// (`proj_in.forward_gpu` matmul + bias, custom split/gelu/multiply kernel,
/// `proj_out.forward_gpu` matmul + bias) with a single compiled graph that
/// the MPSGraph compiler fuses end-to-end.
///
/// Same f16-storage / f32-internal trick as SDPA: the graph casts to f32
/// for the matmuls and gelu, then back to f16 for the output.
pub fn mpsgraph_geglu_ffn_f16_gpu(
    x: &GpuTensor,
    proj_in_w: &[f32],   proj_in_b:  Option<&[f32]>,
    proj_out_w: &[f32],  proj_out_b: Option<&[f32]>,
    n_rows: usize, in_dim: usize, hidden: usize, out_dim: usize,
) -> GpuTensor {
    let mut out = GpuTensor::new_f16(vec![n_rows, out_dim]);
    objc2::rc::autoreleasepool(|_| {
        mpsgraph_geglu_ffn_inner(
            x, proj_in_w, proj_in_b, proj_out_w, proj_out_b,
            n_rows, in_dim, hidden, out_dim, &mut out,
        );
    });
    out
}

fn mpsgraph_geglu_ffn_inner(
    x: &GpuTensor,
    proj_in_w: &[f32],   proj_in_b:  Option<&[f32]>,
    proj_out_w: &[f32],  proj_out_b: Option<&[f32]>,
    n_rows: usize, in_dim: usize, hidden: usize, out_dim: usize,
    out: &mut GpuTensor,
) {
    use metal::foreign_types::ForeignType;
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    debug_assert_eq!(proj_in_w.len(), 2 * hidden * in_dim);
    debug_assert_eq!(proj_out_w.len(), out_dim * hidden);

    let key = GegluGraphKey {
        n_rows, in_dim, hidden, out_dim,
        proj_in_ptr: proj_in_w.as_ptr() as usize,
        proj_out_ptr: proj_out_w.as_ptr() as usize,
        has_in_bias: proj_in_b.is_some(),
        has_out_bias: proj_out_b.is_some(),
    };

    let cached = GEGLU_GRAPH_CACHE.with_borrow_mut(|cache| {
        if let Some(c) = cache.get(&key) {
            (c.graph, c.x_tensor, c.proj_in_w_tensor, c.proj_in_b_tensor,
             c.proj_out_w_tensor, c.proj_out_b_tensor, c.o_tensor)
        } else {
            let built = match unsafe {
                objc2::exception::catch(std::panic::AssertUnwindSafe(|| {
                    build_geglu_graph(key)
                }))
            } {
                Ok(g) => g,
                Err(exc) => {
                    if let Some(exc) = exc {
                        let reason: *mut AnyObject = unsafe { msg_send![exc.as_ref() as *const _ as *mut AnyObject, reason] };
                        let utf8: *const std::os::raw::c_char = unsafe { msg_send![reason, UTF8String] };
                        if !utf8.is_null() {
                            let cstr = unsafe { std::ffi::CStr::from_ptr(utf8) };
                            eprintln!("MPSGraph GeGLU build threw: {}", cstr.to_string_lossy());
                            eprintln!("  shape: n_rows={n_rows} in_dim={in_dim} hidden={hidden} out_dim={out_dim}");
                        }
                    }
                    panic!("MPSGraph GeGLU build failed");
                }
            };
            let r = (built.graph, built.x_tensor, built.proj_in_w_tensor, built.proj_in_b_tensor,
                    built.proj_out_w_tensor, built.proj_out_b_tensor, built.o_tensor);
            cache.insert(key, built);
            r
        }
    });
    let (graph, x_t, win_t, bin_t, wout_t, bout_t, o_t) = cached;

    // weight_f16_buffer stores values as bf16 (project convention).
    // MPSGraph rejects bf16 placeholders for matmul/SDPA, so real f16
    // is required — cast once per weight via mpsgraph_weight_f16_cached
    // and retained for the process lifetime.
    let proj_in_w_bf16 = weight_f16_buffer(proj_in_w);
    let proj_out_w_bf16 = weight_f16_buffer(proj_out_w);
    let proj_in_b_bf16 = proj_in_b.map(weight_f16_buffer);
    let proj_out_b_bf16 = proj_out_b.map(weight_f16_buffer);
    // These weights are owned by a model struct that outlives the call, so the
    // address is stable and unique for as long as it serves as an id. That
    // holds only because nothing here creates and drops weight buffers in a
    // loop; see `next_weight_id` for why an address is otherwise no identity.
    let proj_in_w_buf = mpsgraph_weight_f16_cached(
        &proj_in_w_bf16, proj_in_w.as_ptr() as u64, proj_in_w.len());
    let proj_out_w_buf = mpsgraph_weight_f16_cached(
        &proj_out_w_bf16, proj_out_w.as_ptr() as u64, proj_out_w.len());
    let proj_in_b_buf = proj_in_b_bf16.as_ref().zip(proj_in_b).map(|(b, src)| {
        mpsgraph_weight_f16_cached(b, src.as_ptr() as u64, src.len())
    });
    let proj_out_b_buf = proj_out_b_bf16.as_ref().zip(proj_out_b).map(|(b, src)| {
        mpsgraph_weight_f16_cached(b, src.as_ptr() as u64, src.len())
    });

    // x input is bf16 — cast to f16 scratch for the graph. Output also
    // f16 scratch; cast back to bf16 destination at the end.
    let x_elems = n_rows * in_dim;
    let o_elems = n_rows * out_dim;
    let x_f16 = acquire_buffer(x_elems * 2);
    let o_f16 = acquire_buffer(o_elems * 2);
    cast_bf16_to_half_buf(&x.buffer, &x_f16, x_elems);

    unsafe {
        let nsnumber_cls = AnyClass::get("NSNumber").unwrap();
        let make_shape = |dims: &[usize]| -> *mut AnyObject {
            let arr_cls = AnyClass::get("NSMutableArray").unwrap();
            let arr: *mut AnyObject = msg_send![arr_cls, arrayWithCapacity: dims.len()];
            for &dd in dims {
                let nn: *mut AnyObject = msg_send![nsnumber_cls, numberWithUnsignedLongLong: dd as u64];
                let _: () = msg_send![arr, addObject: nn];
            }
            arr
        };

        let x_shape = make_shape(&[n_rows, in_dim]);
        let win_shape = make_shape(&[2 * hidden, in_dim]);
        let bin_shape = make_shape(&[2 * hidden]);
        let wout_shape = make_shape(&[out_dim, hidden]);
        let bout_shape = make_shape(&[out_dim]);
        let o_shape = make_shape(&[n_rows, out_dim]);

        let tensor_data_cls = AnyClass::get("MPSGraphTensorData").unwrap();

        let make_data = |buf: &Buffer, shape: *mut AnyObject| -> *mut AnyObject {
            let buf_ptr: *mut AnyObject = ForeignType::as_ptr(buf) as *mut AnyObject;
            let d: *mut AnyObject = msg_send![tensor_data_cls, alloc];
            let d: *mut AnyObject = msg_send![d,
                initWithMTLBuffer: buf_ptr shape: shape dataType: MPS_DATA_TYPE_FLOAT16];
            d
        };

        let x_data = make_data(&x_f16, x_shape);
        let win_data = make_data(&proj_in_w_buf, win_shape);
        let wout_data = make_data(&proj_out_w_buf, wout_shape);
        let bin_data = proj_in_b_buf.as_ref().map(|b| make_data(b, bin_shape));
        let bout_data = proj_out_b_buf.as_ref().map(|b| make_data(b, bout_shape));
        let o_data = make_data(&o_f16, o_shape);

        let dict_cls = AnyClass::get("NSMutableDictionary").unwrap();
        let n_feeds = 3usize + bin_data.is_some() as usize + bout_data.is_some() as usize;
        let feeds: *mut AnyObject = msg_send![dict_cls, dictionaryWithCapacity: n_feeds];
        let _: () = msg_send![feeds, setObject: x_data forKey: x_t];
        let _: () = msg_send![feeds, setObject: win_data forKey: win_t];
        let _: () = msg_send![feeds, setObject: wout_data forKey: wout_t];
        if let Some(bd) = bin_data {
            let _: () = msg_send![feeds, setObject: bd forKey: bin_t];
        }
        if let Some(bd) = bout_data {
            let _: () = msg_send![feeds, setObject: bd forKey: bout_t];
        }

        let results: *mut AnyObject = msg_send![dict_cls, dictionaryWithCapacity: 1usize];
        let _: () = msg_send![results, setObject: o_data forKey: o_t];

        flush();
        let backend = get_metal();
        let queue_ptr: *mut AnyObject = ForeignType::as_ptr(&backend.queue) as *mut AnyObject;
        let mps_cmd_cls = AnyClass::get("MPSCommandBuffer").unwrap();
        let result = objc2::exception::catch(std::panic::AssertUnwindSafe(|| {
            let mps_cmd: *mut AnyObject = msg_send![mps_cmd_cls,
                commandBufferFromCommandQueue: queue_ptr];
            let _: () = msg_send![graph,
                encodeToCommandBuffer: mps_cmd
                feeds: feeds
                targetOperations: std::ptr::null_mut::<AnyObject>()
                resultsDictionary: results
                executionDescriptor: std::ptr::null_mut::<AnyObject>()];
            // Wait on the *root* MTLCommandBuffer, not the MPSCommandBuffer.
            // MPSCommandBuffer satisfies MTLCommandBuffer by message forwarding,
            // so `class_getInstanceMethod(MPSCommandBuffer, waitUntilCompleted)`
            // is NULL: the send works at runtime but objc2's debug-build
            // signature check rejects it, and because we are inside
            // `objc2::exception::catch` that panic cannot unwind — the process
            // aborts. `rootCommandBuffer` hands back the concrete MTLCommandBuffer,
            // which implements the selector for real. Captured after encoding so
            // any buffer rotation MPSGraph performed is already reflected.
            let root: *mut AnyObject = msg_send![mps_cmd, rootCommandBuffer];
            let _: *mut AnyObject = msg_send![root, retain];
            let _: () = msg_send![mps_cmd, commit];
            let _: () = msg_send![root, waitUntilCompleted];
            let _: () = msg_send![root, release];
        }));
        if let Err(exc) = result {
            if let Some(exc) = exc {
                let reason: *mut AnyObject = msg_send![exc.as_ref() as *const _ as *mut AnyObject, reason];
                let utf8: *const std::os::raw::c_char = msg_send![reason, UTF8String];
                if !utf8.is_null() {
                    let cstr = std::ffi::CStr::from_ptr(utf8);
                    eprintln!("MPSGraph GeGLU dispatch threw: {}", cstr.to_string_lossy());
                }
            }
            panic!("MPSGraph GeGLU dispatch failed");
        }

        let _: () = msg_send![x_data, release];
        let _: () = msg_send![win_data, release];
        let _: () = msg_send![wout_data, release];
        if let Some(bd) = bin_data { let _: () = msg_send![bd, release]; }
        if let Some(bd) = bout_data { let _: () = msg_send![bd, release]; }
        let _: () = msg_send![o_data, release];
    }

    // f16 graph output → bf16 destination buffer.
    cast_half_to_bf16_buf(&o_f16, &out.buffer, o_elems);
    release_buffer(x_f16);
    release_buffer(o_f16);
}

unsafe fn build_geglu_graph(key: GegluGraphKey) -> CachedGegluGraph {
    let nsnumber_cls = AnyClass::get("NSNumber").unwrap();
    let make_shape = |dims: &[usize]| -> *mut AnyObject {
        let arr_cls = AnyClass::get("NSMutableArray").unwrap();
        let arr: *mut AnyObject = msg_send![arr_cls, arrayWithCapacity: dims.len()];
        for &d in dims {
            let n: *mut AnyObject = msg_send![nsnumber_cls, numberWithUnsignedLongLong: d as u64];
            let _: () = msg_send![arr, addObject: n];
        }
        arr
    };

    let graph_cls = AnyClass::get("MPSGraph").expect("MPSGraph missing");
    let graph: *mut AnyObject = msg_send![graph_cls, alloc];
    let graph: *mut AnyObject = msg_send![graph, init];
    let _: *mut AnyObject = msg_send![graph, retain];

    // Placeholders. Weights/biases stored row-major, dtype f16.
    let x_tensor: *mut AnyObject = msg_send![graph,
        placeholderWithShape: make_shape(&[key.n_rows, key.in_dim])
        dataType: MPS_DATA_TYPE_FLOAT16
        name: std::ptr::null_mut::<AnyObject>()];
    let win_tensor: *mut AnyObject = msg_send![graph,
        placeholderWithShape: make_shape(&[2 * key.hidden, key.in_dim])
        dataType: MPS_DATA_TYPE_FLOAT16
        name: std::ptr::null_mut::<AnyObject>()];
    let wout_tensor: *mut AnyObject = msg_send![graph,
        placeholderWithShape: make_shape(&[key.out_dim, key.hidden])
        dataType: MPS_DATA_TYPE_FLOAT16
        name: std::ptr::null_mut::<AnyObject>()];
    let _: *mut AnyObject = msg_send![x_tensor, retain];
    let _: *mut AnyObject = msg_send![win_tensor, retain];
    let _: *mut AnyObject = msg_send![wout_tensor, retain];

    let bin_tensor: *mut AnyObject = if key.has_in_bias {
        let t: *mut AnyObject = msg_send![graph,
            placeholderWithShape: make_shape(&[2 * key.hidden])
            dataType: MPS_DATA_TYPE_FLOAT16
            name: std::ptr::null_mut::<AnyObject>()];
        let _: *mut AnyObject = msg_send![t, retain];
        t
    } else { std::ptr::null_mut() };

    let bout_tensor: *mut AnyObject = if key.has_out_bias {
        let t: *mut AnyObject = msg_send![graph,
            placeholderWithShape: make_shape(&[key.out_dim])
            dataType: MPS_DATA_TYPE_FLOAT16
            name: std::ptr::null_mut::<AnyObject>()];
        let _: *mut AnyObject = msg_send![t, retain];
        t
    } else { std::ptr::null_mut() };

    // Cast everything to f32 inside the graph for stable matmul accumulation
    // and gelu. Compiler folds the casts into the matmul kernel.
    let x_f32: *mut AnyObject = msg_send![graph,
        castTensor: x_tensor toType: MPS_DATA_TYPE_FLOAT32
        name: std::ptr::null_mut::<AnyObject>()];
    let win_f32: *mut AnyObject = msg_send![graph,
        castTensor: win_tensor toType: MPS_DATA_TYPE_FLOAT32
        name: std::ptr::null_mut::<AnyObject>()];
    let wout_f32: *mut AnyObject = msg_send![graph,
        castTensor: wout_tensor toType: MPS_DATA_TYPE_FLOAT32
        name: std::ptr::null_mut::<AnyObject>()];

    // Weight is [out, in]; we want x @ W^T → transpose to [in, out].
    let perm = make_shape(&[1, 0]);
    let win_t: *mut AnyObject = msg_send![graph,
        transposeTensor: win_f32 permutation: perm
        name: std::ptr::null_mut::<AnyObject>()];
    let perm2 = make_shape(&[1, 0]);
    let wout_t: *mut AnyObject = msg_send![graph,
        transposeTensor: wout_f32 permutation: perm2
        name: std::ptr::null_mut::<AnyObject>()];

    let mut h: *mut AnyObject = msg_send![graph,
        matrixMultiplicationWithPrimaryTensor: x_f32 secondaryTensor: win_t
        name: std::ptr::null_mut::<AnyObject>()];
    if key.has_in_bias {
        let bin_f32: *mut AnyObject = msg_send![graph,
            castTensor: bin_tensor toType: MPS_DATA_TYPE_FLOAT32
            name: std::ptr::null_mut::<AnyObject>()];
        h = msg_send![graph,
            additionWithPrimaryTensor: h secondaryTensor: bin_f32
            name: std::ptr::null_mut::<AnyObject>()];
    }

    // Split h [n_rows, 2·hidden] into a, b each [n_rows, hidden] along axis 1.
    let a: *mut AnyObject = msg_send![graph,
        sliceTensor: h dimension: 1isize start: 0isize length: key.hidden as isize
        name: std::ptr::null_mut::<AnyObject>()];
    let b: *mut AnyObject = msg_send![graph,
        sliceTensor: h dimension: 1isize start: key.hidden as isize length: key.hidden as isize
        name: std::ptr::null_mut::<AnyObject>()];
    // GELU via tanh approximation, matching the existing `gelu` MSL kernel:
    //   gelu(b) = 0.5 · b · (1 + tanh(√(2/π) · (b + 0.044715 · b³)))
    // MPSGraph's `geLUWithTensor:` selector availability is version-dependent;
    // building from primitives (tanh + arithmetic) is portable and the graph
    // compiler fuses the chain.
    let null = || std::ptr::null_mut::<AnyObject>();
    let c_half: *mut AnyObject = msg_send![graph,
        constantWithScalar: 0.5f64 dataType: MPS_DATA_TYPE_FLOAT32];
    let c_one: *mut AnyObject = msg_send![graph,
        constantWithScalar: 1.0f64 dataType: MPS_DATA_TYPE_FLOAT32];
    let c_c1: *mut AnyObject = msg_send![graph,
        constantWithScalar: 0.7978845608028654f64 dataType: MPS_DATA_TYPE_FLOAT32];
    let c_c2: *mut AnyObject = msg_send![graph,
        constantWithScalar: 0.044715f64 dataType: MPS_DATA_TYPE_FLOAT32];

    let b_sq: *mut AnyObject = msg_send![graph,
        multiplicationWithPrimaryTensor: b secondaryTensor: b name: null()];
    let b_cubed: *mut AnyObject = msg_send![graph,
        multiplicationWithPrimaryTensor: b_sq secondaryTensor: b name: null()];
    let c2_b3: *mut AnyObject = msg_send![graph,
        multiplicationWithPrimaryTensor: c_c2 secondaryTensor: b_cubed name: null()];
    let inner_sum: *mut AnyObject = msg_send![graph,
        additionWithPrimaryTensor: b secondaryTensor: c2_b3 name: null()];
    let arg: *mut AnyObject = msg_send![graph,
        multiplicationWithPrimaryTensor: c_c1 secondaryTensor: inner_sum name: null()];
    let t: *mut AnyObject = msg_send![graph,
        tanhWithTensor: arg name: null()];
    let plus1: *mut AnyObject = msg_send![graph,
        additionWithPrimaryTensor: c_one secondaryTensor: t name: null()];
    let half_b: *mut AnyObject = msg_send![graph,
        multiplicationWithPrimaryTensor: c_half secondaryTensor: b name: null()];
    let gelu_b: *mut AnyObject = msg_send![graph,
        multiplicationWithPrimaryTensor: half_b secondaryTensor: plus1 name: null()];

    let gated: *mut AnyObject = msg_send![graph,
        multiplicationWithPrimaryTensor: a secondaryTensor: gelu_b name: null()];

    let mut y: *mut AnyObject = msg_send![graph,
        matrixMultiplicationWithPrimaryTensor: gated secondaryTensor: wout_t
        name: std::ptr::null_mut::<AnyObject>()];
    if key.has_out_bias {
        let bout_f32: *mut AnyObject = msg_send![graph,
            castTensor: bout_tensor toType: MPS_DATA_TYPE_FLOAT32
            name: std::ptr::null_mut::<AnyObject>()];
        y = msg_send![graph,
            additionWithPrimaryTensor: y secondaryTensor: bout_f32
            name: std::ptr::null_mut::<AnyObject>()];
    }

    let o_tensor: *mut AnyObject = msg_send![graph,
        castTensor: y toType: MPS_DATA_TYPE_FLOAT16
        name: std::ptr::null_mut::<AnyObject>()];
    let _: *mut AnyObject = msg_send![o_tensor, retain];

    CachedGegluGraph {
        graph,
        x_tensor, proj_in_w_tensor: win_tensor, proj_in_b_tensor: bin_tensor,
        proj_out_w_tensor: wout_tensor, proj_out_b_tensor: bout_tensor,
        o_tensor,
    }
}

/// f16 MPS sgemm with per-buffer byte offsets. Storage f16, compute fp32.
pub fn sgemm_f16_buf_with_offsets(
    out_buf: &Buffer, out_off_elems: usize,
    a_buf: &Buffer, a_off_elems: usize,
    b_buf: &Buffer, b_off_elems: usize,
    m: usize, n: usize, k: usize,
) {
    let backend = get_metal();
    unsafe {
        dispatch_mps_matmul_bf16_via_f32_with_offsets(
            backend,
            a_buf, a_off_elems * 2,
            b_buf, b_off_elems * 2,
            out_buf, out_off_elems * 2,
            m, n, k,
            false, false, k, n,
        );
    }
}

/// Allocate a fresh fp16 GPU buffer of `n_elements` and return as a Buffer
/// that the caller releases via `release_buffer` when done.
pub fn acquire_f16_buffer(n_elements: usize) -> Buffer {
    acquire_buffer(n_elements * 2)
}

/// Non-caching f32→f16 upload for **small per-step values** like resnet
/// time-embedding projections. Allocates fresh (no pool) so we don't have
/// to flush pending GPU work to safely CPU-write — Metal retains the buffer
/// for the lifetime of any cmd buffer that references it, so dropping our
/// handle (or `release_pool_buffer`) just removes our ownership; the GPU
/// keeps it alive until its dispatching cmd completes.
///
/// Per SDXL inference: ~50 resnets × 25 steps × 2 CFG ≈ 2500 calls. With
/// the pool path each call would force a `flush()` (because the pool may
/// hand back an in-flight buffer), serializing the entire async pipeline.
/// Fresh allocation of <4KB is sub-microsecond on Metal, so we eat the
/// allocator cost (~12ms total) instead of ~thousands of sync waits.
/// Upload weights as IEEE f16, for the half-storage decoder path.
pub fn upload_f32_as_half_buffer(slice: &[f32]) -> Buffer {
    use half::slice::HalfFloatSliceExt;
    let backend = get_metal();
    let buf = backend.device.new_buffer(
        (slice.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let dst: *mut half::f16 = buf.contents() as *mut half::f16;
        let dst_slice: &mut [half::f16] = std::slice::from_raw_parts_mut(dst, slice.len());
        dst_slice.convert_from_f32_slice(slice);
    }
    buf
}

/// Convert a half tensor to bf16, for the few ops that have no half variant
/// (currently only the mid-block attention's MPS matmuls).
pub fn tensor_half_to_bf16(x: &GpuTensor) -> GpuTensor {
    debug_assert!(x.dtype.is_half());
    let out = GpuTensor::new_f16(x.shape.clone());
    cast_half_to_bf16_buf(&x.buffer, &out.buffer, x.elements());
    out
}

/// Inverse of [`tensor_half_to_bf16`].
pub fn tensor_bf16_to_half(x: &GpuTensor) -> GpuTensor {
    debug_assert!(!x.dtype.is_half());
    let out = GpuTensor::new_half(x.shape.clone());
    cast_bf16_to_half_buf(&x.buffer, &out.buffer, x.elements());
    out
}

pub fn upload_f32_as_f16_buffer(slice: &[f32]) -> Buffer {
    use half::bf16;
    use half::slice::HalfFloatSliceExt;
    let backend = get_metal();
    let buf = backend.device.new_buffer(
        (slice.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let dst: *mut bf16 = buf.contents() as *mut bf16;
        let dst_slice: &mut [bf16] = std::slice::from_raw_parts_mut(dst, slice.len());
        dst_slice.convert_from_f32_slice(slice);
    }
    buf
}

/// Upload f32 data as an f32 buffer, unconverted. For small parameters whose
/// precision matters more than their size (e.g. zero-centred norm scales).
pub fn upload_f32_buffer(slice: &[f32]) -> Buffer {
    let backend = get_metal();
    backend.device.new_buffer_with_data(
        slice.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(slice) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Upload raw **little-endian bf16 bytes** straight into a fresh GPU buffer.
///
/// The companion to [`crate::weight::loader::ComponentTensors::get_bf16`]. For
/// a bf16-native checkpoint the source is an mmap view, so a weight goes
/// disk → page cache → `MTLBuffer` as a single `memcpy` with no decode step
/// and no f32 temporary. The previous route
/// (`get_f32()` → `upload_f32_as_f16_buffer()`) was bit-identical but paid a
/// full f32 materialisation — 2x the bytes and a per-element convert — for
/// every tensor in the model.
///
/// `bytes.len()` must be even; the buffer is sized to match exactly.
///
/// Alignment: safetensors payloads are not guaranteed 2-byte aligned, which is
/// why the input is `&[u8]` and this copies rather than reinterpreting. Do not
/// "optimise" this into a `bytemuck::cast_slice` — it will fault or silently
/// misread on odd header sizes.
///
/// Allocates fresh rather than from `BUF_POOL`, matching
/// `upload_f32_as_f16_buffer`: weights are long-lived, and a pool buffer may
/// still be in flight, which would force a `flush()` per weight.
/// Upload a bf16 payload as IEEE f16.
///
/// Lossless: bfloat's 8 mantissa bits fit inside half's 10, and every bf16
/// magnitude is representable. Needed because a half-storage model cannot be
/// fed weights that are still bfloat — the matmul would reinterpret the bits.
pub fn upload_bf16_bytes_as_half(bytes: &[u8]) -> Buffer {
    debug_assert_eq!(bytes.len() % 2, 0, "bf16 payload must be an even byte count");
    let n = bytes.len() / 2;
    let backend = get_metal();
    let buf = backend.device.new_buffer(
        bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let dst: *mut half::f16 = buf.contents() as *mut half::f16;
        for i in 0..n {
            let raw = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            *dst.add(i) = half::f16::from_f32(half::bf16::from_bits(raw).to_f32());
        }
    }
    buf
}

pub fn upload_bf16_bytes(bytes: &[u8]) -> Buffer {
    debug_assert_eq!(bytes.len() % 2, 0, "bf16 payload must be an even byte count");
    let backend = get_metal();
    let buf = backend.device.new_buffer(
        bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buf.contents() as *mut u8,
            bytes.len(),
        );
    }
    buf
}

/// In-place per-batch-per-channel bias add for `[N, C, HW]` × `[N, C]` bias.
/// `bias_buf` is f16. Used by ResnetBlock to inject the time-embedding
/// projection (which is per-batch) into the conv output.
pub fn bias_add_nc_to_nchw_f16_gpu(
    x: &mut GpuTensor, bias_buf: &Buffer, c: usize, hw: usize,
) {
    use metal::MTLSize;
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.bias_add_nc_pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(bias_buf), 0);
        let c_u = c as u32;
        let hw_u = hw as u32;
        enc.set_bytes(2, 4, &c_u as *const _ as *const _);
        enc.set_bytes(3, 4, &hw_u as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(x.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// fp16 GPU-resident flash attention. Q/K/V/O are GpuTensors in `[N, H, L, D]`
/// head-contiguous layout. Internal accumulators are fp32 for stability.
pub fn flash_attention_f16_gpu(
    q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, o: &mut GpuTensor,
    n: usize, h: usize, l_q: usize, l_kv: usize, d: usize, scale: f32,
) {
    use metal::MTLSize;
    debug_assert_eq!(q.dtype, GpuDtype::F16);
    debug_assert_eq!(k.dtype, GpuDtype::F16);
    debug_assert_eq!(v.dtype, GpuDtype::F16);
    debug_assert_eq!(o.dtype, GpuDtype::F16);

    let backend = get_metal();
    let tpg = (d as u64).next_power_of_two().max(1);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.flash_attn_f16_pipe);
        enc.set_buffer(0, Some(&q.buffer), 0);
        enc.set_buffer(1, Some(&k.buffer), 0);
        enc.set_buffer(2, Some(&v.buffer), 0);
        enc.set_buffer(3, Some(&o.buffer), 0);
        let lq_u = l_q as u32;
        let lkv_u = l_kv as u32;
        let d_u = d as u32;
        enc.set_bytes(4, 4, &lq_u as *const _ as *const _);
        enc.set_bytes(5, 4, &lkv_u as *const _ as *const _);
        enc.set_bytes(6, 4, &d_u as *const _ as *const _);
        enc.set_bytes(7, 4, &scale as *const _ as *const _);
        let smem_bytes = (3 * tpg * 4) as u64;  // fp32 smem
        enc.set_threadgroup_memory_length(0, smem_bytes);
        let groups = (n * h * l_q) as u64;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    });
}

/// Row softmax over `[rows, width]`, with `scale` applied to the logits first.
pub fn softmax_rows_scaled_f16_gpu(x: &mut GpuTensor, rows: usize, width: usize, scale: f32) {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if x.dtype.is_half() { &get_metal().softmax_rows_scaled_h16_pipe } else { &get_metal().softmax_rows_scaled_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(x.dtype, GpuDtype::F16 | GpuDtype::Half));
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        let wu = width as u32;
        enc.set_bytes(1, 4, &wu as *const _ as *const _);
        enc.set_bytes(2, 4, &scale as *const _ as *const _);
        enc.dispatch_threads(MTLSize::new(rows as u64, 1, 1), MTLSize::new(64, 1, 1));
        enc.end_encoding();
    });
}

/// Frame `t_idx`, channels `[c0, c0 + c)`, as a `[p, c]` tensor.
pub fn extract_frame_pc_f16_gpu(
    src: &GpuTensor, c0: usize, c: usize, t_idx: usize, t: usize, p: usize,
) -> GpuTensor {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if src.dtype.is_half() { &get_metal().extract_frame_pc_h16_pipe } else { &get_metal().extract_frame_pc_f16_pipe };
    use metal::MTLSize;
    let out = GpuTensor::new_like(src, vec![p, c]);
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        for (i, v) in [c0, c, t_idx, t, p].iter().enumerate() {
            let u = *v as u32;
            enc.set_bytes(2 + i as u64, 4, &u as *const _ as *const _);
        }
        enc.dispatch_threads(MTLSize::new((p * c) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
    out
}

/// Scatter a `[p, c]` frame back into a `[c, t, p]` volume at `t_idx`.
pub fn insert_frame_pc_f16_gpu(
    src: &GpuTensor, dst: &GpuTensor, c: usize, t_idx: usize, t: usize, p: usize,
) {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if src.dtype.is_half() { &get_metal().insert_frame_pc_h16_pipe } else { &get_metal().insert_frame_pc_f16_pipe };
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&dst.buffer), 0);
        for (i, v) in [c, t_idx, t, p].iter().enumerate() {
            let u = *v as u32;
            enc.set_bytes(2 + i as u64, 4, &u as *const _ as *const _);
        }
        enc.dispatch_threads(MTLSize::new((p * c) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
}

/// Wan VAE `DupUp3D` on the GPU. Returns `[out_c, t*ft - drop, h*fs, w*fs]`.
#[allow(clippy::too_many_arguments)]
pub fn dup_up_3d_f16_gpu(
    src: &GpuTensor, in_c: usize, out_c: usize,
    t: usize, h: usize, w: usize, ft: usize, fs: usize, first_chunk: bool,
) -> GpuTensor {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if src.dtype.is_half() { &get_metal().dup_up_3d_h16_pipe } else { &get_metal().dup_up_3d_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(src.dtype, GpuDtype::F16 | GpuDtype::Half));
    let drop = if first_chunk { ft - 1 } else { 0 };
    let (ot, oh, ow) = (t * ft - drop, h * fs, w * fs);
    let out = GpuTensor::new_like(src, vec![out_c, ot, oh, ow]);
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        for (i, v) in [in_c, out_c, t, h, w, ft, fs, drop].iter().enumerate() {
            let u = *v as u32;
            enc.set_bytes(2 + i as u64, 4, &u as *const _ as *const _);
        }
        enc.dispatch_threads(
            MTLSize::new((out_c * ot * oh * ow) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
    out
}

/// `upsample3d`'s channel-into-time interleave: `[2c, t, p] -> [c, 2t, p]`.
pub fn interleave_time_f16_gpu(src: &GpuTensor, c: usize, t: usize, p: usize) -> GpuTensor {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if src.dtype.is_half() { &get_metal().interleave_time_h16_pipe } else { &get_metal().interleave_time_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(src.dtype, GpuDtype::F16 | GpuDtype::Half));
    debug_assert_eq!(src.elements(), 2 * c * t * p);
    let out = GpuTensor::new_like(src, vec![c, 2 * t, p]);
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        let (cu, tu, pu) = (c as u32, t as u32, p as u32);
        enc.set_bytes(2, 4, &cu as *const _ as *const _);
        enc.set_bytes(3, 4, &tu as *const _ as *const _);
        enc.set_bytes(4, 4, &pu as *const _ as *const _);
        enc.dispatch_threads(MTLSize::new((c * 2 * t * p) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
    out
}

/// Swap the outer two axes: `[a, b, p] -> [b, a, p]`, into a fresh tensor.
pub fn transpose_outer_f16_gpu(src: &GpuTensor, a: usize, b: usize, p: usize) -> GpuTensor {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if src.dtype.is_half() { &get_metal().transpose_outer_h16_pipe } else { &get_metal().transpose_outer_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(src.dtype, GpuDtype::F16 | GpuDtype::Half));
    debug_assert_eq!(src.elements(), a * b * p);
    let mut out = GpuTensor::new_like(src, vec![b, a, p]);
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        let (au, bu, pu) = (a as u32, b as u32, p as u32);
        enc.set_bytes(2, 4, &au as *const _ as *const _);
        enc.set_bytes(3, 4, &bu as *const _ as *const _);
        enc.set_bytes(4, 4, &pu as *const _ as *const _);
        enc.dispatch_threads(MTLSize::new((a * b * p) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
    out.shape = vec![b, a, p];
    out
}

/// Zero a buffer on the GPU via a blit fill.
///
/// `acquire_buffer` clears buffers it recycles, but a freshly created
/// `MTLBuffer` is not contractually zeroed — so anything that accumulates into
/// a new tensor must clear it explicitly.
pub fn zero_buffer_gpu(buf: &Buffer, bytes: usize) {
    with_cmd(|cmd| {
        let blit = cmd.new_blit_command_encoder();
        blit.fill_buffer(buf, metal::NSRange::new(0, bytes as u64), 0);
        blit.end_encoding();
    });
}

/// Zero `len_bytes` of `buf` starting at `offset_bytes` (GPU blit fill).
pub fn zero_range_gpu(buf: &Buffer, offset_bytes: usize, len_bytes: usize) {
    with_cmd(|cmd| {
        let blit = cmd.new_blit_command_encoder();
        blit.fill_buffer(buf, metal::NSRange::new(offset_bytes as u64, len_bytes as u64), 0);
        blit.end_encoding();
    });
}

/// Wan VAE channel norm over a `[C, P]` volume (`P = T*H*W`).
///
/// Reduces across the **outer** channel axis, so the access is strided by `P`
/// — this is not the same operation as [`rms_norm_f16_gpu`], which reduces a
/// contiguous row.
pub fn rms_norm_channels_f16_gpu(x: &mut GpuTensor, gamma: &Buffer, c: usize, p: usize) {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if x.dtype.is_half() { &get_metal().rms_norm_channels_h16_pipe } else { &get_metal().rms_norm_channels_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(x.dtype, GpuDtype::F16 | GpuDtype::Half));
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(gamma), 0);
        let cu = c as u32;
        let pu = p as u32;
        enc.set_bytes(2, 4, &cu as *const _ as *const _);
        enc.set_bytes(3, 4, &pu as *const _ as *const _);
        enc.dispatch_threads(MTLSize::new(p as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
}

/// `dst[dst_off ..][..len] += src[src_off ..][..len]`.
///
/// Takes raw buffers, so unlike the `GpuTensor` dispatchers it cannot read the
/// storage type off its arguments — the caller passes it.
pub fn add_shifted_f16_gpu(
    dst: &Buffer, dst_off: usize, src: &Buffer, src_off: usize, len: usize,
    dtype: GpuDtype,
) {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if dtype.is_half() { &get_metal().add_shifted_h16_pipe } else { &get_metal().add_shifted_f16_pipe };
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(src), 0);
        let d = dst_off as u32;
        let s = src_off as u32;
        enc.set_bytes(2, 4, &d as *const _ as *const _);
        enc.set_bytes(3, 4, &s as *const _ as *const _);
        enc.dispatch_threads(MTLSize::new(len as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
}

/// How Wan's AdaLN modulation is laid out.
///
/// `PerToken` is the Wan 2.2 TI2V path (one modulation per token, so some
/// frames can sit at timestep 0 while others denoise); `PerSample` is the
/// Wan 2.1 / 14B path. The kernels take a stride, and 0 collapses per-token
/// indexing into a broadcast, so both share one code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModLayout {
    PerSample,
    PerToken,
}

impl ModLayout {
    /// Row stride into the timestep projection, in elements.
    fn stride(self, vectors: usize, d: usize) -> u32 {
        match self {
            ModLayout::PerSample => 0,
            ModLayout::PerToken => (vectors * d) as u32,
        }
    }
}

/// `x[t] = x[t] * (1 + (table[scale_i] + temb[t][scale_i])) + (table[shift_i] + temb[t][shift_i])`
///
/// `table` is the block's `[6, d]` scale_shift_table; `temb` the timestep
/// projection, `[n, 6, d]` for `PerToken` or `[6, d]` for `PerSample`.
#[allow(clippy::too_many_arguments)]
pub fn modulate_rows_f16_gpu(
    x: &mut GpuTensor, table: &Buffer, temb: &Buffer,
    n: usize, d: usize, layout: ModLayout, scale_i: usize, shift_i: usize,
) {
    // bfloat or IEEE f16 storage; same kernel body either way.
    let pipe = if x.dtype.is_half() { &get_metal().modulate_rows_f16_pipe_h } else { &get_metal().modulate_rows_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(x.dtype, GpuDtype::F16 | GpuDtype::Half));
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(table), 0);
        enc.set_buffer(2, Some(temb), 0);
        let du = d as u32;
        let st = layout.stride(6, d);
        let si = scale_i as u32;
        let hi = shift_i as u32;
        enc.set_bytes(3, 4, &du as *const _ as *const _);
        enc.set_bytes(4, 4, &st as *const _ as *const _);
        enc.set_bytes(5, 4, &si as *const _ as *const _);
        enc.set_bytes(6, 4, &hi as *const _ as *const _);
        enc.dispatch_threads(MTLSize::new((n * d) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
}

/// `x[t] *= table[gate_i] + temb[t][gate_i]`
pub fn gate_rows_f16_gpu(
    x: &mut GpuTensor, table: &Buffer, temb: &Buffer,
    n: usize, d: usize, layout: ModLayout, gate_i: usize,
) {
    // bfloat or IEEE f16 storage; same kernel body either way.
    let pipe = if x.dtype.is_half() { &get_metal().gate_rows_f16_pipe_h } else { &get_metal().gate_rows_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(x.dtype, GpuDtype::F16 | GpuDtype::Half));
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(table), 0);
        enc.set_buffer(2, Some(temb), 0);
        let du = d as u32;
        let st = layout.stride(6, d);
        let gi = gate_i as u32;
        enc.set_bytes(3, 4, &du as *const _ as *const _);
        enc.set_bytes(4, 4, &st as *const _ as *const _);
        enc.set_bytes(5, 4, &gi as *const _ as *const _);
        enc.dispatch_threads(MTLSize::new((n * d) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
}

/// Final output-norm modulation: shift and scale add the **same** `temb` row,
/// with the `[2, d]` table separating them. `temb` is `[n, d]` or `[d]`.
pub fn final_modulate_f16_gpu(
    x: &mut GpuTensor, table: &Buffer, temb: &Buffer,
    n: usize, d: usize, layout: ModLayout,
) {
    // bfloat or IEEE f16 storage; same kernel body either way.
    let pipe = if x.dtype.is_half() { &get_metal().final_modulate_f16_pipe_h } else { &get_metal().final_modulate_f16_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(x.dtype, GpuDtype::F16 | GpuDtype::Half));
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(table), 0);
        enc.set_buffer(2, Some(temb), 0);
        let du = d as u32;
        let st = layout.stride(1, d);
        enc.set_bytes(3, 4, &du as *const _ as *const _);
        enc.set_bytes(4, 4, &st as *const _ as *const _);
        enc.dispatch_threads(MTLSize::new((n * d) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    });
}

/// Head dims with a tiled-attention pipeline. Others fall back to the scalar
/// kernel; adding one is a single `FA_TILED_KERNEL` line in the MSL.
pub const FLASH_ATTN_TILED_DIMS: [usize; 5] = [16, 32, 64, 96, 128];

/// Tiled flash attention. Same contract and same layout as
/// [`flash_attention_f16_gpu`] — Q/K/V/O are bf16 `[N, H, L, D]` — but tiles
/// 32 queries per threadgroup and runs the two matmuls through simdgroup MMA.
///
/// Returns `false` without dispatching when `d` has no compiled pipeline, so
/// callers can fall back. Use [`flash_attention_best_f16_gpu`] to get that
/// fallback automatically.
///
/// Why it matters: the scalar kernel re-reads all of K and V once per query
/// and takes ~9 threadgroup barriers per (query, kv) pair. At video sequence
/// lengths (a 5s 720p Wan clip is ~27k tokens) that is the entire runtime.
pub fn flash_attention_tiled_f16_gpu(
    q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, o: &mut GpuTensor,
    n: usize, h: usize, l_q: usize, l_kv: usize, d: usize, scale: f32,
) -> bool {
    flash_attention_tiled_impl(q, k, v, o, n, h, l_q, l_kv, l_kv, d, scale, false)
}

/// Flash attention over a key set **padded** to a stride: K/V are
/// `[N, H, l_kv, D]` but only the first `kv_valid` rows of each head are
/// keys; the rest (any finite values) get exactly zero weight.
///
/// This exists because the fastest kernel (register-resident, ~2.75 TF/s at
/// D=128) needs `l_kv % 8 == 0`, and a model whose real key count is ragged —
/// Krea 2's image + prompt tokens — otherwise falls back to the tiled kernel
/// at under a third of that. Pad the stride to a multiple of 8, pass the true
/// count here.
///
/// Dispatches only the register-resident kernel. Returns `false` without
/// doing anything when it does not apply (head dim without a pipeline,
/// `l_kv % 8 != 0`, or the A/B env overrides), so the caller can fall back.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_masked_f16_gpu(
    q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, o: &mut GpuTensor,
    n: usize, h: usize, l_q: usize, l_kv: usize, kv_valid: usize, d: usize, scale: f32,
) -> bool {
    assert!(kv_valid >= 1 && kv_valid <= l_kv, "kv_valid {kv_valid} out of 1..={l_kv}");
    flash_attention_tiled_impl(q, k, v, o, n, h, l_q, l_kv, kv_valid, d, scale, true)
}

#[allow(clippy::too_many_arguments)]
fn flash_attention_tiled_impl(
    q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, o: &mut GpuTensor,
    n: usize, h: usize, l_q: usize, l_kv: usize, kv_valid: usize, d: usize, scale: f32,
    reg_only: bool,
) -> bool {
    use metal::MTLSize;
    debug_assert_eq!(q.dtype, GpuDtype::F16);
    debug_assert_eq!(k.dtype, GpuDtype::F16);
    debug_assert_eq!(v.dtype, GpuDtype::F16);
    debug_assert_eq!(o.dtype, GpuDtype::F16);

    // Must match FA_BM / FA_BN / FA_SG / FA_TPG in the MSL.
    const BM: usize = 32;
    const SG: usize = 4;
    const TPG: usize = 128;

    let backend = get_metal();
    if l_q == 0 || l_kv == 0 {
        return false;
    }
    // KLEARU_FLASH_SCALAR=1 falls back to the (slow, simple) scalar kernel.
    // An A/B against the tiled kernels is the cheapest way to tell whether an
    // intermittent wrong answer lives in the tiled/split code or elsewhere.
    if std::env::var_os("KLEARU_FLASH_SCALAR").is_some() {
        return false;
    }

    // Region A is BM*D floats, which is exactly the 2*BN*D bfloats the K|V
    // tiles need. Then the f32 score tile, the bf16 probability tile at half
    // that, three per-row softmax vectors, and the diagonal scratch.
    //
    // The split kernel prefers itself where available: it halves the live
    // fragment count, which is what limits throughput at D >= 96. It carries
    // two score buffers (one per head-dim half) and runs 256 threads.
    const BN: usize = 32;
    // Must match FA_BNS in the MSL — the split kernel's wider traversal
    // block, affordable because it no longer stages K|V in threadgroup memory.
    const BNS: usize = 64;
    // Must match FA_BMR in the MSL — the register-resident kernel's row block.
    // A mismatch here does not fail loudly: `q_blocks` is derived from it, so
    // too large a value simply leaves the tail of every head uncomputed, which
    // reads as roughly cos_sim 0.72 rather than as an error.
    const BMR: usize = 32;
    // Must match FA_NREG in the MSL — output fragments kept in registers.
    const NREG: usize = 6;
    // `KLEARU_FLASH_TILED=1` forces the tiled kernel where the split one would
    // normally win. The two differ in threadgroup budget (the split kernel
    // carries two score buffers), so which is faster is a measurement, not a
    // deduction — and the answer moves with D and sequence length.
    let force_tiled = std::env::var_os("KLEARU_FLASH_TILED").is_some();
    // The register-resident kernel is the default where it applies: measured
    // 2.75 TF/s against the split kernel's 2.19 at D=128, 11440 tokens.
    // `KLEARU_FLASH_SPLIT=1` forces the older one for an A/B.
    let force_split = std::env::var_os("KLEARU_FLASH_SPLIT").is_some();
    // The split kernel reads K/V straight from device memory and clamps a
    // past-the-end 8-row fragment onto the last full one, which is only sound
    // when L_kv is a multiple of 8. A ragged length takes the tiled kernel,
    // which still stages and zero-fills. Both real generation shapes (11440
    // native, 5070 at 480p... which is *not* a multiple of 8) matter here:
    // only the first gets the fast path, so a ragged 480p run is slower.
    let split_ok = l_kv >= 8 && l_kv % 8 == 0;
    // Only the register-resident kernel honours a `kv_valid` short of `l_kv`.
    let reg_available = !force_split && !force_tiled && split_ok
        && backend.flash_attn_reg_pipes.iter().any(|(dim, _)| *dim == d);
    if reg_only && !reg_available {
        return false;
    }
    debug_assert!(kv_valid == l_kv || reg_available);
    let (pipe, tpg, smem_floats) =
        if let Some((_, p)) = backend
            .flash_attn_reg_pipes
            .iter()
            .find(|(dim, _)| *dim == d)
            .filter(|_| !force_split && !force_tiled && split_ok)
        {
            // Region A holds the Q tile as bfloat and later an 8 x D f32 slab;
            // sized by the larger. Nothing else is shared — S and P live in
            // registers — so this is the whole threadgroup budget.
            // Region A (Q tile as bfloat, later the 8 x D output slab) plus
            // the hand-managed spill arena: one fragment set per simdgroup for
            // the head-dim steps that do not stay in registers.
            (p, 128usize, BMR * d / 2 + (BMR / 8) * (d / 8 - NREG) * 64)
        } else if let Some((_, p)) = backend
            .flash_attn_split_pipes
            .iter()
            .find(|(dim, _)| *dim == d)
            .filter(|_| !force_tiled && split_ok)
        {
            // Region A is the Q tile as bfloat (BM*d/2 floats), no longer a
            // full f32 output stage. Then the two score buffers, the bf16
            // probability tile at half that, the per-row softmax vectors and
            // the diagonal scratch.
            (p, 256usize,
             BM * d / 2 + 2 * BM * BNS + (BM * BNS) / 2 + 3 * BM)
        } else if let Some((_, p)) =
            backend.flash_attn_tiled_pipes.iter().find(|(dim, _)| *dim == d)
        {
            (p, TPG, BM * d + BM * BN + (BM * BN) / 2 + 3 * BM + SG * 64)
        } else {
            return false;
        };
    let smem_bytes = (smem_floats * 4) as u64;
    debug_assert!(
        smem_bytes <= 32 * 1024,
        "tiled attention threadgroup memory {smem_bytes} exceeds 32 KB at d={d}"
    );

    // The register-resident kernel covers BMR queries a threadgroup, not BM.
    let reg_selected = !force_split && !force_tiled && split_ok && d >= 96;
    let q_blocks = if reg_selected {
        l_q.div_ceil(BMR)
    } else {
        l_q.div_ceil(BM)
    };
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&q.buffer), 0);
        enc.set_buffer(1, Some(&k.buffer), 0);
        enc.set_buffer(2, Some(&v.buffer), 0);
        enc.set_buffer(3, Some(&o.buffer), 0);
        let lq_u = l_q as u32;
        let lkv_u = l_kv as u32;
        enc.set_bytes(4, 4, &lq_u as *const _ as *const _);
        enc.set_bytes(5, 4, &lkv_u as *const _ as *const _);
        enc.set_bytes(6, 4, &scale as *const _ as *const _);
        // Total KV rows across every head — the bound the split kernel clamps
        // its direct device reads against. Harmless for the tiled kernel,
        // which ignores buffer(7).
        let kv_rows_u = (n * h * l_kv) as u32;
        enc.set_bytes(7, 4, &kv_rows_u as *const _ as *const _);
        // Keys per head actually in play; read only by the register kernel.
        let kv_valid_u = kv_valid as u32;
        enc.set_bytes(8, 4, &kv_valid_u as *const _ as *const _);
        enc.set_threadgroup_memory_length(0, smem_bytes);
        let groups = (n * h * q_blocks) as u64;
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(tpg as u64, 1, 1));
        enc.end_encoding();
    });
    true
}

/// Flash attention with the best available kernel: tiled when the head dim has
/// a compiled pipeline, the scalar kernel otherwise. Identical semantics either
/// way — both read bf16 and accumulate the online softmax in f32.
pub fn flash_attention_best_f16_gpu(
    q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, o: &mut GpuTensor,
    n: usize, h: usize, l_q: usize, l_kv: usize, d: usize, scale: f32,
) {
    if !flash_attention_tiled_f16_gpu(q, k, v, o, n, h, l_q, l_kv, d, scale) {
        flash_attention_f16_gpu(q, k, v, o, n, h, l_q, l_kv, d, scale);
    }
}

/// Permute fp16 [N, L, H*D] → [N, H, L, D]. Returns fresh GpuTensor.
pub fn permute_lh_to_hl_f16_gpu(
    src: &GpuTensor, n: usize, l: usize, h: usize, d: usize,
) -> GpuTensor {
    // bfloat or IEEE f16 storage; same kernel body either way.
    let pipe = if src.dtype.is_half() { &get_metal().permute_lh_to_hl_pipe_h } else { &get_metal().permute_lh_to_hl_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(src.dtype, GpuDtype::F16 | GpuDtype::Half));
    let backend = get_metal();
    let dst = GpuTensor::new_like(src, vec![n, h, l, d]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&dst.buffer), 0);
        let n_u = n as u32; let l_u = l as u32; let h_u = h as u32; let d_u = d as u32;
        enc.set_bytes(2, 4, &n_u as *const _ as *const _);
        enc.set_bytes(3, 4, &l_u as *const _ as *const _);
        enc.set_bytes(4, 4, &h_u as *const _ as *const _);
        enc.set_bytes(5, 4, &d_u as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(dst.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    dst
}

/// Strided permute. Reads `src` as [N, L, inner_stride] (in f16 elements)
/// at the given byte offset and writes a fresh [N, H, L, D] GpuTensor.
/// Used for QKV-fused self-attention: the matmul produces a single
/// [N, L, 3·H·D] buffer, and Q/K/V slices are split out by passing
/// `inner_stride = 3·H·D` and offset = 0 / H·D / 2·H·D (in f16 bytes).
pub fn permute_lh_to_hl_f16_gpu_strided(
    src: &Buffer, src_byte_offset: usize,
    n: usize, l: usize, h: usize, d: usize, inner_stride: usize,
) -> GpuTensor {
    use metal::MTLSize;
    let backend = get_metal();
    let dst = GpuTensor::new_f16(vec![n, h, l, d]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.permute_lh_to_hl_strided_pipe);
        enc.set_buffer(0, Some(src), src_byte_offset as u64);
        enc.set_buffer(1, Some(&dst.buffer), 0);
        let n_u = n as u32; let l_u = l as u32; let h_u = h as u32;
        let d_u = d as u32; let stride_u = inner_stride as u32;
        enc.set_bytes(2, 4, &n_u as *const _ as *const _);
        enc.set_bytes(3, 4, &l_u as *const _ as *const _);
        enc.set_bytes(4, 4, &h_u as *const _ as *const _);
        enc.set_bytes(5, 4, &d_u as *const _ as *const _);
        enc.set_bytes(6, 4, &stride_u as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(dst.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    dst
}

/// Inverse permute fp16 [N, H, L, D] → [N, L, H*D]. Returns fresh GpuTensor.
pub fn permute_hl_to_lh_f16_gpu(
    src: &GpuTensor, n: usize, l: usize, h: usize, d: usize,
) -> GpuTensor {
    // bfloat or IEEE f16 storage; same kernel body either way.
    let pipe = if src.dtype.is_half() { &get_metal().permute_hl_to_lh_pipe_h } else { &get_metal().permute_hl_to_lh_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(src.dtype, GpuDtype::F16 | GpuDtype::Half));
    let backend = get_metal();
    let dst = GpuTensor::new_like(src, vec![n, l, h * d]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&dst.buffer), 0);
        let n_u = n as u32; let l_u = l as u32; let h_u = h as u32; let d_u = d as u32;
        enc.set_bytes(2, 4, &n_u as *const _ as *const _);
        enc.set_bytes(3, 4, &l_u as *const _ as *const _);
        enc.set_bytes(4, 4, &h_u as *const _ as *const _);
        enc.set_bytes(5, 4, &d_u as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(dst.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    dst
}

/// Fused GeGLU split: input fp16 `[n_rows, 2*hidden]` → output fp16
/// `[n_rows, hidden]` where `out[i,j] = in[i,j] * gelu(in[i, hidden+j])`.
/// Single kernel dispatch — fuses what would otherwise be three ops
/// (split, GELU, elementwise multiply).
pub fn geglu_split_f16_gpu(input: &GpuTensor, n_rows: usize, hidden: usize) -> GpuTensor {
    use metal::MTLSize;
    debug_assert_eq!(input.dtype, GpuDtype::F16);
    debug_assert_eq!(input.elements(), n_rows * 2 * hidden);
    let backend = get_metal();
    let out = GpuTensor::new_f16(vec![n_rows, hidden]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.geglu_split_pipe);
        enc.set_buffer(0, Some(&input.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        let h_u = hidden as u32;
        enc.set_bytes(2, 4, &h_u as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(out.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    out
}

/// fp16 NCHW→NHWC permute. Returns fresh GpuTensor.
pub fn nchw_to_nhwc_f16_gpu(src: &GpuTensor, n: usize, c: usize, hw: usize) -> GpuTensor {
    use metal::MTLSize;
    debug_assert_eq!(src.dtype, GpuDtype::F16);
    let backend = get_metal();
    let dst = GpuTensor::new_f16(vec![n, hw, c]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.nchw_to_nhwc_pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&dst.buffer), 0);
        let c_u = c as u32; let hw_u = hw as u32;
        enc.set_bytes(2, 4, &c_u as *const _ as *const _);
        enc.set_bytes(3, 4, &hw_u as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(dst.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    dst
}

/// fp16 NHWC→NCHW permute (inverse). Returns fresh GpuTensor.
pub fn nhwc_to_nchw_f16_gpu(src: &GpuTensor, n: usize, c: usize, hw: usize) -> GpuTensor {
    use metal::MTLSize;
    debug_assert_eq!(src.dtype, GpuDtype::F16);
    let backend = get_metal();
    let dst = GpuTensor::new_f16(vec![n, c, hw]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.nhwc_to_nchw_pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&dst.buffer), 0);
        let c_u = c as u32; let hw_u = hw as u32;
        enc.set_bytes(2, 4, &c_u as *const _ as *const _);
        enc.set_bytes(3, 4, &hw_u as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(dst.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    dst
}

/// Temporal reshape: [B*F, C, H, W] → [B*H*W, F, C]. bf16.
pub fn temporal_reshape_to_f16_gpu(
    src: &GpuTensor, batch: usize, frames: usize, c: usize, h: usize, w: usize,
) -> GpuTensor {
    use metal::MTLSize;
    debug_assert_eq!(src.dtype, GpuDtype::F16);
    let dst = GpuTensor::new_f16(vec![batch * h * w, frames, c]);
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.temporal_reshape_to_pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&dst.buffer), 0);
        let b = batch as u32; let f = frames as u32;
        let cu = c as u32; let hu = h as u32; let wu = w as u32;
        enc.set_bytes(2, 4, &b as *const _ as *const _);
        enc.set_bytes(3, 4, &f as *const _ as *const _);
        enc.set_bytes(4, 4, &cu as *const _ as *const _);
        enc.set_bytes(5, 4, &hu as *const _ as *const _);
        enc.set_bytes(6, 4, &wu as *const _ as *const _);
        let grid = MTLSize::new(dst.elements() as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    dst
}

/// Inverse temporal reshape: [B*H*W, F, C] → [B*F, C, H, W]. bf16.
pub fn temporal_reshape_from_f16_gpu(
    src: &GpuTensor, batch: usize, frames: usize, c: usize, h: usize, w: usize,
) -> GpuTensor {
    use metal::MTLSize;
    debug_assert_eq!(src.dtype, GpuDtype::F16);
    let dst = GpuTensor::new_f16(vec![batch * frames, c, h, w]);
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.temporal_reshape_from_pipe);
        enc.set_buffer(0, Some(&src.buffer), 0);
        enc.set_buffer(1, Some(&dst.buffer), 0);
        let b = batch as u32; let f = frames as u32;
        let cu = c as u32; let hu = h as u32; let wu = w as u32;
        enc.set_bytes(2, 4, &b as *const _ as *const _);
        enc.set_bytes(3, 4, &f as *const _ as *const _);
        enc.set_bytes(4, 4, &cu as *const _ as *const _);
        enc.set_bytes(5, 4, &hu as *const _ as *const _);
        enc.set_bytes(6, 4, &wu as *const _ as *const _);
        let grid = MTLSize::new(dst.elements() as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    dst
}

/// Add learned positional embedding in-place. x: [n_seq, frames, d], pe: [max_f, d].
pub fn pos_embed_add_f16_gpu(
    x: &mut GpuTensor, pe_buf: &metal::Buffer, frames: usize, max_frames: usize, d: usize,
) {
    use metal::MTLSize;
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.pos_embed_add_pipe);
        enc.set_buffer(0, Some(&x.buffer), 0);
        enc.set_buffer(1, Some(pe_buf), 0);
        let f = frames as u32; let mf = max_frames as u32; let du = d as u32;
        enc.set_bytes(2, 4, &f as *const _ as *const _);
        enc.set_bytes(3, 4, &mf as *const _ as *const _);
        enc.set_bytes(4, 4, &du as *const _ as *const _);
        let grid = MTLSize::new(x.elements() as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// Channel-wise concat: `[N, Ca, HW]` and `[N, Cb, HW]` → `[N, Ca+Cb, HW]`,
/// fp16. Returns a fresh GpuTensor. Used by UNet up-path skip-connection merge.
pub fn cat_channels_f16_gpu(
    a: &GpuTensor, ca: usize,
    b: &GpuTensor, cb: usize,
    n: usize, h: usize, w: usize,
) -> GpuTensor {
    use metal::MTLSize;
    debug_assert_eq!(a.dtype, GpuDtype::F16);
    debug_assert_eq!(b.dtype, GpuDtype::F16);
    let backend = get_metal();
    let total_c = ca + cb;
    let hw = h * w;
    let out = GpuTensor::new_f16(vec![n, total_c, h, w]);

    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.cat_channels_pipe);
        enc.set_buffer(0, Some(&a.buffer), 0);
        enc.set_buffer(1, Some(&b.buffer), 0);
        enc.set_buffer(2, Some(&out.buffer), 0);
        let ca_u = ca as u32;
        let cb_u = cb as u32;
        let hw_u = hw as u32;
        enc.set_bytes(3, 4, &ca_u as *const _ as *const _);
        enc.set_bytes(4, 4, &cb_u as *const _ as *const _);
        enc.set_bytes(5, 4, &hw_u as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(out.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    out
}

/// Nearest-neighbor 2× upsample on fp16 GPU buffers. Returns a fresh
/// `[N, C, 2H, 2W]` GpuTensor. Used by `Upsample::forward_gpu`.
/// Zero-pad one row and one column onto the end of each `[h, w]` plane.
///
/// `c` counts planes, so a `[C, T, H, W]` volume passes `C * T`.
/// `avg_down_3d` on the GPU — the encoder down block's averaged shortcut.
#[allow(clippy::too_many_arguments)]
pub fn avg_down_3d_f16_gpu(
    input: &GpuTensor,
    in_c: usize, in_t: usize, in_h: usize, in_w: usize,
    out_c: usize, factor_t: usize, factor_s: usize,
) -> (GpuTensor, usize, usize, usize) {
    use metal::MTLSize;
    debug_assert!(matches!(input.dtype, GpuDtype::F16 | GpuDtype::Half));
    let pipe = if input.dtype.is_half() {
        &get_metal().avg_down_3d_h_pipe
    } else {
        &get_metal().avg_down_3d_pipe
    };
    let pad_t = (factor_t - in_t % factor_t) % factor_t;
    let (ot, oh, ow) = ((in_t + pad_t) / factor_t, in_h / factor_s, in_w / factor_s);
    let out = GpuTensor::new_like(input, vec![out_c, ot, oh, ow]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&input.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        for (i, v) in [in_c, in_t, in_h, in_w, out_c, factor_t, factor_s, pad_t]
            .iter()
            .enumerate()
        {
            let u = *v as u32;
            enc.set_bytes(2 + i as u64, 4, &u as *const _ as *const _);
        }
        enc.dispatch_threads(
            MTLSize::new(out.elements() as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    });
    (out, ot, oh, ow)
}

pub fn pad_end_1_f16_gpu(input: &GpuTensor, c: usize, h: usize, w: usize) -> GpuTensor {
    use metal::MTLSize;
    debug_assert!(matches!(input.dtype, GpuDtype::F16 | GpuDtype::Half));
    let pipe = if input.dtype.is_half() {
        &get_metal().pad_end_1_h_pipe
    } else {
        &get_metal().pad_end_1_pipe
    };
    let out = GpuTensor::new_like(input, vec![c, h + 1, w + 1]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&input.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        let (c_u, h_u, w_u) = (c as u32, h as u32, w as u32);
        enc.set_bytes(2, 4, &c_u as *const _ as *const _);
        enc.set_bytes(3, 4, &h_u as *const _ as *const _);
        enc.set_bytes(4, 4, &w_u as *const _ as *const _);
        enc.dispatch_threads(
            MTLSize::new(out.elements() as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    });
    out
}

pub fn nearest_upsample_2x_f16_gpu(
    input: &GpuTensor, n: usize, c: usize, h: usize, w: usize,
) -> GpuTensor {
    // bfloat or IEEE f16 storage; the kernels are otherwise identical.
    let pipe = if input.dtype.is_half() { &get_metal().upsample_2x_h_pipe } else { &get_metal().upsample_2x_pipe };
    use metal::MTLSize;
    debug_assert!(matches!(input.dtype, GpuDtype::F16 | GpuDtype::Half));
    let backend = get_metal();
    let h_up = h * 2;
    let w_up = w * 2;
    let out = GpuTensor::new_like(input, vec![n, c, h_up, w_up]);

    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&input.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        let c_u = c as u32;
        let h_u = h as u32;
        let w_u = w as u32;
        enc.set_bytes(2, 4, &c_u as *const _ as *const _);
        enc.set_bytes(3, 4, &h_u as *const _ as *const _);
        enc.set_bytes(4, 4, &w_u as *const _ as *const _);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(out.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
    out
}

impl GpuTensor {
    /// Clone the tensor into a fresh GPU buffer (data + shape + dtype).
    /// Required when the caller needs the original after an in-place op.
    /// On M1 unified memory this is a single memcpy via `.contents()`.
    /// GPU→GPU copy via blit encoder. Avoids CPU `contents()` access entirely
    /// — critical with async-committed kernels where the source may still
    /// have writes in flight (a CPU memcpy would race; a blit serializes
    /// after prior queued work via Metal's hazard tracking).
    pub fn clone_data(&self) -> Self {
        let elements = self.elements();
        let bytes = elements * self.dtype.bytes();
        let buf = acquire_buffer(bytes);
        let backend = get_metal();
        with_cmd(|cmd| {
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(&self.buffer, 0, &buf, 0, bytes as u64);
            blit.end_encoding();
        });
        Self { buffer: buf, shape: self.shape.clone(), dtype: self.dtype }
    }
}

/// Release a buffer back to the shared pool.
pub fn release_pool_buffer(buf: Buffer) {
    release_buffer(buf);
}

// GpuTensor deliberately has no Drop impl. metal::Buffer is itself a
// reference-counted Objective-C handle; when GpuTensor drops, the Buffer
// drop runs and releases its retain. Returning buffers to the pool would
// require taking ownership out of `self` — Rust forbids that in Drop
// without unsafe. Pool reuse is therefore opt-in via direct
// release_buffer() calls from sgemm_metal / groupnorm_metal / etc., which
// know when a buffer is done.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_download_roundtrip_f16() {
        let v: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let t = GpuTensor::upload_f32_as_f16(vec![32, 32], &v);
        let back = t.download_to_f32();
        // f16 has 7-bit mantissa (vs fp16's 10) — for values in ±5 range
        // worst-case quantization is ~5 * 2^-7 ≈ 0.04. Tolerance reflects that.
        for (got, want) in back.iter().zip(v.iter()) {
            assert!((got - want).abs() < 0.05,
                "f32→f16→f32 roundtrip diverged: got {got}, want {want}");
        }
    }

    #[test]
    fn silu_f16_gpu_matches_cpu() {
        let v: Vec<f32> = (0..512).map(|i| (i as f32) * 0.1 - 25.0).collect();
        let mut t = GpuTensor::upload_f32_as_f16(vec![v.len()], &v);
        silu_f16_gpu(&mut t);
        let got = t.download_to_f32();
        for (g, x) in got.iter().zip(v.iter()) {
            let want = x / (1.0 + (-x).exp());
            // f16 worst-case for ±25-magnitude values is ~25 * 2^-7 ≈ 0.2.
            assert!((g - want).abs() < 0.25,
                "silu_f16_gpu diverged: x={x}, got={g}, want={want}");
        }
    }

    #[test]
    fn flux_kernels_match_cpu() {
        // ---- gelu_erf (exact GELU) ----
        let v: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
        let mut t = GpuTensor::upload_f32_as_f16(vec![v.len()], &v);
        gelu_erf_f16_gpu(&mut t);
        let got = t.download_to_f32();
        let erf = |x: f32| {
            let s = if x < 0.0 { -1.0 } else { 1.0 };
            let xa = x.abs();
            let tt = 1.0 / (1.0 + 0.3275911 * xa);
            let y = 1.0
                - (((((1.061405429 * tt - 1.453152027) * tt) + 1.421413741) * tt - 0.284496736) * tt
                    + 0.254829592)
                    * tt
                    * (-xa * xa).exp();
            s * y
        };
        for (g, &x) in got.iter().zip(v.iter()) {
            let want = 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2));
            assert!((g - want).abs() < 0.05, "gelu_erf x={x} got={g} want={want}");
        }

        // ---- modulate: x*(1+scale)+shift ----
        let (n, d) = (5usize, 16usize);
        let x: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.03).sin()).collect();
        let scale: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).cos() * 0.5).collect();
        let shift: Vec<f32> = (0..d).map(|i| (i as f32 * 0.2).sin() * 0.3).collect();
        let mut xt = GpuTensor::upload_f32_as_f16(vec![n, d], &x);
        let sc_buf = upload_f32_as_f16_buffer(&scale);
        let sh_buf = upload_f32_as_f16_buffer(&shift);
        modulate_f16_gpu(&mut xt, &sc_buf, 0, &sh_buf, 0, n, d);
        let got = xt.download_to_f32();
        for t in 0..n {
            for di in 0..d {
                let want = x[t * d + di] * (1.0 + scale[di]) + shift[di];
                assert!((got[t * d + di] - want).abs() < 0.02, "modulate diverged");
            }
        }

        // ---- gate_mul: x*gate ----
        let gate: Vec<f32> = (0..d).map(|i| (i as f32 * 0.15).sin()).collect();
        let mut xt2 = GpuTensor::upload_f32_as_f16(vec![n, d], &x);
        let g_buf = upload_f32_as_f16_buffer(&gate);
        gate_mul_f16_gpu(&mut xt2, &g_buf, 0, n, d);
        let got2 = xt2.download_to_f32();
        for t in 0..n {
            for di in 0..d {
                let want = x[t * d + di] * gate[di];
                assert!((got2[t * d + di] - want).abs() < 0.02, "gate_mul diverged");
            }
        }

        // ---- rope_pairs (interleaved 2k/2k+1) ----
        let (nt, nh, dh) = (4usize, 2usize, 8usize);
        let xq: Vec<f32> = (0..nt * nh * dh).map(|i| (i as f32 * 0.07).cos()).collect();
        // cos/sin [nt, dh] with cos[2k]==cos[2k+1] (build_rope_2d convention).
        let mut cos = vec![0.0f32; nt * dh];
        let mut sin = vec![0.0f32; nt * dh];
        for tk in 0..nt {
            for k in 0..dh / 2 {
                let ang = (tk as f32 + 1.0) * 0.3 * (k as f32 + 1.0);
                cos[tk * dh + 2 * k] = ang.cos();
                cos[tk * dh + 2 * k + 1] = ang.cos();
                sin[tk * dh + 2 * k] = ang.sin();
                sin[tk * dh + 2 * k + 1] = ang.sin();
            }
        }
        let mut qt = GpuTensor::upload_f32_as_f16(vec![nt * nh * dh], &xq);
        let cos_buf = upload_f32_as_f16_buffer(&cos);
        let sin_buf = upload_f32_as_f16_buffer(&sin);
        rope_pairs_f16_gpu(&mut qt, &cos_buf, &sin_buf, nt, nh, dh);
        let gotq = qt.download_to_f32();
        let mut want = xq.clone();
        for tk in 0..nt {
            for h in 0..nh {
                let off = (tk * nh + h) * dh;
                for k in 0..dh / 2 {
                    let a = xq[off + 2 * k];
                    let b = xq[off + 2 * k + 1];
                    let c = cos[tk * dh + 2 * k];
                    let s = sin[tk * dh + 2 * k];
                    want[off + 2 * k] = a * c - b * s;
                    want[off + 2 * k + 1] = a * s + b * c;
                }
            }
        }
        for (g, w) in gotq.iter().zip(want.iter()) {
            assert!((g - w).abs() < 0.02, "rope_pairs diverged got={g} want={w}");
        }
    }

    #[test]
    fn eadd_f16_gpu_matches_cpu() {
        let a: Vec<f32> = (0..256).map(|i| i as f32 * 0.05).collect();
        let b: Vec<f32> = (0..256).map(|i| (256 - i) as f32 * 0.07).collect();
        let mut ta = GpuTensor::upload_f32_as_f16(vec![256], &a);
        let tb = GpuTensor::upload_f32_as_f16(vec![256], &b);
        eadd_f16_gpu(&mut ta, &tb);
        let got = ta.download_to_f32();
        for (i, (g, _)) in got.iter().zip(a.iter()).enumerate() {
            let want = a[i] + b[i];
            // f16 quantization for sums up to ~30 is ~0.25.
            assert!((g - want).abs() < 0.3,
                "eadd diverged at {i}: got={g}, want={want}");
        }
    }

    #[test]
    fn sgemm_f16_gpu_matches_cpu() {
        // Small matmul end-to-end on GPU.
        let m = 16; let n = 16; let k = 16;
        let a_f32: Vec<f32> = (0..m*k).map(|i| ((i * 7) % 13) as f32 / 13.0 - 0.5).collect();
        let b_f32: Vec<f32> = (0..k*n).map(|i| ((i * 11) % 17) as f32 / 17.0 - 0.5).collect();

        // Reference: f32 sgemm.
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k { s += a_f32[i*k + kk] * b_f32[kk*n + j]; }
            c_ref[i*n + j] = s;
        }}

        // GPU-resident: upload, run, download.
        let a_gpu = GpuTensor::upload_f32_as_f16(vec![m, k], &a_f32);
        let b_gpu = GpuTensor::upload_f32_as_f16(vec![k, n], &b_f32);
        let mut c_gpu = GpuTensor::new_f16(vec![m, n]);
        sgemm_f16_gpu(&mut c_gpu, &a_gpu, &b_gpu, m, n, k);
        let c_got = c_gpu.download_to_f32();

        let mut max_diff = 0.0f32;
        for (g, w) in c_got.iter().zip(c_ref.iter()) {
            let d = (g - w).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 0.05,
            "sgemm_f16_gpu diverged from f32 reference: max_diff={max_diff}");
    }

    #[test]
    fn layer_norm_f16_out_matches_inplace() {
        // Compares out-of-place LN against in-place LN for the same input.
        // Catches bugs where the new kernel diverges from the in-place reference.
        let n_rows = 7; let dim = 257;  // intentionally non-power-of-2 dim
        let x_f32: Vec<f32> = (0..n_rows*dim)
            .map(|i| ((i * 13) % 19) as f32 / 19.0 - 0.4).collect();
        let gamma: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 * 0.003)).collect();
        let beta:  Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002) - 0.1).collect();
        let g_buf = weight_f16_buffer(&gamma);
        let b_buf = weight_f16_buffer(&beta);

        // In-place reference.
        let mut x_in = GpuTensor::upload_f32_as_f16(vec![n_rows, dim], &x_f32);
        layer_norm_f16_gpu(&mut x_in, &g_buf, &b_buf, dim, 1e-5);
        let want = x_in.download_to_f32();

        // Out-of-place: source unchanged, fresh dst.
        let src = GpuTensor::upload_f32_as_f16(vec![n_rows, dim], &x_f32);
        let dst = layer_norm_f16_gpu_out(&src, &g_buf, &b_buf, dim, 1e-5);
        let got = dst.download_to_f32();

        let mut max_diff = 0.0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            let d = (g - w).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 1e-3,
            "layer_norm_f16_gpu_out diverges from in-place: max_diff={max_diff}");
    }

    #[test]
    fn groupnorm_silu_f16_out_matches_split() {
        // Verifies fused GroupNorm+SiLU (out-of-place) matches groupnorm+silu
        // applied separately. Tests both out-of-place and in-place variants.
        let n = 1; let c = 16; let h = 8; let w = 8;
        let groups = 4;
        let x_f32: Vec<f32> = (0..n*c*h*w)
            .map(|i| ((i * 11) % 17) as f32 / 17.0 - 0.4).collect();
        let gamma: Vec<f32> = (0..c).map(|i| 1.0 + (i as f32 * 0.01)).collect();
        let beta:  Vec<f32> = (0..c).map(|i| (i as f32 * 0.005) - 0.05).collect();
        let g_buf = weight_f16_buffer(&gamma);
        let b_buf = weight_f16_buffer(&beta);

        // Reference: separate groupnorm + silu (in-place).
        let mut x_ref = GpuTensor::upload_f32_as_f16(vec![n, c, h, w], &x_f32);
        groupnorm_f16_gpu(&mut x_ref, &g_buf, &b_buf, n, c, h, w, groups, 1e-5);
        silu_f16_gpu(&mut x_ref);
        let want = x_ref.download_to_f32();

        // Fused out-of-place.
        let src = GpuTensor::upload_f32_as_f16(vec![n, c, h, w], &x_f32);
        let dst = groupnorm_silu_f16_gpu_out(&src, &g_buf, &b_buf, n, c, h, w, groups, 1e-5);
        let got_out = dst.download_to_f32();

        let mut max_diff = 0.0f32;
        for (g, w) in got_out.iter().zip(want.iter()) {
            let d = (g - w).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 5e-2,
            "groupnorm_silu_out vs split: max_diff={max_diff}");

        // Fused in-place (src == dst).
        let mut x_inp = GpuTensor::upload_f32_as_f16(vec![n, c, h, w], &x_f32);
        groupnorm_silu_f16_gpu_inplace(&mut x_inp, &g_buf, &b_buf, n, c, h, w, groups, 1e-5);
        let got_inp = x_inp.download_to_f32();

        let mut max_diff = 0.0f32;
        for (g, w) in got_inp.iter().zip(want.iter()) {
            let d = (g - w).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 5e-2,
            "groupnorm_silu_inplace vs split: max_diff={max_diff}");
    }

    #[test]
    fn fused_norms_at_sdxl_scale() {
        // Catch scale-dependent bugs. SDXL ResnetBlock at stage-1 has
        // c=320, h=w=128, num_groups=32 → group_size=163840.
        // BasicTransformerBlock LN at stage-2 has seq=4096, dim=640.
        let n = 2;

        // GroupNorm + SiLU at SDXL stage-1 scale.
        let c = 320; let h = 32; let w = 32;  // smaller spatial to keep test fast
        let groups = 32;
        let len = n * c * h * w;
        let x_f32: Vec<f32> = (0..len).map(|i| ((i * 13) % 19) as f32 / 19.0 - 0.4).collect();
        let gamma: Vec<f32> = (0..c).map(|i| 1.0 + (i as f32 * 0.001)).collect();
        let beta:  Vec<f32> = (0..c).map(|i| (i as f32 * 0.0005) - 0.05).collect();
        let g_buf = weight_f16_buffer(&gamma);
        let b_buf = weight_f16_buffer(&beta);

        // Reference: split groupnorm + silu, in-place.
        let mut x_ref = GpuTensor::upload_f32_as_f16(vec![n, c, h, w], &x_f32);
        groupnorm_f16_gpu(&mut x_ref, &g_buf, &b_buf, n, c, h, w, groups, 1e-5);
        silu_f16_gpu(&mut x_ref);
        let want = x_ref.download_to_f32();

        // Fused out-of-place.
        let src = GpuTensor::upload_f32_as_f16(vec![n, c, h, w], &x_f32);
        let dst = groupnorm_silu_f16_gpu_out(&src, &g_buf, &b_buf, n, c, h, w, groups, 1e-5);
        let got_out = dst.download_to_f32();
        let mut max_diff = 0.0f32;
        for (g, w) in got_out.iter().zip(want.iter()) {
            let d = (g - w).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 5e-2,
            "groupnorm_silu_out at SDXL scale: max_diff={max_diff}");

        // Fused in-place.
        let mut x_inp = GpuTensor::upload_f32_as_f16(vec![n, c, h, w], &x_f32);
        groupnorm_silu_f16_gpu_inplace(&mut x_inp, &g_buf, &b_buf, n, c, h, w, groups, 1e-5);
        let got_inp = x_inp.download_to_f32();
        let mut max_diff = 0.0f32;
        for (g, w) in got_inp.iter().zip(want.iter()) {
            let d = (g - w).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 5e-2,
            "groupnorm_silu_inplace at SDXL scale: max_diff={max_diff}");

        // LayerNorm at SDXL transformer scale.
        let seq = 4096; let dim = 640;
        let x_f32: Vec<f32> = (0..n*seq*dim).map(|i| ((i * 7) % 11) as f32 / 11.0 - 0.4).collect();
        let gamma: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
        let beta:  Vec<f32> = (0..dim).map(|i| (i as f32 * 0.0001) - 0.02).collect();
        let g_buf = weight_f16_buffer(&gamma);
        let b_buf = weight_f16_buffer(&beta);
        let mut x_ref = GpuTensor::upload_f32_as_f16(vec![n, seq, dim], &x_f32);
        layer_norm_f16_gpu(&mut x_ref, &g_buf, &b_buf, dim, 1e-5);
        let want = x_ref.download_to_f32();
        let src = GpuTensor::upload_f32_as_f16(vec![n, seq, dim], &x_f32);
        let dst = layer_norm_f16_gpu_out(&src, &g_buf, &b_buf, dim, 1e-5);
        let got = dst.download_to_f32();
        let mut max_diff = 0.0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            let d = (g - w).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 5e-3,
            "layer_norm_f16_out at SDXL scale: max_diff={max_diff}");
    }

    #[test]
    fn rms_norm_metal_matches_cpu() {
        let dim = 128;
        let n_rows = 32;
        let mut x: Vec<f32> = (0..n_rows * dim)
            .map(|i| ((i * 13) % 97) as f32 / 50.0 - 0.7)
            .collect();
        let gamma: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 * 0.003)).collect();
        let eps = 1e-5_f32;

        // CPU reference.
        let mut want = x.clone();
        for r in 0..n_rows {
            let row = &mut want[r * dim..(r + 1) * dim];
            let mut s = 0.0_f64;
            for &v in row.iter() { s += (v as f64) * (v as f64); }
            let inv = ((s / dim as f64) + eps as f64).sqrt().recip() as f32;
            for (xi, &g) in row.iter_mut().zip(gamma.iter()) {
                *xi = *xi * inv * g;
            }
        }
        // GPU.
        rms_norm_metal(&mut x, &gamma, dim, eps);

        let mut max_diff = 0.0_f32;
        for (g, w) in x.iter().zip(want.iter()) {
            let d = (g - w).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 1e-4,
            "rms_norm_metal vs CPU max_diff={max_diff}");
    }

    #[test]
    fn flash_attention_causal_metal_matches_cpu() {
        // Small problem: 1 batch, 2 heads, l_q = l_kv = 8, d = 16.
        let n = 1; let h = 2; let l_q = 8; let l_kv = 8; let d = 16;
        let total = n * h * l_q * d;
        let q: Vec<f32> = (0..total).map(|i| ((i as i32 % 23) - 11) as f32 * 0.07).collect();
        let k: Vec<f32> = (0..total).map(|i| ((i as i32 % 29) - 14) as f32 * 0.06).collect();
        let v: Vec<f32> = (0..total).map(|i| ((i as i32 % 31) - 15) as f32 * 0.05).collect();
        let scale = 1.0_f32 / (d as f32).sqrt();

        // CPU causal reference.
        let mut want = vec![0.0_f32; total];
        for nh in 0..(n * h) {
            let qkv_base = nh * l_q * d;
            for qi in 0..l_q {
                let q_row = &q[qkv_base + qi * d..qkv_base + (qi + 1) * d];
                let mut scores = vec![0.0_f32; qi + 1];
                for kj in 0..=qi {
                    let k_row = &k[qkv_base + kj * d..qkv_base + (kj + 1) * d];
                    let mut s = 0.0_f32;
                    for di in 0..d { s += q_row[di] * k_row[di]; }
                    scores[kj] = s * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
                let inv = 1.0 / sum;
                for s in scores.iter_mut() { *s *= inv; }
                for kj in 0..=qi {
                    let v_row = &v[qkv_base + kj * d..qkv_base + (kj + 1) * d];
                    let w = scores[kj];
                    let out_row = &mut want[qkv_base + qi * d..qkv_base + (qi + 1) * d];
                    for di in 0..d { out_row[di] += w * v_row[di]; }
                }
            }
        }

        let mut got = vec![0.0_f32; total];
        flash_attention_causal_metal(&q, &k, &v, &mut got, n, h, l_q, l_kv, d, scale);

        let mut max_diff = 0.0_f32;
        for (g, w) in got.iter().zip(want.iter()) {
            let d = (g - w).abs();
            if d > max_diff { max_diff = d; }
        }
        assert!(max_diff < 1e-4,
            "flash_attention_causal_metal vs CPU max_diff={max_diff}");
    }
    // -- tiled flash attention -------------------------------------------

    /// Deterministic pseudo-random fill; distinct `seed` gives distinct data.
    fn fa_fill(len: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                ((x >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    /// f32 reference for non-causal attention, matching both GPU kernels.
    fn fa_reference(
        q: &[f32], k: &[f32], v: &[f32],
        n: usize, h: usize, l_q: usize, l_kv: usize, d: usize, scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; n * h * l_q * d];
        for nh in 0..(n * h) {
            let qb = nh * l_q * d;
            let kb = nh * l_kv * d;
            for qi in 0..l_q {
                let mut scores = vec![0.0_f32; l_kv];
                for kj in 0..l_kv {
                    let mut acc = 0.0_f32;
                    for di in 0..d {
                        acc += q[qb + qi * d + di] * k[kb + kj * d + di];
                    }
                    scores[kj] = acc * scale;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                for kj in 0..l_kv {
                    let w = scores[kj] / sum;
                    for di in 0..d {
                        out[qb + qi * d + di] += w * v[kb + kj * d + di];
                    }
                }
            }
        }
        out
    }

    /// The tiled kernel must agree with both the scalar GPU kernel it replaces
    /// and an independent f32 CPU reference, across head dims and across
    /// sequence lengths that do and do not divide the 32x16 tiling.
    #[test]
    fn flash_attention_tiled_matches_scalar_and_cpu() {
        // (n, h, l_q, l_kv) — covers exact tile multiples, ragged tails in Q
        // and in KV independently, and lengths shorter than one tile.
        let shapes = [
            (1, 1, 32, 16),
            (1, 1, 33, 17),
            (1, 1, 7, 5),
            (1, 1, 1, 1),
            (2, 3, 64, 48),
            (1, 2, 100, 70),
            (1, 1, 16, 129),
        ];
        for d in FLASH_ATTN_TILED_DIMS {
            for (n, h, l_q, l_kv) in shapes {
                let scale = 1.0 / (d as f32).sqrt();
                let q = fa_fill(n * h * l_q * d, d as u32 * 7 + 1);
                let k = fa_fill(n * h * l_kv * d, d as u32 * 7 + 2);
                let v = fa_fill(n * h * l_kv * d, d as u32 * 7 + 3);

                let qt = GpuTensor::upload_f32_as_f16(vec![n * h * l_q * d], &q);
                let kt = GpuTensor::upload_f32_as_f16(vec![n * h * l_kv * d], &k);
                let vt = GpuTensor::upload_f32_as_f16(vec![n * h * l_kv * d], &v);

                let mut o_tiled = GpuTensor::new_f16(vec![n * h * l_q * d]);
                assert!(
                    flash_attention_tiled_f16_gpu(
                        &qt, &kt, &vt, &mut o_tiled, n, h, l_q, l_kv, d, scale),
                    "no tiled pipeline for d={d}"
                );
                let got_tiled = o_tiled.download_to_f32();

                let mut o_scalar = GpuTensor::new_f16(vec![n * h * l_q * d]);
                flash_attention_f16_gpu(
                    &qt, &kt, &vt, &mut o_scalar, n, h, l_q, l_kv, d, scale);
                let got_scalar = o_scalar.download_to_f32();

                let want = fa_reference(&q, &k, &v, n, h, l_q, l_kv, d, scale);

                let worst = |a: &[f32], b: &[f32]| {
                    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
                };
                let vs_scalar = worst(&got_tiled, &got_scalar);
                let vs_cpu = worst(&got_tiled, &want);
                // Storage is bf16 (~3 decimal digits), so agreement is limited
                // by the output cast, not by the accumulation order.
                assert!(
                    vs_scalar < 8e-3,
                    "d={d} shape=({n},{h},{l_q},{l_kv}) tiled vs scalar max_diff={vs_scalar}"
                );
                assert!(
                    vs_cpu < 8e-3,
                    "d={d} shape=({n},{h},{l_q},{l_kv}) tiled vs cpu max_diff={vs_cpu}"
                );
                assert!(
                    got_tiled.iter().all(|x| x.is_finite()),
                    "d={d} shape=({n},{h},{l_q},{l_kv}) produced non-finite output"
                );
            }
        }
    }

    /// Unsupported head dims must decline rather than dispatch garbage, so
    /// `flash_attention_best_f16_gpu` can fall back.
    #[test]
    fn flash_attention_tiled_declines_unsupported_head_dim() {
        let (n, h, l_q, l_kv, d) = (1, 1, 8, 8, 40);
        let q = fa_fill(n * h * l_q * d, 11);
        let k = fa_fill(n * h * l_kv * d, 12);
        let v = fa_fill(n * h * l_kv * d, 13);
        let qt = GpuTensor::upload_f32_as_f16(vec![n * h * l_q * d], &q);
        let kt = GpuTensor::upload_f32_as_f16(vec![n * h * l_kv * d], &k);
        let vt = GpuTensor::upload_f32_as_f16(vec![n * h * l_kv * d], &v);
        let mut o = GpuTensor::new_f16(vec![n * h * l_q * d]);
        let scale = 1.0 / (d as f32).sqrt();
        assert!(!flash_attention_tiled_f16_gpu(
            &qt, &kt, &vt, &mut o, n, h, l_q, l_kv, d, scale));

        // The auto-selecting entry point still produces the right answer.
        flash_attention_best_f16_gpu(&qt, &kt, &vt, &mut o, n, h, l_q, l_kv, d, scale);
        let want = fa_reference(&q, &k, &v, n, h, l_q, l_kv, d, scale);
        let worst = o.download_to_f32().iter().zip(&want)
            .map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
        assert!(worst < 8e-3, "fallback path max_diff={worst}");
    }
    /// Not a correctness check — prints the timing curve at Wan-like shapes
    /// (head dim 128, 24 heads).
    ///
    /// Reading the columns: `tiled` should be 4-6x `scalar`. `mpsgraph` is
    /// Apple's SDPA and is *faster* than both at these lengths — but that path
    /// materialises the full L x L x H score matrix (6.4 GB at 8192 tokens,
    /// ~70 GB at Wan's 27k), so its advantage collapses as L grows and it
    /// cannot run at video scale at all. Watch its throughput fall from ~4.8
    /// TFLOPS at 2048 to ~0.95 at 8192 while the tiled kernel stays flat.
    /// It also casts to f16 and so cannot be used for Qwen-family activations.
    ///
    /// Warm-up runs at the measured shape on purpose: MPSGraph caches compiled
    /// graphs keyed on shape, and warming at a different size leaves a graph
    /// compile inside the timed region.
    ///
    /// The (h, d) sweep holds h*d = 3072 so every configuration does identical
    /// total FLOPs and the only variable is fragment count. It is what
    /// diagnosed the register cliff: before the head-dim split, d=64 and d=32
    /// plateaued near 1180 GF/s while d=128 collapsed to ~708 because 32 live
    /// fragments spilled. With `flash_attention_split_impl` routing d>=96,
    /// d=128 reaches ~1114 and the sweep is flat. Keep this sweep flat — a
    /// dip at one head dim means fragments started spilling again.
    ///
    /// Run with:
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture flash_attention_tiled_speed
    #[test]
    #[ignore]
    fn flash_attention_tiled_speed() {
        use std::time::Instant;
        // h*d is held at 3072 so every (h, d) pair does identical total FLOPs;
        // any difference is register pressure, since fragment count is D/8.
        for (h, d) in [(24usize, 128usize), (48, 64), (96, 32)] {
        let n = 1usize;
        let scale = 1.0 / (d as f32).sqrt();
        println!("\n=== h={h} d={d} (fragments per simdgroup: 2*{}) ===", d / 8);
        println!("{:>8} {:>12} {:>12} {:>9} {:>12} {:>10}",
                 "tokens", "scalar (ms)", "tiled (ms)", "speedup", "mpsgraph(ms)",
                 "tiled GF/s");
        for &l in &[2048usize, 4096, 8192] {
            let q = fa_fill(n * h * l * d, 1);
            let k = fa_fill(n * h * l * d, 2);
            let v = fa_fill(n * h * l * d, 3);
            let qt = GpuTensor::upload_f32_as_f16(vec![n * h * l * d], &q);
            let kt = GpuTensor::upload_f32_as_f16(vec![n * h * l * d], &k);
            let vt = GpuTensor::upload_f32_as_f16(vec![n * h * l * d], &v);
            let mut o = GpuTensor::new_f16(vec![n * h * l * d]);

            // Warm up at the *measured* shape. MPSGraph caches compiled
            // graphs keyed on shape, so warming at a different size leaves a
            // full graph compile inside the timed region and makes MPSGraph
            // look several times slower than it is.
            set_force_flash_attn(false);
            flash_attention_tiled_f16_gpu(&qt, &kt, &vt, &mut o, n, h, l, l, d, scale);
            flash_attention_f16_gpu(&qt, &kt, &vt, &mut o, n, h, l, l, d, scale);
            mpsgraph_sdpa_bf16_gpu(&qt, &kt, &vt, &mut o, n, h, l, l, d, scale);
            flush();

            let t0 = Instant::now();
            flash_attention_f16_gpu(&qt, &kt, &vt, &mut o, n, h, l, l, d, scale);
            flush();
            let scalar_ms = t0.elapsed().as_secs_f64() * 1e3;

            let t1 = Instant::now();
            flash_attention_tiled_f16_gpu(&qt, &kt, &vt, &mut o, n, h, l, l, d, scale);
            flush();
            let tiled_ms = t1.elapsed().as_secs_f64() * 1e3;

            // Apple's own SDPA, as a ceiling reference. Not usable as the
            // default path (it casts to f16 and overflows on Qwen-family
            // activations) but it says what this GPU can actually do.
            let t2 = Instant::now();
            mpsgraph_sdpa_bf16_gpu(&qt, &kt, &vt, &mut o, n, h, l, l, d, scale);
            flush();
            let mps_ms = t2.elapsed().as_secs_f64() * 1e3;

            // 4*L^2*(h*d) FLOP: QK^T and PV, each 2*L*L*(h*d).
            let gflop = 4.0 * (l as f64) * (l as f64) * (h * d) as f64 / 1e9;
            println!("{l:>8} {scalar_ms:>12.1} {tiled_ms:>12.1} {:>8.1}x {mps_ms:>12.1} {:>10.0}",
                     scalar_ms / tiled_ms, gflop / (tiled_ms / 1e3));
        }
        }
    }
    /// Probe: does this Metal version expose mixed-precision simdgroup MMA
    /// (half inputs accumulating into float), the way CUDA's WMMA does? If it
    /// does, the tiled attention kernel can halve its tile memory and roughly
    /// double MMA throughput without giving up f32 accumulation.
    #[test]
    #[ignore]
    fn probe_mixed_precision_simdgroup_mma() {
        let backend = get_metal();
        let cases: &[(&str, &str)] = &[
            ("half in / float accum",
             "simdgroup_float8x8 d, c; simdgroup_half8x8 a, b; \
              simdgroup_multiply_accumulate(d, a, b, c);"),
            ("all half",
             "simdgroup_half8x8 d, a, b, c; \
              simdgroup_multiply_accumulate(d, a, b, c);"),
            ("bfloat matrix type",
             "simdgroup_matrix<bfloat, 8, 8> d, a, b, c; \
              simdgroup_multiply_accumulate(d, a, b, c);"),
            ("bfloat in / float accum",
             "simdgroup_float8x8 d, c; simdgroup_matrix<bfloat, 8, 8> a, b; \
              simdgroup_multiply_accumulate(d, a, b, c);"),
            ("simdgroup_load bfloat from threadgroup",
             "threadgroup bfloat tg[64]; simdgroup_matrix<bfloat, 8, 8> a; \
              simdgroup_load(a, tg, 8);"),
            ("simdgroup_load bfloat from device",
             "simdgroup_matrix<bfloat, 8, 8> a; \
              simdgroup_load(a, (const device bfloat*)out, 8);"),
            ("simdgroup_load bfloat transposed",
             "threadgroup bfloat tg[64]; simdgroup_matrix<bfloat, 8, 8> a; \
              simdgroup_load(a, tg, 8, ulong2(0,0), true);"),
        ];
        for (label, body) in cases {
            let src = format!(
                "#include <metal_stdlib>\nusing namespace metal;\n\
                 kernel void probe(device float* out [[buffer(0)]]) {{ {body} out[0] = 0; }}\n"
            );
            let opts = metal::CompileOptions::new();
            match backend.device.new_library_with_source(&src, &opts) {
                Ok(_) => println!("SUPPORTED   {label}"),
                Err(e) => {
                    let first = e.lines().find(|l| l.contains("error")).unwrap_or("");
                    println!("unsupported {label}: {}", first.trim());
                }
            }
        }
    }
    // -- Wan per-token modulation -----------------------------------------

    /// CPU reference for the three modulation kernels, written directly from
    /// the block's arithmetic in `wan::model`.
    fn mod_reference(
        x: &[f32], table: &[f32], temb: &[f32],
        n: usize, d: usize, per_token: bool, scale_i: usize, shift_i: usize,
    ) -> Vec<f32> {
        let mut out = x.to_vec();
        for t in 0..n {
            let base = if per_token { t * 6 * d } else { 0 };
            for i in 0..d {
                let scale = table[scale_i * d + i] + temb[base + scale_i * d + i];
                let shift = table[shift_i * d + i] + temb[base + shift_i * d + i];
                out[t * d + i] = out[t * d + i] * (1.0 + scale) + shift;
            }
        }
        out
    }

    #[test]
    fn wan_modulation_kernels_match_cpu() {
        let (n, d) = (7usize, 16usize);
        let x = fa_fill(n * d, 1);
        let table = fa_fill(6 * d, 2);

        for per_token in [false, true] {
            let temb = fa_fill(if per_token { n * 6 * d } else { 6 * d }, 3);
            let layout = if per_token { ModLayout::PerToken } else { ModLayout::PerSample };
            let tbuf = upload_f32_as_f16_buffer(&table);
            let ebuf = upload_f32_as_f16_buffer(&temb);

            // modulate: shift/scale pair (0,1) is attention, (3,4) is the FFN.
            for (shift_i, scale_i) in [(0usize, 1usize), (3, 4)] {
                let mut g = GpuTensor::upload_f32_as_f16(vec![n * d], &x);
                modulate_rows_f16_gpu(&mut g, &tbuf, &ebuf, n, d, layout, scale_i, shift_i);
                let got = g.download_to_f32();
                let want = mod_reference(&x, &table, &temb, n, d, per_token, scale_i, shift_i);
                let worst =
                    got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
                assert!(worst < 5e-2, "modulate per_token={per_token} pair=({shift_i},{scale_i}): {worst}");
            }

            // gate
            let mut g = GpuTensor::upload_f32_as_f16(vec![n * d], &x);
            gate_rows_f16_gpu(&mut g, &tbuf, &ebuf, n, d, layout, 2);
            let got = g.download_to_f32();
            let mut want = x.clone();
            for t in 0..n {
                let base = if per_token { t * 6 * d } else { 0 };
                for i in 0..d {
                    want[t * d + i] *= table[2 * d + i] + temb[base + 2 * d + i];
                }
            }
            let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            assert!(worst < 5e-2, "gate per_token={per_token}: {worst}");
        }
    }

    #[test]
    fn wan_final_modulation_adds_the_same_temb_to_shift_and_scale() {
        // The distinguishing property: one temb row feeds both, and only the
        // [2, d] table separates them.
        let (n, d) = (5usize, 12usize);
        let x = fa_fill(n * d, 4);
        let table = fa_fill(2 * d, 5);

        for per_token in [false, true] {
            let temb = fa_fill(if per_token { n * d } else { d }, 6);
            let layout = if per_token { ModLayout::PerToken } else { ModLayout::PerSample };
            let tbuf = upload_f32_as_f16_buffer(&table);
            let ebuf = upload_f32_as_f16_buffer(&temb);
            let mut g = GpuTensor::upload_f32_as_f16(vec![n * d], &x);
            final_modulate_f16_gpu(&mut g, &tbuf, &ebuf, n, d, layout);
            let got = g.download_to_f32();

            let mut want = x.clone();
            for t in 0..n {
                let base = if per_token { t * d } else { 0 };
                for i in 0..d {
                    let te = temb[base + i];
                    let shift = table[i] + te;
                    let scale = table[d + i] + te;
                    want[t * d + i] = want[t * d + i] * (1.0 + scale) + shift;
                }
            }
            let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            assert!(worst < 5e-2, "final modulate per_token={per_token}: {worst}");
        }
    }

    #[test]
    fn per_token_modulation_with_a_uniform_temb_equals_broadcast() {
        // The same invariant the CPU path has, at the kernel level: repeating
        // one modulation across every row must match the broadcast stride-0
        // path. Catches a wrong row stride, which otherwise only shows up as
        // subtly wrong video.
        let (n, d) = (6usize, 16usize);
        let x = fa_fill(n * d, 7);
        let table = fa_fill(6 * d, 8);
        let one = fa_fill(6 * d, 9);
        let repeated: Vec<f32> = (0..n).flat_map(|_| one.clone()).collect();

        let tbuf = upload_f32_as_f16_buffer(&table);
        let mut a = GpuTensor::upload_f32_as_f16(vec![n * d], &x);
        modulate_rows_f16_gpu(
            &mut a, &tbuf, &upload_f32_as_f16_buffer(&one), n, d, ModLayout::PerSample, 1, 0);
        let mut b = GpuTensor::upload_f32_as_f16(vec![n * d], &x);
        modulate_rows_f16_gpu(
            &mut b, &tbuf, &upload_f32_as_f16_buffer(&repeated), n, d, ModLayout::PerToken, 1, 0);
        assert_eq!(a.download_to_f32(), b.download_to_f32());
    }
    /// Where does the Wan DiT's time actually go? Times one sgemm at each of
    /// the shapes the 5B model uses, so the matmul cost can be separated from
    /// the surrounding dispatch and dtype-cast overhead.
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture wan_sgemm_shapes
    #[test]
    #[ignore]
    fn wan_sgemm_shapes() {
        use std::time::Instant;
        // (m = tokens, n = out, k = in), as used by a 1560-token forward.
        let shapes = [
            ("attn qkv/out  3072x3072", 1560usize, 3072usize, 3072usize),
            ("ffn up       14336x3072", 1560, 14336, 3072),
            ("ffn down     3072x14336", 1560, 3072, 14336),
        ];
        for f32_scratch in [false, true] {
            set_use_f32_sgemm(f32_scratch);
            println!("\n--- f32 scratch: {f32_scratch} ---");
            for (label, m, n, k) in shapes {
                let a = GpuTensor::upload_f32_as_f16(vec![m * k], &vec![0.01_f32; m * k]);
                let w = upload_f32_as_f16_buffer(&vec![0.01_f32; n * k]);
                let out = GpuTensor::new_f16(vec![m * n]);
                // Warm up (graph/pipeline creation out of the timed region).
                sgemm_f16_a_btrans_buf(&out.buffer, &a.buffer, &w, m, n, k);
                flush();

                let reps = 5;
                let t0 = Instant::now();
                for _ in 0..reps {
                    sgemm_f16_a_btrans_buf(&out.buffer, &a.buffer, &w, m, n, k);
                }
                flush();
                let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
                let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
                println!("{label}: {ms:8.2} ms  {:8.0} GFLOP/s", gflop / (ms / 1e3));
            }
        }
        set_use_f32_sgemm(false);
    }

    /// The native bf16 GEMM must agree with the MPS f32 route, which is the
    /// path already validated end-to-end against diffusers.
    ///
    /// Shapes deliberately include ragged M and K: Wan runs 5070 tokens, which
    /// is neither a multiple of the 64-row tile nor of the 32-deep K slice, so
    /// the edge handling is the common case here rather than a corner.
    #[test]
    fn bf16_gemm_matches_mps_f32() {
        let shapes = [
            (64usize, 64usize, 32usize),
            (64, 128, 64),
            (130, 96, 96),   // ragged M
            (65, 64, 33),    // ragged M and K
            (5070, 128, 96), // Wan's token count
        ];
        let mut seed = 12345u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        for (m, n, k) in shapes {
            let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
            let b: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
            let ga = GpuTensor::upload_f32_as_f16(vec![m, k], &a);
            let gb = GpuTensor::upload_f32_as_f16(vec![n, k], &b);

            set_use_f32_sgemm(true);
            set_use_bf16_gemm(false);
            let ref_out = GpuTensor::new_f16(vec![m, n]);
            sgemm_f16_a_btrans_buf(&ref_out.buffer, &ga.buffer, &gb.buffer, m, n, k);
            let expect = ref_out.download_to_f32();

            set_use_bf16_gemm(true);
            let got_t = GpuTensor::new_f16(vec![m, n]);
            sgemm_f16_a_btrans_buf(&got_t.buffer, &ga.buffer, &gb.buffer, m, n, k);
            let got = got_t.download_to_f32();
            set_use_bf16_gemm(false);
            set_use_f32_sgemm(false);

            let (mut dot, mut na, mut nb, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
            for (x, y) in expect.iter().zip(got.iter()) {
                assert!(y.is_finite(), "{m}x{n}x{k}: non-finite output");
                dot += (*x as f64) * (*y as f64);
                na += (*x as f64) * (*x as f64);
                nb += (*y as f64) * (*y as f64);
                worst = worst.max((x - y).abs());
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
            // Both sides round their result to bf16 (8 mantissa bits), so a
            // small absolute gap is expected; the cosine is the real check.
            assert!(cos > 0.9999, "{m}x{n}x{k}: cos_sim {cos:.8}, max_abs {worst:.3e}");
        }
    }

    /// MPS f32 sgemm goes non-finite past a certain M at the FFN's width.
    ///
    /// Found by sweeping Wan's token count: 5070 and 7488 tokens produce a
    /// clean forward, 9360 is 100% NaN. Attention was ruled out first (the
    /// scalar path fails identically), leaving the matmul. This isolates it
    /// with no model involved — the native bf16 kernel runs the same shapes as
    /// a control, since it shares nothing with MPS.
    #[test]
    fn sgemm_is_finite_at_large_m() {
        for (n, k, label) in [(3072usize, 3072usize, "q_proj"), (3072, 14336, "ffn_out")] {
        println!("--- {label}: [M,{k}] x [{n},{k}]^T ---");
        let mut seed = 999u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let b: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
        let gb = GpuTensor::upload_f32_as_f16(vec![n, k], &b);

        for m in [4096usize, 8190, 8192, 8448, 9360, 11440] {
            let a: Vec<f32> = (0..m * k).map(|_| rnd() * 0.1).collect();
            let ga = GpuTensor::upload_f32_as_f16(vec![m, k], &a);
            let mut bad = [0usize; 2];
            for (i, use_bf16) in [false, true].into_iter().enumerate() {
                set_use_f32_sgemm(!use_bf16);
                set_use_bf16_gemm(use_bf16);
                let out = GpuTensor::new_f16(vec![m, n]);
                sgemm_f16_a_btrans_buf(&out.buffer, &ga.buffer, &gb.buffer, m, n, k);
                bad[i] = out.download_to_f32().iter().filter(|v| !v.is_finite()).count();
                set_use_bf16_gemm(false);
                set_use_f32_sgemm(false);
            }
            println!("m={m:>6}  mps_f32 non-finite={:>10}  bf16_mma non-finite={:>10}",
                     bad[0], bad[1]);
        }
        }
    }

    /// The chunked MPS path must be numerically right, not merely finite.
    ///
    /// `sgemm_f16_a_btrans_buf` splits M into fixed-height chunks above
    /// `MPS_MAX_ROWS`, with the last aligned to the bottom of the matrix so it
    /// overlaps the previous one. That path only runs at native resolution, and
    /// the DiT-vs-diffusers validation runs at 256 tokens — so nothing has ever
    /// checked its *values*. A wrong offset or a mishandled overlap would
    /// silently degrade every large generation while still producing finite
    /// output.
    ///
    /// The native bf16 MMA kernel is the reference here: it shares no code with
    /// MPS and is bit-exact against it below the chunking threshold.
    #[test]
    /// The UNTRANSPOSED matmul's chunking, which every DiT `Linear` now uses.
    ///
    /// **`#[ignore]`d because it reproduces a real, unfixed bug** — it is not a
    /// flaky test, it is a flaky *product*. Roughly 2 runs in 3 it reports
    /// cos_sim ~0.33 at some M (which M varies), and the rest of the time
    /// 1.00000000. Within a single process both matmul paths are perfectly
    /// self-consistent (`which_matmul_is_unstable`: 0/7 repeats disagree), so
    /// whatever is wrong is fixed at process start and then stable — the same
    /// signature as the intermittent all-NaN generations. Run it on demand:
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture chunked_sgemm_f16_buf
    ///
    /// It reproduces in ~6 s where a generation took 40 minutes, so it is the
    /// right harness for chasing that bug.
    // Ignored because it must run alone. It is a minute of sustained GEMM, and
    // with anything else on the GPU macOS starts killing command buffers
    // (kIOGPUCommandBufferCallbackErrorImpactingInteractivity), which `flush`
    // now reports as a panic. That is the correct behaviour, but it makes the
    // test a contention detector rather than a correctness one when run
    // alongside the rest of the suite. Run it with `--ignored --test-threads=1`.
    #[ignore]
    ///
    /// `chunked_sgemm_matches_unchunked` covers `sgemm_f16_a_btrans_buf` only.
    /// When the weight transpose moved every `Linear` onto `sgemm_f16_buf`,
    /// nothing checked that path above the 8192-row threshold — and the DiT
    /// correctness test runs at 192 tokens, far below it. A 30-step 704x1280
    /// generation came out as pure noise before this test existed.
    #[test]
    fn chunked_sgemm_f16_buf_matches_reference() {
        let (n, k) = (3072usize, 3072usize);
        let mut seed = 99991u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        // B in [k, n] layout, as the untransposed matmul wants.
        let b: Vec<f32> = (0..k * n).map(|_| rnd() * 0.05).collect();
        let gb = GpuTensor::upload_f32_as_f16(vec![k, n], &b);
        // Both storage modes, and repeated, because the failure is
        // intermittent. The half path is the one that matters: it diverges only
        // above MPS_MAX_ROWS, because the chunked path
        // (`..._with_offsets`) originally ignored `half_io` and cast half data
        // as bfloat — giving a DiT that was correct at 192 tokens and pure
        // noise at 11440, which a bf16-only test could never have caught.
        let gb_half = GpuTensor::upload_f32_as_half(vec![k, n], &b);
        for half_io in [false, true] {
        for m in [4096usize, 8192, 8193, 11440] {
            let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
            let ga = if half_io {
                GpuTensor::upload_f32_as_half(vec![m, k], &a)
            } else {
                GpuTensor::upload_f32_as_f16(vec![m, k], &a)
            };
            let bmat = if half_io { &gb_half } else { &gb };

            set_use_f32_sgemm(true);
            set_sgemm_half_io(half_io);
            let got_t = if half_io {
                GpuTensor::new_half(vec![m, n])
            } else {
                GpuTensor::new_f16(vec![m, n])
            };
            sgemm_f16_buf(&got_t.buffer, &ga.buffer, &bmat.buffer, m, n, k);
            let got = got_t.download_to_f32();
            set_sgemm_half_io(false);
            set_use_f32_sgemm(false);

            // Reference: the bf16 MMA kernel, which shares no code with MPS.
            // It computes A x B^T, so hand it B transposed to [n, k].
            let mut bt = vec![0.0f32; n * k];
            for i in 0..k {
                for j in 0..n {
                    bt[j * k + i] = b[i * n + j];
                }
            }
            let gbt = GpuTensor::upload_f32_as_f16(vec![n, k], &bt);
            let ga_bf = GpuTensor::upload_f32_as_f16(vec![m, k], &a);
            let want_t = GpuTensor::new_f16(vec![m, n]);
            sgemm_bf16_btrans_gpu(&want_t.buffer, &ga_bf.buffer, &gbt.buffer, m, n, k);
            let want = want_t.download_to_f32();

            let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for (x, y) in want.iter().zip(got.iter()) {
                dot += (*x as f64) * (*y as f64);
                na += (*x as f64) * (*x as f64);
                nb += (*y as f64) * (*y as f64);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
            let bad = got.iter().filter(|v| !v.is_finite()).count();
            let store = if half_io { "half" } else { "bf16" };
            println!("{store:<5} m={m:>6} chunks={:<2} cos_sim={cos:.8} non-finite={bad}",
                     m.div_ceil(MPS_MAX_ROWS));
            // On disagreement, settle it against exact f32 rather than assuming
            // the reference is right: `want` is the MMA kernel, and either side
            // could be the broken one. One row is n*k multiply-adds, cheap.
            if cos <= 0.999 {
                let row = m - 1;
                let (mut e_mps, mut e_mma, mut refmax) = (0.0f32, 0.0f32, 0.0f32);
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for i in 0..k {
                        acc += (a[row * k + i] as f64) * (b[i * n + j] as f64);
                    }
                    let w = acc as f32;
                    refmax = refmax.max(w.abs());
                    e_mps = e_mps.max((w - got[row * n + j]).abs());
                    e_mma = e_mma.max((w - want[row * n + j]).abs());
                }
                println!(
                    "  CONVICT row {row}: |ref|max={refmax:.4}  mps_err={e_mps:.4e}                       mma_err={e_mma:.4e}  -> {} is wrong",
                    if e_mps > e_mma { "MPS" } else { "MMA(reference)" }
                );
            }
            assert_eq!(bad, 0, "{store} m={m}: non-finite");
            assert!(cos > 0.999, "{store} m={m}: sgemm_f16_buf diverges, cos_sim {cos}");
        }
        }
    }

    /// The chunked MPS path (B-transposed) must be numerically right, not merely
    /// finite: it splits M into fixed-height chunks with the last aligned to the
    /// bottom of the matrix, and that only runs at generation scale.
    #[test]
    fn chunked_sgemm_matches_unchunked() {
        let (n, k) = (3072usize, 3072usize);
        let mut seed = 271828u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let b: Vec<f32> = (0..n * k).map(|_| rnd() * 0.05).collect();
        let gb = GpuTensor::upload_f32_as_f16(vec![n, k], &b);

        // Straddle the threshold: below it MPS runs unchunked, above it the
        // chunked path runs. 11440 is Wan's native token count.
        for m in [8192usize, 8193, 9360, 11440, 16384, 16385] {
            let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
            let ga = GpuTensor::upload_f32_as_f16(vec![m, k], &a);

            set_use_f32_sgemm(true);
            set_use_bf16_gemm(false);
            let mps = GpuTensor::new_f16(vec![m, n]);
            sgemm_f16_a_btrans_buf(&mps.buffer, &ga.buffer, &gb.buffer, m, n, k);
            let got = mps.download_to_f32();

            set_use_bf16_gemm(true);
            let reference = GpuTensor::new_f16(vec![m, n]);
            sgemm_f16_a_btrans_buf(&reference.buffer, &ga.buffer, &gb.buffer, m, n, k);
            let want = reference.download_to_f32();
            set_use_bf16_gemm(false);
            set_use_f32_sgemm(false);

            let (mut dot, mut na, mut nb, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
            let mut worst_row = 0usize;
            for (i, (x, y)) in want.iter().zip(got.iter()).enumerate() {
                dot += (*x as f64) * (*y as f64);
                na += (*x as f64) * (*x as f64);
                nb += (*y as f64) * (*y as f64);
                if (x - y).abs() > worst { worst = (x - y).abs(); worst_row = i / n; }
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
            let bad = got.iter().filter(|v| !v.is_finite()).count();
            println!(
                "m={m:>6} chunks={:<2} cos_sim={cos:.8} max_abs={worst:.3e}                  (worst row {worst_row}) non-finite={bad}",
                m.div_ceil(MPS_MAX_ROWS)
            );
            assert_eq!(bad, 0, "m={m}: non-finite output");
            assert!(cos > 0.99999, "m={m}: chunked result diverges, cos_sim {cos}");
        }
    }

    /// Flash attention at Wan's actual sequence length.
    ///
    /// `flash_attention_tiled_matches_scalar_and_cpu` tops out at l_q=100.
    /// Wan runs 11440 tokens with 24 heads at d=128 — 114x longer than
    /// anything previously tested. The tiled kernels carry a running max and
    /// sum and rescale the accumulator once per KV tile, so at 358 tiles any
    /// error in that rescaling compounds in a way short sequences cannot show.
    /// This is the shape the model actually runs.
    #[test]
    fn flash_attention_matches_scalar_at_generation_length() {
        let (n, h, d) = (1usize, 24usize, 128usize);
        for l in [2048usize, 11440] {
            let mut seed = 8675309u32;
            let mut rnd = || {
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                ((seed >> 8) as f32 / 8_388_608.0) - 1.0
            };
            let elems = n * h * l * d;
            let qd: Vec<f32> = (0..elems).map(|_| rnd()).collect();
            let kd: Vec<f32> = (0..elems).map(|_| rnd()).collect();
            let vd: Vec<f32> = (0..elems).map(|_| rnd()).collect();
            let q = GpuTensor::upload_f32_as_f16(vec![n, h, l, d], &qd);
            let k = GpuTensor::upload_f32_as_f16(vec![n, h, l, d], &kd);
            let v = GpuTensor::upload_f32_as_f16(vec![n, h, l, d], &vd);
            let scale = 1.0 / (d as f32).sqrt();

            let mut tiled = GpuTensor::new_f16(vec![n, h, l, d]);
            flash_attention_best_f16_gpu(&q, &k, &v, &mut tiled, n, h, l, l, d, scale);
            let got = tiled.download_to_f32();

            // The scalar reference over all heads at 11440 is one dispatch of
            // several seconds, long enough for the watchdog to kill whenever
            // the display wants the GPU. One head at a time keeps each short.
            let head = l * d;
            let mut want = Vec::with_capacity(elems);
            for hi in 0..h {
                let slice = |x: &[f32]| GpuTensor::upload_f32_as_f16(vec![n, 1, l, d], &x[hi * head..(hi + 1) * head]);
                let mut scalar = GpuTensor::new_f16(vec![n, 1, l, d]);
                flash_attention_f16_gpu(&slice(&qd), &slice(&kd), &slice(&vd), &mut scalar, n, 1, l, l, d, scale);
                want.extend(scalar.download_to_f32());
            }

            let (mut dot, mut na, mut nb, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
            for (x, y) in want.iter().zip(got.iter()) {
                dot += (*x as f64) * (*y as f64);
                na += (*x as f64) * (*x as f64);
                nb += (*y as f64) * (*y as f64);
                worst = worst.max((x - y).abs());
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
            let bad = got.iter().filter(|x| !x.is_finite()).count();
            println!("l={l:>6} h={h} d={d}: cos_sim={cos:.8} max_abs={worst:.3e} non-finite={bad}");
            assert_eq!(bad, 0, "l={l}: tiled attention produced non-finite output");
            assert!(cos > 0.9999, "l={l}: tiled attention diverges, cos_sim {cos}");
        }
    }
}

#[cfg(test)]
mod gemm_bench {
    use super::*;

    /// Whether unfenced MPS matmuls corrupt a dependent chain at each model's
    /// shapes: `up` then `down` projections, each output feeding the next as
    /// in a forward, compared bitwise with a fenced run.
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture mps_unfenced_chain
    #[test]
    #[ignore]
    fn mps_unfenced_chain() {
        // (label, rows, model dim, ffn dim, layers)
        let sets = [
            ("sd15 64x64", 4096usize, 320usize, 1280usize, 16usize),
            ("sd15 16x16", 256, 1280, 5120, 16),
            ("sdxl 32x32", 1024, 1280, 5120, 16),
            ("flux 1024²", 4608, 3072, 12288, 8),
            ("krea 768²", 2336, 6144, 16384, 8),
        ];
        // `KLEARU_CHAIN_SHAPES="m,d,f,layers;..."` sweeps other shapes instead.
        let custom: Vec<(String, usize, usize, usize, usize)> = std::env::var("KLEARU_CHAIN_SHAPES")
            .map(|v| v.split(';').map(|t| {
                let n: Vec<usize> = t.split(',').map(|x| x.trim().parse().unwrap()).collect();
                (format!("{}x{}x{}", n[0], n[1], n[2]), n[0], n[1], n[2], n[3])
            }).collect())
            .unwrap_or_default();
        let sets: Vec<(String, usize, usize, usize, usize)> = if custom.is_empty() {
            sets.iter().map(|&(l, m, d, f, n)| (l.to_string(), m, d, f, n)).collect()
        } else {
            custom
        };
        let runs: usize = std::env::var("KLEARU_CHAIN_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
        let mut seed = 11u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let n_sets = sets.len();
        for (si, (label, m, d, f, layers)) in sets.into_iter().enumerate() {
            let label = label.as_str();
            // Diagnostics: `KLEARU_CHAIN_DRAIN=1` empties the buffer pool before
            // each set; `KLEARU_CHAIN_PRE_FENCED=1` runs every set but the last
            // fenced, so only state they leave behind can reach it.
            if std::env::var_os("KLEARU_CHAIN_DRAIN").is_some() { flush(); drain_pool(); }
            let pre_fenced = std::env::var_os("KLEARU_CHAIN_PRE_FENCED").is_some() && si + 1 < n_sets;
            // `KLEARU_CHAIN_LAST_FENCED=1`: the last set's runs are fenced too, so
            // any difference left is damage done by the unfenced sets before it.
            let pre_fenced = pre_fenced
                || (std::env::var_os("KLEARU_CHAIN_LAST_FENCED").is_some() && si + 1 == n_sets);
            // Scaled so activations stay O(1) through the chain.
            let wu: Vec<f32> = (0..f * d).map(|_| rnd() * (3.0 / d as f32).sqrt()).collect();
            let wd: Vec<f32> = (0..d * f).map(|_| rnd() * (3.0 / f as f32).sqrt()).collect();
            let x0: Vec<f32> = (0..m * d).map(|_| rnd()).collect();
            let (gu, gd) = (GpuTensor::upload_f32_as_f16(vec![f, d], &wu), GpuTensor::upload_f32_as_f16(vec![d, f], &wd));
            let chain = |batched: bool| -> Vec<f32> {
                let x = GpuTensor::upload_f32_as_f16(vec![m, d], &x0);
                let h = GpuTensor::new_f16(vec![m, f]);
                if batched { begin_batch(); }
                for _ in 0..layers {
                    sgemm_f16_a_btrans_buf(&h.buffer, &x.buffer, &gu.buffer, m, f, d);
                    sgemm_f16_a_btrans_buf(&x.buffer, &h.buffer, &gd.buffer, m, d, f);
                }
                x.download_to_f32()
            };
            let modes: &[bool] = if std::env::var_os("KLEARU_CHAIN_SHAPES").is_some() { &[false] } else { &[false, true] };
            for &batched in modes {
                let prev = set_mps_barrier(true);
                let want = chain(batched);
                set_mps_barrier(pre_fenced);
                let (mut bad_runs, mut bad_rows, mut nonfinite) = (0, 0, 0);
                for _ in 0..runs {
                    let got = chain(batched);
                    let rows = (0..m).filter(|r| {
                        got[r * d..(r + 1) * d].iter().zip(&want[r * d..(r + 1) * d]).any(|(a, b)| a.to_bits() != b.to_bits())
                    }).count();
                    if rows > 0 { bad_runs += 1; bad_rows += rows; }
                    nonfinite += got.iter().filter(|v| !v.is_finite()).count();
                }
                set_mps_barrier(prev);
                let wf = want.iter().filter(|v| !v.is_finite()).count();
                println!("{label:<11} batched={batched:<5} fenced ref non-finite={wf}; unfenced: {bad_runs}/{runs} runs differ ({bad_rows} rows), {nonfinite} non-finite values");
            }
        }
    }

    /// `mps_unfenced_chain`'s chain with every matmul writing its own output,
    /// so the first one to go wrong — and its rows — can be read off.
    /// `KLEARU_CHAIN_SHAPES`/`KLEARU_CHAIN_RUNS` as there.
    #[test]
    #[ignore]
    fn mps_unfenced_chain_locate() {
        let shapes: Vec<(usize, usize, usize, usize)> = std::env::var("KLEARU_CHAIN_SHAPES")
            .unwrap_or_else(|_| "4096,320,1280,16;256,1280,5120,16;1024,1280,5120,16;4608,3072,12288,8;2336,6144,16384,8".into())
            .split(';').map(|t| {
                let n: Vec<usize> = t.split(',').map(|x| x.trim().parse().unwrap()).collect();
                (n[0], n[1], n[2], n[3])
            }).collect();
        let runs: usize = std::env::var("KLEARU_CHAIN_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        let mut seed = 11u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        for (m, d, f, layers) in shapes {
            let wu: Vec<f32> = (0..f * d).map(|_| rnd() * (3.0 / d as f32).sqrt()).collect();
            let wd: Vec<f32> = (0..d * f).map(|_| rnd() * (3.0 / f as f32).sqrt()).collect();
            let x0: Vec<f32> = (0..m * d).map(|_| rnd()).collect();
            let (gu, gd) = (GpuTensor::upload_f32_as_f16(vec![f, d], &wu), GpuTensor::upload_f32_as_f16(vec![d, f], &wd));
            // outs[2i] = up i [m, f], outs[2i+1] = down i [m, d]
            let chain = || -> Vec<Vec<f32>> {
                let x = GpuTensor::upload_f32_as_f16(vec![m, d], &x0);
                let outs: Vec<GpuTensor> = (0..2 * layers)
                    .map(|j| GpuTensor::new_f16(vec![m, if j % 2 == 0 { f } else { d }]))
                    .collect();
                for i in 0..layers {
                    let src = if i == 0 { &x } else { &outs[2 * i - 1] };
                    sgemm_f16_a_btrans_buf(&outs[2 * i].buffer, &src.buffer, &gu.buffer, m, f, d);
                    sgemm_f16_a_btrans_buf(&outs[2 * i + 1].buffer, &outs[2 * i].buffer, &gd.buffer, m, d, f);
                }
                outs.iter().map(|o| o.download_to_f32()).collect()
            };
            let prev = set_mps_barrier(true);
            let want = chain();
            set_mps_barrier(false);
            for run in 0..runs {
                let got = chain();
                let first = (0..got.len()).find(|&j| got[j].iter().zip(&want[j]).any(|(a, b)| a.to_bits() != b.to_bits()));
                if let Some(j) = first {
                    let cols = if j % 2 == 0 { f } else { d };
                    let rows: Vec<usize> = (0..m).filter(|r| {
                        got[j][r * cols..(r + 1) * cols].iter().zip(&want[j][r * cols..(r + 1) * cols]).any(|(a, b)| a.to_bits() != b.to_bits())
                    }).collect();
                    let nf = got[j].iter().filter(|v| !v.is_finite()).count();
                    println!("{m}x{d}x{f} run {run}: first bad = {} {} ({m}x{cols}, k={}), {} rows {}..={}, {nf} non-finite",
                        if j % 2 == 0 { "up" } else { "down" }, j / 2, if j % 2 == 0 { d } else { f },
                        rows.len(), rows[0], rows[rows.len() - 1]);
                }
            }
            set_mps_barrier(prev);
            println!("{m}x{d}x{f}: done");
        }
    }

    /// MPSMatrixMultiplication with its output across a 4 GiB boundary of GPU
    /// address space, at the Krea VAE attention's score shape: raw, then
    /// through the wrapper with the pool seeded with that same buffer, which
    /// `acquire_mps_buffer` must refuse.
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture mps_matmul_across_4gib
    #[test]
    #[ignore]
    fn mps_matmul_across_4gib() {
        let backend = get_metal();
        let (m, n, k) = (3640usize, 9216usize, 384usize);
        let mut seed = 3u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let new = |bytes: usize| backend.device.new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared);
        let upload = |v: &[f32]| {
            let b = new(v.len() * 4);
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.contents() as *mut f32, v.len()) };
            b
        };
        let av: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let bv: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
        let (a, b) = (upload(&av), upload(&bv));
        let read = |c: &Buffer| -> Vec<f32> {
            flush();
            unsafe { std::slice::from_raw_parts(c.contents() as *const f32, m * n) }.to_vec()
        };
        let bad_rows = |got: &[f32], want: &[f32]| -> Vec<usize> {
            (0..m).filter(|r| got[r * n..(r + 1) * n].iter().zip(&want[r * n..(r + 1) * n]).any(|(x, y)| x.to_bits() != y.to_bits())).collect()
        };
        let mm = |c: &Buffer| unsafe {
            dispatch_mps_matmul_with_transpose(backend, &a, &b, c, m, n, k, MPS_DATA_TYPE_FLOAT32, 4, false, true, n, k)
        };
        let clean = new(m * n * 4);
        mm(&clean);
        let want = read(&clean);
        let mut held = Vec::new();
        let crossing = loop {
            let c = new(m * n * 4);
            if crosses_4gib(&c) { break c; }
            held.push(c);
        };
        let boundary_row = (((crossing.gpu_address() >> 32) + 1) << 32).saturating_sub(crossing.gpu_address()) as usize / (n * 4);
        mm(&crossing);
        let rows = bad_rows(&read(&crossing), &want);
        println!("raw MPS, output crossing at row {boundary_row}: {} bad rows {:?}..{:?}", rows.len(), rows.first(), rows.last());

        // The wrapper, with the pool primed to hand out the crossing buffer.
        let av16: Vec<f32> = av.iter().map(|v| half::bf16::from_f32(*v).to_f32()).collect();
        let bv16: Vec<f32> = bv.iter().map(|v| half::bf16::from_f32(*v).to_f32()).collect();
        let (ga, gb) = (GpuTensor::upload_f32_as_f16(vec![m, k], &av16), GpuTensor::upload_f32_as_f16(vec![n, k], &bv16));
        let out = GpuTensor::new_f16(vec![m, n]);
        set_use_f32_sgemm(true);
        sgemm_f16_a_btrans_buf(&out.buffer, &ga.buffer, &gb.buffer, m, n, k);
        let want16 = out.download_to_f32();
        flush();
        release_buffer(crossing);
        sgemm_f16_a_btrans_buf(&out.buffer, &ga.buffer, &gb.buffer, m, n, k);
        let got16 = out.download_to_f32();
        set_use_f32_sgemm(false);
        let rows: Vec<usize> = (0..m).filter(|r| got16[r * n..(r + 1) * n].iter().zip(&want16[r * n..(r + 1) * n]).any(|(x, y)| x.to_bits() != y.to_bits())).collect();
        println!("wrapper with a crossing buffer pooled: {} bad rows", rows.len());
        assert!(rows.is_empty());
    }

    /// Whether MPSGraph convolution shares MPSMatrixMultiplication's 4 GiB
    /// bug (see `acquire_mps_buffer`): the VAE's 3x3 conv at 768² with its
    /// input, then its output, placed across a 4 GiB boundary, against the
    /// same conv on clean buffers.
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture mpsgraph_conv_across_4gib
    #[test]
    #[ignore]
    fn mpsgraph_conv_across_4gib() {
        use half::slice::HalfFloatSliceExt;
        let (c, h, w) = (96usize, 768usize, 768usize);
        let mut seed = 5u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let x: Vec<f32> = (0..c * h * w).map(|_| rnd()).collect();
        let wt: Vec<f32> = (0..c * c * 9).map(|_| rnd() * 0.05).collect();
        let wid = next_weight_id();
        let bytes = c * h * w * 2;
        let fill = |buf: &Buffer| unsafe {
            std::slice::from_raw_parts_mut(buf.contents() as *mut half::f16, x.len()).convert_from_f32_slice(&x);
        };
        let read = |buf: &Buffer| -> Vec<f32> {
            flush();
            let mut out = vec![0.0f32; c * h * w];
            unsafe { std::slice::from_raw_parts(buf.contents() as *const half::f16, out.len()) }.convert_to_f32_slice(&mut out);
            out
        };
        // Allocate until a buffer of `bytes` crosses; the rest are held so
        // each attempt lands somewhere new.
        let crossing = || -> (Buffer, Vec<Buffer>) {
            let mut held = Vec::new();
            loop {
                let b = get_metal().device.new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared);
                if crosses_4gib(&b) { return (b, held); }
                held.push(b);
            }
        };
        let clean_in = get_metal().device.new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared);
        fill(&clean_in);
        let (want_buf, ..) = mpsgraph_conv2d_half(&clean_in, &wt, wid, 1, c, h, w, c, 3, 3, 1, 1);
        assert!(!crosses_4gib(&want_buf) && !crosses_4gib(&clean_in));
        let want = read(&want_buf);
        let compare = |what: &str, got: &[f32]| {
            let rows = (0..c * h).filter(|r| got[r * w..(r + 1) * w].iter().zip(&want[r * w..(r + 1) * w]).any(|(a, b)| a.to_bits() != b.to_bits())).count();
            println!("{what}: {rows} of {} channel-rows differ, {} non-finite", c * h, got.iter().filter(|v| !v.is_finite()).count());
        };
        let (cin, _held) = crossing();
        fill(&cin);
        let (out, ..) = mpsgraph_conv2d_half(&cin, &wt, wid, 1, c, h, w, c, 3, 3, 1, 1);
        compare("input across 4 GiB", &read(&out));
        // The conv takes its output from the pool: seed the pool with a
        // crossing buffer of exactly that size.
        let (cout, _held2) = crossing();
        flush();
        release_buffer(cout);
        let (out, ..) = mpsgraph_conv2d_half(&clean_in, &wt, wid, 1, c, h, w, c, 3, 3, 1, 1);
        println!("output buffer crosses 4 GiB: {}", crosses_4gib(&out));
        compare("output across 4 GiB", &read(&out));
    }

    /// What the MPS fence (`set_mps_barrier`) costs at other models' matmul
    /// shapes, batched as SD's UNet batches and unbatched as the DiTs run —
    /// so turning it on for them can be judged without their weights.
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture mps_fence_cost
    #[test]
    #[ignore]
    fn mps_fence_cost() {
        // (label, [(m, n, k)]) — one transformer block's projections each.
        let sd15_64: Vec<(usize, usize, usize)> = [(4096, 320, 320); 4].into_iter()
            .chain([(77, 320, 768); 2]).chain([(4096, 2560, 320), (4096, 320, 1280)]).collect();
        let sd15_16: Vec<(usize, usize, usize)> = [(256, 1280, 1280); 4].into_iter()
            .chain([(77, 1280, 768); 2]).chain([(256, 10240, 1280), (256, 1280, 5120)]).collect();
        let sdxl_32: Vec<(usize, usize, usize)> = [(1024, 1280, 1280); 4].into_iter()
            .chain([(77, 1280, 2048); 2]).chain([(1024, 10240, 1280), (1024, 1280, 5120)]).collect();
        let flux: Vec<(usize, usize, usize)> =
            vec![(4096, 9216, 3072), (4096, 3072, 3072), (4096, 12288, 3072), (4096, 3072, 12288)];
        let sets = [("sd15 64x64", sd15_64), ("sd15 16x16", sd15_16), ("sdxl 32x32", sdxl_32), ("flux 1024²", flux)];
        let mut seed = 7u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        println!("{:<11} {:>7} {:>12} {:>12} {:>9}", "set", "batched", "off ms/mm", "on ms/mm", "overhead");
        for (label, shapes) in sets {
            let bufs: Vec<_> = shapes.iter().map(|&(m, n, k)| {
                let a: Vec<f32> = (0..m * k).map(|_| rnd() * 0.1).collect();
                let b: Vec<f32> = (0..n * k).map(|_| rnd() * 0.1).collect();
                (GpuTensor::upload_f32_as_f16(vec![m, k], &a), GpuTensor::upload_f32_as_f16(vec![n, k], &b),
                 GpuTensor::new_f16(vec![m, n]), m, n, k)
            }).collect();
            let reps = if label.starts_with("flux") { 3 } else { 40 };
            for batched in [true, false] {
                let time = |fenced: bool| -> f64 {
                    let prev = set_mps_barrier(fenced);
                    let run = || {
                        if batched { begin_batch(); }
                        for (a, b, c, m, n, k) in &bufs {
                            sgemm_f16_a_btrans_buf(&c.buffer, &a.buffer, &b.buffer, *m, *n, *k);
                        }
                        flush();
                    };
                    run();
                    let t = std::time::Instant::now();
                    for _ in 0..reps { run(); }
                    set_mps_barrier(prev);
                    t.elapsed().as_secs_f64() * 1e3 / (reps * bufs.len()) as f64
                };
                // Interleave to spread drift across both.
                let (mut off, mut on) = (0.0, 0.0);
                for _ in 0..2 { off += time(false); on += time(true); }
                println!("{label:<11} {batched:>7} {:>12.3} {:>12.3} {:>8.1}%", off / 2.0, on / 2.0, (on / off - 1.0) * 100.0);
            }
        }
    }

    /// Krea 2's matmuls at 1024² (4120 rows) through the production path and
    /// through bare MPS on operands already in f16, to separate MPS's own
    /// speed from the wrapper's casts, zero-fills, chunking and fences.
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture krea_matmul_ceiling
    #[test]
    #[ignore]
    fn krea_matmul_ceiling() {
        let backend = get_metal();
        let m = 4120usize;
        let half = m / 2;
        // (label, n, k, calls per block)
        let shapes = [
            ("q/gate/out", 6144usize, 6144usize, 3usize),
            ("k/v", 1536, 6144, 2),
            ("ff gate/up", 16384, 6144, 2),
            ("ff down", 6144, 16384, 1),
            ("gate+up as one", 32768, 6144, 0),
        ];
        let mut seed = 5u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let mut held = Vec::new();
        let mut half_buf = |v: &[f32]| loop {
            let b = upload_f32_as_half_buffer(v);
            if !crosses_4gib(&b) { break b; }
            held.push(b);
        };
        let time = |f: &dyn Fn()| -> f64 {
            f();
            flush();
            let reps = 4;
            let t = std::time::Instant::now();
            for _ in 0..reps { f(); }
            flush();
            t.elapsed().as_secs_f64() / reps as f64
        };
        let mut totals = [0.0f64; 5];
        for (label, n, k, per_block) in shapes {
            let av: Vec<f32> = (0..m * k).map(|_| rnd() * 0.1).collect();
            let bv: Vec<f32> = (0..k * n).map(|_| rnd() * 0.05).collect();
            let (a, b) = (GpuTensor::upload_f32_as_f16(vec![m, k], &av), GpuTensor::upload_f32_as_f16(vec![k, n], &bv));
            let out = GpuTensor::new_f16(vec![m, n]);
            let (a16, b16, c16) = (half_buf(&av), half_buf(&bv), half_buf(&vec![0.0; m * n]));
            let wrapper_chunks = || for c in 0..2 {
                sgemm_f16_buf_with_offsets(&out.buffer, c * half * n, &a.buffer, c * half * k, &b.buffer, 0, half, n, k);
            };
            let fenced = |on: bool| move || {
                let prev = set_mps_barrier(on);
                wrapper_chunks();
                set_mps_barrier(prev);
            };
            let runs: [(&str, Box<dyn Fn()>); 5] = [
                ("wrapper 2x2060 fenced", Box::new(fenced(true))),
                ("wrapper 2x2060", Box::new(fenced(false))),
                ("wrapper 1x4120", Box::new(|| unsafe {
                    dispatch_mps_matmul_bf16_via_f32(backend, &a.buffer, &b.buffer, &out.buffer, m, n, k, false, false, k, n)
                })),
                ("MPS f16 2x2060", Box::new(|| unsafe {
                    for c in 0..2 {
                        dispatch_mps_matmul_with_offsets(backend, &a16, c * half * k * 2, &b16, 0, &c16, c * half * n * 2,
                            half, n, k, MPS_DATA_TYPE_FLOAT16, 2, false, false, k, n);
                    }
                })),
                ("MPS f16 1x4120", Box::new(|| unsafe {
                    dispatch_mps_matmul_with_transpose(backend, &a16, &b16, &c16, m, n, k, MPS_DATA_TYPE_FLOAT16, 2, false, false, k, n)
                })),
            ];
            let gflop = 2.0 * (m * n * k) as f64 / 1e9;
            println!("{label} ({m}x{n}x{k}, {gflop:.0} GFLOP, {per_block}/block):");
            for (i, (what, f)) in runs.iter().enumerate() {
                let dt = time(f.as_ref());
                totals[i] += dt * (per_block * 28) as f64;
                println!("  {what:<22} {:7.1} ms  {:5.2} TF/s", dt * 1e3, gflop / 1e3 / dt);
            }
        }
        println!("28 blocks' matmuls (gate/up as two calls):");
        let names = ["wrapper 2x2060 fenced", "wrapper 2x2060", "wrapper 1x4120", "MPS f16 2x2060", "MPS f16 1x4120"];
        for (what, t) in names.iter().zip(totals) {
            println!("  {what:<22} {t:6.2} s");
        }
    }

    /// Krea's FFN up-projection back to back for `KLEARU_SUSTAIN_SECS`
    /// (default 120), printing throughput per 5 s window — whether the GPU
    /// holds its burst rate under a load as long as a generation. Windows
    /// alternate between the production wrapper (bf16 operands cast to f16
    /// every call) and bare MPS on operands already in f16, so both see the
    /// same thermal state; the averages skip the first 30 s.
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture krea_matmul_sustained
    #[test]
    #[ignore]
    fn krea_matmul_sustained() {
        let backend = get_metal();
        let secs: f64 = std::env::var("KLEARU_SUSTAIN_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(120.0);
        let (m, n, k) = (2060usize, 16384usize, 6144usize);
        let mut seed = 9u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let av: Vec<f32> = (0..m * k).map(|_| rnd() * 0.1).collect();
        let bv: Vec<f32> = (0..k * n).map(|_| rnd() * 0.05).collect();
        let a = GpuTensor::upload_f32_as_f16(vec![m, k], &av);
        let b = GpuTensor::upload_f32_as_f16(vec![k, n], &bv);
        let out = GpuTensor::new_f16(vec![m, n]);
        let (a16, b16, c16) = (upload_f32_as_half_buffer(&av), upload_f32_as_half_buffer(&bv), upload_f32_as_half_buffer(&vec![0.0; m * n]));
        assert!(![&a16, &b16, &c16].iter().any(|x| crosses_4gib(x)), "rerun: a bare-MPS operand crosses 4 GiB");
        let gflop = 2.0 * (m * n * k) as f64 / 1e9;
        let names = ["wrapper", "bare MPS f16"];
        let mut sums = [(0.0f64, 0usize); 2];
        let start = std::time::Instant::now();
        let (mut window, mut calls, mut mode) = (std::time::Instant::now(), 0usize, 0usize);
        while start.elapsed().as_secs_f64() < secs {
            if mode == 0 {
                sgemm_f16_buf(&out.buffer, &a.buffer, &b.buffer, m, n, k);
            } else {
                unsafe { dispatch_mps_matmul_with_transpose(backend, &a16, &b16, &c16, m, n, k, MPS_DATA_TYPE_FLOAT16, 2, false, false, k, n) };
            }
            calls += 1;
            if calls % 8 == 0 {
                flush();
                let dt = window.elapsed().as_secs_f64();
                if dt >= 5.0 {
                    let tf = calls as f64 * gflop / 1e3 / dt;
                    let t = start.elapsed().as_secs_f64();
                    println!("{t:6.1} s  {:<12} {tf:5.2} TF/s", names[mode]);
                    if t > 30.0 {
                        sums[mode].0 += tf;
                        sums[mode].1 += 1;
                    }
                    (window, calls, mode) = (std::time::Instant::now(), 0, 1 - mode);
                }
            }
        }
        flush();
        for (name, (s, c)) in names.iter().zip(sums) {
            println!("after 30 s, {name:<12} {:5.2} TF/s over {c} windows", s / c.max(1) as f64);
        }
    }

    /// Throughput of the matmul paths at the DiT's real shapes.
    ///
    /// The whole forward runs at ~1.8 TFLOP/s against roughly 10 the hardware
    /// can reach, and matmul is ~65% of it. MPS is pinned to its f32 path
    /// (f16 is still all-NaN here, and only 1.38x anyway), so the native
    /// simdgroup kernel is the only way off that ceiling — this measures the
    /// gap without needing the 10 GB model.
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture gemm_throughput
    #[test]
    #[ignore]
    fn gemm_throughput() {
        // m, n, k for one 11440-token forward: QKV/out projections, then the
        // two FFN matmuls that dominate.
        let shapes = [
            ("qkv/out ", 11440usize, 3072usize, 3072usize),
            ("ffn_proj", 11440, 14336, 3072),
            ("ffn_out ", 11440, 3072, 14336),
        ];
        let mut seed = 7u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        println!("{:<9} {:>22} {:>12} {:>12}", "shape", "GFLOP", "mps_f32", "bf16_mma");
        let (mut tot_mps, mut tot_mma) = (0.0f64, 0.0f64);
        for (label, m, n, k) in shapes {
            let a: Vec<f32> = (0..m * k).map(|_| rnd() * 0.1).collect();
            let b: Vec<f32> = (0..n * k).map(|_| rnd() * 0.1).collect();
            let ga = GpuTensor::upload_f32_as_f16(vec![m, k], &a);
            let gb = GpuTensor::upload_f32_as_f16(vec![n, k], &b);
            let out = GpuTensor::new_f16(vec![m, n]);
            let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;

            let mut time = |use_mma: bool| -> f64 {
                set_use_bf16_gemm(use_mma);
                set_use_f32_sgemm(!use_mma);
                // Warm up: first call compiles/caches, and MPS pays graph setup.
                sgemm_f16_a_btrans_buf(&out.buffer, &ga.buffer, &gb.buffer, m, n, k);
                flush();
                let t = std::time::Instant::now();
                let reps = 3;
                for _ in 0..reps {
                    sgemm_f16_a_btrans_buf(&out.buffer, &ga.buffer, &gb.buffer, m, n, k);
                }
                flush();
                let dt = t.elapsed().as_secs_f64() / reps as f64;
                set_use_bf16_gemm(false);
                set_use_f32_sgemm(false);
                dt
            };
            let t_mps = time(false);
            let t_mma = time(true);
            tot_mps += t_mps;
            tot_mma += t_mma;
            println!(
                "{label} {gflop:>12.1} GFLOP  {:>7.2} TF/s  {:>7.2} TF/s",
                gflop / 1e3 / t_mps, gflop / 1e3 / t_mma
            );
        }
        println!("per-block matmul total: mps {:.1} ms, bf16_mma {:.1} ms  ({:.1}x)",
                 tot_mps * 1e3, tot_mma * 1e3, tot_mma / tot_mps);

        // `ffn_out` is 3x slower than `ffn_proj` at identical FLOPs. They differ
        // only in orientation: proj is A[m,k] x W[n,k]^T with k=3072, out is the
        // same call with k=14336. If the B-transposed path is what costs, the
        // untransposed one at the same shape will be much faster — and weights
        // are static, so pre-transposing at load time is free at runtime.
        println!("\n--- ffn_out shape: B-transposed vs untransposed ---");
        let (m, n, k) = (11440usize, 3072usize, 14336usize);
        let a: Vec<f32> = (0..m * k).map(|_| rnd() * 0.1).collect();
        let bt: Vec<f32> = (0..n * k).map(|_| rnd() * 0.1).collect();
        let ga = GpuTensor::upload_f32_as_f16(vec![m, k], &a);
        let gbt = GpuTensor::upload_f32_as_f16(vec![n, k], &bt);
        let gbn = GpuTensor::upload_f32_as_f16(vec![k, n], &bt);
        let out = GpuTensor::new_f16(vec![m, n]);
        let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
        set_use_f32_sgemm(true);
        for (label, transposed) in [("B-transposed", true), ("untransposed", false)] {
            if transposed {
                sgemm_f16_a_btrans_buf(&out.buffer, &ga.buffer, &gbt.buffer, m, n, k);
            } else {
                sgemm_f16_buf(&out.buffer, &ga.buffer, &gbn.buffer, m, n, k);
            }
            flush();
            let t = std::time::Instant::now();
            for _ in 0..3 {
                if transposed {
                    sgemm_f16_a_btrans_buf(&out.buffer, &ga.buffer, &gbt.buffer, m, n, k);
                } else {
                    sgemm_f16_buf(&out.buffer, &ga.buffer, &gbn.buffer, m, n, k);
                }
            }
            flush();
            let dt = t.elapsed().as_secs_f64() / 3.0;
            println!("{label:<14} {:>7.2} TF/s  ({:.1} ms)", gflop / 1e3 / dt, dt * 1e3);
        }
        set_use_f32_sgemm(false);
    }
}

#[cfg(test)]
mod transpose_semantics {
    use super::*;

    /// Pin down what `transpose_outer_f16_gpu` actually does, since a
    /// weight-layout change built on the wrong assumption produced a DiT with

    /// Confirm MLX's simdgroup fragment mapping on this hardware.
    ///
    /// `mlx/backend/metal/kernels/steel/gemm/mma.h` documents that each lane of
    /// an 8x8 fragment owns one row and two consecutive columns. Two attention
    /// optimisations depend on that being true here — a per-row rescale of the
    /// accumulator without a diagonal-matrix MMA, and keeping the score tile in
    /// registers — and a wrong mapping corrupts output without failing to
    /// compile, so it is checked rather than assumed.
    #[test]
    fn simdgroup_fragment_layout_matches_mlx() {
        use metal::MTLSize;
        let backend = get_metal();
        let src: Vec<f32> = (0..64).map(|i| i as f32).collect();   // r*8 + c
        let inbuf = GpuTensor::new_f32(vec![64]);
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(), inbuf.buffer.contents() as *mut f32, src.len());
        }
        let out = GpuTensor::new_f32(vec![64]);
        with_cmd(|cmd| {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&backend.simdgroup_layout_probe_pipe);
            enc.set_buffer(0, Some(&inbuf.buffer), 0);
            enc.set_buffer(1, Some(&out.buffer), 0);
            enc.dispatch_threads(MTLSize::new(32, 1, 1), MTLSize::new(32, 1, 1));
            enc.end_encoding();
        });
        flush();
        let got = out.download_to_f32();

        let mut mismatches = Vec::new();
        for lane in 0..32usize {
            let qid = lane / 4;
            let fm = (qid & 4) + ((lane / 2) % 4);          // row
            let fn_ = (qid & 2) * 2 + (lane % 2) * 2;       // first column
            let want = [(fm * 8 + fn_) as f32, (fm * 8 + fn_ + 1) as f32];
            let have = [got[lane * 2], got[lane * 2 + 1]];
            if have != want {
                mismatches.push((lane, have, want));
            }
        }
        for (lane, have, want) in mismatches.iter().take(6) {
            println!("lane {lane:2}: got {have:?} want {want:?}");
        }
        // Every element exactly once, so the two per lane really do partition.
        let mut seen = vec![0u32; 64];
        for v in &got {
            assert!(*v >= 0.0 && *v < 64.0, "lane reported {v}");
            seen[*v as usize] += 1;
        }
        assert!(seen.iter().all(|c| *c == 1), "elements do not partition");
        assert!(
            mismatches.is_empty(),
            "{} of 32 lanes disagree with MLX's mapping",
            mismatches.len()
        );
        println!("MLX mapping confirmed: each lane owns row fm, columns fn and fn+1");
    }

    /// `pad_end_1` must put the data in the top-left and zeros on the new
    /// bottom row and right column — not centre it, and not wrap.
    ///
    /// Getting this backwards shifts every downsample in the VAE encoder by a
    /// pixel, which does not crash and does not look obviously wrong; it just
    /// makes the encoded latent subtly not the image.
    #[test]
    fn pad_end_1_puts_zeros_only_on_the_far_edges() {
        let (c, h, w) = (2usize, 3usize, 4usize);
        let src: Vec<f32> = (0..c * h * w).map(|i| (i + 1) as f32).collect();
        for half in [false, true] {
            let t = if half {
                GpuTensor::upload_f32_as_half(vec![c, h, w], &src)
            } else {
                GpuTensor::upload_f32_as_f16(vec![c, h, w], &src)
            };
            let out = pad_end_1_f16_gpu(&t, c, h, w).download_to_f32();
            assert_eq!(out.len(), c * (h + 1) * (w + 1));
            let (ho, wo) = (h + 1, w + 1);
            for ci in 0..c {
                for y in 0..ho {
                    for x in 0..wo {
                        let got = out[(ci * ho + y) * wo + x];
                        let want = if y < h && x < w {
                            src[(ci * h + y) * w + x]
                        } else {
                            0.0
                        };
                        assert_eq!(got, want, "half={half} c={ci} y={y} x={x}");
                    }
                }
            }
        }
    }

    /// cos_sim -0.116.
    #[test]
    fn transpose_outer_is_a_2d_transpose_when_p_is_one() {
        let (a, b) = (3usize, 4usize);
        // src[i][j] = i*10 + j
        let src: Vec<f32> = (0..a * b).map(|e| ((e / b) * 10 + (e % b)) as f32).collect();
        let g = GpuTensor::upload_f32_as_f16(vec![a, b], &src);
        let t = transpose_outer_f16_gpu(&g, a, b, 1);
        let got = t.download_to_f32();
        assert_eq!(got.len(), a * b);
        let mut want = vec![0.0f32; a * b];
        for i in 0..a {
            for j in 0..b {
                want[j * a + i] = src[i * b + j];
            }
        }
        println!("src  {src:?}");
        println!("got  {got:?}");
        println!("want {want:?}");
        assert_eq!(got, want, "transpose_outer(p=1) is not [a,b] -> [b,a]");
    }
}

#[cfg(test)]
mod attn_bench {
    use super::*;

    /// Flash-attention throughput at the DiT's real shape.
    ///
    /// After the weight-transpose fix, attention is 68.6% of a native-resolution
    /// forward and runs at roughly 1.1 TF/s — about a tenth of what the
    /// simdgroup units should reach. This measures it directly so the kernel can
    /// be iterated on without loading the 10 GB model.
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture attention_throughput
    #[test]
    #[ignore]
    fn attention_throughput() {
        let (n, h, d) = (1usize, 24usize, 128usize);
        let mut seed = 31337u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        println!("{:<8} {:>12} {:>12} {:>12}", "L", "GFLOP", "time", "TFLOP/s");
        for l in [2048usize, 5070, 11440] {
            let elems = n * h * l * d;
            let q = GpuTensor::upload_f32_as_f16(
                vec![n, h, l, d], &(0..elems).map(|_| rnd()).collect::<Vec<_>>());
            let k = GpuTensor::upload_f32_as_f16(
                vec![n, h, l, d], &(0..elems).map(|_| rnd()).collect::<Vec<_>>());
            let v = GpuTensor::upload_f32_as_f16(
                vec![n, h, l, d], &(0..elems).map(|_| rnd()).collect::<Vec<_>>());
            let mut o = GpuTensor::new_f16(vec![n, h, l, d]);
            let scale = 1.0 / (d as f32).sqrt();
            // QK^T and PV: 2 * L * L * d each.
            let gflop = 2.0 * 2.0 * l as f64 * l as f64 * d as f64 * h as f64 / 1e9;

            flash_attention_best_f16_gpu(&q, &k, &v, &mut o, n, h, l, l, d, scale);
            flush();
            let t = std::time::Instant::now();
            let reps = 3;
            for _ in 0..reps {
                flash_attention_best_f16_gpu(&q, &k, &v, &mut o, n, h, l, l, d, scale);
            }
            flush();
            let dt = t.elapsed().as_secs_f64() / reps as f64;
            println!("{l:<8} {gflop:>10.1} GF {:>10.1} ms {:>10.2} TF/s",
                     dt * 1e3, gflop / 1e3 / dt);
        }
    }
}

#[cfg(test)]
mod attn_occupancy {
    use super::*;

    /// How much threadgroup memory the flash kernels ask for, against the
    /// device limit. Occupancy is capped by this: at D=128 the split kernel's
    /// staging is large enough that few threadgroups fit per core, which would
    /// explain a throughput plateau that does not move with head-dim.
    #[test]
    #[ignore]
    fn flash_threadgroup_memory_budget() {
        let dev = &get_metal().device;
        let limit = dev.max_threadgroup_memory_length() as usize;
        println!("device max threadgroup memory: {} bytes ({} KB)", limit, limit / 1024);
        const BM: usize = 32;
        const BN: usize = 32;
        const BNS: usize = 64;   // split kernel's traversal block
        for d in [64usize, 96, 128] {
            // split kernel's formula, in floats
            // Split: Q tile as bfloat (not a full f32 output stage) plus the two
            // score buffers at the wider traversal block.
            let split = BM * d / 2 + 2 * BM * BNS + (BM * BNS) / 2 + 3 * BM;
            let tiled = BM * d + BM * BN + (BM * BN) / 2 + 3 * BM + 4 * 64;
            let sb = split * 4;
            let tb = tiled * 4;
            println!(
                "d={d:<4} split {:>6} B ({:>4.1} KB, {:.2} threadgroups/core)   \
                 tiled {:>6} B ({:>4.1} KB, {:.2}/core)",
                sb, sb as f64 / 1024.0, limit as f64 / sb as f64,
                tb, tb as f64 / 1024.0, limit as f64 / tb as f64,
            );
        }
    }
}

#[cfg(test)]
mod matmul_stability {
    use super::*;

    /// Which matmul is the intermittent one?
    ///
    /// `chunked_sgemm_f16_buf_matches_reference` fails ~2 runs in 3 with
    /// cos_sim ~0.33, at varying M — but it compares MPS against the bf16 MMA
    /// kernel, so either side could be at fault. This runs each path against
    /// ITSELF repeatedly: whichever one disagrees with its own previous output
    /// on identical inputs is the unstable one.
    #[test]
    #[ignore]
    fn which_matmul_is_unstable() {
        let (m, n, k) = (11440usize, 3072usize, 3072usize);
        let mut seed = 5150u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rnd() * 0.05).collect();
        let ga = GpuTensor::upload_f32_as_f16(vec![m, k], &a);
        let gb = GpuTensor::upload_f32_as_f16(vec![n, k], &b);

        for (label, use_mma) in [("mps_f32 ", false), ("bf16_mma", true)] {
            let mut first: Option<Vec<f32>> = None;
            let mut mismatches = 0usize;
            let reps = 8;
            for _ in 0..reps {
                set_use_bf16_gemm(use_mma);
                set_use_f32_sgemm(!use_mma);
                let out = GpuTensor::new_f16(vec![m, n]);
                sgemm_f16_a_btrans_buf(&out.buffer, &ga.buffer, &gb.buffer, m, n, k);
                let v = out.download_to_f32();
                set_use_bf16_gemm(false);
                set_use_f32_sgemm(false);
                match &first {
                    None => first = Some(v),
                    Some(f) => {
                        let worst = f.iter().zip(v.iter())
                            .fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()));
                        if worst > 1e-6 {
                            mismatches += 1;
                            println!("  {label}: run differs from run 0, max_abs={worst:.4e}");
                        }
                    }
                }
            }
            println!("{label}: {mismatches}/{} repeats disagreed with the first",
                     reps - 1);
        }
    }
}

#[cfg(test)]
mod matmul_truth {
    use super::*;

    /// When MPS and the bf16 MMA kernel disagree, which one is wrong?
    ///
    /// `chunked_sgemm_f16_buf_matches_reference` fails ~2 processes in 3 with
    /// cos_sim ~0.33, and both paths are self-consistent within a process — so
    /// one of them is deterministically wrong for that process, and comparing
    /// them to each other cannot say which.
    ///
    /// A full CPU matmul at these sizes is 108 GFLOP, but a single output row
    /// is only n*k = 9.4 MFLOP. Checking a handful of rows against exact f32
    /// arithmetic is enough to convict.
    ///
    ///   cargo test -p klearu-diffusion --features metal --lib --release \
    ///     -- --ignored --nocapture matmul_ground_truth
    #[test]
    #[ignore]
    fn matmul_ground_truth() {
        let (m, n, k) = (11440usize, 3072usize, 3072usize);
        let mut seed = 20260904u32;
        let mut rnd = || {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        // Round the inputs to bf16 up front so the CPU reference sees exactly
        // the values the GPU does; otherwise the comparison carries an
        // irrelevant input-quantisation error.
        let q = |v: f32| half::bf16::from_f32(v).to_f32();
        let a: Vec<f32> = (0..m * k).map(|_| q(rnd())).collect();
        let b: Vec<f32> = (0..n * k).map(|_| q(rnd() * 0.05)).collect();
        let ga = GpuTensor::upload_f32_as_f16(vec![m, k], &a);
        let gbt = GpuTensor::upload_f32_as_f16(vec![n, k], &b);

        let run = |use_mma: bool| -> Vec<f32> {
            set_use_bf16_gemm(use_mma);
            set_use_f32_sgemm(!use_mma);
            let out = GpuTensor::new_f16(vec![m, n]);
            sgemm_f16_a_btrans_buf(&out.buffer, &ga.buffer, &gbt.buffer, m, n, k);
            let v = out.download_to_f32();
            set_use_bf16_gemm(false);
            set_use_f32_sgemm(false);
            v
        };
        // Mirror the failing test's structure: several sizes and both storage
        // modes in one process. The single-shape version above never
        // reproduced, which already says the trigger is not the arithmetic of
        // any one matmul but something about the sequence of allocations.
        let gb_half = GpuTensor::upload_f32_as_half(vec![n, k], &b);
        let ga_half = GpuTensor::upload_f32_as_half(vec![m, k], &a);
        for half_io in [false, true] {
            for mm in [4096usize, 8192, 8193, 11440] {
                let (aa, bb) = if half_io { (&ga_half, &gb_half) } else { (&ga, &gbt) };
                set_use_f32_sgemm(true);
                set_sgemm_half_io(half_io);
                let o = if half_io {
                    GpuTensor::new_half(vec![mm, n])
                } else {
                    GpuTensor::new_f16(vec![mm, n])
                };
                sgemm_f16_a_btrans_buf(&o.buffer, &aa.buffer, &bb.buffer, mm, n, k);
                let got = o.download_to_f32();
                set_sgemm_half_io(false);
                set_use_f32_sgemm(false);

                // Convict against exact f32 on the last row, which is where the
                // bottom-aligned final chunk lands.
                let row = mm - 1;
                let mut err = 0.0f32;
                let mut refmax = 0.0f32;
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for i in 0..k {
                        acc += (a[row * k + i] as f64) * (b[j * k + i] as f64);
                    }
                    let want = acc as f32;
                    refmax = refmax.max(want.abs());
                    err = err.max((want - got[row * n + j]).abs());
                }
                let store = if half_io { "half" } else { "bf16" };
                let verdict = if err > 0.05 * refmax { "  <-- WRONG" } else { "" };
                println!("{store:<5} m={mm:>6} last-row err={err:.4e} |ref|max={refmax:.4}{verdict}");
            }
        }

        let mps = run(false);
        let mma = run(true);

        // Agreement between the two GPU paths, so this run is labelled.
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in mma.iter().zip(mps.iter()) {
            d += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        println!("mps vs mma: cos_sim={:.8}", d / (na.sqrt() * nb.sqrt() + 1e-12));

        // Exact f32 reference for a few rows, including the chunk boundaries.
        let bound = MPS_MAX_ROWS;
        for &row in &[0usize, bound - 1, bound, bound + 1, m / 2, m - 1] {
            if row >= m { continue; }
            let mut err_mps = 0.0f32;
            let mut err_mma = 0.0f32;
            let mut ref_absmax = 0.0f32;
            for j in 0..n {
                let mut acc = 0.0f64;
                for i in 0..k {
                    acc += (a[row * k + i] as f64) * (b[j * k + i] as f64);
                }
                let want = acc as f32;
                ref_absmax = ref_absmax.max(want.abs());
                err_mps = err_mps.max((want - mps[row * n + j]).abs());
                err_mma = err_mma.max((want - mma[row * n + j]).abs());
            }
            println!(
                "row {row:>6}: |ref|max={ref_absmax:8.4}  mps_err={err_mps:.4e}  \
                 mma_err={err_mma:.4e}"
            );
        }
    }
}
