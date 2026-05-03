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
    weight_ptr: usize,
    n: usize, in_c: usize, h_in: usize, w_in: usize,
    out_c: usize, kh: usize, kw: usize,
    stride: usize, pad: usize,
}

struct CachedConvGraph {
    graph: *mut AnyObject,           // MPSGraph
    input_tensor: *mut AnyObject,    // MPSGraphTensor (placeholder)
    weight_tensor: *mut AnyObject,   // MPSGraphTensor (placeholder)
    output_tensor: *mut AnyObject,   // MPSGraphTensor
    h_out: usize, w_out: usize,
}

// Send/Sync: the cached objc objects are never sent across threads — the
// thread_local enforces single-thread access. We mark them Sync via raw
// pointer wrapper so the thread_local with! works.
unsafe impl Send for CachedConvGraph {}
unsafe impl Sync for CachedConvGraph {}

thread_local! {
    static CONV_GRAPH_CACHE: std::cell::RefCell<HashMap<ConvGraphKey, CachedConvGraph>>
        = std::cell::RefCell::new(HashMap::new());
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
const MPSGRAPH_LAYOUT_NCHW: i64 = 0;
const MPSGRAPH_LAYOUT_OIHW: i64 = 2;
const MPSGRAPH_PADDING_EXPLICIT: i64 = 0;

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
    // fp16 GPU-residence kernels.
    silu_f16_pipe: ComputePipelineState,
    gelu_f16_pipe: ComputePipelineState,
    quick_gelu_f16_pipe: ComputePipelineState,
    eadd_f16_pipe: ComputePipelineState,
    layernorm_f16_pipe: ComputePipelineState,
    layernorm_f16_out_pipe: ComputePipelineState,
    groupnorm_f16_pipe: ComputePipelineState,
    groupnorm_silu_f16_out_pipe: ComputePipelineState,
    groupnorm_silu_f16_inplace_pipe: ComputePipelineState,
    cast_f32_to_f16_pipe: ComputePipelineState,
    cast_f16_to_f32_pipe: ComputePipelineState,
    cast_bf16_to_half_pipe: ComputePipelineState,
    cast_half_to_bf16_pipe: ComputePipelineState,
    bias_add_f16_pipe: ComputePipelineState,
    im2col_f16_pipe: ComputePipelineState,
    bias_add_nc_pipe: ComputePipelineState,
    upsample_2x_pipe: ComputePipelineState,
    cat_channels_pipe: ComputePipelineState,
    flash_attn_f16_pipe: ComputePipelineState,
    permute_lh_to_hl_pipe: ComputePipelineState,
    permute_lh_to_hl_strided_pipe: ComputePipelineState,
    permute_hl_to_lh_pipe: ComputePipelineState,
    geglu_split_pipe: ComputePipelineState,
    nchw_to_nhwc_pipe: ComputePipelineState,
    nhwc_to_nchw_pipe: ComputePipelineState,
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
            silu_f16_pipe: make_pipe("silu_f16"),
            gelu_f16_pipe: make_pipe("gelu_f16"),
            quick_gelu_f16_pipe: make_pipe("quick_gelu_f16"),
            eadd_f16_pipe: make_pipe("elementwise_add_f16"),
            layernorm_f16_pipe: make_pipe("layer_norm_f16"),
            layernorm_f16_out_pipe: make_pipe("layer_norm_f16_out"),
            groupnorm_f16_pipe: make_pipe("groupnorm_f16"),
            groupnorm_silu_f16_out_pipe: make_pipe("groupnorm_silu_f16_out"),
            groupnorm_silu_f16_inplace_pipe: make_pipe("groupnorm_silu_f16_inplace"),
            cast_f32_to_f16_pipe: make_pipe("cast_f32_to_f16"),
            cast_f16_to_f32_pipe: make_pipe("cast_f16_to_f32"),
            cast_bf16_to_half_pipe: make_pipe("cast_bf16_to_half"),
            cast_half_to_bf16_pipe: make_pipe("cast_half_to_bf16"),
            bias_add_f16_pipe: make_pipe("bias_add_strided_f16"),
            im2col_f16_pipe: make_pipe("im2col_f16"),
            bias_add_nc_pipe: make_pipe("bias_add_nc_to_nchw_f16"),
            upsample_2x_pipe: make_pipe("nearest_upsample_2x_f16"),
            cat_channels_pipe: make_pipe("cat_channels_f16"),
            flash_attn_f16_pipe: make_pipe("flash_attention_f16"),
            permute_lh_to_hl_pipe: make_pipe("permute_lh_to_hl_f16"),
            permute_lh_to_hl_strided_pipe: make_pipe("permute_lh_to_hl_f16_strided"),
            permute_hl_to_lh_pipe: make_pipe("permute_hl_to_lh_f16"),
            geglu_split_pipe: make_pipe("geglu_split_f16"),
            nchw_to_nhwc_pipe: make_pipe("nchw_to_nhwc_f16"),
            nhwc_to_nchw_pipe: make_pipe("nhwc_to_nchw_f16"),
            device,
            queue,
        }
    })
}

// ---------- Buffer pool ----------------------------------------------------

thread_local! {
    static BUF_POOL: RefCell<HashMap<usize, Vec<Buffer>>> = RefCell::new(HashMap::new());
}

/// Acquire a `[bytes]`-sized MTLBuffer (Shared storage). Pulled from a
/// thread-local pool when one of the same size is available, else allocated.
fn acquire_buffer(bytes: usize) -> Buffer {
    BUF_POOL.with_borrow_mut(|pool| {
        if let Some(bufs) = pool.get_mut(&bytes) {
            if let Some(buf) = bufs.pop() {
                return buf;
            }
        }
        let backend = get_metal();
        backend.device.new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared)
    })
}

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
    static PENDING_COMMITS: std::cell::Cell<usize> = std::cell::Cell::new(0);

    /// Shared command buffer for the current "batch". When set, every
    /// `*_f16_gpu` kernel dispatch and MPS sgemm appends a new encoder to
    /// this buffer instead of creating its own — saving the
    /// `new_command_buffer + commit` overhead (~50-200μs CPU each) per
    /// kernel call. With ~600+ kernels per SDXL UNet step, this collapses
    /// to a single command-buffer creation+commit per `flush()`.
    static CURRENT_CMD: std::cell::RefCell<Option<metal::CommandBuffer>>
        = const { std::cell::RefCell::new(None) };

    /// When true, the MPS sgemm wrapper uses f32 scratch + MPS f32 sgemm
    /// (the slower-but-safer path). Default is false (fast f16 scratch
    /// path). Set true around code paths whose intermediate matmul
    /// outputs can exceed fp16 max (≈65504) — notably SDXL VAE decoder,
    /// where some activations grow to ~23K and MPS-f16-output overflow
    /// produces Inf → NaN in the final image.
    static USE_F32_SGEMM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Switch the MPS sgemm wrapper to the f32-precision path. Call before
/// VAE decode (or any other matmul-heavy code with large intermediate
/// magnitudes) and reset to false afterward.
pub fn set_use_f32_sgemm(v: bool) {
    USE_F32_SGEMM.with(|c| c.set(v));
}

/// Open a "batch" — subsequent kernel dispatches and MPS sgemms append to
/// a single shared command buffer, committed at the next `flush()`.
/// Idempotent if a batch is already open.
pub fn begin_batch() {
    CURRENT_CMD.with(|c| {
        let mut bw = c.borrow_mut();
        if bw.is_none() {
            let backend = get_metal();
            let cmd = backend.queue.new_command_buffer().to_owned();
            *bw = Some(cmd);
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
        } else {
            drop(bw);
            // No batch. Create + commit our own.
            let backend = get_metal();
            let cmd = backend.queue.new_command_buffer();
            f(cmd);
            cmd.commit();
            PENDING_COMMITS.with(|p| p.set(p.get() + 1));
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
    cmd.commit();
    PENDING_COMMITS.with(|c| c.set(c.get() + 1));
}

/// Wait for all in-flight GPU work to complete. Cheap if no async cmds are
/// pending. Call before any CPU read or fresh CPU write to a buffer that
/// may have been touched by an async-committed kernel. If a batch is open,
/// commits and clears it first.
pub fn flush() {
    // Commit and clear any open batch.
    CURRENT_CMD.with(|c| {
        if let Some(cmd) = c.borrow_mut().take() {
            cmd.commit();
            PENDING_COMMITS.with(|p| p.set(p.get() + 1));
        }
    });
    PENDING_COMMITS.with(|c| {
        if c.get() > 0 {
            let backend = get_metal();
            let barrier = backend.queue.new_command_buffer();
            barrier.commit();
            barrier.wait_until_completed();
            c.set(0);
        }
    });
}

/// Return a buffer to the pool for reuse.
fn release_buffer(buf: Buffer) {
    let bytes = buf.length() as usize;
    BUF_POOL.with_borrow_mut(|pool| {
        pool.entry(bytes).or_default().push(buf);
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
                // Hit. Clone bumps Objective-C retain; caller drops their copy
                // after use, leaving the cache entry alive at refcount 1.
                return cached.buffer.clone();
            }
            // Stale entry. Replace below — return its buffer to the pool first
            // so we don't leak GPU memory.
            if let Some(stale) = cache.remove(&key) {
                release_buffer(stale.buffer);
            }
        }

        // Bound cache size to prevent unbounded growth from activation pointers.
        if cache.len() >= F16_CACHE_MAX {
            if let Some(evict_key) = cache.keys().next().copied() {
                if let Some(evicted) = cache.remove(&evict_key) {
                    release_buffer(evicted.buffer);
                }
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
    let buf_a = acquire_buffer(a_bytes);
    let buf_b = acquire_buffer(b_bytes);
    let buf_c = acquire_buffer(c_bytes);

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
    let a_elems = m * k;
    let b_elems = b_rows * b_cols;
    let c_elems = m * n;

    if use_f32 {
        let a_f32 = acquire_buffer(a_elems * 4);
        let b_f32 = acquire_buffer(b_elems * 4);
        let c_f32 = acquire_buffer(c_elems * 4);
        cast_bf16_to_f32_buf(a_bf16, &a_f32, a_elems);
        cast_bf16_to_f32_buf(b_bf16, &b_f32, b_elems);
        dispatch_mps_matmul_with_transpose(
            backend, &a_f32, &b_f32, &c_f32,
            m, n, k, MPS_DATA_TYPE_FLOAT32, 4,
            transpose_left, transpose_right, b_rows, b_cols,
        );
        cast_f32_to_bf16_buf(&c_f32, c_bf16, c_elems);
        release_buffer(a_f32);
        release_buffer(b_f32);
        release_buffer(c_f32);
    } else {
        let a_f16 = acquire_buffer(a_elems * 2);
        let b_f16 = acquire_buffer(b_elems * 2);
        let c_f16 = acquire_buffer(c_elems * 2);
        cast_bf16_to_half_buf(a_bf16, &a_f16, a_elems);
        cast_bf16_to_half_buf(b_bf16, &b_f16, b_elems);
        dispatch_mps_matmul_with_transpose(
            backend, &a_f16, &b_f16, &c_f16,
            m, n, k, MPS_DATA_TYPE_FLOAT16, 2,
            transpose_left, transpose_right, b_rows, b_cols,
        );
        cast_half_to_bf16_buf(&c_f16, c_bf16, c_elems);
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
    let a_elems = m * k;
    let b_elems = b_rows * b_cols;
    let c_elems = m * n;
    let bytes = if use_f32 { 4 } else { 2 };
    let a_buf = acquire_buffer(a_elems * bytes);
    let b_buf = acquire_buffer(b_elems * bytes);
    let c_buf = acquire_buffer(c_elems * bytes);

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

    if use_f32 {
        dispatch_cast(&backend.cast_f16_to_f32_pipe, a_bf16, a_off_bytes_bf16, &a_buf, 0, a_elems);
        dispatch_cast(&backend.cast_f16_to_f32_pipe, b_bf16, b_off_bytes_bf16, &b_buf, 0, b_elems);
        dispatch_mps_matmul_with_offsets(
            backend, &a_buf, 0, &b_buf, 0, &c_buf, 0,
            m, n, k, MPS_DATA_TYPE_FLOAT32, 4,
            transpose_left, transpose_right, b_rows, b_cols,
        );
        dispatch_cast(&backend.cast_f32_to_f16_pipe, &c_buf, 0, c_bf16, c_off_bytes_bf16, c_elems);
    } else {
        dispatch_cast(&backend.cast_bf16_to_half_pipe, a_bf16, a_off_bytes_bf16, &a_buf, 0, a_elems);
        dispatch_cast(&backend.cast_bf16_to_half_pipe, b_bf16, b_off_bytes_bf16, &b_buf, 0, b_elems);
        dispatch_mps_matmul_with_offsets(
            backend, &a_buf, 0, &b_buf, 0, &c_buf, 0,
            m, n, k, MPS_DATA_TYPE_FLOAT16, 2,
            transpose_left, transpose_right, b_rows, b_cols,
        );
        dispatch_cast(&backend.cast_half_to_bf16_pipe, &c_buf, 0, c_bf16, c_off_bytes_bf16, c_elems);
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

// ---------- GpuTensor (GPU-resident pipeline) -----------------------------

/// Element dtype of a `GpuTensor`. The GPU-resident pipeline stores
/// activations as `F16` (bfloat the bandwidth, 4× the M1 GPU throughput vs
/// fp32). Boundary I/O is f32 — `upload_f32_as_f16` and `download_to_f32`
/// handle the conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDtype { F16, F32 }

impl GpuDtype {
    pub fn bytes(self) -> usize { match self { GpuDtype::F16 => 2, GpuDtype::F32 => 4 } }
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
            GpuDtype::F32 => unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buffer.contents() as *const f32, out.as_mut_ptr(), elements);
            },
        }
        out
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
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    dispatch_elementwise_buf_inplace(&get_metal().silu_f16_pipe, &x.buffer, x.elements());
}

/// In-place GELU on an fp16 GPU buffer.
pub fn gelu_f16_gpu(x: &mut GpuTensor) {
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    dispatch_elementwise_buf_inplace(&get_metal().gelu_f16_pipe, &x.buffer, x.elements());
}

/// In-place quick_gelu on an fp16 GPU buffer.
pub fn quick_gelu_f16_gpu(x: &mut GpuTensor) {
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    dispatch_elementwise_buf_inplace(&get_metal().quick_gelu_f16_pipe, &x.buffer, x.elements());
}

/// In-place a += b for fp16 GPU tensors of equal length.
pub fn eadd_f16_gpu(a: &mut GpuTensor, b: &GpuTensor) {
    use metal::MTLSize;
    debug_assert_eq!(a.dtype, GpuDtype::F16);
    debug_assert_eq!(b.dtype, GpuDtype::F16);
    debug_assert_eq!(a.elements(), b.elements());
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.eadd_f16_pipe);
        enc.set_buffer(0, Some(&a.buffer), 0);
        enc.set_buffer(1, Some(&b.buffer), 0);
        let group = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(a.elements() as u64, 1, 1);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
    });
}

/// In-place LayerNorm on an fp16 GPU buffer. `gamma`, `beta` are f16
/// GPU buffers (typically pre-uploaded weights from the cache).
pub fn layer_norm_f16_gpu(x: &mut GpuTensor, gamma: &Buffer, beta: &Buffer, dim: usize, eps: f32) {
    use metal::MTLSize;
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    debug_assert_eq!(x.elements() % dim, 0);
    let backend = get_metal();
    let n_rows = x.elements() / dim;
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.layernorm_f16_pipe);
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
    use metal::MTLSize;
    debug_assert_eq!(x.dtype, GpuDtype::F16);
    let backend = get_metal();
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.bias_add_f16_pipe);
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
pub fn sgemm_f16_a_btrans_buf(out: &Buffer, a: &Buffer, b: &Buffer,
                              m: usize, n: usize, k: usize) {
    let backend = get_metal();
    unsafe {
        dispatch_mps_matmul_bf16_via_f32(
            backend, a, b, out, m, n, k,
            false, true, n, k,
        );
    }
}

pub fn sgemm_f16_buf(out: &Buffer, a: &Buffer, b: &Buffer,
                     m: usize, n: usize, k: usize) {
    let backend = get_metal();
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
pub fn mpsgraph_conv2d(
    input_bf16: &Buffer,
    weight_f32: &[f32],
    n: usize, in_c: usize, h_in: usize, w_in: usize,
    out_c: usize, kh: usize, kw: usize,
    stride: usize, pad: usize,
) -> (Buffer, usize, usize) {
    let h_out = (h_in + 2 * pad - kh) / stride + 1;
    let w_out = (w_in + 2 * pad - kw) / stride + 1;

    // Cast input bf16 → f16 scratch.
    let input_elems = n * in_c * h_in * w_in;
    let input_f16 = acquire_buffer(input_elems * 2);
    cast_bf16_to_half_buf(input_bf16, &input_f16, input_elems);

    // Weight: bf16 cache → f16 cache. We use the bf16 weight cache as the
    // canonical storage and cast to f16 once (cached per weight slice).
    let weight_bf16 = weight_f16_buffer(weight_f32);
    let weight_elems = out_c * in_c * kh * kw;
    let weight_f16 = mpsgraph_weight_f16_cached(&weight_bf16, weight_f32.as_ptr() as usize, weight_elems);

    // Output buffer.
    let out_elems = n * out_c * h_out * w_out;
    let output_f16 = acquire_buffer(out_elems * 2);

    let key = ConvGraphKey {
        weight_ptr: weight_f32.as_ptr() as usize,
        n, in_c, h_in, w_in, out_c, kh, kw, stride, pad,
    };

    unsafe {
        encode_mpsgraph_conv(key, &input_f16, &weight_f16, &output_f16, h_out, w_out);
    }

    // Cast output f16 → bf16.
    let output_bf16 = acquire_buffer(out_elems * 2);
    cast_half_to_bf16_buf(&output_f16, &output_bf16, out_elems);

    release_buffer(input_f16);
    release_buffer(output_f16);

    (output_bf16, h_out, w_out)
}

thread_local! {
    static MPSGRAPH_WEIGHT_F16_CACHE: std::cell::RefCell<HashMap<usize, Buffer>>
        = std::cell::RefCell::new(HashMap::new());
}

/// One-time conversion of a bf16 weight buffer to a f16 buffer that lives
/// for the process lifetime. Keyed by source CPU pointer.
fn mpsgraph_weight_f16_cached(bf16_buf: &Buffer, src_ptr: usize, n_elements: usize) -> Buffer {
    MPSGRAPH_WEIGHT_F16_CACHE.with_borrow_mut(|cache| {
        if let Some(b) = cache.get(&src_ptr) {
            return b.clone();
        }
        let f16 = acquire_buffer(n_elements * 2);
        cast_bf16_to_half_buf(bf16_buf, &f16, n_elements);
        // Block here once to ensure cast completes before first use. Subsequent
        // hits never block.
        flush();
        let h = f16.clone();
        cache.insert(src_ptr, f16);
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

    let cached = CONV_GRAPH_CACHE.with_borrow_mut(|cache| {
        if let Some(c) = cache.get(&key) {
            // Existing cached pointers are valid (objc retain count maintained).
            (c.graph, c.input_tensor, c.weight_tensor, c.output_tensor)
        } else {
            // Cache miss — build the graph. Catch exceptions so we report
            // the actual NSException reason rather than the opaque "Rust
            // cannot catch foreign exceptions" abort.
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

    let input_data: *mut AnyObject = msg_send![tensor_data_cls, alloc];
    let input_data: *mut AnyObject = msg_send![input_data,
        initWithMTLBuffer: input_buf_ptr
        shape: input_shape
        dataType: MPS_DATA_TYPE_FLOAT16];

    let weight_data: *mut AnyObject = msg_send![tensor_data_cls, alloc];
    let weight_data: *mut AnyObject = msg_send![weight_data,
        initWithMTLBuffer: weight_buf_ptr
        shape: weight_shape
        dataType: MPS_DATA_TYPE_FLOAT16];

    let output_data: *mut AnyObject = msg_send![tensor_data_cls, alloc];
    let output_data: *mut AnyObject = msg_send![output_data,
        initWithMTLBuffer: output_buf_ptr
        shape: output_shape
        dataType: MPS_DATA_TYPE_FLOAT16];

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
        let _: () = msg_send![mps_cmd, commit];
        let _: () = msg_send![mps_cmd, waitUntilCompleted];
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
        let _: () = msg_send![graph, retain];

        let input_shape = make_shape(&[key.n, key.in_c, key.h_in, key.w_in]);
        let weight_shape = make_shape(&[key.out_c, key.in_c, key.kh, key.kw]);

        let input_tensor: *mut AnyObject = msg_send![graph,
            placeholderWithShape: input_shape
            dataType: MPS_DATA_TYPE_FLOAT16
            name: std::ptr::null_mut::<AnyObject>()];
        let weight_tensor: *mut AnyObject = msg_send![graph,
            placeholderWithShape: weight_shape
            dataType: MPS_DATA_TYPE_FLOAT16
            name: std::ptr::null_mut::<AnyObject>()];
        let _: () = msg_send![input_tensor, retain];
        let _: () = msg_send![weight_tensor, retain];

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
        let _: () = msg_send![output_tensor, retain];

        CachedConvGraph {
            graph,
            input_tensor,
            weight_tensor,
            output_tensor,
            h_out, w_out,
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
    // Drain MPSGraph's transient Cocoa allocations (NSNumber shapes,
    // NSArray feeds, MPSGraphTensorData wrappers, graph compilation
    // intermediates) on each dispatch. Without this, ~1750 attention
    // calls per generation accumulate GBs of autoreleased state.
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
            let _: () = msg_send![mps_cmd, commit];
            let _: () = msg_send![mps_cmd, waitUntilCompleted];
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
    let _: () = msg_send![graph, retain];

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
    let _: () = msg_send![q_tensor, retain];
    let _: () = msg_send![k_tensor, retain];
    let _: () = msg_send![v_tensor, retain];

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
    let _: () = msg_send![o_tensor, retain];

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
    let proj_in_w_buf = mpsgraph_weight_f16_cached(
        &proj_in_w_bf16, proj_in_w.as_ptr() as usize, proj_in_w.len());
    let proj_out_w_buf = mpsgraph_weight_f16_cached(
        &proj_out_w_bf16, proj_out_w.as_ptr() as usize, proj_out_w.len());
    let proj_in_b_buf = proj_in_b_bf16.as_ref().zip(proj_in_b).map(|(b, src)| {
        mpsgraph_weight_f16_cached(b, src.as_ptr() as usize, src.len())
    });
    let proj_out_b_buf = proj_out_b_bf16.as_ref().zip(proj_out_b).map(|(b, src)| {
        mpsgraph_weight_f16_cached(b, src.as_ptr() as usize, src.len())
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
            let _: () = msg_send![mps_cmd, commit];
            let _: () = msg_send![mps_cmd, waitUntilCompleted];
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
    let _: () = msg_send![graph, retain];

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
    let _: () = msg_send![x_tensor, retain];
    let _: () = msg_send![win_tensor, retain];
    let _: () = msg_send![wout_tensor, retain];

    let bin_tensor: *mut AnyObject = if key.has_in_bias {
        let t: *mut AnyObject = msg_send![graph,
            placeholderWithShape: make_shape(&[2 * key.hidden])
            dataType: MPS_DATA_TYPE_FLOAT16
            name: std::ptr::null_mut::<AnyObject>()];
        let _: () = msg_send![t, retain];
        t
    } else { std::ptr::null_mut() };

    let bout_tensor: *mut AnyObject = if key.has_out_bias {
        let t: *mut AnyObject = msg_send![graph,
            placeholderWithShape: make_shape(&[key.out_dim])
            dataType: MPS_DATA_TYPE_FLOAT16
            name: std::ptr::null_mut::<AnyObject>()];
        let _: () = msg_send![t, retain];
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
    let _: () = msg_send![o_tensor, retain];

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

/// Permute fp16 [N, L, H*D] → [N, H, L, D]. Returns fresh GpuTensor.
pub fn permute_lh_to_hl_f16_gpu(
    src: &GpuTensor, n: usize, l: usize, h: usize, d: usize,
) -> GpuTensor {
    use metal::MTLSize;
    debug_assert_eq!(src.dtype, GpuDtype::F16);
    let backend = get_metal();
    let dst = GpuTensor::new_f16(vec![n, h, l, d]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.permute_lh_to_hl_pipe);
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
    use metal::MTLSize;
    debug_assert_eq!(src.dtype, GpuDtype::F16);
    let backend = get_metal();
    let dst = GpuTensor::new_f16(vec![n, l, h * d]);
    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.permute_hl_to_lh_pipe);
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
pub fn nearest_upsample_2x_f16_gpu(
    input: &GpuTensor, n: usize, c: usize, h: usize, w: usize,
) -> GpuTensor {
    use metal::MTLSize;
    debug_assert_eq!(input.dtype, GpuDtype::F16);
    let backend = get_metal();
    let h_up = h * 2;
    let w_up = w * 2;
    let out = GpuTensor::new_f16(vec![n, c, h_up, w_up]);

    with_cmd(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&backend.upsample_2x_pipe);
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
}
