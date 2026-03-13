pub mod wasm_transport;
pub mod wasm_party;

use wasm_bindgen::prelude::*;

use klearu_mpc::fixed_point::{
    from_fixed, from_fixed64, to_fixed, to_fixed64, FRAC_BITS_64, SCALE_64,
};
use klearu_mpc::beaver::{DummyTripleGen, DummyTripleGen128};

// --- Fixed-point conversions ---

#[wasm_bindgen]
pub fn fixed_to_f32(x: u32) -> f32 {
    from_fixed(x)
}

#[wasm_bindgen]
pub fn f32_to_fixed(x: f32) -> u32 {
    to_fixed(x)
}

#[wasm_bindgen]
pub fn fixed64_to_f32(x: u64) -> f32 {
    from_fixed64(x)
}

#[wasm_bindgen]
pub fn f32_to_fixed64(x: f32) -> u64 {
    to_fixed64(x)
}

// --- Embedding share (Lower security) ---

/// Create a Q16.16 embedding share from f32 values.
/// party=0 (client): share = to_fixed(v) for each v
/// party=1 (server): share = zeros
/// Returns raw bytes (u32 LE array).
#[wasm_bindgen]
pub fn create_embedding_share(party: u8, embedding_f32: &[f32]) -> Vec<u8> {
    let share: Vec<u32> = if party == 0 {
        embedding_f32.iter().map(|&v| to_fixed(v)).collect()
    } else {
        vec![0u32; embedding_f32.len()]
    };
    u32_slice_to_bytes(&share)
}

/// Reconstruct f32 embedding from two u32 shares (as raw bytes).
#[wasm_bindgen]
pub fn reconstruct_embedding(my_share: &[u8], other_share: &[u8]) -> Vec<f32> {
    let a = bytes_to_u32_slice(my_share);
    let b = bytes_to_u32_slice(other_share);
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| from_fixed(x.wrapping_add(y)))
        .collect()
}

// --- Linear forward (public weights, shared input) ---

/// Q16.16 shared linear forward: y_share = W * x_share.
/// Weights are public f32, input is Q16.16 share (raw bytes).
/// Returns Q16.16 output share (raw bytes).
#[wasm_bindgen]
pub fn shared_linear_forward(
    weights: &[f32],
    in_features: u32,
    out_features: u32,
    x_share_bytes: &[u8],
) -> Vec<u8> {
    let in_f = in_features as usize;
    let out_f = out_features as usize;
    let x_share = bytes_to_u32_slice(x_share_bytes);
    let stride = if out_f > 0 { weights.len() / out_f } else { in_f };

    let mut output = Vec::with_capacity(out_f);
    for j in 0..out_f {
        let row_offset = j * stride;
        let mut acc = 0.0f64;
        for i in 0..in_f {
            acc += weights[row_offset + i] as f64 * x_share[i] as i32 as f64;
        }
        output.push(acc.round() as i64 as i32 as u32);
    }

    u32_slice_to_bytes(&output)
}

/// Q32.32 shared linear forward: y_share = W * x_share.
/// Returns Q32.32 output share (raw bytes as u64 LE).
#[wasm_bindgen]
pub fn shared_linear_forward_64(
    weights: &[f32],
    in_features: u32,
    out_features: u32,
    x_share_bytes: &[u8],
) -> Vec<u8> {
    let in_f = in_features as usize;
    let out_f = out_features as usize;
    let x_share = bytes_to_u64_slice(x_share_bytes);
    let stride = if out_f > 0 { weights.len() / out_f } else { in_f };

    let mut output = Vec::with_capacity(out_f);
    for j in 0..out_f {
        let row_offset = j * stride;
        let mut acc = 0i128;
        for i in 0..in_f {
            let w_q32 = (weights[row_offset + i] as f64 * SCALE_64).round() as i64;
            acc += (w_q32 as i128) * (x_share[i] as i64 as i128);
        }
        output.push(((acc >> 32) as i64) as u64);
    }

    u64_slice_to_bytes(&output)
}

// --- Beaver multiply helpers (stateless, for async WS orchestration) ---

/// Compute d and e shares for Beaver multiply exchange.
/// d = x_share - triple_a, e = y_share - triple_b
/// Returns concatenated [d_shares..., e_shares...] as raw bytes (u64 LE).
#[wasm_bindgen]
pub fn beaver_compute_d_e(
    x_share_bytes: &[u8],
    triple_a_bytes: &[u8],
    y_share_bytes: &[u8],
    triple_b_bytes: &[u8],
) -> Vec<u8> {
    let x = bytes_to_u64_slice(x_share_bytes);
    let a = bytes_to_u64_slice(triple_a_bytes);
    let y = bytes_to_u64_slice(y_share_bytes);
    let b = bytes_to_u64_slice(triple_b_bytes);

    let n = x.len();
    let mut result = Vec::with_capacity(2 * n);
    for i in 0..n {
        result.push(x[i].wrapping_sub(a[i]));
    }
    for i in 0..n {
        result.push(y[i].wrapping_sub(b[i]));
    }

    u64_slice_to_bytes(&result)
}

/// Finalize Beaver multiply after d,e exchange.
/// z = c + a*e + d*b + (party==0 ? d*e : 0), truncated >> 16
/// Returns Q16.16 product shares as raw bytes.
#[wasm_bindgen]
pub fn beaver_finalize(
    party: u8,
    triple_a_bytes: &[u8],
    triple_b_bytes: &[u8],
    triple_c_bytes: &[u8],
    d_full_bytes: &[u8],
    e_full_bytes: &[u8],
) -> Vec<u8> {
    let a = bytes_to_u64_slice(triple_a_bytes);
    let b = bytes_to_u64_slice(triple_b_bytes);
    let c = bytes_to_u64_slice(triple_c_bytes);
    let d = bytes_to_u64_slice(d_full_bytes);
    let e = bytes_to_u64_slice(e_full_bytes);

    let n = a.len();
    let mut output = Vec::with_capacity(n);
    for i in 0..n {
        let mut z = c[i];
        z = z.wrapping_add(a[i].wrapping_mul(e[i]));
        z = z.wrapping_add(d[i].wrapping_mul(b[i]));
        if party == 0 {
            z = z.wrapping_add(d[i].wrapping_mul(e[i]));
        }
        // Truncate Q32.32 product → Q16.16
        let result = ((z as i64) >> 16) as i32 as u32;
        output.push(result);
    }

    u32_slice_to_bytes(&output)
}

/// Compute d and e shares for Beaver multiply in Z_{2^128} (Q32.32 mode).
/// Returns concatenated [d_shares..., e_shares...] as raw bytes (u128 LE).
#[wasm_bindgen]
pub fn beaver_compute_d_e_128(
    x_share_bytes: &[u8],
    triple_a_bytes: &[u8],
    y_share_bytes: &[u8],
    triple_b_bytes: &[u8],
) -> Vec<u8> {
    let x = bytes_to_u64_slice(x_share_bytes);
    let a = bytes_to_u128_slice(triple_a_bytes);
    let y = bytes_to_u64_slice(y_share_bytes);
    let b = bytes_to_u128_slice(triple_b_bytes);

    let n = x.len();
    let mut result = Vec::with_capacity(2 * n);
    for i in 0..n {
        let x_ext = x[i] as i64 as i128 as u128;
        result.push(x_ext.wrapping_sub(a[i]));
    }
    for i in 0..n {
        let y_ext = y[i] as i64 as i128 as u128;
        result.push(y_ext.wrapping_sub(b[i]));
    }

    u128_slice_to_bytes(&result)
}

/// Finalize Beaver multiply in Z_{2^128} after d,e exchange.
/// Returns Q32.32 product shares as raw bytes (u64 LE).
#[wasm_bindgen]
pub fn beaver_finalize_128(
    party: u8,
    triple_a_bytes: &[u8],
    triple_b_bytes: &[u8],
    triple_c_bytes: &[u8],
    d_full_bytes: &[u8],
    e_full_bytes: &[u8],
) -> Vec<u8> {
    let a = bytes_to_u128_slice(triple_a_bytes);
    let b = bytes_to_u128_slice(triple_b_bytes);
    let c = bytes_to_u128_slice(triple_c_bytes);
    let d = bytes_to_u128_slice(d_full_bytes);
    let e = bytes_to_u128_slice(e_full_bytes);

    let n = a.len();
    let mut output = Vec::with_capacity(n);
    for i in 0..n {
        let mut z = c[i];
        z = z.wrapping_add(a[i].wrapping_mul(e[i]));
        z = z.wrapping_add(d[i].wrapping_mul(b[i]));
        if party == 0 {
            z = z.wrapping_add(d[i].wrapping_mul(e[i]));
        }
        let result = ((z as i128) >> FRAC_BITS_64) as i64 as u64;
        output.push(result);
    }

    u64_slice_to_bytes(&output)
}

// --- RMSNorm (stateless finalization after sum_sq reveal) ---

/// Finalize RMSNorm scaling locally after sum_sq has been revealed.
/// Computes inv_rms = 1/sqrt(sum_sq/n + eps), then scales each share by
/// inv_rms * weight[i] using f64 mixed-precision.
/// Input: Q16.16 x_share (raw bytes), weights (f32), eps, revealed sum_sq.
/// Returns: Q16.16 scaled share (raw bytes).
#[wasm_bindgen]
pub fn rmsnorm_finalize(
    x_share_bytes: &[u8],
    weights: &[f32],
    eps: f32,
    revealed_sum_sq: f64,
) -> Vec<u8> {
    let x_share = bytes_to_u32_slice(x_share_bytes);
    let n = x_share.len();

    let mean_sq = revealed_sum_sq / n as f64;
    let inv_rms = 1.0 / (mean_sq + eps as f64).sqrt();

    let mut output = Vec::with_capacity(n);
    for i in 0..n {
        let scale = inv_rms * weights[i] as f64;
        let x = x_share[i] as i32 as f64;
        output.push((scale * x).round() as i64 as i32 as u32);
    }

    u32_slice_to_bytes(&output)
}

/// Q32.32 RMSNorm finalization.
#[wasm_bindgen]
pub fn rmsnorm_finalize_64(
    x_share_bytes: &[u8],
    weights: &[f32],
    eps: f32,
    revealed_sum_sq: f64,
) -> Vec<u8> {
    let x_share = bytes_to_u64_slice(x_share_bytes);
    let n = x_share.len();

    let mean_sq = revealed_sum_sq / n as f64;
    let inv_rms = 1.0 / (mean_sq + eps as f64).sqrt();

    let mut output = Vec::with_capacity(n);
    for i in 0..n {
        let scale = inv_rms * weights[i] as f64;
        let scale_q32 = (scale * SCALE_64).round() as i64;
        let x = x_share[i] as i64;
        output.push((((scale_q32 as i128) * (x as i128)) >> FRAC_BITS_64) as i64 as u64);
    }

    u64_slice_to_bytes(&output)
}

// --- Dummy triple generation (for dev/testing with trusted dealer) ---

/// Generate dummy Beaver triples for one party.
/// Returns raw bytes: [a0, b0, c0, a1, b1, c1, ...] as u64 LE (3 * count values).
#[wasm_bindgen]
pub fn dummy_triple_gen(party: u8, seed: f64, count: u32) -> Vec<u8> {
    let mut gen = DummyTripleGen::new(party, seed as u64);
    let triples = klearu_mpc::beaver::TripleGenerator::generate(&mut gen, count as usize);
    let mut result = Vec::with_capacity(triples.len() * 3);
    for t in &triples {
        result.push(t.a);
        result.push(t.b);
        result.push(t.c);
    }
    u64_slice_to_bytes(&result)
}

/// Generate dummy Beaver128 triples for one party.
/// Returns raw bytes: [a0, b0, c0, a1, b1, c1, ...] as u128 LE (3 * count values).
#[wasm_bindgen]
pub fn dummy_triple_gen_128(party: u8, seed: f64, count: u32) -> Vec<u8> {
    let mut gen = DummyTripleGen128::new(party, seed as u64);
    let triples = klearu_mpc::beaver::TripleGenerator128::generate(&mut gen, count as usize);
    let mut result = Vec::with_capacity(triples.len() * 3);
    for t in &triples {
        result.push(t.a);
        result.push(t.b);
        result.push(t.c);
    }
    u128_slice_to_bytes(&result)
}

// --- Share addition/subtraction (local ops, no communication) ---

/// Element-wise wrapping add of two u32 share vectors (raw bytes).
#[wasm_bindgen]
pub fn shares_add(a_bytes: &[u8], b_bytes: &[u8]) -> Vec<u8> {
    let a = bytes_to_u32_slice(a_bytes);
    let b = bytes_to_u32_slice(b_bytes);
    let result: Vec<u32> = a.iter().zip(b.iter()).map(|(&x, &y)| x.wrapping_add(y)).collect();
    u32_slice_to_bytes(&result)
}

/// Element-wise wrapping add of two u64 share vectors (raw bytes).
#[wasm_bindgen]
pub fn shares_add_64(a_bytes: &[u8], b_bytes: &[u8]) -> Vec<u8> {
    let a = bytes_to_u64_slice(a_bytes);
    let b = bytes_to_u64_slice(b_bytes);
    let result: Vec<u64> = a.iter().zip(b.iter()).map(|(&x, &y)| x.wrapping_add(y)).collect();
    u64_slice_to_bytes(&result)
}

// --- Byte helpers ---

fn u32_slice_to_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn bytes_to_u32_slice(bytes: &[u8]) -> Vec<u32> {
    let n = bytes.len() / 4;
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 4;
        values.push(u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]));
    }
    values
}

fn u64_slice_to_bytes(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn bytes_to_u64_slice(bytes: &[u8]) -> Vec<u64> {
    let n = bytes.len() / 8;
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 8;
        values.push(u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]));
    }
    values
}

fn u128_slice_to_bytes(values: &[u128]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 16);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn bytes_to_u128_slice(bytes: &[u8]) -> Vec<u128> {
    let n = bytes.len() / 16;
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 16;
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&bytes[offset..offset + 16]);
        values.push(u128::from_le_bytes(arr));
    }
    values
}
