//! Private (2PC) MLP forward pass.
//!
//! SwiGLU MLP: output = down_proj(silu(gate_proj(x)) * up_proj(x))
//!
//! Since weights are public to both parties (semi-honest model),
//! linear projections are local.
//!
//! SwiGLU uses reveal-and-compute: gate values are revealed (leaks gate
//! activations, acceptable in semi-honest model) so exact SiLU can be
//! applied. The result multiplies the up_share locally (public * share).
//! This avoids the polynomial SiLU approximation which diverges for
//! values outside [-3, 3].

use klearu_llm::model::mlp::Mlp;
use klearu_mpc::beaver::{TripleGenerator, TripleGenerator128};
use klearu_mpc::activation::swiglu_noreveal_64;
use klearu_mpc::fixed_point::{from_fixed, from_fixed64, SCALE_64};
use klearu_mpc::linear::{shared_linear_forward, shared_linear_forward_64_pq, shared_linear_forward_sparse};
use klearu_mpc::transport::Transport;
use klearu_mpc::{SharedVec, SharedVec64};
use std::io;

/// Exact SiLU: x * sigmoid(x).
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Reveal-and-compute SwiGLU on shared gate and up values.
///
/// 1. Reveal gate_share → gate_plain (leaks gate activations)
/// 2. Compute silu(gate_plain) → public values
/// 3. silu(gate) * up_share → local multiply (public * share)
///
/// No Beaver triples needed.
fn swiglu_reveal(
    _party: u8,
    gate_share: &SharedVec,
    up_share: &SharedVec,
    transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    let n = gate_share.len();
    assert_eq!(n, up_share.len());

    // Reveal gate shares
    transport.send_u32_slice(&gate_share.0)?;
    let gate_other = transport.recv_u32_slice(n)?;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let gate_plain = from_fixed(gate_share.0[i].wrapping_add(gate_other[i]));
        let silu_val = silu(gate_plain);
        // silu(gate) * up_share: public scalar * share = local multiply
        // Use f64 mixed-precision to avoid Q16.16 quantization of silu_val
        let silu_f64 = silu_val as f64;
        let up_f64 = up_share.0[i] as i32 as f64;
        result.push((silu_f64 * up_f64).round() as i64 as i32 as u32);
    }

    Ok(SharedVec(result))
}

/// Private sparse MLP forward pass.
///
/// `mlp`: the plaintext model weights (both parties have these).
/// `x_share`: this party's share of the input hidden state.
/// `neuron_indices`: selected neurons from OPRF-based lookup.
pub fn private_sparse_mlp_forward(
    party: u8,
    mlp: &Mlp,
    x_share: &SharedVec,
    neuron_indices: &[usize],
    _triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    let in_features = mlp.gate_proj.in_features();

    let gate_out_features = mlp.gate_proj.out_features();

    // 1. gate_proj
    let gate_weights = mlp.gate_proj.weights.as_raw_slice();
    let gate_share = shared_linear_forward_sparse(
        party, gate_weights, in_features, gate_out_features, neuron_indices, x_share, &[], transport,
    )?;

    // 2. up_proj
    let up_weights = mlp.up_proj.weights.as_raw_slice();
    let up_out_features = mlp.up_proj.out_features();
    let up_share = shared_linear_forward_sparse(
        party, up_weights, in_features, up_out_features, neuron_indices, x_share, &[], transport,
    )?;

    // 3. SwiGLU via reveal-and-compute (no triples needed)
    let activated = swiglu_reveal(party, &gate_share, &up_share, transport)?;

    // 4. down_proj: gather selected rows and project back
    let down_weights = mlp.down_proj.weights.as_raw_slice();
    let out_features = mlp.down_proj.out_features();
    let down_total_rows = out_features;
    let down_stride = if down_total_rows > 0 { down_weights.len() / down_total_rows } else { 0 };

    let n = neuron_indices.len();
    let mut sub_weights = Vec::with_capacity(out_features * n);
    for j in 0..out_features {
        for &idx in neuron_indices {
            sub_weights.push(down_weights[j * down_stride + idx]);
        }
    }

    shared_linear_forward(party, &sub_weights, n, out_features, &activated, &[], transport)
}

/// Private dense MLP forward pass.
pub fn private_dense_mlp_forward(
    party: u8,
    mlp: &Mlp,
    x_share: &SharedVec,
    _triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    let in_features = mlp.gate_proj.in_features();
    let out_features = mlp.gate_proj.out_features();

    let gate_share = shared_linear_forward(
        party, mlp.gate_proj.weights.as_raw_slice(), in_features, out_features, x_share, &[], transport,
    )?;
    let up_share = shared_linear_forward(
        party, mlp.up_proj.weights.as_raw_slice(), in_features, out_features, x_share, &[], transport,
    )?;

    // SwiGLU via reveal-and-compute (no triples needed)
    let activated = swiglu_reveal(party, &gate_share, &up_share, transport)?;

    let down_in = mlp.down_proj.in_features();
    let down_out = mlp.down_proj.out_features();
    shared_linear_forward(
        party, mlp.down_proj.weights.as_raw_slice(), down_in, down_out, &activated, &[], transport,
    )
}

// --- Q32.32 secure MLP ---

/// Reveal-and-compute SwiGLU on Q32.32 shared gate and up values.
///
/// Same protocol as Q16.16: reveal gate shares, compute SiLU(gate_plain),
/// multiply by up_share locally using i128 mixed-precision.
fn swiglu_reveal_64(
    _party: u8,
    gate_share: &SharedVec64,
    up_share: &SharedVec64,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let n = gate_share.len();
    assert_eq!(n, up_share.len());

    // Reveal gate shares (u64)
    transport.send_u64_slice(&gate_share.0)?;
    let gate_other = transport.recv_u64_slice(n)?;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let gate_plain = from_fixed64(gate_share.0[i].wrapping_add(gate_other[i]));
        let silu_val = silu(gate_plain);
        // silu(gate) * up_share: public scalar × Q32.32 share via i128
        let silu_q32 = (silu_val as f64 * SCALE_64).round() as i64;
        let up = up_share.0[i] as i64;
        result.push((((silu_q32 as i128) * (up as i128)) >> 32) as i64 as u64);
    }

    Ok(SharedVec64(result))
}

/// Secure (Q32.32) dense MLP forward pass using pre-quantized weights.
pub fn private_dense_mlp_forward_secure(
    party: u8,
    mlp: &Mlp,
    x_share: &SharedVec64,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let in_features = mlp.gate_proj.in_features();
    let out_features = mlp.gate_proj.out_features();

    let gate_share = shared_linear_forward_64_pq(
        &mlp.gate_proj.q32_weights(), in_features, out_features, x_share,
    );
    let up_share = shared_linear_forward_64_pq(
        &mlp.up_proj.q32_weights(), in_features, out_features, x_share,
    );

    let activated = swiglu_reveal_64(party, &gate_share, &up_share, transport)?;

    let down_in = mlp.down_proj.in_features();
    let down_out = mlp.down_proj.out_features();
    Ok(shared_linear_forward_64_pq(
        &mlp.down_proj.q32_weights(), down_in, down_out, &activated,
    ))
}

/// No-reveal dense MLP forward pass (Q32.32).
///
/// Uses polynomial SiLU (no gate reveal). 3 triples/neuron, 3 round-trips.
///
/// Protocol:
/// 1. gate_share = shared_linear_forward_64_pq(gate_proj, x_share) — local
/// 2. up_share = shared_linear_forward_64_pq(up_proj, x_share) — local
/// 3. activated = swiglu_noreveal_64(gate_share, up_share, triples, transport) — 3 RTs
/// 4. output = shared_linear_forward_64_pq(down_proj, activated) — local
///
/// Leakage: none (only Beaver d/e values exchanged).
pub fn private_dense_mlp_forward_noreveal(
    party: u8,
    mlp: &Mlp,
    x_share: &SharedVec64,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let in_features = mlp.gate_proj.in_features();
    let out_features = mlp.gate_proj.out_features();

    // 1. gate_proj (local)
    let gate_share = shared_linear_forward_64_pq(
        &mlp.gate_proj.q32_weights(), in_features, out_features, x_share,
    );

    // 2. up_proj (local)
    let up_share = shared_linear_forward_64_pq(
        &mlp.up_proj.q32_weights(), in_features, out_features, x_share,
    );

    // 3. SwiGLU via polynomial (no reveals): 3 triples/element, 3 round-trips
    let n = out_features;
    let needed = 3 * n; // swiglu_noreveal_64 needs 3n triples
    let swiglu_triples = triples.generate(needed);
    let activated = swiglu_noreveal_64(party, &gate_share, &up_share, &swiglu_triples, transport)?;

    // 4. down_proj (local)
    let down_in = mlp.down_proj.in_features();
    let down_out = mlp.down_proj.out_features();
    Ok(shared_linear_forward_64_pq(
        &mlp.down_proj.q32_weights(), down_in, down_out, &activated,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_llm::model::mlp::Mlp;
    use klearu_mpc::beaver::dummy_triple_pair;
    use klearu_mpc::fixed_point::to_fixed;
    use klearu_mpc::transport::memory_transport_pair;

    #[test]
    fn test_private_dense_mlp_compiles() {
        // Smoke test: just verify the code compiles and the types work
        let mlp = Mlp::new(4, 8);

        let input = vec![0.1f32, 0.2, -0.1, 0.05];
        let x_fixed: Vec<u32> = input.iter().map(|&v| to_fixed(v)).collect();

        let (mut gen0, mut gen1) = dummy_triple_pair(1000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let share0 = SharedVec(x_fixed);
        let share1 = SharedVec(vec![0u32; 4]);

        let handle = std::thread::spawn(move || {
            private_dense_mlp_forward(1, &Mlp::new(4, 8), &share1, &mut gen1, &mut trans_b)
        });

        let result0 = private_dense_mlp_forward(0, &mlp, &share0, &mut gen0, &mut trans_a);
        let result1 = handle.join().unwrap();

        // Both should succeed (zero weights, so output should be near zero)
        assert!(result0.is_ok());
        assert!(result1.is_ok());
    }
}
