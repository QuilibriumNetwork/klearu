//! DPF-based Private Information Retrieval for embedding table lookup.
//!
//! Server A generates DPF keys for the target token_id, sends key_1 to Server B.
//! Both servers evaluate their DPF key over the full vocabulary, compute the
//! inner product with the pre-quantized embedding table, and get Q16.16 shares
//! of the embedding that can be upconverted to Q32.32.
//!
//! DPF depth = 16 covers vocabulary up to 65536 (padded). Key size ~276 bytes.

use crate::fixed_point::to_fixed;
use crate::transport::Transport;
use klearu_dpf::{dpf_full_eval, dpf_gen, AesPrg, DpfKey};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::io;

/// Pre-quantize an f32 embedding table to Q16.16 (u32).
///
/// `embedding_f32`: flat `[vocab_size × hidden_size]` row-major.
/// Returns flat `[vocab_size × hidden_size]` in Q16.16.
pub fn quantize_embedding_table(
    embedding_f32: &[f32],
    vocab_size: usize,
    hidden_size: usize,
) -> Vec<u32> {
    assert_eq!(embedding_f32.len(), vocab_size * hidden_size);
    embedding_f32.iter().map(|&v| to_fixed(v)).collect()
}

/// Compute the DPF depth needed to cover a vocabulary of given size.
/// Returns the smallest d such that 2^d >= vocab_size.
pub fn dpf_depth_for_vocab(vocab_size: usize) -> u8 {
    let mut d = 1u8;
    while (1usize << d) < vocab_size {
        d += 1;
    }
    d
}

/// Server A: generate DPF keys for PIR, send key_1 to Server B.
///
/// Returns key_0 for Server A to use locally.
pub fn pir_keygen_and_send(
    prg: &AesPrg,
    token_id: u32,
    vocab_size: usize,
    transport: &mut impl Transport,
) -> io::Result<DpfKey> {
    let depth = dpf_depth_for_vocab(vocab_size);
    let (key_0, key_1) = dpf_gen(prg, token_id, 1, depth);

    // Serialize and send key_1
    send_dpf_key(&key_1, transport)?;

    Ok(key_0)
}

/// Server B: receive DPF key from Server A.
pub fn pir_recv_key(
    vocab_size: usize,
    transport: &mut impl Transport,
) -> io::Result<DpfKey> {
    let depth = dpf_depth_for_vocab(vocab_size);
    recv_dpf_key(depth, transport)
}

/// Evaluate DPF key, inner product with Q16.16 embedding → Q16.16 share.
///
/// The DPF full evaluation produces a selection vector over the padded domain.
/// The inner product with each column of the embedding gives the share of
/// that embedding dimension.
///
/// With the `parallel` feature, uses rayon to parallelize over the vocab
/// dimension: each thread accumulates a chunk of vocab entries into a local
/// buffer, then results are merged.
pub fn pir_compute_embedding_share(
    prg: &AesPrg,
    key: &DpfKey,
    embedding_table_q16: &[u32],
    vocab_size: usize,
    hidden_size: usize,
) -> Vec<u32> {
    assert_eq!(embedding_table_q16.len(), vocab_size * hidden_size);

    let selection = dpf_full_eval(prg, key);

    // Use u32 wrapping arithmetic: the lower 32 bits of the i64 product sum
    // are identical to the u32 wrapping product sum, and the final result is
    // Q16.16 (u32). This halves accumulator memory vs i64 and is more cache-friendly.
    #[cfg(feature = "parallel")]
    let acc = {
        (0..vocab_size)
            .into_par_iter()
            .fold(
                || vec![0u32; hidden_size],
                |mut local_acc, v| {
                    let sel = selection[v];
                    let emb_base = v * hidden_size;
                    for d in 0..hidden_size {
                        local_acc[d] = local_acc[d].wrapping_add(sel.wrapping_mul(embedding_table_q16[emb_base + d]));
                    }
                    local_acc
                },
            )
            .reduce(
                || vec![0u32; hidden_size],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai = ai.wrapping_add(*bi);
                    }
                    a
                },
            )
    };

    #[cfg(not(feature = "parallel"))]
    let acc = {
        let mut acc = vec![0u32; hidden_size];
        for v in 0..vocab_size {
            let sel = selection[v];
            let emb_base = v * hidden_size;
            for d in 0..hidden_size {
                acc[d] = acc[d].wrapping_add(sel.wrapping_mul(embedding_table_q16[emb_base + d]));
            }
        }
        acc
    };

    acc
}

/// Batch-evaluate DPF-PIR for multiple tokens in a single pass over the embedding table.
///
/// This is much faster than calling `pir_compute_embedding_share` N times because:
/// 1. The embedding table (~620MB for vocab=151665, hidden=1024) is read once instead of N times
/// 2. Sequential access pattern instead of strided (cache-friendly)
/// 3. DPF evaluations and vocab accumulation run in parallel (with `parallel` feature)
///
/// Returns one `Vec<u32>` (Q16.16 embedding share) per key.
pub fn pir_compute_embedding_shares_batch(
    prg: &AesPrg,
    keys: &[DpfKey],
    embedding_table_q16: &[u32],
    vocab_size: usize,
    hidden_size: usize,
) -> Vec<Vec<u32>> {
    let n = keys.len();
    if n == 0 {
        return Vec::new();
    }

    // Evaluate all DPF keys (parallel when available)
    #[cfg(feature = "parallel")]
    let selections: Vec<Vec<u32>> = keys.par_iter().map(|k| dpf_full_eval(prg, k)).collect();
    #[cfg(not(feature = "parallel"))]
    let selections: Vec<Vec<u32>> = keys.iter().map(|k| dpf_full_eval(prg, k)).collect();

    let acc_size = n * hidden_size;

    // Use u32 wrapping arithmetic (lower 32 bits match i64 product sum).
    // Loop order: outer=vocab, middle=tokens, inner=dimensions for sequential
    // accumulator access (stride-1 over acc[t*hs + d] with d varying).
    #[cfg(feature = "parallel")]
    let accumulators = {
        (0..vocab_size)
            .into_par_iter()
            .fold(
                || vec![0u32; acc_size],
                |mut local_acc, v| {
                    let emb_base = v * hidden_size;
                    for t in 0..n {
                        let sel = selections[t][v];
                        let base = t * hidden_size;
                        for d in 0..hidden_size {
                            local_acc[base + d] = local_acc[base + d]
                                .wrapping_add(sel.wrapping_mul(embedding_table_q16[emb_base + d]));
                        }
                    }
                    local_acc
                },
            )
            .reduce(
                || vec![0u32; acc_size],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai = ai.wrapping_add(*bi);
                    }
                    a
                },
            )
    };

    #[cfg(not(feature = "parallel"))]
    let accumulators = {
        let mut accumulators = vec![0u32; acc_size];
        for v in 0..vocab_size {
            let emb_base = v * hidden_size;
            for t in 0..n {
                let sel = selections[t][v];
                let base = t * hidden_size;
                for d in 0..hidden_size {
                    accumulators[base + d] = accumulators[base + d]
                        .wrapping_add(sel.wrapping_mul(embedding_table_q16[emb_base + d]));
                }
            }
        }
        accumulators
    };

    // Split flat u32 accumulator into per-token results (already Q16.16)
    (0..n)
        .map(|t| {
            let base = t * hidden_size;
            accumulators[base..base + hidden_size].to_vec()
        })
        .collect()
}

/// Convert Q16.16 shares to Q32.32 (sign-extend, shift left 16).
pub fn q16_to_q32_share(q16: &[u32]) -> Vec<u64> {
    q16.iter()
        .map(|&v| ((v as i32 as i64) << 16) as u64)
        .collect()
}

/// Serialize a DPF key to bytes.
///
/// Format: `party(1) + seed(16) + output_correction(4) + depth × (cw_seed(16) + cw_bits(1))`
pub fn serialize_dpf_key(key: &DpfKey) -> Vec<u8> {
    let depth = key.correction_words.len();
    let len = 1 + 16 + 4 + depth * 17;
    let mut buf = Vec::with_capacity(len);
    buf.push(key.party);
    buf.extend_from_slice(&key.seed);
    buf.extend_from_slice(&key.output_correction.to_le_bytes());
    for cw in &key.correction_words {
        buf.extend_from_slice(&cw.seed);
        let bits = (cw.control_left as u8) | ((cw.control_right as u8) << 1);
        buf.push(bits);
    }
    buf
}

/// Deserialize a DPF key from bytes.
///
/// `depth` must match the number of correction words expected.
pub fn deserialize_dpf_key(bytes: &[u8], depth: u8) -> io::Result<DpfKey> {
    let expected_len = 1 + 16 + 4 + (depth as usize) * 17;
    if bytes.len() != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("DPF key length {}, expected {}", bytes.len(), expected_len),
        ));
    }
    let party = bytes[0];
    let mut seed = [0u8; 16];
    seed.copy_from_slice(&bytes[1..17]);
    let output_correction = u32::from_le_bytes([bytes[17], bytes[18], bytes[19], bytes[20]]);

    let mut correction_words = Vec::with_capacity(depth as usize);
    for i in 0..depth as usize {
        let offset = 21 + i * 17;
        let mut cw_seed = [0u8; 16];
        cw_seed.copy_from_slice(&bytes[offset..offset + 16]);
        let bits = bytes[offset + 16];
        correction_words.push(klearu_dpf::CorrectionWord {
            seed: cw_seed,
            control_left: (bits & 1) != 0,
            control_right: (bits & 2) != 0,
        });
    }

    Ok(DpfKey {
        party,
        seed,
        correction_words,
        output_correction,
    })
}

// --- DPF key serialization (transport-based, for MPC between servers) ---

fn send_dpf_key(key: &DpfKey, transport: &mut impl Transport) -> io::Result<()> {
    // party (1 byte) + seed (16 bytes) + output_correction (4 bytes) + correction_words
    let depth = key.correction_words.len();
    transport.send(&[key.party])?;
    transport.send(&key.seed)?;
    transport.send(&key.output_correction.to_le_bytes())?;

    // Each correction word: 16 bytes seed + 1 byte (2 control bits packed)
    for cw in &key.correction_words {
        transport.send(&cw.seed)?;
        let bits = (cw.control_left as u8) | ((cw.control_right as u8) << 1);
        transport.send(&[bits])?;
    }

    let _ = depth;
    Ok(())
}

fn recv_dpf_key(depth: u8, transport: &mut impl Transport) -> io::Result<DpfKey> {
    let party_bytes = transport.recv(1)?;
    let party = party_bytes[0];
    let seed_bytes = transport.recv(16)?;
    let mut seed = [0u8; 16];
    seed.copy_from_slice(&seed_bytes);
    let oc_bytes = transport.recv(4)?;
    let output_correction = u32::from_le_bytes([oc_bytes[0], oc_bytes[1], oc_bytes[2], oc_bytes[3]]);

    let mut correction_words = Vec::with_capacity(depth as usize);
    for _ in 0..depth {
        let cw_seed_bytes = transport.recv(16)?;
        let mut cw_seed = [0u8; 16];
        cw_seed.copy_from_slice(&cw_seed_bytes);
        let bits_bytes = transport.recv(1)?;
        let bits = bits_bytes[0];
        correction_words.push(klearu_dpf::CorrectionWord {
            seed: cw_seed,
            control_left: (bits & 1) != 0,
            control_right: (bits & 2) != 0,
        });
    }

    Ok(DpfKey {
        party,
        seed,
        correction_words,
        output_correction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::{from_fixed, from_fixed64};
    use crate::transport::memory_transport_pair;

    #[test]
    fn test_quantize_embedding_table() {
        let emb = vec![1.0f32, -0.5, 0.25, 2.0];
        let q = quantize_embedding_table(&emb, 2, 2);
        assert_eq!(q.len(), 4);
        assert!((from_fixed(q[0]) - 1.0).abs() < 1e-4);
        assert!((from_fixed(q[1]) - (-0.5)).abs() < 1e-4);
    }

    #[test]
    fn test_dpf_depth_for_vocab() {
        assert_eq!(dpf_depth_for_vocab(1), 1);
        assert_eq!(dpf_depth_for_vocab(2), 1);
        assert_eq!(dpf_depth_for_vocab(3), 2);
        assert_eq!(dpf_depth_for_vocab(256), 8);
        assert_eq!(dpf_depth_for_vocab(49152), 16);
        assert_eq!(dpf_depth_for_vocab(65536), 16);
    }

    #[test]
    fn test_pir_embedding_correctness() {
        let vocab_size = 8;
        let hidden_size = 4;
        let prg = AesPrg::new(&[42u8; 16]);

        // Create a simple embedding table
        let emb_f32: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6)
            .collect();

        let emb_q16 = quantize_embedding_table(&emb_f32, vocab_size, hidden_size);

        // Test multiple token IDs
        for token_id in 0..vocab_size as u32 {
            let (mut trans_a, mut trans_b) = memory_transport_pair();

            let prg_clone = AesPrg::new(&[42u8; 16]);
            let emb_q16_clone = emb_q16.clone();

            let handle = std::thread::spawn(move || {
                let key_1 = pir_recv_key(vocab_size, &mut trans_b).unwrap();
                pir_compute_embedding_share(&prg_clone, &key_1, &emb_q16_clone, vocab_size, hidden_size)
            });

            let key_0 = pir_keygen_and_send(&prg, token_id, vocab_size, &mut trans_a).unwrap();
            let emb_q16_local = quantize_embedding_table(&emb_f32, vocab_size, hidden_size);
            let share_0 = pir_compute_embedding_share(&prg, &key_0, &emb_q16_local, vocab_size, hidden_size);
            let share_1 = handle.join().unwrap();

            // Reconstruct and verify
            for d in 0..hidden_size {
                let reconstructed = from_fixed(share_0[d].wrapping_add(share_1[d]));
                let expected = emb_f32[token_id as usize * hidden_size + d];
                assert!(
                    (reconstructed - expected).abs() < 0.01,
                    "PIR token={token_id} dim={d}: got {reconstructed}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_q16_to_q32_share() {
        let q16 = vec![to_fixed(1.0), to_fixed(-0.5)];
        let q32 = q16_to_q32_share(&q16);
        assert!((from_fixed64(q32[0]) - 1.0).abs() < 1e-4);
        assert!((from_fixed64(q32[1]) - (-0.5)).abs() < 1e-4);
    }

    #[test]
    fn test_serialize_deserialize_dpf_key_roundtrip() {
        let prg = AesPrg::new(&[77u8; 16]);
        let vocab_size = 16;
        let depth = dpf_depth_for_vocab(vocab_size);
        let (key_0, key_1) = klearu_dpf::dpf_gen(&prg, 5, 1, depth);

        // Roundtrip key_0
        let bytes_0 = serialize_dpf_key(&key_0);
        assert_eq!(bytes_0.len(), 1 + 16 + 4 + (depth as usize) * 17);
        let restored_0 = deserialize_dpf_key(&bytes_0, depth).unwrap();
        assert_eq!(restored_0.party, key_0.party);
        assert_eq!(restored_0.seed, key_0.seed);
        assert_eq!(restored_0.output_correction, key_0.output_correction);
        assert_eq!(restored_0.correction_words.len(), key_0.correction_words.len());

        // Roundtrip key_1
        let bytes_1 = serialize_dpf_key(&key_1);
        let restored_1 = deserialize_dpf_key(&bytes_1, depth).unwrap();

        // Evaluate and verify DPF correctness with restored keys
        let hidden_size = 4;
        let emb_f32: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| ((i * 11 + 5) % 17) as f32 * 0.05 - 0.4)
            .collect();
        let emb_q16 = quantize_embedding_table(&emb_f32, vocab_size, hidden_size);

        let share_0 = pir_compute_embedding_share(&prg, &restored_0, &emb_q16, vocab_size, hidden_size);
        let share_1 = pir_compute_embedding_share(&prg, &restored_1, &emb_q16, vocab_size, hidden_size);

        for d in 0..hidden_size {
            let reconstructed = from_fixed(share_0[d].wrapping_add(share_1[d]));
            let expected = emb_f32[5 * hidden_size + d];
            assert!(
                (reconstructed - expected).abs() < 0.01,
                "Roundtrip PIR dim={d}: got {reconstructed}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_pir_end_to_end_q32() {
        let vocab_size = 16;
        let hidden_size = 4;
        let prg = AesPrg::new(&[99u8; 16]);
        let token_id = 7u32;

        let emb_f32: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| ((i * 11 + 5) % 17) as f32 * 0.05 - 0.4)
            .collect();

        let emb_q16 = quantize_embedding_table(&emb_f32, vocab_size, hidden_size);

        let (mut trans_a, mut trans_b) = memory_transport_pair();
        let prg_b = AesPrg::new(&[99u8; 16]);
        let emb_q16_b = emb_q16.clone();

        let handle = std::thread::spawn(move || {
            let key_1 = pir_recv_key(vocab_size, &mut trans_b).unwrap();
            let share_q16 = pir_compute_embedding_share(&prg_b, &key_1, &emb_q16_b, vocab_size, hidden_size);
            q16_to_q32_share(&share_q16)
        });

        let key_0 = pir_keygen_and_send(&prg, token_id, vocab_size, &mut trans_a).unwrap();
        let share_0_q16 = pir_compute_embedding_share(&prg, &key_0, &emb_q16, vocab_size, hidden_size);
        let share_0_q32 = q16_to_q32_share(&share_0_q16);
        let share_1_q32 = handle.join().unwrap();

        for d in 0..hidden_size {
            let result = from_fixed64(share_0_q32[d].wrapping_add(share_1_q32[d]));
            let expected = emb_f32[token_id as usize * hidden_size + d];
            assert!(
                (result - expected).abs() < 0.01,
                "PIR Q32 dim={d}: got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_batch_pir_matches_individual() {
        let vocab_size = 16;
        let hidden_size = 8;
        let prg = AesPrg::new(&[55u8; 16]);
        let depth = dpf_depth_for_vocab(vocab_size);

        let emb_f32: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6)
            .collect();
        let emb_q16 = quantize_embedding_table(&emb_f32, vocab_size, hidden_size);

        let token_ids = [2u32, 7, 0, 11];

        // Generate keys for all tokens
        let keys: Vec<DpfKey> = token_ids
            .iter()
            .map(|&tid| {
                let (key, _) = dpf_gen(&prg, tid, 1, depth);
                key
            })
            .collect();

        // Individual PIR
        let individual: Vec<Vec<u32>> = keys
            .iter()
            .map(|k| pir_compute_embedding_share(&prg, k, &emb_q16, vocab_size, hidden_size))
            .collect();

        // Batch PIR
        let batched = pir_compute_embedding_shares_batch(
            &prg, &keys, &emb_q16, vocab_size, hidden_size,
        );

        assert_eq!(individual.len(), batched.len());
        for t in 0..token_ids.len() {
            assert_eq!(individual[t].len(), batched[t].len());
            for d in 0..hidden_size {
                assert_eq!(
                    individual[t][d], batched[t][d],
                    "Mismatch at token={}, dim={}: individual={}, batch={}",
                    t, d, individual[t][d], batched[t][d]
                );
            }
        }
    }
}
