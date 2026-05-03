//! Beaver triple generation via Gilboa multiplication using Ferret ROT.
//!
//! Ferret's `send_rot`/`recv_rot` produce random correlated pairs — the sender
//! gets random `(m0[k], m1[k])` and the receiver (with choice bit b_k) gets
//! `m_{b_k}[k]`. These are NOT chosen-message OT; the values are random.
//!
//! Gilboa multiplication is implemented via ROT + arithmetic corrections:
//!
//! For each triple (a, b, c) where c = a*b mod 2^64:
//! 1. Run ROT: sender gets (m0[k], m1[k]), receiver gets m_{b_k}[k]
//! 2. Sender sets r_k = low64(m0[k]) and computes correction:
//!    c_k = (r_k + a * 2^k) - low64(m1[k])
//! 3. Sender sends corrections to receiver via a side channel
//! 4. Receiver applies: if b_k=1, value_k = low64(received[k]) + c_k
//!    else value_k = low64(received[k])
//! 5. Sender's c_share = -sum(r_k), Receiver's c_share = sum(value_k) = sum(r_k + b_k * a * 2^k) = sum(r_k) + a*b
//!
//! Correctness: c_server + c_client = a*b mod 2^64
//!
//! Cost: 64 ROTs + 64 correction u64s per u64 triple; 128 ROTs + 128 correction u128s per u128 triple.

use ferret::{BlockArray, FerretCOT, NetIO, ALICE, BOB};
use klearu_mpc::beaver::{BeaverTriple, BeaverTriple128, TripleGenerator, TripleGenerator128};
use klearu_mpc::transport::Transport;
use std::io;

use crate::tcp_transport::TcpTransport;

/// Generates [`BeaverTriple`]s (u64) using Ferret ROT + Gilboa corrections.
///
/// Holds a `FerretCOT` for ROT and a separate `TcpTransport` for correction exchange.
/// Server (party=1) is ALICE (OT sender), client (party=0) is BOB (OT receiver).
pub struct FerretTripleGen {
    party: u8,
    cot: FerretCOT,
    corrections: TcpTransport,
}

impl FerretTripleGen {
    pub fn new(party: u8, cot: FerretCOT, corrections: TcpTransport) -> Self {
        Self { party, cot, corrections }
    }
}

impl TripleGenerator for FerretTripleGen {
    fn generate(&mut self, n: usize) -> Vec<BeaverTriple> {
        self.generate_io(n).expect("Ferret triple generation failed")
    }
}

impl FerretTripleGen {
    fn generate_io(&mut self, n: usize) -> io::Result<Vec<BeaverTriple>> {
        let bits = 64u64;
        let total = n as u64 * bits;
        let mut triples = Vec::with_capacity(n);

        if self.party == 1 {
            // Server = ALICE = OT sender
            let b0 = BlockArray::new(total);
            let b1 = BlockArray::new(total);
            self.cot.send_rot(&b0, &b1, total);

            let mut a_values = Vec::with_capacity(n);
            let mut c_shares = Vec::with_capacity(n);
            let mut all_corrections = Vec::with_capacity(total as usize);

            for j in 0..n {
                let a: u64 = rand::random();
                a_values.push(a);
                let mut neg_sum_r = 0u64;

                for k in 0..64u64 {
                    let idx = j as u64 * bits + k;
                    let m0 = block_to_u64(b0.get_block_data(idx));
                    let m1 = block_to_u64(b1.get_block_data(idx));

                    let r_k = m0;
                    neg_sum_r = neg_sum_r.wrapping_sub(r_k);

                    // Correction: what receiver needs to add when choice=1
                    // to turn m1 into r_k + a*2^k
                    let correction = r_k
                        .wrapping_add(a.wrapping_mul(1u64 << k))
                        .wrapping_sub(m1);
                    all_corrections.push(correction);
                }
                c_shares.push(neg_sum_r);
            }

            // Send corrections to receiver
            self.corrections.send_u64_slice(&all_corrections)?;

            for j in 0..n {
                triples.push(BeaverTriple {
                    a: a_values[j],
                    b: 0,
                    c: c_shares[j],
                });
            }
        } else {
            // Client = BOB = OT receiver
            let br = BlockArray::new(total);
            let mut b_values = Vec::with_capacity(n);
            let mut choices = Vec::with_capacity(total as usize);

            for _j in 0..n {
                let b: u64 = rand::random();
                b_values.push(b);
                for k in 0..64 {
                    choices.push(((b >> k) & 1) != 0);
                }
            }

            self.cot.recv_rot(&br, &choices, total);

            // Receive corrections from sender
            let all_corrections = self.corrections.recv_u64_slice(total as usize)?;

            for j in 0..n {
                let b = b_values[j];
                let mut c_share = 0u64;
                for k in 0..64u64 {
                    let idx = (j as u64 * bits + k) as usize;
                    let received = block_to_u64(br.get_block_data(idx as u64));
                    let b_k = ((b >> k) & 1) != 0;
                    let value = if b_k {
                        received.wrapping_add(all_corrections[idx])
                    } else {
                        received
                    };
                    c_share = c_share.wrapping_add(value);
                }

                triples.push(BeaverTriple {
                    a: 0,
                    b,
                    c: c_share,
                });
            }
        }

        Ok(triples)
    }
}

/// Generates [`BeaverTriple128`]s (u128) using Ferret ROT + Gilboa corrections.
pub struct FerretTripleGen128 {
    party: u8,
    cot: FerretCOT,
    corrections: TcpTransport,
}

impl FerretTripleGen128 {
    pub fn new(party: u8, cot: FerretCOT, corrections: TcpTransport) -> Self {
        Self { party, cot, corrections }
    }
}

impl TripleGenerator128 for FerretTripleGen128 {
    fn generate(&mut self, n: usize) -> Vec<BeaverTriple128> {
        self.generate_io(n).expect("Ferret triple generation failed")
    }
}

impl FerretTripleGen128 {
    fn generate_io(&mut self, n: usize) -> io::Result<Vec<BeaverTriple128>> {
        let bits = 128u64;
        let total = n as u64 * bits;
        let mut triples = Vec::with_capacity(n);

        if self.party == 1 {
            let b0 = BlockArray::new(total);
            let b1 = BlockArray::new(total);
            self.cot.send_rot(&b0, &b1, total);

            let mut a_values = Vec::with_capacity(n);
            let mut c_shares = Vec::with_capacity(n);
            let mut all_corrections = Vec::with_capacity(total as usize);

            for j in 0..n {
                let a: u128 = rand_u128();
                a_values.push(a);
                let mut neg_sum_r = 0u128;

                for k in 0..128u64 {
                    let idx = j as u64 * bits + k;
                    let m0 = block_to_u128(b0.get_block_data(idx));
                    let m1 = block_to_u128(b1.get_block_data(idx));

                    let r_k = m0;
                    neg_sum_r = neg_sum_r.wrapping_sub(r_k);

                    let correction = r_k
                        .wrapping_add(a.wrapping_mul(1u128 << k))
                        .wrapping_sub(m1);
                    all_corrections.push(correction);
                }
                c_shares.push(neg_sum_r);
            }

            self.corrections.send_u128_slice(&all_corrections)?;

            for j in 0..n {
                triples.push(BeaverTriple128 {
                    a: a_values[j],
                    b: 0,
                    c: c_shares[j],
                });
            }
        } else {
            let br = BlockArray::new(total);
            let mut b_values = Vec::with_capacity(n);
            let mut choices = Vec::with_capacity(total as usize);

            for _j in 0..n {
                let b: u128 = rand_u128();
                b_values.push(b);
                for k in 0..128 {
                    choices.push(((b >> k) & 1) != 0);
                }
            }

            self.cot.recv_rot(&br, &choices, total);

            let all_corrections = self.corrections.recv_u128_slice(total as usize)?;

            for j in 0..n {
                let b = b_values[j];
                let mut c_share = 0u128;
                for k in 0..128u64 {
                    let idx = (j as u64 * bits + k) as usize;
                    let received = block_to_u128(br.get_block_data(idx as u64));
                    let b_k = ((b >> k) & 1) != 0;
                    let value = if b_k {
                        received.wrapping_add(all_corrections[idx])
                    } else {
                        received
                    };
                    c_share = c_share.wrapping_add(value);
                }

                triples.push(BeaverTriple128 {
                    a: 0,
                    b,
                    c: c_share,
                });
            }
        }

        Ok(triples)
    }
}

// --- Ferret connection setup helpers ---

/// Server-side: pick random ports for Ferret OT and correction channel,
/// send them to the client, then create both connections.
///
/// Returns `(FerretCOT, TcpTransport)` for the OT and correction channels.
pub fn setup_ferret_server(
    transport: &mut impl Transport,
) -> io::Result<(FerretCOT, TcpTransport)> {
    use rand::Rng;
    use std::sync::mpsc;

    let ferret_port: u32 = rand::thread_rng().gen_range(20000..50000);
    let correction_port: u32 = ferret_port + 1;

    // Spawn thread for NetIO (blocks on accept)
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let netio = NetIO::new(ALICE, None, ferret_port as i32);
        let cot = FerretCOT::new(ALICE, 1, &netio, false);
        std::mem::forget(netio);
        tx.send(cot).ok();
    });

    // Also listen for the correction channel
    let correction_listener = std::net::TcpListener::bind(format!("0.0.0.0:{correction_port}"))?;

    // Wait a moment for both listeners to be ready
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Tell client both ports
    transport.send_u32(ferret_port)?;
    transport.send_u32(correction_port)?;

    // Accept correction connection
    let (correction_stream, _) = correction_listener.accept()?;
    let correction_transport = TcpTransport::new(correction_stream)?;

    // Wait for Ferret setup
    let cot = rx.recv().map_err(|e| {
        io::Error::new(io::ErrorKind::Other, format!("Ferret setup failed: {e}"))
    })?;

    Ok((cot, correction_transport))
}

/// Client-side: receive ports, connect to both Ferret OT and correction channels.
pub fn setup_ferret_client(
    server_ip: &str,
    transport: &mut impl Transport,
) -> io::Result<(FerretCOT, TcpTransport)> {
    let ferret_port = transport.recv_u32()?;
    let correction_port = transport.recv_u32()?;

    // Connect correction channel first (non-blocking relative to Ferret)
    let correction_stream =
        std::net::TcpStream::connect(format!("{server_ip}:{correction_port}"))?;
    let correction_transport = TcpTransport::new(correction_stream)?;

    // Connect Ferret OT channel
    let netio = NetIO::new(BOB, Some(server_ip.to_string()), ferret_port as i32);
    let cot = FerretCOT::new(BOB, 1, &netio, false);

    std::mem::forget(netio);

    Ok((cot, correction_transport))
}

// --- Helper functions ---

fn block_to_u64(block: Vec<u8>) -> u64 {
    assert!(block.len() >= 8, "block too short: {} bytes", block.len());
    u64::from_le_bytes(block[..8].try_into().unwrap())
}

fn block_to_u128(block: Vec<u8>) -> u128 {
    assert!(block.len() >= 16, "block too short: {} bytes", block.len());
    u128::from_le_bytes(block[..16].try_into().unwrap())
}

fn rand_u128() -> u128 {
    let lo: u64 = rand::random();
    let hi: u64 = rand::random();
    (lo as u128) | ((hi as u128) << 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_roundtrip_u64() {
        let v = 0xDEADBEEF_CAFEBABE_u64;
        // Simulate what ROT would produce: 16-byte block with random upper bytes
        let mut block = vec![0u8; 16];
        block[..8].copy_from_slice(&v.to_le_bytes());
        block[8..].copy_from_slice(&[0xFF; 8]);
        assert_eq!(block_to_u64(block), v);
    }

    #[test]
    fn test_block_roundtrip_u128() {
        let v = 0xDEADBEEF_CAFEBABE_01234567_89ABCDEFu128;
        let block = v.to_le_bytes().to_vec();
        assert_eq!(block_to_u128(block), v);
    }

    // Gilboa correctness proof:
    //   c_server + c_client
    //   = -sum(r_k) + sum(value_k)
    //   where value_k = r_k (if b_k=0) or m1_k + correction_k (if b_k=1)
    //   and correction_k = r_k + a*2^k - m1_k
    //   so value_k = r_k + b_k * a * 2^k
    //   Thus: c_server + c_client = sum(b_k * a * 2^k) = a * b
}
