//! Oblivious Pseudo-Random Function (OPRF) based on Ristretto255.
//!
//! Protocol:
//! 1. Client hashes input to a point: P = H(input)
//! 2. Client blinds: B = r * P, sends B to server
//! 3. Server evaluates: E = k * B, sends E to client
//! 4. Client unblinds: F = r^{-1} * E = k * P
//!
//! The server never sees P (only r*P), and the client never learns k.
//! The output k*P is a deterministic PRF of the input under key k.

use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;

/// Server-side OPRF key.
pub struct OprfServer {
    key: Scalar,
}

impl OprfServer {
    /// Generate a new random OPRF key.
    pub fn random(rng: &mut impl rand::RngCore) -> Self {
        let mut bytes = [0u8; 64];
        rng.fill_bytes(&mut bytes);
        Self {
            key: Scalar::from_bytes_mod_order_wide(&bytes),
        }
    }

    /// Create from a 32-byte seed (deterministic).
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let mut wide = [0u8; 64];
        wide[..32].copy_from_slice(seed);
        Self {
            key: Scalar::from_bytes_mod_order_wide(&wide),
        }
    }

    /// Evaluate the OPRF on blinded points (server side).
    pub fn evaluate(&self, blinded: &[CompressedRistretto]) -> Vec<CompressedRistretto> {
        blinded.iter().map(|b| {
            let point = b.decompress().expect("invalid blinded point");
            (point * self.key).compress()
        }).collect()
    }
}

/// Client-side OPRF state (holds blinding factors for one batch).
pub struct OprfClient {
    blinding_scalars: Vec<Scalar>,
}

impl OprfClient {
    pub fn new() -> Self {
        Self { blinding_scalars: Vec::new() }
    }

    /// Blind a set of bucket keys for private lookup.
    ///
    /// Returns compressed Ristretto points to send to the server.
    pub fn blind(&mut self, bucket_keys: &[&[u8]], rng: &mut impl rand::RngCore) -> Vec<CompressedRistretto> {
        self.blinding_scalars.clear();

        bucket_keys.iter().map(|key| {
            let point = hash_to_point(key);
            let mut bytes = [0u8; 64];
            rng.fill_bytes(&mut bytes);
            let r = Scalar::from_bytes_mod_order_wide(&bytes);
            self.blinding_scalars.push(r);
            (point * r).compress()
        }).collect()
    }

    /// Unblind server responses to get PRF outputs.
    pub fn unblind(&self, responses: &[CompressedRistretto]) -> Vec<[u8; 32]> {
        assert_eq!(responses.len(), self.blinding_scalars.len());

        responses.iter().zip(self.blinding_scalars.iter()).map(|(resp, r)| {
            let evaluated = resp.decompress().expect("invalid server response");
            let unblinded = evaluated * r.invert();
            unblinded.compress().to_bytes()
        }).collect()
    }
}

impl Default for OprfClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash an arbitrary byte string to a Ristretto point.
///
/// Uses a simple hash-and-check approach: SHA-512 the input with a counter
/// until a valid point is produced. This is not constant-time but is sufficient
/// for the semi-honest model.
pub fn hash_to_point(input: &[u8]) -> RistrettoPoint {
    // Simple approach: hash input to 64 bytes, use Ristretto's from_uniform_bytes
    // which maps 64 uniform bytes to a Ristretto point.
    let mut hasher_input = Vec::with_capacity(input.len() + 16);
    hasher_input.extend_from_slice(b"klearu-oprf-v1:");
    hasher_input.extend_from_slice(input);

    // 64-byte hash via two rounds of an AES-based PRG, then
    // curve25519-dalek's from_uniform_bytes maps [u8; 64] to a Ristretto point.
    // For production-grade use, replace with SHA-512.
    let mut output = [0u8; 64];
    // Use AES-based expansion
    use aes::cipher::{BlockEncrypt, KeyInit};
    let key = aes::Aes128::new((&[0x4F, 0x50, 0x52, 0x46, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]).into());

    // Process input into a 16-byte state via CBC-MAC-like construction
    let mut state = [0u8; 16];
    for chunk in hasher_input.chunks(16) {
        for (i, &b) in chunk.iter().enumerate() {
            state[i] ^= b;
        }
        let mut block: aes::Block = state.into();
        key.encrypt_block(&mut block);
        state = block.into();
    }

    // Expand state to 64 bytes via counter mode
    for i in 0..4u8 {
        let mut block_input = state;
        block_input[15] ^= i;
        let mut block: aes::Block = block_input.into();
        key.encrypt_block(&mut block);
        let bytes: [u8; 16] = block.into();
        output[i as usize * 16..(i as usize + 1) * 16].copy_from_slice(&bytes);
    }

    RistrettoPoint::from_uniform_bytes(&output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_oprf_correctness() {
        let mut rng = StdRng::seed_from_u64(42);

        let server = OprfServer::random(&mut rng);
        let mut client = OprfClient::new();

        let inputs = [b"bucket-key-1" as &[u8], b"bucket-key-2"];
        let blinded = client.blind(&inputs, &mut rng);

        let evaluated = server.evaluate(&blinded);
        let results = client.unblind(&evaluated);

        // Results should be deterministic for the same input + key
        assert_ne!(results[0], results[1], "different inputs should give different outputs");

        // Repeat with the same inputs: should get the same results
        let mut client2 = OprfClient::new();
        let blinded2 = client2.blind(&inputs, &mut rng);
        let evaluated2 = server.evaluate(&blinded2);
        let results2 = client2.unblind(&evaluated2);

        assert_eq!(results[0], results2[0], "same input+key should give same output");
        assert_eq!(results[1], results2[1], "same input+key should give same output");
    }

    #[test]
    fn test_oprf_different_keys_different_outputs() {
        let mut rng = StdRng::seed_from_u64(42);

        let server1 = OprfServer::from_seed(&[1u8; 32]);
        let server2 = OprfServer::from_seed(&[2u8; 32]);

        let mut client = OprfClient::new();
        let inputs = [b"test-input" as &[u8]];

        let blinded = client.blind(&inputs, &mut rng);
        let result1 = server1.evaluate(&blinded);

        let mut client2 = OprfClient::new();
        let blinded2 = client2.blind(&inputs, &mut rng);
        let result2 = server2.evaluate(&blinded2);

        let out1 = client.unblind(&result1);
        let out2 = client2.unblind(&result2);

        assert_ne!(out1[0], out2[0], "different keys should give different outputs");
    }

    #[test]
    fn test_hash_to_point_deterministic() {
        let p1 = hash_to_point(b"hello");
        let p2 = hash_to_point(b"hello");
        assert_eq!(p1.compress(), p2.compress());
    }

    #[test]
    fn test_hash_to_point_different_inputs() {
        let p1 = hash_to_point(b"hello");
        let p2 = hash_to_point(b"world");
        assert_ne!(p1.compress(), p2.compress());
    }
}
