//! Densified One Permutation Hashing (DOPH).
//!
//! A single random permutation of the input dimensions is partitioned into K
//! bins.  Within each bin the hash value is the minimum permuted index among
//! non-zero entries.  Empty bins are "densified" by borrowing from the
//! previous non-empty bin (circular).  The K per-bin hashes are combined into
//! a single code using a polynomial hash: sum(h_i * prime^i) mod 2^range_pow.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::tensor::SparseVector;

use super::HashFamily;

/// Densified One Permutation MinHash family.
pub struct MinHash {
    /// For each table, the permutation of [0..input_dim].
    /// `permutations[table][original_dim] = permuted_index`.
    permutations: Vec<Vec<u32>>,
    input_dim: usize,
    k: usize,
    num_tables: usize,
    /// Size of each bin: `input_dim / k` (rounded up if needed, with the last
    /// bin potentially smaller).
    bin_size: usize,
}

impl MinHash {
    /// Create a new MinHash (DOPH) family.
    ///
    /// # Arguments
    /// * `input_dim` - Dimensionality of input vectors.
    /// * `k` - Number of bins (hash functions) per table.
    /// * `num_tables` - Number of hash tables (L).
    /// * `seed` - RNG seed for reproducibility.
    pub fn new(input_dim: usize, k: usize, num_tables: usize, seed: u64) -> Self {
        assert!(k > 0, "k must be > 0");
        assert!(input_dim > 0, "input_dim must be > 0");
        // Bin size: ceiling division so all dims are covered.
        let bin_size = (input_dim + k - 1) / k;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut permutations = Vec::with_capacity(num_tables);

        for _ in 0..num_tables {
            let mut perm: Vec<u32> = (0..input_dim as u32).collect();
            perm.shuffle(&mut rng);
            permutations.push(perm);
        }

        Self {
            permutations,
            input_dim,
            k,
            num_tables,
            bin_size,
        }
    }

    /// Compute the K per-bin minimum permuted indices for a given table.
    ///
    /// Returns a Vec of K values, where each value is the minimum permuted
    /// index among non-zero dimensions that fall into that bin, or `u32::MAX`
    /// for empty bins (before densification).
    fn compute_bin_mins(&self, nonzero_dims: &[u32], table: usize) -> Vec<u32> {
        let perm = &self.permutations[table];
        let mut bin_mins = vec![u32::MAX; self.k];

        for &dim in nonzero_dims {
            let permuted = perm[dim as usize];
            let bin = permuted as usize / self.bin_size;
            if bin < self.k {
                bin_mins[bin] = bin_mins[bin].min(permuted);
            }
        }

        bin_mins
    }

    /// Densify empty bins by borrowing from the nearest non-empty bin
    /// (searching forward, wrapping circularly).
    fn densify(bin_mins: &mut [u32]) {
        let k = bin_mins.len();
        if k == 0 {
            return;
        }

        // If all bins are empty, fill them all with 0 (degenerate case).
        if bin_mins.iter().all(|&v| v == u32::MAX) {
            for m in bin_mins.iter_mut() {
                *m = 0;
            }
            return;
        }

        // For each empty bin, search forward (circularly) for the first
        // non-empty bin and borrow its value.
        for i in 0..k {
            if bin_mins[i] == u32::MAX {
                let mut j = (i + 1) % k;
                while bin_mins[j] == u32::MAX {
                    j = (j + 1) % k;
                }
                bin_mins[i] = bin_mins[j];
            }
        }
    }

    /// Combine K per-bin hash values into a single u64 using polynomial hashing.
    ///
    /// hash = sum(h_i * PRIME^i) mod 2^k, where k = self.k (the range_pow).
    fn combine(&self, bin_mins: &[u32]) -> u64 {
        const PRIME: u64 = 0x517cc1b727220a95; // large prime
        let mask = if self.k >= 64 {
            u64::MAX
        } else {
            (1u64 << self.k) - 1
        };

        let mut hash = 0u64;
        let mut prime_pow = 1u64;
        for &h in bin_mins {
            hash = hash.wrapping_add((h as u64).wrapping_mul(prime_pow));
            prime_pow = prime_pow.wrapping_mul(PRIME);
        }
        hash & mask
    }

    /// Core hashing logic shared by dense and sparse paths.
    fn hash_impl(&self, nonzero_dims: &[u32], table: usize) -> u64 {
        let mut bin_mins = self.compute_bin_mins(nonzero_dims, table);
        Self::densify(&mut bin_mins);
        self.combine(&bin_mins)
    }
}

impl HashFamily for MinHash {
    fn hash_dense(&self, input: &[f32], table: usize) -> u64 {
        debug_assert_eq!(input.len(), self.input_dim);
        let nonzero_dims: Vec<u32> = input
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, _)| i as u32)
            .collect();
        self.hash_impl(&nonzero_dims, table)
    }

    fn hash_sparse(&self, input: &SparseVector, table: usize) -> u64 {
        self.hash_impl(&input.indices, table)
    }

    fn k(&self) -> usize {
        self.k
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn num_tables(&self) -> usize {
        self.num_tables
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::test_helpers::*;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    const DIM: usize = 128;
    const K: usize = 8;
    const L: usize = 4;
    const SEED: u64 = 42;

    fn random_dense(seed: u64, dim: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn deterministic_with_same_seed() {
        let v = random_dense(99, DIM);
        assert_deterministic(|| Box::new(MinHash::new(DIM, K, L, SEED)), &v);
    }

    #[test]
    fn hash_in_range() {
        let h = MinHash::new(DIM, K, L, SEED);
        let v = random_dense(100, DIM);
        assert_hash_in_range(&h, &v);
    }

    #[test]
    fn dense_sparse_agree() {
        let h = MinHash::new(DIM, K, L, SEED);
        let v = random_dense(101, DIM);
        assert_dense_sparse_agree(&h, &v);
    }

    #[test]
    fn different_inputs_different_hashes() {
        let h = MinHash::new(DIM, K, L, SEED);
        // MinHash only considers which dimensions are nonzero, so we need
        // vectors with genuinely different sparsity patterns.
        let mut v1 = vec![0.0f32; DIM];
        let mut v2 = vec![0.0f32; DIM];
        // v1: first half nonzero
        for i in 0..DIM / 2 {
            v1[i] = 1.0;
        }
        // v2: second half nonzero
        for i in DIM / 2..DIM {
            v2[i] = 1.0;
        }
        let any_differ = (0..L).any(|t| h.hash_dense(&v1, t) != h.hash_dense(&v2, t));
        assert!(any_differ, "all tables gave same hash for different inputs");
    }

    #[test]
    fn densification_handles_empty_bins() {
        // Very sparse input: should still hash without panicking.
        let h = MinHash::new(DIM, K, L, SEED);
        let sparse = SparseVector::from_pairs(DIM, vec![(10, 1.0)]);
        let max = 1u64 << K;
        for t in 0..L {
            let hv = h.hash_sparse(&sparse, t);
            assert!(hv < max, "table {t}: hash {hv} >= {max}");
        }
    }

    #[test]
    fn all_empty_vector_hashes() {
        let h = MinHash::new(DIM, K, L, SEED);
        let sparse = SparseVector::new(DIM);
        // All bins empty => densify fills all with 0 => deterministic.
        let h1: Vec<u64> = (0..L).map(|t| h.hash_sparse(&sparse, t)).collect();
        let h2: Vec<u64> = (0..L).map(|t| h.hash_sparse(&sparse, t)).collect();
        assert_eq!(h1, h2);
    }

    #[test]
    fn sparse_input_agrees_with_dense_conversion() {
        let h = MinHash::new(DIM, K, L, SEED);
        let mut v = vec![0.0f32; DIM];
        v[3] = 1.0;
        v[17] = -0.5;
        v[50] = 2.0;
        assert_dense_sparse_agree(&h, &v);
    }

    #[test]
    fn densify_circular_borrow() {
        // Test that densification wraps around.
        let mut bins = vec![u32::MAX, 5, u32::MAX, u32::MAX];
        MinHash::densify(&mut bins);
        // Original: [MAX, 5, MAX, MAX]
        // Densification iterates in-place; later searches see earlier fills.
        // i=0: search from 1: bins[1]=5, so bins[0]=5
        // i=2: search from 3: bins[3]=MAX, wrap to 0: bins[0]=5 => bins[2]=5
        // i=3: search from 0: bins[0]=5 => bins[3]=5
        assert_eq!(bins, vec![5, 5, 5, 5]);
    }
}
