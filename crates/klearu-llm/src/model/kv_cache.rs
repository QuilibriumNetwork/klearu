/// Per-layer KV cache with head-major layout.
///
/// Layout: `[num_kv_heads][max_seq_len][head_dim]` -- all positions for a single
/// KV head are contiguous, optimal for attention score dot products.
pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    current_len: usize,
}

impl KvCache {
    pub fn new(num_kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let total = num_kv_heads * max_seq_len * head_dim;
        Self {
            k: vec![0.0; total],
            v: vec![0.0; total],
            num_kv_heads,
            head_dim,
            max_seq_len,
            current_len: 0,
        }
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Append K and V vectors for a single position.
    /// `k_new` layout: `[num_kv_heads × head_dim]`, `v_new` same.
    pub fn append(&mut self, k_new: &[f32], v_new: &[f32]) {
        debug_assert_eq!(k_new.len(), self.num_kv_heads * self.head_dim);
        debug_assert_eq!(v_new.len(), self.num_kv_heads * self.head_dim);
        assert!(self.current_len < self.max_seq_len, "KV cache is full");

        let pos = self.current_len;
        for h in 0..self.num_kv_heads {
            let src_offset = h * self.head_dim;
            let dst_offset = h * self.max_seq_len * self.head_dim + pos * self.head_dim;
            self.k[dst_offset..dst_offset + self.head_dim]
                .copy_from_slice(&k_new[src_offset..src_offset + self.head_dim]);
            self.v[dst_offset..dst_offset + self.head_dim]
                .copy_from_slice(&v_new[src_offset..src_offset + self.head_dim]);
        }
        self.current_len += 1;
    }

    /// Get K values for a specific KV head, for positions `0..num_positions`.
    /// Returns a slice of `[num_positions × head_dim]`.
    pub fn k_head_positions(&self, kv_head: usize, num_positions: usize) -> &[f32] {
        let offset = kv_head * self.max_seq_len * self.head_dim;
        &self.k[offset..offset + num_positions * self.head_dim]
    }

    /// Get V values for a specific KV head, for positions `0..num_positions`.
    pub fn v_head_positions(&self, kv_head: usize, num_positions: usize) -> &[f32] {
        let offset = kv_head * self.max_seq_len * self.head_dim;
        &self.v[offset..offset + num_positions * self.head_dim]
    }

    /// Get K at a specific head and position.
    pub fn k_at(&self, kv_head: usize, pos: usize) -> &[f32] {
        let offset = kv_head * self.max_seq_len * self.head_dim + pos * self.head_dim;
        &self.k[offset..offset + self.head_dim]
    }

    /// Get V at a specific head and position.
    pub fn v_at(&self, kv_head: usize, pos: usize) -> &[f32] {
        let offset = kv_head * self.max_seq_len * self.head_dim + pos * self.head_dim;
        &self.v[offset..offset + self.head_dim]
    }

    pub fn clear(&mut self) {
        self.current_len = 0;
        // No need to zero memory; current_len gates access.
    }
}

/// Per-layer KV cache storing Q32.32 shares (u64).
///
/// Layout: `[num_kv_heads][max_seq_len][head_dim]` -- same as KvCache.
pub struct KvCache64 {
    k: Vec<u64>,
    v: Vec<u64>,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    current_len: usize,
}

impl KvCache64 {
    pub fn new(num_kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let total = num_kv_heads * max_seq_len * head_dim;
        Self {
            k: vec![0u64; total],
            v: vec![0u64; total],
            num_kv_heads,
            head_dim,
            max_seq_len,
            current_len: 0,
        }
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Append K and V vectors for a single position.
    /// `k_new` layout: `[num_kv_heads × head_dim]`, `v_new` same.
    pub fn append(&mut self, k_new: &[u64], v_new: &[u64]) {
        debug_assert_eq!(k_new.len(), self.num_kv_heads * self.head_dim);
        debug_assert_eq!(v_new.len(), self.num_kv_heads * self.head_dim);
        assert!(self.current_len < self.max_seq_len, "KvCache64 is full");

        let pos = self.current_len;
        for h in 0..self.num_kv_heads {
            let src_offset = h * self.head_dim;
            let dst_offset = h * self.max_seq_len * self.head_dim + pos * self.head_dim;
            self.k[dst_offset..dst_offset + self.head_dim]
                .copy_from_slice(&k_new[src_offset..src_offset + self.head_dim]);
            self.v[dst_offset..dst_offset + self.head_dim]
                .copy_from_slice(&v_new[src_offset..src_offset + self.head_dim]);
        }
        self.current_len += 1;
    }

    /// Get K at a specific head and position.
    pub fn k_at(&self, kv_head: usize, pos: usize) -> &[u64] {
        let offset = kv_head * self.max_seq_len * self.head_dim + pos * self.head_dim;
        &self.k[offset..offset + self.head_dim]
    }

    /// Get V at a specific head and position.
    pub fn v_at(&self, kv_head: usize, pos: usize) -> &[u64] {
        let offset = kv_head * self.max_seq_len * self.head_dim + pos * self.head_dim;
        &self.v[offset..offset + self.head_dim]
    }

    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

/// Collection of Q32.32 KV caches, one per layer.
pub struct KvCacheStore64 {
    caches: Vec<KvCache64>,
}

impl KvCacheStore64 {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Self {
        let caches = (0..num_layers)
            .map(|_| KvCache64::new(num_kv_heads, max_seq_len, head_dim))
            .collect();
        Self { caches }
    }

    pub fn layer_mut(&mut self, idx: usize) -> &mut KvCache64 {
        &mut self.caches[idx]
    }

    pub fn clear(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }
}

/// Collection of KV caches, one per layer.
pub struct KvCacheStore {
    caches: Vec<KvCache>,
}

impl KvCacheStore {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Self {
        let caches = (0..num_layers)
            .map(|_| KvCache::new(num_kv_heads, max_seq_len, head_dim))
            .collect();
        Self { caches }
    }

    pub fn layer(&self, idx: usize) -> &KvCache {
        &self.caches[idx]
    }

    pub fn layer_mut(&mut self, idx: usize) -> &mut KvCache {
        &mut self.caches[idx]
    }

    pub fn clear(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_append_and_read() {
        let mut cache = KvCache::new(2, 8, 4);
        assert_eq!(cache.current_len(), 0);

        // Append position 0: head0=[1,2,3,4], head1=[5,6,7,8]
        let k0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v0 = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        cache.append(&k0, &v0);
        assert_eq!(cache.current_len(), 1);

        assert_eq!(cache.k_at(0, 0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.k_at(1, 0), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.v_at(0, 0), &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(cache.v_at(1, 0), &[50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn test_kv_cache_multiple_positions() {
        let mut cache = KvCache::new(1, 8, 2);

        cache.append(&[1.0, 2.0], &[10.0, 20.0]);
        cache.append(&[3.0, 4.0], &[30.0, 40.0]);
        cache.append(&[5.0, 6.0], &[50.0, 60.0]);

        assert_eq!(cache.current_len(), 3);
        assert_eq!(
            cache.k_head_positions(0, 3),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_kv_cache_clear_and_reuse() {
        let mut cache = KvCache::new(1, 4, 2);
        cache.append(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(cache.current_len(), 1);

        cache.clear();
        assert_eq!(cache.current_len(), 0);

        // Can append again
        cache.append(&[5.0, 6.0], &[7.0, 8.0]);
        assert_eq!(cache.current_len(), 1);
        assert_eq!(cache.k_at(0, 0), &[5.0, 6.0]);
    }

    #[test]
    fn test_kv_cache_store() {
        let mut store = KvCacheStore::new(3, 2, 8, 4);
        let k = vec![0.0; 8]; // 2 heads × 4 dim
        let v = vec![0.0; 8];
        store.layer_mut(1).append(&k, &v);
        assert_eq!(store.layer(1).current_len(), 1);
        assert_eq!(store.layer(0).current_len(), 0);

        store.clear();
        assert_eq!(store.layer(1).current_len(), 0);
    }
}
