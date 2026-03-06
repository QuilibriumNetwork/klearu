use klearu_dejavu::predictor::SparsityPredictor;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Stores trained sparsity predictors for a vision model.
///
/// Each transformer block can have:
/// - An MLP predictor (selects active neurons in fc1)
/// - A head predictor (selects active attention heads)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionPredictorStore {
    /// Per-block MLP sparsity predictors.
    pub mlp_predictors: Vec<SparsityPredictor>,
    /// Per-block attention head sparsity predictors.
    pub head_predictors: Vec<SparsityPredictor>,
    /// Fraction of MLP neurons to keep (0.0 = none, 1.0 = all).
    pub neuron_sparsity: f32,
    /// Fraction of attention heads to keep (0.0 = none, 1.0 = all).
    pub head_sparsity: f32,
}

impl VisionPredictorStore {
    /// Create a new store with the given number of blocks.
    pub fn new(
        num_blocks: usize,
        input_dim: usize,
        mlp_hidden_dim: usize,
        num_heads: usize,
        predictor_hidden: usize,
        neuron_sparsity: f32,
        head_sparsity: f32,
        seed: u64,
    ) -> Self {
        let mlp_predictors: Vec<SparsityPredictor> = (0..num_blocks)
            .map(|i| {
                SparsityPredictor::new(input_dim, predictor_hidden, mlp_hidden_dim, seed + i as u64)
            })
            .collect();

        let head_predictors: Vec<SparsityPredictor> = (0..num_blocks)
            .map(|i| {
                SparsityPredictor::new(
                    input_dim,
                    predictor_hidden,
                    num_heads,
                    seed + num_blocks as u64 + i as u64,
                )
            })
            .collect();

        Self {
            mlp_predictors,
            head_predictors,
            neuron_sparsity,
            head_sparsity,
        }
    }

    /// Number of blocks this store covers.
    pub fn num_blocks(&self) -> usize {
        self.mlp_predictors.len()
    }

    /// Save to a JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load from a JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Select active MLP neurons for a given block.
    pub fn select_neurons(&self, block_idx: usize, input: &[f32], mlp_dim: usize) -> Vec<usize> {
        let k = ((mlp_dim as f32 * self.neuron_sparsity).ceil() as usize)
            .max(1)
            .min(mlp_dim);
        self.mlp_predictors[block_idx].select_top_k(input, k)
    }

    /// Select active attention heads for a given block.
    pub fn select_heads(&self, block_idx: usize, input: &[f32], num_heads: usize) -> Vec<usize> {
        let k = ((num_heads as f32 * self.head_sparsity).ceil() as usize)
            .max(1)
            .min(num_heads);
        self.head_predictors[block_idx].select_top_k(input, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_store_creation() {
        let store = VisionPredictorStore::new(4, 64, 256, 8, 32, 0.5, 0.5, 42);
        assert_eq!(store.num_blocks(), 4);
        assert_eq!(store.mlp_predictors.len(), 4);
        assert_eq!(store.head_predictors.len(), 4);
    }

    #[test]
    fn test_select_neurons_respects_sparsity() {
        let store = VisionPredictorStore::new(1, 16, 32, 4, 8, 0.5, 0.5, 42);
        let input = vec![1.0f32; 16];
        let selected = store.select_neurons(0, &input, 32);
        assert_eq!(selected.len(), 16); // ceil(32 * 0.5) = 16
    }

    #[test]
    fn test_select_heads_respects_sparsity() {
        let store = VisionPredictorStore::new(1, 16, 32, 8, 8, 0.5, 0.5, 42);
        let input = vec![1.0f32; 16];
        let selected = store.select_heads(0, &input, 8);
        assert_eq!(selected.len(), 4); // ceil(8 * 0.5) = 4
    }
}
