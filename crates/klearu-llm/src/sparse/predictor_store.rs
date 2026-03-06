use std::path::Path;

use klearu_dejavu::predictor::SparsityPredictor;

/// Per-layer storage for head and neuron sparsity predictors.
pub struct PredictorStore {
    head_predictors: Vec<Option<SparsityPredictor>>,
    neuron_predictors: Vec<Option<SparsityPredictor>>,
    num_layers: usize,
}

impl PredictorStore {
    pub fn new(num_layers: usize) -> Self {
        Self {
            head_predictors: (0..num_layers).map(|_| None).collect(),
            neuron_predictors: (0..num_layers).map(|_| None).collect(),
            num_layers,
        }
    }

    pub fn set_head_predictor(&mut self, layer: usize, predictor: SparsityPredictor) {
        self.head_predictors[layer] = Some(predictor);
    }

    pub fn set_neuron_predictor(&mut self, layer: usize, predictor: SparsityPredictor) {
        self.neuron_predictors[layer] = Some(predictor);
    }

    /// Predict top-k important head indices for a layer.
    /// Returns all heads if no predictor is set.
    pub fn predict_heads(&self, layer: usize, input: &[f32], k: usize) -> Vec<usize> {
        match &self.head_predictors[layer] {
            Some(predictor) => predictor.select_top_k(input, k),
            None => (0..k).collect(), // fallback: return first k heads
        }
    }

    /// Predict top-k important neuron indices for a layer.
    /// Returns all neurons if no predictor is set.
    pub fn predict_neurons(&self, layer: usize, input: &[f32], k: usize) -> Vec<usize> {
        match &self.neuron_predictors[layer] {
            Some(predictor) => predictor.select_top_k(input, k),
            None => (0..k).collect(),
        }
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Returns true if any layer has a calibrated head or neuron predictor.
    pub fn is_calibrated(&self) -> bool {
        self.head_predictors.iter().any(|p| p.is_some())
            || self.neuron_predictors.iter().any(|p| p.is_some())
    }

    /// Save all predictors to a directory.
    /// Each predictor is saved as `layer_{i}_head.json` / `layer_{i}_neuron.json`.
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        for (i, pred) in self.head_predictors.iter().enumerate() {
            if let Some(p) = pred {
                p.save(&dir.join(format!("layer_{i}_head.json")))?;
            }
        }
        for (i, pred) in self.neuron_predictors.iter().enumerate() {
            if let Some(p) = pred {
                p.save(&dir.join(format!("layer_{i}_neuron.json")))?;
            }
        }
        Ok(())
    }

    /// Load predictors from a directory. Missing files are silently skipped
    /// (those layers will use the fallback strategy).
    pub fn load(dir: &Path, num_layers: usize) -> std::io::Result<Self> {
        let mut store = Self::new(num_layers);
        for i in 0..num_layers {
            let head_path = dir.join(format!("layer_{i}_head.json"));
            if head_path.exists() {
                store.head_predictors[i] = Some(SparsityPredictor::load(&head_path)?);
            }
            let neuron_path = dir.join(format!("layer_{i}_neuron.json"));
            if neuron_path.exists() {
                store.neuron_predictors[i] = Some(SparsityPredictor::load(&neuron_path)?);
            }
        }
        Ok(store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_store_fallback() {
        let store = PredictorStore::new(4);
        let input = vec![0.1; 16];
        let heads = store.predict_heads(0, &input, 4);
        assert_eq!(heads, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_predictor_store_save_load_roundtrip() {
        let mut store = PredictorStore::new(2);
        let head_pred = SparsityPredictor::new(8, 4, 6, 42);
        let neuron_pred = SparsityPredictor::new(8, 4, 12, 43);
        store.set_head_predictor(0, head_pred);
        store.set_neuron_predictor(0, neuron_pred);

        let dir = std::env::temp_dir().join("klearu_test_store");
        store.save(&dir).unwrap();

        let loaded = PredictorStore::load(&dir, 2).unwrap();
        let input = vec![0.1f32; 8];

        // Layer 0 should have trained predictors
        let heads_orig = store.predict_heads(0, &input, 3);
        let heads_loaded = loaded.predict_heads(0, &input, 3);
        assert_eq!(heads_orig, heads_loaded);

        // Layer 1 should still use fallback
        let heads_fallback = loaded.predict_heads(1, &input, 3);
        assert_eq!(heads_fallback, vec![0, 1, 2]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_predictor_store_with_predictor() {
        let mut store = PredictorStore::new(2);
        let predictor = SparsityPredictor::new(8, 4, 6, 42);
        store.set_head_predictor(0, predictor);

        let input = vec![0.1; 8];
        let heads = store.predict_heads(0, &input, 3);
        assert_eq!(heads.len(), 3);
        // All indices should be valid (< 6)
        assert!(heads.iter().all(|&h| h < 6));
    }
}
