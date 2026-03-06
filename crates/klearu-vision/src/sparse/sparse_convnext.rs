use crate::layers::gelu_inplace;
use crate::model::convnext::ConvNextModel;
use crate::sparse::predictor_store::VisionPredictorStore;
use crate::sparse::sparse_mlp::forward_sparse_mlp;

/// ConvNeXt model with SLIDE-based sparsity for the MLP (fc1→GELU→fc2) path.
///
/// ConvNeXt has NO attention, only Conv2d + MLP per block, so sparsity applies
/// exclusively to the MLP neurons within each `ConvNextBlock`. The depthwise
/// convolution, downsampling, and all normalization layers remain dense.
///
/// Block indices for predictors are flattened across all four stages.
pub struct SparseConvNextModel {
    pub model: ConvNextModel,
    pub predictors: VisionPredictorStore,
}

impl SparseConvNextModel {
    pub fn new(model: ConvNextModel, predictors: VisionPredictorStore) -> Self {
        Self { model, predictors }
    }

    /// Forward pass with sparse MLP within each ConvNeXt block.
    ///
    /// Mirrors `ConvNextModel::forward` but replaces the fc1→GELU→fc2 path in
    /// each block with `forward_sparse_mlp` using predictor-selected neurons.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        let features = self.forward_features(image);
        let mut logits = vec![0.0f32; self.model.config.num_classes];
        self.model.head.forward(&features, &mut logits);
        logits
    }

    /// Forward pass to extract pooled features, with sparse MLPs.
    pub fn forward_features(&self, image: &[f32]) -> Vec<f32> {
        let image_size = 224; // Assume standard size (matches ConvNextModel)

        let (stem_h, stem_w) = self.model.stem_conv.output_dims(image_size, image_size);
        let mut features = vec![0.0f32; self.model.config.dims[0] * stem_h * stem_w];
        self.model.stem_conv.forward(image, image_size, image_size, &mut features);

        // Stem norm (channel-wise)
        self.model.stem_norm.forward_2d(&mut features, stem_h, stem_w);

        let mut h = stem_h;
        let mut w = stem_w;
        let mut dim = self.model.config.dims[0];
        let mut block_idx = 0usize;

        for s in 0..4 {
            // Downsample between stages (except first)
            if let Some(ref ds) = self.model.downsamples[s] {
                let (new_features, new_h, new_w) = ds.forward(&features, dim, h, w);
                features = new_features;
                h = new_h;
                w = new_w;
                dim = self.model.config.dims[s];
            }

            // Stage blocks with sparse MLP
            for block in &self.model.stages[s] {
                self.forward_block_sparse(block, &mut features, dim, h, w, block_idx);
                block_idx += 1;
            }
        }

        // Global average pool: [C, H, W] → [C]
        let spatial = h * w;
        let mut pooled = vec![0.0f32; dim];
        for c in 0..dim {
            let mut sum = 0.0f32;
            for p in 0..spatial {
                sum += features[c * spatial + p];
            }
            pooled[c] = sum / spatial as f32;
        }

        self.model.final_norm.forward(&mut pooled);
        pooled
    }

    /// Sparse block forward: same as `ConvNextBlock::forward` but with sparse MLP.
    ///
    /// Layout: `[C, H, W]` channel-first with residual connection.
    fn forward_block_sparse(
        &self,
        block: &crate::model::convnext::ConvNextBlock,
        input: &mut [f32],
        dim: usize,
        h: usize,
        w: usize,
        block_idx: usize,
    ) {
        let spatial = h * w;
        let mlp_hidden = block.fc1.out_features();

        // Depthwise conv (dense, unchanged)
        let mut conv_out = vec![0.0f32; dim * spatial];
        block.dwconv.forward(input, h, w, &mut conv_out);

        // Compute a mean channel vector for the predictor input.
        // Average across all spatial positions after dwconv.
        let mut mean_token = vec![0.0f32; dim];
        let inv_spatial = 1.0 / spatial as f32;
        for c in 0..dim {
            let mut sum = 0.0f32;
            for p in 0..spatial {
                sum += conv_out[c * spatial + p];
            }
            mean_token[c] = sum * inv_spatial;
        }

        // Predict active MLP neurons for this block
        let active_neurons = self.predictors.select_neurons(block_idx, &mean_token, mlp_hidden);

        // Per spatial position: gather → LayerNorm → sparse MLP → optional LayerScale/GRN → residual
        for p in 0..spatial {
            // Gather channel values for this spatial position
            let mut token = vec![0.0f32; dim];
            for c in 0..dim {
                token[c] = conv_out[c * spatial + p];
            }

            // LayerNorm
            block.norm.forward(&mut token);

            // Sparse fc1 → GELU → fc2
            let mut mlp_out = forward_sparse_mlp(
                &block.fc1,
                &block.fc2,
                &token,
                &active_neurons,
            );

            // V2 path: GRN is applied between GELU and fc2, requiring
            // a separate code path from the combined forward_sparse_mlp.
            if block.grn.is_some() {
                mlp_out = self.forward_block_mlp_with_grn(
                    block,
                    &token,
                    &active_neurons,
                );
            }

            // Optional LayerScale (V1)
            if let Some(ref ls) = block.layer_scale {
                ls.forward(&mut mlp_out);
            }

            // Residual: scatter back to channel-first layout
            for c in 0..dim {
                input[c * spatial + p] += mlp_out[c];
            }
        }
    }

    /// MLP forward with GRN inserted between GELU and fc2 (ConvNeXt V2 path).
    ///
    /// Computes sparse fc1, expands to full hidden dim for GRN, then fc2.
    fn forward_block_mlp_with_grn(
        &self,
        block: &crate::model::convnext::ConvNextBlock,
        input: &[f32],
        active_neurons: &[usize],
    ) -> Vec<f32> {
        let dim = block.fc2.out_features();
        let mlp_hidden = block.fc1.out_features();
        let k = active_neurons.len();

        // Sparse fc1: only compute selected neurons
        let mut fc1_sparse = vec![0.0f32; k];
        block.fc1.forward_sparse(input, active_neurons, &mut fc1_sparse);
        gelu_inplace(&mut fc1_sparse);

        // Expand sparse activations to full hidden dim for GRN (zeros for inactive)
        let mut fc1_full = vec![0.0f32; mlp_hidden];
        for (sparse_idx, &neuron_idx) in active_neurons.iter().enumerate() {
            fc1_full[neuron_idx] = fc1_sparse[sparse_idx];
        }

        // Apply GRN on the full intermediate representation
        if let Some(ref grn) = block.grn {
            grn.forward(&mut fc1_full);
        }

        // fc2: dense forward from the (GRN-processed) full hidden
        let mut output = vec![0.0f32; dim];
        block.fc2.forward(&fc1_full, &mut output);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConvNextConfig;
    use crate::model::convnext::ConvNextModel;

    fn tiny_test_config() -> ConvNextConfig {
        ConvNextConfig {
            in_channels: 3,
            dims: [8, 16, 32, 64],
            depths: [1, 1, 1, 1],
            num_classes: 5,
            layer_scale_init: 1e-6,
            layer_norm_eps: 1e-6,
            is_v2: false,
        }
    }

    /// Total number of blocks across all stages.
    fn total_blocks(config: &ConvNextConfig) -> usize {
        config.depths.iter().sum()
    }

    #[test]
    fn test_sparse_convnext_runs() {
        let config = tiny_test_config();
        let num_blocks = total_blocks(&config);

        // Use largest stage dim for the uniform predictor store.
        let max_dim = *config.dims.iter().max().unwrap();
        let max_mlp_hidden = 4 * max_dim;

        let model = ConvNextModel::new(config.clone());
        let predictors = VisionPredictorStore::new(
            num_blocks,
            max_dim,       // input_dim for predictors
            max_mlp_hidden, // mlp_hidden_dim
            1,             // num_heads
            8,             // predictor_hidden
            0.5,           // neuron_sparsity
            0.5,           // head_sparsity
            42,            // seed
        );
        let sparse_model = SparseConvNextModel::new(model, predictors);

        let input = vec![0.1f32; 3 * 224 * 224];
        let logits = sparse_model.forward(&input);
        assert_eq!(logits.len(), 5);
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_sparse_convnext_full_sparsity() {
        let config = tiny_test_config();
        let num_blocks = total_blocks(&config);
        let max_dim = *config.dims.iter().max().unwrap();
        let max_mlp_hidden = 4 * max_dim;

        // Dense reference
        let model = ConvNextModel::new(config.clone());
        let dense_out = model.forward(&vec![0.1f32; 3 * 224 * 224]);

        // Sparse with sparsity=1.0 (keep all neurons)
        let model2 = ConvNextModel::new(config.clone());
        let predictors = VisionPredictorStore::new(
            num_blocks,
            max_dim,
            max_mlp_hidden,
            1,
            8,
            1.0, // keep all neurons
            1.0, // keep all heads (unused)
            42,
        );
        let sparse_model = SparseConvNextModel::new(model2, predictors);
        let sparse_out = sparse_model.forward(&vec![0.1f32; 3 * 224 * 224]);

        assert_eq!(dense_out.len(), sparse_out.len());

        // With sparsity=1.0 (keep all neurons), results should match dense forward.
        // The sparse path gathers all neurons, so the fc1→GELU→fc2 should produce
        // identical results. Note: due to the scatter-back approach in
        // forward_sparse_mlp (computing fc2 via per-row dot products) vs the
        // dense forward (matrix-vector multiply), floating-point ordering may
        // differ slightly.
        for i in 0..dense_out.len() {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-3,
                "logit[{i}]: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i],
            );
        }
    }
}
