pub mod sparse_mlp;
pub mod sparse_attention;
pub mod sparse_convnext;
pub mod sparse_eva02;
pub mod sparse_hiera;
pub mod sparse_qwen_vision;
pub mod sparse_swin;
pub mod sparse_vit;
pub mod predictor_store;
pub mod calibrate;

use crate::config::DaViTConfig;
use crate::model::DaViTModel;
use crate::sparse::predictor_store::VisionPredictorStore;
use crate::sparse::sparse_mlp::forward_sparse_mlp;

/// DaViT model wrapper with SLIDE-based sparsity for MLPs and attention heads.
///
/// Each `(SpatialBlock, ChannelBlock)` pair produces two flat predictor indices:
/// the SpatialBlock at `2*i` and the ChannelBlock at `2*i+1`. Total predictors:
/// `2 * sum(depths)`.
pub struct SparseDaViTModel {
    pub model: DaViTModel,
    pub predictors: VisionPredictorStore,
}

impl SparseDaViTModel {
    pub fn new(model: DaViTModel, predictors: VisionPredictorStore) -> Self {
        Self { model, predictors }
    }

    /// Build a `VisionPredictorStore` for DaViT with per-stage dimensions.
    ///
    /// DaViT has varying `embed_dims`, `num_heads`, and `mlp_hidden_dim` per stage,
    /// and each depth step contains a SpatialBlock + ChannelBlock pair (2 predictors).
    pub fn build_predictors(
        config: &DaViTConfig,
        predictor_hidden: usize,
        neuron_sparsity: f32,
        head_sparsity: f32,
        seed: u64,
    ) -> VisionPredictorStore {
        use klearu_dejavu::predictor::SparsityPredictor;

        let total_blocks: usize = config.depths.iter().sum::<usize>() * 2;
        let mut mlp_predictors = Vec::with_capacity(total_blocks);
        let mut head_predictors = Vec::with_capacity(total_blocks);

        let mut flat_idx: u64 = 0;
        for s in 0..4 {
            let dim = config.embed_dims[s];
            let mlp_hidden = config.mlp_hidden_dim(s);
            let num_heads = config.num_heads[s];

            for _ in 0..config.depths[s] {
                // SpatialBlock predictor
                mlp_predictors.push(SparsityPredictor::new(
                    dim,
                    predictor_hidden,
                    mlp_hidden,
                    seed + flat_idx,
                ));
                head_predictors.push(SparsityPredictor::new(
                    dim,
                    predictor_hidden,
                    num_heads,
                    seed + total_blocks as u64 + flat_idx,
                ));
                flat_idx += 1;

                // ChannelBlock predictor
                mlp_predictors.push(SparsityPredictor::new(
                    dim,
                    predictor_hidden,
                    mlp_hidden,
                    seed + flat_idx,
                ));
                head_predictors.push(SparsityPredictor::new(
                    dim,
                    predictor_hidden,
                    num_heads,
                    seed + total_blocks as u64 + flat_idx,
                ));
                flat_idx += 1;
            }
        }

        VisionPredictorStore {
            mlp_predictors,
            head_predictors,
            neuron_sparsity,
            head_sparsity,
        }
    }

    /// Forward pass with sparse MLP and sparse attention.
    ///
    /// When sparsity = 1.0, this matches dense forward within floating-point tolerance.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        let config = &self.model.config;

        // Patch embedding (dense)
        let (mut features, mut h, mut w) = self.model.stem.forward(
            image,
            config.image_size,
            config.image_size,
        );

        let mut flat_block_idx = 0usize;

        for (s, stage) in self.model.stages.iter().enumerate() {
            // Downsample (dense)
            if let Some(ref ds) = stage.downsample {
                let result = ds.forward(&features, h, w);
                features = result.0;
                h = result.1;
                w = result.2;
            }

            let dim = config.embed_dims[s];
            let num_heads = config.num_heads[s];
            let mlp_hidden = config.mlp_hidden_dim(s);

            for (spatial, channel) in &stage.blocks {
                // ────────────────────────────────────────────────────
                // SpatialBlock: CPE1 → norm1 → sparse WindowAttn → CPE2 → norm2 → sparse MLP
                // ────────────────────────────────────────────────────

                // CPE1 (dense, in-place on [C,H,W])
                spatial.cpe1.forward(&mut features, h, w);

                // Convert [C,H,W] → [N,C] for attention
                let n = h * w;
                let mut tokens = channel_first_to_tokens(&features, dim, h, w);

                // norm1
                let mut normed = tokens.clone();
                for t in 0..n {
                    spatial.norm1.forward(&mut normed[t * dim..(t + 1) * dim]);
                }

                // Mean token for predictor
                let mean_token = compute_mean_token(&tokens, dim, n);

                // Sparse window attention
                let active_heads = self.predictors.select_heads(
                    flat_block_idx, &mean_token, num_heads,
                );
                let attn_out = forward_sparse_window_attention(
                    &spatial.attn, &normed, h, w, &active_heads,
                );
                for (x, a) in tokens.iter_mut().zip(attn_out.iter()) {
                    *x += a;
                }

                // Convert [N,C] → [C,H,W] for CPE2
                tokens_to_channel_first(&tokens, &mut features, dim, h, w);

                // CPE2 (dense)
                spatial.cpe2.forward(&mut features, h, w);

                // Convert [C,H,W] → [N,C] for MLP
                tokens = channel_first_to_tokens(&features, dim, h, w);

                // Sparse MLP
                let active_neurons = self.predictors.select_neurons(
                    flat_block_idx, &mean_token, mlp_hidden,
                );
                for t in 0..n {
                    let token = &mut tokens[t * dim..(t + 1) * dim];
                    let mut normed_token = token.to_vec();
                    spatial.norm2.forward(&mut normed_token);

                    let mlp_out = forward_sparse_mlp(
                        &spatial.mlp_fc1, &spatial.mlp_fc2,
                        &normed_token, &active_neurons,
                    );
                    for d in 0..dim {
                        token[d] += mlp_out[d];
                    }
                }

                // Convert [N,C] → [C,H,W]
                tokens_to_channel_first(&tokens, &mut features, dim, h, w);

                flat_block_idx += 1;

                // ────────────────────────────────────────────────────
                // ChannelBlock: CPE1 → norm1 → sparse ChannelAttn → CPE2 → norm2 → sparse MLP
                // ────────────────────────────────────────────────────

                // CPE1 (dense)
                channel.cpe1.forward(&mut features, h, w);

                // Convert [C,H,W] → [N,C]
                let mut tokens = channel_first_to_tokens(&features, dim, h, w);

                // norm1
                let mut normed = tokens.clone();
                for t in 0..n {
                    channel.norm1.forward(&mut normed[t * dim..(t + 1) * dim]);
                }

                // Mean token for predictor
                let mean_token = compute_mean_token(&tokens, dim, n);

                // Sparse channel attention
                let active_heads = self.predictors.select_heads(
                    flat_block_idx, &mean_token, num_heads,
                );
                let attn_out = forward_sparse_channel_attention(
                    &channel.attn, &normed, n, &active_heads,
                );
                for (x, a) in tokens.iter_mut().zip(attn_out.iter()) {
                    *x += a;
                }

                // Convert [N,C] → [C,H,W] for CPE2
                tokens_to_channel_first(&tokens, &mut features, dim, h, w);

                // CPE2 (dense)
                channel.cpe2.forward(&mut features, h, w);

                // Convert [C,H,W] → [N,C] for MLP
                tokens = channel_first_to_tokens(&features, dim, h, w);

                // Sparse MLP
                let active_neurons = self.predictors.select_neurons(
                    flat_block_idx, &mean_token, mlp_hidden,
                );
                for t in 0..n {
                    let token = &mut tokens[t * dim..(t + 1) * dim];
                    let mut normed_token = token.to_vec();
                    channel.norm2.forward(&mut normed_token);

                    let mlp_out = forward_sparse_mlp(
                        &channel.mlp_fc1, &channel.mlp_fc2,
                        &normed_token, &active_neurons,
                    );
                    for d in 0..dim {
                        token[d] += mlp_out[d];
                    }
                }

                // Convert [N,C] → [C,H,W]
                tokens_to_channel_first(&tokens, &mut features, dim, h, w);

                flat_block_idx += 1;
            }
        }

        // Classification head (dense)
        self.model.head.forward(&features, config.embed_dims[3], h, w)
    }
}

// ─── Layout conversion helpers ──────────────────────────────────────────────

/// Convert `[C, H, W]` (channel-first) to `[N, C]` (token layout).
fn channel_first_to_tokens(chw: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let n = h * w;
    let mut tokens = vec![0.0f32; n * c];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let token_idx = y * w + x;
                tokens[token_idx * c + ch] = chw[ch * n + y * w + x];
            }
        }
    }
    tokens
}

/// Convert `[N, C]` (token layout) to `[C, H, W]` (channel-first).
fn tokens_to_channel_first(tokens: &[f32], chw: &mut [f32], c: usize, h: usize, w: usize) {
    let n = h * w;
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let token_idx = y * w + x;
                chw[ch * n + y * w + x] = tokens[token_idx * c + ch];
            }
        }
    }
}

/// Compute mean token across spatial dimension.
fn compute_mean_token(tokens: &[f32], dim: usize, n: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; dim];
    for t in 0..n {
        for d in 0..dim {
            mean[d] += tokens[t * dim + d];
        }
    }
    let inv = 1.0 / n as f32;
    for v in &mut mean {
        *v *= inv;
    }
    mean
}

// ─── Sparse window attention ────────────────────────────────────────────────

/// Sparse window attention: partition into windows, then only compute attention
/// for active heads within each window.
///
/// Full QKV projection is done densely (cheap), then only active heads are
/// computed. Inactive head positions are zeroed before output projection.
fn forward_sparse_window_attention(
    attn: &crate::model::window_attention::WindowAttention,
    input: &[f32],
    h: usize,
    w: usize,
    active_heads: &[usize],
) -> Vec<f32> {
    let dim = attn.num_heads * attn.head_dim;
    let head_dim = attn.head_dim;
    let n = h * w;
    let ws = attn.window_size;
    let num_windows_h = h / ws;
    let num_windows_w = w / ws;
    let window_tokens = ws * ws;

    // Get relative position bias
    let rel_bias = attn.get_relative_position_bias();

    let mut output = vec![0.0f32; n * dim];

    for wy in 0..num_windows_h {
        for wx in 0..num_windows_w {
            // Gather tokens for this window
            let mut window_input = vec![0.0f32; window_tokens * dim];
            for sy in 0..ws {
                for sx in 0..ws {
                    let global_y = wy * ws + sy;
                    let global_x = wx * ws + sx;
                    let global_idx = global_y * w + global_x;
                    let local_idx = sy * ws + sx;
                    window_input[local_idx * dim..(local_idx + 1) * dim]
                        .copy_from_slice(&input[global_idx * dim..(global_idx + 1) * dim]);
                }
            }

            // Dense QKV projection per window
            let mut qkv = vec![0.0f32; window_tokens * 3 * dim];
            for t in 0..window_tokens {
                attn.qkv.forward(
                    &window_input[t * dim..(t + 1) * dim],
                    &mut qkv[t * 3 * dim..(t + 1) * 3 * dim],
                );
            }

            // Sparse attention: only compute for active heads
            let scale = (head_dim as f32).powf(-0.5);
            let mut attn_output = vec![0.0f32; window_tokens * dim];

            for &head in active_heads {
                let head_offset = head * head_dim;

                let mut scores = vec![0.0f32; window_tokens * window_tokens];

                for qi in 0..window_tokens {
                    for ki in 0..window_tokens {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            let q = qkv[qi * 3 * dim + head_offset + d];
                            let k = qkv[ki * 3 * dim + dim + head_offset + d];
                            dot += q * k;
                        }
                        scores[qi * window_tokens + ki] = dot * scale;
                    }
                }

                // Add relative position bias
                if let Some(ref bias) = rel_bias {
                    let bias_offset = head * window_tokens * window_tokens;
                    for i in 0..window_tokens * window_tokens {
                        scores[i] += bias[bias_offset + i];
                    }
                }

                // Softmax per row
                for qi in 0..window_tokens {
                    let row = &mut scores[qi * window_tokens..(qi + 1) * window_tokens];
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max).exp();
                        sum += *v;
                    }
                    if sum > 0.0 {
                        for v in row.iter_mut() {
                            *v /= sum;
                        }
                    }
                }

                // Weighted V
                for qi in 0..window_tokens {
                    for d in 0..head_dim {
                        let mut val = 0.0f32;
                        for vi in 0..window_tokens {
                            let v = qkv[vi * 3 * dim + 2 * dim + head_offset + d];
                            val += scores[qi * window_tokens + vi] * v;
                        }
                        attn_output[qi * dim + head_offset + d] = val;
                    }
                }
            }

            // Output projection
            let mut proj_out = vec![0.0f32; window_tokens * dim];
            for t in 0..window_tokens {
                attn.proj.forward(
                    &attn_output[t * dim..(t + 1) * dim],
                    &mut proj_out[t * dim..(t + 1) * dim],
                );
            }

            // Scatter back to global output
            for sy in 0..ws {
                for sx in 0..ws {
                    let global_y = wy * ws + sy;
                    let global_x = wx * ws + sx;
                    let global_idx = global_y * w + global_x;
                    let local_idx = sy * ws + sx;
                    output[global_idx * dim..(global_idx + 1) * dim]
                        .copy_from_slice(&proj_out[local_idx * dim..(local_idx + 1) * dim]);
                }
            }
        }
    }

    output
}

// ─── Sparse channel attention ───────────────────────────────────────────────

/// Sparse channel attention: only compute attention for active heads.
///
/// Channel attention transposes Q/K/V so head_dim positions become the sequence
/// dimension and N spatial positions become the feature dimension.
/// Score matrix is `[head_dim, head_dim]` per head, scale = `N^(-0.5)`.
fn forward_sparse_channel_attention(
    attn: &crate::model::channel_attention::ChannelAttention,
    input: &[f32],
    n: usize,
    active_heads: &[usize],
) -> Vec<f32> {
    let dim = attn.num_heads * attn.head_dim;
    let head_dim = attn.head_dim;

    // Dense QKV projection
    let mut qkv = vec![0.0f32; n * 3 * dim];
    for t in 0..n {
        attn.qkv.forward(
            &input[t * dim..(t + 1) * dim],
            &mut qkv[t * 3 * dim..(t + 1) * 3 * dim],
        );
    }

    let scale = (n as f32).powf(-0.5);
    let mut attn_output = vec![0.0f32; n * dim];

    for &head in active_heads {
        let head_offset = head * head_dim;

        // Channel attention: scores[d1, d2] = sum_n(Q[n, d1] * K[n, d2]) * scale
        let mut scores = vec![0.0f32; head_dim * head_dim];

        for d1 in 0..head_dim {
            for d2 in 0..head_dim {
                let mut dot = 0.0f32;
                for t in 0..n {
                    let q = qkv[t * 3 * dim + head_offset + d1];
                    let k = qkv[t * 3 * dim + dim + head_offset + d2];
                    dot += q * k;
                }
                scores[d1 * head_dim + d2] = dot * scale;
            }
        }

        // Softmax per row (over d2)
        for d1 in 0..head_dim {
            let row = &mut scores[d1 * head_dim..(d1 + 1) * head_dim];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }

        // Weighted sum: output[n, d1] = sum_d2(scores[d1, d2] * V[n, d2])
        for t in 0..n {
            for d1 in 0..head_dim {
                let mut val = 0.0f32;
                for d2 in 0..head_dim {
                    let v = qkv[t * 3 * dim + 2 * dim + head_offset + d2];
                    val += scores[d1 * head_dim + d2] * v;
                }
                attn_output[t * dim + head_offset + d1] = val;
            }
        }
    }

    // Output projection
    let mut proj_out = vec![0.0f32; n * dim];
    for t in 0..n {
        attn.proj.forward(
            &attn_output[t * dim..(t + 1) * dim],
            &mut proj_out[t * dim..(t + 1) * dim],
        );
    }

    proj_out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DaViTConfig;

    fn tiny_test_config() -> DaViTConfig {
        // image_size=224 → stem(7x7,s4,p3) → 56x56
        // stage0: no downsample → 56x56 (window=7 fits: 56/7=8)
        // stage1: downsample → 28x28 (window=7 fits: 28/7=4)
        // stage2: downsample → 14x14 (window=7 fits: 14/7=2)
        // stage3: downsample → 7x7 (window=7 fits: 7/7=1)
        DaViTConfig {
            image_size: 224,
            in_channels: 3,
            embed_dims: [8, 16, 32, 64],
            num_heads: [2, 2, 4, 8],
            depths: [1, 1, 1, 1],
            window_size: 7,
            mlp_ratio: 2.0,
            num_classes: 10,
            layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn test_sparse_davit_full_sparsity_matches_dense() {
        let config = tiny_test_config();

        // Dense forward
        let model = DaViTModel::new(config.clone());
        let input = vec![0.1f32; 3 * 224 * 224];
        let dense_out = model.forward(&input);

        // Sparse forward with sparsity=1.0 (keep all)
        let predictors = SparseDaViTModel::build_predictors(&config, 4, 1.0, 1.0, 42);
        let model2 = DaViTModel::new(config);
        let sparse_model = SparseDaViTModel::new(model2, predictors);
        let sparse_out = sparse_model.forward(&input);

        assert_eq!(dense_out.len(), sparse_out.len());
        for i in 0..dense_out.len() {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-3,
                "logit[{i}]: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i],
            );
        }
    }

    #[test]
    fn test_sparse_davit_runs_with_sparsity() {
        let config = tiny_test_config();
        let predictors = SparseDaViTModel::build_predictors(&config, 4, 0.5, 0.5, 42);
        let model = DaViTModel::new(config);
        let sparse_model = SparseDaViTModel::new(model, predictors);

        let input = vec![0.1f32; 3 * 224 * 224];
        let logits = sparse_model.forward(&input);
        assert_eq!(logits.len(), 10);
        for &v in &logits {
            assert!(v.is_finite(), "logit is not finite: {}", v);
        }
    }

    #[test]
    fn test_sparse_davit_predictor_counts() {
        let config = tiny_test_config();
        let total_predictors: usize = config.depths.iter().sum::<usize>() * 2;
        let predictors = SparseDaViTModel::build_predictors(&config, 4, 0.5, 0.5, 42);

        assert_eq!(predictors.mlp_predictors.len(), total_predictors);
        assert_eq!(predictors.head_predictors.len(), total_predictors);
    }
}
