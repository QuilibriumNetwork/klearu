use crate::layers::{LayerNorm, LinearBias, gelu_inplace};
use super::cpe::ConvPosEnc;
use super::window_attention::WindowAttention;
use super::channel_attention::ChannelAttention;

/// Spatial block: CPE → LayerNorm → WindowAttention → CPE → LayerNorm → MLP.
pub struct SpatialBlock {
    pub cpe1: ConvPosEnc,
    pub norm1: LayerNorm,
    pub attn: WindowAttention,
    pub cpe2: ConvPosEnc,
    pub norm2: LayerNorm,
    pub mlp_fc1: LinearBias,
    pub mlp_fc2: LinearBias,
}

/// Channel block: CPE → LayerNorm → ChannelAttention → CPE → LayerNorm → MLP.
pub struct ChannelBlock {
    pub cpe1: ConvPosEnc,
    pub norm1: LayerNorm,
    pub attn: ChannelAttention,
    pub cpe2: ConvPosEnc,
    pub norm2: LayerNorm,
    pub mlp_fc1: LinearBias,
    pub mlp_fc2: LinearBias,
}

impl SpatialBlock {
    pub fn new(dim: usize, num_heads: usize, mlp_hidden: usize, window_size: usize, eps: f32) -> Self {
        Self {
            cpe1: ConvPosEnc::new(dim),
            norm1: LayerNorm::new(dim, eps),
            attn: WindowAttention::new(dim, num_heads, window_size),
            cpe2: ConvPosEnc::new(dim),
            norm2: LayerNorm::new(dim, eps),
            mlp_fc1: LinearBias::new(dim, mlp_hidden),
            mlp_fc2: LinearBias::new(mlp_hidden, dim),
        }
    }

    /// Forward pass.
    ///
    /// Input: `[C, H, W]` (channel-first). Modified in-place.
    pub fn forward(&self, x: &mut Vec<f32>, h: usize, w: usize) {
        let dim = self.norm1.dim();
        let n = h * w;
        debug_assert_eq!(x.len(), dim * h * w);

        // x = x + cpe1(x)
        self.cpe1.forward(x, h, w);

        // Convert [C, H, W] → [N, C] for attention
        let mut tokens = channel_first_to_tokens(x, dim, h, w);

        // x = x + attn(norm1(x))
        let mut normed = tokens.clone();
        for t in 0..n {
            self.norm1.forward(&mut normed[t * dim..(t + 1) * dim]);
        }
        let attn_out = self.attn.forward(&normed, h, w);
        for (tv, av) in tokens.iter_mut().zip(attn_out.iter()) {
            *tv += av;
        }

        // Convert back [N, C] → [C, H, W] for CPE2
        tokens_to_channel_first(&tokens, x, dim, h, w);

        // x = x + cpe2(x)
        self.cpe2.forward(x, h, w);

        // Convert [C, H, W] → [N, C] for MLP
        tokens = channel_first_to_tokens(x, dim, h, w);

        // x = x + mlp(norm2(x))
        let mut mlp_buf = vec![0.0f32; self.mlp_fc1.out_features()];
        let mut mlp_out = vec![0.0f32; dim];
        for t in 0..n {
            let token = &mut tokens[t * dim..(t + 1) * dim];
            let mut normed_token = token.to_vec();
            self.norm2.forward(&mut normed_token);

            self.mlp_fc1.forward(&normed_token, &mut mlp_buf);
            gelu_inplace(&mut mlp_buf);
            self.mlp_fc2.forward(&mlp_buf, &mut mlp_out);

            for d in 0..dim {
                token[d] += mlp_out[d];
            }
        }

        // Convert [N, C] → [C, H, W]
        tokens_to_channel_first(&tokens, x, dim, h, w);
    }
}

impl ChannelBlock {
    pub fn new(dim: usize, num_heads: usize, mlp_hidden: usize, eps: f32) -> Self {
        Self {
            cpe1: ConvPosEnc::new(dim),
            norm1: LayerNorm::new(dim, eps),
            attn: ChannelAttention::new(dim, num_heads),
            cpe2: ConvPosEnc::new(dim),
            norm2: LayerNorm::new(dim, eps),
            mlp_fc1: LinearBias::new(dim, mlp_hidden),
            mlp_fc2: LinearBias::new(mlp_hidden, dim),
        }
    }

    /// Forward pass.
    ///
    /// Input: `[C, H, W]` (channel-first). Modified in-place.
    pub fn forward(&self, x: &mut Vec<f32>, h: usize, w: usize) {
        let dim = self.norm1.dim();
        let n = h * w;
        debug_assert_eq!(x.len(), dim * h * w);

        // x = x + cpe1(x)
        self.cpe1.forward(x, h, w);

        // Convert [C, H, W] → [N, C] for attention
        let mut tokens = channel_first_to_tokens(x, dim, h, w);

        // x = x + attn(norm1(x))
        let mut normed = tokens.clone();
        for t in 0..n {
            self.norm1.forward(&mut normed[t * dim..(t + 1) * dim]);
        }
        let attn_out = self.attn.forward(&normed, n);
        for (tv, av) in tokens.iter_mut().zip(attn_out.iter()) {
            *tv += av;
        }

        // Convert back [N, C] → [C, H, W] for CPE2
        tokens_to_channel_first(&tokens, x, dim, h, w);

        // x = x + cpe2(x)
        self.cpe2.forward(x, h, w);

        // Convert [C, H, W] → [N, C] for MLP
        tokens = channel_first_to_tokens(x, dim, h, w);

        // x = x + mlp(norm2(x))
        let mut mlp_buf = vec![0.0f32; self.mlp_fc1.out_features()];
        let mut mlp_out = vec![0.0f32; dim];
        for t in 0..n {
            let token = &mut tokens[t * dim..(t + 1) * dim];
            let mut normed_token = token.to_vec();
            self.norm2.forward(&mut normed_token);

            self.mlp_fc1.forward(&normed_token, &mut mlp_buf);
            gelu_inplace(&mut mlp_buf);
            self.mlp_fc2.forward(&mlp_buf, &mut mlp_out);

            for d in 0..dim {
                token[d] += mlp_out[d];
            }
        }

        // Convert [N, C] → [C, H, W]
        tokens_to_channel_first(&tokens, x, dim, h, w);
    }
}

/// Convert `[C, H, W]` (channel-first) to `[N, C]` (token layout).
fn channel_first_to_tokens(chw: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let n = h * w;
    let mut tokens = vec![0.0f32; n * c];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let token_idx = y * w + x;
                tokens[token_idx * c + ch] = chw[ch * h * w + y * w + x];
            }
        }
    }
    tokens
}

/// Convert `[N, C]` (token layout) to `[C, H, W]` (channel-first).
fn tokens_to_channel_first(tokens: &[f32], chw: &mut [f32], c: usize, h: usize, w: usize) {
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let token_idx = y * w + x;
                chw[ch * h * w + y * w + x] = tokens[token_idx * c + ch];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_first_roundtrip() {
        let c = 4;
        let h = 3;
        let w = 5;
        let chw: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let tokens = channel_first_to_tokens(&chw, c, h, w);
        let mut roundtrip = vec![0.0f32; c * h * w];
        tokens_to_channel_first(&tokens, &mut roundtrip, c, h, w);
        assert_eq!(chw, roundtrip);
    }

    #[test]
    fn test_spatial_block_forward() {
        let sb = SpatialBlock::new(8, 2, 16, 7, 1e-5);
        let mut x = vec![0.1f32; 8 * 7 * 7];
        sb.forward(&mut x, 7, 7);
        for (i, &v) in x.iter().enumerate() {
            assert!(v.is_finite(), "x[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_channel_block_forward() {
        let cb = ChannelBlock::new(8, 2, 16, 1e-5);
        let mut x = vec![0.1f32; 8 * 7 * 7];
        cb.forward(&mut x, 7, 7);
        for (i, &v) in x.iter().enumerate() {
            assert!(v.is_finite(), "x[{i}] is not finite: {v}");
        }
    }
}
