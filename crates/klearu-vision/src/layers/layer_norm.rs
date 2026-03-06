/// LayerNorm with learnable weight and bias.
///
/// Unlike RMSNorm, this includes mean subtraction and an additive bias.
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; dim],
            bias: vec![0.0; dim],
            eps,
        }
    }

    pub fn dim(&self) -> usize {
        self.weight.len()
    }

    /// Standard LayerNorm: normalize over the last dimension.
    /// `x` has length `dim`.
    pub fn forward(&self, x: &mut [f32]) {
        let n = x.len();
        debug_assert_eq!(n, self.weight.len());

        let mean = x.iter().sum::<f32>() / n as f32;
        for v in x.iter_mut() {
            *v -= mean;
        }

        let var = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + self.eps).sqrt();

        for i in 0..n {
            x[i] = x[i] * inv_std * self.weight[i] + self.bias[i];
        }
    }

    /// LayerNorm2d: normalize over channel dimension for each spatial position.
    ///
    /// Input layout: `[C, H, W]` (channel-first). Normalizes the C-dimensional
    /// vector at each (h, w) position independently.
    pub fn forward_2d(&self, x: &mut [f32], h: usize, w: usize) {
        let c = self.weight.len();
        debug_assert_eq!(x.len(), c * h * w);

        for y in 0..h {
            for xp in 0..w {
                // Gather the C values at position (y, xp)
                let mut mean = 0.0f32;
                for ch in 0..c {
                    mean += x[ch * h * w + y * w + xp];
                }
                mean /= c as f32;

                let mut var = 0.0f32;
                for ch in 0..c {
                    let v = x[ch * h * w + y * w + xp] - mean;
                    var += v * v;
                }
                var /= c as f32;
                let inv_std = 1.0 / (var + self.eps).sqrt();

                for ch in 0..c {
                    let idx = ch * h * w + y * w + xp;
                    x[idx] = (x[idx] - mean) * inv_std * self.weight[ch] + self.bias[ch];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_zero_mean() {
        let ln = LayerNorm::new(4, 1e-5);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        ln.forward(&mut x);

        // Mean should be ~0
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean={mean}");

        // Variance should be ~1 (with weight=1, bias=0)
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 0.01, "var={var}");
    }

    #[test]
    fn test_layernorm_known_values() {
        let ln = LayerNorm::new(4, 0.0);
        let mut x = vec![1.0, 3.0, 5.0, 7.0];
        // mean = 4.0, var = (9+1+1+9)/4 = 5.0, std = sqrt(5)
        ln.forward(&mut x);

        let std5 = 5.0f32.sqrt();
        assert!((x[0] - (-3.0 / std5)).abs() < 1e-5);
        assert!((x[1] - (-1.0 / std5)).abs() < 1e-5);
        assert!((x[2] - (1.0 / std5)).abs() < 1e-5);
        assert!((x[3] - (3.0 / std5)).abs() < 1e-5);
    }

    #[test]
    fn test_layernorm_with_weight_bias() {
        let mut ln = LayerNorm::new(2, 0.0);
        ln.weight = vec![2.0, 3.0];
        ln.bias = vec![1.0, -1.0];

        let mut x = vec![1.0, 3.0];
        // mean=2, var=1, std=1
        // norm = [-1, 1]
        // output = [-1*2+1, 1*3-1] = [-1, 2]
        ln.forward(&mut x);
        assert!((x[0] - (-1.0)).abs() < 1e-5, "x[0]={}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-5, "x[1]={}", x[1]);
    }

    #[test]
    fn test_layernorm_2d() {
        // 2 channels, 2x2 spatial
        let ln = LayerNorm::new(2, 0.0);
        // Input: [C=2, H=2, W=2]
        let mut x = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0
            5.0, 6.0, 7.0, 8.0, // channel 1
        ];
        ln.forward_2d(&mut x, 2, 2);

        // At position (0,0): values are [1, 5], mean=3, var=4, std=2
        // norm = [-1, 1]
        assert!((x[0] - (-1.0)).abs() < 1e-5, "x[0,0,0]={}", x[0]);
        assert!((x[4] - 1.0).abs() < 1e-5, "x[1,0,0]={}", x[4]);

        // At position (0,1): values are [2, 6], mean=4, var=4, std=2
        // norm = [-1, 1]
        assert!((x[1] - (-1.0)).abs() < 1e-5, "x[0,0,1]={}", x[1]);
        assert!((x[5] - 1.0).abs() < 1e-5, "x[1,0,1]={}", x[5]);
    }
}
