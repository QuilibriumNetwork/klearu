use super::linear::Linear;

/// SwiGLU MLP (LLaMA-style).
///
/// `output = down_proj(silu(gate_proj(x)) * up_proj(x))`
pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    intermediate_size: usize,
}

impl Mlp {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::new(hidden_size, intermediate_size),
            up_proj: Linear::new(hidden_size, intermediate_size),
            down_proj: Linear::new(intermediate_size, hidden_size),
            intermediate_size,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut gate = vec![0.0f32; self.intermediate_size];
        let mut up = vec![0.0f32; self.intermediate_size];

        if self.intermediate_size >= 512 {
            rayon::join(
                || self.gate_proj.forward(input, &mut gate),
                || self.up_proj.forward(input, &mut up),
            );
        } else {
            self.gate_proj.forward(input, &mut gate);
            self.up_proj.forward(input, &mut up);
        }

        // SiLU(gate) * up
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g = silu(*g) * u;
        }

        // Down projection
        let hidden_size = self.down_proj.out_features();
        let mut output = vec![0.0f32; hidden_size];
        self.down_proj.forward(&gate, &mut output);

        output
    }
}

/// SiLU (Sigmoid Linear Unit): `x * sigmoid(x)`
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_known_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // silu(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ≈ 0.7311
        assert!((silu(1.0) - 0.7311).abs() < 0.001);
        // silu(-1) = -1 * sigmoid(-1) = -1 / (1 + e) ≈ -0.2689
        assert!((silu(-1.0) - (-0.2689)).abs() < 0.001);
    }

    #[test]
    fn test_mlp_output_shape() {
        let mlp = Mlp::new(8, 16);
        let input = vec![0.1; 8];
        let output = mlp.forward(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_mlp_output_finite() {
        let mlp = Mlp::new(8, 16);
        let input = vec![1.0; 8];
        let output = mlp.forward(&input);
        assert!(output.iter().all(|x| x.is_finite()));
    }
}
