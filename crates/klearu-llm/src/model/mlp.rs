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
        let hidden_size = self.down_proj.out_features();
        let mut output = vec![0.0f32; hidden_size];
        let mut gate = vec![0.0f32; self.intermediate_size];
        let mut up = vec![0.0f32; self.intermediate_size];
        self.forward_into(input, &mut output, &mut gate, &mut up);
        output
    }

    /// Forward pass writing into pre-allocated buffers.
    /// `gate_buf` and `up_buf` must each be >= `intermediate_size`.
    pub fn forward_into(
        &self,
        input: &[f32],
        output: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
    ) {
        self.gate_proj.forward(input, gate_buf);
        self.up_proj.forward(input, up_buf);

        // SiLU(gate) * up
        for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()).take(self.intermediate_size) {
            *g = silu(*g) * u;
        }

        // Down projection
        output.iter_mut().for_each(|v| *v = 0.0);
        self.down_proj.forward(&gate_buf[..self.intermediate_size], output);
    }

    /// Forward pass that additionally copies the pre-SiLU gate projection
    /// (i.e. `gate_proj(input)`) into `gate_preact_out` before the in-place
    /// SiLU multiply overwrites it.
    ///
    /// Functionally equivalent to `forward_into` for the main output; the
    /// capture is a side-channel used by research tooling for
    /// polytope-boundary / stratification analyses. `gate_preact_out` must
    /// have length >= `intermediate_size`.
    pub fn forward_into_capture_gate(
        &self,
        input: &[f32],
        output: &mut [f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        gate_preact_out: &mut [f32],
    ) {
        self.gate_proj.forward(input, gate_buf);
        self.up_proj.forward(input, up_buf);

        gate_preact_out[..self.intermediate_size]
            .copy_from_slice(&gate_buf[..self.intermediate_size]);

        for (g, u) in gate_buf.iter_mut().zip(up_buf.iter()).take(self.intermediate_size) {
            *g = silu(*g) * u;
        }

        output.iter_mut().for_each(|v| *v = 0.0);
        self.down_proj.forward(&gate_buf[..self.intermediate_size], output);
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
