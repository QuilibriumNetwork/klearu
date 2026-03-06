/// SwiGLU MLP layer used in EVA-02.
///
/// Forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`.

use crate::layers::LinearBias;

pub struct SwiGluMlp {
    pub gate_proj: LinearBias,
    pub up_proj: LinearBias,
    pub down_proj: LinearBias,
}

impl SwiGluMlp {
    pub fn new(in_dim: usize, hidden_dim: usize) -> Self {
        Self {
            gate_proj: LinearBias::new(in_dim, hidden_dim),
            up_proj: LinearBias::new(in_dim, hidden_dim),
            down_proj: LinearBias::new(hidden_dim, in_dim),
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        let hidden = self.gate_proj.out_features();
        let mut gate = vec![0.0f32; hidden];
        let mut up = vec![0.0f32; hidden];

        self.gate_proj.forward(input, &mut gate);
        self.up_proj.forward(input, &mut up);

        // SiLU(gate) * up
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g = *g * (1.0 / (1.0 + (-*g).exp())) * u;
        }

        self.down_proj.forward(&gate, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_mlp() {
        let mlp = SwiGluMlp::new(8, 16);
        let input = vec![0.1f32; 8];
        let mut output = vec![0.0f32; 8];
        mlp.forward(&input, &mut output);
        for &v in &output {
            assert!(v.is_finite());
        }
    }
}
