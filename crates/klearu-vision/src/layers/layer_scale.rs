/// Per-channel learnable scaling (used in ConvNeXt).
///
/// Forward: `output[i] = gamma[i] * input[i]`.

pub struct LayerScale {
    pub gamma: Vec<f32>,
}

impl LayerScale {
    pub fn new(dim: usize, init_value: f32) -> Self {
        Self {
            gamma: vec![init_value; dim],
        }
    }

    pub fn dim(&self) -> usize {
        self.gamma.len()
    }

    /// In-place forward: `x[i] *= gamma[i]`.
    pub fn forward(&self, x: &mut [f32]) {
        for (v, &g) in x.iter_mut().zip(self.gamma.iter()) {
            *v *= g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_scale() {
        let ls = LayerScale::new(4, 0.5);
        let mut x = vec![2.0, 4.0, 6.0, 8.0];
        ls.forward(&mut x);
        assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
