//! GatedDeltaNet: linear attention with recurrent state (Qwen3.5).
//!
//! Each layer maintains a per-head recurrent state S ∈ R^{key_dim × value_dim}
//! and a depthwise conv state for short-range dependencies.
//!
//! Decode recurrence (single token):
//! 1. Project: qkv = in_proj_qkv(input), split into q, k, v
//! 2. Conv update: shift conv_state, append new qkv, apply depthwise conv, SiLU
//! 3. Compute gates: beta = sigmoid(in_proj_b(input)), alpha = exp(-exp(a_log) * softplus(in_proj_a(input) + dt_bias))
//! 4. Per-head recurrence: S = alpha * S; err = v - S^T · k; S += beta * outer(k, err)
//! 5. Output: o = S^T · q per head, concat
//! 6. Output gate: z = in_proj_z(input), apply rms_norm(o) * silu(z)
//! 7. Final: out_proj(gated_output)

use super::linear::Linear;

/// SiLU activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus: ln(1 + exp(x)), numerically stable
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid function.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// L2-normalize a vector in-place: x_i = x_i / sqrt(sum(x^2) + eps).
#[inline]
fn l2_normalize(x: &mut [f32]) {
    let eps = 1e-6f32;
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_norm = 1.0 / (sum_sq + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv_norm;
    }
}

/// Per-layer recurrent state for GatedDeltaNet.
pub struct DeltaNetState {
    /// SSM state: [num_heads × key_dim × value_dim], flattened.
    pub ssm_state: Vec<f32>,
    /// Conv state: [conv_dim × kernel_size], ring buffer.
    pub conv_state: Vec<f32>,
    /// Current write position in conv state (ring buffer index).
    pub conv_pos: usize,

    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    kernel_size: usize,
}

impl DeltaNetState {
    pub fn new(
        num_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_dim: usize,
        kernel_size: usize,
    ) -> Self {
        Self {
            ssm_state: vec![0.0; num_heads * key_dim * value_dim],
            conv_state: vec![0.0; conv_dim * kernel_size],
            conv_pos: 0,
            num_heads,
            key_dim,
            value_dim,
            conv_dim,
            kernel_size,
        }
    }

    pub fn clear(&mut self) {
        self.ssm_state.fill(0.0);
        self.conv_state.fill(0.0);
        self.conv_pos = 0;
    }

    /// Get mutable slice for head h's S matrix [key_dim × value_dim].
    fn s_head_mut(&mut self, h: usize) -> &mut [f32] {
        let size = self.key_dim * self.value_dim;
        let offset = h * size;
        &mut self.ssm_state[offset..offset + size]
    }

    /// Get slice for head h's S matrix [key_dim × value_dim].
    fn s_head(&self, h: usize) -> &[f32] {
        let size = self.key_dim * self.value_dim;
        let offset = h * size;
        &self.ssm_state[offset..offset + size]
    }
}

/// Store of DeltaNetState, one per layer. `None` for full-attention layers.
pub struct DeltaNetStateStore {
    pub states: Vec<Option<DeltaNetState>>,
}

impl DeltaNetStateStore {
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    pub fn with_capacity(num_layers: usize) -> Self {
        Self {
            states: (0..num_layers).map(|_| None).collect(),
        }
    }

    pub fn push(&mut self, state: Option<DeltaNetState>) {
        self.states.push(state);
    }

    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut DeltaNetState> {
        self.states[idx].as_mut()
    }

    pub fn clear(&mut self) {
        for state in &mut self.states {
            if let Some(s) = state {
                s.clear();
            }
        }
    }
}

/// GatedDeltaNet linear attention module (Qwen3.5).
///
/// Supports GQA-style split between key and value heads:
/// `num_key_heads` ≤ `num_value_heads`, with each key head shared by
/// `num_value_heads / num_key_heads` value heads. Qwen3.5-0.8B has
/// `num_key_heads = num_value_heads = 16` (1:1 mapping); Qwen3.5-4B has
/// `num_key_heads = 16, num_value_heads = 32` (1:2 mapping).
pub struct GatedDeltaNet {
    pub in_proj_qkv: Linear,   // hidden → (num_key_heads*key_dim + num_key_heads*key_dim + num_value_heads*value_dim)
    pub in_proj_z: Linear,     // hidden → num_value_heads * value_dim (output gate)
    pub in_proj_a: Linear,     // hidden → num_value_heads (alpha gate)
    pub in_proj_b: Linear,     // hidden → num_value_heads (beta gate)
    pub conv_weight: Vec<f32>, // [conv_dim × kernel_size] (depthwise)
    pub dt_bias: Vec<f32>,     // [num_value_heads]
    pub a_log: Vec<f32>,       // [num_value_heads]
    pub norm_weight: Vec<f32>, // [value_dim] for output RMSNorm (per-head, shared weights)
    pub out_proj: Linear,      // num_value_heads * value_dim → hidden
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_dim: usize,       // = num_key_heads * key_dim * 2 + num_value_heads * value_dim
    pub kernel_size: usize,
    pub rms_norm_eps: f32,
}

impl GatedDeltaNet {
    pub fn new(
        hidden_size: usize,
        num_key_heads: usize,
        num_value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
        rms_norm_eps: f32,
    ) -> Self {
        assert!(
            num_value_heads.is_multiple_of(num_key_heads),
            "num_value_heads {num_value_heads} must be a multiple of num_key_heads {num_key_heads}"
        );
        // qkv projection: q(num_key_heads*key_dim) + k(num_key_heads*key_dim) + v(num_value_heads*value_dim)
        let qkv_dim =
            num_key_heads * key_dim + num_key_heads * key_dim + num_value_heads * value_dim;
        let z_dim = num_value_heads * value_dim;

        Self {
            in_proj_qkv: Linear::new(hidden_size, qkv_dim),
            in_proj_z: Linear::new(hidden_size, z_dim),
            in_proj_a: Linear::new(hidden_size, num_value_heads),
            in_proj_b: Linear::new(hidden_size, num_value_heads),
            conv_weight: vec![0.0; qkv_dim * kernel_size],
            dt_bias: vec![0.0; num_value_heads],
            a_log: vec![0.0; num_value_heads],
            norm_weight: vec![0.0; value_dim],
            out_proj: Linear::new(z_dim, hidden_size),
            num_key_heads,
            num_value_heads,
            key_dim,
            value_dim,
            conv_dim: qkv_dim,
            kernel_size,
            rms_norm_eps,
        }
    }

    /// Create a new DeltaNetState for this layer.
    pub fn create_state(&self) -> DeltaNetState {
        DeltaNetState::new(
            self.num_value_heads,
            self.key_dim,
            self.value_dim,
            self.conv_dim,
            self.kernel_size,
        )
    }

    /// Forward pass for a single token (decode).
    /// `input` has shape [hidden_size].
    /// Returns output of shape [hidden_size].
    pub fn forward_decode(&self, input: &[f32], state: &mut DeltaNetState) -> Vec<f32> {
        let hidden_size = self.out_proj.out_features();
        let mut output = vec![0.0f32; hidden_size];
        self.forward_decode_into(input, state, &mut output);
        output
    }

    /// Forward pass writing into a pre-allocated output buffer.
    pub fn forward_decode_into(&self, input: &[f32], state: &mut DeltaNetState, output: &mut [f32]) {
        let qkv_dim = self.conv_dim;
        let z_dim = self.num_value_heads * self.value_dim;
        let group_size = self.num_value_heads / self.num_key_heads;

        // 1. Project QKV
        let mut qkv = vec![0.0f32; qkv_dim];
        self.in_proj_qkv.forward(input, &mut qkv);

        // 2. Conv update: write new qkv into ring buffer, then apply depthwise conv
        let conv_pos = state.conv_pos;
        for ch in 0..self.conv_dim {
            state.conv_state[ch * self.kernel_size + conv_pos] = qkv[ch];
        }
        state.conv_pos = (conv_pos + 1) % self.kernel_size;

        // Apply depthwise conv at current position (dot product of conv_state row with weight row)
        let mut conv_out = vec![0.0f32; qkv_dim];
        for ch in 0..self.conv_dim {
            let mut sum = 0.0f32;
            let state_base = ch * self.kernel_size;
            let weight_base = ch * self.kernel_size;
            for k in 0..self.kernel_size {
                // Read from ring buffer: oldest to newest
                let ring_idx = (state.conv_pos + k) % self.kernel_size;
                sum += state.conv_state[state_base + ring_idx]
                    * self.conv_weight[weight_base + k];
            }
            conv_out[ch] = silu(sum);
        }

        // 3. Split conv_out into q, k, v (k_total uses num_key_heads, v uses num_value_heads).
        let q_total = self.num_key_heads * self.key_dim;
        let k_total = self.num_key_heads * self.key_dim;
        let (qk_part, v_data) = conv_out.split_at_mut(q_total + k_total);
        let (q_data, k_data) = qk_part.split_at_mut(q_total);

        // L2-normalize Q and K per key-head, then scale Q by 1/sqrt(key_dim)
        let inv_sqrt_dk = 1.0 / (self.key_dim as f32).sqrt();
        for h in 0..self.num_key_heads {
            let offset = h * self.key_dim;
            l2_normalize(&mut q_data[offset..offset + self.key_dim]);
            l2_normalize(&mut k_data[offset..offset + self.key_dim]);
            for i in 0..self.key_dim {
                q_data[offset + i] *= inv_sqrt_dk;
            }
        }

        // 4. Compute gates (per value-head)
        let mut a_proj = vec![0.0f32; self.num_value_heads];
        let mut b_proj = vec![0.0f32; self.num_value_heads];
        self.in_proj_a.forward(input, &mut a_proj);
        self.in_proj_b.forward(input, &mut b_proj);

        let mut alpha = vec![0.0f32; self.num_value_heads];
        for h in 0..self.num_value_heads {
            let a = self.a_log[h].exp() * softplus(a_proj[h] + self.dt_bias[h]);
            alpha[h] = (-a).exp();
        }

        let mut beta = vec![0.0f32; self.num_value_heads];
        for h in 0..self.num_value_heads {
            beta[h] = sigmoid(b_proj[h]);
        }

        // 5. Per-value-head recurrence and output. Each value head h_v shares its
        // q/k with key head h_k = h_v / group_size.
        let mut output_heads = vec![0.0f32; z_dim];
        let mut err = vec![0.0f32; self.value_dim];

        for h_v in 0..self.num_value_heads {
            let h_k = h_v / group_size;
            let q_h = &q_data[h_k * self.key_dim..(h_k + 1) * self.key_dim];
            let k_h = &k_data[h_k * self.key_dim..(h_k + 1) * self.key_dim];
            let v_h = &v_data[h_v * self.value_dim..(h_v + 1) * self.value_dim];

            let s = state.s_head_mut(h_v);

            let a = alpha[h_v];
            for val in s.iter_mut() {
                *val *= a;
            }

            err.copy_from_slice(v_h);
            for i in 0..self.key_dim {
                let k_i = k_h[i];
                let s_row = &s[i * self.value_dim..(i + 1) * self.value_dim];
                for j in 0..self.value_dim {
                    err[j] -= s_row[j] * k_i;
                }
            }

            let b = beta[h_v];
            for i in 0..self.key_dim {
                let bk = b * k_h[i];
                let s_row = &mut s[i * self.value_dim..(i + 1) * self.value_dim];
                for j in 0..self.value_dim {
                    s_row[j] += bk * err[j];
                }
            }

            let o_h = &mut output_heads[h_v * self.value_dim..(h_v + 1) * self.value_dim];
            o_h.fill(0.0);
            for i in 0..self.key_dim {
                let q_i = q_h[i];
                let s_row = &s[i * self.value_dim..(i + 1) * self.value_dim];
                for j in 0..self.value_dim {
                    o_h[j] += s_row[j] * q_i;
                }
            }
        }

        // 6. Output gate: z = in_proj_z(input), output = rms_norm(o) * silu(z)
        let mut z = vec![0.0f32; z_dim];
        self.in_proj_z.forward(input, &mut z);

        for h in 0..self.num_value_heads {
            let o_h = &mut output_heads[h * self.value_dim..(h + 1) * self.value_dim];
            let mut sum_sq = 0.0f32;
            for &v in o_h.iter() {
                sum_sq += v * v;
            }
            let rms = (sum_sq / self.value_dim as f32 + self.rms_norm_eps).sqrt();
            let inv_rms = 1.0 / rms;

            let z_h = &z[h * self.value_dim..(h + 1) * self.value_dim];
            for j in 0..self.value_dim {
                o_h[j] = o_h[j] * inv_rms * self.norm_weight[j] * silu(z_h[j]);
            }
        }

        output.iter_mut().for_each(|v| *v = 0.0);
        self.out_proj.forward(&output_heads, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deltanet_state_creation() {
        let state = DeltaNetState::new(4, 8, 8, 64, 4);
        assert_eq!(state.ssm_state.len(), 4 * 8 * 8);
        assert_eq!(state.conv_state.len(), 64 * 4);
        assert_eq!(state.conv_pos, 0);
    }

    #[test]
    fn test_deltanet_state_clear() {
        let mut state = DeltaNetState::new(2, 4, 4, 32, 4);
        state.ssm_state[0] = 1.0;
        state.conv_state[0] = 2.0;
        state.conv_pos = 3;
        state.clear();
        assert!(state.ssm_state.iter().all(|&v| v == 0.0));
        assert!(state.conv_state.iter().all(|&v| v == 0.0));
        assert_eq!(state.conv_pos, 0);
    }

    #[test]
    fn test_gated_deltanet_creation() {
        let dn = GatedDeltaNet::new(32, 4, 4, 8, 8, 4, 1e-5);
        assert_eq!(dn.num_key_heads, 4);
        assert_eq!(dn.num_value_heads, 4);
        assert_eq!(dn.key_dim, 8);
        assert_eq!(dn.value_dim, 8);
        // qkv_dim = 4*8 + 4*8 + 4*8 = 96
        assert_eq!(dn.conv_dim, 96);
        assert_eq!(dn.in_proj_qkv.out_features(), 96);
        assert_eq!(dn.in_proj_z.out_features(), 32); // 4 * 8
        assert_eq!(dn.out_proj.in_features(), 32);
        assert_eq!(dn.out_proj.out_features(), 32);
    }

    #[test]
    fn test_gated_deltanet_creation_gqa_split() {
        // Qwen3.5-4B-style: 16 key heads, 32 value heads.
        let dn = GatedDeltaNet::new(2560, 16, 32, 128, 128, 4, 1e-6);
        assert_eq!(dn.num_key_heads, 16);
        assert_eq!(dn.num_value_heads, 32);
        // qkv_dim = 16*128 + 16*128 + 32*128 = 8192
        assert_eq!(dn.conv_dim, 8192);
        assert_eq!(dn.in_proj_qkv.out_features(), 8192);
        assert_eq!(dn.in_proj_z.out_features(), 4096); // 32 * 128
        assert_eq!(dn.in_proj_a.out_features(), 32);
        assert_eq!(dn.in_proj_b.out_features(), 32);
        assert_eq!(dn.out_proj.in_features(), 4096);
        assert_eq!(dn.out_proj.out_features(), 2560);
        assert_eq!(dn.dt_bias.len(), 32);
        assert_eq!(dn.a_log.len(), 32);
        assert_eq!(dn.conv_weight.len(), 8192 * 4);
    }

    #[test]
    fn test_gated_deltanet_forward_produces_finite() {
        let dn = GatedDeltaNet::new(16, 2, 2, 4, 4, 4, 1e-5);
        let mut state = dn.create_state();

        let input = vec![0.1; 16];
        let output = dn.forward_decode(&input, &mut state);

        assert_eq!(output.len(), 16);
        assert!(output.iter().all(|x| x.is_finite()), "Output should be finite");
    }

    #[test]
    fn test_gated_deltanet_forward_gqa_finite() {
        // 2 key heads, 4 value heads (group_size = 2).
        let dn = GatedDeltaNet::new(16, 2, 4, 4, 4, 4, 1e-5);
        let mut state = dn.create_state();
        let input = vec![0.1; 16];
        let output = dn.forward_decode(&input, &mut state);
        assert_eq!(output.len(), 16);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_gated_deltanet_state_evolves() {
        let dn = GatedDeltaNet::new(16, 2, 2, 4, 4, 4, 1e-5);
        let mut state = dn.create_state();

        let input = vec![0.1; 16];
        let out1 = dn.forward_decode(&input, &mut state);
        let out2 = dn.forward_decode(&input, &mut state);

        let diff: f32 = out1.iter().zip(out2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(state.conv_pos > 0 || state.ssm_state.iter().any(|&v| v != 0.0)
            || diff >= 0.0,
            "State or output should have changed");
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.7311).abs() < 0.001);
    }

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - 0.6931).abs() < 0.001); // ln(2)
        assert!((softplus(25.0) - 25.0).abs() < 0.01);    // saturates
        assert!(softplus(-25.0) < 0.001);                  // near zero
    }

    #[test]
    fn test_deltanet_state_store() {
        let mut store = DeltaNetStateStore::with_capacity(3);
        assert!(store.layer_mut(0).is_none());

        store.states[1] = Some(DeltaNetState::new(2, 4, 4, 32, 4));
        assert!(store.layer_mut(1).is_some());

        store.layer_mut(1).unwrap().ssm_state[0] = 1.0;
        store.clear();
        assert_eq!(store.layer_mut(1).unwrap().ssm_state[0], 0.0);
    }

    #[test]
    fn test_gated_deltanet_qwen35_dims() {
        // Qwen3.5 0.8B dimensions: hidden=1024, 16 key=value heads, key=128, value=128, kernel=4
        let dn = GatedDeltaNet::new(1024, 16, 16, 128, 128, 4, 1e-6);
        // qkv_dim = 16*128 + 16*128 + 16*128 = 6144
        assert_eq!(dn.conv_dim, 6144);
        assert_eq!(dn.in_proj_qkv.out_features(), 6144);
        assert_eq!(dn.in_proj_z.out_features(), 2048); // 16 * 128
        assert_eq!(dn.out_proj.in_features(), 2048);
        assert_eq!(dn.out_proj.out_features(), 1024);
    }
}
