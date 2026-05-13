//! AdamW optimizer for [`ImageTransformer`].
//!
//! Standard Adam with decoupled weight decay (Loshchilov & Hutter, 2019):
//!
//!   m_t = β1·m_{t-1} + (1-β1)·grad
//!   v_t = β2·v_{t-1} + (1-β2)·grad²
//!   m̂   = m_t / (1 - β1^t)
//!   v̂   = v_t / (1 - β2^t)
//!   θ   = θ - lr·(m̂/(√v̂+ε) + wd·θ)
//!
//! Memory: two extra param-sized buffers (m, v). On CPU with ~50M params
//! and f32, that's ~400 MB of optimizer state — tractable on a 64 GB box.
//!
//! [`ImageTransformer`]: crate::model::ImageTransformer

use crate::grad::Gradients;
use crate::model::ImageTransformer;

#[derive(Debug, Clone)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Decoupled weight decay coefficient. Applied to non-norm-gamma
    /// params (norm gammas are NOT decayed — standard practice).
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self { lr: 3e-4, beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay: 0.1 }
    }
}

/// AdamW optimizer state. Holds first- and second-moment buffers
/// shape-identical to the model's params, plus a step counter for
/// bias correction.
pub struct AdamW {
    pub config: AdamWConfig,
    pub m: Gradients,
    pub v: Gradients,
    pub step: u64,
}

impl AdamW {
    pub fn new(model: &ImageTransformer, config: AdamWConfig) -> Self {
        Self {
            config,
            m: Gradients::zeros_for(model),
            v: Gradients::zeros_for(model),
            step: 0,
        }
    }

    /// Apply one optimizer step to `model` using gradients in `grad`.
    /// Weight decay is applied to Linear weights and the embedding /
    /// LM-head tables; RMSNorm gammas and the positional embedding are
    /// NOT decayed (standard convention).
    pub fn step(&mut self, model: &mut ImageTransformer, grad: &Gradients) {
        self.step += 1;
        let cfg = self.config.clone();
        let bc1 = 1.0 - cfg.beta1.powi(self.step as i32);
        let bc2 = 1.0 - cfg.beta2.powi(self.step as i32);
        let lr_t = cfg.lr * bc2.sqrt() / bc1;

        update_buf(&mut model.embed, &grad.embed,
            &mut self.m.embed, &mut self.v.embed, &cfg, lr_t, true);
        update_buf(&mut model.pos_embed, &grad.pos_embed,
            &mut self.m.pos_embed, &mut self.v.pos_embed, &cfg, lr_t, false);

        for i in 0..model.blocks.len() {
            let b = &mut model.blocks[i];
            let bg = &grad.blocks[i];
            let mb = &mut self.m.blocks[i];
            let vb = &mut self.v.blocks[i];
            update_buf(&mut b.norm_attn.gamma, &bg.norm_attn_gamma,
                &mut mb.norm_attn_gamma, &mut vb.norm_attn_gamma, &cfg, lr_t, false);
            update_buf(&mut b.q_proj.weight, &bg.q_proj_w,
                &mut mb.q_proj_w, &mut vb.q_proj_w, &cfg, lr_t, true);
            update_buf(&mut b.k_proj.weight, &bg.k_proj_w,
                &mut mb.k_proj_w, &mut vb.k_proj_w, &cfg, lr_t, true);
            update_buf(&mut b.v_proj.weight, &bg.v_proj_w,
                &mut mb.v_proj_w, &mut vb.v_proj_w, &cfg, lr_t, true);
            update_buf(&mut b.o_proj.weight, &bg.o_proj_w,
                &mut mb.o_proj_w, &mut vb.o_proj_w, &cfg, lr_t, true);
            update_buf(&mut b.norm_mlp.gamma, &bg.norm_mlp_gamma,
                &mut mb.norm_mlp_gamma, &mut vb.norm_mlp_gamma, &cfg, lr_t, false);
            update_buf(&mut b.mlp_gate.weight, &bg.mlp_gate_w,
                &mut mb.mlp_gate_w, &mut vb.mlp_gate_w, &cfg, lr_t, true);
            update_buf(&mut b.mlp_up.weight, &bg.mlp_up_w,
                &mut mb.mlp_up_w, &mut vb.mlp_up_w, &cfg, lr_t, true);
            update_buf(&mut b.mlp_down.weight, &bg.mlp_down_w,
                &mut mb.mlp_down_w, &mut vb.mlp_down_w, &cfg, lr_t, true);
        }

        update_buf(&mut model.final_norm.gamma, &grad.final_norm_gamma,
            &mut self.m.final_norm_gamma, &mut self.v.final_norm_gamma, &cfg, lr_t, false);
        update_buf(&mut model.lm_head.weight, &grad.lm_head_w,
            &mut self.m.lm_head_w, &mut self.v.lm_head_w, &cfg, lr_t, true);
    }
}

/// One AdamW update for a single flat parameter buffer.
fn update_buf(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    cfg: &AdamWConfig,
    lr_t: f32,
    decay: bool,
) {
    debug_assert_eq!(param.len(), grad.len());
    debug_assert_eq!(param.len(), m.len());
    debug_assert_eq!(param.len(), v.len());
    let beta1 = cfg.beta1;
    let beta2 = cfg.beta2;
    let eps = cfg.eps;
    let wd = if decay { cfg.weight_decay } else { 0.0 };
    for i in 0..param.len() {
        let g = grad[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        // m and v are already bias-corrected outside via lr_t.
        let update = m[i] / (v[i].sqrt() + eps) + wd * param[i];
        param[i] -= lr_t * update;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImageTransformer, ImageTransformerConfig};

    fn tiny_cfg() -> ImageTransformerConfig {
        ImageTransformerConfig {
            vocab_text: 100, vocab_image: 16,
            hidden_size: 16, num_layers: 1, num_heads: 4,
            mlp_intermediate: 32,
            max_text_len: 8,
            image_grid_h: 2, image_grid_w: 2,
            bos_token: 100, sep_image_token: 101, eos_token: 102,
            rms_norm_eps: 1e-5, rope_theta: 10_000.0,
        }
    }

    #[test]
    fn step_lowers_constant_gradient() {
        // For a constant positive gradient, AdamW should always reduce
        // the param. With weight decay applied, the magnitude doesn't
        // matter for the sign of motion.
        let mut model = ImageTransformer::from_config(tiny_cfg());
        // Seed lm_head.weight to 1.0 so we can see the change.
        for w in model.lm_head.weight.iter_mut() { *w = 1.0; }
        let mut opt = AdamW::new(&model, AdamWConfig::default());
        let mut grad = Gradients::zeros_for(&model);
        for g in grad.lm_head_w.iter_mut() { *g = 0.5; } // positive grad → param decreases

        let before = model.lm_head.weight[0];
        opt.step(&mut model, &grad);
        let after = model.lm_head.weight[0];
        assert!(after < before,
            "AdamW should decrease param with positive grad: before={before} after={after}");
    }

    #[test]
    fn norm_gamma_is_not_decayed() {
        // With zero grad and weight_decay=0.5, a Linear weight should
        // shrink, but a RMSNorm gamma should NOT.
        let mut model = ImageTransformer::from_config(tiny_cfg());
        for w in model.lm_head.weight.iter_mut() { *w = 1.0; }
        for g in model.final_norm.gamma.iter_mut() { *g = 1.0; }
        let mut opt = AdamW::new(&model, AdamWConfig {
            weight_decay: 0.5, lr: 0.1, ..Default::default()
        });
        let grad = Gradients::zeros_for(&model);
        opt.step(&mut model, &grad);
        // lm_head got decayed.
        assert!(model.lm_head.weight[0] < 1.0,
            "lm_head should be decayed: got {}", model.lm_head.weight[0]);
        // Norm gamma should be untouched.
        assert!((model.final_norm.gamma[0] - 1.0).abs() < 1e-7,
            "norm gamma should NOT be decayed: got {}", model.final_norm.gamma[0]);
    }
}
