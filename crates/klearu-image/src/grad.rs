//! Gradient buffers for [`ImageTransformer`]. Shape-identical to the
//! model's param tensors so we can update params in place via the
//! optimizer.
//!
//! [`ImageTransformer`]: crate::model::ImageTransformer

use crate::model::{ImageBlock, ImageTransformer, ImageTransformerConfig};

/// Per-block gradient buffer. Field-for-field mirror of `ImageBlock`.
#[derive(Clone)]
pub struct BlockGrad {
    pub norm_attn_gamma: Vec<f32>,
    pub q_proj_w: Vec<f32>,
    pub k_proj_w: Vec<f32>,
    pub v_proj_w: Vec<f32>,
    pub o_proj_w: Vec<f32>,
    pub norm_mlp_gamma: Vec<f32>,
    pub mlp_gate_w: Vec<f32>,
    pub mlp_up_w: Vec<f32>,
    pub mlp_down_w: Vec<f32>,
}

impl BlockGrad {
    pub fn zeros_for(block: &ImageBlock) -> Self {
        Self {
            norm_attn_gamma: vec![0.0; block.norm_attn.gamma.len()],
            q_proj_w:        vec![0.0; block.q_proj.weight.len()],
            k_proj_w:        vec![0.0; block.k_proj.weight.len()],
            v_proj_w:        vec![0.0; block.v_proj.weight.len()],
            o_proj_w:        vec![0.0; block.o_proj.weight.len()],
            norm_mlp_gamma:  vec![0.0; block.norm_mlp.gamma.len()],
            mlp_gate_w:      vec![0.0; block.mlp_gate.weight.len()],
            mlp_up_w:        vec![0.0; block.mlp_up.weight.len()],
            mlp_down_w:      vec![0.0; block.mlp_down.weight.len()],
        }
    }

    pub fn zero_inplace(&mut self) {
        for v in self.norm_attn_gamma.iter_mut() { *v = 0.0; }
        for v in self.q_proj_w.iter_mut() { *v = 0.0; }
        for v in self.k_proj_w.iter_mut() { *v = 0.0; }
        for v in self.v_proj_w.iter_mut() { *v = 0.0; }
        for v in self.o_proj_w.iter_mut() { *v = 0.0; }
        for v in self.norm_mlp_gamma.iter_mut() { *v = 0.0; }
        for v in self.mlp_gate_w.iter_mut() { *v = 0.0; }
        for v in self.mlp_up_w.iter_mut() { *v = 0.0; }
        for v in self.mlp_down_w.iter_mut() { *v = 0.0; }
    }
}

/// Full model gradient buffer. One per training step (or accumulated
/// across micro-batches).
#[derive(Clone)]
pub struct Gradients {
    pub embed: Vec<f32>,           // [unified_vocab, hidden]
    pub pos_embed: Vec<f32>,       // [max_seq_len, hidden]
    pub blocks: Vec<BlockGrad>,    // length == num_layers
    pub final_norm_gamma: Vec<f32>, // [hidden]
    pub lm_head_w: Vec<f32>,       // [vocab_image, hidden]
}

impl Gradients {
    pub fn zeros_for(model: &ImageTransformer) -> Self {
        Self {
            embed:            vec![0.0; model.embed.len()],
            pos_embed:        vec![0.0; model.pos_embed.len()],
            blocks:           model.blocks.iter().map(BlockGrad::zeros_for).collect(),
            final_norm_gamma: vec![0.0; model.final_norm.gamma.len()],
            lm_head_w:        vec![0.0; model.lm_head.weight.len()],
        }
    }

    pub fn zero_inplace(&mut self) {
        for v in self.embed.iter_mut() { *v = 0.0; }
        for v in self.pos_embed.iter_mut() { *v = 0.0; }
        for b in self.blocks.iter_mut() { b.zero_inplace(); }
        for v in self.final_norm_gamma.iter_mut() { *v = 0.0; }
        for v in self.lm_head_w.iter_mut() { *v = 0.0; }
    }

    pub fn from_config(cfg: &ImageTransformerConfig) -> Self {
        let d = cfg.hidden_size;
        let m = cfg.mlp_intermediate;
        let block = || BlockGrad {
            norm_attn_gamma: vec![0.0; d],
            q_proj_w:        vec![0.0; d * d],
            k_proj_w:        vec![0.0; d * d],
            v_proj_w:        vec![0.0; d * d],
            o_proj_w:        vec![0.0; d * d],
            norm_mlp_gamma:  vec![0.0; d],
            mlp_gate_w:      vec![0.0; d * m],
            mlp_up_w:        vec![0.0; d * m],
            mlp_down_w:      vec![0.0; m * d],
        };
        Self {
            embed: vec![0.0; cfg.unified_vocab_size() * d],
            pos_embed: vec![0.0; cfg.max_seq_len() * d],
            blocks: (0..cfg.num_layers).map(|_| block()).collect(),
            final_norm_gamma: vec![0.0; d],
            lm_head_w: vec![0.0; cfg.vocab_image * d],
        }
    }
}
