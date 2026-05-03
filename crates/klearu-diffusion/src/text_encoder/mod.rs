//! CLIP text encoder for SD's prompt → embedding path.
//!
//! Pre-norm Transformer with LayerNorm + multi-head self-attention +
//! quick_gelu MLP. Learned absolute position embeddings (77 max).
//! Causal mask during attention so each token only sees prior tokens
//! (this is how OpenAI's CLIP was trained).
//!
//! Same architecture serves CLIP-L (768d/12 layers) and CLIP-G
//! (1280d/32 layers); only the config dimensions differ.

use crate::config::CLIPTextConfig;
use crate::error::{DiffusionError, Result};
use crate::layers::{Attention, LayerNorm, Linear, gelu_inplace, quick_gelu_inplace};
use crate::weight::{ComponentTensors, load_embedding, load_layer_norm, load_linear};

/// CLIP MLP activation. CLIP-L uses `quick_gelu` (OpenAI's Hendrycks-Gimpel
/// approximation); CLIP-G (OpenCLIP) uses true `gelu`.
#[derive(Clone, Copy, Debug)]
pub enum CLIPActivation {
    QuickGelu,
    Gelu,
}

impl CLIPActivation {
    pub fn from_name(s: &str) -> Self {
        match s {
            "gelu" | "GELU" | "gelu_pytorch_tanh" => CLIPActivation::Gelu,
            // Default and "quick_gelu" → QuickGelu
            _ => CLIPActivation::QuickGelu,
        }
    }
    fn apply(self, x: &mut [f32]) {
        match self {
            CLIPActivation::QuickGelu => quick_gelu_inplace(x),
            CLIPActivation::Gelu => gelu_inplace(x),
        }
    }
}

pub struct CLIPMLP {
    pub fc1: Linear,
    pub fc2: Linear,
    pub activation: CLIPActivation,
}

impl CLIPMLP {
    pub fn new(hidden: usize, intermediate: usize, activation: CLIPActivation) -> Self {
        Self {
            fc1: Linear::new(hidden, intermediate, true),
            fc2: Linear::new(intermediate, hidden, true),
            activation,
        }
    }
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() / self.fc1.in_features;
        let mut h = vec![0.0f32; n * self.fc1.out_features];
        self.fc1.forward_batch(x, &mut h);
        self.activation.apply(&mut h);
        let mut out = vec![0.0f32; n * self.fc2.out_features];
        self.fc2.forward_batch(&h, &mut out);
        out
    }
}

pub struct CLIPEncoderLayer {
    pub layer_norm1: LayerNorm,
    pub self_attn: Attention,
    pub layer_norm2: LayerNorm,
    pub mlp: CLIPMLP,
}

impl CLIPEncoderLayer {
    pub fn new(
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        intermediate: usize,
        eps: f32,
        activation: CLIPActivation,
    ) -> Self {
        Self {
            layer_norm1: LayerNorm::new(hidden, eps),
            // OpenAI CLIP has biases on q_proj/k_proj/v_proj (unlike SD's UNet
            // attention). Without qkv_bias=true, load_linear silently drops the
            // bias values from the safetensors, producing wildly off CLIP
            // embeddings (~cos_sim 0.4 vs diffusers' reference).
            self_attn: Attention::new_with_qkv_bias(
                hidden, hidden, num_heads, head_dim, true, true,
            ),
            layer_norm2: LayerNorm::new(hidden, eps),
            mlp: CLIPMLP::new(hidden, intermediate, activation),
        }
    }

    /// Forward over [n × seq × hidden].
    pub fn forward(&self, x: &mut Vec<f32>, n: usize, seq: usize) {
        // pre-norm + self-attn (causal)
        let mut x_norm = x.clone();
        self.layer_norm1.forward_inplace(&mut x_norm);
        let mut sa_out = vec![0.0f32; x.len()];
        self.self_attn.forward(&x_norm, &x_norm, n, seq, seq, /*causal=*/ true, &mut sa_out);
        for i in 0..x.len() { x[i] += sa_out[i]; }

        // pre-norm + mlp
        let mut x_norm = x.clone();
        self.layer_norm2.forward_inplace(&mut x_norm);
        let mlp_out = self.mlp.forward(&x_norm);
        for i in 0..x.len() { x[i] += mlp_out[i]; }
    }
}

pub struct CLIPTextModel {
    pub config: CLIPTextConfig,
    pub token_embedding: Vec<f32>,    // [vocab_size × hidden_size]
    pub position_embedding: Vec<f32>, // [max_position × hidden_size]
    pub layers: Vec<CLIPEncoderLayer>,
    pub final_layer_norm: LayerNorm,
    /// SDXL's CLIP-G has a learned text_projection that produces the pooled
    /// embedding consumed by the SDXL UNet. CLIP-L for SD 1.5 doesn't use it.
    pub text_projection: Option<Linear>,
}

impl CLIPTextModel {
    pub fn from_config(config: CLIPTextConfig) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let activation = CLIPActivation::from_name(&config.hidden_act);
        let layers = (0..config.num_hidden_layers)
            .map(|_| CLIPEncoderLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                head_dim,
                config.intermediate_size,
                config.layer_norm_eps,
                activation,
            ))
            .collect();
        Self {
            token_embedding: vec![0.0; config.vocab_size * config.hidden_size],
            position_embedding: vec![0.0; config.max_position_embeddings * config.hidden_size],
            layers,
            final_layer_norm: LayerNorm::new(config.hidden_size, config.layer_norm_eps),
            text_projection: None,
            config,
        }
    }

    /// Encode tokenized prompt → final hidden states [seq_len × hidden_size]
    /// (after `final_layer_norm`). Used for SD 1.5 cross-attention and for
    /// the SDXL pooled-projection path.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        self.forward_layers(input_ids, None)
    }

    /// Generalised encode: returns the hidden state at `return_layer_index`
    /// (0-indexed; `Some(n_layers - 2)` is the penultimate hidden used by
    /// SDXL for cross-attention; `None` means apply all layers AND
    /// `final_layer_norm`, matching the original `forward` semantics).
    ///
    /// SDXL convention: cross-attn embedding = hidden_states[-2] = the
    /// pre-final-norm hidden after the last-but-one encoder layer. The
    /// final layer-norm is skipped for that path.
    pub fn forward_layers(
        &self,
        input_ids: &[u32],
        return_layer_index: Option<usize>,
    ) -> Result<Vec<f32>> {
        let seq = input_ids.len();
        if seq > self.config.max_position_embeddings {
            return Err(DiffusionError::Unsupported(
                format!("seq_len {seq} > max_position {}", self.config.max_position_embeddings),
            ));
        }
        let h = self.config.hidden_size;

        // Embed tokens + add positional embeddings.
        let mut x = vec![0.0f32; seq * h];
        for (i, &tok) in input_ids.iter().enumerate() {
            let tok = tok as usize;
            if tok >= self.config.vocab_size {
                return Err(DiffusionError::Unsupported(
                    format!("token id {tok} >= vocab_size {}", self.config.vocab_size),
                ));
            }
            for j in 0..h {
                x[i * h + j] = self.token_embedding[tok * h + j]
                             + self.position_embedding[i * h + j];
            }
        }

        // Encoder stack. Stop at return_layer_index (inclusive) if given.
        let stop_at = return_layer_index.unwrap_or(self.layers.len() - 1);
        let stop_at = stop_at.min(self.layers.len() - 1);
        for layer in self.layers.iter().take(stop_at + 1) {
            layer.forward(&mut x, /*n=*/ 1, seq);
        }

        // Apply final_layer_norm only when caller wants the FULL forward.
        if return_layer_index.is_none() {
            self.final_layer_norm.forward_inplace(&mut x);
        }

        Ok(x)
    }

    /// Load weights from an opened CLIP component (text_encoder/ or text_encoder_2/).
    /// Tensor naming follows HF transformers CLIPTextModel convention.
    pub fn load_from(&mut self, comp: &ComponentTensors) -> Result<()> {
        // Embeddings
        load_embedding(comp, "text_model.embeddings.token_embedding.weight",
            &mut self.token_embedding)?;
        load_embedding(comp, "text_model.embeddings.position_embedding.weight",
            &mut self.position_embedding)?;
        // Encoder layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("text_model.encoder.layers.{i}");
            load_layer_norm(comp, &format!("{p}.layer_norm1"), &mut layer.layer_norm1)?;
            load_linear(comp, &format!("{p}.self_attn.q_proj"), &mut layer.self_attn.to_q)?;
            load_linear(comp, &format!("{p}.self_attn.k_proj"), &mut layer.self_attn.to_k)?;
            load_linear(comp, &format!("{p}.self_attn.v_proj"), &mut layer.self_attn.to_v)?;
            load_linear(comp, &format!("{p}.self_attn.out_proj"), &mut layer.self_attn.to_out)?;
            load_layer_norm(comp, &format!("{p}.layer_norm2"), &mut layer.layer_norm2)?;
            load_linear(comp, &format!("{p}.mlp.fc1"), &mut layer.mlp.fc1)?;
            load_linear(comp, &format!("{p}.mlp.fc2"), &mut layer.mlp.fc2)?;
        }
        load_layer_norm(comp, "text_model.final_layer_norm", &mut self.final_layer_norm)?;
        // CLIP-G has a learned projection used for the pooled output.
        if comp.has("text_projection.weight") {
            let mut proj = Linear::new(self.config.hidden_size, self.config.hidden_size, false);
            load_linear(comp, "text_projection", &mut proj)?;
            self.text_projection = Some(proj);
        }
        Ok(())
    }

    /// CLIP-G's pooled embedding: take the hidden state at the EOS position,
    /// project through `text_projection`. SDXL feeds this into TextTimeEmbedding.
    /// `eos_token_position` is the index of the EOS token in `input_ids`.
    pub fn pooled_forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let h_states = self.forward(input_ids)?;
        let eos_pos = input_ids.iter()
            .position(|&t| t == 49407 /* CLIP <|endoftext|> */)
            .unwrap_or(input_ids.len() - 1);
        let h = self.config.hidden_size;
        let pooled = &h_states[eos_pos * h..(eos_pos + 1) * h];
        let proj = match &self.text_projection {
            Some(p) => {
                let mut out = vec![0.0f32; p.out_features];
                p.forward(pooled, &mut out);
                out
            }
            None => pooled.to_vec(),
        };
        Ok(proj)
    }
}

/// SDXL dual encoder: CLIP-L (text_encoder) + CLIP-G (text_encoder_2).
/// Returns concatenated hidden states + CLIP-G pooled embedding.
pub struct SDXLDualTextEncoder {
    pub clip_l: CLIPTextModel,
    pub clip_g: CLIPTextModel,
}

impl SDXLDualTextEncoder {
    pub fn from_configs(clip_l: CLIPTextConfig, clip_g: CLIPTextConfig) -> Self {
        Self {
            clip_l: CLIPTextModel::from_config(clip_l),
            clip_g: CLIPTextModel::from_config(clip_g),
        }
    }

    /// Returns (concat_hidden [seq × 2048], pooled_g [1280]).
    ///
    /// SDXL convention: cross-attention conditioning uses the *penultimate*
    /// hidden state (pre-final-norm) of both CLIP-L and CLIP-G. The pooled
    /// embedding for time-text conditioning uses the *final* hidden of
    /// CLIP-G at the EOS position, projected through `text_projection`.
    pub fn forward(&self, ids_l: &[u32], ids_g: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        // Penultimate-layer hidden states for cross-attention.
        let penult_l = self.clip_l.config.num_hidden_layers.saturating_sub(2);
        let penult_g = self.clip_g.config.num_hidden_layers.saturating_sub(2);
        let h_l = self.clip_l.forward_layers(ids_l, Some(penult_l))?;
        let h_g = self.clip_g.forward_layers(ids_g, Some(penult_g))?;
        // Pooled CLIP-G uses the FULL forward (with final_layer_norm) and
        // a learned text_projection. pooled_forward already does this.
        let pooled_g = self.clip_g.pooled_forward(ids_g)?;

        let dl = self.clip_l.config.hidden_size;
        let dg = self.clip_g.config.hidden_size;
        let nl = h_l.len() / dl;
        let ng = h_g.len() / dg;
        if nl != ng {
            return Err(DiffusionError::ShapeMismatch {
                expected: format!("CLIP-L tokens == CLIP-G tokens"),
                got: format!("L={nl}, G={ng}"),
            });
        }
        let total = dl + dg;
        let mut out = vec![0.0f32; nl * total];
        for i in 0..nl {
            out[i * total..i * total + dl].copy_from_slice(&h_l[i * dl..(i + 1) * dl]);
            out[i * total + dl..i * total + total].copy_from_slice(&h_g[i * dg..(i + 1) * dg]);
        }
        Ok((out, pooled_g))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> CLIPTextConfig {
        CLIPTextConfig {
            vocab_size: 100,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            max_position_embeddings: 16,
            layer_norm_eps: 1e-5,
            hidden_act: "quick_gelu".into(),
        }
    }

    #[test]
    fn clip_forward_correct_shape() {
        let model = CLIPTextModel::from_config(tiny_config());
        let ids = vec![1u32, 2, 3, 4];
        let out = model.forward(&ids).unwrap();
        assert_eq!(out.len(), 4 * 16);
    }

    #[test]
    fn clip_dual_concat() {
        let mut cfg_l = tiny_config(); cfg_l.hidden_size = 8;
        let mut cfg_g = tiny_config(); cfg_g.hidden_size = 16;
        let dual = SDXLDualTextEncoder::from_configs(cfg_l, cfg_g);
        let ids = vec![1u32, 2, 3, 4];
        let (concat, _) = dual.forward(&ids, &ids).unwrap();
        // 4 tokens × (8+16) = 96
        assert_eq!(concat.len(), 4 * 24);
    }
}
