use crate::config::{LayerType, LlmConfig};

use super::attention::Attention;
use super::gated_deltanet::GatedDeltaNet;
use super::mlp::Mlp;
use super::rms_norm::RmsNorm;

/// The attention component of a transformer block.
pub enum AttentionLayer {
    /// Standard full attention (LLaMA or Qwen3.5 gated full attention).
    Standard(Attention),
    /// GatedDeltaNet linear attention (Qwen3.5).
    GatedDeltaNet(GatedDeltaNet),
}

/// A single transformer block: pre-norm -> attention -> residual -> pre-norm -> MLP -> residual.
pub struct TransformerBlock {
    pub attn_norm: RmsNorm,
    pub attention: AttentionLayer,
    pub mlp_norm: RmsNorm,
    pub mlp: Mlp,
}

impl TransformerBlock {
    pub fn new(config: &LlmConfig) -> Self {
        Self::new_for_layer(config, 0)
    }

    pub fn new_for_layer(config: &LlmConfig, layer_idx: usize) -> Self {
        let layer_type = config.layer_type(layer_idx);
        let use_one_plus_weight = config.is_qwen35();

        let attn_norm = if use_one_plus_weight {
            RmsNorm::new_one_plus(config.hidden_size, config.rms_norm_eps)
        } else {
            RmsNorm::new(config.hidden_size, config.rms_norm_eps)
        };

        let mlp_norm = if use_one_plus_weight {
            RmsNorm::new_one_plus(config.hidden_size, config.rms_norm_eps)
        } else {
            RmsNorm::new(config.hidden_size, config.rms_norm_eps)
        };

        let attention = match layer_type {
            LayerType::LinearAttention => {
                AttentionLayer::GatedDeltaNet(GatedDeltaNet::new(
                    config.hidden_size,
                    config.linear_num_key_heads,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    config.linear_conv_kernel_dim,
                    config.rms_norm_eps,
                ))
            }
            LayerType::FullAttention => {
                if config.is_qwen35() && config.attn_output_gate {
                    AttentionLayer::Standard(Attention::new_gated(
                        config.hidden_size,
                        config.num_heads,
                        config.num_kv_heads,
                        config.head_dim,
                        config.rms_norm_eps,
                    ))
                } else {
                    AttentionLayer::Standard(Attention::new(
                        config.hidden_size,
                        config.num_heads,
                        config.num_kv_heads,
                        config.head_dim,
                    ))
                }
            }
        };

        Self {
            attn_norm,
            attention,
            mlp_norm,
            mlp: Mlp::new(config.hidden_size, config.intermediate_size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let config = LlmConfig {
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            ..LlmConfig::default()
        };
        let block = TransformerBlock::new(&config);
        match &block.attention {
            AttentionLayer::Standard(attn) => {
                assert_eq!(attn.q_proj.in_features(), 32);
                assert_eq!(attn.q_proj.out_features(), 32); // 4 heads * 8 dim
            }
            AttentionLayer::GatedDeltaNet(_) => panic!("Expected Standard attention"),
        }
        assert_eq!(block.mlp.gate_proj.in_features(), 32);
        assert_eq!(block.mlp.gate_proj.out_features(), 64);
    }
}
