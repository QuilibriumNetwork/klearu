/// Identifies where a HuggingFace weight tensor should be loaded into the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightTarget {
    Embedding,
    LmHead,
    FinalNorm,
    LayerAttnNorm(usize),
    LayerMlpNorm(usize),
    LayerQProj(usize),
    LayerKProj(usize),
    LayerVProj(usize),
    LayerOProj(usize),
    LayerGateProj(usize),
    LayerUpProj(usize),
    LayerDownProj(usize),

    // Qwen3.5 gated full attention
    LayerQNorm(usize),
    LayerKNorm(usize),

    // Qwen3.5 GatedDeltaNet linear attention
    LayerDeltaNetQKV(usize),
    LayerDeltaNetZ(usize),
    LayerDeltaNetA(usize),
    LayerDeltaNetB(usize),
    LayerDeltaNetConv(usize),
    LayerDeltaNetDtBias(usize),
    LayerDeltaNetALog(usize),
    LayerDeltaNetNorm(usize),
    LayerDeltaNetOutProj(usize),
}

/// Parse a HuggingFace weight name to a WeightTarget.
///
/// Supports both LLaMA-style (`model.layers.{i}.*`) and
/// Qwen3.5-style (`model.language_model.layers.{i}.*`) prefixes.
pub fn parse_weight_name(name: &str) -> Option<WeightTarget> {
    // Try to normalize the prefix. Qwen3.5 uses "model.language_model." while LLaMA uses "model."
    let normalized = if name.starts_with("model.language_model.") {
        name.strip_prefix("model.language_model.").unwrap()
    } else if name.starts_with("model.") {
        name.strip_prefix("model.").unwrap()
    } else {
        name
    };

    // Embedding
    if normalized == "embed_tokens.weight" {
        return Some(WeightTarget::Embedding);
    }

    // LM head
    if name == "lm_head.weight" {
        return Some(WeightTarget::LmHead);
    }

    // Final norm
    if normalized == "norm.weight" {
        return Some(WeightTarget::FinalNorm);
    }

    // layers.{i}.*
    if let Some(rest) = normalized.strip_prefix("layers.") {
        let dot_pos = rest.find('.')?;
        let layer_idx: usize = rest[..dot_pos].parse().ok()?;
        let suffix = &rest[dot_pos + 1..];

        return match suffix {
            // Standard attention projections
            "self_attn.q_proj.weight" => Some(WeightTarget::LayerQProj(layer_idx)),
            "self_attn.k_proj.weight" => Some(WeightTarget::LayerKProj(layer_idx)),
            "self_attn.v_proj.weight" => Some(WeightTarget::LayerVProj(layer_idx)),
            "self_attn.o_proj.weight" => Some(WeightTarget::LayerOProj(layer_idx)),

            // Q/K RMSNorm (Qwen3.5 gated full attention)
            "self_attn.q_norm.weight" => Some(WeightTarget::LayerQNorm(layer_idx)),
            "self_attn.k_norm.weight" => Some(WeightTarget::LayerKNorm(layer_idx)),

            // MLP projections
            "mlp.gate_proj.weight" => Some(WeightTarget::LayerGateProj(layer_idx)),
            "mlp.up_proj.weight" => Some(WeightTarget::LayerUpProj(layer_idx)),
            "mlp.down_proj.weight" => Some(WeightTarget::LayerDownProj(layer_idx)),

            // Layer norms
            "input_layernorm.weight" => Some(WeightTarget::LayerAttnNorm(layer_idx)),
            "post_attention_layernorm.weight" => Some(WeightTarget::LayerMlpNorm(layer_idx)),

            // GatedDeltaNet linear attention (Qwen3.5)
            "linear_attn.in_proj_qkv.weight" => Some(WeightTarget::LayerDeltaNetQKV(layer_idx)),
            "linear_attn.in_proj_z.weight" => Some(WeightTarget::LayerDeltaNetZ(layer_idx)),
            "linear_attn.in_proj_a.weight" => Some(WeightTarget::LayerDeltaNetA(layer_idx)),
            "linear_attn.in_proj_b.weight" => Some(WeightTarget::LayerDeltaNetB(layer_idx)),
            "linear_attn.conv1d.weight" => Some(WeightTarget::LayerDeltaNetConv(layer_idx)),
            "linear_attn.dt_bias" => Some(WeightTarget::LayerDeltaNetDtBias(layer_idx)),
            "linear_attn.A_log" => Some(WeightTarget::LayerDeltaNetALog(layer_idx)),
            "linear_attn.norm.weight" => Some(WeightTarget::LayerDeltaNetNorm(layer_idx)),
            "linear_attn.out_proj.weight" => Some(WeightTarget::LayerDeltaNetOutProj(layer_idx)),

            _ => None,
        };
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding() {
        assert_eq!(
            parse_weight_name("model.embed_tokens.weight"),
            Some(WeightTarget::Embedding)
        );
    }

    #[test]
    fn test_lm_head() {
        assert_eq!(
            parse_weight_name("lm_head.weight"),
            Some(WeightTarget::LmHead)
        );
    }

    #[test]
    fn test_final_norm() {
        assert_eq!(
            parse_weight_name("model.norm.weight"),
            Some(WeightTarget::FinalNorm)
        );
    }

    #[test]
    fn test_layer_attn_projections() {
        assert_eq!(
            parse_weight_name("model.layers.0.self_attn.q_proj.weight"),
            Some(WeightTarget::LayerQProj(0))
        );
        assert_eq!(
            parse_weight_name("model.layers.21.self_attn.k_proj.weight"),
            Some(WeightTarget::LayerKProj(21))
        );
        assert_eq!(
            parse_weight_name("model.layers.5.self_attn.v_proj.weight"),
            Some(WeightTarget::LayerVProj(5))
        );
        assert_eq!(
            parse_weight_name("model.layers.10.self_attn.o_proj.weight"),
            Some(WeightTarget::LayerOProj(10))
        );
    }

    #[test]
    fn test_layer_mlp_projections() {
        assert_eq!(
            parse_weight_name("model.layers.3.mlp.gate_proj.weight"),
            Some(WeightTarget::LayerGateProj(3))
        );
        assert_eq!(
            parse_weight_name("model.layers.3.mlp.up_proj.weight"),
            Some(WeightTarget::LayerUpProj(3))
        );
        assert_eq!(
            parse_weight_name("model.layers.3.mlp.down_proj.weight"),
            Some(WeightTarget::LayerDownProj(3))
        );
    }

    #[test]
    fn test_layer_norms() {
        assert_eq!(
            parse_weight_name("model.layers.0.input_layernorm.weight"),
            Some(WeightTarget::LayerAttnNorm(0))
        );
        assert_eq!(
            parse_weight_name("model.layers.0.post_attention_layernorm.weight"),
            Some(WeightTarget::LayerMlpNorm(0))
        );
    }

    #[test]
    fn test_unknown_name() {
        assert_eq!(parse_weight_name("model.layers.0.self_attn.rotary_emb.inv_freq"), None);
        assert_eq!(parse_weight_name("some.random.tensor"), None);
    }

    // Qwen3.5-specific tests
    #[test]
    fn test_qwen35_embedding() {
        assert_eq!(
            parse_weight_name("model.language_model.embed_tokens.weight"),
            Some(WeightTarget::Embedding)
        );
    }

    #[test]
    fn test_qwen35_final_norm() {
        assert_eq!(
            parse_weight_name("model.language_model.norm.weight"),
            Some(WeightTarget::FinalNorm)
        );
    }

    #[test]
    fn test_qwen35_linear_attn() {
        assert_eq!(
            parse_weight_name("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
            Some(WeightTarget::LayerDeltaNetQKV(0))
        );
        assert_eq!(
            parse_weight_name("model.language_model.layers.5.linear_attn.in_proj_z.weight"),
            Some(WeightTarget::LayerDeltaNetZ(5))
        );
        assert_eq!(
            parse_weight_name("model.language_model.layers.2.linear_attn.in_proj_a.weight"),
            Some(WeightTarget::LayerDeltaNetA(2))
        );
        assert_eq!(
            parse_weight_name("model.language_model.layers.2.linear_attn.in_proj_b.weight"),
            Some(WeightTarget::LayerDeltaNetB(2))
        );
        assert_eq!(
            parse_weight_name("model.language_model.layers.1.linear_attn.conv1d.weight"),
            Some(WeightTarget::LayerDeltaNetConv(1))
        );
        assert_eq!(
            parse_weight_name("model.language_model.layers.0.linear_attn.dt_bias"),
            Some(WeightTarget::LayerDeltaNetDtBias(0))
        );
        assert_eq!(
            parse_weight_name("model.language_model.layers.0.linear_attn.A_log"),
            Some(WeightTarget::LayerDeltaNetALog(0))
        );
        assert_eq!(
            parse_weight_name("model.language_model.layers.4.linear_attn.norm.weight"),
            Some(WeightTarget::LayerDeltaNetNorm(4))
        );
        assert_eq!(
            parse_weight_name("model.language_model.layers.3.linear_attn.out_proj.weight"),
            Some(WeightTarget::LayerDeltaNetOutProj(3))
        );
    }

    #[test]
    fn test_qwen35_gated_attn() {
        assert_eq!(
            parse_weight_name("model.language_model.layers.3.self_attn.q_norm.weight"),
            Some(WeightTarget::LayerQNorm(3))
        );
        assert_eq!(
            parse_weight_name("model.language_model.layers.7.self_attn.k_norm.weight"),
            Some(WeightTarget::LayerKNorm(7))
        );
        assert_eq!(
            parse_weight_name("model.language_model.layers.3.self_attn.q_proj.weight"),
            Some(WeightTarget::LayerQProj(3))
        );
    }

    #[test]
    fn test_qwen35_mlp() {
        assert_eq!(
            parse_weight_name("model.language_model.layers.0.mlp.gate_proj.weight"),
            Some(WeightTarget::LayerGateProj(0))
        );
    }

    #[test]
    fn test_visual_weights_skipped() {
        // Vision encoder weights should return None (not loaded)
        assert_eq!(
            parse_weight_name("model.visual.blocks.0.attn.proj.weight"),
            None
        );
    }
}
