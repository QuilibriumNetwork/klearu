/// Identifies where a timm DaViT weight tensor should be loaded into the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaViTWeightTarget {
    // Stem
    StemConvWeight,
    StemConvBias,
    StemNormWeight,
    StemNormBias,

    // Stage downsample
    StageDownsampleNormWeight(usize),
    StageDownsampleNormBias(usize),
    StageDownsampleConvWeight(usize),
    StageDownsampleConvBias(usize),

    // Block components (stage_idx, block_idx within stage)
    // In timm, blocks alternate: even=Spatial, odd=Channel.
    // block_idx here maps to the (SpatialBlock, ChannelBlock) tuple.
    BlockCpe1Weight(usize, usize, BlockType),
    BlockCpe1Bias(usize, usize, BlockType),
    BlockNorm1Weight(usize, usize, BlockType),
    BlockNorm1Bias(usize, usize, BlockType),
    BlockAttnQkvWeight(usize, usize, BlockType),
    BlockAttnQkvBias(usize, usize, BlockType),
    BlockAttnProjWeight(usize, usize, BlockType),
    BlockAttnProjBias(usize, usize, BlockType),
    BlockCpe2Weight(usize, usize, BlockType),
    BlockCpe2Bias(usize, usize, BlockType),
    BlockNorm2Weight(usize, usize, BlockType),
    BlockNorm2Bias(usize, usize, BlockType),
    BlockMlpFc1Weight(usize, usize, BlockType),
    BlockMlpFc1Bias(usize, usize, BlockType),
    BlockMlpFc2Weight(usize, usize, BlockType),
    BlockMlpFc2Bias(usize, usize, BlockType),

    // Classification head
    HeadNormWeight,
    HeadNormBias,
    HeadFcWeight,
    HeadFcBias,
}

/// Whether this block is spatial (even index in timm) or channel (odd index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    Spatial,
    Channel,
}

/// Parse a timm DaViT weight name to a DaViTWeightTarget.
///
/// timm weight name patterns (current timm format):
/// ```text
/// stem.conv.weight / stem.conv.bias                                → StemConv
/// stem.norm.weight / stem.norm.bias                                → StemNorm
/// stages.{s}.downsample.norm.weight / .bias                        → StageDownsampleNorm
/// stages.{s}.downsample.conv.weight / .bias                        → StageDownsampleConv
/// stages.{s}.blocks.{pair}.{0|1}.cpe1.proj.weight / .bias          → BlockCpe1
/// stages.{s}.blocks.{pair}.{0|1}.cpe2.proj.weight / .bias          → BlockCpe2
/// stages.{s}.blocks.{pair}.{0|1}.norm1.weight / .bias              → BlockNorm1
/// stages.{s}.blocks.{pair}.{0|1}.norm2.weight / .bias              → BlockNorm2
/// stages.{s}.blocks.{pair}.{0|1}.attn.qkv.weight / .bias          → BlockAttnQkv
/// stages.{s}.blocks.{pair}.{0|1}.attn.proj.weight / .bias          → BlockAttnProj
/// stages.{s}.blocks.{pair}.{0|1}.mlp.fc1.weight / .bias           → BlockMlpFc1
/// stages.{s}.blocks.{pair}.{0|1}.mlp.fc2.weight / .bias           → BlockMlpFc2
/// head.norm.weight / head.norm.bias                                → HeadNorm
/// head.fc.weight / head.fc.bias                                    → HeadFc
/// ```
///
/// Block sub-index: 0 = Spatial, 1 = Channel.
/// `pair` is the dual-block pair index within the stage.
pub fn parse_davit_weight_name(name: &str) -> Option<DaViTWeightTarget> {
    // Stem
    if name == "stem.conv.weight" {
        return Some(DaViTWeightTarget::StemConvWeight);
    }
    if name == "stem.conv.bias" {
        return Some(DaViTWeightTarget::StemConvBias);
    }
    if name == "stem.norm.weight" {
        return Some(DaViTWeightTarget::StemNormWeight);
    }
    if name == "stem.norm.bias" {
        return Some(DaViTWeightTarget::StemNormBias);
    }

    // Head
    if name == "head.norm.weight" {
        return Some(DaViTWeightTarget::HeadNormWeight);
    }
    if name == "head.norm.bias" {
        return Some(DaViTWeightTarget::HeadNormBias);
    }
    if name == "head.fc.weight" {
        return Some(DaViTWeightTarget::HeadFcWeight);
    }
    if name == "head.fc.bias" {
        return Some(DaViTWeightTarget::HeadFcBias);
    }

    // stages.{s}.*
    let rest = name.strip_prefix("stages.")?;
    let dot_pos = rest.find('.')?;
    let stage_idx: usize = rest[..dot_pos].parse().ok()?;
    let suffix = &rest[dot_pos + 1..];

    // Downsample
    if let Some(ds_suffix) = suffix.strip_prefix("downsample.") {
        return match ds_suffix {
            "norm.weight" => Some(DaViTWeightTarget::StageDownsampleNormWeight(stage_idx)),
            "norm.bias" => Some(DaViTWeightTarget::StageDownsampleNormBias(stage_idx)),
            "conv.weight" => Some(DaViTWeightTarget::StageDownsampleConvWeight(stage_idx)),
            "conv.bias" => Some(DaViTWeightTarget::StageDownsampleConvBias(stage_idx)),
            _ => None,
        };
    }

    // blocks.{pair}.{sub_idx}.component
    // Format: stages.{s}.blocks.{pair_idx}.{0=spatial|1=channel}.component
    let blocks_rest = suffix.strip_prefix("blocks.")?;
    let dot_pos = blocks_rest.find('.')?;
    let pair_idx: usize = blocks_rest[..dot_pos].parse().ok()?;
    let after_pair = &blocks_rest[dot_pos + 1..];

    let dot_pos2 = after_pair.find('.')?;
    let sub_idx: usize = after_pair[..dot_pos2].parse().ok()?;
    let block_suffix = &after_pair[dot_pos2 + 1..];

    let block_type = if sub_idx == 0 {
        BlockType::Spatial
    } else {
        BlockType::Channel
    };

    match block_suffix {
        "cpe1.proj.weight" => Some(DaViTWeightTarget::BlockCpe1Weight(stage_idx, pair_idx, block_type)),
        "cpe1.proj.bias" => Some(DaViTWeightTarget::BlockCpe1Bias(stage_idx, pair_idx, block_type)),
        "norm1.weight" => Some(DaViTWeightTarget::BlockNorm1Weight(stage_idx, pair_idx, block_type)),
        "norm1.bias" => Some(DaViTWeightTarget::BlockNorm1Bias(stage_idx, pair_idx, block_type)),
        "attn.qkv.weight" => Some(DaViTWeightTarget::BlockAttnQkvWeight(stage_idx, pair_idx, block_type)),
        "attn.qkv.bias" => Some(DaViTWeightTarget::BlockAttnQkvBias(stage_idx, pair_idx, block_type)),
        "attn.proj.weight" => Some(DaViTWeightTarget::BlockAttnProjWeight(stage_idx, pair_idx, block_type)),
        "attn.proj.bias" => Some(DaViTWeightTarget::BlockAttnProjBias(stage_idx, pair_idx, block_type)),
        "cpe2.proj.weight" => Some(DaViTWeightTarget::BlockCpe2Weight(stage_idx, pair_idx, block_type)),
        "cpe2.proj.bias" => Some(DaViTWeightTarget::BlockCpe2Bias(stage_idx, pair_idx, block_type)),
        "norm2.weight" => Some(DaViTWeightTarget::BlockNorm2Weight(stage_idx, pair_idx, block_type)),
        "norm2.bias" => Some(DaViTWeightTarget::BlockNorm2Bias(stage_idx, pair_idx, block_type)),
        "mlp.fc1.weight" => Some(DaViTWeightTarget::BlockMlpFc1Weight(stage_idx, pair_idx, block_type)),
        "mlp.fc1.bias" => Some(DaViTWeightTarget::BlockMlpFc1Bias(stage_idx, pair_idx, block_type)),
        "mlp.fc2.weight" => Some(DaViTWeightTarget::BlockMlpFc2Weight(stage_idx, pair_idx, block_type)),
        "mlp.fc2.bias" => Some(DaViTWeightTarget::BlockMlpFc2Bias(stage_idx, pair_idx, block_type)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stem_weights() {
        assert_eq!(parse_davit_weight_name("stem.conv.weight"), Some(DaViTWeightTarget::StemConvWeight));
        assert_eq!(parse_davit_weight_name("stem.conv.bias"), Some(DaViTWeightTarget::StemConvBias));
        assert_eq!(parse_davit_weight_name("stem.norm.weight"), Some(DaViTWeightTarget::StemNormWeight));
        assert_eq!(parse_davit_weight_name("stem.norm.bias"), Some(DaViTWeightTarget::StemNormBias));
    }

    #[test]
    fn test_head_weights() {
        assert_eq!(parse_davit_weight_name("head.norm.weight"), Some(DaViTWeightTarget::HeadNormWeight));
        assert_eq!(parse_davit_weight_name("head.norm.bias"), Some(DaViTWeightTarget::HeadNormBias));
        assert_eq!(parse_davit_weight_name("head.fc.weight"), Some(DaViTWeightTarget::HeadFcWeight));
        assert_eq!(parse_davit_weight_name("head.fc.bias"), Some(DaViTWeightTarget::HeadFcBias));
    }

    #[test]
    fn test_downsample_weights() {
        assert_eq!(
            parse_davit_weight_name("stages.1.downsample.norm.weight"),
            Some(DaViTWeightTarget::StageDownsampleNormWeight(1))
        );
        assert_eq!(
            parse_davit_weight_name("stages.2.downsample.conv.weight"),
            Some(DaViTWeightTarget::StageDownsampleConvWeight(2))
        );
        assert_eq!(
            parse_davit_weight_name("stages.1.downsample.conv.bias"),
            Some(DaViTWeightTarget::StageDownsampleConvBias(1))
        );
    }

    #[test]
    fn test_spatial_block_weights() {
        // Sub-index 0 = spatial
        assert_eq!(
            parse_davit_weight_name("stages.0.blocks.0.0.attn.qkv.weight"),
            Some(DaViTWeightTarget::BlockAttnQkvWeight(0, 0, BlockType::Spatial))
        );
        assert_eq!(
            parse_davit_weight_name("stages.2.blocks.2.0.mlp.fc1.weight"),
            Some(DaViTWeightTarget::BlockMlpFc1Weight(2, 2, BlockType::Spatial))
        );
    }

    #[test]
    fn test_channel_block_weights() {
        // Sub-index 1 = channel
        assert_eq!(
            parse_davit_weight_name("stages.0.blocks.0.1.attn.qkv.weight"),
            Some(DaViTWeightTarget::BlockAttnQkvWeight(0, 0, BlockType::Channel))
        );
        assert_eq!(
            parse_davit_weight_name("stages.2.blocks.1.1.norm1.weight"),
            Some(DaViTWeightTarget::BlockNorm1Weight(2, 1, BlockType::Channel))
        );
    }

    #[test]
    fn test_cpe_weights() {
        assert_eq!(
            parse_davit_weight_name("stages.0.blocks.0.0.cpe1.proj.weight"),
            Some(DaViTWeightTarget::BlockCpe1Weight(0, 0, BlockType::Spatial))
        );
        assert_eq!(
            parse_davit_weight_name("stages.1.blocks.0.1.cpe2.proj.bias"),
            Some(DaViTWeightTarget::BlockCpe2Bias(1, 0, BlockType::Channel))
        );
    }

    #[test]
    fn test_unknown_name() {
        assert_eq!(parse_davit_weight_name("some.random.tensor"), None);
        assert_eq!(parse_davit_weight_name("stages.0.blocks.0.0.something.weight"), None);
    }
}
