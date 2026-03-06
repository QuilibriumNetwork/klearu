use crate::error::{VisionError, Result};
use crate::model::vit::PoolType;

/// Configuration for DaViT (Dual Attention Vision Transformer).
#[derive(Debug, Clone)]
pub struct DaViTConfig {
    pub image_size: usize,
    pub in_channels: usize,
    pub embed_dims: [usize; 4],
    pub num_heads: [usize; 4],
    pub depths: [usize; 4],
    pub window_size: usize,
    pub mlp_ratio: f32,
    pub num_classes: usize,
    pub layer_norm_eps: f32,
}

impl DaViTConfig {
    pub fn tiny() -> Self {
        Self {
            image_size: 224,
            in_channels: 3,
            embed_dims: [96, 192, 384, 768],
            num_heads: [3, 6, 12, 24],
            depths: [1, 1, 3, 1],
            window_size: 7,
            mlp_ratio: 4.0,
            num_classes: 1000,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn small() -> Self {
        Self {
            depths: [1, 1, 9, 1],
            ..Self::tiny()
        }
    }

    pub fn base() -> Self {
        Self {
            embed_dims: [128, 256, 512, 1024],
            num_heads: [4, 8, 16, 32],
            depths: [1, 1, 9, 1],
            ..Self::tiny()
        }
    }

    /// Head dim is always embed_dim / num_heads.
    pub fn head_dim(&self, stage: usize) -> usize {
        self.embed_dims[stage] / self.num_heads[stage]
    }

    /// MLP hidden dim for a given stage.
    pub fn mlp_hidden_dim(&self, stage: usize) -> usize {
        (self.embed_dims[stage] as f32 * self.mlp_ratio) as usize
    }

    /// Parse from a timm-style JSON config.
    ///
    /// Supports two formats:
    /// 1. Explicit fields: `embed_dims`, `num_heads`, `depths`, etc.
    /// 2. Architecture name only: `"architecture": "davit_tiny"` — falls back
    ///    to known configs, overriding `num_classes` from the JSON if present.
    pub fn from_timm_config(json: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| VisionError::InvalidConfig(e.to_string()))?;

        // Try architecture-name fallback first if explicit fields are missing
        let has_explicit = v["embed_dims"].is_array()
            && v["num_heads"].is_array()
            && v["depths"].is_array();

        if !has_explicit {
            if let Some(arch) = v["architecture"].as_str() {
                let num_classes = v["num_classes"].as_u64().unwrap_or(1000) as usize;
                let base = match arch {
                    "davit_tiny" => Self::tiny(),
                    "davit_small" => Self::small(),
                    "davit_base" => Self::base(),
                    _ => return Err(VisionError::InvalidConfig(
                        format!("Unknown architecture '{arch}' and no explicit embed_dims/depths/num_heads"),
                    )),
                };
                return Ok(Self { num_classes, ..base });
            }
            return Err(VisionError::InvalidConfig(
                "Missing embed_dims/num_heads/depths and no architecture field".into(),
            ));
        }

        let image_size = v["img_size"].as_u64().unwrap_or(224) as usize;
        let in_channels = v["in_chans"].as_u64().unwrap_or(3) as usize;
        let num_classes = v["num_classes"].as_u64().unwrap_or(1000) as usize;
        let window_size = v["window_size"].as_u64().unwrap_or(7) as usize;
        let mlp_ratio = v["mlp_ratio"].as_f64().unwrap_or(4.0) as f32;

        let embed_dims = parse_array_4(&v, "embed_dims")
            .ok_or_else(|| VisionError::InvalidConfig("Missing embed_dims".into()))?;
        let num_heads = parse_array_4(&v, "num_heads")
            .ok_or_else(|| VisionError::InvalidConfig("Missing num_heads".into()))?;
        let depths = parse_array_4(&v, "depths")
            .ok_or_else(|| VisionError::InvalidConfig("Missing depths".into()))?;

        Ok(Self {
            image_size,
            in_channels,
            embed_dims,
            num_heads,
            depths,
            window_size,
            mlp_ratio,
            num_classes,
            layer_norm_eps: 1e-5,
        })
    }
}

/// Configuration for standard Vision Transformer (ViT).
#[derive(Debug, Clone, Copy)]
pub struct ViTConfig {
    pub image_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub mlp_ratio: f32,
    pub num_classes: usize,
    pub layer_norm_eps: f32,
    pub pool_type: PoolType,
}

impl ViTConfig {
    pub fn tiny_patch16() -> Self {
        Self {
            image_size: 224,
            in_channels: 3,
            patch_size: 16,
            embed_dim: 192,
            num_heads: 3,
            num_layers: 12,
            mlp_ratio: 4.0,
            num_classes: 1000,
            layer_norm_eps: 1e-6,
            pool_type: PoolType::Cls,
        }
    }

    pub fn small_patch16() -> Self {
        Self {
            embed_dim: 384,
            num_heads: 6,
            ..Self::tiny_patch16()
        }
    }

    pub fn base_patch16() -> Self {
        Self {
            embed_dim: 768,
            num_heads: 12,
            ..Self::tiny_patch16()
        }
    }

    pub fn large_patch16() -> Self {
        Self {
            embed_dim: 1024,
            num_heads: 16,
            num_layers: 24,
            ..Self::tiny_patch16()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    pub fn mlp_hidden_dim(&self) -> usize {
        (self.embed_dim as f32 * self.mlp_ratio) as usize
    }

    /// Parse from a timm-style JSON config.
    pub fn from_timm_config(json: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| VisionError::InvalidConfig(e.to_string()))?;

        // Try architecture name fallback
        if let Some(arch) = v["architecture"].as_str() {
            let num_classes = v["num_classes"].as_u64().unwrap_or(1000) as usize;
            // Match model size from architecture name (vit_tiny, eva02_tiny, etc.)
            let base = match arch {
                a if a.contains("_tiny") => Self::tiny_patch16(),
                a if a.contains("_small") => Self::small_patch16(),
                a if a.contains("_base") => Self::base_patch16(),
                a if a.contains("_large") => Self::large_patch16(),
                _ => return Err(VisionError::InvalidConfig(
                    format!("Unknown ViT architecture '{arch}'"),
                )),
            };
            // Extract patch_size from architecture name (e.g., "patch14" → 14)
            let patch_size = extract_patch_size(arch)
                .or_else(|| v["patch_size"].as_u64().map(|v| v as usize))
                .unwrap_or(base.patch_size);
            // Extract image_size from pretrained_cfg.input_size or architecture name
            let image_size = v["pretrained_cfg"]["input_size"][1].as_u64()
                .map(|v| v as usize)
                .or_else(|| v["img_size"].as_u64().map(|v| v as usize))
                .or_else(|| extract_image_size(arch))
                .unwrap_or(base.image_size);
            let pool_type = match v["global_pool"].as_str() {
                Some("avg") | Some("mean") | Some("map") => PoolType::Mean,
                _ => base.pool_type,
            };
            return Ok(Self { num_classes, patch_size, image_size, pool_type, ..base });
        }

        let image_size = v["img_size"].as_u64().unwrap_or(224) as usize;
        let in_channels = v["in_chans"].as_u64().unwrap_or(3) as usize;
        let patch_size = v["patch_size"].as_u64().unwrap_or(16) as usize;
        let embed_dim = v["embed_dim"].as_u64()
            .ok_or_else(|| VisionError::InvalidConfig("Missing embed_dim".into()))? as usize;
        let num_heads = v["num_heads"].as_u64()
            .ok_or_else(|| VisionError::InvalidConfig("Missing num_heads".into()))? as usize;
        let num_layers = v["depth"].as_u64()
            .or_else(|| v["num_layers"].as_u64())
            .ok_or_else(|| VisionError::InvalidConfig("Missing depth/num_layers".into()))? as usize;
        let mlp_ratio = v["mlp_ratio"].as_f64().unwrap_or(4.0) as f32;
        let num_classes = v["num_classes"].as_u64().unwrap_or(1000) as usize;

        let pool_type = match v["global_pool"].as_str() {
            Some("avg") | Some("mean") | Some("map") => PoolType::Mean,
            _ => PoolType::Cls,
        };

        Ok(Self {
            image_size,
            in_channels,
            patch_size,
            embed_dim,
            num_heads,
            num_layers,
            mlp_ratio,
            num_classes,
            layer_norm_eps: 1e-6,
            pool_type,
        })
    }
}

/// Extract patch size from a timm architecture name (e.g., "vit_small_patch14_dinov2" → 14).
fn extract_patch_size(arch: &str) -> Option<usize> {
    let patch_idx = arch.find("patch")?;
    let after = &arch[patch_idx + 5..];
    let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Extract image size from a timm architecture name (e.g., "eva02_tiny_patch14_336" → 336).
fn extract_image_size(arch: &str) -> Option<usize> {
    // Look for a trailing number after the last underscore (e.g., "_336", "_224")
    let last_underscore = arch.rfind('_')?;
    let after = &arch[last_underscore + 1..];
    let size: usize = after.parse().ok()?;
    // Sanity: image sizes are typically 56..1024
    if size >= 56 { Some(size) } else { None }
}

/// Configuration for Qwen3.5 Vision Encoder (VL model).
#[derive(Debug, Clone)]
pub struct QwenVisionConfig {
    /// Number of transformer blocks (12 for Qwen3.5-0.8B).
    pub depth: usize,
    /// Hidden size of the vision encoder (768).
    pub hidden_size: usize,
    /// Number of attention heads (12).
    pub num_heads: usize,
    /// MLP intermediate size (3072).
    pub intermediate_size: usize,
    /// Spatial patch size (16).
    pub patch_size: usize,
    /// Temporal patch size for video (2).
    pub temporal_patch_size: usize,
    /// Spatial merge size for PatchMerger (2).
    pub spatial_merge_size: usize,
    /// Input channels (3).
    pub in_channels: usize,
    /// Output hidden size (projects to LLM hidden dim, e.g. 1024).
    pub out_hidden_size: usize,
    /// Number of learned position embeddings (2304).
    pub num_position_embeddings: usize,
    /// Layer norm epsilon (1e-6).
    pub layer_norm_eps: f32,
}

impl QwenVisionConfig {
    /// Default config matching Qwen3.5-0.8B vision encoder.
    pub fn qwen35_0_8b() -> Self {
        Self {
            depth: 12,
            hidden_size: 768,
            num_heads: 12,
            intermediate_size: 3072,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            out_hidden_size: 1024,
            num_position_embeddings: 2304,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Parse from the `vision_config` section of a Qwen3.5 config.json.
    pub fn from_hf_config(json: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| VisionError::InvalidConfig(e.to_string()))?;

        // Try to extract vision_config from nested format
        let vc = if let Some(vc) = v.get("vision_config") {
            vc.clone()
        } else {
            v
        };

        Ok(Self {
            depth: vc["depth"].as_u64().unwrap_or(12) as usize,
            hidden_size: vc["hidden_size"].as_u64().unwrap_or(768) as usize,
            num_heads: vc["num_heads"].as_u64().unwrap_or(12) as usize,
            intermediate_size: vc["intermediate_size"].as_u64().unwrap_or(3072) as usize,
            patch_size: vc["patch_size"].as_u64().unwrap_or(16) as usize,
            temporal_patch_size: vc["temporal_patch_size"].as_u64().unwrap_or(2) as usize,
            spatial_merge_size: vc["spatial_merge_size"].as_u64().unwrap_or(2) as usize,
            in_channels: vc["in_channels"].as_u64().unwrap_or(3) as usize,
            out_hidden_size: vc["out_hidden_size"].as_u64().unwrap_or(1024) as usize,
            num_position_embeddings: vc["num_position_embeddings"].as_u64().unwrap_or(2304) as usize,
            layer_norm_eps: vc["layer_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
        })
    }
}

/// Configuration for ConvNeXt models.
#[derive(Debug, Clone)]
pub struct ConvNextConfig {
    /// Number of input channels (3).
    pub in_channels: usize,
    /// Per-stage channel dimensions.
    pub dims: [usize; 4],
    /// Per-stage block depths.
    pub depths: [usize; 4],
    /// Number of output classes.
    pub num_classes: usize,
    /// Layer scale initial value (1e-6).
    pub layer_scale_init: f32,
    /// Layer norm epsilon.
    pub layer_norm_eps: f32,
    /// Whether this is a ConvNeXt V2 (uses GRN instead of LayerScale).
    pub is_v2: bool,
}

impl ConvNextConfig {
    pub fn tiny() -> Self {
        Self {
            in_channels: 3,
            dims: [96, 192, 384, 768],
            depths: [3, 3, 9, 3],
            num_classes: 1000,
            layer_scale_init: 1e-6,
            layer_norm_eps: 1e-6,
            is_v2: false,
        }
    }

    pub fn small() -> Self {
        Self {
            depths: [3, 3, 27, 3],
            ..Self::tiny()
        }
    }

    pub fn base() -> Self {
        Self {
            dims: [128, 256, 512, 1024],
            depths: [3, 3, 27, 3],
            ..Self::tiny()
        }
    }

    pub fn large() -> Self {
        Self {
            dims: [192, 384, 768, 1536],
            depths: [3, 3, 27, 3],
            ..Self::tiny()
        }
    }

    pub fn tiny_v2() -> Self {
        Self { is_v2: true, ..Self::tiny() }
    }

    pub fn base_v2() -> Self {
        Self { is_v2: true, ..Self::base() }
    }
}

/// Configuration for Swin Transformer.
#[derive(Debug, Clone)]
pub struct SwinConfig {
    pub image_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    /// Per-stage embedding dimensions.
    pub embed_dims: [usize; 4],
    /// Per-stage attention head counts.
    pub num_heads: [usize; 4],
    /// Per-stage block depths.
    pub depths: [usize; 4],
    /// Window size for attention.
    pub window_size: usize,
    /// MLP ratio.
    pub mlp_ratio: f32,
    pub num_classes: usize,
    pub layer_norm_eps: f32,
}

impl SwinConfig {
    pub fn tiny() -> Self {
        Self {
            image_size: 224,
            in_channels: 3,
            patch_size: 4,
            embed_dims: [96, 192, 384, 768],
            num_heads: [3, 6, 12, 24],
            depths: [2, 2, 6, 2],
            window_size: 7,
            mlp_ratio: 4.0,
            num_classes: 1000,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn small() -> Self {
        Self {
            depths: [2, 2, 18, 2],
            ..Self::tiny()
        }
    }

    pub fn base() -> Self {
        Self {
            embed_dims: [128, 256, 512, 1024],
            num_heads: [4, 8, 16, 32],
            depths: [2, 2, 18, 2],
            ..Self::tiny()
        }
    }

    pub fn head_dim(&self, stage: usize) -> usize {
        self.embed_dims[stage] / self.num_heads[stage]
    }

    pub fn mlp_hidden_dim(&self, stage: usize) -> usize {
        (self.embed_dims[stage] as f32 * self.mlp_ratio) as usize
    }
}

/// Configuration for Hiera (Hierarchical ViT, SAM2 backbone).
#[derive(Debug, Clone)]
pub struct HieraConfig {
    pub image_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    /// Per-stage embedding dimensions.
    pub embed_dims: [usize; 4],
    /// Per-stage attention head counts.
    pub num_heads: [usize; 4],
    /// Per-stage block depths.
    pub depths: [usize; 4],
    /// Mask unit size (window for attention).
    pub mask_unit_size: usize,
    /// MLP ratio.
    pub mlp_ratio: f32,
    pub num_classes: usize,
}

impl HieraConfig {
    pub fn tiny() -> Self {
        Self {
            image_size: 224,
            in_channels: 3,
            patch_size: 4,
            embed_dims: [96, 192, 384, 768],
            num_heads: [1, 2, 4, 8],
            depths: [1, 2, 7, 2],
            mask_unit_size: 8,
            mlp_ratio: 4.0,
            num_classes: 1000,
        }
    }

    pub fn small() -> Self {
        Self {
            depths: [1, 2, 11, 2],
            ..Self::tiny()
        }
    }

    pub fn base() -> Self {
        Self {
            embed_dims: [96, 192, 384, 768],
            num_heads: [1, 2, 4, 8],
            depths: [2, 3, 16, 3],
            ..Self::tiny()
        }
    }

    pub fn head_dim(&self, stage: usize) -> usize {
        self.embed_dims[stage] / self.num_heads[stage]
    }

    pub fn mlp_hidden_dim(&self, stage: usize) -> usize {
        (self.embed_dims[stage] as f32 * self.mlp_ratio) as usize
    }
}

fn parse_array_4(v: &serde_json::Value, key: &str) -> Option<[usize; 4]> {
    let arr = v[key].as_array()?;
    if arr.len() != 4 {
        return None;
    }
    Some([
        arr[0].as_u64()? as usize,
        arr[1].as_u64()? as usize,
        arr[2].as_u64()? as usize,
        arr[3].as_u64()? as usize,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_config() {
        let c = DaViTConfig::tiny();
        assert_eq!(c.embed_dims, [96, 192, 384, 768]);
        assert_eq!(c.head_dim(0), 32);
        assert_eq!(c.head_dim(3), 32);
        assert_eq!(c.mlp_hidden_dim(0), 384);
    }

    #[test]
    fn test_base_config() {
        let c = DaViTConfig::base();
        assert_eq!(c.embed_dims, [128, 256, 512, 1024]);
        assert_eq!(c.head_dim(0), 32);
    }

    #[test]
    fn test_vit_tiny() {
        let c = ViTConfig::tiny_patch16();
        assert_eq!(c.embed_dim, 192);
        assert_eq!(c.num_heads, 3);
        assert_eq!(c.num_layers, 12);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.num_patches(), 196); // (224/16)^2
    }

    #[test]
    fn test_vit_base() {
        let c = ViTConfig::base_patch16();
        assert_eq!(c.embed_dim, 768);
        assert_eq!(c.num_heads, 12);
        assert_eq!(c.head_dim(), 64);
    }

    #[test]
    fn test_qwen_vision_config() {
        let c = QwenVisionConfig::qwen35_0_8b();
        assert_eq!(c.depth, 12);
        assert_eq!(c.hidden_size, 768);
        assert_eq!(c.num_heads, 12);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.out_hidden_size, 1024);
    }

    #[test]
    fn test_qwen_vision_from_hf() {
        let json = r#"{
            "vision_config": {
                "depth": 12,
                "hidden_size": 768,
                "num_heads": 12,
                "intermediate_size": 3072,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "in_channels": 3,
                "out_hidden_size": 1024,
                "num_position_embeddings": 2304
            }
        }"#;
        let c = QwenVisionConfig::from_hf_config(json).unwrap();
        assert_eq!(c.depth, 12);
        assert_eq!(c.hidden_size, 768);
        assert_eq!(c.out_hidden_size, 1024);
    }

    #[test]
    fn test_convnext_configs() {
        let c = ConvNextConfig::tiny();
        assert_eq!(c.dims, [96, 192, 384, 768]);
        assert_eq!(c.depths, [3, 3, 9, 3]);
        assert!(!c.is_v2);

        let c2 = ConvNextConfig::tiny_v2();
        assert!(c2.is_v2);
    }

    #[test]
    fn test_swin_config() {
        let c = SwinConfig::tiny();
        assert_eq!(c.embed_dims, [96, 192, 384, 768]);
        assert_eq!(c.head_dim(0), 32);
        assert_eq!(c.mlp_hidden_dim(0), 384);
    }

    #[test]
    fn test_hiera_config() {
        let c = HieraConfig::tiny();
        assert_eq!(c.embed_dims, [96, 192, 384, 768]);
        assert_eq!(c.head_dim(0), 96);
        assert_eq!(c.mlp_hidden_dim(0), 384);
    }

    #[test]
    fn test_vit_from_timm_config() {
        let json = r#"{
            "embed_dim": 384,
            "num_heads": 6,
            "depth": 12,
            "patch_size": 16,
            "img_size": 224,
            "num_classes": 1000,
            "global_pool": "avg"
        }"#;
        let c = ViTConfig::from_timm_config(json).unwrap();
        assert_eq!(c.embed_dim, 384);
        assert_eq!(c.num_heads, 6);
        assert!(matches!(c.pool_type, PoolType::Mean));
    }

    #[test]
    fn test_from_timm_config() {
        let json = r#"{
            "img_size": 224,
            "in_chans": 3,
            "num_classes": 1000,
            "embed_dims": [96, 192, 384, 768],
            "num_heads": [3, 6, 12, 24],
            "depths": [1, 1, 3, 1],
            "window_size": 7,
            "mlp_ratio": 4.0
        }"#;
        let c = DaViTConfig::from_timm_config(json).unwrap();
        assert_eq!(c.embed_dims, [96, 192, 384, 768]);
        assert_eq!(c.depths, [1, 1, 3, 1]);
    }
}
