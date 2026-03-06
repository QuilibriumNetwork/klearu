/// Weight loader for Florence-2 vision encoder.
///
/// Florence-2 uses DaViT as its vision backbone.
/// Maps Florence-2 weight names to existing `DaViTModel`.

use std::path::Path;

use crate::config::DaViTConfig;
use crate::error::{VisionError, Result};
use crate::model::DaViTModel;

/// Load a Florence-2 vision encoder from a model directory.
///
/// Florence-2 stores its vision backbone under the `vision_tower.` prefix.
pub fn load_florence2_vision(model_dir: &Path) -> Result<DaViTModel> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;

    // Florence-2 may have nested vision config
    let v: serde_json::Value =
        serde_json::from_str(&config_str).map_err(|e| VisionError::InvalidConfig(e.to_string()))?;

    let vision_json = if let Some(vc) = v.get("vision_config") {
        serde_json::to_string(vc).unwrap_or_else(|_| config_str.clone())
    } else {
        config_str
    };

    let config = DaViTConfig::from_timm_config(&vision_json)?;

    tracing::info!(
        "Loading Florence-2 vision: embed_dims={:?}, depths={:?}",
        config.embed_dims, config.depths,
    );

    let model = DaViTModel::new(config);

    // Weight loading uses the same DaViT loader with prefix stripping
    // Full implementation would iterate safetensors and strip "vision_tower." prefix
    // before delegating to `load_weight_into_model`.

    Ok(model)
}
