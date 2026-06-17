//! Vision engine — image classification via `klearu-vision`.
//!
//! Phase 4 supports two model families because they cover ~all timm
//! classification checkpoints:
//!
//!   - **DaViT** (`load_davit_model`) — the hierarchical dual-attention
//!     model. Microsoft's Florence-2 vision tower uses this; the timm
//!     `davit_*` variants land here.
//!   - **ViT** (`load_vit_model`) — vanilla Vision Transformer. CLIP,
//!     DINOv2, SigLIP, DeiT, etc.
//!
//! Other families (Swin, ConvNext, Hiera, EVA02) have loaders in
//! klearu-vision and slot in here with an extra enum arm when needed.
//!
//! The dispatch is by directory layout: we sniff `config.json`'s shape
//! to decide DaViT vs ViT. If it's ambiguous the caller can pass
//! `force_kind` (Phase 4 settings expose this).
//!
//! Labels resolution order: `labels.json` (plain array) in the model dir,
//! then `id2label` in `config.json`, then `"class_N"` placeholders.

use std::path::{Path, PathBuf};

use image::{imageops::FilterType, GenericImageView};
use klearu_vision::preprocess::{
    center_crop, normalize, rgb_bytes_to_chw, Interpolation, PreprocessConfig, ResizeMode,
};
use klearu_vision::weight::{load_davit_model, load_vit_model};
use klearu_vision::{DaViTModel, ViTModel};
use serde::{Deserialize, Serialize};

use crate::error::{DesktopError, Result};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VisionKind {
    Davit,
    Vit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionStatus {
    pub loaded: bool,
    pub model_dir: Option<String>,
    pub kind: Option<VisionKind>,
    pub num_classes: Option<usize>,
    pub input_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub class_index: usize,
    pub label: String,
    pub probability: f32,
}

enum Model {
    Davit(DaViTModel),
    Vit(ViTModel),
}

struct Loaded {
    model: Model,
    preprocess: PreprocessConfig,
    labels: Vec<String>,
    model_dir: PathBuf,
    kind: VisionKind,
    num_classes: usize,
}

#[derive(Default)]
pub struct VisionEngine {
    loaded: Option<Loaded>,
}

impl VisionEngine {
    pub fn new() -> Self { Self::default() }

    pub fn status(&self) -> VisionStatus {
        match &self.loaded {
            Some(l) => VisionStatus {
                loaded: true,
                model_dir: Some(l.model_dir.to_string_lossy().into_owned()),
                kind: Some(l.kind),
                num_classes: Some(l.num_classes),
                input_size: Some(l.preprocess.input_size),
            },
            None => VisionStatus {
                loaded: false, model_dir: None, kind: None, num_classes: None, input_size: None,
            },
        }
    }

    pub fn unload(&mut self) { self.loaded = None; }

    /// Load a model. `force_kind` overrides the auto-detect; pass `None`
    /// to let us sniff `config.json`.
    pub fn load(&mut self, model_dir: &Path, force_kind: Option<VisionKind>) -> Result<VisionStatus> {
        tracing::info!(path = %model_dir.display(), ?force_kind, "loading vision model");
        let preprocess = preprocess_from_dir(model_dir);
        let kind = force_kind.unwrap_or_else(|| detect_kind(model_dir));

        let (model, num_classes) = match kind {
            VisionKind::Davit => {
                let m = load_davit_model(model_dir)
                    .map_err(|e| DesktopError::engine(format!("load davit: {e}")))?;
                let n = m.config.num_classes;
                (Model::Davit(m), n)
            }
            VisionKind::Vit => {
                let m = load_vit_model(model_dir)
                    .map_err(|e| DesktopError::engine(format!("load vit: {e}")))?;
                let n = m.config.num_classes;
                (Model::Vit(m), n)
            }
        };

        let labels = load_labels(model_dir, num_classes);

        self.loaded = Some(Loaded {
            model,
            preprocess,
            labels,
            model_dir: model_dir.to_path_buf(),
            kind,
            num_classes,
        });
        Ok(self.status())
    }

    /// Classify the image at `path` and return the top-`k` predictions.
    pub fn classify_path(&self, path: &Path, k: usize) -> Result<Vec<Prediction>> {
        let bytes = std::fs::read(path)
            .map_err(|e| DesktopError::Io(format!("read {}: {e}", path.display())))?;
        self.classify_bytes(&bytes, k)
    }

    /// Classify decoded-image bytes (PNG/JPEG/etc).
    pub fn classify_bytes(&self, bytes: &[u8], k: usize) -> Result<Vec<Prediction>> {
        let Some(loaded) = self.loaded.as_ref() else {
            return Err(DesktopError::NotLoaded("vision"));
        };

        let img = image::load_from_memory(bytes)
            .map_err(|e| DesktopError::bad_input(format!("image decode: {e}")))?;
        let tensor = preprocess_to_tensor(&img, &loaded.preprocess);

        let logits = match &loaded.model {
            Model::Davit(m) => m.forward(&tensor),
            Model::Vit(m) => m.forward(&tensor),
        };

        let probs = softmax(&logits);
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = k.max(1).min(indexed.len());

        Ok(indexed
            .into_iter()
            .take(k)
            .map(|(idx, p)| Prediction {
                class_index: idx,
                label: loaded
                    .labels
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{idx}")),
                probability: p,
            })
            .collect())
    }
}

// ---- preprocessing ---------------------------------------------------------

fn preprocess_from_dir(model_dir: &Path) -> PreprocessConfig {
    let config_path = model_dir.join("config.json");
    if let Ok(json) = std::fs::read_to_string(&config_path) {
        PreprocessConfig::from_model_config(&json)
    } else {
        PreprocessConfig::default()
    }
}

/// Decode an in-memory image, resize per the preprocess config, center-crop
/// to `input_size`, normalize with mean/std. Returns a `[3, H, W]` f32 buf.
fn preprocess_to_tensor(img: &image::DynamicImage, config: &PreprocessConfig) -> Vec<f32> {
    let filter = match config.interpolation {
        Interpolation::Bicubic => FilterType::CatmullRom,
        Interpolation::Bilinear => FilterType::Triangle,
    };
    let target = config.input_size;

    let resized = match config.resize_mode {
        ResizeMode::ShortEdge => {
            let (w, h) = img.dimensions();
            let scale = config.resize_size as f32 / w.min(h) as f32;
            img.resize_exact(
                (w as f32 * scale).round() as u32,
                (h as f32 * scale).round() as u32,
                filter,
            )
        }
        ResizeMode::LongEdge => {
            let (w, h) = img.dimensions();
            let scale = config.resize_size as f32 / w.max(h) as f32;
            img.resize_exact(
                (w as f32 * scale).round() as u32,
                (h as f32 * scale).round() as u32,
                filter,
            )
        }
        ResizeMode::Exact => img.resize_exact(
            config.resize_size as u32,
            config.resize_size as u32,
            filter,
        ),
    };

    let rgb = resized.to_rgb8();
    let (rw, rh) = (rgb.width() as usize, rgb.height() as usize);
    let tensor = rgb_bytes_to_chw(rgb.as_raw(), rh, rw);
    let mut cropped = if rh > target || rw > target {
        center_crop(&tensor, 3, rh, rw, target, target)
    } else {
        tensor
    };
    normalize(&mut cropped, 3, target, target, &config.mean, &config.std);
    cropped
}

// ---- model-kind sniffing ---------------------------------------------------

fn detect_kind(model_dir: &Path) -> VisionKind {
    let config_path = model_dir.join("config.json");
    let Ok(text) = std::fs::read_to_string(&config_path) else {
        return VisionKind::Davit; // default — most checkpoints we ship are DaViT
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return VisionKind::Davit;
    };
    // DaViT configs have `embed_dims` (list-per-stage) and `depths`. ViT
    // configs have a flat `hidden_size` or `embed_dim` scalar.
    let arch = v.get("architectures").and_then(|x| x.as_array());
    if let Some(arch) = arch {
        for a in arch {
            if let Some(s) = a.as_str() {
                let sl = s.to_lowercase();
                if sl.contains("davit") { return VisionKind::Davit; }
                if sl.contains("vit") || sl.contains("clip") || sl.contains("siglip") || sl.contains("dino") {
                    return VisionKind::Vit;
                }
            }
        }
    }
    if v.get("embed_dims").and_then(|x| x.as_array()).is_some() {
        return VisionKind::Davit;
    }
    if v.get("hidden_size").is_some() || v.get("embed_dim").is_some() {
        return VisionKind::Vit;
    }
    VisionKind::Davit
}

// ---- labels ----------------------------------------------------------------

fn load_labels(model_dir: &Path, num_classes: usize) -> Vec<String> {
    // 1. `labels.json` as a flat array.
    if let Ok(text) = std::fs::read_to_string(model_dir.join("labels.json")) {
        if let Ok(arr) = serde_json::from_str::<Vec<String>>(&text) {
            if arr.len() >= num_classes {
                return arr.into_iter().take(num_classes).collect();
            }
        }
    }
    // 2. `id2label` in config.json. HF transformers format: { "0": "tench", ... }
    if let Ok(text) = std::fs::read_to_string(model_dir.join("config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(id2label) = v.get("id2label").and_then(|x| x.as_object()) {
                let mut out = vec![String::new(); num_classes];
                let mut got_any = false;
                for (k, val) in id2label {
                    if let (Ok(i), Some(s)) = (k.parse::<usize>(), val.as_str()) {
                        if i < num_classes {
                            out[i] = s.to_string();
                            got_any = true;
                        }
                    }
                }
                if got_any {
                    for (i, s) in out.iter_mut().enumerate() {
                        if s.is_empty() { *s = format!("class_{i}"); }
                    }
                    return out;
                }
            }
        }
    }
    // 3. Fallback: placeholders.
    (0..num_classes).map(|i| format!("class_{i}")).collect()
}

// ---- softmax ---------------------------------------------------------------

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.into_iter().map(|e| e / sum).collect()
    } else {
        vec![0.0; logits.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum = {s}");
        // monotonic
        assert!(p[0] < p[1]);
        assert!(p[1] < p[2]);
    }

    #[test]
    fn softmax_handles_degenerate() {
        let p = softmax(&[]);
        assert!(p.is_empty());
    }
}
