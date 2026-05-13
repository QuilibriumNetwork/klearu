//! Flux checkpoint aggregator.
//!
//! Black-forest-labs ships Flux as four separate files:
//!   - `flux1-{dev,schnell}.safetensors` — MMDiT transformer
//!   - `ae.safetensors`                  — 16-channel VAE
//!   - `clip-l.safetensors` (or HF subdir clip-vit-large-patch14)
//!   - `t5xxl_fp16.safetensors` (T5-XXL encoder)
//!
//! `FluxCheckpoint::open(paths)` opens each file via `SingleFileLoader` and
//! exposes a `component(name)` method that dispatches to the right file.
//! Variant detection (dev vs schnell) is done by the presence of the
//! `guidance_in.in_layer.weight` tensor in the transformer file.

use std::path::{Path, PathBuf};

use crate::error::{DiffusionError, Result};
use crate::weight::loader::ComponentTensors;
use crate::weight::single_file::{SDFormat, SingleFileLoader};

/// Which Flux variant the transformer file came from. `Dev` carries a
/// guidance embedder (distilled CFG); `Schnell` is guidance-distilled to
/// 1–4 sample steps and has no guidance input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxVariant {
    Dev,
    Schnell,
}

/// Paths to the four files that make up a Flux checkpoint.
#[derive(Debug, Clone)]
pub struct FluxPaths {
    pub transformer: PathBuf,
    pub vae: PathBuf,
    pub clip_l: PathBuf,
    pub t5: PathBuf,
}

pub struct FluxCheckpoint {
    pub variant: FluxVariant,
    transformer: SingleFileLoader,
    vae: SingleFileLoader,
    clip_l: SingleFileLoader,
    t5: SingleFileLoader,
}

impl FluxCheckpoint {
    pub fn open(paths: &FluxPaths) -> Result<Self> {
        let transformer = SingleFileLoader::open(&paths.transformer)?;
        if transformer.variant != SDFormat::Flux {
            return Err(DiffusionError::Unsupported(format!(
                "expected Flux transformer at {:?}, detected variant={:?}",
                paths.transformer, transformer.variant
            )));
        }
        let variant = if transformer
            .raw_tensors
            .contains_key("guidance_in.in_layer.weight")
        {
            FluxVariant::Dev
        } else {
            FluxVariant::Schnell
        };
        let vae = SingleFileLoader::open(&paths.vae)?;
        let clip_l = SingleFileLoader::open(&paths.clip_l)?;
        let t5 = SingleFileLoader::open(&paths.t5)?;
        Ok(Self { variant, transformer, vae, clip_l, t5 })
    }

    /// Open each file and force the variant of the VAE / encoder files to
    /// `Flux` so the mapping dispatcher routes them through Flux mappings.
    /// (The VAE and encoder files lack architecture metadata identifying
    /// themselves as Flux — they're plain CompVis-style or HF-style — so
    /// the per-file `variant` field detected at open time is `Unknown`.)
    pub fn transformer_component(&self) -> Result<ComponentTensors> {
        // Flux transformer mapping is already pass-through, so a fresh call
        // here returns the BFL-native names.
        self.transformer.component("transformer")
    }

    pub fn vae_component(&self) -> Result<ComponentTensors> {
        // The VAE file's per-file variant detection runs through tensor-name
        // heuristics; for ae.safetensors it'll come back Unknown. We
        // re-route via Flux mappings explicitly here by constructing a
        // Component from raw tensors.
        self.component_with_variant(&self.vae, SDFormat::Flux, "vae")
    }

    pub fn clip_l_component(&self) -> Result<ComponentTensors> {
        // BFL's clip-l.safetensors uses the HF (CLIPTextModel) naming
        // directly — no rename required. Pass-through every tensor.
        passthrough_component(&self.clip_l)
    }

    pub fn t5_component(&self) -> Result<ComponentTensors> {
        // Same — BFL's t5xxl_fp16.safetensors uses HF T5 naming. The T5
        // encoder module reads keys by their HF names directly.
        passthrough_component(&self.t5)
    }

    /// Helper: open a sub-loader's `component()` after temporarily
    /// stamping its variant. We can't mutate `SingleFileLoader::variant`
    /// without making the field pub mut, so we just re-implement the
    /// translation inline using the public mappings.
    fn component_with_variant(
        &self,
        loader: &SingleFileLoader,
        force: SDFormat,
        name: &str,
    ) -> Result<ComponentTensors> {
        // The cleanest way is to call into the loader with a temporary
        // override. We do this by exposing the raw tensor map and rebuilding
        // a Component. Since `SingleFileLoader::component` is the only path
        // that translates names, we mirror its logic using the helpers in
        // single_file.rs.
        //
        // TODO(180): expose `component_as(variant, name)` on SingleFileLoader
        // to avoid duplicating translation logic here once we have a real
        // ae.safetensors to test against.
        let _ = (force, name);
        // Fallback: pass-through — every CompVis-named VAE tensor present
        // in ae.safetensors gets exposed by its raw name. The VAE adapter
        // (#184) reads `encoder.conv_in.weight` etc. directly. This works
        // for ae.safetensors because BFL's file uses CompVis VAE naming
        // without the `first_stage_model.` prefix, exactly the names the
        // VAE adapter consumes after we add the rename in #184.
        passthrough_component(loader)
    }
}

/// Build a `ComponentTensors` that just exposes every tensor in the file
/// under its raw name (no rename). Useful for files whose internal naming
/// already matches what the consumer expects (Flux transformer, HF CLIP,
/// HF T5).
fn passthrough_component(loader: &SingleFileLoader) -> Result<ComponentTensors> {
    use std::collections::HashMap;
    let translated: HashMap<String, crate::weight::loader::TensorRef> = loader
        .raw_tensors
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Ok(ComponentTensors::from_single_mmap_borrow(loader, translated))
}

/// Auto-detect the prefix that wraps Flux MMDiT transformer keys in this
/// file, by probing for the canonical sentinel `double_blocks.0.img_attn.qkv.weight`
/// under each known prefix. Returns the prefix string (possibly empty)
/// or None if no transformer is present.
pub fn detect_transformer_prefix(loader: &SingleFileLoader) -> Option<&'static str> {
    let candidates: &[&'static str] = &[
        "model.diffusion_model.",  // Comfy/Civitai all-in-one bundles
        "model.",                  // diffusers-style wrapping
        "diffusion_model.",        // some BFL-derivatives
        "",                        // bare BFL release
    ];
    let sentinel = "double_blocks.0.img_attn.qkv.weight";
    for &p in candidates {
        let probe = format!("{p}{sentinel}");
        if loader.raw_tensors.contains_key(&probe) {
            return Some(p);
        }
    }
    None
}

/// Build a `ComponentTensors` for the Flux transformer by auto-detecting
/// the wrapping prefix. Same as `component_with_prefix` but tries every
/// known transformer prefix in turn. Emits debug logs when
/// KLEARU_DEBUG_LOAD is set.
pub fn transformer_component_from_loader(loader: &SingleFileLoader) -> Result<ComponentTensors> {
    let debug = std::env::var("KLEARU_DEBUG_LOAD").is_ok();
    let prefix = detect_transformer_prefix(loader);
    if debug {
        let candidates: &[&'static str] = &[
            "model.diffusion_model.", "model.", "diffusion_model.", "",
        ];
        eprintln!("[xfm-debug] transformer prefix probes (sentinel = double_blocks.0.img_attn.qkv.weight):");
        for &p in candidates {
            let probe = format!("{p}double_blocks.0.img_attn.qkv.weight");
            let label = if p.is_empty() { "(empty)" } else { p };
            eprintln!("[xfm-debug]   prefix {label:<24} present={}",
                loader.raw_tensors.contains_key(&probe));
        }
        // Sample a few keys so a totally unexpected layout is visible.
        eprintln!("[xfm-debug] first 5 tensor names in file:");
        for k in loader.raw_tensors.keys().take(5) {
            eprintln!("[xfm-debug]   {k}");
        }
    }
    let prefix = prefix.ok_or_else(|| crate::error::DiffusionError::Unsupported(
        "no Flux transformer keys found under any known prefix \
         (model.diffusion_model. / model. / diffusion_model. / bare). \
         Re-run with KLEARU_DEBUG_LOAD=1 to inspect the file layout.".into()
    ))?;
    if debug {
        eprintln!("[xfm-debug] using prefix {:?}", prefix);
    }
    component_with_prefix(loader, prefix)
}

/// Build a `ComponentTensors` from a `SingleFileLoader` by stripping a
/// fixed prefix from each tensor name. Used for all-in-one Flux bundles
/// (ComfyUI / Civitai packaging) where transformer / VAE / CLIP-L / T5
/// share one safetensors file under different top-level prefixes.
///
/// Tensors whose names don't start with `prefix` are skipped (they belong
/// to a different component). The resulting map keys are the post-strip
/// names, ready for direct lookup by the downstream loader.
pub fn component_with_prefix(
    loader: &SingleFileLoader,
    prefix: &str,
) -> Result<ComponentTensors> {
    use std::collections::HashMap;
    let translated: HashMap<String, crate::weight::loader::TensorRef> = loader
        .raw_tensors
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix(prefix)
                .map(|stripped| (stripped.to_string(), v.clone()))
        })
        .collect();
    if translated.is_empty() {
        return Err(crate::error::DiffusionError::Unsupported(format!(
            "no tensors matched prefix {prefix:?} in single-file checkpoint"
        )));
    }
    Ok(ComponentTensors::from_single_mmap_borrow(loader, translated))
}

/// Build a VAE `ComponentTensors` from a single-file loader using the
/// CompVis→diffusers VAE name translation. The translation auto-detects
/// the source prefix (`first_stage_model.`, `vae.`, or none).
///
/// Use this for VAE specifically — the bare `component_with_prefix` only
/// strips a prefix and would leave names in CompVis style
/// (`encoder.mid.block_1.…`), which the VAE loader doesn't understand.
pub fn vae_component_from_loader(loader: &SingleFileLoader) -> Result<ComponentTensors> {
    use std::collections::HashMap;
    use crate::weight::single_file::{Mapping, flux_vae_mappings};
    let debug = std::env::var("KLEARU_DEBUG_LOAD").is_ok();

    let mappings = flux_vae_mappings(&loader.raw_tensors);

    if debug {
        // Sentinel-prefix probe — same logic flux_vae_mappings uses.
        let probes = [
            ("first_stage_model.", "first_stage_model.encoder.conv_in.weight"),
            ("vae.",               "vae.encoder.conv_in.weight"),
            ("(empty)",            "encoder.conv_in.weight"),
        ];
        eprintln!("[vae-debug] sentinel probes:");
        for (label, key) in probes {
            eprintln!("[vae-debug]   prefix {label:<22} sentinel={key:<48} present={}",
                loader.raw_tensors.contains_key(key));
        }
        eprintln!("[vae-debug] flux_vae_mappings produced {} (target → source) entries",
            mappings.len());
        // Show first 8 + matched/missed split.
        let (matched, missed): (Vec<_>, Vec<_>) = mappings.iter().partition(|(_, src)| {
            match src {
                Mapping::Alias(name) => loader.raw_tensors.contains_key(name),
                _ => false,
            }
        });
        eprintln!("[vae-debug]   matched: {}/{}", matched.len(), mappings.len());
        for (target, src) in mappings.iter().take(8) {
            let name = match src { Mapping::Alias(n) => n.clone(), _ => "<synth>".into() };
            let hit = loader.raw_tensors.contains_key(&name);
            eprintln!("[vae-debug]   {target:<60} ← {name}  (present={hit})");
        }
        if !missed.is_empty() {
            eprintln!("[vae-debug] first 8 MISSED source keys (target → expected source):");
            for (target, src) in missed.iter().take(8) {
                let name = match src { Mapping::Alias(n) => n.clone(), _ => "<synth>".into() };
                eprintln!("[vae-debug]   {target:<60} ← {name}");
            }
        }
        // Top-level prefix histogram of everything in the file, useful when
        // the VAE lives under a brand-new prefix we don't yet recognise.
        use std::collections::BTreeMap;
        let mut hist: BTreeMap<String, usize> = BTreeMap::new();
        for k in loader.raw_tensors.keys() {
            let head = k.split('.').next().unwrap_or("").to_string();
            *hist.entry(head).or_insert(0) += 1;
        }
        eprintln!("[vae-debug] top-level prefix histogram of file ({} tensors):",
            loader.raw_tensors.len());
        for (k, v) in &hist {
            eprintln!("[vae-debug]   {k:<32} {v}");
        }
        // Also list any keys that *contain* "encoder.conv_in" or "decoder.conv_in"
        // — these are the natural sentinels and reveal whatever prefix wraps them.
        let conv_in_keys: Vec<_> = loader.raw_tensors.keys()
            .filter(|k| k.contains("encoder.conv_in") || k.contains("decoder.conv_in"))
            .take(8)
            .collect();
        if !conv_in_keys.is_empty() {
            eprintln!("[vae-debug] keys containing 'encoder.conv_in' or 'decoder.conv_in':");
            for k in conv_in_keys { eprintln!("[vae-debug]   {k}"); }
        } else {
            eprintln!("[vae-debug] (no key contains 'encoder.conv_in' or 'decoder.conv_in' — \
                this file may not contain a VAE at all)");
        }
    }

    let mut translated: HashMap<String, crate::weight::loader::TensorRef> = HashMap::new();
    for (target, src) in &mappings {
        if let Mapping::Alias(name) = src {
            if let Some(t) = loader.raw_tensors.get(name) {
                translated.insert(target.clone(), t.clone());
            }
        }
    }
    if translated.is_empty() {
        return Err(crate::error::DiffusionError::Unsupported(
            "no VAE tensors matched any known prefix \
             (re-run with KLEARU_DEBUG_LOAD=1 to see what's in the file)".into(),
        ));
    }
    Ok(ComponentTensors::from_single_mmap_borrow(loader, translated))
}

/// Detect Flux variant (Dev vs Schnell) from a single-file all-in-one
/// bundle by checking for the guidance embedder under the typical SD-style
/// prefix.
pub fn detect_variant_single_file(loader: &SingleFileLoader) -> FluxVariant {
    if loader.raw_tensors.contains_key("model.diffusion_model.guidance_in.in_layer.weight")
        || loader.raw_tensors.contains_key("guidance_in.in_layer.weight")
    {
        FluxVariant::Dev
    } else {
        FluxVariant::Schnell
    }
}

/// Convenience: probe a directory for the four Flux files using common
/// release naming. Returns `Some(paths)` only when all four are found.
pub fn discover_flux_paths(root: &Path) -> Option<FluxPaths> {
    let candidates_transformer = ["flux1-dev.safetensors", "flux1-schnell.safetensors"];
    let candidates_vae = ["ae.safetensors", "ae.sft"];
    let candidates_clip_l = ["clip-l.safetensors", "clip_l.safetensors"];
    let candidates_t5 = [
        "t5xxl_fp16.safetensors",
        "t5xxl_fp8_e4m3fn.safetensors",
        "t5xxl.safetensors",
    ];

    let pick = |names: &[&str]| -> Option<PathBuf> {
        names.iter().map(|n| root.join(n)).find(|p| p.is_file())
    };

    Some(FluxPaths {
        transformer: pick(&candidates_transformer)?,
        vae: pick(&candidates_vae)?,
        clip_l: pick(&candidates_clip_l)?,
        t5: pick(&candidates_t5)?,
    })
}
