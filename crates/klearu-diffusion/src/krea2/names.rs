//! Tensor names for the Krea 2 transformer.
//!
//! The canonical names are diffusers' (`transformer/` in the HF repo). The
//! official single-file checkpoints (`turbo.safetensors`, `raw.safetensors`,
//! and ComfyUI repackagings under `model.diffusion_model.`) use the reference
//! `mmdit.py` module tree instead; [`canonicalize`] renames those in place so
//! every loader sees one naming scheme. The rename table mirrors diffusers'
//! `convert_krea2_transformer_checkpoint_to_diffusers`.

use crate::weight::loader::ComponentTensors;

/// Present in every diffusers-named checkpoint.
const DIFFUSERS_MARKER: &str = "text_fusion.projector.weight";
const ORIGINAL_MARKER: &str = "txtfusion.projector.weight";
const CHECKPOINT_PREFIXES: [&str; 5] =
    ["", "model.diffusion_model.", "diffusion_model.", "transformer.", "model."];

fn transformer_prefix(comp: &ComponentTensors) -> Option<&'static str> {
    CHECKPOINT_PREFIXES.into_iter().find(|prefix| {
        comp.has(&format!("{prefix}{DIFFUSERS_MARKER}"))
            || comp.has(&format!("{prefix}{ORIGINAL_MARKER}"))
    })
}

/// Whether `comp` holds a Krea 2 transformer under either naming scheme.
pub fn is_krea2_transformer(comp: &ComponentTensors) -> bool {
    transformer_prefix(comp).is_some()
}

/// Rename reference (`mmdit.py`) tensor names to diffusers names. A no-op on
/// a checkpoint that is already diffusers-named.
///
/// Only the map keys change; the data stays where it is in the mmap. The one
/// shape difference — the reference stores each block's six modulation
/// vectors flat as `[6 * d]` against diffusers' `[6, d]` — is the same bytes
/// in the same order, and every loader here checks element counts rather
/// than ranks.
pub fn canonicalize(comp: &mut ComponentTensors) {
    let Some(prefix) = transformer_prefix(comp) else { return };
    let original = comp.has(&format!("{prefix}{ORIGINAL_MARKER}"));
    let old = std::mem::take(&mut comp.tensors);
    comp.tensors = old
        .into_iter()
        .filter_map(|(name, r)| {
            let key = name.strip_prefix(prefix)?;
            Some((if original { original_to_diffusers(key) } else { key.to_string() }, r))
        })
        .collect();
}

const PREFIX_RENAMES: [(&str, &str); 12] = [
    ("first.", "img_in."),
    ("tmlp.0.", "time_embed.linear_1."),
    ("tmlp.2.", "time_embed.linear_2."),
    ("tproj.1.", "time_mod_proj."),
    ("txtmlp.0.scale", "txt_in.norm.weight"),
    ("txtmlp.1.", "txt_in.linear_1."),
    ("txtmlp.3.", "txt_in.linear_2."),
    ("txtfusion.", "text_fusion."),
    ("blocks.", "transformer_blocks."),
    ("last.linear.", "final_layer.linear."),
    ("last.norm.scale", "final_layer.norm.weight"),
    ("last.modulation.lin", "final_layer.scale_shift_table"),
];

const BLOCK_RENAMES: [(&str, &str); 11] = [
    (".attn.wq.", ".attn.to_q."),
    (".attn.wk.", ".attn.to_k."),
    (".attn.wv.", ".attn.to_v."),
    (".attn.wo.", ".attn.to_out.0."),
    (".attn.gate.", ".attn.to_gate."),
    (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"),
    (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"),
    (".mlp.", ".ff."),
    (".prenorm.scale", ".norm1.weight"),
    (".postnorm.scale", ".norm2.weight"),
    (".mod.lin", ".scale_shift_table"),
];

/// Map one reference name to its diffusers equivalent.
pub fn original_to_diffusers(name: &str) -> String {
    let mut key = name.strip_prefix("model.diffusion_model.").unwrap_or(name).to_string();
    if let Some((old, new)) = PREFIX_RENAMES.iter().find(|(old, _)| key.starts_with(old)) {
        key = format!("{new}{}", &key[old.len()..]);
    }
    for (old, new) in BLOCK_RENAMES {
        key = key.replace(old, new);
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_and_normalizes_checkpoint_layouts_without_copying_weights() {
        for prefix in CHECKPOINT_PREFIXES {
            for original in [false, true] {
                let mut comp = ComponentTensors {
                    mmaps: Vec::new(), owned: Vec::new(), tensors: Default::default(),
                };
                let marker = if original { ORIGINAL_MARKER } else { DIFFUSERS_MARKER };
                let weight = if original { "first.weight" } else { "img_in.weight" };
                for name in [marker, weight] {
                    comp.append_synth_tensor(&format!("{prefix}{name}"), safetensors::Dtype::F32,
                        vec![1], &1.25_f32.to_le_bytes()).unwrap();
                }
                assert!(is_krea2_transformer(&comp), "prefix={prefix} original={original}");
                let ptr = comp.owned.as_ptr();
                canonicalize(&mut comp);
                assert!(comp.has(DIFFUSERS_MARKER));
                assert_eq!(comp.get_f32("img_in.weight").unwrap(), vec![1.25]);
                assert_eq!(comp.owned.as_ptr(), ptr);
                canonicalize(&mut comp);
                assert_eq!(comp.tensors.len(), 2);
                assert_eq!(comp.get_f32("img_in.weight").unwrap(), vec![1.25]);
            }
        }
    }

    #[test]
    fn leaves_other_model_families_untouched() {
        let mut comp = ComponentTensors {
            mmaps: Vec::new(), owned: Vec::new(), tensors: Default::default(),
        };
        comp.append_synth_tensor("double_blocks.0.img_attn.qkv.weight", safetensors::Dtype::F32,
            vec![1], &1.0_f32.to_le_bytes()).unwrap();
        assert!(!is_krea2_transformer(&comp));
        canonicalize(&mut comp);
        assert!(comp.has("double_blocks.0.img_attn.qkv.weight"));
    }

    #[test]
    fn renames_match_the_diffusers_converter() {
        let cases = [
            ("first.weight", "img_in.weight"),
            ("tmlp.2.bias", "time_embed.linear_2.bias"),
            ("tproj.1.weight", "time_mod_proj.weight"),
            ("txtmlp.0.scale", "txt_in.norm.weight"),
            ("txtmlp.3.weight", "txt_in.linear_2.weight"),
            ("txtfusion.projector.weight", "text_fusion.projector.weight"),
            (
                "txtfusion.layerwise_blocks.1.attn.qknorm.knorm.scale",
                "text_fusion.layerwise_blocks.1.attn.norm_k.weight",
            ),
            ("txtfusion.refiner_blocks.0.mlp.down.weight", "text_fusion.refiner_blocks.0.ff.down.weight"),
            ("blocks.27.mod.lin", "transformer_blocks.27.scale_shift_table"),
            ("blocks.3.attn.wo.weight", "transformer_blocks.3.attn.to_out.0.weight"),
            ("blocks.3.attn.gate.weight", "transformer_blocks.3.attn.to_gate.weight"),
            ("blocks.0.prenorm.scale", "transformer_blocks.0.norm1.weight"),
            ("last.modulation.lin", "final_layer.scale_shift_table"),
            ("last.norm.scale", "final_layer.norm.weight"),
            ("model.diffusion_model.last.linear.bias", "final_layer.linear.bias"),
        ];
        for (orig, want) in cases {
            assert_eq!(original_to_diffusers(orig), want, "{orig}");
        }
    }
}
