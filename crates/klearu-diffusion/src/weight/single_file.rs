//! Single-file (all-in-one) checkpoint loader.
//!
//! Many SD checkpoints (Civitai, A1111-format, original CompVis releases)
//! ship as one `.safetensors` containing UNet + VAE + text_encoder weights
//! together, with the *original CompVis tensor naming* rather than the
//! HF Diffusers convention.
//!
//! This module provides `SingleFileLoader::open(path)` and
//! `loader.component(component_name)` which returns a `ComponentTensors`
//! whose `get_f32(diffusers_name)` works as if the checkpoint had been
//! pre-split into Diffusers subdirs. Implementation: a name-translation
//! table built once at open time, mapping `text_model.encoder.layers.0.…`
//! style names to `cond_stage_model.transformer.text_model.encoder.…`
//! style names.
//!
//! Status:
//!   - SD 1.5 / 2.x (single CLIP-L, no OpenCLIP) — supported
//!   - SDXL — partial: UNet + VAE + CLIP-L work; CLIP-G needs splitting
//!     `conditioner.embedders.1.model.transformer.resblocks.<n>.attn.in_proj_weight`
//!     into separate q/k/v tensors, which our existing TensorRef API can't
//!     do (it points at slices of mmap, can't synthesise a new tensor from
//!     a re-slice). SDXL CLIP-G returns NotImplemented.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::error::{DiffusionError, Result};
use crate::weight::loader::{ComponentTensors, TensorRef};

pub struct SingleFileLoader {
    pub mmap: Mmap,
    /// All tensor names present in the file → their (offset, dtype, shape).
    pub raw_tensors: HashMap<String, TensorRef>,
    pub variant: SDFormat,
    /// Standardized "modelspec.*" metadata read from the safetensors
    /// header. Civitai and A1111-style tooling populate this with
    /// architecture, prediction_type, resolution, title, license, etc.
    /// See https://github.com/Stability-AI/ModelSpec for the spec.
    pub metadata: ModelSpec,
}

/// Parsed modelspec metadata. All fields are optional (publishers may set
/// some, all, or none of them).
#[derive(Debug, Default, Clone)]
pub struct ModelSpec {
    pub architecture: Option<String>, // e.g. "stable-diffusion-v1-5" or "stable-diffusion-xl-base-v1-0"
    pub implementation: Option<String>,
    pub title: Option<String>,
    pub author: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub date: Option<String>,
    pub prediction_type: Option<String>, // "epsilon" / "v_prediction"
    pub resolution: Option<String>,      // "512x512" / "1024x1024"
    pub timestep_range: Option<String>,
    pub usage_hint: Option<String>,
    /// Any other keys we didn't promote to typed fields.
    pub raw: HashMap<String, String>,
}

impl ModelSpec {
    fn from_metadata(meta: &HashMap<String, String>) -> Self {
        let take = |k: &str| meta.get(k).cloned();
        Self {
            architecture: take("modelspec.architecture"),
            implementation: take("modelspec.implementation"),
            title: take("modelspec.title"),
            author: take("modelspec.author"),
            description: take("modelspec.description"),
            license: take("modelspec.license"),
            date: take("modelspec.date"),
            prediction_type: take("modelspec.prediction_type"),
            resolution: take("modelspec.resolution"),
            timestep_range: take("modelspec.timestep_range"),
            usage_hint: take("modelspec.usage_hint"),
            raw: meta.clone(),
        }
    }

    /// Map the architecture string to an SDFormat. Returns Unknown when
    /// the field is absent or doesn't match a known variant.
    pub fn architecture_variant(&self) -> SDFormat {
        match self.architecture.as_deref() {
            Some(a) => {
                let a = a.to_ascii_lowercase();
                if a.contains("xl") { SDFormat::Sdxl }
                else if a.contains("stable-diffusion") || a.contains("sd-") || a.contains("sd-v") {
                    SDFormat::Sd15
                } else { SDFormat::Unknown }
            }
            None => SDFormat::Unknown,
        }
    }

    /// Width or height in pixels, parsed from "<W>x<H>" strings.
    /// Returns None when the field is absent or malformed.
    pub fn resolution_pixels(&self) -> Option<(u32, u32)> {
        let s = self.resolution.as_deref()?;
        let (w, h) = s.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SDFormat {
    Sd15,    // CompVis SD 1.x or 2.x with single CLIP-L
    Sdxl,    // CompVis SDXL with CLIP-L + OpenCLIP CLIP-G
    Unknown,
}

impl SingleFileLoader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // First read the metadata-only header (cheap; doesn't deserialize tensor slices).
        let (_header_size, raw_metadata) = SafeTensors::read_metadata(&mmap)?;
        // The safetensors crate exposes the JSON metadata dict via .metadata()
        // (Option<HashMap<String, String>>). Default to empty when absent.
        let metadata_map: HashMap<String, String> = raw_metadata
            .metadata()
            .clone()
            .unwrap_or_default();
        let metadata = ModelSpec::from_metadata(&metadata_map);

        // Now deserialize tensors (uses the same mmap, no extra IO).
        let st = SafeTensors::deserialize(&mmap)?;
        let mut raw_tensors = HashMap::new();
        for name in st.names() {
            let tensor = st.tensor(name)?;
            let data = tensor.data();
            let base = mmap.as_ptr() as usize;
            let dptr = data.as_ptr() as usize;
            let offset = dptr - base;
            raw_tensors.insert(name.to_string(), TensorRef {
                mmap_index: 0,
                offset,
                end: offset + data.len(),
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
            });
        }

        // Variant detection: prefer modelspec.architecture, fall back to tensor-name heuristic.
        let variant = match metadata.architecture_variant() {
            SDFormat::Unknown => {
                if raw_tensors.contains_key("conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight") {
                    SDFormat::Sdxl
                } else if raw_tensors.contains_key("cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight") {
                    SDFormat::Sd15
                } else {
                    SDFormat::Unknown
                }
            }
            v => v,
        };

        Ok(Self { mmap, raw_tensors, variant, metadata })
    }

    /// Build a virtual ComponentTensors for the named component
    /// ("unet" / "vae" / "text_encoder" / "text_encoder_2"), using the
    /// CompVis-to-Diffusers name translation table appropriate for the
    /// detected variant.
    pub fn component(&self, name: &str) -> Result<ComponentTensors> {
        let mappings = match (self.variant, name) {
            (SDFormat::Sd15, "unet") => sd15_unet_mappings(),
            (SDFormat::Sd15, "vae") => sd_vae_mappings(),
            (SDFormat::Sd15, "text_encoder") => sd15_clip_l_mappings(),
            (SDFormat::Sdxl, "unet") => sdxl_unet_mappings(),
            (SDFormat::Sdxl, "vae") => sd_vae_mappings(),
            (SDFormat::Sdxl, "text_encoder") => sdxl_clip_l_mappings(),
            (SDFormat::Sdxl, "text_encoder_2") => sdxl_clip_g_mappings(),
            _ => {
                return Err(DiffusionError::Unsupported(
                    format!("no single-file mapping for variant={:?}, component={name}", self.variant)
                ));
            }
        };

        // Build a translated tensor map. For each diffusers name, look up the
        // corresponding compvis name in the file. Skip if missing (will surface
        // as MissingTensor on later get_f32 calls — same UX as the directory loader).
        let mut translated: HashMap<String, TensorRef> = HashMap::new();
        for (diffusers_name, source) in mappings.iter() {
            match source {
                Mapping::Alias(compvis_name) => {
                    if let Some(r) = self.raw_tensors.get(compvis_name.as_str()) {
                        translated.insert(diffusers_name.clone(), r.clone());
                    }
                }
                // Synth mappings are handled below with file-extending logic.
                Mapping::Synth(_) => {}
            }
        }

        let mut comp = ComponentTensors::from_single_mmap_borrow(self, translated);

        // Now process synth mappings: read the source tensor's bytes, split or
        // reshape, append to comp's owned buffer (via a second helper).
        for (diffusers_name, source) in mappings.iter() {
            if let Mapping::Synth(spec) = source {
                self.apply_synth(spec, diffusers_name, &mut comp)?;
            }
        }

        Ok(comp)
    }

    /// Apply a synthesised tensor spec by reading source bytes from the
    /// single-file mmap, transforming them, and appending the result to
    /// the component's owned mmap-backed buffer via in-memory tensor
    /// injection.
    fn apply_synth(
        &self,
        spec: &SynthSpec,
        target_name: &str,
        comp: &mut ComponentTensors,
    ) -> Result<()> {
        match spec {
            SynthSpec::SplitQKV { source, slice, expected_dtype } => {
                let src = self.raw_tensors.get(source.as_str())
                    .ok_or_else(|| DiffusionError::MissingTensor(source.clone()))?;
                if src.dtype != *expected_dtype {
                    // Permit f16/bf16 sources too; convert via existing get_f32 path.
                }
                let src_bytes = &self.mmap[src.offset..src.end];
                let bytes_per_elem = bytes_per_element(src.dtype)?;
                // attn.in_proj_weight has shape [3 * embed_dim, embed_dim] (or [3*embed_dim] for bias).
                // We split along the leading axis into three equal chunks.
                let total_elems: usize = src.shape.iter().product();
                if !total_elems.is_multiple_of(3) {
                    return Err(DiffusionError::ShapeMismatch {
                        expected: format!("{source}.shape[0] divisible by 3"),
                        got: format!("{:?}", src.shape),
                    });
                }
                let chunk_elems = total_elems / 3;
                let chunk_bytes = chunk_elems * bytes_per_elem;
                let start = match slice {
                    QKVSlice::Q => 0,
                    QKVSlice::K => 1,
                    QKVSlice::V => 2,
                } * chunk_bytes;
                let chunk_data = &src_bytes[start..start + chunk_bytes];
                // New shape: leading axis becomes embed_dim, rest unchanged.
                let mut new_shape = src.shape.clone();
                new_shape[0] /= 3;
                comp.append_synth_tensor(target_name, src.dtype, new_shape, chunk_data)?;
            }
            SynthSpec::Transpose2D { source } => {
                let src = self.raw_tensors.get(source.as_str())
                    .ok_or_else(|| DiffusionError::MissingTensor(source.clone()))?;
                if src.shape.len() != 2 {
                    return Err(DiffusionError::ShapeMismatch {
                        expected: format!("{source}: 2-D tensor for Transpose2D"),
                        got: format!("{:?}", src.shape),
                    });
                }
                let rows = src.shape[0];
                let cols = src.shape[1];
                let bpe = bytes_per_element(src.dtype)?;
                let src_bytes = &self.mmap[src.offset..src.end];
                let mut t = vec![0u8; rows * cols * bpe];
                // Out shape: [cols, rows]. out[c, r] = in[r, c].
                for r in 0..rows {
                    for c in 0..cols {
                        let from = (r * cols + c) * bpe;
                        let to = (c * rows + r) * bpe;
                        t[to..to + bpe].copy_from_slice(&src_bytes[from..from + bpe]);
                    }
                }
                comp.append_synth_tensor(target_name, src.dtype, vec![cols, rows], &t)?;
            }
        }
        Ok(())
    }
}

/// One mapping from a diffusers name to a source. Most are `Alias`
/// (rename); CLIP-G's QKV needs `Synth` to slice an in_proj_weight.
#[derive(Debug, Clone)]
pub enum Mapping {
    Alias(String),
    Synth(SynthSpec),
}

#[derive(Debug, Clone)]
pub enum SynthSpec {
    /// Slice OpenCLIP's `attn.in_proj_weight` (or bias) into one of {Q, K, V}.
    SplitQKV {
        source: String,
        slice: QKVSlice,
        expected_dtype: safetensors::Dtype,
    },
    /// Transpose a 2-D tensor at load time. Used for OpenCLIP's
    /// `text_projection` raw `nn.Parameter` which is applied as
    /// `pooled @ M` (no transposition) — vs HF `nn.Linear` semantics
    /// (`pooled @ W^T`). Without this transpose we'd be computing the
    /// transpose of the intended projection, corrupting SDXL's pooled
    /// embedding and (via TextTimeEmbedding) every UNet timestep.
    Transpose2D {
        source: String,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum QKVSlice { Q, K, V }

fn bytes_per_element(d: safetensors::Dtype) -> Result<usize> {
    Ok(match d {
        safetensors::Dtype::F32 => 4,
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
        safetensors::Dtype::F64 => 8,
        other => return Err(DiffusionError::Unsupported(format!("dtype {other:?}"))),
    })
}

// ============================================================================
// SD 1.5 mapping tables (Diffusers name → CompVis name)
// ============================================================================

/// Generate UNet name mappings for SD 1.5.
/// Returns a Vec of (diffusers_name, compvis_name) pairs.
///
/// SD 1.5 UNet structure:
///   input_blocks.0  = conv_in
///   input_blocks.1,2 = down_block.0 resnets (with cross-attn at .1)
///   input_blocks.3  = down_block.0 downsample
///   input_blocks.4,5 = down_block.1 resnets+attn
///   input_blocks.6  = down_block.1 downsample
///   input_blocks.7,8 = down_block.2 resnets+attn
///   input_blocks.9  = down_block.2 downsample
///   input_blocks.10,11 = down_block.3 resnets (no attn — DownBlock2D)
///   middle_block.0,2 = mid_block resnets (.1 = attn)
///   output_blocks.0,1,2 = up_block.0 (UpBlock2D, 3 resnets)
///   output_blocks.3,4,5 = up_block.1 (CrossAttnUpBlock2D, 3 resnets+attn)
///   output_blocks.6,7,8 = up_block.2
///   output_blocks.9,10,11 = up_block.3
///   out.0 = conv_norm_out, out.2 = conv_out
fn sd15_unet_mappings() -> Vec<(String, Mapping)> {
    let mut m: Vec<(String, Mapping)> = Vec::new();
    let prefix = "model.diffusion_model";

    // Time embedding
    push_linear(&mut m, "time_embedding.linear_1", &format!("{prefix}.time_embed.0"));
    push_linear(&mut m, "time_embedding.linear_2", &format!("{prefix}.time_embed.2"));

    // conv_in
    push_conv(&mut m, "conv_in", &format!("{prefix}.input_blocks.0.0"));

    // Down-block layout for SD 1.5 (input block index → (down_block, resnet_idx, has_attn))
    // Pattern: 4 down blocks, each with 2 resnets + (optional) downsample.
    // Block 0: input_blocks 1,2 (with attn), input_blocks 3 (downsample)
    // Block 1: input_blocks 4,5 (with attn), input_blocks 6 (downsample)
    // Block 2: input_blocks 7,8 (with attn), input_blocks 9 (downsample)
    // Block 3: input_blocks 10,11 (NO attn), no downsample
    let down_layout: &[(&[(usize, usize, bool)], Option<usize>)] = &[
        (&[(0, 1, true),  (0, 2, true)],   Some(3)),
        (&[(1, 4, true),  (1, 5, true)],   Some(6)),
        (&[(2, 7, true),  (2, 8, true)],   Some(9)),
        (&[(3, 10, false), (3, 11, false)], None),
    ];
    for (resnets, downsample_ib) in down_layout {
        for (block_i, ib_i, has_attn) in resnets.iter() {
            let resnet_j = ib_i % 3 - 1; // 1→0, 2→1, 4→0, 5→1, etc. roughly
            // Map input_blocks.<ib_i>.0 → down_blocks.<block_i>.resnets.<idx_in_block>
            let resnet_in_block = if *ib_i == 1 || *ib_i == 4 || *ib_i == 7 || *ib_i == 10 { 0 } else { 1 };
            push_resnet(&mut m,
                &format!("down_blocks.{block_i}.resnets.{resnet_in_block}"),
                &format!("{prefix}.input_blocks.{ib_i}.0"));
            let _ = resnet_j;
            if *has_attn {
                push_transformer(&mut m,
                    &format!("down_blocks.{block_i}.attentions.{resnet_in_block}"),
                    &format!("{prefix}.input_blocks.{ib_i}.1"),
                    1, // SD 1.5 has 1 transformer block per attention
                );
            }
        }
        if let Some(ds) = downsample_ib {
            // Downsample: input_blocks.<ds>.0 → down_blocks.<block_i>.downsamplers.0.conv
            // (the contained conv is at .op)
            let block_i = down_layout.iter().position(|(_, x)| x.as_ref() == Some(ds)).unwrap();
            push_conv(&mut m,
                &format!("down_blocks.{block_i}.downsamplers.0.conv"),
                &format!("{prefix}.input_blocks.{ds}.0.op"));
        }
    }

    // Mid block: middle_block.0 = resnet, .1 = transformer, .2 = resnet
    push_resnet(&mut m, "mid_block.resnets.0", &format!("{prefix}.middle_block.0"));
    push_transformer(&mut m, "mid_block.attentions.0", &format!("{prefix}.middle_block.1"), 1);
    push_resnet(&mut m, "mid_block.resnets.1", &format!("{prefix}.middle_block.2"));

    // Up blocks (SD 1.5: 4 up blocks, 3 resnets each, with attn on blocks 1-3)
    // output_blocks.0,1,2 = up_block.0 (no attn — UpBlock2D)
    // output_blocks.3,4,5 = up_block.1 (CrossAttnUpBlock2D)
    // output_blocks.6,7,8 = up_block.2
    // output_blocks.9,10,11 = up_block.3
    // Each output block: .0 = resnet, .1 = (attn or upsample), .2 = upsample-or-nothing
    let up_layout: &[(usize, &[(usize, bool, bool)])] = &[
        // (block_i, [(ob_i, has_attn, has_upsample)])
        (0, &[(0, false, false), (1, false, false), (2, false, true)]),
        (1, &[(3, true, false), (4, true, false), (5, true, true)]),
        (2, &[(6, true, false), (7, true, false), (8, true, true)]),
        (3, &[(9, true, false), (10, true, false), (11, true, false)]),
    ];
    for (block_i, ob_specs) in up_layout {
        for (resnet_in_block, (ob_i, has_attn, has_upsample)) in ob_specs.iter().enumerate() {
            push_resnet(&mut m,
                &format!("up_blocks.{block_i}.resnets.{resnet_in_block}"),
                &format!("{prefix}.output_blocks.{ob_i}.0"));
            if *has_attn {
                push_transformer(&mut m,
                    &format!("up_blocks.{block_i}.attentions.{resnet_in_block}"),
                    &format!("{prefix}.output_blocks.{ob_i}.1"),
                    1,
                );
            }
            if *has_upsample {
                // Upsample is at the end of the output block. Its index varies:
                // for blocks with attn, it's at .2 (after attn); for no-attn, at .1.
                let ups_idx = if *has_attn { 2 } else { 1 };
                push_conv(&mut m,
                    &format!("up_blocks.{block_i}.upsamplers.0.conv"),
                    &format!("{prefix}.output_blocks.{ob_i}.{ups_idx}.conv"));
            }
        }
    }

    // out: GroupNorm + Conv
    push_group_norm(&mut m, "conv_norm_out", &format!("{prefix}.out.0"));
    push_conv(&mut m, "conv_out", &format!("{prefix}.out.2"));

    m
}

/// SDXL UNet mappings. SDXL UNet structure (CompVis name → Diffusers name):
///
///   block_out_channels = [320, 640, 1280]   (3 levels, not 4)
///   layers_per_block   = 2
///   transformer_layers_per_block = [1, 2, 10]   (varies by block)
///   down_block_types = [DownBlock2D, CrossAttnDownBlock2D, CrossAttnDownBlock2D]
///     ⇒ first down block has NO attention
///
///   input_blocks layout:
///     0 = conv_in
///     1, 2 = block 0 (resnets only, no attn)
///     3 = block 0 downsample (.0.op)
///     4, 5 = block 1 (resnet + transformer×2)
///     6 = block 1 downsample
///     7, 8 = block 2 (resnet + transformer×10)
///     (no downsample on last block)
///
///   middle_block: 0=resnet, 1=transformer×10, 2=resnet
///
///   output_blocks layout (mirrors input_blocks reverse):
///     0, 1, 2 = up block 0 (3 resnets + transformer×10, then upsample)
///     3, 4, 5 = up block 1 (3 resnets + transformer×2, then upsample)
///     6, 7, 8 = up block 2 (3 resnets, NO attn, no upsample)
///
///   label_emb.0 = add_embedding (size/crop conditioning MLP)
fn sdxl_unet_mappings() -> Vec<(String, Mapping)> {
    let mut m: Vec<(String, Mapping)> = Vec::new();
    let prefix = "model.diffusion_model";

    // Time embedding
    push_linear(&mut m, "time_embedding.linear_1", &format!("{prefix}.time_embed.0"));
    push_linear(&mut m, "time_embedding.linear_2", &format!("{prefix}.time_embed.2"));

    // Add embedding (SDXL size/crop conditioning) — CompVis stores under label_emb.0
    push_linear(&mut m, "add_embedding.linear_1", &format!("{prefix}.label_emb.0.0"));
    push_linear(&mut m, "add_embedding.linear_2", &format!("{prefix}.label_emb.0.2"));

    // conv_in
    push_conv(&mut m, "conv_in", &format!("{prefix}.input_blocks.0.0"));

    // Down blocks. Layout: (block_index, [(input_block_idx, has_attn, n_transformer_layers)], downsample_input_block)
    // SDXL: block 0 has no attn; blocks 1 and 2 have transformer with 2 / 10 layers respectively.
    let down_layout: &[(usize, &[(usize, bool, usize)], Option<usize>)] = &[
        (0, &[(1, false, 0), (2, false, 0)], Some(3)),
        (1, &[(4, true, 2),  (5, true, 2)],  Some(6)),
        (2, &[(7, true, 10), (8, true, 10)], None),
    ];
    for (block_i, resnets, downsample_ib) in down_layout {
        for (j, (ib_i, has_attn, n_layers)) in resnets.iter().enumerate() {
            push_resnet(&mut m,
                &format!("down_blocks.{block_i}.resnets.{j}"),
                &format!("{prefix}.input_blocks.{ib_i}.0"));
            if *has_attn {
                push_transformer(&mut m,
                    &format!("down_blocks.{block_i}.attentions.{j}"),
                    &format!("{prefix}.input_blocks.{ib_i}.1"),
                    *n_layers);
            }
        }
        if let Some(ds) = downsample_ib {
            push_conv(&mut m,
                &format!("down_blocks.{block_i}.downsamplers.0.conv"),
                &format!("{prefix}.input_blocks.{ds}.0.op"));
        }
    }

    // Mid block: resnet + transformer×10 + resnet
    push_resnet(&mut m, "mid_block.resnets.0", &format!("{prefix}.middle_block.0"));
    push_transformer(&mut m, "mid_block.attentions.0", &format!("{prefix}.middle_block.1"), 10);
    push_resnet(&mut m, "mid_block.resnets.1", &format!("{prefix}.middle_block.2"));

    // Up blocks. SDXL: block 0 has transformer×10, block 1 transformer×2, block 2 no attn.
    // Each output_block: .0 = resnet, .1 = (attn or upsample), .2 = upsample-or-nothing
    let up_layout: &[(usize, &[(usize, bool, bool, usize)])] = &[
        (0, &[(0, true,  false, 10), (1, true,  false, 10), (2, true,  true,  10)]),
        (1, &[(3, true,  false, 2),  (4, true,  false, 2),  (5, true,  true,  2)]),
        (2, &[(6, false, false, 0),  (7, false, false, 0),  (8, false, false, 0)]),
    ];
    for (block_i, ob_specs) in up_layout {
        for (resnet_in_block, (ob_i, has_attn, has_upsample, n_layers)) in ob_specs.iter().enumerate() {
            push_resnet(&mut m,
                &format!("up_blocks.{block_i}.resnets.{resnet_in_block}"),
                &format!("{prefix}.output_blocks.{ob_i}.0"));
            if *has_attn {
                push_transformer(&mut m,
                    &format!("up_blocks.{block_i}.attentions.{resnet_in_block}"),
                    &format!("{prefix}.output_blocks.{ob_i}.1"),
                    *n_layers);
            }
            if *has_upsample {
                let ups_idx = if *has_attn { 2 } else { 1 };
                push_conv(&mut m,
                    &format!("up_blocks.{block_i}.upsamplers.0.conv"),
                    &format!("{prefix}.output_blocks.{ob_i}.{ups_idx}.conv"));
            }
        }
    }

    // out: GroupNorm + Conv
    push_group_norm(&mut m, "conv_norm_out", &format!("{prefix}.out.0"));
    push_conv(&mut m, "conv_out", &format!("{prefix}.out.2"));

    m
}

fn sd_vae_mappings() -> Vec<(String, Mapping)> {
    let mut m: Vec<(String, Mapping)> = Vec::new();
    let prefix = "first_stage_model";

    // Encoder
    push_conv(&mut m, "encoder.conv_in", &format!("{prefix}.encoder.conv_in"));
    // 4 down blocks: encoder.down.<i>.block.<j> (resnets), encoder.down.<i>.downsample.conv
    for i in 0..4 {
        for j in 0..2 {
            push_vae_resnet(&mut m,
                &format!("encoder.down_blocks.{i}.resnets.{j}"),
                &format!("{prefix}.encoder.down.{i}.block.{j}"));
        }
        if i < 3 {
            push_conv(&mut m,
                &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                &format!("{prefix}.encoder.down.{i}.downsample.conv"));
        }
    }
    push_vae_resnet(&mut m, "encoder.mid_block.resnets.0", &format!("{prefix}.encoder.mid.block_1"));
    push_vae_attn(&mut m, "encoder.mid_block.attentions.0", &format!("{prefix}.encoder.mid.attn_1"));
    push_vae_resnet(&mut m, "encoder.mid_block.resnets.1", &format!("{prefix}.encoder.mid.block_2"));
    push_group_norm(&mut m, "encoder.conv_norm_out", &format!("{prefix}.encoder.norm_out"));
    push_conv(&mut m, "encoder.conv_out", &format!("{prefix}.encoder.conv_out"));
    push_conv(&mut m, "quant_conv", &format!("{prefix}.quant_conv"));

    // Decoder
    push_conv(&mut m, "decoder.conv_in", &format!("{prefix}.decoder.conv_in"));
    push_vae_resnet(&mut m, "decoder.mid_block.resnets.0", &format!("{prefix}.decoder.mid.block_1"));
    push_vae_attn(&mut m, "decoder.mid_block.attentions.0", &format!("{prefix}.decoder.mid.attn_1"));
    push_vae_resnet(&mut m, "decoder.mid_block.resnets.1", &format!("{prefix}.decoder.mid.block_2"));
    // Decoder up blocks (in reverse order vs encoder; decoder.up.<3-i> in single file)
    for i in 0..4 {
        let cv_i = 3 - i;
        for j in 0..3 {
            push_vae_resnet(&mut m,
                &format!("decoder.up_blocks.{i}.resnets.{j}"),
                &format!("{prefix}.decoder.up.{cv_i}.block.{j}"));
        }
        if i < 3 {
            push_conv(&mut m,
                &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                &format!("{prefix}.decoder.up.{cv_i}.upsample.conv"));
        }
    }
    push_group_norm(&mut m, "decoder.conv_norm_out", &format!("{prefix}.decoder.norm_out"));
    push_conv(&mut m, "decoder.conv_out", &format!("{prefix}.decoder.conv_out"));
    push_conv(&mut m, "post_quant_conv", &format!("{prefix}.post_quant_conv"));

    m
}

fn sd15_clip_l_mappings() -> Vec<(String, Mapping)> {
    clip_l_mappings_with_prefix("cond_stage_model.transformer")
}

fn sdxl_clip_l_mappings() -> Vec<(String, Mapping)> {
    // SDXL CLIP-L lives at conditioner.embedders.0.transformer.text_model.*
    clip_l_mappings_with_prefix("conditioner.embedders.0.transformer")
}

/// SDXL CLIP-G (OpenCLIP bigG/14) — different naming convention than CLIP-L.
/// Lives under `conditioner.embedders.1.model.*` with OpenCLIP's
/// transformer naming (resblocks, ln_1, ln_2, c_fc, c_proj, attn.in_proj_*).
/// Each resblock's QKV is packed into `attn.in_proj_weight`/`bias` and must
/// be split into three separate q/k/v tensors.
fn sdxl_clip_g_mappings() -> Vec<(String, Mapping)> {
    let mut m: Vec<(String, Mapping)> = Vec::new();
    let root = "conditioner.embedders.1.model";

    // Embeddings: OpenCLIP uses `token_embedding.weight` and `positional_embedding`
    // (the latter as a tensor, not nn.Embedding); we treat it as the position table.
    m.push((
        "text_model.embeddings.token_embedding.weight".into(),
        Mapping::Alias(format!("{root}.token_embedding.weight")),
    ));
    m.push((
        "text_model.embeddings.position_embedding.weight".into(),
        Mapping::Alias(format!("{root}.positional_embedding")),
    ));

    // 32 transformer resblocks for CLIP-G.
    for i in 0..32 {
        let cv = format!("{root}.transformer.resblocks.{i}");
        let dp = format!("text_model.encoder.layers.{i}");

        // Layer norms: ln_1 → layer_norm1, ln_2 → layer_norm2.
        push_layer_norm(&mut m, &format!("{dp}.layer_norm1"), &format!("{cv}.ln_1"));
        push_layer_norm(&mut m, &format!("{dp}.layer_norm2"), &format!("{cv}.ln_2"));

        // Self-attention QKV: split attn.in_proj_weight ([3*D, D]) and
        // attn.in_proj_bias ([3*D]) into three separate tensors.
        for (slice, qkv_name) in [(QKVSlice::Q, "q_proj"), (QKVSlice::K, "k_proj"), (QKVSlice::V, "v_proj")] {
            m.push((
                format!("{dp}.self_attn.{qkv_name}.weight"),
                Mapping::Synth(SynthSpec::SplitQKV {
                    source: format!("{cv}.attn.in_proj_weight"),
                    slice,
                    expected_dtype: safetensors::Dtype::F16, // common; overridden by actual dtype
                }),
            ));
            m.push((
                format!("{dp}.self_attn.{qkv_name}.bias"),
                Mapping::Synth(SynthSpec::SplitQKV {
                    source: format!("{cv}.attn.in_proj_bias"),
                    slice,
                    expected_dtype: safetensors::Dtype::F16,
                }),
            ));
        }
        // out_proj
        push_linear(&mut m, &format!("{dp}.self_attn.out_proj"), &format!("{cv}.attn.out_proj"));

        // MLP: c_fc → fc1, c_proj → fc2.
        push_linear(&mut m, &format!("{dp}.mlp.fc1"), &format!("{cv}.mlp.c_fc"));
        push_linear(&mut m, &format!("{dp}.mlp.fc2"), &format!("{cv}.mlp.c_proj"));
    }

    // Final layer norm.
    push_layer_norm(&mut m, "text_model.final_layer_norm", &format!("{root}.ln_final"));

    // Text projection (used by CLIPTextModel::pooled_forward for SDXL).
    // OpenCLIP saves this as a raw nn.Parameter [hidden, hidden] applied as
    // `pooled @ M`. Our Linear matches HF nn.Linear semantics (`pooled @ W^T`)
    // so we must transpose at load.
    m.push((
        "text_projection.weight".into(),
        Mapping::Synth(SynthSpec::Transpose2D {
            source: format!("{root}.text_projection"),
        }),
    ));

    m
}

fn clip_l_mappings_with_prefix(root: &str) -> Vec<(String, Mapping)> {
    let mut m: Vec<(String, Mapping)> = Vec::new();
    m.push(("text_model.embeddings.token_embedding.weight".into(),
        Mapping::Alias(format!("{root}.text_model.embeddings.token_embedding.weight"))));
    m.push(("text_model.embeddings.position_embedding.weight".into(),
        Mapping::Alias(format!("{root}.text_model.embeddings.position_embedding.weight"))));
    for i in 0..12 {
        let p = format!("{root}.text_model.encoder.layers.{i}");
        push_layer_norm(&mut m,
            &format!("text_model.encoder.layers.{i}.layer_norm1"),
            &format!("{p}.layer_norm1"));
        push_linear(&mut m,
            &format!("text_model.encoder.layers.{i}.self_attn.q_proj"),
            &format!("{p}.self_attn.q_proj"));
        push_linear(&mut m,
            &format!("text_model.encoder.layers.{i}.self_attn.k_proj"),
            &format!("{p}.self_attn.k_proj"));
        push_linear(&mut m,
            &format!("text_model.encoder.layers.{i}.self_attn.v_proj"),
            &format!("{p}.self_attn.v_proj"));
        push_linear(&mut m,
            &format!("text_model.encoder.layers.{i}.self_attn.out_proj"),
            &format!("{p}.self_attn.out_proj"));
        push_layer_norm(&mut m,
            &format!("text_model.encoder.layers.{i}.layer_norm2"),
            &format!("{p}.layer_norm2"));
        push_linear(&mut m,
            &format!("text_model.encoder.layers.{i}.mlp.fc1"),
            &format!("{p}.mlp.fc1"));
        push_linear(&mut m,
            &format!("text_model.encoder.layers.{i}.mlp.fc2"),
            &format!("{p}.mlp.fc2"));
    }
    push_layer_norm(&mut m, "text_model.final_layer_norm",
        &format!("{root}.text_model.final_layer_norm"));
    m
}

// ============================================================================
// Helpers — push (diffusers_name.weight, compvis_name.weight) and similarly for bias
// ============================================================================

fn push_linear(m: &mut Vec<(String, Mapping)>, diffusers: &str, compvis: &str) {
    m.push((format!("{diffusers}.weight"), Mapping::Alias(format!("{compvis}.weight"))));
    m.push((format!("{diffusers}.bias"),   Mapping::Alias(format!("{compvis}.bias"))));
}

fn push_conv(m: &mut Vec<(String, Mapping)>, diffusers: &str, compvis: &str) {
    m.push((format!("{diffusers}.weight"), Mapping::Alias(format!("{compvis}.weight"))));
    m.push((format!("{diffusers}.bias"),   Mapping::Alias(format!("{compvis}.bias"))));
}

fn push_group_norm(m: &mut Vec<(String, Mapping)>, diffusers: &str, compvis: &str) {
    m.push((format!("{diffusers}.weight"), Mapping::Alias(format!("{compvis}.weight"))));
    m.push((format!("{diffusers}.bias"),   Mapping::Alias(format!("{compvis}.bias"))));
}

fn push_layer_norm(m: &mut Vec<(String, Mapping)>, diffusers: &str, compvis: &str) {
    m.push((format!("{diffusers}.weight"), Mapping::Alias(format!("{compvis}.weight"))));
    m.push((format!("{diffusers}.bias"),   Mapping::Alias(format!("{compvis}.bias"))));
}

fn push_resnet(m: &mut Vec<(String, Mapping)>, diffusers: &str, compvis: &str) {
    push_group_norm(m, &format!("{diffusers}.norm1"), &format!("{compvis}.in_layers.0"));
    push_conv(m, &format!("{diffusers}.conv1"), &format!("{compvis}.in_layers.2"));
    push_linear(m, &format!("{diffusers}.time_emb_proj"), &format!("{compvis}.emb_layers.1"));
    push_group_norm(m, &format!("{diffusers}.norm2"), &format!("{compvis}.out_layers.0"));
    push_conv(m, &format!("{diffusers}.conv2"), &format!("{compvis}.out_layers.3"));
    // skip_connection (only when in_channels != out_channels — both names get added; missing
    // tensors are silently skipped during translation).
    push_conv(m, &format!("{diffusers}.conv_shortcut"), &format!("{compvis}.skip_connection"));
}

fn push_vae_resnet(m: &mut Vec<(String, Mapping)>, diffusers: &str, compvis: &str) {
    push_group_norm(m, &format!("{diffusers}.norm1"), &format!("{compvis}.norm1"));
    push_conv(m, &format!("{diffusers}.conv1"), &format!("{compvis}.conv1"));
    push_group_norm(m, &format!("{diffusers}.norm2"), &format!("{compvis}.norm2"));
    push_conv(m, &format!("{diffusers}.conv2"), &format!("{compvis}.conv2"));
    push_conv(m, &format!("{diffusers}.conv_shortcut"), &format!("{compvis}.nin_shortcut"));
}

fn push_vae_attn(m: &mut Vec<(String, Mapping)>, diffusers: &str, compvis: &str) {
    push_group_norm(m, &format!("{diffusers}.group_norm"), &format!("{compvis}.norm"));
    push_linear(m, &format!("{diffusers}.to_q"), &format!("{compvis}.q"));
    push_linear(m, &format!("{diffusers}.to_k"), &format!("{compvis}.k"));
    push_linear(m, &format!("{diffusers}.to_v"), &format!("{compvis}.v"));
    push_linear(m, &format!("{diffusers}.to_out.0"), &format!("{compvis}.proj_out"));
}

fn push_transformer(m: &mut Vec<(String, Mapping)>, diffusers: &str, compvis: &str, n_blocks: usize) {
    push_group_norm(m, &format!("{diffusers}.norm"), &format!("{compvis}.norm"));
    push_conv(m, &format!("{diffusers}.proj_in"), &format!("{compvis}.proj_in"));
    for k in 0..n_blocks {
        let dp = format!("{diffusers}.transformer_blocks.{k}");
        let cp = format!("{compvis}.transformer_blocks.{k}");
        push_layer_norm(m, &format!("{dp}.norm1"), &format!("{cp}.norm1"));
        push_linear(m, &format!("{dp}.attn1.to_q"), &format!("{cp}.attn1.to_q"));
        push_linear(m, &format!("{dp}.attn1.to_k"), &format!("{cp}.attn1.to_k"));
        push_linear(m, &format!("{dp}.attn1.to_v"), &format!("{cp}.attn1.to_v"));
        push_linear(m, &format!("{dp}.attn1.to_out.0"), &format!("{cp}.attn1.to_out.0"));
        push_layer_norm(m, &format!("{dp}.norm2"), &format!("{cp}.norm2"));
        push_linear(m, &format!("{dp}.attn2.to_q"), &format!("{cp}.attn2.to_q"));
        push_linear(m, &format!("{dp}.attn2.to_k"), &format!("{cp}.attn2.to_k"));
        push_linear(m, &format!("{dp}.attn2.to_v"), &format!("{cp}.attn2.to_v"));
        push_linear(m, &format!("{dp}.attn2.to_out.0"), &format!("{cp}.attn2.to_out.0"));
        push_layer_norm(m, &format!("{dp}.norm3"), &format!("{cp}.norm3"));
        push_linear(m, &format!("{dp}.ff.net.0.proj"), &format!("{cp}.ff.net.0.proj"));
        push_linear(m, &format!("{dp}.ff.net.2"), &format!("{cp}.ff.net.2"));
    }
    push_conv(m, &format!("{diffusers}.proj_out"), &format!("{compvis}.proj_out"));
}
