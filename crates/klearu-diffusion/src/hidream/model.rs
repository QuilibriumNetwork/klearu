//! HiDream-O1 Qwen3-VL decoder + text-to-image forward (CPU reference).
//!
//! The forward mirrors `_forward_generation` in the reference for the
//! text-to-image case (no reference images → vision encoder + DeepStack are
//! skipped):
//!   1. embed the text tokens
//!   2. overwrite the `<tms>` token's embedding with the timestep embedding
//!   3. patch-embed the noised image `z` and append it to the sequence
//!   4. run the 36 Qwen3-VL decoder layers under the hybrid attention mask
//!      (text = causal prefix; `<tms>`+image tokens = bidirectional)
//!   5. final-norm, then unembed at the image positions → predicted x₀ patches

use crate::hidream::config::{HiDreamConfig, FREQ_EMBED_SIZE, PATCH_DIM, TMS_TOKEN_ID};
use crate::hidream::layers::{
    silu, BottleneckPatchEmbed, FinalLayer, Linear, RmsNorm, TimestepEmbedder,
};
use crate::hidream::mrope::{self, RopeTables};
use crate::weight::loader::ComponentTensors;
use crate::Result;

/// Fixed-point anchor for image RoPE positions (reference `fix_point=4096`):
/// image-grid positions are offset by this constant so an image always sits at
/// the same rotary coordinates regardless of prompt length.
pub const ROPE_FIX_POINT: i64 = 4096;

/// One Qwen3-VL grouped-query attention block with per-head q/k RMSNorm.
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: RmsNorm, // over head_dim
    pub k_norm: RmsNorm, // over head_dim
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    pub fn new(cfg: &crate::hidream::config::TextConfig) -> Self {
        let h = cfg.hidden_size;
        let dh = cfg.head_dim();
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let bias = cfg.attention_bias;
        Self {
            q_proj: Linear::new(h, nh * dh, bias),
            k_proj: Linear::new(h, nkv * dh, bias),
            v_proj: Linear::new(h, nkv * dh, bias),
            o_proj: Linear::new(nh * dh, h, bias),
            q_norm: RmsNorm::new(dh, cfg.rms_norm_eps),
            k_norm: RmsNorm::new(dh, cfg.rms_norm_eps),
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: dh,
        }
    }

    /// `hidden`: [n, hidden_size]. `gen_mask[i]` true → token i attends
    /// bidirectionally to all; false → causal (attends only to j ≤ i).
    pub fn forward(
        &self,
        hidden: &[f32],
        n: usize,
        rope: &RopeTables,
        gen_mask: &[bool],
    ) -> Vec<f32> {
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let dh = self.head_dim;
        let n_rep = nh / nkv;

        // Projections, then per-head q/k norm (applied before RoPE).
        let mut q = self.q_proj.forward_vec(hidden, n); // [n, nh*dh]
        let mut k = self.k_proj.forward_vec(hidden, n); // [n, nkv*dh]
        let v = self.v_proj.forward_vec(hidden, n); // [n, nkv*dh]
        self.q_norm.apply_per_head(&mut q, n, nh);
        self.k_norm.apply_per_head(&mut k, n, nkv);

        // RoPE on q (nh heads) and k (nkv heads).
        mrope::apply(&mut q, rope, n, nh);
        mrope::apply(&mut k, rope, n, nkv);

        let scale = 1.0_f32 / (dh as f32).sqrt();
        let mut out = vec![0.0_f32; n * nh * dh];

        use rayon::prelude::*;
        out.par_chunks_mut(nh * dh)
            .enumerate()
            .for_each(|(qi, out_row)| {
                let bidir = gen_mask[qi];
                let last = if bidir { n } else { qi + 1 };
                for h in 0..nh {
                    let kv_h = h / n_rep;
                    let q_off = (qi * nh + h) * dh;
                    // scores over allowed keys.
                    let mut scores = vec![f32::NEG_INFINITY; n];
                    let mut max = f32::NEG_INFINITY;
                    for ki in 0..last {
                        let k_off = (ki * nkv + kv_h) * dh;
                        let mut s = 0.0_f32;
                        for d in 0..dh {
                            s += q[q_off + d] * k[k_off + d];
                        }
                        s *= scale;
                        scores[ki] = s;
                        if s > max {
                            max = s;
                        }
                    }
                    let mut sum = 0.0_f32;
                    for s in scores.iter_mut().take(last) {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    let o_off = h * dh;
                    for ki in 0..last {
                        let w = scores[ki] * inv;
                        if w == 0.0 {
                            continue;
                        }
                        let v_off = (ki * nkv + kv_h) * dh;
                        for d in 0..dh {
                            out_row[o_off + d] += w * v[v_off + d];
                        }
                    }
                }
            });

        self.o_proj.forward_vec(&out, n)
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        self.q_proj.load(comp, &format!("{prefix}.q_proj"))?;
        self.k_proj.load(comp, &format!("{prefix}.k_proj"))?;
        self.v_proj.load(comp, &format!("{prefix}.v_proj"))?;
        self.o_proj.load(comp, &format!("{prefix}.o_proj"))?;
        self.q_norm.load(comp, &format!("{prefix}.q_norm.weight"))?;
        self.k_norm.load(comp, &format!("{prefix}.k_norm.weight"))?;
        Ok(())
    }
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Mlp {
    pub fn new(hidden: usize, intermediate: usize) -> Self {
        Self {
            gate_proj: Linear::new(hidden, intermediate, false),
            up_proj: Linear::new(hidden, intermediate, false),
            down_proj: Linear::new(intermediate, hidden, false),
        }
    }

    pub fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut g = self.gate_proj.forward_vec(x, n);
        let u = self.up_proj.forward_vec(x, n);
        for (gi, ui) in g.iter_mut().zip(u.iter()) {
            *gi = silu(*gi) * ui;
        }
        self.down_proj.forward_vec(&g, n)
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        self.gate_proj.load(comp, &format!("{prefix}.gate_proj"))?;
        self.up_proj.load(comp, &format!("{prefix}.up_proj"))?;
        self.down_proj.load(comp, &format!("{prefix}.down_proj"))?;
        Ok(())
    }
}

/// Pre-norm Qwen3-VL decoder layer.
pub struct DecoderLayer {
    pub input_layernorm: RmsNorm,
    pub self_attn: Attention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Mlp,
}

impl DecoderLayer {
    pub fn new(cfg: &crate::hidream::config::TextConfig) -> Self {
        Self {
            input_layernorm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps),
            self_attn: Attention::new(cfg),
            post_attention_layernorm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps),
            mlp: Mlp::new(cfg.hidden_size, cfg.intermediate_size),
        }
    }

    pub fn forward(&self, hidden: &mut [f32], n: usize, rope: &RopeTables, gen_mask: &[bool]) {
        let hsz = self.input_layernorm.weight.len();
        // Attention sub-block.
        let mut normed = hidden.to_vec();
        self.input_layernorm.apply(&mut normed, n);
        let attn = self.self_attn.forward(&normed, n, rope, gen_mask);
        for (h, a) in hidden.iter_mut().zip(attn.iter()) {
            *h += a;
        }
        // MLP sub-block.
        let mut normed2 = hidden.to_vec();
        self.post_attention_layernorm.apply(&mut normed2, n);
        let mlp = self.mlp.forward(&normed2, n);
        for (h, m) in hidden.iter_mut().zip(mlp.iter()) {
            *h += m;
        }
        debug_assert_eq!(hidden.len(), n * hsz);
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        self.input_layernorm
            .load(comp, &format!("{prefix}.input_layernorm.weight"))?;
        self.self_attn.load(comp, &format!("{prefix}.self_attn"))?;
        self.post_attention_layernorm
            .load(comp, &format!("{prefix}.post_attention_layernorm.weight"))?;
        self.mlp.load(comp, &format!("{prefix}.mlp"))?;
        Ok(())
    }
}

/// 3-axis position ids for the text-to-image case, row-major `[3, seq]`.
///
/// Text tokens get sequential `(i,i,i)`; the `image_len` image tokens get grid
/// positions `(0, row, col)` offset by `ROPE_FIX_POINT`, where the grid is
/// flattened row-major (`row = j / w_patches`, `col = j % w_patches`).
pub fn build_position_ids_t2i(txt_seq_len: usize, h_patches: usize, w_patches: usize) -> Vec<i64> {
    let image_len = h_patches * w_patches;
    let seq = txt_seq_len + image_len;
    let mut pos = vec![0i64; 3 * seq];
    for i in 0..txt_seq_len {
        let p = i as i64;
        pos[i] = p; // t
        pos[seq + i] = p; // h
        pos[2 * seq + i] = p; // w
    }
    for j in 0..image_len {
        let idx = txt_seq_len + j;
        let row = (j / w_patches) as i64;
        let col = (j % w_patches) as i64;
        pos[idx] = ROPE_FIX_POINT; // t = 0 + fix_point
        pos[seq + idx] = row + ROPE_FIX_POINT;
        pos[2 * seq + idx] = col + ROPE_FIX_POINT;
    }
    pos
}

/// General multi-image 3-axis position ids (the reference
/// `get_rope_index_fix_point` with `spatial_merge_size=1`). Used for
/// reference-image editing, where the sequence interleaves VLM reference-image
/// tokens (normal sequential positions) with the target + pixel-reference image
/// grids (anchored at `ROPE_FIX_POINT`).
///
/// - `input_ids`: the position sequence (`input_ids_pad` — text + vision tokens)
/// - `grids`: each image's `(t, h, w)` grid in **token units**, in order of
///   appearance (VLM refs pre-divided by spatial_merge; target/pixel-refs at
///   patch-32 grid).
/// - `skip_vision_start[i]`: 0 = the image's `<vision_start>` counts as text
///   (VLM refs); 1 = the image is fix-point-anchored (target + pixel refs).
///
/// Returns row-major `[3, seq]`.
pub fn build_position_ids_multi(
    input_ids: &[u32],
    grids: &[(usize, usize, usize)],
    skip_vision_start: &[usize],
    image_token_id: u32,
    fix_point_init: i64,
) -> Vec<i64> {
    let seq = input_ids.len();
    // Accumulate position blocks as [3][len].
    let mut blocks: Vec<[Vec<i64>; 3]> = Vec::new();
    let mut st = 0usize;
    let mut fix_point = fix_point_init;
    let block_max = |b: &[Vec<i64>; 3]| -> i64 {
        b.iter().flat_map(|axis| axis.iter().copied()).max().unwrap_or(-1)
    };
    let push_text = |blocks: &mut Vec<[Vec<i64>; 3]>, text_len: usize, st_idx: i64| {
        let row: Vec<i64> = (0..text_len).map(|j| st_idx + j as i64).collect();
        blocks.push([row.clone(), row.clone(), row]);
    };

    for (img_i, &(gt, gh, gw)) in grids.iter().enumerate() {
        // ed = next image_token at/after st.
        let ed = (st..seq)
            .find(|&j| input_ids[j] == image_token_id)
            .unwrap_or(seq + 1);
        let mut text_len = ed.saturating_sub(st) as i64 - skip_vision_start[img_i] as i64;
        if text_len < 0 {
            text_len = 0;
        }
        let st_idx = blocks.last().map(|b| block_max(b) + 1).unwrap_or(0);
        push_text(&mut blocks, text_len as usize, st_idx);

        let n = gt * gh * gw;
        let mut ti = vec![0i64; n];
        let mut hi = vec![0i64; n];
        let mut wi = vec![0i64; n];
        for k in 0..n {
            ti[k] = (k / (gh * gw)) as i64;
            hi[k] = ((k / gw) % gh) as i64;
            wi[k] = (k % gw) as i64;
        }
        let offset = if skip_vision_start[img_i] != 0 {
            if fix_point > 0 {
                fix_point -= st_idx;
            }
            let off = fix_point + st_idx;
            fix_point = 0;
            off
        } else {
            text_len + st_idx
        };
        for k in 0..n {
            ti[k] += offset;
            hi[k] += offset;
            wi[k] += offset;
        }
        blocks.push([ti, hi, wi]);
        st = ed + n;
    }

    if st < seq {
        let st_idx = blocks.last().map(|b| block_max(b) + 1).unwrap_or(0);
        push_text(&mut blocks, seq - st, st_idx);
    }

    // Concatenate blocks into [3, seq].
    let mut pos = vec![0i64; 3 * seq];
    let mut col = 0usize;
    for b in &blocks {
        let len = b[0].len();
        for axis in 0..3 {
            pos[axis * seq + col..axis * seq + col + len].copy_from_slice(&b[axis]);
        }
        col += len;
    }
    debug_assert_eq!(col, seq, "position blocks must cover the sequence");
    pos
}

/// The full HiDream-O1 generation model: Qwen3-VL text backbone plus the three
/// pixel-generation modules.
pub struct HiDreamModel {
    pub cfg: HiDreamConfig,
    pub embed_tokens: Vec<f32>, // [vocab, hidden]
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub x_embedder: BottleneckPatchEmbed,
    pub t_embedder1: TimestepEmbedder,
    pub final_layer2: FinalLayer,
}

impl HiDreamModel {
    pub fn new(cfg: HiDreamConfig) -> Self {
        let tc = &cfg.text_config;
        let hidden = tc.hidden_size;
        let layers = (0..tc.num_hidden_layers).map(|_| DecoderLayer::new(tc)).collect();
        let bottleneck = hidden / 4;
        Self {
            embed_tokens: vec![0.0; tc.vocab_size * hidden],
            layers,
            norm: RmsNorm::new(hidden, tc.rms_norm_eps),
            x_embedder: BottleneckPatchEmbed::new(PATCH_DIM, bottleneck, hidden),
            t_embedder1: TimestepEmbedder::new(FREQ_EMBED_SIZE, hidden),
            final_layer2: FinalLayer::new(hidden, PATCH_DIM),
            cfg,
        }
    }

    fn hidden(&self) -> usize {
        self.cfg.text_config.hidden_size
    }

    fn embed_row(&self, token_id: u32) -> &[f32] {
        let h = self.hidden();
        let off = token_id as usize * h;
        &self.embed_tokens[off..off + h]
    }

    /// One denoising forward for text-to-image. Returns the predicted clean
    /// image (x₀) as patch tokens `[image_len, PATCH_DIM]`.
    ///
    /// - `text_ids`: the text prompt tokens (chat template + `<|boi|>` + `<tms>`)
    /// - `z`: noised image patches `[image_len, PATCH_DIM]`
    /// - `timestep`: scalar `t_pixeldit = 1 - step_t/1000`
    /// - `h_patches`/`w_patches`: image grid (`H/32`, `W/32`)
    pub fn forward_t2i(
        &self,
        text_ids: &[u32],
        z: &[f32],
        timestep: f32,
        h_patches: usize,
        w_patches: usize,
    ) -> Vec<f32> {
        let hidden = self.hidden();
        let txt_len = text_ids.len();
        let image_len = h_patches * w_patches;
        let seq = txt_len + image_len;
        debug_assert_eq!(z.len(), image_len * PATCH_DIM);

        // 1. Embed text tokens.
        let mut states = vec![0.0_f32; seq * hidden];
        for (i, &id) in text_ids.iter().enumerate() {
            states[i * hidden..(i + 1) * hidden].copy_from_slice(self.embed_row(id));
        }

        // 2. Overwrite the <tms> token embedding with the timestep embedding.
        let t_emb = self.t_embedder1.forward(timestep);
        for (i, &id) in text_ids.iter().enumerate() {
            if id == TMS_TOKEN_ID {
                states[i * hidden..(i + 1) * hidden].copy_from_slice(&t_emb);
            }
        }

        // 3. Patch-embed the noised image and append to the sequence.
        let v_emb = self.x_embedder.forward(z, image_len);
        states[txt_len * hidden..].copy_from_slice(&v_emb);

        // 4. Build masks + RoPE tables.
        //    gen_mask: the <tms> token (last text token) and all image tokens
        //    attend bidirectionally; the rest of the text is a causal prefix.
        let mut gen_mask = vec![false; seq];
        for (i, &id) in text_ids.iter().enumerate() {
            if id == TMS_TOKEN_ID {
                gen_mask[i] = true;
            }
        }
        for g in gen_mask.iter_mut().skip(txt_len) {
            *g = true;
        }

        let position_ids = build_position_ids_t2i(txt_len, h_patches, w_patches);
        let tc = &self.cfg.text_config;
        let rope = mrope::build_tables(
            &position_ids,
            seq,
            tc.head_dim(),
            tc.rope_theta,
            &tc.mrope_section(),
        );

        // 5. Decoder layers.
        for layer in &self.layers {
            layer.forward(&mut states, seq, &rope, &gen_mask);
        }
        self.norm.apply(&mut states, seq);

        // 6. Unembed at image positions only → predicted x₀ patches.
        let img_states = &states[txt_len * hidden..];
        self.final_layer2.forward(img_states, image_len)
    }

    /// Reference-editing forward (CPU oracle). Mirrors `_forward_generation`
    /// for the `ref_patches is not None` path. Takes the vision encoder's output
    /// (`vision_embeds` + `deepstack`) precomputed so the forward is validated
    /// independently of the (separately-validated) encoder.
    ///
    /// - `text_ids`: input_ids — chat template text with `image_token`
    ///   placeholders for the VLM reference tokens, ending in `<boi><tms>`.
    /// - `vision_embeds`: `[num_vlm, hidden]` scattered at the image_token
    ///   placeholders; `deepstack`: 3 × `[num_vlm, hidden]`.
    /// - `vinputs`: `[gen_len, PATCH_DIM]` = `cat(z_target, ref_patches)`.
    /// - `tgt_len`: target token count (output is read here).
    /// - `positions`: `[3, seq]` from [`build_position_ids_multi`].
    /// Returns predicted x₀ for the target `[tgt_len, PATCH_DIM]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_ref(
        &self,
        text_ids: &[u32],
        vision_embeds: &[f32],
        deepstack: &[Vec<f32>],
        vinputs: &[f32],
        gen_len: usize,
        tgt_len: usize,
        positions: &[i64],
        timestep: f32,
        image_token_id: u32,
        tms_token_id: u32,
    ) -> Vec<f32> {
        let hidden = self.hidden();
        let txt_len = text_ids.len();
        let seq = txt_len + gen_len;
        debug_assert_eq!(vinputs.len(), gen_len * PATCH_DIM);
        debug_assert_eq!(positions.len(), 3 * seq);

        let t_emb = self.t_embedder1.forward(timestep);
        let mut states = vec![0.0_f32; seq * hidden];

        // 1. Text region: embeddings, with vision_embeds at image placeholders
        //    and the timestep embedding at the <tms> token. Collect VLM positions.
        let mut vlm_positions = Vec::new();
        let mut vlm_idx = 0usize;
        for (i, &id) in text_ids.iter().enumerate() {
            let dst = &mut states[i * hidden..(i + 1) * hidden];
            if id == image_token_id {
                dst.copy_from_slice(&vision_embeds[vlm_idx * hidden..(vlm_idx + 1) * hidden]);
                vlm_positions.push(i);
                vlm_idx += 1;
            } else if id == tms_token_id {
                dst.copy_from_slice(&t_emb);
            } else {
                dst.copy_from_slice(self.embed_row(id));
            }
        }
        // 2. Gen region: x_embedder over (target z ++ ref patches).
        let v_emb = self.x_embedder.forward(vinputs, gen_len);
        states[txt_len * hidden..].copy_from_slice(&v_emb);

        // 3. Attention mask: text + VLM-image are causal; <tms> + target + ref
        //    are bidirectional ("gen").
        let mut gen_mask = vec![false; seq];
        for (i, &id) in text_ids.iter().enumerate() {
            if id == tms_token_id {
                gen_mask[i] = true;
            }
        }
        for g in gen_mask.iter_mut().skip(txt_len) {
            *g = true;
        }

        let tc = &self.cfg.text_config;
        let rope = mrope::build_tables(positions, seq, tc.head_dim(), tc.rope_theta, &tc.mrope_section());

        // 4. Decoder layers, with DeepStack injection at VLM positions (first 3).
        for (li, layer) in self.layers.iter().enumerate() {
            layer.forward(&mut states, seq, &rope, &gen_mask);
            if li < deepstack.len() {
                let ds = &deepstack[li];
                for (j, &p) in vlm_positions.iter().enumerate() {
                    let s = &mut states[p * hidden..(p + 1) * hidden];
                    for (x, d) in s.iter_mut().zip(&ds[j * hidden..(j + 1) * hidden]) {
                        *x += d;
                    }
                }
            }
        }
        self.norm.apply(&mut states, seq);

        // 5. Unembed at the target positions (first tgt_len of the gen region).
        let tgt = &states[txt_len * hidden..(txt_len + tgt_len) * hidden];
        self.final_layer2.forward(tgt, tgt_len)
    }

    /// Load all weights from a HiDream-O1 model directory's transformer
    /// component. Qwen3-VL names: `model.language_model.layers.{i}.*`,
    /// `model.language_model.embed_tokens.weight`, `model.language_model.norm`;
    /// the generation modules live at `model.{x_embedder,t_embedder1,final_layer2}`.
    pub fn load(&mut self, comp: &ComponentTensors) -> Result<()> {
        let lm = "model.language_model";
        self.embed_tokens = comp.get_f32(&format!("{lm}.embed_tokens.weight"))?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load(comp, &format!("{lm}.layers.{i}"))?;
        }
        self.norm.load(comp, &format!("{lm}.norm.weight"))?;
        self.x_embedder.load(comp, "model.x_embedder")?;
        self.t_embedder1.load(comp, "model.t_embedder1")?;
        self.final_layer2.load(comp, "model.final_layer2")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hidream::config::{RopeScaling, TextConfig};

    fn tiny_cfg() -> HiDreamConfig {
        HiDreamConfig {
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            text_config: TextConfig {
                hidden_size: 16,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                head_dim: 4,
                intermediate_size: 32,
                vocab_size: 200000,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                attention_bias: false,
                rope_scaling: Some(RopeScaling {
                    mrope_section: Some(vec![1, 1, 0]),
                    rope_type: None,
                }),
                max_position_embeddings: 4096,
            },
        }
    }

    #[test]
    fn position_ids_multi_matches_reference() {
        // Synthetic reference-editing sequence (see the Python dump): 1 VLM ref
        // (grid 2×2), target grid 3×3, pixel-ref grid 2×2.
        const VS: u32 = 151652;
        const IMG: u32 = 151655;
        const VE: u32 = 151653;
        const BOI: u32 = 151669;
        const TMS: u32 = 151673;
        let mut ids = vec![100, VS, IMG, IMG, IMG, IMG, VE, 200, 201, BOI, TMS];
        ids.push(VS);
        for _ in 0..8 {
            ids.push(IMG);
        }
        ids.push(VS);
        for _ in 0..3 {
            ids.push(IMG);
        }
        let grids = [(1, 2, 2), (1, 3, 3), (1, 2, 2)];
        let skip = [0, 1, 1];
        let pos = build_position_ids_multi(&ids, &grids, &skip, IMG, ROPE_FIX_POINT);
        let seq = ids.len();
        let t = &pos[0..seq];
        let h = &pos[seq..2 * seq];
        let w = &pos[2 * seq..3 * seq];
        let exp_t = [0i64, 1, 2, 2, 2, 2, 4, 5, 6, 7, 8, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4099, 4099, 4099, 4099];
        let exp_h = [0i64, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 4096, 4096, 4096, 4097, 4097, 4097, 4098, 4098, 4098, 4099, 4099, 4100, 4100];
        let exp_w = [0i64, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 4096, 4097, 4098, 4096, 4097, 4098, 4096, 4097, 4098, 4099, 4100, 4099, 4100];
        assert_eq!(t, &exp_t, "T axis");
        assert_eq!(h, &exp_h, "H axis");
        assert_eq!(w, &exp_w, "W axis");
    }

    #[test]
    fn position_ids_t2i_layout() {
        // txt_len=3, 2x2 grid → image_len=4, seq=7.
        let pos = build_position_ids_t2i(3, 2, 2);
        let seq = 7;
        // Text: sequential on all axes.
        assert_eq!(&pos[0..3], &[0, 1, 2]); // t row, text portion
        // Image t-axis all fixed-point.
        assert_eq!(pos[3], ROPE_FIX_POINT);
        // h-axis (row): grid rows 0,0,1,1 + fix_point.
        assert_eq!(pos[seq + 3], ROPE_FIX_POINT);
        assert_eq!(pos[seq + 3 + 3], ROPE_FIX_POINT + 1);
        // w-axis (col): grid cols 0,1,0,1 + fix_point.
        assert_eq!(pos[2 * seq + 3], ROPE_FIX_POINT);
        assert_eq!(pos[2 * seq + 4], ROPE_FIX_POINT + 1);
    }

    #[test]
    fn forward_t2i_shapes_and_finite() {
        let cfg = tiny_cfg();
        let mut model = HiDreamModel::new(cfg);
        // Random-ish weights so the forward isn't trivially zero.
        for (i, w) in model.embed_tokens.iter_mut().enumerate() {
            *w = ((i % 7) as f32 - 3.0) * 0.01;
        }
        let fill = |lin: &mut Linear, seed: f32| {
            for (i, w) in lin.weight.iter_mut().enumerate() {
                *w = ((i as f32 * seed).sin()) * 0.05;
            }
        };
        for layer in model.layers.iter_mut() {
            fill(&mut layer.self_attn.q_proj, 0.1);
            fill(&mut layer.self_attn.k_proj, 0.2);
            fill(&mut layer.self_attn.v_proj, 0.3);
            fill(&mut layer.self_attn.o_proj, 0.4);
            fill(&mut layer.mlp.gate_proj, 0.5);
            fill(&mut layer.mlp.up_proj, 0.6);
            fill(&mut layer.mlp.down_proj, 0.7);
        }
        fill(&mut model.x_embedder.proj1, 0.8);
        fill(&mut model.x_embedder.proj2, 0.9);
        fill(&mut model.final_layer2.linear, 0.15);
        fill(&mut model.t_embedder1.lin1, 0.25);
        fill(&mut model.t_embedder1.lin2, 0.35);

        let h_patches = 2;
        let w_patches = 2;
        let image_len = h_patches * w_patches;
        // text_ids: 2 normal tokens then the <tms> token.
        let text_ids = vec![5u32, 6u32, TMS_TOKEN_ID];
        let z = vec![0.01_f32; image_len * PATCH_DIM];
        let out = model.forward_t2i(&text_ids, &z, 0.5, h_patches, w_patches);
        assert_eq!(out.len(), image_len * PATCH_DIM);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
