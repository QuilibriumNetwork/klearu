//! Qwen3-VL vision encoder for HiDream-O1 reference-image editing.
//!
//! Encodes reference images into LLM-space tokens (+ DeepStack features) that
//! are injected into the generation sequence. CPU reference implementation
//! (runs once per generation on small ~384px images, so it doesn't need the
//! GPU). Mirrors `Qwen3VLVisionModel` in the reference exactly:
//!   patch_embed (Conv3d ≡ Linear over 2×p×p patches) → + interpolated learned
//!   position embedding → 27 ViT blocks (LayerNorm, qkv attention with 2D RoPE,
//!   gelu-tanh MLP) → final merger (4 patches → out_hidden). DeepStack mergers
//!   at blocks [8,16,24] tap intermediate features.

use crate::hidream::layers::{gelu_erf, gelu_tanh, LayerNorm, Linear};
use crate::weight::loader::ComponentTensors;
use crate::Result;

/// A weight getter — abstracts over `ComponentTensors` (production) and a
/// safetensors dump (validation test).
type Getter<'a> = &'a dyn Fn(&str) -> Result<Vec<f32>>;

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
    /// Flattened patch dim fed to patch_embed: `in_ch · temporal · p · p`.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
    /// Side length of the learned position grid (√num_position_embeddings).
    pub fn num_grid_per_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt() as usize
    }
}

const LN_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10000.0;

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig) -> Self {
        let h = cfg.hidden_size;
        Self {
            norm1: LayerNorm::new(h, LN_EPS),
            norm2: LayerNorm::new(h, LN_EPS),
            qkv: Linear::new(h, 3 * h, true),
            proj: Linear::new(h, h, true),
            fc1: Linear::new(h, cfg.intermediate_size, true),
            fc2: Linear::new(cfg.intermediate_size, h, true),
        }
    }

    fn load(&mut self, get: Getter, prefix: &str) -> Result<()> {
        load_ln(&mut self.norm1, get, &format!("{prefix}.norm1"))?;
        load_ln(&mut self.norm2, get, &format!("{prefix}.norm2"))?;
        load_lin(&mut self.qkv, get, &format!("{prefix}.attn.qkv"))?;
        load_lin(&mut self.proj, get, &format!("{prefix}.attn.proj"))?;
        load_lin(&mut self.fc1, get, &format!("{prefix}.mlp.linear_fc1"))?;
        load_lin(&mut self.fc2, get, &format!("{prefix}.mlp.linear_fc2"))?;
        Ok(())
    }

    /// `hidden`: [n, hidden]; `cos`/`sin`: [n, head_dim] 2D-RoPE tables.
    fn forward(&self, hidden: &mut [f32], n: usize, nh: usize, dh: usize, cos: &[f32], sin: &[f32]) {
        let h = self.norm1.weight.len();
        // Attention sub-block.
        let mut normed = hidden.to_vec();
        self.norm1.apply(&mut normed, n);
        let qkv = self.qkv.forward_vec(&normed, n); // [n, 3h]
        // Split into q,k,v [n, nh, dh] (reference reshape(n,3,nh,dh).permute(1,0,2,3)).
        let mut q = vec![0.0_f32; n * h];
        let mut k = vec![0.0_f32; n * h];
        let mut v = vec![0.0_f32; n * h];
        for t in 0..n {
            for hh in 0..nh {
                for d in 0..dh {
                    let base = t * 3 * h;
                    q[(t * nh + hh) * dh + d] = qkv[base + hh * dh + d];
                    k[(t * nh + hh) * dh + d] = qkv[base + h + hh * dh + d];
                    v[(t * nh + hh) * dh + d] = qkv[base + 2 * h + hh * dh + d];
                }
            }
        }
        apply_rope_vision(&mut q, n, nh, dh, cos, sin);
        apply_rope_vision(&mut k, n, nh, dh, cos, sin);

        // Full bidirectional attention (single image = one chunk, no mask).
        let scale = 1.0 / (dh as f32).sqrt();
        let mut attn = vec![0.0_f32; n * h];
        use rayon::prelude::*;
        attn.par_chunks_mut(h).enumerate().for_each(|(qi, out_row)| {
            for hh in 0..nh {
                let q_off = (qi * nh + hh) * dh;
                let mut scores = vec![0.0_f32; n];
                let mut max = f32::NEG_INFINITY;
                for (ki, sc) in scores.iter_mut().enumerate() {
                    let k_off = (ki * nh + hh) * dh;
                    let mut s = 0.0;
                    for d in 0..dh {
                        s += q[q_off + d] * k[k_off + d];
                    }
                    s *= scale;
                    *sc = s;
                    if s > max {
                        max = s;
                    }
                }
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                let o_off = hh * dh;
                for (ki, &w) in scores.iter().enumerate() {
                    let w = w * inv;
                    let v_off = (ki * nh + hh) * dh;
                    for d in 0..dh {
                        out_row[o_off + d] += w * v[v_off + d];
                    }
                }
            }
        });
        let attn = self.proj.forward_vec(&attn, n);
        for (hi, a) in hidden.iter_mut().zip(attn.iter()) {
            *hi += a;
        }

        // MLP sub-block (gelu-tanh).
        let mut normed2 = hidden.to_vec();
        self.norm2.apply(&mut normed2, n);
        let mut mid = self.fc1.forward_vec(&normed2, n);
        for x in mid.iter_mut() {
            *x = gelu_tanh(*x);
        }
        let mlp = self.fc2.forward_vec(&mid, n);
        for (hi, m) in hidden.iter_mut().zip(mlp.iter()) {
            *hi += m;
        }
    }
}

/// PatchMerger: groups `merge²` patches, norms, and projects to out_hidden.
/// `use_postshuffle_norm` controls whether the LayerNorm is over the merged
/// (post-shuffle) dim or the per-patch hidden dim.
struct PatchMerger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    merged_dim: usize, // hidden · merge²
    use_postshuffle_norm: bool,
}

impl PatchMerger {
    fn new(cfg: &VisionConfig, use_postshuffle_norm: bool) -> Self {
        let merged = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let norm_dim = if use_postshuffle_norm { merged } else { cfg.hidden_size };
        Self {
            norm: LayerNorm::new(norm_dim, LN_EPS),
            fc1: Linear::new(merged, merged, true),
            fc2: Linear::new(merged, cfg.out_hidden_size, true),
            merged_dim: merged,
            use_postshuffle_norm,
        }
    }

    fn load(&mut self, get: Getter, prefix: &str) -> Result<()> {
        load_ln(&mut self.norm, get, &format!("{prefix}.norm"))?;
        load_lin(&mut self.fc1, get, &format!("{prefix}.linear_fc1"))?;
        load_lin(&mut self.fc2, get, &format!("{prefix}.linear_fc2"))?;
        Ok(())
    }

    /// `hidden`: [n_patches, per_patch_hidden]. Returns [n_patches/merge², out].
    fn forward(&self, hidden: &[f32], n_patches: usize, per_patch: usize) -> Vec<f32> {
        let group = self.merged_dim / per_patch; // merge²
        let n_merged = n_patches / group;
        // Group `group` consecutive patches into a merged row [merged_dim].
        let mut merged = if self.use_postshuffle_norm {
            // norm over the merged dim — view first, then norm.
            let mut m = hidden.to_vec();
            self.norm.apply(&mut m, n_merged); // n_merged rows of merged_dim
            m
        } else {
            // norm over the per-patch hidden dim — norm first, then view.
            let mut m = hidden.to_vec();
            self.norm.apply(&mut m, n_patches);
            m
        };
        // merged is now [n_merged, merged_dim] either way (same memory layout).
        let mut mid = self.fc1.forward_vec(&merged, n_merged);
        for x in mid.iter_mut() {
            *x = gelu_erf(*x);
        }
        merged.clear();
        self.fc2.forward_vec(&mid, n_merged)
    }
}

/// The Qwen3-VL vision encoder.
pub struct VisionEncoder {
    pub cfg: VisionConfig,
    patch_embed: Linear,    // patch_dim -> hidden
    pos_embed: Vec<f32>,    // [num_position_embeddings, hidden]
    rot_inv_freq: Vec<f32>, // [head_dim/4]
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    deepstack_mergers: Vec<PatchMerger>,
}

impl VisionEncoder {
    pub fn new(cfg: VisionConfig) -> Self {
        let rot_dim = cfg.head_dim() / 2; // Qwen3VLVisionRotaryEmbedding(head_dim/2)
        let rot_inv_freq: Vec<f32> = (0..rot_dim / 2)
            .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / rot_dim as f32))
            .collect();
        let blocks = (0..cfg.depth).map(|_| VisionBlock::new(&cfg)).collect();
        let deepstack_mergers =
            (0..cfg.deepstack_visual_indexes.len()).map(|_| PatchMerger::new(&cfg, true)).collect();
        Self {
            patch_embed: Linear::new(cfg.patch_dim(), cfg.hidden_size, true),
            pos_embed: vec![0.0; cfg.num_position_embeddings * cfg.hidden_size],
            rot_inv_freq,
            merger: PatchMerger::new(&cfg, false),
            deepstack_mergers,
            blocks,
            cfg,
        }
    }

    pub fn load(&mut self, comp: &ComponentTensors, prefix: &str) -> Result<()> {
        self.load_with(&|n| comp.get_f32(n), prefix)
    }

    /// Load via an arbitrary weight getter (used by the validation test).
    pub fn load_with(&mut self, get: Getter, prefix: &str) -> Result<()> {
        // patch_embed Conv3d weight [hidden, in, t, p, p] flattens to [hidden, patch_dim].
        self.patch_embed.weight = get(&format!("{prefix}.patch_embed.proj.weight"))?;
        self.patch_embed.bias = Some(get(&format!("{prefix}.patch_embed.proj.bias"))?);
        self.pos_embed = get(&format!("{prefix}.pos_embed.weight"))?;
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            blk.load(get, &format!("{prefix}.blocks.{i}"))?;
        }
        self.merger.load(get, &format!("{prefix}.merger"))?;
        for (i, m) in self.deepstack_mergers.iter_mut().enumerate() {
            m.load(get, &format!("{prefix}.deepstack_merger_list.{i}"))?;
        }
        Ok(())
    }

    /// Encode one image. `pixel_values`: `[n_patches, patch_dim]` in
    /// merge-block order (as the Qwen image processor produces). `grid` =
    /// `(t, h, w)` patch grid. Returns `(image_embeds [n_merged, out_hidden],
    /// deepstack_features Vec<[n_merged, out_hidden]>)`.
    pub fn forward(&self, pixel_values: &[f32], grid: (usize, usize, usize)) -> (Vec<f32>, Vec<Vec<f32>>) {
        let (t, gh, gw) = grid;
        let n = t * gh * gw;
        let hidden_dim = self.cfg.hidden_size;
        let nh = self.cfg.num_heads;
        let dh = self.cfg.head_dim();

        // 1. patch_embed + interpolated position embedding.
        let mut hidden = self.patch_embed.forward_vec(pixel_values, n);
        let pos = self.pos_embed_interpolate(grid);
        for (hi, p) in hidden.iter_mut().zip(pos.iter()) {
            *hi += p;
        }

        // 2. 2D RoPE tables.
        let (cos, sin) = self.rope_tables(grid);

        // 3. Blocks + DeepStack taps.
        let mut deepstack = Vec::new();
        for (li, blk) in self.blocks.iter().enumerate() {
            blk.forward(&mut hidden, n, nh, dh, &cos, &sin);
            if let Some(di) = self.cfg.deepstack_visual_indexes.iter().position(|&x| x == li) {
                deepstack.push(self.deepstack_mergers[di].forward(&hidden, n, hidden_dim));
            }
        }

        // 4. Final merge.
        let image_embeds = self.merger.forward(&hidden, n, hidden_dim);
        (image_embeds, deepstack)
    }

    /// Bilinearly interpolate the learned `[grid×grid, hidden]` position table
    /// to the image's `h×w` grid, then permute into merge-block order.
    fn pos_embed_interpolate(&self, grid: (usize, usize, usize)) -> Vec<f32> {
        let (t, h, w) = grid;
        let hidden = self.cfg.hidden_size;
        let side = self.cfg.num_grid_per_side();
        let m = self.cfg.spatial_merge_size;

        let lin = |n_out: usize| -> Vec<f32> {
            // linspace(0, side-1, n_out).
            if n_out == 1 {
                return vec![0.0];
            }
            let step = (side - 1) as f32 / (n_out - 1) as f32;
            (0..n_out).map(|i| i as f32 * step).collect()
        };
        let h_idx = lin(h);
        let w_idx = lin(w);

        // Bilinear lookup in h-major order [h*w, hidden].
        let mut hmajor = vec![0.0_f32; h * w * hidden];
        for (hi, &hy) in h_idx.iter().enumerate() {
            let hf = hy.floor() as usize;
            let hc = (hf + 1).min(side - 1);
            let dh = hy - hf as f32;
            for (wi, &wx) in w_idx.iter().enumerate() {
                let wf = wx.floor() as usize;
                let wc = (wf + 1).min(side - 1);
                let dw = wx - wf as f32;
                let corners = [
                    (hf * side + wf, (1.0 - dh) * (1.0 - dw)),
                    (hf * side + wc, (1.0 - dh) * dw),
                    (hc * side + wf, dh * (1.0 - dw)),
                    (hc * side + wc, dh * dw),
                ];
                let dst = (hi * w + wi) * hidden;
                for (idx, wgt) in corners {
                    let src = idx * hidden;
                    for d in 0..hidden {
                        hmajor[dst + d] += wgt * self.pos_embed[src + d];
                    }
                }
            }
        }

        // Permute h-major → merge-block order, repeated over t frames.
        let mut out = vec![0.0_f32; t * h * w * hidden];
        let mw = w / m;
        let mh = h / m;
        let mut o = 0;
        for _ in 0..t {
            for br in 0..mh {
                for bc in 0..mw {
                    for ir in 0..m {
                        for ic in 0..m {
                            let hi = br * m + ir;
                            let wi = bc * m + ic;
                            let src = (hi * w + wi) * hidden;
                            out[o * hidden..o * hidden + hidden]
                                .copy_from_slice(&hmajor[src..src + hidden]);
                            o += 1;
                        }
                    }
                }
            }
        }
        out
    }

    /// 2D-RoPE cos/sin tables `[n, head_dim]` in merge-block patch order.
    fn rope_tables(&self, grid: (usize, usize, usize)) -> (Vec<f32>, Vec<f32>) {
        let (t, h, w) = grid;
        let n = t * h * w;
        let dh = self.cfg.head_dim();
        let m = self.cfg.spatial_merge_size;
        let nfreq = self.rot_inv_freq.len(); // head_dim/4
        let half = dh / 2; // = 2*nfreq
        let mut cos = vec![0.0_f32; n * dh];
        let mut sin = vec![0.0_f32; n * dh];
        let mh = h / m;
        let mw = w / m;
        let mut k = 0usize;
        for _ in 0..t {
            for br in 0..mh {
                for bc in 0..mw {
                    for ir in 0..m {
                        for ic in 0..m {
                            let row = (br * m + ir) as f32;
                            let col = (bc * m + ic) as f32;
                            // emb half = [row·invf (nfreq), col·invf (nfreq)]; then cat(half,half).
                            for f in 0..nfreq {
                                let a_row = row * self.rot_inv_freq[f];
                                let a_col = col * self.rot_inv_freq[f];
                                let (sr, cr) = a_row.sin_cos();
                                let (sc, cc) = a_col.sin_cos();
                                // positions f and f+nfreq in the half; duplicated at +half.
                                let base = k * dh;
                                cos[base + f] = cr;
                                sin[base + f] = sr;
                                cos[base + nfreq + f] = cc;
                                sin[base + nfreq + f] = sc;
                                cos[base + half + f] = cr;
                                sin[base + half + f] = sr;
                                cos[base + half + nfreq + f] = cc;
                                sin[base + half + nfreq + f] = sc;
                            }
                            k += 1;
                        }
                    }
                }
            }
        }
        (cos, sin)
    }
}

fn load_lin(l: &mut Linear, get: Getter, prefix: &str) -> Result<()> {
    l.weight = get(&format!("{prefix}.weight"))?;
    l.bias = Some(get(&format!("{prefix}.bias"))?);
    Ok(())
}

fn load_ln(l: &mut LayerNorm, get: Getter, prefix: &str) -> Result<()> {
    l.weight = get(&format!("{prefix}.weight"))?;
    l.bias = get(&format!("{prefix}.bias"))?;
    Ok(())
}

/// rotate_half RoPE on `x[n, num_heads, head_dim]` with cos/sin `[n, head_dim]`.
fn apply_rope_vision(x: &mut [f32], n: usize, nh: usize, dh: usize, cos: &[f32], sin: &[f32]) {
    let half = dh / 2;
    for t in 0..n {
        let c = &cos[t * dh..(t + 1) * dh];
        let s = &sin[t * dh..(t + 1) * dh];
        for hh in 0..nh {
            let off = (t * nh + hh) * dh;
            let row = &mut x[off..off + dh];
            // rotate_half: out[i<half] = x[i]·c[i] - x[i+half]·s[i];
            //              out[i>=half] = x[i]·c[i] + x[i-half]·s[i].
            let orig: Vec<f32> = row.to_vec();
            for i in 0..half {
                row[i] = orig[i] * c[i] - orig[i + half] * s[i];
                row[i + half] = orig[i + half] * c[i + half] + orig[i] * s[i + half];
            }
        }
    }
}
