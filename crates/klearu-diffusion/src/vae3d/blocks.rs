//! Wan VAE blocks: the feature cache, residual/attention/mid blocks, and the
//! up/down stages — in both the residual form Wan 2.2 (`is_residual: true`) is
//! built from and the plain form of Wan 2.1 / Qwen-Image ([`StageKind`]).
//!
//! # The feature cache is not an optimisation
//!
//! The reference never decodes a clip in one pass. `_decode` calls the decoder
//! **one latent frame at a time**, and `_encode` feeds chunks of 1 then 4
//! frames. All temporal context flows through `feat_cache`: a slot per causal
//! convolution, holding that conv's trailing input frames from the previous
//! chunk.
//!
//! Two layers go further and skip work entirely on the first chunk —
//! `upsample3d` does not run its temporal conv, and `downsample3d` passes
//! through — which is what makes the frame arithmetic come out. A 31-frame
//! latent decodes to `1 + 4 * 30 = 121` frames, exactly 5s at 24fps.
//!
//! So a `FeatCache` is required for a correct decode, not optional.

use super::{avg_down_3d, dup_up_3d, silu, CausalConv3d, RmsNormChannel, Vol, CACHE_T};

/// One cache slot.
#[derive(Clone, Debug)]
pub enum Slot {
    /// Nothing cached yet — this is the first chunk.
    Empty,
    /// First chunk already seen by a layer that skips its temporal conv once.
    /// The reference spells this `"Rep"`.
    Rep,
    /// Trailing frames from the previous chunk.
    Frames(Vol),
}

/// Per-conv trailing-frame cache, walked in a fixed order by every forward
/// pass. `idx` is reset before each chunk; the slots persist across chunks.
#[derive(Debug)]
pub struct FeatCache {
    pub slots: Vec<Slot>,
    pub idx: usize,
}

impl FeatCache {
    pub fn new(n_convs: usize) -> Self {
        Self { slots: vec![Slot::Empty; n_convs], idx: 0 }
    }

    /// Start a new chunk. Slots keep their contents; only the walk position
    /// rewinds.
    pub fn rewind(&mut self) {
        self.idx = 0;
    }

    fn next(&mut self) -> usize {
        let i = self.idx;
        if i >= self.slots.len() {
            self.slots.resize(i + 1, Slot::Empty);
        }
        self.idx += 1;
        i
    }
}

/// Run a causal conv through its cache slot.
///
/// Mirrors the reference exactly: the new cache is taken from the tensor
/// *before* the convolution, back-filled from the previous slot when this
/// chunk is shorter than `CACHE_T`, and the conv itself runs against the
/// *previous* slot's frames.
pub fn cached_conv(conv: &CausalConv3d, x: &Vol, cache: Option<&mut FeatCache>) -> Vol {
    let Some(cache) = cache else {
        return conv.forward(x, None);
    };
    let i = cache.next();
    let mut cache_x = x.tail_frames(CACHE_T);
    if cache_x.t < 2 {
        if let Slot::Frames(prev) = &cache.slots[i] {
            cache_x = Vol::concat_time(&prev.tail_frames(1), &cache_x);
        }
    }
    let prev = match &cache.slots[i] {
        Slot::Frames(v) => Some(v.clone()),
        _ => None,
    };
    let out = conv.forward(x, prev.as_ref());
    cache.slots[i] = Slot::Frames(cache_x);
    out
}

// ---------------------------------------------------------------------------
// Residual block
// ---------------------------------------------------------------------------

/// `norm -> silu -> conv -> norm -> silu -> conv`, plus a 1x1x1 shortcut when
/// the channel count changes.
#[derive(Clone, Debug)]
pub struct ResidualBlock {
    pub norm1: RmsNormChannel,
    pub conv1: CausalConv3d,
    pub norm2: RmsNormChannel,
    pub conv2: CausalConv3d,
    /// Present only when `in_dim != out_dim`.
    pub shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            norm1: RmsNormChannel::new(in_dim, false),
            conv1: CausalConv3d::new(in_dim, out_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), true),
            norm2: RmsNormChannel::new(out_dim, false),
            conv2: CausalConv3d::new(out_dim, out_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), true),
            shortcut: (in_dim != out_dim).then(|| {
                CausalConv3d::new(in_dim, out_dim, (1, 1, 1), (1, 1, 1), (0, 0, 0), true)
            }),
        }
    }

    /// Number of cache slots this block consumes.
    pub const CONVS: usize = 2;

    pub fn forward(&self, x: &Vol, mut cache: Option<&mut FeatCache>) -> Vol {
        // The shortcut is a kernel-1 conv: no temporal extent, no cache slot.
        let h = match &self.shortcut {
            Some(sc) => sc.forward(x, None),
            None => x.clone(),
        };

        let mut y = self.norm1.forward(x);
        silu(&mut y.data);
        let mut y = cached_conv(&self.conv1, &y, cache.as_deref_mut());

        y = self.norm2.forward(&y);
        silu(&mut y.data);
        let mut y = cached_conv(&self.conv2, &y, cache);

        for (o, r) in y.data.iter_mut().zip(&h.data) {
            *o += *r;
        }
        y
    }
}

// ---------------------------------------------------------------------------
// Attention block
// ---------------------------------------------------------------------------

/// Single-head spatial self-attention over `H*W` positions, applied per frame.
///
/// Present in the mid block of **both** encoder and decoder regardless of
/// `is_residual` or `attn_scales` — `attn_scales` only governs the down/up
/// stages, and this VAE's is empty, but the mid block is unconditional.
#[derive(Clone, Debug)]
pub struct AttentionBlock {
    pub norm: RmsNormChannel,
    /// 1x1 conv producing q|k|v concatenated: `[3*dim, dim, 1, 1, 1]`.
    pub to_qkv: CausalConv3d,
    pub proj: CausalConv3d,
    pub dim: usize,
}

impl AttentionBlock {
    pub fn new(dim: usize) -> Self {
        Self {
            norm: RmsNormChannel::new(dim, false),
            to_qkv: CausalConv3d::conv2d(dim, dim * 3, 1, 1, 0, true),
            proj: CausalConv3d::conv2d(dim, dim, 1, 1, 0, true),
            dim,
        }
    }

    /// No causal convs, so no cache slots.
    pub const CONVS: usize = 0;

    pub fn forward(&self, x: &Vol) -> Vol {
        let n = x.h * x.w;
        let d = self.dim;
        let scale = 1.0 / (d as f32).sqrt();

        let y = self.norm.forward(x);
        let qkv = self.to_qkv.forward(&y, None);

        // Attention is per frame; positions are the H*W spatial sites.
        let mut attn = Vol::zeros(d, x.t, x.h, x.w);
        let mut scores = vec![0.0_f32; n];
        for t in 0..x.t {
            for i in 0..n {
                let (ih, iw) = (i / x.w, i % x.w);
                let mut max = f32::NEG_INFINITY;
                for (j, s) in scores.iter_mut().enumerate() {
                    let (jh, jw) = (j / x.w, j % x.w);
                    let mut acc = 0.0_f32;
                    for c in 0..d {
                        acc += qkv.at(c, t, ih, iw) * qkv.at(d + c, t, jh, jw);
                    }
                    *s = acc * scale;
                    if *s > max {
                        max = *s;
                    }
                }
                let mut sum = 0.0_f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for c in 0..d {
                    let mut acc = 0.0_f32;
                    for (j, s) in scores.iter().enumerate() {
                        acc += s * qkv.at(2 * d + c, t, j / x.w, j % x.w);
                    }
                    let k = attn.idx(c, t, ih, iw);
                    attn.data[k] = acc * inv;
                }
            }
        }

        let mut out = self.proj.forward(&attn, None);
        for (o, r) in out.data.iter_mut().zip(&x.data) {
            *o += *r;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Mid block
// ---------------------------------------------------------------------------

/// `resnet -> (attention -> resnet) * num_layers`.
#[derive(Clone, Debug)]
pub struct MidBlock {
    pub resnets: Vec<ResidualBlock>,
    pub attentions: Vec<AttentionBlock>,
}

impl MidBlock {
    pub fn new(dim: usize, num_layers: usize) -> Self {
        Self {
            resnets: (0..=num_layers).map(|_| ResidualBlock::new(dim, dim)).collect(),
            attentions: (0..num_layers).map(|_| AttentionBlock::new(dim)).collect(),
        }
    }

    pub fn convs(&self) -> usize {
        self.resnets.len() * ResidualBlock::CONVS
    }

    pub fn forward(&self, x: &Vol, mut cache: Option<&mut FeatCache>) -> Vol {
        let mut y = self.resnets[0].forward(x, cache.as_deref_mut());
        for (attn, resnet) in self.attentions.iter().zip(&self.resnets[1..]) {
            y = attn.forward(&y);
            y = resnet.forward(&y, cache.as_deref_mut());
        }
        y
    }
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Nearest 2x spatial upsample. For an integer scale, PyTorch's
/// `nearest-exact` maps destination `i` to source `floor(i/2)`, the same as
/// plain nearest.
fn upsample_nearest_2x(x: &Vol) -> Vol {
    let mut out = Vol::zeros(x.c, x.t, x.h * 2, x.w * 2);
    for c in 0..x.c {
        for t in 0..x.t {
            for h in 0..out.h {
                for w in 0..out.w {
                    let v = x.at(c, t, h / 2, w / 2);
                    let i = out.idx(c, t, h, w);
                    out.data[i] = v;
                }
            }
        }
    }
    out
}

/// Which resampling a [`Resample`] performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResampleMode {
    Upsample2d,
    Upsample3d,
    Downsample2d,
    Downsample3d,
}

/// Spatial resample, optionally paired with a temporal conv.
///
/// Ordering differs by direction, and it matters: `upsample3d` runs its
/// temporal conv **before** the spatial resample, `downsample3d` runs it
/// **after**.
#[derive(Clone, Debug)]
pub struct Resample {
    pub mode: ResampleMode,
    /// Per-frame spatial conv.
    pub conv: CausalConv3d,
    /// Temporal conv, for the 3d modes.
    pub time_conv: Option<CausalConv3d>,
    pub dim: usize,
}

impl Resample {
    pub fn new(dim: usize, mode: ResampleMode, upsample_out_dim: Option<usize>) -> Self {
        let out = upsample_out_dim.unwrap_or(dim / 2);
        match mode {
            ResampleMode::Upsample2d => Self {
                mode,
                conv: CausalConv3d::conv2d(dim, out, 3, 1, 1, true),
                time_conv: None,
                dim,
            },
            ResampleMode::Upsample3d => Self {
                mode,
                conv: CausalConv3d::conv2d(dim, out, 3, 1, 1, true),
                // Doubles channels; the two halves interleave into time.
                time_conv: Some(CausalConv3d::new(
                    dim,
                    dim * 2,
                    (3, 1, 1),
                    (1, 1, 1),
                    (1, 0, 0),
                    true,
                )),
                dim,
            },
            ResampleMode::Downsample2d => Self {
                mode,
                conv: CausalConv3d::conv2d_pad_end(dim, dim, 3, 2),
                time_conv: None,
                dim,
            },
            ResampleMode::Downsample3d => Self {
                mode,
                conv: CausalConv3d::conv2d_pad_end(dim, dim, 3, 2),
                // pad_t = 0, stride 2: consumes the manually-prepended cache
                // frame rather than zero padding.
                time_conv: Some(CausalConv3d::new(
                    dim,
                    dim,
                    (3, 1, 1),
                    (2, 1, 1),
                    (0, 0, 0),
                    true,
                )),
                dim,
            },
        }
    }

    /// Cache slots consumed: the 3d modes take one for their temporal conv.
    pub fn convs(&self) -> usize {
        usize::from(self.time_conv.is_some())
    }

    pub fn forward(&self, x: &Vol, cache: Option<&mut FeatCache>) -> Vol {
        let mut cache = cache;
        let mut y = x.clone();

        if self.mode == ResampleMode::Upsample3d {
            if let Some(cache) = cache.as_deref_mut() {
                let i = cache.next();
                match &cache.slots[i] {
                    // First chunk: skip the temporal conv entirely, so the
                    // first latent frame expands to one output frame.
                    Slot::Empty => {
                        cache.slots[i] = Slot::Rep;
                    }
                    slot => {
                        let prev = match slot {
                            Slot::Frames(v) => Some(v.clone()),
                            _ => None,
                        };
                        let mut cache_x = y.tail_frames(CACHE_T);
                        if cache_x.t < 2 {
                            match &cache.slots[i] {
                                Slot::Frames(p) => {
                                    cache_x = Vol::concat_time(&p.tail_frames(1), &cache_x);
                                }
                                // After "Rep", front-pad with zeros instead.
                                Slot::Rep => {
                                    let z = Vol::zeros(cache_x.c, cache_x.t, cache_x.h, cache_x.w);
                                    cache_x = Vol::concat_time(&z, &cache_x);
                                }
                                Slot::Empty => {}
                            }
                        }
                        let tc = self.time_conv.as_ref().unwrap();
                        let conv_out = tc.forward(&y, prev.as_ref());
                        cache.slots[i] = Slot::Frames(cache_x);

                        // [2*c, t] -> interleave the two halves into 2t.
                        let c = y.c;
                        let mut up = Vol::zeros(c, conv_out.t * 2, conv_out.h, conv_out.w);
                        for ci in 0..c {
                            for t in 0..conv_out.t {
                                for k in 0..2 {
                                    for h in 0..conv_out.h {
                                        for w in 0..conv_out.w {
                                            let v = conv_out.at(k * c + ci, t, h, w);
                                            let idx = up.idx(ci, t * 2 + k, h, w);
                                            up.data[idx] = v;
                                        }
                                    }
                                }
                            }
                        }
                        y = up;
                    }
                }
            }
        }

        y = match self.mode {
            ResampleMode::Upsample2d | ResampleMode::Upsample3d => {
                self.conv.forward(&upsample_nearest_2x(&y), None)
            }
            _ => self.conv.forward(&y, None),
        };

        if self.mode == ResampleMode::Downsample3d {
            if let Some(cache) = cache {
                let i = cache.next();
                match &cache.slots[i] {
                    // First chunk: pass through, just seed the cache.
                    Slot::Empty | Slot::Rep => {
                        cache.slots[i] = Slot::Frames(y.clone());
                    }
                    Slot::Frames(prev) => {
                        let cache_x = y.tail_frames(1);
                        // The reference concatenates the cached frame itself
                        // rather than routing it through the conv's padding,
                        // because this conv has pad_t = 0.
                        let joined = Vol::concat_time(&prev.tail_frames(1), &y);
                        y = self.time_conv.as_ref().unwrap().forward(&joined, None);
                        cache.slots[i] = Slot::Frames(cache_x);
                    }
                }
            }
        }
        y
    }
}

// ---------------------------------------------------------------------------
// Up / down stages
// ---------------------------------------------------------------------------

/// Which of the two Wan VAE stage designs a down/up stage follows.
///
/// Both run the same resnets and the same [`Resample`]; they differ in three
/// places, each of which produces a correctly-shaped wrong answer if mixed up:
///
/// | | `Plain` (Wan 2.1, Qwen-Image) | `Residual` (Wan 2.2) |
/// |---|---|---|
/// | stage shortcut | none | `AvgDown3D` / `DupUp3D` |
/// | upsampler conv | halves channels, so the next stage takes `dim / 2` | keeps them |
/// | weight names | encoder flattened into one `down_blocks` list; `upsamplers.0` | nested per stage; `upsampler` |
///
/// The structs keep their Wan 2.2 names (`ResidualDownBlock`,
/// `ResidualUpBlock`) because that is the layout they were written for; a
/// `Plain` stage is the same struct with the shortcut switched off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKind {
    Plain,
    Residual,
}

impl StageKind {
    pub fn of(cfg: &super::WanVaeConfig) -> Self {
        if cfg.is_residual { StageKind::Residual } else { StageKind::Plain }
    }
}

/// Encoder stage: `num_res_blocks` residual blocks and an optional
/// downsampler, plus — for [`StageKind::Residual`] — an `AvgDown3D` shortcut
/// added on top.
#[derive(Clone, Debug)]
pub struct ResidualDownBlock {
    pub kind: StageKind,
    pub resnets: Vec<ResidualBlock>,
    /// One attention block after each resnet, or none. Only a `Plain` stage
    /// whose scale is listed in `attn_scales` has them, and no shipped
    /// checkpoint lists any — but the reference supports it, and getting it
    /// wrong would shift every later flat weight index.
    pub attentions: Vec<AttentionBlock>,
    pub downsampler: Option<Resample>,
    pub out_dim: usize,
    pub factor_t: usize,
    pub factor_s: usize,
}

impl ResidualDownBlock {
    /// Wan 2.2 stage.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        num_res_blocks: usize,
        temporal_downsample: bool,
        down_flag: bool,
    ) -> Self {
        Self::build(
            StageKind::Residual,
            in_dim,
            out_dim,
            num_res_blocks,
            temporal_downsample,
            down_flag,
            false,
        )
    }

    /// Wan 2.1 / Qwen-Image stage: no shortcut, optional attention.
    pub fn new_plain(
        in_dim: usize,
        out_dim: usize,
        num_res_blocks: usize,
        temporal_downsample: bool,
        down_flag: bool,
        with_attention: bool,
    ) -> Self {
        Self::build(
            StageKind::Plain,
            in_dim,
            out_dim,
            num_res_blocks,
            temporal_downsample,
            down_flag,
            with_attention,
        )
    }

    fn build(
        kind: StageKind,
        in_dim: usize,
        out_dim: usize,
        num_res_blocks: usize,
        temporal_downsample: bool,
        down_flag: bool,
        with_attention: bool,
    ) -> Self {
        let mut dim = in_dim;
        let resnets = (0..num_res_blocks)
            .map(|_| {
                let b = ResidualBlock::new(dim, out_dim);
                dim = out_dim;
                b
            })
            .collect();
        let attentions = if with_attention {
            (0..num_res_blocks).map(|_| AttentionBlock::new(out_dim)).collect()
        } else {
            Vec::new()
        };
        // The downsampler keeps its width in both designs, so `None`.
        let downsampler = down_flag.then(|| {
            let mode = if temporal_downsample {
                ResampleMode::Downsample3d
            } else {
                ResampleMode::Downsample2d
            };
            Resample::new(out_dim, mode, None)
        });
        Self {
            kind,
            resnets,
            attentions,
            downsampler,
            out_dim,
            factor_t: if temporal_downsample { 2 } else { 1 },
            factor_s: if down_flag { 2 } else { 1 },
        }
    }

    pub fn convs(&self) -> usize {
        self.resnets.len() * ResidualBlock::CONVS
            + self.downsampler.as_ref().map_or(0, Resample::convs)
    }

    pub fn forward(&self, x: &Vol, mut cache: Option<&mut FeatCache>) -> Vol {
        let mut y = x.clone();
        for (i, r) in self.resnets.iter().enumerate() {
            y = r.forward(&y, cache.as_deref_mut());
            if let Some(a) = self.attentions.get(i) {
                y = a.forward(&y);
            }
        }
        if let Some(d) = &self.downsampler {
            y = d.forward(&y, cache);
        }
        if self.kind == StageKind::Residual {
            let sc = avg_down_3d(x, self.out_dim, self.factor_t, self.factor_s);
            for (o, r) in y.data.iter_mut().zip(&sc.data) {
                *o += *r;
            }
        }
        y
    }
}

/// Decoder stage: `num_res_blocks + 1` residual blocks and an optional
/// upsampler, plus — for [`StageKind::Residual`] — a `DupUp3D` shortcut added
/// on top when upsampling.
#[derive(Clone, Debug)]
pub struct ResidualUpBlock {
    pub kind: StageKind,
    pub resnets: Vec<ResidualBlock>,
    pub upsampler: Option<Resample>,
    pub out_dim: usize,
    pub factor_t: usize,
    pub factor_s: usize,
    pub has_shortcut: bool,
}

impl ResidualUpBlock {
    /// Wan 2.1 / Qwen-Image stage (`WanUpBlock` / `QwenImageUpBlock`).
    ///
    /// The upsampler's spatial conv **halves** the channel count here (Wan
    /// 2.2's keeps it), which is why the decoder hands every stage after the
    /// first `in_dim / 2`. There is no shortcut, so `first_chunk` has nothing
    /// to trim: the first-chunk frame arithmetic comes entirely from
    /// `upsample3d` skipping its temporal conv.
    pub fn new_plain(
        in_dim: usize,
        out_dim: usize,
        num_res_blocks: usize,
        temporal_upsample: bool,
        up_flag: bool,
    ) -> Self {
        let mut dim = in_dim;
        let resnets = (0..=num_res_blocks)
            .map(|_| {
                let b = ResidualBlock::new(dim, out_dim);
                dim = out_dim;
                b
            })
            .collect();
        let upsampler = up_flag.then(|| {
            let mode = if temporal_upsample {
                ResampleMode::Upsample3d
            } else {
                ResampleMode::Upsample2d
            };
            Resample::new(out_dim, mode, None)
        });
        Self {
            kind: StageKind::Plain,
            resnets,
            upsampler,
            out_dim,
            factor_t: if temporal_upsample { 2 } else { 1 },
            factor_s: 2,
            has_shortcut: false,
        }
    }

    /// Wan 2.2 stage.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        num_res_blocks: usize,
        temporal_upsample: bool,
        up_flag: bool,
    ) -> Self {
        let mut dim = in_dim;
        // Note the `+ 1`: decoder stages carry one more residual block than
        // encoder stages.
        let resnets = (0..=num_res_blocks)
            .map(|_| {
                let b = ResidualBlock::new(dim, out_dim);
                dim = out_dim;
                b
            })
            .collect();
        let upsampler = up_flag.then(|| {
            let mode = if temporal_upsample {
                ResampleMode::Upsample3d
            } else {
                ResampleMode::Upsample2d
            };
            Resample::new(out_dim, mode, Some(out_dim))
        });
        Self {
            kind: StageKind::Residual,
            resnets,
            upsampler,
            out_dim,
            factor_t: if temporal_upsample { 2 } else { 1 },
            factor_s: 2,
            has_shortcut: up_flag,
        }
    }

    pub fn convs(&self) -> usize {
        self.resnets.len() * ResidualBlock::CONVS
            + self.upsampler.as_ref().map_or(0, Resample::convs)
    }

    pub fn forward(
        &self,
        x: &Vol,
        mut cache: Option<&mut FeatCache>,
        first_chunk: bool,
    ) -> Vol {
        let mut y = x.clone();
        for r in &self.resnets {
            y = r.forward(&y, cache.as_deref_mut());
        }
        if let Some(u) = &self.upsampler {
            y = u.forward(&y, cache);
        }
        if self.has_shortcut {
            let sc = dup_up_3d(x, self.out_dim, self.factor_t, self.factor_s, first_chunk);
            assert_eq!(
                (sc.c, sc.t, sc.h, sc.w),
                (y.c, y.t, y.h, y.w),
                "DupUp3D shortcut shape disagrees with the main path"
            );
            for (o, r) in y.data.iter_mut().zip(&sc.data) {
                *o += *r;
            }
        }
        y
    }
}
