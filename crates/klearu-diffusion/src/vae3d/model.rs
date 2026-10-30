//! Wan VAE encoder and decoder, and the chunked drivers that run them.
//!
//! Both are inherently chunked (see [`super::blocks`]): the decoder consumes
//! one latent frame per call and the encoder consumes 1 then 4 pixel frames
//! per call, with a [`FeatCache`] carrying all temporal context between calls.

use super::blocks::{
    cached_conv, FeatCache, MidBlock, ResidualDownBlock, ResidualUpBlock, StageKind,
};
use super::{patchify, silu, unpatchify, CausalConv3d, RmsNormChannel, Vol, WanVaeConfig};

/// Decoder: `conv_in -> mid -> up blocks -> norm -> silu -> conv_out`.
pub struct Decoder3d {
    pub conv_in: CausalConv3d,
    pub mid_block: MidBlock,
    pub up_blocks: Vec<ResidualUpBlock>,
    pub norm_out: RmsNormChannel,
    pub conv_out: CausalConv3d,
}

impl Decoder3d {
    /// Build from config. Stage widths are
    /// `dim * [dim_mult[-1]] ++ reverse(dim_mult)`, so for Wan 2.2
    /// (`decoder_base_dim` 256, `dim_mult` [1,2,4,4]) that is
    /// `[1024, 1024, 1024, 512, 256]` — four up blocks.
    ///
    /// Wan 2.1 / Qwen-Image (`base_dim` 96) gives `[384, 384, 384, 192, 96]`,
    /// but every stage after the first *enters* at half its listed width,
    /// because the previous stage's upsampler halved it: the up blocks run
    /// 384->384, 192->384, 192->192, 96->96.
    pub fn new(cfg: &WanVaeConfig) -> Self {
        let dim = cfg.decoder_dim();
        let mut dims = vec![dim * cfg.dim_mult[cfg.dim_mult.len() - 1]];
        dims.extend(cfg.dim_mult.iter().rev().map(|m| dim * m));

        // Temporal upsampling mirrors the downsample schedule reversed.
        let temporal_up: Vec<bool> = cfg.temperal_downsample.iter().rev().copied().collect();

        let n = cfg.dim_mult.len();
        let kind = StageKind::of(cfg);
        let up_blocks = (0..dims.len() - 1)
            .map(|i| {
                let up_flag = i != n - 1;
                let temporal = up_flag && *temporal_up.get(i).unwrap_or(&false);
                match kind {
                    StageKind::Residual => ResidualUpBlock::new(
                        dims[i],
                        dims[i + 1],
                        cfg.num_res_blocks,
                        temporal,
                        up_flag,
                    ),
                    StageKind::Plain => ResidualUpBlock::new_plain(
                        if i > 0 { dims[i] / 2 } else { dims[i] },
                        dims[i + 1],
                        cfg.num_res_blocks,
                        temporal,
                        up_flag,
                    ),
                }
            })
            .collect::<Vec<_>>();

        let out_dim = *dims.last().unwrap();
        let out_channels = if cfg.out_channels == 0 { cfg.in_channels } else { cfg.out_channels };
        Self {
            conv_in: CausalConv3d::new(cfg.z_dim, dims[0], (3, 3, 3), (1, 1, 1), (1, 1, 1), true),
            mid_block: MidBlock::new(dims[0], 1),
            up_blocks,
            norm_out: RmsNormChannel::new(out_dim, false),
            conv_out: CausalConv3d::new(
                out_dim,
                out_channels,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                true,
            ),
        }
    }

    /// Total cache slots the decoder walks per chunk.
    pub fn convs(&self) -> usize {
        1 + self.mid_block.convs()
            + self.up_blocks.iter().map(ResidualUpBlock::convs).sum::<usize>()
            + 1
    }

    /// Decode one latent chunk. `first_chunk` must be true for the first call
    /// after a cache reset.
    pub fn forward(&self, z: &Vol, cache: &mut FeatCache, first_chunk: bool) -> Vol {
        let mut x = cached_conv(&self.conv_in, z, Some(cache));
        x = self.mid_block.forward(&x, Some(cache));
        for up in &self.up_blocks {
            x = up.forward(&x, Some(cache), first_chunk);
        }
        x = self.norm_out.forward(&x);
        silu(&mut x.data);
        cached_conv(&self.conv_out, &x, Some(cache))
    }
}

/// Encoder: `conv_in -> down blocks -> mid -> norm -> silu -> conv_out`.
///
/// Emits `2 * z_dim` channels (mean and log-variance).
pub struct Encoder3d {
    pub conv_in: CausalConv3d,
    pub down_blocks: Vec<ResidualDownBlock>,
    pub mid_block: MidBlock,
    pub norm_out: RmsNormChannel,
    pub conv_out: CausalConv3d,
}

impl Encoder3d {
    pub fn new(cfg: &WanVaeConfig) -> Self {
        let dim = cfg.base_dim;
        let mut dims = vec![dim];
        dims.extend(cfg.dim_mult.iter().map(|m| dim * m));

        let n = cfg.dim_mult.len();
        let kind = StageKind::of(cfg);
        // Attention placement for the plain stages: the reference tracks a
        // `scale` that starts at 1 and halves after every downsample, and puts
        // attention after each resnet of a stage whose scale is listed. Powers
        // of two are exact in f32, so the float comparison is safe.
        let mut scale = 1.0_f32;
        let down_blocks = (0..dims.len() - 1)
            .map(|i| {
                let down_flag = i != n - 1;
                let temporal = down_flag && cfg.temperal_downsample[i];
                let b = match kind {
                    StageKind::Residual => ResidualDownBlock::new(
                        dims[i],
                        dims[i + 1],
                        cfg.num_res_blocks,
                        temporal,
                        down_flag,
                    ),
                    StageKind::Plain => ResidualDownBlock::new_plain(
                        dims[i],
                        dims[i + 1],
                        cfg.num_res_blocks,
                        temporal,
                        down_flag,
                        cfg.attn_scales.contains(&scale),
                    ),
                };
                if down_flag {
                    scale /= 2.0;
                }
                b
            })
            .collect::<Vec<_>>();

        let out_dim = *dims.last().unwrap();
        Self {
            conv_in: CausalConv3d::new(
                cfg.in_channels,
                dims[0],
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                true,
            ),
            down_blocks,
            mid_block: MidBlock::new(out_dim, 1),
            norm_out: RmsNormChannel::new(out_dim, false),
            conv_out: CausalConv3d::new(
                out_dim,
                cfg.z_dim * 2,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                true,
            ),
        }
    }

    pub fn convs(&self) -> usize {
        1 + self.down_blocks.iter().map(ResidualDownBlock::convs).sum::<usize>()
            + self.mid_block.convs()
            + 1
    }

    pub fn forward(&self, x: &Vol, cache: &mut FeatCache) -> Vol {
        let mut y = cached_conv(&self.conv_in, x, Some(cache));
        for d in &self.down_blocks {
            y = d.forward(&y, Some(cache));
        }
        y = self.mid_block.forward(&y, Some(cache));
        y = self.norm_out.forward(&y);
        silu(&mut y.data);
        cached_conv(&self.conv_out, &y, Some(cache))
    }
}

/// The full VAE: encoder, decoder, and the two 1x1 quant convs.
pub struct WanVae {
    pub cfg: WanVaeConfig,
    pub encoder: Encoder3d,
    pub decoder: Decoder3d,
    pub quant_conv: CausalConv3d,
    pub post_quant_conv: CausalConv3d,
}

impl WanVae {
    pub fn new(cfg: WanVaeConfig) -> Self {
        let encoder = Encoder3d::new(&cfg);
        let decoder = Decoder3d::new(&cfg);
        let quant_conv =
            CausalConv3d::new(cfg.z_dim * 2, cfg.z_dim * 2, (1, 1, 1), (1, 1, 1), (0, 0, 0), true);
        let post_quant_conv =
            CausalConv3d::new(cfg.z_dim, cfg.z_dim, (1, 1, 1), (1, 1, 1), (0, 0, 0), true);
        Self { cfg, encoder, decoder, quant_conv, post_quant_conv }
    }

    /// Output frames produced by decoding `latent_frames` latent frames.
    ///
    /// The first latent frame yields a single frame — `upsample3d` skips its
    /// temporal conv on the first chunk and `DupUp3D` drops its leading
    /// frames — and every later one yields `temporal_compression`. So 31
    /// latent frames give `1 + 4 * 30 = 121`, exactly 5s at 24fps.
    pub fn decoded_frame_count(&self, latent_frames: usize) -> usize {
        if latent_frames == 0 {
            return 0;
        }
        1 + (latent_frames - 1) * self.cfg.temporal_compression()
    }

    /// Latent frames produced by encoding `pixel_frames` frames.
    pub fn encoded_frame_count(&self, pixel_frames: usize) -> usize {
        if pixel_frames == 0 {
            return 0;
        }
        1 + (pixel_frames - 1).div_ceil(self.cfg.temporal_compression())
    }

    /// Decode a full latent `[z_dim, T, H, W]` to pixels, one latent frame at
    /// a time, then unpatchify and clamp to `[-1, 1]`.
    pub fn decode(&self, z: &Vol) -> Vol {
        assert_eq!(z.c, self.cfg.z_dim, "decode: expected {} latent channels", self.cfg.z_dim);
        let x = self.post_quant_conv.forward(z, None);
        let mut cache = FeatCache::new(self.decoder.convs());
        let mut out: Option<Vol> = None;
        for t in 0..x.t {
            cache.rewind();
            let frame = frame_slice(&x, t);
            let dec = self.decoder.forward(&frame, &mut cache, t == 0);
            out = Some(match out {
                None => dec,
                Some(prev) => Vol::concat_time(&prev, &dec),
            });
        }
        let mut out = unpatchify(&out.expect("decode: empty latent"), self.cfg.patch_size);
        for v in out.data.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        out
    }

    /// GPU encode of a **single** pixel frame — the conditioning-image case.
    ///
    /// Same result as [`Self::encode`] for `T = 1`, but seconds instead of
    /// minutes: the CPU path costs 102 s at 320x192 and 409 s at 640x352,
    /// scaling with pixel count, which put a start frame at native resolution
    /// in front of an hour-long generation for roughly 25 minutes of nothing.
    ///
    /// Returns `None` when the GPU encoder cannot serve the request, so callers
    /// fall back to [`Self::encode`] rather than silently changing behaviour.
    /// That is the case for multi-frame input, whose second chunk needs
    /// Downsample3d's stride-2 temporal conv.
    #[cfg(feature = "metal")]
    pub fn encode_gpu(&self, pixels: &Vol) -> Option<Vol> {
        let enc = super::gpu::GpuEncoder3d::from_cpu(&self.encoder, &self.quant_conv);
        let x = patchify(pixels, self.cfg.patch_size);
        let mut cache = super::gpu::GpuFeatCache::new(self.encoder.convs());
        let iters = self.encoded_frame_count(x.t);
        let mut out: Option<Vol> = None;
        for i in 0..iters {
            cache.rewind();
            // 1 frame, then 4 at a time: the causal VAE's chunking, mirrored
            // from `encode` so the two produce the same latent.
            let (start, end) = if i == 0 {
                (0, 1)
            } else {
                let s = 1 + 4 * (i - 1);
                (s, (s + 4).min(x.t))
            };
            if start >= end {
                break;
            }
            let chunk = frame_range(&x, start, end);
            // `upload_vol`, not `upload_f32_as_f16`: the VAE stack runs in IEEE
            // half by default and `GpuDtype::F16` means *bfloat*. Uploading
            // bf16 data against half weights does not fail, it returns noise.
            let gx = super::gpu::upload_vol(&chunk);
            let (o, t, h, w) = enc.forward_chunk(&gx, chunk.t, chunk.h, chunk.w, &mut cache);
            let data = o.download_to_f32();
            let c = data.len() / (t * h * w);
            let v = Vol { data, c, t, h, w };
            out = Some(match out {
                None => v,
                Some(prev) => Vol::concat_time(&prev, &v),
            });
        }
        out
    }

    /// Single-frame convenience over [`Self::encode_gpu`].
    #[cfg(feature = "metal")]
    pub fn encode_frame_gpu(&self, pixels: &Vol) -> Option<Vol> {
        self.encode_gpu(pixels)
    }

    /// One-shot GPU decode of an image latent `[z_dim, h, w]` (already
    /// de-normalised) to RGB `[3, h*s, w*s]` in `[-1, 1]`.
    ///
    /// Builds a decoder-only [`super::gpu::GpuImageVae`] for the call and
    /// drops it. Anything decoding more than one image should hold a
    /// `GpuImageVae` instead; see there for why rebuilding is not free.
    #[cfg(feature = "metal")]
    pub fn decode_image_gpu(&self, z: &[f32], h: usize, w: usize) -> Vec<f32> {
        super::gpu::GpuImageVae::decoder_only(self).decode(z, h, w)
    }

    /// One-shot GPU encode of RGB `[3, h, w]` in `[-1, 1]` to the posterior
    /// mean `[z_dim, h/s, w/s]` (un-normalised). Same caveat as
    /// [`Self::decode_image_gpu`].
    #[cfg(feature = "metal")]
    pub fn encode_image_gpu(&self, rgb: &[f32], h: usize, w: usize) -> Vec<f32> {
        super::gpu::GpuImageVae::encoder_only(self).encode(rgb, h, w)
    }

    /// Encode pixels `[3, T, H, W]` in `[-1, 1]` to `[2 * z_dim, T', H', W']`
    /// (mean and log-variance), in chunks of 1 then 4 frames.
    pub fn encode(&self, pixels: &Vol) -> Vol {
        let x = patchify(pixels, self.cfg.patch_size);
        let mut cache = FeatCache::new(self.encoder.convs());
        let iters = self.encoded_frame_count(x.t);
        let mut out: Option<Vol> = None;
        for i in 0..iters {
            cache.rewind();
            let (start, end) = if i == 0 {
                (0, 1)
            } else {
                let s = 1 + 4 * (i - 1);
                (s, (s + 4).min(x.t))
            };
            if start >= end {
                break;
            }
            let chunk = frame_range(&x, start, end);
            let enc = self.encoder.forward(&chunk, &mut cache);
            out = Some(match out {
                None => enc,
                Some(prev) => Vol::concat_time(&prev, &enc),
            });
        }
        self.quant_conv.forward(&out.expect("encode: empty input"), None)
    }
}

/// One frame as its own volume.
fn frame_slice(x: &Vol, t: usize) -> Vol {
    frame_range(x, t, t + 1)
}

/// Frames `[start, end)` as their own volume.
fn frame_range(x: &Vol, start: usize, end: usize) -> Vol {
    let mut out = Vol::zeros(x.c, end - start, x.h, x.w);
    for c in 0..x.c {
        for t in start..end {
            for h in 0..x.h {
                for w in 0..x.w {
                    let v = x.at(c, t, h, w);
                    let i = out.idx(c, t - start, h, w);
                    out.data[i] = v;
                }
            }
        }
    }
    out
}
