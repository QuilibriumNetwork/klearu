//! Motion module — the collection of all temporal transformers that get
//! injected into the UNet. One TemporalTransformer per spatial-attention
//! site in the down / mid / up blocks.

use super::temporal::TemporalTransformer;

/// All temporal transformers for one UNet, indexed to match the UNet's
/// block layout. `down[i][j]` is the j-th motion module in down_block i;
/// `mid[0]` is the single mid-block temporal module; `up[i][j]` matches
/// up_block i.
pub struct MotionModules {
    pub down: Vec<Vec<TemporalTransformer>>,
    pub mid: Vec<TemporalTransformer>,
    pub up: Vec<Vec<TemporalTransformer>>,
}

impl MotionModules {
    /// Build the skeleton (zero-weight) motion modules for an SDXL UNet.
    ///
    /// SDXL block layout (channels per stage):
    ///   down_blocks: [320, 640, 1280] — 2 resnets each, attentions on
    ///     blocks 1 and 2 (block 0 has no attention → no temporal module)
    ///   mid_block: 1 attention at 1280 channels
    ///   up_blocks: [1280, 640, 320] — 3 resnets each, attentions on
    ///     blocks 0 and 1
    ///
    /// AnimateDiff injects one TemporalTransformer per resnet that has a
    /// spatial attention companion. So down_block 0 (no attention) gets
    /// no temporal modules; down_blocks 1–2 each get 2.
    /// Build the skeleton for the SDXL AnimateDiff beta motion adapter.
    ///
    /// Actual adapter layout (from `diffusion_pytorch_model.fp16.safetensors`):
    ///   down_blocks: [320×2, 640×2, 1280×2] — motion modules at ALL blocks
    ///   mid_block: NONE (no motion modules in this adapter)
    ///   up_blocks: [1280×3, 640×3, 320×3] — motion modules at ALL blocks
    pub fn sdxl_default(max_frames: usize) -> Self {
        let num_heads = 8;
        let num_layers = 1;
        let ff_mult = 4;

        let make = |channels: usize, count: usize| -> Vec<TemporalTransformer> {
            let inner = channels;
            let ff_inner = inner * ff_mult;
            (0..count)
                .map(|_| TemporalTransformer::new(channels, inner, num_heads, num_layers, ff_inner, max_frames))
                .collect()
        };

        let down = vec![
            make(320, 2),   // block 0
            make(640, 2),   // block 1
            make(1280, 2),  // block 2
        ];

        let mid = vec![]; // no mid-block motion modules in SDXL beta adapter

        let up = vec![
            make(1280, 3), // block 0
            make(640, 3),  // block 1
            make(320, 3),  // block 2
        ];

        Self { down, mid, up }
    }

    /// Build for SD 1.5 layout.
    ///
    /// SD 1.5 block layout:
    ///   down: [320(2,no-attn), 640(2,attn), 1280(2,attn), 1280(2,no-attn)]
    ///   mid: 1280 (1 attn)
    ///   up: [1280(3,no-attn), 1280(3,attn), 640(3,attn), 320(3,no-attn)]
    pub fn sd15_default(max_frames: usize) -> Self {
        let num_heads = 8;
        let num_layers = 1;
        let ff_mult = 4;

        let make = |channels: usize, count: usize| -> Vec<TemporalTransformer> {
            let inner = channels;
            let ff_inner = inner * ff_mult;
            (0..count)
                .map(|_| TemporalTransformer::new(channels, inner, num_heads, num_layers, ff_inner, max_frames))
                .collect()
        };

        let down = vec![
            vec![],          // block 0: 320, no attn
            make(640, 2),    // block 1: 640, attn
            make(1280, 2),   // block 2: 1280, attn
            vec![],          // block 3: 1280, no attn
        ];
        let mid = make(1280, 1);
        let up = vec![
            vec![],          // block 0: 1280, no attn
            make(1280, 3),   // block 1: 1280, attn
            make(640, 3),    // block 2: 640, attn
            vec![],          // block 3: 320, no attn
        ];

        Self { down, mid, up }
    }
}
