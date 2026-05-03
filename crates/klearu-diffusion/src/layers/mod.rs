//! Layer primitives for SD's conv-heavy architecture.
//!
//! Tensor layout convention: NCHW row-major, flattened as Vec<f32>.
//! Index: `idx(n, c, h, w) = ((n*C + c)*H + h)*W + w`.

pub mod activations;
pub mod attention;
pub mod conv2d;
pub mod group_norm;
pub mod layer_norm;
pub mod linear;
pub mod sample;

pub use activations::{gelu_inplace, quick_gelu_inplace, silu_inplace};
pub use attention::Attention;
pub use conv2d::Conv2d;
pub use group_norm::GroupNorm;
pub use layer_norm::LayerNorm;
pub use linear::Linear;
pub use sample::{Downsample, Upsample};
