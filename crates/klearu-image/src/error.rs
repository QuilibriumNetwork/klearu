//! Crate-level error type.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ImageGenError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("missing tensor: {0}")]
    MissingTensor(String),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("config error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, ImageGenError>;
