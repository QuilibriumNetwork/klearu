use thiserror::Error;

#[derive(Debug, Error)]
pub enum DiffusionError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("safetensors: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("checkpoint layout: {0}")]
    Layout(String),

    #[error("missing tensor: {0}")]
    MissingTensor(String),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("unsupported feature: {0}")]
    Unsupported(String),

    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),
}

pub type Result<T> = std::result::Result<T, DiffusionError>;
