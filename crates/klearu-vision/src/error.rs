use std::io;

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensors(String),

    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    #[error("Weight loading error: {0}")]
    WeightLoad(String),

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Missing weight: {0}")]
    MissingWeight(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),
}

pub type Result<T> = std::result::Result<T, VisionError>;
