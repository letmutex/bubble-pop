use thiserror::Error;

/// Error types for BubblePop.
#[derive(Error, Debug)]
pub enum BubblePopError {
    #[error("I/O error for path '{1}': {0}")]
    IoError(#[source] std::io::Error, String),

    #[error("Image decode error: {0}")]
    ImageDecodeError(String),

    #[error("Inference runtime error: {0}")]
    RuntimeError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Model load error: {0}")]
    ModelLoadError(String),
}
