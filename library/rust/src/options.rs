use std::path::{Path, PathBuf};

/// Hardware execution backend selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Pure CPU execution backend.
    #[default]
    Cpu,
}

/// Unified options for speech bubble detection.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// Hardware execution backend (default: `Backend::Cpu`).
    pub backend: Backend,
    /// Minimum confidence threshold for detected bubbles in range [0.0, 1.0] (default: 0.60).
    pub confidence_threshold: f32,
    /// Number of worker threads for inference execution (default: 4).
    pub num_threads: usize,
    /// Optional explicit path to the LiteRT runtime shared library (`libLiteRt.[ext]`).
    pub runtime_lib_path: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            confidence_threshold: 0.60,
            num_threads: 4,
            runtime_lib_path: None,
        }
    }
}

impl Options {
    /// Sets the hardware execution backend.
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Sets the minimum confidence threshold in range [0.0, 1.0].
    pub fn with_confidence(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Sets the number of worker threads for inference.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads.max(1);
        self
    }

    /// Sets an explicit path to the TensorFlow Lite C runtime shared library.
    pub fn with_library_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.runtime_lib_path = Some(path.as_ref().to_path_buf());
        self
    }
}
