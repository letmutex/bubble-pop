//! # BubblePop
//!
//! Fast, lightweight speech bubble extraction for manga and comics in Rust.
//!
//! ## Quick Start
//!
//! ```no_run
//! use bubblepop::detect;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // One-shot detection from a file path
//!     let result = detect("manga_page.jpg")?;
//!     println!("Found {} bubbles in {:.2}ms", result.bubbles.len(), result.total_ms);
//!
//!     for bubble in &result.bubbles {
//!         println!("Confidence: {:.2}, Points: {}", bubble.confidence, bubble.points.len());
//!     }
//!     Ok(())
//! }
//! ```

mod bindings;
mod detector;
mod error;
mod input;
mod options;
mod pipeline;
mod runtime;
mod types;

pub use detector::BubbleDetector;
pub use error::BubblePopError;
pub use input::{ImageSource, IntoImageSource, PixelFormat, RawPixelView};
pub use options::{Backend, Options};
pub use types::{BubblePath, BubblePoint, DetectionResult, Phase, PhaseTiming};

/// Global function to detect speech bubbles from an image path, byte slice, or `DynamicImage`.
///
/// # Example
/// ```no_run
/// let result = bubblepop::detect("manga_page.jpg").unwrap();
/// println!("Found {} bubbles", result.bubbles.len());
/// ```
pub fn detect<'a, I: IntoImageSource<'a>>(input: I) -> Result<DetectionResult, BubblePopError> {
    BubbleDetector::new()?.detect(input)
}

/// Global detection function with custom detection options.
pub fn detect_with_options<'a, I: IntoImageSource<'a>>(
    input: I,
    options: Options,
) -> Result<DetectionResult, BubblePopError> {
    BubbleDetector::with_options(options)?.detect(input)
}
