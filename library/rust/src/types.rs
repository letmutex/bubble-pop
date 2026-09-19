use std::fmt::Write;

/// A normalized coordinate in range [0.0, 1.0] relative to the source image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BubblePoint {
    pub x: f32,
    pub y: f32,
}

impl BubblePoint {
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Converts normalized coordinates [0.0, 1.0] to pixel coordinates for an image of given dimensions.
    #[inline]
    pub fn to_pixel_coords(&self, image_width: u32, image_height: u32) -> (f32, f32) {
        (self.x * image_width as f32, self.y * image_height as f32)
    }
}

/// A closed bubble contour in normalized coordinates with model scores.
#[derive(Debug, Clone, PartialEq)]
pub struct BubblePath {
    /// Normalized polygon vertices [0.0, 1.0] forming a closed boundary.
    pub points: Vec<BubblePoint>,
    /// Overall confidence score (mean of mask probability and instance score).
    pub confidence: f32,
    /// Mean mask probability within the bubble region.
    pub mask_probability: f32,
    /// Estimated unscaled area in source image pixels.
    pub area_pixels: u32,
}

impl BubblePath {
    /// Generates an SVG path string (e.g., `"M12.5,34.2 L56.1,78.9 ... Z"`) scaled to the given dimensions.
    pub fn to_svg_path(&self, image_width: u32, image_height: u32) -> String {
        let mut path = String::with_capacity(self.points.len() * 18);
        for (i, pt) in self.points.iter().enumerate() {
            let (px, py) = pt.to_pixel_coords(image_width, image_height);
            if i == 0 {
                let _ = write!(path, "M{:.1},{:.1}", px, py);
            } else {
                let _ = write!(path, " L{:.1},{:.1}", px, py);
            }
        }
        path.push('Z');
        path
    }
}

/// Distinct execution phases measured during speech bubble detection and decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    ImagePreparation,
    Preprocessing,
    Inference,
    ComponentsScoring,
    ContoursSimplification,
    ResultPacking,
}

/// Non-overlapping elapsed time duration for one detection phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseTiming {
    pub phase: Phase,
    pub duration_ms: f64,
}

/// Result of a speech bubble detection run, containing extracted contours, dimensions, and timing metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionResult {
    /// Extracted speech bubbles sorted in descending order of confidence score.
    pub bubbles: Vec<BubblePath>,
    /// Source image width in pixels.
    pub image_width: u32,
    /// Source image height in pixels.
    pub image_height: u32,
    /// Model inference duration in milliseconds.
    pub inference_ms: f64,
    /// Total wall-clock detection duration in milliseconds.
    pub total_ms: f64,
    /// Non-overlapping breakdown of all detection phases.
    pub phase_timings: Vec<PhaseTiming>,
}
