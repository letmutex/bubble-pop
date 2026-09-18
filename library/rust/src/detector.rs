use std::borrow::Cow;
use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::BubblePopError;
use crate::input::{ImageSource, IntoImageSource, PixelFormat, RawPixelView};
use crate::options::Options;
use crate::pipeline::{
    INPUT_HEIGHT, INPUT_WIDTH, Letterbox, PostprocessContext, extract_bubbles, preprocess,
};
use crate::runtime::RuntimeSession;
use crate::types::{DetectionResult, Phase, PhaseTiming};

thread_local! {
    static PREPROCESS_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static POSTPROCESS_CTX: RefCell<PostprocessContext> = RefCell::new(PostprocessContext::default());
}

/// Speech bubble detector with embedded model weights.
#[derive(Clone)]
pub struct BubbleDetector {
    session: Arc<RuntimeSession>,
    options: Options,
}

impl BubbleDetector {
    /// Creates a detector using the default embedded model and default options (CPU, 0.60 confidence).
    pub fn new() -> Result<Self, BubblePopError> {
        Self::with_options(Options::default())
    }

    /// Creates a detector with custom options (backend + confidence threshold) using the embedded model.
    pub fn with_options(options: Options) -> Result<Self, BubblePopError> {
        let session = RuntimeSession::new(&options)?;
        Ok(Self {
            session: Arc::new(session),
            options,
        })
    }

    /// Creates a detector from a custom TFLite model file.
    pub fn from_model_file<P: AsRef<Path>>(
        path: P,
        options: Options,
    ) -> Result<Self, BubblePopError> {
        let session = RuntimeSession::from_file(path, &options)?;
        Ok(Self {
            session: Arc::new(session),
            options,
        })
    }

    /// Creates a detector from TFLite model bytes in memory.
    pub fn from_model_bytes(bytes: &[u8], options: Options) -> Result<Self, BubblePopError> {
        let session = RuntimeSession::from_bytes(Cow::Owned(bytes.to_vec()), &options)?;
        Ok(Self {
            session: Arc::new(session),
            options,
        })
    }

    /// Prewarms the inference runtime.
    pub fn prewarm(&self) -> Result<Duration, BubblePopError> {
        self.session.prewarm()
    }

    /// Returns the current detection options.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Sets new detection options.
    pub fn set_options(&mut self, options: Options) {
        self.options = options;
    }

    /// Detects bubbles from an image source (file path, byte slice, DynamicImage, etc.).
    pub fn detect<'a, I: IntoImageSource<'a>>(
        &self,
        input: I,
    ) -> Result<DetectionResult, BubblePopError> {
        let prep_start = Instant::now();
        let source = input.into_image_source()?;
        let prep_ms = prep_start.elapsed().as_secs_f64() * 1000.0;

        self.detect_source(&source, prep_ms)
    }

    /// Detects bubbles from a raw pixel buffer (e.g. RGBA8, RGB8, ARGB8, Gray8).
    pub fn detect_raw(&self, view: RawPixelView<'_>) -> Result<DetectionResult, BubblePopError> {
        let bpp = view.format.bytes_per_pixel();
        let stride = view.stride_bytes.unwrap_or(view.width as usize * bpp);
        self.detect_raw_impl(
            view.pixels,
            view.width,
            view.height,
            view.format,
            stride,
            0.0,
        )
    }

    fn detect_source(
        &self,
        source: &ImageSource<'_>,
        prep_ms: f64,
    ) -> Result<DetectionResult, BubblePopError> {
        self.detect_raw_impl(
            &source.data,
            source.width,
            source.height,
            source.format,
            source.stride_bytes,
            prep_ms,
        )
    }

    fn detect_raw_impl(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        format: PixelFormat,
        stride_bytes: usize,
        image_prep_ms: f64,
    ) -> Result<DetectionResult, BubblePopError> {
        if width == 0 || height == 0 {
            return Err(BubblePopError::InvalidInput(
                "Image dimensions must be positive".into(),
            ));
        }

        let bpp = format.bytes_per_pixel();
        let min_stride = (width as usize).checked_mul(bpp).ok_or_else(|| {
            BubblePopError::InvalidInput("Image width overflowed stride calculation".into())
        })?;

        if stride_bytes < min_stride {
            return Err(BubblePopError::InvalidInput(format!(
                "Stride {stride_bytes} bytes is smaller than row bytes {min_stride}"
            )));
        }

        let total_required_len = ((height as usize) - 1)
            .checked_mul(stride_bytes)
            .and_then(|row_offset| row_offset.checked_add(min_stride))
            .ok_or_else(|| {
                BubblePopError::InvalidInput(
                    "Image dimensions overflowed buffer size calculation".into(),
                )
            })?;

        if pixels.len() < total_required_len {
            return Err(BubblePopError::InvalidInput(format!(
                "Pixel buffer too small: expected at least {total_required_len} bytes, got {}",
                pixels.len()
            )));
        }

        let total_start = Instant::now();
        let letterbox = Letterbox::new(width as usize, height as usize);

        // 1. Preprocessing (thread-local buffer for zero lock contention)
        let preprocess_start = Instant::now();
        let ((bubbles, timings), inference_duration, preprocess_duration) =
            PREPROCESS_BUF.with(|buf_cell| {
                let mut buf = buf_cell.borrow_mut();
                let target_len = INPUT_HEIGHT * INPUT_WIDTH;
                if buf.len() != target_len {
                    buf.resize(target_len, 0.0);
                }

                preprocess(
                    pixels,
                    width as usize,
                    height as usize,
                    stride_bytes,
                    format,
                    &letterbox,
                    &mut buf,
                );
                let preprocess_duration = preprocess_start.elapsed();

                // 2. Inference & 3. Postprocessing
                let (out_channels, out_width, out_height) = self.session.output_shape();
                let (res, inf_dur) = POSTPROCESS_CTX.with(|ctx_cell| {
                    let mut ctx = ctx_cell.borrow_mut();
                    self.session.infer_with(&buf, |logits| {
                        extract_bubbles(
                            &mut ctx,
                            logits,
                            out_channels,
                            out_width,
                            out_height,
                            &letterbox,
                            self.options.confidence_threshold,
                        )
                    })
                })?;

                Ok::<_, BubblePopError>((res, inf_dur, preprocess_duration))
            })?;

        let total_duration = total_start.elapsed();
        let inference_ms = inference_duration.as_secs_f64() * 1000.0;
        let total_ms = total_duration.as_secs_f64() * 1000.0 + image_prep_ms;

        let phase_timings = vec![
            PhaseTiming {
                phase: Phase::ImagePreparation,
                duration_ms: image_prep_ms,
            },
            PhaseTiming {
                phase: Phase::Preprocessing,
                duration_ms: preprocess_duration.as_secs_f64() * 1000.0,
            },
            PhaseTiming {
                phase: Phase::Inference,
                duration_ms: inference_ms,
            },
            PhaseTiming {
                phase: Phase::ComponentsScoring,
                duration_ms: timings.scoring_ms,
            },
            PhaseTiming {
                phase: Phase::ContoursSimplification,
                duration_ms: timings.contour_ms,
            },
            PhaseTiming {
                phase: Phase::ResultPacking,
                duration_ms: timings.packing_ms,
            },
        ];

        Ok(DetectionResult {
            bubbles,
            image_width: width,
            image_height: height,
            inference_ms,
            total_ms,
            phase_timings,
        })
    }
}
