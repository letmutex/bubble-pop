use std::borrow::Cow;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use crate::bindings::{InterpreterState, TfLiteBindings};
use crate::error::BubblePopError;
use crate::options::Options;
use crate::pipeline::{INPUT_HEIGHT, INPUT_WIDTH};

pub(crate) const EMBEDDED_MODEL_BYTES: &[u8] = include_bytes!("../models/model.tflite");

#[allow(dead_code)]
pub(crate) struct RuntimeSession {
    state: Mutex<InterpreterState>,
}

impl RuntimeSession {
    pub(crate) fn new(options: &Options) -> Result<Self, BubblePopError> {
        Self::from_bytes(Cow::Borrowed(EMBEDDED_MODEL_BYTES), options)
    }

    pub(crate) fn from_bytes(
        bytes: Cow<'static, [u8]>,
        options: &Options,
    ) -> Result<Self, BubblePopError> {
        let bindings = TfLiteBindings::load(options.runtime_lib_path.as_deref())?;
        let state = InterpreterState::from_bytes(&bindings, bytes, options)?;

        Ok(Self {
            state: Mutex::new(state),
        })
    }

    pub(crate) fn from_file<P: AsRef<Path>>(
        path: P,
        options: &Options,
    ) -> Result<Self, BubblePopError> {
        let bindings = TfLiteBindings::load(options.runtime_lib_path.as_deref())?;
        let state = InterpreterState::from_file(&bindings, path, options)?;

        Ok(Self {
            state: Mutex::new(state),
        })
    }

    pub(crate) fn prewarm(&self) -> Result<Duration, BubblePopError> {
        let dummy = vec![0.0f32; INPUT_HEIGHT * INPUT_WIDTH];
        let (_res, duration) = self.infer_with(&dummy, |_| ())?;
        Ok(duration)
    }

    /// Runs model inference with zero-copy borrowed logits.
    pub(crate) fn infer_with<R, F>(
        &self,
        input_tensor: &[f32],
        f: F,
    ) -> Result<(R, Duration), BubblePopError>
    where
        F: FnOnce(&[f32]) -> R,
    {
        if input_tensor.len() != INPUT_HEIGHT * INPUT_WIDTH {
            return Err(BubblePopError::InvalidInput(format!(
                "Expected input tensor of size {}, got {}",
                INPUT_HEIGHT * INPUT_WIDTH,
                input_tensor.len()
            )));
        }

        let state = self.state.lock().unwrap();
        state.infer_with(input_tensor, f)
    }

    #[inline]
    pub(crate) fn output_shape(&self) -> (usize, usize, usize) {
        let state = self.state.lock().unwrap();
        (
            state.output_channels(),
            state.output_width(),
            state.output_height(),
        )
    }
}
