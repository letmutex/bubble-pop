use libloading::{Library, Symbol};
use std::borrow::Cow;
use std::ffi::{CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::BubblePopError;
use crate::options::Options;
use crate::pipeline::{INPUT_HEIGHT, INPUT_WIDTH};

// ============================================================
// Raw C Types & Enums (internal to bindings module)
// ============================================================

type TfLiteModel = c_void;
type TfLiteInterpreterOptions = c_void;
type TfLiteInterpreter = c_void;
type TfLiteTensor = c_void;
type TfLiteDelegate = c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct TfLiteXNNPackDelegateOptions {
    pub num_threads: i32,
    pub runtime_flags: u32,
    pub flags: u32,
    pub weights_cache: *mut c_void,
    pub handle_variable_ops: bool,
    pub weight_cache_file_path: *const c_char,
    pub weight_cache_file_descriptor: i32,
    pub weight_cache_provider: *mut c_void,
    pub weight_cache_lock_memory: bool,
}

impl Default for TfLiteXNNPackDelegateOptions {
    fn default() -> Self {
        Self {
            num_threads: 0,
            runtime_flags: 0,
            flags: 0x3,
            weights_cache: std::ptr::null_mut(),
            handle_variable_ops: false,
            weight_cache_file_path: std::ptr::null(),
            weight_cache_file_descriptor: 0,
            weight_cache_provider: std::ptr::null_mut(),
            weight_cache_lock_memory: false,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(dead_code)]
enum TfLiteStatus {
    Ok = 0,
    Error = 1,
    DelegateError = 2,
    ApplicationError = 3,
    DelegateDataNotFound = 4,
    DelegateDataWriteError = 5,
    DelegateDataReadError = 6,
    UnresolvedOps = 7,
    Cancelled = 8,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(dead_code)]
enum TfLiteType {
    NoType = 0,
    Float32 = 1,
    Int32 = 2,
    UInt8 = 3,
    Int64 = 4,
    String = 5,
    Bool = 6,
    Int16 = 7,
    Complex64 = 8,
    Int8 = 9,
    Float16 = 10,
    Float64 = 11,
    Complex128 = 12,
    UInt64 = 13,
    Resource = 14,
    Variant = 15,
    UInt32 = 16,
    UInt16 = 17,
    Int4 = 18,
}

// ============================================================
// C API Function Signatures
// ============================================================

type FnModelCreate =
    unsafe extern "C" fn(model_data: *const c_void, model_size: usize) -> *mut TfLiteModel;
type FnModelCreateFromFile = unsafe extern "C" fn(model_path: *const c_char) -> *mut TfLiteModel;
type FnModelDelete = unsafe extern "C" fn(model: *mut TfLiteModel);

type FnInterpreterOptionsCreate = unsafe extern "C" fn() -> *mut TfLiteInterpreterOptions;
type FnInterpreterOptionsSetNumThreads =
    unsafe extern "C" fn(options: *mut TfLiteInterpreterOptions, num_threads: i32);
type FnInterpreterOptionsAddDelegate =
    unsafe extern "C" fn(options: *mut TfLiteInterpreterOptions, delegate: *mut TfLiteDelegate);
type FnInterpreterOptionsDelete = unsafe extern "C" fn(options: *mut TfLiteInterpreterOptions);

type FnXNNPackDelegateCreate =
    unsafe extern "C" fn(options: *const TfLiteXNNPackDelegateOptions) -> *mut TfLiteDelegate;
type FnXNNPackDelegateDelete = unsafe extern "C" fn(delegate: *mut TfLiteDelegate);
type FnXNNPackOptionsDefault = unsafe extern "C" fn() -> TfLiteXNNPackDelegateOptions;

type FnInterpreterCreate = unsafe extern "C" fn(
    model: *const TfLiteModel,
    optional_options: *const TfLiteInterpreterOptions,
) -> *mut TfLiteInterpreter;
type FnInterpreterDelete = unsafe extern "C" fn(interpreter: *mut TfLiteInterpreter);
type FnInterpreterAllocateTensors =
    unsafe extern "C" fn(interpreter: *mut TfLiteInterpreter) -> TfLiteStatus;
type FnInterpreterInvoke =
    unsafe extern "C" fn(interpreter: *mut TfLiteInterpreter) -> TfLiteStatus;

type FnInterpreterGetInputTensorCount =
    unsafe extern "C" fn(interpreter: *const TfLiteInterpreter) -> i32;
type FnInterpreterGetOutputTensorCount =
    unsafe extern "C" fn(interpreter: *const TfLiteInterpreter) -> i32;
type FnInterpreterGetInputTensor = unsafe extern "C" fn(
    interpreter: *const TfLiteInterpreter,
    input_index: i32,
) -> *mut TfLiteTensor;
type FnInterpreterGetOutputTensor = unsafe extern "C" fn(
    interpreter: *const TfLiteInterpreter,
    output_index: i32,
) -> *const TfLiteTensor;

type FnTensorType = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> TfLiteType;
type FnTensorNumDims = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> i32;
type FnTensorDim = unsafe extern "C" fn(tensor: *const TfLiteTensor, dim_index: i32) -> i32;
type FnTensorByteSize = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> usize;
type FnTensorData = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> *mut c_void;

type FnVersion = unsafe extern "C" fn() -> *const c_char;

// ============================================================
// Dynamic Library Loader & Symbol Table
// ============================================================

#[allow(dead_code)]
pub(crate) struct TfLiteBindings {
    _lib: Library,
    model_create: Symbol<'static, FnModelCreate>,
    model_create_from_file: Symbol<'static, FnModelCreateFromFile>,
    model_delete: Symbol<'static, FnModelDelete>,
    options_create: Symbol<'static, FnInterpreterOptionsCreate>,
    options_set_num_threads: Symbol<'static, FnInterpreterOptionsSetNumThreads>,
    options_add_delegate: Symbol<'static, FnInterpreterOptionsAddDelegate>,
    options_delete: Symbol<'static, FnInterpreterOptionsDelete>,
    xnnpack_delegate_create: Option<Symbol<'static, FnXNNPackDelegateCreate>>,
    xnnpack_delegate_delete: Option<Symbol<'static, FnXNNPackDelegateDelete>>,
    xnnpack_options_default: Option<Symbol<'static, FnXNNPackOptionsDefault>>,
    interpreter_create: Symbol<'static, FnInterpreterCreate>,
    interpreter_delete: Symbol<'static, FnInterpreterDelete>,
    interpreter_allocate_tensors: Symbol<'static, FnInterpreterAllocateTensors>,
    interpreter_invoke: Symbol<'static, FnInterpreterInvoke>,
    interpreter_get_input_tensor_count: Symbol<'static, FnInterpreterGetInputTensorCount>,
    interpreter_get_output_tensor_count: Symbol<'static, FnInterpreterGetOutputTensorCount>,
    interpreter_get_input_tensor: Symbol<'static, FnInterpreterGetInputTensor>,
    interpreter_get_output_tensor: Symbol<'static, FnInterpreterGetOutputTensor>,
    tensor_type: Symbol<'static, FnTensorType>,
    tensor_num_dims: Symbol<'static, FnTensorNumDims>,
    tensor_dim: Symbol<'static, FnTensorDim>,
    tensor_byte_size: Symbol<'static, FnTensorByteSize>,
    tensor_data: Symbol<'static, FnTensorData>,
    version: Symbol<'static, FnVersion>,
}

impl TfLiteBindings {
    pub(crate) fn load(custom_path: Option<&Path>) -> Result<Arc<Self>, BubblePopError> {
        let last_err: Option<String>;
        let path = TfLiteBindings::resolve_shared_lib_path(custom_path);
        let Some(path) = path else {
            return Err(BubblePopError::RuntimeError(
                "Cannot resolve TensorFlow Lite runtime shared library. \
                Please ensure 'libtensorflowlite_c' is next to the executable."
                    .to_string(),
            ));
        };

        unsafe {
            match Library::new(&path) {
                Ok(lib) => match Self::from_library(lib) {
                    Ok(bindings) => return Ok(Arc::new(bindings)),
                    Err(e) => {
                        last_err = Some(format!(
                            "Failed to resolve symbols in {}: {e}",
                            path.display()
                        ))
                    }
                },
                Err(e) => {
                    last_err = Some(format!("Failed to load {}: {e}", path.display()));
                }
            }
        }

        Err(BubblePopError::RuntimeError(format!(
            "Failed to load TensorFlow Lite shared library (last error: {:?}). \
            Please ensure 'libtensorflowlite_c' is in your executable path.",
            last_err
        )))
    }

    fn resolve_shared_lib_path(custom_path: Option<&Path>) -> Option<PathBuf> {
        // Explicit path in Options
        if let Some(path) = custom_path {
            return Some(path.to_path_buf());
        }

        // Executable directory
        #[cfg(target_os = "windows")]
        let name = "libtensorflowlite_c.dll";
        #[cfg(target_os = "linux")]
        let name = "libtensorflowlite_c.so";
        #[cfg(target_os = "macos")]
        let name = "libtensorflowlite_c.dylib";
        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        let name = "libtensorflowlite_c.so";

        if let Ok(exe_path) = std::env::current_exe()
            && let Some(exe_dir) = exe_path.parent()
        {
            let path = exe_dir.join(name);
            if path.exists() {
                return Some(path);
            }
        }

        // Compile-time path injected by build.rs (downloaded or fetched prebuilt binary)
        if let Some(builtin_path) = option_env!("TFLITE_BUILTIN_LIB_PATH") {
            let path = PathBuf::from(builtin_path);
            if path.exists() {
                return Some(path);
            }
        }

        None
    }

    unsafe fn from_library(lib: Library) -> Result<Self, Box<dyn std::error::Error>> {
        macro_rules! load_sym {
            ($sym_name:ident, $sym_type:ty, $raw_name:literal) => {
                let $sym_name: Symbol<$sym_type> = unsafe { lib.get($raw_name)? };
                let $sym_name: Symbol<'static, $sym_type> =
                    unsafe { std::mem::transmute($sym_name) };
            };
        }

        load_sym!(model_create, FnModelCreate, b"TfLiteModelCreate\0");
        load_sym!(
            model_create_from_file,
            FnModelCreateFromFile,
            b"TfLiteModelCreateFromFile\0"
        );
        load_sym!(model_delete, FnModelDelete, b"TfLiteModelDelete\0");

        load_sym!(
            options_create,
            FnInterpreterOptionsCreate,
            b"TfLiteInterpreterOptionsCreate\0"
        );
        load_sym!(
            options_set_num_threads,
            FnInterpreterOptionsSetNumThreads,
            b"TfLiteInterpreterOptionsSetNumThreads\0"
        );
        load_sym!(
            options_add_delegate,
            FnInterpreterOptionsAddDelegate,
            b"TfLiteInterpreterOptionsAddDelegate\0"
        );
        load_sym!(
            options_delete,
            FnInterpreterOptionsDelete,
            b"TfLiteInterpreterOptionsDelete\0"
        );

        let xnnpack_options_default: Option<Symbol<FnXNNPackOptionsDefault>> =
            unsafe { lib.get(b"TfLiteXNNPackDelegateOptionsDefault\0").ok() };
        let xnnpack_options_default =
            xnnpack_options_default.map(|s| unsafe { std::mem::transmute(s) });

        let xnnpack_delegate_create: Option<Symbol<FnXNNPackDelegateCreate>> =
            unsafe { lib.get(b"TfLiteXNNPackDelegateCreate\0").ok() };
        let xnnpack_delegate_create =
            xnnpack_delegate_create.map(|s| unsafe { std::mem::transmute(s) });

        let xnnpack_delegate_delete: Option<Symbol<FnXNNPackDelegateDelete>> =
            unsafe { lib.get(b"TfLiteXNNPackDelegateDelete\0").ok() };
        let xnnpack_delegate_delete =
            xnnpack_delegate_delete.map(|s| unsafe { std::mem::transmute(s) });

        load_sym!(
            interpreter_create,
            FnInterpreterCreate,
            b"TfLiteInterpreterCreate\0"
        );
        load_sym!(
            interpreter_delete,
            FnInterpreterDelete,
            b"TfLiteInterpreterDelete\0"
        );
        load_sym!(
            interpreter_allocate_tensors,
            FnInterpreterAllocateTensors,
            b"TfLiteInterpreterAllocateTensors\0"
        );
        load_sym!(
            interpreter_invoke,
            FnInterpreterInvoke,
            b"TfLiteInterpreterInvoke\0"
        );

        load_sym!(
            interpreter_get_input_tensor_count,
            FnInterpreterGetInputTensorCount,
            b"TfLiteInterpreterGetInputTensorCount\0"
        );
        load_sym!(
            interpreter_get_output_tensor_count,
            FnInterpreterGetOutputTensorCount,
            b"TfLiteInterpreterGetOutputTensorCount\0"
        );
        load_sym!(
            interpreter_get_input_tensor,
            FnInterpreterGetInputTensor,
            b"TfLiteInterpreterGetInputTensor\0"
        );
        load_sym!(
            interpreter_get_output_tensor,
            FnInterpreterGetOutputTensor,
            b"TfLiteInterpreterGetOutputTensor\0"
        );

        load_sym!(tensor_type, FnTensorType, b"TfLiteTensorType\0");
        load_sym!(tensor_num_dims, FnTensorNumDims, b"TfLiteTensorNumDims\0");
        load_sym!(tensor_dim, FnTensorDim, b"TfLiteTensorDim\0");
        load_sym!(
            tensor_byte_size,
            FnTensorByteSize,
            b"TfLiteTensorByteSize\0"
        );
        load_sym!(tensor_data, FnTensorData, b"TfLiteTensorData\0");

        load_sym!(version, FnVersion, b"TfLiteVersion\0");

        Ok(Self {
            _lib: lib,
            model_create,
            model_create_from_file,
            model_delete,
            options_create,
            options_set_num_threads,
            options_add_delegate,
            options_delete,
            xnnpack_delegate_create,
            xnnpack_delegate_delete,
            xnnpack_options_default,
            interpreter_create,
            interpreter_delete,
            interpreter_allocate_tensors,
            interpreter_invoke,
            interpreter_get_input_tensor_count,
            interpreter_get_output_tensor_count,
            interpreter_get_input_tensor,
            interpreter_get_output_tensor,
            tensor_type,
            tensor_num_dims,
            tensor_dim,
            tensor_byte_size,
            tensor_data,
            version,
        })
    }
}

// ============================================================
// Safe RAII Interpreter Wrapper
// ============================================================

pub(crate) struct InterpreterState {
    _model_bytes: Option<Cow<'static, [u8]>>,
    bindings: Arc<TfLiteBindings>,
    model: *mut TfLiteModel,
    options: *mut TfLiteInterpreterOptions,
    xnnpack_delegate: *mut TfLiteDelegate,
    interpreter: *mut TfLiteInterpreter,
    input_tensor: *mut TfLiteTensor,
    output_tensor: *const TfLiteTensor,
    output_height: usize,
    output_width: usize,
    output_channels: usize,
}

unsafe impl Send for InterpreterState {}

impl Drop for InterpreterState {
    fn drop(&mut self) {
        unsafe {
            if !self.interpreter.is_null() {
                (self.bindings.interpreter_delete)(self.interpreter);
            }
            if !self.xnnpack_delegate.is_null()
                && let Some(xnn_delete) = &self.bindings.xnnpack_delegate_delete
            {
                xnn_delete(self.xnnpack_delegate);
            }
            if !self.options.is_null() {
                (self.bindings.options_delete)(self.options);
            }
            if !self.model.is_null() {
                (self.bindings.model_delete)(self.model);
            }
        }
    }
}

impl InterpreterState {
    pub(crate) fn from_bytes(
        bindings: &Arc<TfLiteBindings>,
        bytes: Cow<'static, [u8]>,
        options: &Options,
    ) -> Result<Self, BubblePopError> {
        let ptr = bytes.as_ptr() as *const c_void;
        let len = bytes.len();
        let mut state = Self::create(bindings, options, |b| unsafe { (b.model_create)(ptr, len) })?;
        state._model_bytes = Some(bytes);
        Ok(state)
    }

    pub(crate) fn from_file<P: AsRef<Path>>(
        bindings: &Arc<TfLiteBindings>,
        path: P,
        options: &Options,
    ) -> Result<Self, BubblePopError> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            BubblePopError::InvalidInput("Model path must be valid UTF-8".to_string())
        })?;
        let c_path = CString::new(path_str)
            .map_err(|e| BubblePopError::InvalidInput(format!("Invalid path string: {e}")))?;

        Self::create(bindings, options, |b| unsafe {
            (b.model_create_from_file)(c_path.as_ptr())
        })
    }

    #[inline]
    pub(crate) fn output_channels(&self) -> usize {
        self.output_channels
    }

    #[inline]
    pub(crate) fn output_width(&self) -> usize {
        self.output_width
    }

    #[inline]
    pub(crate) fn output_height(&self) -> usize {
        self.output_height
    }

    fn create<F>(
        bindings: &Arc<TfLiteBindings>,
        options: &Options,
        create_model_fn: F,
    ) -> Result<Self, BubblePopError>
    where
        F: FnOnce(&TfLiteBindings) -> *mut TfLiteModel,
    {
        let model = create_model_fn(bindings);
        if model.is_null() {
            return Err(BubblePopError::ModelLoadError(
                "TfLiteModelCreate failed".to_string(),
            ));
        }

        let interp_options = unsafe { (bindings.options_create)() };
        if interp_options.is_null() {
            unsafe { (bindings.model_delete)(model) };
            return Err(BubblePopError::ModelLoadError(
                "TfLiteInterpreterOptionsCreate failed".to_string(),
            ));
        }

        unsafe {
            (bindings.options_set_num_threads)(interp_options, options.num_threads as i32);
        }

        // Enable XNNPACK acceleration if available
        let xnnpack_delegate = if let Some(xnn_create) = &bindings.xnnpack_delegate_create {
            let mut xnn_opts = if let Some(xnn_default) = &bindings.xnnpack_options_default {
                unsafe { xnn_default() }
            } else {
                TfLiteXNNPackDelegateOptions::default()
            };
            xnn_opts.num_threads = options.num_threads as i32;

            let del = unsafe { xnn_create(&xnn_opts) };
            if !del.is_null() {
                unsafe { (bindings.options_add_delegate)(interp_options, del) };
                del
            } else {
                std::ptr::null_mut()
            }
        } else {
            std::ptr::null_mut()
        };

        let cleanup = |interpreter: *mut TfLiteInterpreter,
                       interp_options: *mut TfLiteInterpreterOptions,
                       xnnpack_del: *mut TfLiteDelegate,
                       model: *mut TfLiteModel| unsafe {
            if !interpreter.is_null() {
                (bindings.interpreter_delete)(interpreter);
            }
            if !xnnpack_del.is_null()
                && let Some(xnn_delete) = &bindings.xnnpack_delegate_delete
            {
                xnn_delete(xnnpack_del);
            }
            if !interp_options.is_null() {
                (bindings.options_delete)(interp_options);
            }
            if !model.is_null() {
                (bindings.model_delete)(model);
            }
        };

        let interpreter = unsafe { (bindings.interpreter_create)(model, interp_options) };
        if interpreter.is_null() {
            cleanup(
                std::ptr::null_mut(),
                interp_options,
                xnnpack_delegate,
                model,
            );
            return Err(BubblePopError::ModelLoadError(
                "TfLiteInterpreterCreate failed".to_string(),
            ));
        }

        let alloc_status = unsafe { (bindings.interpreter_allocate_tensors)(interpreter) };
        if alloc_status != TfLiteStatus::Ok {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::RuntimeError(format!(
                "TfLiteInterpreterAllocateTensors failed with status: {:?}",
                alloc_status
            )));
        }

        let in_count = unsafe { (bindings.interpreter_get_input_tensor_count)(interpreter) };
        let out_count = unsafe { (bindings.interpreter_get_output_tensor_count)(interpreter) };
        if in_count < 1 || out_count < 1 {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::ModelLoadError(format!(
                "Expected at least 1 input and 1 output tensor, found inputs: {in_count}, outputs: {out_count}"
            )));
        }

        let input_tensor = unsafe { (bindings.interpreter_get_input_tensor)(interpreter, 0) };
        let output_tensor = unsafe { (bindings.interpreter_get_output_tensor)(interpreter, 0) };

        if input_tensor.is_null() || output_tensor.is_null() {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::ModelLoadError(
                "Failed to retrieve model tensor handles".to_string(),
            ));
        }

        // Validate Input Tensor: Type, Dimension Count, Dimensions, and Byte Size
        let in_type = unsafe { (bindings.tensor_type)(input_tensor) };
        if in_type != TfLiteType::Float32 {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::ModelLoadError(format!(
                "Expected Float32 input tensor, got type ID {in_type:?}"
            )));
        }

        let in_dims = unsafe { (bindings.tensor_num_dims)(input_tensor) };
        if in_dims != 4 {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::ModelLoadError(format!(
                "Expected 4-dimensional input tensor [1, H, W, C], got {in_dims} dimensions"
            )));
        }

        let in_batch = unsafe { (bindings.tensor_dim)(input_tensor, 0) };
        let in_height = unsafe { (bindings.tensor_dim)(input_tensor, 1) };
        let in_width = unsafe { (bindings.tensor_dim)(input_tensor, 2) };
        let in_channels = unsafe { (bindings.tensor_dim)(input_tensor, 3) };
        if in_batch != 1
            || in_height != INPUT_HEIGHT as i32
            || in_width != INPUT_WIDTH as i32
            || in_channels != 1
        {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::ModelLoadError(format!(
                "Expected input tensor dimensions [1, {INPUT_HEIGHT}, {INPUT_WIDTH}, 1], got [{in_batch}, {in_height}, {in_width}, {in_channels}]"
            )));
        }

        let in_bytes = unsafe { (bindings.tensor_byte_size)(input_tensor) };
        let expected_in_bytes = INPUT_HEIGHT * INPUT_WIDTH * std::mem::size_of::<f32>();
        if in_bytes != expected_in_bytes {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::ModelLoadError(format!(
                "Input tensor byte size mismatch: expected {expected_in_bytes} bytes, got {in_bytes}"
            )));
        }

        // Validate Output Tensor: Type, Dimension Count, Dimensions, and Byte Size
        let out_type = unsafe { (bindings.tensor_type)(output_tensor) };
        if out_type != TfLiteType::Float32 {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::ModelLoadError(format!(
                "Expected Float32 output tensor, got type ID {out_type:?}"
            )));
        }

        let out_dims = unsafe { (bindings.tensor_num_dims)(output_tensor) };
        if out_dims != 4 {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::ModelLoadError(format!(
                "Expected 4-dimensional output tensor [1, H, W, C], got {out_dims} dimensions"
            )));
        }

        let out_batch = unsafe { (bindings.tensor_dim)(output_tensor, 0) };
        let out_height = unsafe { (bindings.tensor_dim)(output_tensor, 1) };
        let out_width = unsafe { (bindings.tensor_dim)(output_tensor, 2) };
        let out_channels = unsafe { (bindings.tensor_dim)(output_tensor, 3) };

        if out_batch <= 0 || out_height <= 0 || out_width <= 0 || out_channels <= 0 {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::ModelLoadError(format!(
                "Invalid output tensor dimensions: [{out_batch}, {out_height}, {out_width}, {out_channels}]"
            )));
        }

        let output_height = out_height as usize;
        let output_width = out_width as usize;
        let output_channels = out_channels as usize;

        let out_bytes = unsafe { (bindings.tensor_byte_size)(output_tensor) };
        let expected_out_bytes =
            output_height * output_width * output_channels * std::mem::size_of::<f32>();
        if out_bytes != expected_out_bytes {
            cleanup(interpreter, interp_options, xnnpack_delegate, model);
            return Err(BubblePopError::ModelLoadError(format!(
                "Output tensor byte size mismatch: expected {expected_out_bytes} bytes, got {out_bytes}"
            )));
        }

        Ok(Self {
            _model_bytes: None,
            bindings: Arc::clone(bindings),
            model,
            options: interp_options,
            xnnpack_delegate,
            interpreter,
            input_tensor,
            output_tensor,
            output_height,
            output_width,
            output_channels,
        })
    }

    pub(crate) fn infer_with<R, F>(
        &self,
        input_tensor: &[f32],
        f: F,
    ) -> Result<(R, Duration), BubblePopError>
    where
        F: FnOnce(&[f32]) -> R,
    {
        unsafe {
            let input_data_ptr = (self.bindings.tensor_data)(self.input_tensor) as *mut f32;
            if input_data_ptr.is_null() {
                return Err(BubblePopError::RuntimeError(
                    "Input tensor data pointer is null".to_string(),
                ));
            }

            // Direct buffer copy into input tensor
            std::ptr::copy_nonoverlapping(
                input_tensor.as_ptr(),
                input_data_ptr,
                input_tensor.len(),
            );

            let started = Instant::now();
            let status = (self.bindings.interpreter_invoke)(self.interpreter);
            let inference_duration = started.elapsed();

            if status != TfLiteStatus::Ok {
                return Err(BubblePopError::RuntimeError(format!(
                    "TfLiteInterpreterInvoke failed with status: {:?}",
                    status
                )));
            }

            let output_data_ptr = (self.bindings.tensor_data)(self.output_tensor) as *const f32;
            if output_data_ptr.is_null() {
                return Err(BubblePopError::RuntimeError(
                    "Output tensor data pointer is null".to_string(),
                ));
            }

            let output_len = self.output_height * self.output_width * self.output_channels;
            let output_slice = std::slice::from_raw_parts(output_data_ptr, output_len);
            let result = f(output_slice);

            Ok((result, inference_duration))
        }
    }
}
