//! Safe, ergonomic Rust bindings for Apple CoreML inference with ANE acceleration.
//!
//! # Platform Support
//!
//! Requires macOS or iOS. On non-Apple targets, types exist as stubs
//! returning `Error::UnsupportedPlatform`.

pub mod compile;
pub mod description;
pub mod error;
pub(crate) mod ffi;
pub mod state;
pub mod tensor;
pub mod batch;
pub mod compute;

pub use batch::{BatchPrediction, BatchProvider};
pub use compile::compile_model;
pub use compute::{available_devices, ComputeDevice};
pub use description::{FeatureDescription, FeatureType, ModelMetadata, ShapeConstraint};
pub use error::{Error, ErrorKind, Result};
pub use state::State;
pub use tensor::{AsMultiArray, BorrowedTensor, DataType, OwnedTensor};

/// Compute unit selection for CoreML model loading.
///
/// Default is `All` — uses CPU, GPU (Metal), and Apple Neural Engine
/// for maximum throughput. This is the whole point of native CoreML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ComputeUnits {
    /// CPU only — no GPU or ANE.
    CpuOnly,
    /// CPU + GPU (Metal) — no ANE.
    CpuAndGpu,
    /// CPU + Apple Neural Engine — no GPU.
    CpuAndNeuralEngine,
    /// All available: CPU + GPU + ANE. **Use this.**
    #[default]
    All,
}

impl std::fmt::Display for ComputeUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuOnly => write!(f, "CPU only"),
            Self::CpuAndGpu => write!(f, "CPU + GPU"),
            Self::CpuAndNeuralEngine => write!(f, "CPU + Neural Engine"),
            Self::All => write!(f, "All (CPU + GPU + ANE)"),
        }
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

#[cfg(target_vendor = "apple")]
pub struct Model {
    inner: objc2::rc::Retained<objc2_core_ml::MLModel>,
    path: std::path::PathBuf,
}

#[cfg(target_vendor = "apple")]
impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").field("path", &self.path).finish()
    }
}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct Model {
    _private: (),
}

// Apple documents MLModel.predictionFromFeatures as thread-safe for
// concurrent read-only predictions on the same model instance.
#[cfg(target_vendor = "apple")]
unsafe impl Send for Model {}
#[cfg(target_vendor = "apple")]
unsafe impl Sync for Model {}

impl Model {
    #[cfg(target_vendor = "apple")]
    pub fn load(path: impl AsRef<std::path::Path>, compute_units: ComputeUnits) -> Result<Self> {
        use objc2_core_ml::{MLComputeUnits, MLModel, MLModelConfiguration};

        let path = path.as_ref();
        let path_str = path.to_str().ok_or_else(|| {
            Error::new(ErrorKind::ModelLoad, "path contains non-UTF8 characters")
        })?;

        let url = objc2_foundation::NSURL::fileURLWithPath(&ffi::str_to_nsstring(path_str));
        let config = unsafe { MLModelConfiguration::new() };
        let ml_units = match compute_units {
            ComputeUnits::CpuOnly => MLComputeUnits(1),
            ComputeUnits::CpuAndGpu => MLComputeUnits::CPUAndGPU,
            ComputeUnits::CpuAndNeuralEngine => MLComputeUnits(2),
            ComputeUnits::All => MLComputeUnits::All,
        };
        unsafe { config.setComputeUnits(ml_units) };

        let inner = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
            .map_err(|e| Error::from_nserror(ErrorKind::ModelLoad, &e))?;

        Ok(Self { inner, path: path.to_path_buf() })
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn load(_path: impl AsRef<std::path::Path>, _compute_units: ComputeUnits) -> Result<Self> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
    }

    /// The filesystem path this model was loaded from.
    pub fn path(&self) -> &std::path::Path {
        #[cfg(target_vendor = "apple")]
        { &self.path }
        #[cfg(not(target_vendor = "apple"))]
        { std::path::Path::new("") }
    }

    /// Run a synchronous prediction with named input tensors.
    ///
    /// Accepts any type implementing `AsMultiArray` (both `BorrowedTensor` and `OwnedTensor`).
    #[cfg(target_vendor = "apple")]
    pub fn predict(&self, inputs: &[(&str, &dyn AsMultiArray)]) -> Result<Prediction> {
        use objc2::AnyThread;
        use objc2_core_ml::{MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue};
        use objc2_foundation::{NSDictionary, NSString};

        objc2::rc::autoreleasepool(|_pool| {
            let mut keys: Vec<objc2::rc::Retained<NSString>> = Vec::with_capacity(inputs.len());
            let mut vals: Vec<objc2::rc::Retained<MLFeatureValue>> = Vec::with_capacity(inputs.len());

            for &(name, tensor) in inputs {
                keys.push(ffi::str_to_nsstring(name));
                vals.push(unsafe { MLFeatureValue::featureValueWithMultiArray(tensor.as_ml_multi_array()) });
            }

            let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
            let val_refs: Vec<&MLFeatureValue> = vals.iter().map(|v| &**v).collect();

            let dict: objc2::rc::Retained<NSDictionary<NSString, MLFeatureValue>> =
                NSDictionary::from_slices(&key_refs, &val_refs);

            let dict_any: &NSDictionary<NSString, objc2::runtime::AnyObject> =
                unsafe { &*((&*dict) as *const NSDictionary<NSString, MLFeatureValue>
                    as *const NSDictionary<NSString, objc2::runtime::AnyObject>) };

            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    dict_any,
                )
            }
            .map_err(|e| Error::from_nserror(ErrorKind::Prediction, &e))?;

            let provider_ref: &objc2::runtime::ProtocolObject<dyn MLFeatureProvider> =
                objc2::runtime::ProtocolObject::from_ref(&*provider);

            let result = unsafe { self.inner.predictionFromFeatures_error(provider_ref) }
                .map_err(|e| Error::from_nserror(ErrorKind::Prediction, &e))?;

            Ok(Prediction { inner: result })
        })
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn predict(&self, _inputs: &[(&str, &dyn AsMultiArray)]) -> Result<Prediction> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
    }

    /// Get descriptions of all model inputs.
    #[cfg(target_vendor = "apple")]
    pub fn inputs(&self) -> Vec<FeatureDescription> {
        let desc = unsafe { self.inner.modelDescription() };
        let input_map = unsafe { desc.inputDescriptionsByName() };
        description::extract_features(&input_map)
    }

    /// Get descriptions of all model outputs.
    #[cfg(target_vendor = "apple")]
    pub fn outputs(&self) -> Vec<FeatureDescription> {
        let desc = unsafe { self.inner.modelDescription() };
        let output_map = unsafe { desc.outputDescriptionsByName() };
        description::extract_features(&output_map)
    }

    /// Get model metadata (author, description, version, license).
    #[cfg(target_vendor = "apple")]
    pub fn metadata(&self) -> ModelMetadata {
        let desc = unsafe { self.inner.modelDescription() };
        description::extract_metadata(&desc)
    }

    /// Create a new state for stateful prediction (macOS 15+ / iOS 18+).
    #[cfg(target_vendor = "apple")]
    pub fn new_state(&self) -> Result<State> {
        let inner = unsafe { self.inner.newState() };
        Ok(State { inner })
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn new_state(&self) -> Result<State> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
    }

    /// Run prediction with persistent state (macOS 15+ / iOS 18+).
    #[cfg(target_vendor = "apple")]
    pub fn predict_stateful(
        &self,
        inputs: &[(&str, &dyn AsMultiArray)],
        state: &State,
    ) -> Result<Prediction> {
        use objc2::AnyThread;
        use objc2_core_ml::{MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue};
        use objc2_foundation::{NSDictionary, NSString};

        objc2::rc::autoreleasepool(|_pool| {
            let mut keys: Vec<objc2::rc::Retained<NSString>> = Vec::with_capacity(inputs.len());
            let mut vals: Vec<objc2::rc::Retained<MLFeatureValue>> = Vec::with_capacity(inputs.len());

            for &(name, tensor) in inputs {
                keys.push(ffi::str_to_nsstring(name));
                vals.push(unsafe { MLFeatureValue::featureValueWithMultiArray(tensor.as_ml_multi_array()) });
            }

            let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
            let val_refs: Vec<&MLFeatureValue> = vals.iter().map(|v| &**v).collect();
            let dict: objc2::rc::Retained<NSDictionary<NSString, MLFeatureValue>> =
                NSDictionary::from_slices(&key_refs, &val_refs);
            let dict_any: &NSDictionary<NSString, objc2::runtime::AnyObject> =
                unsafe { &*((&*dict) as *const NSDictionary<NSString, MLFeatureValue>
                    as *const NSDictionary<NSString, objc2::runtime::AnyObject>) };

            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(), dict_any,
                )
            }
            .map_err(|e| Error::from_nserror(ErrorKind::Prediction, &e))?;

            let provider_ref: &objc2::runtime::ProtocolObject<dyn MLFeatureProvider> =
                objc2::runtime::ProtocolObject::from_ref(&*provider);

            let result = unsafe {
                self.inner.predictionFromFeatures_usingState_error(provider_ref, &state.inner)
            }
            .map_err(|e| Error::from_nserror(ErrorKind::Prediction, &e))?;

            Ok(Prediction { inner: result })
        })
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn predict_stateful(
        &self,
        _inputs: &[(&str, &dyn AsMultiArray)],
        _state: &State,
    ) -> Result<Prediction> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
    }

    /// Run batch prediction for multiple input sets at once.
    ///
    /// More efficient than calling `predict()` in a loop.
    #[cfg(target_vendor = "apple")]
    pub fn predict_batch(&self, batch: &batch::BatchProvider) -> Result<batch::BatchPrediction> {
        use objc2_core_ml::MLBatchProvider;

        let batch_ref: &objc2::runtime::ProtocolObject<dyn MLBatchProvider> =
            objc2::runtime::ProtocolObject::from_ref(&*batch.inner);

        let result = unsafe { self.inner.predictionsFromBatch_error(batch_ref) }
            .map_err(|e| Error::from_nserror(ErrorKind::Prediction, &e))?;

        Ok(batch::BatchPrediction { inner: result })
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn predict_batch(&self, _batch: &batch::BatchProvider) -> Result<batch::BatchPrediction> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn inputs(&self) -> Vec<FeatureDescription> { vec![] }

    #[cfg(not(target_vendor = "apple"))]
    pub fn outputs(&self) -> Vec<FeatureDescription> { vec![] }

    #[cfg(not(target_vendor = "apple"))]
    pub fn metadata(&self) -> ModelMetadata { ModelMetadata::default() }
}

// ─── Prediction result ──────────────────────────────────────────────────────

#[cfg(target_vendor = "apple")]
pub struct Prediction {
    inner: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_core_ml::MLFeatureProvider>>,
}

#[cfg(not(target_vendor = "apple"))]
pub struct Prediction {
    _private: (),
}

// The Prediction holds a Retained MLFeatureProvider which is reference-counted.
// Safe to move to another thread for output extraction.
#[cfg(target_vendor = "apple")]
unsafe impl Send for Prediction {}
#[cfg(target_vendor = "apple")]
unsafe impl Sync for Prediction {}

impl Prediction {
    /// Get an output as (Vec<f32>, shape), converting from the model's native data type.
    /// Allocates a new Vec for the output data.
    #[cfg(target_vendor = "apple")]
    #[allow(deprecated)]
    pub fn get_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        objc2::rc::autoreleasepool(|_pool| {
            let (count, shape, data_type, array) = self.get_output_array(name)?;
            let mut buf = vec![0.0f32; count];
            Self::copy_array_to_f32(&array, data_type, count, &mut buf)?;
            Ok((buf, shape))
        })
    }

    /// Copy an output into a caller-provided f32 buffer (zero-alloc hot path).
    ///
    /// Returns the shape. The buffer must be large enough to hold all elements.
    #[cfg(target_vendor = "apple")]
    #[allow(deprecated)]
    pub fn get_f32_into(&self, name: &str, buf: &mut [f32]) -> Result<Vec<usize>> {
        objc2::rc::autoreleasepool(|_pool| {
            let (count, shape, data_type, array) = self.get_output_array(name)?;
            if buf.len() < count {
                return Err(Error::new(
                    ErrorKind::InvalidShape,
                    format!("buffer length {} < output element count {count}", buf.len()),
                ));
            }
            Self::copy_array_to_f32(&array, data_type, count, buf)?;
            Ok(shape)
        })
    }

    #[cfg(target_vendor = "apple")]
    #[allow(deprecated)]
    #[allow(clippy::type_complexity)]
    fn get_output_array(
        &self,
        name: &str,
    ) -> Result<(
        usize,
        Vec<usize>,
        Option<DataType>,
        objc2::rc::Retained<objc2_core_ml::MLMultiArray>,
    )> {
        use objc2_core_ml::MLFeatureProvider;

        let ns_name = ffi::str_to_nsstring(name);
        let feature_val = unsafe { self.inner.featureValueForName(&ns_name) }.ok_or_else(|| {
            Error::new(ErrorKind::Prediction, format!("output '{name}' not found"))
        })?;

        let array = unsafe { feature_val.multiArrayValue() }.ok_or_else(|| {
            Error::new(ErrorKind::Prediction, format!("output '{name}' is not a multi-array"))
        })?;

        let shape = ffi::nsarray_to_shape(unsafe { &array.shape() });
        let count = tensor::element_count(&shape);
        let dt_raw = unsafe { array.dataType() };
        let data_type = ffi::ml_to_datatype(dt_raw.0);

        Ok((count, shape, data_type, array))
    }

    /// Copy MLMultiArray data into a flat f32 buffer in row-major (C-contiguous) order.
    ///
    /// CoreML MLMultiArray outputs may have non-row-major strides (especially when
    /// inference runs on GPU or ANE). This function reads the array's actual strides
    /// and iterates in logical (row-major) index order, computing the physical offset
    /// for each element using the strides.
    #[cfg(target_vendor = "apple")]
    #[allow(deprecated)]
    #[allow(clippy::needless_range_loop)]
    fn copy_array_to_f32(
        array: &objc2_core_ml::MLMultiArray,
        data_type: Option<DataType>,
        count: usize,
        buf: &mut [f32],
    ) -> Result<()> {
        unsafe {
            let ptr = array.dataPointer();
            let shape = ffi::nsarray_to_shape(&array.shape());
            let strides = ffi::nsarray_to_shape(&array.strides());
            let row_major_strides = tensor::compute_strides(&shape);
            let is_contiguous = strides == row_major_strides;

            if is_contiguous {
                // Fast path: data is already row-major contiguous
                match data_type {
                    Some(DataType::Float32) => {
                        let src = ptr.as_ptr() as *const f32;
                        std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), count);
                    }
                    Some(DataType::Float16) => {
                        let src = ptr.as_ptr() as *const u16;
                        for i in 0..count {
                            buf[i] = f16_to_f32(*src.add(i));
                        }
                    }
                    Some(DataType::Float64) => {
                        let src = ptr.as_ptr() as *const f64;
                        for i in 0..count {
                            buf[i] = *src.add(i) as f32;
                        }
                    }
                    Some(DataType::Int32) => {
                        let src = ptr.as_ptr() as *const i32;
                        for i in 0..count {
                            buf[i] = *src.add(i) as f32;
                        }
                    }
                    None => {
                        return Err(Error::new(
                            ErrorKind::Prediction,
                            "unsupported output data type",
                        ));
                    }
                }
            } else {
                // Slow path: non-contiguous strides — iterate in logical row-major order,
                // compute physical offset for each element using the actual strides.
                let ndims = shape.len();
                let mut indices = vec![0usize; ndims];

                macro_rules! strided_copy {
                    ($src_type:ty, $convert:expr) => {{
                        let src = ptr.as_ptr() as *const $src_type;
                        for logical_idx in 0..count {
                            let physical: usize = indices.iter()
                                .zip(strides.iter())
                                .map(|(&i, &s)| i * s)
                                .sum();
                            buf[logical_idx] = $convert(*src.add(physical));
                            // Increment indices in row-major order (last dim fastest)
                            for d in (0..ndims).rev() {
                                indices[d] += 1;
                                if indices[d] < shape[d] {
                                    break;
                                }
                                indices[d] = 0;
                            }
                        }
                    }};
                }

                match data_type {
                    Some(DataType::Float32) => strided_copy!(f32, |v: f32| v),
                    Some(DataType::Float16) => strided_copy!(u16, |v: u16| f16_to_f32(v)),
                    Some(DataType::Float64) => strided_copy!(f64, |v: f64| v as f32),
                    Some(DataType::Int32) => strided_copy!(i32, |v: i32| v as f32),
                    None => {
                        return Err(Error::new(
                            ErrorKind::Prediction,
                            "unsupported output data type",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn get_f32(&self, _name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn get_f32_into(&self, _name: &str, _buf: &mut [f32]) -> Result<Vec<usize>> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
    }

    /// Get an output as (Vec<i32>, shape). Only works if the output is Int32.
    #[cfg(target_vendor = "apple")]
    #[allow(deprecated)]
    pub fn get_i32(&self, name: &str) -> Result<(Vec<i32>, Vec<usize>)> {
        objc2::rc::autoreleasepool(|_pool| {
            let (count, shape, data_type, array) = self.get_output_array(name)?;
            match data_type {
                Some(DataType::Int32) => {
                    let mut buf = vec![0i32; count];
                    unsafe {
                        let ptr = array.dataPointer();
                        let src = ptr.as_ptr() as *const i32;
                        std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), count);
                    }
                    Ok((buf, shape))
                }
                Some(dt) => Err(Error::new(
                    ErrorKind::Prediction,
                    format!("output '{name}' is {dt}, not Int32"),
                )),
                None => Err(Error::new(ErrorKind::Prediction, "unsupported output data type")),
            }
        })
    }

    /// Get an output as (Vec<f64>, shape). Only works if the output is Float64.
    #[cfg(target_vendor = "apple")]
    #[allow(deprecated)]
    pub fn get_f64(&self, name: &str) -> Result<(Vec<f64>, Vec<usize>)> {
        objc2::rc::autoreleasepool(|_pool| {
            let (count, shape, data_type, array) = self.get_output_array(name)?;
            match data_type {
                Some(DataType::Float64) => {
                    let mut buf = vec![0.0f64; count];
                    unsafe {
                        let ptr = array.dataPointer();
                        let src = ptr.as_ptr() as *const f64;
                        std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), count);
                    }
                    Ok((buf, shape))
                }
                Some(dt) => Err(Error::new(
                    ErrorKind::Prediction,
                    format!("output '{name}' is {dt}, not Float64"),
                )),
                None => Err(Error::new(ErrorKind::Prediction, "unsupported output data type")),
            }
        })
    }

    /// Get an output as raw bytes and its shape + data type.
    #[cfg(target_vendor = "apple")]
    #[allow(deprecated)]
    pub fn get_raw(&self, name: &str) -> Result<(Vec<u8>, Vec<usize>, Option<DataType>)> {
        objc2::rc::autoreleasepool(|_pool| {
            let (count, shape, data_type, array) = self.get_output_array(name)?;
            let byte_size = data_type.map(|dt| dt.byte_size()).unwrap_or(4);
            let total_bytes = count * byte_size;
            let mut buf = vec![0u8; total_bytes];
            unsafe {
                let ptr = array.dataPointer();
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr() as *const u8,
                    buf.as_mut_ptr(),
                    total_bytes,
                );
            }
            Ok((buf, shape, data_type))
        })
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn get_i32(&self, _name: &str) -> Result<(Vec<i32>, Vec<usize>)> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn get_f64(&self, _name: &str) -> Result<(Vec<f64>, Vec<usize>)> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn get_raw(&self, _name: &str) -> Result<(Vec<u8>, Vec<usize>, Option<DataType>)> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
    }
}

/// Convert a half-precision float (u16 bits) to f32.
#[cfg(target_vendor = "apple")]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = 0i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            let exp32 = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
        }
    } else {
        let exp32 = exp + (127 - 15);
        f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_units_default_is_all() {
        assert_eq!(ComputeUnits::default(), ComputeUnits::All);
    }

    #[test]
    fn compute_units_display() {
        assert_eq!(format!("{}", ComputeUnits::CpuAndGpu), "CPU + GPU");
        assert_eq!(format!("{}", ComputeUnits::All), "All (CPU + GPU + ANE)");
    }

    #[test]
    fn compute_units_display_cpu_only() {
        assert_eq!(format!("{}", ComputeUnits::CpuOnly), "CPU only");
    }

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn model_load_fails_on_non_apple() {
        let err = Model::load("/tmp/fake.mlmodelc", ComputeUnits::All).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::UnsupportedPlatform);
    }
}
