//! Safe, ergonomic Rust bindings for Apple CoreML inference with ANE acceleration.
//!
//! # Platform Support
//!
//! Requires macOS or iOS. On non-Apple targets, types exist as stubs
//! returning `Error::UnsupportedPlatform`.

pub mod description;
pub mod error;
pub(crate) mod ffi;
pub mod tensor;

pub use description::{FeatureDescription, FeatureType, ModelMetadata};
pub use error::{Error, ErrorKind, Result};
pub use tensor::{BorrowedTensor, DataType, OwnedTensor};

/// Compute unit selection for CoreML model loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ComputeUnits {
    CpuOnly,
    CpuAndGpu,
    CpuAndNeuralEngine,
    #[default]
    All,
}

impl std::fmt::Display for ComputeUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuOnly => write!(f, "CPU Only"),
            Self::CpuAndGpu => write!(f, "CPU + GPU"),
            Self::CpuAndNeuralEngine => write!(f, "CPU + Neural Engine"),
            Self::All => write!(f, "All (CPU + GPU + ANE)"),
        }
    }
}

// ─── Model (Apple) ──────────────────────────────────────────────────────────

#[cfg(target_vendor = "apple")]
pub struct Model {
    inner: objc2::rc::Retained<objc2_core_ml::MLModel>,
    #[allow(dead_code)]
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
            ComputeUnits::CpuOnly => MLComputeUnits::CPUOnly,
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

    #[cfg(target_vendor = "apple")]
    pub fn predict(&self, inputs: &[(&str, &BorrowedTensor<'_>)]) -> Result<Prediction> {
        use objc2::AnyThread;
        use objc2_core_ml::{MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue};
        use objc2_foundation::{NSDictionary, NSString};

        objc2::rc::autoreleasepool(|_pool| {
            // Build NSDictionary<NSString, MLFeatureValue>
            let mut keys: Vec<objc2::rc::Retained<NSString>> = Vec::with_capacity(inputs.len());
            let mut vals: Vec<objc2::rc::Retained<MLFeatureValue>> = Vec::with_capacity(inputs.len());

            for &(name, tensor) in inputs {
                keys.push(ffi::str_to_nsstring(name));
                vals.push(unsafe { MLFeatureValue::featureValueWithMultiArray(&tensor.inner) });
            }

            let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
            let val_refs: Vec<&MLFeatureValue> = vals.iter().map(|v| &**v).collect();

            // Create NSDictionary manually via from_retained_slice or from_slices
            let dict: objc2::rc::Retained<NSDictionary<NSString, MLFeatureValue>> =
                NSDictionary::from_slices(&key_refs, &val_refs);

            // MLDictionaryFeatureProvider expects NSDictionary<NSString, AnyObject>
            // We need to cast. MLFeatureValue is an NSObject subclass.
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

            // Cast to protocol object for prediction
            let provider_ref: &objc2::runtime::ProtocolObject<dyn MLFeatureProvider> =
                objc2::runtime::ProtocolObject::from_ref(&*provider);

            let result = unsafe { self.inner.predictionFromFeatures_error(provider_ref) }
                .map_err(|e| Error::from_nserror(ErrorKind::Prediction, &e))?;

            Ok(Prediction { inner: result })
        })
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn predict(&self, _inputs: &[(&str, &BorrowedTensor<'_>)]) -> Result<Prediction> {
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

    #[cfg(not(target_vendor = "apple"))]
    pub fn inputs(&self) -> Vec<FeatureDescription> { vec![] }

    #[cfg(not(target_vendor = "apple"))]
    pub fn outputs(&self) -> Vec<FeatureDescription> { vec![] }

    #[cfg(not(target_vendor = "apple"))]
    pub fn metadata(&self) -> ModelMetadata { ModelMetadata::default() }
}

#[cfg(target_vendor = "apple")]
unsafe impl Send for Model {}

// ─── Prediction result ──────────────────────────────────────────────────────

#[cfg(target_vendor = "apple")]
pub struct Prediction {
    inner: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_core_ml::MLFeatureProvider>>,
}

#[cfg(not(target_vendor = "apple"))]
pub struct Prediction {
    _private: (),
}

impl Prediction {
    /// Get an output as (Vec<f32>, shape), converting from the model's native data type.
    #[cfg(target_vendor = "apple")]
    #[allow(deprecated)]
    pub fn get_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
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

        let mut buf = vec![0.0f32; count];

        unsafe {
            let ptr = array.dataPointer();
            match data_type {
                Some(DataType::Float32) => {
                    let src = ptr.as_ptr() as *const f32;
                    std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), count);
                }
                Some(DataType::Float16) => {
                    // Convert f16 to f32
                    let src = ptr.as_ptr() as *const u16;
                    for i in 0..count {
                        let bits = *src.add(i);
                        buf[i] = f16_to_f32(bits);
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
                        format!("unsupported output data type for '{name}'"),
                    ));
                }
            }
        }

        Ok((buf, shape))
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn get_f32(&self, _name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
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
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize
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
        // Inf or NaN
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
        }
    } else {
        // Normal
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
        assert_eq!(format!("{}", ComputeUnits::CpuOnly), "CPU Only");
        assert_eq!(format!("{}", ComputeUnits::All), "All (CPU + GPU + ANE)");
    }

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn model_load_fails_on_non_apple() {
        let err = Model::load("/tmp/fake.mlmodelc", ComputeUnits::All).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::UnsupportedPlatform);
    }
}
