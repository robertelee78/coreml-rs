//! Safe, ergonomic Rust bindings for Apple CoreML inference with ANE acceleration.
//!
//! # Platform Support
//!
//! Requires macOS or iOS. On non-Apple targets, types exist as stubs
//! returning `Error::UnsupportedPlatform`.

pub mod error;
pub(crate) mod ffi;
pub mod tensor;

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

#[cfg(not(target_vendor = "apple"))]
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
    /// Get an output as (Vec<f32>, shape).
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
        let mut buf = vec![0.0f32; count];

        unsafe {
            let ptr = array.dataPointer();
            let src = ptr.as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), count);
        }

        Ok((buf, shape))
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn get_f32(&self, _name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
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
