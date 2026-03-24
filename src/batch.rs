//! Batch prediction support via MLArrayBatchProvider.
//!
//! More efficient than calling `predict()` in a loop when you have
//! multiple inputs to process. CoreML can optimize execution across
//! the entire batch.

use crate::error::{Error, ErrorKind, Result};
use crate::tensor::AsMultiArray;

// ─── BatchProvider ──────────────────────────────────────────────────────────

/// A batch of input feature sets for bulk prediction.
///
/// Wraps `MLArrayBatchProvider` to collect multiple input sets that can
/// be submitted to `Model::predict_batch` in a single call.
///
/// # Example
///
/// ```ignore
/// let inputs_0: &[(&str, &dyn AsMultiArray)] = &[("x", &tensor_a)];
/// let inputs_1: &[(&str, &dyn AsMultiArray)] = &[("x", &tensor_b)];
/// let batch = BatchProvider::new(&[&inputs_0[..], &inputs_1[..]])?;
/// assert_eq!(batch.count(), 2);
/// ```
#[cfg(target_vendor = "apple")]
pub struct BatchProvider {
    pub(crate) inner: objc2::rc::Retained<objc2_core_ml::MLArrayBatchProvider>,
}

#[cfg(target_vendor = "apple")]
impl BatchProvider {
    /// Create a batch provider from a slice of named input sets.
    ///
    /// Each element in `inputs` is a complete set of named tensors for one prediction.
    pub fn new(inputs: &[&[(&str, &dyn AsMultiArray)]]) -> Result<Self> {
        use objc2::AnyThread;
        use objc2::runtime::ProtocolObject;
        use objc2_core_ml::{
            MLArrayBatchProvider, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue,
        };
        use objc2_foundation::{NSDictionary, NSString};

        let mut providers: Vec<
            objc2::rc::Retained<ProtocolObject<dyn MLFeatureProvider>>,
        > = Vec::with_capacity(inputs.len());

        for input_set in inputs {
            let mut keys: Vec<objc2::rc::Retained<NSString>> =
                Vec::with_capacity(input_set.len());
            let mut vals: Vec<objc2::rc::Retained<MLFeatureValue>> =
                Vec::with_capacity(input_set.len());

            for &(name, tensor) in *input_set {
                keys.push(crate::ffi::str_to_nsstring(name));
                vals.push(unsafe {
                    MLFeatureValue::featureValueWithMultiArray(tensor.as_ml_multi_array())
                });
            }

            let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
            let val_refs: Vec<&MLFeatureValue> = vals.iter().map(|v| &**v).collect();

            let dict: objc2::rc::Retained<NSDictionary<NSString, MLFeatureValue>> =
                NSDictionary::from_slices(&key_refs, &val_refs);

            // Safety: MLFeatureValue is an NSObject subclass, so the pointer cast is valid.
            let dict_any: &NSDictionary<NSString, objc2::runtime::AnyObject> = unsafe {
                &*((&*dict) as *const NSDictionary<NSString, MLFeatureValue>
                    as *const NSDictionary<NSString, objc2::runtime::AnyObject>)
            };

            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    dict_any,
                )
            }
            .map_err(|e| Error::from_nserror(ErrorKind::Prediction, &e))?;

            let proto = ProtocolObject::from_retained(provider);
            providers.push(proto);
        }

        let provider_refs: Vec<&ProtocolObject<dyn MLFeatureProvider>> =
            providers.iter().map(|p| &**p).collect();
        let array = objc2_foundation::NSArray::from_slice(&provider_refs);

        let batch = unsafe {
            MLArrayBatchProvider::initWithFeatureProviderArray(
                MLArrayBatchProvider::alloc(),
                &array,
            )
        };

        Ok(Self { inner: batch })
    }

    /// Returns the number of input sets in this batch.
    pub fn count(&self) -> usize {
        use objc2_core_ml::MLBatchProvider;
        (unsafe { self.inner.count() }) as usize
    }
}

#[cfg(target_vendor = "apple")]
impl std::fmt::Debug for BatchProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchProvider")
            .field("count", &self.count())
            .finish()
    }
}

// Safety: MLArrayBatchProvider holds immutable data after construction.
#[cfg(target_vendor = "apple")]
unsafe impl Send for BatchProvider {}

// ─── Non-Apple stub ─────────────────────────────────────────────────────────

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct BatchProvider {
    _private: (),
}

#[cfg(not(target_vendor = "apple"))]
impl BatchProvider {
    pub fn new(_inputs: &[&[(&str, &dyn AsMultiArray)]]) -> Result<Self> {
        Err(Error::new(
            ErrorKind::UnsupportedPlatform,
            "CoreML requires Apple platform",
        ))
    }

    pub fn count(&self) -> usize {
        0
    }
}

// ─── BatchPrediction ────────────────────────────────────────────────────────

/// Result of a batch prediction, wrapping an `MLBatchProvider`.
///
/// Provides indexed access to individual prediction results.
#[cfg(target_vendor = "apple")]
pub struct BatchPrediction {
    pub(crate) inner:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_core_ml::MLBatchProvider>>,
}

#[cfg(target_vendor = "apple")]
impl BatchPrediction {
    /// Number of results in the batch.
    pub fn count(&self) -> usize {
        use objc2_core_ml::MLBatchProvider;
        (unsafe { self.inner.count() }) as usize
    }

    /// Get an output tensor as `(Vec<f32>, shape)` from the result at `index`.
    ///
    /// This is the batch equivalent of `Prediction::get_f32`.
    #[allow(deprecated)]
    #[allow(clippy::needless_range_loop)]
    pub fn get_f32(&self, index: usize, output_name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        use objc2_core_ml::{MLBatchProvider, MLFeatureProvider};

        if index >= self.count() {
            return Err(Error::new(
                ErrorKind::Prediction,
                format!(
                    "batch index {index} out of range (count: {})",
                    self.count()
                ),
            ));
        }

        objc2::rc::autoreleasepool(|_pool| {
            let provider = unsafe { self.inner.featuresAtIndex(index as isize) };

            let ns_name = crate::ffi::str_to_nsstring(output_name);
            let feature_val =
                unsafe { provider.featureValueForName(&ns_name) }.ok_or_else(|| {
                    Error::new(
                        ErrorKind::Prediction,
                        format!("output '{output_name}' not found at batch index {index}"),
                    )
                })?;

            let array = unsafe { feature_val.multiArrayValue() }.ok_or_else(|| {
                Error::new(
                    ErrorKind::Prediction,
                    format!(
                        "output '{output_name}' is not a multi-array at batch index {index}"
                    ),
                )
            })?;

            let shape = crate::ffi::nsarray_to_shape(unsafe { &array.shape() });
            let count = crate::tensor::element_count(&shape);
            let dt_raw = unsafe { array.dataType() };
            let data_type = crate::ffi::ml_to_datatype(dt_raw.0);

            let mut buf = vec![0.0f32; count];
            unsafe {
                let ptr = array.dataPointer();
                match data_type {
                    Some(crate::tensor::DataType::Float32) => {
                        let src = ptr.as_ptr() as *const f32;
                        std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), count);
                    }
                    Some(crate::tensor::DataType::Float16) => {
                        let src = ptr.as_ptr() as *const u16;
                        for i in 0..count {
                            buf[i] = crate::f16_to_f32(*src.add(i));
                        }
                    }
                    Some(crate::tensor::DataType::Float64) => {
                        let src = ptr.as_ptr() as *const f64;
                        for i in 0..count {
                            buf[i] = *src.add(i) as f32;
                        }
                    }
                    Some(crate::tensor::DataType::Int32) => {
                        let src = ptr.as_ptr() as *const i32;
                        for i in 0..count {
                            buf[i] = *src.add(i) as f32;
                        }
                    }
                    Some(crate::tensor::DataType::Int16) => {
                        let src = ptr.as_ptr() as *const i16;
                        for i in 0..count {
                            buf[i] = *src.add(i) as f32;
                        }
                    }
                    Some(crate::tensor::DataType::Int8) => {
                        let src = ptr.as_ptr() as *const i8;
                        for i in 0..count {
                            buf[i] = *src.add(i) as f32;
                        }
                    }
                    Some(crate::tensor::DataType::UInt32) => {
                        let src = ptr.as_ptr() as *const u32;
                        for i in 0..count {
                            buf[i] = *src.add(i) as f32;
                        }
                    }
                    Some(crate::tensor::DataType::UInt16) => {
                        let src = ptr.as_ptr() as *const u16;
                        for i in 0..count {
                            buf[i] = *src.add(i) as f32;
                        }
                    }
                    Some(crate::tensor::DataType::UInt8) => {
                        let src = ptr.as_ptr() as *const u8;
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
            }

            Ok((buf, shape))
        })
    }

    /// Get the feature provider at the given index (for advanced use).
    ///
    /// Returns a retained protocol object that can be queried for any output features.
    pub fn feature_provider(
        &self,
        index: usize,
    ) -> Result<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_core_ml::MLFeatureProvider>,
        >,
    > {
        use objc2_core_ml::MLBatchProvider;

        if index >= self.count() {
            return Err(Error::new(
                ErrorKind::Prediction,
                format!(
                    "batch index {index} out of range (count: {})",
                    self.count()
                ),
            ));
        }

        Ok(unsafe { self.inner.featuresAtIndex(index as isize) })
    }
}

#[cfg(target_vendor = "apple")]
impl std::fmt::Debug for BatchPrediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchPrediction")
            .field("count", &self.count())
            .finish()
    }
}

// Safety: the retained MLBatchProvider is reference-counted and read-only after creation.
#[cfg(target_vendor = "apple")]
unsafe impl Send for BatchPrediction {}

// ─── Non-Apple stub ─────────────────────────────────────────────────────────

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct BatchPrediction {
    _private: (),
}

#[cfg(not(target_vendor = "apple"))]
impl BatchPrediction {
    pub fn count(&self) -> usize {
        0
    }

    pub fn get_f32(
        &self,
        _index: usize,
        _output_name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        Err(Error::new(
            ErrorKind::UnsupportedPlatform,
            "CoreML requires Apple platform",
        ))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_vendor = "apple"))]
    use super::*;

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn batch_provider_fails_on_non_apple() {
        let inputs: &[&[(&str, &dyn AsMultiArray)]] = &[];
        let err = BatchProvider::new(inputs).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::UnsupportedPlatform);
    }

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn batch_prediction_fails_on_non_apple() {
        let pred = BatchPrediction { _private: () };
        assert_eq!(pred.count(), 0);
        let err = pred.get_f32(0, "output").unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::UnsupportedPlatform);
    }
}
