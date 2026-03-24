//! Async model loading and prediction.
//!
//! Extends [`Model`](crate::Model) with async variants of load and predict
//! that use Apple's completion-handler-based APIs under the hood.

use crate::async_bridge::{self, CompletionFuture};
use crate::error::{Error, ErrorKind, Result};
use crate::{ComputeUnits, Prediction};

#[cfg(target_vendor = "apple")]
impl crate::Model {
    /// Load a compiled model asynchronously.
    ///
    /// Returns a `CompletionFuture` that resolves when loading completes.
    /// Use `.await` in async contexts or `.block_on()` for synchronous callers.
    ///
    /// Requires macOS 12+ / iOS 15+.
    pub fn load_async(
        path: impl AsRef<std::path::Path>,
        compute_units: ComputeUnits,
    ) -> Result<CompletionFuture<Self>> {
        use objc2_core_ml::{MLComputeUnits, MLModel, MLModelConfiguration};

        let path = path.as_ref();
        let path_str = path.to_str().ok_or_else(|| {
            Error::new(ErrorKind::ModelLoad, "path contains non-UTF8 characters")
        })?;

        let url =
            objc2_foundation::NSURL::fileURLWithPath(&crate::ffi::str_to_nsstring(path_str));
        let config = unsafe { MLModelConfiguration::new() };
        let ml_units = match compute_units {
            ComputeUnits::CpuOnly => MLComputeUnits(1),
            ComputeUnits::CpuAndGpu => MLComputeUnits::CPUAndGPU,
            ComputeUnits::CpuAndNeuralEngine => MLComputeUnits(2),
            ComputeUnits::All => MLComputeUnits::All,
        };
        unsafe { config.setComputeUnits(ml_units) };

        let (sender, future) = async_bridge::completion_channel();
        let sender_cell = std::cell::Cell::new(Some(sender));
        let owned_path = path.to_path_buf();

        let block = block2::RcBlock::new(
            move |model_ptr: *mut MLModel, error_ptr: *mut objc2_foundation::NSError| {
                let sender = sender_cell
                    .take()
                    .expect("completion handler called more than once");
                if model_ptr.is_null() {
                    if error_ptr.is_null() {
                        sender.send(Err(Error::new(
                            ErrorKind::ModelLoad,
                            "model load returned null with no error",
                        )));
                    } else {
                        let err = unsafe { &*error_ptr };
                        sender.send(Err(Error::from_nserror(ErrorKind::ModelLoad, err)));
                    }
                } else {
                    // Safety: model_ptr is non-null. Use retain to get +1 refcount.
                    let retained = unsafe { objc2::rc::Retained::retain(model_ptr) };
                    match retained {
                        Some(inner) => {
                            sender.send(Ok(crate::Model {
                                inner,
                                path: owned_path.clone(),
                            }));
                        }
                        None => {
                            sender.send(Err(Error::new(
                                ErrorKind::ModelLoad,
                                "failed to retain MLModel pointer",
                            )));
                        }
                    }
                }
            },
        );

        unsafe {
            MLModel::loadContentsOfURL_configuration_completionHandler(&url, &config, &block);
        }

        Ok(future)
    }

    /// Load a model from in-memory specification bytes asynchronously.
    ///
    /// Creates an `MLModelAsset` from the specification data (synchronously),
    /// then loads the model asynchronously via the completion handler API.
    ///
    /// The `data` parameter should contain the contents of a `.mlmodel` file
    /// (the protobuf specification, not a compiled `.mlmodelc`).
    ///
    /// Requires macOS 14.4+ / iOS 17.4+.
    pub fn load_from_bytes(
        data: &[u8],
        compute_units: ComputeUnits,
    ) -> Result<CompletionFuture<Self>> {
        use objc2_core_ml::{MLComputeUnits, MLModel, MLModelAsset, MLModelConfiguration};
        use objc2_foundation::NSData;

        // Step 1: Create NSData from the byte slice (copies data).
        let ns_data = NSData::with_bytes(data);

        // Step 2: Create MLModelAsset synchronously.
        let asset =
            unsafe { MLModelAsset::modelAssetWithSpecificationData_error(&ns_data) }
                .map_err(|e| Error::from_nserror(ErrorKind::ModelLoad, &e))?;

        // Step 3: Configure compute units.
        let config = unsafe { MLModelConfiguration::new() };
        let ml_units = match compute_units {
            ComputeUnits::CpuOnly => MLComputeUnits(1),
            ComputeUnits::CpuAndGpu => MLComputeUnits::CPUAndGPU,
            ComputeUnits::CpuAndNeuralEngine => MLComputeUnits(2),
            ComputeUnits::All => MLComputeUnits::All,
        };
        unsafe { config.setComputeUnits(ml_units) };

        // Step 4: Load asynchronously via completion handler.
        let (sender, future) = async_bridge::completion_channel();
        let sender_cell = std::cell::Cell::new(Some(sender));

        let block = block2::RcBlock::new(
            move |model_ptr: *mut MLModel, error_ptr: *mut objc2_foundation::NSError| {
                let sender = sender_cell
                    .take()
                    .expect("completion handler called more than once");
                if model_ptr.is_null() {
                    if error_ptr.is_null() {
                        sender.send(Err(Error::new(
                            ErrorKind::ModelLoad,
                            "model load from bytes returned null with no error",
                        )));
                    } else {
                        let err = unsafe { &*error_ptr };
                        sender.send(Err(Error::from_nserror(ErrorKind::ModelLoad, err)));
                    }
                } else {
                    let retained = unsafe { objc2::rc::Retained::retain(model_ptr) };
                    match retained {
                        Some(inner) => {
                            sender.send(Ok(crate::Model {
                                inner,
                                path: std::path::PathBuf::from("<in-memory>"),
                            }));
                        }
                        None => {
                            sender.send(Err(Error::new(
                                ErrorKind::ModelLoad,
                                "failed to retain MLModel pointer",
                            )));
                        }
                    }
                }
            },
        );

        unsafe {
            MLModel::loadModelAsset_configuration_completionHandler(&asset, &config, &block);
        }

        Ok(future)
    }

    /// Run a prediction asynchronously.
    ///
    /// Builds the feature provider from the input tensors, then calls the
    /// async prediction API with a completion handler.
    ///
    /// Requires macOS 14+ / iOS 17+.
    pub fn predict_async(
        &self,
        inputs: &[(&str, &dyn crate::tensor::AsMultiArray)],
    ) -> Result<CompletionFuture<Prediction>> {
        use objc2::AnyThread;
        use objc2_core_ml::{MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue};
        use objc2_foundation::{NSDictionary, NSString};

        // Build the feature provider (same as sync predict).
        let provider = objc2::rc::autoreleasepool(|_pool| {
            let mut keys: Vec<objc2::rc::Retained<NSString>> =
                Vec::with_capacity(inputs.len());
            let mut vals: Vec<objc2::rc::Retained<MLFeatureValue>> =
                Vec::with_capacity(inputs.len());

            for &(name, tensor) in inputs {
                keys.push(crate::ffi::str_to_nsstring(name));
                vals.push(unsafe {
                    MLFeatureValue::featureValueWithMultiArray(tensor.as_ml_multi_array())
                });
            }

            let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
            let val_refs: Vec<&MLFeatureValue> = vals.iter().map(|v| &**v).collect();

            let dict: objc2::rc::Retained<NSDictionary<NSString, MLFeatureValue>> =
                NSDictionary::from_slices(&key_refs, &val_refs);

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

            Ok(provider)
        })?;

        let provider_ref: &objc2::runtime::ProtocolObject<dyn MLFeatureProvider> =
            objc2::runtime::ProtocolObject::from_ref(&*provider);

        let (sender, future) = async_bridge::completion_channel();
        let sender_cell = std::cell::Cell::new(Some(sender));

        let block = block2::RcBlock::new(
            move |result_ptr: *mut objc2::runtime::ProtocolObject<dyn MLFeatureProvider>,
                  error_ptr: *mut objc2_foundation::NSError| {
                let sender = sender_cell
                    .take()
                    .expect("completion handler called more than once");
                if result_ptr.is_null() {
                    if error_ptr.is_null() {
                        sender.send(Err(Error::new(
                            ErrorKind::Prediction,
                            "async prediction returned null with no error",
                        )));
                    } else {
                        let err = unsafe { &*error_ptr };
                        sender.send(Err(Error::from_nserror(ErrorKind::Prediction, err)));
                    }
                } else {
                    let retained = unsafe { objc2::rc::Retained::retain(result_ptr) };
                    match retained {
                        Some(inner) => {
                            sender.send(Ok(Prediction { inner }));
                        }
                        None => {
                            sender.send(Err(Error::new(
                                ErrorKind::Prediction,
                                "failed to retain prediction result pointer",
                            )));
                        }
                    }
                }
            },
        );

        unsafe {
            self.inner
                .predictionFromFeatures_completionHandler(provider_ref, &block);
        }

        Ok(future)
    }
}

// Non-Apple stubs
#[cfg(not(target_vendor = "apple"))]
impl crate::Model {
    /// Load a compiled model asynchronously (stub for non-Apple platforms).
    pub fn load_async(
        _path: impl AsRef<std::path::Path>,
        _compute_units: ComputeUnits,
    ) -> Result<CompletionFuture<Self>> {
        Err(Error::new(
            ErrorKind::UnsupportedPlatform,
            "CoreML requires Apple platform",
        ))
    }

    /// Load a model from in-memory bytes (stub for non-Apple platforms).
    pub fn load_from_bytes(
        _data: &[u8],
        _compute_units: ComputeUnits,
    ) -> Result<CompletionFuture<Self>> {
        Err(Error::new(
            ErrorKind::UnsupportedPlatform,
            "CoreML requires Apple platform",
        ))
    }

    /// Run a prediction asynchronously (stub for non-Apple platforms).
    pub fn predict_async(
        &self,
        _inputs: &[(&str, &dyn crate::tensor::AsMultiArray)],
    ) -> Result<CompletionFuture<Prediction>> {
        Err(Error::new(
            ErrorKind::UnsupportedPlatform,
            "CoreML requires Apple platform",
        ))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_vendor = "apple"))]
    use crate::{ComputeUnits, ErrorKind, Model};

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn load_async_fails_on_non_apple() {
        let err = Model::load_async("/tmp/fake.mlmodelc", ComputeUnits::All).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::UnsupportedPlatform);
    }

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn load_from_bytes_fails_on_non_apple() {
        let err = Model::load_from_bytes(&[0u8; 10], ComputeUnits::All).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::UnsupportedPlatform);
    }
}
