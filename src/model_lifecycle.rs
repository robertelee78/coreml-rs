//! Model lifecycle management — unload/reload for memory efficiency.
//!
//! When managing multiple large models on resource-constrained devices,
//! unloading idle models reclaims GPU/ANE memory without losing the
//! ability to quickly reload them.
//!
//! # Design
//!
//! [`ModelHandle`] is a move-based state machine wrapping [`Model`]. State
//! transitions (`unload`, `reload`) consume `self` and return a new handle,
//! so the Rust type system prevents use-after-unload at compile time.
//!
//! ```text
//!   load()       unload()       reload()
//!  --------> Loaded -------> Unloaded -------> Loaded
//!                <-----------          <-----------
//!                  reload()              unload()
//! ```

use crate::error::{Error, ErrorKind, Result};
use crate::{ComputeUnits, Model};
use std::path::PathBuf;

/// A model handle that supports unloading and reloading.
///
/// Wraps a [`Model`] with lifecycle management. Unloading releases the
/// model's GPU/ANE resources (by dropping the inner `MLModel`) while
/// retaining the filesystem path and compute-unit configuration for
/// efficient reloading.
///
/// State transitions consume `self`, so the compiler prevents calling
/// `predict` on an unloaded model or double-unloading.
///
/// # Example
///
/// ```ignore
/// use coreml_native::{ComputeUnits, ModelHandle};
///
/// let handle = ModelHandle::load("model.mlmodelc", ComputeUnits::All)?;
/// let prediction = handle.predict(&[("input", &tensor)])?;
///
/// // Free GPU/ANE memory when the model is idle.
/// let handle = handle.unload()?;
/// assert!(!handle.is_loaded());
///
/// // Reload when needed again.
/// let handle = handle.reload()?;
/// let prediction = handle.predict(&[("input", &tensor)])?;
/// ```
pub enum ModelHandle {
    /// Model is loaded and ready for inference.
    Loaded {
        /// The loaded model instance.
        model: Model,
        /// The compute units used to load this model (preserved for reload).
        compute_units: ComputeUnits,
    },
    /// Model has been unloaded from memory. The path and configuration are
    /// retained so the model can be reloaded without the caller needing to
    /// remember them.
    Unloaded {
        /// Filesystem path to the compiled `.mlmodelc` bundle.
        path: PathBuf,
        /// Compute units to use when reloading.
        compute_units: ComputeUnits,
    },
}

impl ModelHandle {
    /// Load a compiled CoreML model and wrap it in a lifecycle handle.
    ///
    /// This is equivalent to [`Model::load`] but returns a `ModelHandle`
    /// that supports later unloading and reloading.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded (invalid path,
    /// corrupt model, or non-Apple platform).
    pub fn load(
        path: impl AsRef<std::path::Path>,
        compute_units: ComputeUnits,
    ) -> Result<Self> {
        let model = Model::load(&path, compute_units)?;
        Ok(Self::Loaded {
            model,
            compute_units,
        })
    }

    /// Wrap an already-loaded [`Model`] in a lifecycle handle.
    ///
    /// Use this when you already have a `Model` instance (e.g. loaded via
    /// [`Model::load_async`]) and want to add lifecycle management.
    pub fn from_model(model: Model, compute_units: ComputeUnits) -> Self {
        Self::Loaded {
            model,
            compute_units,
        }
    }

    /// Returns `true` if the model is currently loaded and ready for
    /// inference.
    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::Loaded { .. })
    }

    /// Returns the filesystem path this model was (or will be) loaded from.
    pub fn path(&self) -> &std::path::Path {
        match self {
            Self::Loaded { model, .. } => model.path(),
            Self::Unloaded { path, .. } => path,
        }
    }

    /// Returns the compute-unit configuration.
    pub fn compute_units(&self) -> ComputeUnits {
        match self {
            Self::Loaded { compute_units, .. } | Self::Unloaded { compute_units, .. } => {
                *compute_units
            }
        }
    }

    /// Get a reference to the loaded model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is currently unloaded.
    pub fn model(&self) -> Result<&Model> {
        match self {
            Self::Loaded { model, .. } => Ok(model),
            Self::Unloaded { .. } => Err(Error::new(
                ErrorKind::ModelLoad,
                "model is unloaded; call reload() first",
            )),
        }
    }

    /// Unload the model from memory, releasing GPU/ANE resources.
    ///
    /// The path and compute-unit configuration are preserved so the model
    /// can be reloaded later via [`reload`](Self::reload).
    ///
    /// This method consumes `self` and returns a new `ModelHandle` in the
    /// `Unloaded` state. The inner `MLModel` is dropped, which tells CoreML
    /// to release its GPU and Neural Engine allocations.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is already unloaded.
    pub fn unload(self) -> Result<Self> {
        match self {
            Self::Loaded {
                model,
                compute_units,
            } => {
                let path = model.path().to_path_buf();
                // Dropping the Model releases the Retained<MLModel> and its
                // associated GPU/ANE resources.
                drop(model);
                Ok(Self::Unloaded {
                    path,
                    compute_units,
                })
            }
            Self::Unloaded { .. } => Err(Error::new(
                ErrorKind::ModelLoad,
                "model is already unloaded",
            )),
        }
    }

    /// Reload a previously unloaded model from its original path.
    ///
    /// Uses the same compute-unit configuration that was active when the
    /// model was first loaded.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is already loaded, or if reloading
    /// fails (e.g. the model file was deleted while unloaded).
    pub fn reload(self) -> Result<Self> {
        match self {
            Self::Unloaded {
                path,
                compute_units,
            } => {
                let model = Model::load(&path, compute_units)?;
                Ok(Self::Loaded {
                    model,
                    compute_units,
                })
            }
            Self::Loaded { .. } => Err(Error::new(
                ErrorKind::ModelLoad,
                "model is already loaded",
            )),
        }
    }

    /// Run a prediction on the loaded model.
    ///
    /// This is a convenience method that delegates to [`Model::predict`].
    ///
    /// # Errors
    ///
    /// Returns an error if the model is unloaded, or if prediction fails.
    pub fn predict(
        &self,
        inputs: &[(&str, &dyn crate::tensor::AsMultiArray)],
    ) -> Result<crate::Prediction> {
        self.model()?.predict(inputs)
    }

    /// Get descriptions of all model inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is unloaded.
    pub fn inputs(&self) -> Result<Vec<crate::FeatureDescription>> {
        Ok(self.model()?.inputs())
    }

    /// Get descriptions of all model outputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is unloaded.
    pub fn outputs(&self) -> Result<Vec<crate::FeatureDescription>> {
        Ok(self.model()?.outputs())
    }

    /// Get model metadata (author, description, version, license).
    ///
    /// # Errors
    ///
    /// Returns an error if the model is unloaded.
    pub fn metadata(&self) -> Result<crate::ModelMetadata> {
        Ok(self.model()?.metadata())
    }
}

impl std::fmt::Debug for ModelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loaded {
                model,
                compute_units,
            } => f
                .debug_struct("ModelHandle")
                .field("state", &"Loaded")
                .field("path", &model.path())
                .field("compute_units", compute_units)
                .finish(),
            Self::Unloaded {
                path,
                compute_units,
            } => f
                .debug_struct("ModelHandle")
                .field("state", &"Unloaded")
                .field("path", path)
                .field("compute_units", compute_units)
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unloaded_handle_is_not_loaded() {
        let handle = ModelHandle::Unloaded {
            path: PathBuf::from("/test.mlmodelc"),
            compute_units: ComputeUnits::All,
        };
        assert!(!handle.is_loaded());
    }

    #[test]
    fn unloaded_handle_preserves_path() {
        let handle = ModelHandle::Unloaded {
            path: PathBuf::from("/models/my_model.mlmodelc"),
            compute_units: ComputeUnits::CpuAndGpu,
        };
        assert_eq!(
            handle.path(),
            std::path::Path::new("/models/my_model.mlmodelc")
        );
    }

    #[test]
    fn unloaded_handle_preserves_compute_units() {
        let handle = ModelHandle::Unloaded {
            path: PathBuf::from("/test.mlmodelc"),
            compute_units: ComputeUnits::CpuAndNeuralEngine,
        };
        assert_eq!(handle.compute_units(), ComputeUnits::CpuAndNeuralEngine);
    }

    #[test]
    fn unloaded_handle_rejects_model_access() {
        let handle = ModelHandle::Unloaded {
            path: PathBuf::from("/test.mlmodelc"),
            compute_units: ComputeUnits::All,
        };
        let err = handle.model().unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::ModelLoad);
        assert!(err.message().contains("unloaded"));
    }

    #[test]
    fn unloaded_handle_rejects_double_unload() {
        let handle = ModelHandle::Unloaded {
            path: PathBuf::from("/test.mlmodelc"),
            compute_units: ComputeUnits::All,
        };
        let err = handle.unload().unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::ModelLoad);
        assert!(err.message().contains("already unloaded"));
    }

    #[test]
    fn load_nonexistent_model_fails() {
        let result = ModelHandle::load("/nonexistent.mlmodelc", ComputeUnits::All);
        assert!(result.is_err());
    }

    #[test]
    fn debug_format_unloaded() {
        let handle = ModelHandle::Unloaded {
            path: PathBuf::from("/test.mlmodelc"),
            compute_units: ComputeUnits::All,
        };
        let debug = format!("{:?}", handle);
        assert!(debug.contains("Unloaded"));
        assert!(debug.contains("/test.mlmodelc"));
    }

    #[test]
    fn unloaded_handle_rejects_inputs() {
        let handle = ModelHandle::Unloaded {
            path: PathBuf::from("/test.mlmodelc"),
            compute_units: ComputeUnits::All,
        };
        assert!(handle.inputs().is_err());
    }

    #[test]
    fn unloaded_handle_rejects_outputs() {
        let handle = ModelHandle::Unloaded {
            path: PathBuf::from("/test.mlmodelc"),
            compute_units: ComputeUnits::All,
        };
        assert!(handle.outputs().is_err());
    }

    #[test]
    fn unloaded_handle_rejects_metadata() {
        let handle = ModelHandle::Unloaded {
            path: PathBuf::from("/test.mlmodelc"),
            compute_units: ComputeUnits::All,
        };
        assert!(handle.metadata().is_err());
    }

    #[cfg(target_vendor = "apple")]
    mod apple_tests {
        use super::*;

        #[test]
        fn load_unload_reload_cycle() {
            let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/test_linear.mlmodelc");
            if !model_path.exists() {
                // Skip if fixture is not available.
                return;
            }

            let handle = ModelHandle::load(&model_path, ComputeUnits::All).unwrap();
            assert!(handle.is_loaded());
            assert!(handle.model().is_ok());

            // Unload releases GPU/ANE resources.
            let handle = handle.unload().unwrap();
            assert!(!handle.is_loaded());
            assert!(handle.model().is_err());
            assert_eq!(handle.path(), model_path);

            // Reload brings the model back.
            let handle = handle.reload().unwrap();
            assert!(handle.is_loaded());
            assert!(handle.model().is_ok());
        }

        #[test]
        fn loaded_handle_rejects_double_reload() {
            let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/test_linear.mlmodelc");
            if !model_path.exists() {
                return;
            }

            let handle = ModelHandle::load(&model_path, ComputeUnits::All).unwrap();
            let err = handle.reload().unwrap_err();
            assert_eq!(err.kind(), &ErrorKind::ModelLoad);
            assert!(err.message().contains("already loaded"));
        }

        #[test]
        fn from_model_wraps_existing() {
            let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/test_linear.mlmodelc");
            if !model_path.exists() {
                return;
            }

            let model = Model::load(&model_path, ComputeUnits::All).unwrap();
            let handle = ModelHandle::from_model(model, ComputeUnits::All);
            assert!(handle.is_loaded());
            assert_eq!(handle.compute_units(), ComputeUnits::All);
        }

        #[test]
        fn debug_format_loaded() {
            let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/test_linear.mlmodelc");
            if !model_path.exists() {
                return;
            }

            let handle = ModelHandle::load(&model_path, ComputeUnits::All).unwrap();
            let debug = format!("{:?}", handle);
            assert!(debug.contains("Loaded"));
        }
    }
}
