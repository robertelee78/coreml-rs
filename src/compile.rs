/// Model compilation (.mlmodel/.mlpackage -> .mlmodelc).
///
/// Covers FR-7.1, FR-7.2.

use crate::error::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

/// Compile a `.mlmodel` or `.mlpackage` to a `.mlmodelc` directory.
///
/// Returns the path to the compiled model directory.
/// The compiled model is placed in a temporary directory by CoreML;
/// you should copy it to a permanent location.
#[cfg(target_vendor = "apple")]
#[allow(deprecated)] // sync API is deprecated but async requires run loop
pub fn compile_model(source: impl AsRef<Path>) -> Result<PathBuf> {
    use objc2_core_ml::MLModel;

    let source = source.as_ref();
    let source_str = source.to_str().ok_or_else(|| {
        Error::new(ErrorKind::ModelLoad, "source path contains non-UTF8 characters")
    })?;

    let url = objc2_foundation::NSURL::fileURLWithPath(
        &crate::ffi::str_to_nsstring(source_str),
    );

    let compiled_url = unsafe { MLModel::compileModelAtURL_error(&url) }
        .map_err(|e| Error::from_nserror(ErrorKind::ModelLoad, &e))?;

    let compiled_path = compiled_url.path()
        .ok_or_else(|| Error::new(ErrorKind::ModelLoad, "compiled URL has no path"))?;

    Ok(PathBuf::from(compiled_path.to_string()))
}

#[cfg(not(target_vendor = "apple"))]
pub fn compile_model(_source: impl AsRef<Path>) -> Result<PathBuf> {
    Err(Error::new(
        ErrorKind::UnsupportedPlatform,
        "CoreML requires Apple platform",
    ))
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn compile_fails_on_non_apple() {
        let err = super::compile_model("/tmp/model.mlmodel").unwrap_err();
        assert_eq!(err.kind(), &crate::ErrorKind::UnsupportedPlatform);
    }
}
