//! Model compilation (.mlmodel/.mlpackage -> .mlmodelc).
//!
//! Covers FR-7.1, FR-7.2.

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

/// Compile a `.mlmodel` or `.mlpackage` asynchronously.
///
/// Returns a [`CompletionFuture`](crate::async_bridge::CompletionFuture) that
/// resolves to the path of the compiled `.mlmodelc` directory.
///
/// The compiled model is placed in a temporary directory by CoreML;
/// you should copy it to a permanent location.
///
/// Requires macOS 14.4+ / iOS 17.4+.
#[cfg(target_vendor = "apple")]
pub fn compile_model_async(
    source: impl AsRef<Path>,
) -> Result<crate::async_bridge::CompletionFuture<PathBuf>> {
    use objc2_core_ml::MLModel;

    let source = source.as_ref();
    let source_str = source.to_str().ok_or_else(|| {
        Error::new(ErrorKind::ModelLoad, "source path contains non-UTF8 characters")
    })?;

    let url = objc2_foundation::NSURL::fileURLWithPath(
        &crate::ffi::str_to_nsstring(source_str),
    );

    let (sender, future) = crate::async_bridge::completion_channel();
    let sender_cell = std::cell::Cell::new(Some(sender));

    let block = block2::RcBlock::new(
        move |compiled_url: *mut objc2_foundation::NSURL,
              error: *mut objc2_foundation::NSError| {
            let sender = sender_cell
                .take()
                .expect("completion handler called more than once");
            if compiled_url.is_null() {
                if error.is_null() {
                    sender.send(Err(Error::new(
                        ErrorKind::ModelLoad,
                        "compile returned null with no error",
                    )));
                } else {
                    let err = unsafe { &*error };
                    sender.send(Err(Error::from_nserror(ErrorKind::ModelLoad, err)));
                }
            } else {
                let url = unsafe { &*compiled_url };
                match url.path() {
                    Some(p) => sender.send(Ok(PathBuf::from(p.to_string()))),
                    None => sender.send(Err(Error::new(
                        ErrorKind::ModelLoad,
                        "compiled URL has no path",
                    ))),
                }
            }
        },
    );

    unsafe {
        MLModel::compileModelAtURL_completionHandler(&url, &block);
    }

    Ok(future)
}

/// Compile a `.mlmodel` or `.mlpackage` asynchronously (stub for non-Apple platforms).
#[cfg(not(target_vendor = "apple"))]
pub fn compile_model_async(
    _source: impl AsRef<Path>,
) -> Result<crate::async_bridge::CompletionFuture<PathBuf>> {
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

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn compile_async_fails_on_non_apple() {
        let err = super::compile_model_async("/tmp/model.mlmodel").unwrap_err();
        assert_eq!(err.kind(), &crate::ErrorKind::UnsupportedPlatform);
    }
}
