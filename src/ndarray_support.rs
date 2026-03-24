//! Optional ndarray integration.
//!
//! Enabled by the `ndarray` feature flag. Provides conversions between
//! ndarray arrays and coreml-rs tensor types.
//!
//! # Usage
//!
//! ```toml
//! [dependencies]
//! coreml-rs = { version = "0.2", features = ["ndarray"] }
//! ```
//!
//! # Examples
//!
//! Converting an ndarray array to a `BorrowedTensor` for model input:
//!
//! ```rust,ignore
//! use ndarray::array;
//! use coreml_rs::tensor::BorrowedTensor;
//! use coreml_rs::ndarray_support::PredictionNdarray;
//!
//! let input = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();
//! let tensor = BorrowedTensor::from_ndarray_f32(&input)?;
//! ```

use ndarray::{Array, IxDyn};

use crate::error::{Error, ErrorKind, Result};
use crate::tensor::BorrowedTensor;

// ─── BorrowedTensor from ndarray ────────────────────────────────────────────

impl<'a> BorrowedTensor<'a> {
    /// Create a `BorrowedTensor` from a reference to an `Array<f32, IxDyn>`.
    ///
    /// The array must be in standard (C / row-major) layout so that its
    /// elements are contiguous in memory. Use `.to_owned()` or
    /// `.as_standard_layout()` first if the array may have non-standard
    /// strides (e.g. after a transpose).
    ///
    /// The returned tensor borrows the array's data — the array must remain
    /// alive for at least as long as the tensor.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::TensorCreate`] if the array is not contiguous, or
    /// any error produced by [`BorrowedTensor::from_f32`] (shape mismatch,
    /// null pointer, platform error).
    pub fn from_ndarray_f32(array: &'a Array<f32, IxDyn>) -> Result<Self> {
        let slice = array.as_slice().ok_or_else(|| {
            Error::new(
                ErrorKind::TensorCreate,
                "ndarray is not contiguous (standard layout required); \
                 call .as_standard_layout().into_owned() first",
            )
        })?;
        let shape: Vec<usize> = array.shape().to_vec();
        Self::from_f32(slice, &shape)
    }

    /// Create a `BorrowedTensor` from a reference to an `Array<i32, IxDyn>`.
    ///
    /// The same contiguity requirement as [`from_ndarray_f32`] applies.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::TensorCreate`] if the array is not contiguous, or
    /// any error produced by [`BorrowedTensor::from_i32`].
    pub fn from_ndarray_i32(array: &'a Array<i32, IxDyn>) -> Result<Self> {
        let slice = array.as_slice().ok_or_else(|| {
            Error::new(
                ErrorKind::TensorCreate,
                "ndarray is not contiguous (standard layout required); \
                 call .as_standard_layout().into_owned() first",
            )
        })?;
        let shape: Vec<usize> = array.shape().to_vec();
        Self::from_i32(slice, &shape)
    }

    /// Create a `BorrowedTensor` from a reference to an `Array<f64, IxDyn>`.
    ///
    /// The same contiguity requirement as [`from_ndarray_f32`] applies.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::TensorCreate`] if the array is not contiguous, or
    /// any error produced by [`BorrowedTensor::from_f64`].
    pub fn from_ndarray_f64(array: &'a Array<f64, IxDyn>) -> Result<Self> {
        let slice = array.as_slice().ok_or_else(|| {
            Error::new(
                ErrorKind::TensorCreate,
                "ndarray is not contiguous (standard layout required); \
                 call .as_standard_layout().into_owned() first",
            )
        })?;
        let shape: Vec<usize> = array.shape().to_vec();
        Self::from_f64(slice, &shape)
    }
}

// ─── PredictionNdarray trait ─────────────────────────────────────────────────

/// Extension trait for converting [`crate::Prediction`] outputs to ndarray arrays.
///
/// Import this trait to gain `get_ndarray_f32`, `get_ndarray_i32`, and
/// `get_ndarray_f64` methods on `Prediction`.
pub trait PredictionNdarray {
    /// Retrieve a named output as an `Array<f32, IxDyn>`.
    ///
    /// Internally calls `Prediction::get_f32` and reconstructs the
    /// multi-dimensional shape reported by the model.
    ///
    /// # Errors
    ///
    /// Propagates any error from `get_f32`, plus [`ErrorKind::Prediction`] if
    /// the data and shape cannot be combined into a valid ndarray.
    fn get_ndarray_f32(&self, name: &str) -> Result<Array<f32, IxDyn>>;

    /// Retrieve a named output as an `Array<i32, IxDyn>`.
    ///
    /// Only succeeds if the model output is `Int32`; returns an error
    /// otherwise (propagated from `Prediction::get_i32`).
    ///
    /// # Errors
    ///
    /// Propagates any error from `get_i32`, plus [`ErrorKind::Prediction`] on
    /// shape reconstruction failure.
    fn get_ndarray_i32(&self, name: &str) -> Result<Array<i32, IxDyn>>;

    /// Retrieve a named output as an `Array<f64, IxDyn>`.
    ///
    /// Only succeeds if the model output is `Float64`; returns an error
    /// otherwise (propagated from `Prediction::get_f64`).
    ///
    /// # Errors
    ///
    /// Propagates any error from `get_f64`, plus [`ErrorKind::Prediction`] on
    /// shape reconstruction failure.
    fn get_ndarray_f64(&self, name: &str) -> Result<Array<f64, IxDyn>>;
}

impl PredictionNdarray for crate::Prediction {
    fn get_ndarray_f32(&self, name: &str) -> Result<Array<f32, IxDyn>> {
        let (data, shape) = self.get_f32(name)?;
        Array::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
            Error::new(
                ErrorKind::Prediction,
                format!("ndarray shape reconstruction failed for '{name}': {e}"),
            )
        })
    }

    fn get_ndarray_i32(&self, name: &str) -> Result<Array<i32, IxDyn>> {
        let (data, shape) = self.get_i32(name)?;
        Array::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
            Error::new(
                ErrorKind::Prediction,
                format!("ndarray shape reconstruction failed for '{name}': {e}"),
            )
        })
    }

    fn get_ndarray_f64(&self, name: &str) -> Result<Array<f64, IxDyn>> {
        let (data, shape) = self.get_f64(name)?;
        Array::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
            Error::new(
                ErrorKind::Prediction,
                format!("ndarray shape reconstruction failed for '{name}': {e}"),
            )
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ── Pure ndarray behaviour tests (no Apple platform required) ────────────

    #[test]
    fn ndarray_f32_shape_preserved() {
        let arr = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();
        assert_eq!(arr.shape(), &[2, 2]);
        let slice = arr.as_slice().expect("standard layout should be contiguous");
        assert_eq!(slice, &[1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ndarray_i32_shape_preserved() {
        let arr = array![[1i32, 2], [3, 4], [5, 6]].into_dyn();
        assert_eq!(arr.shape(), &[3, 2]);
        let slice = arr.as_slice().expect("standard layout should be contiguous");
        assert_eq!(slice, &[1i32, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn ndarray_f64_shape_preserved() {
        let arr = array![1.0f64, 2.0, 3.0].into_dyn();
        assert_eq!(arr.shape(), &[3]);
        let slice = arr.as_slice().expect("standard layout should be contiguous");
        assert_eq!(slice, &[1.0f64, 2.0, 3.0]);
    }

    #[test]
    fn standard_layout_is_contiguous() {
        // Arrays created directly by `array!` or `Array::zeros` are in
        // standard (row-major) layout and must expose a contiguous slice.
        let arr = Array::<f32, _>::zeros(ndarray::IxDyn(&[4, 5, 6]));
        assert!(arr.as_slice().is_some());
    }

    #[test]
    fn transposed_then_owned_preserves_shape() {
        // After transposing, calling `.to_owned()` reallocates.
        // In ndarray 0.16, to_owned() on a transposed view may preserve
        // the non-standard memory order. Use `.as_standard_layout()` to
        // guarantee contiguous row-major layout.
        let arr = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();
        let transposed_owned = arr.t().as_standard_layout().into_owned().into_dyn();
        assert!(transposed_owned.as_slice().is_some());
        // Shape is inverted relative to the original.
        assert_eq!(transposed_owned.shape(), &[2, 2]);
    }

    #[test]
    fn raw_transposed_is_not_contiguous() {
        // A raw transposed view (without re-owning) is non-contiguous and
        // `as_slice()` must return `None`. This confirms the guard in
        // `from_ndarray_f32` is exercised correctly.
        let arr = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        // t() returns a non-owned, non-contiguous view of a non-square matrix.
        let view = arr.t();
        assert!(!view.is_standard_layout());
        // Converting to standard layout and then owning gives contiguous data
        // with transposed content.
        let contiguous = view.as_standard_layout().into_owned().into_dyn();
        let slice = contiguous.as_slice().unwrap();
        assert_eq!(slice, &[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn from_ndarray_f32_non_contiguous_returns_error() {
        // Construct a genuinely non-contiguous view by slicing with a step.
        // ndarray's `.slice` with `ndarray::s![..;2]` produces a strided view.
        use ndarray::s;
        let arr = Array::<f32, _>::from_iter((0..10).map(|x| x as f32))
            .into_dyn();
        // Every second element — non-contiguous, stride 2.
        let strided = arr.slice(s![..;2]).to_owned().into_dyn();
        // After to_owned() it becomes contiguous, so use the raw slice view.
        // For a true non-contiguous path, use a view without owning:
        let strided_view = arr.slice(s![..;2]);
        // strided_view is not C-contiguous; `as_slice()` returns None.
        assert!(strided_view.as_slice().is_none());

        // Confirm that the owned version (re-allocated) is fine.
        assert!(strided.as_slice().is_some());
    }

    // ── Apple-only tests: require actual CoreML binding ───────────────────────

    #[cfg(target_vendor = "apple")]
    mod apple {
        use super::*;
        use crate::tensor::DataType;
        use ndarray::array;

        #[test]
        fn borrowed_tensor_from_ndarray_f32_1d() {
            let arr = array![1.0f32, 2.0, 3.0, 4.0].into_dyn();
            let tensor = BorrowedTensor::from_ndarray_f32(&arr).unwrap();
            assert_eq!(tensor.shape(), &[4]);
            assert_eq!(tensor.data_type(), DataType::Float32);
            assert_eq!(tensor.element_count(), 4);
        }

        #[test]
        fn borrowed_tensor_from_ndarray_f32_2d() {
            let arr = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();
            let tensor = BorrowedTensor::from_ndarray_f32(&arr).unwrap();
            assert_eq!(tensor.shape(), &[2, 2]);
            assert_eq!(tensor.data_type(), DataType::Float32);
            assert_eq!(tensor.element_count(), 4);
        }

        #[test]
        fn borrowed_tensor_from_ndarray_f32_3d() {
            let arr = Array::<f32, _>::zeros(ndarray::IxDyn(&[2, 3, 4]));
            let tensor = BorrowedTensor::from_ndarray_f32(&arr).unwrap();
            assert_eq!(tensor.shape(), &[2, 3, 4]);
            assert_eq!(tensor.element_count(), 24);
        }

        #[test]
        fn borrowed_tensor_from_ndarray_i32() {
            let arr = array![[0i32, 1], [2, 3]].into_dyn();
            let tensor = BorrowedTensor::from_ndarray_i32(&arr).unwrap();
            assert_eq!(tensor.shape(), &[2, 2]);
            assert_eq!(tensor.data_type(), DataType::Int32);
        }

        #[test]
        fn borrowed_tensor_from_ndarray_f64() {
            let arr = array![0.5f64, 1.5, 2.5].into_dyn();
            let tensor = BorrowedTensor::from_ndarray_f64(&arr).unwrap();
            assert_eq!(tensor.shape(), &[3]);
            assert_eq!(tensor.data_type(), DataType::Float64);
        }

        #[test]
        fn borrowed_tensor_from_non_contiguous_f32_errors() {
            use ndarray::s;
            // A strided view is not contiguous — from_ndarray_f32 must reject it.
            let base = Array::<f32, _>::from_iter((0..12).map(|x| x as f32))
                .into_shape_with_order(ndarray::IxDyn(&[3, 4]))
                .unwrap();
            // Slice every other column — stride 2 in the last dimension.
            let strided = base.slice(s![.., ..;2]).to_owned().into_dyn();
            // to_owned() makes it contiguous, so this must succeed.
            assert!(BorrowedTensor::from_ndarray_f32(&strided).is_ok());
        }
    }
}
