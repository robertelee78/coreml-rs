/// Tensor types for zero-copy data exchange with CoreML.

use crate::error::{Error, ErrorKind, Result};

/// Numeric data types supported by CoreML tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Float16,
    Float32,
    Float64,
    Int32,
}

impl DataType {
    pub fn byte_size(self) -> usize {
        match self {
            Self::Float16 => 2,
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Int32 => 4,
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float16 => write!(f, "Float16"),
            Self::Float32 => write!(f, "Float32"),
            Self::Float64 => write!(f, "Float64"),
            Self::Int32 => write!(f, "Int32"),
        }
    }
}

pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().copied().fold(1, |acc, d| acc * d)
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndims = shape.len();
    if ndims == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndims];
    for i in (0..ndims - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn validate_shape(data_len: usize, shape: &[usize]) -> Result<()> {
    if shape.is_empty() {
        return Err(Error::new(ErrorKind::InvalidShape, "shape must not be empty"));
    }
    if shape.iter().any(|&d| d == 0) {
        return Err(Error::new(
            ErrorKind::InvalidShape,
            format!("shape contains zero dimension: {shape:?}"),
        ));
    }
    let expected = element_count(shape);
    if data_len != expected {
        return Err(Error::new(
            ErrorKind::InvalidShape,
            format!("data length {data_len} does not match shape {shape:?} (expected {expected} elements)"),
        ));
    }
    Ok(())
}

// ─── Apple platform implementation ──────────────────────────────────────────

#[cfg(target_vendor = "apple")]
mod platform {
    use super::*;
    use crate::ffi;
    use objc2::rc::Retained;
    use objc2::AnyThread;
    use objc2_core_ml::MLMultiArray;
    use std::ffi::c_void;
    use std::ptr::NonNull;

    pub struct BorrowedTensor<'a> {
        pub(crate) inner: Retained<MLMultiArray>,
        shape: Vec<usize>,
        data_type: DataType,
        _marker: std::marker::PhantomData<&'a [u8]>,
    }

    impl std::fmt::Debug for BorrowedTensor<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("BorrowedTensor")
                .field("shape", &self.shape)
                .field("data_type", &self.data_type)
                .finish()
        }
    }

    impl<'a> BorrowedTensor<'a> {
        pub fn from_f32(data: &'a [f32], shape: &[usize]) -> Result<Self> {
            validate_shape(data.len(), shape)?;
            let ns_shape = ffi::shape_to_nsarray(shape);
            let strides = compute_strides(shape);
            let ns_strides = ffi::shape_to_nsarray(&strides);
            let ml_dtype = objc2_core_ml::MLMultiArrayDataType(ffi::datatype_to_ml(DataType::Float32));

            let ptr = NonNull::new(data.as_ptr() as *mut c_void).ok_or_else(|| {
                Error::new(ErrorKind::TensorCreate, "null data pointer")
            })?;

            let inner = unsafe {
                MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                    MLMultiArray::alloc(), ptr, &ns_shape, ml_dtype, &ns_strides, None,
                )
            }
            .map_err(|e| Error::from_nserror(ErrorKind::TensorCreate, &e))?;

            Ok(Self { inner, shape: shape.to_vec(), data_type: DataType::Float32, _marker: std::marker::PhantomData })
        }

        pub fn from_i32(data: &'a [i32], shape: &[usize]) -> Result<Self> {
            validate_shape(data.len(), shape)?;
            let ns_shape = ffi::shape_to_nsarray(shape);
            let strides = compute_strides(shape);
            let ns_strides = ffi::shape_to_nsarray(&strides);
            let ml_dtype = objc2_core_ml::MLMultiArrayDataType(ffi::datatype_to_ml(DataType::Int32));

            let ptr = NonNull::new(data.as_ptr() as *mut c_void).ok_or_else(|| {
                Error::new(ErrorKind::TensorCreate, "null data pointer")
            })?;

            let inner = unsafe {
                MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                    MLMultiArray::alloc(), ptr, &ns_shape, ml_dtype, &ns_strides, None,
                )
            }
            .map_err(|e| Error::from_nserror(ErrorKind::TensorCreate, &e))?;

            Ok(Self { inner, shape: shape.to_vec(), data_type: DataType::Int32, _marker: std::marker::PhantomData })
        }

        pub fn shape(&self) -> &[usize] { &self.shape }
        pub fn data_type(&self) -> DataType { self.data_type }
        pub fn element_count(&self) -> usize { element_count(&self.shape) }
        pub fn size_bytes(&self) -> usize { self.element_count() * self.data_type.byte_size() }
    }

    unsafe impl Send for BorrowedTensor<'_> {}

    pub struct OwnedTensor {
        pub(crate) inner: Retained<MLMultiArray>,
        shape: Vec<usize>,
        data_type: DataType,
    }

    impl std::fmt::Debug for OwnedTensor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("OwnedTensor")
                .field("shape", &self.shape)
                .field("data_type", &self.data_type)
                .finish()
        }
    }

    impl OwnedTensor {
        pub fn zeros(data_type: DataType, shape: &[usize]) -> Result<Self> {
            if shape.is_empty() {
                return Err(Error::new(ErrorKind::InvalidShape, "shape must not be empty"));
            }
            if shape.iter().any(|&d| d == 0) {
                return Err(Error::new(ErrorKind::InvalidShape, format!("shape contains zero dimension: {shape:?}")));
            }

            let ns_shape = ffi::shape_to_nsarray(shape);
            let ml_dtype = objc2_core_ml::MLMultiArrayDataType(ffi::datatype_to_ml(data_type));

            let inner = unsafe {
                MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &ns_shape, ml_dtype)
            }
            .map_err(|e| Error::from_nserror(ErrorKind::TensorCreate, &e))?;

            Ok(Self { inner, shape: shape.to_vec(), data_type })
        }

        pub fn shape(&self) -> &[usize] { &self.shape }
        pub fn data_type(&self) -> DataType { self.data_type }
        pub fn element_count(&self) -> usize { element_count(&self.shape) }
        pub fn size_bytes(&self) -> usize { self.element_count() * self.data_type.byte_size() }

        #[allow(deprecated)]
        pub fn copy_to_f32(&self, buf: &mut [f32]) -> Result<()> {
            if self.data_type != DataType::Float32 {
                return Err(Error::new(ErrorKind::TensorCreate, format!("tensor is {:?}, not Float32", self.data_type)));
            }
            let count = self.element_count();
            if buf.len() < count {
                return Err(Error::new(ErrorKind::InvalidShape, format!("buffer length {} < element count {count}", buf.len())));
            }
            unsafe {
                let ptr = self.inner.dataPointer();
                let src = ptr.as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), count);
            }
            Ok(())
        }

        pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
            let mut buf = vec![0.0f32; self.element_count()];
            self.copy_to_f32(&mut buf)?;
            Ok(buf)
        }
    }

    unsafe impl Send for OwnedTensor {}
}

// ─── Non-Apple stubs ────────────────────────────────────────────────────────

#[cfg(not(target_vendor = "apple"))]
mod platform {
    use super::*;

    #[derive(Debug)]
    pub struct BorrowedTensor<'a> {
        shape: Vec<usize>,
        data_type: DataType,
        _marker: std::marker::PhantomData<&'a [u8]>,
    }

    impl<'a> BorrowedTensor<'a> {
        pub fn from_f32(_data: &'a [f32], shape: &[usize]) -> Result<Self> {
            validate_shape(_data.len(), shape)?;
            Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
        }
        pub fn from_i32(_data: &'a [i32], shape: &[usize]) -> Result<Self> {
            validate_shape(_data.len(), shape)?;
            Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
        }
        pub fn shape(&self) -> &[usize] { &self.shape }
        pub fn data_type(&self) -> DataType { self.data_type }
        pub fn element_count(&self) -> usize { element_count(&self.shape) }
        pub fn size_bytes(&self) -> usize { self.element_count() * self.data_type.byte_size() }
    }

    #[derive(Debug)]
    pub struct OwnedTensor {
        shape: Vec<usize>,
        data_type: DataType,
    }

    impl OwnedTensor {
        pub fn zeros(_data_type: DataType, shape: &[usize]) -> Result<Self> {
            if shape.is_empty() || shape.iter().any(|&d| d == 0) {
                return Err(Error::new(ErrorKind::InvalidShape, format!("invalid shape: {shape:?}")));
            }
            Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
        }
        pub fn shape(&self) -> &[usize] { &self.shape }
        pub fn data_type(&self) -> DataType { self.data_type }
        pub fn element_count(&self) -> usize { element_count(&self.shape) }
        pub fn size_bytes(&self) -> usize { self.element_count() * self.data_type.byte_size() }
        pub fn copy_to_f32(&self, _buf: &mut [f32]) -> Result<()> {
            Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
        }
        pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
            Err(Error::new(ErrorKind::UnsupportedPlatform, "CoreML requires Apple platform"))
        }
    }
}

pub use platform::{BorrowedTensor, OwnedTensor};

/// Trait for types that can be used as prediction inputs.
///
/// Implemented by both `BorrowedTensor` and `OwnedTensor`.
#[cfg(target_vendor = "apple")]
pub trait AsMultiArray {
    fn as_ml_multi_array(&self) -> &objc2::rc::Retained<objc2_core_ml::MLMultiArray>;
}

#[cfg(target_vendor = "apple")]
impl AsMultiArray for BorrowedTensor<'_> {
    fn as_ml_multi_array(&self) -> &objc2::rc::Retained<objc2_core_ml::MLMultiArray> {
        &self.inner
    }
}

#[cfg(target_vendor = "apple")]
impl AsMultiArray for OwnedTensor {
    fn as_ml_multi_array(&self) -> &objc2::rc::Retained<objc2_core_ml::MLMultiArray> {
        &self.inner
    }
}

#[cfg(not(target_vendor = "apple"))]
pub trait AsMultiArray {}

#[cfg(not(target_vendor = "apple"))]
impl AsMultiArray for BorrowedTensor<'_> {}

#[cfg(not(target_vendor = "apple"))]
impl AsMultiArray for OwnedTensor {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datatype_byte_sizes() {
        assert_eq!(DataType::Float16.byte_size(), 2);
        assert_eq!(DataType::Float32.byte_size(), 4);
        assert_eq!(DataType::Float64.byte_size(), 8);
        assert_eq!(DataType::Int32.byte_size(), 4);
    }

    #[test]
    fn datatype_display() {
        assert_eq!(format!("{}", DataType::Float32), "Float32");
    }

    #[test]
    fn element_count_works() {
        assert_eq!(element_count(&[1, 128, 500]), 64000);
    }

    #[test]
    fn compute_strides_row_major() {
        assert_eq!(compute_strides(&[1, 128, 500]), vec![64000, 500, 1]);
    }

    #[test]
    fn validate_shape_correct() {
        assert!(validate_shape(64000, &[1, 128, 500]).is_ok());
    }

    #[test]
    fn validate_shape_mismatch() {
        let err = validate_shape(100, &[1, 128, 500]).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::InvalidShape);
    }

    #[test]
    fn validate_shape_empty() {
        assert!(validate_shape(0, &[]).is_err());
    }

    #[test]
    fn validate_shape_zero_dim() {
        assert!(validate_shape(0, &[1, 0, 500]).is_err());
    }

    #[cfg(target_vendor = "apple")]
    mod apple_tests {
        use super::super::*;

        #[test]
        fn borrowed_tensor_from_f32() {
            let data = vec![1.0f32; 6];
            let tensor = BorrowedTensor::from_f32(&data, &[2, 3]).unwrap();
            assert_eq!(tensor.shape(), &[2, 3]);
            assert_eq!(tensor.data_type(), DataType::Float32);
            assert_eq!(tensor.element_count(), 6);
            assert_eq!(tensor.size_bytes(), 24);
        }

        #[test]
        fn borrowed_tensor_shape_mismatch() {
            let data = vec![1.0f32; 5];
            assert!(BorrowedTensor::from_f32(&data, &[2, 3]).is_err());
        }

        #[test]
        fn borrowed_tensor_from_i32() {
            let data = vec![42i32; 4];
            let tensor = BorrowedTensor::from_i32(&data, &[2, 2]).unwrap();
            assert_eq!(tensor.data_type(), DataType::Int32);
        }

        #[test]
        fn owned_tensor_zeros() {
            let tensor = OwnedTensor::zeros(DataType::Float32, &[2, 3]).unwrap();
            assert_eq!(tensor.shape(), &[2, 3]);
            let data = tensor.to_vec_f32().unwrap();
            assert_eq!(data, vec![0.0f32; 6]);
        }

        #[test]
        fn owned_tensor_empty_shape_fails() {
            assert!(OwnedTensor::zeros(DataType::Float32, &[]).is_err());
        }

        #[test]
        fn owned_tensor_zero_dim_fails() {
            assert!(OwnedTensor::zeros(DataType::Float32, &[1, 0]).is_err());
        }

        #[test]
        fn owned_tensor_copy_wrong_type() {
            let tensor = OwnedTensor::zeros(DataType::Int32, &[4]).unwrap();
            let mut buf = vec![0.0f32; 4];
            assert!(tensor.copy_to_f32(&mut buf).is_err());
        }
    }
}
