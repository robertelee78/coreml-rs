//! Internal FFI helpers for converting between Rust and Foundation types.

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2_foundation::{NSArray, NSNumber, NSString};

use crate::tensor::DataType;

#[cfg(target_vendor = "apple")]
pub(crate) fn shape_to_nsarray(shape: &[usize]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = shape
        .iter()
        .map(|&d| NSNumber::new_isize(d as isize))
        .collect();
    let refs: Vec<&NSNumber> = numbers.iter().map(|n| &**n).collect();
    NSArray::from_slice(&refs)
}

#[cfg(target_vendor = "apple")]
pub(crate) fn nsarray_to_shape(array: &NSArray<NSNumber>) -> Vec<usize> {
    let count = array.len();
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let num = array.objectAtIndex(i);
        result.push(num.as_isize() as usize);
    }
    result
}

#[cfg(target_vendor = "apple")]
pub(crate) fn str_to_nsstring(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

#[cfg(target_vendor = "apple")]
pub(crate) fn nsstring_to_string(s: &NSString) -> String {
    s.to_string()
}

/// Map our DataType to the raw isize value of MLMultiArrayDataType.
pub(crate) fn datatype_to_ml(dt: DataType) -> isize {
    match dt {
        DataType::Float16 => 65552,
        DataType::Float32 => 65568,
        DataType::Float64 => 65600,
        DataType::Int32 => 131104,
    }
}

pub(crate) fn ml_to_datatype(raw: isize) -> Option<DataType> {
    match raw {
        65552 => Some(DataType::Float16),
        65568 => Some(DataType::Float32),
        65600 => Some(DataType::Float64),
        131104 => Some(DataType::Int32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datatype_roundtrip() {
        for dt in [DataType::Float16, DataType::Float32, DataType::Float64, DataType::Int32] {
            let raw = datatype_to_ml(dt);
            let back = ml_to_datatype(raw).unwrap();
            assert_eq!(dt, back);
        }
    }

    #[test]
    fn ml_to_datatype_unknown() {
        assert_eq!(ml_to_datatype(999), None);
    }

    #[cfg(target_vendor = "apple")]
    mod apple_tests {
        use super::super::*;

        #[test]
        fn shape_roundtrip() {
            let shape = vec![1, 128, 500];
            let ns = shape_to_nsarray(&shape);
            let back = nsarray_to_shape(&ns);
            assert_eq!(shape, back);
        }

        #[test]
        fn shape_empty() {
            let shape: Vec<usize> = vec![];
            let ns = shape_to_nsarray(&shape);
            let back = nsarray_to_shape(&ns);
            assert_eq!(shape, back);
        }

        #[test]
        fn string_roundtrip() {
            let s = "audio_signal";
            let ns = str_to_nsstring(s);
            let back = nsstring_to_string(&ns);
            assert_eq!(s, back);
        }

        #[test]
        fn string_unicode() {
            let s = "input_\u{2581}test";
            let ns = str_to_nsstring(s);
            let back = nsstring_to_string(&ns);
            assert_eq!(s, back);
        }
    }
}
