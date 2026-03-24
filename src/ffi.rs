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
///
/// Verified values from objc2-core-ml-0.3.2 generated source:
///   Float16 = 0x10000 | 16 = 65552
///   Float32 = 0x10000 | 32 = 65568
///   Float64 = 0x10000 | 64 = 65600
///   Int32   = 0x20000 | 32 = 131104
///   Int8    = 0x20000 | 8  = 131080  (defined in CoreML headers)
///
/// Computed from Apple's pattern (not defined in CoreML headers):
///   Int16   = 0x20000 | 16 = 131088  (signed int, 16-bit; not a CoreML constant)
///
/// Unsigned integer types have no MLMultiArrayDataType constant in CoreML headers.
/// Values below use a private sentinel range (0x30000 | bit_width) for internal
/// tracking only. CoreML will reject these values at runtime; callers must be
/// aware that UInt32/UInt16/UInt8 tensors cannot be passed to CoreML models directly.
///   UInt32 = 0x30000 | 32 = 196640
///   UInt16 = 0x30000 | 16 = 196624
///   UInt8  = 0x30000 | 8  = 196616
pub(crate) fn datatype_to_ml(dt: DataType) -> isize {
    match dt {
        DataType::Float16 => 0x10000 | 16,   // 65552
        DataType::Float32 => 0x10000 | 32,   // 65568
        DataType::Float64 => 0x10000 | 64,   // 65600
        DataType::Int32   => 0x20000 | 32,   // 131104
        DataType::Int8    => 0x20000 | 8,    // 131080 — native CoreML constant
        DataType::Int16   => 0x20000 | 16,   // 131088 — computed; not a CoreML constant
        DataType::UInt32  => 0x30000 | 32,   // 196640 — sentinel; no CoreML mapping
        DataType::UInt16  => 0x30000 | 16,   // 196624 — sentinel; no CoreML mapping
        DataType::UInt8   => 0x30000 | 8,    // 196616 — sentinel; no CoreML mapping
    }
}

pub(crate) fn ml_to_datatype(raw: isize) -> Option<DataType> {
    match raw {
        65552  => Some(DataType::Float16),
        65568  => Some(DataType::Float32),
        65600  => Some(DataType::Float64),
        131104 => Some(DataType::Int32),
        131080 => Some(DataType::Int8),   // native CoreML constant
        131088 => Some(DataType::Int16),  // computed; not a CoreML constant
        // UInt32/UInt16/UInt8 use sentinel values with no CoreML header definition;
        // they are excluded from reverse mapping since CoreML will not produce them.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datatype_roundtrip() {
        // These types have a defined (or computed) MLMultiArrayDataType mapping and
        // support a full to/from roundtrip.
        for dt in [
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::Int32,
            DataType::Int8,   // native CoreML constant
            DataType::Int16,  // computed from Apple pattern
        ] {
            let raw = datatype_to_ml(dt);
            let back = ml_to_datatype(raw).unwrap();
            assert_eq!(dt, back);
        }
    }

    #[test]
    fn unsigned_types_no_coreml_mapping() {
        // UInt32, UInt16, UInt8 have no MLMultiArrayDataType constant in CoreML
        // headers. Their sentinel values do not reverse-map to a DataType.
        for dt in [DataType::UInt32, DataType::UInt16, DataType::UInt8] {
            let raw = datatype_to_ml(dt);
            assert_eq!(ml_to_datatype(raw), None,
                "unsigned type {dt} sentinel value {raw} should not reverse-map");
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
