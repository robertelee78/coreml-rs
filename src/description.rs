/// Model introspection types.
///
/// Covers FR-4.1, FR-4.2, FR-4.3.

use crate::tensor::DataType;

/// Description of a model feature (input or output).
#[derive(Debug, Clone)]
pub struct FeatureDescription {
    name: String,
    feature_type: FeatureType,
    shape: Option<Vec<usize>>,
    data_type: Option<DataType>,
    is_optional: bool,
}

impl FeatureDescription {
    pub fn name(&self) -> &str { &self.name }
    pub fn feature_type(&self) -> &FeatureType { &self.feature_type }
    pub fn shape(&self) -> Option<&[usize]> { self.shape.as_deref() }
    pub fn data_type(&self) -> Option<DataType> { self.data_type }
    pub fn is_optional(&self) -> bool { self.is_optional }
}

/// The type of a model feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureType {
    MultiArray,
    Image,
    Dictionary,
    Sequence,
    String,
    Int64,
    Double,
    Invalid,
}

impl std::fmt::Display for FeatureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MultiArray => write!(f, "MultiArray"),
            Self::Image => write!(f, "Image"),
            Self::Dictionary => write!(f, "Dictionary"),
            Self::Sequence => write!(f, "Sequence"),
            Self::String => write!(f, "String"),
            Self::Int64 => write!(f, "Int64"),
            Self::Double => write!(f, "Double"),
            Self::Invalid => write!(f, "Invalid"),
        }
    }
}

/// Model metadata.
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    pub author: Option<String>,
    pub description: Option<String>,
    pub version: Option<String>,
    pub license: Option<String>,
}

// ─── Apple platform builders ────────────────────────────────────────────────

#[cfg(target_vendor = "apple")]
pub(crate) fn extract_features(
    descriptions: &objc2_foundation::NSDictionary<
        objc2_foundation::NSString,
        objc2_core_ml::MLFeatureDescription,
    >,
) -> Vec<FeatureDescription> {
    use crate::ffi;
    use objc2_core_ml::MLFeatureType;

    let mut result = Vec::new();
    let keys = descriptions.allKeys();

    for key in keys.iter() {
        let name = ffi::nsstring_to_string(&key);

        if let Some(desc) = descriptions.objectForKey(&key) {
            let ft = unsafe { desc.r#type() };
            let is_optional = unsafe { desc.isOptional() };

            let feature_type = match ft {
                MLFeatureType::MultiArray => FeatureType::MultiArray,
                MLFeatureType::Image => FeatureType::Image,
                MLFeatureType::Dictionary => FeatureType::Dictionary,
                MLFeatureType::Sequence => FeatureType::Sequence,
                MLFeatureType::String => FeatureType::String,
                MLFeatureType::Int64 => FeatureType::Int64,
                MLFeatureType::Double => FeatureType::Double,
                _ => FeatureType::Invalid,
            };

            let (shape, data_type) = if feature_type == FeatureType::MultiArray {
                let constraint = unsafe { desc.multiArrayConstraint() };
                match constraint {
                    Some(c) => {
                        let ns_shape = unsafe { c.shape() };
                        let shape = ffi::nsarray_to_shape(&ns_shape);
                        let dt_raw = unsafe { c.dataType() };
                        let dt = ffi::ml_to_datatype(dt_raw.0);
                        (Some(shape), dt)
                    }
                    None => (None, None),
                }
            } else {
                (None, None)
            };

            result.push(FeatureDescription {
                name,
                feature_type,
                shape,
                data_type,
                is_optional,
            });
        }
    }

    result.sort_by(|a, b| a.name.cmp(&b.name));
    result
}

#[cfg(target_vendor = "apple")]
pub(crate) fn extract_metadata(
    model_desc: &objc2_core_ml::MLModelDescription,
) -> ModelMetadata {
    use crate::ffi;

    let meta = unsafe { model_desc.metadata() };
    let mut result = ModelMetadata::default();

    // Metadata keys are NSStrings. Try known keys.
    let author_key = ffi::str_to_nsstring("MLModelAuthorKey");
    let desc_key = ffi::str_to_nsstring("MLModelDescriptionKey");
    let version_key = ffi::str_to_nsstring("MLModelVersionStringKey");
    let license_key = ffi::str_to_nsstring("MLModelLicenseKey");

    if let Some(v) = meta.objectForKey(&author_key) {
        // Try to downcast to NSString
        if let Some(s) = v.downcast_ref::<objc2_foundation::NSString>() {
            result.author = Some(ffi::nsstring_to_string(s));
        }
    }
    if let Some(v) = meta.objectForKey(&desc_key) {
        if let Some(s) = v.downcast_ref::<objc2_foundation::NSString>() {
            result.description = Some(ffi::nsstring_to_string(s));
        }
    }
    if let Some(v) = meta.objectForKey(&version_key) {
        if let Some(s) = v.downcast_ref::<objc2_foundation::NSString>() {
            result.version = Some(ffi::nsstring_to_string(s));
        }
    }
    if let Some(v) = meta.objectForKey(&license_key) {
        if let Some(s) = v.downcast_ref::<objc2_foundation::NSString>() {
            result.license = Some(ffi::nsstring_to_string(s));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_type_display() {
        assert_eq!(format!("{}", FeatureType::MultiArray), "MultiArray");
        assert_eq!(format!("{}", FeatureType::Image), "Image");
    }

    #[test]
    fn feature_type_equality() {
        assert_eq!(FeatureType::MultiArray, FeatureType::MultiArray);
        assert_ne!(FeatureType::MultiArray, FeatureType::Image);
    }

    #[test]
    fn metadata_default() {
        let m = ModelMetadata::default();
        assert!(m.author.is_none());
        assert!(m.description.is_none());
        assert!(m.version.is_none());
        assert!(m.license.is_none());
    }

    #[test]
    fn feature_description_accessors() {
        let fd = FeatureDescription {
            name: "input".into(),
            feature_type: FeatureType::MultiArray,
            shape: Some(vec![1, 128, 500]),
            data_type: Some(DataType::Float32),
            is_optional: false,
        };
        assert_eq!(fd.name(), "input");
        assert_eq!(fd.feature_type(), &FeatureType::MultiArray);
        assert_eq!(fd.shape(), Some(&[1, 128, 500][..]));
        assert_eq!(fd.data_type(), Some(DataType::Float32));
        assert!(!fd.is_optional());
    }
}
