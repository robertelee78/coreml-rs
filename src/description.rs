//! Model introspection types.
//!
//! Covers FR-4.1, FR-4.2, FR-4.3.

use crate::tensor::DataType;

/// Constraint on the shape of a multi-array feature.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeConstraint {
    /// Fixed shape -- only one shape is allowed.
    Fixed(Vec<usize>),
    /// One of several enumerated shapes.
    Enumerated(Vec<Vec<usize>>),
    /// Each dimension has an independent range (min, max).
    Range(Vec<(usize, usize)>),
    /// Unknown or unspecified constraint.
    Unspecified,
}

/// Description of a model feature (input or output).
#[derive(Debug, Clone)]
pub struct FeatureDescription {
    name: String,
    feature_type: FeatureType,
    shape: Option<Vec<usize>>,
    data_type: Option<DataType>,
    is_optional: bool,
    /// For MultiArray features, the shape constraint type.
    shape_constraint: Option<ShapeConstraint>,
}

impl FeatureDescription {
    pub fn name(&self) -> &str { &self.name }
    pub fn feature_type(&self) -> &FeatureType { &self.feature_type }
    pub fn shape(&self) -> Option<&[usize]> { self.shape.as_deref() }
    pub fn data_type(&self) -> Option<DataType> { self.data_type }
    pub fn is_optional(&self) -> bool { self.is_optional }
    pub fn shape_constraint(&self) -> Option<&ShapeConstraint> { self.shape_constraint.as_ref() }
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
    /// The name of the predicted feature (for classifier models).
    pub predicted_feature_name: Option<String>,
    /// The name of the predicted probabilities feature (for classifier models).
    pub predicted_probabilities_name: Option<String>,
    /// Whether the model supports on-device updates.
    pub is_updatable: bool,
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
    use objc2_core_ml::{MLFeatureType, MLMultiArrayShapeConstraintType};

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

            let (shape, data_type, shape_constraint) =
                if feature_type == FeatureType::MultiArray {
                    let constraint = unsafe { desc.multiArrayConstraint() };
                    match constraint {
                        Some(c) => {
                            let ns_shape = unsafe { c.shape() };
                            let shape = ffi::nsarray_to_shape(&ns_shape);
                            let dt_raw = unsafe { c.dataType() };
                            let dt = ffi::ml_to_datatype(dt_raw.0);

                            let sc = unsafe { c.shapeConstraint() };
                            let sc_type = unsafe { sc.r#type() };
                            let sc_val = match sc_type {
                                MLMultiArrayShapeConstraintType::Enumerated => {
                                    let enum_shapes = unsafe { sc.enumeratedShapes() };
                                    let mut shapes = Vec::new();
                                    for i in 0..enum_shapes.len() {
                                        let s = enum_shapes.objectAtIndex(i);
                                        shapes.push(ffi::nsarray_to_shape(&s));
                                    }
                                    ShapeConstraint::Enumerated(shapes)
                                }
                                MLMultiArrayShapeConstraintType::Range => {
                                    let range_vals = unsafe { sc.sizeRangeForDimension() };
                                    let mut ranges = Vec::new();
                                    for i in 0..range_vals.len() {
                                        let val = range_vals.objectAtIndex(i);
                                        let r = unsafe { val.rangeValue() };
                                        let lower = r.location;
                                        let upper = lower + r.length;
                                        ranges.push((lower, upper));
                                    }
                                    ShapeConstraint::Range(ranges)
                                }
                                _ => ShapeConstraint::Unspecified,
                            };

                            (Some(shape), dt, Some(sc_val))
                        }
                        None => (None, None, None),
                    }
                } else {
                    (None, None, None)
                };

            result.push(FeatureDescription {
                name,
                feature_type,
                shape,
                data_type,
                is_optional,
                shape_constraint,
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

    result.predicted_feature_name = unsafe { model_desc.predictedFeatureName() }
        .map(|s| ffi::nsstring_to_string(&s));
    result.predicted_probabilities_name = unsafe { model_desc.predictedProbabilitiesName() }
        .map(|s| ffi::nsstring_to_string(&s));
    result.is_updatable = unsafe { model_desc.isUpdatable() };

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
        assert!(m.predicted_feature_name.is_none());
        assert!(m.predicted_probabilities_name.is_none());
        assert!(!m.is_updatable);
    }

    #[test]
    fn feature_description_accessors() {
        let fd = FeatureDescription {
            name: "input".into(),
            feature_type: FeatureType::MultiArray,
            shape: Some(vec![1, 128, 500]),
            data_type: Some(DataType::Float32),
            is_optional: false,
            shape_constraint: Some(ShapeConstraint::Fixed(vec![1, 128, 500])),
        };
        assert_eq!(fd.name(), "input");
        assert_eq!(fd.feature_type(), &FeatureType::MultiArray);
        assert_eq!(fd.shape(), Some(&[1, 128, 500][..]));
        assert_eq!(fd.data_type(), Some(DataType::Float32));
        assert!(!fd.is_optional());
        assert_eq!(
            fd.shape_constraint(),
            Some(&ShapeConstraint::Fixed(vec![1, 128, 500])),
        );
    }

    #[test]
    fn shape_constraint_types() {
        let fixed = ShapeConstraint::Fixed(vec![1, 128]);
        let enum_c = ShapeConstraint::Enumerated(vec![vec![1, 128], vec![1, 256]]);
        let range_c = ShapeConstraint::Range(vec![(1, 10), (64, 512)]);
        let unspec = ShapeConstraint::Unspecified;

        assert_ne!(fixed, enum_c);
        assert_ne!(enum_c, range_c);
        assert_ne!(range_c, unspec);
        assert_eq!(fixed, ShapeConstraint::Fixed(vec![1, 128]));
    }
}
