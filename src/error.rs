/// Error types for the coreml crate.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    ModelLoad,
    TensorCreate,
    Prediction,
    Introspection,
    InvalidShape,
    UnsupportedPlatform,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelLoad => write!(f, "model load"),
            Self::TensorCreate => write!(f, "tensor create"),
            Self::Prediction => write!(f, "prediction"),
            Self::Introspection => write!(f, "introspection"),
            Self::InvalidShape => write!(f, "invalid shape"),
            Self::UnsupportedPlatform => write!(f, "unsupported platform"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }

    pub fn kind(&self) -> &ErrorKind { &self.kind }
    pub fn message(&self) -> &str { &self.message }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "coreml {}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(target_vendor = "apple")]
impl Error {
    pub(crate) fn from_nserror(kind: ErrorKind, err: &objc2_foundation::NSError) -> Self {
        let desc = err.localizedDescription();
        Self::new(kind, desc.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = Error::new(ErrorKind::ModelLoad, "file not found");
        let s = format!("{err}");
        assert!(s.contains("model load"));
        assert!(s.contains("file not found"));
    }

    #[test]
    fn error_implements_std_error() {
        let err = Error::new(ErrorKind::Prediction, "fail");
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn error_kind_accessor() {
        let err = Error::new(ErrorKind::InvalidShape, "mismatch");
        assert_eq!(err.kind(), &ErrorKind::InvalidShape);
    }

    #[test]
    fn all_error_kinds_distinct() {
        let kinds = vec![
            ErrorKind::ModelLoad, ErrorKind::TensorCreate, ErrorKind::Prediction,
            ErrorKind::Introspection, ErrorKind::InvalidShape, ErrorKind::UnsupportedPlatform,
        ];
        for (i, a) in kinds.iter().enumerate() {
            for (j, b) in kinds.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }
}
