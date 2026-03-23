//! Stateful prediction support (MLState).
//!
//! Requires macOS 15+ / iOS 18+.
//! Covers FR-5.1, FR-5.2, FR-5.3.

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;

/// A persistent state for stateful CoreML models (RNN, KV-cache, etc.).
///
/// State is mutated in-place during prediction and carries forward
/// across multiple predict calls.
#[cfg(target_vendor = "apple")]
pub struct State {
    pub(crate) inner: Retained<objc2_core_ml::MLState>,
}

#[cfg(target_vendor = "apple")]
impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State").finish()
    }
}

#[cfg(target_vendor = "apple")]
unsafe impl Send for State {}

#[cfg(not(target_vendor = "apple"))]
#[derive(Debug)]
pub struct State {
    _private: (),
}
