//! Compute device enumeration and inspection.
//!
//! Discover available compute devices (CPU, GPU, Neural Engine)
//! and their capabilities.

use crate::error::{Error, ErrorKind, Result};

// Suppress unused warnings until callers use the error types.
const _: () = {
    fn _assert_imports() {
        let _ = Error::new(ErrorKind::UnsupportedPlatform, "");
        let _: Result<()> = Ok(());
    }
};

/// A compute device available for CoreML inference.
#[derive(Debug, Clone, PartialEq)]
pub enum ComputeDevice {
    /// CPU compute device.
    Cpu,
    /// GPU (Metal) compute device.
    Gpu {
        /// Metal device name, if available.
        name: Option<String>,
    },
    /// Apple Neural Engine.
    NeuralEngine,
}

impl std::fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu { name: Some(n) } => write!(f, "GPU ({n})"),
            Self::Gpu { name: None } => write!(f, "GPU"),
            Self::NeuralEngine => write!(f, "Neural Engine"),
        }
    }
}

/// Returns a list of all compute devices available for CoreML on this system.
#[cfg(target_vendor = "apple")]
pub fn available_devices() -> Vec<ComputeDevice> {
    // Try to use MLAllComputeDevices if available, otherwise return a static list.
    // MLAllComputeDevices is available on macOS 14+ / iOS 17+.
    // For older systems, return a reasonable default.

    // Since MLAllComputeDevices may require newer SDK, provide a safe default:
    let mut devices = vec![ComputeDevice::Cpu];

    // GPU is always available on macOS
    devices.push(ComputeDevice::Gpu { name: None });

    // Neural Engine is available on Apple Silicon
    #[cfg(target_arch = "aarch64")]
    devices.push(ComputeDevice::NeuralEngine);

    devices
}

#[cfg(not(target_vendor = "apple"))]
pub fn available_devices() -> Vec<ComputeDevice> {
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_device_display() {
        assert_eq!(format!("{}", ComputeDevice::Cpu), "CPU");
        assert_eq!(
            format!(
                "{}",
                ComputeDevice::Gpu {
                    name: Some("M1 Pro".into())
                }
            ),
            "GPU (M1 Pro)"
        );
        assert_eq!(
            format!("{}", ComputeDevice::Gpu { name: None }),
            "GPU"
        );
        assert_eq!(format!("{}", ComputeDevice::NeuralEngine), "Neural Engine");
    }

    #[test]
    fn compute_device_equality() {
        assert_eq!(ComputeDevice::Cpu, ComputeDevice::Cpu);
        assert_ne!(ComputeDevice::Cpu, ComputeDevice::NeuralEngine);
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn available_devices_non_empty() {
        let devices = available_devices();
        assert!(!devices.is_empty());
        assert!(devices.contains(&ComputeDevice::Cpu));
    }

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn available_devices_empty_on_non_apple() {
        assert!(available_devices().is_empty());
    }
}
