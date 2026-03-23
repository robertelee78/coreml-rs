# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-23

### Added

- Safe Rust bindings for Apple CoreML via Objective-C FFI
- Model loading from `.mlmodelc` compiled bundles and `.mlpackage` archives
- On-device model compilation with `MLModelConfiguration` support
- Zero-copy tensor interop with `ndarray` and raw buffer views
- Synchronous and asynchronous prediction APIs
- Model introspection: input/output feature descriptions, metadata access
- Stateful prediction support via `MLState` bindings
- Configurable compute units (CPU, GPU, Neural Engine, or all)
- Mixed-dtype input support and multi-output model handling
- `OwnedTensor` type for owned tensor data passed as model input
- Cross-platform compilation stubs for non-macOS targets (Linux, Windows)
- Integration test suite covering 15+ inference scenarios
- CI pipeline with macOS and cross-platform build verification

[Unreleased]: https://github.com/robertelee78/coreml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/robertelee78/coreml/releases/tag/v0.1.0
