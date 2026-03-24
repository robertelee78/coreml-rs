# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-23

### Added

- Async APIs: `load_async`, `predict_async`, `compile_model_async` with runtime-agnostic `CompletionFuture`
- `load_from_bytes` for loading models from in-memory data
- Batch prediction via `BatchProvider` and `MLArrayBatchProvider`
- `ModelHandle` with type-safe `unload`/`reload` for GPU/ANE memory management
- 9 tensor data types: Float16, Float32, Float64, Int32, Int16, Int8, UInt32, UInt16, UInt8
- Device enumeration for discovering available CPU, GPU, and Neural Engine devices
- Optional `ndarray` feature flag for zero-copy `ndarray` ↔ tensor conversion
- Shape constraint introspection on model inputs/outputs

### Fixed

- Stride-aware output copy for non-contiguous GPU/ANE tensor layouts

### Changed

- Renamed crate from `coreml-rs` to `coreml-native`

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

[Unreleased]: https://github.com/robertelee78/coreml-native/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/robertelee78/coreml-native/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/robertelee78/coreml-native/releases/tag/v0.1.0
