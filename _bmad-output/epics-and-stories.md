---
stepsCompleted: [1,2,3,4]
inputDocuments: ['_bmad-output/prd-coreml-crate.md', 'docs/research-native-rust-coreml.md']
---

# coreml Crate - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for the `coreml` Rust crate, decomposing the PRD requirements into implementable stories across 7 epics spanning 3 release phases.

## Requirements Inventory

### Functional Requirements

- FR-1: Model Loading (FR-1.1 through FR-1.4)
- FR-2: Tensor Creation (FR-2.1 through FR-2.7)
- FR-3: Prediction (FR-3.1 through FR-3.5)
- FR-4: Model Introspection (FR-4.1 through FR-4.4)
- FR-5: Stateful Prediction (FR-5.1 through FR-5.3)
- FR-6: Async Prediction (FR-6.1 through FR-6.3)
- FR-7: Model Compilation (FR-7.1 through FR-7.3)
- FR-8: Output Tensor Access (FR-8.1 through FR-8.3)
- FR-9: Error Handling (FR-9.1 through FR-9.3)
- FR-10: Cross-Platform Compilation (FR-10.1 through FR-10.3)

### Non-Functional Requirements

- NFR-1: Performance (NFR-1.1 through NFR-1.4)
- NFR-2: Memory Safety (NFR-2.1 through NFR-2.4)
- NFR-3: API Ergonomics (NFR-3.1 through NFR-3.4)
- NFR-4: Compatibility (NFR-4.1 through NFR-4.4)
- NFR-5: Packaging (NFR-5.1 through NFR-5.4)
- NFR-6: Testing (NFR-6.1 through NFR-6.4)

### FR Coverage Map

| FR | Epic | Stories |
|----|------|---------|
| FR-1 (Model Loading) | Epic 3 | 3.1, 3.2 |
| FR-2 (Tensor Creation) | Epic 2 | 2.1, 2.2, 2.3 |
| FR-3 (Prediction) | Epic 4 | 4.1, 4.2, 4.3 |
| FR-4 (Model Introspection) | Epic 3 | 3.3 |
| FR-5 (Stateful Prediction) | Epic 6 | 6.1 |
| FR-6 (Async Prediction) | Epic 6 | 6.2 |
| FR-7 (Model Compilation) | Epic 6 | 6.3 |
| FR-8 (Output Tensor Access) | Epic 4 | 4.1, 4.2 |
| FR-9 (Error Handling) | Epic 1 | 1.1 |
| FR-10 (Cross-Platform) | Epic 1 | 1.3 |
| NFR-6 (Testing) | Epic 5 | 5.1, 5.2, 5.3 |
| NFR-5 (Packaging) | Epic 7 | 7.1, 7.2, 7.3 |

## Epic List

| Epic | Title | Phase | Stories | Priority |
|------|-------|-------|---------|----------|
| 1 | Foundation & Project Setup | Phase 1 (v0.1.0) | 3 | P0 |
| 2 | Tensor System | Phase 1 (v0.1.0) | 3 | P0 |
| 3 | Model Loading & Introspection | Phase 1 (v0.1.0) | 3 | P0 |
| 4 | Prediction Engine | Phase 1 (v0.1.0) | 3 | P0 |
| 5 | Testing, Benchmarks & Examples | Phase 1 (v0.1.0) | 3 | P0 |
| 6 | Advanced Features | Phase 2 (v0.2.0) | 3 | P1 |
| 7 | Ecosystem Release | Phase 3 (v0.3.0) | 3 | P1 |

---

## Epic 1: Foundation & Project Setup

**Goal:** Establish crate structure, dependencies, error types, FFI helpers, and cross-platform compilation gates. Every subsequent epic depends on this foundation.

**Dependencies:** None
**FRs:** FR-9.1, FR-9.2, FR-9.3, FR-10.1, FR-10.2, FR-10.3

### Story 1.1: Error Types and Result Handling

As a crate consumer,
I want all fallible operations to return `Result<T, coreml::Error>` with descriptive error messages,
So that I can handle CoreML failures idiomatically in Rust.

**Acceptance Criteria:**

**Given** a CoreML operation fails (e.g., invalid model path)
**When** the operation returns an Err variant
**Then** the Error contains: an ErrorKind enum variant, a human-readable message, and the source NSError localized description
**And** the Error type implements `std::error::Error`, `Display`, and `Debug`

**Given** an Error is created from an NSError
**When** displayed via `Display` trait
**Then** the output includes both the error kind and the CoreML error message

**Technical Notes:**
- Error kinds: ModelLoad, TensorCreate, Prediction, Introspection, InvalidShape, UnsupportedPlatform
- Wrap `Retained<NSError>` -> extract `localizedDescription()` -> store as `String`
- File: `src/error.rs`

**FRs covered:** FR-9.1, FR-9.2, FR-9.3

---

### Story 1.2: FFI Helper Layer

As a crate developer,
I want internal helper functions for converting between Rust and Foundation types,
So that every module can safely bridge Rust <-> Obj-C data without duplicating conversion logic.

**Acceptance Criteria:**

**Given** a Rust `&[usize]` shape array
**When** converted to `NSArray<NSNumber>`
**Then** each element is correctly represented as an NSNumber with i64 value

**Given** an `NSArray<NSNumber>` from CoreML
**When** converted to `Vec<usize>`
**Then** all elements are correctly extracted

**Given** a Rust `&str`
**When** converted to `NSString`
**Then** the string content is preserved (including Unicode)

**Given** a set of `(&str, &MLFeatureValue)` pairs
**When** converted to an `NSDictionary<NSString, AnyObject>`
**Then** the dictionary has the correct count and all keys map to their values

**Technical Notes:**
- Functions: `shape_to_nsarray`, `nsarray_to_shape`, `str_to_nsstring`, `nsstring_to_str`, `compute_strides`
- All conversions must be safe wrappers around unsafe objc2 calls
- File: `src/ffi.rs`

**FRs covered:** Internal (supports all FRs)

---

### Story 1.3: Crate Structure and Cross-Platform Gates

As a Rust developer on Linux or Windows,
I want the crate to compile on non-Apple targets (with no functionality),
So that my cross-platform project can depend on it behind a feature flag without breaking CI.

**Acceptance Criteria:**

**Given** `cargo build` runs on a Linux target
**When** the `coreml` crate is a dependency
**Then** compilation succeeds with no errors
**And** all public types exist as empty stubs or type aliases

**Given** a developer tries to call `Model::load()` on a non-Apple target
**When** the function is invoked
**Then** it returns `Err(Error::UnsupportedPlatform)` with a helpful message

**Given** the `Cargo.toml` is properly structured
**When** inspected
**Then** it contains: package metadata (name, version, license, repository, keywords, categories), objc2 dependencies gated by `[target.'cfg(target_vendor = "apple")'.dependencies]`, edition = "2021", rust-version = "1.75.0"

**Technical Notes:**
- Use `#[cfg(target_vendor = "apple")]` for all platform-specific code
- Provide stub types on non-Apple targets so downstream code compiles
- No `build.rs` required
- Files: `src/lib.rs`, `Cargo.toml`

**FRs covered:** FR-10.1, FR-10.2, FR-10.3

---

## Epic 2: Tensor System

**Goal:** Implement zero-copy tensor creation from Rust slices, shape validation, data type abstraction, and safe tensor access. This is the core data bridge between Rust and CoreML.

**Dependencies:** Epic 1
**FRs:** FR-2.1, FR-2.4, FR-2.5, FR-2.6, FR-2.7

### Story 2.1: Tensor from Borrowed f32 Slice (Zero-Copy)

As a crate consumer,
I want to create a CoreML tensor from a `&[f32]` slice without copying data,
So that I can feed Rust-owned audio/image/text features directly into CoreML inference.

**Acceptance Criteria:**

**Given** a `&[f32]` slice of length 64000 and shape `[1, 128, 500]`
**When** `Tensor::from_slice(data, &[1, 128, 500])` is called
**Then** an MLMultiArray is created wrapping the Rust buffer pointer with Float32 data type
**And** no data is copied (the underlying pointer matches the Rust slice)
**And** the deallocator is None (Rust retains ownership)

**Given** a `&[f32]` slice of length 100 and shape `[1, 128, 500]`
**When** `Tensor::from_slice(data, &[1, 128, 500])` is called
**Then** an `Err(Error::InvalidShape)` is returned because 100 != 1*128*500

**Given** a Tensor created from a borrowed slice
**When** `tensor.shape()` is called
**Then** it returns `&[1, 128, 500]`

**Given** a Tensor created from a borrowed slice
**When** `tensor.data_type()` is called
**Then** it returns `DataType::Float32`

**Technical Notes:**
- Use `MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error`
- Strides computed as row-major contiguous: `[dim1*dim2, dim2, 1]`
- Tensor struct must hold a reference to the underlying `Retained<MLMultiArray>`
- CRITICAL: The Tensor must NOT outlive the source slice. Use a lifetime parameter: `Tensor<'a>`
- File: `src/tensor.rs`

**FRs covered:** FR-2.1, FR-2.5, FR-2.6, FR-2.7

---

### Story 2.2: Owned Tensor (Crate-Allocated)

As a crate consumer,
I want to create a tensor that the crate allocates and owns,
So that I can use it for pre-allocated output buffers and intermediate computations.

**Acceptance Criteria:**

**Given** shape `[1, 640]` and data type Float32
**When** `Tensor::zeros(DataType::Float32, &[1, 640])` is called
**Then** an MLMultiArray is created with the given shape, zero-initialized
**And** the tensor owns its buffer (no external lifetime dependency)

**Given** an owned tensor
**When** `tensor.as_mut_slice::<f32>()` is called
**Then** a mutable slice to the underlying buffer is returned for writing

**Given** an owned tensor
**When** `tensor.as_slice::<f32>()` is called
**Then** a read-only slice to the underlying buffer is returned

**Technical Notes:**
- Use `MLMultiArray::initWithShape_dataType_error` for crate-allocated tensors
- OwnedTensor is a separate type or `Tensor<'static>` with owned buffer
- Access via `getBytesWithHandler` (read) and `getMutableBytesWithHandler` (write)
- File: `src/tensor.rs`

**FRs covered:** FR-2.4

---

### Story 2.3: DataType Enum and Tensor Utilities

As a crate consumer,
I want a DataType enum and utility methods on tensors,
So that I can work with different numeric types and inspect tensor properties.

**Acceptance Criteria:**

**Given** the DataType enum
**When** inspected
**Then** it has variants: Float16, Float32, Float64, Int32
**And** it implements Debug, Clone, Copy, PartialEq, Eq

**Given** a DataType variant
**When** converted to/from `MLMultiArrayDataType`
**Then** the conversion is bidirectional and lossless

**Given** a tensor of any type
**When** `tensor.element_count()` is called
**Then** it returns the product of all shape dimensions

**Given** a tensor of any type
**When** `tensor.size_bytes()` is called
**Then** it returns element_count * data_type.byte_size()

**Technical Notes:**
- DataType::byte_size() returns 2, 4, 8, or 4 respectively
- File: `src/tensor.rs`

**FRs covered:** FR-2.5, FR-2.6

---

## Epic 3: Model Loading & Introspection

**Goal:** Load compiled CoreML models, configure compute units, and inspect model input/output metadata. This is the entry point for all inference workflows.

**Dependencies:** Epic 1, Epic 2
**FRs:** FR-1.1, FR-1.2, FR-1.3, FR-1.4, FR-4.1, FR-4.2, FR-4.3

### Story 3.1: Model Loading with Compute Unit Configuration

As a crate consumer,
I want to load a `.mlmodelc` bundle from a path and configure which hardware backends to use,
So that I can control whether inference runs on CPU, GPU, ANE, or all.

**Acceptance Criteria:**

**Given** a valid `.mlmodelc` path and `ComputeUnits::All`
**When** `Model::load(path, ComputeUnits::All)` is called
**Then** an MLModel is loaded with MLComputeUnits set to All
**And** a `Model` struct is returned

**Given** an invalid path
**When** `Model::load(invalid_path, ComputeUnits::All)` is called
**Then** `Err(Error::ModelLoad("..."))` is returned with the CoreML error message

**Given** `ComputeUnits::CpuOnly`
**When** a model is loaded
**Then** MLModelConfiguration.computeUnits is set to MLComputeUnitsCPUOnly

**Given** a loaded model
**When** the Model struct is dropped
**Then** the underlying MLModel is released (ARC via Retained<T> drop)
**And** no memory is leaked

**Technical Notes:**
- ComputeUnits enum: CpuOnly, CpuAndGpu, CpuAndNeuralEngine, All
- Model struct holds `Retained<MLModel>` and the path
- Model is Send (can move between threads) but NOT Sync
- File: `src/model.rs`, `src/config.rs`

**FRs covered:** FR-1.1, FR-1.2, FR-1.3, FR-1.4

---

### Story 3.2: ComputeUnits Enum and Configuration Builder

As a crate consumer,
I want a clean ComputeUnits enum and optional configuration builder,
So that I can configure model loading with precision control.

**Acceptance Criteria:**

**Given** the ComputeUnits enum
**When** inspected
**Then** it has variants: CpuOnly, CpuAndGpu, CpuAndNeuralEngine, All
**And** it implements Debug, Clone, Copy, PartialEq, Default (default = All)

**Given** a ComputeUnits variant
**When** converted to MLComputeUnits
**Then** the mapping is correct (CpuOnly=0, CpuAndGpu=1, CpuAndNeuralEngine=2, All=3)

**Given** an optional ModelConfig builder
**When** `ModelConfig::new().compute_units(CpuOnly).low_precision_gpu(true).build()`
**Then** an MLModelConfiguration is created with the specified settings

**Technical Notes:**
- File: `src/config.rs`

**FRs covered:** FR-1.2

---

### Story 3.3: Model Introspection (Inputs, Outputs, Metadata)

As a crate consumer,
I want to inspect a loaded model's input/output requirements and metadata,
So that I can construct correct tensors and understand the model contract at runtime.

**Acceptance Criteria:**

**Given** a loaded model
**When** `model.inputs()` is called
**Then** it returns a `Vec<FeatureDescription>` with name, data_type, and shape for each input

**Given** a loaded model
**When** `model.outputs()` is called
**Then** it returns a `Vec<FeatureDescription>` with name, data_type, and shape for each output

**Given** a FeatureDescription for a multi-array input
**When** `desc.shape()` is called
**Then** it returns `Option<Vec<usize>>` (None if shape is flexible/unknown)

**Given** a FeatureDescription for a multi-array input
**When** `desc.data_type()` is called
**Then** it returns the DataType enum variant

**Given** a loaded model
**When** `model.metadata()` is called
**Then** it returns a struct with optional author, description, version, license fields

**Technical Notes:**
- Access via `model.modelDescription().inputDescriptionsByName()`
- FeatureDescription wraps MLFeatureDescription with safe accessors
- Shape extracted from MLMultiArrayConstraint
- Metadata extracted from MLModelDescription.metadata()
- Files: `src/description.rs`, `src/model.rs`

**FRs covered:** FR-4.1, FR-4.2, FR-4.3

---

## Epic 4: Prediction Engine

**Goal:** Implement synchronous prediction with named inputs/outputs, autorelease pool management, and zero-copy output access. This is the core inference path.

**Dependencies:** Epic 1, Epic 2, Epic 3
**FRs:** FR-3.1, FR-3.2, FR-3.3, FR-3.5, FR-8.1, FR-8.2, FR-8.3

### Story 4.1: Synchronous Prediction with Named Inputs

As a crate consumer,
I want to run a prediction by providing named input tensors and receiving named output tensors,
So that I can perform ML inference in my Rust application.

**Acceptance Criteria:**

**Given** a loaded model and a tensor matching the input shape
**When** `model.predict(&[("input_name", &tensor)])` is called
**Then** the prediction runs synchronously via CoreML
**And** a `Prediction` struct is returned containing the output tensors

**Given** a prediction with mismatched input name
**When** `model.predict(&[("wrong_name", &tensor)])` is called
**Then** `Err(Error::Prediction("..."))` is returned with the CoreML error

**Given** the predict call
**When** executed internally
**Then** it is wrapped in `objc2::rc::autoreleasepool()` to prevent Obj-C object leaks

**Technical Notes:**
- Build MLDictionaryFeatureProvider from input name-tensor pairs
- Each tensor's inner MLMultiArray wrapped in MLFeatureValue
- Call `model.predictionFromFeatures_error()`
- Every prediction wrapped in autoreleasepool
- File: `src/model.rs`

**FRs covered:** FR-3.1, FR-3.2, FR-3.5

---

### Story 4.2: Prediction Output Access

As a crate consumer,
I want to access prediction outputs by name and read the tensor data as Rust slices,
So that I can use inference results in my application logic.

**Acceptance Criteria:**

**Given** a Prediction result
**When** `prediction.get("output_name")` is called
**Then** it returns `Option<OutputTensor>` for the named output

**Given** an OutputTensor
**When** `output.to_vec::<f32>()` is called
**Then** it returns `Vec<f32>` with a copy of the output data

**Given** an OutputTensor
**When** `output.copy_to(buf: &mut [f32])` is called
**Then** the output data is copied into the caller's buffer
**And** returns an error if buffer length doesn't match output size

**Given** an OutputTensor
**When** `output.shape()` and `output.data_type()` are called
**Then** they return the correct shape and data type of the output

**Given** a Prediction result
**When** `prediction.output_names()` is called
**Then** it returns the set of output feature names

**Technical Notes:**
- Use `getBytesWithHandler` for scoped read access to output MLMultiArray
- OutputTensor holds `Retained<MLMultiArray>` from the prediction result
- Copy data within the handler scope (handler provides raw pointer + strides)
- File: `src/prediction.rs`

**FRs covered:** FR-3.3, FR-8.1, FR-8.2, FR-8.3

---

### Story 4.3: Multi-Input Prediction

As a crate consumer with a model requiring multiple inputs (e.g., encoder with "audio_signal" + "length"),
I want to provide multiple named tensors in a single predict call,
So that I can run models with complex input contracts.

**Acceptance Criteria:**

**Given** a model with 2 inputs: "audio_signal" (shape [1, 128, 500]) and "length" (shape [1])
**When** `model.predict(&[("audio_signal", &audio_tensor), ("length", &length_tensor)])` is called
**Then** both inputs are packed into the MLDictionaryFeatureProvider
**And** prediction succeeds

**Given** a model with 3 inputs
**When** only 2 are provided
**Then** CoreML returns an error which is propagated as `Err(Error::Prediction(...))`

**Technical Notes:**
- Same predict method handles single and multi-input (it's a slice of tuples)
- File: `src/model.rs`

**FRs covered:** FR-3.1, FR-3.2

---

## Epic 5: Testing, Benchmarks & Examples

**Goal:** Validate correctness, measure performance, and provide working examples. Creates confidence for Phase 2 and crates.io publication.

**Dependencies:** Epic 1-4
**FRs:** NFR-6.1, NFR-6.2, NFR-6.4, NFR-3.2

### Story 5.1: Unit Tests for Tensor and Error Types

As a crate developer,
I want comprehensive unit tests for tensor creation, shape validation, and error types,
So that the foundation is proven correct before integration testing.

**Acceptance Criteria:**

**Given** the tensor module
**When** tests run
**Then** the following are verified:
- Shape validation: correct shape passes, mismatched length fails
- DataType enum: all conversions to/from MLMultiArrayDataType
- Strides computation: correct for 1D, 2D, 3D, 4D shapes
- Error types: Display output, Error trait, ErrorKind variants

**Given** shape `[0, 128]` (zero dimension)
**When** tensor creation attempted
**Then** it returns an appropriate error

**Technical Notes:**
- Tests can run without a CoreML model (pure logic tests)
- Use `#[cfg(test)]` module in each source file
- File: tests within `src/tensor.rs`, `src/error.rs`

**FRs covered:** NFR-6.1

---

### Story 5.2: Integration Test with Test Model

As a crate developer,
I want an integration test that loads a real .mlmodelc, runs a prediction, and validates output,
So that the full pipeline is proven end-to-end on macOS.

**Acceptance Criteria:**

**Given** a small test model (.mlmodelc) included in the test fixtures
**When** `cargo test` runs on macOS
**Then** the test loads the model, creates an input tensor, runs prediction, and reads output
**And** the output values are within expected bounds

**Given** `cargo test` runs on Linux
**When** the integration test is encountered
**Then** it is skipped via `#[cfg(target_vendor = "apple")]`

**Technical Notes:**
- Create a minimal test model using coremltools (e.g., y = 2*x + 1, single input/output)
- Store compiled .mlmodelc in `tests/fixtures/`
- Alternatively: use Python script in `scripts/` to generate test model
- File: `tests/integration_test.rs`

**FRs covered:** NFR-6.2

---

### Story 5.3: Examples and Benchmark Suite

As a crate consumer,
I want working examples showing common usage patterns,
So that I can quickly understand how to use the crate.

**Acceptance Criteria:**

**Given** the `examples/` directory
**When** inspected
**Then** it contains:
- `load_and_predict.rs` -- loads a model, creates tensor, runs prediction, prints output
- `inspect_model.rs` -- loads a model, prints all input/output descriptions

**Given** the benchmark suite
**When** `cargo bench` runs
**Then** it measures: model load time, single prediction latency, 100-prediction throughput

**Technical Notes:**
- Examples use paths that can be overridden via env vars or CLI args
- Benchmark uses criterion crate
- File: `examples/*.rs`, `benches/predict_bench.rs`

**FRs covered:** NFR-6.4, NFR-3.2

---

## Epic 6: Advanced Features

**Goal:** Add stateful prediction (MLState), async prediction, Float16/Int32 tensor support, output backings, and model compilation. Enables swictation C bridge replacement.

**Dependencies:** Epic 1-5
**FRs:** FR-2.2, FR-2.3, FR-3.4, FR-5.1-5.3, FR-6.1-6.3, FR-7.1-7.3

### Story 6.1: Stateful Prediction (MLState)

As a crate consumer building an RNN or transformer model,
I want to create persistent state that carries across prediction calls,
So that decoder hidden state or KV-cache is managed efficiently by CoreML.

**Acceptance Criteria:**

**Given** a model with state descriptors (macOS 15+ / iOS 18+)
**When** `model.new_state()` is called
**Then** a State struct is returned wrapping MLState

**Given** a State struct and input tensor
**When** `model.predict_stateful(&[("input", &tensor)], &state)` is called
**Then** prediction runs with the state modified in-place
**And** subsequent calls with the same state reflect accumulated state

**Given** a model on macOS < 15
**When** `model.new_state()` is called
**Then** `Err(Error::UnsupportedPlatform("MLState requires macOS 15+"))` is returned

**Technical Notes:**
- Use `MLModel::newState()` and `predictionFromFeatures_usingState_error`
- State must be Send (matches MLState threading model)
- Gate behind `#[cfg(target_os = "macos")]` + runtime version check
- File: `src/state.rs`, `src/model.rs`

**FRs covered:** FR-5.1, FR-5.2, FR-5.3

---

### Story 6.2: Float16 and Int32 Tensor Support + Output Backings

As a crate consumer,
I want to create Float16 and Int32 tensors and pre-allocate output buffers,
So that I can match ANE-native precision and eliminate per-prediction allocation.

**Acceptance Criteria:**

**Given** a `&[half::f16]` slice and shape
**When** `Tensor::from_slice_f16(data, &shape)` is called
**Then** an MLMultiArray with Float16 data type is created, zero-copy

**Given** a `&[i32]` slice and shape
**When** `Tensor::from_slice_i32(data, &shape)` is called
**Then** an MLMultiArray with Int32 data type is created, zero-copy

**Given** pre-allocated output tensors
**When** `model.predict_with_backings(&inputs, &output_backings)` is called
**Then** CoreML writes output directly into the pre-allocated buffers via MLPredictionOptions.outputBackings
**And** no new output buffer is allocated

**Technical Notes:**
- Float16 requires the `half` crate as optional dependency
- Output backings: create NSDictionary of output name -> MLMultiArray, set on MLPredictionOptions
- File: `src/tensor.rs`, `src/model.rs`

**FRs covered:** FR-2.2, FR-2.3, FR-3.4

---

### Story 6.3: Async Prediction and Model Compilation

As a crate consumer,
I want async prediction and runtime model compilation,
So that I can run non-blocking inference and compile models at runtime.

**Acceptance Criteria:**

**Given** a loaded model and input tensor
**When** `model.predict_async(&inputs, callback)` is called
**Then** prediction runs on a CoreML-managed thread
**And** the callback receives `Result<Prediction, Error>` on completion

**Given** a `.mlmodel` or `.mlpackage` path
**When** `Model::compile(source_path)` is called
**Then** it returns the path to the compiled `.mlmodelc` directory

**Given** a large model compilation
**When** `Model::compile_async(source_path, callback)` is called
**Then** compilation runs asynchronously and the callback receives the result

**Technical Notes:**
- Async predict: use `predictionFromFeatures_completionHandler` with `block2::RcBlock`
- Model compile: use `MLModel::compileModelAtURL_error`
- Async compile: use `MLModel::compileModelAtURL_completionHandler`
- File: `src/model.rs`, `src/compile.rs`

**FRs covered:** FR-6.1, FR-6.2, FR-6.3, FR-7.1, FR-7.2, FR-7.3

---

## Epic 7: Ecosystem Release

**Goal:** Publish to crates.io, comprehensive documentation, CI pipeline, and community examples. Makes the crate available to the Rust+Apple ecosystem.

**Dependencies:** Epic 1-6
**FRs:** NFR-5.1-5.4, NFR-3.2, NFR-6.3

### Story 7.1: crates.io Publication Preparation

As a crate author,
I want the crate ready for crates.io publication,
So that any Rust developer can `cargo add coreml` and start using it.

**Acceptance Criteria:**

**Given** the Cargo.toml
**When** `cargo publish --dry-run` is executed
**Then** it passes with no errors

**Given** the package metadata
**When** inspected
**Then** it includes: description, license (Apache-2.0 OR MIT), repository URL, keywords (coreml, apple, neural-engine, inference, machine-learning), categories (os::macos-apis, science)

**Given** the crate
**When** dependency tree is inspected
**Then** it has minimal dependencies: only objc2 ecosystem crates + optional `half`

**Technical Notes:**
- Check crate name availability on crates.io
- Ensure `include` in Cargo.toml excludes test fixtures from published crate
- File: `Cargo.toml`

**FRs covered:** NFR-5.1, NFR-5.2, NFR-5.3, NFR-5.4

---

### Story 7.2: Comprehensive Documentation

As a crate consumer,
I want complete rustdoc documentation with examples on every public type and method,
So that I can learn the API without reading source code.

**Acceptance Criteria:**

**Given** any public type or method
**When** `cargo doc` is built
**Then** it has a doc comment with description and at least one usage example

**Given** the crate root documentation
**When** viewed on docs.rs
**Then** it includes: overview, quickstart, feature flags, platform requirements, comparison table vs alternatives

**Given** `cargo doc --no-deps`
**When** warnings are checked
**Then** there are zero missing-docs warnings

**Technical Notes:**
- Add `#![warn(missing_docs)]` to lib.rs
- File: all `src/*.rs` files

**FRs covered:** NFR-3.2

---

### Story 7.3: GitHub Actions CI and README

As a crate maintainer,
I want CI running tests on every push and a polished README,
So that contributors and users have confidence in the project quality.

**Acceptance Criteria:**

**Given** a push to the main branch
**When** GitHub Actions runs
**Then** it executes: `cargo build`, `cargo test`, `cargo clippy`, `cargo doc` on macOS runner
**And** all checks pass

**Given** a push from a non-macOS environment
**When** CI also runs on Ubuntu
**Then** `cargo build` succeeds (cross-platform compilation stub)
**And** macOS-only tests are skipped

**Given** the README.md
**When** viewed on GitHub/crates.io
**Then** it includes: badges (CI, crates.io version, docs.rs), feature overview, quickstart code, comparison table, minimum requirements, license

**Technical Notes:**
- CI config: `.github/workflows/ci.yml`
- macOS runner: `macos-14` (Apple Silicon)
- File: `.github/workflows/ci.yml`, `README.md`

**FRs covered:** NFR-6.3

---

## Sprint Sequencing (Recommended)

### Sprint 1 (Phase 1a): Foundation + Tensor
- Story 1.1: Error Types
- Story 1.2: FFI Helper Layer
- Story 1.3: Crate Structure
- Story 2.1: Tensor from f32 Slice
- Story 2.2: Owned Tensor
- Story 2.3: DataType Enum

### Sprint 2 (Phase 1b): Model + Prediction
- Story 3.1: Model Loading
- Story 3.2: ComputeUnits + Config
- Story 3.3: Model Introspection
- Story 4.1: Synchronous Prediction
- Story 4.2: Prediction Output Access
- Story 4.3: Multi-Input Prediction

### Sprint 3 (Phase 1c): Testing + Validation
- Story 5.1: Unit Tests
- Story 5.2: Integration Test
- Story 5.3: Examples + Benchmarks

### Sprint 4 (Phase 2): Advanced Features
- Story 6.1: Stateful Prediction
- Story 6.2: Float16/Int32 + Output Backings
- Story 6.3: Async + Compilation

### Sprint 5 (Phase 3): Release
- Story 7.1: crates.io Preparation
- Story 7.2: Documentation
- Story 7.3: CI + README
