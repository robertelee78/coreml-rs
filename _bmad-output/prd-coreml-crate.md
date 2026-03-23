---
stepsCompleted: [1,2,3,4,5,6,7,8,9,10,11,12]
inputDocuments: ['docs/research-native-rust-coreml.md']
workflowType: 'prd'
---

# Product Requirements Document - coreml (Rust Crate)

**Author:** Robert
**Date:** 2026-03-23
**Version:** 1.0

---

## Executive Summary

### Vision

An ergonomic, safe Rust crate providing native Apple CoreML inference on macOS and iOS. The `coreml` crate wraps Apple's CoreML framework via `objc2-core-ml` auto-generated bindings, exposing a safe, idiomatic Rust API for loading compiled ML models, creating tensors, running predictions, and inspecting model metadata -- with full Apple Neural Engine (ANE) acceleration.

### Differentiator

No standalone, safe, ergonomic Rust CoreML inference crate exists today. The ecosystem gap:
- `objc2-core-ml` (v0.3.2): 100% API coverage but raw `unsafe`, no ergonomic layer
- `coreml-rs` (v0.4.0): Requires Swift runtime, barely maintained (14 stars)
- `candle-coreml` (v0.3.1): Coupled to HuggingFace Candle tensor ecosystem
- ONNX Runtime CoreML EP: Only dispatches subset of operators to ANE

The `coreml` crate provides safe wrappers over `objc2-core-ml` with zero-copy tensor I/O, automatic memory management, and no external language dependencies (no C, Obj-C, or Swift code).

### Target Users

1. **Rust developers deploying ML models on Apple platforms** -- image classification, NLP, audio, computer vision
2. **Rust audio/speech projects** -- STT/TTS inference needing ANE acceleration (e.g., swictation)
3. **Rust+Apple ecosystem contributors** -- building applications leveraging Apple Silicon ML hardware
4. **ML framework authors** -- needing a CoreML backend for their Rust ML framework

### Key Constraints

- macOS/iOS only (Apple CoreML framework dependency)
- Requires Apple Silicon for ANE acceleration (Intel Macs: CPU/GPU only)
- Pre-compiled `.mlmodelc` bundles as primary input format
- objc2 ecosystem is pre-1.0 (v0.6.4) -- pin dependency versions
- Synchronous prediction API is NOT thread-safe (Send but not Sync)

---

## Success Criteria

| ID | Criterion | Measurement | Target |
|----|-----------|-------------|--------|
| SC-1 | Prediction accuracy matches baseline | Output comparison vs ONNX Runtime on same model | < 0.01% relative error (Float32) |
| SC-2 | Zero-copy tensor I/O | No memcpy between Rust buffer and CoreML | Verified via profiling (Instruments) |
| SC-3 | ANE utilization on supported ops | MLComputePlan inspection shows ANE dispatch | Encoder convolutions on ANE |
| SC-4 | No memory leaks | Instruments Leaks tool during sustained inference | 0 leaks over 1000 predictions |
| SC-5 | API ergonomics | Load-predict workflow in < 10 lines of user code | Verified via examples |
| SC-6 | Build with no external toolchain | `cargo build` succeeds with no Xcode/Swift/cc dependency beyond SDK | Pure Rust compilation |
| SC-7 | Publishable to crates.io | Passes `cargo publish --dry-run` | Clean publish |
| SC-8 | Drop-in replacement for swictation C bridge | Same output, same performance class | Verified via integration test |

---

## Product Scope

### Phase 1: MVP -- Core Inference (v0.1.0)

Load a compiled CoreML model, create input tensors from Rust slices, run synchronous prediction, read output tensors, inspect model metadata. Sufficient to replace swictation's C bridge.

### Phase 2: Advanced Features (v0.2.0)

Stateful prediction (MLState for RNN/KV-cache), async prediction via completion handlers, Float16 tensor support, pre-allocated output backings, model compilation (.mlmodel -> .mlmodelc).

### Phase 3: Ecosystem Integration (v0.3.0)

Publish to crates.io, comprehensive documentation, integration examples for common ML tasks (image classification, speech, NLP), optional ndarray/nalgebra interop features.

### Out of Scope

- Model training or fine-tuning (CoreML on-device update API)
- Model conversion (coremltools is Python -- separate tooling)
- Cross-platform inference (this crate is Apple-only by design)
- Custom MLLayer or MLModel protocol implementation in Rust (future consideration)
- Direct ANE access via private APIs (Rustane's approach)

---

## User Journeys

### UJ-1: Load and Run a Pre-Compiled Model

**Actor:** Rust developer with a `.mlmodelc` bundle
**Goal:** Run inference from Rust code

```
1. Developer adds `coreml` to Cargo.toml
2. Developer calls Model::load("path/to/model.mlmodelc", ComputeUnits::All)
3. Developer inspects model inputs: model.inputs() returns name, shape, data type
4. Developer creates Tensor from &[f32] slice with shape
5. Developer calls model.predict(&[("input_name", &tensor)])
6. Developer reads output tensor as &[f32] slice
7. Developer uses results in application logic
```

**Success:** Prediction completes, output matches expected values, no unsafe code in user code.

### UJ-2: Inspect Model Metadata Before Prediction

**Actor:** Developer integrating an unfamiliar model
**Goal:** Understand model I/O contract at runtime

```
1. Developer loads model
2. Developer iterates model.inputs() -- gets name, data type, shape constraints
3. Developer iterates model.outputs() -- gets name, data type, expected shape
4. Developer constructs inputs matching the discovered contract
5. Developer runs prediction with confidence in tensor shapes
```

**Success:** All input/output metadata accessible without external documentation.

### UJ-3: Multi-Model Pipeline (Encoder + Decoder + Joiner)

**Actor:** Speech recognition developer (e.g., RNN-T architecture)
**Goal:** Run a multi-model inference pipeline

```
1. Developer loads 3 models (encoder, decoder, joiner)
2. Developer processes audio -> mel features -> encoder input tensor
3. Developer runs encoder.predict() -> encoder output
4. In a loop:
   a. Developer runs decoder.predict() with token input
   b. Developer runs joiner.predict() with encoder + decoder outputs
   c. Developer selects token, updates state
5. Developer returns decoded text
```

**Success:** Multi-model pipeline produces correct transcription, each model uses appropriate compute backend.

### UJ-4: Stateful Prediction (RNN Decoder State Persistence)

**Actor:** Developer with a stateful model (LSTM, KV-cache)
**Goal:** Maintain state across prediction calls without copying

```
1. Developer loads model requiring state (macOS 15+ / iOS 18+)
2. Developer creates state: let state = model.new_state()
3. Developer runs prediction with state: model.predict_stateful(&input, &state)
4. State mutated in-place by CoreML runtime
5. Developer runs subsequent predictions -- state carries forward
6. No manual state tensor copy between calls
```

**Success:** State persists correctly, matching non-stateful baseline accuracy.

### UJ-5: Integration into Existing Rust Project

**Actor:** Maintainer of a Rust ML project wanting CoreML backend
**Goal:** Add CoreML as an optional backend behind a feature flag

```
1. Developer adds `coreml = { version = "0.1", optional = true }` to Cargo.toml
2. Developer adds `#[cfg(feature = "coreml")]` conditional code
3. On macOS: cargo build --features coreml compiles and links CoreML.framework
4. On Linux: cargo build succeeds (coreml feature not enabled), no compilation errors
5. Developer's CI pipeline works cross-platform without modification
```

**Success:** Feature-gated compilation, zero impact on non-Apple platforms.

---

## Functional Requirements

### FR-1: Model Loading

- FR-1.1: Load a compiled `.mlmodelc` directory from a filesystem path
- FR-1.2: Configure compute units at load time: CpuOnly, CpuAndGpu, CpuAndNeuralEngine, All (default)
- FR-1.3: Return a typed error on load failure containing the CoreML error description
- FR-1.4: Support loading multiple model instances concurrently (each on its own thread)

### FR-2: Tensor Creation

- FR-2.1: Create a tensor from a `&[f32]` slice and a shape `&[usize]`, zero-copy (Rust owns the buffer)
- FR-2.2: Create a tensor from a `&[f16]` (half) slice and a shape, zero-copy
- FR-2.3: Create a tensor from a `&[i32]` slice and a shape, zero-copy
- FR-2.4: Create an owned tensor (crate allocates and owns the buffer) from shape and data type
- FR-2.5: Access tensor shape as `&[usize]`
- FR-2.6: Access tensor data type as an enum (Float16, Float32, Float64, Int32)
- FR-2.7: Validate that the borrowed slice length matches the product of shape dimensions

### FR-3: Prediction

- FR-3.1: Run synchronous prediction with named input tensors, returning named output tensors
- FR-3.2: Accept input as slice of `(&str, &Tensor)` tuples (name-tensor pairs)
- FR-3.3: Return output as a `Prediction` struct supporting `get(name) -> Option<Tensor>`
- FR-3.4: Pre-allocate output buffers via output backings to avoid allocation per prediction
- FR-3.5: Wrap every prediction call in an autorelease pool automatically

### FR-4: Model Introspection

- FR-4.1: List input feature names, data types, and shape constraints
- FR-4.2: List output feature names, data types, and shape constraints
- FR-4.3: Access model metadata (author, description, version, license)
- FR-4.4: Report state descriptions for stateful models (when available)

### FR-5: Stateful Prediction (Phase 2)

- FR-5.1: Create a new state object from a loaded model (macOS 15+ / iOS 18+)
- FR-5.2: Run prediction with mutable state reference (state modified in-place by CoreML)
- FR-5.3: Support multiple independent state instances per model

### FR-6: Async Prediction (Phase 2)

- FR-6.1: Run prediction asynchronously, returning result via a callback or channel
- FR-6.2: Support cancellation of in-flight async predictions
- FR-6.3: Async predictions are thread-safe (CoreML async API guarantee)

### FR-7: Model Compilation (Phase 2)

- FR-7.1: Compile a `.mlmodel` or `.mlpackage` to `.mlmodelc` at runtime
- FR-7.2: Return the path to the compiled model directory
- FR-7.3: Support async compilation for large models

### FR-8: Output Tensor Access

- FR-8.1: Read output tensor data as `&[f32]` slice (scoped access via getBytesWithHandler)
- FR-8.2: Copy output tensor data into a caller-provided `&mut [f32]` buffer
- FR-8.3: Access output tensor shape and data type

### FR-9: Error Handling

- FR-9.1: All fallible operations return `Result<T, coreml::Error>`
- FR-9.2: Error type contains: error kind enum, human-readable message, source NSError description
- FR-9.3: Error type implements `std::error::Error` and `Display`

### FR-10: Cross-Platform Compilation

- FR-10.1: Crate compiles on non-Apple targets with all types stubbed (empty impls or compile errors with helpful messages)
- FR-10.2: `#[cfg(target_os = "macos")]` or `#[cfg(target_vendor = "apple")]` gates all platform-specific code
- FR-10.3: No build.rs required (pure Rust, no cc/swift compilation step)

---

## Non-Functional Requirements

### NFR-1: Performance

- NFR-1.1: Model loading latency < 500ms for a 100MB .mlmodelc on Apple Silicon as measured by benchmark
- NFR-1.2: Prediction throughput within 5% of raw objc2 calls as measured by criterion benchmark
- NFR-1.3: Zero memcpy for input tensors created from borrowed slices as verified by profiling
- NFR-1.4: Zero memcpy for output tensors when using pre-allocated output backings

### NFR-2: Memory Safety

- NFR-2.1: No memory leaks under sustained prediction workloads (1000+ iterations) as measured by Instruments Leaks
- NFR-2.2: All autorelease pools drain correctly -- no unbounded Obj-C object accumulation
- NFR-2.3: Tensor borrows enforce Rust lifetime rules -- borrowed tensors cannot outlive source data
- NFR-2.4: Model handles are Send (safe to move between threads)

### NFR-3: API Ergonomics

- NFR-3.1: No `unsafe` in user-facing code -- all unsafety confined to crate internals
- NFR-3.2: Complete rustdoc documentation with examples for all public types and methods
- NFR-3.3: Meaningful compile errors when used on non-Apple targets
- NFR-3.4: Type-safe tensor creation prevents shape/data mismatch at compile time where possible

### NFR-4: Compatibility

- NFR-4.1: Minimum macOS 12.0 (Monterey) for core features (getBytesWithHandler availability)
- NFR-4.2: macOS 15.0+ for stateful prediction (MLState)
- NFR-4.3: Rust edition 2021, MSRV 1.75.0 (objc2 requirement)
- NFR-4.4: Compatible with objc2 0.6.x, objc2-core-ml 0.3.x, block2 0.6.x

### NFR-5: Packaging

- NFR-5.1: Published to crates.io under the `coreml` name (or `coreml-safe` if taken)
- NFR-5.2: Dual-licensed Apache-2.0 OR MIT (standard Rust ecosystem)
- NFR-5.3: Minimal dependency tree -- only objc2 ecosystem crates
- NFR-5.4: No system dependencies beyond Xcode Command Line Tools SDK

### NFR-6: Testing

- NFR-6.1: Unit tests for tensor creation, shape validation, error types
- NFR-6.2: Integration tests against a small test model (.mlmodelc included in test fixtures)
- NFR-6.3: CI via GitHub Actions on macOS runners
- NFR-6.4: Benchmark suite (criterion) for load time, prediction throughput, memory usage

---

## Implementation Phases

### Phase 1: MVP Core Inference (v0.1.0)

**Goal:** Load model, create tensors, predict, read output, introspect model. Replace swictation C bridge.

| Component | FRs Covered | Priority |
|-----------|-------------|----------|
| `error.rs` -- Error types | FR-9.* | P0 |
| `config.rs` -- ComputeUnits, ModelConfiguration | FR-1.2 | P0 |
| `tensor.rs` -- Tensor from slice, shape, data type | FR-2.1, 2.4-2.7 | P0 |
| `model.rs` -- Model load, predict, drop | FR-1.1-1.4, FR-3.1-3.3, 3.5 | P0 |
| `prediction.rs` -- Prediction result, output access | FR-3.3, FR-8.1-8.3 | P0 |
| `description.rs` -- Model introspection | FR-4.1-4.3 | P0 |
| `lib.rs` -- Public API re-exports | FR-10.1-10.3 | P0 |
| `ffi.rs` -- Internal objc2 helpers (NSArray/NSDictionary conversion) | Internal | P0 |
| Integration test with small model | NFR-6.2 | P0 |
| Benchmark: load + predict | NFR-6.4 | P1 |
| Example: `load_and_predict.rs` | NFR-3.2 | P1 |
| Example: `inspect_model.rs` | NFR-3.2 | P1 |

### Phase 2: Advanced Features (v0.2.0)

**Goal:** Stateful prediction, async prediction, Float16, output backings, model compilation.

| Component | FRs Covered | Priority |
|-----------|-------------|----------|
| `tensor.rs` -- Float16 tensor support | FR-2.2 | P0 |
| `tensor.rs` -- Int32 tensor support | FR-2.3 | P0 |
| `model.rs` -- Output backings (pre-allocated output) | FR-3.4 | P0 |
| `state.rs` -- MLState create + stateful predict | FR-5.1-5.3 | P0 |
| `model.rs` -- Async predict via block2 | FR-6.1-6.3 | P1 |
| `compile.rs` -- Runtime model compilation | FR-7.1-7.3 | P1 |
| `description.rs` -- State descriptions | FR-4.4 | P1 |
| Swictation integration: replace C bridge | SC-8 | P0 |

### Phase 3: Ecosystem Release (v0.3.0)

**Goal:** Publish to crates.io, documentation, community examples, optional interop features.

| Component | FRs Covered | Priority |
|-----------|-------------|----------|
| crates.io publish | NFR-5.1-5.4, SC-7 | P0 |
| Comprehensive rustdoc | NFR-3.2 | P0 |
| Example: image classification | UJ-1 | P1 |
| Example: multi-model pipeline | UJ-3 | P1 |
| Optional feature: `ndarray` interop | Ecosystem | P2 |
| Optional feature: `half` crate interop | Ecosystem | P2 |
| GitHub Actions CI | NFR-6.3 | P0 |
| README with badges, quickstart, comparison table | Community | P0 |

---

## Architecture Overview

```
coreml (safe public API)
  src/
    lib.rs          -- Re-exports, cfg gates, crate docs
    error.rs        -- Error enum wrapping NSError
    config.rs       -- ComputeUnits enum, ModelConfig builder
    model.rs        -- Model struct (load, predict, predict_stateful, describe)
    tensor.rs       -- Tensor struct (from_slice, from_owned, shape, dtype, as_slice)
    prediction.rs   -- Prediction result (get output by name)
    description.rs  -- ModelDescription, FeatureDescription, TensorConstraint
    state.rs        -- State struct for stateful prediction [Phase 2]
    compile.rs      -- Model compilation [Phase 2]
    ffi.rs          -- Internal: NSArray<->Vec, NSDictionary<->HashMap, NSString<->str
```

### Dependency Graph

```
coreml
  +-- objc2 (0.6)           -- Message sending, ARC, autorelease
  +-- objc2-foundation (0.3) -- NSString, NSArray, NSDictionary, NSNumber, NSURL, NSError, NSSet
  +-- objc2-core-ml (0.3)   -- MLModel, MLMultiArray, MLFeatureValue, etc.
  +-- block2 (0.6)          -- Obj-C blocks for async + MLMultiArray deallocator
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure objc2 (no C/Swift bridge) | Zero external toolchain dependency, auto-updated with new SDKs |
| Send but not Sync for Model | Sync prediction is not thread-safe per Apple docs |
| Autorelease pools in every predict call | Prevents Obj-C object leaks from CoreML temporaries |
| Zero-copy input via initWithDataPointer | Avoids memcpy, Rust retains buffer ownership |
| Zero-copy output via outputBackings | Pre-allocated buffers eliminate per-prediction allocation |
| getBytesWithHandler for output reads | Replaces deprecated dataPointer, handles non-contiguous layouts |
| EnumeratedShapes over RangeDim | Better ANE compatibility for flexible input models |

---

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| objc2 breaking changes (pre-1.0) | API refactoring required | Medium | Pin versions, thin internal abstraction layer |
| ANE not dispatching expected ops | Performance regression | Low | MLComputePlan verification, benchmark suite |
| Autorelease pool mismanagement | Memory leaks | Medium | Comprehensive leak testing with Instruments |
| Tensor lifetime violations | Use-after-free | Low | Rust borrow checker enforces lifetimes at compile time |
| `coreml` crate name taken on crates.io | Need alternate name | Medium | Fallback names: `coreml-safe`, `coreml-rs`, `apple-coreml` |

---

## Dependencies and Prerequisites

- macOS 12.0+ development machine with Xcode Command Line Tools
- Apple Silicon recommended (for ANE testing); Intel Macs supported (CPU/GPU only)
- Rust 1.75.0+ (objc2 MSRV)
- Test model: Small `.mlmodelc` fixture (e.g., MobileNetV2 or custom minimal model)
- For swictation integration: FluidInference Parakeet-TDT CoreML models from HuggingFace

---

## Glossary

| Term | Definition |
|------|-----------|
| ANE | Apple Neural Engine -- fixed-function ML accelerator in Apple Silicon |
| MLMultiArray | CoreML's multi-dimensional numeric tensor type |
| .mlmodelc | Compiled (device-optimized) CoreML model bundle directory |
| MLFeatureProvider | CoreML protocol for providing named input/output feature values |
| objc2 | Rust crate ecosystem for calling Objective-C frameworks |
| RNN-T | Recurrent Neural Network Transducer -- streaming speech recognition architecture |
| Zero-copy | Data sharing without memory duplication between Rust and CoreML |
| Retained<T> | objc2's ARC-aware smart pointer (equivalent to Obj-C strong reference) |
| ComputeUnits | CoreML configuration for selecting CPU, GPU, and/or ANE backends |
