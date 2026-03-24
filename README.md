# coreml-native

Safe, ergonomic Rust bindings for Apple CoreML inference with full Apple Neural Engine (ANE) acceleration. Built on [`objc2-core-ml`](https://docs.rs/objc2-core-ml) — no C bridge, no Swift runtime, pure Rust.

## Features

- **Load** compiled `.mlmodelc` models with configurable compute units (CPU/GPU/ANE)
- **Zero-copy tensors** from Rust slices via `MLMultiArray::initWithDataPointer`
- **Predict** with named inputs/outputs, automatic Float16-to-Float32 conversion
- **Async APIs** — `load_async`, `predict_async`, `compile_model_async` with runtime-agnostic `CompletionFuture`
- **Batch prediction** — submit multiple inputs in one call via `BatchProvider`
- **Model lifecycle** — `ModelHandle` with type-safe `unload`/`reload` for GPU/ANE memory management
- **9 tensor data types** — Float16, Float32, Float64, Int32, Int16, Int8, UInt32, UInt16, UInt8
- **Introspect** model inputs, outputs, shapes, data types, shape constraints, and metadata
- **Stateful prediction** via `MLState` for RNN decoders and KV-cache (macOS 15+)
- **Compile** `.mlmodel`/`.mlpackage` to `.mlmodelc` at runtime (sync and async)
- **Device enumeration** — discover available CPU, GPU, and Neural Engine devices
- **ndarray integration** — optional feature flag for zero-copy `ndarray` ↔ tensor conversion
- **Stride-aware output copy** — correctly handles non-contiguous GPU/ANE tensor layouts
- **Thread-safe** — `Model`, `Prediction`, and tensor types are `Send + Sync`
- **Cross-platform** — compiles on Linux/Windows with stub types (no-op)
- **No external toolchain** — pure `cargo build`, no Xcode project needed

## Quick Start

```rust
use coreml_native::{Model, BorrowedTensor, ComputeUnits};

let model = Model::load("model.mlmodelc", ComputeUnits::All)?;

let data = vec![1.0f32, 2.0, 3.0, 4.0];
let tensor = BorrowedTensor::from_f32(&data, &[1, 4])?;

let prediction = model.predict(&[("input", &tensor)])?;
let (output, shape) = prediction.get_f32("output")?;

println!("shape: {shape:?}, output: {output:?}");
```

## Async Loading & Prediction

`CompletionFuture` works with any async runtime (tokio, async-std, smol) or can be blocked on synchronously:

```rust
use coreml_native::{Model, BorrowedTensor, ComputeUnits};

// Async load — .await or .block_on()
let model = Model::load_async("model.mlmodelc", ComputeUnits::All)?
    .block_on()?;

// Async prediction
let data = vec![1.0f32; 4];
let tensor = BorrowedTensor::from_f32(&data, &[1, 4])?;
let prediction = model.predict_async(&[("input", &tensor)])?
    .block_on()?;

// Load from in-memory bytes (macOS 14.4+)
let spec_bytes = std::fs::read("model.mlmodel")?;
let model = Model::load_from_bytes(&spec_bytes, ComputeUnits::All)?
    .block_on()?;
```

## Batch Prediction

More efficient than calling `predict()` in a loop — CoreML optimizes across the batch:

```rust
use coreml_native::{Model, BorrowedTensor, ComputeUnits, BatchProvider};

let model = Model::load("model.mlmodelc", ComputeUnits::All)?;

let data_a = vec![1.0f32; 4];
let data_b = vec![2.0f32; 4];
let tensor_a = BorrowedTensor::from_f32(&data_a, &[1, 4])?;
let tensor_b = BorrowedTensor::from_f32(&data_b, &[1, 4])?;

let inputs: Vec<Vec<(&str, &dyn coreml_native::AsMultiArray)>> = vec![
    vec![("input", &tensor_a)],
    vec![("input", &tensor_b)],
];
let input_refs: Vec<&[(&str, &dyn coreml_native::AsMultiArray)]> =
    inputs.iter().map(|v| v.as_slice()).collect();

let batch = BatchProvider::new(&input_refs)?;
let results = model.predict_batch(&batch)?;

for i in 0..results.count() {
    let (output, shape) = results.get_f32(i, "output")?;
    println!("batch {i}: shape={shape:?}, output={output:?}");
}
```

## Model Lifecycle Management

`ModelHandle` uses move semantics to prevent use-after-unload at compile time:

```rust
use coreml_native::{ModelHandle, ComputeUnits};

let handle = ModelHandle::load("model.mlmodelc", ComputeUnits::All)?;

// Use the model
let prediction = handle.predict(&[("input", &tensor)])?;

// Free GPU/ANE memory when idle
let handle = handle.unload()?;
assert!(!handle.is_loaded());

// Reload when needed — same path and compute config preserved
let handle = handle.reload()?;
```

## Tensor Types

### BorrowedTensor — zero-copy from Rust slices

```rust
use coreml_native::BorrowedTensor;

// Supported: f32, i32, f64, f16 (as u16 bits), i16, i8, u32, u16, u8
let tensor = BorrowedTensor::from_f32(&data, &[1, 4])?;
let tensor = BorrowedTensor::from_i32(&int_data, &[2, 3])?;
let tensor = BorrowedTensor::from_f16_bits(&half_data, &[1, 8])?;
```

### OwnedTensor — CoreML-allocated memory

```rust
use coreml_native::{OwnedTensor, DataType};

let tensor = OwnedTensor::zeros(DataType::Float32, &[1, 4])?;
let data = tensor.to_vec_f32()?;   // copy out as Vec<f32>
let data = tensor.to_vec_i32()?;   // copy out as Vec<i32>
let bytes = tensor.to_raw_bytes()?; // raw byte copy
```

### Output Extraction

```rust
// Allocating
let (f32_data, shape) = prediction.get_f32("output")?;
let (i32_data, shape) = prediction.get_i32("output")?;
let (f64_data, shape) = prediction.get_f64("output")?;
let (bytes, shape, dtype) = prediction.get_raw("output")?;

// Zero-alloc — reuse a buffer across predictions
let mut buf = vec![0.0f32; 1024];
let shape = prediction.get_f32_into("output", &mut buf)?;
```

## Model Introspection

```rust
let model = Model::load("model.mlmodelc", ComputeUnits::CpuOnly)?;

for input in model.inputs() {
    println!("{}: {:?} shape={:?} dtype={:?} optional={}",
        input.name(), input.feature_type(), input.shape(),
        input.data_type(), input.is_optional());
    if let Some(constraint) = input.shape_constraint() {
        println!("  constraint: {:?}", constraint);
    }
}

let meta = model.metadata();
println!("author={:?} version={:?} updatable={}",
    meta.author, meta.version, meta.is_updatable);
```

## Device Enumeration

```rust
use coreml_native::{available_devices, ComputeDevice};

for device in available_devices() {
    println!("{device}"); // "CPU", "GPU (M1 Pro)", "Neural Engine"
}
```

## Stateful Prediction (macOS 15+)

For RNN decoders, KV-cache models, and other stateful architectures:

```rust
let model = Model::load("decoder.mlmodelc", ComputeUnits::All)?;
let state = model.new_state()?;

// State persists across calls
let pred1 = model.predict_stateful(&[("input", &tensor1)], &state)?;
let pred2 = model.predict_stateful(&[("input", &tensor2)], &state)?;
```

## Runtime Compilation

```rust
use coreml_native::compile_model;

// Sync
let compiled_path = compile_model("model.mlpackage")?;

// Async (macOS 14.4+)
let compiled_path = coreml_native::compile_model_async("model.mlpackage")?
    .block_on()?;
```

## ndarray Integration

Enable with the `ndarray` feature flag:

```toml
[dependencies]
coreml-native = { version = "0.2", features = ["ndarray"] }
```

```rust
use ndarray::array;
use coreml_native::ndarray_support::PredictionNdarray;

// ndarray → tensor (zero-copy, must be standard layout)
let input = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();
let tensor = BorrowedTensor::from_ndarray_f32(&input)?;

// prediction → ndarray
let output = prediction.get_ndarray_f32("output")?;
println!("shape: {:?}, sum: {}", output.shape(), output.sum());
```

## Requirements

- macOS 12+ (Monterey) for core features
- macOS 14+ for async prediction
- macOS 14.4+ for `load_from_bytes` and `compile_model_async`
- macOS 15+ for stateful prediction (`MLState`)
- Apple Silicon recommended for ANE acceleration
- Rust 1.75+

## Comparison

| Crate | Approach | ANE | Standalone | Maintained |
|-------|----------|-----|-----------|-----------|
| **coreml-native** | objc2 bindings | Full | Yes | Yes |
| `objc2-core-ml` | Raw auto-gen | Full | Yes* | Yes |
| `coreml-rs` (swarnimarun) | Swift bridge | Yes | No (Swift runtime) | Minimal |
| `candle-coreml` | objc2 + Candle | Yes | No (Candle dep) | Yes |
| `ort` CoreML EP | ONNX Runtime | Partial | Yes | Yes |

*Raw `unsafe` API without ergonomic wrappers.

## License

Apache-2.0 OR MIT
