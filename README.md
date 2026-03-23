# coreml-rs

Rust implementation of CoreML.

Safe, ergonomic Rust bindings for Apple CoreML inference with full Apple Neural Engine (ANE) acceleration. Built on [`objc2-core-ml`](https://docs.rs/objc2-core-ml) — no C bridge, no Swift runtime, pure Rust.

## Features

- **Load** compiled `.mlmodelc` models with configurable compute units (CPU/GPU/ANE)
- **Zero-copy tensors** from Rust slices via `MLMultiArray::initWithDataPointer`
- **Predict** with named inputs/outputs, automatic Float16-to-Float32 conversion
- **Introspect** model inputs, outputs, shapes, data types, and metadata
- **Stateful prediction** via `MLState` for RNN decoders and KV-cache (macOS 15+)
- **Compile** `.mlmodel`/`.mlpackage` to `.mlmodelc` at runtime
- **Cross-platform** — compiles on Linux/Windows with stub types (no-op)
- **No external toolchain** — pure `cargo build`, no Xcode project needed

## Quick Start

```rust
use coreml_rs::{Model, BorrowedTensor, ComputeUnits};

let model = Model::load("model.mlmodelc", ComputeUnits::All)?;

let data = vec![1.0f32, 2.0, 3.0, 4.0];
let tensor = BorrowedTensor::from_f32(&data, &[1, 4])?;

let prediction = model.predict(&[("input", &tensor)])?;
let (output, shape) = prediction.get_f32("output")?;

println!("shape: {shape:?}, output: {output:?}");
```

## Model Introspection

```rust
let model = Model::load("model.mlmodelc", ComputeUnits::CpuOnly)?;

for input in model.inputs() {
    println!("{}: {:?} shape={:?}", input.name(), input.feature_type(), input.shape());
}
```

## Requirements

- macOS 12+ (Monterey) for core features
- macOS 15+ for stateful prediction (`MLState`)
- Apple Silicon recommended for ANE acceleration
- Rust 1.75+

## Comparison

| Crate | Approach | ANE | Standalone | Maintained |
|-------|----------|-----|-----------|-----------|
| **coreml-rs** | objc2 bindings | Full | Yes | Yes |
| `objc2-core-ml` | Raw auto-gen | Full | Yes* | Yes |
| `coreml-rs` (swarnimarun) | Swift bridge | Yes | No (Swift runtime) | Minimal |
| `candle-coreml` | objc2 + Candle | Yes | No (Candle dep) | Yes |
| `ort` CoreML EP | ONNX Runtime | Partial | Yes | Yes |

*Raw `unsafe` API without ergonomic wrappers.

## License

Apache-2.0 OR MIT
