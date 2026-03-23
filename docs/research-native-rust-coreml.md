# Native Rust CoreML Inference Crate - Research Findings

**Date**: 2026-03-23
**Scope**: Comprehensive research for building an ergonomic, safe Rust crate wrapping Apple's CoreML framework for ML inference, targeting STT (Parakeet-TDT) and TTS (Kokoro-82M) models on Apple Silicon.

---

## 1. Ecosystem Landscape

### Existing Rust CoreML Crates

| Crate | Version | Approach | Status | Limitations |
|-------|---------|----------|--------|-------------|
| `objc2-core-ml` | 0.3.2 | Auto-generated objc2 bindings | Active, comprehensive | Raw `unsafe` API, no ergonomic wrapper |
| `coreml-rs` | 0.4.0 | Swift bridge via `swift-bridge` | 14 stars, WIP | Requires Swift runtime, barely maintained |
| `candle-coreml` | 0.3.1 | objc2-based, Candle tensors | Active | Coupled to HuggingFace Candle ecosystem |
| `adamnemecek/coreml` | N/A | Direct Obj-C FFI | Abandoned (1 star) | Not viable |

### Related Projects

| Project | Description | Relevance |
|---------|-------------|-----------|
| `Rustane` | Drives ANE directly via private `_ANEClient` APIs | Experimental, bypasses CoreML entirely |
| `FluidAudio` | Swift CoreML SDK (1,697 stars) | Has pre-converted Parakeet-TDT + Kokoro CoreML models |
| `whisper.cpp` | C bridge to CoreML for Whisper encoder | Reference pattern for C bridge approach |
| `Kokoros`/`kokorox` | Kokoro-82M TTS in Rust via ONNX/Candle | Potential CoreML backend target |

### Gap Analysis

**No standalone, ergonomic, safe Rust CoreML inference crate exists.** The `objc2-core-ml` bindings are complete (100% API coverage) but raw. Everyone else is either coupled to a specific ML framework (candle-coreml), barely maintained (coreml-rs), or abandoned.

---

## 2. Architecture Decision: objc2 vs C Bridge vs Swift Bridge

### Recommendation: **objc2 ecosystem** (Pure Rust, no C/Obj-C/Swift code)

**Dependencies:**
- `objc2` v0.6.4 -- Message sending, ARC via `Retained<T>`, autorelease pools
- `objc2-foundation` v0.3 -- NSString, NSArray, NSDictionary, NSNumber, NSURL, NSError
- `objc2-core-ml` v0.3.2 -- Complete CoreML bindings (62 structs, 6 traits)
- `block2` v0.6 -- Obj-C block support for async callbacks and MLMultiArray deallocators

**Why not C bridge (current swictation approach):**
- Manual memory management, no type safety, boilerplate for every new API
- Every CoreML feature addition requires hand-written Obj-C

**Why not swift-bridge:**
- Requires Swift runtime (`libswift_Concurrency.dylib`)
- Extra build complexity, double indirection (Rust -> C FFI -> Swift -> CoreML)
- Less mature for framework interop

**Why objc2:**
- Auto-generated from Xcode 16.4 SDKs -- always up-to-date
- Type-safe: nullable -> `Option<Retained<T>>`, NSError** -> `Result<T, Retained<NSError>>`
- ARC handled automatically via `Retained<T>` (retain on create, release on Drop)
- Servo ecosystem converging on objc2, deprecating old hand-written crates
- 874 stars, 60K+ dependents, 69 contributors, active maintenance
- Protocol conformance possible from Rust via `define_class!`

### Comparison Table

| Challenge | objc2 | C Bridge | swift-bridge |
|-----------|-------|----------|-------------|
| NSError handling | Auto `Result<T, Retained<NSError>>` via `_` marker | Manual error codes/strings | Via Swift's throws |
| ARC | Automatic `Retained<T>` | `-fobjc-arc` + CFBridging | Automatic (Swift side) |
| Zero-copy MLMultiArray | `initWithDataPointer` with `block2` deallocator | Direct pointer, manual lifetime | Extra copy likely |
| Async callbacks | `block2::RcBlock` | Callback function pointers | Native async/await |
| Protocol conformance | `define_class!` + `unsafe impl` | Must implement in Obj-C | Implement in Swift |
| Maintenance | Auto-generated, zero effort | Manual for each new API | Manual Swift wrapper |

---

## 3. CoreML API Surface (Complete Reference)

### MLModel

**Loading:**
```rust
// Sync (objc2)
MLModel::modelWithContentsOfURL_configuration_error(&url, &config) -> Result<Retained<MLModel>, Retained<NSError>>

// Async (requires block2)
MLModel::loadContentsOfURL_configuration_completionHandler(&url, &config, &handler_block)
```

**Prediction:**
```rust
// Sync
model.predictionFromFeatures_error(&provider) -> Result<Retained<ProtocolObject<dyn MLFeatureProvider>>, Retained<NSError>>

// With options (for output backings)
model.predictionFromFeatures_options_error(&provider, &options) -> Result<...>

// Async (iOS 17+, requires block2)
model.predictionFromFeatures_completionHandler(&provider, &completion_block)

// Stateful (iOS 18+ / macOS 15+)
model.predictionFromFeatures_usingState_error(&provider, &state) -> Result<...>

// Batch
model.predictionsFromBatch_error(&batch_provider) -> Result<Retained<ProtocolObject<dyn MLBatchProvider>>, ...>
```

**Introspection:**
```rust
model.modelDescription() -> Retained<MLModelDescription>
model.configuration() -> Retained<MLModelConfiguration>
```

**Compilation:**
```rust
MLModel::compileModelAtURL_error(&source_url) -> Result<Retained<NSURL>, ...>
```

### MLModelConfiguration

```rust
config.setComputeUnits(MLComputeUnits)  // .cpuOnly=0, .cpuAndGPU=1, .cpuAndNeuralEngine=2, .all=3
config.setAllowLowPrecisionAccumulationOnGPU(bool)
config.setModelDisplayName(Option<&NSString>)
config.setOptimizationHints(&MLOptimizationHints)
config.setPreferredMetalDevice(Option<&ProtocolObject<dyn MTLDevice>>)
config.setFunctionName(Option<&NSString>)  // Multi-function models, iOS 18+
```

### MLMultiArray

**Creation (zero-copy from Rust buffer):**
```rust
MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
    alloc,
    data_ptr: NonNull<c_void>,     // Rust-owned buffer
    shape: &NSArray<NSNumber>,
    data_type: MLMultiArrayDataType,  // Float16, Float32, Float64, Int32
    strides: &NSArray<NSNumber>,
    deallocator: Option<&DynBlock<dyn Fn(NonNull<c_void>)>>,  // None = Rust owns memory
) -> Result<Retained<Self>, Retained<NSError>>
```

**Data access:**
```rust
// DEPRECATED (but still works):
array.dataPointer() -> NonNull<c_void>

// PREFERRED (scoped access with correct strides):
array.getBytesWithHandler(&handler_block)  // Read-only
array.getMutableBytesWithHandler(&handler_block)  // Mutable, strides may differ

// Properties:
array.shape() -> Retained<NSArray<NSNumber>>
array.strides() -> Retained<NSArray<NSNumber>>
array.dataType() -> MLMultiArrayDataType
array.count() -> NSInteger
```

**Data types:**
- `MLMultiArrayDataType::Float16` -- ANE native precision
- `MLMultiArrayDataType::Float32` -- GPU/CPU standard
- `MLMultiArrayDataType::Float64` -- Double precision
- `MLMultiArrayDataType::Int32` -- Integer

### MLFeatureValue

```rust
// Creation:
MLFeatureValue::featureValueWithMultiArray(&array) -> Retained<Self>
MLFeatureValue::featureValueWithInt64(i64) -> Retained<Self>
MLFeatureValue::featureValueWithDouble(f64) -> Retained<Self>
MLFeatureValue::featureValueWithString(&NSString) -> Retained<Self>

// Access:
value.multiArrayValue() -> Option<Retained<MLMultiArray>>
value.int64Value() -> i64
value.doubleValue() -> f64
value.stringValue() -> Retained<NSString>
value.r#type() -> MLFeatureType
```

### MLDictionaryFeatureProvider

```rust
MLDictionaryFeatureProvider::initWithDictionary_error(
    alloc,
    &NSDictionary<NSString, AnyObject>,
) -> Result<Retained<Self>, Retained<NSError>>

// Implements MLFeatureProvider protocol:
provider.featureNames() -> Retained<NSSet<NSString>>
provider.featureValueForName(&NSString) -> Option<Retained<MLFeatureValue>>
```

### MLModelDescription (Introspection)

```rust
desc.inputDescriptionsByName() -> Retained<NSDictionary<NSString, MLFeatureDescription>>
desc.outputDescriptionsByName() -> Retained<NSDictionary<NSString, MLFeatureDescription>>
desc.stateDescriptionsByName() -> Retained<NSDictionary<NSString, MLFeatureDescription>>  // iOS 18+

// MLFeatureDescription:
feat.name() -> Retained<NSString>
feat.r#type() -> MLFeatureType
feat.isOptional() -> bool
feat.multiArrayConstraint() -> Option<Retained<MLMultiArrayConstraint>>

// MLMultiArrayConstraint:
constraint.shape() -> Retained<NSArray<NSNumber>>
constraint.dataType() -> MLMultiArrayDataType
```

### MLPredictionOptions

```rust
options.setOutputBackings(&NSDictionary<NSString, MLMultiArray>)  // Pre-allocated output buffers
```

### MLState (iOS 18+ / macOS 15+)

```rust
let state = model.newState()  // Create fresh state
model.predictionFromFeatures_usingState_error(&input, &state)  // State mutated in-place
```

---

## 4. Critical Implementation Details

### Autorelease Pool Requirement

**Every Rust call into CoreML prediction MUST be wrapped in an autorelease pool.** CoreML allocates temporary Obj-C objects that are autoreleased. Without a draining pool, memory leaks.

```rust
use objc2::rc::autoreleasepool;

pub fn predict(&self, input: &FeatureProvider) -> Result<FeatureProvider> {
    autoreleasepool(|_pool| {
        let result = unsafe {
            self.inner.predictionFromFeatures_error(&input.inner)
        }.map_err(|e| Error::Prediction(nserror_to_string(&e)))?;
        Ok(FeatureProvider { inner: result })
    })
}
```

### Zero-Copy Input Pattern

```rust
// Allocate page-aligned buffer in Rust for best ANE performance
let data: Vec<f32> = vec![0.0; total_elements];
let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();

let array = unsafe {
    MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
        MLMultiArray::alloc(),
        ptr,
        &shape_nsarray,
        MLMultiArrayDataType::Float32,
        &strides_nsarray,
        None,  // No deallocator -- Rust owns the buffer
    )
}?;
// CRITICAL: `data` must outlive `array`
```

### Zero-Copy Output Pattern

```rust
// Pre-allocate output buffer
let mut output_buf: Vec<f32> = vec![0.0; output_size];
let output_array = create_multiarray_from_slice(&mut output_buf, &output_shape)?;

// Register as output backing
let options = unsafe { MLPredictionOptions::new() };
let backings = create_dict(&[("output_name", &output_array)]);
unsafe { options.setOutputBackings(&backings) };

// Predict -- CoreML writes directly into our buffer
let result = unsafe {
    model.predictionFromFeatures_options_error(&input, &options)
}?;
// output_buf now contains the result, no copy needed
```

### NSError to Rust Error Conversion

```rust
fn nserror_to_string(error: &NSError) -> String {
    unsafe { error.localizedDescription() }.to_string()
}

// objc2's msg_send! auto-converts NSError** via the `_` marker:
let result: Result<Retained<MLModel>, Retained<NSError>> = unsafe {
    msg_send![MLModel::class(), modelWithContentsOfURL: &url, configuration: &config, error: _]
};
// But the generated methods in objc2-core-ml already do this for us.
```

### Thread Safety

- **Synchronous `prediction(from:)`**: NOT documented as thread-safe
- **Async prediction API (iOS 17+)**: Explicitly thread-safe
- **Model struct**: Should be `Send` but NOT `Sync` for synchronous API
- **For concurrent access**: Either use async API or maintain a model instance pool

---

## 5. Apple Neural Engine (ANE) Technical Details

### Architecture (M4)

- 16 cores with independent DVFS
- 38 TOPS (INT8) / ~19 TFLOPS (FP16)
- Fixed-function graph execution engine (not programmable)
- Primary primitive: convolution (1x1 conv is 3x faster than general matmul)
- Tensor layout: NCDHW + Interleave, last axis must be 64-byte aligned
- FP16-only through M3; A17 Pro/M4+ add INT8 arithmetic

### ANE-Compatible Operations

**Fully supported:** Convolution (all standard), MatMul, Pooling (kernel<=13, stride<=2), BatchNorm, LayerNorm, InstanceNorm, ReLU, sigmoid, tanh, element-wise add/multiply, upsampling (scale<=2), transposed convolution, softmax, reshape, transpose, concat, split

**NOT supported (CPU/GPU fallback):**
- **LSTM/GRU/RNN layers** -- critical for Parakeet-TDT decoder
- Custom layers (no public ANE API)
- Gather operations
- Dilated convolutions
- Broadcastable layers (AddBroadcastableLayer, MultiplyBroadcastableLayer)
- "ND" layers (ConcatND, SplitND, LoadConstantND)
- Pooling kernel >13 or stride >2
- Upsampling scale >2
- Dynamic reshape

### ANE Optimization for Transformers (Apple Research)

1. Use 4D channels-first `(B, C, 1, S)` layout, not 3D `(B, S, C)`
2. Replace `nn.Linear` with `nn.Conv2d`
3. Chunk multi-head attention into single-head operations
4. Avoid reshape ops -- use einsum `bchq,bkhc->bkhq`
5. For short sequences, increase batch size or apply quantization

### Performance (distilbert, iPhone 13, seq=128, batch=1)

- ANE-optimized: 3.47ms (up to 10x faster, 14x less memory vs baseline)
- Comparable to AWS Inferentia server-side ASIC

### Performance by Device

| Chip | TOPS | Key Devices |
|------|------|-------------|
| A14 | 11 | iPhone 12 |
| A15 | 15.8 | iPhone 13/14 |
| A17 Pro | 35 | iPhone 15 Pro |
| M1 | 11 | MacBook Air/Pro |
| M2 | 15.8 | MacBook Air/Pro |
| M4 | 38 | iPad Pro, MacBook Air |

---

## 6. Model Format & Conversion

### Format Hierarchy

- `.mlmodel` -- Source format (protobuf, editable)
- `.mlpackage` -- Directory format (architecture + weights separated, required for MLProgram)
- `.mlmodelc` -- Compiled format (runtime, device-optimized)

### Conversion Pipeline

**Recommended: PyTorch -> CoreML direct**
```python
import coremltools as ct
traced = torch.jit.trace(model, example_input)
mlmodel = ct.convert(traced, inputs=[ct.TensorType(shape=input_shape)],
                     convert_to="mlprogram",
                     compute_precision=ct.precision.FLOAT16)
mlmodel.save("model.mlpackage")
```

**NOT recommended:** ONNX -> CoreML (deprecated converter, not actively developed)

### Flexible Input Shapes

**EnumeratedShapes (preferred for ANE):** Up to 128 pre-defined shape buckets
```python
ct.TensorType(name="input", shape=ct.EnumeratedShapes(
    shapes=[[1, 80, 100], [1, 80, 200], [1, 80, 500]],
    default=[1, 80, 500]
))
```

**RangeDim:** Bounded ranges per dimension (may impact ANE performance)
```python
ct.TensorType(name="input", shape=(1, 80, ct.RangeDim(1, 3000)))
```

### Quantization

| Technique | ANE Support | Effect |
|-----------|-------------|--------|
| Float16 | Full (all ANE generations) | 2x size reduction, native precision |
| INT8 weights-only | All ANE | Reduced memory bandwidth |
| INT8 compute (W8A8) | A17 Pro/M4+ only | Full INT8 arithmetic |
| 4-bit palettization | Yes | ~4x compression, minimal quality loss |

### coremltools 8.3.0 (Latest, April 2025)

New utilities: `MLModelValidator`, `MLModelComparator`, `MLModelInspector`, `MLModelBenchmarker`, remote device profiling

### Stateful Models (iOS 18+ / macOS 15+)

```python
mlmodel = ct.convert(traced,
    inputs=[ct.TensorType(shape=(1,))],
    states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1, 640)), name="decoder_state")],
    minimum_deployment_target=ct.target.iOS18)
```

Performance: KV-cache benchmark M3 Max: 238ms with state vs 4,245ms without (17.8x speedup)

---

## 7. Pre-Converted Models Available

### STT (Speech-to-Text)

- `FluidInference/parakeet-tdt-0.6b-v3-coreml` on HuggingFace
  - ~110x RTF on M4 Pro (1 min audio in ~0.5s)
  - ~800MB peak memory, macOS 14+ / iOS 17+
  - Conversion scripts: `github.com/FluidInference/mobius`

### TTS (Text-to-Speech)

- `FluidInference/kokoro-82m-coreml` on HuggingFace
  - Kokoro-82M (StyleTTS 2 based), 87M parameters
  - 30-50% vocoder speedup via ANE
  - Bucketing strategy for variable-length synthesis
  - Apache 2.0 license

---

## 8. Swictation Current Implementation

### Architecture

Three-model RNN-T pipeline: Encoder (FastConformer) + Decoder (BiLSTM) + Joiner (FFN)

### Model Specifications

| Aspect | 0.6B | 1.1B |
|--------|------|------|
| Mel Features | 128 | 80 |
| Encoder Dim | 640 | 1024 |
| Decoder Hidden | 640 | 640 |
| Vocab Size | ~7400 | ~4000+ |
| WER | 7-8% | 5.77% |

### Current C Bridge (`coreml_bridge.m`)

- 225 lines of Obj-C compiled with `cc` crate
- Functions: `coreml_load_model`, `coreml_predict`, `coreml_predict_multi`, `coreml_free_model`, `coreml_free_string`
- Uses `initWithDataPointer` for zero-copy input, `memcpy` for output
- ARC via `-fobjc-arc`, `__bridge_retained`/`__bridge_transfer` for lifetime management

### ANE Execution Reality

- **Encoder** (conv-heavy FastConformer): Runs on ANE
- **Decoder** (BiLSTM): Falls back to CPU (LSTM not ANE-supported)
- **Joiner** (FFN): Runs on ANE
- This hybrid execution is expected and matches whisper.cpp's pattern

---

## 9. TTS Model Recommendations

### Primary: Kokoro-82M

- 87M parameters, single forward pass (non-iterative)
- Proven CoreML conversion with ANE acceleration
- Pre-converted models on HuggingFace
- Rust implementations exist (Kokoros, kokorox)
- Conv-heavy HiFi-GAN vocoder maps well to ANE

### Secondary: Piper TTS (VITS-based)

- 40+ languages, MIT license, extremely lightweight
- Native ONNX export, works with `ort` CoreML EP
- Rust bindings: `piper-rs`, `piper-tts-rs`

### Tertiary: Matcha-TTS

- Ultra-light models under 10MB
- 2-4 ODE steps for good quality
- Available via sherpa-onnx with CoreML

---

## 10. Proposed Crate Architecture

```
coreml/
  src/
    lib.rs          -- Public API: Model, Tensor, Prediction, ComputeUnits
    model.rs        -- MLModel wrapper (load, predict, describe, compile)
    tensor.rs       -- MLMultiArray wrapper (zero-copy, typed access)
    feature.rs      -- MLFeatureValue/Provider builders
    config.rs       -- MLModelConfiguration (compute units, precision)
    description.rs  -- Model introspection (input/output shapes/types)
    error.rs        -- Error types wrapping NSError
    state.rs        -- MLState for stateful RNN/KV-cache models
  Cargo.toml
  examples/
    load_and_predict.rs
    inspect_model.rs
    stateful_prediction.rs
```

### Key Design Principles

1. **Safe public API** -- all `unsafe` confined to internal implementation
2. **Zero-copy I/O** -- `initWithDataPointer` for input, `outputBackings` for output
3. **Autorelease pools** -- automatically wrapped around every CoreML call
4. **Typed tensors** -- `Tensor<f32>`, `Tensor<f16>`, `Tensor<i32>`
5. **Builder pattern** -- for feature providers and configurations
6. **`cfg(target_os = "macos")`** -- conditional compilation throughout
7. **`Send` but not `Sync`** -- matches CoreML threading model for sync API

---

## Sources

### Core Documentation
- [objc2-core-ml docs.rs](https://docs.rs/objc2-core-ml) (v0.3.2, 62 structs, 6 traits)
- [objc2 GitHub](https://github.com/madsmtm/objc2) (874 stars, 60K+ dependents)
- [block2 docs.rs](https://docs.rs/block2)
- [MLModel Apple Docs](https://developer.apple.com/documentation/coreml/mlmodel)
- [MLMultiArray Apple Docs](https://developer.apple.com/documentation/coreml/mlmultiarray)
- [MLComputeUnits Apple Docs](https://developer.apple.com/documentation/coreml/mlcomputeunits)
- [MLFeatureProvider Apple Docs](https://developer.apple.com/documentation/coreml/mlfeatureprovider)

### Apple Research & WWDC
- [Deploying Transformers on ANE](https://machinelearning.apple.com/research/neural-engine-transformers)
- [WWDC24: Deploy ML on-device](https://developer.apple.com/videos/play/wwdc2024/10161/)
- [WWDC23: Async prediction](https://developer.apple.com/videos/play/wwdc2023/10049/)
- [coremltools 8.3.0](https://github.com/apple/coremltools/releases)
- [Stateful Models Guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [Flexible Input Shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)

### ANE Reference
- [hollance/neural-engine](https://github.com/hollance/neural-engine)
- [Inside M4 ANE](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [ONNX Runtime CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)

### Existing Projects
- [coreml-rs](https://github.com/swarnimarun/coreml-rs) (Swift bridge, 14 stars)
- [candle-coreml](https://crates.io/crates/candle-coreml) (Candle integration)
- [Rustane](https://github.com/ncdrone/rustane) (Direct ANE access)
- [FluidAudio](https://github.com/FluidInference/FluidAudio) (Swift CoreML SDK)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (C bridge reference)
- [kokoro-coreml](https://github.com/mattmireles/kokoro-coreml) (TTS conversion)

### Pre-Converted Models
- [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)
- [FluidInference/kokoro-82m-coreml](https://huggingface.co/FluidInference/kokoro-82m-coreml)

### Rust TTS Ecosystem
- [Kokoros](https://github.com/lucasjinreal/Kokoros) (Kokoro-82M in Rust)
- [kokorox](https://lib.rs/crates/kokorox) (Community fork)
- [piper-rs](https://lib.rs/crates/piper-rs) (Piper VITS in Rust)
