//! Inspect a CoreML model's inputs, outputs, and metadata.
//!
//! Usage: cargo run --example inspect_model -- <path/to/model.mlmodelc>

use coreml_native::{ComputeUnits, Model};

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/test_linear.mlmodelc".to_string());

    println!("Loading model: {model_path}");
    let model = Model::load(&model_path, ComputeUnits::All).expect("Failed to load model");

    println!("\n--- Inputs ---");
    for desc in model.inputs() {
        println!(
            "  {}: {:?} shape={:?} optional={}",
            desc.name(),
            desc.feature_type(),
            desc.shape(),
            desc.is_optional(),
        );
        if let Some(dt) = desc.data_type() {
            println!("    data_type: {dt}");
        }
    }

    println!("\n--- Outputs ---");
    for desc in model.outputs() {
        println!(
            "  {}: {:?} shape={:?}",
            desc.name(),
            desc.feature_type(),
            desc.shape(),
        );
        if let Some(dt) = desc.data_type() {
            println!("    data_type: {dt}");
        }
    }

    let meta = model.metadata();
    println!("\n--- Metadata ---");
    println!("  author: {:?}", meta.author);
    println!("  description: {:?}", meta.description);
    println!("  version: {:?}", meta.version);
    println!("  license: {:?}", meta.license);
}
