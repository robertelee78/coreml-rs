//! Load a CoreML model and run prediction.
//!
//! Usage: cargo run --example load_and_predict -- <path/to/model.mlmodelc>

use coreml_rs::{BorrowedTensor, ComputeUnits, Model};

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/test_linear.mlmodelc".to_string());

    println!("Loading model: {model_path}");
    let model = Model::load(&model_path, ComputeUnits::All).expect("Failed to load model");

    // Create a simple input tensor
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor =
        BorrowedTensor::from_f32(&input_data, &[1, 4]).expect("Failed to create tensor");

    println!("Running prediction...");
    let prediction = model.predict(&[("input", &tensor)]).expect("Prediction failed");

    let (output, shape) = prediction.get_f32("output").expect("Failed to get output");
    println!("Output shape: {shape:?}");
    println!("Output data: {output:?}");
}
