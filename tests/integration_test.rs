//! Integration test: load test model, predict, verify output.

#[cfg(target_vendor = "apple")]
#[test]
fn load_and_predict_linear_model() {
    use coreml_rs::{BorrowedTensor, ComputeUnits, Model};

    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/test_linear.mlmodelc");
    let model = Model::load(model_path, ComputeUnits::All)
        .expect("failed to load test model");

    // Input: [1, 2, 3, 4]
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = BorrowedTensor::from_f32(&input_data, &[1, 4])
        .expect("failed to create tensor");

    let prediction = model.predict(&[("input", &tensor)])
        .expect("prediction failed");

    // Expected: 2*x + 1 = [3, 5, 7, 9]
    let (output, shape) = prediction.get_f32("output")
        .expect("failed to get output");

    assert_eq!(shape, vec![1, 4]);
    assert_eq!(output.len(), 4);

    let expected = vec![3.0f32, 5.0, 7.0, 9.0];
    for (got, want) in output.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 0.01,
            "output mismatch: got {got}, expected {want}"
        );
    }
}

#[cfg(target_vendor = "apple")]
#[test]
fn model_introspection() {
    use coreml_rs::{ComputeUnits, Model};

    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/test_linear.mlmodelc");
    let model = Model::load(model_path, ComputeUnits::CpuOnly)
        .expect("failed to load test model");

    let inputs = model.inputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].name(), "input");

    let outputs = model.outputs();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].name(), "output");
}

#[cfg(target_vendor = "apple")]
#[test]
fn model_load_invalid_path() {
    use coreml_rs::{ComputeUnits, ErrorKind, Model};

    let err = Model::load("/nonexistent/model.mlmodelc", ComputeUnits::All)
        .unwrap_err();
    assert_eq!(err.kind(), &ErrorKind::ModelLoad);
}
