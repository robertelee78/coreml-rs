//! Integration tests: real CoreML model loading, inference, and introspection.
//!
//! These tests use tests/fixtures/test_linear.mlmodelc (y = 2x + 1).
//! They only run on Apple platforms.

#[cfg(target_vendor = "apple")]
const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/test_linear.mlmodelc");

#[cfg(target_vendor = "apple")]
mod inference {
    use super::MODEL_PATH;
    use coreml_rs::{BorrowedTensor, ComputeUnits, Model};

    #[test]
    fn predict_basic_linear() {
        let model = Model::load(MODEL_PATH, ComputeUnits::All)
            .expect("failed to load test model");

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = BorrowedTensor::from_f32(&input_data, &[1, 4]).unwrap();
        let prediction = model.predict(&[("input", &tensor)]).unwrap();
        let (output, shape) = prediction.get_f32("output").unwrap();

        assert_eq!(shape, vec![1, 4]);
        let expected = vec![3.0f32, 5.0, 7.0, 9.0];
        for (got, want) in output.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 0.01,
                "output mismatch: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn predict_zeros() {
        let model = Model::load(MODEL_PATH, ComputeUnits::All).unwrap();

        let input_data = vec![0.0f32; 4];
        let tensor = BorrowedTensor::from_f32(&input_data, &[1, 4]).unwrap();
        let prediction = model.predict(&[("input", &tensor)]).unwrap();
        let (output, _) = prediction.get_f32("output").unwrap();

        // 2*0 + 1 = 1
        for val in &output {
            assert!(
                (val - 1.0).abs() < 0.01,
                "expected ~1.0 for zero input, got {val}"
            );
        }
    }

    #[test]
    fn predict_negative_values() {
        let model = Model::load(MODEL_PATH, ComputeUnits::All).unwrap();

        let input_data = vec![-1.0f32, -0.5, 0.0, 0.5];
        let tensor = BorrowedTensor::from_f32(&input_data, &[1, 4]).unwrap();
        let prediction = model.predict(&[("input", &tensor)]).unwrap();
        let (output, _) = prediction.get_f32("output").unwrap();

        // 2*(-1)+1=-1, 2*(-0.5)+1=0, 2*(0)+1=1, 2*(0.5)+1=2
        let expected = vec![-1.0f32, 0.0, 1.0, 2.0];
        for (got, want) in output.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 0.05,
                "output mismatch: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn predict_large_values() {
        let model = Model::load(MODEL_PATH, ComputeUnits::All).unwrap();

        let input_data = vec![100.0f32, 500.0, -200.0, 0.001];
        let tensor = BorrowedTensor::from_f32(&input_data, &[1, 4]).unwrap();
        let prediction = model.predict(&[("input", &tensor)]).unwrap();
        let (output, _) = prediction.get_f32("output").unwrap();

        let expected = vec![201.0f32, 1001.0, -399.0, 1.002];
        for (got, want) in output.iter().zip(expected.iter()) {
            let tol = want.abs() * 0.01 + 0.1; // relative + absolute tolerance for f16
            assert!(
                (got - want).abs() < tol,
                "output mismatch: got {got}, expected {want} (tol={tol})"
            );
        }
    }

    #[test]
    fn predict_multiple_times_same_model() {
        let model = Model::load(MODEL_PATH, ComputeUnits::All).unwrap();

        for i in 0..10 {
            let val = i as f32;
            let input_data = vec![val; 4];
            let tensor = BorrowedTensor::from_f32(&input_data, &[1, 4]).unwrap();
            let prediction = model.predict(&[("input", &tensor)]).unwrap();
            let (output, _) = prediction.get_f32("output").unwrap();

            let expected = 2.0 * val + 1.0;
            for got in &output {
                assert!(
                    (got - expected).abs() < expected.abs() * 0.01 + 0.1,
                    "iteration {i}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn predict_gpu_ane_accelerated() {
        let model = Model::load(MODEL_PATH, ComputeUnits::All).unwrap();

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = BorrowedTensor::from_f32(&input_data, &[1, 4]).unwrap();
        let prediction = model.predict(&[("input", &tensor)]).unwrap();
        let (output, _) = prediction.get_f32("output").unwrap();

        let expected = vec![3.0f32, 5.0, 7.0, 9.0];
        for (got, want) in output.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 0.01);
        }
    }
}

#[cfg(target_vendor = "apple")]
mod introspection {
    use super::MODEL_PATH;
    use coreml_rs::{ComputeUnits, DataType, FeatureType, Model};

    #[test]
    fn input_descriptions() {
        let model = Model::load(MODEL_PATH, ComputeUnits::All).unwrap();

        let inputs = model.inputs();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].name(), "input");
        assert_eq!(inputs[0].feature_type(), &FeatureType::MultiArray);
        assert_eq!(inputs[0].shape(), Some(&[1, 4][..]));
        assert!(inputs[0].data_type().is_some());
        assert!(!inputs[0].is_optional());
    }

    #[test]
    fn output_descriptions() {
        let model = Model::load(MODEL_PATH, ComputeUnits::All).unwrap();

        let outputs = model.outputs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name(), "output");
        assert_eq!(outputs[0].feature_type(), &FeatureType::MultiArray);
        assert_eq!(outputs[0].shape(), Some(&[1, 4][..]));
    }

    #[test]
    fn input_data_type_is_float() {
        let model = Model::load(MODEL_PATH, ComputeUnits::All).unwrap();

        let inputs = model.inputs();
        let dt = inputs[0].data_type().unwrap();
        assert!(
            dt == DataType::Float16 || dt == DataType::Float32,
            "expected Float16 or Float32, got {:?}",
            dt
        );
    }

    #[test]
    fn metadata_is_accessible() {
        let model = Model::load(MODEL_PATH, ComputeUnits::All).unwrap();
        let meta = model.metadata();
        // Metadata exists (may be empty strings for test model)
        let _ = meta.author;
        let _ = meta.description;
        let _ = meta.version;
        let _ = meta.license;
    }
}

#[cfg(target_vendor = "apple")]
mod error_handling {
    use coreml_rs::{BorrowedTensor, ComputeUnits, ErrorKind, Model};

    #[test]
    fn load_invalid_path() {
        let err = Model::load("/nonexistent/model.mlmodelc", ComputeUnits::All).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::ModelLoad);
        assert!(!err.message().is_empty());
    }

    #[test]
    fn predict_wrong_output_name() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/test_linear.mlmodelc");
        let model = Model::load(model_path, ComputeUnits::All).unwrap();

        let input_data = vec![1.0f32; 4];
        let tensor = BorrowedTensor::from_f32(&input_data, &[1, 4]).unwrap();
        let prediction = model.predict(&[("input", &tensor)]).unwrap();

        let err = prediction.get_f32("nonexistent_output").unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::Prediction);
        assert!(err.message().contains("not found"));
    }

    #[test]
    fn tensor_shape_mismatch() {
        let data = vec![1.0f32; 5];
        let err = BorrowedTensor::from_f32(&data, &[1, 4]).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::InvalidShape);
    }

    #[test]
    fn tensor_zero_dimension() {
        let data: Vec<f32> = vec![];
        let err = BorrowedTensor::from_f32(&data, &[0, 4]).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::InvalidShape);
    }

    #[test]
    fn tensor_empty_shape() {
        let data = vec![1.0f32];
        let err = BorrowedTensor::from_f32(&data, &[]).unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::InvalidShape);
    }
}
