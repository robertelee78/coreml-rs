#!/usr/bin/env python3
"""Create a minimal CoreML test model: y = 2*x + 1 (single f32 input/output).

Requires: pip install coremltools torch
Outputs: tests/fixtures/test_linear.mlmodelc/
"""

import sys
import os
import shutil
import torch
import coremltools as ct

class LinearModel(torch.nn.Module):
    def forward(self, x):
        return 2.0 * x + 1.0

model = LinearModel()
model.eval()

example_input = torch.randn(1, 4)
traced = torch.jit.trace(model, example_input)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=(1, 4))],
    outputs=[ct.TensorType(name="output")],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS13,
)

# Save as .mlpackage
pkg_path = "/tmp/test_linear.mlpackage"
mlmodel.save(pkg_path)

# Compile to .mlmodelc
import subprocess
out_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(out_dir, exist_ok=True)

# Remove old compiled model if exists
compiled_path = os.path.join(out_dir, "test_linear.mlmodelc")
if os.path.exists(compiled_path):
    shutil.rmtree(compiled_path)

result = subprocess.run(
    ["xcrun", "coremlcompiler", "compile", pkg_path, out_dir],
    capture_output=True, text=True
)

if result.returncode != 0:
    print(f"Compilation failed: {result.stderr}", file=sys.stderr)
    sys.exit(1)

print(f"Test model created at: {compiled_path}")

# Verify
import numpy as np
pred = mlmodel.predict({"input": np.array([[1.0, 2.0, 3.0, 4.0]])})
print(f"Prediction for [1,2,3,4]: {pred['output']}")
print(f"Expected: [3, 5, 7, 9]")
