#!/usr/bin/env python3
"""Create a multi-input, multi-output CoreML test model with mixed dtypes.

Model: Takes float_input (Float32 [1,4]) and int_input (Int32 [1,2])
       Outputs: sum_output = float_input (passthrough), count_output = float(int_input)

This tests mixed-dtype inputs and multi-output extraction.
"""

import os
import shutil
import subprocess
import sys

import coremltools as ct
import torch


class MultiIOModel(torch.nn.Module):
    def forward(self, float_in: torch.Tensor, int_in: torch.Tensor):
        # Output 1: 2*float_in + 1 (same as linear model)
        out_a = 2.0 * float_in + 1.0
        # Output 2: int_in cast to float, doubled
        out_b = 2.0 * int_in.float()
        return out_a, out_b


model = MultiIOModel()
model.eval()

example_float = torch.randn(1, 4)
example_int = torch.randint(0, 10, (1, 2)).float()  # trace needs float

traced = torch.jit.trace(model, (example_float, example_int))

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="float_input", shape=(1, 4), dtype=ct.converters.mil.mil.types.fp32),
        ct.TensorType(name="int_input", shape=(1, 2), dtype=ct.converters.mil.mil.types.fp32),
    ],
    outputs=[
        ct.TensorType(name="sum_output"),
        ct.TensorType(name="count_output"),
    ],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS13,
)

pkg_path = "/tmp/test_multi_io.mlpackage"
mlmodel.save(pkg_path)

out_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(out_dir, exist_ok=True)

compiled_path = os.path.join(out_dir, "test_multi_io.mlmodelc")
if os.path.exists(compiled_path):
    shutil.rmtree(compiled_path)

result = subprocess.run(
    ["xcrun", "coremlcompiler", "compile", pkg_path, out_dir],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print(f"Compilation failed: {result.stderr}", file=sys.stderr)
    sys.exit(1)

print(f"Multi-IO model created at: {compiled_path}")

import numpy as np
pred = mlmodel.predict({
    "float_input": np.array([[1.0, 2.0, 3.0, 4.0]]),
    "int_input": np.array([[5.0, 10.0]]),
})
print(f"sum_output for [1,2,3,4]: {pred['sum_output']}")
print(f"count_output for [5,10]: {pred['count_output']}")
