#!/usr/bin/env python3
"""Check short serial/MPI dumps with scale-aware float32 tolerances."""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def compare(left, right):
    metrics = {}
    with h5py.File(left) as a, h5py.File(right) as b:
        for name in ("history", "moving/displacement", "moving/velocity", "moving/stress",
                     "stationary/displacement", "stationary/stress", "interface/cumulative_slip",
                     "interface/plastic_slip", "interface/friction_coefficient", "interface/slip_rate"):
            x, y = a[name][:], b[name][:]
            if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise AssertionError(f"Shape/non-finite mismatch: {name}")
            # History columns have different units: each has its own scale.
            scale = np.max(np.abs(x), axis=0) if name == "history" else np.max(np.abs(x))
            tolerance = 1e-5 + 3e-4 * scale
            error = np.abs(x.astype(float) - y.astype(float))
            if np.any(error > tolerance):
                raise AssertionError(f"MPI mismatch: {name}, max error={error.max()}, max scaled={np.max(error / tolerance)}")
            metrics[name] = {"max_abs_error": float(error.max()), "max_tolerance_fraction": float(np.max(error / tolerance))}
        for name in ("phase_id", "step_id", "moving/elements", "stationary/elements"):
            np.testing.assert_array_equal(a[name][:], b[name][:])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("mpi", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"passed": True, "datasets": compare(args.serial, args.mpi)}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
