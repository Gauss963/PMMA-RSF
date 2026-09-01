#!/usr/bin/env python3
"""Compare serial and MPI PMMA HDF5 dumps with physical-field tolerances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ABSOLUTE_TOLERANCES = {
    "displacement": 1.0e-7,
    "stress": 1.0e-4,
    "history": 1.0e-4,
    "friction_coefficient": 2.0e-3,
    "friction_strength": 1.0e-8,
    "friction_velocity": 1.0e-8,
    "slip_rate": 1.0e-8,
    "plastic_slip": 1.0e-8,
    "cumulative_slip": 1.0e-8,
    "rsf_state": 1.0e-8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("mpi", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def dataset_names(handle: h5py.File) -> list[str]:
    names: list[str] = []
    handle.visititems(
        lambda name, value: names.append(name)
        if isinstance(value, h5py.Dataset)
        else None
    )
    return names


def tolerance_for(name: str) -> float:
    leaf = name.rsplit("/", maxsplit=1)[-1]
    return ABSOLUTE_TOLERANCES.get(leaf, 0.0)


def main() -> int:
    args = parse_args()
    report: dict[str, object] = {"serial": str(args.serial), "mpi": str(args.mpi)}
    failures: list[str] = []
    metrics: list[dict[str, object]] = []
    with h5py.File(args.serial, "r") as serial, h5py.File(args.mpi, "r") as mpi:
        serial_names = dataset_names(serial)
        mpi_names = dataset_names(mpi)
        if serial_names != mpi_names:
            failures.append("Dataset paths differ between serial and MPI dumps.")
        for name in sorted(set(serial_names) & set(mpi_names)):
            serial_value = serial[name][...]
            mpi_value = mpi[name][...]
            if serial_value.shape != mpi_value.shape or serial_value.dtype != mpi_value.dtype:
                failures.append(f"Shape or dtype mismatch: {name}")
                continue
            if serial_value.dtype.kind in "SUO":
                if not np.array_equal(serial_value, mpi_value):
                    failures.append(f"Non-numeric dataset mismatch: {name}")
                continue
            if not np.array_equal(np.isfinite(serial_value), np.isfinite(mpi_value)):
                failures.append(f"Finite-value mask mismatch: {name}")
                continue
            finite = np.isfinite(serial_value)
            difference = np.abs(
                serial_value[finite].astype(np.float64)
                - mpi_value[finite].astype(np.float64)
            )
            maximum = float(difference.max(initial=0.0))
            tolerance = tolerance_for(name)
            metrics.append(
                {
                    "dataset": name,
                    "max_abs_difference": maximum,
                    "absolute_tolerance": tolerance,
                }
            )
            if maximum > tolerance:
                failures.append(
                    f"{name}: max abs difference {maximum:.6e} exceeds {tolerance:.6e}"
                )

        report["serial_mpi_ranks"] = int(serial.attrs["mpi_ranks"])
        report["mpi_ranks"] = int(mpi.attrs["mpi_ranks"])
    report["datasets_compared"] = len(metrics)
    report["metrics"] = metrics
    report["failures"] = failures
    report["passed"] = not failures
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
