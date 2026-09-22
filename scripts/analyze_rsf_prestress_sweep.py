#!/usr/bin/env python3
"""Measure realized prestress, early loading stop, and rupture diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tomllib

import h5py
import numpy as np

from scripts.analyze_rsf_rate_sweep import analyze as analyze_rate_case


def analyze(run_dir: Path) -> dict:
    metrics = analyze_rate_case(run_dir)
    case_path = next((run_dir / "input").glob("*.toml"))
    case = tomllib.loads(case_path.read_text(encoding="utf-8"))
    loading = case["loading"]
    rsf = case["rsf"]
    prestress_displacement = loading["prestress_shear_displacement"]
    dynamic_increment = (
        loading["shear_displacement_final"] - prestress_displacement
    )
    peak_friction = rsf["initial_friction"] + rsf["middle"]["a"] * math.log(
        rsf["dynamic_calibration_velocity"] / rsf["initial_steady_velocity"]
    )
    residual_friction = rsf["target_middle_dynamic_friction"]

    h5_path = run_dir / "data/simulation.h5"
    with h5py.File(h5_path, "r") as h5:
        handoff = {
            name: float(h5.attrs[f"rsf_state_handoff_{name}"])
            for name in (
                "seed_velocity",
                "mean_shear_traction",
                "mean_normal_traction",
                "mean_friction_coefficient",
                "mean_state",
                "min_state",
                "max_state",
                "active_fraction",
                "state_saturation_fraction",
                "mean_strength_relative_error",
                "max_strength_relative_error",
            )
        }
        high = h5["interface_high_rate"]
        columns = [value.decode() for value in high["history_columns"][:]]
        history = np.asarray(high["history"][:], dtype=np.float64)
        phase = np.asarray(high["phase_id"][:])
        y_mm = np.asarray(high["contact_line_y"][:], dtype=np.float64)
        slip_rate = np.asarray(high["slip_rate"][:], dtype=np.float64)

    mean_mu = handoff["mean_shear_traction"] / handoff["mean_normal_traction"]
    realized_pi = (mean_mu - residual_friction) / (
        peak_friction - residual_friction
    )
    shear_rows = np.flatnonzero(phase == 2)
    nucleus = (y_mm >= loading["stop_min_y"]) & (
        y_mm <= loading["stop_max_y"]
    )
    trigger_rows = shear_rows[
        np.any(
            slip_rate[np.ix_(shear_rows, np.flatnonzero(nucleus))]
            >= loading["stop_velocity"],
            axis=1,
        )
    ]
    stopped_rows = shear_rows[
        history[shear_rows, columns.index("shear_loading_stopped")] > 0.5
    ]
    trigger_row = int(trigger_rows[0]) if trigger_rows.size else None
    stopped_row = int(stopped_rows[0]) if stopped_rows.size else None

    target_pi = rsf["target_normalized_prestress"]
    metrics.update(
        {
            "sweep_parameter": "normalized prestress Pi",
            "normalized_prestress_definition": "(tau0/sigma_n - mu_residual) / (mu_peak - mu_residual)",
            "target_normalized_prestress": target_pi,
            "realized_normalized_prestress": float(realized_pi),
            "normalized_prestress_error": float(realized_pi - target_pi),
            "peak_friction_frozen_state": float(peak_friction),
            "residual_friction": float(residual_friction),
            "prestress_shear_displacement_mm": float(prestress_displacement),
            "dynamic_shear_increment_mm": float(dynamic_increment),
            "peak_prescribed_speed_mm_s": float(
                math.pi
                * dynamic_increment
                / (2.0 * loading["shear_ramp_time"])
            ),
            "rsf_state_handoff": handoff,
            "first_saved_500_mm_s_row": trigger_row,
            "first_saved_loading_stopped_row": stopped_row,
            "loading_stop_lag_high_rate_frames": (
                None
                if trigger_row is None or stopped_row is None
                else int(stopped_row - trigger_row)
            ),
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    metrics = analyze(args.run_dir)
    output = args.run_dir / "stats/rsf_prestress_sweep_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
