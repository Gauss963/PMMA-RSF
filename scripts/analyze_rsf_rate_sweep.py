#!/usr/bin/env python3
"""Measure front arrivals and post-nucleation boundary work for an RSF run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np


def first_crossings(
    slip: np.ndarray, time_ms: np.ndarray, threshold: np.ndarray
) -> np.ndarray:
    crossed = slip >= threshold[None, :]
    available = np.any(crossed, axis=0)
    arrivals = np.full(slip.shape[1], np.nan, dtype=np.float64)
    stations = np.flatnonzero(available)
    upper = np.argmax(crossed[:, available], axis=0)
    lower = np.maximum(upper - 1, 0)
    before = slip[lower, stations]
    after = slip[upper, stations]
    fraction = np.clip(
        (threshold[stations] - before) / np.maximum(after - before, 1.0e-30),
        0.0,
        1.0,
    )
    arrivals[stations] = time_ms[lower] + fraction * (
        time_ms[upper] - time_ms[lower]
    )
    return arrivals


def front_fit(y_mm: np.ndarray, arrivals_ms: np.ndarray, end_mm: float) -> dict:
    smoothed = np.full(arrivals_ms.shape, np.nan)
    for station, position in enumerate(y_mm):
        local = (
            (np.abs(y_mm - position) <= 5.0) & np.isfinite(arrivals_ms)
        )
        if np.any(local):
            smoothed[station] = np.median(arrivals_ms[local])
    fit_mask = (
        (y_mm >= 200.0)
        & (y_mm <= end_mm)
        & np.isfinite(smoothed)
    )
    if np.count_nonzero(fit_mask) < 20:
        return {"available": False, "point_count": int(np.count_nonzero(fit_mask))}
    x = y_mm[fit_mask]
    t = smoothed[fit_mask]
    slope, intercept = np.polyfit(x, t, 1)
    residual = t - (slope * x + intercept)
    variance = np.sum((t - np.mean(t)) ** 2)
    return {
        "available": bool(slope > 0),
        "point_count": int(x.size),
        "fit_interval_mm": [200.0, float(end_mm)],
        "speed_m_per_s": float(1.0 / slope) if slope > 0 else None,
        "slope_ms_per_mm": float(slope),
        "r_squared": float(1.0 - np.sum(residual**2) / variance)
        if variance > 0
        else None,
    }


def analyze(run_dir: Path) -> dict:
    case_path = next((run_dir / "input").glob("*.toml"))
    case = tomllib.loads(case_path.read_text(encoding="utf-8"))
    h5_path = run_dir / "data/simulation.h5"
    with h5py.File(h5_path, "r") as h5:
        high = h5["interface_high_rate"]
        columns = [value.decode() for value in high["history_columns"][:]]
        if "shear_boundary_reaction" not in columns:
            raise ValueError("This run lacks the shear boundary reaction diagnostic.")
        history = np.asarray(high["history"][:], dtype=np.float64)
        phase = np.asarray(high["phase_id"][:])
        steps = np.asarray(high["step_id"][:], dtype=np.int64)
        normal_rows = np.flatnonzero(phase == 1)
        shear_rows = np.flatnonzero(phase == 2)
        if normal_rows.size == 0 or shear_rows.size < 2:
            raise ValueError("Both normal and shear interface histories are required.")
        time_ms = (
            steps[shear_rows] + int(h5.attrs["pressure_steps"])
        ) * float(h5.attrs["dt"]) * 1e3
        shear_time_ms = time_ms - case["loading"]["normal_phase_time"] * 1e3
        y_mm = np.asarray(high["contact_line_y"][:], dtype=np.float64)
        normal_end = np.asarray(high["cumulative_slip"][normal_rows[-1]], dtype=np.float64)
        slip = np.asarray(high["cumulative_slip"][shear_rows], dtype=np.float64)
        slip -= normal_end[None, :]
        critical_slip = np.asarray(
            h5["interface/rsf_characteristic_slip_profile"][:], dtype=np.float64
        )
        nucleus = (y_mm >= 5.0) & (y_mm <= 25.0)
        rate_nucleus = np.asarray(
            high["slip_rate"][shear_rows[0] : shear_rows[-1] + 1, np.flatnonzero(nucleus)],
            dtype=np.float64,
        )
        dynamic_rows = np.flatnonzero(np.any(rate_nucleus >= 500.0, axis=1))
        nucleus_time_ms = (
            float(shear_time_ms[dynamic_rows[0]]) if dynamic_rows.size else None
        )
        vw = (y_mm >= 200.0) & (y_mm < 470.0)
        rate_vw = np.asarray(
            high["slip_rate"][shear_rows[0] : shear_rows[-1] + 1, np.flatnonzero(vw)],
            dtype=np.float64,
        )
        peak_rates = np.max(rate_vw, axis=0)

    arrivals_005 = first_crossings(slip, shear_time_ms, 0.05 * critical_slip)
    arrivals_1 = first_crossings(slip, shear_time_ms, critical_slip)
    fit_end_mm = case["geometry"]["moving"]["dimensions"][1] - case["rsf"]["leading_length"] - 0.5

    displacement = history[:, columns.index("applied_shear_displacement")]
    reaction = history[:, columns.index("shear_boundary_reaction")]
    shear_displacement = displacement[shear_rows]
    shear_reaction = reaction[shear_rows]
    increments = 0.5 * (shear_reaction[:-1] + shear_reaction[1:]) * np.diff(
        shear_displacement
    )
    cumulative_work = np.concatenate(([0.0], np.cumsum(increments)))
    stop_flag = history[shear_rows, columns.index("shear_loading_stopped")] > 0.5
    stop_rows = np.flatnonzero(stop_flag)
    stop_index = int(stop_rows[0]) if stop_rows.size else None
    work_end_index = stop_index if stop_index is not None else len(shear_rows) - 1
    if nucleus_time_ms is None:
        work_after_nucleation = None
    else:
        work_at_nucleation = np.interp(
            nucleus_time_ms, shear_time_ms, cumulative_work
        )
        work_after_nucleation = float(
            cumulative_work[work_end_index] - work_at_nucleation
        )

    return {
        "run_id": run_dir.name,
        "case": case_path.name,
        "dynamic_calibration_velocity_mm_s": case["rsf"]["dynamic_calibration_velocity"],
        "middle_b": case["rsf"]["middle"]["b"],
        "dc_mm": case["rsf"]["middle"]["dc"],
        "shear_ramp_time_ms": case["loading"]["shear_ramp_time"] * 1e3,
        "peak_prescribed_speed_mm_s": float(
            np.pi
            * case["loading"]["shear_displacement_final"]
            / (2.0 * case["loading"]["shear_ramp_time"])
        ),
        "rupture_speed_1dc": front_fit(y_mm, arrivals_1, fit_end_mm),
        "rupture_speed_0p05dc": front_fit(y_mm, arrivals_005, fit_end_mm),
        "nucleation_definition": "first V >= 500 mm/s in y = 5-25 mm after shear starts",
        "nucleation_time_in_shear_ms": nucleus_time_ms,
        "loading_stop_time_in_shear_ms": (
            float(shear_time_ms[stop_index]) if stop_index is not None else None
        ),
        "loading_stop_displacement_mm": (
            float(shear_displacement[stop_index]) if stop_index is not None else None
        ),
        "boundary_work_definition": "trapezoidal integral of imposed-face reaction d(displacement), model energy per unit out-of-plane thickness",
        "boundary_work_after_nucleation_until_stop": work_after_nucleation,
        "boundary_work_total_shear": float(cumulative_work[-1]),
        "peak_local_slip_rate_vw_mm_s": float(np.max(peak_rates)),
        "vw_station_fraction_reaching_200_mm_s": float(np.mean(peak_rates >= 200.0)),
        "vw_station_fraction_reaching_500_mm_s": float(np.mean(peak_rates >= 500.0)),
        "post_shear_1dc_station_fraction": float(np.mean(np.isfinite(arrivals_1))),
        "note": "RSF has no kinetic-friction floor; the 1Dc front is a weakening contour, not automatically an autonomous crack tip.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    metrics = analyze(args.run_dir)
    output = args.run_dir / "stats/rsf_rate_sweep_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
