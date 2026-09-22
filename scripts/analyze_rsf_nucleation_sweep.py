#!/usr/bin/env python3
"""Record continuous-loading RSF diagnostics without rendering on the GPU node."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np

try:
    from .analyze_rsf_rate_sweep import analyze as analyze_rate
except ImportError:
    from analyze_rsf_rate_sweep import analyze as analyze_rate


def arrival_fit(y: np.ndarray, arrivals: np.ndarray, end: float) -> dict:
    mask = (y >= 200.0) & (y < end)
    available = mask & np.isfinite(arrivals) & (arrivals >= 0.0)
    count = int(np.count_nonzero(available))
    result = {
        "available": False,
        "fit_interval_mm": [200.0, float(end)],
        "point_count": count,
        "station_fraction": float(count / max(np.count_nonzero(mask), 1)),
    }
    if count < 20:
        return result
    position, time_ms = y[available], arrivals[available]
    slope, intercept = np.polyfit(position, time_ms, 1)
    variance = float(np.sum((time_ms - np.mean(time_ms)) ** 2))
    result.update(
        slope_ms_per_mm=float(slope),
        r_squared=(float(1 - np.sum((time_ms - slope * position - intercept) ** 2) / variance)
                   if variance > 0.0 else None),
        speed_m_per_s=float(1 / slope) if slope > 0.0 else None,
        available=bool(slope > 0.0 and variance > 0.0),
    )
    return result


def velocity_diagnostics(group: h5py.Group, dt: float, pressure_steps: int,
                         dc: np.ndarray) -> dict:
    """Stream the interface, keeping at most 256 frames resident at a time."""
    y = np.asarray(group["contact_line_y"], dtype=float)
    phase = np.asarray(group["phase_id"])
    steps = np.asarray(group["step_id"], dtype=np.int64)
    absolute_ms = (steps + np.where(phase == 2, pressure_steps, 0)) * dt * 1e3
    thresholds = (100.0, 500.0, 1000.0)
    arrivals = {v: np.full(y.shape, np.nan) for v in thresholds}
    peak = np.zeros(y.shape)
    normal_peak = np.zeros(y.shape)
    largest_state = 0.0
    max_slip_per_step_dc = 0.0
    previous = np.zeros(y.shape)
    previous_ms = 0.0
    for start in range(0, len(steps), 256):
        stop = min(start + 256, len(steps))
        rates = np.abs(np.asarray(group["slip_rate"][start:stop], dtype=float))
        states = np.asarray(group["rsf_state"][start:stop], dtype=float)
        if not np.all(np.isfinite(rates)) or not np.all(np.isfinite(states)):
            raise ValueError("Non-finite interface rate or state in the completed dump.")
        peak = np.maximum(peak, np.max(rates, axis=0))
        normal = phase[start:stop] == 1
        if np.any(normal):
            normal_peak = np.maximum(normal_peak, np.max(rates[normal], axis=0))
        largest_state = max(largest_state, float(np.max(states)))
        max_slip_per_step_dc = max(
            max_slip_per_step_dc, float(np.max(rates * dt / dc[None, :]))
        )
        for threshold, first in arrivals.items():
            crossed = rates >= threshold
            stations = np.flatnonzero(np.isnan(first) & np.any(crossed, axis=0))
            if not len(stations):
                continue
            upper = np.argmax(crossed[:, stations], axis=0)
            lower = np.maximum(upper - 1, 0)
            before = np.where(upper > 0, rates[lower, stations], previous[stations])
            before_ms = np.where(upper > 0, absolute_ms[start + lower], previous_ms)
            after = rates[upper, stations]
            fraction = np.clip((threshold - before) / np.maximum(after - before, 1e-30), 0, 1)
            first[stations] = before_ms + fraction * (absolute_ms[start + upper] - before_ms)
        previous = rates[-1]
        previous_ms = absolute_ms[stop - 1]
    return {
        "y": y, "arrivals": arrivals, "peak": peak, "normal_peak": normal_peak,
        "max_state_s": largest_state,
        "max_saved_V_dt_over_Dc": max_slip_per_step_dc,
    }


def analyze(run_dir: Path) -> dict:
    result = analyze_rate(run_dir)
    case = tomllib.loads(next((run_dir / "input").glob("*.toml")).read_text())
    with h5py.File(run_dir / "data/simulation.h5", "r") as h5:
        high = h5["interface_high_rate"]
        phase = np.asarray(high["phase_id"])
        rows = np.flatnonzero(phase == 2)
        normal = np.flatnonzero(phase == 1)
        dc = np.asarray(h5["interface/rsf_characteristic_slip_profile"], dtype=float)
        diagnostics = velocity_diagnostics(high, float(h5.attrs["dt"]),
                                           int(h5.attrs["pressure_steps"]), dc)
        columns = [v.decode() if isinstance(v, bytes) else str(v) for v in high["history_columns"]]
        history = np.asarray(high["history"], dtype=float)
        normal_state = np.asarray(high["rsf_state"][normal[-1]], dtype=float)
        first_state = np.asarray(high["rsf_state"][rows[0]], dtype=float)
        normal_slip = np.asarray(high["cumulative_slip"][normal[-1]], dtype=float)
        final_slip = np.asarray(high["cumulative_slip"][rows[-1]], dtype=float)
        reinitialized = bool(h5.attrs.get("rsf_state_handoff_reinitialized", 0))
        step = np.asarray(high["step_id"], dtype=np.int64)[rows]
        shear_ms = step * float(h5.attrs["dt"]) * 1e3

    y = diagnostics["y"]
    normal_ms = case["loading"]["normal_phase_time"] * 1e3
    end = case["geometry"]["moving"]["dimensions"][1] - case["rsf"]["leading_length"]
    arrivals = {v: t - normal_ms for v, t in diagnostics["arrivals"].items()}
    stopped = history[rows, columns.index("shear_loading_stopped")] > 0.5
    stop_rows = np.flatnonzero(stopped)
    displacement = history[rows, columns.index("applied_shear_displacement")]
    reaction = history[rows, columns.index("shear_boundary_reaction")]
    work = 0.5 * (reaction[1:] + reaction[:-1]) * np.diff(displacement)
    stop_index = int(stop_rows[0]) if len(stop_rows) else None
    elastic = history[rows, columns.index("elastic_energy")]
    interface = history[rows, columns.index("interface_energy")]
    kinetic = history[rows, columns.index("kinetic_energy")]
    state_jump = np.max(np.abs(np.log(np.maximum(first_state, 1e-38) /
                                      np.maximum(normal_state, 1e-38))))
    result.update({
        "middle_a": case["rsf"]["middle"]["a"],
        "middle_a_over_b": case["rsf"]["middle"]["a"] / case["rsf"]["middle"]["b"],
        "initial_state_mode": case["rsf"]["initial_state_mode"],
        "state_reinitialized_at_handoff": reinitialized,
        "max_abs_log_state_change_across_phase_boundary": float(state_jump),
        "maximum_saved_state_s": diagnostics["max_state_s"],
        "maximum_saved_V_dt_over_Dc": diagnostics["max_saved_V_dt_over_Dc"],
        "normal_phase_max_slip_mm": float(np.max(normal_slip)),
        "normal_phase_max_saved_rate_mm_s": float(np.max(diagnostics["normal_peak"])),
        "normal_phase_fraction_reaching_500_mm_s": float(np.mean(diagnostics["normal_peak"] >= 500)),
        "sampling_source": "interface_high_rate",
        "shear_saved_dt_us": float(np.median(np.diff(shear_ms)) * 1e3),
        "first_crossing_500_fit": arrival_fit(y, arrivals[500.0], end),
        "first_crossing_1000_fit": arrival_fit(y, arrivals[1000.0], end),
        "first_crossing_fit_note": "Absolute first crossings; stations already dynamic during normal loading are excluded. Positive fit slope alone does not prove a self-sustained rupture.",
        "normal_to_shear_state_jump_note": "Difference between adjacent saved frames, without imposing a static friction threshold at zero speed.",
        "stop_time_over_nominal_ramp": (float(shear_ms[stop_index] / (case["loading"]["shear_ramp_time"] * 1e3))
                                        if stop_index is not None else None),
        "post_stop_boundary_work": float(np.sum(work[stop_index:])) if stop_index is not None else None,
        "post_stop_elastic_energy_release": (float((elastic + interface)[stop_index] - np.min((elastic + interface)[stop_index:]))
                                            if stop_index is not None else None),
        "post_stop_peak_kinetic_over_stored_energy": (float(np.max(kinetic[stop_index:]) /
                                                        max(abs(elastic[stop_index] + interface[stop_index]), 1e-30))
                                                    if stop_index is not None else None),
        "post_stop_500_crossing_fraction": (float(np.mean(np.isfinite(arrivals[500.0]) &
                                                            (arrivals[500.0] > shear_ms[stop_index])))
                                           if stop_index is not None else None),
        "final_post_normal_slip_Dc_min": float(np.min((final_slip - normal_slip) / dc)),
        "calibration_note": "b-a and b*Dc are fixed in loading/middle zones; the velocity-step reference energy is fixed, while realized breakdown work must be measured.",
    })
    stats = run_dir / "stats"
    stats.mkdir(exist_ok=True)
    with (stats / "rsf_nucleation_station_arrivals.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["y_mm", "first_100_time_in_shear_ms", "first_500_time_in_shear_ms",
                         "first_1000_time_in_shear_ms", "normal_peak_V_mm_s", "peak_V_mm_s",
                         "normal_end_theta_s", "normal_slip_mm", "post_normal_slip_Dc"])
        for j, position in enumerate(y):
            writer.writerow([position, *[arrivals[v][j] if np.isfinite(arrivals[v][j]) else ""
                                         for v in (100.0, 500.0, 1000.0)],
                             diagnostics["normal_peak"][j], diagnostics["peak"][j],
                             normal_state[j], normal_slip[j], (final_slip[j] - normal_slip[j]) / dc[j]])
    return result


def summarize(root: Path) -> Path:
    records = []
    for number in range(336, 352):
        run_id = f"TS{number:04d}"
        path = root / "runs" / run_id / "stats/rsf_nucleation_sweep_metrics.json"
        if not path.is_file():
            records.append({"run_id": run_id, "status": "missing_metrics"})
            continue
        try:
            record = json.loads(path.read_text())
            records.append({"run_id": run_id, "status": "complete", "metrics": record})
        except (OSError, ValueError) as exc:
            records.append({"run_id": run_id, "status": "unreadable_metrics", "error": str(exc)})
    out = root / "stats/TS0336_TS0351_nucleation_sweep_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--summarize", type=Path, metavar="REPO_ROOT")
    args = parser.parse_args()
    if args.summarize:
        print(summarize(args.summarize))
        return 0
    if args.run_dir is None:
        parser.error("run_dir is required unless --summarize is given")
    result = analyze(args.run_dir)
    out = args.run_dir / "stats/rsf_nucleation_sweep_metrics.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
