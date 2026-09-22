#!/usr/bin/env python3
"""Collect TS0320-TS0335 prestress diagnostics without requiring all runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/TS0320_TS0335_rsf_prestress_sweep_summary.csv"


def main() -> int:
    rows = []
    for number in range(320, 336):
        run_id = f"TS{number:04d}"
        path = ROOT / "runs" / run_id / "stats/rsf_prestress_sweep_metrics.json"
        if not path.is_file():
            rows.append({"run_id": run_id, "status": "missing_metrics"})
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        fit = metrics["rupture_speed_1dc"]
        handoff = metrics["rsf_state_handoff"]
        rows.append(
            {
                "run_id": run_id,
                "status": "complete",
                "target_pi": metrics["target_normalized_prestress"],
                "realized_pi": metrics["realized_normalized_prestress"],
                "pi_error": metrics["normalized_prestress_error"],
                "prestress_displacement_mm": metrics[
                    "prestress_shear_displacement_mm"
                ],
                "handoff_tau_mpa": handoff["mean_shear_traction"],
                "handoff_sigma_n_mpa": handoff["mean_normal_traction"],
                "handoff_state_min_s": handoff["min_state"],
                "handoff_state_max_s": handoff["max_state"],
                "handoff_state_saturation_fraction": handoff[
                    "state_saturation_fraction"
                ],
                "handoff_mean_strength_relative_error": handoff[
                    "mean_strength_relative_error"
                ],
                "loading_stop_time_ms": metrics[
                    "loading_stop_time_in_shear_ms"
                ],
                "loading_stop_displacement_mm": metrics[
                    "loading_stop_displacement_mm"
                ],
                "stop_lag_frames": metrics[
                    "loading_stop_lag_high_rate_frames"
                ],
                "rupture_speed_1dc_m_s": fit.get("speed_m_per_s"),
                "rupture_fit_r_squared": fit.get("r_squared"),
                "boundary_work_after_nucleation_until_stop": metrics[
                    "boundary_work_after_nucleation_until_stop"
                ],
            }
        )
    columns = [
        "run_id",
        "status",
        "target_pi",
        "realized_pi",
        "pi_error",
        "prestress_displacement_mm",
        "handoff_tau_mpa",
        "handoff_sigma_n_mpa",
        "handoff_state_min_s",
        "handoff_state_max_s",
        "handoff_state_saturation_fraction",
        "handoff_mean_strength_relative_error",
        "loading_stop_time_ms",
        "loading_stop_displacement_mm",
        "stop_lag_frames",
        "rupture_speed_1dc_m_s",
        "rupture_fit_r_squared",
        "boundary_work_after_nucleation_until_stop",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
