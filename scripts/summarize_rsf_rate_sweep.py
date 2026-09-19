#!/usr/bin/env python3
"""Collect per-run RSF-rate diagnostics without requiring every run to finish."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/TS0304_TS0319_rsf_rate_sweep_summary.csv"


def main() -> int:
    rows = []
    for number in range(304, 320):
        run_id = f"TS{number:04d}"
        path = ROOT / "runs" / run_id / "stats/rsf_rate_sweep_metrics.json"
        if not path.is_file():
            rows.append({"run_id": run_id, "status": "missing_metrics"})
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        fit = metrics["rupture_speed_1dc"]
        rows.append({
            "run_id": run_id,
            "status": "complete",
            "shear_ramp_time_ms": metrics["shear_ramp_time_ms"],
            "peak_prescribed_speed_mm_s": metrics["peak_prescribed_speed_mm_s"],
            "rupture_speed_1dc_m_s": fit.get("speed_m_per_s"),
            "rupture_fit_r_squared": fit.get("r_squared"),
            "nucleation_time_in_shear_ms": metrics["nucleation_time_in_shear_ms"],
            "loading_stop_time_in_shear_ms": metrics["loading_stop_time_in_shear_ms"],
            "loading_stop_displacement_mm": metrics["loading_stop_displacement_mm"],
            "boundary_work_after_nucleation_until_stop": metrics[
                "boundary_work_after_nucleation_until_stop"
            ],
            "vw_station_fraction_reaching_200_mm_s": metrics[
                "vw_station_fraction_reaching_200_mm_s"
            ],
        })
    columns = [
        "run_id",
        "status",
        "shear_ramp_time_ms",
        "peak_prescribed_speed_mm_s",
        "rupture_speed_1dc_m_s",
        "rupture_fit_r_squared",
        "nucleation_time_in_shear_ms",
        "loading_stop_time_in_shear_ms",
        "loading_stop_displacement_mm",
        "boundary_work_after_nucleation_until_stop",
        "vw_station_fraction_reaching_200_mm_s",
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
