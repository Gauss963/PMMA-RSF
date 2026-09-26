#!/usr/bin/env python3
"""Compare extension energy release and external work after actuator arrest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np

try:
    from .analyze_rsf_nucleation_sweep import analyze as analyze_nucleation
except ImportError:
    from analyze_rsf_nucleation_sweep import analyze as analyze_nucleation


def analyze(run_dir: Path) -> dict:
    result = analyze_nucleation(run_dir)
    case = tomllib.loads(next((run_dir / "input").glob("*.toml")).read_text())
    with h5py.File(run_dir / "data/simulation.h5", "r") as h5:
        group = h5["interface_high_rate"]
        columns = [v.decode() if isinstance(v, bytes) else str(v) for v in group["history_columns"]]
        phase = np.asarray(group["phase_id"])
        history = np.asarray(group["history"], dtype=float)[phase == 2]
    def column(name):
        return history[:, columns.index(name)]
    stopped = np.flatnonzero(column("shear_loading_stopped") > 0.5)
    index = int(stopped[0]) if len(stopped) else None
    extension = column("extension_elastic_energy")
    coordinate = column("normal_loading_coordinate")
    force = column("normal_external_force")
    normal_work = (force[:-1] + force[1:]) * 0.5 * np.diff(coordinate)
    result.update(
        extension_length_mm=case["geometry"]["moving"].get("loading_extension_length", 0.0),
        extension_energy_at_stop=float(extension[index]) if index is not None else None,
        extension_energy_final=float(extension[-1]),
        extension_max_release_after_stop=(float(extension[index] - np.min(extension[index:]))
                                          if index is not None else None),
        normal_external_work_after_stop=(float(np.sum(normal_work[index:])) if index is not None else None),
        energy_units="N mm per mm out-of-plane thickness (2-D model)",
        loading_extension_note="Same PMMA, no external normal load, no contact for y<0; original fault profiles unchanged.",
        calibration_note="All constitutive and loading parameters fixed to TS0343; only loading extension length changes. All cases use log-speed RSF projection, including the new zero-extension control.",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize:
        records = []
        for number in range(352, 368):
            run_id = f"TS{number:04d}"
            path = args.summarize / "runs" / run_id / "stats/rsf_extension_sweep_metrics.json"
            records.append({"run_id": run_id, "metrics": json.loads(path.read_text()) if path.exists() else None})
        out = args.summarize / "stats/TS0352_TS0367_extension_sweep_summary.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    else:
        if args.run_dir is None:
            parser.error("run_dir is required unless --summarize is specified")
        result = analyze(args.run_dir)
        out = args.run_dir / "stats/rsf_extension_sweep_metrics.json"
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
