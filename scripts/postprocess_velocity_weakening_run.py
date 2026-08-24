from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from plot_contact_friction_map import plot_mu_eff_maps
from plot_contact_mu_disp import plot_contact_mu_disp
from tatva.pmma.dynamics import save_history_plots


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DEFAULT_Y_POINTS = [125.0, 250.0, 375.0, 450.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the complete analysis bundle for an existing PMMA dump."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--fit-start", type=float, default=120.0)
    parser.add_argument("--fit-end", type=float, default=440.0)
    parser.add_argument("--y-points", type=float, nargs="*", default=DEFAULT_Y_POINTS)
    return parser.parse_args()


def _read_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "stats" / "summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return dict(payload.get("summary", payload))


def _history_result(input_path: Path, run_dir: Path) -> dict[str, Any]:
    summary = _read_summary(run_dir)
    with h5py.File(input_path, "r") as h5:
        history = np.asarray(h5["history"], dtype=np.float64)
        columns = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in h5["history_columns"][:]
        ]
        summary.setdefault(
            "shear_loading_mode",
            str(h5.attrs.get("shear_loading_mode", "stress")),
        )
    return {"history": history, "columns": columns, "summary": summary}


def _run(command: list[str]) -> None:
    print(f"\nRunning:\n{shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    run_dir = input_path.parent.parent
    plot_dir = run_dir / "Plot"
    stats_dir = run_dir / "stats"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    history_paths = save_history_plots(
        _history_result(input_path, run_dir),
        plot_dir,
        prefix="simulation",
        extension=".pdf",
    )
    mu_maps = plot_mu_eff_maps(
        input_path,
        plot_dir / "mu_eff_map.pdf",
        plot_dir / "mu_eff_map_phase_split.pdf",
    )
    mu_disp = plot_contact_mu_disp(
        input_path,
        plot_dir / "contact_mu_disp.pdf",
        selection="max-final-slip",
        y_points=list(args.y_points),
    )

    dense_stem = "near_fault_peak_to_peak_amplitude_along_fault_dense"
    _run(
        [
            sys.executable,
            str(SRC_DIR / "plot_dense_near_fault_peak_to_peak.py"),
            "--input",
            str(input_path),
            "--output",
            str(plot_dir / f"{dense_stem}.pdf"),
            "--stats-dir",
            str(stats_dir),
            "--pdf-only",
            "--dpi",
            str(args.dpi),
        ]
    )
    _run(
        [
            sys.executable,
            str(SRC_DIR / "run_velocity_weakening_analysis_suite.py"),
            "--input",
            str(input_path),
            "--fit-start",
            str(args.fit_start),
            "--fit-end",
            str(args.fit_end),
            "--dpi",
            str(args.dpi),
        ]
    )

    result = {
        "input": str(input_path),
        "history_plots": [str(path) for path in history_paths],
        "mu_maps": mu_maps,
        "mu_disp": mu_disp,
        "dense_peak_to_peak": {
            "plot": str(plot_dir / f"{dense_stem}.pdf"),
            "csv": str(stats_dir / f"{dense_stem}.csv"),
            "json": str(stats_dir / f"{dense_stem}.json"),
        },
        "analysis_suite": str(plot_dir / "analysis_suite_summary.json"),
    }
    output_path = stats_dir / "postprocess_summary.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nPost-processing complete:\n{json.dumps(result, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
