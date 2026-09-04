from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis_orchestration import run_callable_task, run_subprocess_task
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
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Skip analysis tasks whose complete output set already exists.",
    )
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

    task_values: dict[str, Any] = {}

    def capture(name: str, action: Callable[[], Any]) -> Callable[[], None]:
        def wrapped() -> None:
            task_values[name] = action()

        return wrapped

    history_paths = [
        plot_dir / "simulation_tractions.pdf",
        plot_dir / "simulation_interface_state.pdf",
        plot_dir / "simulation_energies.pdf",
    ]
    tasks = [
        run_callable_task(
            "history_plots",
            capture(
                "history_plots",
                lambda: save_history_plots(
                    _history_result(input_path, run_dir),
                    plot_dir,
                    prefix="simulation",
                    extension=".pdf",
                ),
            ),
            expected_outputs=history_paths,
            missing_only=args.missing_only,
        )
    ]
    mu_map_paths = [
        plot_dir / "mu_eff_map.pdf",
        plot_dir / "mu_eff_map_phase_split.pdf",
    ]
    tasks.append(
        run_callable_task(
            "mu_maps",
            capture(
                "mu_maps",
                lambda: plot_mu_eff_maps(
                    input_path,
                    mu_map_paths[0],
                    mu_map_paths[1],
                ),
            ),
            expected_outputs=mu_map_paths,
            missing_only=args.missing_only,
        )
    )
    mu_disp_path = plot_dir / "contact_mu_disp.pdf"
    tasks.append(
        run_callable_task(
            "contact_mu_disp",
            capture(
                "contact_mu_disp",
                lambda: plot_contact_mu_disp(
                    input_path,
                    mu_disp_path,
                    selection="max-final-slip",
                    y_points=list(args.y_points),
                ),
            ),
            expected_outputs=[mu_disp_path],
            missing_only=args.missing_only,
        )
    )

    dense_stem = "near_fault_peak_to_peak_amplitude_along_fault_dense"
    dense_paths = [
        plot_dir / f"{dense_stem}.pdf",
        stats_dir / f"{dense_stem}.csv",
        stats_dir / f"{dense_stem}.json",
    ]
    tasks.append(
        run_subprocess_task(
            "dense_near_fault_peak_to_peak",
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
            ],
            expected_outputs=dense_paths,
            cwd=REPO_ROOT,
            missing_only=args.missing_only,
        )
    )
    suite_command = [
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
    if args.missing_only:
        suite_command.append("--missing-only")
    tasks.append(
        run_subprocess_task(
            "extended_analysis_suite",
            suite_command,
            expected_outputs=[],
            cwd=REPO_ROOT,
            missing_only=False,
        )
    )

    suite_summary_path = plot_dir / "analysis_suite_summary.json"
    suite_summary: dict[str, Any] = {}
    if suite_summary_path.is_file():
        suite_summary = json.loads(suite_summary_path.read_text(encoding="utf-8"))

    failed_tasks = [task["name"] for task in tasks if task["status"] == "failed"]
    failed_tasks.extend(
        f"extended_analysis_suite:{name}"
        for name in suite_summary.get("failed_tasks", [])
    )
    result = {
        "status": "complete" if not failed_tasks else "complete_with_failures",
        "input": str(input_path),
        "history_plots": [str(path) for path in history_paths],
        "mu_maps": task_values.get(
            "mu_maps",
            {
                "output": str(mu_map_paths[0]),
                "phase_split_output": str(mu_map_paths[1]),
            },
        ),
        "mu_disp": task_values.get(
            "contact_mu_disp",
            {"output": str(mu_disp_path)},
        ),
        "dense_peak_to_peak": {
            "plot": str(dense_paths[0]),
            "csv": str(dense_paths[1]),
            "json": str(dense_paths[2]),
        },
        "analysis_suite": str(suite_summary_path),
        "analysis_suite_status": suite_summary.get("status", "unavailable"),
        "failed_tasks": failed_tasks,
        "tasks": tasks,
    }
    output_path = stats_dir / "postprocess_summary.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nPost-processing complete:\n{json.dumps(result, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
