from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from analysis_orchestration import run_subprocess_task


SRC_DIR = Path(__file__).resolve().parent
DEFAULT_STATIONS = [160.0, 240.0, 320.0, 400.0]
DEFAULT_DISTANCES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the extended run-0114 analysis suite for one dump."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fit-start", type=float, default=120.0)
    parser.add_argument("--fit-end", type=float, default=440.0)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Skip analysis tasks whose complete output set already exists.",
    )
    return parser.parse_args()


def contiguous_rupture_interval(
    contact_y: np.ndarray,
    cumulative_slip: np.ndarray,
    critical_slip: np.ndarray,
    *,
    fraction: float,
) -> tuple[float, float] | None:
    if fraction <= 0.0:
        raise ValueError("fraction must be positive")
    order = np.argsort(contact_y)
    y = np.asarray(contact_y, dtype=np.float64)[order]
    reached = (
        np.asarray(cumulative_slip, dtype=np.float64)[order]
        >= fraction * np.asarray(critical_slip, dtype=np.float64)[order]
    )
    reached_indices = np.flatnonzero(reached)
    if not len(reached_indices):
        return None
    start = int(reached_indices[0])
    missing_after_start = np.flatnonzero(~reached[start:])
    stop = len(y) if not len(missing_after_start) else start + int(missing_after_start[0])
    return float(y[start]), float(y[stop - 1])


def inspect_rupture(data_path: Path) -> dict[str, object]:
    with h5py.File(data_path, "r") as h5:
        friction_law = str(h5.attrs.get("friction_law", "slip-weakening"))
        contact_y = np.asarray(h5["interface/contact_line_y"], dtype=np.float64)
        critical_slip = np.asarray(
            h5["interface/critical_slip_profile"], dtype=np.float64
        )
        final_slip = np.asarray(
            h5["interface/cumulative_slip"][-1], dtype=np.float64
        )

    intervals = {
        f"{fraction:g}_dc": contiguous_rupture_interval(
            contact_y,
            final_slip,
            critical_slip,
            fraction=fraction,
        )
        for fraction in (0.05, 0.5, 1.0)
    }
    endpoints = {
        name: None if interval is None else interval[1]
        for name, interval in intervals.items()
    }
    starts = {
        name: None if interval is None else interval[0]
        for name, interval in intervals.items()
    }
    reached = final_slip >= 0.05 * critical_slip
    return {
        "friction_law": friction_law,
        "definition": (
            "first contiguous reached interval at final saved frame; constrained "
            "end nodes before the first reached station are excluded"
        ),
        "starts_mm": starts,
        "endpoints_mm": endpoints,
        "maximum_reached_y_mm": (
            float(np.max(contact_y[reached])) if np.any(reached) else None
        ),
        "reached_node_fraction_0.05_dc": float(np.mean(reached)),
        "final_max_cumulative_slip_mm": float(np.max(final_slip)),
    }


def resolved_stations(start_mm: float, endpoint_mm: float) -> list[float]:
    span = endpoint_mm - start_mm
    if span <= 5.0:
        raise ValueError(
            f"Ruptured interval {start_mm:.3f}-{endpoint_mm:.3f} mm is too short."
        )
    stations = [
        station
        for station in DEFAULT_STATIONS
        if start_mm < station <= endpoint_mm - 5.0
    ]
    if len(stations) == len(DEFAULT_STATIONS):
        return stations
    margin = min(5.0, 0.12 * span)
    # The slide-ready near-fault figure has a fixed four-panel layout.
    return np.linspace(
        start_mm + margin,
        endpoint_mm - margin,
        num=4,
        dtype=np.float64,
    ).tolist()


def resolved_fit_interval(
    start_mm: float,
    endpoint_mm: float,
    preferred_start: float,
    preferred_end: float,
) -> tuple[float, float]:
    span = endpoint_mm - start_mm
    fit_start = max(start_mm, preferred_start)
    fit_end = min(endpoint_mm, preferred_end)
    if fit_end - fit_start >= max(10.0, 0.35 * span):
        return fit_start, fit_end
    return start_mm + 0.20 * span, start_mm + 0.85 * span


def python_command(script: str, *arguments: object) -> list[str]:
    return [
        sys.executable,
        str(SRC_DIR / script),
        *(str(argument) for argument in arguments),
    ]


def sigma_trace_command(
    *,
    data_path: Path,
    plot_dir: Path,
    stem: str,
    stations: list[float],
    distances: list[float],
    baseline_mode: str,
    time_origin: str,
    time_scale: str = "linear",
    time_start_ms: float | None = None,
    time_end_ms: float | None = None,
    layout_stem: str | None = None,
    dpi: int,
) -> list[str]:
    command = python_command(
        "plot_sigma_xy_probe_traces.py",
        "--input",
        data_path,
        "--output-traces",
        plot_dir / f"{stem}.png",
        "--output-metrics",
        plot_dir / f"{stem}_metrics.json",
        "--y-points",
        *stations,
        "--off-fault-distances",
        *distances,
        "--baseline-mode",
        baseline_mode,
        "--time-origin",
        time_origin,
        "--time-scale",
        time_scale,
        "--dpi",
        dpi,
    )
    if layout_stem is None:
        command.append("--skip-layout")
    else:
        command.extend(["--output-layout", plot_dir / f"{layout_stem}.png"])
    if time_start_ms is not None:
        command.extend(["--time-start-ms", time_start_ms])
    if time_end_ms is not None:
        command.extend(["--time-end-ms", time_end_ms])
    return [str(value) for value in command]


def output_pair(path: Path) -> tuple[Path, Path]:
    stem = path.with_suffix("") if path.suffix else path
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


def main() -> int:
    args = parse_args()
    data_path = args.input.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    plot_dir = data_path.parent.parent / "Plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rupture = inspect_rupture(data_path)
    endpoint_005 = rupture["endpoints_mm"]["0.05_dc"]
    endpoint_05 = rupture["endpoints_mm"]["0.5_dc"]
    start_005 = rupture["starts_mm"]["0.05_dc"]
    start_05 = rupture["starts_mm"]["0.5_dc"]
    if endpoint_005 is None:
        raise ValueError("The final frame has no contiguous ruptured interval.")
    assert start_005 is not None
    stations = resolved_stations(float(start_005), float(endpoint_005))
    fit_limit = endpoint_05 if endpoint_05 is not None else endpoint_005
    fit_origin = start_05 if start_05 is not None else start_005
    fit_start, fit_end = resolved_fit_interval(
        float(fit_origin),
        float(fit_limit),
        float(args.fit_start),
        float(args.fit_end),
    )

    station_args = ["--stations", *stations]
    y_point_args = ["--y-points", *stations]
    rupture_script = (
        "plot_rsf_rupture_analysis.py"
        if str(rupture["friction_law"]).startswith("rate-state")
        else "plot_rupture_speed_and_fault_profile.py"
    )
    commands: list[tuple[str, list[str], list[Path]]] = [
        (
            "rupture_speed_and_rsf_profiles",
            python_command(
                rupture_script,
                data_path,
                "--output-dir",
                plot_dir,
                "--fit-start",
                fit_start,
                "--fit-end",
                fit_end,
                "--dpi",
                args.dpi,
            ),
            [
                *output_pair(plot_dir / "rupture_speed_stable_fit.png"),
                *output_pair(plot_dir / "fault_interface_profile.png"),
                *output_pair(plot_dir / "rsf_mechanism.png"),
                plot_dir / "rsf_rupture_analysis_metrics.json",
            ],
        ),
        (
            "interface_stress_slip",
            python_command(
                "plot_interface_stress_slip.py",
                data_path,
                "--output",
                plot_dir / "interface_stress_slip_selected_points.png",
                *y_point_args,
                "--dpi",
                args.dpi,
            ),
            list(output_pair(plot_dir / "interface_stress_slip_selected_points.png")),
        ),
        (
            "near_fault_stress_fluctuation",
            python_command(
                "plot_near_fault_stress_fluctuation.py",
                data_path,
                "--output-dir",
                plot_dir,
                *station_args,
                "--fit-start",
                fit_start,
                "--fit-end",
                fit_end,
                "--off-fault-distances",
                *DEFAULT_DISTANCES,
                "--dpi",
                args.dpi,
            ),
            [
                *output_pair(plot_dir / "near_fault_stress_fluctuation_by_station.png"),
                *output_pair(plot_dir / "near_fault_stress_fluctuation_collapse.png"),
                *output_pair(plot_dir / "near_fault_on_fault_triangle_zoom.png"),
                plot_dir / "near_fault_stress_fluctuation_metrics.json",
            ],
        ),
        (
            "on_fault_slip_stress_rates",
            python_command(
                "plot_on_fault_slip_stress_rates.py",
                "--input",
                data_path,
                "--output",
                plot_dir / "on_fault_slip_stress_rates_by_station.png",
                "--output-metrics",
                plot_dir / "on_fault_slip_stress_rates_by_station_metrics.json",
                *y_point_args,
                "--dpi",
                args.dpi,
            ),
            [
                *output_pair(plot_dir / "on_fault_slip_stress_rates_by_station.png"),
                plot_dir / "on_fault_slip_stress_rates_by_station_metrics.json",
            ],
        ),
        (
            "near_fault_delta_tau_time_by_station_5mm",
            sigma_trace_command(
                data_path=data_path,
                plot_dir=plot_dir,
                stem="near_fault_delta_tau_time_by_station_5mm",
                layout_stem="near_fault_delta_tau_probe_layout_5mm",
                stations=stations,
                distances=DEFAULT_DISTANCES,
                baseline_mode="residual",
                time_origin="shear-start",
                dpi=args.dpi,
            ),
            [
                *output_pair(plot_dir / "near_fault_delta_tau_time_by_station_5mm.png"),
                *output_pair(plot_dir / "near_fault_delta_tau_probe_layout_5mm.png"),
                plot_dir / "near_fault_delta_tau_time_by_station_5mm_metrics.json",
            ],
        ),
    ]

    permanent_views = [
        ("", "linear", -0.5, 20.3),
        ("_centered_5ms", "linear", -5.0, 5.0),
        ("_symlog", "symlog", -0.5, 20.3),
    ]
    for distance_label, distances in (
        ("5mm", [5.0]),
        ("0to5mm", DEFAULT_DISTANCES),
    ):
        for suffix, scale, start, end in permanent_views:
            stem = (
                "near_fault_permanent_stress_drop_time_by_station_"
                f"{distance_label}{suffix}"
            )
            layout = (
                f"near_fault_permanent_stress_drop_probe_layout_{distance_label}"
                if suffix == ""
                else None
            )
            expected_outputs = [
                *output_pair(plot_dir / f"{stem}.png"),
                plot_dir / f"{stem}_metrics.json",
            ]
            if layout is not None:
                expected_outputs.extend(output_pair(plot_dir / f"{layout}.png"))
            commands.append(
                (
                    stem,
                    sigma_trace_command(
                        data_path=data_path,
                        plot_dir=plot_dir,
                        stem=stem,
                        layout_stem=layout,
                        stations=stations,
                        distances=distances,
                        baseline_mode="pre-event",
                        time_origin="local-tip",
                        time_scale=scale,
                        time_start_ms=start,
                        time_end_ms=end,
                        dpi=args.dpi,
                    ),
                    expected_outputs,
                )
            )

    task_results = [
        run_subprocess_task(
            name,
            command,
            expected_outputs=expected_outputs,
            cwd=SRC_DIR.parent,
            missing_only=args.missing_only,
        )
        for name, command, expected_outputs in commands
    ]
    failed_tasks = [
        task["name"] for task in task_results if task["status"] == "failed"
    ]

    summary = {
        "input": str(data_path),
        "rupture": rupture,
        "analysis_stations_mm": stations,
        "requested_rupture_speed_fit_interval_mm": [
            float(args.fit_start),
            float(args.fit_end),
        ],
        "rupture_speed_fit_interval_mm": [fit_start, fit_end],
        "status": "complete" if not failed_tasks else "complete_with_failures",
        "commands_total": len(commands),
        "commands_completed": sum(
            task["status"] == "completed" for task in task_results
        ),
        "commands_skipped_existing": sum(
            task["status"] == "skipped_existing" for task in task_results
        ),
        "failed_tasks": failed_tasks,
        "tasks": task_results,
    }
    summary_path = plot_dir / "analysis_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nAnalysis suite complete:\n{json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
