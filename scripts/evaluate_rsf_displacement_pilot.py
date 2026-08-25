#!/usr/bin/env python3
"""Evaluate rupture coverage and loading-induced kinetic energy for an RSF pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Pilot simulation HDF5 dump.")
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=10.0,
        help="Low slip-rate threshold used for rupture coverage [mm/s].",
    )
    parser.add_argument(
        "--dynamic-velocity-threshold",
        type=float,
        default=1000.0,
        help=(
            "Slip-rate threshold used to measure the dynamic rupture-front "
            "direction [mm/s]."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Metrics JSON path (default: run stats/rupture_pilot_metrics.json).",
    )
    return parser.parse_args()


def _columns(dataset: h5py.Dataset) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in np.asarray(dataset)
    ]


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _front_metrics(
    station_y: np.ndarray,
    first_reached: np.ndarray,
    frame_time: np.ndarray,
    shear_start_time: float,
) -> dict[str, object]:
    """Measure whether first dynamic arrivals form one forward-moving front."""
    reached = first_reached >= 0
    arrival_ms = np.full(station_y.shape, np.nan, dtype=np.float64)
    arrival_ms[reached] = 1.0e3 * (
        frame_time[first_reached[reached]] - shear_start_time
    )
    if not np.any(reached):
        return {
            "nucleation_y_min_mm": None,
            "nucleation_y_max_mm": None,
            "forward_step_fraction": None,
            "backward_step_count": 0,
            "largest_backward_step_ms": None,
            "largest_backward_step_from_y_mm": None,
            "largest_backward_step_to_y_mm": None,
            "largest_forward_stall_step_ms": None,
            "largest_forward_stall_from_y_mm": None,
            "largest_forward_stall_to_y_mm": None,
            "rupture_duration_ms": None,
        }

    earliest_frame = int(np.min(first_reached[reached]))
    nucleation = first_reached == earliest_frame
    adjacent = reached[:-1] & reached[1:]
    adjacent_left_indices = np.flatnonzero(adjacent)
    steps = np.diff(arrival_ms)[adjacent]
    positive_frame_intervals = np.diff(frame_time)
    positive_frame_intervals = positive_frame_intervals[positive_frame_intervals > 0.0]
    sampling_tolerance_ms = (
        0.5e3 * float(np.median(positive_frame_intervals))
        if positive_frame_intervals.size
        else 0.0
    )
    backward = steps < -sampling_tolerance_ms
    reached_arrivals = arrival_ms[reached]
    largest_backward_index = int(np.argmin(steps)) if steps.size else None
    largest_stall_index = int(np.argmax(steps)) if steps.size else None
    return {
        "nucleation_y_min_mm": float(np.min(station_y[nucleation])),
        "nucleation_y_max_mm": float(np.max(station_y[nucleation])),
        "sampling_tolerance_ms": sampling_tolerance_ms,
        "forward_step_fraction": (
            float(np.mean(~backward)) if steps.size else None
        ),
        "backward_step_count": int(np.count_nonzero(backward)),
        "largest_backward_step_ms": (
            float(np.min(steps)) if steps.size else None
        ),
        "largest_backward_step_from_y_mm": (
            float(station_y[adjacent_left_indices[largest_backward_index]])
            if largest_backward_index is not None
            else None
        ),
        "largest_backward_step_to_y_mm": (
            float(station_y[adjacent_left_indices[largest_backward_index] + 1])
            if largest_backward_index is not None
            else None
        ),
        "largest_forward_stall_step_ms": (
            float(np.max(steps)) if steps.size else None
        ),
        "largest_forward_stall_from_y_mm": (
            float(station_y[adjacent_left_indices[largest_stall_index]])
            if largest_stall_index is not None
            else None
        ),
        "largest_forward_stall_to_y_mm": (
            float(station_y[adjacent_left_indices[largest_stall_index] + 1])
            if largest_stall_index is not None
            else None
        ),
        "rupture_duration_ms": float(
            np.max(reached_arrivals) - np.min(reached_arrivals)
        ),
    }


def _record_first_crossings(
    block: np.ndarray,
    block_rows: np.ndarray,
    threshold: float,
    first_reached: np.ndarray,
) -> None:
    """Update first-arrival rows without retaining the full velocity history."""
    unreached = first_reached < 0
    if not np.any(unreached):
        return
    hits = block[:, unreached] >= threshold
    hit_columns = np.any(hits, axis=0)
    if not np.any(hit_columns):
        return
    local_first = np.argmax(hits[:, hit_columns], axis=0)
    target_columns = np.flatnonzero(unreached)[hit_columns]
    first_reached[target_columns] = block_rows[local_first]


def _run_metadata(data_path: Path) -> tuple[float, float, float, str]:
    run_dir = data_path.parent.parent
    summary_path = run_dir / "stats" / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
        return (
            float(payload["mesh_size"]),
            float(payload["shear_displacement_s"]),
            float(payload["shear_ramp_time"]),
            "summary.json",
        )

    resolved_path = run_dir / "input" / "resolved_case.json"
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Neither {summary_path} nor {resolved_path} is available."
        )
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    return (
        float(resolved["numerics"]["mesh_size"]),
        float(resolved["loading"]["shear_displacement_final"]),
        float(resolved["loading"]["shear_ramp_time"]),
        "resolved_case.json",
    )


def _read_valid_history(
    dataset: h5py.Dataset,
    *,
    time_column: int,
) -> tuple[np.ndarray, bool]:
    chunk_rows = dataset.chunks[0] if dataset.chunks else 2048
    blocks: list[np.ndarray] = []
    stopped_at_corrupt_chunk = False
    for start in range(0, dataset.shape[0], chunk_rows):
        stop = min(start + chunk_rows, dataset.shape[0])
        try:
            block = np.asarray(dataset[start:stop], dtype=np.float64)
        except OSError:
            stopped_at_corrupt_chunk = True
            break
        positive = np.flatnonzero(block[:, time_column] > 0.0)
        if not positive.size:
            break
        last_written = int(positive[-1]) + 1
        blocks.append(block[:last_written])
        if last_written < len(block):
            break
    if not blocks:
        raise ValueError("No readable, completed time frames were found.")
    return np.concatenate(blocks, axis=0), stopped_at_corrupt_chunk


def evaluate(
    data_path: Path,
    velocity_threshold: float,
    dynamic_velocity_threshold: float = 1000.0,
) -> dict[str, object]:
    data_path = data_path.expanduser().resolve()
    mesh_size, configured_displacement, ramp_time, metadata_source = _run_metadata(
        data_path
    )

    with h5py.File(data_path, "r") as h5:
        group_name = "interface_high_rate" if "interface_high_rate" in h5 else "interface"
        group = h5[group_name]
        y = np.asarray(group["contact_line_y"], dtype=np.float64)
        columns = _columns(group["history_columns"])
        col = {name: index for index, name in enumerate(columns)}
        required = {
            "time",
            "elastic_energy",
            "interface_energy",
            "kinetic_energy",
            "applied_shear_displacement",
            "shear_loading_stopped",
        }
        missing = sorted(required.difference(col))
        if missing:
            raise ValueError(f"History is missing required columns: {missing}")

        history, corrupt_tail_ignored = _read_valid_history(
            group["history"], time_column=col["time"]
        )
        last_valid_index = len(history) - 1
        phase_id = np.asarray(
            group["phase_id"][: len(history)], dtype=np.int32
        )
        shear_phase_id = int(np.max(phase_id))
        shear_phase = phase_id == shear_phase_id
        shear_indices = np.flatnonzero(shear_phase)
        if not shear_indices.size:
            raise ValueError("No explicit shear-phase frames were found.")
        shear_start_index = int(shear_indices[0])
        shear_start_time = float(history[shear_start_index, col["time"]])
        interior = (y >= mesh_size) & (y <= 500.0 - mesh_size)
        interior_y = y[interior]
        peak_speed = np.zeros(interior_y.size, dtype=np.float64)
        first_reached = np.full(interior_y.size, -1, dtype=np.int64)
        first_dynamic_reached = np.full(interior_y.size, -1, dtype=np.int64)

        slip_rate = group["slip_rate"]
        chunk_rows = 2048
        for start in range(shear_start_index, last_valid_index + 1, chunk_rows):
            stop = min(start + chunk_rows, last_valid_index + 1)
            row_mask = shear_phase[start:stop]
            if not np.any(row_mask):
                continue
            block = np.abs(
                np.asarray(slip_rate[start:stop, :], dtype=np.float64)[:, interior]
            )[row_mask]
            block_rows = np.arange(start, stop, dtype=np.int64)[row_mask]
            peak_speed = np.maximum(peak_speed, np.max(block, axis=0))
            _record_first_crossings(
                block, block_rows, velocity_threshold, first_reached
            )
            _record_first_crossings(
                block,
                block_rows,
                dynamic_velocity_threshold,
                first_dynamic_reached,
            )

        speed_reached = first_reached >= 0
        dynamic_speed_reached = first_dynamic_reached >= 0
        final_slip = np.asarray(
            group["cumulative_slip"][last_valid_index], dtype=np.float64
        )
        dc = np.asarray(
            h5["interface/rsf_characteristic_slip_profile"], dtype=np.float64
        )
        slip_reached = final_slip[interior] >= dc[interior]
        full_rupture = bool(np.all(speed_reached) and np.all(slip_reached))
        reached_indices = first_reached[speed_reached]
        onset_index = int(np.min(reached_indices)) if reached_indices.size else None
        completion_index = (
            int(np.max(first_reached)) if np.all(speed_reached) else None
        )
        dynamic_reached_indices = first_dynamic_reached[dynamic_speed_reached]
        dynamic_onset_index = (
            int(np.min(dynamic_reached_indices))
            if dynamic_reached_indices.size
            else None
        )
        dynamic_completion_index = (
            int(np.max(first_dynamic_reached))
            if np.all(dynamic_speed_reached)
            else None
        )

    elastic = np.abs(history[:, col["elastic_energy"]])
    interface = np.abs(history[:, col["interface_energy"]])
    kinetic = np.maximum(history[:, col["kinetic_energy"]], 0.0)
    stored = np.maximum(elastic + interface, np.finfo(np.float64).tiny)
    kinetic_ratio = kinetic / stored
    pre_onset_stop = len(history) if onset_index is None else onset_index + 1
    pre_onset = shear_phase.copy()
    pre_onset[pre_onset_stop:] = False
    pre_dynamic_onset_stop = (
        len(history) if dynamic_onset_index is None else dynamic_onset_index + 1
    )
    pre_dynamic_onset = shear_phase.copy()
    pre_dynamic_onset[pre_dynamic_onset_stop:] = False

    stopped_indices = np.flatnonzero(
        history[:, col["shear_loading_stopped"]] > 0.5
    )
    stop_index = int(stopped_indices[0]) if stopped_indices.size else None
    applied_displacement = history[:, col["applied_shear_displacement"]]

    reached_y = interior_y[speed_reached]
    weakened_y = interior_y[slip_reached]
    front_metrics = _front_metrics(
        interior_y,
        first_reached,
        history[:, col["time"]],
        shear_start_time,
    )
    dynamic_front_metrics = _front_metrics(
        interior_y,
        first_dynamic_reached,
        history[:, col["time"]],
        shear_start_time,
    )
    metrics: dict[str, object] = {
        "data_path": str(data_path),
        "metadata_source": metadata_source,
        "corrupt_tail_ignored": corrupt_tail_ignored,
        "valid_frame_count": last_valid_index + 1,
        "last_valid_time_ms": 1.0e3 * float(history[last_valid_index, col["time"]]),
        "mesh_size_mm": mesh_size,
        "configured_shear_displacement_mm": configured_displacement,
        "configured_ramp_time_ms": 1.0e3 * ramp_time,
        "velocity_threshold_mm_s": velocity_threshold,
        "dynamic_velocity_threshold_mm_s": dynamic_velocity_threshold,
        "full_rupture": full_rupture,
        "interior_station_count": int(np.count_nonzero(interior)),
        "velocity_coverage_fraction": float(np.mean(speed_reached)),
        "dynamic_velocity_coverage_fraction": float(
            np.mean(dynamic_speed_reached)
        ),
        "dc_slip_coverage_fraction": float(np.mean(slip_reached)),
        "velocity_front_max_y_mm": float(np.max(reached_y)) if reached_y.size else None,
        "dc_slip_front_max_y_mm": float(np.max(weakened_y)) if weakened_y.size else None,
        "peak_slip_rate_mm_s": _distribution(peak_speed),
        "rupture_front": front_metrics,
        "dynamic_rupture_front": dynamic_front_metrics,
        "dynamic_rupture_onset_time_in_shear_ms": (
            None
            if dynamic_onset_index is None
            else 1.0e3
            * (
                float(history[dynamic_onset_index, col["time"]])
                - shear_start_time
            )
        ),
        "dynamic_rupture_completion_time_in_shear_ms": (
            None
            if dynamic_completion_index is None
            else 1.0e3
            * (
                float(history[dynamic_completion_index, col["time"]])
                - shear_start_time
            )
        ),
        "rupture_onset_time_ms": (
            None
            if onset_index is None
            else 1.0e3 * float(history[onset_index, col["time"]])
        ),
        "rupture_onset_time_in_shear_ms": (
            None
            if onset_index is None
            else 1.0e3
            * (float(history[onset_index, col["time"]]) - shear_start_time)
        ),
        "rupture_onset_applied_displacement_mm": (
            None if onset_index is None else float(applied_displacement[onset_index])
        ),
        "rupture_completion_time_ms": (
            None
            if completion_index is None
            else 1.0e3 * float(history[completion_index, col["time"]])
        ),
        "rupture_completion_time_in_shear_ms": (
            None
            if completion_index is None
            else 1.0e3
            * (float(history[completion_index, col["time"]]) - shear_start_time)
        ),
        "rupture_completion_applied_displacement_mm": (
            None
            if completion_index is None
            else float(applied_displacement[completion_index])
        ),
        "loading_stopped_at_rupture_completion": (
            None
            if completion_index is None
            else bool(
                history[completion_index, col["shear_loading_stopped"]] > 0.5
            )
        ),
        "kinetic_energy_ratio_at_rupture_completion": (
            None
            if completion_index is None
            else float(kinetic_ratio[completion_index])
        ),
        "loading_stop_time_ms": (
            None
            if stop_index is None
            else 1.0e3 * float(history[stop_index, col["time"]])
        ),
        "loading_stop_time_in_shear_ms": (
            None
            if stop_index is None
            else 1.0e3
            * (float(history[stop_index, col["time"]]) - shear_start_time)
        ),
        "loading_stop_displacement_mm": (
            None if stop_index is None else float(applied_displacement[stop_index])
        ),
        "maximum_applied_displacement_mm": float(np.max(applied_displacement)),
        "kinetic_energy": {
            "max_pre_rupture_ratio": float(np.max(kinetic_ratio[pre_onset])),
            "ratio_at_rupture_onset": (
                None if onset_index is None else float(kinetic_ratio[onset_index])
            ),
            "max_pre_rupture": float(np.max(kinetic[pre_onset])),
            "max_pre_dynamic_rupture_ratio": float(
                np.max(kinetic_ratio[pre_dynamic_onset])
            ),
            "ratio_at_dynamic_rupture_onset": (
                None
                if dynamic_onset_index is None
                else float(kinetic_ratio[dynamic_onset_index])
            ),
            "max_dynamic_ratio_in_shear": float(np.max(kinetic_ratio[shear_phase])),
            "max_dynamic_in_shear": float(np.max(kinetic[shear_phase])),
        },
    }
    return metrics


def main() -> int:
    args = parse_args()
    if args.velocity_threshold <= 0.0:
        raise ValueError("velocity-threshold must be positive.")
    if args.dynamic_velocity_threshold <= 0.0:
        raise ValueError("dynamic-velocity-threshold must be positive.")
    if args.dynamic_velocity_threshold <= args.velocity_threshold:
        raise ValueError(
            "dynamic-velocity-threshold must exceed velocity-threshold."
        )
    metrics = evaluate(
        args.data_path,
        args.velocity_threshold,
        args.dynamic_velocity_threshold,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.data_path.expanduser().resolve().parent.parent
        / "stats"
        / "rupture_pilot_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
