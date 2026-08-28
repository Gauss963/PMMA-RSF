from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from tatva.pmma.plotting import configure_journal_style, style_axis


DEFAULT_Y_POINTS = [0.0, 50.0, 100.0, 200.0, 400.0, 480.0, 495.0, 500.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot local contact shear stress against cumulative fault slip."
    )
    parser.add_argument("data_path", type=Path, help="Simulation HDF5 file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output stem or .png/.pdf path. Both PNG and PDF are written.",
    )
    parser.add_argument(
        "--y-points",
        type=float,
        nargs="+",
        default=DEFAULT_Y_POINTS,
        help="Contact-line y coordinates [mm].",
    )
    parser.add_argument(
        "--slip-samples-per-point",
        type=int,
        default=350,
        help="Target number of slip-spaced samples per station.",
    )
    parser.add_argument(
        "--uniform-time-samples",
        type=int,
        default=600,
        help="Additional uniformly time-spaced shear-phase samples.",
    )
    parser.add_argument(
        "--normal-penalty",
        type=float,
        default=None,
        help=(
            "Normal contact penalty [stress/mm]. When omitted, infer the default "
            "as ten times the maximum tangential penalty."
        ),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def output_paths(data_path: Path, requested: Path | None) -> tuple[Path, Path]:
    if requested is None:
        stem = data_path.parent.parent / "Plot" / "interface_stress_slip_selected_points"
    elif requested.suffix.lower() in {".png", ".pdf"}:
        stem = requested.with_suffix("")
    else:
        stem = requested
    stem.parent.mkdir(parents=True, exist_ok=True)
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


def nearest_station_indices(
    contact_y: np.ndarray, requested_y: list[float]
) -> np.ndarray:
    indices = np.asarray(
        [int(np.argmin(np.abs(contact_y - target))) for target in requested_y],
        dtype=np.int64,
    )
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Two requested y-points resolve to the same contact station.")
    return indices


def choose_frame_indices(
    cumulative_slip: np.ndarray,
    shear_indices: np.ndarray,
    *,
    slip_samples_per_point: int,
    uniform_time_samples: int,
) -> np.ndarray:
    if slip_samples_per_point < 2:
        raise ValueError("slip_samples_per_point must be at least 2.")
    if uniform_time_samples < 2:
        raise ValueError("uniform_time_samples must be at least 2.")

    chosen: set[int] = set(
        np.linspace(
            int(shear_indices[0]),
            int(shear_indices[-1]),
            min(uniform_time_samples, len(shear_indices)),
            dtype=np.int64,
        ).tolist()
    )
    for station in range(cumulative_slip.shape[1]):
        station_slip = cumulative_slip[shear_indices, station]
        final_slip = float(station_slip[-1])
        if final_slip <= 0.0:
            continue
        targets = np.linspace(
            0.0,
            final_slip,
            min(slip_samples_per_point, len(shear_indices)),
        )
        local_indices = np.searchsorted(station_slip, targets, side="left")
        local_indices = np.clip(local_indices, 0, len(shear_indices) - 1)
        chosen.update(shear_indices[local_indices].astype(int).tolist())

    chosen.add(int(shear_indices[0]))
    chosen.add(int(shear_indices[-1]))
    return np.asarray(sorted(chosen), dtype=np.int64)


def colored_curve(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color_value: np.ndarray,
    *,
    norm: Normalize,
) -> LineCollection:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(color_value)
    x = x[valid]
    y = y[valid]
    color_value = color_value[valid]
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=norm,
        linewidth=1.15,
        alpha=0.95,
    )
    collection.set_array(0.5 * (color_value[:-1] + color_value[1:]))
    axis.add_collection(collection)
    axis.autoscale_view()
    return collection


def subplot_grid_shape(station_count: int) -> tuple[int, int]:
    if station_count < 1:
        raise ValueError("At least one station is required.")
    if station_count <= 2:
        return 1, station_count
    return math.ceil(station_count / 2), 2


def plot_stress_slip(
    data_path: Path,
    png_path: Path,
    pdf_path: Path,
    *,
    requested_y: list[float],
    slip_samples_per_point: int,
    uniform_time_samples: int,
    normal_penalty_override: float | None,
    dpi: int,
) -> dict[str, object]:
    configure_journal_style()
    run_name = data_path.parent.parent.name
    with h5py.File(data_path, "r") as h5:
        contact_y = np.asarray(h5["interface/contact_line_y"], dtype=np.float64)
        station_indices = nearest_station_indices(contact_y, requested_y)
        resolved_y = contact_y[station_indices]
        master_nodes = np.asarray(
            h5["interface/master_nodes"][station_indices], dtype=np.int64
        )
        slave_nodes = np.asarray(
            h5["interface/slave_nodes"][station_indices], dtype=np.int64
        )
        critical_slip = np.asarray(
            h5["interface/critical_slip_profile"][station_indices],
            dtype=np.float64,
        )
        creep_weight = np.asarray(
            h5["interface/creep_weight_profile"][station_indices],
            dtype=np.float64,
        )
        penalty_t = np.asarray(
            h5["interface/tangential_penalty_profile"][station_indices],
            dtype=np.float64,
        )
        all_penalty_t = np.asarray(
            h5["interface/tangential_penalty_profile"], dtype=np.float64
        )
        normal_penalty = (
            10.0 * float(np.max(all_penalty_t))
            if normal_penalty_override is None
            else float(normal_penalty_override)
        )
        if normal_penalty <= 0.0:
            raise ValueError("normal_penalty must be positive.")

        history = np.asarray(h5["history"], dtype=np.float64)
        history_columns = [
            item.decode() for item in np.asarray(h5["history_columns"])
        ]
        time = history[:, history_columns.index("time")]
        pressure_time = float(
            h5.attrs["pressure_steps"] * h5.attrs["dt"]
        )
        shear_time_ms = (time - pressure_time) * 1e3
        phase_id = np.asarray(h5["phase_id"])
        shear_indices = np.flatnonzero(phase_id == 2)
        cumulative_slip = np.asarray(
            h5["interface/cumulative_slip"][:, station_indices],
            dtype=np.float64,
        )
        frame_indices = choose_frame_indices(
            cumulative_slip,
            shear_indices,
            slip_samples_per_point=slip_samples_per_point,
            uniform_time_samples=uniform_time_samples,
        )

        plastic_slip = np.asarray(
            h5["interface/plastic_slip"][frame_indices, :][:, station_indices],
            dtype=np.float64,
        )
        relative_displacement = np.empty(
            (len(frame_indices), len(station_indices), 2), dtype=np.float64
        )
        moving_displacement = h5["moving/displacement"]
        stationary_displacement = h5["stationary/displacement"]
        for output_index, frame_index in enumerate(frame_indices):
            relative_displacement[output_index] = (
                np.asarray(
                    moving_displacement[frame_index, master_nodes, :],
                    dtype=np.float64,
                )
                - np.asarray(
                    stationary_displacement[frame_index, slave_nodes, :],
                    dtype=np.float64,
                )
            )

        penetration = np.maximum(relative_displacement[:, :, 0], 0.0)
        normal_traction = normal_penalty * penetration
        shear_traction = penalty_t[None, :] * (
            relative_displacement[:, :, 1] - plastic_slip
        )
        shear_traction = np.where(penetration > 0.0, shear_traction, np.nan)
        sampled_slip = cumulative_slip[frame_indices]
        sampled_time_ms = shear_time_ms[frame_indices]

        stopped = history[:, history_columns.index("shear_loading_stopped")] > 0.5
        stop_indices = np.flatnonzero(stopped)
        stop_time_ms = (
            float(shear_time_ms[stop_indices[0]]) if len(stop_indices) else np.nan
        )

    station_count = len(station_indices)
    subplot_rows, subplot_columns = subplot_grid_shape(station_count)
    figure, axes = plt.subplots(
        subplot_rows,
        subplot_columns,
        figsize=(3.5 * subplot_columns, 2.35 * subplot_rows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    norm = Normalize(
        vmin=max(0.0, float(np.min(sampled_time_ms))),
        vmax=float(np.max(sampled_time_ms)),
    )
    last_collection: LineCollection | None = None
    station_summaries: list[dict[str, float]] = []

    for station in range(station_count):
        axis = axes_flat[station]
        last_collection = colored_curve(
            axis,
            sampled_slip[:, station],
            shear_traction[:, station],
            sampled_time_ms,
            norm=norm,
        )
        axis.axvline(
            critical_slip[station],
            color="#d1495b",
            linestyle="--",
            linewidth=0.9,
            label=r"local $D_c$",
        )
        if np.isfinite(stop_time_ms):
            stop_sample = int(np.argmin(np.abs(sampled_time_ms - stop_time_ms)))
            axis.scatter(
                sampled_slip[stop_sample, station],
                shear_traction[stop_sample, station],
                s=19,
                facecolor="white",
                edgecolor="black",
                linewidth=0.9,
                zorder=4,
                label="loading stopped",
            )
        panel = chr(ord("a") + station)
        axis.set_title(
            f"({panel})  $y = {resolved_y[station]:.0f}$ mm",
            loc="left",
        )
        axis.text(
            0.98,
            0.95,
            rf"$D_c = {critical_slip[station]:.3g}$ mm",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )
        axis.margins(x=0.02, y=0.08)
        style_axis(axis)
        if station % subplot_columns == 0:
            axis.set_ylabel(r"Contact shear traction $\tau$ [MPa]")
        if station // subplot_columns == subplot_rows - 1:
            axis.set_xlabel(r"Cumulative slip $\delta$ [mm]")
        if station == 0:
            axis.legend(loc="best")

        peak_index = int(np.nanargmax(shear_traction[:, station]))
        dc_indices = np.flatnonzero(
            sampled_slip[:, station] >= critical_slip[station]
        )
        dc_index = int(dc_indices[0]) if len(dc_indices) else -1
        station_summaries.append(
            {
                "y_mm": float(resolved_y[station]),
                "critical_slip_mm": float(critical_slip[station]),
                "creep_weight": float(creep_weight[station]),
                "peak_tau_mpa": float(shear_traction[peak_index, station]),
                "slip_at_peak_tau_mm": float(sampled_slip[peak_index, station]),
                "tau_at_dc_mpa": (
                    float(shear_traction[dc_index, station])
                    if dc_index >= 0
                    else float("nan")
                ),
                "final_slip_mm": float(sampled_slip[-1, station]),
                "final_tau_mpa": float(shear_traction[-1, station]),
                "final_sigma_n_mpa": float(normal_traction[-1, station]),
            }
        )

    for axis in axes_flat[station_count:]:
        axis.set_visible(False)

    if last_collection is None:
        raise RuntimeError("No stress-slip curves were generated.")
    colorbar = figure.colorbar(
        last_collection,
        ax=axes_flat.tolist(),
        shrink=0.88,
        pad=0.015,
    )
    colorbar.set_label("Shear-phase time [ms]")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)

    return {
        "data_path": str(data_path),
        "run_name": run_name,
        "png": str(png_path),
        "pdf": str(pdf_path),
        "sampled_frames": int(len(frame_indices)),
        "normal_penalty": normal_penalty,
        "normal_penalty_source": (
            "10 * max(tangential_penalty_profile)"
            if normal_penalty_override is None
            else "command line"
        ),
        "loading_stop_time_in_shear_ms": stop_time_ms,
        "stations": station_summaries,
    }


def main() -> int:
    args = parse_args()
    data_path = args.data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    png_path, pdf_path = output_paths(data_path, args.output)
    payload = plot_stress_slip(
        data_path,
        png_path,
        pdf_path,
        requested_y=list(args.y_points),
        slip_samples_per_point=args.slip_samples_per_point,
        uniform_time_samples=args.uniform_time_samples,
        normal_penalty_override=args.normal_penalty,
        dpi=args.dpi,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
