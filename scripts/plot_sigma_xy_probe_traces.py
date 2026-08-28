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
from matplotlib.patches import Rectangle

from plot_near_fault_stress_fluctuation import (
    BACKGROUND,
    DISTANCE_COLORS,
    GOLD,
    GRID,
    INK,
    MUTED,
    PANEL,
    RED,
    STATION_COLORS,
    choose_probe_patches,
    configure_style,
    first_crossing_times,
    saved_time_ms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot traction-positive near-fault delta-tau time histories. "
            "The residual reference and rupture-tip definition match "
            "plot_near_fault_stress_fluctuation.py."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-traces", type=Path, required=True)
    parser.add_argument(
        "--output-layout",
        type=Path,
        default=None,
        help="Probe-layout output. Defaults beside --output-traces.",
    )
    parser.add_argument(
        "--skip-layout",
        action="store_true",
        help="Write only the stress-history figure and metrics.",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=None,
        help="JSON metrics output. Defaults beside --output-traces.",
    )
    parser.add_argument(
        "--y-points",
        type=float,
        nargs="+",
        default=[160.0, 240.0, 320.0, 400.0],
        help="Along-fault probe positions [mm].",
    )
    parser.add_argument(
        "--even-y-count",
        type=int,
        default=None,
        help=(
            "Ignore --y-points and sample this many evenly spaced positions "
            "between the two fault ends."
        ),
    )
    parser.add_argument(
        "--x-from-fault",
        type=float,
        default=5.0,
        help="Legacy single-distance value; retained for API compatibility.",
    )
    parser.add_argument(
        "--off-fault-distances",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        help=(
            "Normal distances from the fault [mm]. Zero uses reconstructed "
            "interface traction; positive values use moving-block bulk stress."
        ),
    )
    parser.add_argument(
        "--probe-half-size",
        type=float,
        default=0.5,
        help="Half-size of the square averaging patch [mm].",
    )
    parser.add_argument(
        "--tip-slip-fraction",
        type=float,
        default=0.05,
        help="Local D_c fraction used as the rupture-tip arrival proxy.",
    )
    parser.add_argument(
        "--residual-start-us",
        type=float,
        default=40.0,
        help="Start of the post-tip residual averaging window [microseconds].",
    )
    parser.add_argument(
        "--residual-end-us",
        type=float,
        default=50.0,
        help="End of the post-tip residual averaging window [microseconds].",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["residual", "pre-event"],
        default="residual",
        help=(
            "residual matches the rupture-tip figures; pre-event exposes the "
            "permanent stress drop between pre- and post-event plateaus."
        ),
    )
    parser.add_argument(
        "--pre-baseline-start-us",
        type=float,
        default=300.0,
        help="Far edge of the pre-tip baseline window [microseconds before tip].",
    )
    parser.add_argument(
        "--pre-baseline-end-us",
        type=float,
        default=200.0,
        help="Near edge of the pre-tip baseline window [microseconds before tip].",
    )
    parser.add_argument(
        "--post-window-duration-ms",
        type=float,
        default=2.0,
        help="Duration of the late plateau window ending at the dump end [ms].",
    )
    parser.add_argument(
        "--time-before-ms",
        type=float,
        default=0.30,
        help="Automatic plot padding before the earliest selected arrival [ms].",
    )
    parser.add_argument(
        "--time-after-ms",
        type=float,
        default=0.35,
        help="Automatic plot padding after the latest selected arrival [ms].",
    )
    parser.add_argument(
        "--time-start-ms",
        type=float,
        default=None,
        help="Explicit plot start in the selected time-origin coordinates [ms].",
    )
    parser.add_argument(
        "--time-end-ms",
        type=float,
        default=None,
        help="Explicit plot end in the selected time-origin coordinates [ms].",
    )
    parser.add_argument(
        "--time-origin",
        choices=["shear-start", "local-tip"],
        default="shear-start",
        help="Use global shear-phase time or align each trace to its local tip.",
    )
    parser.add_argument(
        "--time-scale",
        choices=["linear", "symlog"],
        default="linear",
        help=(
            "Time-axis scale. symlog supports pre-tip negative time and zero "
            "while compressing the post-event tail."
        ),
    )
    parser.add_argument(
        "--symlog-linthresh-us",
        type=float,
        default=100.0,
        help="Half-width of the symlog axis's central linear interval [us].",
    )
    parser.add_argument(
        "--tail-sample-interval-us",
        type=float,
        default=20.0,
        help=(
            "Display sampling interval after the dense rupture window "
            "[microseconds]; no smoothing is applied."
        ),
    )
    parser.add_argument(
        "--dense-after-full-rupture-ms",
        type=float,
        default=0.50,
        help="Keep every frame this long after the full-fault tip arrival [ms].",
    )
    parser.add_argument(
        "--arrival-chunk-frames",
        type=int,
        default=2048,
        help="Frames per cumulative-slip chunk while finding arrivals.",
    )
    parser.add_argument(
        "--batch-frames",
        type=int,
        default=512,
        help="Frames per stress/displacement HDF5 read.",
    )
    parser.add_argument(
        "--raw-sigma",
        action="store_true",
        help="Plot traction-positive raw shear stress instead of delta tau.",
    )
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def _output_pair(path: Path) -> tuple[Path, Path]:
    suffix = path.suffix.lower()
    if suffix in {".png", ".pdf"}:
        stem = path.with_suffix("")
    else:
        stem = path
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


def _default_sibling(path: Path, stem_suffix: str, extension: str) -> Path:
    stem = path.with_suffix("") if path.suffix else path
    return stem.with_name(f"{stem.name}{stem_suffix}").with_suffix(extension)


def _save_figure(
    figure: plt.Figure,
    requested_path: Path,
    dpi: int,
) -> tuple[Path, Path]:
    png_path, pdf_path = _output_pair(requested_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=dpi)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def _run_id(input_path: Path) -> str:
    run_dir = input_path.parent.parent if input_path.parent.name == "data" else input_path.parent
    return run_dir.name.split("_", maxsplit=1)[0]


def _critical_slip_profile(h5: h5py.File, contact_y: np.ndarray) -> np.ndarray:
    if "critical_slip_profile" in h5["interface"]:
        return np.asarray(
            h5["interface/critical_slip_profile"],
            dtype=np.float64,
        )
    return np.full(
        contact_y.shape,
        float(h5["interface"].attrs["critical_slip"]),
        dtype=np.float64,
    )


def _resolve_stations(
    contact_y: np.ndarray,
    requested_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    station_indices = np.asarray(
        [int(np.argmin(np.abs(contact_y - target))) for target in requested_y],
        dtype=np.int64,
    )
    if len(np.unique(station_indices)) != len(station_indices):
        raise ValueError("Two requested y positions resolve to the same interface node.")
    return station_indices, contact_y[station_indices]


def _read_interface_traction_frames(
    h5: h5py.File,
    frame_indices: np.ndarray,
    dense_frame_count: int,
    station_indices: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Reconstruct interface traction at dense event and sparse tail frames."""
    master_nodes = np.asarray(
        h5["interface/master_nodes"][station_indices],
        dtype=np.int64,
    )
    slave_nodes = np.asarray(
        h5["interface/slave_nodes"][station_indices],
        dtype=np.int64,
    )
    tangential_penalty = np.asarray(
        h5["interface/tangential_penalty_profile"][station_indices],
        dtype=np.float64,
    )
    traction = np.empty(
        (len(frame_indices), len(station_indices)),
        dtype=np.float32,
    )
    dense_frames = frame_indices[:dense_frame_count]
    tail_frames = frame_indices[dense_frame_count:]
    selectors: list[tuple[slice | np.ndarray, slice]] = []
    if len(dense_frames):
        if np.any(np.diff(dense_frames) != 1):
            raise ValueError("Dense interface frames must be contiguous.")
        selectors.append(
            (
                slice(int(dense_frames[0]), int(dense_frames[-1]) + 1),
                slice(0, len(dense_frames)),
            )
        )
    if len(tail_frames):
        selectors.append(
            (
                np.asarray(tail_frames, dtype=np.int64),
                slice(dense_frame_count, len(frame_indices)),
            )
        )

    for station_position, station_index in enumerate(station_indices):
        master_node = int(master_nodes[station_position])
        slave_node = int(slave_nodes[station_position])
        for selector, destination in selectors:
            moving_tangential = np.asarray(
                h5["moving/displacement"][selector, master_node, 1],
                dtype=np.float64,
            )
            stationary_tangential = np.asarray(
                h5["stationary/displacement"][selector, slave_node, 1],
                dtype=np.float64,
            )
            plastic_slip = np.asarray(
                h5["interface/plastic_slip"][selector, int(station_index)],
                dtype=np.float64,
            )
            traction[destination, station_position] = (
                (
                    moving_tangential
                    - stationary_tangential
                    - plastic_slip
                )
                * tangential_penalty[station_position]
            ).astype(np.float32)

    contact_y = np.asarray(
        h5["interface/contact_line_y"][station_indices],
        dtype=np.float64,
    )
    metadata = [
        {
            "source": "reconstructed_interface_contact_traction",
            "requested_station_mm": float(contact_y[index]),
            "requested_off_fault_mm": 0.0,
            "mean_station_mm": float(contact_y[index]),
            "mean_off_fault_mm": 0.0,
            "element_count": 0,
            "master_node": int(master_nodes[index]),
            "slave_node": int(slave_nodes[index]),
            "tangential_penalty": float(tangential_penalty[index]),
        }
        for index in range(len(station_indices))
    ]
    return traction, metadata


def _read_probe_stress_frames(
    dataset: h5py.Dataset,
    frame_indices: np.ndarray,
    groups: list[np.ndarray],
) -> np.ndarray:
    if not len(frame_indices):
        raise ValueError("No stress frames were selected.")
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if np.any(np.diff(frame_indices) <= 0):
        raise ValueError("Selected stress frames must increase strictly.")

    unique_elements = np.unique(np.concatenate(groups))
    element_position = {
        int(element): position
        for position, element in enumerate(unique_elements)
    }
    group_positions = [
        np.asarray([element_position[int(element)] for element in group])
        for group in groups
    ]
    output = np.empty((len(frame_indices), len(groups)), dtype=np.float32)
    split_points = np.flatnonzero(np.diff(frame_indices) != 1) + 1
    runs = np.split(frame_indices, split_points)
    output_start = 0
    for run_index, run in enumerate(runs):
        start = int(run[0])
        stop = int(run[-1]) + 1
        values = np.asarray(
            dataset[start:stop, unique_elements, 0, 1],
            dtype=np.float32,
        )
        output_stop = output_start + len(run)
        for group_index, positions in enumerate(group_positions):
            output[output_start:output_stop, group_index] = np.mean(
                values[:, positions],
                axis=1,
                dtype=np.float64,
            )
        output_start = output_stop
        if run_index == 0 or run_index == len(runs) - 1:
            print(
                f"Loaded stress run {run_index + 1}/{len(runs)}: "
                f"frames {start}:{stop}"
            )
    return output


def _select_dense_and_tail_frames(
    shear_indices: np.ndarray,
    shear_time_ms: np.ndarray,
    read_start_ms: float,
    read_end_ms: float,
    dense_end_ms: float,
    tail_sample_interval_us: float,
) -> tuple[np.ndarray, int]:
    available = shear_indices[
        (shear_time_ms[shear_indices] >= read_start_ms)
        & (shear_time_ms[shear_indices] <= read_end_ms)
    ]
    if len(available) < 2:
        raise ValueError("No saved shear frames cover the requested time window.")
    dense = available[shear_time_ms[available] <= dense_end_ms]
    tail = available[shear_time_ms[available] > dense_end_ms]
    if len(tail) and tail_sample_interval_us > 0.0:
        saved_dt_ms = float(np.median(np.diff(shear_time_ms[shear_indices])))
        stride = max(
            1,
            int(math.ceil(tail_sample_interval_us * 1e-3 / saved_dt_ms)),
        )
        sampled_tail = tail[::stride]
        if sampled_tail[-1] != tail[-1]:
            sampled_tail = np.append(sampled_tail, tail[-1])
        tail = sampled_tail
    selected = np.unique(np.concatenate([dense, tail])).astype(np.int64)
    return selected, len(dense)


def _subplot_grid(station_count: int) -> tuple[int, int, tuple[float, float]]:
    if station_count <= 4:
        return 2, 2, (7.2, 5.0)
    columns = 3
    rows = math.ceil(station_count / columns)
    return rows, columns, (7.2, max(5.0, 1.65 * rows + 1.8))


def _configure_time_axis(
    axis: plt.Axes,
    *,
    time_scale: str,
    symlog_linthresh_us: float,
) -> None:
    if time_scale == "linear":
        return
    axis.set_xscale(
        "symlog",
        base=10,
        linthresh=symlog_linthresh_us * 1e-3,
        linscale=1.0,
    )


def _plot_time_histories(
    *,
    run_id: str,
    time_ms: np.ndarray,
    plotted_stress: np.ndarray,
    stations: np.ndarray,
    arrivals_ms: np.ndarray,
    residual_stress: np.ndarray,
    distances: np.ndarray,
    tip_slip_fraction: float,
    residual_start_us: float,
    residual_end_us: float,
    view_start_ms: float,
    view_end_ms: float,
    time_scale: str,
    symlog_linthresh_us: float,
    raw_sigma: bool,
    output_path: Path,
    dpi: int,
) -> tuple[Path, Path]:
    rows, columns, figure_size = _subplot_grid(len(stations))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.76,
        bottom=0.115,
        wspace=0.16,
        hspace=0.28,
    )
    axes_flat = axes.ravel()
    view_mask = (time_ms >= view_start_ms) & (time_ms <= view_end_ms)
    if np.count_nonzero(view_mask) < 2:
        raise ValueError("The requested time window contains fewer than two frames.")
    values_in_view = plotted_stress[view_mask]
    y_min = float(np.nanmin(values_in_view))
    y_max = float(np.nanmax(values_in_view))
    y_margin = 0.08 * max(y_max - y_min, 1.0)

    for station_index, (axis, station, arrival) in enumerate(
        zip(axes_flat, stations, arrivals_ms, strict=False)
    ):
        axis.set_facecolor(PANEL)
        distance_lines = []
        for distance_index, distance in enumerate(distances):
            distance_lines.append(
                axis.plot(
                    time_ms,
                    plotted_stress[:, station_index, distance_index],
                    color=DISTANCE_COLORS[
                        distance_index % len(DISTANCE_COLORS)
                    ],
                    linewidth=1.10,
                    alpha=0.96,
                    zorder=3 + len(distances) - distance_index,
                    label=rf"$d_\perp={distance:g}$ mm",
                )[0]
            )
        axis.axhline(0.0, color=GRID, linewidth=1.0)
        tip_line = axis.axvline(
            arrival,
            color=RED,
            linestyle=(0, (3, 3)),
            linewidth=1.25,
            label=rf"tip: ${tip_slip_fraction:g}D_c$",
        )
        residual_start_ms = arrival + residual_start_us * 1e-3
        residual_end_ms = arrival + residual_end_us * 1e-3
        residual_patch = axis.axvspan(
            residual_start_ms,
            residual_end_ms,
            color=GOLD,
            alpha=0.22,
            linewidth=0.0,
            label="residual window",
        )
        axis.set_xlim(view_start_ms, view_end_ms)
        _configure_time_axis(
            axis,
            time_scale=time_scale,
            symlog_linthresh_us=symlog_linthresh_us,
        )
        axis.set_ylim(y_min - y_margin, y_max + y_margin)
        axis.set_title(
            rf"Station $y={station:.0f}$ mm  |  "
            rf"$t_{{tip}}={arrival:.4f}$ ms",
            loc="left",
            pad=7,
        )
        if not raw_sigma:
            axis.text(
                0.018,
                0.965,
                (
                    rf"$\tau^r$ range="
                    rf"{np.min(residual_stress[station_index]):.3f}-"
                    rf"{np.max(residual_stress[station_index]):.3f} MPa"
                ),
                transform=axis.transAxes,
                color=MUTED,
                fontsize=6.5,
                va="top",
            )
        axis.grid()
        axis.spines[["top", "right"]].set_visible(False)
        if station_index % columns == 0:
            axis.set_ylabel(
                r"$\Delta\tau$ [MPa]" if not raw_sigma else r"$\tau^+$ [MPa]"
            )
        if station_index >= (rows - 1) * columns:
            axis.set_xlabel("Time from shear-phase start [ms]")
        if station_index == 0:
            axis.legend(
                handles=[tip_line, residual_patch],
                loc="lower left",
                fontsize=6.4,
                ncol=2,
            )

    for axis in axes_flat[len(stations) :]:
        axis.set_visible(False)

    figure.legend(
        handles=distance_lines,
        labels=[rf"$d_\perp={distance:g}$ mm" for distance in distances],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.865),
        ncol=len(distances),
        frameon=False,
        fontsize=6.7,
        handlelength=1.8,
        columnspacing=0.9,
    )

    quantity = "Traction-positive shear stress" if raw_sigma else r"$\Delta\tau$ time histories"
    figure.text(
        0.075,
        0.985,
        f"{quantity} ({run_id})",
        fontsize=10.5,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    figure.text(
        0.075,
        0.951,
        (
            r"Interface traction at $d_\perp=0$ mm; moving-block "
            r"$\sigma_{xy}$ at $d_\perp=1$-5 mm, all in the same "
            r"traction-positive convention; "
            rf"tip=${tip_slip_fraction:g}D_c$; residual=+{residual_start_us:g}--"
            rf"{residual_end_us:g} $\mu$s."
        ),
        fontsize=7.2,
        color=MUTED,
        va="top",
    )
    footer = (
        r"$\Delta\tau(t)=\tau^+(t)-\tau^r$, where "
        r"$\tau^+$ is moving-side $\sigma_{xy}$ reoriented to positive "
        r"contact resistance. No temporal smoothing; each trace averages a "
        r"1$\times$1 mm probe patch."
        if not raw_sigma
        else (
            r"$\tau^+$ is moving-side $\sigma_{xy}$ reoriented to positive "
            r"contact resistance. No temporal smoothing."
        )
    )
    if time_scale == "symlog":
        footer += rf" Symmetric-log time is linear within $\pm{symlog_linthresh_us:g}\,\mu$s."
    figure.text(0.075, 0.022, footer, fontsize=6.3, color=MUTED)
    return _save_figure(figure, output_path, dpi)


def _plot_permanent_drop_histories(
    *,
    run_id: str,
    time_ms: np.ndarray,
    delta_tau: np.ndarray,
    stations: np.ndarray,
    arrivals_ms: np.ndarray,
    pre_baseline_stress: np.ndarray,
    post_baseline_stress: np.ndarray,
    post_baseline_std: np.ndarray,
    permanent_drop: np.ndarray,
    distances: np.ndarray,
    tip_slip_fraction: float,
    pre_baseline_start_us: float,
    pre_baseline_end_us: float,
    post_window_start_ms: float,
    post_window_end_ms: float,
    full_rupture_time_ms: float,
    view_start_ms: float,
    view_end_ms: float,
    time_origin: str,
    time_scale: str,
    symlog_linthresh_us: float,
    tail_sample_interval_us: float,
    output_path: Path,
    dpi: int,
) -> tuple[Path, Path]:
    rows, columns, figure_size = _subplot_grid(len(stations))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.76,
        bottom=0.115,
        wspace=0.16,
        hspace=0.28,
    )
    axes_flat = axes.ravel()
    displayed_values: list[np.ndarray] = []
    for station_index, arrival in enumerate(arrivals_ms):
        axis_time = time_ms - arrival if time_origin == "local-tip" else time_ms
        mask = (axis_time >= view_start_ms) & (axis_time <= view_end_ms)
        displayed_values.append(delta_tau[mask, station_index])
    finite_values = np.concatenate(displayed_values)
    y_min = float(np.nanmin(finite_values))
    y_max = float(np.nanmax(finite_values))
    y_margin = 0.08 * max(y_max - y_min, 1.0)
    if time_origin == "local-tip":
        post_window_visible = bool(
            np.any(
                (post_window_end_ms - arrivals_ms >= view_start_ms)
                & (post_window_start_ms - arrivals_ms <= view_end_ms)
            )
        )
    else:
        post_window_visible = (
            post_window_end_ms >= view_start_ms
            and post_window_start_ms <= view_end_ms
        )

    for station_index, (axis, station, arrival) in enumerate(
        zip(axes_flat, stations, arrivals_ms, strict=False)
    ):
        axis_time = time_ms - arrival if time_origin == "local-tip" else time_ms
        tip_x = 0.0 if time_origin == "local-tip" else arrival
        full_rupture_x = (
            full_rupture_time_ms - arrival
            if time_origin == "local-tip"
            else full_rupture_time_ms
        )
        pre_start_x = (
            -pre_baseline_start_us * 1e-3
            if time_origin == "local-tip"
            else arrival - pre_baseline_start_us * 1e-3
        )
        pre_end_x = (
            -pre_baseline_end_us * 1e-3
            if time_origin == "local-tip"
            else arrival - pre_baseline_end_us * 1e-3
        )
        post_start_x = (
            post_window_start_ms - arrival
            if time_origin == "local-tip"
            else post_window_start_ms
        )
        post_end_x = (
            post_window_end_ms - arrival
            if time_origin == "local-tip"
            else post_window_end_ms
        )

        axis.set_facecolor(PANEL)
        distance_lines = []
        for distance_index, distance in enumerate(distances):
            line = axis.plot(
                axis_time,
                delta_tau[:, station_index, distance_index],
                color=DISTANCE_COLORS[
                    distance_index % len(DISTANCE_COLORS)
                ],
                linewidth=1.10,
                alpha=0.96,
                zorder=3 + len(distances) - distance_index,
                label=rf"$d_\perp={distance:g}$ mm",
            )[0]
            distance_lines.append(line)
        axis.axhline(0.0, color=GRID, linewidth=1.0)
        pre_patch = axis.axvspan(
            pre_start_x,
            pre_end_x,
            color=STATION_COLORS[0],
            alpha=0.10,
            linewidth=0.0,
            label="pre-event baseline",
        )
        tip_line = axis.axvline(
            tip_x,
            color=RED,
            linestyle=(0, (3, 3)),
            linewidth=1.25,
            label=rf"local tip: ${tip_slip_fraction:g}D_c$",
        )
        full_tip_line = axis.axvline(
            full_rupture_x,
            color=GOLD,
            linestyle=(0, (4, 3)),
            linewidth=1.35,
            label="full-fault tip",
        )
        event_handles = [pre_patch, tip_line, full_tip_line]
        if post_end_x >= view_start_ms and post_start_x <= view_end_ms:
            post_patch = axis.axvspan(
                post_start_x,
                post_end_x,
                color=GOLD,
                alpha=0.16,
                linewidth=0.0,
                label="late plateau",
            )
            event_handles.append(post_patch)
            for distance_index in range(len(distances)):
                axis.hlines(
                    permanent_drop[station_index, distance_index],
                    post_start_x,
                    post_end_x,
                    color=DISTANCE_COLORS[
                        distance_index % len(DISTANCE_COLORS)
                    ],
                    linestyle=(0, (5, 3)),
                    linewidth=1.2,
                )
        axis.set_xlim(view_start_ms, view_end_ms)
        _configure_time_axis(
            axis,
            time_scale=time_scale,
            symlog_linthresh_us=symlog_linthresh_us,
        )
        axis.set_ylim(y_min - y_margin, y_max + y_margin)
        axis.set_title(
            rf"Station $y={station:.0f}$ mm  |  "
            rf"$t_{{tip}}={arrival:.4f}$ ms",
            loc="left",
            pad=7,
        )
        station_drop = permanent_drop[station_index]
        station_late_std = post_baseline_std[station_index]
        axis.text(
            0.018,
            0.965,
            (
                rf"late $\Delta\tau$ range="
                rf"{np.min(station_drop):+.3f} to "
                rf"{np.max(station_drop):+.3f} MPa  |  "
                rf"$\sigma_{{late}}$ range="
                rf"{np.min(station_late_std):.3f}-"
                rf"{np.max(station_late_std):.3f} MPa"
            ),
            transform=axis.transAxes,
            color=MUTED,
            fontsize=6.4,
            va="top",
        )
        axis.grid()
        axis.spines[["top", "right"]].set_visible(False)
        if station_index % columns == 0:
            axis.set_ylabel(r"$\Delta\tau=\tau^+-\tau^0$ [MPa]")
        if station_index >= (rows - 1) * columns:
            axis.set_xlabel(
                r"Time relative to local rupture tip, $t-t_{tip}$ [ms]"
                if time_origin == "local-tip"
                else "Time from shear-phase start [ms]"
            )
        if station_index == 0:
            axis.legend(
                handles=event_handles,
                loc="lower left",
                fontsize=6.2,
                ncol=2,
            )

    for axis in axes_flat[len(stations) :]:
        axis.set_visible(False)

    figure.legend(
        handles=distance_lines,
        labels=[rf"$d_\perp={distance:g}$ mm" for distance in distances],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.865),
        ncol=len(distances),
        frameon=False,
        fontsize=6.7,
        handlelength=1.8,
        columnspacing=0.9,
    )

    figure.text(
        0.075,
        0.985,
        f"Permanent near-fault stress drop ({run_id})",
        fontsize=10.5,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    figure.text(
        0.075,
        0.951,
        (
            r"Interface traction at $d_\perp=0$ mm; moving-block "
            r"$\sigma_{xy}$ at $d_\perp=1$-5 mm. "
            rf"pre-event mean=$-{pre_baseline_start_us:g}$ to "
            rf"$-{pre_baseline_end_us:g}$ $\mu$s; full rupture="
            rf"{full_rupture_time_ms:.4f} ms."
        ),
        fontsize=7.2,
        color=MUTED,
        va="top",
    )
    figure.text(
        0.075,
        0.022,
        (
            r"At 0 mm, $\tau^+$ is reconstructed contact traction; for "
            r"$d_\perp>0$, moving-side $\sigma_{xy}$ is reoriented to the same "
            r"positive resistance. Event: every frame; quiet tail: every "
            rf"{tail_sample_interval_us:g} $\mu$s, no smoothing. Late plateau="
            rf"{post_window_start_ms:.1f}--{post_window_end_ms:.1f} ms"
            + ("." if post_window_visible else " (outside displayed window).")
            + (
                rf" Symlog linear within $\pm{symlog_linthresh_us:g}\,\mu$s."
                if time_scale == "symlog"
                else ""
            )
        ),
        fontsize=6.3,
        color=MUTED,
    )
    return _save_figure(figure, output_path, dpi)


def _plot_probe_layout(
    *,
    run_id: str,
    coords: np.ndarray,
    stations: np.ndarray,
    probe_metadata: list[dict[str, object]],
    x_from_fault: float,
    baseline_mode: str,
    pre_baseline_start_us: float,
    pre_baseline_end_us: float,
    output_path: Path,
    dpi: int,
) -> tuple[Path, Path]:
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    fault_x = float(x_max)
    target_x = fault_x - x_from_fault

    figure = plt.figure(figsize=(7.2, 4.2))
    grid = figure.add_gridspec(
        1,
        2,
        left=0.08,
        right=0.98,
        top=0.87,
        bottom=0.12,
        width_ratios=[1.25, 0.75],
        wspace=0.10,
    )
    geometry_axis = figure.add_subplot(grid[0])
    note_axis = figure.add_subplot(grid[1])
    geometry_axis.set_facecolor(PANEL)
    geometry_axis.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="#E9EFEF",
            edgecolor=INK,
            linewidth=1.6,
        )
    )
    geometry_axis.axvline(
        fault_x,
        color=RED,
        linewidth=2.2,
        label="fault",
    )
    geometry_axis.axvline(
        target_x,
        color=GOLD,
        linestyle=(0, (4, 3)),
        linewidth=2.0,
        label=rf"probe line: $d_\perp={x_from_fault:g}$ mm",
    )
    mean_off_fault = np.asarray(
        [float(item["mean_off_fault_mm"]) for item in probe_metadata]
    )
    mean_y = np.asarray(
        [float(item["mean_station_mm"]) for item in probe_metadata]
    )
    centers_x = fault_x - mean_off_fault
    for index, (x_value, y_value, station) in enumerate(
        zip(centers_x, mean_y, stations, strict=True)
    ):
        color = STATION_COLORS[index % len(STATION_COLORS)]
        geometry_axis.scatter(
            x_value,
            y_value,
            s=28,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
        geometry_axis.annotate(
            rf"$y={station:.0f}$ mm",
            (x_value, y_value),
            xytext=(-12, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=color,
            fontsize=7.0,
            fontweight="semibold",
        )
    geometry_axis.set_xlim(x_min - 20.0, x_max + 20.0)
    geometry_axis.set_ylim(y_min - 15.0, y_max + 15.0)
    geometry_axis.set_aspect("equal")
    geometry_axis.set_xlabel("x, normal to fault [mm]")
    geometry_axis.set_ylabel("y, along fault [mm]")
    geometry_axis.set_title("Moving-block probe geometry", loc="left")
    geometry_axis.grid()
    geometry_axis.legend(loc="lower left")
    geometry_axis.spines[["top", "right"]].set_visible(False)

    note_axis.set_facecolor(PANEL)
    note_axis.set_xticks([])
    note_axis.set_yticks([])
    note_axis.spines[:].set_visible(False)
    note_axis.text(
        0.08,
        0.84,
        "Stress convention",
        transform=note_axis.transAxes,
        color=INK,
        fontsize=9.0,
        fontweight="semibold",
    )
    equation_text = (
        (
            r"$\tau^+=s\,\sigma_{xy}^{moving}$" "\n"
            r"$\Delta\tau(t)=\tau^+(t)-\tau^0$" "\n"
            rf"$\tau^0=\langle\tau^+\rangle_"
            rf"{{t_{{tip}}-{pre_baseline_start_us:g}\,\mu s}}^"
            rf"{{t_{{tip}}-{pre_baseline_end_us:g}\,\mu s}}$"
        )
        if baseline_mode == "pre-event"
        else (
            r"$\tau^+=s\,\sigma_{xy}^{moving}$" "\n"
            r"$\Delta\tau(t)=\tau^+(t)-\tau^r$" "\n"
            r"$\tau^r=\langle\tau^+\rangle_{t_{tip}+40\ldots50\,\mu s}$"
        )
    )
    note_axis.text(
        0.08,
        0.70,
        equation_text,
        transform=note_axis.transAxes,
        color=INK,
        fontsize=8.5,
        linespacing=1.6,
        va="top",
    )
    note_axis.text(
        0.08,
        0.36,
        (
            "The sign s is inferred by matching the moving-side bulk "
            "stress fluctuation to the reconstructed interface contact "
            "traction. This preserves the paper's positive resisting "
            "traction convention."
        ),
        transform=note_axis.transAxes,
        color=MUTED,
        fontsize=7.0,
        linespacing=1.35,
        va="top",
        wrap=True,
    )

    figure.text(
        0.08,
        0.985,
        f"Near-fault time-history probes ({run_id})",
        fontsize=10.5,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    figure.text(
        0.08,
        0.949,
        (
            "Probe positions and the pre-event-referenced stress-drop definition"
            if baseline_mode == "pre-event"
            else "Probe positions and the residual-referenced shear-stress definition"
        ),
        fontsize=7.2,
        color=MUTED,
        va="top",
    )
    return _save_figure(figure, output_path, dpi)


def validate_off_fault_distances(
    distances: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved = np.asarray(distances, dtype=np.float64)
    if not len(resolved):
        raise ValueError("At least one off-fault distance is required.")
    if np.any(resolved < 0.0) or np.any(np.diff(resolved) <= 0.0):
        raise ValueError(
            "off_fault_distances must be unique, non-negative, and increasing."
        )
    zero_indices = np.flatnonzero(np.isclose(resolved, 0.0))
    if len(zero_indices) > 1:
        raise ValueError("At most one 0 mm interface probe is allowed.")
    positive_indices = np.flatnonzero(~np.isclose(resolved, 0.0))
    if not len(positive_indices):
        raise ValueError("At least one positive bulk-probe distance is required.")
    return resolved, zero_indices, positive_indices


def plot_sigma_xy_probe_traces(
    input_path: Path,
    output_traces: Path,
    output_layout: Path | None,
    *,
    y_points: list[float],
    x_from_fault: float = 5.0,
    off_fault_distances: list[float] | None = None,
    probe_half_size: float = 0.5,
    tip_slip_fraction: float = 0.05,
    residual_start_us: float = 40.0,
    residual_end_us: float = 50.0,
    baseline_mode: str = "residual",
    pre_baseline_start_us: float = 300.0,
    pre_baseline_end_us: float = 200.0,
    post_window_duration_ms: float = 2.0,
    time_before_ms: float = 0.30,
    time_after_ms: float = 0.35,
    time_start_ms: float | None = None,
    time_end_ms: float | None = None,
    time_origin: str = "shear-start",
    time_scale: str = "linear",
    symlog_linthresh_us: float = 100.0,
    tail_sample_interval_us: float = 20.0,
    dense_after_full_rupture_ms: float = 0.50,
    arrival_chunk_frames: int = 2048,
    batch_frames: int = 512,
    raw_sigma: bool = False,
    dpi: int = 260,
    output_metrics: Path | None = None,
    skip_layout: bool = False,
) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if x_from_fault <= 0.0:
        raise ValueError("x_from_fault must be positive for a bulk-stress probe.")
    requested_distances = (
        [0.0, 1.0, 2.0, 3.0, 4.0, x_from_fault]
        if off_fault_distances is None
        else off_fault_distances
    )
    distances, zero_distance_indices, positive_distance_indices = (
        validate_off_fault_distances(requested_distances)
    )
    if probe_half_size <= 0.0:
        raise ValueError("probe_half_size must be positive.")
    if not 0.0 < tip_slip_fraction < 1.0:
        raise ValueError("tip_slip_fraction must lie strictly between 0 and 1.")
    if not 0.0 <= residual_start_us < residual_end_us:
        raise ValueError("The residual averaging window is invalid.")
    if baseline_mode not in {"residual", "pre-event"}:
        raise ValueError(f"Unsupported baseline mode: {baseline_mode}")
    if not 0.0 < pre_baseline_end_us < pre_baseline_start_us:
        raise ValueError("The pre-event baseline window is invalid.")
    if post_window_duration_ms <= 0.0:
        raise ValueError("post_window_duration_ms must be positive.")
    if time_before_ms < 0.0 or time_after_ms < 0.0:
        raise ValueError("Automatic time padding cannot be negative.")
    if time_origin not in {"shear-start", "local-tip"}:
        raise ValueError(f"Unsupported time origin: {time_origin}")
    if time_scale not in {"linear", "symlog"}:
        raise ValueError(f"Unsupported time scale: {time_scale}")
    if symlog_linthresh_us <= 0.0:
        raise ValueError("symlog_linthresh_us must be positive.")
    if tail_sample_interval_us < 0.0:
        raise ValueError("tail_sample_interval_us cannot be negative.")
    if dense_after_full_rupture_ms < 0.0:
        raise ValueError("dense_after_full_rupture_ms cannot be negative.")
    if raw_sigma and baseline_mode == "pre-event":
        raise ValueError("--raw-sigma cannot be combined with pre-event baseline mode.")

    configure_style()
    requested_y = np.asarray(y_points, dtype=np.float64)
    if not len(requested_y):
        raise ValueError("At least one y point is required.")

    with h5py.File(input_path, "r") as h5:
        shear_time_ms, shear_indices = saved_time_ms(h5)
        contact_y = np.asarray(
            h5["interface/contact_line_y"],
            dtype=np.float64,
        )
        station_indices, stations = _resolve_stations(contact_y, requested_y)
        critical_slip = _critical_slip_profile(h5, contact_y)
        arrival_all = first_crossing_times(
            h5["interface/cumulative_slip"],
            shear_indices,
            shear_time_ms,
            tip_slip_fraction * critical_slip,
            chunk_frames=arrival_chunk_frames,
        )
        arrivals_ms = arrival_all[station_indices]
        if np.any(~np.isfinite(arrivals_ms)):
            missing = stations[~np.isfinite(arrivals_ms)]
            raise ValueError(f"No rupture arrival found at y={missing.tolist()} mm.")

        shear_start_ms = float(shear_time_ms[shear_indices[0]])
        shear_end_ms = float(shear_time_ms[shear_indices[-1]])
        full_rupture_time_ms = float(np.nanmax(arrival_all))
        post_window_end_ms = shear_end_ms
        post_window_start_ms = post_window_end_ms - post_window_duration_ms
        if post_window_start_ms <= full_rupture_time_ms:
            raise ValueError(
                "The requested late plateau begins before full-fault rupture."
            )

        if time_origin == "local-tip":
            view_start_ms = -time_before_ms
            view_end_ms = time_after_ms
            if time_start_ms is not None:
                view_start_ms = float(time_start_ms)
            if time_end_ms is not None:
                view_end_ms = float(time_end_ms)
            read_start_ms = float(np.min(arrivals_ms) + view_start_ms)
            read_end_ms = float(np.max(arrivals_ms) + view_end_ms)
        else:
            auto_start = float(np.min(arrivals_ms) - time_before_ms)
            auto_end = float(np.max(arrivals_ms) + time_after_ms)
            view_start_ms = max(shear_start_ms, auto_start)
            view_end_ms = min(shear_end_ms, auto_end)
            if time_start_ms is not None:
                view_start_ms = float(time_start_ms)
            if time_end_ms is not None:
                view_end_ms = float(time_end_ms)
            read_start_ms = view_start_ms
            read_end_ms = view_end_ms
        if view_start_ms >= view_end_ms:
            raise ValueError("The resolved plot time window is empty.")

        read_start_ms = min(
            read_start_ms,
            float(np.min(arrivals_ms) - pre_baseline_start_us * 1e-3),
        )
        read_end_ms = max(
            read_end_ms,
            float(np.max(arrivals_ms) + residual_end_us * 1e-3),
        )
        if baseline_mode == "pre-event":
            read_end_ms = max(read_end_ms, post_window_end_ms)
        read_start_ms = max(read_start_ms, shear_start_ms)
        read_end_ms = min(read_end_ms, shear_end_ms)
        dense_end_ms = min(
            read_end_ms,
            full_rupture_time_ms + dense_after_full_rupture_ms,
        )
        selected_frames, dense_frame_count = _select_dense_and_tail_frames(
            shear_indices,
            shear_time_ms,
            read_start_ms,
            read_end_ms,
            dense_end_ms,
            tail_sample_interval_us,
        )
        if dense_frame_count < 2:
            raise ValueError("The dense rupture window contains fewer than two frames.")

        positive_distances = distances[positive_distance_indices]
        groups, bulk_probe_metadata = choose_probe_patches(
            h5,
            stations,
            positive_distances,
            probe_half_size,
        )
        bulk_sigma_xy_flat = _read_probe_stress_frames(
            h5["moving/stress"],
            selected_frames,
            groups,
        )
        interface_traction, interface_metadata = _read_interface_traction_frames(
            h5,
            selected_frames,
            dense_frame_count,
            station_indices,
        )
        coords = np.asarray(h5["moving/coords"], dtype=np.float64)

    time_ms = shear_time_ms[selected_frames]
    bulk_sigma_xy = bulk_sigma_xy_flat.reshape(
        len(time_ms),
        len(stations),
        len(positive_distances),
    )
    reference_distance_index = int(np.argmin(positive_distances))
    centered_bulk = (
        bulk_sigma_xy[:dense_frame_count, :, reference_distance_index]
        - np.mean(
            bulk_sigma_xy[:dense_frame_count, :, reference_distance_index],
            axis=0,
            keepdims=True,
        )
    )
    centered_interface = interface_traction[:dense_frame_count] - np.mean(
        interface_traction[:dense_frame_count],
        axis=0,
        keepdims=True,
    )
    sign_metric = float(
        np.sum(centered_interface * centered_bulk, dtype=np.float64)
    )
    bulk_to_contact_orientation = 1.0 if sign_metric >= 0.0 else -1.0
    traction_positive_stress = np.empty(
        (len(time_ms), len(stations), len(distances)),
        dtype=np.float32,
    )
    if len(zero_distance_indices):
        traction_positive_stress[:, :, zero_distance_indices[0]] = (
            interface_traction
        )
    traction_positive_stress[:, :, positive_distance_indices] = (
        bulk_to_contact_orientation * bulk_sigma_xy
    )

    bulk_metadata_grid = np.asarray(
        bulk_probe_metadata,
        dtype=object,
    ).reshape(len(stations), len(positive_distances))
    interface_metadata_array = np.asarray(interface_metadata, dtype=object)
    probe_metadata_grid = np.empty(
        (len(stations), len(distances)),
        dtype=object,
    )
    if len(zero_distance_indices):
        probe_metadata_grid[:, zero_distance_indices[0]] = (
            interface_metadata_array
        )
    probe_metadata_grid[:, positive_distance_indices] = bulk_metadata_grid
    farthest_distance_index = int(np.argmax(distances))
    layout_probe_metadata = probe_metadata_grid[:, farthest_distance_index].tolist()

    metric_shape = (len(stations), len(distances))
    residual_stress = np.empty(metric_shape, dtype=np.float64)
    pre_tip_stress = np.empty(metric_shape, dtype=np.float64)
    pre_baseline_stress = np.empty(metric_shape, dtype=np.float64)
    residual_sample_count = np.empty(len(stations), dtype=np.int64)
    pre_baseline_sample_count = np.empty(len(stations), dtype=np.int64)
    for station_index, arrival in enumerate(arrivals_ms):
        relative_time_ms = time_ms - arrival
        residual_mask = (
            (relative_time_ms >= residual_start_us * 1e-3)
            & (relative_time_ms <= residual_end_us * 1e-3)
        )
        pre_tip_mask = (
            (relative_time_ms >= -residual_end_us * 1e-3)
            & (relative_time_ms <= -residual_start_us * 1e-3)
        )
        pre_baseline_mask = (
            (relative_time_ms >= -pre_baseline_start_us * 1e-3)
            & (relative_time_ms <= -pre_baseline_end_us * 1e-3)
        )
        residual_sample_count[station_index] = np.count_nonzero(residual_mask)
        pre_baseline_sample_count[station_index] = np.count_nonzero(
            pre_baseline_mask
        )
        if residual_sample_count[station_index] < 2:
            raise ValueError(
                f"Residual window at y={stations[station_index]:g} mm "
                "contains fewer than two saved frames."
            )
        if np.count_nonzero(pre_tip_mask) < 2:
            raise ValueError(
                f"Pre-tip comparison window at y={stations[station_index]:g} mm "
                "contains fewer than two saved frames."
            )
        if pre_baseline_sample_count[station_index] < 2:
            raise ValueError(
                f"Pre-event baseline at y={stations[station_index]:g} mm "
                "contains fewer than two saved frames."
            )
        residual_stress[station_index] = np.mean(
            traction_positive_stress[residual_mask, station_index, :],
            axis=0,
            dtype=np.float64,
        )
        pre_tip_stress[station_index] = np.mean(
            traction_positive_stress[pre_tip_mask, station_index, :],
            axis=0,
            dtype=np.float64,
        )
        pre_baseline_stress[station_index] = np.mean(
            traction_positive_stress[pre_baseline_mask, station_index, :],
            axis=0,
            dtype=np.float64,
        )

    post_baseline_mask = (
        (time_ms >= post_window_start_ms) & (time_ms <= post_window_end_ms)
    )
    post_sample_count = int(np.count_nonzero(post_baseline_mask))
    if baseline_mode == "pre-event" and post_sample_count < 10:
        raise ValueError("The late plateau contains fewer than ten sampled frames.")
    if post_sample_count:
        post_baseline_stress = np.mean(
            traction_positive_stress[post_baseline_mask],
            axis=0,
            dtype=np.float64,
        )
        post_baseline_std = np.std(
            traction_positive_stress[post_baseline_mask],
            axis=0,
            dtype=np.float64,
        )
        post_time = time_ms[post_baseline_mask]
        post_slope = np.asarray(
            [
                np.polyfit(
                    post_time,
                    traction_positive_stress[
                        post_baseline_mask,
                        station,
                        :,
                    ],
                    1,
                )[0]
                for station in range(len(stations))
            ],
            dtype=np.float64,
        )
    else:
        post_baseline_stress = np.full(metric_shape, np.nan)
        post_baseline_std = np.full(metric_shape, np.nan)
        post_slope = np.full(metric_shape, np.nan)

    permanent_drop = post_baseline_stress - pre_baseline_stress
    reference_stress = (
        pre_baseline_stress if baseline_mode == "pre-event" else residual_stress
    )
    delta_tau = traction_positive_stress - reference_stress[None, :, :]
    plotted_stress = traction_positive_stress if raw_sigma else delta_tau

    run_id = _run_id(input_path)
    if baseline_mode == "pre-event":
        trace_paths = _plot_permanent_drop_histories(
            run_id=run_id,
            time_ms=time_ms,
            delta_tau=delta_tau,
            stations=stations,
            arrivals_ms=arrivals_ms,
            pre_baseline_stress=pre_baseline_stress,
            post_baseline_stress=post_baseline_stress,
            post_baseline_std=post_baseline_std,
            permanent_drop=permanent_drop,
            distances=distances,
            tip_slip_fraction=tip_slip_fraction,
            pre_baseline_start_us=pre_baseline_start_us,
            pre_baseline_end_us=pre_baseline_end_us,
            post_window_start_ms=post_window_start_ms,
            post_window_end_ms=post_window_end_ms,
            full_rupture_time_ms=full_rupture_time_ms,
            view_start_ms=view_start_ms,
            view_end_ms=view_end_ms,
            time_origin=time_origin,
            time_scale=time_scale,
            symlog_linthresh_us=symlog_linthresh_us,
            tail_sample_interval_us=tail_sample_interval_us,
            output_path=output_traces,
            dpi=dpi,
        )
    else:
        trace_paths = _plot_time_histories(
            run_id=run_id,
            time_ms=time_ms,
            plotted_stress=plotted_stress,
            stations=stations,
            arrivals_ms=arrivals_ms,
            residual_stress=residual_stress,
            distances=distances,
            tip_slip_fraction=tip_slip_fraction,
            residual_start_us=residual_start_us,
            residual_end_us=residual_end_us,
            view_start_ms=view_start_ms,
            view_end_ms=view_end_ms,
            time_scale=time_scale,
            symlog_linthresh_us=symlog_linthresh_us,
            raw_sigma=raw_sigma,
            output_path=output_traces,
            dpi=dpi,
        )
    layout_paths: tuple[Path, Path] | None = None
    if not skip_layout:
        if output_layout is None:
            output_layout = _default_sibling(
                output_traces,
                "_probe_layout",
                output_traces.suffix or ".png",
            )
        layout_paths = _plot_probe_layout(
            run_id=run_id,
            coords=coords,
            stations=stations,
            probe_metadata=layout_probe_metadata,
            x_from_fault=float(np.max(distances)),
            baseline_mode=baseline_mode,
            pre_baseline_start_us=pre_baseline_start_us,
            pre_baseline_end_us=pre_baseline_end_us,
            output_path=output_layout,
            dpi=dpi,
        )

    result: dict[str, object] = {
        "run_id": run_id,
        "input": str(input_path.resolve()),
        "outputs": {
            "traces_png": str(trace_paths[0].resolve()),
            "traces_pdf": str(trace_paths[1].resolve()),
        },
        "baseline_mode": baseline_mode,
        "definition": (
            (
                "delta_tau(t) = traction_positive_sigma_xy(t) - tau_pre; "
                f"tau_pre is the mean {pre_baseline_start_us:g}-"
                f"{pre_baseline_end_us:g} microseconds before local tip arrival"
            )
            if baseline_mode == "pre-event"
            else (
                "delta_tau(t) = traction_positive_sigma_xy(t) - tau_residual; "
                f"tau_residual is the mean {residual_start_us:g}-"
                f"{residual_end_us:g} microseconds after local tip arrival"
            )
        ),
        "tip_arrival_definition": (
            f"first cumulative-slip crossing of {tip_slip_fraction:g} * local D_c"
        ),
        "stress_sign_convention": (
            (
                "0 mm is reconstructed positive resisting contact traction; "
                "positive-distance moving-side sigma_xy is reoriented to match"
            )
            if len(zero_distance_indices)
            else (
                "Positive-distance moving-side sigma_xy is reoriented using "
                "the reconstructed interface contact traction."
            )
        ),
        "bulk_sigma_xy_orientation_to_positive_contact_traction": (
            bulk_to_contact_orientation
        ),
        "interface_bulk_dynamic_correlation_sign_metric": sign_metric,
        "off_fault_distances_mm": distances.astype(float).tolist(),
        "probe_half_size_mm": float(probe_half_size),
        "requested_y_mm": requested_y.astype(float).tolist(),
        "resolved_y_mm": stations.astype(float).tolist(),
        "tip_arrival_ms": arrivals_ms.astype(float).tolist(),
        "full_fault_tip_arrival_ms": full_rupture_time_ms,
        "residual_traction_positive_stress_mpa": residual_stress.tolist(),
        "pre_tip_traction_positive_stress_mpa": pre_tip_stress.tolist(),
        "pre_event_baseline_traction_positive_stress_mpa": (
            pre_baseline_stress.tolist()
        ),
        "late_plateau_traction_positive_stress_mpa": (
            post_baseline_stress.tolist()
        ),
        "late_plateau_standard_deviation_mpa": post_baseline_std.tolist(),
        "late_plateau_slope_mpa_per_ms": post_slope.tolist(),
        "permanent_stress_change_mpa": permanent_drop.tolist(),
        "pre_to_residual_stress_drop_mpa": (
            pre_tip_stress - residual_stress
        ).tolist(),
        "residual_sample_count": residual_sample_count.astype(int).tolist(),
        "pre_event_baseline_sample_count": (
            pre_baseline_sample_count.astype(int).tolist()
        ),
        "late_plateau_sample_count": post_sample_count,
        "late_plateau_window_ms": [
            post_window_start_ms,
            post_window_end_ms,
        ],
        "time_origin": time_origin,
        "time_scale": time_scale,
        "symlog_linthresh_us": float(symlog_linthresh_us),
        "time_window_ms": [view_start_ms, view_end_ms],
        "frames_read": {
            "first": int(selected_frames[0]),
            "last": int(selected_frames[-1]),
            "selected_count": int(len(selected_frames)),
            "dense_count": int(dense_frame_count),
            "tail_sample_interval_us": float(tail_sample_interval_us),
        },
        "probe_patches": bulk_probe_metadata,
        "interface_probes": interface_metadata,
    }
    if layout_paths is not None:
        result["outputs"].update(
            {
                "layout_png": str(layout_paths[0].resolve()),
                "layout_pdf": str(layout_paths[1].resolve()),
            }
        )
    if output_metrics is None:
        output_metrics = _default_sibling(output_traces, "_metrics", ".json")
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    result["outputs"]["metrics_json"] = str(output_metrics.resolve())
    output_metrics.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    args = parse_args()
    y_points = list(args.y_points)
    if args.even_y_count is not None:
        if args.even_y_count <= 0:
            raise ValueError(
                f"--even-y-count must be positive, got {args.even_y_count}"
            )
        with h5py.File(args.input, "r") as h5:
            contact_y = np.asarray(
                h5["interface/contact_line_y"],
                dtype=np.float64,
            )
        y_points = np.linspace(
            float(np.min(contact_y)),
            float(np.max(contact_y)),
            num=args.even_y_count + 2,
            dtype=np.float64,
        )[1:-1].tolist()

    result = plot_sigma_xy_probe_traces(
        args.input,
        args.output_traces,
        args.output_layout,
        y_points=y_points,
        x_from_fault=args.x_from_fault,
        off_fault_distances=args.off_fault_distances,
        probe_half_size=args.probe_half_size,
        tip_slip_fraction=args.tip_slip_fraction,
        residual_start_us=args.residual_start_us,
        residual_end_us=args.residual_end_us,
        baseline_mode=args.baseline_mode,
        pre_baseline_start_us=args.pre_baseline_start_us,
        pre_baseline_end_us=args.pre_baseline_end_us,
        post_window_duration_ms=args.post_window_duration_ms,
        time_before_ms=args.time_before_ms,
        time_after_ms=args.time_after_ms,
        time_start_ms=args.time_start_ms,
        time_end_ms=args.time_end_ms,
        time_origin=args.time_origin,
        time_scale=args.time_scale,
        symlog_linthresh_us=args.symlog_linthresh_us,
        tail_sample_interval_us=args.tail_sample_interval_us,
        dense_after_full_rupture_ms=args.dense_after_full_rupture_ms,
        arrival_chunk_frames=args.arrival_chunk_frames,
        batch_frames=args.batch_frames,
        raw_sigma=args.raw_sigma,
        dpi=args.dpi,
        output_metrics=args.output_metrics,
        skip_layout=args.skip_layout,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
