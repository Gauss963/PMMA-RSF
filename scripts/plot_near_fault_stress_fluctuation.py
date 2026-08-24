from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BACKGROUND = "#F7F4EE"
PANEL = "#FFFEFA"
INK = "#18313A"
MUTED = "#68767B"
GRID = "#D9D6CE"
TEAL = "#087F8C"
NAVY = "#1E4D6B"
ORANGE = "#E87524"
RED = "#B84A3A"
GOLD = "#E6B655"
STATION_COLORS = ("#087F8C", "#E87524", "#1E4D6B", "#B84A3A")
DISTANCE_COLORS = (
    "#D66A2C",
    "#3C2A78",
    "#315DA8",
    "#168B91",
    "#55A868",
    "#D5B52E",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot near-fault sigma_xy fluctuations in a rupture-tip coordinate, "
            "following the construction of Kammer and McLaskey (2019), Fig. 2."
        )
    )
    parser.add_argument("data_path", type=Path, help="Simulation HDF5 dump.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to the run's Plot directory.",
    )
    parser.add_argument(
        "--stations",
        type=float,
        nargs="+",
        default=[160.0, 240.0, 320.0, 400.0],
        help="Along-fault station positions in the stable segment [mm].",
    )
    parser.add_argument(
        "--off-fault-distances",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        help=(
            "Distances normal to the fault [mm]. Zero uses reconstructed "
            "interface contact traction; positive values use moving-block "
            "bulk stress oriented to the same traction-positive convention."
        ),
    )
    parser.add_argument(
        "--fit-start",
        type=float,
        default=120.0,
        help="Start of the rupture-speed fit interval [mm].",
    )
    parser.add_argument(
        "--fit-end",
        type=float,
        default=440.0,
        help="End of the rupture-speed fit interval [mm].",
    )
    parser.add_argument(
        "--tip-slip-fraction",
        type=float,
        default=0.05,
        help="Fraction of local D_c used as the rupture-tip arrival proxy.",
    )
    parser.add_argument(
        "--xi-min",
        type=float,
        default=-80.0,
        help="Minimum tip-relative along-fault coordinate [mm].",
    )
    parser.add_argument(
        "--xi-max",
        type=float,
        default=80.0,
        help="Maximum tip-relative along-fault coordinate [mm].",
    )
    parser.add_argument(
        "--residual-start-us",
        type=float,
        default=40.0,
        help="Start of the post-tip residual-stress averaging window [microseconds].",
    )
    parser.add_argument(
        "--residual-end-us",
        type=float,
        default=50.0,
        help="End of the post-tip residual-stress averaging window [microseconds].",
    )
    parser.add_argument(
        "--probe-half-size",
        type=float,
        default=0.5,
        help="Half-size of each square element-center averaging patch [mm].",
    )
    parser.add_argument(
        "--arrival-chunk-frames",
        type=int,
        default=2048,
        help="Frames per cumulative-slip chunk while finding rupture arrivals.",
    )
    parser.add_argument(
        "--stress-chunk-frames",
        type=int,
        default=64,
        help="Frames per near-fault stress read.",
    )
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Avenir", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.70,
            "legend.frameon": False,
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def saved_time_ms(h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    phase_id = np.asarray(h5["phase_id"], dtype=np.int8)
    shear_indices = np.flatnonzero(phase_id == 2)
    if len(shear_indices) < 2:
        raise ValueError("At least two saved shear-phase frames are required.")

    if "step_id" in h5 and "dt" in h5.attrs and "pressure_steps" in h5.attrs:
        step_id = np.asarray(h5["step_id"], dtype=np.int64)
        absolute_step = step_id + np.where(
            phase_id == 2,
            int(h5.attrs["pressure_steps"]),
            0,
        )
        absolute_time_ms = (
            absolute_step.astype(np.float64) * float(h5.attrs["dt"]) * 1e3
        )
    else:
        columns = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in np.asarray(h5["history_columns"])
        ]
        absolute_time_ms = (
            np.asarray(
                h5["history"][:, columns.index("time")],
                dtype=np.float64,
            )
            * 1e3
        )

    shear_time_ms = absolute_time_ms - absolute_time_ms[shear_indices[0]]
    if np.any(np.diff(shear_time_ms[shear_indices]) <= 0.0):
        raise ValueError("Saved shear-frame times must increase monotonically.")
    if not np.all(np.diff(shear_indices) == 1):
        raise ValueError("Saved shear frames must be contiguous.")
    return shear_time_ms, shear_indices


def first_crossing_times(
    dataset: h5py.Dataset,
    frame_indices: np.ndarray,
    time_ms: np.ndarray,
    thresholds: np.ndarray,
    *,
    chunk_frames: int,
) -> np.ndarray:
    if chunk_frames < 2:
        raise ValueError("arrival_chunk_frames must be at least 2.")

    station_count = int(dataset.shape[1])
    crossing_time = np.full(station_count, np.nan, dtype=np.float64)
    unresolved = np.ones(station_count, dtype=bool)
    previous_values: np.ndarray | None = None
    previous_time: float | None = None
    first_frame = int(frame_indices[0])
    last_frame = int(frame_indices[-1])

    for start in range(first_frame, last_frame + 1, chunk_frames):
        stop = min(start + chunk_frames, last_frame + 1)
        values = np.asarray(dataset[start:stop], dtype=np.float64)
        times = time_ms[start:stop]
        if previous_values is not None and previous_time is not None:
            values = np.vstack([previous_values, values])
            times = np.concatenate([[previous_time], times])

        before = values[:-1]
        after = values[1:]
        crossing = (
            (before < thresholds[None, :])
            & (after >= thresholds[None, :])
            & unresolved[None, :]
        )
        crossed_here = np.any(crossing, axis=0)
        if np.any(crossed_here):
            columns = np.flatnonzero(crossed_here)
            rows = np.argmax(crossing[:, columns], axis=0)
            low = before[rows, columns]
            high = after[rows, columns]
            fraction = np.divide(
                thresholds[columns] - low,
                high - low,
                out=np.zeros_like(low),
                where=high > low,
            )
            crossing_time[columns] = (
                times[rows] + fraction * (times[rows + 1] - times[rows])
            )
            unresolved[columns] = False

        previous_values = values[-1].copy()
        previous_time = float(times[-1])

    return crossing_time


def linear_arrival_fit(
    coordinate: np.ndarray,
    arrival_time_ms: np.ndarray,
    fit_start: float,
    fit_end: float,
) -> dict[str, object]:
    mask = (
        (coordinate >= fit_start)
        & (coordinate <= fit_end)
        & np.isfinite(arrival_time_ms)
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError("Not enough finite rupture arrivals in the fit interval.")

    slope, intercept = np.polyfit(coordinate[mask], arrival_time_ms[mask], 1)
    fitted = slope * coordinate[mask] + intercept
    residual = arrival_time_ms[mask] - fitted
    centered = arrival_time_ms[mask] - np.mean(arrival_time_ms[mask])
    r_squared = 1.0 - np.sum(residual**2) / np.sum(centered**2)
    return {
        "mask": mask,
        "slope_ms_per_mm": float(slope),
        "intercept_ms": float(intercept),
        "speed_m_per_s": float(1.0 / slope),
        "r_squared": float(r_squared),
    }


def descending_crossing_position(
    coordinate: np.ndarray,
    values: np.ndarray,
    level: float,
    target_coordinate: float,
) -> float:
    crossings = np.flatnonzero(
        (values[:-1] >= level) & (values[1:] < level)
    )
    if not len(crossings):
        return float("nan")
    lower = int(
        crossings[
            np.argmin(np.abs(coordinate[crossings] - target_coordinate))
        ]
    )
    value_span = values[lower + 1] - values[lower]
    if value_span == 0.0:
        return float(coordinate[lower])
    fraction = (level - values[lower]) / value_span
    return float(
        coordinate[lower]
        + fraction * (coordinate[lower + 1] - coordinate[lower])
    )


def estimate_cohesive_zone(
    cumulative_slip: h5py.Dataset,
    contact_y: np.ndarray,
    critical_slip: np.ndarray,
    shear_indices: np.ndarray,
    shear_time_ms: np.ndarray,
    half_dc_arrival: np.ndarray,
    fit_start: float,
    fit_end: float,
) -> dict[str, object]:
    span = fit_end - fit_start
    if span >= 60.0:
        positions = np.arange(fit_start + 20.0, fit_end - 39.9, 20.0)
    else:
        positions = np.linspace(
            fit_start + 0.20 * span,
            fit_end - 0.20 * span,
            num=3,
        )
    measured_positions: list[float] = []
    cohesive_zones: list[float] = []
    for target in positions:
        station = int(np.argmin(np.abs(contact_y - target)))
        arrival = float(half_dc_arrival[station])
        if not np.isfinite(arrival):
            continue
        frame = int(
            shear_indices[
                np.argmin(np.abs(shear_time_ms[shear_indices] - arrival))
            ]
        )
        progress = (
            np.asarray(cumulative_slip[frame], dtype=np.float64)
            / critical_slip
        )
        position_05 = descending_crossing_position(
            contact_y,
            progress,
            0.05,
            float(contact_y[station]),
        )
        position_95 = descending_crossing_position(
            contact_y,
            progress,
            0.95,
            float(contact_y[station]),
        )
        cohesive_zone = (position_05 - position_95) / 0.90
        if np.isfinite(cohesive_zone) and cohesive_zone > 0.0:
            measured_positions.append(float(contact_y[station]))
            cohesive_zones.append(float(cohesive_zone))

    if not cohesive_zones:
        return {
            "sample_positions_mm": [],
            "values_mm": [],
            "median_mm": float("nan"),
        }
    return {
        "sample_positions_mm": measured_positions,
        "values_mm": cohesive_zones,
        "median_mm": float(np.median(cohesive_zones)),
    }


def choose_probe_patches(
    h5: h5py.File,
    stations: np.ndarray,
    distances: np.ndarray,
    half_size: float,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    coords = np.asarray(h5["moving/coords"], dtype=np.float64)
    plot_elements = np.asarray(h5["moving/plot_elements"], dtype=np.int64)
    parent_elements = np.asarray(
        h5["moving/plot_parent_elements"],
        dtype=np.int64,
    )
    centers = coords[plot_elements].mean(axis=1)
    fault_x = float(np.max(coords[:, 0]))

    groups: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    tolerance = max(1e-8, half_size * 1e-7)
    for station in stations:
        for distance in distances:
            target_x = fault_x - float(distance)
            mask = (
                (np.abs(centers[:, 0] - target_x) <= half_size + tolerance)
                & (np.abs(centers[:, 1] - station) <= half_size + tolerance)
            )
            selected_plot = np.flatnonzero(mask)
            if not len(selected_plot):
                squared_distance = (
                    (centers[:, 0] - target_x) ** 2
                    + (centers[:, 1] - station) ** 2
                )
                selected_plot = np.asarray(
                    [int(np.argmin(squared_distance))],
                    dtype=np.int64,
                )
            selected_parent = np.unique(parent_elements[selected_plot])
            selected_centers = centers[selected_plot]
            groups.append(selected_parent)
            metadata.append(
                {
                    "source": "moving_bulk_sigma_xy_patch",
                    "requested_station_mm": float(station),
                    "requested_off_fault_mm": float(distance),
                    "mean_station_mm": float(np.mean(selected_centers[:, 1])),
                    "mean_off_fault_mm": float(
                        fault_x - np.mean(selected_centers[:, 0])
                    ),
                    "element_count": int(len(selected_parent)),
                    "element_indices": selected_parent.astype(int).tolist(),
                }
            )
    return groups, metadata


def read_probe_stress(
    dataset: h5py.Dataset,
    frame_start: int,
    frame_stop: int,
    groups: list[np.ndarray],
    *,
    chunk_frames: int,
) -> np.ndarray:
    if chunk_frames < 1:
        raise ValueError("stress_chunk_frames must be positive.")
    unique_elements = np.unique(np.concatenate(groups))
    element_position = {
        int(element): index for index, element in enumerate(unique_elements)
    }
    group_positions = [
        np.asarray([element_position[int(element)] for element in group])
        for group in groups
    ]
    frame_count = frame_stop - frame_start
    group_stress = np.empty((frame_count, len(groups)), dtype=np.float32)

    for start in range(frame_start, frame_stop, chunk_frames):
        stop = min(start + chunk_frames, frame_stop)
        values = np.asarray(
            dataset[start:stop, unique_elements, 0, 1],
            dtype=np.float32,
        )
        local_start = start - frame_start
        local_stop = stop - frame_start
        for group_index, positions in enumerate(group_positions):
            group_stress[local_start:local_stop, group_index] = np.mean(
                values[:, positions],
                axis=1,
                dtype=np.float64,
            )
        if (
            start == frame_start
            or stop == frame_stop
            or (start - frame_start) % (10 * chunk_frames) == 0
        ):
            print(
                f"Loaded near-fault stress frames {start}:{stop} / "
                f"{frame_stop - 1}"
            )
    return group_stress


def read_interface_traction(
    h5: h5py.File,
    frame_start: int,
    frame_stop: int,
    station_indices: np.ndarray,
    *,
    chunk_frames: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if chunk_frames < 1:
        raise ValueError("stress_chunk_frames must be positive.")

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
    master_order = np.argsort(master_nodes)
    slave_order = np.argsort(slave_nodes)
    master_restore = np.argsort(master_order)
    slave_restore = np.argsort(slave_order)
    frame_count = frame_stop - frame_start
    traction = np.empty((frame_count, len(station_indices)), dtype=np.float32)

    for start in range(frame_start, frame_stop, chunk_frames):
        stop = min(start + chunk_frames, frame_stop)
        moving_tangential = np.asarray(
            h5["moving/displacement"][
                start:stop,
                master_nodes[master_order],
                1,
            ],
            dtype=np.float64,
        )[:, master_restore]
        stationary_tangential = np.asarray(
            h5["stationary/displacement"][
                start:stop,
                slave_nodes[slave_order],
                1,
            ],
            dtype=np.float64,
        )[:, slave_restore]
        plastic_slip = np.asarray(
            h5["interface/plastic_slip"][start:stop, station_indices],
            dtype=np.float64,
        )
        elastic_slip = (
            moving_tangential - stationary_tangential - plastic_slip
        )
        traction[start - frame_start : stop - frame_start] = (
            elastic_slip * tangential_penalty[None, :]
        ).astype(np.float32)
        if (
            start == frame_start
            or stop == frame_stop
            or (start - frame_start) % (10 * chunk_frames) == 0
        ):
            print(
                f"Loaded interface traction frames {start}:{stop} / "
                f"{frame_stop - 1}"
            )

    metadata = [
        {
            "source": "reconstructed_interface_contact_traction",
            "requested_station_mm": float(
                h5["interface/contact_line_y"][station]
            ),
            "requested_off_fault_mm": 0.0,
            "mean_station_mm": float(
                h5["interface/contact_line_y"][station]
            ),
            "mean_off_fault_mm": 0.0,
            "element_count": 0,
            "master_node": int(master_node),
            "slave_node": int(slave_node),
            "tangential_penalty": float(penalty),
        }
        for station, master_node, slave_node, penalty in zip(
            station_indices,
            master_nodes,
            slave_nodes,
            tangential_penalty,
            strict=True,
        )
    ]
    return traction, metadata


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=dpi)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def add_cohesive_zone(axis: plt.Axes, cohesive_zone_mm: float) -> None:
    if np.isfinite(cohesive_zone_mm) and cohesive_zone_mm > 0.0:
        axis.axvspan(
            -cohesive_zone_mm,
            0.0,
            color=GOLD,
            alpha=0.18,
            linewidth=0.0,
            zorder=0,
        )
    axis.axvline(0.0, color=RED, linestyle=(0, (3, 3)), linewidth=1.2)


def plot_station_panels(
    *,
    run_id: str,
    stations: np.ndarray,
    distances: np.ndarray,
    xi: np.ndarray,
    delta_stress: np.ndarray,
    tip_arrival_ms: np.ndarray,
    speed_fit: dict[str, object],
    half_dc_fit: dict[str, object],
    cohesive_zone_mm: float,
    xi_min: float,
    xi_max: float,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16.0, 9.0),
        sharex=True,
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.965,
        top=0.80,
        bottom=0.12,
        wspace=0.13,
        hspace=0.25,
    )
    axes_flat = axes.ravel()

    for station_index, (axis, station) in enumerate(
        zip(axes_flat, stations, strict=True)
    ):
        axis.set_facecolor(PANEL)
        local_curves = delta_stress[:, station_index, :]
        robust_span = float(
            np.nanpercentile(local_curves, 99.5)
            - np.nanpercentile(local_curves, 0.5)
        )
        spacing = max(1.0, 1.08 * robust_span)
        for distance_index, distance in reversed(
            list(enumerate(distances))
        ):
            offset = distance_index * spacing
            axis.plot(
                xi,
                local_curves[:, distance_index] + offset,
                color=DISTANCE_COLORS[
                    distance_index % len(DISTANCE_COLORS)
                ],
                linewidth=1.35,
            )
            label_x = xi_max - 2.0
            label_y = float(
                np.interp(
                    label_x,
                    xi,
                    local_curves[:, distance_index],
                )
                + offset
            )
            axis.text(
                label_x,
                label_y,
                rf"$d_\perp={distance:g}$ mm",
                color=DISTANCE_COLORS[
                    distance_index % len(DISTANCE_COLORS)
                ],
                fontsize=10.2,
                ha="right",
                va="bottom",
                bbox={
                    "facecolor": PANEL,
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.0,
                },
            )

        add_cohesive_zone(axis, cohesive_zone_mm)
        axis.set_xlim(xi_min, xi_max)
        axis.set_yticks([])
        axis.set_title(
            rf"Station $y={station:.0f}$ mm  |  "
            rf"$t_{{tip}}={tip_arrival_ms[station_index]:.4f}$ ms",
            loc="left",
            pad=8,
        )
        axis.grid(axis="x")
        axis.spines[["top", "right", "left"]].set_visible(False)

        scale_mpa = 2.0
        scale_x = xi_min + 6.0
        lower, upper = axis.get_ylim()
        scale_bottom = lower + 0.10 * (upper - lower)
        axis.plot(
            [scale_x, scale_x],
            [scale_bottom, scale_bottom + scale_mpa],
            color=INK,
            linewidth=2.0,
            clip_on=False,
        )
        axis.plot(
            [scale_x - 1.0, scale_x + 1.0],
            [scale_bottom, scale_bottom],
            color=INK,
            linewidth=1.5,
            clip_on=False,
        )
        axis.plot(
            [scale_x - 1.0, scale_x + 1.0],
            [scale_bottom + scale_mpa, scale_bottom + scale_mpa],
            color=INK,
            linewidth=1.5,
            clip_on=False,
        )
        axis.text(
            scale_x + 2.0,
            scale_bottom + 0.5 * scale_mpa,
            "2 MPa",
            color=INK,
            fontsize=9.5,
            va="center",
        )

    for axis in axes[-1]:
        axis.set_xlabel(
            r"Tip-relative along-fault coordinate, "
            r"$\xi=y-y_{\mathrm{tip}}$ [mm]"
        )

    figure.text(
        0.07,
        0.955,
        f"Run {run_id}  |  Near-fault shear-stress fluctuation",
        fontsize=24,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(
        0.07,
        0.905,
        (
            r"Kammer-McLaskey Fig. 2 construction: "
            r"$\xi=-C_f(t-t_{\mathrm{tip}})$, "
            rf"$C_f={float(speed_fit['speed_m_per_s']) / 1e3:.3f}$ km/s "
            rf"($0.05D_c$, $R^2={float(speed_fit['r_squared']):.4f}$)."
        ),
        fontsize=13.5,
        color=MUTED,
        va="top",
    )
    figure.text(
        0.07,
        0.867,
        (
            rf"Consistency check: $0.5D_c$ fit = "
            rf"{float(half_dc_fit['speed_m_per_s']) / 1e3:.3f} km/s. "
            r"Gold = measured cohesive zone; red = tip proxy. "
            r"All curves use the paper's traction-positive polarity; "
            r"moving-side $\sigma_{xy}$ is sign-corrected accordingly."
        ),
        fontsize=11.5,
        color=MUTED,
        va="top",
    )
    figure.text(
        0.07,
        0.035,
        (
            r"At 0 mm, $\Delta\tau=\tau-\tau^r$; for $d_\perp>0$, "
            r"$\Delta\sigma_{xy}=\sigma_{xy}-\sigma_{xy}^{r}$ in the same "
            r"traction-positive polarity. Residuals use the local 40-50 $\mu$s "
            "post-tip mean; traces are vertically shifted within each station."
        ),
        fontsize=10.5,
        color=MUTED,
    )
    return save_figure(
        figure,
        output_dir,
        "near_fault_stress_fluctuation_by_station",
        dpi,
    )


def plot_collapse_panels(
    *,
    run_id: str,
    stations: np.ndarray,
    distances: np.ndarray,
    xi: np.ndarray,
    delta_stress: np.ndarray,
    speed_fit: dict[str, object],
    cohesive_zone_mm: float,
    xi_min: float,
    xi_max: float,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    figure = plt.figure(figsize=(16.0, 9.0))
    grid = figure.add_gridspec(
        2,
        4,
        left=0.07,
        right=0.965,
        top=0.80,
        bottom=0.12,
        wspace=0.27,
        hspace=0.31,
    )
    distance_axes = [figure.add_subplot(grid[0, column]) for column in range(4)]
    distance_axes.extend(
        figure.add_subplot(grid[1, column]) for column in range(2)
    )

    global_low, global_high = np.nanpercentile(
        delta_stress,
        [0.2, 99.8],
    )
    margin = 0.06 * max(global_high - global_low, 1.0)
    for distance_index, distance in enumerate(distances):
        axis = distance_axes[distance_index]
        axis.set_facecolor(PANEL)
        station_curves = delta_stress[:, :, distance_index]
        for station_index, station in enumerate(stations):
            axis.plot(
                xi,
                station_curves[:, station_index],
                color=STATION_COLORS[
                    station_index % len(STATION_COLORS)
                ],
                linewidth=1.05,
                alpha=0.82,
                label=rf"$y={station:.0f}$ mm",
            )
        axis.plot(
            xi,
            np.median(station_curves, axis=1),
            color=INK,
            linewidth=2.2,
            label="station median",
            zorder=5,
        )
        add_cohesive_zone(axis, cohesive_zone_mm)
        axis.axhline(0.0, color=GRID, linewidth=0.9)
        axis.set_xlim(xi_min, xi_max)
        axis.set_ylim(global_low - margin, global_high + margin)
        axis.set_title(rf"$d_\perp={distance:g}$ mm", loc="left")
        axis.grid()
        axis.spines[["top", "right"]].set_visible(False)
        if distance_index >= 4:
            axis.set_xlabel(r"$\xi=y-y_{\mathrm{tip}}$ [mm]")
        if distance_index in (0, 4):
            axis.set_ylabel(
                r"$\Delta\tau$ [MPa]"
                if distance == 0.0
                else r"$\Delta\sigma_{xy}$ [MPa]"
            )
        if distance_index == 0:
            axis.legend(loc="upper left", fontsize=9.2, ncol=2)

    amplitude_axis = figure.add_subplot(grid[1, 2:])
    amplitude_axis.set_facecolor(PANEL)
    peak_to_peak = np.ptp(delta_stress, axis=0)
    for distance_index, distance in enumerate(distances):
        amplitude_axis.plot(
            stations,
            peak_to_peak[:, distance_index],
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=DISTANCE_COLORS[
                distance_index % len(DISTANCE_COLORS)
            ],
            label=rf"$d_\perp={distance:g}$ mm",
        )
    amplitude_axis.set_title(
        "Waveform amplitude evolves along fault",
        loc="left",
    )
    amplitude_axis.set_xlabel("Station along fault, y [mm]")
    amplitude_axis.set_ylabel("Peak-to-peak stress fluctuation [MPa]")
    amplitude_axis.grid()
    amplitude_axis.spines[["top", "right"]].set_visible(False)
    amplitude_axis.legend(fontsize=9.2, ncol=2)

    figure.text(
        0.07,
        0.955,
        f"Run {run_id}  |  Does the near-tip stress field collapse?",
        fontsize=24,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(
        0.07,
        0.905,
        (
            r"All stations use the same stable-front transform "
            rf"$C_f={float(speed_fit['speed_m_per_s']) / 1e3:.3f}$ km/s. "
            "A strictly steady cohesive-crack field would collapse onto one "
            "curve at each off-fault distance."
        ),
        fontsize=13.5,
        color=MUTED,
        va="top",
    )
    figure.text(
        0.07,
        0.865,
        (
            "The partial mismatch and downstream amplitude growth diagnose "
            "non-steady wave content superposed on the propagating front; "
            "they should not be hidden by cherry-picking one station."
        ),
        fontsize=11.5,
        color=MUTED,
        va="top",
    )
    figure.text(
        0.07,
        0.035,
        (
            r"Coordinate note: the simulation uses $y$ along the fault and "
            r"$x$ normal to it. Thus $\xi=y-y_{\mathrm{tip}}$ here is the "
            r"paper's $x-x_{\mathrm{tip}}$. Stress polarity follows the "
            r"paper's positive resisting traction."
        ),
        fontsize=10.5,
        color=MUTED,
    )
    return save_figure(
        figure,
        output_dir,
        "near_fault_stress_fluctuation_collapse",
        dpi,
    )


def fit_on_fault_triangles(
    *,
    frame_time_ms: np.ndarray,
    station_tip_arrival: np.ndarray,
    speed_m_per_s: float,
    on_fault_stress: np.ndarray,
    residual_stress: np.ndarray,
    cohesive_zone_mm: float,
) -> list[dict[str, object]]:
    fits: list[dict[str, object]] = []
    for station_index, arrival in enumerate(station_tip_arrival):
        xi = -speed_m_per_s * (frame_time_ms - arrival)
        delta = on_fault_stress[:, station_index] - residual_stress[station_index]
        mask = (xi >= -cohesive_zone_mm) & (xi <= 0.0)
        fit_x = np.asarray(xi[mask], dtype=np.float64)
        fit_y = np.asarray(delta[mask], dtype=np.float64)
        order = np.argsort(fit_x)
        fit_x = fit_x[order]
        fit_y = fit_y[order]
        if len(fit_x) < 3:
            fits.append(
                {
                    "sample_xi_mm": fit_x.tolist(),
                    "sample_delta_stress_mpa": fit_y.tolist(),
                    "sample_count": int(len(fit_x)),
                    "slope_mpa_per_mm": float("nan"),
                    "intercept_mpa": float("nan"),
                    "r_squared": float("nan"),
                }
            )
            continue
        slope, intercept = np.polyfit(fit_x, fit_y, 1)
        predicted = slope * fit_x + intercept
        centered = fit_y - np.mean(fit_y)
        denominator = float(np.sum(centered**2))
        r_squared = (
            1.0 - float(np.sum((fit_y - predicted) ** 2)) / denominator
            if denominator > 0.0
            else float("nan")
        )
        fits.append(
            {
                "sample_xi_mm": fit_x.tolist(),
                "sample_delta_stress_mpa": fit_y.tolist(),
                "sample_count": int(len(fit_x)),
                "slope_mpa_per_mm": float(slope),
                "intercept_mpa": float(intercept),
                "r_squared": r_squared,
            }
        )
    return fits


def plot_on_fault_triangle_zoom(
    *,
    run_id: str,
    stations: np.ndarray,
    frame_time_ms: np.ndarray,
    station_tip_arrival: np.ndarray,
    speed_m_per_s: float,
    on_fault_stress: np.ndarray,
    residual_stress: np.ndarray,
    cohesive_zone_mm: float,
    triangle_fits: list[dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 9.0))
    figure.subplots_adjust(
        left=0.075,
        right=0.965,
        top=0.80,
        bottom=0.12,
        wspace=0.17,
        hspace=0.31,
    )
    zoom_extent = max(5.0, 3.0 * cohesive_zone_mm)
    for station_index, (axis, station) in enumerate(
        zip(axes.ravel(), stations, strict=True)
    ):
        axis.set_facecolor(PANEL)
        xi = -speed_m_per_s * (
            frame_time_ms - station_tip_arrival[station_index]
        )
        delta = (
            on_fault_stress[:, station_index]
            - residual_stress[station_index]
        )
        window = (xi >= -zoom_extent) & (xi <= zoom_extent)
        order = np.argsort(xi[window])
        plot_x = xi[window][order]
        plot_y = delta[window][order]
        axis.plot(
            plot_x,
            plot_y,
            color=ORANGE,
            linewidth=2.0,
            marker="o",
            markersize=3.8,
            label="raw saved samples",
            zorder=4,
        )

        fit = triangle_fits[station_index]
        if np.isfinite(float(fit["slope_mpa_per_mm"])):
            fit_x = np.linspace(-cohesive_zone_mm, 0.0, 120)
            fit_y = (
                float(fit["slope_mpa_per_mm"]) * fit_x
                + float(fit["intercept_mpa"])
            )
            axis.plot(
                fit_x,
                fit_y,
                color=INK,
                linewidth=2.0,
                linestyle=(0, (5, 3)),
                label="linear fit inside measured $X_c$",
                zorder=5,
            )
        add_cohesive_zone(axis, cohesive_zone_mm)
        axis.axhline(0.0, color=GRID, linewidth=1.0)
        axis.set_xlim(-zoom_extent, zoom_extent)
        axis.set_title(
            rf"Station $y={station:.0f}$ mm  |  "
            rf"$n={int(fit['sample_count'])}$, "
            rf"$R^2={float(fit['r_squared']):.3f}$",
            loc="left",
        )
        axis.set_xlabel(r"$\xi=y-y_{\mathrm{tip}}$ [mm]")
        axis.set_ylabel(r"On-fault $\Delta\tau$ [MPa]")
        axis.grid()
        axis.spines[["top", "right"]].set_visible(False)
        if station_index == 0:
            axis.legend(loc="lower left", fontsize=10.5)

    figure.text(
        0.075,
        0.955,
        f"Run {run_id}  |  Does the on-fault cohesive triangle emerge?",
        fontsize=24,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(
        0.075,
        0.905,
        (
            r"Positive resisting interface traction, no temporal smoothing or spatial "
            r"interpolation. Black fits use only raw samples in "
            r"$-X_c\leq\xi\leq0$."
        ),
        fontsize=13.5,
        color=MUTED,
        va="top",
    )
    figure.text(
        0.075,
        0.865,
        (
            rf"Measured $X_c={cohesive_zone_mm:.2f}$ mm; gold marks the "
            r"cohesive zone and red marks the $0.05D_c$ tip proxy. "
            r"Only 3-4 saved points resolve $X_c$, so shape is visible but "
            r"the fitted slope is not yet mesh-converged."
        ),
        fontsize=11.5,
        color=MUTED,
        va="top",
    )
    figure.text(
        0.075,
        0.035,
        (
            r"Fig. 2 expectation: the on-fault traction changes linearly "
            r"through the cohesive zone, while off-fault stress develops an "
            r"oscillatory waveform."
        ),
        fontsize=10.5,
        color=MUTED,
    )
    return save_figure(
        figure,
        output_dir,
        "near_fault_on_fault_triangle_zoom",
        dpi,
    )


def main() -> int:
    args = parse_args()
    data_path = args.data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_path.parent.parent / "Plot"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stations_requested = np.asarray(args.stations, dtype=np.float64)
    distances = np.asarray(args.off_fault_distances, dtype=np.float64)
    if len(stations_requested) != 4:
        raise ValueError(
            "The slide-ready station figure requires exactly four stations."
        )
    if len(distances) != 6:
        raise ValueError(
            "The figures require exactly six distances (0 through 5 mm)."
        )
    if np.any(distances < 0.0):
        raise ValueError("Off-fault distances cannot be negative.")
    zero_distance_indices = np.flatnonzero(np.isclose(distances, 0.0))
    if len(zero_distance_indices) != 1:
        raise ValueError("Exactly one 0 mm interface trace is required.")
    positive_distance_indices = np.flatnonzero(distances > 0.0)
    if len(positive_distance_indices) != 5:
        raise ValueError("Exactly five positive off-fault distances are required.")
    if not 0.0 < args.tip_slip_fraction < 1.0:
        raise ValueError("tip_slip_fraction must lie strictly between 0 and 1.")
    if args.xi_min >= args.xi_max:
        raise ValueError("xi_min must be smaller than xi_max.")
    if not 0.0 <= args.residual_start_us < args.residual_end_us:
        raise ValueError("The residual-stress window is invalid.")

    configure_style()
    run_id = data_path.parent.parent.name.split("_", maxsplit=1)[0]

    with h5py.File(data_path, "r") as h5:
        shear_time_ms, shear_indices = saved_time_ms(h5)
        contact_y = np.asarray(
            h5["interface/contact_line_y"],
            dtype=np.float64,
        )
        critical_slip = np.asarray(
            h5["interface/critical_slip_profile"],
            dtype=np.float64,
        )
        cumulative_slip = h5["interface/cumulative_slip"]

        tip_arrival = first_crossing_times(
            cumulative_slip,
            shear_indices,
            shear_time_ms,
            args.tip_slip_fraction * critical_slip,
            chunk_frames=args.arrival_chunk_frames,
        )
        half_dc_arrival = first_crossing_times(
            cumulative_slip,
            shear_indices,
            shear_time_ms,
            0.5 * critical_slip,
            chunk_frames=args.arrival_chunk_frames,
        )
        speed_fit = linear_arrival_fit(
            contact_y,
            tip_arrival,
            args.fit_start,
            args.fit_end,
        )
        half_dc_fit = linear_arrival_fit(
            contact_y,
            half_dc_arrival,
            args.fit_start,
            args.fit_end,
        )
        speed_m_per_s = float(speed_fit["speed_m_per_s"])

        station_indices = np.asarray(
            [
                int(np.argmin(np.abs(contact_y - target)))
                for target in stations_requested
            ],
            dtype=np.int64,
        )
        stations = contact_y[station_indices]
        station_tip_arrival = tip_arrival[station_indices]
        if np.any(~np.isfinite(station_tip_arrival)):
            missing = stations[~np.isfinite(station_tip_arrival)]
            raise ValueError(
                f"No tip-arrival crossing at stations {missing.tolist()}."
            )

        cohesive_zone = estimate_cohesive_zone(
            cumulative_slip,
            contact_y,
            critical_slip,
            shear_indices,
            shear_time_ms,
            half_dc_arrival,
            args.fit_start,
            args.fit_end,
        )
        cohesive_zone_mm = float(cohesive_zone["median_mm"])

        positive_distances = distances[positive_distance_indices]
        groups, bulk_probe_metadata = choose_probe_patches(
            h5,
            stations,
            positive_distances,
            args.probe_half_size,
        )

        first_needed_time = float(
            np.min(station_tip_arrival - args.xi_max / speed_m_per_s)
        )
        last_needed_time = float(
            np.max(station_tip_arrival - args.xi_min / speed_m_per_s)
        )
        needed = shear_indices[
            (shear_time_ms[shear_indices] >= first_needed_time)
            & (shear_time_ms[shear_indices] <= last_needed_time)
        ]
        if len(needed) < 2:
            raise ValueError("No saved stress frames cover the requested xi range.")
        frame_start = int(needed[0])
        frame_stop = int(needed[-1]) + 1
        bulk_stress_flat = read_probe_stress(
            h5["moving/stress"],
            frame_start,
            frame_stop,
            groups,
            chunk_frames=args.stress_chunk_frames,
        )
        interface_traction, interface_probe_metadata = read_interface_traction(
            h5,
            frame_start,
            frame_stop,
            station_indices,
            chunk_frames=args.stress_chunk_frames,
        )

    frame_time_ms = shear_time_ms[frame_start:frame_stop]
    bulk_stress = bulk_stress_flat.reshape(
        len(frame_time_ms),
        len(stations),
        len(positive_distances),
    )
    reference_index = int(np.argmin(positive_distances))
    centered_interface = interface_traction - np.mean(
        interface_traction,
        axis=0,
        keepdims=True,
    )
    centered_bulk = bulk_stress[:, :, reference_index] - np.mean(
        bulk_stress[:, :, reference_index],
        axis=0,
        keepdims=True,
    )
    interface_bulk_correlation = float(
        np.sum(centered_interface * centered_bulk, dtype=np.float64)
    )
    interface_to_bulk_orientation = (
        1.0 if interface_bulk_correlation >= 0.0 else -1.0
    )
    bulk_to_contact_orientation = interface_to_bulk_orientation

    probe_stress = np.empty(
        (len(frame_time_ms), len(stations), len(distances)),
        dtype=np.float32,
    )
    # Kammer and McLaskey plot positive resisting traction. On the moving
    # block, the signed bulk sigma_xy has the opposite polarity because its
    # fault-surface outward normal points toward +x.
    probe_stress[:, :, zero_distance_indices[0]] = interface_traction
    probe_stress[:, :, positive_distance_indices] = (
        bulk_to_contact_orientation * bulk_stress
    )

    bulk_metadata_grid = np.asarray(
        bulk_probe_metadata,
        dtype=object,
    ).reshape(len(stations), len(positive_distances))
    positive_lookup = {
        int(distance_index): local_index
        for local_index, distance_index in enumerate(positive_distance_indices)
    }
    probe_metadata: list[dict[str, object]] = []
    for station_index in range(len(stations)):
        for distance_index in range(len(distances)):
            if distance_index == int(zero_distance_indices[0]):
                probe_metadata.append(interface_probe_metadata[station_index])
            else:
                probe_metadata.append(
                    bulk_metadata_grid[
                        station_index,
                        positive_lookup[distance_index],
                    ]
                )
    common_xi = np.linspace(args.xi_min, args.xi_max, 1201)
    delta_stress = np.empty(
        (len(common_xi), len(stations), len(distances)),
        dtype=np.float64,
    )
    residual_stress = np.empty(
        (len(stations), len(distances)),
        dtype=np.float64,
    )
    pre_stress = np.empty_like(residual_stress)
    residual_start_ms = args.residual_start_us * 1e-3
    residual_end_ms = args.residual_end_us * 1e-3

    for station_index, arrival in enumerate(station_tip_arrival):
        local_time_ms = frame_time_ms - arrival
        local_xi = -speed_m_per_s * local_time_ms
        residual_mask = (
            (local_time_ms >= residual_start_ms)
            & (local_time_ms <= residual_end_ms)
        )
        pre_mask = (
            (local_time_ms >= -residual_end_ms)
            & (local_time_ms <= -residual_start_ms)
        )
        if np.count_nonzero(residual_mask) < 2:
            raise ValueError(
                f"Residual window contains fewer than two frames at y="
                f"{stations[station_index]:g} mm."
            )
        if np.count_nonzero(pre_mask) < 2:
            raise ValueError(
                f"Pre-tip window contains fewer than two frames at y="
                f"{stations[station_index]:g} mm."
            )
        order = np.argsort(local_xi)
        for distance_index in range(len(distances)):
            trace = probe_stress[:, station_index, distance_index]
            residual = float(np.mean(trace[residual_mask], dtype=np.float64))
            residual_stress[station_index, distance_index] = residual
            pre_stress[station_index, distance_index] = float(
                np.mean(trace[pre_mask], dtype=np.float64)
            )
            delta_trace = trace.astype(np.float64) - residual
            delta_stress[:, station_index, distance_index] = np.interp(
                common_xi,
                local_xi[order],
                delta_trace[order],
            )

    zero_distance_index = int(zero_distance_indices[0])
    triangle_fits = fit_on_fault_triangles(
        frame_time_ms=frame_time_ms,
        station_tip_arrival=station_tip_arrival,
        speed_m_per_s=speed_m_per_s,
        on_fault_stress=probe_stress[:, :, zero_distance_index],
        residual_stress=residual_stress[:, zero_distance_index],
        cohesive_zone_mm=cohesive_zone_mm,
    )
    station_paths = plot_station_panels(
        run_id=run_id,
        stations=stations,
        distances=distances,
        xi=common_xi,
        delta_stress=delta_stress,
        tip_arrival_ms=station_tip_arrival,
        speed_fit=speed_fit,
        half_dc_fit=half_dc_fit,
        cohesive_zone_mm=cohesive_zone_mm,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    triangle_paths = plot_on_fault_triangle_zoom(
        run_id=run_id,
        stations=stations,
        frame_time_ms=frame_time_ms,
        station_tip_arrival=station_tip_arrival,
        speed_m_per_s=speed_m_per_s,
        on_fault_stress=probe_stress[:, :, zero_distance_index],
        residual_stress=residual_stress[:, zero_distance_index],
        cohesive_zone_mm=cohesive_zone_mm,
        triangle_fits=triangle_fits,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    collapse_paths = plot_collapse_panels(
        run_id=run_id,
        stations=stations,
        distances=distances,
        xi=common_xi,
        delta_stress=delta_stress,
        speed_fit=speed_fit,
        cohesive_zone_mm=cohesive_zone_mm,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    peak_to_peak = np.ptp(delta_stress, axis=0)
    payload = {
        "run_id": run_id,
        "source_method": (
            "Kammer and McLaskey (2019), Fig. 2 and Section 3.1"
        ),
        "coordinate_definition": (
            "xi = simulation y - y_tip = -C_f * (t - t_tip); this is the "
            "paper's x - x_tip because the simulation uses y along the fault"
        ),
        "tip_arrival_definition": (
            f"first cumulative-slip crossing of "
            f"{args.tip_slip_fraction:g} * local D_c"
        ),
        "speed_fit_interval_mm": [args.fit_start, args.fit_end],
        "tip_speed_m_per_s": speed_m_per_s,
        "tip_speed_r_squared": float(speed_fit["r_squared"]),
        "half_dc_speed_m_per_s": float(half_dc_fit["speed_m_per_s"]),
        "half_dc_speed_r_squared": float(half_dc_fit["r_squared"]),
        "cohesive_zone": cohesive_zone,
        "stations_mm": stations.astype(float).tolist(),
        "tip_arrival_ms": station_tip_arrival.astype(float).tolist(),
        "off_fault_distances_mm": distances.astype(float).tolist(),
        "interface_traction_orientation_to_bulk_sigma_xy": (
            interface_to_bulk_orientation
        ),
        "bulk_sigma_xy_orientation_to_positive_contact_traction": (
            bulk_to_contact_orientation
        ),
        "stress_sign_convention": (
            "positive resisting contact traction, matching Kammer and "
            "McLaskey (2019) Fig. 2; moving-side bulk sigma_xy is reoriented"
        ),
        "interface_bulk_dynamic_correlation_sign_metric": (
            interface_bulk_correlation
        ),
        "residual_window_us": [
            args.residual_start_us,
            args.residual_end_us,
        ],
        "residual_sigma_xy_mpa": residual_stress.tolist(),
        "pre_tip_sigma_xy_mpa": pre_stress.tolist(),
        "peak_to_peak_delta_sigma_xy_mpa": peak_to_peak.tolist(),
        "residual_traction_positive_stress_mpa": residual_stress.tolist(),
        "pre_tip_traction_positive_stress_mpa": pre_stress.tolist(),
        "peak_to_peak_traction_positive_stress_mpa": peak_to_peak.tolist(),
        "on_fault_triangle_fits": triangle_fits,
        "probe_patches": probe_metadata,
        "frames_read": {
            "start": frame_start,
            "stop_exclusive": frame_stop,
            "count": frame_stop - frame_start,
        },
        "outputs": {
            "station_png": str(station_paths[0]),
            "station_pdf": str(station_paths[1]),
            "collapse_png": str(collapse_paths[0]),
            "collapse_pdf": str(collapse_paths[1]),
            "triangle_zoom_png": str(triangle_paths[0]),
            "triangle_zoom_pdf": str(triangle_paths[1]),
        },
    }
    metrics_path = output_dir / "near_fault_stress_fluctuation_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
