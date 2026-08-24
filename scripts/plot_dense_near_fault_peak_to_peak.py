from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_near_fault_stress_fluctuation import (
    first_crossing_times,
    linear_arrival_fit,
    saved_time_ms,
)


DEFAULT_DISTANCES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
PRE_RUPTURE_DISTANCES = [0.0, 1.0]
PRE_RUPTURE_COMPONENTS = ["von_mises", "sigma_xx", "sigma_yy", "sigma_xy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot dense along-fault peak-to-peak near-tip stress amplitude "
            "using the same rupture coordinate as the fluctuation-collapse plot."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output stem or PNG/PDF path. PNG, PDF, CSV, and JSON are written.",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON outputs. Defaults beside the plot.",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Write the PDF plot without an additional PNG copy.",
    )
    parser.add_argument(
        "--off-fault-distances",
        type=float,
        nargs="+",
        default=DEFAULT_DISTANCES,
    )
    parser.add_argument(
        "--station-stride",
        type=int,
        default=1,
        help="Use every Nth interface mesh station; 1 follows the mesh density.",
    )
    parser.add_argument("--fit-start", type=float, default=120.0)
    parser.add_argument("--fit-end", type=float, default=440.0)
    parser.add_argument("--tip-slip-fraction", type=float, default=0.05)
    parser.add_argument("--xi-min", type=float, default=-80.0)
    parser.add_argument("--xi-max", type=float, default=80.0)
    parser.add_argument("--probe-half-size", type=float, default=0.5)
    parser.add_argument("--arrival-chunk-frames", type=int, default=2048)
    parser.add_argument("--stress-chunk-frames", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def output_paths(
    input_path: Path,
    requested: Path | None,
    stats_dir: Path | None,
) -> dict[str, Path]:
    if requested is None:
        stem = (
            input_path.parent.parent
            / "Plot"
            / "near_fault_peak_to_peak_amplitude_along_fault_dense"
        )
    elif requested.suffix.lower() in {".png", ".pdf", ".csv", ".json"}:
        stem = requested.with_suffix("")
    else:
        stem = requested
    resolved_stats_dir = (
        stats_dir.expanduser().resolve() if stats_dir is not None else stem.parent
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    resolved_stats_dir.mkdir(parents=True, exist_ok=True)
    return {
        "png": stem.with_suffix(".png"),
        "pdf": stem.with_suffix(".pdf"),
        "csv": resolved_stats_dir / f"{stem.name}.csv",
        "json": resolved_stats_dir / f"{stem.name}.json",
    }


def sampled_station_indices(station_count: int, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("station_stride must be positive.")
    indices = np.arange(0, station_count, stride, dtype=np.int64)
    if indices[-1] != station_count - 1:
        indices = np.append(indices, station_count - 1)
    return indices


def resolved_fit_interval(
    coordinate: np.ndarray,
    arrival_time_ms: np.ndarray,
    preferred_start: float,
    preferred_end: float,
) -> tuple[float, float]:
    finite = np.isfinite(arrival_time_ms)
    preferred = (
        finite
        & (coordinate >= preferred_start)
        & (coordinate <= preferred_end)
    )
    if np.count_nonzero(preferred) >= 3:
        fit_start = float(preferred_start)
        fit_end = float(preferred_end)
    else:
        available = coordinate[finite]
        if len(available) < 3:
            raise ValueError("At least three rupture arrivals are required.")
        trim = min(max(0, len(available) // 10), (len(available) - 3) // 2)
        fit_start = float(available[trim])
        fit_end = float(available[-trim - 1])

    candidate = (
        finite & (coordinate >= fit_start) & (coordinate <= fit_end)
    )
    candidate_coordinate = coordinate[candidate]
    candidate_arrival = arrival_time_ms[candidate]
    peak_coordinate = float(
        candidate_coordinate[int(np.argmax(candidate_arrival))]
    )
    spacing = float(np.median(np.diff(candidate_coordinate)))
    if peak_coordinate >= fit_start + 2.0 * spacing:
        trimmed = candidate & (coordinate <= peak_coordinate)
        if np.count_nonzero(trimmed) >= 3:
            # A downstream decrease in arrival time indicates a reverse front.
            fit_end = min(fit_end, peak_coordinate)
    return fit_start, fit_end


def bulk_probe_groups(
    h5: h5py.File,
    stations: np.ndarray,
    distances: np.ndarray,
    half_size: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    if half_size <= 0.0:
        raise ValueError("probe_half_size must be positive.")
    coords = np.asarray(h5["moving/coords"], dtype=np.float64)
    plot_elements = np.asarray(h5["moving/plot_elements"], dtype=np.int64)
    parent_elements = np.asarray(
        h5["moving/plot_parent_elements"], dtype=np.int64
    )
    centers = coords[plot_elements].mean(axis=1)
    fault_x = float(np.max(coords[:, 0]))
    targets = np.asarray(
        [
            (fault_x - float(distance), float(station))
            for station in stations
            for distance in distances
        ],
        dtype=np.float64,
    )

    tree = cKDTree(centers)
    tolerance = max(1e-8, half_size * 1e-7)
    selected_plot_groups = tree.query_ball_point(
        targets,
        r=half_size + tolerance,
        p=np.inf,
        workers=1,
    )
    groups: list[np.ndarray] = []
    realized_station = np.empty(len(targets), dtype=np.float64)
    realized_distance = np.empty(len(targets), dtype=np.float64)
    for index, selected in enumerate(selected_plot_groups):
        selected_plot = np.asarray(selected, dtype=np.int64)
        if not len(selected_plot):
            _, nearest = tree.query(targets[index], k=1, workers=1)
            selected_plot = np.asarray([int(nearest)], dtype=np.int64)
        groups.append(np.unique(parent_elements[selected_plot]))
        selected_centers = centers[selected_plot]
        realized_station[index] = float(np.mean(selected_centers[:, 1]))
        realized_distance[index] = float(
            fault_x - np.mean(selected_centers[:, 0])
        )
    return groups, realized_station, realized_distance


def group_positions(
    groups: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    unique_elements = np.unique(np.concatenate(groups))
    positions = {
        int(element): index for index, element in enumerate(unique_elements)
    }
    mapped = [
        np.asarray([positions[int(element)] for element in group], dtype=np.int64)
        for group in groups
    ]
    return unique_elements, mapped


def von_mises_2d(stress: np.ndarray) -> np.ndarray:
    sigma_xx = stress[..., 0, 0]
    sigma_yy = stress[..., 1, 1]
    sigma_xy = 0.5 * (stress[..., 0, 1] + stress[..., 1, 0])
    return np.sqrt(
        np.maximum(
            sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3.0 * sigma_xy**2,
            0.0,
        )
    )


def shear_displacement_stop_frame(
    frame_indices: np.ndarray,
    applied_displacement: np.ndarray,
) -> int:
    values = np.asarray(applied_displacement[frame_indices], dtype=np.float64)
    value_range = float(np.ptp(values))
    tolerance = max(1e-12, value_range * 1e-9)
    changing_intervals = np.flatnonzero(np.abs(np.diff(values)) > tolerance)
    if not len(changing_intervals):
        raise ValueError(
            "Applied shear displacement does not change during the saved shear phase."
        )
    local_stop = int(changing_intervals[-1]) + 1
    return int(frame_indices[local_stop])


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    finite = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(finite) < 3:
        return None
    first_finite = first[finite]
    second_finite = second[finite]
    if np.std(first_finite) == 0.0 or np.std(second_finite) == 0.0:
        return None
    return float(np.corrcoef(first_finite, second_finite)[0, 1])


def read_pre_rupture_bulk_stress(
    h5: h5py.File,
    *,
    frame_index: int,
    stations: np.ndarray,
    distances: np.ndarray,
    half_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups, realized_station, realized_distance = bulk_probe_groups(
        h5,
        stations,
        distances,
        half_size,
    )
    unique_elements, mapped_groups = group_positions(groups)
    selected_stress = np.asarray(
        h5["moving/stress"][frame_index, unique_elements, :, :],
        dtype=np.float64,
    )
    tensors = np.empty(
        (len(stations), len(distances), 2, 2),
        dtype=np.float64,
    )
    for group_index, positions in enumerate(mapped_groups):
        station = group_index // len(distances)
        distance = group_index % len(distances)
        tensors[station, distance] = np.mean(
            selected_stress[positions],
            axis=0,
            dtype=np.float64,
        )

    components = np.empty(
        (len(stations), len(distances), len(PRE_RUPTURE_COMPONENTS)),
        dtype=np.float64,
    )
    components[:, :, 0] = von_mises_2d(tensors)
    components[:, :, 1] = tensors[:, :, 0, 0]
    components[:, :, 2] = tensors[:, :, 1, 1]
    components[:, :, 3] = 0.5 * (
        tensors[:, :, 0, 1] + tensors[:, :, 1, 0]
    )
    return components, realized_station, realized_distance


def update_extrema(
    running_min: np.ndarray,
    running_max: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
) -> None:
    finite_valid = valid & np.isfinite(values)
    local_min = np.min(
        np.where(finite_valid, values, np.inf),
        axis=0,
    )
    local_max = np.max(
        np.where(finite_valid, values, -np.inf),
        axis=0,
    )
    np.minimum(running_min, local_min, out=running_min)
    np.maximum(running_max, local_max, out=running_max)


def stream_peak_to_peak(
    h5: h5py.File,
    *,
    frame_start: int,
    frame_stop: int,
    time_ms: np.ndarray,
    window_start_ms: np.ndarray,
    window_end_ms: np.ndarray,
    station_indices: np.ndarray,
    bulk_groups: list[np.ndarray],
    positive_distance_count: int,
    chunk_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if chunk_frames < 1:
        raise ValueError("stress_chunk_frames must be positive.")
    station_count = len(station_indices)
    master_nodes = np.asarray(
        h5["interface/master_nodes"][station_indices], dtype=np.int64
    )
    slave_nodes = np.asarray(
        h5["interface/slave_nodes"][station_indices], dtype=np.int64
    )
    tangential_penalty = np.asarray(
        h5["interface/tangential_penalty_profile"][station_indices],
        dtype=np.float64,
    )
    master_order = np.argsort(master_nodes)
    slave_order = np.argsort(slave_nodes)
    master_restore = np.argsort(master_order)
    slave_restore = np.argsort(slave_order)

    unique_elements, mapped_groups = group_positions(bulk_groups)
    interface_min = np.full(station_count, np.inf, dtype=np.float64)
    interface_max = np.full(station_count, -np.inf, dtype=np.float64)
    bulk_min = np.full(
        (station_count, positive_distance_count), np.inf, dtype=np.float64
    )
    bulk_max = np.full(
        (station_count, positive_distance_count), -np.inf, dtype=np.float64
    )
    sample_count = np.zeros(station_count, dtype=np.int64)

    for start in range(frame_start, frame_stop, chunk_frames):
        stop = min(start + chunk_frames, frame_stop)
        chunk_time = time_ms[start:stop]
        valid = (
            (chunk_time[:, None] >= window_start_ms[None, :])
            & (chunk_time[:, None] <= window_end_ms[None, :])
        )
        sample_count += np.count_nonzero(valid, axis=0)

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
        interface_traction = (
            moving_tangential - stationary_tangential - plastic_slip
        ) * tangential_penalty[None, :]
        update_extrema(
            interface_min,
            interface_max,
            interface_traction,
            valid,
        )

        selected_stress = np.asarray(
            h5["moving/stress"][start:stop, unique_elements, 0, 1],
            dtype=np.float32,
        )
        bulk_stress = np.empty(
            (stop - start, station_count, positive_distance_count),
            dtype=np.float32,
        )
        for group_index, positions in enumerate(mapped_groups):
            station = group_index // positive_distance_count
            distance = group_index % positive_distance_count
            bulk_stress[:, station, distance] = np.mean(
                selected_stress[:, positions], axis=1, dtype=np.float64
            )
        update_extrema(
            bulk_min,
            bulk_max,
            bulk_stress,
            valid[:, :, None],
        )

        if start == frame_start or stop == frame_stop or (
            start - frame_start
        ) % (20 * chunk_frames) == 0:
            print(
                f"Processed stress frames {start}:{stop} / {frame_stop - 1}",
                flush=True,
            )

    if np.any(sample_count < 2):
        missing = np.flatnonzero(sample_count < 2)
        raise ValueError(
            "Fewer than two saved frames in the xi window at station indices "
            f"{missing.tolist()}."
        )
    peak_to_peak = np.column_stack(
        [interface_max - interface_min, bulk_max - bulk_min]
    )
    return peak_to_peak, sample_count


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_amplitude(
    stations: np.ndarray,
    distances: np.ndarray,
    peak_to_peak: np.ndarray,
    pre_rupture_distances: np.ndarray,
    pre_rupture_stress: np.ndarray,
    *,
    fit_start: float,
    fit_end: float,
    paths: dict[str, Path],
    dpi: int,
    write_png: bool,
) -> None:
    configure_style()
    figure, axis = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    stress_axis = axis.twinx()
    colors = plt.get_cmap("viridis")(
        np.linspace(0.06, 0.92, len(distances))
    )
    axis.axvspan(
        fit_start,
        fit_end,
        color="#7F7F7F",
        alpha=0.07,
        linewidth=0.0,
        zorder=0,
    )
    for index, (distance, color) in enumerate(zip(distances, colors, strict=True)):
        axis.plot(
            stations,
            peak_to_peak[:, index],
            color=color,
            linewidth=1.25,
            marker="o",
            markersize=2.1,
            markevery=max(1, len(stations) // 25),
            markeredgewidth=0.0,
            label=rf"${distance:g}$ mm",
            zorder=3,
        )
    stress_colors = {
        "von_mises": "#202020",
        "sigma_xx": "#2166AC",
        "sigma_yy": "#B2182B",
        "sigma_xy": "#D6604D",
    }
    stress_labels = {
        "von_mises": r"$\sigma_{\mathrm{VM}}$",
        "sigma_xx": r"$\sigma_{xx}$",
        "sigma_yy": r"$\sigma_{yy}$",
        "sigma_xy": r"$\sigma_{xy}$",
    }
    distance_styles = ["-", (0, (4, 2))]
    for distance_index, distance in enumerate(pre_rupture_distances):
        for component_index, component in enumerate(PRE_RUPTURE_COMPONENTS):
            stress_axis.plot(
                stations,
                pre_rupture_stress[:, distance_index, component_index],
                color=stress_colors[component],
                linestyle=distance_styles[distance_index],
                linewidth=0.9,
                alpha=0.72,
                label=(
                    rf"{stress_labels[component]}, "
                    rf"$d_\perp={distance:g}$ mm"
                ),
                zorder=1,
            )
    axis.set_xlim(float(np.min(stations)), float(np.max(stations)))
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(r"Station along fault, $y$ [mm]")
    axis.set_ylabel("Peak-to-peak stress fluctuation [MPa]")
    stress_axis.set_ylabel("Stress when shear displacement stops [MPa]")
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.75)
    amplitude_legend = axis.legend(
        title=r"Distance from fault, $d_\perp$",
        ncol=2,
        frameon=False,
        loc="upper left",
        handlelength=2.2,
        columnspacing=1.2,
    )
    stress_axis.legend(
        title="Bulk stress at displacement stop",
        ncol=2,
        frameon=False,
        loc="upper right",
        fontsize=7.8,
        title_fontsize=8.5,
        handlelength=2.4,
        columnspacing=0.9,
    )
    axis.add_artist(amplitude_legend)
    axis.set_zorder(stress_axis.get_zorder() + 1)
    axis.patch.set_visible(False)
    if write_png:
        figure.savefig(paths["png"], dpi=dpi, bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(figure)


def write_csv(
    path: Path,
    stations: np.ndarray,
    arrivals: np.ndarray,
    sample_count: np.ndarray,
    distances: np.ndarray,
    peak_to_peak: np.ndarray,
    pre_rupture_distances: np.ndarray,
    pre_rupture_stress: np.ndarray,
) -> None:
    headers = [
        "station_y_mm",
        "tip_arrival_ms",
        "samples_in_xi_window",
        *[
            f"peak_to_peak_mpa_{distance:g}mm_from_fault"
            for distance in distances
        ],
        *[
            f"shear_displacement_stop_{component}_mpa_{distance:g}mm_from_fault"
            for distance in pre_rupture_distances
            for component in PRE_RUPTURE_COMPONENTS
        ],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for station, arrival, count, amplitudes, pre_stress in zip(
            stations,
            arrivals,
            sample_count,
            peak_to_peak,
            pre_rupture_stress,
            strict=True,
        ):
            writer.writerow(
                [
                    f"{station:.9g}",
                    f"{arrival:.12g}",
                    int(count),
                    *[f"{value:.12g}" for value in amplitudes],
                    *[
                        f"{value:.12g}"
                        for distance_values in pre_stress
                        for value in distance_values
                    ],
                ]
            )


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    paths = output_paths(input_path, args.output, args.stats_dir)
    distances = np.asarray(args.off_fault_distances, dtype=np.float64)
    pre_rupture_distances = np.asarray(
        PRE_RUPTURE_DISTANCES,
        dtype=np.float64,
    )
    if len(distances) < 1 or np.any(distances < 0.0):
        raise ValueError("off_fault_distances must be non-negative.")
    zero = np.flatnonzero(np.isclose(distances, 0.0))
    if len(zero) != 1 or int(zero[0]) != 0:
        raise ValueError("off_fault_distances must start with exactly one 0 mm trace.")
    positive_distances = distances[1:]
    if np.any(positive_distances <= 0.0):
        raise ValueError("Positive off-fault distances must follow 0 mm.")
    if not 0.0 < args.tip_slip_fraction < 1.0:
        raise ValueError("tip_slip_fraction must lie between zero and one.")
    if args.xi_min >= args.xi_max:
        raise ValueError("xi_min must be smaller than xi_max.")

    with h5py.File(input_path, "r") as h5:
        time_ms, shear_indices = saved_time_ms(h5)
        contact_y_all = np.asarray(
            h5["interface/contact_line_y"], dtype=np.float64
        )
        critical_slip = np.asarray(
            h5["interface/critical_slip_profile"], dtype=np.float64
        )
        arrival_all = first_crossing_times(
            h5["interface/cumulative_slip"],
            shear_indices,
            time_ms,
            args.tip_slip_fraction * critical_slip,
            chunk_frames=args.arrival_chunk_frames,
        )
        fit_start, fit_end = resolved_fit_interval(
            contact_y_all,
            arrival_all,
            args.fit_start,
            args.fit_end,
        )
        speed_fit = linear_arrival_fit(
            contact_y_all,
            arrival_all,
            fit_start,
            fit_end,
        )
        speed_m_per_s = float(speed_fit["speed_m_per_s"])
        candidate_station_indices = sampled_station_indices(
            len(contact_y_all), args.station_stride
        )
        candidate_arrivals = arrival_all[candidate_station_indices]
        finite_arrivals = np.isfinite(candidate_arrivals)
        station_indices = candidate_station_indices[finite_arrivals]
        if len(station_indices) < 3:
            raise ValueError(
                "Dense amplitude analysis requires rupture arrivals at least "
                "three sampled interface stations."
            )
        stations = contact_y_all[station_indices]
        arrivals = arrival_all[station_indices]
        history_columns = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in h5["history_columns"][...]
        ]
        if "applied_shear_displacement" not in history_columns:
            raise ValueError(
                "History does not contain applied_shear_displacement."
            )
        history = np.asarray(h5["history"], dtype=np.float64)
        applied_displacement = history[
            :,
            history_columns.index("applied_shear_displacement"),
        ]
        pre_rupture_frame = shear_displacement_stop_frame(
            shear_indices,
            applied_displacement,
        )
        pre_rupture_time_ms = float(time_ms[pre_rupture_frame])
        displacement_stop_absolute_time_ms = float(
            history[pre_rupture_frame, history_columns.index("time")] * 1e3
        )
        displacement_stop_value = float(applied_displacement[pre_rupture_frame])
        (
            pre_rupture_stress,
            pre_realized_station,
            pre_realized_distance,
        ) = read_pre_rupture_bulk_stress(
            h5,
            frame_index=pre_rupture_frame,
            stations=stations,
            distances=pre_rupture_distances,
            half_size=args.probe_half_size,
        )

        window_start_ms = arrivals - args.xi_max / speed_m_per_s
        window_end_ms = arrivals - args.xi_min / speed_m_per_s
        in_global_window = shear_indices[
            (time_ms[shear_indices] >= float(np.min(window_start_ms)))
            & (time_ms[shear_indices] <= float(np.max(window_end_ms)))
        ]
        if len(in_global_window) < 2:
            raise ValueError("No saved stress frames cover the requested xi window.")
        frame_start = max(int(shear_indices[0]), int(in_global_window[0]) - 1)
        frame_stop = min(
            int(shear_indices[-1]) + 1,
            int(in_global_window[-1]) + 2,
        )
        groups, realized_station, realized_distance = bulk_probe_groups(
            h5,
            stations,
            positive_distances,
            args.probe_half_size,
        )
        peak_to_peak, sample_count = stream_peak_to_peak(
            h5,
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_ms=time_ms,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            station_indices=station_indices,
            bulk_groups=groups,
            positive_distance_count=len(positive_distances),
            chunk_frames=args.stress_chunk_frames,
        )

    plot_amplitude(
        stations,
        distances,
        peak_to_peak,
        pre_rupture_distances,
        pre_rupture_stress,
        fit_start=fit_start,
        fit_end=fit_end,
        paths=paths,
        dpi=args.dpi,
        write_png=not args.pdf_only,
    )
    write_csv(
        paths["csv"],
        stations,
        arrivals,
        sample_count,
        distances,
        peak_to_peak,
        pre_rupture_distances,
        pre_rupture_stress,
    )
    fit_mask = (stations >= fit_start) & (stations <= fit_end)
    correlation_metrics = {
        f"amplitude_{amplitude_distance:g}mm__stopstress_{pre_distance:g}mm_{component}": {
            "raw_pearson_r": pearson_correlation(
                peak_to_peak[fit_mask, amplitude_index],
                pre_rupture_stress[
                    fit_mask,
                    pre_distance_index,
                    component_index,
                ],
            ),
            "first_difference_pearson_r": pearson_correlation(
                np.diff(peak_to_peak[fit_mask, amplitude_index]),
                np.diff(
                    pre_rupture_stress[
                        fit_mask,
                        pre_distance_index,
                        component_index,
                    ]
                ),
            ),
        }
        for amplitude_index, amplitude_distance in enumerate(distances)
        for pre_distance_index, pre_distance in enumerate(pre_rupture_distances)
        for component_index, component in enumerate(PRE_RUPTURE_COMPONENTS)
    }
    metrics = {
        "input": str(input_path),
        "run_id": input_path.parent.parent.name.split("_", maxsplit=1)[0],
        "definition": (
            "Peak-to-peak = max(stress) - min(stress) over each local "
            "rupture-tip window. Subtracting the 40-50 us residual baseline "
            "does not change this amplitude."
        ),
        "rupture_coordinate": "xi = -C_f * (t - t_tip)",
        "xi_window_mm": [args.xi_min, args.xi_max],
        "tip_arrival_definition": (
            f"first cumulative-slip crossing of {args.tip_slip_fraction:g} * local D_c"
        ),
        "speed_fit_interval_mm": [fit_start, fit_end],
        "speed_m_per_s": speed_m_per_s,
        "speed_fit_r_squared": float(speed_fit["r_squared"]),
        "station_count": int(len(stations)),
        "interface_station_count": int(len(contact_y_all)),
        "ruptured_station_fraction": float(
            len(station_indices) / len(candidate_station_indices)
        ),
        "station_stride_in_mesh_nodes": int(args.station_stride),
        "station_spacing_mm_median": float(np.median(np.diff(stations))),
        "off_fault_distances_mm": distances.astype(float).tolist(),
        "shear_displacement_stop_bulk_stress": {
            "definition": (
                "Moving-block bulk stress at the first saved frame on the "
                "final plateau after applied shear displacement stops changing."
            ),
            "frame": pre_rupture_frame,
            "time_ms_from_shear_start": pre_rupture_time_ms,
            "absolute_time_ms": displacement_stop_absolute_time_ms,
            "applied_shear_displacement_mm": displacement_stop_value,
            "off_fault_distances_mm": pre_rupture_distances.astype(float).tolist(),
            "components": PRE_RUPTURE_COMPONENTS,
            "von_mises_definition": (
                "sqrt(sigma_xx^2 - sigma_xx*sigma_yy + sigma_yy^2 "
                "+ 3*sigma_xy^2)"
            ),
            "realized_station_range_mm": [
                float(np.min(pre_realized_station)),
                float(np.max(pre_realized_station)),
            ],
            "realized_distance_range_mm": [
                float(np.min(pre_realized_distance)),
                float(np.max(pre_realized_distance)),
            ],
            "range_mpa_by_distance_and_component": {
                f"{distance:g}mm_{component}": [
                    float(np.min(pre_rupture_stress[:, distance_index, component_index])),
                    float(np.max(pre_rupture_stress[:, distance_index, component_index])),
                ]
                for distance_index, distance in enumerate(pre_rupture_distances)
                for component_index, component in enumerate(PRE_RUPTURE_COMPONENTS)
            },
            "amplitude_correlation": {
                "station_interval_mm": [fit_start, fit_end],
                "interpretation": (
                    "Raw Pearson r measures shared long-wavelength spatial "
                    "variation. First-difference r tests point-to-point local "
                    "covariation and should not be interpreted as causation."
                ),
                "values": correlation_metrics,
            },
        },
        "probe_half_size_mm": args.probe_half_size,
        "bulk_probe_realized_station_range_mm": [
            float(np.min(realized_station)),
            float(np.max(realized_station)),
        ],
        "bulk_probe_realized_distance_range_mm": [
            float(np.min(realized_distance)),
            float(np.max(realized_distance)),
        ],
        "frames_read": {
            "start": frame_start,
            "stop_exclusive": frame_stop,
            "count": frame_stop - frame_start,
        },
        "peak_to_peak_range_mpa_by_distance": {
            f"{distance:g}": [
                float(np.min(peak_to_peak[:, index])),
                float(np.max(peak_to_peak[:, index])),
            ]
            for index, distance in enumerate(distances)
        },
        "outputs": {
            name: str(path)
            for name, path in paths.items()
            if name != "png" or not args.pdf_only
        },
    }
    paths["json"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
