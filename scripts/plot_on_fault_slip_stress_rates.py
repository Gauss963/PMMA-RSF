from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from plot_near_fault_stress_fluctuation import (
    BACKGROUND,
    GOLD,
    GRID,
    INK,
    MUTED,
    PANEL,
    RED,
    configure_style,
    first_crossing_times,
    saved_time_ms,
)
from plot_sigma_xy_probe_traces import (
    _critical_slip_profile,
    _output_pair,
    _read_interface_traction_frames,
    _resolve_stations,
    _run_id,
)


STRESS_BLUE = "#2676AE"
STRESS_RAW = "#9EC9E2"
SLIP_GREEN = "#2A9D3F"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot on-fault slip rate and signed shear-stress rate at selected "
            "stations, aligned to each local rupture arrival."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-metrics", type=Path, default=None)
    parser.add_argument(
        "--y-points",
        type=float,
        nargs="+",
        default=[160.0, 240.0, 320.0, 400.0],
    )
    parser.add_argument("--time-before-ms", type=float, default=0.25)
    parser.add_argument("--time-after-ms", type=float, default=1.75)
    parser.add_argument("--tip-slip-fraction", type=float, default=0.05)
    parser.add_argument(
        "--stress-smoothing-us",
        type=float,
        default=200.0,
        help="Gaussian derivative standard deviation for stress rate [us].",
    )
    parser.add_argument(
        "--slip-smoothing-us",
        type=float,
        default=50.0,
        help="Gaussian derivative standard deviation for slip rate [us].",
    )
    parser.add_argument(
        "--show-raw-stress-rate",
        action="store_true",
        help="Overlay the unsmoothed stress-rate derivative as a faint line.",
    )
    parser.add_argument(
        "--stress-rate-linthresh",
        type=float,
        default=1.0,
        help="Central linear half-width of the signed symlog stress-rate axis [MPa/ms].",
    )
    parser.add_argument("--arrival-chunk-frames", type=int, default=2048)
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def _default_output(input_path: Path) -> Path:
    run_dir = (
        input_path.parent.parent
        if input_path.parent.name == "data"
        else input_path.parent
    )
    return run_dir / "Plot" / "on_fault_slip_stress_rates_by_station.png"


def _save_figure(
    figure: plt.Figure,
    output_path: Path,
    dpi: int,
) -> tuple[Path, Path]:
    png_path, pdf_path = _output_pair(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=dpi)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def plot_on_fault_rates(
    *,
    input_path: Path,
    output_path: Path,
    output_metrics: Path | None,
    y_points: list[float],
    time_before_ms: float,
    time_after_ms: float,
    tip_slip_fraction: float,
    stress_smoothing_us: float,
    slip_smoothing_us: float,
    show_raw_stress_rate: bool,
    stress_rate_linthresh: float,
    arrival_chunk_frames: int,
    dpi: int,
) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if time_before_ms < 0.0 or time_after_ms <= 0.0:
        raise ValueError("The local time window is invalid.")
    if not 0.0 < tip_slip_fraction < 1.0:
        raise ValueError("tip_slip_fraction must lie between zero and one.")
    if stress_smoothing_us <= 0.0 or slip_smoothing_us <= 0.0:
        raise ValueError("Gaussian derivative scales must be positive.")
    if stress_rate_linthresh <= 0.0:
        raise ValueError("stress_rate_linthresh must be positive.")

    requested_y = np.asarray(y_points, dtype=np.float64)
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
            raise ValueError(f"No rupture arrival at y={missing.tolist()} mm.")
        full_fault_arrival_ms = float(np.nanmax(arrival_all))

        read_start_ms = float(np.min(arrivals_ms) - time_before_ms)
        read_end_ms = float(np.max(arrivals_ms) + time_after_ms)
        frame_mask = (
            (shear_time_ms[shear_indices] >= read_start_ms)
            & (shear_time_ms[shear_indices] <= read_end_ms)
        )
        frame_indices = shear_indices[frame_mask]
        if len(frame_indices) < 3:
            raise ValueError("The selected time window contains too few frames.")
        if np.any(np.diff(frame_indices) != 1):
            raise ValueError("Rate calculation requires contiguous saved frames.")

        interface_traction, interface_metadata = _read_interface_traction_frames(
            h5,
            frame_indices,
            len(frame_indices),
            station_indices,
        )
        cumulative_slip = np.asarray(
            h5["interface/cumulative_slip"][
                int(frame_indices[0]) : int(frame_indices[-1]) + 1,
                station_indices,
            ],
            dtype=np.float64,
        )

    frame_time_ms = shear_time_ms[frame_indices]
    frame_time_s = frame_time_ms * 1e-3
    saved_dt_ms = float(np.median(np.diff(frame_time_ms)))
    saved_dt_us = saved_dt_ms * 1e3
    stress_rate = np.gradient(
        interface_traction.astype(np.float64),
        frame_time_ms,
        axis=0,
        edge_order=2,
    )
    slip_rate = np.maximum(
        np.gradient(
            cumulative_slip,
            frame_time_s,
            axis=0,
            edge_order=2,
        )
        * 1e-3,
        0.0,
    )
    stress_smoothing_samples = max(
        1.0,
        stress_smoothing_us / saved_dt_us,
    )
    slip_smoothing_samples = max(
        1.0,
        slip_smoothing_us / saved_dt_us,
    )
    stress_rate_smooth = gaussian_filter1d(
        interface_traction.astype(np.float64),
        sigma=stress_smoothing_samples,
        order=1,
        axis=0,
        mode="nearest",
    ) / saved_dt_ms
    slip_rate_smooth = np.maximum(
        gaussian_filter1d(
            cumulative_slip,
            sigma=slip_smoothing_samples,
            order=1,
            axis=0,
            mode="nearest",
        )
        / (saved_dt_ms * 1e-3)
        * 1e-3,
        0.0,
    )

    configure_style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.94,
        top=0.76,
        bottom=0.115,
        wspace=0.20,
        hspace=0.28,
    )
    axes_flat = axes.ravel()
    slip_axes: list[plt.Axes] = []
    visible_slip_values: list[np.ndarray] = []
    visible_stress_values: list[np.ndarray] = []
    for station_index, arrival_ms in enumerate(arrivals_ms):
        local_time_ms = frame_time_ms - arrival_ms
        visible = (
            (local_time_ms >= -time_before_ms)
            & (local_time_ms <= time_after_ms)
        )
        visible_slip_values.append(slip_rate_smooth[visible, station_index])
        visible_stress_values.append(
            stress_rate_smooth[visible, station_index]
        )
    slip_max = max(float(np.nanmax(values)) for values in visible_slip_values)
    stress_abs_max = max(
        float(np.nanmax(np.abs(values))) for values in visible_stress_values
    )
    stress_limit = max(
        stress_rate_linthresh * 10.0,
        1.08 * stress_abs_max,
    )

    legend_handles: list[plt.Line2D] = []
    event_handles: list[plt.Line2D] = []
    metrics: list[dict[str, float]] = []
    for station_index, (axis, station, arrival_ms) in enumerate(
        zip(axes_flat, stations, arrivals_ms, strict=True)
    ):
        local_time_ms = frame_time_ms - arrival_ms
        full_fault_local_ms = full_fault_arrival_ms - arrival_ms
        slip_axis = axis.twinx()
        slip_axes.append(slip_axis)
        axis.set_facecolor(PANEL)
        slip_axis.set_facecolor("none")

        if show_raw_stress_rate:
            axis.plot(
                local_time_ms,
                stress_rate[:, station_index],
                color=STRESS_RAW,
                linewidth=0.45,
                alpha=0.12,
                zorder=2,
            )
        stress_line = axis.plot(
            local_time_ms,
            stress_rate_smooth[:, station_index],
            color=STRESS_BLUE,
            linewidth=1.0,
            zorder=4,
            label=r"stress rate $d\tau/dt$",
        )[0]
        slip_line = slip_axis.plot(
            local_time_ms,
            slip_rate_smooth[:, station_index],
            color=SLIP_GREEN,
            linewidth=1.05,
            zorder=5,
            label=r"slip rate $d\delta/dt$",
        )[0]
        tip_line = axis.axvline(
            0.0,
            color=RED,
            linestyle=(0, (3, 3)),
            linewidth=1.2,
            label="local rupture arrival",
        )
        full_line = axis.axvline(
            full_fault_local_ms,
            color=GOLD,
            linestyle=(0, (4, 3)),
            linewidth=1.25,
            label="full-fault arrival",
        )
        axis.axhline(0.0, color=GRID, linewidth=0.9)

        axis.set_xlim(-time_before_ms, time_after_ms)
        axis.set_ylim(-stress_limit, stress_limit)
        axis.set_yscale(
            "symlog",
            linthresh=stress_rate_linthresh,
            linscale=1.0,
            base=10,
        )
        slip_axis.set_ylim(0.0, max(1e-12, 1.08 * slip_max))
        axis.set_title(
            rf"Station $y={station:.0f}$ mm  |  "
            rf"$t_{{tip}}={arrival_ms:.4f}$ ms",
            loc="left",
            pad=7,
        )
        axis.grid()
        axis.spines[["top", "right"]].set_visible(False)
        slip_axis.spines[["top", "left"]].set_visible(False)
        slip_axis.spines["right"].set_color(SLIP_GREEN)
        slip_axis.tick_params(axis="y", colors=SLIP_GREEN)

        if station_index % 2 == 0:
            axis.set_ylabel(r"Stress rate $d\tau/dt$ [MPa/ms]", color=STRESS_BLUE)
            axis.tick_params(axis="y", colors=STRESS_BLUE)
        else:
            axis.tick_params(axis="y", labelleft=False)
        if station_index % 2 == 1:
            slip_axis.set_ylabel(r"Slip rate $d\delta/dt$ [m/s]", color=SLIP_GREEN)
        else:
            slip_axis.tick_params(axis="y", labelright=False)
        if station_index >= 2:
            axis.set_xlabel(
                r"Time relative to local rupture arrival, $t-t_{tip}$ [ms]"
            )

        visible = (
            (local_time_ms >= -time_before_ms)
            & (local_time_ms <= time_after_ms)
        )
        metrics.append(
            {
                "station_y_mm": float(station),
                "tip_arrival_ms": float(arrival_ms),
                "peak_slip_rate_m_per_s": float(
                    np.max(slip_rate_smooth[visible, station_index])
                ),
                "peak_abs_stress_rate_mpa_per_ms": float(
                    np.max(np.abs(stress_rate_smooth[visible, station_index]))
                ),
            }
        )
        if station_index == 0:
            legend_handles = [stress_line, slip_line]
            event_handles = [tip_line, full_line]

    figure.text(
        0.075,
        0.985,
        f"On-fault slip and stress rates ({_run_id(input_path)})",
        fontsize=10.5,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    figure.text(
        0.075,
        0.951,
        (
            r"Interface contact traction and cumulative frictional slip; "
            rf"Gaussian derivatives: $\sigma_\tau={stress_smoothing_us:g}$ "
            rf"$\mu$s and $\sigma_\delta={slip_smoothing_us:g}$ $\mu$s."
        ),
        fontsize=7.2,
        color=MUTED,
        va="top",
    )
    figure.legend(
        handles=legend_handles + event_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.865),
        ncol=4,
        frameon=False,
        fontsize=6.6,
        handlelength=1.9,
        columnspacing=1.0,
    )
    figure.text(
        0.075,
        0.022,
        (
            r"Stress rate is signed and uses a symmetric-log axis; slip rate "
            r"is the non-negative derivative of cumulative frictional slip. "
            + (
                r"Faint blue shows the unsmoothed stress-rate derivative."
                if show_raw_stress_rate
                else r"Raw derivative is omitted to expose the event-scale response."
            )
        ),
        fontsize=6.3,
        color=MUTED,
    )

    png_path, pdf_path = _save_figure(figure, output_path, dpi)
    result: dict[str, object] = {
        "run_id": _run_id(input_path),
        "input": str(input_path.resolve()),
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
        },
        "stations_y_mm": stations.astype(float).tolist(),
        "time_window_ms": [-float(time_before_ms), float(time_after_ms)],
        "tip_arrival_definition": (
            f"first cumulative-slip crossing of {tip_slip_fraction:g} * local D_c"
        ),
        "full_fault_arrival_ms": full_fault_arrival_ms,
        "saved_dt_us": saved_dt_us,
        "stress_smoothing_us": float(stress_smoothing_us),
        "stress_smoothing_sigma_samples": float(stress_smoothing_samples),
        "slip_smoothing_us": float(slip_smoothing_us),
        "slip_smoothing_sigma_samples": float(slip_smoothing_samples),
        "smoothing_method": "Gaussian derivative",
        "raw_stress_rate_shown": bool(show_raw_stress_rate),
        "stress_rate_units": "MPa/ms",
        "slip_rate_units": "m/s",
        "station_metrics": metrics,
        "interface_probes": interface_metadata,
    }
    if output_metrics is None:
        output_metrics = png_path.with_name(f"{png_path.stem}_metrics.json")
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    result["outputs"]["metrics_json"] = str(output_metrics.resolve())
    output_metrics.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    args = parse_args()
    output_path = args.output or _default_output(args.input)
    result = plot_on_fault_rates(
        input_path=args.input,
        output_path=output_path,
        output_metrics=args.output_metrics,
        y_points=list(args.y_points),
        time_before_ms=args.time_before_ms,
        time_after_ms=args.time_after_ms,
        tip_slip_fraction=args.tip_slip_fraction,
        stress_smoothing_us=args.stress_smoothing_us,
        slip_smoothing_us=args.slip_smoothing_us,
        show_raw_stress_rate=args.show_raw_stress_rate,
        stress_rate_linthresh=args.stress_rate_linthresh,
        arrival_chunk_frames=args.arrival_chunk_frames,
        dpi=args.dpi,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
