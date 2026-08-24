from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


BACKGROUND = "#F7F4EE"
PANEL = "#FFFEFA"
INK = "#18313A"
MUTED = "#68767B"
GRID = "#D9D6CE"
TEAL = "#087F8C"
NAVY = "#1E4D6B"
ORANGE = "#E87524"
GOLD = "#E6B655"
RED = "#B84A3A"
PALE_GOLD = "#F3E5BD"
PALE_BLUE = "#DCEBEA"
PALE_ORANGE = "#F3D5BF"
PALE_RED = "#E9C7C1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create slide-ready rupture-speed and fault-profile figures from a "
            "velocity-weakening Tatva dump."
        )
    )
    parser.add_argument("data_path", type=Path, help="Simulation HDF5 file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to the run's Plot directory.",
    )
    parser.add_argument(
        "--fit-start",
        type=float,
        default=120.0,
        help="Stable-rupture fit start along the contact line [mm].",
    )
    parser.add_argument(
        "--fit-end",
        type=float,
        default=440.0,
        help="Stable-rupture fit end along the contact line [mm].",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=2048,
        help="Number of HDF5 frames read at once.",
    )
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Avenir", "DejaVu Sans"],
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def first_crossing_and_peak_rate(
    dataset: h5py.Dataset,
    frame_indices: np.ndarray,
    shear_time_ms: np.ndarray,
    thresholds: np.ndarray,
    *,
    chunk_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if chunk_frames < 2:
        raise ValueError("chunk_frames must be at least 2.")
    if len(frame_indices) < 2:
        raise ValueError("At least two shear frames are required.")
    if not np.all(np.diff(frame_indices) == 1):
        raise ValueError("Shear frames must be contiguous.")

    station_count = dataset.shape[1]
    crossing_time = np.full(station_count, np.nan, dtype=np.float64)
    peak_rate = np.full(station_count, -np.inf, dtype=np.float64)
    peak_rate_time = np.full(station_count, np.nan, dtype=np.float64)
    unresolved = np.ones(station_count, dtype=bool)

    first_frame = int(frame_indices[0])
    last_frame = int(frame_indices[-1])
    previous_values: np.ndarray | None = None
    previous_time: float | None = None

    for start in range(first_frame, last_frame + 1, chunk_frames):
        end = min(start + chunk_frames, last_frame + 1)
        values = np.asarray(dataset[start:end], dtype=np.float64)
        times = shear_time_ms[start:end]

        if previous_values is not None and previous_time is not None:
            values = np.vstack([previous_values, values])
            times = np.concatenate([[previous_time], times])

        before = values[:-1]
        after = values[1:]
        interval_ms = np.diff(times)

        if np.any(interval_ms <= 0.0):
            raise ValueError("Saved shear-frame times must increase monotonically.")

        if np.any(unresolved):
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
                    times[rows] + fraction * interval_ms[rows]
                )
                unresolved[columns] = False

        rate = (after - before) / (interval_ms[:, None] * 1e-3)
        local_rows = np.argmax(rate, axis=0)
        columns = np.arange(station_count)
        local_max = rate[local_rows, columns]
        improved = local_max > peak_rate
        peak_rate[improved] = local_max[improved]
        peak_rate_time[improved] = times[1:][local_rows[improved]]

        previous_values = values[-1].copy()
        previous_time = float(times[-1])

    return crossing_time, peak_rate_time


def linear_arrival_fit(
    contact_y: np.ndarray,
    arrival_time_ms: np.ndarray,
    fit_start: float,
    fit_end: float,
) -> dict[str, object]:
    mask = (
        (contact_y >= fit_start)
        & (contact_y <= fit_end)
        & np.isfinite(arrival_time_ms)
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError("Not enough finite arrival times in the fit interval.")

    slope, intercept = np.polyfit(contact_y[mask], arrival_time_ms[mask], 1)
    fitted = slope * contact_y[mask] + intercept
    residual = arrival_time_ms[mask] - fitted
    total = arrival_time_ms[mask] - np.mean(arrival_time_ms[mask])
    r_squared = 1.0 - np.sum(residual**2) / np.sum(total**2)
    return {
        "mask": mask,
        "slope_ms_per_mm": float(slope),
        "intercept_ms": float(intercept),
        "speed_m_per_s": float(1.0 / slope),
        "r_squared": float(r_squared),
    }


def material_wave_speeds(
    young_modulus_mpa: float,
    poisson_ratio: float,
    density_tonne_per_mm3: float,
) -> dict[str, float]:
    shear_modulus_mpa = young_modulus_mpa / (2.0 * (1.0 + poisson_ratio))
    lame_lambda_mpa = (
        young_modulus_mpa
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    shear_speed_mm_s = np.sqrt(shear_modulus_mpa / density_tonne_per_mm3)
    pressure_speed_mm_s = np.sqrt(
        (lame_lambda_mpa + 2.0 * shear_modulus_mpa)
        / density_tonne_per_mm3
    )
    rayleigh_low = 0.80 * shear_speed_mm_s
    rayleigh_high = 0.999999 * shear_speed_mm_s
    for _ in range(80):
        rayleigh_mid = 0.5 * (rayleigh_low + rayleigh_high)
        alpha_s = np.sqrt(
            1.0 - (rayleigh_mid / shear_speed_mm_s) ** 2
        )
        alpha_d = np.sqrt(
            1.0 - (rayleigh_mid / pressure_speed_mm_s) ** 2
        )
        rayleigh_function = (
            4.0 * alpha_s * alpha_d - (1.0 + alpha_s**2) ** 2
        )
        if rayleigh_function > 0.0:
            rayleigh_low = rayleigh_mid
        else:
            rayleigh_high = rayleigh_mid
    rayleigh_speed_mm_s = 0.5 * (rayleigh_low + rayleigh_high)
    return {
        "c_s": float(shear_speed_mm_s / 1e3),
        "c_p": float(pressure_speed_mm_s / 1e3),
        "c_r": float(rayleigh_speed_mm_s / 1e3),
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


def kammer_czm_speed_from_state(
    *,
    fracture_energy_j_m2: float,
    cohesive_zone_mm: float,
    peak_to_residual_mpa: float,
    young_modulus_mpa: float,
    poisson_ratio: float,
    wave_speeds: dict[str, float],
) -> float:
    young_modulus_pa = young_modulus_mpa * 1e6
    cohesive_zone_m = cohesive_zone_mm * 1e-3
    peak_to_residual_pa = peak_to_residual_mpa * 1e6
    target_fii = (
        fracture_energy_j_m2
        * young_modulus_pa
        / (1.0 - poisson_ratio**2)
        * 9.0
        * np.pi
        / (32.0 * cohesive_zone_m)
        / peak_to_residual_pa**2
    )
    shear_speed = wave_speeds["c_s"]
    pressure_speed = wave_speeds["c_p"]
    rayleigh_speed = wave_speeds["c_r"]

    def dynamic_factor(speed: float) -> float:
        alpha_s = np.sqrt(1.0 - (speed / shear_speed) ** 2)
        alpha_d = np.sqrt(1.0 - (speed / pressure_speed) ** 2)
        rayleigh_function = (
            4.0 * alpha_s * alpha_d - (1.0 + alpha_s**2) ** 2
        )
        return (
            alpha_s
            * (speed / shear_speed) ** 2
            / ((1.0 - poisson_ratio) * rayleigh_function)
        )

    lower = 1e-5 * rayleigh_speed
    upper = (1.0 - 1e-7) * rayleigh_speed
    if target_fii <= dynamic_factor(lower):
        return float("nan")
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if dynamic_factor(middle) < target_fii:
            lower = middle
        else:
            upper = middle
    return float(0.5 * (lower + upper))


def estimate_kammer_czm_speed(
    *,
    h5: h5py.File,
    contact_y: np.ndarray,
    critical_slip: np.ndarray,
    mu_static: np.ndarray,
    mu_kinetic: np.ndarray,
    half_dc_arrival: np.ndarray,
    shear_indices: np.ndarray,
    shear_time_ms: np.ndarray,
    master_nodes: np.ndarray,
    slave_nodes: np.ndarray,
    normal_penalty: float,
    fit_start: float,
    fit_end: float,
    young_modulus_mpa: float,
    poisson_ratio: float,
    wave_speeds: dict[str, float],
) -> dict[str, object]:
    sample_start = fit_start + 20.0
    sample_end = fit_end - 40.0
    if sample_end <= sample_start:
        sample_start = fit_start
        sample_end = fit_end
    sample_positions = np.arange(sample_start, sample_end + 0.1, 20.0)

    cohesive_zones: list[float] = []
    normal_stresses: list[float] = []
    fracture_energies: list[float] = []
    theoretical_speeds: list[float] = []
    used_positions: list[float] = []
    cumulative_slip = h5["interface/cumulative_slip"]

    for sample_position in sample_positions:
        station = int(np.argmin(np.abs(contact_y - sample_position)))
        arrival_time = float(half_dc_arrival[station])
        if not np.isfinite(arrival_time):
            continue
        frame = int(
            shear_indices[
                np.argmin(
                    np.abs(
                        shear_time_ms[shear_indices]
                        - arrival_time
                    )
                )
            ]
        )
        weakening_progress = (
            np.asarray(cumulative_slip[frame], dtype=np.float64)
            / critical_slip
        )
        position_05 = descending_crossing_position(
            contact_y,
            weakening_progress,
            0.05,
            sample_position,
        )
        position_95 = descending_crossing_position(
            contact_y,
            weakening_progress,
            0.95,
            sample_position,
        )
        cohesive_zone_mm = (position_05 - position_95) / 0.90
        if not np.isfinite(cohesive_zone_mm) or cohesive_zone_mm <= 0.0:
            continue

        normal_separation = float(
            h5["moving/displacement"][frame, master_nodes[station], 0]
            - h5["stationary/displacement"][frame, slave_nodes[station], 0]
        )
        normal_stress_mpa = normal_penalty * max(normal_separation, 0.0)
        peak_to_residual_mpa = (
            mu_static[station] - mu_kinetic[station]
        ) * normal_stress_mpa
        if peak_to_residual_mpa <= 0.0:
            continue
        fracture_energy_j_m2 = (
            0.5
            * peak_to_residual_mpa
            * critical_slip[station]
            * 1e3
        )
        theoretical_speed = kammer_czm_speed_from_state(
            fracture_energy_j_m2=fracture_energy_j_m2,
            cohesive_zone_mm=cohesive_zone_mm,
            peak_to_residual_mpa=peak_to_residual_mpa,
            young_modulus_mpa=young_modulus_mpa,
            poisson_ratio=poisson_ratio,
            wave_speeds=wave_speeds,
        )
        if not np.isfinite(theoretical_speed):
            continue

        used_positions.append(float(contact_y[station]))
        cohesive_zones.append(cohesive_zone_mm)
        normal_stresses.append(normal_stress_mpa)
        fracture_energies.append(fracture_energy_j_m2)
        theoretical_speeds.append(theoretical_speed)

    if len(theoretical_speeds) < 3:
        raise ValueError(
            "Not enough stable-front stations for the Kammer-McLaskey "
            "cohesive-zone speed estimate."
        )

    def distribution(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        percentile_16, percentile_84 = np.percentile(array, [16.0, 84.0])
        return {
            "median": float(np.median(array)),
            "p16": float(percentile_16),
            "p84": float(percentile_84),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }

    return {
        "source": (
            "Kammer and McLaskey (2019), Eqs. (A.3), (A.4), and (A.12)"
        ),
        "sample_positions_mm": used_positions,
        "cohesive_zone_mm": distribution(cohesive_zones),
        "normal_stress_mpa": distribution(normal_stresses),
        "fracture_energy_j_m2": distribution(fracture_energies),
        "speed_m_per_s": distribution(theoretical_speeds),
    }


def add_zone_bands(axis: plt.Axes) -> None:
    axis.axvspan(0.0, 120.0, color=PALE_GOLD, alpha=0.58, linewidth=0)
    axis.axvspan(120.0, 480.0, color=PALE_BLUE, alpha=0.36, linewidth=0)
    axis.axvspan(480.0, 490.0, color=PALE_ORANGE, alpha=0.72, linewidth=0)
    axis.axvspan(490.0, 500.0, color=PALE_RED, alpha=0.62, linewidth=0)


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


def plot_rupture_speed(
    *,
    run_id: str,
    contact_y: np.ndarray,
    half_dc_arrival: np.ndarray,
    peak_rate_arrival: np.ndarray,
    half_dc_fit: dict[str, object],
    peak_rate_fit: dict[str, object],
    stop_time_ms: float,
    fit_start: float,
    fit_end: float,
    wave_speeds: dict[str, float],
    czm_prediction: dict[str, object],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    czm_speed_stats = dict(czm_prediction["speed_m_per_s"])
    czm_xc_stats = dict(czm_prediction["cohesive_zone_mm"])
    czm_speed = float(czm_speed_stats["median"])
    czm_speed_p16 = float(czm_speed_stats["p16"])
    czm_speed_p84 = float(czm_speed_stats["p84"])

    figure = plt.figure(figsize=(16.0, 9.0))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(3.2, 1.25),
        height_ratios=(1.05, 1.0),
        left=0.065,
        right=0.965,
        top=0.86,
        bottom=0.16,
        wspace=0.18,
        hspace=0.30,
    )
    main_axis = figure.add_subplot(grid[:, 0])
    summary_axis = figure.add_subplot(grid[0, 1])
    speed_axis = figure.add_subplot(grid[1, 1])

    figure.text(
        0.065,
        0.945,
        f"Run {run_id}  |  Rupture arrival and stable-front speed",
        fontsize=25,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(
        0.065,
        0.902,
        (
            "Arrival-time fits and the Kammer-McLaskey cohesive-zone model "
            f"agree in the stable segment ({fit_start:.0f}-{fit_end:.0f} mm)."
        ),
        fontsize=14.5,
        color=MUTED,
        va="top",
    )

    main_axis.set_facecolor(PANEL)
    add_zone_bands(main_axis)
    main_axis.plot(
        contact_y,
        half_dc_arrival,
        color=TEAL,
        linewidth=2.7,
        marker="o",
        markersize=4.2,
        markevery=20,
        label=r"$0.5D_c$ crossing",
        zorder=4,
    )
    main_axis.plot(
        contact_y,
        peak_rate_arrival,
        color=NAVY,
        linewidth=2.0,
        marker="s",
        markersize=3.8,
        markevery=20,
        alpha=0.88,
        label="Peak slip-rate time",
        zorder=3,
    )

    fit_y = np.linspace(fit_start, fit_end, 200)
    fit_t = (
        float(half_dc_fit["slope_ms_per_mm"]) * fit_y
        + float(half_dc_fit["intercept_ms"])
    )
    main_axis.plot(
        fit_y,
        fit_t,
        color=ORANGE,
        linewidth=3.0,
        linestyle=(0, (7, 4)),
        label=f"Linear fit, {fit_start:.0f}-{fit_end:.0f} mm",
        zorder=5,
    )
    fit_anchor_y = 0.5 * (fit_start + fit_end)
    fit_anchor_t = (
        float(half_dc_fit["slope_ms_per_mm"]) * fit_anchor_y
        + float(half_dc_fit["intercept_ms"])
    )
    czm_fit_t = fit_anchor_t + (fit_y - fit_anchor_y) / czm_speed
    czm_p16_t = fit_anchor_t + (fit_y - fit_anchor_y) / czm_speed_p16
    czm_p84_t = fit_anchor_t + (fit_y - fit_anchor_y) / czm_speed_p84
    main_axis.fill_between(
        fit_y,
        np.minimum(czm_p16_t, czm_p84_t),
        np.maximum(czm_p16_t, czm_p84_t),
        color=RED,
        alpha=0.10,
        linewidth=0.0,
        zorder=2,
    )
    main_axis.plot(
        fit_y,
        czm_fit_t,
        color=RED,
        linewidth=2.3,
        linestyle=(0, (2, 2, 7, 2)),
        label=(
            "Kammer-McLaskey CZM, "
            f"{czm_speed / 1e3:.2f} km/s"
        ),
        zorder=4,
    )
    main_axis.axhline(
        stop_time_ms,
        color=RED,
        linewidth=1.7,
        linestyle=(0, (3, 3)),
        zorder=2,
    )
    main_axis.text(
        494.0,
        stop_time_ms - 0.035,
        f"Loading stopped  {stop_time_ms:.3f} ms",
        color=RED,
        fontsize=12.5,
        ha="right",
        va="top",
        bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 2.2, "alpha": 0.9},
    )

    main_axis.text(
        58.0,
        10.60,
        "Nucleation\n$D_c$ taper",
        ha="center",
        va="top",
        fontsize=12.5,
        color="#6D5721",
    )
    main_axis.text(
        300.0,
        10.60,
        "Uniform dynamic fault",
        ha="center",
        va="top",
        fontsize=12.5,
        color="#315D60",
    )
    main_axis.annotate(
        "Tail creep",
        xy=(494.5, half_dc_arrival[np.argmin(np.abs(contact_y - 495.0))]),
        xytext=(432.0, 10.42),
        color=RED,
        fontsize=12.5,
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.4},
    )
    main_axis.set_xlim(0.0, 500.0)
    finite_arrivals = np.concatenate(
        [
            half_dc_arrival[np.isfinite(half_dc_arrival)],
            peak_rate_arrival[np.isfinite(peak_rate_arrival)],
        ]
    )
    lower = min(8.25, float(np.min(finite_arrivals)) - 0.08)
    upper = max(10.70, float(np.max(finite_arrivals)) + 0.08)
    main_axis.set_ylim(lower, upper)
    main_axis.set_xlabel("Position along fault, y [mm]")
    main_axis.set_ylabel("Arrival time after shear phase begins [ms]")
    main_axis.grid(axis="y")
    main_axis.legend(loc="lower right", fontsize=12.5)
    main_axis.spines[["top", "right"]].set_visible(False)

    half_speed = float(half_dc_fit["speed_m_per_s"])
    peak_speed = float(peak_rate_fit["speed_m_per_s"])
    representative_speed = 0.5 * (half_speed + peak_speed)

    summary_axis.set_facecolor(PANEL)
    summary_axis.axis("off")
    summary_axis.text(
        0.04,
        0.93,
        "Stable rupture",
        fontsize=14,
        color=MUTED,
        va="top",
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.04,
        0.72,
        f"{representative_speed / 1e3:.2f}",
        fontsize=42,
        fontweight="bold",
        color=ORANGE,
        va="top",
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.49,
        0.70,
        "km/s",
        fontsize=18,
        color=INK,
        va="top",
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.04,
        0.49,
        (
            rf"$0.5D_c$:  {half_speed / 1e3:.3f} km/s"
            "\n"
            rf"$R^2={float(half_dc_fit['r_squared']):.4f}$"
        ),
        fontsize=13.5,
        color=TEAL,
        linespacing=1.35,
        va="top",
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.04,
        0.27,
        (
            f"Peak rate:  {peak_speed / 1e3:.3f} km/s"
            "\n"
            rf"$R^2={float(peak_rate_fit['r_squared']):.4f}$"
        ),
        fontsize=13.5,
        color=NAVY,
        linespacing=1.35,
        va="top",
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.04,
        0.045,
        (
            f"CZM: {czm_speed / 1e3:.3f} km/s"
            rf"  |  $X_c={float(czm_xc_stats['median']):.2f}$ mm"
        ),
        fontsize=10.3,
        color=RED,
        va="bottom",
        transform=summary_axis.transAxes,
    )
    summary_axis.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=summary_axis.transAxes,
            fill=False,
            edgecolor=GRID,
            linewidth=1.0,
        )
    )

    speed_axis.set_facecolor(PANEL)
    names = [
        r"$v_r$",
        r"$v_{\mathrm{CZM}}$",
        r"$c_R$",
        r"$c_s$",
        r"$c_p$",
    ]
    values = [
        representative_speed / 1e3,
        czm_speed / 1e3,
        wave_speeds["c_r"] / 1e3,
        wave_speeds["c_s"] / 1e3,
        wave_speeds["c_p"] / 1e3,
    ]
    colors = [ORANGE, RED, GOLD, TEAL, NAVY]
    positions = np.arange(len(names))
    speed_axis.hlines(
        positions,
        0.0,
        values,
        color=[f"{color}66" for color in colors],
        linewidth=5.0,
    )
    speed_axis.scatter(values, positions, s=115, color=colors, zorder=3)
    speed_axis.errorbar(
        czm_speed / 1e3,
        1,
        xerr=np.asarray(
            [
                [(czm_speed - czm_speed_p16) / 1e3],
                [(czm_speed_p84 - czm_speed) / 1e3],
            ]
        ),
        fmt="none",
        ecolor=RED,
        elinewidth=2.0,
        capsize=4.0,
        zorder=4,
    )
    for position, value in zip(positions, values, strict=True):
        speed_axis.text(
            value + 0.055,
            position,
            f"{value:.2f}",
            va="center",
            fontsize=12.5,
            color=INK,
        )
    speed_axis.set_yticks(positions, names)
    speed_axis.invert_yaxis()
    speed_axis.set_xlim(0.0, 3.05)
    speed_axis.set_xlabel("Speed [km/s]")
    speed_axis.set_title("CZM and material-wave comparison", loc="left", pad=10)
    speed_axis.grid(axis="x")
    speed_axis.spines[["top", "right", "left"]].set_visible(False)
    speed_axis.tick_params(axis="y", length=0)

    figure.text(
        0.065,
        0.052,
        (
            r"Arrival criteria: cumulative slip = $0.5D_c(y)$ and local peak "
            "slip rate. Speeds are least-squares slopes over the highlighted "
            "stable segment."
        ),
        fontsize=10.8,
        color=MUTED,
    )
    figure.text(
        0.065,
        0.025,
        (
            "CZM: Kammer and McLaskey (2019), Eq. (A.12); "
            r"$X_c=(y_{5\%}-y_{95\%})/0.90$. "
            "The red band spans the 16th-84th percentiles across stable-front "
            "stations; the line is midpoint-anchored, so only its slope is "
            "predicted."
        ),
        fontsize=10.2,
        color=MUTED,
    )
    return save_figure(figure, output_dir, "rupture_speed_stable_fit", dpi)


def plot_fault_profile(
    *,
    run_id: str,
    contact_y: np.ndarray,
    critical_slip: np.ndarray,
    mu_static: np.ndarray,
    mu_kinetic: np.ndarray,
    creep_weight: np.ndarray,
    stop_time_ms: float,
    young_modulus_mpa: float,
    poisson_ratio: float,
    nucleation_initial_tau_mpa: float,
    nucleation_normal_stress_mpa: float,
    nucleation_reference_time_ms: float,
    nucleation_taper_length_mm: float,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    shear_modulus_mpa = young_modulus_mpa / (2.0 * (1.0 + poisson_ratio))
    lame_lambda_mpa = (
        young_modulus_mpa
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    elastic_prefactor_mpa = (
        16.0
        / np.pi
        * shear_modulus_mpa
        * (lame_lambda_mpa + shear_modulus_mpa)
        / (lame_lambda_mpa + 2.0 * shear_modulus_mpa)
    )
    representative_mu_s = float(np.median(mu_static))
    representative_mu_k = float(np.median(mu_kinetic))
    tau_peak_mpa = representative_mu_s * nucleation_normal_stress_mpa
    tau_residual_mpa = representative_mu_k * nucleation_normal_stress_mpa
    initial_stress_drop_mpa = nucleation_initial_tau_mpa - tau_residual_mpa
    if initial_stress_drop_mpa <= 0.0:
        raise ValueError(
            "The Ke et al. critical-length estimate requires tau_i > tau_r. "
            f"Got tau_i={nucleation_initial_tau_mpa:.6g} MPa and "
            f"tau_r={tau_residual_mpa:.6g} MPa."
        )
    fracture_energy_per_dc_mpa = 0.5 * (tau_peak_mpa - tau_residual_mpa)
    lc_per_dc = (
        elastic_prefactor_mpa
        * fracture_energy_per_dc_mpa
        / initial_stress_drop_mpa**2
    )
    local_lc = lc_per_dc * critical_slip
    distance_from_loading_edge = contact_y - float(np.min(contact_y))
    taper_mask = (
        distance_from_loading_edge >= 0.0
    ) & (distance_from_loading_edge <= nucleation_taper_length_mm)
    taper_distance = distance_from_loading_edge[taper_mask]
    taper_difference = local_lc[taper_mask] - taper_distance
    sign_changes = np.flatnonzero(
        taper_difference[:-1] * taper_difference[1:] <= 0.0
    )
    if len(sign_changes):
        lower = int(sign_changes[0])
        x0 = taper_distance[lower]
        x1 = taper_distance[lower + 1]
        d0 = taper_difference[lower]
        d1 = taper_difference[lower + 1]
        effective_hstar_mm = float(x0 - d0 * (x1 - x0) / (d1 - d0))
    else:
        effective_hstar_mm = float("nan")

    figure = plt.figure(figsize=(16.0, 9.0))
    grid = figure.add_gridspec(
        4,
        1,
        height_ratios=(0.47, 1.40, 1.05, 1.05),
        left=0.09,
        right=0.925,
        top=0.81,
        bottom=0.16,
        hspace=0.56,
    )
    zone_axis = figure.add_subplot(grid[0])
    dc_axis = figure.add_subplot(grid[1])
    friction_axis = figure.add_subplot(grid[2], sharex=dc_axis)
    creep_axis = figure.add_subplot(grid[3], sharex=dc_axis)

    figure.text(
        0.065,
        0.945,
        f"Run {run_id}  |  Fault-interface profile",
        fontsize=25,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(
        0.065,
        0.902,
        (
            "Spatial friction law, far-end relaxation, and the supplied "
            r"Ke et al. LEFM $h^*$ estimate."
        ),
        fontsize=14.5,
        color=MUTED,
        va="top",
    )

    zone_axis.set_xlim(-12.0, 505.0)
    zone_axis.set_ylim(0.0, 1.0)
    zone_axis.axis("off")
    zones = [
        (0.0, 120.0, PALE_GOLD, "Nucleation", "#6D5721"),
        (120.0, 480.0, PALE_BLUE, "Uniform dynamic fault", "#315D60"),
        (480.0, 490.0, PALE_ORANGE, "", ORANGE),
        (490.0, 500.0, PALE_RED, "", RED),
    ]
    for start, end, color, label, text_color in zones:
        zone_axis.add_patch(
            Rectangle(
                (start, 0.17),
                end - start,
                0.50,
                facecolor=color,
                edgecolor=BACKGROUND,
                linewidth=1.5,
            )
        )
        if label:
            zone_axis.text(
                0.5 * (start + end),
                0.42,
                label,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="semibold",
                color=text_color,
            )
    zone_axis.annotate(
        "Creep\ntransition",
        xy=(485.0, 0.67),
        xytext=(452.0, 0.91),
        ha="center",
        va="bottom",
        fontsize=11.5,
        color=ORANGE,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1.2},
    )
    zone_axis.annotate(
        "Creep\nplateau",
        xy=(495.0, 0.67),
        xytext=(494.0, 0.91),
        ha="center",
        va="bottom",
        fontsize=11.5,
        color=RED,
        arrowprops={"arrowstyle": "-", "color": RED, "lw": 1.2},
    )
    zone_axis.add_patch(
        FancyArrowPatch(
            (-9.0, 0.42),
            (-0.5, 0.42),
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=2.0,
            color=ORANGE,
        )
    )
    zone_axis.text(
        0.0,
        0.02,
        "Loading end, y = 0",
        ha="left",
        va="top",
        fontsize=10.5,
        color=INK,
    )
    if np.isfinite(effective_hstar_mm):
        zone_axis.add_patch(
            FancyArrowPatch(
                (0.0, 0.84),
                (effective_hstar_mm, 0.84),
                arrowstyle="<->",
                mutation_scale=13,
                linewidth=1.8,
                color=ORANGE,
            )
        )
        zone_axis.text(
            0.5 * effective_hstar_mm,
            0.93,
            rf"$h^*\approx {effective_hstar_mm:.0f}$ mm",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="semibold",
            color=ORANGE,
        )

    for axis in (dc_axis, friction_axis, creep_axis):
        axis.set_facecolor(PANEL)
        add_zone_bands(axis)
        axis.set_xlim(0.0, 500.0)
        axis.grid(axis="y")
        axis.spines[["top", "right"]].set_visible(False)

    dc_axis.plot(contact_y, critical_slip, color=ORANGE, linewidth=3.2)
    dc_axis.fill_between(
        contact_y,
        0.0,
        critical_slip,
        color=ORANGE,
        alpha=0.16,
    )
    dc_axis.scatter(
        [contact_y[0], contact_y[np.argmin(np.abs(contact_y - 120.0))]],
        [
            critical_slip[0],
            critical_slip[np.argmin(np.abs(contact_y - 120.0))],
        ],
        color=ORANGE,
        s=55,
        zorder=4,
    )
    dc_axis.annotate(
        (
            f"$D_c$ = {critical_slip[0]:.3f} mm\n"
            f"local $L_c$ = {local_lc[0]:.0f} mm"
        ),
        xy=(contact_y[0], critical_slip[0]),
        xytext=(24.0, 0.285),
        fontsize=11.5,
        color=ORANGE,
        va="top",
        arrowprops={"arrowstyle": "-", "color": ORANGE},
    )
    base_dc_index = int(np.argmin(np.abs(contact_y - 120.0)))
    dc_axis.annotate(
        (
            f"Base $D_c$ = {critical_slip[base_dc_index]:.4f} mm\n"
            f"local $L_c$ = {local_lc[base_dc_index]:.1f} mm"
        ),
        xy=(contact_y[base_dc_index], critical_slip[base_dc_index]),
        xytext=(195.0, 0.065),
        fontsize=11.5,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.2},
    )
    if np.isfinite(effective_hstar_mm):
        crossover_dc = float(
            np.interp(
                effective_hstar_mm,
                distance_from_loading_edge,
                critical_slip,
            )
        )
        dc_axis.vlines(
            effective_hstar_mm,
            0.0,
            crossover_dc,
            color=TEAL,
            linewidth=1.6,
            linestyle=(0, (4, 3)),
            zorder=3,
        )
        dc_axis.scatter(
            [effective_hstar_mm],
            [crossover_dc],
            color=TEAL,
            s=60,
            zorder=5,
        )
        dc_axis.annotate(
            rf"$L_c(y)=y$ at $h^*\approx {effective_hstar_mm:.0f}$ mm",
            xy=(effective_hstar_mm, crossover_dc),
            xytext=(145.0, 0.145),
            fontsize=11.5,
            color=TEAL,
            arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.2},
        )
    dc_axis.text(
        0.99,
        0.91,
        r"Half-cosine taper, $0 \leq y \leq 120$ mm",
        transform=dc_axis.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        color=MUTED,
    )
    dc_axis.set_ylim(0.0, max(0.34, 1.08 * float(np.max(critical_slip))))
    dc_axis.set_ylabel("$D_c$ [mm]")
    dc_axis.set_title(
        r"Slip-weakening distance and local critical length $L_c(y)$",
        loc="left",
        pad=8,
    )
    dc_axis.tick_params(axis="x", labelbottom=False)
    lc_axis = dc_axis.secondary_yaxis(
        "right",
        functions=(
            lambda dc: dc * lc_per_dc,
            lambda lc: lc / lc_per_dc,
        ),
    )
    lc_axis.set_ylabel(r"Local $L_c(y)$ [mm]", color=TEAL)
    lc_axis.tick_params(axis="y", colors=TEAL)
    lc_axis.spines["right"].set_color(TEAL)

    friction_axis.plot(
        contact_y,
        mu_static,
        color=TEAL,
        linewidth=3.0,
        label=rf"$\mu_s={float(np.median(mu_static)):.2f}$",
    )
    friction_axis.plot(
        contact_y,
        mu_kinetic,
        color=NAVY,
        linewidth=3.0,
        label=rf"$\mu_k={float(np.median(mu_kinetic)):.2f}$",
    )
    friction_axis.set_ylim(
        max(0.0, float(np.min(mu_kinetic)) - 0.08),
        min(1.0, float(np.max(mu_static)) + 0.08),
    )
    friction_axis.set_ylabel("Friction\ncoefficient")
    friction_axis.set_title(
        "Frictional strength: spatially uniform along the entire fault",
        loc="left",
        pad=8,
    )
    friction_axis.legend(loc="center right", ncol=2, fontsize=13)
    friction_axis.text(
        0.01,
        0.50,
        (
            rf"No $\mu_s$ gradient   |   "
            rf"$\Delta\mu={float(np.median(mu_static - mu_kinetic)):.2f}$"
        ),
        transform=friction_axis.transAxes,
        fontsize=12,
        color=MUTED,
        va="center",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": PANEL,
            "edgecolor": "none",
            "alpha": 0.92,
        },
    )
    friction_axis.tick_params(axis="x", labelbottom=False)

    creep_axis.plot(contact_y, creep_weight, color=RED, linewidth=3.2)
    creep_axis.fill_between(
        contact_y,
        0.0,
        creep_weight,
        color=RED,
        alpha=0.14,
    )
    creep_axis.set_ylim(-0.05, 1.14)
    creep_axis.set_ylabel("Creep\nweight")
    creep_axis.set_xlabel("Position along fault, y [mm]")
    creep_axis.set_title(
        "Far-end buffer: viscoplastic relaxation without changing friction",
        loc="left",
        pad=8,
    )
    creep_axis.annotate(
        "Half-cosine transition\n480-490 mm",
        xy=(486.5, creep_weight[np.argmin(np.abs(contact_y - 486.5))]),
        xytext=(372.0, 0.46),
        fontsize=12.5,
        color=ORANGE,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.3},
    )
    creep_axis.annotate(
        "Full creep\n490-500 mm",
        xy=(496.0, 1.0),
        xytext=(450.0, 0.92),
        fontsize=12.5,
        color=RED,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.3},
    )

    figure.text(
        0.065,
        0.047,
        (
            "Boundary control: shear displacement is applied at the loading-end "
            f"face; loading stops at rupture detection ({stop_time_ms:.3f} ms). "
            "Creep relaxation time = 0.05 ms."
        ),
        fontsize=10.8,
        color=MUTED,
    )
    figure.text(
        0.065,
        0.020,
        (
            rf"$L_c=\frac{{16}}{{\pi}}\frac{{\mu(\lambda+\mu)}}"
            rf"{{\lambda+2\mu}}\frac{{G}}{{(\tau_i-\tau_r)^2}}$, "
            rf"$G=\frac{{1}}{{2}}(\tau_p-\tau_r)D_c$. "
            rf"Pointwise estimate using $\tau_i={nucleation_initial_tau_mpa:.2f}$ "
            rf"MPa and $\sigma_n={nucleation_normal_stress_mpa:.2f}$ MPa, "
            rf"averaged over 0-{nucleation_taper_length_mm:.0f} mm at "
            rf"$t={nucleation_reference_time_ms:.3f}$ ms."
        ),
        fontsize=9.8,
        color=MUTED,
    )
    return save_figure(figure, output_dir, "fault_interface_profile", dpi)


def plot_creep_mechanism(
    *,
    run_id: str,
    contact_y: np.ndarray,
    creep_weight: np.ndarray,
    mu_static: np.ndarray,
    mu_kinetic: np.ndarray,
    dt_s: float,
    relaxation_time_s: float,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    figure = plt.figure(figsize=(16.0, 9.0))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.18, 1.42),
        height_ratios=(0.88, 1.12),
        left=0.055,
        right=0.965,
        top=0.82,
        bottom=0.15,
        wspace=0.15,
        hspace=0.36,
    )
    algorithm_axis = figure.add_subplot(grid[:, 0])
    spatial_axis = figure.add_subplot(grid[0, 1])
    relaxation_axis = figure.add_subplot(grid[1, 1])

    figure.text(
        0.055,
        0.945,
        f"Run {run_id}  |  How the far-end creep regularization works",
        fontsize=25,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(
        0.055,
        0.902,
        (
            "The Coulomb strength is unchanged; only the rate of plastic return "
            "is blended from instantaneous to viscoplastic."
        ),
        fontsize=14.5,
        color=MUTED,
        va="top",
    )

    algorithm_axis.set_facecolor("#F1EEE7")
    algorithm_axis.set_xlim(0.0, 1.0)
    algorithm_axis.set_ylim(0.0, 1.0)
    algorithm_axis.axis("off")
    algorithm_axis.text(
        0.04,
        0.965,
        "Tatva contact update, each explicit time step",
        transform=algorithm_axis.transAxes,
        fontsize=17,
        fontweight="bold",
        color=INK,
        va="top",
    )

    alpha = (
        dt_s / (dt_s + relaxation_time_s)
        if relaxation_time_s > 0.0
        else 1.0
    )

    algorithm_axis.text(
        0.04,
        0.845,
        "01",
        transform=algorithm_axis.transAxes,
        fontsize=16,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.845,
        "Elastic predictor",
        transform=algorithm_axis.transAxes,
        fontsize=16,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.765,
        r"$\tau_{\mathrm{trial}}=k_t(\delta_t-p)$",
        transform=algorithm_axis.transAxes,
        fontsize=15,
        color=INK,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.705,
        r"$p$: plastic slip;  $\delta_t$: relative tangential displacement",
        transform=algorithm_axis.transAxes,
        fontsize=11.5,
        color=MUTED,
        va="center",
    )
    algorithm_axis.plot(
        [0.04, 0.96],
        [0.650, 0.650],
        transform=algorithm_axis.transAxes,
        color=GRID,
        linewidth=1.4,
    )

    algorithm_axis.text(
        0.04,
        0.585,
        "02",
        transform=algorithm_axis.transAxes,
        fontsize=16,
        fontweight="bold",
        color=TEAL,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.585,
        "Coulomb check",
        transform=algorithm_axis.transAxes,
        fontsize=16,
        fontweight="bold",
        color=TEAL,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.505,
        r"$\tau_y=\mu_{\mathrm{eff}}(D)\,\sigma_n$",
        transform=algorithm_axis.transAxes,
        fontsize=15,
        color=INK,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.435,
        r"$|\tau_{\mathrm{trial}}|\leq\tau_y$: stick,  $\Delta p=0$",
        transform=algorithm_axis.transAxes,
        fontsize=13,
        color=INK,
        va="center",
    )
    algorithm_axis.plot(
        [0.04, 0.96],
        [0.375, 0.375],
        transform=algorithm_axis.transAxes,
        color=GRID,
        linewidth=1.4,
    )

    algorithm_axis.text(
        0.04,
        0.315,
        "03",
        transform=algorithm_axis.transAxes,
        fontsize=16,
        fontweight="bold",
        color=ORANGE,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.315,
        "Rate-dependent plastic return",
        transform=algorithm_axis.transAxes,
        fontsize=16,
        fontweight="bold",
        color=ORANGE,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.245,
        r"$\Delta p=f(w)\,\Delta p_C$",
        transform=algorithm_axis.transAxes,
        fontsize=14,
        color=INK,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.175,
        (
            r"$\Delta p_C=\mathrm{sgn}(\tau_{\mathrm{trial}})"
            r"\dfrac{|\tau_{\mathrm{trial}}|-\tau_y}{k_t}$"
        ),
        transform=algorithm_axis.transAxes,
        fontsize=13,
        color=INK,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.095,
        (
            r"$f(w)=(1-w)+w\alpha$,   "
            r"$\alpha=\dfrac{\Delta t}{\Delta t+t_c}$"
        ),
        transform=algorithm_axis.transAxes,
        fontsize=12.5,
        color=INK,
        va="center",
    )
    algorithm_axis.text(
        0.13,
        0.025,
        (
            rf"$w=0$: $f=1$ instant return   |   "
            rf"$w=1$: $f=\alpha={alpha:.2e}$ per step"
        ),
        transform=algorithm_axis.transAxes,
        fontsize=11.5,
        fontweight="semibold",
        color=RED,
        va="center",
    )

    spatial_axis.set_facecolor(PANEL)
    spatial_axis.axvspan(450.0, 480.0, color=PALE_BLUE, alpha=0.55, linewidth=0)
    spatial_axis.axvspan(480.0, 490.0, color=PALE_ORANGE, alpha=0.75, linewidth=0)
    spatial_axis.axvspan(490.0, 500.0, color=PALE_RED, alpha=0.65, linewidth=0)
    spatial_axis.plot(
        contact_y,
        creep_weight,
        color=RED,
        linewidth=3.2,
    )
    spatial_axis.fill_between(
        contact_y,
        0.0,
        creep_weight,
        color=RED,
        alpha=0.16,
    )
    spatial_axis.set_xlim(450.0, 500.0)
    spatial_axis.set_ylim(-0.04, 1.14)
    spatial_axis.set_xticks([450.0, 480.0, 490.0, 500.0])
    spatial_axis.set_yticks([0.0, 0.5, 1.0])
    spatial_axis.set_xlabel("Fault-tail position, y [mm]")
    spatial_axis.set_ylabel("Creep weight, $w(y)$")
    spatial_axis.set_title(
        "Spatial blend: only the last 20 mm is regularized",
        loc="left",
        pad=9,
    )
    spatial_axis.grid(axis="y")
    spatial_axis.spines[["top", "right"]].set_visible(False)
    spatial_axis.text(
        465.0,
        0.56,
        "Standard\nCoulomb",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#315D60",
    )
    spatial_axis.text(
        485.0,
        0.58,
        "Half-cosine\ntransition",
        ha="center",
        va="center",
        fontsize=11.5,
        color=ORANGE,
    )
    spatial_axis.text(
        495.0,
        0.56,
        "Full\ncreep",
        ha="center",
        va="center",
        fontsize=12.5,
        color=RED,
    )

    relaxation_axis.set_facecolor(PANEL)
    relaxation_time_us = relaxation_time_s * 1e6
    dt_us = dt_s * 1e6
    end_time_us = max(4.0 * relaxation_time_us, 20.0 * dt_us)
    time_us = np.linspace(0.0, end_time_us, 600)
    if relaxation_time_s > 0.0:
        full_creep_excess = np.power(
            1.0 - alpha,
            time_us / dt_us,
        )
    else:
        full_creep_excess = np.zeros_like(time_us)
        full_creep_excess[0] = 1.0
    relaxation_axis.plot(
        time_us,
        full_creep_excess,
        color=RED,
        linewidth=3.2,
        label=rf"Full creep, $t_c={relaxation_time_us:.0f}\ \mu$s",
    )
    relaxation_axis.fill_between(
        time_us,
        0.0,
        full_creep_excess,
        color=RED,
        alpha=0.12,
    )
    standard_drop_time = min(max(2.0 * dt_us, end_time_us * 0.004), end_time_us)
    relaxation_axis.plot(
        [0.0, standard_drop_time, end_time_us],
        [1.0, 0.0, 0.0],
        color=TEAL,
        linewidth=2.7,
        linestyle=(0, (6, 3)),
        label="Standard Coulomb, one-step return",
    )
    if relaxation_time_s > 0.0:
        relaxation_axis.axvline(
            relaxation_time_us,
            color=ORANGE,
            linewidth=1.6,
            linestyle=(0, (4, 3)),
        )
        relaxation_axis.scatter(
            [relaxation_time_us],
            [np.exp(-1.0)],
            color=ORANGE,
            s=65,
            zorder=4,
        )
        relaxation_axis.annotate(
            r"After $t_c$: 37% of the initial overstress remains",
            xy=(relaxation_time_us, np.exp(-1.0)),
            xytext=(1.55 * relaxation_time_us, 0.57),
            fontsize=12.5,
            color=ORANGE,
            arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.3},
        )
    relaxation_axis.set_xlim(0.0, end_time_us)
    relaxation_axis.set_ylim(-0.03, 1.08)
    relaxation_axis.set_xlabel(r"Time after exceeding Coulomb strength [$\mu$s]")
    relaxation_axis.set_ylabel(
        r"Normalized overstress, $(|\tau|-\tau_y)/(|\tau_0|-\tau_y)$"
    )
    relaxation_axis.set_title(
        "Temporal effect at fixed tangential displacement",
        loc="left",
        pad=9,
    )
    relaxation_axis.grid()
    relaxation_axis.spines[["top", "right"]].set_visible(False)
    relaxation_axis.legend(loc="upper right", fontsize=12)

    figure.text(
        0.055,
        0.035,
        (
            rf"0114 values: $\mu_s={float(np.median(mu_static)):.2f}$ and "
            rf"$\mu_k={float(np.median(mu_kinetic)):.2f}$ everywhere; "
            rf"$\Delta t={dt_us:.4f}\ \mu$s; "
            rf"$t_c={relaxation_time_us:.0f}\ \mu$s. "
            "The regularization smooths an abrupt far-end plastic correction "
            "into gradual dissipative slip."
        ),
        fontsize=11.5,
        color=MUTED,
    )
    return save_figure(figure, output_dir, "creep_mechanism", dpi)


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
    run_id = data_path.parent.parent.name.split("_", maxsplit=1)[0]
    configure_style()

    with h5py.File(data_path, "r") as h5:
        contact_y = np.asarray(h5["interface/contact_line_y"], dtype=np.float64)
        critical_slip = np.asarray(
            h5["interface/critical_slip_profile"], dtype=np.float64
        )
        mu_static = np.asarray(
            h5["interface/mu_static_profile"], dtype=np.float64
        )
        mu_kinetic = np.asarray(
            h5["interface/mu_kinetic_profile"], dtype=np.float64
        )
        creep_weight = np.asarray(
            h5["interface/creep_weight_profile"], dtype=np.float64
        )
        history = np.asarray(h5["history"], dtype=np.float64)
        history_columns = decode_strings(np.asarray(h5["history_columns"]))
        phase_id = np.asarray(h5["phase_id"], dtype=np.int8)
        time = history[:, history_columns.index("time")]
        pressure_time = float(h5.attrs["pressure_steps"] * h5.attrs["dt"])
        shear_time_ms = (time - pressure_time) * 1e3
        shear_indices = np.flatnonzero(phase_id == 2)

        half_dc_arrival, peak_rate_arrival = first_crossing_and_peak_rate(
            h5["interface/cumulative_slip"],
            shear_indices,
            shear_time_ms,
            0.5 * critical_slip,
            chunk_frames=args.chunk_frames,
        )
        nucleation_taper_length_mm = float(
            h5.attrs.get("loading_edge_nucleation_length", 0.0)
        )
        loading_edge_index = int(
            np.argmin(contact_y - float(np.min(contact_y)))
        )
        nucleation_reference_time_ms = float(
            half_dc_arrival[loading_edge_index]
        )
        nucleation_reference_frame = int(
            shear_indices[
                np.argmin(
                    np.abs(
                        shear_time_ms[shear_indices]
                        - nucleation_reference_time_ms
                    )
                )
            ]
        )
        master_nodes = np.asarray(
            h5["interface/master_nodes"], dtype=np.int64
        )
        slave_nodes = np.asarray(
            h5["interface/slave_nodes"], dtype=np.int64
        )
        tangential_penalty = np.asarray(
            h5["interface/tangential_penalty_profile"], dtype=np.float64
        )
        relative_displacement = (
            np.asarray(
                h5["moving/displacement"][
                    nucleation_reference_frame, master_nodes, :
                ],
                dtype=np.float64,
            )
            - np.asarray(
                h5["stationary/displacement"][
                    nucleation_reference_frame, slave_nodes, :
                ],
                dtype=np.float64,
            )
        )
        plastic_slip = np.asarray(
            h5["interface/plastic_slip"][nucleation_reference_frame],
            dtype=np.float64,
        )
        nucleation_tau_profile = tangential_penalty * (
            relative_displacement[:, 1] - plastic_slip
        )
        normal_penalty = 10.0 * float(np.max(tangential_penalty))
        nucleation_sigma_profile = normal_penalty * np.maximum(
            relative_displacement[:, 0],
            0.0,
        )
        distance_from_loading_edge = contact_y - float(np.min(contact_y))
        nucleation_zone = (
            (distance_from_loading_edge >= 0.0)
            & (
                distance_from_loading_edge
                <= nucleation_taper_length_mm
            )
        )
        nucleation_initial_tau_mpa = float(
            np.mean(nucleation_tau_profile[nucleation_zone])
        )
        nucleation_normal_stress_mpa = float(
            np.mean(nucleation_sigma_profile[nucleation_zone])
        )
        stopped = (
            history[:, history_columns.index("shear_loading_stopped")] > 0.5
        )
        stop_indices = np.flatnonzero(stopped)
        stop_time_ms = (
            float(shear_time_ms[stop_indices[0]])
            if len(stop_indices)
            else float("nan")
        )

        young_modulus_mpa = float(
            h5.attrs.get("young_modulus", h5.attrs.get("E", 7662.0))
        )
        poisson_ratio = float(
            h5.attrs.get("poisson_ratio", h5.attrs.get("nu", 0.2))
        )
        density_tonne_per_mm3 = float(
            h5.attrs.get("density", h5.attrs.get("rho", 1.148e-9))
        )
        dt_s = float(h5.attrs["dt"])
        creep_relaxation_time_s = float(
            h5.attrs.get("leading_edge_creep_relaxation_time", 0.0)
        )
        wave_speeds = material_wave_speeds(
            young_modulus_mpa,
            poisson_ratio,
            density_tonne_per_mm3,
        )
        czm_prediction = estimate_kammer_czm_speed(
            h5=h5,
            contact_y=contact_y,
            critical_slip=critical_slip,
            mu_static=mu_static,
            mu_kinetic=mu_kinetic,
            half_dc_arrival=half_dc_arrival,
            shear_indices=shear_indices,
            shear_time_ms=shear_time_ms,
            master_nodes=master_nodes,
            slave_nodes=slave_nodes,
            normal_penalty=normal_penalty,
            fit_start=args.fit_start,
            fit_end=args.fit_end,
            young_modulus_mpa=young_modulus_mpa,
            poisson_ratio=poisson_ratio,
            wave_speeds=wave_speeds,
        )

    half_dc_fit = linear_arrival_fit(
        contact_y,
        half_dc_arrival,
        args.fit_start,
        args.fit_end,
    )
    peak_rate_fit = linear_arrival_fit(
        contact_y,
        peak_rate_arrival,
        args.fit_start,
        args.fit_end,
    )
    speed_paths = plot_rupture_speed(
        run_id=run_id,
        contact_y=contact_y,
        half_dc_arrival=half_dc_arrival,
        peak_rate_arrival=peak_rate_arrival,
        half_dc_fit=half_dc_fit,
        peak_rate_fit=peak_rate_fit,
        stop_time_ms=stop_time_ms,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
        wave_speeds=wave_speeds,
        czm_prediction=czm_prediction,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    profile_paths = plot_fault_profile(
        run_id=run_id,
        contact_y=contact_y,
        critical_slip=critical_slip,
        mu_static=mu_static,
        mu_kinetic=mu_kinetic,
        creep_weight=creep_weight,
        stop_time_ms=stop_time_ms,
        young_modulus_mpa=young_modulus_mpa,
        poisson_ratio=poisson_ratio,
        nucleation_initial_tau_mpa=nucleation_initial_tau_mpa,
        nucleation_normal_stress_mpa=nucleation_normal_stress_mpa,
        nucleation_reference_time_ms=nucleation_reference_time_ms,
        nucleation_taper_length_mm=nucleation_taper_length_mm,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    creep_paths = plot_creep_mechanism(
        run_id=run_id,
        contact_y=contact_y,
        creep_weight=creep_weight,
        mu_static=mu_static,
        mu_kinetic=mu_kinetic,
        dt_s=dt_s,
        relaxation_time_s=creep_relaxation_time_s,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    payload = {
        "run_id": run_id,
        "fit_interval_mm": [args.fit_start, args.fit_end],
        "half_dc_speed_m_per_s": half_dc_fit["speed_m_per_s"],
        "half_dc_r_squared": half_dc_fit["r_squared"],
        "peak_rate_speed_m_per_s": peak_rate_fit["speed_m_per_s"],
        "peak_rate_r_squared": peak_rate_fit["r_squared"],
        "representative_speed_m_per_s": 0.5
        * (
            float(half_dc_fit["speed_m_per_s"])
            + float(peak_rate_fit["speed_m_per_s"])
        ),
        "wave_speeds_m_per_s": wave_speeds,
        "kammer_mclaskey_czm": czm_prediction,
        "loading_stop_time_ms": stop_time_ms,
        "nucleation_length_reference": {
            "source": "Ke, McLaskey, and Kammer (2022), Supplementary Note S1",
            "reference_time_ms": nucleation_reference_time_ms,
            "averaging_interval_mm": [0.0, nucleation_taper_length_mm],
            "tau_i_mpa": nucleation_initial_tau_mpa,
            "sigma_n_mpa": nucleation_normal_stress_mpa,
        },
        "outputs": [
            str(path)
            for path in (*speed_paths, *profile_paths, *creep_paths)
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
