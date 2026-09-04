from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_rupture_speed_and_fault_profile import (
    decode_strings,
    linear_arrival_fit,
    material_wave_speeds,
)
from tatva.pmma.profiles import regularized_steady_friction


INK = "#202124"
MUTED = "#5F6368"
GRID = "#D7DADD"
NAVY = "#24557A"
TEAL = "#00838F"
ORANGE = "#D55E00"
GOLD = "#B8860B"
RED = "#A33A2B"
PALE_GOLD = "#E8C66A"
PALE_BLUE = "#70A9B0"
PALE_RED = "#D58A7B"


def configure_journal_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "axes.titleweight": "normal",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.7,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_journal_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    save_options = {"bbox_inches": "tight", "pad_inches": 0.04, "facecolor": "white"}
    figure.savefig(png_path, dpi=dpi, **save_options)
    figure.savefig(pdf_path, **save_options)
    plt.close(figure)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RSF rupture speed, spatial profiles, and constitutive mechanism."
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fit-start", type=float, default=120.0)
    parser.add_argument("--fit-end", type=float, default=440.0)
    parser.add_argument("--chunk-frames", type=int, default=2048)
    parser.add_argument(
        "--velocity-thresholds",
        type=float,
        nargs=2,
        default=(500.0, 1000.0),
        metavar=("LOW", "HIGH"),
        help="Dynamic slip-rate thresholds used for rupture arrival [mm/s].",
    )
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def first_velocity_crossing(
    dataset: h5py.Dataset,
    frame_indices: np.ndarray,
    shear_time_ms: np.ndarray,
    threshold_mm_s: float,
    *,
    chunk_frames: int,
) -> np.ndarray:
    if threshold_mm_s <= 0.0:
        raise ValueError("The slip-rate threshold must be positive.")
    if chunk_frames < 2:
        raise ValueError("chunk_frames must be at least 2.")
    if len(frame_indices) < 2 or not np.all(np.diff(frame_indices) == 1):
        raise ValueError("At least two contiguous shear frames are required.")

    arrivals = np.full(dataset.shape[1], np.nan, dtype=np.float64)
    unresolved = np.ones(dataset.shape[1], dtype=bool)
    first_frame = int(frame_indices[0])
    last_frame = int(frame_indices[-1])
    previous_values: np.ndarray | None = None
    previous_time: float | None = None

    for start in range(first_frame, last_frame + 1, chunk_frames):
        end = min(start + chunk_frames, last_frame + 1)
        values = np.abs(np.asarray(dataset[start:end], dtype=np.float64))
        times = shear_time_ms[start:end]
        if previous_values is not None and previous_time is not None:
            values = np.vstack([previous_values, values])
            times = np.concatenate([[previous_time], times])

        before = values[:-1]
        after = values[1:]
        crossing = (
            (before < threshold_mm_s)
            & (after >= threshold_mm_s)
            & unresolved[None, :]
        )
        crossed = np.any(crossing, axis=0)
        if np.any(crossed):
            stations = np.flatnonzero(crossed)
            intervals = np.argmax(crossing[:, crossed], axis=0)
            lower = before[intervals, stations]
            upper = after[intervals, stations]
            fraction = np.clip(
                (threshold_mm_s - lower) / np.maximum(upper - lower, 1.0e-30),
                0.0,
                1.0,
            )
            arrivals[stations] = times[intervals] + fraction * (
                times[intervals + 1] - times[intervals]
            )
            unresolved[stations] = False

        previous_values = values[-1].copy()
        previous_time = float(times[-1])
        if not np.any(unresolved):
            break
    return arrivals


def optional_linear_arrival_fit(
    position_mm: np.ndarray,
    arrival_time_ms: np.ndarray,
    fit_start_mm: float,
    fit_end_mm: float,
) -> dict[str, object]:
    fit_mask = (
        (position_mm >= fit_start_mm)
        & (position_mm <= fit_end_mm)
        & np.isfinite(arrival_time_ms)
    )
    finite_point_count = int(np.count_nonzero(fit_mask))
    if finite_point_count < 3:
        return {
            "available": False,
            "finite_point_count": finite_point_count,
            "reason": "fewer than three finite arrivals in the requested fit interval",
            "mask": fit_mask,
            "slope_ms_per_mm": None,
            "intercept_ms": None,
            "speed_m_per_s": None,
            "r_squared": None,
        }

    fit = linear_arrival_fit(
        position_mm,
        arrival_time_ms,
        fit_start_mm,
        fit_end_mm,
    )
    return {
        "available": True,
        "finite_point_count": finite_point_count,
        "reason": None,
        **fit,
    }


def _available_fit_speeds(*fits: dict[str, object]) -> list[float]:
    return [
        float(fit["speed_m_per_s"])
        for fit in fits
        if bool(fit["available"])
    ]


def _serializable_fit(fit: dict[str, object]) -> dict[str, object]:
    return {
        key: fit[key]
        for key in (
            "available",
            "finite_point_count",
            "reason",
            "slope_ms_per_mm",
            "intercept_ms",
            "speed_m_per_s",
            "r_squared",
        )
    }


def _zone_metadata(h5: h5py.File, contact_y: np.ndarray) -> dict[str, float]:
    spec = json.loads(str(h5.attrs.get("rsf_profile_spec_json", "{}")))
    y_min = float(np.min(contact_y))
    y_max = float(np.max(contact_y))
    loading_length = float(spec.get("loading_length", 30.0))
    leading_length = float(spec.get("leading_length", 30.0))
    transition = float(spec.get("transition_length", 10.0))
    loading_transition = float(
        spec.get("loading_transition_length", transition)
    )
    leading_transition = float(
        spec.get("leading_transition_length", transition)
    )
    return {
        "y_min": y_min,
        "y_max": y_max,
        "loading_end": y_min + loading_length,
        "loading_transition_end": (
            y_min + loading_length + loading_transition
        ),
        "leading_transition_start": (
            y_max - leading_length - leading_transition
        ),
        "leading_start": y_max - leading_length,
    }


def _shade_zones(axis: plt.Axes, zones: dict[str, float]) -> None:
    axis.axvspan(
        zones["y_min"], zones["loading_end"], color=PALE_GOLD, alpha=0.16, lw=0
    )
    axis.axvspan(
        zones["loading_end"],
        zones["loading_transition_end"],
        color=PALE_GOLD,
        alpha=0.08,
        lw=0,
    )
    axis.axvspan(
        zones["loading_transition_end"],
        zones["leading_transition_start"],
        color=PALE_BLUE,
        alpha=0.08,
        lw=0,
    )
    axis.axvspan(
        zones["leading_transition_start"],
        zones["leading_start"],
        color=PALE_RED,
        alpha=0.08,
        lw=0,
    )
    axis.axvspan(
        zones["leading_start"], zones["y_max"], color=PALE_RED, alpha=0.15, lw=0
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
    )


def _plot_speed(
    *,
    contact_y: np.ndarray,
    low_arrival: np.ndarray,
    high_arrival: np.ndarray,
    low_fit: dict[str, object],
    high_fit: dict[str, object],
    velocity_thresholds: tuple[float, float],
    fit_start: float,
    fit_end: float,
    stop_time_ms: float,
    zones: dict[str, float],
    wave_speeds: dict[str, float],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    figure, (axis, speed_axis) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.15),
        gridspec_kw={"width_ratios": (2.25, 1.0)},
        constrained_layout=True,
    )
    _shade_zones(axis, zones)
    low_threshold, high_threshold = velocity_thresholds
    axis.plot(
        contact_y,
        low_arrival,
        color=TEAL,
        lw=1.35,
        label=rf"$|V|={low_threshold:g}$ mm s$^{{-1}}$",
    )
    axis.plot(
        contact_y,
        high_arrival,
        color=NAVY,
        lw=1.05,
        label=rf"$|V|={high_threshold:g}$ mm s$^{{-1}}$",
    )
    fit_y = np.linspace(fit_start, fit_end, 300)
    for threshold, fit, color, line_style in (
        (low_threshold, low_fit, ORANGE, (0, (5, 3))),
        (high_threshold, high_fit, GOLD, (0, (2, 2))),
    ):
        if not bool(fit["available"]):
            continue
        axis.plot(
            fit_y,
            float(fit["slope_ms_per_mm"]) * fit_y
            + float(fit["intercept_ms"]),
            color=color,
            lw=1.35,
            ls=line_style,
            label=(
                rf"Fit at {threshold:g} mm s$^{{-1}}$, "
                f"{fit_start:.0f}-{fit_end:.0f} mm"
            ),
        )
    if np.isfinite(stop_time_ms):
        axis.axhline(
            stop_time_ms,
            color=RED,
            lw=0.9,
            ls=(0, (3, 2)),
            label="Loading stopped",
        )
    finite = np.concatenate(
        [low_arrival[np.isfinite(low_arrival)], high_arrival[np.isfinite(high_arrival)]]
    )
    if finite.size:
        pad = max(0.05, 0.06 * float(np.ptp(finite)))
        axis.set_ylim(float(np.min(finite)) - pad, float(np.max(finite)) + pad)
    axis.set_xlim(zones["y_min"], zones["y_max"])
    axis.set_xlabel("Position along fault, y [mm]")
    axis.set_ylabel("Arrival time after shear phase begins [ms]")
    axis.set_title("Rupture-front arrival", loc="left")
    axis.grid(axis="y")
    axis.legend(loc="best", handlelength=2.5)
    axis.spines[["top", "right"]].set_visible(False)
    _panel_label(axis, "(a)")

    available_speeds = _available_fit_speeds(low_fit, high_fit)
    labels = [r"$c_R$", r"$c_s$", r"$c_p$"]
    values = [
        wave_speeds["c_r"] / 1e3,
        wave_speeds["c_s"] / 1e3,
        wave_speeds["c_p"] / 1e3,
    ]
    colors = [GOLD, TEAL, NAVY]
    if available_speeds:
        representative_speed = float(np.mean(available_speeds))
        measured_label = r"$v_r$"
        if len(available_speeds) == 1:
            available_threshold = (
                low_threshold if bool(low_fit["available"]) else high_threshold
            )
            measured_label = rf"$v_r$ ({available_threshold:g})"
        labels.insert(0, measured_label)
        values.insert(0, representative_speed / 1e3)
        colors.insert(0, ORANGE)
    positions = np.arange(len(values))
    speed_axis.hlines(positions, 0.0, values, color=colors, alpha=0.55, lw=2.1)
    speed_axis.scatter(values, positions, color=colors, s=25, zorder=3)
    for position, value in zip(positions, values, strict=True):
        speed_axis.text(
            value + 0.045,
            position,
            f"{value:.2f}",
            va="center",
            color=INK,
            fontsize=7.5,
        )
    speed_axis.set_yticks(positions, labels)
    speed_axis.invert_yaxis()
    speed_axis.set_xlim(0.0, 1.12 * max(values))
    speed_axis.set_xlabel("Speed [km/s]")
    fit_notes = []
    for label, fit in (("low", low_fit), ("high", high_fit)):
        if bool(fit["available"]):
            fit_notes.append(
                rf"$R^2_{{V_{{{label[0]}}}}}="
                rf"{float(fit['r_squared']):.4f}$"
            )
        else:
            fit_notes.append(
                f"{label} fit unavailable (n={int(fit['finite_point_count'])})"
            )
    speed_axis.set_title(
        "Measured and material speeds\n" + "; ".join(fit_notes),
        loc="left",
        fontsize=8.2,
    )
    speed_axis.grid(axis="x")
    speed_axis.spines[["top", "right", "left"]].set_visible(False)
    _panel_label(speed_axis, "(b)")
    return save_journal_figure(figure, output_dir, "rupture_speed_stable_fit", dpi)


def _plot_profile(
    *,
    contact_y: np.ndarray,
    direct_effect: np.ndarray,
    state_effect: np.ndarray,
    characteristic_slip: np.ndarray,
    zones: dict[str, float],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 5.4),
        sharex=True,
        constrained_layout=True,
    )
    for axis in axes:
        _shade_zones(axis, zones)
        axis.grid(axis="y")
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].plot(contact_y, direct_effect, color=TEAL, lw=1.35, label=r"Direct effect, $a$")
    axes[0].plot(contact_y, state_effect, color=NAVY, lw=1.35, label=r"State effect, $b$")
    axes[0].set_ylabel("RSF parameter")
    axes[0].set_title("Constitutive coefficients", loc="left")
    axes[0].legend(ncol=2, loc="upper center")

    weakening = direct_effect - state_effect
    axes[1].axhline(0.0, color=INK, lw=0.75)
    axes[1].plot(contact_y, weakening, color=ORANGE, lw=1.45)
    axes[1].fill_between(
        contact_y,
        0.0,
        weakening,
        where=weakening < 0.0,
        color=ORANGE,
        alpha=0.14,
        label=r"$a-b<0$: velocity weakening",
    )
    axes[1].fill_between(
        contact_y,
        0.0,
        weakening,
        where=weakening > 0.0,
        color=TEAL,
        alpha=0.14,
        label=r"$a-b>0$: velocity strengthening",
    )
    axes[1].set_ylabel(r"$a-b$")
    axes[1].set_title("Steady-state velocity dependence", loc="left")
    axes[1].legend(ncol=2, loc="lower center")

    axes[2].plot(contact_y, 1.0e3 * characteristic_slip, color=RED, lw=1.45)
    axes[2].set_ylabel(r"$D_c$ [$\mu$m]")
    axes[2].set_xlabel("Position along fault, y [mm]")
    axes[2].set_title("Characteristic slip distance", loc="left")
    axes[2].set_xlim(zones["y_min"], zones["y_max"])
    for axis, label in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        _panel_label(axis, label)
    return save_journal_figure(figure, output_dir, "fault_interface_profile", dpi)


def _plot_mechanism(
    *,
    direct_effect: np.ndarray,
    state_effect: np.ndarray,
    reference_friction: np.ndarray,
    reference_velocity: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    figure, (equation_axis, curve_axis) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        gridspec_kw={"width_ratios": (1.0, 1.2)},
        constrained_layout=True,
    )
    equation_axis.axis("off")
    equation_axis.set_title("RSF formulation (ageing law)", loc="left", pad=8.0)
    equations = [
        (0.86, "State evolution", (r"$\dot{\theta}=1-|V|\theta/D_c$",)),
        (
            0.57,
            "Regularized shear strength",
            (
                r"$\tau=a\sigma_n\,\mathrm{asinh}(\Xi)$",
                r"$\Xi=\dfrac{|V|}{2V_0}\exp\!\left[\dfrac{f_0+b\ln(V_0\theta/D_c)}{a}\right]$",
            ),
        ),
        (
            0.20,
            "Steady state",
            (
                r"$\theta_{ss}=D_c/|V|$",
                r"$\partial f_{ss}/\partial\ln V\approx a-b$",
            ),
        ),
    ]
    for y, title, equation_lines in equations:
        equation_axis.text(
            0.0,
            y,
            title,
            transform=equation_axis.transAxes,
            fontsize=8.2,
            fontweight="bold",
            color=INK,
        )
        for line_index, equation in enumerate(equation_lines):
            equation_axis.text(
                0.0,
                y - 0.12 - 0.095 * line_index,
                equation,
                transform=equation_axis.transAxes,
                fontsize=8.0 if len(equation_lines) == 1 else 7.4,
                color=NAVY,
            )
    _panel_label(equation_axis, "(a)")

    velocity = np.logspace(-5, 4, 500)
    indices = [0, len(direct_effect) // 2, len(direct_effect) - 1]
    labels = ["Loading end", "Middle", "Leading edge"]
    colors = [ORANGE, NAVY, TEAL]
    for index, label, color in zip(indices, labels, colors, strict=True):
        friction = np.asarray(
            [
                regularized_steady_friction(
                    velocity=float(value),
                    reference_friction=float(reference_friction[index]),
                    direct_effect=float(direct_effect[index]),
                    state_effect=float(state_effect[index]),
                    reference_velocity=float(reference_velocity[index]),
                )
                for value in velocity
            ]
        )
        curve_axis.semilogx(
            velocity,
            friction,
            color=color,
            lw=1.45,
            label=rf"{label}: $a-b={direct_effect[index] - state_effect[index]:+.4f}$",
        )
    curve_axis.set_xlabel(r"Steady slip rate, $|V|$ [mm s$^{-1}$]")
    curve_axis.set_ylabel(r"Steady friction coefficient, $f_{ss}$")
    curve_axis.set_title("Steady-state friction", loc="left")
    curve_axis.grid()
    curve_axis.legend(loc="best")
    curve_axis.spines[["top", "right"]].set_visible(False)
    _panel_label(curve_axis, "(b)")
    return save_journal_figure(figure, output_dir, "rsf_mechanism", dpi)


def main() -> int:
    args = parse_args()
    velocity_thresholds = tuple(sorted(float(value) for value in args.velocity_thresholds))
    if velocity_thresholds[0] <= 0.0:
        raise ValueError("Velocity thresholds must be positive.")
    data_path = args.data_path.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_path.parent.parent / "Plot"
    )
    configure_journal_style()

    with h5py.File(data_path, "r") as h5:
        friction_law = str(h5.attrs.get("friction_law", ""))
        if not friction_law.startswith("rate-state"):
            raise ValueError(f"Expected rate-state dump, got {friction_law!r}.")
        interface = h5["interface"]
        contact_y = np.asarray(interface["contact_line_y"], dtype=np.float64)
        characteristic_slip = np.asarray(
            interface["rsf_characteristic_slip_profile"], dtype=np.float64
        )
        direct_effect = np.asarray(interface["rsf_direct_effect_profile"], dtype=np.float64)
        state_effect = np.asarray(interface["rsf_state_effect_profile"], dtype=np.float64)
        reference_friction = np.asarray(
            interface["rsf_reference_friction_profile"], dtype=np.float64
        )
        reference_velocity = np.asarray(
            interface["rsf_reference_velocity_profile"], dtype=np.float64
        )
        history = np.asarray(h5["history"], dtype=np.float64)
        columns = decode_strings(np.asarray(h5["history_columns"]))
        phase_id = np.asarray(h5["phase_id"], dtype=np.int8)
        pressure_time = float(h5.attrs["pressure_steps"] * h5.attrs["dt"])
        shear_time_ms = (history[:, columns.index("time")] - pressure_time) * 1e3
        shear_indices = np.flatnonzero(phase_id == 2)
        low_arrival = first_velocity_crossing(
            interface["slip_rate"],
            shear_indices,
            shear_time_ms,
            velocity_thresholds[0],
            chunk_frames=args.chunk_frames,
        )
        high_arrival = first_velocity_crossing(
            interface["slip_rate"],
            shear_indices,
            shear_time_ms,
            velocity_thresholds[1],
            chunk_frames=args.chunk_frames,
        )
        stopped = history[:, columns.index("shear_loading_stopped")] > 0.5
        stop_indices = np.flatnonzero(stopped)
        stop_time_ms = float(shear_time_ms[stop_indices[0]]) if len(stop_indices) else float("nan")
        young = float(h5.attrs.get("young_modulus", h5.attrs.get("E", 7662.0)))
        poisson = float(h5.attrs.get("poisson_ratio", h5.attrs.get("nu", 0.2)))
        density = float(h5.attrs.get("density", h5.attrs.get("rho", 1.148e-9)))
        zones = _zone_metadata(h5, contact_y)

    low_fit = optional_linear_arrival_fit(
        contact_y, low_arrival, args.fit_start, args.fit_end
    )
    high_fit = optional_linear_arrival_fit(
        contact_y, high_arrival, args.fit_start, args.fit_end
    )
    wave_speeds = material_wave_speeds(young, poisson, density)
    speed_paths = _plot_speed(
        contact_y=contact_y,
        low_arrival=low_arrival,
        high_arrival=high_arrival,
        low_fit=low_fit,
        high_fit=high_fit,
        velocity_thresholds=velocity_thresholds,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
        stop_time_ms=stop_time_ms,
        zones=zones,
        wave_speeds=wave_speeds,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    profile_paths = _plot_profile(
        contact_y=contact_y,
        direct_effect=direct_effect,
        state_effect=state_effect,
        characteristic_slip=characteristic_slip,
        zones=zones,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    mechanism_paths = _plot_mechanism(
        direct_effect=direct_effect,
        state_effect=state_effect,
        reference_friction=reference_friction,
        reference_velocity=reference_velocity,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    available_speeds = _available_fit_speeds(low_fit, high_fit)
    stable_speed = (
        float(np.mean(available_speeds)) if len(available_speeds) == 2 else None
    )
    representative_speed = (
        float(np.mean(available_speeds)) if available_speeds else None
    )
    payload = {
        "friction_law": friction_law,
        "fit_interval_mm": [args.fit_start, args.fit_end],
        "velocity_thresholds_mm_s": velocity_thresholds,
        "low_threshold_speed_m_per_s": low_fit["speed_m_per_s"],
        "high_threshold_speed_m_per_s": high_fit["speed_m_per_s"],
        "stable_speed_m_per_s": stable_speed,
        "representative_speed_m_per_s": representative_speed,
        "arrival_fits": {
            "low_velocity_threshold": _serializable_fit(low_fit),
            "high_velocity_threshold": _serializable_fit(high_fit),
        },
        "arrival_definition": (
            "Stable speed is the mean of linear fits to the first crossings of two "
            "dynamic slip-rate thresholds when both fits are available. The "
            "representative speed uses whichever threshold fits are available. "
            "The thresholds exceed the imposed loading velocity, so quasistatic "
            "creep and later global peak-rate arrivals are excluded."
        ),
        "loading_stop_time_ms": stop_time_ms if np.isfinite(stop_time_ms) else None,
        "wave_speeds_m_per_s": wave_speeds,
        "lsw_czm_prediction": "not applicable to this RSF run",
        "outputs": [
            str(path)
            for path in (*speed_paths, *profile_paths, *mechanism_paths)
        ],
    }
    metrics_path = output_dir / "rsf_rupture_analysis_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
