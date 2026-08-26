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
    first_crossing_and_peak_rate,
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
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def _zone_metadata(h5: h5py.File, contact_y: np.ndarray) -> dict[str, float]:
    spec = json.loads(str(h5.attrs.get("rsf_profile_spec_json", "{}")))
    y_min = float(np.min(contact_y))
    y_max = float(np.max(contact_y))
    loading_length = float(spec.get("loading_length", 30.0))
    leading_length = float(spec.get("leading_length", 30.0))
    transition = float(spec.get("transition_length", 10.0))
    return {
        "y_min": y_min,
        "y_max": y_max,
        "loading_end": y_min + loading_length,
        "loading_transition_end": y_min + loading_length + transition,
        "leading_transition_start": y_max - leading_length - transition,
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
    half_arrival: np.ndarray,
    dc_arrival: np.ndarray,
    half_fit: dict[str, object],
    dc_fit: dict[str, object],
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
    axis.plot(contact_y, half_arrival, color=TEAL, lw=1.35, label=r"$0.5D_c$ crossing")
    axis.plot(contact_y, dc_arrival, color=NAVY, lw=1.05, label=r"$D_c$ crossing")
    fit_y = np.linspace(fit_start, fit_end, 300)
    axis.plot(
        fit_y,
        float(half_fit["slope_ms_per_mm"]) * fit_y + float(half_fit["intercept_ms"]),
        color=ORANGE,
        lw=1.45,
        ls=(0, (5, 3)),
        label=f"Linear fit, {fit_start:.0f}-{fit_end:.0f} mm",
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
        [half_arrival[np.isfinite(half_arrival)], dc_arrival[np.isfinite(dc_arrival)]]
    )
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

    half_speed = float(half_fit["speed_m_per_s"])
    dc_speed = float(dc_fit["speed_m_per_s"])
    measured = 0.5 * (half_speed + dc_speed)
    labels = [r"$v_r$", r"$c_R$", r"$c_s$", r"$c_p$"]
    values = [
        measured / 1e3,
        wave_speeds["c_r"] / 1e3,
        wave_speeds["c_s"] / 1e3,
        wave_speeds["c_p"] / 1e3,
    ]
    colors = [ORANGE, GOLD, TEAL, NAVY]
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
    speed_axis.set_title(
        "Measured and material speeds\n"
        rf"$v_r={measured / 1e3:.3f}$ km s$^{{-1}}$; "
        rf"$R^2_{{0.5D_c}}={float(half_fit['r_squared']):.4f}$, "
        rf"$R^2_{{D_c}}={float(dc_fit['r_squared']):.4f}$",
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
        half_arrival, _ = first_crossing_and_peak_rate(
            interface["cumulative_slip"],
            shear_indices,
            shear_time_ms,
            0.5 * characteristic_slip,
            chunk_frames=args.chunk_frames,
        )
        dc_arrival, _ = first_crossing_and_peak_rate(
            interface["cumulative_slip"],
            shear_indices,
            shear_time_ms,
            characteristic_slip,
            chunk_frames=args.chunk_frames,
        )
        stopped = history[:, columns.index("shear_loading_stopped")] > 0.5
        stop_indices = np.flatnonzero(stopped)
        stop_time_ms = float(shear_time_ms[stop_indices[0]]) if len(stop_indices) else float("nan")
        young = float(h5.attrs.get("young_modulus", h5.attrs.get("E", 7662.0)))
        poisson = float(h5.attrs.get("poisson_ratio", h5.attrs.get("nu", 0.2)))
        density = float(h5.attrs.get("density", h5.attrs.get("rho", 1.148e-9)))
        zones = _zone_metadata(h5, contact_y)

    half_fit = linear_arrival_fit(contact_y, half_arrival, args.fit_start, args.fit_end)
    dc_fit = linear_arrival_fit(contact_y, dc_arrival, args.fit_start, args.fit_end)
    wave_speeds = material_wave_speeds(young, poisson, density)
    speed_paths = _plot_speed(
        contact_y=contact_y,
        half_arrival=half_arrival,
        dc_arrival=dc_arrival,
        half_fit=half_fit,
        dc_fit=dc_fit,
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
    payload = {
        "friction_law": friction_law,
        "fit_interval_mm": [args.fit_start, args.fit_end],
        "half_dc_speed_m_per_s": half_fit["speed_m_per_s"],
        "dc_speed_m_per_s": dc_fit["speed_m_per_s"],
        "stable_speed_m_per_s": 0.5
        * (float(half_fit["speed_m_per_s"]) + float(dc_fit["speed_m_per_s"])),
        "arrival_fits": {
            "half_dc": {
                key: half_fit[key]
                for key in (
                    "slope_ms_per_mm",
                    "intercept_ms",
                    "speed_m_per_s",
                    "r_squared",
                )
            },
            "dc": {
                key: dc_fit[key]
                for key in (
                    "slope_ms_per_mm",
                    "intercept_ms",
                    "speed_m_per_s",
                    "r_squared",
                )
            },
        },
        "arrival_definition": (
            "Mean of linear fits to the first 0.5D_c and first D_c crossings. "
            "The global peak slip-rate time is excluded because later wave arrivals "
            "can make it noncausal along the fault."
        ),
        "loading_stop_time_ms": stop_time_ms,
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
