#!/usr/bin/env python3
"""Zoom the shear-phase friction map onto the rupture and overlay multi-threshold arrivals.

The full mu_eff_map_phase_split view spans the whole shear window, so a front that
crosses the fault in a fraction of a millisecond collapses into a near-horizontal
line. This figure restricts time to the rupture, overlays the first local
D_c-slip crossing and the first saved slip-rate crossings of several thresholds,
fits each over the same interval, and marks the shear-loading stop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tatva.pmma.plotting import configure_journal_style, style_axis
from plot_contact_friction_map import (
    RUPTURE_FIT_START_MM,
    _automatic_mu_norm,
    _effective_friction_colormap,
    _first_slip_distance_crossing,
    _fit_end_before_leading_zone,
    _read_wave_speeds,
    _save_with_png,
)
from plot_rsf_rupture_analysis import optional_linear_arrival_fit


# Slip-rate thresholds in mm/s (numerically mm/ms = m/s * 1e3).
SLIP_RATE_THRESHOLDS_MM_PER_S = (100.0, 500.0, 1000.0)
LOCAL_SPEED_BIN_MM = 50.0
ARRIVAL_STYLES = {
    "slip_dc": ("white", "-", r"$\Delta\delta=D_c$"),
    "rate_100": ("#ffb000", "--", r"$V\geq0.1$ m s$^{-1}$"),
    "rate_500": ("#ff6f00", ":", r"$V\geq0.5$ m s$^{-1}$"),
    "rate_1000": ("#e8384f", "-.", r"$V\geq1$ m s$^{-1}$"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="PDF path; a PNG is written beside it.")
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--t-min", type=float, help="Shear-relative window start [ms]; default automatic.")
    parser.add_argument("--t-max", type=float, help="Shear-relative window end [ms]; default automatic.")
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def first_threshold_crossing(values: np.ndarray, time_ms: np.ndarray, threshold: float) -> np.ndarray:
    """First saved time each station reaches ``threshold`` (NaN if never)."""
    reached = np.abs(values) >= threshold
    arrivals = np.full(values.shape[1], np.nan, dtype=np.float64)
    hit = np.any(reached, axis=0)
    arrivals[hit] = time_ms[np.argmax(reached[:, hit], axis=0)]
    return arrivals


def local_speeds(y: np.ndarray, arrival_ms: np.ndarray, width_mm: float = LOCAL_SPEED_BIN_MM) -> list[dict]:
    """Least-squares speed in consecutive bins; signed, so backward fronts show as negative."""
    rows = []
    for start in np.arange(float(y.min()), float(y.max()), width_mm):
        mask = (y >= start) & (y < start + width_mm) & np.isfinite(arrival_ms)
        speed = None
        if np.count_nonzero(mask) >= 3 and np.ptp(arrival_ms[mask]) > 0.0:
            slope = np.polyfit(y[mask], arrival_ms[mask], 1)[0]
            speed = None if slope == 0.0 else float(1.0 / slope)
        rows.append({"y_start_mm": float(start), "y_end_mm": float(start + width_mm), "speed_m_per_s": speed})
    return rows


def automatic_window(arrivals: dict[str, np.ndarray], shear_end_ms: float) -> tuple[float, float]:
    """Span from the earliest slip-rate arrival to the late D_c arrivals, padded."""
    early = [np.nanmin(a) for a in arrivals.values() if np.isfinite(a).any()]
    late = arrivals["slip_dc"][np.isfinite(arrivals["slip_dc"])]
    if not early or late.size == 0:
        return 0.0, shear_end_ms
    start, end = min(early), float(np.percentile(late, 99.0))
    pad = max(0.15 * (end - start), 0.2)
    return max(0.0, start - pad), min(shear_end_ms, end + pad)


def plot_rupture_arrival_zoom(
    input_path: Path,
    output_path: Path,
    stats_dir: Path,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    dpi: int = 260,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as h5:
        interface = h5["interface"]
        y = np.asarray(h5["moving/coords"], dtype=np.float64)[np.asarray(interface["master_nodes"]), 1]
        order = np.argsort(y)
        y = y[order]
        phase_id = np.asarray(h5["phase_id"])
        shear = np.flatnonzero(phase_id == 2)
        if shear.size < 2:
            raise ValueError("The dump has fewer than two shear-phase frames.")
        dt = float(h5.attrs["dt"])
        # Exact integer-step clock, relative to the start of the shear phase.
        time_ms = np.asarray(h5["step_id"], dtype=np.float64)[shear] * dt * 1e3
        if "rsf_characteristic_slip_profile" in interface:
            critical_slip = np.asarray(interface["rsf_characteristic_slip_profile"], dtype=np.float64)
        else:
            critical_slip = np.asarray(interface["critical_slip_profile"], dtype=np.float64)
        critical_slip = critical_slip[order]
        creep_weight = (
            np.asarray(interface["creep_weight_profile"], dtype=np.float64)[order]
            if "creep_weight_profile" in interface else None
        )
        profile_spec = json.loads(str(h5.attrs.get("rsf_profile_spec_json", "{}")))
        # The last normal frame is the slip baseline, as in the mu map.
        base = max(int(shear[0]) - 1, 0)
        slip = np.asarray(interface["cumulative_slip"][base:shear[-1] + 1], dtype=np.float64)[:, order]
        slip_time = np.concatenate([[0.0], time_ms]) if base < shear[0] else time_ms
        rate = np.asarray(interface["slip_rate"][shear[0]:shear[-1] + 1], dtype=np.float64)[:, order]
        columns = [c.decode() if isinstance(c, bytes) else str(c) for c in h5["history_columns"][:]]
        stopped = np.asarray(h5["history"][shear[0]:shear[-1] + 1, columns.index("shear_loading_stopped")]) > 0.5
        wave_speeds = _read_wave_speeds(input_path, h5)

        arrivals = {"slip_dc": _first_slip_distance_crossing(slip, slip_time, critical_slip)}
        for threshold in SLIP_RATE_THRESHOLDS_MM_PER_S:
            arrivals[f"rate_{int(threshold)}"] = first_threshold_crossing(rate, time_ms, threshold)
        del slip, rate
        auto_start, auto_end = automatic_window(arrivals, float(time_ms[-1]))
        window_start = auto_start if t_min is None else float(t_min)
        window_end = auto_end if t_max is None else float(t_max)
        frames = np.flatnonzero((time_ms >= window_start) & (time_ms <= window_end))
        if frames.size < 2:
            raise ValueError("The requested time window holds fewer than two saved frames.")
        mu = np.asarray(
            interface["friction_coefficient"][shear[frames[0]]:shear[frames[-1]] + 1],
            dtype=np.float64,
        )[:, order]
        window_time = time_ms[frames[0]:frames[-1] + 1]

    stop_time = float(time_ms[np.argmax(stopped)]) if stopped.any() else None
    fit_end, vs_start, creep_start = _fit_end_before_leading_zone(y, profile_spec, creep_weight)
    fits = {name: optional_linear_arrival_fit(y, arrival, RUPTURE_FIT_START_MM, fit_end)
            for name, arrival in arrivals.items()}

    configure_journal_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    mesh = ax.pcolormesh(y, window_time, mu, shading="auto", cmap=_effective_friction_colormap(),
                         norm=_automatic_mu_norm(mu), rasterized=True)
    labels = []
    for name, arrival in arrivals.items():
        color, style, label = ARRIVAL_STYLES[name]
        speed = fits[name]["speed_m_per_s"]
        text = label if speed is None else f"{label}: $v_r$={speed:.0f} m s$^{{-1}}$"
        ax.plot(y, arrival, color=color, ls=style, lw=1.1, label=text)
        labels.append(text)
    if stop_time is not None:
        ax.axhline(stop_time, color="#d0d0d0", lw=0.8, ls=(0, (2, 2)))
        ax.text(0.01, stop_time, f" shear loading stop {stop_time:.3f} ms", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", color="#e0e0e0", fontsize=6.8)
    ax.axvspan(RUPTURE_FIT_START_MM, fit_end, ymin=0.0, ymax=0.015, color="#ff6f00", lw=0)
    ax.set_xlim(float(y.min()), float(y.max()))
    ax.set_ylim(float(window_time[0]), float(window_time[-1]))
    ax.set_xlabel(r"Position along fault, $y$ [mm]")
    ax.set_ylabel("Shear phase time [ms]")
    zone = " (creep excluded)" if creep_start is not None else " (VS excluded)" if vs_start is not None else ""
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.04), fontsize=6.6, frameon=True, facecolor=(0.08, 0.08, 0.08, 0.82),
              edgecolor="none", labelcolor="white",
              title=f"Fit {RUPTURE_FIT_START_MM:.0f}-{fit_end:.0f} mm{zone};"
                    f" $C_R$={wave_speeds['c_r']:.0f} m s$^{{-1}}$",
              title_fontsize=6.6)
    ax.get_legend().get_title().set_color("white")
    style_axis(ax, grid=False)
    fig.colorbar(mesh, ax=ax, label="Effective friction coefficient")
    with matplotlib.rc_context({"savefig.dpi": dpi}):
        png_path = _save_with_png(fig, output_path)
    plt.close(fig)

    def finite_or_none(value):
        return None if value is None or not np.isfinite(value) else float(value)

    summary = {
        "input": str(input_path),
        "outputs": {"pdf": str(output_path), "png": str(png_path)},
        "time_reference": "shear-phase start, exact integer steps",
        "window_ms": [float(window_time[0]), float(window_time[-1])],
        "shear_loading_stop_first_saved_ms": stop_time,
        "fit_interval_mm": [RUPTURE_FIT_START_MM, fit_end],
        "leading_vs_start_mm": vs_start,
        "leading_creep_start_mm": creep_start,
        "rayleigh_wave_speed_m_per_s": wave_speeds["c_r"],
        "shear_wave_speed_m_per_s": wave_speeds["c_s"],
        "arrival_definitions": {
            "slip_dc": "first post-shear cumulative slip increment = local D_c (interpolated)",
            **{f"rate_{int(t)}": f"first saved |slip rate| >= {t / 1000:g} m/s"
               for t in SLIP_RATE_THRESHOLDS_MM_PER_S},
        },
        "arrivals": {},
    }
    for name, arrival in arrivals.items():
        finite = np.isfinite(arrival)
        fit = fits[name]
        summary["arrivals"][name] = {
            "reached_fraction": float(np.mean(finite)),
            "first_time_ms": finite_or_none(np.nanmin(arrival)) if finite.any() else None,
            "first_position_mm": float(y[np.nanargmin(arrival)]) if finite.any() else None,
            "last_time_ms": finite_or_none(np.nanmax(arrival)) if finite.any() else None,
            "fit_available": bool(fit["available"]),
            "fit_point_count": int(fit["finite_point_count"]),
            "speed_m_per_s": fit["speed_m_per_s"],
            "speed_over_rayleigh": (None if fit["speed_m_per_s"] is None
                                    else fit["speed_m_per_s"] / wave_speeds["c_r"]),
            "r_squared": fit["r_squared"],
            "local_speed_by_bin": local_speeds(y, arrival),
        }
    stats_path = stats_dir / f"{output_path.stem}.json"
    stats_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["outputs"]["json"] = str(stats_path)
    return summary


def main() -> int:
    args = parse_args()
    summary = plot_rupture_arrival_zoom(args.input, args.output, args.stats_dir,
                                        t_min=args.t_min, t_max=args.t_max, dpi=args.dpi)
    print(json.dumps({k: summary[k] for k in ("window_ms", "shear_loading_stop_first_saved_ms", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
