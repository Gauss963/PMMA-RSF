from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tatva.pmma.plotting import configure_journal_style, panel_label, style_axis
from plot_rupture_speed_and_fault_profile import material_wave_speeds


MU_COLOR_FLOOR = 0.6
_DEFAULT_MATERIAL = {
    "young_modulus": 7662.0,
    "poisson_ratio": 0.2,
    "density": 1.148e-9,
}


def _effective_friction_colormap():
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_under("black")
    return cmap


def _read_wave_speeds(input_path: Path, h5: h5py.File) -> dict[str, float]:
    material = dict(_DEFAULT_MATERIAL)
    resolved_case = input_path.parent.parent / "input" / "resolved_case.json"
    if resolved_case.is_file():
        payload = json.loads(resolved_case.read_text(encoding="utf-8"))
        material.update(payload.get("material", {}))
    for key, legacy_key in (
        ("young_modulus", "E"),
        ("poisson_ratio", "nu"),
        ("density", "rho"),
    ):
        value = h5.attrs.get(key, h5.attrs.get(legacy_key))
        if value is not None:
            material[key] = float(value)
    return material_wave_speeds(
        float(material["young_modulus"]),
        float(material["poisson_ratio"]),
        float(material["density"]),
    )


def _add_wave_speed_guides(
    axis: plt.Axes,
    y_bounds: tuple[float, float],
    time_bounds: tuple[float, float],
    wave_speeds: dict[str, float],
) -> plt.Axes | None:
    """Add a physically scaled travel-time ruler for the wave speeds."""
    y_min, y_max = y_bounds
    time_min, time_max = time_bounds
    y_span = y_max - y_min
    time_span = time_max - time_min
    if y_span <= 0.0 or time_span <= 0.0:
        return None

    # Across the full map these travel times occupy less than 1% of the shear
    # phase. A local ruler keeps the units physical while making the slopes
    # distinguishable instead of presenting them as nearly horizontal swatches.
    ruler = axis.inset_axes([0.065, 0.63, 0.49, 0.30])
    guide_distance = np.linspace(0.0, y_span, 100)
    guide_specs = (
        ("c_s", 1.0, r"$C_S$", "#ffb000", (0, (2, 1.5))),
        ("c_r", 1.0, r"$C_R$", "white", (0, (5, 2))),
        ("c_r", 0.8, r"$0.8C_R$", "#56b4e9", (0, (4, 1.5))),
        ("c_r", 0.5, r"$0.5C_R$", "#e78ac3", (0, (1, 1.5))),
    )
    for key, speed_fraction, label, color, line_style in guide_specs:
        speed = speed_fraction * float(wave_speeds[key])
        travel_time = guide_distance / speed
        ruler.plot(
            guide_distance,
            travel_time,
            color=color,
            lw=1.15,
            ls=line_style,
            solid_capstyle="round",
            label=rf"{label}  {speed / 1e3:.2f} km s$^{{-1}}$",
        )

    slowest_speed = 0.5 * float(wave_speeds["c_r"])
    ruler.set_xlim(0.0, y_span)
    ruler.set_ylim(0.0, 1.08 * y_span / slowest_speed)
    ruler.set_xlabel(r"Travel distance, $\Delta y$ [mm]", fontsize=6.2, labelpad=1.0)
    ruler.set_ylabel(r"Travel time, $\Delta t$ [ms]", fontsize=6.2, labelpad=1.0)
    ruler.set_title("Physical wave-speed slopes", fontsize=6.8, pad=2.0)
    ruler.set_facecolor((0.02, 0.02, 0.02, 0.88))
    ruler.tick_params(colors="white", labelsize=5.8, width=0.6, length=2.2, pad=1.5)
    ruler.xaxis.label.set_color("white")
    ruler.yaxis.label.set_color("white")
    ruler.title.set_color("white")
    for spine in ruler.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.6)
    legend = ruler.legend(
        loc="upper left",
        fontsize=5.7,
        frameon=False,
        handlelength=2.4,
        borderaxespad=0.35,
        labelspacing=0.25,
    )
    for text in legend.get_texts():
        text.set_color("white")
    return ruler


def _save_with_png(fig: plt.Figure, output_path: Path) -> Path:
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    png_path = output_path.with_suffix(".png")
    if png_path != output_path:
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0.04)
    return png_path


def _cell_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    if values.size == 1:
        delta = 0.5
        return np.array([values[0] - delta, values[0] + delta], dtype=np.float64)
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate(([first], mids, [last]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot effective friction coefficient along the contact line over time."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--phase-split-output",
        type=Path,
        default=None,
    )
    parser.add_argument("--mu-s", type=float, default=0.8)
    parser.add_argument("--mu-k", type=float, default=0.6)
    parser.add_argument("--d-c", type=float, default=8.0)
    return parser.parse_args()


def plot_mu_eff_maps(
    input_path: Path,
    output_path: Path,
    phase_split_output_path: Path,
    *,
    mu_s: float = 0.8,
    mu_k: float = 0.6,
    d_c: float = 8.0,
) -> dict[str, float | str]:
    configure_journal_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        master_nodes = np.asarray(h5["interface/master_nodes"], dtype=np.int32)
        y_coords = np.asarray(h5["moving/coords"], dtype=np.float32)[master_nodes, 1]
        history = np.asarray(h5["history"], dtype=np.float32)
        phase_id = np.asarray(h5["phase_id"], dtype=np.int32)
        step_id = np.asarray(h5["step_id"], dtype=np.int64) if "step_id" in h5 else None
        dt = float(h5.attrs["dt"]) if "dt" in h5.attrs else None
        pressure_steps = int(h5.attrs["pressure_steps"]) if "pressure_steps" in h5.attrs else 0
        cumulative_slip = np.asarray(h5["interface/cumulative_slip"], dtype=np.float32)
        friction_law = str(h5.attrs.get("friction_law", "slip-weakening"))
        wave_speeds = _read_wave_speeds(input_path, h5)
        saved_mu_eff = (
            np.asarray(h5["interface/friction_coefficient"], dtype=np.float32)
            if "friction_coefficient" in h5["interface"]
            else None
        )
        if "mu_static_profile" in h5["interface"]:
            mu_s_profile = np.asarray(h5["interface/mu_static_profile"], dtype=np.float32)
        else:
            mu_s_profile = np.full(y_coords.shape, mu_s, dtype=np.float32)
        if "mu_kinetic_profile" in h5["interface"]:
            mu_k_profile = np.asarray(h5["interface/mu_kinetic_profile"], dtype=np.float32)
        else:
            mu_k_profile = np.full(y_coords.shape, mu_k, dtype=np.float32)

    order = np.argsort(y_coords)
    y_sorted = y_coords[order]
    cum_sorted = cumulative_slip[:, order]
    mu_s_sorted = mu_s_profile[order]
    mu_k_sorted = mu_k_profile[order]
    if step_id is not None and dt is not None:
        absolute_steps = step_id + np.where(phase_id == 2, pressure_steps, 0)
        time_ms = absolute_steps.astype(np.float64) * dt * 1e3
    else:
        time_ms = history[:, 0] * 1e3
    y_edges = _cell_edges(y_sorted)
    time_edges = _cell_edges(time_ms)

    if saved_mu_eff is not None:
        mu_eff = saved_mu_eff[:, order]
    else:
        mu_eff = np.maximum(
            mu_k_sorted[None, :],
            mu_s_sorted[None, :]
            - (mu_s_sorted[None, :] - mu_k_sorted[None, :])
            * np.minimum(cum_sorted / d_c, 1.0),
        )
    mu_plot_min = float(np.nanmin(mu_eff))
    mu_plot_max = float(np.nanmax(mu_eff))
    mu_display_max = max(mu_plot_max, MU_COLOR_FLOOR + 1.0e-6)
    friction_cmap = _effective_friction_colormap()

    normal_idx = np.where(phase_id == 1)[0]
    normal_end_idx = int(normal_idx[-1]) if normal_idx.size else None

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180, layout="constrained")
    im = ax.pcolormesh(
        y_edges,
        time_edges,
        mu_eff,
        cmap=friction_cmap,
        vmin=MU_COLOR_FLOOR,
        vmax=mu_display_max,
        shading="auto",
        rasterized=True,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02, extend="min")
    cbar.set_label("Effective friction coefficient")

    if normal_end_idx is not None:
        ax.axhline(time_ms[normal_end_idx], color="white", lw=1.2, ls="--", alpha=0.9)
        ax.text(
            float(y_sorted[0]) + 8.0,
            float(time_ms[normal_end_idx]) + 0.15,
            "normal/shear boundary",
            color="white",
            fontsize=9,
            va="bottom",
        )

    normal_end_min = float(mu_eff[normal_end_idx].min()) if normal_end_idx is not None else float("nan")
    final_min = float(mu_eff[-1].min())
    ax.set_title("Effective friction along the fault", loc="left")
    ax.set_xlabel(r"Position along fault, $y$ [mm]")
    ax.set_ylabel("Time [ms]")
    style_axis(ax, grid=False)
    output_png_path = _save_with_png(fig, output_path)
    plt.close(fig)

    normal_mask = phase_id == 1
    shear_mask = phase_id == 2
    normal_time_ms = (
        time_ms[normal_mask] - float(time_ms[normal_mask][0])
        if np.any(normal_mask)
        else np.zeros(0, dtype=np.float32)
    )
    shear_time_ms = (
        time_ms[shear_mask] - float(time_ms[shear_mask][0])
        if np.any(shear_mask)
        else np.zeros(0, dtype=np.float32)
    )
    normal_time_edges = (
        _cell_edges(normal_time_ms)
        if normal_time_ms.size
        else np.array([0.0, 1.0], dtype=np.float64)
    )
    shear_time_edges = (
        _cell_edges(shear_time_ms)
        if shear_time_ms.size
        else np.array([0.0, 1.0], dtype=np.float64)
    )
    fig = plt.figure(figsize=(7.2, 5.2), dpi=180)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[30.0, 1.0],
        height_ratios=[4.0, 1.0],
        wspace=0.08,
        hspace=0.12,
    )
    ax_shear = fig.add_subplot(gs[0, 0])
    ax_normal = fig.add_subplot(gs[1, 0], sharex=ax_shear)
    cax = fig.add_subplot(gs[:, 1])

    shear_im = ax_shear.pcolormesh(
        y_edges,
        shear_time_edges,
        mu_eff[shear_mask],
        cmap=friction_cmap,
        vmin=MU_COLOR_FLOOR,
        vmax=mu_display_max,
        shading="auto",
        rasterized=True,
    )
    ax_shear.set_ylabel("Shear phase time [ms]")
    panel_label(ax_shear, "(a) Shear phase")

    normal_im = ax_normal.pcolormesh(
        y_edges,
        normal_time_edges,
        mu_eff[normal_mask],
        cmap=friction_cmap,
        vmin=MU_COLOR_FLOOR,
        vmax=mu_display_max,
        shading="auto",
        rasterized=True,
    )
    ax_normal.set_xlabel(r"Position along fault, $y$ [mm]")
    ax_normal.set_ylabel("Normal phase time [ms]")
    panel_label(ax_normal, "(b) Normal loading")
    plt.setp(ax_shear.get_xticklabels(), visible=False)
    style_axis(ax_shear, grid=False)
    style_axis(ax_normal, grid=False)
    _add_wave_speed_guides(
        ax_shear,
        (float(y_edges[0]), float(y_edges[-1])),
        (float(shear_time_edges[0]), float(shear_time_edges[-1])),
        wave_speeds,
    )

    cbar = fig.colorbar(normal_im, cax=cax, extend="min")
    cbar.set_label("Effective friction coefficient")
    phase_split_png_path = _save_with_png(fig, phase_split_output_path)
    plt.close(fig)

    return {
        "output": str(output_path),
        "output_png": str(output_png_path),
        "phase_split_output": str(phase_split_output_path),
        "phase_split_output_png": str(phase_split_png_path),
        "mu_color_floor": MU_COLOR_FLOOR,
        "rayleigh_wave_speed_m_per_s": wave_speeds["c_r"],
        "rayleigh_80_percent_speed_m_per_s": 0.8 * wave_speeds["c_r"],
        "rayleigh_50_percent_speed_m_per_s": 0.5 * wave_speeds["c_r"],
        "shear_wave_speed_m_per_s": wave_speeds["c_s"],
        "mu_min_normal_end": normal_end_min,
        "mu_min_final": final_min,
        "mu_mean_normal_end": float(mu_eff[normal_end_idx].mean()) if normal_end_idx is not None else float("nan"),
        "mu_mean_final": float(mu_eff[-1].mean()),
        "cum_slip_max_normal_end": float(cum_sorted[normal_end_idx].max()) if normal_end_idx is not None else float("nan"),
        "cum_slip_max_final": float(cum_sorted[-1].max()),
    }


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    plot_dir = input_path.parent.parent / "Plot"
    result = plot_mu_eff_maps(
        input_path,
        args.output or plot_dir / "mu_eff_map.pdf",
        args.phase_split_output or plot_dir / "mu_eff_map_phase_split.pdf",
        mu_s=args.mu_s,
        mu_k=args.mu_k,
        d_c=args.d_c,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
