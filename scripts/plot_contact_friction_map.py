from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tatva.pmma.plotting import configure_journal_style, panel_label, style_axis


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

    normal_idx = np.where(phase_id == 1)[0]
    normal_end_idx = int(normal_idx[-1]) if normal_idx.size else None

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180, layout="constrained")
    im = ax.pcolormesh(
        y_edges,
        time_edges,
        mu_eff,
        cmap="viridis",
        vmin=mu_plot_min,
        vmax=mu_plot_max,
        shading="auto",
        rasterized=True,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
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
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
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
        cmap="viridis",
        vmin=mu_plot_min,
        vmax=mu_plot_max,
        shading="auto",
        rasterized=True,
    )
    ax_shear.set_ylabel("Shear phase time [ms]")
    panel_label(ax_shear, "(a) Shear phase")

    normal_im = ax_normal.pcolormesh(
        y_edges,
        normal_time_edges,
        mu_eff[normal_mask],
        cmap="viridis",
        vmin=mu_plot_min,
        vmax=mu_plot_max,
        shading="auto",
        rasterized=True,
    )
    ax_normal.set_xlabel(r"Position along fault, $y$ [mm]")
    ax_normal.set_ylabel("Normal phase time [ms]")
    panel_label(ax_normal, "(b) Normal loading")
    plt.setp(ax_shear.get_xticklabels(), visible=False)
    style_axis(ax_shear, grid=False)
    style_axis(ax_normal, grid=False)

    cbar = fig.colorbar(normal_im, cax=cax)
    cbar.set_label("Effective friction coefficient")
    fig.savefig(phase_split_output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    return {
        "output": str(output_path),
        "phase_split_output": str(phase_split_output_path),
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
