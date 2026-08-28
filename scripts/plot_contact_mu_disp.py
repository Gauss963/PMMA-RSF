from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tatva.pmma.plotting import (
    GREY,
    JOURNAL_COLORS,
    configure_journal_style,
    style_axis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot effective friction coefficient versus cumulative slip for one contact point."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection",
        choices=["max-final-slip", "midpoint"],
        default="max-final-slip",
    )
    parser.add_argument(
        "--y-points",
        type=float,
        nargs="*",
        default=None,
        help="Specific contact-line y positions [mm] to plot together.",
    )
    return parser.parse_args()


def plot_contact_mu_disp(
    input_path: Path,
    output_path: Path,
    *,
    selection: str = "max-final-slip",
    y_points: list[float] | None = None,
) -> dict[str, float | int | str]:
    configure_journal_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def local_mu_eff(
        mu_s_local: float,
        slip: np.ndarray,
        mu_k_local: float,
        d_c: float,
    ) -> np.ndarray:
        return np.maximum(
            mu_k_local,
            mu_s_local
            - (mu_s_local - mu_k_local) * np.minimum(slip / d_c, 1.0),
        )

    with h5py.File(input_path, "r") as h5:
        history = np.asarray(h5["history"], dtype=np.float64)
        phase_id = np.asarray(h5["phase_id"], dtype=np.int32)
        step_id = np.asarray(h5["step_id"], dtype=np.int64) if "step_id" in h5 else None
        dt = float(h5.attrs["dt"]) if "dt" in h5.attrs else None
        pressure_steps = int(h5.attrs["pressure_steps"]) if "pressure_steps" in h5.attrs else 0
        cumulative_slip = np.asarray(h5["interface/cumulative_slip"], dtype=np.float64)
        y_coords = np.asarray(h5["interface/contact_line_y"], dtype=np.float64)
        friction_law = str(h5.attrs.get("friction_law", "slip-weakening"))
        saved_mu_eff = (
            np.asarray(h5["interface/friction_coefficient"], dtype=np.float64)
            if "friction_coefficient" in h5["interface"]
            else None
        )
        mu_s = float(h5["interface"].attrs["mu_static"])
        mu_k = float(h5["interface"].attrs["mu_kinetic"])
        d_c = float(h5["interface"].attrs["critical_slip"])
        if "mu_static_profile" in h5["interface"]:
            mu_s_profile = np.asarray(h5["interface/mu_static_profile"], dtype=np.float64)
        else:
            mu_s_profile = np.full_like(y_coords, mu_s, dtype=np.float64)
        if "mu_kinetic_profile" in h5["interface"]:
            mu_k_profile = np.asarray(h5["interface/mu_kinetic_profile"], dtype=np.float64)
        else:
            mu_k_profile = np.full_like(y_coords, mu_k, dtype=np.float64)

    if step_id is not None and dt is not None:
        absolute_steps = step_id + np.where(phase_id == 2, pressure_steps, 0)
        time_ms = absolute_steps.astype(np.float64) * dt * 1e3
    else:
        time_ms = history[:, 0] * 1e3
    n_frames = int(history.shape[0])
    normal_end_idx = int(np.where(phase_id == 1)[0][-1])

    def point_mu_eff(point_idx: int) -> np.ndarray:
        if saved_mu_eff is not None:
            return saved_mu_eff[:, point_idx]
        return local_mu_eff(
            float(mu_s_profile[point_idx]),
            cumulative_slip[:, point_idx],
            float(mu_k_profile[point_idx]),
            d_c,
        )

    if y_points:
        point_indices = [int(np.argmin(np.abs(y_coords - y_val))) for y_val in y_points]
        colors = JOURNAL_COLORS
        ncols = min(2, len(point_indices))
        nrows = int(np.ceil(len(point_indices) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7.2, 2.45 * nrows),
            dpi=180,
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
        flat_axes = list(axes_arr.ravel())
        for i, point_idx in enumerate(point_indices):
            ax = flat_axes[i]
            slip = cumulative_slip[:, point_idx]
            mu_s_local = float(mu_s_profile[point_idx])
            mu_k_local = float(mu_k_profile[point_idx])
            mu_eff = point_mu_eff(point_idx)
            color = colors[i % len(colors)]
            ax.plot(
                slip,
                mu_eff,
                lw=1.25,
                color=color,
            )
            ax.scatter(
                slip[normal_end_idx],
                mu_eff[normal_end_idx],
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                s=19,
                zorder=3,
            )
            ax.scatter(
                slip[-1],
                mu_eff[-1],
                marker="s",
                color=color,
                linewidth=0.0,
                s=18,
                zorder=3,
            )
            if friction_law == "slip-weakening":
                ax.axhline(mu_s_local, color=GREY, ls="--", lw=0.8)
                ax.axhline(mu_k_local, color=GREY, ls=":", lw=0.8)
                ax.axvline(d_c, color=GREY, ls="-.", lw=0.8)
            ax.set_title(f"$y = {y_coords[point_idx]:.0f}$ mm", loc="left")
            ax.text(
                0.98,
                0.95,
                rf"$\mu_{{\rm final}} = {mu_eff[-1]:.3f}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=color,
                fontsize=7.5,
            )
            ax.margins(x=0.04, y=0.06)
            style_axis(ax)
        for ax in flat_axes[len(point_indices) :]:
            ax.set_visible(False)
        for row_axes in axes_arr:
            row_axes[0].set_ylabel(r"Friction coefficient, $\mu$")
        for ax in axes_arr[-1]:
            if ax.get_visible():
                ax.set_xlabel(r"Cumulative slip, $\delta$ [mm]")
        plot_kind = "multi-point"
        point_idx = point_indices[0]
        mu_s_local = float(mu_s_profile[point_idx])
        mu_k_local = float(mu_k_profile[point_idx])
        final_slip = float(cumulative_slip[-1, point_idx])
        selected_mu_eff = point_mu_eff(point_idx)
        final_mu = float(selected_mu_eff[-1])
        normal_end_slip = float(cumulative_slip[normal_end_idx, point_idx])
        normal_end_mu = float(selected_mu_eff[normal_end_idx])
        legend_handles = [
            plt.Line2D(
                [],
                [],
                markeredgecolor=GREY,
                markerfacecolor="white",
                marker="o",
                linestyle="None",
                label="End of normal loading",
            ),
            plt.Line2D(
                [],
                [],
                color=GREY,
                marker="s",
                linestyle="None",
                label="Final state",
            ),
        ]
        if friction_law == "slip-weakening":
            legend_handles[:0] = [
                plt.Line2D([], [], color=GREY, ls="--", lw=0.8, label=r"Local $\mu_s$"),
                plt.Line2D([], [], color=GREY, ls=":", lw=0.8, label=r"Local $\mu_k$"),
                plt.Line2D([], [], color=GREY, ls="-.", lw=0.8, label=rf"$D_c = {d_c:.3f}$ mm"),
            ]
        fig.legend(
            handles=legend_handles,
            loc="outside upper center",
            ncol=min(3, len(legend_handles)),
        )
    else:
        if selection == "midpoint":
            point_idx = int(
                np.argmin(np.abs(y_coords - 0.5 * (y_coords.min() + y_coords.max())))
            )
        else:
            point_idx = int(np.argmax(cumulative_slip[-1]))
        slip = cumulative_slip[:, point_idx]
        mu_s_local = float(mu_s_profile[point_idx])
        mu_k_local = float(mu_k_profile[point_idx])
        mu_eff = point_mu_eff(point_idx)
        fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=180)
        ax.plot(
            slip,
            mu_eff,
            lw=1.25,
            color=JOURNAL_COLORS[0],
            label=friction_law,
        )
        ax.scatter(
            [slip[0], slip[normal_end_idx], slip[-1]],
            [mu_eff[0], mu_eff[normal_end_idx], mu_eff[-1]],
            color=[GREY, JOURNAL_COLORS[1], JOURNAL_COLORS[3]],
            s=20,
            zorder=3,
        )
        ax.annotate("start", (slip[0], mu_eff[0]), textcoords="offset points", xytext=(6, 6))
        ax.annotate(
            f"normal end ({time_ms[normal_end_idx]:.3f} ms)",
            (slip[normal_end_idx], mu_eff[normal_end_idx]),
            textcoords="offset points",
            xytext=(6, -14),
        )
        ax.annotate(
            f"final ({time_ms[-1]:.3f} ms)",
            (slip[-1], mu_eff[-1]),
            textcoords="offset points",
            xytext=(6, 6),
        )
        plot_kind = selection
        final_slip = float(slip[-1])
        final_mu = float(mu_eff[-1])
        normal_end_slip = float(slip[normal_end_idx])
        normal_end_mu = float(mu_eff[normal_end_idx])

    if not y_points:
        if friction_law == "slip-weakening":
            ax.axhline(mu_s_local, color="#999999", ls="--", lw=1.0, label=f"local mu_s = {mu_s_local:.3f}")
            ax.axhline(
                mu_k_local,
                color="#999999",
                ls=":",
                lw=1.0,
                label=f"local mu_k = {mu_k_local:.3f}",
            )
            ax.axvline(d_c, color="#999999", ls="-.", lw=1.0, label=f"d_c = {d_c:.3f}")
        ax.set_xlabel(r"Cumulative slip, $\delta$ [mm]")
        ax.set_ylabel(r"Friction coefficient, $\mu$")
        ax.margins(x=0.04, y=0.06)
        ax.set_title(f"$y = {y_coords[point_idx]:.1f}$ mm", loc="left")
        style_axis(ax)
        ax.legend()
        fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    return {
        "output": str(output_path),
        "plot_kind": plot_kind,
        "point_index": point_idx,
        "y_mm": float(y_coords[point_idx]),
        "final_slip": final_slip,
        "final_mu": final_mu,
        "normal_end_slip": normal_end_slip,
        "normal_end_mu": normal_end_mu,
    }


def main() -> int:
    args = parse_args()
    print(
        plot_contact_mu_disp(
            args.input,
            args.output,
            selection=args.selection,
            y_points=args.y_points,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
