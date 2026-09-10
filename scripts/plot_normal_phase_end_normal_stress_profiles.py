from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tatva.pmma.plotting import BLUE, GREY, ORANGE, configure_journal_style, style_axis


DEFAULT_DISTANCE_FROM_FAULT_MM = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot moving-block normal stress along the loading boundary and "
            "at a fixed distance from the fault at the end of normal loading."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument(
        "--distance-from-fault",
        type=float,
        default=DEFAULT_DISTANCE_FROM_FAULT_MM,
    )
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def last_normal_frame(phase_id: np.ndarray) -> int:
    indices = np.flatnonzero(np.asarray(phase_id) == 1)
    if indices.size == 0:
        raise ValueError("The dump contains no normal-phase bulk frame.")
    return int(indices[-1])


def _frame_time_ms(h5: h5py.File, frame_index: int) -> float:
    if "step_id" in h5 and "dt" in h5.attrs:
        return float(h5["step_id"][frame_index]) * float(h5.attrs["dt"]) * 1.0e3
    return float(h5["history"][frame_index, 0]) * 1.0e3


def profile_at_nearest_element_columns(
    element_centers: np.ndarray,
    values: np.ndarray,
    target_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average tied nearest element columns and return values ordered along y."""
    centers = np.asarray(element_centers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] < 2:
        raise ValueError("element_centers must have shape (elements, dimension>=2).")
    if values.shape != (centers.shape[0],):
        raise ValueError("values must contain one scalar per element.")

    x_columns = np.unique(centers[:, 0])
    minimum_distance = float(np.min(np.abs(x_columns - target_x)))
    selected_x = x_columns[
        np.isclose(np.abs(x_columns - target_x), minimum_distance, atol=1.0e-9)
    ]
    selected = np.any(
        np.isclose(centers[:, 0, None], selected_x[None, :], atol=1.0e-9),
        axis=1,
    )
    y_selected = centers[selected, 1]
    values_selected = values[selected]
    y_values, inverse = np.unique(y_selected, return_inverse=True)
    sums = np.bincount(inverse, weights=values_selected)
    counts = np.bincount(inverse)
    return y_values, sums / counts, selected_x


def output_paths(output_path: Path, stats_dir: Path) -> dict[str, Path]:
    stem = output_path.stem
    return {
        "pdf": output_path,
        "csv": stats_dir / f"{stem}.csv",
        "json": stats_dir / f"{stem}.json",
    }


def plot_normal_phase_end_normal_stress_profiles(
    input_path: Path,
    output_path: Path,
    stats_dir: Path,
    *,
    distance_from_fault_mm: float = DEFAULT_DISTANCE_FROM_FAULT_MM,
    dpi: int = 260,
) -> dict[str, Any]:
    if distance_from_fault_mm <= 0.0:
        raise ValueError("distance_from_fault_mm must be positive.")

    paths = output_paths(output_path, stats_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        frame_index = last_normal_frame(np.asarray(h5["phase_id"], dtype=np.int8))
        time_ms = _frame_time_ms(h5, frame_index)
        coords = np.asarray(h5["moving/coords"], dtype=np.float64)
        elements = np.asarray(h5["moving/elements"], dtype=np.int64)
        sigma_xx = np.asarray(
            h5["moving/stress"][frame_index, :, 0, 0], dtype=np.float64
        )
        reference_stress = float(h5.attrs.get("normal_stress", np.nan))
        loading_fraction = float(
            h5.attrs.get("normal_displacement_loading_fraction", 1.0)
        )
        leading_fraction = float(
            h5.attrs.get("normal_displacement_leading_fraction", 1.0)
        )

    element_centers = coords[elements].mean(axis=1)
    fault_x = float(np.max(coords[:, 0]))
    boundary_target_x = float(np.min(coords[:, 0]))
    near_fault_target_x = fault_x - distance_from_fault_mm
    compression_positive = -sigma_xx

    boundary_y, boundary_stress, boundary_x = profile_at_nearest_element_columns(
        element_centers,
        compression_positive,
        boundary_target_x,
    )
    near_y, near_stress, near_x = profile_at_nearest_element_columns(
        element_centers,
        compression_positive,
        near_fault_target_x,
    )
    if boundary_y.shape != near_y.shape or not np.allclose(boundary_y, near_y):
        raise ValueError("Boundary and near-fault profiles do not share the same y grid.")

    configure_journal_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.05),
        dpi=180,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    profiles = (
        (
            axes[0],
            boundary_stress,
            BLUE,
            "Normal-loading boundary",
            f"element center $x={boundary_x.mean():.2f}$ mm",
        ),
        (
            axes[1],
            near_stress,
            ORANGE,
            f"Moving side, {distance_from_fault_mm:g} mm from fault",
            "element centers $x="
            + ", ".join(f"{value:.2f}" for value in near_x)
            + "$ mm",
        ),
    )
    for label, (axis, stress, color, title, location) in zip(
        ("a", "b"), profiles, strict=True
    ):
        axis.plot(boundary_y, stress, color=color, lw=1.05)
        if np.isfinite(reference_stress):
            axis.axhline(
                reference_stress,
                color=GREY,
                ls="--",
                lw=0.8,
                label=rf"Reference ${reference_stress:g}$ MPa",
            )
        axis.set_title(f"({label}) {title}\n{location}", loc="left")
        axis.set_xlabel(r"Position along fault, $y$ [mm]")
        axis.margins(x=0.01)
        style_axis(axis)
    axes[0].set_ylabel(r"Compressive normal stress, $-\sigma_{xx}$ [MPa]")
    if np.isfinite(reference_stress):
        axes[1].legend(loc="best")
    fig.suptitle(
        (
            f"End of normal loading, $t={time_ms:.3f}$ ms; "
            f"displacement multipliers {loading_fraction:.3f} to {leading_fraction:.3f}"
        ),
        fontsize=9.0,
    )
    fig.savefig(paths["pdf"], dpi=dpi)
    plt.close(fig)

    with paths["csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "y_mm",
                "loading_boundary_compressive_normal_stress_mpa",
                f"moving_{distance_from_fault_mm:g}mm_from_fault_compressive_normal_stress_mpa",
            ]
        )
        writer.writerows(zip(boundary_y, boundary_stress, near_stress, strict=True))

    metadata: dict[str, Any] = {
        "input": str(input_path),
        "frame_index": frame_index,
        "time_ms": time_ms,
        "stress_component": "-sigma_xx",
        "stress_sign_convention": "positive_in_compression",
        "normal_displacement_loading_fraction": loading_fraction,
        "normal_displacement_leading_fraction": leading_fraction,
        "fault_x_mm": fault_x,
        "loading_boundary_target_x_mm": boundary_target_x,
        "loading_boundary_element_center_x_mm": boundary_x.tolist(),
        "distance_from_fault_mm": distance_from_fault_mm,
        "near_fault_target_x_mm": near_fault_target_x,
        "near_fault_element_center_x_mm": near_x.tolist(),
        "profile_point_count": int(boundary_y.size),
        "loading_boundary_mean_mpa": float(np.mean(boundary_stress)),
        "near_fault_mean_mpa": float(np.mean(near_stress)),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["json"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> int:
    args = parse_args()
    plot_normal_phase_end_normal_stress_profiles(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.stats_dir.expanduser().resolve(),
        distance_from_fault_mm=args.distance_from_fault,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
