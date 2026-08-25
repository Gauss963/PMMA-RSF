from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_CTX: dict[str, object] = {}

SINGLE_STRESS_MODES = ("von_mises", "sigma_xy", "sigma_yy")
COMBINED_STRESS_MODE = "stress_triptych"
PANEL_STRESS_MODES = ("sigma_yy", "von_mises", "sigma_xy")
RANGE_CACHE_VERSION = 1
RANGE_HISTOGRAM_BINS = 65_536
RANGE_CHECKPOINT_FRAMES = 1024
WORKER_FRAME_LIMIT = 250
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB\x60\x82"


def _as_triangles(elements: np.ndarray, scalar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split quad connectivity into triangles for tripcolor(shading="flat").

    Triangle meshes pass through unchanged. Quads are split along the (0, 2)
    diagonal into two triangles that each inherit the quad's scalar value.
    """
    if elements.shape[1] == 3:
        return elements, scalar
    if elements.shape[1] == 4:
        triangles = np.concatenate(
            [elements[:, [0, 1, 2]], elements[:, [0, 2, 3]]], axis=0
        )
        return triangles, np.concatenate([scalar, scalar], axis=0)
    raise ValueError(f"Unsupported element connectivity with {elements.shape[1]} nodes")


def _plot_parent_indices(group: h5py.Group) -> np.ndarray | None:
    if "plot_parent_elements" not in group:
        return None
    parent = np.asarray(group["plot_parent_elements"], dtype=np.int32)
    source_elements = int(group["stress"].shape[1])
    if parent.size == source_elements and np.array_equal(
        parent,
        np.arange(source_elements, dtype=np.int32),
    ):
        return None
    return parent


def _frame_path(output_dir: Path, frame_idx: int) -> Path:
    return output_dir / f"stress_{frame_idx:07d}.png"


def _frame_is_complete(path: Path) -> bool:
    """Check the PNG framing without decoding a multi-megabyte image."""
    try:
        if path.stat().st_size < len(PNG_SIGNATURE) + len(PNG_IEND):
            return False
        with path.open("rb") as stream:
            if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                return False
            stream.seek(-len(PNG_IEND), os.SEEK_END)
            return stream.read(len(PNG_IEND)) == PNG_IEND
    except OSError:
        return False


def von_mises_2d(stress: np.ndarray) -> np.ndarray:
    sxx = stress[..., 0, 0]
    syy = stress[..., 1, 1]
    sxy = stress[..., 0, 1]
    return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))


def von_mises_3d(stress: np.ndarray) -> np.ndarray:
    sxx = stress[..., 0, 0]
    syy = stress[..., 1, 1]
    szz = stress[..., 2, 2]
    sxy = stress[..., 0, 1]
    syz = stress[..., 1, 2]
    szx = stress[..., 2, 0]
    return np.sqrt(
        np.maximum(
            0.5
            * (
                (sxx - syy) ** 2
                + (syy - szz) ** 2
                + (szz - sxx) ** 2
                + 6.0 * (sxy * sxy + syz * syz + szx * szx)
            ),
            0.0,
        )
    )


def stress_scalar(stress: np.ndarray, mode: str) -> np.ndarray:
    dim = int(stress.shape[-1])
    if mode == "von_mises":
        return von_mises_2d(stress) if dim == 2 else von_mises_3d(stress)
    if mode == "sigma_xy":
        return stress[..., 0, 1]
    if mode == "sigma_yy":
        return stress[..., 1, 1]
    raise ValueError(f"Unsupported stress mode: {mode}")


def render_modes(stress_mode: str) -> tuple[str, ...]:
    if stress_mode == COMBINED_STRESS_MODE:
        return PANEL_STRESS_MODES
    if stress_mode in SINGLE_STRESS_MODES:
        return (stress_mode,)
    raise ValueError(f"Unsupported stress mode: {stress_mode}")


def stress_style(mode: str, stress_max: float, stress_percentile: float) -> dict[str, object]:
    if mode == "sigma_xy":
        return {
            "cmap": "RdBu_r",
            "vmin": -stress_max,
            "vmax": stress_max,
            "colorbar_label": "sigma_xy shear stress",
            "title": "sigma_xy",
            "clip_label": f"abs(vmax)=p{stress_percentile:.1f}={stress_max:.2f}",
        }
    if mode == "sigma_yy":
        return {
            "cmap": "RdBu_r",
            "vmin": -stress_max,
            "vmax": stress_max,
            "colorbar_label": "sigma_yy normal stress",
            "title": "sigma_yy",
            "clip_label": f"abs(vmax)=p{stress_percentile:.1f}={stress_max:.2f}",
        }
    if mode == "von_mises":
        return {
            "cmap": "magma",
            "vmin": 0.0,
            "vmax": stress_max,
            "colorbar_label": "von Mises stress",
            "title": "von Mises",
            "clip_label": f"vmax=p{stress_percentile:.1f}={stress_max:.2f}",
        }
    raise ValueError(f"Unsupported stress mode: {mode}")


def _percentile_from_histogram(
    histogram: np.ndarray,
    edges: np.ndarray,
    percentile: float,
) -> float:
    total = int(histogram.sum())
    if total <= 0:
        return 0.0
    rank = np.clip(percentile, 0.0, 100.0) / 100.0 * (total - 1)
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    bin_idx = min(
        int(np.searchsorted(cumulative, rank, side="right")),
        histogram.size - 1,
    )
    previous = int(cumulative[bin_idx - 1]) if bin_idx > 0 else 0
    count = max(1, int(histogram[bin_idx]))
    fraction = min(max((rank - previous) / count, 0.0), 1.0)
    return float(edges[bin_idx] + fraction * (edges[bin_idx + 1] - edges[bin_idx]))


def _save_range_cache(
    cache_path: Path | None,
    *,
    source_size: int,
    source_mtime_ns: int,
    modes: tuple[str, ...],
    stress_percentile: float,
    histogram_bins: int,
    phase: str,
    next_frame: int,
    stress_abs_max: np.ndarray,
    disp_max: float,
    histograms: np.ndarray,
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.tmp")
    with temporary_path.open("wb") as stream:
        np.savez(
            stream,
            version=np.int64(RANGE_CACHE_VERSION),
            source_size=np.int64(source_size),
            source_mtime_ns=np.int64(source_mtime_ns),
            modes=np.asarray(modes, dtype="U16"),
            stress_percentile=np.float64(stress_percentile),
            histogram_bins=np.int64(histogram_bins),
            phase=np.asarray(phase, dtype="U16"),
            next_frame=np.int64(next_frame),
            stress_abs_max=np.asarray(stress_abs_max, dtype=np.float64),
            disp_max=np.float64(disp_max),
            histograms=np.asarray(histograms, dtype=np.uint64),
        )
    os.replace(temporary_path, cache_path)


def _load_range_cache(
    cache_path: Path | None,
    *,
    source_size: int,
    source_mtime_ns: int,
    modes: tuple[str, ...],
    stress_percentile: float,
    histogram_bins: int,
) -> dict[str, object] | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            valid = (
                int(cache["version"]) == RANGE_CACHE_VERSION
                and int(cache["source_size"]) == source_size
                and int(cache["source_mtime_ns"]) == source_mtime_ns
                and tuple(str(value) for value in cache["modes"]) == modes
                and float(cache["stress_percentile"]) == stress_percentile
                and int(cache["histogram_bins"]) == histogram_bins
            )
            if not valid:
                return None
            return {
                "phase": str(cache["phase"]),
                "next_frame": int(cache["next_frame"]),
                "stress_abs_max": np.asarray(
                    cache["stress_abs_max"],
                    dtype=np.float64,
                ),
                "disp_max": float(cache["disp_max"]),
                "histograms": np.asarray(cache["histograms"], dtype=np.uint64),
            }
    except (KeyError, OSError, ValueError):
        return None


def compute_global_ranges(
    h5_path: Path,
    batch_size: int = 8,
    *,
    stress_mode: str,
    stress_percentile: float = 99.5,
    cache_path: Path | None = None,
    histogram_bins: int = RANGE_HISTOGRAM_BINS,
    checkpoint_frames: int = RANGE_CHECKPOINT_FRAMES,
) -> tuple[dict[str, dict[str, float]], float]:
    modes = render_modes(stress_mode)
    source_stat = h5_path.stat()
    source_size = int(source_stat.st_size)
    source_mtime_ns = int(source_stat.st_mtime_ns)
    stress_abs_max = np.zeros(len(modes), dtype=np.float64)
    histograms = np.zeros((len(modes), histogram_bins), dtype=np.uint64)
    disp_max = 0.0
    phase = "max"
    next_frame = 0
    cached = _load_range_cache(
        cache_path,
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
        modes=modes,
        stress_percentile=stress_percentile,
        histogram_bins=histogram_bins,
    )
    if cached is not None:
        phase = str(cached["phase"])
        next_frame = int(cached["next_frame"])
        stress_abs_max = np.asarray(cached["stress_abs_max"], dtype=np.float64)
        disp_max = float(cached["disp_max"])
        histograms = np.asarray(cached["histograms"], dtype=np.uint64)

    with h5py.File(h5_path, "r") as h5:
        n_frames = int(h5["history"].shape[0])
        moving_parent = _plot_parent_indices(h5["moving"])
        stationary_parent = _plot_parent_indices(h5["stationary"])
        if phase == "max":
            print(
                f"[render] full range pass 1/2: frames {next_frame}/{n_frames}",
                flush=True,
            )
            for start in range(next_frame, n_frames, batch_size):
                stop = min(start + batch_size, n_frames)
                moving_stress = np.asarray(
                    h5["moving/stress"][start:stop],
                    dtype=np.float32,
                )
                stationary_stress = np.asarray(
                    h5["stationary/stress"][start:stop],
                    dtype=np.float32,
                )
                if moving_parent is not None:
                    moving_stress = moving_stress[:, moving_parent]
                if stationary_parent is not None:
                    stationary_stress = stationary_stress[:, stationary_parent]
                moving_disp = np.asarray(
                    h5["moving/displacement"][start:stop],
                    dtype=np.float32,
                )
                stationary_disp = np.asarray(
                    h5["stationary/displacement"][start:stop],
                    dtype=np.float32,
                )
                disp_max = max(
                    disp_max,
                    float(np.linalg.norm(moving_disp, axis=-1).max(initial=0.0)),
                    float(np.linalg.norm(stationary_disp, axis=-1).max(initial=0.0)),
                )
                for mode_idx, mode in enumerate(modes):
                    moving_scalar = stress_scalar(moving_stress, mode)
                    stationary_scalar = stress_scalar(stationary_stress, mode)
                    stress_abs_max[mode_idx] = max(
                        stress_abs_max[mode_idx],
                        float(np.abs(moving_scalar).max(initial=0.0)),
                        float(np.abs(stationary_scalar).max(initial=0.0)),
                    )
                if stop == n_frames or stop // checkpoint_frames != start // checkpoint_frames:
                    print(f"[render] range pass 1/2: {stop}/{n_frames}", flush=True)
                    _save_range_cache(
                        cache_path,
                        source_size=source_size,
                        source_mtime_ns=source_mtime_ns,
                        modes=modes,
                        stress_percentile=stress_percentile,
                        histogram_bins=histogram_bins,
                        phase="max",
                        next_frame=stop,
                        stress_abs_max=stress_abs_max,
                        disp_max=disp_max,
                        histograms=histograms,
                    )
            phase = "histogram"
            next_frame = 0
            _save_range_cache(
                cache_path,
                source_size=source_size,
                source_mtime_ns=source_mtime_ns,
                modes=modes,
                stress_percentile=stress_percentile,
                histogram_bins=histogram_bins,
                phase=phase,
                next_frame=next_frame,
                stress_abs_max=stress_abs_max,
                disp_max=disp_max,
                histograms=histograms,
            )

        edges_by_mode = [
            np.linspace(
                0.0,
                max(float(maximum), 1e-6),
                num=histogram_bins + 1,
                dtype=np.float64,
            )
            for maximum in stress_abs_max
        ]
        if phase == "histogram":
            print(
                f"[render] full range pass 2/2: frames {next_frame}/{n_frames}",
                flush=True,
            )
            for start in range(next_frame, n_frames, batch_size):
                stop = min(start + batch_size, n_frames)
                moving_stress = np.asarray(
                    h5["moving/stress"][start:stop],
                    dtype=np.float32,
                )
                stationary_stress = np.asarray(
                    h5["stationary/stress"][start:stop],
                    dtype=np.float32,
                )
                if moving_parent is not None:
                    moving_stress = moving_stress[:, moving_parent]
                if stationary_parent is not None:
                    stationary_stress = stationary_stress[:, stationary_parent]
                for mode_idx, mode in enumerate(modes):
                    moving_scalar = stress_scalar(moving_stress, mode)
                    stationary_scalar = stress_scalar(stationary_stress, mode)
                    if mode != "von_mises":
                        moving_scalar = np.abs(moving_scalar)
                        stationary_scalar = np.abs(stationary_scalar)
                    histograms[mode_idx] += np.histogram(
                        moving_scalar,
                        bins=edges_by_mode[mode_idx],
                    )[0].astype(np.uint64)
                    histograms[mode_idx] += np.histogram(
                        stationary_scalar,
                        bins=edges_by_mode[mode_idx],
                    )[0].astype(np.uint64)
                if stop == n_frames or stop // checkpoint_frames != start // checkpoint_frames:
                    print(f"[render] range pass 2/2: {stop}/{n_frames}", flush=True)
                    _save_range_cache(
                        cache_path,
                        source_size=source_size,
                        source_mtime_ns=source_mtime_ns,
                        modes=modes,
                        stress_percentile=stress_percentile,
                        histogram_bins=histogram_bins,
                        phase="histogram",
                        next_frame=stop,
                        stress_abs_max=stress_abs_max,
                        disp_max=disp_max,
                        histograms=histograms,
                    )
            phase = "complete"
            next_frame = n_frames
            _save_range_cache(
                cache_path,
                source_size=source_size,
                source_mtime_ns=source_mtime_ns,
                modes=modes,
                stress_percentile=stress_percentile,
                histogram_bins=histogram_bins,
                phase=phase,
                next_frame=next_frame,
                stress_abs_max=stress_abs_max,
                disp_max=disp_max,
                histograms=histograms,
            )

    mode_stats: dict[str, dict[str, float]] = {}
    for mode_idx, mode in enumerate(modes):
        stress_plot_max = _percentile_from_histogram(
            histograms[mode_idx],
            edges_by_mode[mode_idx],
            stress_percentile,
        )
        mode_stats[mode] = {
            "stress_abs_max": float(stress_abs_max[mode_idx]),
            "stress_plot_max": max(stress_plot_max, 1e-6),
        }
    return mode_stats, disp_max


def _init_worker(
    h5_path: str,
    output_dir: str,
    deform_scale: float,
    stress_ranges: dict[str, dict[str, float]],
    stress_percentile: float,
    dpi: int,
    figsize: tuple[float, float],
    swap_axes: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    stress_mode: str,
) -> None:
    _CTX["h5_path"] = h5_path
    _CTX["output_dir"] = output_dir
    _CTX["deform_scale"] = deform_scale
    _CTX["stress_ranges"] = stress_ranges
    _CTX["stress_percentile"] = stress_percentile
    _CTX["dpi"] = dpi
    _CTX["figsize"] = figsize
    _CTX["swap_axes"] = swap_axes
    _CTX["xlim"] = xlim
    _CTX["ylim"] = ylim
    _CTX["stress_mode"] = stress_mode
    h5 = h5py.File(h5_path, "r")
    _CTX["file"] = h5
    _CTX["moving_coords"] = np.asarray(h5["moving/coords"], dtype=np.float32)
    _CTX["moving_elements"] = np.asarray(
        h5["moving/plot_elements"]
        if "plot_elements" in h5["moving"]
        else h5["moving/elements"],
        dtype=np.int32,
    )
    _CTX["stationary_coords"] = np.asarray(
        h5["stationary/coords"],
        dtype=np.float32,
    )
    _CTX["stationary_elements"] = np.asarray(
        h5["stationary/plot_elements"]
        if "plot_elements" in h5["stationary"]
        else h5["stationary/elements"],
        dtype=np.int32,
    )
    _CTX["moving_parent"] = _plot_parent_indices(h5["moving"])
    _CTX["stationary_parent"] = _plot_parent_indices(h5["stationary"])


def _render_frame(frame_idx: int) -> str:
    h5: h5py.File = _CTX["file"]  # type: ignore[assignment]
    deform_scale = float(_CTX["deform_scale"])
    stress_ranges = dict(_CTX["stress_ranges"])  # type: ignore[arg-type]
    stress_percentile = float(_CTX["stress_percentile"])
    dpi = int(_CTX["dpi"])
    figsize = tuple(_CTX["figsize"])  # type: ignore[arg-type]
    swap_axes = bool(_CTX["swap_axes"])
    xlim = tuple(_CTX["xlim"])  # type: ignore[arg-type]
    ylim = tuple(_CTX["ylim"])  # type: ignore[arg-type]
    stress_mode = str(_CTX["stress_mode"])
    output_dir = Path(str(_CTX["output_dir"]))

    moving_coords = np.asarray(_CTX["moving_coords"], dtype=np.float32)
    moving_elements = np.asarray(_CTX["moving_elements"], dtype=np.int32)
    stationary_coords = np.asarray(_CTX["stationary_coords"], dtype=np.float32)
    stationary_elements = np.asarray(_CTX["stationary_elements"], dtype=np.int32)
    moving_parent = _CTX["moving_parent"]
    stationary_parent = _CTX["stationary_parent"]

    moving_disp = np.asarray(h5["moving/displacement"][frame_idx], dtype=np.float32)
    stationary_disp = np.asarray(
        h5["stationary/displacement"][frame_idx], dtype=np.float32
    )
    moving_stress = np.asarray(h5["moving/stress"][frame_idx], dtype=np.float32)
    stationary_stress = np.asarray(h5["stationary/stress"][frame_idx], dtype=np.float32)
    hist = np.asarray(h5["history"][frame_idx], dtype=np.float32)

    if moving_parent is not None:
        moving_stress = moving_stress[moving_parent]
    if stationary_parent is not None:
        stationary_stress = stationary_stress[stationary_parent]

    moving_xy = moving_coords + deform_scale * moving_disp
    stationary_xy = stationary_coords + deform_scale * stationary_disp
    if moving_xy.shape[1] == 3:
        moving_xy = moving_xy[:, :2]
        stationary_xy = stationary_xy[:, :2]
    if swap_axes:
        moving_xy = moving_xy[:, [1, 0]]
        stationary_xy = stationary_xy[:, [1, 0]]

    panel_modes = render_modes(stress_mode)
    time_ms = float(hist[0]) * 1e3
    applied_shear = float(hist[1])
    avg_tau = float(hist[2])
    max_slip = float(hist[5])
    fig, axes = plt.subplots(
        1,
        len(panel_modes),
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )
    axes_row = list(axes[0])
    for ax, mode in zip(axes_row, panel_modes, strict=True):
        style = stress_style(
            mode,
            float(stress_ranges[mode]["stress_plot_max"]),
            stress_percentile,
        )
        moving_scalar = stress_scalar(moving_stress, mode)
        stationary_scalar = stress_scalar(stationary_stress, mode)
        moving_triangles, moving_scalar = _as_triangles(moving_elements, moving_scalar)
        stationary_triangles, stationary_scalar = _as_triangles(
            stationary_elements, stationary_scalar
        )
        trip1 = ax.tripcolor(
            moving_xy[:, 0],
            moving_xy[:, 1],
            moving_triangles,
            facecolors=moving_scalar,
            shading="flat",
            cmap=str(style["cmap"]),
            vmin=float(style["vmin"]),
            vmax=float(style["vmax"]),
        )
        ax.tripcolor(
            stationary_xy[:, 0],
            stationary_xy[:, 1],
            stationary_triangles,
            facecolors=stationary_scalar,
            shading="flat",
            cmap=str(style["cmap"]),
            vmin=float(style["vmin"]),
            vmax=float(style["vmax"]),
        )
        cbar = fig.colorbar(trip1, ax=ax, pad=0.02)
        cbar.set_label(str(style["colorbar_label"]))
        ax.set_title(f"{style['title']}\n{style['clip_label']}", fontsize=10, pad=8)
        ax.set_aspect("equal")
        if swap_axes:
            ax.set_xlabel("y")
            ax.set_ylabel("x")
        else:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(False)

    fig.suptitle(
        "PMMA-RSF stress panels\n"
        f"frame={frame_idx:04d}  time={time_ms:.3f} ms  "
        f"applied_shear={applied_shear:.3f}  avg_tau={avg_tau:.4e}  "
        f"max_slip={max_slip:.4e}  deform_scale={deform_scale:.3g}",
        fontsize=11,
        y=0.99,
    )

    output_path = _frame_path(output_dir, frame_idx)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    try:
        fig.savefig(temporary_path, format="png")
        os.replace(temporary_path, output_path)
    finally:
        plt.close(fig)
        temporary_path.unlink(missing_ok=True)
    return str(output_path)


def render_all_frames(
    h5_path: Path,
    output_dir: Path,
    *,
    workers: int,
    dpi: int,
    width: float,
    height: float,
    deform_scale: float | None,
    frame_limit: int | None,
    stress_percentile: float,
    swap_axes: bool,
    margin: float,
    stress_mode: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    range_cache_path = output_dir / ".stress_range_cache.npz"
    stress_ranges, disp_max = compute_global_ranges(
        h5_path,
        stress_mode=stress_mode,
        stress_percentile=stress_percentile,
        cache_path=range_cache_path,
    )
    if deform_scale is None:
        deform_scale = 1.0 if disp_max <= 0.0 else min(5000.0, 20.0 / disp_max)

    with h5py.File(h5_path, "r") as h5:
        n_frames = int(h5["history"].shape[0])
        moving_coords = np.asarray(h5["moving/coords"], dtype=np.float32)
        stationary_coords = np.asarray(h5["stationary/coords"], dtype=np.float32)
    if frame_limit is not None:
        n_frames = min(n_frames, max(1, int(frame_limit)))

    frame_indices = [
        frame_idx
        for frame_idx in range(n_frames)
        if not _frame_is_complete(_frame_path(output_dir, frame_idx))
    ]
    skipped_frames = n_frames - len(frame_indices)
    if skipped_frames:
        print(
            f"[render] resume: {skipped_frames}/{n_frames} complete frames retained; "
            f"{len(frame_indices)} remain",
            flush=True,
        )

    coords = np.vstack([moving_coords, stationary_coords])
    if coords.shape[1] == 3:
        coords = coords[:, :2]
    if swap_axes:
        coords = coords[:, [1, 0]]
    x_min = float(coords[:, 0].min() - margin)
    x_max = float(coords[:, 0].max() + margin)
    y_min = float(coords[:, 1].min() - margin)
    y_max = float(coords[:, 1].max() + margin)

    if frame_indices:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=min(workers, len(frame_indices)),
            maxtasksperchild=WORKER_FRAME_LIMIT,
            initializer=_init_worker,
            initargs=(
                str(h5_path),
                str(output_dir),
                float(deform_scale),
                stress_ranges,
                float(stress_percentile),
                int(dpi),
                (float(width), float(height)),
                bool(swap_axes),
                (x_min, x_max),
                (y_min, y_max),
                str(stress_mode),
            ),
        ) as pool:
            for idx, _ in enumerate(
                pool.imap_unordered(_render_frame, frame_indices),
                start=1,
            ):
                if idx % 200 == 0 or idx == len(frame_indices):
                    print(
                        f"[render] {idx}/{len(frame_indices)} remaining frames done "
                        f"({skipped_frames + idx}/{n_frames} total)",
                        flush=True,
                    )
    else:
        print(f"[render] all {n_frames} frames already complete", flush=True)

    return {
        "frames": float(n_frames),
        "frames_rendered": float(len(frame_indices)),
        "frames_reused": float(skipped_frames),
        "stress_ranges": {
            mode: {
                "stress_max": float(stats["stress_abs_max"]),
                "stress_plot_max": float(stats["stress_plot_max"]),
            }
            for mode, stats in stress_ranges.items()
        },
        "disp_max": float(disp_max),
        "deform_scale": float(deform_scale),
        "swap_axes": bool(swap_axes),
        "stress_mode": stress_mode,
        "range_scan": "all-frames-two-pass-histogram",
        "range_histogram_bins": RANGE_HISTOGRAM_BINS,
        "range_cache": str(range_cache_path),
    }


def make_video(
    frames_dir: Path,
    video_path: Path,
    *,
    fps: int,
    crf: int,
    preset: str,
) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "stress_%07d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        str(video_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one stress image per frame and combine them into an mp4."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--width", type=float, default=15.75)
    parser.add_argument("--height", type=float, default=6.75)
    parser.add_argument("--deform-scale", type=float, default=None)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stress-percentile", type=float, default=99.5)
    parser.add_argument(
        "--stress-mode",
        choices=[*SINGLE_STRESS_MODES, COMBINED_STRESS_MODE],
        default=COMBINED_STRESS_MODE,
    )
    parser.add_argument("--swap-axes", dest="swap_axes", action="store_true")
    parser.add_argument("--no-swap-axes", dest="swap_axes", action="store_false")
    parser.set_defaults(swap_axes=False)
    parser.add_argument("--margin", type=float, default=8.0)
    parser.add_argument(
        "--ranges-only",
        action="store_true",
        help="Scan every frame and checkpoint global stress ranges without rendering.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    plot_dir = input_path.parent.parent / "Plot"
    frames_dir = args.frames_dir or plot_dir / "stress_triptych_frames"
    video = args.video or plot_dir / "stress_triptych_60fps.mp4"
    if args.ranges_only:
        frames_dir.mkdir(parents=True, exist_ok=True)
        range_cache = frames_dir / ".stress_range_cache.npz"
        stress_ranges, disp_max = compute_global_ranges(
            input_path,
            stress_mode=args.stress_mode,
            stress_percentile=args.stress_percentile,
            cache_path=range_cache,
        )
        print(
            {
                "input": str(input_path),
                "range_cache": str(range_cache),
                "stress_ranges": stress_ranges,
                "disp_max": disp_max,
            }
        )
        return 0
    stats = render_all_frames(
        input_path,
        frames_dir,
        workers=max(1, args.workers),
        dpi=args.dpi,
        width=args.width,
        height=args.height,
        deform_scale=args.deform_scale,
        frame_limit=args.max_frames,
        stress_percentile=args.stress_percentile,
        swap_axes=args.swap_axes,
        margin=args.margin,
        stress_mode=args.stress_mode,
    )
    make_video(
        frames_dir,
        video,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
    )
    print(
        {
            "frames_dir": str(frames_dir),
            "video": str(video),
            "fps": args.fps,
            **stats,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
