import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from render_stress_frames import (  # noqa: E402
    PNG_IEND,
    PNG_SIGNATURE,
    _frame_is_complete,
    _percentile_from_histogram,
    compute_global_ranges,
    render_modes,
    stress_scalar,
)


def test_frame_is_complete_checks_png_header_and_terminal_chunk(tmp_path):
    complete = tmp_path / "complete.png"
    complete.write_bytes(PNG_SIGNATURE + b"payload" + PNG_IEND)
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(PNG_SIGNATURE + b"payload")
    wrong_header = tmp_path / "wrong.png"
    wrong_header.write_bytes(b"not-png!" + b"payload" + PNG_IEND)

    assert _frame_is_complete(complete)
    assert not _frame_is_complete(truncated)
    assert not _frame_is_complete(wrong_header)
    assert not _frame_is_complete(tmp_path / "missing.png")


def test_histogram_percentile_uses_all_values_with_bin_limited_error():
    values = np.linspace(0.0, 10.0, 100_001, dtype=np.float64)
    histogram, edges = np.histogram(values, bins=4096)
    percentile = _percentile_from_histogram(histogram, edges, 99.5)

    assert abs(percentile - np.percentile(values, 99.5)) <= edges[1] - edges[0]


def test_render_modes_triptych_order():
    assert render_modes("stress_triptych") == ("sigma_yy", "von_mises", "sigma_xy")


def test_stress_scalar_extracts_sigma_components_and_von_mises():
    stress = np.array(
        [
            [[10.0, 4.0], [4.0, -6.0]],
            [[3.0, -2.0], [-2.0, 5.0]],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(stress_scalar(stress, "sigma_xy"), np.array([4.0, -2.0]))
    np.testing.assert_allclose(stress_scalar(stress, "sigma_yy"), np.array([-6.0, 5.0]))

    expected = np.array(
        [
            math.sqrt(10.0 * 10.0 - 10.0 * (-6.0) + (-6.0) * (-6.0) + 3.0 * 4.0 * 4.0),
            math.sqrt(3.0 * 3.0 - 3.0 * 5.0 + 5.0 * 5.0 + 3.0 * (-2.0) * (-2.0)),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(stress_scalar(stress, "von_mises"), expected)


def test_compute_global_ranges_scans_every_frame_and_reuses_cache(tmp_path):
    h5_path = tmp_path / "frames.h5"
    cache_path = tmp_path / "ranges.npz"
    n_frames = 5
    moving_stress = np.zeros((n_frames, 3, 2, 2), dtype=np.float32)
    stationary_stress = np.zeros((n_frames, 2, 2, 2), dtype=np.float32)
    for frame_idx in range(n_frames):
        moving_stress[frame_idx, :, 0, 0] = frame_idx + np.arange(3)
        moving_stress[frame_idx, :, 1, 1] = 2.0 * frame_idx + np.arange(3)
        moving_stress[frame_idx, :, 0, 1] = -(frame_idx + np.arange(3))
        moving_stress[frame_idx, :, 1, 0] = moving_stress[frame_idx, :, 0, 1]
        stationary_stress[frame_idx] = moving_stress[frame_idx, :2] * 0.5

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("history", data=np.zeros((n_frames, 6), dtype=np.float32))
        moving = h5.create_group("moving")
        stationary = h5.create_group("stationary")
        moving.create_dataset("stress", data=moving_stress)
        stationary.create_dataset("stress", data=stationary_stress)
        moving.create_dataset(
            "displacement",
            data=np.full((n_frames, 4, 2), 3.0, dtype=np.float32),
        )
        stationary.create_dataset(
            "displacement",
            data=np.full((n_frames, 3, 2), 4.0, dtype=np.float32),
        )

    first_ranges, first_disp_max = compute_global_ranges(
        h5_path,
        batch_size=2,
        stress_mode="stress_triptych",
        stress_percentile=99.5,
        cache_path=cache_path,
        histogram_bins=1024,
        checkpoint_frames=1,
    )
    second_ranges, second_disp_max = compute_global_ranges(
        h5_path,
        batch_size=2,
        stress_mode="stress_triptych",
        stress_percentile=99.5,
        cache_path=cache_path,
        histogram_bins=1024,
        checkpoint_frames=1,
    )

    combined_stress = np.concatenate([moving_stress, stationary_stress], axis=1)
    for mode in render_modes("stress_triptych"):
        values = stress_scalar(combined_stress, mode)
        if mode != "von_mises":
            values = np.abs(values)
        bin_width = float(values.max()) / 1024
        assert first_ranges[mode]["stress_abs_max"] == float(values.max())
        assert first_ranges[mode]["stress_plot_max"] == pytest.approx(
            np.percentile(values, 99.5, method="lower"),
            abs=bin_width,
        )
    assert first_disp_max == pytest.approx(math.sqrt(32.0))
    assert second_ranges == first_ranges
    assert second_disp_max == first_disp_max
    assert cache_path.exists()
