import sys
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plot_dense_near_fault_peak_to_peak import (  # noqa: E402
    output_paths,
    pearson_correlation,
    resolved_fit_interval,
    sampled_station_indices,
    shear_displacement_stop_frame,
    von_mises_2d,
)


def test_output_paths_put_plots_and_stats_in_separate_directories(tmp_path):
    input_path = tmp_path / "run" / "data" / "simulation.h5"
    plot_path = tmp_path / "run" / "Plot" / "dense.pdf"
    stats_dir = tmp_path / "run" / "stats"

    paths = output_paths(input_path, plot_path, stats_dir)

    assert paths["pdf"] == plot_path
    assert paths["png"] == plot_path.with_suffix(".png")
    assert paths["csv"] == stats_dir / "dense.csv"
    assert paths["json"] == stats_dir / "dense.json"


def test_resolved_fit_interval_trims_a_reverse_front():
    coordinate = np.arange(0.0, 501.0)
    arrival = 10.0 + coordinate / 1250.0
    reverse = coordinate > 385.0
    arrival[reverse] = arrival[385] - (coordinate[reverse] - 385.0) / 2500.0

    fit_start, fit_end = resolved_fit_interval(
        coordinate,
        arrival,
        120.0,
        440.0,
    )

    assert fit_start == 120.0
    assert fit_end == 385.0


def test_resolved_fit_interval_falls_back_to_partial_rupture_extent():
    coordinate = np.arange(0.0, 501.0, 5.0)
    arrival = np.full_like(coordinate, np.nan)
    reached = coordinate <= 75.0
    arrival[reached] = 2.0 + coordinate[reached] / 1000.0

    fit_start, fit_end = resolved_fit_interval(
        coordinate,
        arrival,
        120.0,
        440.0,
    )

    assert fit_start == 5.0
    assert fit_end == 70.0


def test_station_stride_always_keeps_the_last_mesh_station():
    assert sampled_station_indices(6, 2).tolist() == [0, 2, 4, 5]


def test_von_mises_2d_matches_the_animation_definition():
    stress = np.asarray([[[10.0, 4.0], [4.0, -6.0]]])

    np.testing.assert_allclose(von_mises_2d(stress), np.sqrt([244.0]))


def test_shear_displacement_stop_frame_is_first_frame_on_final_plateau():
    frames = np.asarray([3, 4, 5, 6])
    displacement = np.asarray([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3])

    assert shear_displacement_stop_frame(frames, displacement) == 5


def test_pearson_correlation_handles_linear_and_constant_series():
    first = np.asarray([1.0, 2.0, 3.0, 4.0])

    assert pearson_correlation(first, 2.0 * first) == 1.0
    assert pearson_correlation(first, np.ones_like(first)) is None
