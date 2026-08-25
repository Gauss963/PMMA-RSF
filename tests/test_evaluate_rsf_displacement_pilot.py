from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_rsf_displacement_pilot import (
    _front_metrics,
    _record_first_crossings,
    _record_first_profile_crossings,
)


def test_front_metrics_identify_clean_forward_rupture():
    metrics = _front_metrics(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([1, 2, 4, 5]),
        np.arange(6, dtype=np.float64) * 1.0e-4,
        0.0,
    )

    assert metrics["nucleation_y_min_mm"] == pytest.approx(1.0)
    assert metrics["nucleation_y_max_mm"] == pytest.approx(1.0)
    assert metrics["forward_step_fraction"] == pytest.approx(1.0)
    assert metrics["backward_step_count"] == 0
    assert metrics["largest_backward_step_ms"] is None
    assert metrics["largest_backward_step_from_y_mm"] is None
    assert metrics["largest_backward_step_to_y_mm"] is None
    assert metrics["rupture_duration_ms"] == pytest.approx(0.4)
    assert metrics["largest_forward_stall_from_y_mm"] == pytest.approx(2.0)
    assert metrics["largest_forward_stall_to_y_mm"] == pytest.approx(3.0)


def test_front_metrics_detect_bilateral_or_backward_arrivals():
    metrics = _front_metrics(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.asarray([5, 3, 2, 3, 5]),
        np.arange(7, dtype=np.float64) * 1.0e-4,
        0.0,
    )

    assert metrics["nucleation_y_min_mm"] == pytest.approx(3.0)
    assert metrics["nucleation_y_max_mm"] == pytest.approx(3.0)
    assert metrics["forward_step_fraction"] == pytest.approx(0.5)
    assert metrics["backward_step_count"] == 2
    assert metrics["largest_backward_step_ms"] == pytest.approx(-0.2)
    assert metrics["largest_backward_step_from_y_mm"] == pytest.approx(1.0)
    assert metrics["largest_backward_step_to_y_mm"] == pytest.approx(2.0)


def test_first_crossings_distinguish_creep_from_dynamic_rupture():
    block = np.asarray(
        [
            [20.0, 5.0, 0.0],
            [200.0, 30.0, 5.0],
            [1200.0, 500.0, 20.0],
            [900.0, 1500.0, 30.0],
        ]
    )
    rows = np.asarray([10, 11, 12, 13])
    low = np.full(3, -1, dtype=np.int64)
    dynamic = np.full(3, -1, dtype=np.int64)

    _record_first_crossings(block, rows, 10.0, low)
    _record_first_crossings(block, rows, 1000.0, dynamic)

    assert np.array_equal(low, [10, 11, 12])
    assert np.array_equal(dynamic, [12, 13, -1])


def test_profile_crossings_apply_local_dc_thresholds():
    block = np.asarray(
        [
            [0.2, 0.2, 0.2],
            [0.4, 0.6, 0.8],
            [0.8, 1.1, 1.2],
        ]
    )
    rows = np.asarray([20, 21, 22])
    thresholds = np.asarray([0.5, 1.0, 2.0])
    first = np.full(3, -1, dtype=np.int64)

    _record_first_profile_crossings(block, rows, thresholds, first)

    assert np.array_equal(first, [22, 22, -1])
