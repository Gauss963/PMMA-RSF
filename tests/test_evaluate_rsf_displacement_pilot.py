from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_rsf_displacement_pilot import _front_metrics


def test_front_metrics_identify_clean_forward_rupture():
    metrics = _front_metrics(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([1, 2, 3, 4]),
        np.arange(6, dtype=np.float64) * 1.0e-4,
        0.0,
    )

    assert metrics["nucleation_y_min_mm"] == pytest.approx(1.0)
    assert metrics["nucleation_y_max_mm"] == pytest.approx(1.0)
    assert metrics["forward_step_fraction"] == pytest.approx(1.0)
    assert metrics["backward_step_count"] == 0
    assert metrics["rupture_duration_ms"] == pytest.approx(0.3)


def test_front_metrics_detect_bilateral_or_backward_arrivals():
    metrics = _front_metrics(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.asarray([4, 3, 2, 3, 4]),
        np.arange(6, dtype=np.float64) * 1.0e-4,
        0.0,
    )

    assert metrics["nucleation_y_min_mm"] == pytest.approx(3.0)
    assert metrics["nucleation_y_max_mm"] == pytest.approx(3.0)
    assert metrics["forward_step_fraction"] == pytest.approx(0.5)
    assert metrics["backward_step_count"] == 2
    assert metrics["largest_backward_step_ms"] == pytest.approx(-0.1)
