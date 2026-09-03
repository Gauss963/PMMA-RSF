from __future__ import annotations

import numpy as np
import pytest

from scripts.plot_near_fault_stress_fluctuation import (
    piecewise_linear_window_mean,
    required_frame_time_bounds,
)


@pytest.mark.parametrize("speed_m_per_s", [2000.0, -2000.0])
def test_required_frame_time_bounds_include_residual_window(
    speed_m_per_s: float,
) -> None:
    bounds = required_frame_time_bounds(
        np.asarray([1.0, 2.0]),
        speed_m_per_s=speed_m_per_s,
        xi_min_mm=-80.0,
        xi_max_mm=80.0,
        residual_end_ms=0.05,
    )
    assert bounds == pytest.approx((0.95, 2.05))


def test_required_frame_time_bounds_include_larger_xi_window() -> None:
    bounds = required_frame_time_bounds(
        np.asarray([1.0, 2.0]),
        speed_m_per_s=1000.0,
        xi_min_mm=-80.0,
        xi_max_mm=80.0,
        residual_end_ms=0.05,
    )
    assert bounds == pytest.approx((0.92, 2.08))


def test_required_frame_time_bounds_reject_zero_speed() -> None:
    with pytest.raises(ValueError, match="finite and nonzero"):
        required_frame_time_bounds(
            np.asarray([1.0]),
            speed_m_per_s=0.0,
            xi_min_mm=-80.0,
            xi_max_mm=80.0,
            residual_end_ms=0.05,
        )


def test_piecewise_linear_window_mean_without_interior_sample() -> None:
    result = piecewise_linear_window_mean(
        np.asarray([0.0, 2.0]),
        np.asarray([0.0, 2.0]),
        0.4,
        0.5,
    )
    assert result == pytest.approx(0.45)


def test_piecewise_linear_window_mean_preserves_value_columns() -> None:
    result = piecewise_linear_window_mean(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([[0.0, 2.0], [1.0, 4.0], [2.0, 6.0]]),
        0.25,
        1.75,
    )
    np.testing.assert_allclose(result, [1.0, 4.0])


def test_piecewise_linear_window_mean_rejects_uncovered_window() -> None:
    with pytest.raises(ValueError, match="beyond the saved samples"):
        piecewise_linear_window_mean(
            np.asarray([0.0, 1.0]),
            np.asarray([2.0, 3.0]),
            -0.1,
            0.5,
        )
