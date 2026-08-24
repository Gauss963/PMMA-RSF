import sys
from pathlib import Path

import numpy as np
import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plot_sigma_xy_probe_traces import validate_off_fault_distances  # noqa: E402


def test_single_positive_off_fault_distance_is_valid():
    distances, zero_indices, positive_indices = validate_off_fault_distances([5.0])

    np.testing.assert_array_equal(distances, [5.0])
    np.testing.assert_array_equal(zero_indices, [])
    np.testing.assert_array_equal(positive_indices, [0])


def test_interface_and_positive_off_fault_distances_are_valid():
    distances, zero_indices, positive_indices = validate_off_fault_distances(
        [0.0, 1.0, 5.0]
    )

    np.testing.assert_array_equal(distances, [0.0, 1.0, 5.0])
    np.testing.assert_array_equal(zero_indices, [0])
    np.testing.assert_array_equal(positive_indices, [1, 2])


@pytest.mark.parametrize("distances", [[], [0.0], [2.0, 1.0], [1.0, 1.0]])
def test_invalid_off_fault_distances_are_rejected(distances):
    with pytest.raises(ValueError):
        validate_off_fault_distances(distances)
