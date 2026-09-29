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


@pytest.mark.parametrize("mode", ["residual", "pre-event"])
def test_late_plateau_guard_does_not_block_local_trace(tmp_path, monkeypatch, mode):
    import h5py
    import plot_sigma_xy_probe_traces as plots

    path = tmp_path / "simulation.h5"
    with h5py.File(path, "w") as h5:
        h5["interface/contact_line_y"] = [0., 100., 200.]
        h5["interface/cumulative_slip"] = np.zeros((6, 3))
    monkeypatch.setattr(plots, "configure_style", lambda: None)
    monkeypatch.setattr(plots, "saved_time_ms", lambda h5: (np.arange(6.), np.arange(6)))
    monkeypatch.setattr(plots, "_critical_slip_profile", lambda h5, y: np.ones(3))
    monkeypatch.setattr(plots, "first_crossing_times", lambda *a, **k: np.array([1., 2., 4.]))
    monkeypatch.setattr(plots, "_select_dense_and_tail_frames", lambda *a: (np.arange(6), 6))

    class ReachedProbeSelection(Exception):
        pass

    def reached(*args):
        raise ReachedProbeSelection

    monkeypatch.setattr(plots, "choose_probe_patches", reached)
    error = ReachedProbeSelection if mode == "residual" else ValueError
    with pytest.raises(error):
        plots.plot_sigma_xy_probe_traces(path, tmp_path / "trace.png", None,
                                        y_points=[100.], baseline_mode=mode)
