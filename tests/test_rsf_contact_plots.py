import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


PLOT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(PLOT_DIR))

from plot_contact_friction_map import plot_mu_eff_maps  # noqa: E402
from plot_contact_mu_disp import plot_contact_mu_disp  # noqa: E402
from plot_rsf_rupture_analysis import (  # noqa: E402
    _zone_metadata,
    first_velocity_crossing,
)


def test_first_velocity_crossing_interpolates_across_chunks(tmp_path):
    input_path = tmp_path / "velocity.h5"
    with h5py.File(input_path, "w") as h5:
        rates = h5.create_dataset(
            "rates",
            data=np.asarray(
                [
                    [0.0, 0.0],
                    [400.0, 0.0],
                    [600.0, -200.0],
                    [800.0, -400.0],
                    [1200.0, -600.0],
                ]
            ),
        )
        arrivals = first_velocity_crossing(
            rates,
            np.arange(5, dtype=np.int64),
            np.arange(5, dtype=np.float64),
            500.0,
            chunk_frames=2,
        )

    assert arrivals == pytest.approx([1.5, 3.5])


def test_zone_metadata_uses_independent_transition_lengths(tmp_path):
    input_path = tmp_path / "zones.h5"
    with h5py.File(input_path, "w") as h5:
        h5.attrs["rsf_profile_spec_json"] = (
            '{"loading_length": 30.0, "leading_length": 30.0, '
            '"transition_length": 50.0, '
            '"loading_transition_length": 50.0, '
            '"leading_transition_length": 100.0}'
        )
        zones = _zone_metadata(h5, np.asarray([0.0, 500.0]))

    assert zones["loading_transition_end"] == pytest.approx(80.0)
    assert zones["leading_transition_start"] == pytest.approx(370.0)
    assert zones["leading_start"] == pytest.approx(470.0)


def test_contact_plots_use_saved_rsf_coefficient(tmp_path):
    input_path = tmp_path / "rsf.h5"
    saved_coefficient = np.array([[0.30, 0.31], [0.32, 0.33]], dtype=np.float32)
    with h5py.File(input_path, "w") as h5:
        h5.attrs["dt"] = 1.0e-4
        h5.attrs["pressure_steps"] = 1
        h5.attrs["friction_law"] = "rate-state-vws"
        h5.create_dataset("history", data=np.zeros((2, 13), dtype=np.float32))
        h5.create_dataset("phase_id", data=np.array([1, 2], dtype=np.int32))
        h5.create_dataset("step_id", data=np.array([1, 1], dtype=np.int32))
        moving = h5.create_group("moving")
        moving.create_dataset(
            "coords", data=np.array([[0.0, 0.0], [0.0, 500.0]], dtype=np.float32)
        )
        interface = h5.create_group("interface")
        interface.attrs["mu_static"] = 0.8
        interface.attrs["mu_kinetic"] = 0.6
        interface.attrs["critical_slip"] = 8.0
        interface.create_dataset("master_nodes", data=np.array([0, 1]))
        interface.create_dataset("contact_line_y", data=np.array([0.0, 500.0]))
        interface.create_dataset(
            "cumulative_slip", data=np.zeros((2, 2), dtype=np.float32)
        )
        interface.create_dataset("friction_coefficient", data=saved_coefficient)
        interface.create_dataset("mu_static_profile", data=np.array([0.8, 0.8]))
        interface.create_dataset("mu_kinetic_profile", data=np.array([0.6, 0.6]))

    map_stats = plot_mu_eff_maps(
        input_path,
        tmp_path / "map.pdf",
        tmp_path / "phase-map.pdf",
    )
    path_stats = plot_contact_mu_disp(input_path, tmp_path / "path.pdf")

    assert map_stats["mu_min_final"] == pytest.approx(0.32)
    assert path_stats["final_mu"] == pytest.approx(0.32)
    assert (tmp_path / "map.pdf").exists()
    assert (tmp_path / "phase-map.pdf").exists()
    assert (tmp_path / "path.pdf").exists()
