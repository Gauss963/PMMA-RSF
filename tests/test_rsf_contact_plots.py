import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


PLOT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(PLOT_DIR))

from plot_contact_friction_map import plot_mu_eff_maps  # noqa: E402
from plot_contact_mu_disp import plot_contact_mu_disp  # noqa: E402


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
