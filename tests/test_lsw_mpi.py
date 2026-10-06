from dataclasses import replace
from pathlib import Path
import time

import h5py
import numpy as np
import pytest

from tatva.pmma.dynamics import build_case_model, run_simulation_dumped, SimulationCheckpointed
from tatva.pmma.lsw import load_lsw_case


SOURCE = Path(__file__).resolve().parents[1] / "cases/lsw_0114_normal_stress_16mpa.toml"


def test_reference_0114_settings_and_pressure_face():
    case, cfg, output, _ = load_lsw_case(SOURCE)
    assert cfg.element_type == "tri3"
    assert cfg.mesh_size == 1
    assert cfg.normal_phase_time == 0.04
    assert cfg.shear_phase_time == 0.03
    assert cfg.shear_displacement_s_override == 1.76
    assert cfg.shear_loading_stop_slip is None
    assert cfg.loading_edge_critical_slip == 0.315
    assert cfg.leading_edge_creep_length == 10
    assert cfg.leading_edge_creep_transition_length == 10
    assert cfg.normal_relaxation_time is None
    assert output["shear_frames_per_phase"] == 95000
    model = build_case_model(case, replace(cfg, mesh_size=10))
    assert model["moving"].mesh.elements.shape[1] == 3
    assert model["normal_loading_mode"] == "stress"
    force = np.asarray(model["force_normal"])
    indices = np.flatnonzero(force)
    np.testing.assert_array_equal(indices, np.asarray(model["moving_normal_edge_dofs"]))
    assert force.sum() == pytest.approx(16 * 500)
    assert model["normal_displacement_pressure"].max() == 0


def test_lsw_checkpoint_resume_preserves_solution(tmp_path):
    case, cfg, _, _ = load_lsw_case(SOURCE)
    cfg = replace(cfg, mesh_size=20, normal_phase_time=1e-6, normal_ramp_time=1e-6,
                  shear_phase_time=1e-6, shear_ramp_time=1e-6)
    kwargs = dict(frames_per_phase=4, shear_frames_per_phase=4, include_initial_frame=False)
    clean = run_simulation_dumped(case, cfg, tmp_path / "clean.h5", **kwargs)
    checkpoint = tmp_path / "checkpoint.npz"
    with pytest.raises(SimulationCheckpointed):
        run_simulation_dumped(case, cfg, tmp_path / "resume.h5", **kwargs,
                              checkpoint_path=checkpoint,
                              checkpoint_deadline_monotonic=time.monotonic() - 1)
    resumed = run_simulation_dumped(case, cfg, tmp_path / "resume.h5", **kwargs,
                                  checkpoint_path=checkpoint, resume=True)
    np.testing.assert_array_equal(clean["history"], resumed["history"])
    np.testing.assert_array_equal(clean["final_u"], resumed["final_u"])
    with h5py.File(tmp_path / "resume.h5") as h5:
        assert h5.attrs["element_type"] == "tri3"
        assert h5.attrs["saved_frames"] == 8
