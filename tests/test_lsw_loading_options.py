from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from tatva.pmma.dynamics import build_case_model, run_simulation_dumped
from tatva.pmma.lsw import load_lsw_case


SOURCE = Path(__file__).resolve().parents[1] / "cases/lsw_uniform_q4_normal_stress_16mpa.toml"


def _short(cfg, **changes):
    """Coarse mesh and phases short enough for a unit test, at the production dt."""
    return replace(cfg, mesh_size=20, normal_phase_time=2e-5, normal_ramp_time=1e-5,
                   shear_phase_time=4e-5, shear_ramp_time=2e-5, stop_shear_loading_on_rupture=False,
                   **changes)


def test_uniform_cases_drop_nucleation_zone_and_creep():
    for name in ("lsw_uniform_q4_normal_stress_16mpa.toml",
                 "lsw_uniform_q4_chamfer20x5_normal_stress_16mpa.toml"):
        _, cfg, _, _ = load_lsw_case(SOURCE.parent / name)
        assert cfg.element_type == "quad4"
        assert cfg.loading_edge_nucleation_length == 0.0
        assert cfg.leading_edge_creep_length == 0.0
        assert cfg.leading_edge_creep_relaxation_time == 0.0
        assert cfg.shear_post_ramp_rate == 0.0


def test_post_ramp_rate_continues_actuator_after_ramp():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    cfg = _short(cfg, shear_post_ramp_rate=5.0)
    model = build_case_model(case, cfg)
    dt = float(model["dt"])
    schedule = np.asarray(model["shear_displacement_shear"], dtype=np.float64)
    ramp_steps = int(np.ceil(cfg.shear_ramp_time / dt))
    target = cfg.shear_displacement_s_override
    assert schedule[ramp_steps - 1] == pytest.approx(target, rel=1e-6)
    after = np.arange(1, schedule.size - ramp_steps + 1) * dt
    np.testing.assert_allclose(schedule[ramp_steps:], target + 5.0 * after, rtol=1e-6)
    velocity = np.asarray(model["shear_velocity_shear"], dtype=np.float64)
    np.testing.assert_allclose(velocity[ramp_steps + 1:], 5.0, rtol=1e-3)


def test_post_ramp_rate_rejects_negative_or_stress_loading():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    with pytest.raises(ValueError, match="non-negative"):
        build_case_model(case, _short(cfg, shear_post_ramp_rate=-1.0))
    with pytest.raises(ValueError, match="displacement"):
        build_case_model(case, _short(cfg, shear_post_ramp_rate=1.0, shear_loading_mode="stress"))


def test_spring_reaction_reports_spring_force(tmp_path):
    case, cfg, _, _ = load_lsw_case(SOURCE)
    cfg = _short(cfg, shear_loading_mode="spring-displacement", shear_loading_stiffness=25.0,
                 lock_shear_edge_during_normal=False)
    result = run_simulation_dumped(case, cfg, tmp_path / "spring.h5", frames_per_phase=4,
                                   shear_frames_per_phase=8, include_initial_frame=False)
    columns = list(result["columns"])
    history = np.asarray(result["history"], dtype=np.float64)
    traction = history[:, columns.index("applied_shear")]
    reaction = history[:, columns.index("shear_boundary_reaction")]
    actuator = history[:, columns.index("applied_shear_displacement")]
    face = history[:, columns.index("loading_face_displacement")]
    # Spring force per unit thickness = k (actuator - face) x 200 mm loaded face.
    np.testing.assert_allclose(reaction, traction * 200.0, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(traction, 25.0 * (actuator - face), rtol=1e-4, atol=1e-6)
    assert np.any(np.abs(reaction) > 1.0)


def test_rigid_spring_face_moves_as_one_body(tmp_path):
    case, cfg, _, _ = load_lsw_case(SOURCE)
    cfg = _short(cfg, shear_loading_mode="spring-displacement", shear_loading_stiffness=25.0,
                 shear_spring_rigid_face=True, lock_shear_edge_during_normal=False)
    result = run_simulation_dumped(case, cfg, tmp_path / "rigid.h5", frames_per_phase=4,
                                   shear_frames_per_phase=8, include_initial_frame=False)
    with h5py.File(tmp_path / "rigid.h5") as h5:
        assert h5.attrs["shear_spring_rigid_face"] == 1
        coords = h5["moving/coords"][:]
        face_uy = h5["moving/displacement"][:, coords[:, 1] < 1e-9, 1]
    assert face_uy.shape[1] > 2 and np.abs(face_uy).max() > 0.0
    np.testing.assert_array_equal(np.ptp(face_uy, axis=1), 0.0)
    columns = list(result["columns"])
    history = np.asarray(result["history"], dtype=np.float64)
    np.testing.assert_allclose(history[:, columns.index("loading_face_displacement")],
                               face_uy[:, 0], rtol=1e-5, atol=1e-9)


def test_rigid_spring_face_requires_spring_loading():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    with pytest.raises(ValueError, match="spring-displacement"):
        build_case_model(case, _short(cfg, shear_spring_rigid_face=True))
