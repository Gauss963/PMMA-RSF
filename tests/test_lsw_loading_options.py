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
    settings = dict(mesh_size=20, normal_phase_time=2e-5, normal_ramp_time=1e-5,
                    shear_phase_time=4e-5, shear_ramp_time=2e-5, stop_shear_loading_on_rupture=False)
    settings.update(changes)
    return replace(cfg, **settings)


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


def test_reset_slip_weakening_at_shear_start_clears_normal_phase_slip(tmp_path):
    case, cfg, _, _ = load_lsw_case(SOURCE)
    # Low friction so the contact ends slip during the (short, fast) normal ramp.
    case = replace(case, friction=replace(case.friction, mu_s=0.1, mu_k=0.05))
    cfg = _short(cfg, reset_slip_weakening_at_shear_start=True)
    run_simulation_dumped(case, cfg, tmp_path / "reset.h5", frames_per_phase=4,
                          shear_frames_per_phase=8, include_initial_frame=False)
    with h5py.File(tmp_path / "reset.h5") as h5:
        phase = h5["phase_id"][:]
        slip = h5["interface/cumulative_slip"][:]
        erased = h5.attrs["slip_weakening_reset_max_cumulative_slip"]
        assert h5.attrs["reset_slip_weakening_at_shear_start"] == 1
    last_normal = np.flatnonzero(phase == 1)[-1]
    assert erased > 0.0
    assert slip[last_normal].max() == pytest.approx(erased, rel=1e-6)
    # The first shear frame only holds slip accumulated after the reset.
    assert slip[last_normal + 1].max() < slip[last_normal].max()


def test_reset_slip_weakening_requires_slip_weakening_law():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    with pytest.raises(ValueError, match="slip-weakening"):
        build_case_model(case, _short(cfg, reset_slip_weakening_at_shear_start=True,
                                      friction_law="rate-state-regularized"))


def _static_spring_cfg(cfg, **changes):
    return _short(cfg, shear_loading_mode="spring-displacement", shear_loading_stiffness=25.0,
                  shear_spring_rigid_face=True, lock_shear_edge_during_normal=False,
                  reset_slip_weakening_at_shear_start=True, normal_phase_mode="static",
                  normal_ramp_time=0.0, **changes)


def test_static_normal_solution_is_an_equilibrium():
    from tatva.pmma.static_normal import solve_static_normal_phase

    case, cfg, _, _ = load_lsw_case(SOURCE)
    cfg = _static_spring_cfg(cfg)
    model = build_case_model(case, cfg)
    state = solve_static_normal_phase(model, cfg)
    assert state["relative_residual"] < 1e-9
    assert state["open_nodes"] == 0
    weights = np.asarray(model["interface_weights"], dtype=np.float64)
    # Moving-block x equilibrium: contact normal force balances the applied 16 MPa x 500 mm.
    assert np.sum(weights * state["sigma_n"]) == pytest.approx(16.0 * 500.0, rel=1e-6)
    assert np.all(state["tau_over_sigma"] <= np.asarray(model["mu_s_profile"]) + 1e-9)


def test_static_normal_start_holds_still(tmp_path):
    case, cfg, _, _ = load_lsw_case(SOURCE)
    case = replace(case, friction=replace(case.friction, mu_s=0.25, mu_k=0.15))
    cfg = _static_spring_cfg(cfg)
    result = run_simulation_dumped(case, cfg, tmp_path / "static.h5", frames_per_phase=8,
                                   shear_frames_per_phase=4, include_initial_frame=True)
    columns = list(result["columns"])
    history = np.asarray(result["history"], dtype=np.float64)
    with h5py.File(tmp_path / "static.h5") as h5:
        phase = h5["phase_id"][:]
        elastic = h5.attrs["static_normal_relative_residual"]
        assert h5.attrs["normal_phase_mode"] == "static"
    assert elastic < 1e-9
    normal = phase == 1
    kinetic = history[normal, columns.index("kinetic_energy")]
    stored = history[normal, columns.index("elastic_energy")]
    assert kinetic.max() < 1e-4 * stored.max()


def test_static_normal_requires_zero_ramp_and_shear_only_weakening():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    with pytest.raises(ValueError, match="normal_ramp_time"):
        build_case_model(case, _short(cfg, normal_phase_mode="static", normal_ramp_time=1e-5,
                                      reset_slip_weakening_at_shear_start=True))
    with pytest.raises(ValueError, match="reset_slip_weakening_at_shear_start"):
        build_case_model(case, _short(cfg, normal_phase_mode="static", normal_ramp_time=0.0))
