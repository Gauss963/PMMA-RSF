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


def test_float64_requires_x64_instead_of_silently_truncating():
    import jax

    case, cfg, _, _ = load_lsw_case(SOURCE)
    if jax.config.read("jax_enable_x64"):
        pytest.skip("x64 already enabled in this process")
    with pytest.raises(ValueError, match="x64"):
        build_case_model(case, _short(cfg, dtype="float64"))


def _fillet_model(cfg, case, *, radius=10.0, fine=None, mesh=5.0):
    return build_case_model(case, replace(
        cfg, mesh_size=mesh, time_step_override=None, fault_end_refinement_size=fine,
        moving_leading_fault_fillet_radius=radius, moving_loading_fault_fillet_radius=radius))


def test_fault_corner_fillets_follow_quarter_circles_and_start_with_a_gap():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    model = _fillet_model(cfg, case, radius=10.0)
    coords = np.asarray(model["moving"].mesh.coords, dtype=np.float64)
    front = np.asarray(model["moving"].boundary_nodes["moving-block-front"])
    y, x = coords[front, 1], coords[front, 0]
    recess = np.zeros_like(y)
    d_lead = np.clip(y - 490.0, 0.0, 10.0); d_load = np.clip(10.0 - y, 0.0, 10.0)
    recess += 10.0 - np.sqrt(100.0 - d_lead**2) + 10.0 - np.sqrt(100.0 - d_load**2)
    np.testing.assert_allclose(x, 200.0 - recess, atol=1e-4)
    # Every front node is paired; the gap equals the recess and is zero off the fillets.
    gap = np.asarray(model["interface_initial_gap"], dtype=np.float64)
    master = np.asarray(model["master_nodes"])
    assert master.size == front.size
    paired_y = coords[master, 1]
    expected = np.interp(paired_y, np.sort(y), (200.0 - x)[np.argsort(y)])
    np.testing.assert_allclose(gap, expected, atol=1e-4)
    assert np.all(gap[(paired_y > 10.0) & (paired_y < 490.0)] == 0.0)
    assert gap.max() == pytest.approx(10.0, abs=1e-4)


def test_flat_fault_has_zero_initial_gap():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    model = build_case_model(case, _short(cfg))
    assert np.all(np.asarray(model["interface_initial_gap"]) == 0.0)


def test_fault_end_refinement_matches_nodes_across_the_fault():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    coarse = _fillet_model(cfg, case, radius=0.0, fine=None)
    model = _fillet_model(cfg, case, radius=0.0, fine=0.5)
    moving_y = np.unique(np.asarray(model["moving"].mesh.coords)[:, 1])
    stationary_y = np.unique(np.asarray(model["stationary"].mesh.coords)[:, 1])
    on_fault = stationary_y[stationary_y <= 500.0 + 1e-9]
    np.testing.assert_array_equal(moving_y, on_fault)
    spacing = np.diff(moving_y)
    assert spacing.min() == pytest.approx(0.5, rel=1e-6)
    assert spacing.max() <= 5.0 + 1e-9
    assert spacing[0] == pytest.approx(0.5, rel=1e-6) and spacing[-1] == pytest.approx(0.5, rel=1e-6)
    x = np.unique(np.asarray(model["moving"].mesh.coords)[:, 0])
    assert np.diff(x)[-1] == pytest.approx(0.5, rel=1e-6)
    front = np.asarray(model["moving"].boundary_nodes["moving-block-front"])
    assert np.asarray(model["master_nodes"]).size == front.size
    assert len(model["moving"].mesh.elements) > len(coarse["moving"].mesh.elements)


def test_static_normal_with_fillets_opens_the_recessed_pairs():
    from tatva.pmma.static_normal import solve_static_normal_phase

    case, cfg, _, _ = load_lsw_case(SOURCE)
    cfg = _static_spring_cfg(cfg, mesh_size=5.0, time_step_override=None, fault_end_refinement_size=1.0,
                             moving_leading_fault_fillet_radius=10.0, moving_loading_fault_fillet_radius=10.0)
    model = build_case_model(case, cfg)
    state = solve_static_normal_phase(model, cfg)
    assert state["relative_residual"] < 1e-9
    weights = np.asarray(model["interface_weights"], dtype=np.float64)
    assert np.sum(weights * state["sigma_n"]) == pytest.approx(16.0 * 500.0, rel=1e-6)
    gap = np.asarray(model["interface_initial_gap"], dtype=np.float64)
    assert state["open_nodes"] > 0
    assert np.all(state["sigma_n"][gap > 1.0] == 0.0)


def test_filleted_refined_dynamics_holds_static_start(tmp_path):
    case, cfg, _, _ = load_lsw_case(SOURCE)
    cfg = _static_spring_cfg(cfg, mesh_size=20.0, time_step_override=None, fault_end_refinement_size=5.0,
                             moving_leading_fault_fillet_radius=10.0, moving_loading_fault_fillet_radius=10.0)
    result = run_simulation_dumped(case, cfg, tmp_path / "fillet.h5", frames_per_phase=4,
                                   shear_frames_per_phase=4, include_initial_frame=True)
    columns = list(result["columns"])
    history = np.asarray(result["history"], dtype=np.float64)
    assert np.all(np.isfinite(history))
    with h5py.File(tmp_path / "fillet.h5") as h5:
        assert h5.attrs["moving_leading_fault_fillet_radius"] == 10.0
        assert h5.attrs["fault_end_refinement_size"] == 5.0
        normal = h5["phase_id"][:] == 1
    kinetic = history[normal, columns.index("kinetic_energy")]
    assert kinetic.max() < 1e-4 * history[normal, columns.index("elastic_energy")].max()


def test_restart_continues_from_previous_final_state(tmp_path):
    case, cfg, _, _ = load_lsw_case(SOURCE)
    first_cfg = _static_spring_cfg(cfg, shear_displacement_s_override=0.05)
    first = tmp_path / "first.h5"
    run_simulation_dumped(case, first_cfg, first, frames_per_phase=4, shear_frames_per_phase=4,
                          include_initial_frame=True)
    with h5py.File(first) as h5:
        final = h5["final_state"]
        actuator = float(final.attrs["applied_shear_displacement"])
        plastic = np.asarray(final["plastic_slip"])
        assert final["u"].shape[0] == int(h5["moving/displacement"].shape[1] * 2
                                         + h5["stationary/displacement"].shape[1] * 2)
    assert actuator == pytest.approx(0.05, rel=1e-6)

    second_cfg = _static_spring_cfg(cfg, shear_displacement_s_override=0.0, shear_ramp_time=0.0,
                                    shear_post_ramp_rate=10.0, restart_from_file=str(first))
    model = build_case_model(case, second_cfg)
    schedule = np.asarray(model["shear_displacement_shear"], dtype=np.float64)
    dt = float(model["dt"])
    np.testing.assert_allclose(schedule, actuator + 10.0 * dt * np.arange(1, schedule.size + 1), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(model["shear_displacement_pressure"]), actuator, rtol=1e-6)
    second = tmp_path / "second.h5"
    result = run_simulation_dumped(case, second_cfg, second, frames_per_phase=4,
                                   shear_frames_per_phase=4, include_initial_frame=True)
    with h5py.File(second) as h5:
        assert h5.attrs["restart_actuator_displacement"] == pytest.approx(actuator)
        normal = h5["phase_id"][:] == 1
        start_plastic = h5["interface/plastic_slip"][0]
    # Stuck pairs keep the earlier plastic offset (none slide at mu_s on this short case);
    # on this coarse mesh every pair is also a plotted interface node.
    assert start_plastic.size == plastic.size
    np.testing.assert_allclose(start_plastic, plastic, atol=1e-9)
    columns = list(result["columns"])
    history = np.asarray(result["history"], dtype=np.float64)
    kinetic = history[normal, columns.index("kinetic_energy")]
    assert kinetic.max() < 1e-4 * history[normal, columns.index("elastic_energy")].max()


def test_restart_requires_static_spring_start(tmp_path):
    case, cfg, _, _ = load_lsw_case(SOURCE)
    with pytest.raises(ValueError, match="restart_from_file"):
        build_case_model(case, _short(cfg, restart_from_file=str(tmp_path / "missing.h5")))


def test_shear_loading_face_gap_leaves_a_free_strip_at_the_fault():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    full = build_case_model(case, _short(cfg, mesh_size=5.0))
    gapped = build_case_model(case, _short(cfg, mesh_size=5.0, shear_loading_face_gap=5.0))
    coords = np.asarray(gapped["moving"].mesh.coords)
    loaded = np.asarray(gapped["moving_shear_loading_dofs"]) // 2
    assert coords[loaded, 0].max() == pytest.approx(195.0)
    assert np.all(coords[loaded, 1] == 0.0)
    assert np.asarray(gapped["moving_shear_edge_dofs"]).size == loaded.size
    assert float(np.sum(gapped["force_shear_unit"])) == pytest.approx(195.0)
    assert float(np.sum(full["force_shear_unit"])) == pytest.approx(200.0)
    assert gapped["shear_loading_face_width"] == pytest.approx(195.0)
    corner = np.flatnonzero((coords[:, 0] == 200.0) & (coords[:, 1] == 0.0))
    assert np.asarray(gapped["force_shear_unit"])[2 * corner + 1] == 0.0


def test_contact_time_step_is_per_pair_only_on_refined_meshes():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    uniform = build_case_model(case, _short(cfg, mesh_size=5.0, time_step_override=None))
    masses = np.asarray(uniform["mass_flat"], dtype=np.float64)
    fixed = np.asarray(uniform["fixed_dofs"])
    free = np.setdiff1d(np.arange(masses.size), fixed)
    weights = np.asarray(uniform["interface_weights"], dtype=np.float64)
    global_bound = 0.25 * np.sqrt(masses[free].min() / float(uniform["penalty_n"]) / weights.max())
    assert float(uniform["dt_contact"]) == pytest.approx(global_bound, rel=1e-6)

    refined = build_case_model(case, _short(cfg, mesh_size=5.0, time_step_override=None,
                                            fault_end_refinement_size=0.5))
    masses = np.asarray(refined["mass_flat"], dtype=np.float64)
    master = np.asarray(refined["master_nodes"]); slave = np.asarray(refined["slave_nodes"])
    offset = int(refined["moving_offset"])
    pair_mass = np.minimum(masses[2 * master], masses[offset + 2 * slave])
    weights = np.asarray(refined["interface_weights"], dtype=np.float64)
    per_pair = 0.25 * np.sqrt(np.min(pair_mass / (2.0 * float(refined["penalty_n"]) * weights)))
    assert float(refined["dt_contact"]) == pytest.approx(per_pair, rel=1e-6)
    free = np.setdiff1d(np.arange(masses.size), np.asarray(refined["fixed_dofs"]))
    assert per_pair > 0.25 * np.sqrt(masses[free].min() / float(refined["penalty_n"]) / weights.max())


def test_partial_healing_sets_strength_and_remaining_weakening(tmp_path):
    from tatva.pmma.static_normal import solve_static_normal_phase

    case, cfg, _, _ = load_lsw_case(SOURCE)
    case = replace(case, friction=replace(case.friction, mu_s=0.25, mu_k=0.15))
    cfg = _static_spring_cfg(cfg, healing_fraction=0.5, shear_displacement_s_override=0.0)
    model = build_case_model(case, cfg)
    state = solve_static_normal_phase(model, cfg)
    assert state["tau_over_sigma"].max() == pytest.approx(0.20, abs=1e-6)  # ends slide at the healed strength
    run_simulation_dumped(case, cfg, tmp_path / "heal.h5", frames_per_phase=4,
                          shear_frames_per_phase=4, include_initial_frame=True)
    with h5py.File(tmp_path / "heal.h5") as h5:
        assert h5.attrs["healing_fraction"] == 0.5
        first_shear = int(np.flatnonzero(h5["phase_id"][:] == 2)[0])
        mu = h5["interface/friction_coefficient"][first_shear]
        dc = h5["interface/critical_slip_profile"][:]
        memory = h5["interface/cumulative_slip"][first_shear]
    np.testing.assert_allclose(mu, 0.20, atol=1e-6)
    np.testing.assert_allclose(memory, 0.5 * dc, rtol=1e-5)


def test_healing_fraction_validation():
    case, cfg, _, _ = load_lsw_case(SOURCE)
    with pytest.raises(ValueError, match="healing_fraction"):
        build_case_model(case, _short(cfg, healing_fraction=0.0, reset_slip_weakening_at_shear_start=True))
    with pytest.raises(ValueError, match="reset_slip_weakening_at_shear_start"):
        build_case_model(case, _short(cfg, healing_fraction=0.5))
