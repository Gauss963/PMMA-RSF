from pathlib import Path
import time

import h5py
import numpy as np
import pytest

from tatva.pmma.dynamics import (
    SimulationCheckpointed,
    _shear_loading_stop_candidates,
    _shear_loading_stop_reached,
    build_case_model,
    quadrature_weighted_element_average,
    run_simulation_dumped,
)
from tatva.pmma.config import load_case_config
from tatva.pmma.estimate import estimate_case_size
from tatva.pmma.profiles import build_rate_state_profile, calibrate_state_effect
from tatva.pmma.profiles import regularized_steady_friction
from tatva.pmma.runner import (
    _validate_run_storage,
    allocate_run_directory,
    make_case,
    make_run_config,
    run_case,
)
from CohesiveZoneModel.Lc_estimate import RSF_D_c, RSF_ZONES, mm


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CASE = ROOT / "cases/rsf_0116_hpc.toml"
SIX_HOUR_CASE = ROOT / "cases/rsf_0116_6h.toml"
HYBRID_CASE = ROOT / "cases/rsf_0116_hybrid30_6h.toml"
CORRECTED_HYBRID_CASE = (
    ROOT / "cases/rsf_0116_hybrid10_6h.toml"
)
EXPLICIT_Q4_CASE = (
    ROOT / "cases/rsf_0116_q4_explicit_10h.toml"
)
COHESIVE_CALIBRATED_CASE = (
    ROOT / "cases/rsf_0117_q4_explicit_10h.toml"
)
TS0118_CASE = ROOT / "cases/rsf_0118_q4_explicit_10h.toml"
TS0119_CASE = ROOT / "cases/rsf_0119_q4_explicit_10h.toml"
TS0120_CASE = ROOT / "cases/rsf_0120_q4_fully_explicit_12h.toml"


def test_run_directory_sequence_starts_at_ts0117_and_increments(tmp_path):
    (tmp_path / "0116_legacy-run").mkdir()
    (tmp_path / "TS0017_historical-run").mkdir()

    first = allocate_run_directory(tmp_path)
    second = allocate_run_directory(tmp_path)

    assert first.name == "TS0117"
    assert second.name == "TS0118"


def test_run_directory_sequence_continues_from_highest_ts_id(tmp_path):
    (tmp_path / "TS0121_previous").mkdir()

    allocated = allocate_run_directory(tmp_path)

    assert allocated.name == "TS0122"


def test_cohesive_calibrated_case_matches_lc_estimate():
    config = load_case_config(COHESIVE_CALIBRATED_CASE)
    expected_dc_mm = RSF_D_c / mm

    for name in ("loading", "middle", "leading"):
        zone = getattr(config.rsf, name)
        assert zone.direct_effect == pytest.approx(RSF_ZONES[name]["a"])
        assert zone.state_effect == pytest.approx(RSF_ZONES[name]["b"])
        assert zone.characteristic_slip == pytest.approx(expected_dc_mm)
    assert config.loading.stop_slip == pytest.approx(expected_dc_mm)


def test_storage_preflight_uses_uncompressed_remaining_size_and_reserve(
    tmp_path, monkeypatch
):
    class Usage:
        free = 129

    monkeypatch.setattr("tatva.pmma.runner.shutil.disk_usage", lambda _: Usage())
    estimate = {"estimated_uncompressed_bytes": 100}

    with pytest.raises(OSError, match="Insufficient free space"):
        _validate_run_storage(
            tmp_path,
            estimate,
            existing_dump_bytes=20,
            reserve_bytes=50,
        )

    Usage.free = 130
    report = _validate_run_storage(
        tmp_path,
        estimate,
        existing_dump_bytes=20,
        reserve_bytes=50,
    )
    assert report == {
        "available_bytes": 130,
        "estimated_remaining_uncompressed_bytes": 80,
        "reserve_bytes": 50,
        "required_bytes": 130,
    }


def test_standard_rsf_calibration_recovers_former_lsw_drop():
    b = calibrate_state_effect(
        initial_friction=0.8,
        dynamic_friction=0.45,
        direct_effect=0.005,
        characteristic_slip=0.00629795,
        reference_velocity=1.0e-4,
        reference_state=3.3e-4,
        initial_steady_velocity=1.0e-4,
        dynamic_velocity=2000.0,
    )

    assert b == pytest.approx(0.025819400653936703)
    assert 0.005 - b < 0.0
    dynamic_mu = regularized_steady_friction(
        velocity=2000.0,
        reference_friction=0.8,
        direct_effect=0.005,
        state_effect=b,
        reference_velocity=1.0e-4,
    )
    assert dynamic_mu == pytest.approx(0.45)


def test_three_zone_profile_has_half_cosine_transitions_and_uniform_f0():
    config = load_case_config(PRODUCTION_CASE)
    specification = make_run_config(config).rsf_profile_spec
    y = np.arange(0.0, 501.0, 5.0)

    profile = build_rate_state_profile(y, specification)

    a = profile["direct_effect"]
    b = profile["state_effect"]
    assert a[y == 0.0] - b[y == 0.0] < 0.0
    assert a[y == 250.0] - b[y == 250.0] < 0.0
    assert a[y == 500.0] - b[y == 500.0] > 0.0
    assert a[y == 35.0] == pytest.approx(0.5 * (0.004 + 0.005))
    assert a[y == 465.0] == pytest.approx(0.5 * (0.005 + 0.008))
    assert np.allclose(profile["reference_friction"], 0.8)


def test_production_case_estimate_stays_below_one_tb():
    config = load_case_config(PRODUCTION_CASE)

    estimate = estimate_case_size(config)

    assert make_run_config(config).operator_batch_size == 65_536
    assert estimate["degrees_of_freedom"] == 35_977_904
    assert estimate["bulk_frames"] == 600
    assert estimate["interface_frames"] == 99_800
    assert estimate["estimated_uncompressed_tb"] == pytest.approx(0.532225534836)
    assert estimate["bulk_cfl_dt_estimate_s"] == pytest.approx(
        1.2852546070157183e-8
    )
    assert estimate["bulk_limited_steps_estimate"] == 5_446_392


def test_six_hour_case_preserves_shear_window_and_reduces_step_budget():
    config = load_case_config(SIX_HOUR_CASE)
    estimate = estimate_case_size(config)

    assert config.numerics.mesh_size == pytest.approx(0.5)
    assert config.numerics.cfl == pytest.approx(0.35)
    assert config.numerics.contact_safety_factor == pytest.approx(0.75)
    assert config.loading.normal_phase_time == pytest.approx(0.010)
    assert config.loading.normal_ramp_time == pytest.approx(0.005)
    assert config.loading.shear_phase_time == pytest.approx(0.030)
    assert config.loading.shear_ramp_time == pytest.approx(0.010)
    assert estimate["bulk_cfl_dt_estimate_s"] == pytest.approx(
        6.426273035078592e-8
    )
    assert estimate["bulk_limited_steps_estimate"] == 622_445
    assert estimate["estimated_uncompressed_tb"] < 0.04


def test_hybrid_case_preloads_30_percent_before_explicit_dynamics():
    from dataclasses import replace

    config = load_case_config(HYBRID_CASE)
    estimate = estimate_case_size(config)
    coarse = replace(config, numerics=replace(config.numerics, mesh_size=50.0))
    model = build_case_model(make_case(coarse), make_run_config(coarse))

    expected_target = 0.30 * config.loading.shear_displacement_final
    assert model["quasistatic_shear_target"] == pytest.approx(expected_target)
    assert float(model["shear_displacement_pressure"][0]) == pytest.approx(0.0)
    assert float(model["shear_displacement_pressure"][-1]) == pytest.approx(
        expected_target
    )
    assert float(model["shear_displacement_shear"][0]) == pytest.approx(
        expected_target
    )
    assert float(model["shear_displacement_shear"][-1]) == pytest.approx(
        config.loading.shear_displacement_final
    )
    assert config.loading.normal_phase_time + config.loading.shear_phase_time == (
        pytest.approx(0.032)
    )
    assert estimate["estimated_uncompressed_tb"] < 0.03


def test_corrected_hybrid_case_uses_safe_preload_and_dynamic_front_trigger():
    from dataclasses import replace

    config = load_case_config(CORRECTED_HYBRID_CASE)
    estimate = estimate_case_size(config)
    coarse = replace(config, numerics=replace(config.numerics, mesh_size=50.0))
    model = build_case_model(make_case(coarse), make_run_config(coarse))

    assert config.loading.quasistatic_shear_fraction == pytest.approx(0.10)
    assert config.loading.quasistatic_shear_ramp_time == pytest.approx(0.001)
    assert config.loading.shear_ramp_time == pytest.approx(0.009)
    assert config.loading.normal_phase_time + config.loading.shear_phase_time == (
        pytest.approx(0.032)
    )
    assert model["quasistatic_shear_target"] == pytest.approx(0.176)
    assert model["shear_loading_stop_velocity"] == pytest.approx(10.0)
    y = np.asarray(model["moving"].mesh.coords[np.asarray(model["master_nodes"]), 1])
    mask = np.asarray(model["shear_loading_stop_mask"])
    assert np.array_equal(mask, (y >= 10.0) & (y <= 440.0))
    assert estimate["estimated_uncompressed_tb"] < 0.03


def test_explicit_q4_case_relaxes_only_normal_loading_and_fits_dump_budget():
    from dataclasses import replace

    config = load_case_config(EXPLICIT_Q4_CASE)
    estimate = estimate_case_size(config)
    # The production stop trigger is exactly y=440 mm; use a coarse spacing
    # that still contains that monitored station.
    coarse = replace(config, numerics=replace(config.numerics, mesh_size=20.0))
    model = build_case_model(make_case(coarse), make_run_config(coarse))

    assert config.loading.quasistatic_shear_fraction == pytest.approx(0.0)
    assert config.loading.normal_relaxation_time == pytest.approx(2.0e-4)
    assert config.loading.relax_tangential_contact_during_normal is True
    assert model["normal_relaxation_time"] == pytest.approx(2.0e-4)
    assert model["relax_tangential_contact_during_normal"] is True
    assert float(model["shear_displacement_pressure"][-1]) == pytest.approx(0.0)
    assert float(model["shear_displacement_shear"][0]) == pytest.approx(0.0)
    assert float(model["shear_displacement_shear"][-1]) == pytest.approx(2.45)
    assert config.numerics.contact_safety_factor == pytest.approx(0.50)
    assert config.numerics.time_step == pytest.approx(15.0e-9)
    assert model["dt"] == pytest.approx(15.0e-9)
    assert model["dt_limiter"] == "configured"
    assert model["dt"] < model["dt_stable_limit"]
    assert config.loading.stop_min_y == pytest.approx(440.0)
    assert config.output.bulk_shear_frames == 36_000
    assert config.output.interface_shear_frames == 300_000
    assert estimate["estimated_uncompressed_tb"] == pytest.approx(1.2525534117)
    assert config.output.estimated_compression_ratio == pytest.approx(0.955)
    assert estimate["estimated_uncompressed_tb"] > config.output.maximum_dump_tb


def test_ts0118_uses_terminal_ramp_exact_step_and_calibrated_rsf():
    from dataclasses import replace

    config = load_case_config(TS0118_CASE)
    estimate = estimate_case_size(config)
    coarse = replace(config, numerics=replace(config.numerics, mesh_size=20.0))
    model = build_case_model(make_case(coarse), make_run_config(coarse))

    assert config.loading.shear_displacement_final == pytest.approx(2.22)
    assert config.loading.shear_ramp_time == pytest.approx(0.029)
    assert config.loading.shear_phase_time == pytest.approx(0.029)
    assert config.loading.quasistatic_shear_fraction == pytest.approx(0.0)
    assert config.loading.stop_min_y == pytest.approx(440.0)
    assert config.loading.stop_max_y == pytest.approx(440.0)
    assert config.numerics.time_step == pytest.approx(10.0e-9)
    assert model["dt"] == pytest.approx(10.0e-9)
    assert model["dt_limiter"] == "configured"
    assert model["dt"] < model["dt_stable_limit"]
    assert config.rsf.loading.direct_effect == pytest.approx(
        config.rsf.loading.state_effect
    )
    for name in ("loading", "middle", "leading"):
        zone = getattr(config.rsf, name)
        assert zone.characteristic_slip == pytest.approx(RSF_D_c / mm)
    assert config.output.bulk_shear_frames == 36_000
    assert config.output.interface_shear_frames == 300_000
    assert estimate["estimated_uncompressed_tb"] == pytest.approx(1.2525534117)
    assert (
        estimate["estimated_uncompressed_tb"]
        * config.output.estimated_compression_ratio
    ) == pytest.approx(1.1961885082)


def test_ts0119_uses_full_fault_stop_coverage_and_ts0116_displacement_margin():
    config = load_case_config(TS0119_CASE)
    run_config = make_run_config(config)

    assert config.loading.shear_displacement_final == pytest.approx(2.45)
    assert config.loading.shear_ramp_time == pytest.approx(0.029)
    assert config.loading.quasistatic_shear_fraction == pytest.approx(0.0)
    assert config.loading.stop_velocity == pytest.approx(500.0)
    assert config.loading.stop_min_y == pytest.approx(0.5)
    assert config.loading.stop_max_y == pytest.approx(499.0)
    assert config.loading.stop_coverage_fraction == pytest.approx(1.0)
    assert run_config.shear_loading_stop_coverage_fraction == pytest.approx(1.0)
    assert config.numerics.time_step == pytest.approx(10.0e-9)
    for name in ("loading", "middle", "leading"):
        assert getattr(config.rsf, name).characteristic_slip == pytest.approx(
            RSF_D_c / mm
        )


def test_ts0120_restores_undamped_fully_explicit_0116_normal_loading():
    config = load_case_config(TS0120_CASE)
    estimate = estimate_case_size(config)
    run_config = make_run_config(config)

    assert config.loading.normal_phase_time == pytest.approx(0.040)
    assert config.loading.normal_ramp_time == pytest.approx(0.020)
    assert config.loading.normal_relaxation_time is None
    assert config.loading.quasistatic_damping_time is None
    assert config.loading.effective_normal_relaxation_time is None
    assert config.loading.relax_tangential_contact_during_normal is False
    assert config.loading.quasistatic_shear_fraction == pytest.approx(0.0)
    assert run_config.normal_relaxation_time is None
    assert run_config.relax_tangential_contact_during_normal is False
    assert estimate["configured_steps_estimate"] == 6_900_000
    assert config.output.bulk_normal_frames == 2
    assert config.output.bulk_shear_frames == 36_000
    assert config.output.interface_normal_frames == 100
    assert config.output.interface_shear_frames == 300_000


def test_estimator_matches_remainder_cells_used_by_structured_mesh():
    from dataclasses import replace

    config = load_case_config(EXPLICIT_Q4_CASE)
    config = replace(
        config,
        moving=replace(config.moving, dimensions=(1.0, 1.0)),
        stationary=replace(config.stationary, dimensions=(1.1, 1.0)),
        numerics=replace(config.numerics, mesh_size=0.4),
    )

    estimate = estimate_case_size(config)

    # 1.0 -> [0, .4, .8, 1.0], 1.1 -> [0, .4, .8, 1.1].
    assert estimate["moving_elements"] == 9
    assert estimate["stationary_elements"] == 9
    assert estimate["moving_nodes"] == 16
    assert estimate["stationary_nodes"] == 16
    assert estimate["minimum_cell_size_mm"] == pytest.approx(0.2)


def test_quadrature_output_uses_volume_weighted_element_average():
    values = np.asarray(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
                [[3.0, 0.0], [0.0, 3.0]],
                [[8.0, 0.0], [0.0, 8.0]],
            ]
        ],
        dtype=np.float32,
    )
    weights = np.asarray([[1.0, 1.0, 2.0, 4.0]], dtype=np.float32)

    averaged = quadrature_weighted_element_average(values, weights)

    np.testing.assert_allclose(
        np.asarray(averaged),
        np.asarray([[[41.0 / 8.0, 0.0], [0.0, 41.0 / 8.0]]]),
    )


def test_loading_stop_trigger_requires_slip_rate_at_a_monitored_station():
    cumulative_slip = np.asarray([0.02, 0.02, 0.001], dtype=np.float32)
    slip_rate = np.asarray([100.0, 5.0, 100.0], dtype=np.float32)
    stop_mask = np.asarray([False, True, True])
    critical_slip = np.full(3, 0.006, dtype=np.float32)

    candidates = _shear_loading_stop_candidates(
        cumulative_slip,
        slip_rate,
        stop_mask,
        critical_slip,
        np.asarray(0.006, dtype=np.float32),
        True,
        np.asarray(10.0, dtype=np.float32),
    )
    assert not np.asarray(candidates).any()

    slip_rate[1] = 20.0
    candidates = _shear_loading_stop_candidates(
        cumulative_slip,
        slip_rate,
        stop_mask,
        critical_slip,
        np.asarray(0.006, dtype=np.float32),
        True,
        np.asarray(10.0, dtype=np.float32),
    )
    assert np.array_equal(np.asarray(candidates), [False, True, False])


def test_loading_stop_coverage_rejects_an_isolated_far_edge_event():
    cumulative_slip = np.asarray([0.008, 0.008, 0.001, 0.008], dtype=np.float32)
    slip_rate = np.asarray([0.0, 0.0, 0.0, 1000.0], dtype=np.float32)
    stop_mask = np.asarray([False, True, True, True])
    critical_slip = np.full(4, 0.006, dtype=np.float32)

    reached = _shear_loading_stop_reached(
        cumulative_slip,
        slip_rate,
        stop_mask,
        critical_slip,
        np.asarray(0.006, dtype=np.float32),
        True,
        np.asarray(500.0, dtype=np.float32),
        1.0,
    )
    assert not bool(np.asarray(reached))

    cumulative_slip[2] = 0.008
    reached = _shear_loading_stop_reached(
        cumulative_slip,
        slip_rate,
        stop_mask,
        critical_slip,
        np.asarray(0.006, dtype=np.float32),
        True,
        np.asarray(500.0, dtype=np.float32),
        1.0,
    )
    assert bool(np.asarray(reached))


def test_contact_safety_factor_scales_contact_limit_without_changing_bulk_limit():
    from dataclasses import replace

    config = load_case_config(SIX_HOUR_CASE)
    coarse = replace(
        config,
        numerics=replace(
            config.numerics,
            mesh_size=50.0,
            contact_safety_factor=0.25,
        ),
    )
    baseline = build_case_model(make_case(coarse), make_run_config(coarse))
    raised_config = replace(
        coarse,
        numerics=replace(coarse.numerics, contact_safety_factor=1.0),
    )
    raised = build_case_model(make_case(raised_config), make_run_config(raised_config))

    assert raised["dt_bulk"] == pytest.approx(baseline["dt_bulk"])
    assert raised["dt_contact"] == pytest.approx(4.0 * baseline["dt_contact"])
    assert raised["contact_safety_factor"] == pytest.approx(1.0)


def test_configured_time_step_cannot_exceed_assembled_stability_limit():
    from dataclasses import replace

    config = load_case_config(SIX_HOUR_CASE)
    coarse = replace(config, numerics=replace(config.numerics, mesh_size=50.0))
    stable = build_case_model(make_case(coarse), make_run_config(coarse))
    unsafe = replace(
        coarse,
        numerics=replace(
            coarse.numerics,
            time_step=1.01 * stable["dt_stable_limit"],
        ),
    )

    with pytest.raises(ValueError, match="exceeds the assembled stability limit"):
        build_case_model(make_case(unsafe), make_run_config(unsafe))


def test_regularized_profile_reaches_model_interface():
    config = load_case_config(PRODUCTION_CASE)
    # Dataclass replacement keeps this test independent of TOML serialization.
    from dataclasses import replace

    coarse = replace(config, numerics=replace(config.numerics, mesh_size=50.0))
    model = build_case_model(make_case(coarse), make_run_config(coarse))
    a = np.asarray(model["rsf_parameters"]["direct_effect"])
    b = np.asarray(model["rsf_parameters"]["state_effect"])

    assert model["friction_law"] == "rate-state-regularized"
    assert a[0] - b[0] < 0.0
    assert a[-1] - b[-1] > 0.0


def test_regularized_dump_separates_bulk_and_interface_frames(tmp_path):
    from dataclasses import replace

    config = load_case_config(PRODUCTION_CASE)
    config = replace(
        config,
        numerics=replace(config.numerics, mesh_size=100.0, cfl=0.2),
        loading=replace(
            config.loading,
            normal_phase_time=2.0e-5,
            normal_ramp_time=1.0e-5,
            shear_phase_time=2.0e-5,
            shear_ramp_time=1.0e-5,
            normal_displacement=1.0e-3,
            shear_displacement_final=1.0e-3,
            normal_relaxation_time=2.0e-6,
            relax_tangential_contact_during_normal=True,
            stop_on_rupture=False,
        ),
    )
    output = tmp_path / "simulation.h5"

    result = run_simulation_dumped(
        make_case(config),
        make_run_config(config),
        output,
        frames_per_phase=2,
        shear_frames_per_phase=3,
        interface_frames_per_phase=4,
        shear_interface_frames_per_phase=6,
        include_initial_frame=False,
    )

    with h5py.File(output, "r") as h5:
        assert h5["history"].shape[0] == 5
        assert h5["interface_high_rate/history"].shape[0] == 10
        assert h5["interface/rsf_direct_effect_profile"].shape == (6,)
        assert h5.attrs["friction_law"] == "rate-state-regularized"
    assert result["summary"]["saved_frames"] == 5
    assert result["summary"]["interface_shear_frames_saved"] == 6
    assert result["summary"]["quasistatic_handoff"] is None
    assert result["summary"]["normal_relaxation_handoff"] is not None
    assert result["summary"]["normal_relaxation_handoff"][
        "applied_displacement"
    ] == pytest.approx(0.0)
    assert result["summary"]["normal_relaxation_handoff"][
        "post_reset_kinetic_energy"
    ] == pytest.approx(0.0)
    assert result["summary"]["normal_relaxation_handoff"]["velocity_reset"] is True
    with h5py.File(output, "r") as h5:
        assert h5.attrs["normal_relaxation_handoff_kinetic_ratio"] >= 0.0
        assert h5.attrs["normal_relaxation_handoff_max_slip"] == pytest.approx(0.0)
        normal_rows = np.flatnonzero(h5["phase_id"][:] == 1)
        assert h5["history"][normal_rows[-1], 2] == pytest.approx(0.0)
        normal_interface_rows = np.flatnonzero(
            h5["interface_high_rate/phase_id"][:] == 1
        )
        assert np.all(
            h5["interface_high_rate/plastic_slip"][normal_interface_rows] == 0.0
        )
        first_shear_row = np.flatnonzero(
            h5["interface_high_rate/phase_id"][:] == 2
        )[0]
        assert abs(h5["interface_high_rate/history"][first_shear_row, 2]) < 1.0e-3


def test_quasistatic_handoff_diagnostics_are_saved(tmp_path):
    from dataclasses import replace

    config = load_case_config(PRODUCTION_CASE)
    config = replace(
        config,
        numerics=replace(config.numerics, mesh_size=100.0, cfl=0.2),
        loading=replace(
            config.loading,
            normal_phase_time=2.0e-5,
            normal_ramp_time=2.0e-6,
            shear_phase_time=2.0e-5,
            shear_ramp_time=1.0e-5,
            normal_displacement=1.0e-3,
            shear_displacement_final=1.0e-3,
            quasistatic_shear_fraction=0.7,
            quasistatic_shear_start_time=4.0e-6,
            quasistatic_shear_ramp_time=8.0e-6,
            quasistatic_damping_time=2.0e-6,
            stop_on_rupture=False,
        ),
    )
    output = tmp_path / "hybrid.h5"

    result = run_simulation_dumped(
        make_case(config),
        make_run_config(config),
        output,
        frames_per_phase=2,
        shear_frames_per_phase=2,
        interface_frames_per_phase=3,
        shear_interface_frames_per_phase=3,
        include_initial_frame=False,
    )

    handoff = result["summary"]["quasistatic_handoff"]
    assert handoff is not None
    assert handoff["applied_displacement"] == pytest.approx(7.0e-4, rel=1e-5)
    assert handoff["kinetic_ratio"] >= 0.0
    assert isinstance(handoff["stop_slip_threshold_exceeded"], bool)
    assert isinstance(handoff["stop_velocity_threshold_exceeded"], bool)
    assert isinstance(handoff["stop_trigger_reached"], bool)
    with h5py.File(output, "r") as h5:
        assert h5.attrs["quasistatic_shear_fraction"] == pytest.approx(0.7)
        assert h5.attrs["quasistatic_handoff_kinetic_ratio"] >= 0.0


def test_regularized_dump_resumes_checkpoint_without_changing_solution(tmp_path):
    from dataclasses import replace

    config = load_case_config(PRODUCTION_CASE)
    config = replace(
        config,
        numerics=replace(config.numerics, mesh_size=100.0, cfl=0.2),
        loading=replace(
            config.loading,
            normal_phase_time=2.0e-5,
            normal_ramp_time=1.0e-5,
            shear_phase_time=2.0e-5,
            shear_ramp_time=1.0e-5,
            normal_displacement=1.0e-3,
            shear_displacement_final=1.0e-3,
            stop_on_rupture=False,
        ),
    )
    interrupted_output = tmp_path / "interrupted.h5"
    checkpoint = tmp_path / "checkpoint.npz"
    common = dict(
        frames_per_phase=2,
        shear_frames_per_phase=3,
        interface_frames_per_phase=4,
        shear_interface_frames_per_phase=6,
        include_initial_frame=False,
        checkpoint_path=checkpoint,
    )

    with pytest.raises(SimulationCheckpointed):
        run_simulation_dumped(
            make_case(config),
            make_run_config(config),
            interrupted_output,
            checkpoint_deadline_monotonic=time.monotonic() - 1.0,
            **common,
        )
    assert checkpoint.exists()

    resumed = run_simulation_dumped(
        make_case(config),
        make_run_config(config),
        interrupted_output,
        resume=True,
        **common,
    )
    reference = run_simulation_dumped(
        make_case(config),
        make_run_config(config),
        tmp_path / "reference.h5",
        frames_per_phase=2,
        shear_frames_per_phase=3,
        interface_frames_per_phase=4,
        shear_interface_frames_per_phase=6,
        include_initial_frame=False,
    )

    assert not checkpoint.exists()
    np.testing.assert_allclose(
        resumed["history"], reference["history"], rtol=5.0e-6, atol=1.0e-6
    )
    assert resumed["summary"]["saved_frames"] == 5


def test_case_runner_marks_checkpoint_and_resumes_same_directory(tmp_path):
    from dataclasses import replace
    import json

    config = load_case_config(PRODUCTION_CASE)
    config = replace(
        config,
        numerics=replace(config.numerics, mesh_size=100.0, cfl=0.2),
        loading=replace(
            config.loading,
            normal_phase_time=2.0e-5,
            normal_ramp_time=1.0e-5,
            shear_phase_time=2.0e-5,
            shear_ramp_time=1.0e-5,
            normal_displacement=1.0e-3,
            shear_displacement_final=1.0e-3,
            stop_on_rupture=False,
        ),
        output=replace(
            config.output,
            bulk_normal_frames=2,
            bulk_shear_frames=3,
            interface_normal_frames=4,
            interface_shear_frames=6,
            checkpoint_interval_minutes=1.0,
        ),
    )
    run_dir = tmp_path / "job-test"

    run_case(
        config,
        PRODUCTION_CASE,
        run_root=tmp_path,
        run_dir=run_dir,
        time_limit_seconds=1.0e-9,
    )
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "checkpointed"
    assert (run_dir / "checkpoint.npz").exists()

    run_case(
        config,
        PRODUCTION_CASE,
        run_root=tmp_path,
        run_dir=run_dir,
        resume=True,
    )
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["resume_count"] == 1
    assert not (run_dir / "checkpoint.npz").exists()
