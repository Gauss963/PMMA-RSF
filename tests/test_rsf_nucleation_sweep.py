from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.analyze_rsf_nucleation_sweep import analyze, arrival_fit, velocity_diagnostics
from scripts.generate_ts0336_nucleation_sweep_cases import (
    CASE_COUNT, DIRECT_EFFECTS, RAMP_TIMES_S, TEMPLATE, case_path, render_case,
)
from tatva.pmma.config import load_case_config
from tatva.pmma.dynamics import build_case_model, run_simulation_dumped
from tatva.pmma.profiles import regularized_steady_friction
from tatva.pmma.runner import make_case, make_run_config, preflight


def test_matrix_keeps_steady_curve_energy_and_displacement_boundary():
    template = TEMPLATE.read_text()
    baseline = load_case_config(TEMPLATE)
    energy = baseline.rsf.middle.state_effect * baseline.rsf.middle.characteristic_slip
    combinations = set()
    total_bytes = 0
    assert CASE_COUNT == 16
    for index in range(1, CASE_COUNT + 1):
        path = case_path(index)
        assert path.read_text() == render_case(template, index)
        cfg = load_case_config(path)
        loading = cfg.loading
        assert cfg.rsf.initial_state_mode == "steady-state"
        assert cfg.rsf.target_normalized_prestress is None
        assert loading.prestress_shear_displacement is None
        assert loading.effective_normal_relaxation_time is None
        assert loading.quasistatic_shear_fraction == 0.0
        assert loading.normal_phase_time == 0.04
        assert loading.normal_ramp_time == 0.02
        assert not loading.relax_tangential_contact_during_normal
        assert loading.shear_displacement_initial == 0.0
        assert loading.shear_displacement_final == 2.45
        assert loading.stop_velocity == 500.0
        assert loading.stop_coverage_fraction is None
        assert (loading.stop_min_y, loading.stop_max_y) == (5.0, 25.0)
        assert cfg.moving == baseline.moving
        assert cfg.stationary == baseline.stationary
        assert cfg.material == baseline.material
        assert cfg.numerics == baseline.numerics
        assert cfg.rsf.leading == baseline.rsf.leading
        zone = cfg.rsf.middle
        assert cfg.rsf.loading == zone
        assert zone.state_effect * zone.characteristic_slip == pytest.approx(energy, rel=1e-13)
        for velocity in (1e-4, 0.1, 200.0, 10000.0):
            kwargs = dict(velocity=velocity, reference_friction=cfg.rsf.initial_friction,
                          reference_velocity=cfg.rsf.reference_velocity)
            assert regularized_steady_friction(
                direct_effect=zone.direct_effect, state_effect=zone.state_effect, **kwargs
            ) == pytest.approx(regularized_steady_friction(
                direct_effect=baseline.rsf.middle.direct_effect,
                state_effect=baseline.rsf.middle.state_effect, **kwargs
            ), rel=1e-10)
        combinations.add((zone.direct_effect, loading.shear_ramp_time))
        estimate = preflight(cfg)
        assert not estimate["animation_enabled"]
        assert estimate["within_dump_limit"]
        total_bytes += estimate["estimated_uncompressed_bytes"]
    assert combinations == {(a, t) for a in DIRECT_EFFECTS for t in RAMP_TIMES_S}
    assert total_bytes < 1_400_000_000_000
    control = load_case_config(case_path(8))
    assert control.rsf.middle == baseline.rsf.middle
    assert control.loading.shear_ramp_time == baseline.loading.shear_ramp_time
    with pytest.raises(ValueError):
        case_path(17)


def test_continuous_solver_has_no_prestress_or_state_reset_and_freezes_on_event(tmp_path):
    cfg = load_case_config(case_path(8))
    cfg = replace(
        cfg,
        numerics=replace(cfg.numerics, mesh_size=100.0, time_step=1e-7),
        loading=replace(cfg.loading, normal_phase_time=4e-5, normal_ramp_time=1e-5,
                        shear_phase_time=2e-5, shear_ramp_time=1e-5,
                        shear_displacement_final=3e-3,
                        stop_min_y=0.0, stop_max_y=200.0, stop_velocity=1e-8),
    )
    model = build_case_model(make_case(cfg), make_run_config(cfg))
    assert model["shear_displacement_boundary"] == "moving-block-right"
    assert model["shear_loading_mode"] == "displacement"
    np.testing.assert_array_equal(model["shear_displacement_pressure"], 0.0)
    output = tmp_path / "continuous.h5"
    result = run_simulation_dumped(
        make_case(cfg), make_run_config(cfg), output,
        frames_per_phase=2, shear_frames_per_phase=4,
        interface_frames_per_phase=400, shear_interface_frames_per_phase=200,
        include_initial_frame=False, store_bulk_strain=False, store_bulk_velocity=False,
    )
    assert result["summary"]["rsf_state_handoff"] is None
    assert result["summary"]["normal_relaxation_handoff"] is None
    with h5py.File(output, "r") as h5:
        assert not h5.attrs.get("rsf_state_handoff_reinitialized", 0)
        high = h5["interface_high_rate"]
        phase = np.asarray(high["phase_id"])
        columns = [x.decode() for x in high["history_columns"]]
        history = np.asarray(high["history"])
        normal = np.flatnonzero(phase == 1)
        shear = np.flatnonzero(phase == 2)
        np.testing.assert_array_equal(history[normal, columns.index("applied_shear_displacement")], 0.0)
        theta = np.asarray(high["rsf_state"])
        assert np.all(np.isfinite(theta)) and np.all(theta > 0)
        # Ageing can grow theta only at one second per second, never by inversion.
        theta0 = np.asarray(h5["interface/rsf_characteristic_slip_profile"]) / cfg.rsf.initial_steady_velocity
        assert np.max(theta - theta0[None, :]) < 1e-3
        np.testing.assert_allclose(theta[shear[0]], theta[normal[-1]], rtol=1e-3, atol=1e-6)
        y = np.asarray(high["contact_line_y"])
        mask = (y >= 0) & (y <= 200)
        rates = np.asarray(high["slip_rate"])
        slip = np.asarray(high["cumulative_slip"])
        trigger = np.any((rates[:, mask] >= cfg.loading.stop_velocity) &
                         (slip[:, mask] >= cfg.loading.stop_slip), axis=1) & (phase == 2)
        stopped = history[:, columns.index("shear_loading_stopped")] > 0.5
        assert np.any(trigger)
        assert np.flatnonzero(stopped)[0] == np.flatnonzero(trigger)[0]
        displacement = history[:, columns.index("applied_shear_displacement")]
        np.testing.assert_array_equal(displacement[stopped], displacement[np.flatnonzero(stopped)[0]])


def test_streaming_arrivals_preserve_normal_events_and_chunk_boundary(tmp_path):
    n = 600
    time_ms = np.arange(1, n + 1) * 0.01
    y = np.arange(0.0, 501.0, 10.0)
    # First station slides during normal loading; another crosses at a chunk boundary.
    arrival = 2.0 + y / 200.0
    arrival[0] = 0.5
    arrival[1] = 2.56
    rates = np.maximum(time_ms[:, None] - arrival[None, :] + 0.1, 0.0) * 5000
    phase = np.where(time_ms <= 1.0, 1, 2)
    steps = np.arange(1, n + 1) - np.where(phase == 2, 100, 0)
    with h5py.File(tmp_path / "trace.h5", "w") as h5:
        high = h5.create_group("interface_high_rate")
        high["contact_line_y"] = y
        high["phase_id"] = phase
        high["step_id"] = steps
        high["slip_rate"] = rates
        high["rsf_state"] = np.full(rates.shape, 4.0)
        result = velocity_diagnostics(high, 1e-5, 100, np.full(y.shape, 0.4))
    np.testing.assert_allclose(result["arrivals"][500.0], arrival, atol=1e-12)
    assert result["normal_peak"][0] > 500
    fit = arrival_fit(y, result["arrivals"][500.0] - 1.0, 470.0)
    assert fit["speed_m_per_s"] == pytest.approx(200.0)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert not arrival_fit(y, np.full(y.shape, np.nan), 470)["available"]


def test_scheduler_maps_sixteen_independent_gpus_to_new_runs():
    root = Path(__file__).resolve().parents[1]
    script = (root / "slurm/PMMA-RSF-GB200-R1-NUCLEATION-SWEEP.slurm").read_text()
    assert "#SBATCH --ntasks=16" in script
    assert "#SBATCH --partition=gb200-r1" in script
    assert "#SBATCH --time=16:00:00" in script
    assert "--gpus-per-task=1" in script
    assert "--mpi=none" in script
    assert "SWEEP_RUN_NUMBER=$((335 + SWEEP_INDEX))" in script
    assert "PREFLIGHT_ONLY" in script
    assert "estimated_uncompressed_bytes" in script
    assert "1_400_000_000_000" in script


def test_nucleation_analysis_writes_stats_and_detects_propagation_after_stop(tmp_path):
    run = tmp_path / "TS0343"
    (run / "input").mkdir(parents=True)
    (run / "data").mkdir()
    (run / "input/case.toml").write_text(case_path(8).read_text())
    y = np.arange(0.0, 501.0, 10.0)
    time_ms = np.arange(1, 501) * 0.01
    dc = 0.00042852936846183833
    slip = np.maximum(time_ms[:, None] - (0.5 + y[None, :] / 200), 0) * 0.1
    rates = np.where(slip > 0, 2000.0, 0.0)
    columns = ["time", "applied_shear", "avg_tau", "avg_sigma_n", "max_penetration",
               "max_slip", "mu_eff_mean", "elastic_energy", "interface_energy",
               "kinetic_energy", "applied_shear_displacement", "shear_loading_stopped",
               "loading_face_displacement", "shear_boundary_reaction"]
    history = np.zeros((502, len(columns)))
    history[2:, columns.index("applied_shear_displacement")] = np.minimum(time_ms, 1.0)
    history[2:, columns.index("shear_boundary_reaction")] = 10.0
    history[2:, columns.index("shear_loading_stopped")] = time_ms >= 1.0
    history[2:, columns.index("elastic_energy")] = 100.0 - time_ms
    history[2:, columns.index("kinetic_energy")] = 0.5
    with h5py.File(run / "data/simulation.h5", "w") as h5:
        h5.attrs["dt"] = 1e-5
        h5.attrs["pressure_steps"] = 4000
        iface = h5.create_group("interface")
        iface["rsf_characteristic_slip_profile"] = np.full(y.shape, dc)
        high = h5.create_group("interface_high_rate")
        high["contact_line_y"] = y
        high["history_columns"] = np.asarray(columns, dtype="S")
        high["history"] = history
        high["phase_id"] = np.r_[1, 1, np.full(500, 2)]
        high["step_id"] = np.r_[2000, 4000, np.arange(1, 501)]
        high["cumulative_slip"] = np.vstack((np.zeros((2, len(y))), slip))
        high["slip_rate"] = np.vstack((np.zeros((2, len(y))), rates))
        high["rsf_state"] = np.full((502, len(y)), 4.0)
    result = analyze(run)
    json.dumps(result, allow_nan=False)
    assert result["post_stop_boundary_work"] == 0.0
    assert result["post_stop_500_crossing_fraction"] > 0.5
    assert result["post_stop_elastic_energy_release"] == pytest.approx(4.0)
    assert result["normal_phase_max_saved_rate_mm_s"] == 0.0
    assert result["maximum_saved_state_s"] == 4.0
    assert result["first_crossing_500_fit"]["speed_m_per_s"] == pytest.approx(200, rel=0.01)
    assert (run / "stats/rsf_nucleation_station_arrivals.csv").exists()
