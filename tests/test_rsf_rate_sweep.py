from dataclasses import asdict
import shutil

import h5py
import numpy as np
import pytest

from scripts.analyze_rsf_rate_sweep import analyze, first_crossings, front_fit
from scripts.generate_ts0278_rsf_rate_sweep_cases import (
    CASE_COUNT,
    BASELINE_INDEX,
    DYNAMIC_VELOCITY_MM_S,
    TEMPLATE,
    breakdown_integral,
    calibrated_parameters,
    case_path,
    ramp_time,
    render_case,
)
from tatva.pmma.config import load_case_config
from tatva.pmma.estimate import estimate_case_size


def test_rate_sweep_preserves_czm_energy_and_baseline_geometry():
    template = TEMPLATE.read_text(encoding="utf-8")
    baseline = load_case_config(TEMPLATE)
    old = baseline.rsf
    new_b, new_dc = calibrated_parameters(
        {
            "rsf": {
                "middle": {"a": old.middle.direct_effect, "b": old.middle.state_effect, "dc": old.middle.characteristic_slip},
                "initial_steady_velocity": old.initial_steady_velocity,
                "dynamic_calibration_velocity": old.dynamic_calibration_velocity,
                "initial_friction": old.initial_friction,
                "target_middle_dynamic_friction": old.target_middle_dynamic_friction,
            }
        }
    )
    old_energy_factor = (
        old.middle.state_effect
        * old.middle.characteristic_slip
        * breakdown_integral(
            old.dynamic_calibration_velocity / old.initial_steady_velocity
        )
    )
    new_energy_factor = (
        new_b
        * new_dc
        * breakdown_integral(
            DYNAMIC_VELOCITY_MM_S / old.initial_steady_velocity
        )
    )
    assert new_energy_factor == pytest.approx(old_energy_factor, rel=1e-12)
    assert new_dc == pytest.approx(0.00042852936846183833)

    ramps = []
    total_estimated_dump_bytes = 0
    for index in range(1, CASE_COUNT + 2):
        path = case_path(index)
        assert path.read_text(encoding="utf-8") == render_case(template, index)
        config = load_case_config(path)
        assert config.moving == baseline.moving
        assert config.stationary == baseline.stationary
        assert config.material == baseline.material
        assert config.numerics == baseline.numerics
        assert config.loading.normal_loading_mode == "stress"
        assert config.loading.normal_stress_reference == pytest.approx(16.0)
        assert config.loading.shear_displacement_final == pytest.approx(2.45)
        assert config.rsf.dynamic_calibration_velocity == pytest.approx(200.0)
        assert config.rsf.middle.state_effect == pytest.approx(new_b)
        for zone_name in ("loading", "middle", "leading"):
            assert getattr(config.rsf, zone_name).characteristic_slip == pytest.approx(new_dc)
        estimate = estimate_case_size(config)
        estimated_dump_bytes = int(
            estimate["estimated_uncompressed_bytes"]
            * config.output.estimated_compression_ratio
        )
        assert estimated_dump_bytes < 100_000_000_000
        total_estimated_dump_bytes += estimated_dump_bytes
        if index <= CASE_COUNT:
            ramps.append(config.loading.shear_ramp_time)
            assert config.loading.stop_min_y == baseline.loading.stop_min_y
            assert config.loading.stop_max_y == baseline.loading.stop_max_y
            assert config.loading.stop_slip == pytest.approx(new_dc)
        else:
            anchor = load_case_config(case_path(BASELINE_INDEX))
            payload = asdict(config.loading)
            anchor_payload = asdict(anchor.loading)
            for key in ("stop_slip", "stop_min_y", "stop_max_y", "stop_coverage_fraction"):
                payload.pop(key)
                anchor_payload.pop(key)
            assert payload == anchor_payload
            assert config.loading.stop_slip == pytest.approx(1.0e-12)
            assert config.loading.stop_min_y == 5.0
            assert config.loading.stop_max_y == 25.0
            assert config.loading.stop_coverage_fraction is None
    assert ramps == sorted(ramps, reverse=True)
    assert ramp_time(BASELINE_INDEX) == pytest.approx(0.075)
    assert total_estimated_dump_bytes < 1_400_000_000_000


def test_front_fit_and_crossing_interpolation():
    y = np.arange(0.0, 501.0, 1.0)
    time = np.arange(0.0, 20.1, 0.1)
    arrival = 2.0 + y / 40.0
    slip = np.maximum(time[:, None] - arrival[None, :], 0.0)
    arrivals = first_crossings(slip, time, np.full(y.shape, 0.5))
    fit = front_fit(y, arrivals, 469.5)
    assert fit["available"]
    assert fit["speed_m_per_s"] == pytest.approx(40.0, rel=1e-3)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_rate_analysis_reads_high_rate_dump_and_integrates_boundary_work(tmp_path):
    run_dir = tmp_path / "TS0309"
    (run_dir / "input").mkdir(parents=True)
    (run_dir / "data").mkdir()
    shutil.copyfile(TEMPLATE, run_dir / "input/case.toml")
    y = np.arange(0.0, 501.0, 10.0)
    shear_steps = np.arange(1, 1501)
    shear_time_ms = shear_steps * 0.01
    dc = 0.0003614447313307937
    shear_slip = np.maximum(
        shear_time_ms[:, None] - y[None, :] / 40.0,
        0.0,
    ) * dc / 0.1
    shear_slip_rate = np.zeros_like(shear_slip)
    shear_slip_rate[99, 1] = 500.0
    columns = [
        "time", "applied_shear", "avg_tau", "avg_sigma_n", "max_penetration",
        "max_slip", "mu_eff_mean", "elastic_energy", "interface_energy",
        "kinetic_energy", "applied_shear_displacement", "shear_loading_stopped",
        "loading_face_displacement", "shear_boundary_reaction",
    ]
    history = np.zeros((1502, len(columns)), dtype=np.float32)
    history[2:, columns.index("applied_shear_displacement")] = (
        np.minimum(shear_steps, 1200) / 1500.0
    )
    history[1201:, columns.index("shear_loading_stopped")] = 1.0
    history[2:, columns.index("shear_boundary_reaction")] = 10.0

    with h5py.File(run_dir / "data/simulation.h5", "w") as h5:
        h5.attrs["dt"] = 1.0e-5
        h5.attrs["pressure_steps"] = 4000
        iface = h5.create_group("interface")
        iface.create_dataset(
            "rsf_characteristic_slip_profile", data=np.full(y.shape, dc)
        )
        high = h5.create_group("interface_high_rate")
        high.create_dataset("contact_line_y", data=y)
        high.create_dataset("history_columns", data=np.asarray(columns, dtype="S"))
        high.create_dataset("history", data=history)
        high.create_dataset("phase_id", data=np.concatenate(([1, 1], np.full(1500, 2))))
        high.create_dataset("step_id", data=np.concatenate(([3999, 4000], shear_steps)))
        high.create_dataset(
            "cumulative_slip", data=np.vstack((np.zeros((2, y.size)), shear_slip))
        )
        high.create_dataset(
            "slip_rate", data=np.vstack((np.zeros((2, y.size)), shear_slip_rate))
        )

    result = analyze(run_dir)
    assert result["rupture_speed_1dc"]["speed_m_per_s"] == pytest.approx(40.0, rel=1e-3)
    assert result["nucleation_time_in_shear_ms"] == pytest.approx(1.0)
    assert result["loading_stop_time_in_shear_ms"] == pytest.approx(12.0)
    assert result["loading_stop_displacement_mm"] == pytest.approx(0.8)
    assert result["boundary_work_after_nucleation_until_stop"] == pytest.approx(
        10.0 * (0.8 - 0.1 / 1.5), rel=1e-3
    )
