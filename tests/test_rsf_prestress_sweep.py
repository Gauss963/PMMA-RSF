from dataclasses import replace
import math
import tomllib

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.generate_ts0320_rsf_prestress_sweep_cases import (
    CASE_COUNT,
    DYNAMIC_SHEAR_INCREMENT_MM,
    FIRST_RUN,
    NORMALIZED_PRESTRESS_MIN,
    NORMALIZED_PRESTRESS_STEP,
    TEMPLATE,
    case_path,
    normalized_prestress,
    prestress_displacement,
    render_case,
)
from tatva.friction import regularized_rate_state_strength
from tatva.pmma.config import load_case_config
from tatva.pmma.dynamics import build_case_model, run_simulation_dumped
from tatva.pmma.estimate import estimate_case_size
from tatva.pmma.runner import make_case, make_run_config


def test_prestress_sweep_has_sixteen_monotonic_single_parameter_cases():
    template_text = TEMPLATE.read_text(encoding="utf-8")
    prestress_values = []
    total_estimated_dump = 0

    assert CASE_COUNT == 16
    for index in range(1, CASE_COUNT + 1):
        path = case_path(index)
        assert path.read_text(encoding="utf-8") == render_case(
            template_text, index
        )
        config = load_case_config(path)
        expected_pi = NORMALIZED_PRESTRESS_MIN + (
            index - 1
        ) * NORMALIZED_PRESTRESS_STEP
        prestress = config.loading.prestress_shear_displacement
        assert config.name == (
            f"pmma-rsf-{FIRST_RUN + index - 1:04d}-prestress-{index:02d}of16"
        )
        assert config.rsf.initial_state_mode == "traction-consistent-handoff"
        assert config.rsf.target_normalized_prestress == pytest.approx(
            expected_pi
        )
        assert prestress == pytest.approx(prestress_displacement(
            tomllib.loads(template_text), index
        ))
        assert (
            config.loading.shear_displacement_final - prestress
        ) == pytest.approx(DYNAMIC_SHEAR_INCREMENT_MM)
        peak_speed = math.pi * DYNAMIC_SHEAR_INCREMENT_MM / (
            2.0 * config.loading.shear_ramp_time
        )
        assert peak_speed == pytest.approx(200.0)
        assert config.loading.normal_relaxation_start_time == pytest.approx(
            0.020
        )
        assert config.loading.quasistatic_shear_start_time == pytest.approx(
            0.020
        )
        assert config.loading.quasistatic_shear_ramp_time == pytest.approx(
            0.010
        )
        assert config.loading.stop_velocity == pytest.approx(500.0)
        assert config.loading.stop_min_y == pytest.approx(5.0)
        assert config.loading.stop_max_y == pytest.approx(25.0)
        assert config.loading.stop_coverage_fraction is None
        estimate = estimate_case_size(config)
        total_estimated_dump += int(
            estimate["estimated_uncompressed_bytes"]
            * config.output.estimated_compression_ratio
        )
        prestress_values.append(prestress)

    assert prestress_values == sorted(prestress_values)
    assert normalized_prestress(1) == pytest.approx(0.450)
    assert normalized_prestress(16) == pytest.approx(0.825)
    assert total_estimated_dump < 1_400_000_000_000
    with pytest.raises(ValueError, match="1 through 16"):
        case_path(17)


def test_prestress_schedule_relaxes_only_after_normal_ramp():
    config = load_case_config(case_path(1))
    coarse = replace(
        config,
        numerics=replace(config.numerics, mesh_size=100.0, time_step=None),
        loading=replace(
            config.loading,
            stop_min_y=0.0,
            stop_max_y=100.0,
        ),
    )
    model = build_case_model(make_case(coarse), make_run_config(coarse))
    relaxation = np.asarray(model["normal_relaxation_pressure"])
    start = int(math.floor(0.020 / model["dt"]))

    assert np.all(relaxation[:start] == 0.0)
    assert np.all(relaxation[start:] == 1.0)
    assert model["quasistatic_shear_target"] == pytest.approx(
        coarse.loading.prestress_shear_displacement
    )
    assert np.asarray(model["shear_displacement_pressure"])[-1] == (
        pytest.approx(coarse.loading.prestress_shear_displacement)
    )


def test_traction_consistent_handoff_recovers_prepared_strength_and_stops_same_step(
    tmp_path,
):
    config = load_case_config(case_path(1))
    config = replace(
        config,
        numerics=replace(
            config.numerics,
            mesh_size=100.0,
            cfl=0.2,
            time_step=None,
        ),
        loading=replace(
            config.loading,
            normal_phase_time=4.0e-5,
            normal_ramp_time=1.0e-5,
            normal_relaxation_time=2.0e-6,
            normal_relaxation_start_time=1.0e-5,
            prestress_shear_displacement=2.0e-3,
            quasistatic_shear_start_time=1.0e-5,
            quasistatic_shear_ramp_time=1.0e-5,
            shear_displacement_final=3.0e-3,
            shear_phase_time=2.0e-5,
            shear_ramp_time=1.0e-5,
            stop_slip=1.0e-12,
            stop_velocity=1.0e-8,
            stop_min_y=0.0,
            stop_max_y=100.0,
            stop_coverage_fraction=None,
        ),
    )
    output = tmp_path / "handoff.h5"
    result = run_simulation_dumped(
        make_case(config),
        make_run_config(config),
        output,
        frames_per_phase=1,
        shear_frames_per_phase=4,
        interface_frames_per_phase=2,
        shear_interface_frames_per_phase=12,
        include_initial_frame=False,
        store_bulk_strain=False,
        store_bulk_velocity=False,
    )

    handoff = result["summary"]["rsf_state_handoff"]
    assert handoff is not None
    assert handoff["mean_shear_traction"] > 0.0
    assert handoff["mean_normal_traction"] > 0.0
    assert handoff["min_state"] > 0.0
    assert math.isfinite(handoff["max_state"])

    with h5py.File(output, "r") as h5:
        high = h5["interface_high_rate"]
        phase = np.asarray(high["phase_id"][:])
        normal_rows = np.flatnonzero(phase == 1)
        shear_rows = np.flatnonzero(phase == 2)
        assert np.all(high["cumulative_slip"][normal_rows] == 0.0)

        state = np.asarray(h5["interface/rsf_handoff_state_profile"][:])
        strength = np.asarray(high["friction_strength"][normal_rows[-1]])
        coefficient = np.asarray(
            high["friction_coefficient"][normal_rows[-1]]
        )
        state_floor = np.finfo(np.float32).tiny * 10.0
        state_ceiling = np.finfo(np.float32).max / 10.0
        active = (
            (strength > 0.0)
            & (coefficient > 0.0)
            & (state > state_floor)
            & (state < state_ceiling)
        )
        normal = strength[active] / coefficient[active]
        recovered = regularized_rate_state_strength(
            jnp.full(normal.shape, config.rsf.initial_steady_velocity),
            jnp.asarray(normal),
            jnp.asarray(state[active]),
            reference_friction=jnp.asarray(
                h5["interface/rsf_reference_friction_profile"][:][active]
            ),
            direct_effect=jnp.asarray(
                h5["interface/rsf_direct_effect_profile"][:][active]
            ),
            state_effect=jnp.asarray(
                h5["interface/rsf_state_effect_profile"][:][active]
            ),
            reference_velocity=jnp.asarray(
                h5["interface/rsf_reference_velocity_profile"][:][active]
            ),
            characteristic_slip=jnp.asarray(
                h5["interface/rsf_characteristic_slip_profile"][:][active]
            ),
        )
        np.testing.assert_allclose(
            np.asarray(recovered), strength[active], rtol=2.0e-4, atol=1.0e-6
        )

        columns = [value.decode() for value in high["history_columns"][:]]
        stopped = np.asarray(high["history"][:, columns.index(
            "shear_loading_stopped"
        )]) > 0.5
        y = np.asarray(high["contact_line_y"][:])
        nucleus = (y >= config.loading.stop_min_y) & (
            y <= config.loading.stop_max_y
        )
        trigger = np.any(
            np.asarray(high["slip_rate"][:, nucleus])
            >= config.loading.stop_velocity,
            axis=1,
        ) & (phase == 2)
        first_trigger = np.flatnonzero(trigger)[0]
        first_stopped = np.flatnonzero(stopped & (phase == 2))[0]
        assert first_stopped == first_trigger
        displacement = np.asarray(
            high["history"][:, columns.index("applied_shear_displacement")]
        )
        np.testing.assert_allclose(
            displacement[first_stopped:], displacement[first_stopped]
        )
