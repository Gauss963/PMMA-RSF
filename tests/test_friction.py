import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tatva.friction import (
    linear_slip_weakening_coefficient,
    linear_slip_weakening_strength,
    project_regularized_rate_state_velocity,
    regularized_rate_state_initial_state,
    regularized_rate_state_strength,
    steady_state_age,
    update_ageing_state,
    velocity_weakening_strengthening_coefficient,
)


def test_linear_slip_weakening_coefficient_reaches_kinetic_value():
    slip = jnp.array([0.0, 0.2, 0.4, 0.8, -0.2])
    coefficient = linear_slip_weakening_coefficient(
        slip,
        static_friction=0.8,
        kinetic_friction=0.6,
        characteristic_slip=0.4,
    )
    assert np.allclose(np.asarray(coefficient), [0.8, 0.7, 0.6, 0.6, 0.7])


def test_linear_slip_weakening_supports_spatial_profiles():
    coefficient = linear_slip_weakening_coefficient(
        jnp.array([0.1, 0.2]),
        static_friction=jnp.array([0.8, 0.9]),
        kinetic_friction=jnp.array([0.6, 0.5]),
        characteristic_slip=jnp.array([0.2, 0.4]),
    )
    assert np.allclose(np.asarray(coefficient), [0.7, 0.7])


def test_linear_slip_weakening_strength_is_jittable():
    strength = jax.jit(
        lambda slip, normal_stress: linear_slip_weakening_strength(
            slip,
            normal_stress,
            static_friction=0.8,
            kinetic_friction=0.6,
            characteristic_slip=0.4,
        )
    )(jnp.array([0.0, 0.2, 0.4]), jnp.array([10.0, 10.0, 10.0]))
    assert np.allclose(np.asarray(strength), [8.0, 7.0, 6.0])


def test_ageing_state_grows_linearly_while_sticking():
    state = jnp.array([0.2, 0.7])
    updated = update_ageing_state(state, jnp.zeros(2), 0.05, 0.4)
    assert np.allclose(np.asarray(updated), [0.25, 0.75])


def test_ageing_state_matches_constant_velocity_solution():
    state = jnp.array([0.2, 0.7])
    velocity = jnp.array([2.0, 0.5])
    dt = 0.05
    characteristic_slip = 0.4
    updated = update_ageing_state(state, velocity, dt, characteristic_slip)
    steady = characteristic_slip / np.asarray(velocity)
    expected = steady + (np.asarray(state) - steady) * np.exp(
        -np.asarray(velocity) * dt / characteristic_slip
    )
    assert np.allclose(np.asarray(updated), expected, rtol=2e-7, atol=1e-7)
    assert np.all(np.asarray(updated) > 0.0)


def test_vws_coefficient_matches_rezakhani_equation():
    velocity = jnp.array([0.0, 2.0e-3])
    state = jnp.array([3.3e-4, 8.0e-4])
    coefficient = velocity_weakening_strengthening_coefficient(
        velocity,
        state,
        reference_friction=0.285,
        direct_effect=0.005,
        state_effect=0.0214,
        reference_velocity=1.0e-4,
        reference_state=3.3e-4,
    )
    expected = (
        0.285
        + 0.005 * np.log1p(np.asarray(velocity) / 1.0e-4)
        + 0.0214 * np.log1p(np.asarray(state) / 3.3e-4)
    )
    assert np.allclose(np.asarray(coefficient), expected, rtol=2e-7)
    assert np.isfinite(np.asarray(coefficient)[0])


def test_rezakhani_parameters_are_velocity_weakening_at_steady_state():
    velocity = jnp.array([1.0e-4, 1.0e-3])
    state = steady_state_age(velocity, 5.0e-4)
    coefficient = velocity_weakening_strengthening_coefficient(
        velocity,
        state,
        reference_friction=0.285,
        direct_effect=0.005,
        state_effect=0.0214,
        reference_velocity=1.0e-4,
        reference_state=3.3e-4,
    )
    assert np.asarray(coefficient)[1] < np.asarray(coefficient)[0]


def test_rate_state_updates_are_jittable():
    update = jax.jit(
        lambda state, velocity: velocity_weakening_strengthening_coefficient(
            velocity,
            update_ageing_state(state, velocity, 1.0e-5, 5.0e-4),
            reference_friction=0.285,
            direct_effect=0.005,
            state_effect=0.0214,
            reference_velocity=1.0e-4,
            reference_state=3.3e-4,
        )
    )
    result = update(jnp.array([3.3e-4]), jnp.array([2.0e-3]))
    assert np.isfinite(np.asarray(result)).all()


def test_steady_state_age_rejects_zero_velocity_by_returning_infinity():
    result = steady_state_age(jnp.array([2.0, 0.0]), 0.4)
    assert np.asarray(result)[0] == pytest.approx(0.2)
    assert np.isinf(np.asarray(result)[1])


def test_tpv101_initial_state_matches_prescribed_strength():
    velocity = jnp.asarray(1.0e-12)
    normal_stress = jnp.asarray(120.0e6)
    shear_stress = jnp.asarray(75.0e6)
    direct_effect = jnp.asarray(0.008)
    state = regularized_rate_state_initial_state(
        velocity,
        shear_stress,
        normal_stress,
        reference_friction=0.6,
        direct_effect=direct_effect,
        state_effect=0.012,
        reference_velocity=1.0e-6,
        characteristic_slip=0.02,
    )
    recovered_strength = regularized_rate_state_strength(
        velocity,
        normal_stress,
        state,
        reference_friction=0.6,
        direct_effect=direct_effect,
        state_effect=0.012,
        reference_velocity=1.0e-6,
        characteristic_slip=0.02,
    )
    assert float(state) == pytest.approx(1.606238999213454e9, rel=2e-6)
    assert float(recovered_strength) == pytest.approx(75.0e6, rel=2e-6)


def test_tpv101_regularized_strength_is_finite_across_extreme_rates():
    velocity = jnp.asarray([1.0e-12, 1.0e-6, 1.0, 1.0e3])
    state = jnp.full(velocity.shape, 1.606238999213454e9)
    strength = jax.jit(
        lambda rate, theta: regularized_rate_state_strength(
            rate,
            jnp.asarray(120.0e6),
            theta,
            reference_friction=0.6,
            direct_effect=jnp.asarray(0.008),
            state_effect=0.012,
            reference_velocity=1.0e-6,
            characteristic_slip=0.02,
        )
    )(velocity, state)
    assert np.isfinite(np.asarray(strength)).all()
    assert (np.asarray(strength) > 0.0).all()


def test_regularized_velocity_projection_satisfies_tpv_residual():
    free_velocity = jnp.asarray([0.0, 2.0, -4.0])
    normal_stress = jnp.full((3,), 16.0)
    state = jnp.full((3,), 0.00629795 / 1.0e-4)
    impulse_factor = jnp.asarray([1.0e-3, 2.0e-3, 3.0e-3])

    corrected, strength = project_regularized_rate_state_velocity(
        free_velocity,
        normal_stress,
        state,
        impulse_factor,
        reference_friction=0.8,
        direct_effect=0.005,
        state_effect=0.025819400653936703,
        reference_velocity=1.0e-4,
        characteristic_slip=0.00629795,
    )

    residual = np.abs(np.asarray(corrected)) + np.asarray(impulse_factor) * np.asarray(
        strength
    ) - np.abs(np.asarray(free_velocity))
    assert np.allclose(residual, 0.0, rtol=2e-6, atol=2e-6)
    assert np.sign(np.asarray(corrected)[1:]).tolist() == [1.0, -1.0]
