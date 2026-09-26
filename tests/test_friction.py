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


def test_tpv101_initial_state_inverse_is_stable_for_small_stress_ratio():
    velocity = jnp.asarray(1.0e-4, dtype=jnp.float32)
    normal_stress = jnp.asarray(16.0, dtype=jnp.float32)
    shear_stress = jnp.asarray(1.0e-5, dtype=jnp.float32)
    direct_effect = jnp.asarray(0.005, dtype=jnp.float32)
    state = regularized_rate_state_initial_state(
        velocity,
        shear_stress,
        normal_stress,
        reference_friction=0.8,
        direct_effect=direct_effect,
        state_effect=0.029123527228205267,
        reference_velocity=1.0e-4,
        characteristic_slip=0.00042852936846183833,
    )
    recovered_strength = regularized_rate_state_strength(
        velocity,
        normal_stress,
        state,
        reference_friction=0.8,
        direct_effect=direct_effect,
        state_effect=0.029123527228205267,
        reference_velocity=1.0e-4,
        characteristic_slip=0.00042852936846183833,
    )

    assert np.isfinite(float(state))
    assert float(state) > 0.0
    assert float(recovered_strength) == pytest.approx(1.0e-5, rel=2.0e-4)


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


def test_near_sticking_projection_retains_traction_even_if_speed_underflows():
    a = jnp.asarray([0.0025, 0.005, 0.01, 0.02], dtype=jnp.float32)
    factor = jnp.asarray(69.68641115, dtype=jnp.float32)
    free = jnp.full(a.shape, factor * 16.0 * 0.1)
    corrected, traction = jax.jit(project_regularized_rate_state_velocity)(
        free, jnp.asarray(16.0), jnp.full(a.shape, 4.285293684618383), factor,
        reference_friction=0.8, direct_effect=a, state_effect=a + 0.024123527228205266,
        reference_velocity=1e-4, characteristic_slip=0.00042852936846183833,
    )
    np.testing.assert_allclose(traction, 1.6, rtol=2e-5)
    np.testing.assert_allclose(corrected + factor * traction, free, rtol=2e-5)
    assert np.max(np.abs(corrected)) < 1e-18


def test_log_projection_agrees_with_independently_prescribed_roots():
    # Manufacture roots across near-rest, dynamic and SCEC-scale velocities.
    speed = jnp.asarray([1e-35, 1e-20, 1e-6, 0.1, 1.0, 1000.0], dtype=jnp.float32)
    for f0, a, b, v0, dc, theta, sigma in (
        (0.8, 0.005, 0.0291235272, 1e-4, 0.000428529, 4.28529, 16.0),
        (0.6, 0.008, 0.012, 1e-6, 0.02, 1.606238999e9, 120e6),
    ):
        kw = dict(reference_friction=f0, direct_effect=a, state_effect=b,
                  reference_velocity=v0, characteristic_slip=dc)
        state = jnp.full(speed.shape, theta)
        stress = jnp.asarray(sigma)
        tau = regularized_rate_state_strength(speed, stress, state, **kw)
        factor = jnp.asarray(0.002 if sigma > 1e6 else 69.6864)
        free = speed + factor * tau
        projected, strength = project_regularized_rate_state_velocity(free, stress, state, factor, **kw)
        np.testing.assert_allclose(projected, speed, rtol=2e-4, atol=1e-37)
        np.testing.assert_allclose(strength, tau, rtol=2e-5)
        np.testing.assert_allclose(projected + factor * strength, free, rtol=2e-5)


def test_log_projection_open_contact_and_disabled_impulse():
    free = jnp.asarray([0., -4., 2.], dtype=jnp.float32)
    normal = jnp.asarray([16., 0., 16.], dtype=jnp.float32)
    state = jnp.ones_like(free)
    v, tau = project_regularized_rate_state_velocity(
        free, normal, state, jnp.asarray([1., 1., 0.]),
        reference_friction=.8, direct_effect=.005, state_effect=.02,
        reference_velocity=1e-4, characteristic_slip=.001,
    )
    np.testing.assert_array_equal(v, free)
    assert tau[0] == 0 and tau[1] == 0
    assert np.isfinite(tau).all()
