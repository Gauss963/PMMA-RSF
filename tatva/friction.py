"""Friction constitutive updates for explicit Tatva simulations."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def linear_slip_weakening_coefficient(
    slip: jax.Array,
    *,
    static_friction: float | jax.Array,
    kinetic_friction: float | jax.Array,
    characteristic_slip: float | jax.Array,
) -> jax.Array:
    """Evaluate a linear slip-weakening friction coefficient.

    The coefficient decreases linearly from ``static_friction`` at zero slip
    to ``kinetic_friction`` at ``|slip| = characteristic_slip`` and remains at
    the kinetic value thereafter. Scalar or spatially varying parameters are
    accepted and follow JAX broadcasting rules.
    """
    weakening_fraction = jnp.clip(
        jnp.abs(slip) / characteristic_slip,
        0.0,
        1.0,
    )
    return static_friction - (
        static_friction - kinetic_friction
    ) * weakening_fraction


def linear_slip_weakening_strength(
    slip: jax.Array,
    normal_stress: jax.Array,
    *,
    static_friction: float | jax.Array,
    kinetic_friction: float | jax.Array,
    characteristic_slip: float | jax.Array,
) -> jax.Array:
    """Evaluate linear slip-weakening strength for compressive normal stress.

    ``normal_stress`` is positive in compression. Contact opening and traction
    direction are deliberately left to the caller's contact formulation.
    """
    coefficient = linear_slip_weakening_coefficient(
        slip,
        static_friction=static_friction,
        kinetic_friction=kinetic_friction,
        characteristic_slip=characteristic_slip,
    )
    return coefficient * normal_stress


def _asinh_exp(log_value: jax.Array) -> jax.Array:
    """Evaluate ``asinh(exp(log_value))`` without overflowing."""
    positive_log = jnp.maximum(log_value, 0.0)
    negative_log = jnp.minimum(log_value, 0.0)
    positive_branch = positive_log + jnp.log1p(
        jnp.sqrt(1.0 + jnp.exp(-2.0 * positive_log))
    )
    negative_branch = jnp.arcsinh(jnp.exp(negative_log))
    return jnp.where(log_value >= 0.0, positive_branch, negative_branch)


def update_ageing_state(
    state: jax.Array,
    slip_rate: jax.Array,
    dt: float | jax.Array,
    characteristic_slip: float | jax.Array,
) -> jax.Array:
    """Advance the Dieterich ageing law exactly over one constant-rate step.

    The state variable follows ``state_dot = 1 - |V| state / D``.  Using the
    exact constant-rate update instead of forward Euler keeps the state
    positive for time steps that span a large fraction of ``D / |V|``.
    """
    velocity = jnp.abs(slip_rate)
    step_slip = velocity * dt / characteristic_slip
    decay = jnp.exp(-step_slip)
    safe_step_slip = jnp.maximum(step_slip, jnp.finfo(state.dtype).eps)
    age_increment = dt * (-jnp.expm1(-safe_step_slip)) / safe_step_slip
    sliding_state = state * decay + age_increment
    return jnp.where(velocity > 0.0, sliding_state, state + dt)


def velocity_weakening_strengthening_coefficient(
    slip_rate: jax.Array,
    state: jax.Array,
    *,
    reference_friction: float | jax.Array,
    direct_effect: float | jax.Array,
    state_effect: float | jax.Array,
    reference_velocity: float | jax.Array,
    reference_state: float | jax.Array,
) -> jax.Array:
    """Evaluate the finite-at-rest VWS rate-and-state friction coefficient.

    This is the velocity-weakening/strengthening form used by Rezakhani et al.::

        f = f0 + a log(1 + |V| / V*) + b log(1 + state / state*)
    """
    velocity = jnp.abs(slip_rate)
    positive_state = jnp.maximum(state, 0.0)
    return (
        reference_friction
        + direct_effect * jnp.log1p(velocity / reference_velocity)
        + state_effect * jnp.log1p(positive_state / reference_state)
    )


def regularized_rate_state_strength(
    slip_rate: jax.Array,
    normal_stress: jax.Array,
    state: jax.Array,
    *,
    reference_friction: float | jax.Array,
    direct_effect: float | jax.Array,
    state_effect: float | jax.Array,
    reference_velocity: float | jax.Array,
    characteristic_slip: float | jax.Array,
) -> jax.Array:
    """Evaluate the SCEC TPV101 regularized rate-and-state strength.

    ``normal_stress`` is positive in compression. All dimensional arguments
    must use one consistent unit system.
    """
    velocity = jnp.abs(slip_rate)
    positive_state = jnp.maximum(state, jnp.finfo(state.dtype).tiny)
    log_argument = (
        jnp.log(jnp.maximum(velocity, jnp.finfo(velocity.dtype).tiny))
        - jnp.log(2.0 * reference_velocity)
        + (
            reference_friction
            + state_effect
            * jnp.log(reference_velocity * positive_state / characteristic_slip)
        )
        / direct_effect
    )
    strength = direct_effect * normal_stress * _asinh_exp(log_argument)
    return jnp.where(velocity > 0.0, strength, 0.0)


def regularized_rate_state_initial_state(
    slip_rate: jax.Array,
    shear_stress: jax.Array,
    normal_stress: jax.Array,
    *,
    reference_friction: float | jax.Array,
    direct_effect: jax.Array,
    state_effect: float | jax.Array,
    reference_velocity: float | jax.Array,
    characteristic_slip: float | jax.Array,
) -> jax.Array:
    """Return the TPV101 state that exactly supports a prescribed traction.

    This is the analytic inverse of :func:`regularized_rate_state_strength`.
    """
    stress_ratio = jnp.abs(shear_stress) / (direct_effect * normal_stress)
    log_two_sinh = stress_ratio + jnp.log1p(-jnp.exp(-2.0 * stress_ratio))
    log_state = (
        jnp.log(characteristic_slip / reference_velocity)
        + (
            direct_effect * log_two_sinh
            - reference_friction
            - direct_effect * jnp.log(jnp.abs(slip_rate) / reference_velocity)
        )
        / state_effect
    )
    return jnp.exp(log_state)


def project_regularized_rate_state_velocity(
    free_relative_velocity: jax.Array,
    normal_stress: jax.Array,
    state: jax.Array,
    relative_impulse_factor: jax.Array,
    *,
    reference_friction: float | jax.Array,
    direct_effect: float | jax.Array,
    state_effect: float | jax.Array,
    reference_velocity: float | jax.Array,
    characteristic_slip: float | jax.Array,
    iterations: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Apply the implicit RSF velocity projection used by TPV101/102.

    ``relative_impulse_factor`` is ``dt * interface_weight * (1/m+ + 1/m-)``.
    Bisection solves ``V + factor * tau(V, theta) = V_free`` independently at
    every contact node and returns signed relative velocity plus strength.
    """
    free_speed = jnp.abs(free_relative_velocity)

    def residual(speed: jax.Array) -> jax.Array:
        strength = regularized_rate_state_strength(
            speed,
            normal_stress,
            state,
            reference_friction=reference_friction,
            direct_effect=direct_effect,
            state_effect=state_effect,
            reference_velocity=reference_velocity,
            characteristic_slip=characteristic_slip,
        )
        return speed + relative_impulse_factor * strength - free_speed

    lower = jnp.zeros_like(free_speed)
    upper = free_speed

    def bisect(_iteration: int, bounds: tuple[jax.Array, jax.Array]):
        low, high = bounds
        midpoint = 0.5 * (low + high)
        move_low = residual(midpoint) < 0.0
        return jnp.where(move_low, midpoint, low), jnp.where(
            move_low, high, midpoint
        )

    lower, upper = jax.lax.fori_loop(0, iterations, bisect, (lower, upper))
    corrected_speed = 0.5 * (lower + upper)
    strength = regularized_rate_state_strength(
        corrected_speed,
        normal_stress,
        state,
        reference_friction=reference_friction,
        direct_effect=direct_effect,
        state_effect=state_effect,
        reference_velocity=reference_velocity,
        characteristic_slip=characteristic_slip,
    )
    direction = jnp.where(free_relative_velocity >= 0.0, 1.0, -1.0)
    return direction * corrected_speed, strength


def steady_state_age(
    slip_rate: jax.Array,
    characteristic_slip: float | jax.Array,
) -> jax.Array:
    """Return the steady-state age ``D / |V|`` for non-zero slip rate."""
    velocity = jnp.abs(slip_rate)
    return characteristic_slip / velocity
