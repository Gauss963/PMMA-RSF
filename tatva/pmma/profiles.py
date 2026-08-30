"""Spatial RSF profiles and LSW-to-RSF calibration helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


PROFILE_KEYS = (
    "reference_friction",
    "direct_effect",
    "state_effect",
    "characteristic_slip",
    "initial_state",
)


def regularized_steady_friction(
    *,
    velocity: float,
    reference_friction: float,
    direct_effect: float,
    state_effect: float,
    reference_velocity: float,
) -> float:
    """Evaluate standard regularized RSF at the ageing-law steady state."""
    if velocity <= 0.0:
        raise ValueError("velocity must be positive.")
    log_argument = (
        math.log(velocity / (2.0 * reference_velocity))
        + (
            reference_friction
            + state_effect * math.log(reference_velocity / velocity)
        )
        / direct_effect
    )
    if log_argument >= 0.0:
        asinh_exp = log_argument + math.log1p(
            math.sqrt(1.0 + math.exp(-2.0 * log_argument))
        )
    else:
        asinh_exp = math.asinh(math.exp(log_argument))
    return direct_effect * asinh_exp


def calibrate_state_effect(
    *,
    initial_friction: float,
    dynamic_friction: float,
    direct_effect: float,
    characteristic_slip: float,
    reference_velocity: float,
    reference_state: float,
    initial_steady_velocity: float,
    dynamic_velocity: float,
) -> float:
    """Find ``b`` from the standard steady-state RSF friction drop.

    With the ageing-law steady state ``theta=Dc/V``, standard RSF gives
    ``mu(V2)-mu(V1)=(a-b) ln(V2/V1)``. ``Dc`` and reference-state arguments are
    accepted to keep the calibration call self-documenting.
    """
    del characteristic_slip, reference_state
    if not math.isclose(
        initial_steady_velocity, reference_velocity, rel_tol=1.0e-12
    ):
        raise ValueError(
            "LSW-to-RSF calibration currently requires initial velocity V1=V0."
        )
    log_ratio = math.log(dynamic_velocity / initial_steady_velocity)
    if log_ratio <= 0.0:
        raise ValueError("dynamic_velocity must exceed initial_steady_velocity.")
    state_effect = direct_effect + (
        initial_friction - dynamic_friction
    ) / log_ratio
    if state_effect < 0.0:
        raise ValueError("The requested calibration produces a negative RSF b value.")
    return state_effect


def _half_cosine(progress: np.ndarray) -> np.ndarray:
    clipped = np.clip(progress, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * clipped))


def _blend(
    y: np.ndarray,
    start: float,
    end: float,
    left: float,
    right: float,
) -> np.ndarray:
    if end <= start:
        return np.where(y >= end, right, left)
    weight = _half_cosine((y - start) / (end - start))
    return left + (right - left) * weight


def _zone_value(zone: dict[str, Any], long_name: str, short_name: str) -> float:
    if long_name in zone:
        return float(zone[long_name])
    return float(zone[short_name])


def build_rate_state_profile(
    y_coordinates: np.ndarray,
    specification: dict[str, Any],
) -> dict[str, np.ndarray | dict[str, float]]:
    """Build loading/middle/leading RSF fields with half-cosine transitions."""
    y = np.asarray(y_coordinates, dtype=np.float64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_coordinates must be a non-empty one-dimensional array.")
    y_min = float(y.min())
    y_max = float(y.max())
    actual_length = y_max - y_min
    profile_length = float(specification.get("profile_length", actual_length))
    if profile_length + 1.0e-9 < actual_length:
        raise ValueError(
            "RSF profile_length cannot be shorter than the supplied coordinates."
        )
    profile_y_max = y_min + profile_length
    loading_length = float(specification["loading_length"])
    leading_length = float(specification["leading_length"])
    transition_length = float(specification["transition_length"])
    loading_transition_length = float(
        specification.get("loading_transition_length", transition_length)
    )
    leading_transition_length = float(
        specification.get("leading_transition_length", transition_length)
    )
    if min(
        loading_length,
        leading_length,
        loading_transition_length,
        leading_transition_length,
    ) < 0.0:
        raise ValueError("RSF zone and transition lengths must be non-negative.")
    if (
        loading_length
        + leading_length
        + loading_transition_length
        + leading_transition_length
        >= profile_length
    ):
        raise ValueError("RSF end zones and transitions leave no middle segment.")

    initial_friction = float(specification["initial_friction"])
    zones = {
        name: dict(specification[name])
        for name in ("loading", "middle", "leading")
    }
    for zone in zones.values():
        if "reference_friction" not in zone and "f0" not in zone:
            zone["reference_friction"] = initial_friction
    fields: dict[str, np.ndarray] = {}
    aliases = {
        "direct_effect": "a",
        "state_effect": "b",
        "characteristic_slip": "dc",
        "reference_friction": "f0",
    }
    loading_transition_start = y_min + loading_length
    loading_transition_end = (
        loading_transition_start + loading_transition_length
    )
    leading_transition_end = profile_y_max - leading_length
    leading_transition_start = (
        leading_transition_end - leading_transition_length
    )
    for field_name, alias in aliases.items():
        loading_value = _zone_value(zones["loading"], field_name, alias)
        middle_value = _zone_value(zones["middle"], field_name, alias)
        leading_value = _zone_value(zones["leading"], field_name, alias)
        values = np.full(y.shape, middle_value, dtype=np.float64)
        values[y <= loading_transition_start] = loading_value
        loading_mask = (y > loading_transition_start) & (y < loading_transition_end)
        values[loading_mask] = _blend(
            y[loading_mask],
            loading_transition_start,
            loading_transition_end,
            loading_value,
            middle_value,
        )
        leading_mask = (y > leading_transition_start) & (y < leading_transition_end)
        values[leading_mask] = _blend(
            y[leading_mask],
            leading_transition_start,
            leading_transition_end,
            middle_value,
            leading_value,
        )
        values[y >= leading_transition_end] = leading_value
        fields[field_name] = values

    initial_velocity = float(specification["initial_steady_velocity"])
    reference_velocity = float(specification["reference_velocity"])
    reference_state = float(specification["reference_state"])
    initial_state = fields["characteristic_slip"] / initial_velocity
    fields["initial_state"] = initial_state
    fields["reference_velocity"] = np.full(y.shape, reference_velocity)
    fields["reference_state"] = np.full(y.shape, reference_state)
    fields["metadata"] = {
        "y_min": y_min,
        "y_max": y_max,
        "profile_y_max": profile_y_max,
        "profile_length": profile_length,
        "loading_plateau_end": loading_transition_start,
        "loading_transition_end": loading_transition_end,
        "leading_transition_start": leading_transition_start,
        "leading_plateau_start": leading_transition_end,
    }
    return fields
