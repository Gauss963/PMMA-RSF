"""Derive LSW and rate-and-state parameters from a cohesive-zone anchor.

The cohesive-zone model fixes the process-zone size. Everything else -- the
fracture energy, the LSW characteristic slip, the RSF a/b pairs and their
characteristic slip -- follows from it at the current friction coefficients.
D_c is therefore a derived quantity: carrying an old D_c across a change of
mu_k silently rescales the fracture energy.
"""

import numpy as np

MPa = 1e6
GPa = 1e9
mm = 1e-3

# PMMA properties and loading, matching the [material] and [loading] blocks of
# the PMMA-RSF cases.
E = 7662.0 * MPa
nu = 0.2
mu_s = 0.8
mu_k = 0.45
sigma_n = 16 * MPa

p_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
p_mu = E / (2 * (1 + nu))
elastic_factor = p_mu * (p_lambda + p_mu) / (p_lambda + 2 * p_mu)

tau_p = mu_s * sigma_n
tau_r = mu_k * sigma_n
delta_tau = tau_p - tau_r


# ---------------------------------------------------------------------------
# Linear slip weakening
# ---------------------------------------------------------------------------
def critical_nucleation_length(fracture_energy):
    return (8 / np.pi) * elastic_factor * fracture_energy / delta_tau**2


def quasistatic_cohesive_zone_size(fracture_energy):
    # Kammer-McLaskey Eq. (A.12) in the f_II -> 1 limit.
    return (9 * np.pi / 8) * elastic_factor * fracture_energy / delta_tau**2


def cohesive_slip_for_cohesive_zone_size(cohesive_zone_size):
    """Invert the cohesive-zone size for D_c.

    Substituting Gamma = delta_tau * D_c / 2 into
    `quasistatic_cohesive_zone_size` collapses it to
    X_c = (9 pi / 16) * elastic_factor * D_c / delta_tau, so the inversion
    carries 16 / (9 pi).
    """
    return cohesive_zone_size * (16 / (9 * np.pi)) * delta_tau / elastic_factor


CZM_COHESIVE_ZONE_SIZE = 5 * mm
D_c = cohesive_slip_for_cohesive_zone_size(CZM_COHESIVE_ZONE_SIZE)
G = 0.5 * delta_tau * D_c
X_c = quasistatic_cohesive_zone_size(G)
L_c = critical_nucleation_length(G)


# ---------------------------------------------------------------------------
# Rate and state
# ---------------------------------------------------------------------------
# Reference values shared by every Velocity-weakening RSF case.
V_ref = 1.0e-4        # reference_velocity
theta_ref = 3.3e-4    # reference_state
V_init = 1.0e-4       # initial_steady_velocity
V_dyn = 2000.0        # dynamic_calibration_velocity

# a, b and a-b are dimensionless and set by the velocity ratio alone, so the
# unit of V_init / V_dyn cancels. The cases carry both in mm/s.
log_velocity_ratio = np.log(V_dyn / V_init)

# Rubin & Ampuero (2005) write their nucleation lengths with the mode-II
# plane-strain modulus mu/(1-nu). That is exactly twice `elastic_factor`, which
# carries the extra 1/2 the LSW expressions above expect.
mode_II_modulus = p_mu / (1 - nu)


def rsf_state_effect(direct_effect, dynamic_friction=mu_k):
    """Return the b that puts the steady state at `dynamic_friction` for V_dyn.

    Mirrors ``calibrate_state_effect`` in ``tatva/pmma/profiles.py``: with the
    ageing-law steady state theta = D_c / V, standard RSF gives
    ``mu(V2) - mu(V1) = (a - b) ln(V2 / V1)``, and the cases fix V1 = V_init =
    V_ref so that mu(V1) is the initial friction ``mu_s``.
    """
    state_effect = direct_effect + (mu_s - dynamic_friction) / log_velocity_ratio
    if state_effect < 0.0:
        raise ValueError(
            f"Calibration gives a negative b ({state_effect:.6g}); the requested "
            f"friction drop is too small for a = {direct_effect:.6g}."
        )
    return state_effect


def rsf_steady_friction(direct_effect, state_effect, velocity):
    """Steady-state friction coefficient at `velocity`."""
    return mu_s + (direct_effect - state_effect) * np.log(velocity / V_init)


def rsf_peak_friction(direct_effect):
    """Friction the instant the rate steps to V_dyn, before the state relaxes.

    The direct effect overshoots above mu_s; only afterwards does the state
    term pull the strength down to the steady value.
    """
    return mu_s + direct_effect * log_velocity_ratio


def rsf_stress_drop(state_effect):
    """Peak-to-residual shear stress drop of the RSF weakening step."""
    return sigma_n * state_effect * log_velocity_ratio


def _breakdown_integral(overshoot):
    """Return the dimensionless breakdown work of one ageing-law step.

    Holding the slip rate at V while the state relaxes gives
    ``theta(d) = Dc/V + (theta0 - Dc/V) exp(-d/Dc)``, so the strength above its
    steady value is ``sigma_n b ln(1 + (W - 1) exp(-d/Dc))`` with
    ``W = theta0 V / Dc``. Integrating over slip leaves ``-Li2(-(W - 1))``,
    expanded here for large argument. The neglected term is O(1/W^2), which is
    below 1e-15 at the velocity ratio the cases use and still only 2e-6 at
    W = 100.
    """
    amplitude = overshoot - 1.0
    if amplitude <= 0.0:
        raise ValueError("overshoot must exceed 1 for a weakening step.")
    return 0.5 * np.log(amplitude) ** 2 + np.pi**2 / 6 - 1.0 / amplitude


# The cases start every node at the steady state of V_init, i.e.
# theta0 = D_c / V_init, so the overshoot ratio collapses to the velocity ratio.
overshoot_ratio = V_dyn / V_init
breakdown_integral = _breakdown_integral(overshoot_ratio)


def rsf_slip_for_fracture_energy(state_effect, fracture_energy=G):
    """Return the RSF D_c whose breakdown work equals `fracture_energy`.

    It comes out far below the LSW D_c because the logarithmic tail of the
    ageing law keeps consuming energy long after a linear law has healed.
    """
    return fracture_energy / (sigma_n * state_effect * breakdown_integral)


# The three fault zones. Only `middle` carries the rupture, so it is the one
# calibrated to reach mu_k at V_dyn. `loading` is velocity neutral (a = b), so
# its steady-state friction stays at mu_s and it transmits load without
# weakening; `leading` is velocity strengthening and acts as a barrier.
RSF_ZONES = {
    "loading": {"a": 0.004, "b": 0.004},
    "middle": {"a": 0.005, "b": rsf_state_effect(0.005)},
    "leading": {"a": 0.008, "b": 0.005},
}

# One D_c is shared by every zone, matched on `middle` so that the zone the
# rupture actually runs through carries the cohesive-zone fracture energy.
RSF_D_c = rsf_slip_for_fracture_energy(RSF_ZONES["middle"]["b"], G)


def rsf_fracture_energy(state_effect, characteristic_slip=None):
    """Breakdown work of one RSF weakening step, in J/m^2."""
    if characteristic_slip is None:
        characteristic_slip = RSF_D_c
    return sigma_n * state_effect * characteristic_slip * breakdown_integral


def process_zone_size(state_effect, characteristic_slip=None):
    """Return the RSF state-evolution length L_b for one zone."""
    if characteristic_slip is None:
        characteristic_slip = RSF_D_c
    return mode_II_modulus * characteristic_slip / (state_effect * sigma_n)


def rsf_nucleation_lengths(direct_effect, state_effect, characteristic_slip=None):
    """Return (h*_RR, h*_RA) for one RSF zone.

    ``h*_RR`` is the Rice & Ruina linear-stability size (accurate for a/b well
    below 1) and ``h*_RA`` is the Rubin & Ampuero ageing-law size (accurate as
    a/b approaches 1). Both are undefined where the zone is velocity
    strengthening, so they come back as NaN when b <= a.
    """
    if characteristic_slip is None:
        characteristic_slip = RSF_D_c
    weakening = state_effect - direct_effect
    if weakening <= 0.0:
        return np.nan, np.nan
    h_RR = (np.pi / 4) * mode_II_modulus * characteristic_slip / (
        weakening * sigma_n
    )
    h_RA = (2 / np.pi) * mode_II_modulus * state_effect * characteristic_slip / (
        weakening**2 * sigma_n
    )
    return h_RR, h_RA


# ---------------------------------------------------------------------------
# Mesh resolution
# ---------------------------------------------------------------------------
PRODUCTION_CELL_SIZE = 0.5 * mm
# The usual floor is five cells across the process zone. L_b is the
# quasi-static width; it contracts as the rupture approaches the limiting
# speed, so treat a quasi-static count near the floor as already marginal.
MINIMUM_CELLS_PER_PROCESS_ZONE = 5


def binding_process_zone():
    """Return the smallest L_b across zones sharing one D_c; L_b scales as 1/b."""
    return min(process_zone_size(zone["b"]) for zone in RSF_ZONES.values())


def main():
    print(
        f"=== Case 1: cohesive-zone anchor, X_c = "
        f"{CZM_COHESIVE_ZONE_SIZE / mm:.0f} mm ==="
    )
    print(f"Normal stress (sigma_n):            {sigma_n / MPa:>10.3g} MPa")
    print(f"Static friction (mu_s):             {mu_s:>10.3g}")
    print(f"Kinetic friction (mu_k):            {mu_k:>10.3g}")
    print(f"Peak shear stress (tau_p):          {tau_p / MPa:>10.3g} MPa")
    print(f"Residual shear stress (tau_r):      {tau_r / MPa:>10.3g} MPa")
    print(f"Stress drop (d_tau):                {delta_tau / MPa:>10.3g} MPa")
    print(f"Young's modulus (E):                {E / GPa:>10.3g} GPa")
    print(f"Poisson's ratio (nu):               {nu:>10.3g}")
    print("Derived from the anchor:")
    print(f"  Fracture energy (Gamma):          {G:>10.6g} J/m^2")
    print(f"  LSW cohesive slip (D_c):          {D_c / mm:>10.6g} mm")
    print(f"  Quasistatic cohesive zone (X_c):  {X_c / mm:>10.6g} mm")
    print(f"  Critical nucleation length (L_c): {L_c / mm:>10.6g} mm")

    print()
    print("=== Case 2: LSW -> RSF parameter conversion ===")
    print(f"Initial friction (mu_s -> f0):      {mu_s:>10.3g}")
    print(f"Dynamic friction target (mu_k):     {mu_k:>10.3g}")
    print(f"Reference velocity (V*):            {V_ref:>10.3g} mm/s")
    print(f"Reference state (theta*):           {theta_ref:>10.3g} s")
    print(f"Initial steady velocity (V1):       {V_init:>10.3g} mm/s")
    print(f"Dynamic calibration velocity (V2):  {V_dyn:>10.3g} mm/s")
    print(f"ln(V2 / V1):                        {log_velocity_ratio:>10.6g}")

    print()
    print("The friction drop fixes a-b on its own; a only sets how that drop is")
    print("split between the direct and state effects.")
    print(
        f"a - b required for mu_s -> mu_k:    "
        f"{(mu_k - mu_s) / log_velocity_ratio:>10.6g}"
    )

    print()
    header = f"{'zone':<9}{'a':>10}{'b':>14}{'a-b':>14}{'mu(V2)':>10}"
    print(header)
    print("-" * len(header))
    for name, zone in RSF_ZONES.items():
        a = zone["a"]
        b = zone["b"]
        mu_dynamic = rsf_steady_friction(a, b, V_dyn)
        print(f"{name:<9}{a:>10.6g}{b:>14.10g}{a - b:>+14.8g}{mu_dynamic:>10.6g}")

    print()
    print("=== Case 3: RSF characteristic slip at the same fracture energy ===")
    print(f"Overshoot ratio (theta0 V2 / D_c):  {overshoot_ratio:>10.4g}")
    print(f"Breakdown integral:                 {breakdown_integral:>10.6g}")
    print(f"Target fracture energy (Gamma):     {G:>10.6g} J/m^2")
    print(f"LSW cohesive slip (D_c):            {D_c / mm:>10.6g} mm")
    print(f"RSF characteristic slip (D_c):      {RSF_D_c / mm:>10.6g} mm")
    print()
    print("The ageing law keeps drawing on its logarithmic tail long after a")
    print("linear law has healed, so it reaches the same Gamma on a much")
    print("shorter slip. One D_c is shared, matched on the middle zone; the")
    print("end zones then carry less energy in proportion to their b.")
    header = (
        f"{'zone':<9}{'mu_peak':>9}{'d_tau [MPa]':>13}"
        f"{'Gamma [J/m2]':>14}{'vs target':>11}"
    )
    print(header)
    print("-" * len(header))
    for name, zone in RSF_ZONES.items():
        energy = rsf_fracture_energy(zone["b"])
        print(
            f"{name:<9}{rsf_peak_friction(zone['a']):>9.4f}"
            f"{rsf_stress_drop(zone['b']) / MPa:>13.4g}{energy:>14.6g}"
            f"{energy / G:>10.2f}x"
        )

    print()
    print("=== Case 4: length scales and mesh resolution ===")
    print("With one shared D_c, L_b scales as 1/b, so the largest b binds. h* is")
    print("undefined where a zone is velocity neutral or strengthening.")
    print("Rubin & Ampuero (2005): h*_RR holds for a/b below ~0.378 and h*_RA")
    print("as a/b approaches 1; the applicable one is starred.")
    header = (
        f"{'zone':<9}{'a/b':>8}{'L_b [mm]':>11}{'cells/L_b':>11}"
        f"{'h*_RR [mm]':>13}{'h*_RA [mm]':>13}"
    )
    print(header)
    print("-" * len(header))
    for name, zone in RSF_ZONES.items():
        L_b = process_zone_size(zone["b"])
        cells = L_b / PRODUCTION_CELL_SIZE
        h_RR, h_RA = rsf_nucleation_lengths(zone["a"], zone["b"])
        ratio = zone["a"] / zone["b"]
        rr_mark, ra_mark = "", ""
        if np.isfinite(h_RR):
            rr_mark, ra_mark = ("*", "") if ratio < 0.378 else ("", "*")
        flag = (
            "  <- under-resolved"
            if cells < MINIMUM_CELLS_PER_PROCESS_ZONE
            else ""
        )
        print(
            f"{name:<9}{ratio:>8.4g}{L_b / mm:>11.4g}{cells:>11.1f}"
            f"{h_RR / mm:>12.6g}{rr_mark:<1}{h_RA / mm:>12.6g}{ra_mark:<1}{flag}"
        )

    binding = binding_process_zone()
    print()
    print(f"Binding process zone L_b:           {binding / mm:>10.4g} mm")
    print(f"LSW cohesive zone X_c:              {X_c / mm:>10.4g} mm")
    print(
        f"Coarsest mesh at {MINIMUM_CELLS_PER_PROCESS_ZONE} cells/L_b:      "
        f"{binding / MINIMUM_CELLS_PER_PROCESS_ZONE / mm:>10.4g} mm"
    )
    print(
        f"Production mesh {PRODUCTION_CELL_SIZE / mm:.2g} mm gives         "
        f"{binding / PRODUCTION_CELL_SIZE:>10.1f} cells/L_b"
    )
    print()
    print("L_b is quasi-static and contracts as the rupture speeds up, so aim")
    print("to keep a factor of two in hand rather than sitting on the floor.")
    print()
    print("h* and L_c answer different questions -- h* is a linear-stability")
    print("size for a uniform RSF fault, L_c an energy balance for a")
    print("slip-weakening crack -- so read them as independent indicators")
    print("rather than as a check on each other.")


if __name__ == "__main__":
    main()
