#!/usr/bin/env python3
"""Generate TS0320-TS0335 with traction-consistent RSF prestress."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0319_rsf_rate_16.toml"
FIRST_RUN = 320
CASE_COUNT = 16
NORMALIZED_PRESTRESS_MIN = 0.450
NORMALIZED_PRESTRESS_STEP = 0.025
DYNAMIC_SHEAR_INCREMENT_MM = 2.45
NORMAL_RELAXATION_TIME_S = 2.0e-4
PRESTRESS_START_TIME_S = 0.020
PRESTRESS_RAMP_TIME_S = 0.010

# Linear h -> 0 extrapolation from h = 100, 50, 20 and 10 mm pilots.  The
# intercept includes the shear traction induced by the locked normal-loading
# stage; the slope converts loading-face displacement to mean fault traction.
PRESTRESS_TRACTION_INTERCEPT_MPA = 1.5657202449864727
PRESTRESS_STIFFNESS_MPA_PER_MM = 5.49296788259774


def replace_count(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"Expected {count} occurrences of {old!r}.")
    return text.replace(old, new)


def normalized_prestress(index: int) -> float:
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}.")
    return NORMALIZED_PRESTRESS_MIN + (index - 1) * NORMALIZED_PRESTRESS_STEP


def peak_friction(template: dict) -> float:
    rsf = template["rsf"]
    return rsf["initial_friction"] + rsf["middle"]["a"] * math.log(
        rsf["dynamic_calibration_velocity"] / rsf["initial_steady_velocity"]
    )


def target_mean_shear_traction(template: dict, index: int) -> float:
    rsf = template["rsf"]
    loading = template["loading"]
    residual = rsf["target_middle_dynamic_friction"]
    target_mu = residual + normalized_prestress(index) * (
        peak_friction(template) - residual
    )
    return loading["normal_stress_reference"] * target_mu


def prestress_displacement(template: dict, index: int) -> float:
    target_tau = target_mean_shear_traction(template, index)
    return (
        target_tau - PRESTRESS_TRACTION_INTERCEPT_MPA
    ) / PRESTRESS_STIFFNESS_MPA_PER_MM


def case_path(index: int) -> Path:
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}.")
    run_number = FIRST_RUN + index - 1
    return ROOT / "cases" / f"rsf_{run_number:04d}_rsf_prestress_{index:02d}.toml"


def render_case(template: str, index: int) -> str:
    parsed = tomllib.loads(template)
    run_number = FIRST_RUN + index - 1
    target_pi = normalized_prestress(index)
    target_tau = target_mean_shear_traction(parsed, index)
    prestress = prestress_displacement(parsed, index)
    final_displacement = prestress + DYNAMIC_SHEAR_INCREMENT_MM

    text = replace_count(
        template,
        'name = "pmma-rsf-0319-rsf-rate-16of16"',
        f'name = "pmma-rsf-{run_number:04d}-prestress-{index:02d}of16"',
    )
    text = replace_count(
        text,
        "# Normal-stress sweep uses the full rectangular moving block.",
        "# Retain the TS0319 full rectangular moving block.",
    )
    text = replace_count(
        text,
        "normal_ramp_time = 0.020\n",
        (
            "normal_ramp_time = 0.020\n"
            "# Damping is enabled only for the static initial-condition construction.\n"
            f"normal_relaxation_time = {NORMAL_RELAXATION_TIME_S:.12g}\n"
            f"normal_relaxation_start_time = {PRESTRESS_START_TIME_S:.12g}\n"
        ),
    )
    text = replace_count(
        text,
        "shear_displacement_final = 2.45\n"
        "quasistatic_shear_fraction = 0.0\n"
        "quasistatic_shear_start_time = 0.0\n"
        "quasistatic_shear_ramp_time = 0.0",
        (
            f"shear_displacement_final = {final_displacement:.17g}\n"
            "# Build a static traction field while RSF is inactive, then invert the\n"
            "# regularized law so theta supports that traction at V_init.\n"
            f"prestress_shear_displacement = {prestress:.17g}\n"
            "quasistatic_shear_fraction = 0.0\n"
            f"quasistatic_shear_start_time = {PRESTRESS_START_TIME_S:.12g}\n"
            f"quasistatic_shear_ramp_time = {PRESTRESS_RAMP_TIME_S:.12g}"
        ),
    )
    text = replace_count(
        text,
        "# Fixed 2.45 mm half-cosine ramp; peak speed 200 mm/s.",
        "# Add the same 2.45 mm half-cosine increment; peak speed 200 mm/s.",
    )
    text = replace_count(
        text,
        "# Preserve the TS0278 CZM energy; stop at the recalibrated local D_c.\n"
        "stop_slip = 0.00042852936846183833",
        (
            "# Freeze on the first dynamic loading-end event in this shear phase.\n"
            "stop_slip = 1.0e-12"
        ),
    )
    text = replace_count(text, "stop_min_y = 0.5", "stop_min_y = 5.0")
    text = replace_count(text, "stop_max_y = 499.0", "stop_max_y = 25.0")
    text = replace_count(
        text,
        "# Require rupture through the full, unchamfered active contact.",
        "# Monitor only the loading-end nucleus.",
    )
    text = replace_count(
        text,
        "stop_coverage_fraction = 1.0",
        "# A single station at 500 mm/s is sufficient; no coverage criterion.",
    )
    text = replace_count(
        text,
        "initial_steady_velocity = 1.0e-4\n",
        (
            "initial_steady_velocity = 1.0e-4\n"
            'initial_state_mode = "traction-consistent-handoff"\n'
            f"target_normalized_prestress = {target_pi:.17g}\n"
            f"# Target mean shear traction: {target_tau:.9g} MPa.\n"
        ),
    )
    text = replace_count(
        text,
        "# Leading sweep 04/16, VS-to-VN, fraction=0.428571429.",
        "# Retain the TS0319 leading velocity-strengthening parameters.",
    )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    template = TEMPLATE.read_text(encoding="utf-8")
    stale: list[Path] = []
    for index in range(1, CASE_COUNT + 1):
        path = case_path(index)
        expected = render_case(template, index)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                stale.append(path)
        else:
            path.write_text(expected, encoding="utf-8")
            print(path.relative_to(ROOT))
    if stale:
        for path in stale:
            print(f"Stale generated case: {path.relative_to(ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
