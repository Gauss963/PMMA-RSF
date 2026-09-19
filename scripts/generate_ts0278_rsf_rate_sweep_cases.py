#!/usr/bin/env python3
"""Generate the 16 fixed-fracture-energy RSF loading-rate sweep cases."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0278_czm_xc_07.toml"
FIRST_RUN = 304
CASE_COUNT = 16
CONTROL_RUN = FIRST_RUN + CASE_COUNT
BASELINE_INDEX = 6
BASE_RAMP_TIME_S = 0.075
SLOW_RAMP_TIME_S = 0.150
FAST_PEAK_SPEED_MM_S = 200.0
POST_RAMP_TIME_S = 0.030
DYNAMIC_VELOCITY_MM_S = 200.0
BULK_SHEAR_FRAMES = 4400


def replace_count(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"Expected {count} occurrences of {old!r}.")
    return text.replace(old, new)


def breakdown_integral(velocity_ratio: float) -> float:
    """Ageing-law breakdown integral used by the existing CZM calibration."""
    amplitude = velocity_ratio - 1.0
    return 0.5 * math.log(amplitude) ** 2 + math.pi**2 / 6.0 - 1.0 / amplitude


def calibrated_parameters(template: dict) -> tuple[float, float]:
    rsf = template["rsf"]
    old_b = rsf["middle"]["b"]
    old_dc = rsf["middle"]["dc"]
    initial_velocity = rsf["initial_steady_velocity"]
    old_dynamic_velocity = rsf["dynamic_calibration_velocity"]
    direct_effect = rsf["middle"]["a"]
    new_b = direct_effect + (
        rsf["initial_friction"] - rsf["target_middle_dynamic_friction"]
    ) / math.log(DYNAMIC_VELOCITY_MM_S / initial_velocity)
    old_breakdown = breakdown_integral(old_dynamic_velocity / initial_velocity)
    new_breakdown = breakdown_integral(DYNAMIC_VELOCITY_MM_S / initial_velocity)
    new_dc = old_dc * old_b * old_breakdown / (new_b * new_breakdown)
    return new_b, new_dc


def ramp_time(index: int) -> float:
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}.")
    fast_ramp_time = math.pi * 2.45 / (2.0 * FAST_PEAK_SPEED_MM_S)
    if index <= BASELINE_INDEX:
        fraction = (index - 1) / (BASELINE_INDEX - 1)
        return SLOW_RAMP_TIME_S * (BASE_RAMP_TIME_S / SLOW_RAMP_TIME_S) ** fraction
    fraction = (index - BASELINE_INDEX) / (CASE_COUNT - BASELINE_INDEX)
    return BASE_RAMP_TIME_S * (fast_ramp_time / BASE_RAMP_TIME_S) ** fraction


def case_path(index: int) -> Path:
    if index == CASE_COUNT + 1:
        return ROOT / "cases" / f"rsf_{CONTROL_RUN:04d}_nucleation_stop.toml"
    return ROOT / "cases" / f"rsf_{FIRST_RUN + index - 1:04d}_rsf_rate_{index:02d}.toml"


def render_case(template: str, index: int) -> str:
    parsed = tomllib.loads(template)
    new_b, new_dc = calibrated_parameters(parsed)
    control = index == CASE_COUNT + 1
    time_s = BASE_RAMP_TIME_S if control else ramp_time(index)
    shear_time_s = max(BASE_RAMP_TIME_S, time_s + POST_RAMP_TIME_S)
    run_number = CONTROL_RUN if control else FIRST_RUN + index - 1
    suffix = "nucleation-stop" if control else f"rsf-rate-{index:02d}of16"
    peak_speed = math.pi * 2.45 / (2.0 * time_s)

    text = replace_count(
        template,
        'name = "pmma-rsf-0278-czm-xc-07of16"',
        f'name = "pmma-rsf-{run_number:04d}-{suffix}"',
    )
    text = replace_count(
        text,
        "dynamic_calibration_velocity = 2000.0",
        f"dynamic_calibration_velocity = {DYNAMIC_VELOCITY_MM_S:.1f}",
    )
    text = replace_count(
        text,
        "b = 0.025819400653936703",
        f"b = {new_b:.17g}",
        count=2,
    )
    text = replace_count(
        text,
        "0.0003614447313307937",
        f"{new_dc:.17g}",
        count=4,
    )
    text = replace_count(
        text,
        "shear_phase_time = 0.075",
        f"shear_phase_time = {shear_time_s:.17g}",
    )
    text = replace_count(
        text,
        "# Full 75 ms half-cosine shear-displacement ramp.\nshear_ramp_time = 0.075",
        (
            f"# Fixed 2.45 mm half-cosine ramp; peak speed {peak_speed:.6g} mm/s.\n"
            f"shear_ramp_time = {time_s:.17g}"
        ),
    )
    text = replace_count(
        text,
        "bulk_shear_frames = 5200",
        f"bulk_shear_frames = {BULK_SHEAR_FRAMES}",
    )
    text = replace_count(
        text,
        "# Prescribed CZM anchor X_c = 4.8 mm; stop at one local D_c.",
        "# Preserve the TS0278 CZM energy; stop at the recalibrated local D_c.",
    )
    if control:
        text = replace_count(
            text,
            "# Preserve the TS0278 CZM energy; stop at the recalibrated local D_c.",
            "# The dynamic stop ignores pre-slip accumulated during normal loading.",
        )
        text = replace_count(
            text,
            f"stop_slip = {new_dc:.17g}",
            "stop_slip = 1.0e-12",
        )
        text = replace_count(text, "stop_min_y = 0.5", "stop_min_y = 5.0")
        text = replace_count(text, "stop_max_y = 499.0", "stop_max_y = 25.0")
        text = replace_count(
            text,
            "# Require rupture through the full, unchamfered active contact.",
            "# Freeze at the first 500 mm/s event in the loading-end nucleus.",
        )
        text = replace_count(
            text,
            "stop_coverage_fraction = 1.0",
            "# No slip-coverage test: normal loading already accumulated several D_c.",
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
