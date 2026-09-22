#!/usr/bin/env python3
"""Generate 16 continuous-RSF cases: four direct effects by four ramp times."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0319_rsf_rate_16.toml"
FIRST_RUN = 336
DIRECT_EFFECTS = (0.0025, 0.005, 0.010, 0.020)
RAMP_TIMES_S = (0.075, 0.050, 0.030, 0.01924225500323749)
CASE_COUNT = len(DIRECT_EFFECTS) * len(RAMP_TIMES_S)
SHEAR_TIME_S = 0.095
INTERFACE_NORMAL_FRAMES = 1000
INTERFACE_SHEAR_FRAMES = 100000


def coordinates(index: int) -> tuple[int, int]:
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}.")
    return divmod(index - 1, len(RAMP_TIMES_S))


def parameters(template: dict, index: int) -> tuple[float, float, float, float]:
    row, column = coordinates(index)
    middle = template["rsf"]["middle"]
    a = DIRECT_EFFECTS[row]
    b = a + middle["b"] - middle["a"]
    # Fixed velocity ratio: preserving b*Dc preserves the calibrated velocity-
    # step breakdown integral. Actual dynamic breakdown work is not prescribed.
    dc = middle["dc"] * middle["b"] / b
    return a, b, dc, RAMP_TIMES_S[column]


def case_path(index: int) -> Path:
    coordinates(index)
    return ROOT / "cases" / f"rsf_{FIRST_RUN + index - 1:04d}_nucleation_{index:02d}.toml"


def replace_once(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"Expected {count} occurrences of {old!r}.")
    return text.replace(old, new)


def render_case(template: str, index: int) -> str:
    parsed = tomllib.loads(template)
    a, b, dc, ramp = parameters(parsed, index)
    number = FIRST_RUN + index - 1
    text = replace_once(
        template,
        'name = "pmma-rsf-0319-rsf-rate-16of16"',
        f'name = "pmma-rsf-{number:04d}-nucleation-{index:02d}of16"',
    )
    text = replace_once(
        text,
        "# Normal-stress sweep uses the full rectangular moving block.",
        "# Continuous RSF loading; full rectangular block without a chamfer.",
    )
    text = replace_once(
        text,
        f'shear_phase_time = {parsed["loading"]["shear_phase_time"]:.17g}',
        f"shear_phase_time = {SHEAR_TIME_S:.17g}",
    )
    text = replace_once(
        text,
        "# Fixed 2.45 mm half-cosine ramp; peak speed 200 mm/s.\n"
        f'shear_ramp_time = {parsed["loading"]["shear_ramp_time"]:.17g}',
        f"# Fixed 2.45 mm ramp; nominal peak speed {math.pi * 2.45 / (2 * ramp):.9g} mm/s.\n"
        f"shear_ramp_time = {ramp:.17g}",
    )
    text = replace_once(
        text,
        "# Preserve the TS0278 CZM energy; stop at the recalibrated local D_c.\n"
        f'stop_slip = {parsed["loading"]["stop_slip"]:.17g}',
        "# Freeze on the first loading-end V >= 500 mm/s; no coverage delay.\n"
        "stop_slip = 1.0e-12",
    )
    text = replace_once(text, "stop_min_y = 0.5", "stop_min_y = 5.0")
    text = replace_once(
        text,
        "# Require rupture through the full, unchamfered active contact.\n"
        "stop_max_y = 499.0\nstop_coverage_fraction = 1.0",
        "# Monitor the same y = 5-25 mm interval as TS0320-TS0335.\n"
        "stop_max_y = 25.0",
    )
    text = replace_once(
        text,
        "initial_steady_velocity = 1.0e-4\n",
        "initial_steady_velocity = 1.0e-4\n"
        '# theta0 = Dc/V_init; evolve continuously, without a phase reset.\n'
        'initial_state_mode = "steady-state"\n',
    )
    text = replace_once(
        text, "interface_normal_frames = 100\n",
        f"interface_normal_frames = {INTERFACE_NORMAL_FRAMES}\n",
    )
    text = replace_once(
        text, "interface_shear_frames = 50000\n",
        f"interface_shear_frames = {INTERFACE_SHEAR_FRAMES}\n",
    )
    old_middle = parsed["rsf"]["middle"]
    old_zone = (
        f'a = {old_middle["a"]}\n'
        f'b = {old_middle["b"]}\n'
        f'dc = {old_middle["dc"]}'
    )
    text = replace_once(
        text, old_zone, f"a = {a:.17g}\nb = {b:.17g}\ndc = {dc:.17g}", count=2,
    )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    template = TEMPLATE.read_text(encoding="utf-8")
    stale = []
    for index in range(1, CASE_COUNT + 1):
        path = case_path(index)
        expected = render_case(template, index)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                stale.append(path)
        else:
            path.write_text(expected, encoding="utf-8")
            print(path.relative_to(ROOT))
    for path in stale:
        print(f"Stale generated case: {path.relative_to(ROOT)}")
    return int(bool(stale))


if __name__ == "__main__":
    raise SystemExit(main())
