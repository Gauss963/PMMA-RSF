#!/usr/bin/env python3
"""Generate the 16-case normal-displacement dip sweep based on TS0163."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0163_leading_interp_04.toml"
CASE_COUNT = 16
FIRST_RUN = 208
MINIMUM_END_FRACTION = 0.90
MAXIMUM_END_FRACTION = 1.10
SHEAR_RAMP_TIME_SECONDS = 0.075


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}.")
    return text.replace(old, new)


def normal_displacement_fractions(index: int) -> tuple[float, float]:
    """Return loading- and leading-end multipliers for one sweep index."""
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}, found {index}.")
    fraction = (index - 1) / (CASE_COUNT - 1)
    loading = MINIMUM_END_FRACTION + fraction * (
        MAXIMUM_END_FRACTION - MINIMUM_END_FRACTION
    )
    leading = MAXIMUM_END_FRACTION - fraction * (
        MAXIMUM_END_FRACTION - MINIMUM_END_FRACTION
    )
    return loading, leading


def case_path(index: int) -> Path:
    run_number = FIRST_RUN + index - 1
    return ROOT / "cases" / f"rsf_{run_number:04d}_normal_dip_{index:02d}.toml"


def render_case(template: str, index: int) -> str:
    run_number = FIRST_RUN + index - 1
    loading_fraction, leading_fraction = normal_displacement_fractions(index)

    text = replace_once(
        template,
        'name = "pmma-rsf-0163-leading-ab-04of16"',
        f'name = "pmma-rsf-{run_number:04d}-normal-dip-{index:02d}of16"',
    )
    text = replace_once(
        text,
        "leading_chamfer_along_fault = 20.0\n"
        "leading_chamfer_perpendicular = 5.0",
        (
            "# Normal-dip sweep uses the full rectangular moving block.\n"
            "leading_chamfer_along_fault = 0.0\n"
            "leading_chamfer_perpendicular = 0.0"
        ),
    )
    text = replace_once(
        text,
        "normal_displacement = 0.691620986687549",
        (
            "normal_displacement = 0.691620986687549\n"
            f"# Normal-dip sweep {index:02d}/16; y=0 is the loading end.\n"
            "normal_displacement_loading_fraction = "
            f"{loading_fraction:.15g}\n"
            "normal_displacement_leading_fraction = "
            f"{leading_fraction:.15g}"
        ),
    )
    text = replace_once(
        text,
        "# Fixed at the TS0159 loading rate (peak 153.938040026 mm/s).\n"
        "shear_ramp_time = 0.025",
        (
            "# Full 75 ms half-cosine shear-displacement ramp.\n"
            f"shear_ramp_time = {SHEAR_RAMP_TIME_SECONDS:.3f}"
        ),
    )
    return replace_once(
        text,
        "stop_max_y = 479.0",
        (
            "# Require rupture through the full, unchamfered active contact.\n"
            "stop_max_y = 499.0"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed cases differ from deterministic output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        print("Stale generated cases:")
        for path in stale:
            print(path.relative_to(ROOT))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
