#!/usr/bin/env python3
"""Generate the 16-case TS0126 shear-loading-rate sweep."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0126_q4_chamfer20x5_12h.toml"
CASE_COUNT = 16
FIRST_RUN = 144
BASE_RAMP_TIME_S = 0.075
BASE_DISPLACEMENT_MM = 2.45


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}.")
    return text.replace(old, new)


def case_path(index: int) -> Path:
    run_number = FIRST_RUN + index - 1
    return ROOT / "cases" / f"rsf_{run_number:04d}_shear_rate_{index:02d}.toml"


def render_case(template: str, index: int) -> str:
    run_number = FIRST_RUN + index - 1
    speed_factor = 1.0 + 2.0 * (index - 1) / (CASE_COUNT - 1)
    ramp_time = BASE_RAMP_TIME_S / speed_factor
    peak_speed = (
        math.pi
        * BASE_DISPLACEMENT_MM
        / (2.0 * ramp_time)
    )

    text = replace_once(
        template,
        'name = "pmma-rsf-0126-chamfer20x5"',
        f'name = "pmma-rsf-{run_number:04d}-shear-rate-{index:02d}of16"',
    )
    text = replace_once(
        text,
        "shear_ramp_time = 0.075",
        (
            f"# Rate sweep {index:02d}/16: {speed_factor:.9f}x TS0126 "
            f"(peak {peak_speed:.9f} mm/s).\n"
            f"shear_ramp_time = {ramp_time:.15g}"
        ),
    )
    text = replace_once(text, "bulk_shear_frames = 83200", "bulk_shear_frames = 5200")
    text = replace_once(
        text,
        "interface_shear_frames = 800000",
        "interface_shear_frames = 50000",
    )
    text = replace_once(text, "maximum_dump_tb = 1.40", "maximum_dump_tb = 0.10")
    return text


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
