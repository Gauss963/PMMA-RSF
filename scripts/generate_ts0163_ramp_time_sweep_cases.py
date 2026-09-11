#!/usr/bin/env python3
"""Generate the 16-case ramp-time sweep based on TS0163."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0163_leading_interp_04.toml"
CASE_COUNT = 16
FIRST_RUN = 176
START_RAMP_TIME_S = 0.025
END_RAMP_TIME_S = 0.075


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}.")
    return text.replace(old, new)


def ramp_time(index: int) -> float:
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}, found {index}.")
    fraction = (index - 1) / (CASE_COUNT - 1)
    return START_RAMP_TIME_S + fraction * (END_RAMP_TIME_S - START_RAMP_TIME_S)


def case_path(index: int) -> Path:
    run_number = FIRST_RUN + index - 1
    return ROOT / "cases" / f"rsf_{run_number:04d}_ramp_time_{index:02d}.toml"


def render_case(template: str, index: int) -> str:
    run_number = FIRST_RUN + index - 1
    duration = ramp_time(index)
    fraction = (index - 1) / (CASE_COUNT - 1)

    text = replace_once(
        template,
        'name = "pmma-rsf-0163-leading-ab-04of16"',
        f'name = "pmma-rsf-{run_number:04d}-ramp-time-{index:02d}of16"',
    )
    return replace_once(
        text,
        "# Fixed at the TS0159 loading rate (peak 153.938040026 mm/s).\n"
        "shear_ramp_time = 0.025",
        (
            f"# Ramp-time sweep {index:02d}/16 from TS0163 to 75 ms; "
            f"fraction={fraction:.9f}.\n"
            f"shear_ramp_time = {duration:.15g}"
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
