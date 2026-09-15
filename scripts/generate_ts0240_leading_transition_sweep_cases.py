#!/usr/bin/env python3
"""Generate the 16-case leading-edge transition sweep based on TS0240."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0240_normal_stress_01.toml"
CASE_COUNT = 16
FIRST_RUN = 256
NORMAL_STRESS_MPA = 16.0
MAXIMUM_TRANSITION_LENGTH_MM = 100.0
MINIMUM_TRANSITION_LENGTH_MM = 0.0


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}.")
    return text.replace(old, new)


def transition_length(index: int) -> float:
    """Return the leading half-cosine transition length for one sweep index."""
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}, found {index}.")
    fraction = (index - 1) / (CASE_COUNT - 1)
    return MAXIMUM_TRANSITION_LENGTH_MM + fraction * (
        MINIMUM_TRANSITION_LENGTH_MM - MAXIMUM_TRANSITION_LENGTH_MM
    )


def case_path(index: int) -> Path:
    run_number = FIRST_RUN + index - 1
    return (
        ROOT
        / "cases"
        / f"rsf_{run_number:04d}_leading_transition_{index:02d}.toml"
    )


def render_case(template: str, index: int) -> str:
    run_number = FIRST_RUN + index - 1
    length_mm = transition_length(index)

    text = replace_once(
        template,
        'name = "pmma-rsf-0240-normal-stress-01of16"',
        f'name = "pmma-rsf-{run_number:04d}-leading-transition-{index:02d}of16"',
    )
    text = replace_once(
        text,
        "normal_stress_reference = 17",
        f"normal_stress_reference = {NORMAL_STRESS_MPA:.1f}",
    )
    transition_comment = (
        "# Direct middle-to-leading step; no half-cosine transition."
        if length_mm == 0.0
        else "# Leading-edge half-cosine transition-length sweep."
    )
    return replace_once(
        text,
        "leading_transition_length = 100.0",
        f"{transition_comment}\nleading_transition_length = {length_mm:.15g}",
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
