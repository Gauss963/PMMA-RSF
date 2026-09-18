#!/usr/bin/env python3
"""Generate a 16-case loading-end VW-to-VN sweep at fixed X_c = 5 mm."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0271_leading_transition_16.toml"
CASE_COUNT = 16
FIRST_RUN = 288
MIDDLE_A = 0.005
MIDDLE_B = 0.025819400653936703
NEUTRAL_A = 0.004
NEUTRAL_B = 0.004
FIXED_XC_MM = 5.0
FIXED_DC_MM = 0.0003765049284695767


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}.")
    return text.replace(old, new)


def neutral_fraction(index: int) -> float:
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}, found {index}.")
    return index / CASE_COUNT


def loading_parameters(index: int) -> tuple[float, float]:
    fraction = neutral_fraction(index)
    direct_effect = MIDDLE_A + fraction * (NEUTRAL_A - MIDDLE_A)
    state_effect = MIDDLE_B + fraction * (NEUTRAL_B - MIDDLE_B)
    return direct_effect, state_effect


def case_path(index: int) -> Path:
    run_number = FIRST_RUN + index - 1
    return ROOT / "cases" / f"rsf_{run_number:04d}_loading_vn_{index:02d}.toml"


def render_case(template: str, index: int) -> str:
    run_number = FIRST_RUN + index - 1
    fraction = neutral_fraction(index)
    direct_effect, state_effect = loading_parameters(index)

    text = replace_once(
        template,
        'name = "pmma-rsf-0271-leading-transition-16of16"',
        f'name = "pmma-rsf-{run_number:04d}-loading-vn-{index:02d}of16"',
    )
    old_loading = """[rsf.loading]
# TS0143: no loading-end VN zone; identical to the middle VW law.
a = 0.005
b = 0.025819400653936703
dc = 0.0003765049284695767"""
    new_loading = f"""[rsf.loading]
# Restore the loading-end VN law linearly; VN fraction={fraction:.9f}.
a = {direct_effect:.17g}
b = {state_effect:.17g}
dc = {FIXED_DC_MM:.16g}"""
    return replace_once(text, old_loading, new_loading)


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
