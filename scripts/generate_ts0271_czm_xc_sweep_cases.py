#!/usr/bin/env python3
"""Generate a 16-case prescribed cohesive-zone-size sweep around TS0271."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0271_leading_transition_16.toml"
CASE_COUNT = 16
FIRST_RUN = 272
BASELINE_XC_MM = 5.0
BASELINE_RSF_DC_MM = 0.0003765049284695767
XC_VALUES_MM = tuple(3.6 + 0.2 * offset for offset in range(CASE_COUNT))


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}.")
    return text.replace(old, new)


def cohesive_zone_size(index: int) -> float:
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}, found {index}.")
    return XC_VALUES_MM[index - 1]


def characteristic_slip(index: int) -> float:
    """Return RSF D_c in mm at fixed strength drop and ageing-law calibration."""
    return BASELINE_RSF_DC_MM * cohesive_zone_size(index) / BASELINE_XC_MM


def case_path(index: int) -> Path:
    run_number = FIRST_RUN + index - 1
    return ROOT / "cases" / f"rsf_{run_number:04d}_czm_xc_{index:02d}.toml"


def render_case(template: str, index: int) -> str:
    run_number = FIRST_RUN + index - 1
    x_c_mm = cohesive_zone_size(index)
    d_c_mm = characteristic_slip(index)

    text = replace_once(
        template,
        'name = "pmma-rsf-0271-leading-transition-16of16"',
        f'name = "pmma-rsf-{run_number:04d}-czm-xc-{index:02d}of16"',
    )
    text = replace_once(
        text,
        "stop_slip = 0.0003765049284695767",
        (
            f"# Prescribed CZM anchor X_c = {x_c_mm:.1f} mm; "
            "stop at one local D_c.\n"
            f"stop_slip = {d_c_mm:.16g}"
        ),
    )
    old_dc = "dc = 0.0003765049284695767"
    if text.count(old_dc) != 3:
        raise ValueError("Expected exactly three shared RSF D_c entries.")
    return text.replace(old_dc, f"dc = {d_c_mm:.16g}")


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
