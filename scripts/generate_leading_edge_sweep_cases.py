#!/usr/bin/env python3
"""Generate the 16-case TS0159 leading-edge RSF sweep."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0159_shear_rate_16.toml"
CASE_COUNT = 16
FIRST_RUN = 160
VN_INDEX = 8

VS_A = 0.008
VS_B = 0.005
VN_A = 0.005
VN_B = 0.005
VW_A = 0.005
VW_B = 0.025819400653936703


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}.")
    return text.replace(old, new)


def leading_parameters(index: int) -> tuple[float, float, str, float]:
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}, found {index}.")
    if index <= VN_INDEX:
        fraction = (index - 1) / (VN_INDEX - 1)
        direct_effect = VS_A + fraction * (VN_A - VS_A)
        state_effect = VS_B + fraction * (VN_B - VS_B)
        stage = "VS-to-VN"
    else:
        fraction = (index - VN_INDEX) / (CASE_COUNT - VN_INDEX)
        direct_effect = VN_A + fraction * (VW_A - VN_A)
        state_effect = VN_B + fraction * (VW_B - VN_B)
        stage = "VN-to-VW"
    return direct_effect, state_effect, stage, fraction


def case_path(index: int) -> Path:
    run_number = FIRST_RUN + index - 1
    return ROOT / "cases" / f"rsf_{run_number:04d}_leading_interp_{index:02d}.toml"


def render_case(template: str, index: int) -> str:
    run_number = FIRST_RUN + index - 1
    direct_effect, state_effect, stage, fraction = leading_parameters(index)

    text = replace_once(
        template,
        'name = "pmma-rsf-0159-shear-rate-16of16"',
        f'name = "pmma-rsf-{run_number:04d}-leading-ab-{index:02d}of16"',
    )
    text = replace_once(
        text,
        "# Rate sweep 16/16: 3.000000000x TS0126 (peak 153.938040026 mm/s).\n"
        "shear_ramp_time = 0.025",
        "# Fixed at the TS0159 loading rate (peak 153.938040026 mm/s).\n"
        "shear_ramp_time = 0.025",
    )
    text = replace_once(
        text,
        "[rsf.loading]\n"
        "a = 0.004\n"
        "b = 0.004\n"
        "dc = 0.0003765049284695767",
        "[rsf.loading]\n"
        "# TS0143: no loading-end VN zone; identical to the middle VW law.\n"
        "a = 0.005\n"
        "b = 0.025819400653936703\n"
        "dc = 0.0003765049284695767",
    )
    text = replace_once(
        text,
        "[rsf.leading]\n"
        "# Restore TS0124 friction so the chamfer is the only physical intervention.\n"
        "f0 = 0.8\n"
        "a = 0.008\n"
        "b = 0.005\n"
        "dc = 0.0003765049284695767",
        "[rsf.leading]\n"
        f"# Leading sweep {index:02d}/16, {stage}, fraction={fraction:.9f}.\n"
        f"f0 = 0.8\n"
        f"a = {direct_effect:.17g}\n"
        f"b = {state_effect:.17g}\n"
        "dc = 0.0003765049284695767",
    )
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
