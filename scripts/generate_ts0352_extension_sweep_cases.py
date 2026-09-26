#!/usr/bin/env python3
"""Generate exactly 16 unloaded moving-block extensions, 0 to 30 mm."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .generate_ts0336_nucleation_sweep_cases import replace_once
except ImportError:
    from generate_ts0336_nucleation_sweep_cases import replace_once

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0343_nucleation_08.toml"
FIRST_RUN = 352
LENGTHS_MM = tuple(float(i * 2) for i in range(16))
CASE_COUNT = len(LENGTHS_MM)


def case_path(index: int) -> Path:
    if not 1 <= index <= CASE_COUNT:
        raise ValueError("Sweep index must be 1 through 16.")
    return ROOT / "cases" / f"rsf_{FIRST_RUN + index - 1:04d}_extension_{index:02d}.toml"


def render_case(template: str, index: int) -> str:
    case_path(index)
    text = replace_once(
        template, 'name = "pmma-rsf-0343-nucleation-08of16"',
        f'name = "pmma-rsf-{FIRST_RUN + index - 1:04d}-extension-{index:02d}of16"',
    )
    return replace_once(
        text, "dimensions = [200.0, 500.0]\n",
        "dimensions = [200.0, 500.0]\n"
        "# Original geometry/profile stays at y=0..500. Extend only toward -y.\n"
        "# No normal load or contact on extension; impose +y displacement at its bottom.\n"
        f"loading_extension_length = {LENGTHS_MM[index - 1]:.1f}\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    template = TEMPLATE.read_text()
    stale = []
    for index in range(1, CASE_COUNT + 1):
        path = case_path(index)
        expected = render_case(template, index)
        if args.check:
            if not path.is_file() or path.read_text() != expected:
                stale.append(path)
        else:
            path.write_text(expected)
            print(path.relative_to(ROOT))
    for path in stale:
        print(f"Stale generated case: {path}")
    return int(bool(stale))


if __name__ == "__main__":
    raise SystemExit(main())
