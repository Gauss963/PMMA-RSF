#!/usr/bin/env python3
"""Generate the 16-case chamfer-depth sweep based on TS0163."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0163_leading_interp_04.toml"
CASE_COUNT = 16
FIRST_RUN = 192
CHAMFER_LENGTH_MM = 20.0
MAX_CHAMFER_DEPTH_MM = 8.0


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}.")
    return text.replace(old, new)


def chamfer_geometry(index: int) -> tuple[float, float, float]:
    """Return along-fault length, perpendicular depth, and rupture endpoint."""
    if not 1 <= index <= CASE_COUNT:
        raise ValueError(f"Sweep index must be 1 through {CASE_COUNT}, found {index}.")
    depth = MAX_CHAMFER_DEPTH_MM * (index - 1) / (CASE_COUNT - 1)
    if index == 1:
        # A zero-depth triangular cut is a rectangular, full-length interface.
        return 0.0, 0.0, 499.0
    return CHAMFER_LENGTH_MM, depth, 479.0


def case_path(index: int) -> Path:
    run_number = FIRST_RUN + index - 1
    return ROOT / "cases" / f"rsf_{run_number:04d}_chamfer_depth_{index:02d}.toml"


def render_case(template: str, index: int) -> str:
    run_number = FIRST_RUN + index - 1
    length, depth, stop_max_y = chamfer_geometry(index)

    text = replace_once(
        template,
        'name = "pmma-rsf-0163-leading-ab-04of16"',
        f'name = "pmma-rsf-{run_number:04d}-chamfer-depth-{index:02d}of16"',
    )
    text = replace_once(
        text,
        "leading_chamfer_along_fault = 20.0\n"
        "leading_chamfer_perpendicular = 5.0",
        (
            f"# Chamfer-depth sweep {index:02d}/16; "
            f"perpendicular depth={depth:.9f} mm.\n"
            f"leading_chamfer_along_fault = {length:.15g}\n"
            f"leading_chamfer_perpendicular = {depth:.15g}"
        ),
    )
    return replace_once(
        text,
        "stop_max_y = 479.0",
        (
            "# Require rupture through the end of the active contact.\n"
            f"stop_max_y = {stop_max_y:.15g}"
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
