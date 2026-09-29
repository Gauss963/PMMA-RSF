#!/usr/bin/env python3
"""Exactly sixteen extension/timestep convergence cases, TS0368-TS0383."""

import argparse
from pathlib import Path

try:
    from .generate_ts0336_nucleation_sweep_cases import replace_once
except ImportError:
    from generate_ts0336_nucleation_sweep_cases import replace_once

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases/rsf_0352_extension_01.toml"
FIRST_RUN = 368
LENGTHS = (0.0, 6.0, 28.0, 30.0)
DT_NS = (10.0, 5.0, 2.5, 1.25)
VARIANTS = tuple((length, dt) for length in LENGTHS for dt in DT_NS)
CASE_COUNT = len(VARIANTS)


def case_path(index):
    if not 1 <= index <= CASE_COUNT:
        raise ValueError("Sweep index must be 1 through 16.")
    return ROOT / "cases" / f"rsf_{FIRST_RUN + index - 1:04d}_convergence_{index:02d}.toml"


def render_case(template, index):
    case_path(index)
    length, dt = VARIANTS[index - 1]
    replacements = (
        ('name = "pmma-rsf-0352-extension-01of16"',
         f'name = "pmma-rsf-{FIRST_RUN + index - 1:04d}-convergence-{index:02d}of16"'),
        ('loading_extension_length = 0.0', f'loading_extension_length = {length}'),
        ('time_step = 1.0e-8', f'time_step = {dt * 1e-9:.12g}'),
        ('dtype = "float32"', 'dtype = "float32"\nrsf_state_dtype = "float64"'),
        ('shear_phase_time = 0.095000000000000001', 'shear_phase_time = 0.012'),
        ('bulk_normal_frames = 2', 'bulk_normal_frames = 8'),
        ('bulk_shear_frames = 4400', 'bulk_shear_frames = 100'),
        ('interface_shear_frames = 100000', 'interface_shear_frames = 120001\n'
         '# Every integration step, original loading end y=0..30 mm.\n'
         'integration_probe_max_y = 30.0'),
    )
    for old, new in replacements:
        template = replace_once(template, old, new)
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for index in range(1, CASE_COUNT + 1):
        path = case_path(index)
        expected = render_case(TEMPLATE.read_text(), index)
        if args.check:
            if not path.is_file() or path.read_text() != expected:
                stale.append(str(path))
        else:
            path.write_text(expected)
            print(path.relative_to(ROOT))
    if stale:
        raise SystemExit("Stale generated cases: " + ", ".join(stale))


if __name__ == "__main__":
    main()
