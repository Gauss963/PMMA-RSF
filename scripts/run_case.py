#!/usr/bin/env python3
"""Run one TOML-defined Tatva PMMA case without animation work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tatva.pmma.config import load_case_config
from tatva.pmma.runner import allocate_run_directory, preflight, run_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PMMA case TOML file")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Use an exact run directory, primarily for Slurm job-local logs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the checkpoint in --run-dir after validating the input.",
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
        default=None,
        help="Checkpoint and exit cleanly after this much runner wall time.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate and print mesh/dump estimates without creating a run.",
    )
    parser.add_argument(
        "--allocate-run-dir",
        action="store_true",
        help="Create and print the next TS#### run directory, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    config = load_case_config(source)
    estimate = preflight(config)
    if args.preflight:
        print(json.dumps(estimate, indent=2))
        return 0 if estimate["within_dump_limit"] else 2
    default_root = REPO_ROOT / config.run_root
    run_root = (args.run_root or default_root).expanduser().resolve()
    if args.allocate_run_dir:
        if args.run_dir is not None or args.resume:
            raise ValueError(
                "--allocate-run-dir cannot be combined with --run-dir or --resume."
            )
        print(allocate_run_directory(run_root))
        return 0
    run_dir = run_case(
        config,
        source,
        run_root=run_root,
        run_dir=args.run_dir,
        resume=args.resume,
        time_limit_seconds=args.time_limit_seconds,
    )
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    print(json.dumps({"run_dir": str(run_dir), "status": status["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
