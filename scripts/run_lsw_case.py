#!/usr/bin/env python3
"""Run or benchmark one legacy-compatible PMMA LSW case, serial or MPI."""

from dataclasses import asdict, replace
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tatva.pmma.dynamics import build_case_model, run_simulation_dumped, SimulationCheckpointed
from tatva.pmma.lsw import load_lsw_case, mesh_estimate
from tatva.pmma.mpi import get_mpi_context
from tatva.pmma.runner import allocate_run_directory


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-limit-seconds", type=float)
    parser.add_argument("--benchmark-steps", type=int, default=0,
                        help="Short timing test per phase; not a physical production run.")
    parser.add_argument("--mesh-size", type=float, help="Benchmark-only mesh override")
    args = parser.parse_args()
    started = time.monotonic()
    mpi = get_mpi_context()
    source = args.input.resolve()
    case, cfg, output, description = load_lsw_case(source)
    if cfg.dtype == "float64":
        # Must precede any JAX array creation, or float64 is silently truncated.
        import jax
        jax.config.update("jax_enable_x64", True)
    if args.mesh_size is not None:
        if not args.benchmark_steps:
            raise ValueError("Mesh overrides are restricted to benchmarks.")
        cfg = replace(cfg, mesh_size=args.mesh_size)
    if args.benchmark_steps < 0:
        raise ValueError("benchmark-steps must be nonnegative")
    if args.benchmark_steps:
        duration = args.benchmark_steps * cfg.time_step_override
        cfg = replace(cfg, normal_phase_time=duration, normal_ramp_time=duration,
                      shear_phase_time=duration, shear_ramp_time=duration)
        output = dict(output, frames_per_phase=max(2, math.ceil(args.benchmark_steps / 667)),
                      shear_frames_per_phase=max(2, math.ceil(args.benchmark_steps / 25)))
    model = build_case_model(case, cfg)
    estimate = mesh_estimate(case, cfg, output, model)
    del model
    if mpi.is_root:
        print(json.dumps(estimate, indent=2), flush=True)
    if args.preflight:
        return 0
    run = None
    if mpi.is_root:
        run = args.run_dir.resolve() if args.run_dir else allocate_run_directory(ROOT / "runs")
        if args.resume:
            recorded = json.loads((run / "input/resolved.json").read_text())
            if recorded["solver"] != asdict(cfg) or recorded["output"] != output:
                raise ValueError("Resume input does not match the saved simulation.")
            if recorded["input_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
                raise ValueError("Resume source configuration changed.")
        elif (run / "data/simulation.h5").exists():
            raise FileExistsError(f"Refusing to overwrite {run}")
        for folder in ("data", "stats", "input", "logs", "Plot"):
            (run / folder).mkdir(parents=True, exist_ok=True)
        if not args.resume:
            shutil.copy2(source, run / "input/case.toml")
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            write_json(run / "input/resolved.json", {
                "case": asdict(case), "solver": asdict(cfg), "output": output,
                "description": description, "benchmark": bool(args.benchmark_steps),
                "git_revision": revision,
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            })
        write_json(run / "stats/preflight.json", estimate)
        print(f"Run directory: {run}; MPI ranks: {mpi.size}", flush=True)
        write_json(run / "status.json", {"status": "running", "mpi_ranks": mpi.size})
    if mpi.enabled:
        run = Path(mpi.comm.bcast(str(run) if mpi.is_root else None, root=0))
    deadline = None if args.time_limit_seconds is None else started + args.time_limit_seconds
    try:
        result = run_simulation_dumped(
            case, cfg, run / "data/simulation.h5", **output,
            checkpoint_path=run / "data/checkpoint.npz",
            checkpoint_deadline_monotonic=deadline, resume=args.resume,
        )
    except SimulationCheckpointed:
        if mpi.is_root:
            write_json(run / "status.json", {"status": "checkpointed", "mpi_ranks": mpi.size})
        return 0
    if mpi.is_root:
        summary = result["summary"]
        summary["wall_seconds"] = time.monotonic() - started
        summary["benchmark"] = bool(args.benchmark_steps)
        summary["slurm_job_id"] = os.getenv("SLURM_JOB_ID")
        for key in ("final_max_slip", "final_avg_tau", "final_avg_sigma_n"):
            if not math.isfinite(summary[key]):
                raise RuntimeError(f"Non-finite simulation output: {key}")
        write_json(run / "stats/summary.json", {"summary": summary, "run_dir": str(run)})
        write_json(run / "status.json", {"status": "completed", "mpi_ranks": mpi.size})
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        mpi = get_mpi_context()
        if mpi.enabled:
            mpi.comm.Abort(1)
        raise
