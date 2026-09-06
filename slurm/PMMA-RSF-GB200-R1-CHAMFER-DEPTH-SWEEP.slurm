#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=gb200-r1
#SBATCH --job-name=PMMA-CHAMFER-SWEEP
#SBATCH --output=gb200-chamfer-depth-sweep-%j.out
#SBATCH --error=gb200-chamfer-depth-sweep-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=16:00:00
#SBATCH --signal=USR1@1800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/tatva
ENV_PREFIX=/work/gauss112/.venvs/tatva-gb200
PYTHON=$ENV_PREFIX/bin/python
RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-54000}
MIN_FREE_BYTES=${MIN_FREE_BYTES:-10000000000}

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset LD_LIBRARY_PATH || true
cd "$ROOT"

echo "GB200 chamfer-depth sweep job ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes: ${SLURM_JOB_NODELIST:-unset}; tasks: ${SLURM_NTASKS:-unset}"

"$PYTHON" scripts/generate_ts0163_chamfer_depth_sweep_cases.py --check
"$PYTHON" - <<'PY'
import shutil
from dataclasses import asdict
from pathlib import Path

from scripts.generate_ts0163_chamfer_depth_sweep_cases import chamfer_geometry
from tatva.pmma.config import load_case_config
from tatva.pmma.runner import preflight

root = Path("/work/gauss112/tatva")
case_paths = [
    root / "cases" / f"rsf_{run:04d}_chamfer_depth_{index:02d}.toml"
    for index, run in enumerate(range(192, 208), start=1)
]
configs = [load_case_config(path) for path in case_paths]
baseline = asdict(configs[0])
for index, (path, config) in enumerate(zip(case_paths, configs, strict=True), start=1):
    expected_length, expected_depth, expected_stop = chamfer_geometry(index)
    actual = (
        config.moving.leading_chamfer_along_fault,
        config.moving.leading_chamfer_perpendicular,
        config.loading.stop_max_y,
    )
    expected = (expected_length, expected_depth, expected_stop)
    if any(abs(a - b) > 1.0e-14 for a, b in zip(actual, expected, strict=True)):
        raise SystemExit(f"Unexpected chamfer geometry in {path.name}: {actual}.")
    payload = asdict(config)
    payload["name"] = baseline["name"]
    payload["moving"]["leading_chamfer_along_fault"] = baseline["moving"][
        "leading_chamfer_along_fault"
    ]
    payload["moving"]["leading_chamfer_perpendicular"] = baseline["moving"][
        "leading_chamfer_perpendicular"
    ]
    payload["loading"]["stop_max_y"] = baseline["loading"]["stop_max_y"]
    if payload != baseline:
        raise SystemExit(f"Unexpected non-geometry difference in {path.name}.")

estimates = [preflight(config) for config in configs]
if not all(estimate["within_dump_limit"] for estimate in estimates):
    raise SystemExit("At least one per-case dump estimate exceeds its limit.")
per_case_estimates = [int(estimate["estimated_dump_bytes"]) for estimate in estimates]
estimated_total = sum(per_case_estimates)
existing_bytes = sum(
    min(
        estimate,
        (root / "runs" / f"TS{run:04d}" / "data" / "simulation.h5").stat().st_size,
    )
    for run, estimate in zip(range(192, 208), per_case_estimates, strict=True)
    if (root / "runs" / f"TS{run:04d}" / "data" / "simulation.h5").is_file()
)
estimated_remaining = estimated_total - existing_bytes
free_bytes = shutil.disk_usage(root).free
required_bytes = estimated_remaining + 20_000_000_000
print(
    f"Chamfer-depth storage gate: total={estimated_total / 1e12:.4f} TB; "
    f"existing={existing_bytes / 1e12:.4f} TB; "
    f"remaining={estimated_remaining / 1e12:.4f} TB; "
    f"free={free_bytes / 1e12:.4f} TB; required={required_bytes / 1e12:.4f} TB"
)
if free_bytes < required_bytes:
    raise SystemExit("Insufficient free space for the 16-case chamfer-depth sweep.")
PY

export ROOT ENV_PREFIX RUN_TIME_LIMIT_SECONDS MIN_FREE_BYTES
set +e
srun --exact --mpi=none --kill-on-bad-exit=0 --wait=0 \
  --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=12 \
  --gpus-per-task=1 --gpu-bind=single:1 \
  bash -c '
    export SWEEP_INDEX=$((SLURM_PROCID + 1))
    exec bash "$ROOT/scripts/run_gb200_ts0163_chamfer_depth_sweep_rank.sh"
  '
sweep_status=$?
set -e

echo "GB200 chamfer-depth sweep job ended at $(date -u +%Y-%m-%dT%H:%M:%SZ); status=$sweep_status"
exit "$sweep_status"
