#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=gb200-r1
#SBATCH --job-name=PMMA-AB-SWEEP
#SBATCH --output=gb200-ab-sweep-%j.out
#SBATCH --error=gb200-ab-sweep-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/tatva
ENV_PREFIX=/work/gauss112/.venvs/tatva-gb200
PYTHON=$ENV_PREFIX/bin/python
RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-55800}
MIN_FREE_BYTES=${MIN_FREE_BYTES:-50000000000}

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset LD_LIBRARY_PATH || true
cd "$ROOT"

echo "GB200 loading-end sweep job ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes: ${SLURM_JOB_NODELIST:-unset}; tasks: ${SLURM_NTASKS:-unset}"

"$PYTHON" - <<'PY'
import shutil
from dataclasses import asdict
from pathlib import Path

from tatva.pmma.config import load_case_config
from tatva.pmma.runner import preflight

root = Path("/work/gauss112/tatva")
case_paths = [
    root / "cases" / f"rsf_{run:04d}_loading_interp_{index:02d}.toml"
    for index, run in enumerate(range(128, 144), start=1)
]
configs = [load_case_config(path) for path in case_paths]
baseline = asdict(configs[0])
for path, config in zip(case_paths[1:], configs[1:], strict=True):
    payload = asdict(config)
    for varying_key in ("name", "rsf"):
        payload.pop(varying_key)
    reference = dict(baseline)
    for varying_key in ("name", "rsf"):
        reference.pop(varying_key)
    if payload != reference:
        raise SystemExit(f"Unexpected non-RSF sweep difference in {path.name}.")

estimate = preflight(configs[0])
if not estimate["within_dump_limit"]:
    raise SystemExit("Per-case dump estimate exceeds its configured limit.")
estimated_total = int(estimate["estimated_dump_bytes"]) * len(configs)
free_bytes = shutil.disk_usage(root).free
required_bytes = estimated_total + 50_000_000_000
print(
    f"Sweep storage gate: per_case={estimate['estimated_dump_tb']:.4f} TB; "
    f"total={estimated_total / 1e12:.4f} TB; "
    f"free={free_bytes / 1e12:.4f} TB; required={required_bytes / 1e12:.4f} TB"
)
if free_bytes < required_bytes:
    raise SystemExit("Insufficient free space for the 16-case sweep.")
PY

pids=()
for index in $(seq 1 16); do
  srun --exclusive --exact --mpi=none \
    --nodes=1 --ntasks=1 --cpus-per-task=12 \
    --gpus-per-task=1 --gpu-bind=single:1 \
    env \
      ROOT="$ROOT" \
      ENV_PREFIX="$ENV_PREFIX" \
      SWEEP_INDEX="$index" \
      RUN_TIME_LIMIT_SECONDS="$RUN_TIME_LIMIT_SECONDS" \
      MIN_FREE_BYTES="$MIN_FREE_BYTES" \
      bash "$ROOT/scripts/run_gb200_loading_sweep_rank.sh" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

echo "GB200 loading-end sweep job ended at $(date -u +%Y-%m-%dT%H:%M:%SZ); failed=$failed"
exit "$failed"
