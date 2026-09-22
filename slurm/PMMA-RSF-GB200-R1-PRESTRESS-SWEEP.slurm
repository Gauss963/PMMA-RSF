#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=gb200-r1
#SBATCH --job-name=PMMA-RSF-PRESTRESS
#SBATCH --output=gb200-rsf-prestress-sweep-%j.out
#SBATCH --error=gb200-rsf-prestress-sweep-%j.err
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
MIN_FREE_BYTES=${MIN_FREE_BYTES:-25000000000}

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset LD_LIBRARY_PATH || true
cd "$ROOT"

echo "RSF-prestress sweep job ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes: ${SLURM_JOB_NODELIST:-unset}; tasks: ${SLURM_NTASKS:-unset}"

"$PYTHON" scripts/generate_ts0320_rsf_prestress_sweep_cases.py --check
"$PYTHON" - <<'PY'
import json
import math
import shutil
from pathlib import Path

from scripts.generate_ts0320_rsf_prestress_sweep_cases import (
    CASE_COUNT,
    DYNAMIC_SHEAR_INCREMENT_MM,
    case_path,
    normalized_prestress,
)
from tatva.pmma.config import load_case_config
from tatva.pmma.runner import preflight

root = Path('/work/gauss112/tatva')
if CASE_COUNT != 16:
    raise SystemExit(f'Expected exactly 16 cases, got {CASE_COUNT}.')
paths = [case_path(index) for index in range(1, CASE_COUNT + 1)]
configs = [load_case_config(path) for path in paths]
estimates = [preflight(config) for config in configs]
for index, config in enumerate(configs, start=1):
    loading = config.loading
    if config.rsf.initial_state_mode != 'traction-consistent-handoff':
        raise SystemExit(f'{paths[index - 1]} lacks traction-consistent handoff.')
    if not math.isclose(
        config.rsf.target_normalized_prestress,
        normalized_prestress(index),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise SystemExit(f'{paths[index - 1]} has the wrong prestress target.')
    if not math.isclose(
        loading.shear_displacement_final - loading.prestress_shear_displacement,
        DYNAMIC_SHEAR_INCREMENT_MM,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise SystemExit(f'{paths[index - 1]} changes the dynamic increment.')
    if loading.stop_velocity != 500.0 or loading.stop_coverage_fraction is not None:
        raise SystemExit(f'{paths[index - 1]} has the wrong stop criterion.')
if not all(estimate['within_dump_limit'] for estimate in estimates):
    raise SystemExit('At least one case exceeds its per-run dump limit.')
total = sum(int(estimate['estimated_dump_bytes']) for estimate in estimates)
if total > 1_400_000_000_000:
    raise SystemExit(f'Estimated sweep dump {total / 1e12:.3f} TB exceeds 1.4 TB.')
remaining = 0
for path, estimate in zip(paths, estimates):
    run_dir = root / 'runs' / f'TS{int(path.stem.split("_")[1]):04d}'
    status_path = run_dir / 'status.json'
    status = json.loads(status_path.read_text())['status'] if status_path.exists() else None
    if status == 'complete':
        continue
    dump = run_dir / 'data' / 'simulation.h5'
    checkpoint = run_dir / 'checkpoint.npz'
    if status in ('failed', 'checkpointed'):
        if not (dump.is_file() and checkpoint.is_file()):
            raise SystemExit(f'{run_dir} cannot resume without both dump and checkpoint.')
        existing = dump.stat().st_size
    elif status is None:
        existing = 0
    else:
        raise SystemExit(f'Unsafe existing status {status!r} for {run_dir}.')
    remaining += max(int(estimate['estimated_dump_bytes']) - existing, 0)
free = shutil.disk_usage(root).free
required = remaining + 25_000_000_000
print(f'RSF-prestress storage gate: total={total / 1e12:.3f} TB, '
      f'remaining={remaining / 1e12:.3f} TB, '
      f'free={free / 1e12:.3f} TB, required={required / 1e12:.3f} TB')
if free < required:
    raise SystemExit('Insufficient free space for remaining sweep output.')
PY

export ROOT ENV_PREFIX RUN_TIME_LIMIT_SECONDS MIN_FREE_BYTES
set +e
srun --exact --mpi=none --kill-on-bad-exit=0 --wait=0 \
  --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=12 \
  --gpus-per-task=1 --gpu-bind=single:1 \
  bash -c '
    export SWEEP_INDEX=$((SLURM_PROCID + 1))
    exec bash "$ROOT/scripts/run_gb200_ts0320_rsf_prestress_sweep_rank.sh"
  '
sweep_status=$?
set -e

"$PYTHON" scripts/summarize_rsf_prestress_sweep.py
echo "RSF-prestress sweep ended at $(date -u +%Y-%m-%dT%H:%M:%SZ); srun status=$sweep_status"
exit "$sweep_status"
