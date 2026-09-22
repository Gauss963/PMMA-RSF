#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=gb200-r1
#SBATCH --job-name=PMMA-RSF-NUCLEATION
#SBATCH --output=gb200-nucleation-sweep-%j.out
#SBATCH --error=gb200-nucleation-sweep-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=16:00:00
#SBATCH --signal=USR1@1800

set -euo pipefail

ROOT=/work/gauss112/tatva
ENV_PREFIX=/work/gauss112/.venvs/tatva-gb200
PYTHON=$ENV_PREFIX/bin/python
RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-54000}
MIN_FREE_BYTES=${MIN_FREE_BYTES:-25000000000}
ANALYSIS_SCRIPT=scripts/analyze_rsf_nucleation_sweep.py
ANALYSIS_LOG=rsf_nucleation_analysis.log

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset LD_LIBRARY_PATH || true
cd "$ROOT"

echo "Continuous RSF 4x4 sweep job ${SLURM_JOB_ID:-preflight} at $(date -Is)"
"$PYTHON" scripts/generate_ts0336_nucleation_sweep_cases.py --check
"$PYTHON" - <<'PY'
import json
import math
import shutil
from pathlib import Path

from scripts.generate_ts0336_nucleation_sweep_cases import (
    CASE_COUNT, FIRST_RUN, case_path,
)
from tatva.pmma.config import load_case_config
from tatva.pmma.runner import preflight

root = Path('/work/gauss112/tatva')
if CASE_COUNT != 16:
    raise SystemExit(f'Expected exactly 16 cases, got {CASE_COUNT}.')
paths = [case_path(index) for index in range(1, CASE_COUNT + 1)]
configs = [load_case_config(path) for path in paths]
estimates = [preflight(config) for config in configs]
for config in configs:
    loading = config.loading
    assert config.rsf.initial_state_mode == 'steady-state'
    assert config.rsf.target_normalized_prestress is None
    assert loading.prestress_shear_displacement is None
    assert loading.quasistatic_shear_fraction == 0.0
    assert loading.effective_normal_relaxation_time is None
    assert loading.shear_displacement_initial == 0.0
    assert loading.shear_displacement_final == 2.45
    assert loading.stop_on_rupture and loading.stop_velocity == 500.0
    assert loading.stop_coverage_fraction is None
    assert (loading.stop_min_y, loading.stop_max_y) == (5.0, 25.0)
if not all(estimate['within_dump_limit'] for estimate in estimates):
    raise SystemExit('A case exceeds its per-run dump limit.')
# Gate on the uncompressed estimate; compression savings are not required.
total = sum(int(estimate['estimated_uncompressed_bytes']) for estimate in estimates)
if total > 1_400_000_000_000:
    raise SystemExit(f'Sweep dump estimate {total / 1e12:.3f} TB exceeds 1.4 TB.')
remaining = 0
for index, estimate in enumerate(estimates):
    run_dir = root / 'runs' / f'TS{FIRST_RUN + index:04d}'
    status_path = run_dir / 'status.json'
    status = json.loads(status_path.read_text())['status'] if status_path.exists() else None
    if status == 'complete':
        continue
    dump = run_dir / 'data/simulation.h5'
    checkpoint = run_dir / 'checkpoint.npz'
    if status in ('failed', 'checkpointed'):
        if not (dump.is_file() and checkpoint.is_file()):
            raise SystemExit(f'{run_dir} lacks a resumable dump/checkpoint pair.')
        existing = dump.stat().st_size
    elif status is None:
        if dump.exists() or checkpoint.exists():
            raise SystemExit(f'{run_dir} contains data without a status record.')
        existing = 0
    else:
        raise SystemExit(f'Unsafe existing status {status!r} for {run_dir}.')
    remaining += max(int(estimate['estimated_uncompressed_bytes']) - existing, 0)
free = shutil.disk_usage(root).free
required = remaining + 50_000_000_000
print(f'Storage: total={total / 1e12:.3f} TB; remaining={remaining / 1e12:.3f} TB; '
      f'free={free / 1e12:.3f} TB; required={required / 1e12:.3f} TB')
if free < required:
    raise SystemExit('Insufficient free space for the complete sweep.')
PY

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  exit 0
fi

export ROOT ENV_PREFIX RUN_TIME_LIMIT_SECONDS MIN_FREE_BYTES ANALYSIS_SCRIPT ANALYSIS_LOG
set +e
srun --exact --mpi=none --kill-on-bad-exit=0 --wait=0 \
  --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=12 \
  --gpus-per-task=1 --gpu-bind=single:1 \
  bash -c '
    export SWEEP_INDEX=$((SLURM_PROCID + 1))
    export SWEEP_RUN_NUMBER=$((335 + SWEEP_INDEX))
    printf -v SWEEP_CASE_FILE "rsf_%04d_nucleation_%02d.toml" "$SWEEP_RUN_NUMBER" "$SWEEP_INDEX"
    export SWEEP_CASE_FILE
    exec bash "$ROOT/scripts/run_gb200_ts0320_rsf_prestress_sweep_rank.sh"
  '
sweep_status=$?
set -e

"$PYTHON" "$ANALYSIS_SCRIPT" --summarize "$ROOT"
echo "Continuous RSF sweep finished at $(date -Is); status=$sweep_status"
exit "$sweep_status"
