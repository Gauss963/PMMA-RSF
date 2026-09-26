#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=gb200-r1
#SBATCH --job-name=PMMA-RSF-EXTENSION
#SBATCH --output=gb200-extension-sweep-%j.out
#SBATCH --error=gb200-extension-sweep-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=16:00:00
#SBATCH --signal=USR1@1800

set -euo pipefail
ROOT=${ROOT:-/work/gauss112/tatva}
ENV_PREFIX=${ENV_PREFIX:-/work/gauss112/.venvs/tatva-gb200}
PYTHON=$ENV_PREFIX/bin/python
RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-54000}
MIN_FREE_BYTES=${MIN_FREE_BYTES:-25000000000}
ANALYSIS_SCRIPT=scripts/analyze_rsf_extension_sweep.py
ANALYSIS_LOG=rsf_extension_analysis.log
RANK_PREFLIGHT_SCRIPT=scripts/validate_rsf_extension_backend.py
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
export ROOT
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset LD_LIBRARY_PATH || true
cd "$ROOT"
echo "Extension sweep job ${SLURM_JOB_ID:-preflight} at $(date -Is)"
"$PYTHON" scripts/generate_ts0352_extension_sweep_cases.py --check
"$PYTHON" - <<'PY'
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from scripts.generate_ts0352_extension_sweep_cases import CASE_COUNT, FIRST_RUN, LENGTHS_MM, TEMPLATE, case_path
from tatva.pmma.config import load_case_config
from tatva.pmma.runner import preflight

root = Path(os.environ['ROOT'])
baseline = load_case_config(TEMPLATE)
assert CASE_COUNT == 16 and LENGTHS_MM == tuple(float(2*i) for i in range(16))
total = remaining = 0
for index, length in enumerate(LENGTHS_MM, 1):
    config = load_case_config(case_path(index))
    assert replace(config, name=baseline.name, moving=replace(config.moving, loading_extension_length=0.0)) == baseline
    assert config.moving.loading_extension_length == length
    estimate = preflight(config)
    assert estimate['within_dump_limit'] and not estimate['animation_enabled']
    assert estimate['fault_nodes'] == 1001
    size = int(estimate['estimated_uncompressed_bytes'])
    total += size
    run = root / 'runs' / f'TS{FIRST_RUN + index - 1:04d}'
    status_file = run / 'status.json'
    status = json.loads(status_file.read_text())['status'] if status_file.exists() else None
    dump, checkpoint = run / 'data/simulation.h5', run / 'checkpoint.npz'
    if status == 'complete':
        continue
    if status in ('failed', 'checkpointed'):
        if not (dump.is_file() and checkpoint.is_file()):
            raise SystemExit(f'{run}: no valid dump/checkpoint pair; refusing overwrite')
        remaining += max(size - dump.stat().st_size, 0)
    elif status is None and not dump.exists() and not checkpoint.exists():
        remaining += size
    else:
        raise SystemExit(f'{run}: unsafe existing status {status!r}')
assert total <= 1_400_000_000_000
free = shutil.disk_usage(root).free
print(f'16 cases: {total / 1e12:.4f} TB uncompressed; {remaining / 1e12:.4f} TB remaining; {free / 1e12:.4f} TB free')
if free < remaining + 50_000_000_000:
    raise SystemExit('Insufficient space including the 50 GB reserve')
PY
if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  exit 0
fi
export ROOT ENV_PREFIX RUN_TIME_LIMIT_SECONDS MIN_FREE_BYTES ANALYSIS_SCRIPT ANALYSIS_LOG
export RANK_PREFLIGHT_SCRIPT
set +e
srun --exact --mpi=none --kill-on-bad-exit=0 --wait=0 \
  --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=12 \
  --gpus-per-task=1 --gpu-bind=single:1 \
  bash -c '
    export SWEEP_INDEX=$((SLURM_PROCID + 1))
    export SWEEP_RUN_NUMBER=$((351 + SWEEP_INDEX))
    printf -v SWEEP_CASE_FILE "rsf_%04d_extension_%02d.toml" "$SWEEP_RUN_NUMBER" "$SWEEP_INDEX"
    export SWEEP_CASE_FILE
    exec bash "$ROOT/scripts/run_gb200_ts0320_rsf_prestress_sweep_rank.sh"
  '
sweep_status=$?
set -e
"$PYTHON" "$ANALYSIS_SCRIPT" --summarize "$ROOT"
echo "Extension sweep finished at $(date -Is); status=$sweep_status"
exit "$sweep_status"
