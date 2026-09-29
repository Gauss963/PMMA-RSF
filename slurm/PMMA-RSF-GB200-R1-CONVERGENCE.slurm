#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=gb200-r1
#SBATCH --job-name=PMMA-DT-0368-0383
#SBATCH --output=gb200-convergence-%j.out
#SBATCH --error=gb200-convergence-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@1800

set -euo pipefail
ROOT=${ROOT:-/work/gauss112/tatva}
ENV_PREFIX=${ENV_PREFIX:-/work/gauss112/.venvs/tatva-gb200}
export PATH="$ENV_PREFIX/bin:$PATH" PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1
export MIN_FREE_BYTES=25000000000
export ANALYSIS_SCRIPT=scripts/analyze_rsf_convergence_sweep.py
export ANALYSIS_LOG=rsf_convergence_analysis.log
export RANK_PREFLIGHT_SCRIPT=scripts/validate_rsf_convergence_backend.py
export ROOT ENV_PREFIX
unset LD_LIBRARY_PATH || true
cd "$ROOT"
if [[ ${SKIP_PILOT:-0} == 1 ]]; then
  : "${HARD_DEADLINE_EPOCH:?Direct launch requires the user-approved absolute deadline}"
  unset PILOT_REPORT
  echo "Direct production launch authorized; no queued timing pilot dependency."
fi
if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  export RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-82800}
else
  export RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-$(python scripts/convergence_walltime.py)}
fi
python scripts/preflight_convergence_sweep.py
if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then exit 0; fi
if [[ ${SKIP_PILOT:-0} != 1 ]]; then
  : "${PILOT_REPORT:?Supply the successful GB200 convergence pilot report}"
fi
echo "Convergence sweep ${SLURM_JOB_ID}, code $(git rev-parse HEAD), $(date -Is)"
set +e
srun --exact --mpi=none --kill-on-bad-exit=0 --wait=0 \
  --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=12 \
  --gpus-per-task=1 --gpu-bind=single:1 \
  bash -c '
    export SWEEP_INDEX=$((SLURM_PROCID + 1))
    export SWEEP_RUN_NUMBER=$((367 + SWEEP_INDEX))
    printf -v SWEEP_CASE_FILE "rsf_%04d_convergence_%02d.toml" "$SWEEP_RUN_NUMBER" "$SWEEP_INDEX"
    export SWEEP_CASE_FILE
    exec bash "$ROOT/scripts/run_gb200_ts0320_rsf_prestress_sweep_rank.sh"
  '
status=$?
set -e
python "$ANALYSIS_SCRIPT" --summarize "$ROOT"
exit "$status"
