#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=PMMA-plots
#SBATCH --output=/work1/gauss112/pmma-plots-%A_%a.out
#SBATCH --error=/work1/gauss112/pmma-plots-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --array=128-143%4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work1/gauss112/tatva
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
POSTPROCESS="$ROOT/scripts/postprocess_velocity_weakening_run.py"
RUN_ID=$(printf "TS%04d" "$SLURM_ARRAY_TASK_ID")
RUN_DIR="$ROOT/runs/$RUN_ID"
INPUT="$RUN_DIR/data/simulation.h5"
LOG="$RUN_DIR/logs/analysis-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
STATUS="$RUN_DIR/stats/analysis_status.json"

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$POSTPROCESS" ]] || { echo "Missing postprocessor: $POSTPROCESS" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "Missing simulation dump: $INPUT" >&2; exit 1; }

mkdir -p "$RUN_DIR/Plot" "$RUN_DIR/logs" "$RUN_DIR/stats"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_epoch=$(date +%s)
write_status() {
  local script_exit=$?
  local finished_at
  local elapsed_seconds
  finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  elapsed_seconds=$(( $(date +%s) - start_epoch ))
  cat > "$STATUS" <<EOF
{
  "job_id": "${SLURM_ARRAY_JOB_ID:-interactive}_${SLURM_ARRAY_TASK_ID:-unset}",
  "run_id": "$RUN_ID",
  "started_at": "$started_at",
  "finished_at": "$finished_at",
  "elapsed_seconds": $elapsed_seconds,
  "script_exit": $script_exit,
  "input": "$INPUT",
  "animation_requested": false
}
EOF
}
trap write_status EXIT

echo "Analysis for $RUN_ID started at $started_at"
echo "Host=$(hostname) allocated_cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "Animation and animation-frame rendering are disabled."
/usr/bin/time -v "$PYTHON" "$POSTPROCESS" --input "$INPUT" --dpi 260
echo "Analysis for $RUN_ID completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
