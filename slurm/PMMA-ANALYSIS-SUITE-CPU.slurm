#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=PMMA-suite
#SBATCH --output=/work1/gauss112/pmma-analysis-suite-%j.out
#SBATCH --error=/work1/gauss112/pmma-analysis-suite-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work1/gauss112/PMMA-RSF
RUN_DIR=${RUN_DIR:?Submit with --export=ALL,RUN_DIR=/path/to/TS####}
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
ANALYZER="$ROOT/scripts/run_velocity_weakening_analysis_suite.py"
INPUT="$RUN_DIR/data/simulation.h5"
LOG="$RUN_DIR/logs/analysis-suite-${SLURM_JOB_ID:-interactive}.log"
STATUS="$RUN_DIR/stats/analysis_suite_status.json"

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$ANALYZER" ]] || { echo "Missing analyzer: $ANALYZER" >&2; exit 1; }
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
  "job_id": "${SLURM_JOB_ID:-interactive}",
  "started_at": "$started_at",
  "finished_at": "$finished_at",
  "elapsed_seconds": $elapsed_seconds,
  "script_exit": $script_exit,
  "input": "$INPUT"
}
EOF
}
trap write_status EXIT

echo "Analysis suite job ${SLURM_JOB_ID:-interactive} started at $started_at"
echo "Host=$(hostname) allocated_cpus=${SLURM_CPUS_PER_TASK:-unset}"
/usr/bin/time -v "$PYTHON" "$ANALYZER" \
  --input "$INPUT" \
  --fit-start 120 \
  --fit-end 440 \
  --dpi 260
echo "Analysis suite completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
