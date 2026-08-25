#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=PMMA-analysis
#SBATCH --output=/work1/gauss112/pmma-analysis-%j.out
#SBATCH --error=/work1/gauss112/pmma-analysis-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=144G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work1/gauss112/PMMA-RSF
RUN_DIR=${RUN_DIR:?Submit with --export=ALL,RUN_DIR=/path/to/TS####}
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
POSTPROCESS="$ROOT/scripts/postprocess_velocity_weakening_run.py"
INPUT="$RUN_DIR/data/simulation.h5"
LOG="$RUN_DIR/logs/analysis-${SLURM_JOB_ID:-interactive}.log"
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

echo "Analysis job ${SLURM_JOB_ID:-interactive} started at $started_at"
echo "Host=$(hostname) allocated_cpus=${SLURM_CPUS_PER_TASK:-unset} memory=144G"
/usr/bin/time -v "$PYTHON" "$POSTPROCESS" --input "$INPUT" --dpi 260
echo "Analysis completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
