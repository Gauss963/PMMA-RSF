#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=8gpus
#SBATCH --job-name=TS0118-L-PILOT
#SBATCH --output=pilot-ts0118-loading-%A_%a.out
#SBATCH --error=pilot-ts0118-loading-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-1%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/tatva
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
mapfile -t CASE_FILES < <(
  compgen -G "$ROOT/cases/pilots/ts0118_ramp290_loading???_middle002_gamma_h100.toml" | sort
)
task_id=${SLURM_ARRAY_TASK_ID:-0}

if (( task_id >= ${#CASE_FILES[@]} )); then
  echo "Array index $task_id has no loading-neutral pilot case." >&2
  exit 2
fi

CASE_FILE=${CASE_FILES[$task_id]}
case_name=$(basename "$CASE_FILE" .toml)
RUN_DIR="$ROOT/runs/pilots/ts0118/$case_name"

echo "Loading-neutral pilot: $CASE_FILE"
CASE_FILE="$CASE_FILE" RUN_DIR_OVERRIDE="$RUN_DIR" RUN_TIME_LIMIT_SECONDS=10200 \
  bash "$ROOT/slurm/PMMA-RSF-GPU.slurm"

"$PYTHON" "$ROOT/scripts/evaluate_rsf_displacement_pilot.py" \
  "$RUN_DIR/data/simulation.h5"
