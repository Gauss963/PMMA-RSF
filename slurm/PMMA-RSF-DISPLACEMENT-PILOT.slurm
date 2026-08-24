#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=8gpus
#SBATCH --job-name=PMMA-DISP-PILOT
#SBATCH --output=pilot-%A_%a.out
#SBATCH --error=pilot-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/tatva
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
PILOT_CASE_GLOB=${PILOT_CASE_GLOB:-$ROOT/cases/pilots/rsf_disp_h250_d*.toml}
mapfile -t CASE_FILES < <(compgen -G "$PILOT_CASE_GLOB" | sort)
task_id=${SLURM_ARRAY_TASK_ID:-0}

if (( task_id >= ${#CASE_FILES[@]} )); then
  echo "Array index $task_id has no case in $PILOT_CASE_GLOB" >&2
  exit 2
fi

CASE_FILE=${CASE_FILES[$task_id]}
RUN_DIR=$("$PYTHON" "$ROOT/scripts/run_case.py" \
  "$CASE_FILE" --allocate-run-dir)

echo "Pilot case: $CASE_FILE"
CASE_FILE="$CASE_FILE" RUN_DIR_OVERRIDE="$RUN_DIR" RUN_TIME_LIMIT_SECONDS=3300 \
  bash "$ROOT/slurm/PMMA-RSF-GPU.slurm"

"$PYTHON" "$ROOT/scripts/evaluate_rsf_displacement_pilot.py" \
  "$RUN_DIR/data/simulation.h5"
