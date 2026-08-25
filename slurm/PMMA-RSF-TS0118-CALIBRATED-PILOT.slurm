#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=8gpus
#SBATCH --job-name=TS0118-G-PILOT
#SBATCH --output=pilot-ts0118-gamma-%j.out
#SBATCH --error=pilot-ts0118-gamma-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/tatva
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
CASE_FILE="$ROOT/cases/pilots/ts0118_ramp290_neutral05_a002_gamma_h100.toml"
case_name=$(basename "$CASE_FILE" .toml)
RUN_DIR="$ROOT/runs/pilots/ts0118/$case_name"

echo "Gamma-preserving pilot: $CASE_FILE"
CASE_FILE="$CASE_FILE" RUN_DIR_OVERRIDE="$RUN_DIR" RUN_TIME_LIMIT_SECONDS=10200 \
  bash "$ROOT/slurm/PMMA-RSF-GPU.slurm"

"$PYTHON" "$ROOT/scripts/evaluate_rsf_displacement_pilot.py" \
  "$RUN_DIR/data/simulation.h5"
MPLCONFIGDIR="/tmp/mpl-${SLURM_JOB_ID}" \
  "$PYTHON" "$ROOT/scripts/plot_rsf_rupture_analysis.py" \
  "$RUN_DIR/data/simulation.h5" \
  --output-dir "$RUN_DIR/stats/rsf_diagnostic" \
  --chunk-frames 1024
