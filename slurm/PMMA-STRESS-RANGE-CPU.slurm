#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=PMMA-range
#SBATCH --output=/work1/gauss112/pmma-stress-range-%j.out
#SBATCH --error=/work1/gauss112/pmma-stress-range-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=18G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work1/gauss112/PMMA-RSF
RUN_DIR=${RUN_DIR:?Submit with --export=ALL,RUN_DIR=/path/to/TS####}
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
RENDERER="$ROOT/scripts/render_stress_frames.py"
INPUT="$RUN_DIR/data/simulation.h5"
FRAMES_DIR="$RUN_DIR/Plot/stress_triptych_frames"
LOG="$RUN_DIR/logs/stress-range-${SLURM_JOB_ID:-interactive}.log"

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$RENDERER" ]] || { echo "Missing renderer: $RENDERER" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "Missing simulation dump: $INPUT" >&2; exit 1; }

mkdir -p "$FRAMES_DIR" "$RUN_DIR/logs"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Stress-range scan ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
/usr/bin/time -v "$PYTHON" "$RENDERER" \
  --input "$INPUT" \
  --frames-dir "$FRAMES_DIR" \
  --stress-percentile 99.5 \
  --stress-mode stress_triptych \
  --ranges-only
echo "Stress-range scan completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
