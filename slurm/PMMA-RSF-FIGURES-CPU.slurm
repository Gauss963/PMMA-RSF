#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=PMMA-RSF-fig
#SBATCH --output=/work1/gauss112/pmma-rsf-figures-%j.out
#SBATCH --error=/work1/gauss112/pmma-rsf-figures-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work1/gauss112/PMMA-RSF
RUN_DIR=${RUN_DIR:?Submit with --export=ALL,RUN_DIR=/path/to/TS####}
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
PLOTTER="$ROOT/scripts/plot_rsf_rupture_analysis.py"
INPUT="$RUN_DIR/data/simulation.h5"
PLOT_DIR="$RUN_DIR/Plot"
LOG="$RUN_DIR/logs/rsf-figures-${SLURM_JOB_ID:-interactive}.log"

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$PLOTTER" ]] || { echo "Missing plotter: $PLOTTER" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "Missing simulation dump: $INPUT" >&2; exit 1; }

mkdir -p "$PLOT_DIR" "$RUN_DIR/logs"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "RSF figure job ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
/usr/bin/time -v "$PYTHON" "$PLOTTER" \
  "$INPUT" \
  --output-dir "$PLOT_DIR" \
  --fit-start 120 \
  --fit-end 440 \
  --dpi 260
echo "RSF figures completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
