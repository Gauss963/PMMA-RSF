#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=PMMA-anim56
#SBATCH --output=/work1/gauss112/pmma-animation-%j.out
#SBATCH --error=/work1/gauss112/pmma-animation-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work1/gauss112/PMMA-RSF
RUN_DIR=${RUN_DIR:?Submit with --export=ALL,RUN_DIR=/path/to/TS####}
WORKERS=${WORKERS:-56}
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
FFMPEG_DIR=/home/gauss112/.conda/envs/FFmpeg/bin
RENDERER="$ROOT/scripts/render_stress_frames.py"
INPUT="$RUN_DIR/data/simulation.h5"
PLOT_DIR="$RUN_DIR/Plot"
FRAMES_DIR="$PLOT_DIR/stress_triptych_frames"
VIDEO="$PLOT_DIR/stress_triptych_60fps.mp4"
LOG="$RUN_DIR/logs/animation-${SLURM_JOB_ID:-interactive}.log"

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -x "$FFMPEG_DIR/ffmpeg" ]] || { echo "Missing FFmpeg: $FFMPEG_DIR/ffmpeg" >&2; exit 1; }
[[ -f "$RENDERER" ]] || { echo "Missing renderer: $RENDERER" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "Missing simulation dump: $INPUT" >&2; exit 1; }
[[ "$WORKERS" -eq "${SLURM_CPUS_PER_TASK:-$WORKERS}" ]] || {
  echo "WORKERS=$WORKERS must equal SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}." >&2
  exit 1
}

mkdir -p "$PLOT_DIR" "$FRAMES_DIR" "$RUN_DIR/logs"
exec > >(tee -a "$LOG") 2>&1

export PATH="$FFMPEG_DIR:$PATH"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Animation job ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host=$(hostname) CPUs=${SLURM_CPUS_PER_TASK:-unset} workers=$WORKERS"
echo "Input=$INPUT"
"$FFMPEG_DIR/ffmpeg" -version | head -n 1

"$PYTHON" "$RENDERER" \
  --input "$INPUT" \
  --frames-dir "$FRAMES_DIR" \
  --video "$VIDEO" \
  --workers "$WORKERS" \
  --fps 60 \
  --dpi 320 \
  --width 15.75 \
  --height 6.75 \
  --stress-percentile 99.5 \
  --margin 8.0 \
  --stress-mode stress_triptych \
  --no-swap-axes

echo "Animation completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Video=$VIDEO"
