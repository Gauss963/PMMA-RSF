#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=PMMA-Q4-post56
#SBATCH --output=/work1/gauss112/pmma-q4-postprocess-%j.out
#SBATCH --error=/work1/gauss112/pmma-q4-postprocess-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --hint=nomultithread
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -uo pipefail

ROOT=/work1/gauss112/tatva
RUN_DIR=${RUN_DIR:?Submit with --export=ALL,RUN_DIR=/path/to/TS####_case}
WORKERS=${WORKERS:-56}
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
FFMPEG_DIR=/home/gauss112/.conda/envs/FFmpeg/bin
POSTPROCESS="$ROOT/scripts/postprocess_velocity_weakening_run.py"
RENDERER="$ROOT/scripts/render_stress_frames.py"
INPUT="$RUN_DIR/data/simulation.h5"
PLOT_DIR="$RUN_DIR/Plot"
FRAMES_DIR="$PLOT_DIR/stress_triptych_frames"
VIDEO="$PLOT_DIR/stress_triptych_60fps.mp4"
LOG_DIR="$RUN_DIR/logs"
LOG="$LOG_DIR/postprocess-${SLURM_JOB_ID:-interactive}.log"
STATUS="$RUN_DIR/stats/postprocess_status.json"

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -x "$FFMPEG_DIR/ffmpeg" ]] || { echo "Missing FFmpeg: $FFMPEG_DIR/ffmpeg" >&2; exit 1; }
[[ -f "$POSTPROCESS" ]] || { echo "Missing postprocessor: $POSTPROCESS" >&2; exit 1; }
[[ -f "$RENDERER" ]] || { echo "Missing renderer: $RENDERER" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "Missing simulation dump: $INPUT" >&2; exit 1; }
[[ "$WORKERS" -eq "${SLURM_CPUS_PER_TASK:-$WORKERS}" ]] || {
  echo "WORKERS=$WORKERS must equal SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}." >&2
  exit 1
}

mkdir -p "$PLOT_DIR" "$FRAMES_DIR" "$LOG_DIR" "$RUN_DIR/stats"
exec > >(tee -a "$LOG") 2>&1

export PATH="$FFMPEG_DIR:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_epoch=$(date +%s)
analysis_exit=null
animation_exit=null

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
  "analysis_exit": $analysis_exit,
  "animation_exit": $animation_exit,
  "workers": $WORKERS,
  "input": "$INPUT",
  "video": "$VIDEO"
}
EOF
}
trap write_status EXIT

echo "PMMA Q4 post-processing started at $started_at"
echo "Host=$(hostname) CPUs=${SLURM_CPUS_PER_TASK:-unset} workers=$WORKERS"
echo "Input=$INPUT"
"$PYTHON" --version
"$FFMPEG_DIR/ffmpeg" -version | head -n 1

set +e
/usr/bin/time -v "$PYTHON" "$POSTPROCESS" --input "$INPUT" --dpi 260
analysis_exit=$?

/usr/bin/time -v "$PYTHON" "$RENDERER" \
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
animation_exit=$?
set -e

echo "Analysis exit=$analysis_exit animation exit=$animation_exit"
echo "Plots=$PLOT_DIR"
echo "Video=$VIDEO"
if [[ "$analysis_exit" -ne 0 || "$animation_exit" -ne 0 ]]; then
  exit 1
fi
