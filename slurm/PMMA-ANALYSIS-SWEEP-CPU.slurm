#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=PMMA-plots
#SBATCH --output=/work1/gauss112/pmma-plots-%j.out
#SBATCH --error=/work1/gauss112/pmma-plots-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work1/gauss112/tatva
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
POSTPROCESS="$ROOT/scripts/postprocess_velocity_weakening_run.py"
RUN_FIRST=${RUN_FIRST:-128}
RUN_LAST=${RUN_LAST:-143}
WORKERS=${WORKERS:-4}

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$POSTPROCESS" ]] || { echo "Missing postprocessor: $POSTPROCESS" >&2; exit 1; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "WORKERS must be positive." >&2; exit 1; }

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

process_run() {
  local run_number=$1
  local run_id
  local run_dir
  local input
  local log
  local status
  run_id=$(printf "TS%04d" "$run_number")
  run_dir="$ROOT/runs/$run_id"
  input="$run_dir/data/simulation.h5"
  log="$run_dir/logs/analysis-${SLURM_JOB_ID:-interactive}.log"
  status="$run_dir/stats/analysis_status.json"

  [[ -f "$input" ]] || { echo "Missing simulation dump: $input" >&2; return 1; }
  mkdir -p "$run_dir/Plot" "$run_dir/logs" "$run_dir/stats"

  (
    exec > >(tee -a "$log") 2>&1
    local started_at
    local start_epoch
    started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_epoch=$(date +%s)
    write_status() {
      local script_exit=$?
      local finished_at
      local elapsed_seconds
      finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      elapsed_seconds=$(( $(date +%s) - start_epoch ))
      cat > "$status" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-interactive}",
  "run_id": "$run_id",
  "started_at": "$started_at",
  "finished_at": "$finished_at",
  "elapsed_seconds": $elapsed_seconds,
  "script_exit": $script_exit,
  "input": "$input",
  "animation_requested": false
}
EOF
    }
    trap write_status EXIT

    echo "Analysis for $run_id started at $started_at"
    echo "Host=$(hostname) job=${SLURM_JOB_ID:-interactive}"
    echo "Animation and animation-frame rendering are disabled."
    set +e
    /usr/bin/time -v "$PYTHON" "$POSTPROCESS" \
      --input "$input" \
      --dpi 260 \
      --missing-only
    analysis_exit=$?
    set -e
    if [[ "$analysis_exit" -eq 0 ]]; then
      echo "Analysis for $run_id completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
      echo "Analysis for $run_id failed with exit $analysis_exit" >&2
    fi
    exit "$analysis_exit"
  )
}

worker() {
  local worker_index=$1
  local run_number
  local worker_failed=0
  for ((run_number = RUN_FIRST + worker_index; run_number <= RUN_LAST; run_number += WORKERS)); do
    if ! process_run "$run_number"; then
      worker_failed=1
    fi
  done
  return "$worker_failed"
}

echo "Analysis sweep ${SLURM_JOB_ID:-interactive}: TS$(printf '%04d' "$RUN_FIRST")-TS$(printf '%04d' "$RUN_LAST")"
echo "Host=$(hostname) allocated_cpus=${SLURM_CPUS_PER_TASK:-unset} workers=$WORKERS"

pids=()
for ((worker_index = 0; worker_index < WORKERS; worker_index++)); do
  worker "$worker_index" &
  pids+=("$!")
done

sweep_failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    sweep_failed=1
  fi
done

if [[ "$sweep_failed" -ne 0 ]]; then
  echo "One or more analyses failed; inspect each run's logs and status." >&2
  exit 1
fi
echo "All requested analyses completed."
