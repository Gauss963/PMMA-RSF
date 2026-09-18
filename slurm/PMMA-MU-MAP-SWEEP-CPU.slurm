#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=PMMA-mu-maps
#SBATCH --output=/work1/gauss112/pmma-mu-maps-%j.out
#SBATCH --error=/work1/gauss112/pmma-mu-maps-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work1/gauss112/tatva
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
PLOT_SCRIPT="$ROOT/scripts/plot_contact_friction_map.py"
RUN_FIRST=${RUN_FIRST:-272}
RUN_LAST=${RUN_LAST:-287}
WORKERS=${WORKERS:-4}

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$PLOT_SCRIPT" ]] || { echo "Missing plotter: $PLOT_SCRIPT" >&2; exit 1; }
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
  run_id=$(printf "TS%04d" "$run_number")
  run_dir="$ROOT/runs/$run_id"
  input="$run_dir/data/simulation.h5"
  log="$run_dir/logs/mu-map-${SLURM_JOB_ID:-interactive}.log"

  [[ -f "$input" ]] || { echo "Missing simulation dump: $input" >&2; return 1; }
  mkdir -p "$run_dir/Plot" "$run_dir/logs"

  (
    exec > >(tee -a "$log") 2>&1
    echo "Friction maps for $run_id started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$PYTHON" "$PLOT_SCRIPT" \
      --input "$input" \
      --output "$run_dir/Plot/mu_eff_map.pdf" \
      --phase-split-output "$run_dir/Plot/mu_eff_map_phase_split.pdf"
    for output in \
      "$run_dir/Plot/mu_eff_map.pdf" \
      "$run_dir/Plot/mu_eff_map.png" \
      "$run_dir/Plot/mu_eff_map_phase_split.pdf" \
      "$run_dir/Plot/mu_eff_map_phase_split.png"; do
      [[ -s "$output" ]] || { echo "Missing output: $output" >&2; exit 1; }
    done
    echo "Friction maps for $run_id completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

echo "Friction-map sweep: TS$(printf '%04d' "$RUN_FIRST")-TS$(printf '%04d' "$RUN_LAST")"
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
  echo "One or more friction-map tasks failed; inspect the per-run logs." >&2
  exit 1
fi
echo "All requested friction maps completed."
