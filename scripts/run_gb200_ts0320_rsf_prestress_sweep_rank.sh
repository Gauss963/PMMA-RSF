#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/work/gauss112/tatva}
ENV_PREFIX=${ENV_PREFIX:-/work/gauss112/.venvs/tatva-gb200}
PYTHON=$ENV_PREFIX/bin/python
SWEEP_INDEX=${SWEEP_INDEX:?Set SWEEP_INDEX to 1 through 16.}
RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-54000}
MIN_FREE_BYTES=${MIN_FREE_BYTES:-25000000000}
ANALYSIS_SCRIPT=${ANALYSIS_SCRIPT:-scripts/analyze_rsf_prestress_sweep.py}
ANALYSIS_LOG=${ANALYSIS_LOG:-rsf_prestress_analysis.log}

if (( SWEEP_INDEX < 1 || SWEEP_INDEX > 16 )); then
  echo "SWEEP_INDEX must be 1 through 16." >&2
  exit 2
fi

export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export HDF5_USE_FILE_LOCKING=TRUE
unset LD_LIBRARY_PATH || true

[[ "$(uname -m)" == "aarch64" ]] || { echo "GB200 requires aarch64." >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
cd "$ROOT"

"$PYTHON" - <<'PY'
import jax

devices = jax.devices()
print(f"Independent rank JAX devices: {devices}")
if len(devices) != 1 or devices[0].platform != "gpu":
    raise SystemExit("Expected exactly one visible GPU per task.")
PY

if [[ -n ${RANK_PREFLIGHT_SCRIPT:-} ]]; then
  "$PYTHON" "$RANK_PREFLIGHT_SCRIPT"
fi

simulation_pid=""
guard_pid=""
monitor_pid=""

cleanup() {
  for pid in "$guard_pid" "$monitor_pid"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
request_checkpoint() {
  if [[ -n "$simulation_pid" ]]; then
    kill -USR1 "$simulation_pid" 2>/dev/null || true
  fi
}
disk_guard() {
  while kill -0 "$simulation_pid" 2>/dev/null; do
    free_bytes=$(df --output=avail -B1 "$ROOT" | tail -1 | tr -d ' ')
    if (( free_bytes < MIN_FREE_BYTES )); then
      echo "Free space below guard threshold; requesting checkpoint."
      request_checkpoint
      return
    fi
    sleep 60
  done
}
trap cleanup EXIT
trap request_checkpoint USR1 TERM

run_one() {
  local run_number=$1
  local case_name=$2
  local run_id case_file run_dir log_dir run_status simulation_status
  printf -v run_id 'TS%04d' "$run_number"
  case_file="$ROOT/cases/$case_name"
  run_dir="$ROOT/runs/$run_id"
  log_dir="$run_dir/logs"
  [[ -f "$case_file" ]] || { echo "Missing case: $case_file" >&2; return 1; }
  mkdir -p "$log_dir" "$ROOT/.run_locks"
  exec 9>"$ROOT/.run_locks/$run_id.lock"
  flock -n 9 || { echo "Another task owns $run_id; skip duplicate."; return 0; }

  local resume_args=()
  if [[ -f "$run_dir/status.json" ]]; then
    run_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run_dir/status.json")
    case "$run_status" in
      complete) echo "$run_id is already complete."; return 0 ;;
      checkpointed)
        [[ -f "$run_dir/checkpoint.npz" ]] || { echo "Missing $run_id checkpoint." >&2; return 1; }
        resume_args=(--resume)
        ;;
      failed)
        [[ -f "$run_dir/checkpoint.npz" && -f "$run_dir/data/simulation.h5" ]] || {
          echo "Cannot resume $run_id without its checkpoint and dump." >&2
          return 1
        }
        echo "$run_id failed previously; resuming from its last valid checkpoint."
        rm -f "$run_dir/traceback.txt"
        resume_args=(--resume)
        ;;
      *) echo "Unsafe existing status $run_status for $run_id; refusing overwrite." >&2; return 1 ;;
    esac
  fi

  echo "$run_id started at $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
  echo "Case: $case_file; GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
  "$PYTHON" scripts/run_case.py "$case_file" --preflight

  local gpu_id=${CUDA_VISIBLE_DEVICES%%,*}
  nvidia-smi -i "$gpu_id" \
    --query-gpu=timestamp,name,uuid,memory.total,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -l 2 > "$log_dir/nvidia_smi.csv" &
  monitor_pid=$!

  "$PYTHON" scripts/run_case.py "$case_file" \
    --run-dir "$run_dir" \
    --time-limit-seconds "$RUN_TIME_LIMIT_SECONDS" \
    "${resume_args[@]}" > >(tee -a "$log_dir/job.log") 2>&1 &
  simulation_pid=$!
  disk_guard &
  guard_pid=$!
  set +e
  while true; do
    wait "$simulation_pid"
    simulation_status=$?
    if kill -0 "$simulation_pid" 2>/dev/null; then
      continue
    fi
    break
  done
  set -e
  simulation_pid=""
  cleanup
  guard_pid=""
  monitor_pid=""
  if [[ "$simulation_status" -ne 0 ]]; then
    echo "$run_id simulation failed with status $simulation_status." >&2
    return "$simulation_status"
  fi

  run_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run_dir/status.json")
  echo "$run_id ended with status $run_status at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$run_status" == complete ]]; then
    if ! "$PYTHON" "$ANALYSIS_SCRIPT" "$run_dir" \
      > "$log_dir/$ANALYSIS_LOG" 2>&1; then
      echo "$run_id analysis failed; simulation data remains available." >&2
    fi
  fi
  [[ "$run_status" == complete ]]
}

run_number=$((319 + SWEEP_INDEX))
printf -v case_name 'rsf_%04d_rsf_prestress_%02d.toml' "$run_number" "$SWEEP_INDEX"
run_one "${SWEEP_RUN_NUMBER:-$run_number}" "${SWEEP_CASE_FILE:-$case_name}"
