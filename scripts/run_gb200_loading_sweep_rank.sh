#!/bin/bash

set -euo pipefail

ROOT=${ROOT:-/work/gauss112/tatva}
ENV_PREFIX=${ENV_PREFIX:-/work/gauss112/.venvs/tatva-gb200}
PYTHON=$ENV_PREFIX/bin/python
SWEEP_INDEX=${SWEEP_INDEX:?Set SWEEP_INDEX to an integer from 1 through 16.}
RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-55800}
MIN_FREE_BYTES=${MIN_FREE_BYTES:-50000000000}

if (( SWEEP_INDEX < 1 || SWEEP_INDEX > 16 )); then
  echo "SWEEP_INDEX must be between 1 and 16, found $SWEEP_INDEX." >&2
  exit 2
fi

run_number=$((127 + SWEEP_INDEX))
printf -v run_id "TS%04d" "$run_number"
printf -v case_name "rsf_%04d_loading_interp_%02d.toml" \
  "$run_number" "$SWEEP_INDEX"
CASE_FILE="$ROOT/cases/$case_name"
RUN_DIR="$ROOT/runs/$run_id"
LOG_DIR="$RUN_DIR/logs"
LOCK_DIR="$ROOT/.run_locks"

[[ "$(uname -m)" == "aarch64" ]] || {
  echo "GB200 sweep tasks require aarch64, found $(uname -m)." >&2
  exit 1
}
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$CASE_FILE" ]] || { echo "Missing case: $CASE_FILE" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$LOCK_DIR"
exec 9>"$LOCK_DIR/$run_id.lock"
if ! flock -n 9; then
  echo "Another process is already running $run_id; exiting duplicate task."
  exit 0
fi

RESUME_ARGS=()
if [[ -f "$RUN_DIR/status.json" ]]; then
  run_status=$(
    "$PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "$RUN_DIR/status.json"
  )
  case "$run_status" in
    complete)
      echo "$run_id is already complete; nothing to do."
      exit 0
      ;;
    checkpointed)
      [[ -f "$RUN_DIR/checkpoint.npz" ]] || {
        echo "$run_id is checkpointed but checkpoint.npz is missing." >&2
        exit 1
      }
      RESUME_ARGS=(--resume)
      ;;
    failed|running)
      echo "$run_id has unsafe status '$run_status'; refusing automatic HDF5 resume." >&2
      exit 1
      ;;
    *)
      echo "$run_id has non-resumable status '$run_status'; refusing to overwrite it." >&2
      exit 1
      ;;
  esac
fi

export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# JAX CUDA graphs accumulated until they exhausted GB200 driver memory.
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export HDF5_USE_FILE_LOCKING=TRUE
unset LD_LIBRARY_PATH || true

exec > >(tee -a "$LOG_DIR/job.log") 2>&1
cd "$ROOT"

echo "$run_id sweep task started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Case: $CASE_FILE"
echo "Runner time limit: $RUN_TIME_LIMIT_SECONDS seconds"

if [[ -n "${SLURM_PROCID:-}" ]]; then
  expected_index=$((SLURM_PROCID + 1))
  if [[ "$SWEEP_INDEX" != "$expected_index" ]]; then
    echo "Sweep index $SWEEP_INDEX does not match Slurm process $SLURM_PROCID." >&2
    exit 1
  fi
fi

"$PYTHON" - <<'PY'
import jax

devices = jax.devices()
print(f"JAX devices={devices}; independent serial run")
if len(devices) != 1 or devices[0].platform != "gpu":
    raise SystemExit(f"Expected exactly one GPU, found {devices}.")
PY

gpu_id=${CUDA_VISIBLE_DEVICES%%,*}
nvidia-smi -i "$gpu_id" \
  --query-gpu=timestamp,name,uuid,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.sm \
  --format=csv,noheader,nounits -l 2 > "$LOG_DIR/nvidia_smi.csv" &
monitor_pid=$!
simulation_pid=""
guard_pid=""

cleanup() {
  for pid in "$guard_pid" "$monitor_pid"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
forward_checkpoint_signal() {
  if [[ -n "$simulation_pid" ]]; then
    echo "Forwarding checkpoint request to simulation process $simulation_pid"
    kill -USR1 "$simulation_pid" 2>/dev/null || true
  fi
}
disk_guard() {
  while kill -0 "$simulation_pid" 2>/dev/null; do
    free_bytes=$(df --output=avail -B1 "$ROOT" | tail -1 | tr -d ' ')
    if (( free_bytes < MIN_FREE_BYTES )); then
      echo "Free space $free_bytes is below $MIN_FREE_BYTES; requesting checkpoint."
      kill -USR1 "$simulation_pid" 2>/dev/null || true
      return
    fi
    sleep 60
  done
}

trap cleanup EXIT
trap forward_checkpoint_signal USR1 TERM

"$PYTHON" scripts/run_case.py "$CASE_FILE" --preflight
"$PYTHON" scripts/run_case.py "$CASE_FILE" \
  --run-dir "$RUN_DIR" \
  --time-limit-seconds "$RUN_TIME_LIMIT_SECONDS" \
  "${RESUME_ARGS[@]}" &
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
trap - USR1 TERM

if [[ "$simulation_status" -ne 0 ]]; then
  echo "$run_id simulation exited with status $simulation_status." >&2
  exit "$simulation_status"
fi

run_status=$(
  "$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$RUN_DIR/status.json"
)
echo "$run_id finished this allocation with status: $run_status"
echo "$run_id sweep task ended at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
