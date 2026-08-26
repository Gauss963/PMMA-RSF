#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=8gpus
#SBATCH --job-name=PMMA-RSF
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/tatva
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
CASE_FILE=${CASE_FILE:-$ROOT/cases/rsf_0119_q4_explicit_10h.toml}
RESUME_DIR=${RESUME_DIR:-}
RUN_DIR_OVERRIDE=${RUN_DIR_OVERRIDE:-}
RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-34800}
if [[ -n "$RESUME_DIR" ]]; then
  RUN_DIR="$RESUME_DIR"
  RESUME_ARGS=(--resume)
elif [[ -n "$RUN_DIR_OVERRIDE" ]]; then
  RUN_DIR="$RUN_DIR_OVERRIDE"
  RESUME_ARGS=()
else
  RUN_DIR=$("$PYTHON" "$ROOT/scripts/run_case.py" \
    "$CASE_FILE" --allocate-run-dir)
  RESUME_ARGS=()
fi
LOG_DIR="$RUN_DIR/logs"

module purge
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.94
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset LD_LIBRARY_PATH || true

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/job.log") 2>&1

cd "$ROOT"
echo "PMMA RSF job ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Case: $CASE_FILE"
echo "Run directory: $RUN_DIR"
echo "Runner time limit: $RUN_TIME_LIMIT_SECONDS seconds"
nvidia-smi -L
"$PYTHON" - <<'PY'
import jax
import h5py
import tatva

devices = jax.devices()
print(
    f"JAX {jax.__version__}; h5py={h5py.__version__}; "
    f"devices={devices}; Tatva={tatva.__file__}"
)
if len(devices) != 1 or devices[0].platform != "gpu":
    raise SystemExit(f"Expected exactly one GPU, found {devices}")
PY

gpu_id=${CUDA_VISIBLE_DEVICES%%,*}
nvidia-smi -i "$gpu_id" \
  --query-gpu=timestamp,name,uuid,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.sm \
  --format=csv,noheader,nounits -l 2 > "$LOG_DIR/nvidia_smi.csv" &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

"$PYTHON" scripts/run_case.py "$CASE_FILE" --preflight
"$PYTHON" scripts/run_case.py "$CASE_FILE" \
  --run-dir "$RUN_DIR" \
  --time-limit-seconds "$RUN_TIME_LIMIT_SECONDS" \
  "${RESUME_ARGS[@]}" &
simulation_pid=$!
forward_checkpoint_signal() {
  echo "Forwarding checkpoint request to simulation process $simulation_pid"
  kill -USR1 "$simulation_pid" 2>/dev/null || true
}
trap forward_checkpoint_signal USR1 TERM
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
  "$PYTHON" - "$RUN_DIR/status.json" "$simulation_status" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

status_path = Path(sys.argv[1])
exit_code = int(sys.argv[2])
try:
    status = json.loads(status_path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    status = {}
status.update(
    status="failed",
    failed_utc=datetime.now(timezone.utc).isoformat(),
    native_exit_code=exit_code,
    error=f"Simulation process exited with native status {exit_code}.",
)
status_path.parent.mkdir(parents=True, exist_ok=True)
status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
PY
  echo "Simulation exited with status $simulation_status" >&2
  exit "$simulation_status"
fi

run_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/status.json")
echo "Run status: $run_status"
if [[ "$run_status" == "checkpointed" ]]; then
  echo "Resume with RESUME_DIR=$RUN_DIR and the same CASE_FILE."
fi

cleanup_monitor
trap - EXIT

echo "PMMA RSF job finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
