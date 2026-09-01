#!/bin/bash
#SBATCH --partition=cpu-2g
#SBATCH --job-name=TS0126-CPU
#SBATCH --output=zinfandel-ts0126-%j.out
#SBATCH --error=zinfandel-ts0126-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=450G
#SBATCH --time=28-00:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail

ROOT=${ROOT:-$HOME/PMMA-RSF}
PYTHON=${PYTHON:-$HOME/miniconda3/envs/tatva/bin/python}
CASE_FILE=${CASE_FILE:-$ROOT/cases/rsf_0126_q4_chamfer20x5_zinfandel_sparse.toml}
RUN_DIR_OVERRIDE=${RUN_DIR_OVERRIDE:-$ROOT/runs-zinfandel/TS0126}
RESUME_DIR=${RESUME_DIR:-}
RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-2418000}
MPI_RANKS=${MPI_RANKS:-24}

if (( MPI_RANKS < 1 || SLURM_CPUS_PER_TASK % MPI_RANKS != 0 )); then
  echo "MPI_RANKS must divide SLURM_CPUS_PER_TASK exactly." >&2
  exit 1
fi
THREADS_PER_RANK=$((SLURM_CPUS_PER_TASK / MPI_RANKS))

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }

module purge
module load mpi/2021.15
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
/opt/bin/sbatch_pre.sh

if [[ -n "$RESUME_DIR" ]]; then
  RUN_DIR="$RESUME_DIR"
  RESUME_ARGS=(--resume)
else
  RUN_DIR="$RUN_DIR_OVERRIDE"
  RESUME_ARGS=()
fi
LOG_DIR="$RUN_DIR/logs"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=0
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="$THREADS_PER_RANK"
export OPENBLAS_NUM_THREADS="$THREADS_PER_RANK"
export MKL_NUM_THREADS="$THREADS_PER_RANK"
export NUMEXPR_NUM_THREADS="$THREADS_PER_RANK"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$THREADS_PER_RANK"
export I_MPI_FABRICS=shm
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/job.log") 2>&1

cd "$ROOT"
echo "PMMA TS0126 CPU job ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname); CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo "MPI ranks: $MPI_RANKS; threads per rank: $THREADS_PER_RANK"
echo "Case: $CASE_FILE"
echo "Run directory: $RUN_DIR"
echo "Runner time limit: $RUN_TIME_LIMIT_SECONDS seconds"
free -h
df -h "$HOME"

"$PYTHON" - <<'PY'
import h5py
import jax
import tatva

devices = jax.devices()
print(
    f"JAX {jax.__version__}; h5py={h5py.__version__}; "
    f"devices={devices}; Tatva={tatva.__file__}"
)
if len(devices) != 1 or devices[0].platform != "cpu":
    raise SystemExit(f"Expected exactly one CPU device, found {devices}.")
PY

if (( MPI_RANKS > 1 )); then
  "$PYTHON" - <<'PY'
import mpi4jax
import mpi4py

print(f"mpi4py={mpi4py.__version__}; mpi4jax={mpi4jax.__version__}")
PY
fi

vmstat 10 > "$LOG_DIR/vmstat.log" &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
finish_job() {
  cleanup_monitor
  /opt/bin/sbatch_post.sh || true
}
trap finish_job EXIT

"$PYTHON" scripts/run_case.py "$CASE_FILE" --preflight
runner=(
  "$PYTHON" scripts/run_case.py "$CASE_FILE"
  --run-dir "$RUN_DIR"
  --time-limit-seconds "$RUN_TIME_LIMIT_SECONDS"
  "${RESUME_ARGS[@]}"
)
if (( MPI_RANKS > 1 )); then
  mpirun -n "$MPI_RANKS" -ppn "$MPI_RANKS" "${runner[@]}" &
else
  "${runner[@]}" &
fi
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
/opt/bin/sbatch_post.sh
echo "PMMA TS0126 CPU job finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
