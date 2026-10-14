#!/bin/bash
# Repeated events in one job: event 1 runs CASE, each later event restarts from the
# previous event's final state with RESTART_CASE (full strength recovery). Events land
# in $RUN_DIR/ev1, ev2, ...; resubmitting skips completed events and resumes the rest.
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=LSW-events
#SBATCH --output=/work1/gauss112/lsw-%j.out
#SBATCH --error=/work1/gauss112/lsw-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=96:00:00

set -euo pipefail
module load gcc/11.2.0 openmpi/5.0.2
ROOT=${ROOT:-/work1/gauss112/PMMA-LSW-MPI}
PYTHON=/home/gauss112/.venvs/tatva-lsw-mpi/bin/python
RUN_DIR=${RUN_DIR:?Set the unique production run directory before submission}
EVENTS=${EVENTS:?Set the number of events}
CASE=${CASE:?Set the event-1 case TOML (path relative to ROOT or absolute)}
RESTART_CASE=${RESTART_CASE:?Set the restart case TOML for events 2+}
[[ "$CASE" = /* ]] || CASE="$ROOT/$CASE"
[[ "$RESTART_CASE" = /* ]] || RESTART_CASE="$ROOT/$RESTART_CASE"
[[ -f "$CASE" && -f "$RESTART_CASE" ]] || { echo "Missing case file" >&2; exit 1; }
grep -q '^restart_from_file = "PREVIOUS_EVENT"$' "$RESTART_CASE" \
  || { echo "RESTART_CASE needs restart_from_file = \"PREVIOUS_EVENT\"" >&2; exit 1; }
export PYTHONPATH="$ROOT"
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$OMP_NUM_THREADS TF_NUM_INTEROP_THREADS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$OMP_NUM_THREADS"
# F1's Slurm does not export SLURM_JOB_END_TIME; fall back to the scheduler's end time.
end=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
if [[ "${BIND:-core}" == none ]]; then
  placement=(--map-by slot --bind-to none)
else
  placement=(--map-by "package:PE=$OMP_NUM_THREADS" --bind-to core)
fi
mkdir -p "$RUN_DIR"
# Optional GAP (mm) overrides shear_loading_face_gap in both cases.
gap_edit=()
if [[ -n "${GAP:-}" ]]; then
  grep -q '^shear_loading_face_gap = ' "$CASE" || { echo "CASE has no shear_loading_face_gap" >&2; exit 1; }
  gap_edit=(-e "s/^shear_loading_face_gap = .*/shear_loading_face_gap = $GAP/"
            -e "/^name = /s/-gap[0-9.]*-/-gap$GAP-/")
fi
for ((e = 1; e <= EVENTS; e++)); do
  dir="$RUN_DIR/ev$e"
  if grep -qs '"completed"' "$dir/status.json"; then
    echo "ev$e already completed"; continue
  fi
  case_file="$RUN_DIR/ev$e.toml"
  if ((e == 1)); then
    sed ${gap_edit[@]+"${gap_edit[@]}"} -e '' "$CASE" > "$case_file"
  else
    sed ${gap_edit[@]+"${gap_edit[@]}"} \
      -e "s#^restart_from_file = \"PREVIOUS_EVENT\"\$#restart_from_file = \"$RUN_DIR/ev$((e - 1))/data/simulation.h5\"#" \
      "$RESTART_CASE" > "$case_file"
  fi
  args=()
  [[ -f "$dir/data/checkpoint.npz" ]] && args+=(--resume)
  limit=$((end - $(date +%s) - 600))
  [[ $limit -gt 0 ]]
  echo "ev$e starting $(date)"
  mpirun -np "$SLURM_NTASKS" "${placement[@]}" \
    "$PYTHON" "$ROOT/scripts/run_lsw_case.py" "$case_file" \
    --run-dir "$dir" --time-limit-seconds "$limit" "${args[@]}"
  grep -qs '"completed"' "$dir/status.json" || { echo "ev$e stopped before completing"; exit 0; }
done
