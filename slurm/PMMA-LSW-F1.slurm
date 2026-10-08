#!/bin/bash
# 1 rank x 8 threads: F1 benchmark (job 1848) showed 56-core MPI layouts were only
# 1.4x faster at 5x the core-hours, because rank-0 HDF5 output dominates the shear phase.
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=LSW-F1
#SBATCH --output=/work1/gauss112/lsw-%j.out
#SBATCH --error=/work1/gauss112/lsw-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=36:00:00

set -euo pipefail
module load gcc/11.2.0 openmpi/5.0.2
ROOT=${ROOT:-/work1/gauss112/PMMA-LSW-MPI}
PYTHON=/home/gauss112/.venvs/tatva-lsw-mpi/bin/python
RUN_DIR=${RUN_DIR:?Set the unique production run directory before submission}
CASE=${CASE:?Set the LSW case TOML (path relative to ROOT or absolute)}
[[ "$CASE" = /* ]] || CASE="$ROOT/$CASE"
[[ -f "$CASE" ]] || { echo "Missing case: $CASE" >&2; exit 1; }
export PYTHONPATH="$ROOT"
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$OMP_NUM_THREADS TF_NUM_INTEROP_THREADS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$OMP_NUM_THREADS"
args=()
if [[ -f "$RUN_DIR/data/checkpoint.npz" ]]; then
  args+=(--resume)
fi
# F1's Slurm does not export SLURM_JOB_END_TIME; fall back to the scheduler's end time.
end=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
limit=$((end - $(date +%s) - 600))
[[ $limit -gt 0 ]]
mpirun -np "$SLURM_NTASKS" --map-by "package:PE=$OMP_NUM_THREADS" --bind-to core \
  "$PYTHON" "$ROOT/scripts/run_lsw_case.py" \
  "$CASE" \
  --run-dir "$RUN_DIR" --time-limit-seconds "$limit" "${args[@]}"
