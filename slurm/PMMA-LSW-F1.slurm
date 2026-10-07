#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=LSW-0114-stress16
#SBATCH --output=/work1/gauss112/lsw-0114-%j.out
#SBATCH --error=/work1/gauss112/lsw-0114-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00

set -euo pipefail
module load gcc/11.2.0 openmpi/5.0.2
ROOT=${ROOT:-/work1/gauss112/PMMA-LSW-MPI}
PYTHON=/home/gauss112/.venvs/tatva-lsw-mpi/bin/python
RUN_DIR=${RUN_DIR:?Set the unique production run directory before submission}
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
  "$ROOT/cases/lsw_0114_normal_stress_16mpa.toml" \
  --run-dir "$RUN_DIR" --time-limit-seconds "$limit" "${args[@]}"
