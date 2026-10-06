#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=LSW-MPI-check
#SBATCH --output=/work1/gauss112/lsw-mpi-check-%j.out
#SBATCH --error=/work1/gauss112/lsw-mpi-check-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00

set -euo pipefail
module load gcc/11.2.0 openmpi/5.0.2
ROOT=${ROOT:-/work1/gauss112/PMMA-LSW-MPI}
PYTHON=/home/gauss112/.venvs/tatva-lsw-mpi/bin/python
CASE="$ROOT/cases/lsw_0114_normal_stress_16mpa.toml"
RESULTS=${RESULTS:-"$ROOT/runs/benchmarks/lsw-0114-${SLURM_JOB_ID}"}
mkdir -p "$RESULTS"
export PYTHONPATH="$ROOT"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_test() {
  local ranks=$1 threads=$2 name=$3 steps=$4
  shift 4
  if [[ -f "$RESULTS/$name/stats/summary.json" ]]; then
    echo "Reusing completed benchmark: $name"
    return
  fi
  export OMP_NUM_THREADS=$threads TF_NUM_INTRAOP_THREADS=$threads TF_NUM_INTEROP_THREADS=1
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$threads"
  mpirun -np "$ranks" --map-by "package:PE=$threads" --bind-to core \
    "$PYTHON" "$ROOT/scripts/run_lsw_case.py" "$CASE" \
    --run-dir "$RESULTS/$name" --benchmark-steps "$steps" "$@" \
    > "$RESULTS/$name.log" 2>&1
}

# Validate the full contact update (not just an MPI sum), and checkpoint/resume.
run_test 1 4 serial-small 400 --mesh-size 20
run_test 4 4 mpi-small 400 --mesh-size 20
"$PYTHON" "$ROOT/scripts/compare_lsw_mpi.py" \
  "$RESULTS/serial-small/data/simulation.h5" "$RESULTS/mpi-small/data/simulation.h5" \
  --output "$RESULTS/validation.json"
run_test 4 4 resume-small 400 --mesh-size 20 --time-limit-seconds 1
run_test 4 4 resume-small 400 --mesh-size 20 --resume
"$PYTHON" "$ROOT/scripts/compare_lsw_mpi.py" \
  "$RESULTS/mpi-small/data/simulation.h5" "$RESULTS/resume-small/data/simulation.h5" \
  --output "$RESULTS/resume-validation.json"

# Same 1-mm mesh and original dump cadence; timings exclude first compilations.
run_test 1 8 serial-full 4000
run_test 8 7 mpi8-full 4000
run_test 14 4 mpi14-full 4000
run_test 28 2 mpi28-full 4000
"$PYTHON" "$ROOT/scripts/compare_lsw_mpi.py" \
  "$RESULTS/serial-full/data/simulation.h5" "$RESULTS/mpi14-full/data/simulation.h5" \
  --output "$RESULTS/full-mesh-validation.json"
echo "Benchmark and validation complete: $RESULTS"
