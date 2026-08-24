#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=8gpus
#SBATCH --job-name=PMMA-RSF-DT2
#SBATCH --output=/work/gauss112/tatva/slurm/dt-refine-%j.out
#SBATCH --error=/work/gauss112/tatva/slurm/dt-refine-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT=/work/gauss112/tatva
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python

module purge
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.94
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset LD_LIBRARY_PATH || true

cd "$ROOT"
for factor in 050 075; do
  case_file="$ROOT/cases/rsf_0116_dt_pilot_${factor}.toml"
  run_dir=$("$PYTHON" scripts/run_case.py \
    "$case_file" --allocate-run-dir)
  echo "=== contact factor $factor started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PYTHON" scripts/run_case.py "$case_file" --preflight
  "$PYTHON" scripts/run_case.py "$case_file" --run-dir "$run_dir"
  echo "=== contact factor $factor completed $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done
