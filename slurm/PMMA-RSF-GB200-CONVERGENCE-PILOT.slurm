#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=gb200-dev
#SBATCH --job-name=PMMA-DT-PILOT
#SBATCH --output=gb200-dt-pilot-%j.out
#SBATCH --error=gb200-dt-pilot-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=02:00:00

set -euo pipefail
ROOT=${ROOT:-/work/gauss112/tatva}
ENV_PREFIX=${ENV_PREFIX:-/work/gauss112/.venvs/tatva-gb200}
export PATH="$ENV_PREFIX/bin:$PATH" PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
unset LD_LIBRARY_PATH || true
cd "$ROOT"
python -c 'import jax; d=jax.devices(); assert len(d)==1 and d[0].platform=="gpu", d; print(d)'
python scripts/validate_rsf_convergence_backend.py \
  --benchmark-report "stats/convergence_pilot_${SLURM_JOB_ID}.json"
