#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=hm112
#SBATCH --job-name=PMMA-ext-review
#SBATCH --output=/work1/gauss112/pmma-extension-review-%j.out
#SBATCH --error=/work1/gauss112/pmma-extension-review-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail
ROOT=/work1/gauss112/tatva
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" "$ROOT/scripts/review_rsf_extension_sweep.py" \
  --runs-root "$ROOT/runs" \
  --output-root "$ROOT/reviews/TS0352_TS0367" \
  --dump-diagnostics
