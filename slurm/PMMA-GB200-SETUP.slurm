#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=gb200-dev
#SBATCH --job-name=tatva-gb200-setup
#SBATCH --output=gb200-setup-%j.out
#SBATCH --error=gb200-setup-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/tatva
ENV_NAME=tatva-gb200
ENV_PYTHON=/home/gauss112/.conda/envs/$ENV_NAME/bin/python

module purge
module load miniconda3/26.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$ROOT"
echo "GB200 setup job ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname); architecture: $(uname -m)"
module list
nvidia-smi -L
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader

[[ "$(uname -m)" == "aarch64" ]] || {
  echo "GB200 environment must be built on aarch64, found $(uname -m)." >&2
  exit 1
}

if [[ -x "$ENV_PYTHON" ]]; then
  existing_arch=$($ENV_PYTHON -c 'import platform; print(platform.machine())')
  [[ "$existing_arch" == "aarch64" ]] || {
    echo "Existing $ENV_NAME environment has architecture $existing_arch." >&2
    exit 1
  }
  echo "Reusing existing aarch64 environment: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python=3.12 pip
fi

conda activate "$ENV_NAME"
unset LD_LIBRARY_PATH || true
python -m pip install --upgrade pip
# JAX publishes CUDA 13 wheels for Linux aarch64 and recommends pip-provided
# CUDA libraries so the environment stays independent of x86_64 modules.
python -m pip install --upgrade "jax[cuda13]"
python -m pip install --editable "${ROOT}[dev]"
python -m pip check

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python - <<'PY'
import platform

import h5py
import jax
import jax.numpy as jnp
import tatva

devices = jax.devices()
print(f"platform={platform.platform()}; machine={platform.machine()}")
print(
    f"JAX {jax.__version__}; h5py={h5py.__version__}; "
    f"devices={devices}; Tatva={tatva.__file__}"
)
if platform.machine() != "aarch64":
    raise SystemExit("Python environment is not aarch64.")
if len(devices) != 1 or devices[0].platform != "gpu":
    raise SystemExit(f"Expected exactly one GB200 GPU, found {devices}.")

x = jnp.linspace(0.0, 1.0, 1_000_000, dtype=jnp.float32)
value = jax.jit(lambda values: jnp.sum(jnp.sin(values) ** 2))(x)
value.block_until_ready()
print(f"JAX GB200 compile/execute smoke value={float(value):.6e}")
PY

python scripts/run_case.py \
  cases/rsf_0127_q4_chamfer20x5_gb200_8h.toml --preflight
python -m pytest -q tests/test_friction.py tests/test_pmma_cases.py

echo "GB200 setup and PMMA smoke tests completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
