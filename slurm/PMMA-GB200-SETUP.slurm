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
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/tatva
ENV_PREFIX=/work/gauss112/.venvs/tatva-gb200
ENV_PYTHON=$ENV_PREFIX/bin/python
MINIFORGE_PREFIX=/work/gauss112/.miniforge3-aarch64

module purge

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
    echo "Existing environment has architecture $existing_arch: $ENV_PREFIX" >&2
    exit 1
  }
  echo "Reusing existing aarch64 environment: $ENV_PREFIX"
else
  system_python=$(command -v python3 || command -v python || true)
  use_system_python=0
  if [[ -n "$system_python" ]]; then
    use_system_python=$($system_python - <<'PY'
import sys

print(int(sys.version_info >= (3, 11)))
PY
)
  fi

  needs_miniforge=1
  if [[ "$use_system_python" == "1" ]]; then
    echo "Creating the ARM venv with $system_python ($($system_python --version))."
    if "$system_python" -m venv "$ENV_PREFIX"; then
      needs_miniforge=0
    else
      failed_env="$ENV_PREFIX.failed-${SLURM_JOB_ID:-manual}"
      echo "System venv creation failed; preserving it at $failed_env." >&2
      mv "$ENV_PREFIX" "$failed_env"
    fi
  fi

  if [[ "$needs_miniforge" == "1" ]]; then
    echo "A suitable system venv is unavailable; bootstrapping aarch64 Miniforge."
    if [[ ! -x "$MINIFORGE_PREFIX/bin/python" ]]; then
      installer="${TMPDIR:-/tmp}/Miniforge3-Linux-aarch64-${SLURM_JOB_ID:-manual}.sh"
      curl --fail --location --retry 3 \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh \
        --output "$installer"
      bash "$installer" -b -p "$MINIFORGE_PREFIX"
      rm -f "$installer"
    fi
    "$MINIFORGE_PREFIX/bin/conda" create -y -p "$ENV_PREFIX" python=3.12 pip
  fi
fi

unset LD_LIBRARY_PATH || true
export PATH="$ENV_PREFIX/bin:$PATH"
export PIP_CACHE_DIR=/work/gauss112/.cache/pip-aarch64
"$ENV_PYTHON" -m pip install --upgrade pip
# JAX publishes CUDA 13 wheels for Linux aarch64 and recommends pip-provided
# CUDA libraries so the environment stays independent of x86_64 modules.
"$ENV_PYTHON" -m pip install --upgrade "jax[cuda13]"
"$ENV_PYTHON" -m pip install --editable "${ROOT}[dev]"
"$ENV_PYTHON" -m pip check

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$ENV_PYTHON" - <<'PY'
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

"$ENV_PYTHON" scripts/run_case.py \
  cases/rsf_0127_q4_chamfer20x5_gb200_8h.toml --preflight
"$ENV_PYTHON" - <<'PY'
import shutil
from pathlib import Path

from tatva.pmma.config import load_case_config
from tatva.pmma.runner import preflight

root = Path("/work/gauss112/tatva")
case = load_case_config(root / "cases/rsf_0127_q4_chamfer20x5_gb200_8h.toml")
estimate = preflight(case)
free_bytes = shutil.disk_usage(root).free
required_bytes = estimate["estimated_uncompressed_bytes"] + 50_000_000_000
print(
    f"Storage gate: free={free_bytes / 1e12:.3f} TB; "
    f"required={required_bytes / 1e12:.3f} TB"
)
if free_bytes < required_bytes:
    raise SystemExit("Insufficient storage for TS0127 production dump.")
PY
"$ENV_PYTHON" -m pytest -q tests/test_friction.py tests/test_pmma_cases.py

echo "GB200 setup and PMMA smoke tests completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "${AUTO_SUBMIT_PRODUCTION:-0}" == "1" ]]; then
  production_job=$(sbatch --parsable "$ROOT/slurm/PMMA-RSF-GB200.slurm")
  echo "Submitted validated TS0127 GB200 production job: $production_job"
else
  echo "AUTO_SUBMIT_PRODUCTION is disabled; production job was not submitted."
fi
