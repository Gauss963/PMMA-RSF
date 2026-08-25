#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=8gpus
#SBATCH --job-name=PMMA-GPU-SCALE
#SBATCH --output=gpu-scale-%j.out
#SBATCH --error=gpu-scale-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/tatva
PYTHON=/home/gauss112/.conda/envs/tatva/bin/python
CASE_FILE="$ROOT/cases/pilots/ts0118_gpu_scaling_h050.toml"
RESULT_DIR="$ROOT/runs/benchmarks/gpu-scaling/job-${SLURM_JOB_ID}"
all_visible=${CUDA_VISIBLE_DEVICES:?Slurm did not expose the requested GPUs.}
first_visible=${all_visible%%,*}

module purge
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.94
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset LD_LIBRARY_PATH || true

mkdir -p "$RESULT_DIR"
cd "$ROOT"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -l 1 > "$RESULT_DIR/nvidia_smi.csv" &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

run_benchmark() {
  local label=$1
  local visible=$2
  local run_dir="$RESULT_DIR/$label"
  local start_ns end_ns

  start_ns=$(date +%s%N)
  CUDA_VISIBLE_DEVICES="$visible" "$PYTHON" scripts/run_case.py "$CASE_FILE" \
    --run-dir "$run_dir" \
    --time-limit-seconds 4800 \
    > "$RESULT_DIR/$label.stdout" 2>&1
  end_ns=$(date +%s%N)
  "$PYTHON" -c 'import sys; print((int(sys.argv[2])-int(sys.argv[1]))/1e9)' \
    "$start_ns" "$end_ns"
}

echo "Slurm-visible GPUs: $all_visible"
one_gpu_seconds=$(run_benchmark one-gpu "$first_visible")
two_gpu_seconds=$(run_benchmark two-visible-gpus "$all_visible")

"$PYTHON" - "$one_gpu_seconds" "$two_gpu_seconds" \
  "$RESULT_DIR/scaling.json" <<'PY'
import json
from pathlib import Path
import sys

one = float(sys.argv[1])
two = float(sys.argv[2])
payload = {
    "one_gpu_seconds": one,
    "two_visible_gpus_seconds": two,
    "two_over_one_ratio": two / one,
    "linear_scaling_threshold": 0.55,
    "passes_linear_scaling_threshold": two / one <= 0.55,
    "note": (
        "The current PMMA driver uses ordinary jax.jit arrays on the default "
        "device. This benchmark checks whether merely exposing a second GPU "
        "changes measured runtime or utilization."
    ),
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

cleanup_monitor
trap - EXIT
