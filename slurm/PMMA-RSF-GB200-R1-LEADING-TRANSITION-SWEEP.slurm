#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=gb200-r1
#SBATCH --job-name=PMMA-LEAD-TR
#SBATCH --output=gb200-leading-transition-sweep-%j.out
#SBATCH --error=gb200-leading-transition-sweep-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#SBATCH --time=16:00:00
#SBATCH --signal=USR1@1800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b09501028@ntu.edu.tw

set -euo pipefail

ROOT=/work/gauss112/tatva
ENV_PREFIX=/work/gauss112/.venvs/tatva-gb200
PYTHON=$ENV_PREFIX/bin/python
RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-54000}
MIN_FREE_BYTES=${MIN_FREE_BYTES:-10000000000}

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset LD_LIBRARY_PATH || true
cd "$ROOT"

echo "GB200 leading-transition sweep job ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes: ${SLURM_JOB_NODELIST:-unset}; tasks: ${SLURM_NTASKS:-unset}"

"$PYTHON" scripts/generate_ts0240_leading_transition_sweep_cases.py --check
"$PYTHON" - <<'PY'
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts.generate_ts0240_leading_transition_sweep_cases import transition_length
from tatva.pmma.config import load_case_config
from tatva.pmma.profiles import build_rate_state_profile
from tatva.pmma.runner import make_run_config, preflight

root = Path("/work/gauss112/tatva")
reference = load_case_config(root / "cases/rsf_0240_normal_stress_01.toml")
case_paths = [
    root / "cases" / f"rsf_{run:04d}_leading_transition_{index:02d}.toml"
    for index, run in enumerate(range(256, 272), start=1)
]
configs = [load_case_config(path) for path in case_paths]

def normalized(payload):
    payload = asdict(payload)
    payload["name"] = "normalized"
    payload["loading"].pop("normal_stress_reference")
    payload["rsf"].pop("leading_transition_length")
    return payload

reference_payload = normalized(reference)
for index, (path, config) in enumerate(zip(case_paths, configs, strict=True), start=1):
    expected_length = transition_length(index)
    if config.loading.normal_loading_mode != "stress":
        raise SystemExit(f"{path.name} does not enable normal stress control.")
    if abs(config.loading.normal_stress_reference - 16.0) > 1.0e-14:
        raise SystemExit(f"{path.name} does not use 16 MPa normal traction.")
    if abs(config.rsf.leading_transition_length - expected_length) > 1.0e-12:
        raise SystemExit(
            f"Unexpected leading transition in {path.name}: "
            f"{config.rsf.leading_transition_length}."
        )
    if normalized(config) != reference_payload:
        raise SystemExit(f"Unexpected non-sweep difference in {path.name}.")
    if make_run_config(config).normal_loading_mode != "stress":
        raise SystemExit(f"{path.name} did not propagate stress control.")

step_config = configs[-1]
step_y = np.arange(0.0, 500.5, 0.5)
step_profile = build_rate_state_profile(
    step_y, make_run_config(step_config).rsf_profile_spec
)
if step_profile["metadata"]["leading_transition_start"] != 470.0:
    raise SystemExit("Zero-length transition does not begin at y=470 mm.")
below_step = int(np.flatnonzero(step_y == 469.5)[0])
at_step = int(np.flatnonzero(step_y == 470.0)[0])
if step_profile["direct_effect"][below_step] != step_config.rsf.middle.direct_effect:
    raise SystemExit("Zero-length transition lost the middle value below the step.")
if step_profile["direct_effect"][at_step] != step_config.rsf.leading.direct_effect:
    raise SystemExit("Zero-length transition did not step to the leading value.")

estimates = [preflight(config) for config in configs]
if not all(estimate["within_dump_limit"] for estimate in estimates):
    raise SystemExit("At least one per-case dump estimate exceeds its limit.")
per_case_estimates = [int(estimate["estimated_dump_bytes"]) for estimate in estimates]
estimated_total = sum(per_case_estimates)
existing_bytes = sum(
    min(
        estimate,
        (root / "runs" / f"TS{run:04d}" / "data" / "simulation.h5").stat().st_size,
    )
    for run, estimate in zip(range(256, 272), per_case_estimates, strict=True)
    if (root / "runs" / f"TS{run:04d}" / "data" / "simulation.h5").is_file()
)
estimated_remaining = estimated_total - existing_bytes
free_bytes = shutil.disk_usage(root).free
required_bytes = estimated_remaining + 20_000_000_000
print(
    f"Leading-transition storage gate: total={estimated_total / 1e12:.4f} TB; "
    f"existing={existing_bytes / 1e12:.4f} TB; "
    f"remaining={estimated_remaining / 1e12:.4f} TB; "
    f"free={free_bytes / 1e12:.4f} TB; required={required_bytes / 1e12:.4f} TB"
)
if free_bytes < required_bytes:
    raise SystemExit("Insufficient free space for the 16-case leading-transition sweep.")
PY

export ROOT ENV_PREFIX RUN_TIME_LIMIT_SECONDS MIN_FREE_BYTES
set +e
srun --exact --mpi=none --kill-on-bad-exit=0 --wait=0 \
  --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=12 \
  --gpus-per-task=1 --gpu-bind=single:1 \
  bash -c '
    export SWEEP_INDEX=$((SLURM_PROCID + 1))
    exec bash "$ROOT/scripts/run_gb200_ts0240_leading_transition_sweep_rank.sh"
  '
sweep_status=$?
set -e

echo "GB200 leading-transition sweep job ended at $(date -u +%Y-%m-%dT%H:%M:%SZ); status=$sweep_status"
exit "$sweep_status"
