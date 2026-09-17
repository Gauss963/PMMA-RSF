#!/bin/bash
#SBATCH --account=MST113389
#SBATCH --partition=gb200-r1
#SBATCH --job-name=PMMA-CZM-XC
#SBATCH --output=gb200-czm-xc-sweep-%j.out
#SBATCH --error=gb200-czm-xc-sweep-%j.err
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

echo "GB200 CZM-Xc sweep job ${SLURM_JOB_ID:-interactive} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes: ${SLURM_JOB_NODELIST:-unset}; tasks: ${SLURM_NTASKS:-unset}"

"$PYTHON" scripts/generate_ts0271_czm_xc_sweep_cases.py --check
"$PYTHON" - <<'PY'
import shutil
from dataclasses import asdict
from pathlib import Path

from scripts.generate_ts0271_czm_xc_sweep_cases import (
    BASELINE_RSF_DC_MM,
    BASELINE_XC_MM,
    characteristic_slip,
    cohesive_zone_size,
)
from tatva.pmma.config import load_case_config
from tatva.pmma.runner import preflight

root = Path("/work/gauss112/tatva")
reference = load_case_config(root / "cases/rsf_0271_leading_transition_16.toml")
case_paths = [
    root / "cases" / f"rsf_{run:04d}_czm_xc_{index:02d}.toml"
    for index, run in enumerate(range(272, 288), start=1)
]
configs = [load_case_config(path) for path in case_paths]

def normalized(payload):
    payload = asdict(payload)
    payload["name"] = "normalized"
    payload["loading"].pop("stop_slip")
    for zone in ("loading", "middle", "leading"):
        payload["rsf"][zone].pop("characteristic_slip")
    return payload

reference_payload = normalized(reference)
if abs(reference.rsf.middle.characteristic_slip - BASELINE_RSF_DC_MM) > 1.0e-16:
    raise SystemExit("TS0271 no longer matches the 5 mm cohesive-zone anchor.")

for index, (path, config) in enumerate(zip(case_paths, configs, strict=True), start=1):
    expected_xc = cohesive_zone_size(index)
    expected_dc = characteristic_slip(index)
    if normalized(config) != reference_payload:
        raise SystemExit(f"Unexpected non-sweep difference in {path.name}.")
    if abs(config.loading.stop_slip - expected_dc) > 1.0e-15:
        raise SystemExit(f"{path.name} does not stop at one swept D_c.")
    for zone_name in ("loading", "middle", "leading"):
        zone = getattr(config.rsf, zone_name)
        if abs(zone.characteristic_slip - expected_dc) > 1.0e-15:
            raise SystemExit(f"Unexpected {zone_name} D_c in {path.name}.")
    recovered_xc = BASELINE_XC_MM * expected_dc / BASELINE_RSF_DC_MM
    if abs(recovered_xc - expected_xc) > 1.0e-12:
        raise SystemExit(f"CZM-to-RSF conversion failed for {path.name}.")

# The minimum X_c retains at least five 0.5 mm cells across middle-zone L_b.
baseline_middle_lb_mm = 3.637
minimum_lb_mm = baseline_middle_lb_mm * cohesive_zone_size(1) / BASELINE_XC_MM
if minimum_lb_mm / reference.numerics.mesh_size < 5.0:
    raise SystemExit("Minimum swept X_c is under-resolved by the production mesh.")

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
    for run, estimate in zip(range(272, 288), per_case_estimates, strict=True)
    if (root / "runs" / f"TS{run:04d}" / "data" / "simulation.h5").is_file()
)
estimated_remaining = estimated_total - existing_bytes
free_bytes = shutil.disk_usage(root).free
required_bytes = estimated_remaining + 20_000_000_000
print(
    f"CZM-Xc storage gate: total={estimated_total / 1e12:.4f} TB; "
    f"existing={existing_bytes / 1e12:.4f} TB; "
    f"remaining={estimated_remaining / 1e12:.4f} TB; "
    f"free={free_bytes / 1e12:.4f} TB; required={required_bytes / 1e12:.4f} TB"
)
if free_bytes < required_bytes:
    raise SystemExit("Insufficient free space for the 16-case CZM-Xc sweep.")
PY

export ROOT ENV_PREFIX RUN_TIME_LIMIT_SECONDS MIN_FREE_BYTES
set +e
srun --exact --mpi=none --kill-on-bad-exit=0 --wait=0 \
  --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=12 \
  --gpus-per-task=1 --gpu-bind=single:1 \
  bash -c '
    export SWEEP_INDEX=$((SLURM_PROCID + 1))
    exec bash "$ROOT/scripts/run_gb200_ts0271_czm_xc_sweep_rank.sh"
  '
sweep_status=$?
set -e

echo "GB200 CZM-Xc sweep job ended at $(date -u +%Y-%m-%dT%H:%M:%SZ); status=$sweep_status"
exit "$sweep_status"
