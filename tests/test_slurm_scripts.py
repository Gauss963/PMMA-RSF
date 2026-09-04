import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIRECTIVE = re.compile(
    r"^\s*#SBATCH\s+--mem(?:-per-cpu)?(?:=|\s)",
    flags=re.MULTILINE,
)


def test_non_gb200_slurm_scripts_leave_memory_allocation_to_scheduler():
    scripts = sorted((ROOT / "slurm").glob("*.slurm"))
    explicit_memory_scripts = {
        "PMMA-GB200-SETUP.slurm",
        "PMMA-RSF-GB200.slurm",
        "PMMA-RSF-GB200-R1-LEADING-EDGE-SWEEP.slurm",
        "PMMA-RSF-GB200-R1-SWEEP.slurm",
        "PMMA-RSF-GB200-R1-SHEAR-RATE-SWEEP.slurm",
    }

    assert scripts
    for script in scripts:
        if script.name in explicit_memory_scripts:
            continue
        content = script.read_text(encoding="utf-8")
        assert not MEMORY_DIRECTIVE.search(content), (
            f"{script.name} specifies memory; leave RAM allocation to Slurm."
        )


def test_gb200_setup_builds_an_isolated_arm_environment():
    content = (ROOT / "slurm/PMMA-GB200-SETUP.slurm").read_text(encoding="utf-8")

    assert "#SBATCH --partition=gb200-dev" in content
    assert "#SBATCH --gres=gpu:1" in content
    assert "#SBATCH --mem=64G" in content
    assert "ENV_PREFIX=/work/gauss112/.venvs/tatva-gb200" in content
    assert '"$system_python" -m venv "$ENV_PREFIX"' in content
    assert "Miniforge3-Linux-aarch64.sh" in content
    assert '"$(uname -m)" == "aarch64"' in content
    assert '"jax[cuda13]==0.11.0"' in content
    assert "tests/test_friction.py tests/test_pmma_cases.py" in content
    assert "estimated_uncompressed_bytes" in content
    assert 'AUTO_SUBMIT_PRODUCTION:-0' in content
    assert 'sbatch --parsable "$ROOT/slurm/PMMA-RSF-GB200.slurm"' in content


def test_gb200_development_pilot_uses_one_gpu_for_two_hours():
    content = (ROOT / "slurm/PMMA-RSF-GB200.slurm").read_text(encoding="utf-8")

    assert "#SBATCH --partition=gb200-dev" in content
    assert "#SBATCH --nodes=1" in content
    assert "#SBATCH --gres=gpu:1" in content
    assert "#SBATCH --mem=200G" in content
    assert "#SBATCH --time=02:00:00" in content
    assert "RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-6600}" in content
    assert "RUN_DIR_OVERRIDE=${RUN_DIR_OVERRIDE:-$ROOT/runs/TS0127}" in content
    assert "ENV_PREFIX=/work/gauss112/.venvs/tatva-gb200" in content
    assert "module load miniconda3" not in content
    assert 'flock -n 9' in content
    assert "rsf_0127_q4_chamfer20x5_gb200_8h.toml" in content


def test_gb200_r1_sweep_uses_sixteen_independent_gpu_steps():
    content = (ROOT / "slurm/PMMA-RSF-GB200-R1-SWEEP.slurm").read_text(
        encoding="utf-8"
    )
    rank_runner = (ROOT / "scripts/run_gb200_loading_sweep_rank.sh").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --partition=gb200-r1" in content
    assert "#SBATCH --nodes=4" in content
    assert "#SBATCH --ntasks=16" in content
    assert "#SBATCH --ntasks-per-node=4" in content
    assert "#SBATCH --gres=gpu:4" in content
    assert "#SBATCH --mem=800G" in content
    assert "#SBATCH --time=16:00:00" in content
    assert "#SBATCH --signal=USR1@1800" in content
    assert "--mpi=none" in content
    assert "--gpus-per-task=1" in content
    assert "srun --exact --mpi=none --kill-on-bad-exit=0 --wait=0" in content
    assert "--ntasks=16 --ntasks-per-node=4" in content
    assert "SWEEP_INDEX=$((SLURM_PROCID + 1))" in content
    assert "RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-54000}" in content
    assert "estimated_remaining + 20_000_000_000" in content
    assert "MIN_FREE_BYTES=${MIN_FREE_BYTES:-10000000000}" in content

    assert "run_number=$((127 + SWEEP_INDEX))" in rank_runner
    assert 'RUN_DIR="$ROOT/runs/$run_id"' in rank_runner
    assert "SLURM_PROCID + 1" in rank_runner
    assert "tatva.pmma.mpi" not in rank_runner
    assert "mpi4py" not in rank_runner
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.90" in rank_runner
    assert "XLA_FLAGS=--xla_gpu_enable_command_buffer=" in rank_runner
    assert "refusing automatic HDF5 resume" in rank_runner
    assert "Free space $free_bytes is below" in rank_runner
    assert "RESUME_ARGS=(--resume)" in rank_runner


def test_cpu_analysis_sweep_processes_every_run_without_animation():
    content = (ROOT / "slurm/PMMA-ANALYSIS-SWEEP-CPU.slurm").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --partition=hm112" in content
    assert "#SBATCH --cpus-per-task=32" in content
    assert "#SBATCH --array" not in content
    assert "ROOT=/work1/gauss112/tatva" in content
    assert "RUN_FIRST=${RUN_FIRST:-128}" in content
    assert "RUN_LAST=${RUN_LAST:-143}" in content
    assert "WORKERS=${WORKERS:-4}" in content
    assert 'run_id=$(printf "TS%04d" "$run_number")' in content
    assert "worker \"$worker_index\" &" in content
    assert "postprocess_velocity_weakening_run.py" in content
    assert "--input \"$input\" --dpi 260" in content
    assert "render_stress_frames.py" not in content
    assert "make_stress_animation.py" not in content
    assert "stress_triptych_frames" not in content


def test_single_run_cpu_analysis_uses_current_f1_checkout_without_animation():
    content = (ROOT / "slurm/PMMA-ANALYSIS-CPU.slurm").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --partition=hm112" in content
    assert "#SBATCH --cpus-per-task=8" in content
    assert "ROOT=/work1/gauss112/tatva" in content
    assert "postprocess_velocity_weakening_run.py" in content
    assert "render_stress_frames.py" not in content
    assert "make_stress_animation.py" not in content


def test_gb200_shear_rate_sweep_uses_independent_single_gpu_tasks():
    content = (
        ROOT / "slurm/PMMA-RSF-GB200-R1-SHEAR-RATE-SWEEP.slurm"
    ).read_text(encoding="utf-8")
    rank_runner = (
        ROOT / "scripts/run_gb200_shear_rate_sweep_rank.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=gb200-r1" in content
    assert "#SBATCH --nodes=4" in content
    assert "#SBATCH --ntasks=16" in content
    assert "#SBATCH --ntasks-per-node=4" in content
    assert "#SBATCH --gres=gpu:4" in content
    assert "#SBATCH --time=16:00:00" in content
    assert "--mpi=none" in content
    assert "--gpus-per-task=1" in content
    assert "generate_shear_rate_sweep_cases.py --check" in content
    assert "range(144, 160)" in content
    assert "0.075 / expected_factor" in content

    assert "run_number=$((143 + SWEEP_INDEX))" in rank_runner
    assert 'RUN_DIR="$ROOT/runs/$run_id"' in rank_runner
    assert "tatva.pmma.mpi" not in rank_runner
    assert "mpi4py" not in rank_runner
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.90" in rank_runner
    assert "XLA_FLAGS=--xla_gpu_enable_command_buffer=" in rank_runner
    assert "refusing automatic HDF5 resume" in rank_runner


def test_gb200_leading_edge_sweep_uses_independent_single_gpu_tasks():
    content = (
        ROOT / "slurm/PMMA-RSF-GB200-R1-LEADING-EDGE-SWEEP.slurm"
    ).read_text(encoding="utf-8")
    rank_runner = (
        ROOT / "scripts/run_gb200_leading_edge_sweep_rank.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=gb200-r1" in content
    assert "#SBATCH --nodes=4" in content
    assert "#SBATCH --ntasks=16" in content
    assert "#SBATCH --ntasks-per-node=4" in content
    assert "#SBATCH --gres=gpu:4" in content
    assert "#SBATCH --time=16:00:00" in content
    assert "--mpi=none" in content
    assert "--gpus-per-task=1" in content
    assert "generate_leading_edge_sweep_cases.py --check" in content
    assert "range(160, 176)" in content
    assert "config.rsf.loading != config.rsf.middle" in content
    assert "config.loading.shear_ramp_time - 0.025" in content

    assert "run_number=$((159 + SWEEP_INDEX))" in rank_runner
    assert 'RUN_DIR="$ROOT/runs/$run_id"' in rank_runner
    assert "tatva.pmma.mpi" not in rank_runner
    assert "mpi4py" not in rank_runner
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.90" in rank_runner
    assert "XLA_FLAGS=--xla_gpu_enable_command_buffer=" in rank_runner
    assert "refusing automatic HDF5 resume" in rank_runner
