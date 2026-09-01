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
        # Zinfandel's 96-core node has 512 GB; reserve 450 GB for replicated MPI state.
        "PMMA-RSF-ZINFANDEL-CPU.slurm",
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
    assert '"jax[cuda13]"' in content
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
