import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIRECTIVE = re.compile(
    r"^\s*#SBATCH\s+--mem(?:-per-cpu)?(?:=|\s)",
    flags=re.MULTILINE,
)


def test_slurm_scripts_leave_memory_allocation_to_scheduler():
    scripts = sorted((ROOT / "slurm").glob("*.slurm"))

    assert scripts
    for script in scripts:
        content = script.read_text(encoding="utf-8")
        assert not MEMORY_DIRECTIVE.search(content), (
            f"{script.name} specifies memory; leave RAM allocation to Slurm."
        )


def test_gb200_setup_builds_an_isolated_arm_environment():
    content = (ROOT / "slurm/PMMA-GB200-SETUP.slurm").read_text(encoding="utf-8")

    assert "#SBATCH --partition=gb200-dev" in content
    assert "#SBATCH --gres=gpu:1" in content
    assert "module load miniconda3/26.1.1" in content
    assert "ENV_NAME=tatva-gb200" in content
    assert '"$(uname -m)" == "aarch64"' in content
    assert '"jax[cuda13]"' in content
    assert "tests/test_friction.py tests/test_pmma_cases.py" in content
    assert "estimated_uncompressed_bytes" in content
    assert 'AUTO_SUBMIT_PRODUCTION:-1' in content
    assert 'sbatch --parsable "$ROOT/slurm/PMMA-RSF-GB200.slurm"' in content


def test_gb200_production_uses_one_gpu_for_eight_hours():
    content = (ROOT / "slurm/PMMA-RSF-GB200.slurm").read_text(encoding="utf-8")

    assert "#SBATCH --partition=gb200-r1" in content
    assert "#SBATCH --nodes=1" in content
    assert "#SBATCH --gres=gpu:1" in content
    assert "#SBATCH --time=08:00:00" in content
    assert "RUN_TIME_LIMIT_SECONDS=${RUN_TIME_LIMIT_SECONDS:-27600}" in content
    assert "rsf_0127_q4_chamfer20x5_gb200_8h.toml" in content
