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
