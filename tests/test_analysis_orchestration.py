from __future__ import annotations

import sys
from pathlib import Path

from scripts.analysis_orchestration import run_callable_task, run_subprocess_task


def test_callable_failure_is_recorded_without_raising(tmp_path: Path) -> None:
    def fail() -> None:
        raise RuntimeError("bad plot")

    result = run_callable_task(
        "bad_plot",
        fail,
        expected_outputs=[tmp_path / "missing.pdf"],
        missing_only=False,
    )

    assert result["status"] == "failed"
    assert result["error"] == "RuntimeError: bad plot"


def test_subprocess_failure_does_not_prevent_following_task(tmp_path: Path) -> None:
    failed = run_subprocess_task(
        "bad_plot",
        [sys.executable, "-c", "raise SystemExit(3)"],
        expected_outputs=[tmp_path / "bad.pdf"],
        cwd=tmp_path,
        missing_only=False,
    )
    output = tmp_path / "good.pdf"
    completed = run_subprocess_task(
        "good_plot",
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('good.pdf').write_text('ok')",
        ],
        expected_outputs=[output],
        cwd=tmp_path,
        missing_only=False,
    )

    assert failed["status"] == "failed"
    assert failed["returncode"] == 3
    assert completed["status"] == "completed"
    assert output.read_text(encoding="utf-8") == "ok"


def test_missing_only_skips_a_complete_task(tmp_path: Path) -> None:
    output = tmp_path / "existing.pdf"
    output.write_text("complete", encoding="utf-8")

    result = run_subprocess_task(
        "existing_plot",
        [sys.executable, "-c", "raise SystemExit(9)"],
        expected_outputs=[output],
        cwd=tmp_path,
        missing_only=True,
    )

    assert result["status"] == "skipped_existing"
