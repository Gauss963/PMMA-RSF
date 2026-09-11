from __future__ import annotations

import shlex
import subprocess
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def _expected_paths(paths: Sequence[Path | str]) -> list[Path]:
    return [Path(path) for path in paths]


def _base_result(name: str, expected_outputs: Sequence[Path | str]) -> dict[str, Any]:
    return {
        "name": name,
        "expected_outputs": [str(path) for path in _expected_paths(expected_outputs)],
    }


def _missing_outputs(expected_outputs: Sequence[Path | str]) -> list[str]:
    return [
        str(path)
        for path in _expected_paths(expected_outputs)
        if not path.is_file()
    ]


def run_callable_task(
    name: str,
    action: Callable[[], object],
    *,
    expected_outputs: Sequence[Path | str],
    missing_only: bool,
) -> dict[str, Any]:
    result = _base_result(name, expected_outputs)
    if missing_only and expected_outputs and not _missing_outputs(expected_outputs):
        result["status"] = "skipped_existing"
        print(f"Skipping {name}: all expected outputs already exist.", flush=True)
        return result

    print(f"\nRunning analysis task: {name}", flush=True)
    try:
        action()
    except Exception as error:  # noqa: BLE001 - analysis failures must not stop siblings
        traceback.print_exc()
        result.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        print(f"Analysis task {name} failed; continuing.", flush=True)
        return result

    missing = _missing_outputs(expected_outputs)
    if missing:
        result.update(
            status="failed",
            error="Command completed without all expected outputs.",
            missing_outputs=missing,
        )
        print(f"Analysis task {name} left missing outputs; continuing.", flush=True)
        return result

    result["status"] = "completed"
    return result


def run_subprocess_task(
    name: str,
    command: Sequence[str],
    *,
    expected_outputs: Sequence[Path | str],
    cwd: Path,
    missing_only: bool,
) -> dict[str, Any]:
    result = _base_result(name, expected_outputs)
    result["command"] = list(command)
    if missing_only and expected_outputs and not _missing_outputs(expected_outputs):
        result["status"] = "skipped_existing"
        print(f"Skipping {name}: all expected outputs already exist.", flush=True)
        return result

    print(f"\nRunning {name}:\n{shlex.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        result.update(
            status="failed",
            returncode=int(completed.returncode),
            error=f"Command exited with status {completed.returncode}.",
        )
        print(f"Analysis task {name} failed; continuing.", flush=True)
        return result

    missing = _missing_outputs(expected_outputs)
    if missing:
        result.update(
            status="failed",
            error="Command completed without all expected outputs.",
            missing_outputs=missing,
        )
        print(f"Analysis task {name} left missing outputs; continuing.", flush=True)
        return result

    result["status"] = "completed"
    return result
