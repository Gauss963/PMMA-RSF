#!/usr/bin/env python3
"""Fill missing residual-referenced traces without retrying plateau analyses."""

import argparse
import json
from pathlib import Path
import sys

from analysis_orchestration import run_subprocess_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--first', type=int, required=True)
    parser.add_argument('--last', type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    script = root / 'scripts/plot_sigma_xy_probe_traces.py'
    results = []
    for number in range(args.first, args.last + 1):
        run = args.runs_root / f'TS{number:04d}'
        summary = run / 'Plot/analysis_suite_summary.json'
        if not summary.is_file():
            print(f'{run.name}: suite summary missing; skipping.', flush=True)
            continue
        payload = json.loads(summary.read_text())
        for task in payload['tasks']:
            if task['name'] != 'near_fault_delta_tau_time_by_station_5mm':
                continue
            command = task['command']
            if Path(command[1]).resolve() != script or '--baseline-mode' not in command:
                raise ValueError(f'Unexpected trace command in {summary}')
            if command[command.index('--baseline-mode') + 1] != 'residual':
                raise ValueError(f'Not a residual-referenced trace in {summary}')
            result = run_subprocess_task(
                task['name'], [sys.executable, str(script), *command[2:]],
                expected_outputs=task['expected_outputs'], cwd=root, missing_only=True)
            results.append({'run': run.name, **result})
            (run / 'stats/local_stress_trace_completion.json').write_text(
                json.dumps(result, indent=2) + '\n')
    return int(any(r['status'] == 'failed' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
