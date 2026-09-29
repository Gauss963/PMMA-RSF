#!/usr/bin/env python3
"""Validate the exact 16-case manifest, free space and optional GPU pilot."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_ts0368_convergence_cases import VARIANTS, TEMPLATE, FIRST_RUN, case_path, render_case
from tatva.pmma.config import load_case_config
from tatva.pmma.runner import preflight


def source_hash():
    paths = [ROOT / name for name in (
        'tatva/friction.py', 'tatva/pmma/dynamics.py', 'tatva/pmma/integration_probes.py',
        'tatva/pmma/config.py', 'tatva/pmma/runner.py',
    )] + [case_path(i) for i in range(1, 17)]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def main():
    baseline = load_case_config(TEMPLATE)
    assert len(VARIANTS) == 16
    total = remaining = 0
    for index, (length, dt) in enumerate(VARIANTS, 1):
        assert case_path(index).read_text() == render_case(TEMPLATE.read_text(), index)
        cfg = load_case_config(case_path(index))
        assert cfg.moving.loading_extension_length == length
        assert abs(cfg.numerics.time_step - dt * 1e-9) < 1e-20
        assert cfg.numerics.rsf_state_dtype == 'float64'
        assert replace(cfg, name=baseline.name,
                       moving=replace(cfg.moving, loading_extension_length=0.),
                       numerics=replace(cfg.numerics, time_step=baseline.numerics.time_step, rsf_state_dtype=None),
                       output=baseline.output,
                       loading=replace(cfg.loading, shear_phase_time=baseline.loading.shear_phase_time)) == baseline
        estimate = preflight(cfg)
        assert estimate['fault_nodes'] == 1001 and estimate['integration_probe_nodes'] == 61
        assert estimate['within_dump_limit'] and not estimate['animation_enabled']
        size = int(estimate['estimated_uncompressed_bytes'])
        total += size
        run = ROOT / f'runs/TS{FIRST_RUN + index - 1:04d}'
        path = run / 'status.json'
        status = json.loads(path.read_text())['status'] if path.exists() else None
        dump, checkpoint = run / 'data/simulation.h5', run / 'checkpoint.npz'
        if status == 'complete':
            if not dump.is_file():
                raise SystemExit(f'{run}: complete but dump missing, refusing silent skip')
        elif status in ('failed', 'checkpointed') and dump.is_file() and checkpoint.is_file():
            remaining += max(0, size - dump.stat().st_size)
        elif status is None and not dump.exists() and not checkpoint.exists():
            remaining += size
        else:
            raise SystemExit(f'{run}: unsafe existing status {status!r}')
    assert total < 1.4e12
    free = shutil.disk_usage(ROOT).free
    print(f'16 cases, raw estimate {total / 1e12:.4f} TB, remaining {remaining / 1e12:.4f} TB, free {free / 1e12:.4f} TB')
    if free < remaining + 50_000_000_000:
        raise SystemExit('Insufficient free space including 50 GB reserve')
    report_path = os.environ.get('PILOT_REPORT')
    if report_path:
        report = json.loads(Path(report_path).read_text())
        if not report['passed']:
            raise SystemExit('GPU pilot has not passed')
        if report.get('source_hash') != source_hash():
            raise SystemExit('Pilot is stale relative to the solver/cases; rerun the pilot')
        budget = int(os.environ.get('RUN_TIME_LIMIT_SECONDS', '82800')) / 3600
        print(f"Pilot predicts {report['predicted_finest_hours']:.2f} h; with margin {report['hours_with_25_percent_margin']:.2f} h; runner budget {budget:.2f} h")
        if report['predicted_finest_hours'] > budget:
            raise SystemExit('Pilot exceeds runner budget; split/replan before production')


if __name__ == '__main__':
    main()
