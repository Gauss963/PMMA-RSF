#!/usr/bin/env python3
"""Backend/checkpoint regression, optionally followed by a full-mesh timing pilot."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from tatva.friction import update_ageing_state
from tatva.pmma.config import load_case_config
from tatva.pmma.dynamics import SimulationCheckpointed, run_simulation_dumped
from tatva.pmma.runner import make_case, make_run_config


def execute(cfg, path, normal_frames=40, shear_frames=20, **kwargs):
    return run_simulation_dumped(
        make_case(cfg), make_run_config(cfg), path,
        frames_per_phase=2, shear_frames_per_phase=2,
        interface_frames_per_phase=normal_frames, shear_interface_frames_per_phase=shear_frames,
        rsf_state_dtype='float64', integration_probe_max_y=30.,
        include_initial_frame=False, store_bulk_strain=False, store_bulk_velocity=False,
        **kwargs,
    )


def validate(directory):
    jax.config.update('jax_enable_x64', True)
    initial = jnp.asarray(4.285293684618383, jnp.float64)
    dt = 1.25e-9
    aged = jax.jit(lambda: jax.lax.fori_loop(
        0, 10000, lambda _, theta: update_ageing_state(theta, jnp.float32(0), dt, .00042852936846183833),
        initial))()
    np.testing.assert_allclose(aged, float(initial) + dt * 10000, rtol=0, atol=1e-11)
    cfg = load_case_config(ROOT / 'cases/rsf_0383_convergence_16.toml')
    small = replace(cfg, numerics=replace(cfg.numerics, mesh_size=30., time_step=1e-7),
                    loading=replace(cfg.loading, normal_phase_time=4e-5, normal_ramp_time=1e-5,
                                    shear_phase_time=2e-5, shear_ramp_time=1e-5,
                                    shear_displacement_final=.003, stop_min_y=0, stop_max_y=200,
                                    stop_velocity=1e-8))
    fresh, resumed, checkpoint = (directory / name for name in ('fresh.h5', 'resume.h5', 'checkpoint.npz'))
    execute(small, fresh)
    try:
        execute(small, resumed, checkpoint_path=checkpoint, checkpoint_deadline_monotonic=time.monotonic() - 1)
    except SimulationCheckpointed:
        pass
    else:
        raise AssertionError('Checkpoint test failed to checkpoint')
    with np.load(checkpoint) as cp:
        assert cp['carry_0'].dtype == np.float32 and cp['carry_1'].dtype == np.float32
        assert cp['carry_4'].dtype == np.float64
    execute(small, resumed, checkpoint_path=checkpoint, resume=True)
    with h5py.File(fresh) as a, h5py.File(resumed) as b:
        roundoff = {}
        for field in ('values', 'state', 'history'):
            left = np.asarray(a['integration_probes/' + field])
            right = np.asarray(b['integration_probes/' + field])
            # GPU scatter/reduction order need not be bitwise reproducible.
            # Scale each physical channel separately, not by the largest
            # number across unrelated units (e.g. velocity vs overlap).
            axes = tuple(range(left.ndim - 1))
            scale = np.maximum(np.max(np.abs(left), axis=axes), np.max(np.abs(right), axis=axes))
            error = np.max(np.abs(left - right), axis=axes)
            tolerance = 64 * np.finfo(np.float32).eps * scale + np.finfo(np.float32).tiny
            if not np.all(error <= tolerance):
                raise AssertionError(f'{field}: restart difference {error} exceeds {tolerance}')
            roundoff[field] = (error / np.maximum(scale, np.finfo(np.float32).tiny)).tolist()
        g = a['integration_probes']
        columns = [v.decode() for v in g['history_columns']]
        for name in ('shear_loading_stopped', 'applied_shear_displacement'):
            i = columns.index(name)
            np.testing.assert_array_equal(g['history'][:, i], b['integration_probes/history'][:, i])
        assert g.attrs['saved_steps'] == len(g['history'])
        values = np.asarray(g['values'])
        assert np.isfinite(values).all()
        np.testing.assert_allclose(np.abs(values[:, :, 4]), values[:, :, 3] * values[:, :, 6], atol=1e-5)
    print('Restart scaled roundoff by channel: ' + json.dumps(roundoff), flush=True)
    print('Mixed-precision RSF, every-step probes and checkpoint regression PASSED', flush=True)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-report', type=Path)
    args = parser.parse_args()
    print(f'Convergence backend: JAX={jax.__version__}, devices={jax.devices()}', flush=True)
    with TemporaryDirectory(prefix='rsf-convergence-', dir=ROOT if args.benchmark_report else None) as directory:
        directory = Path(directory)
        cfg = validate(directory)
        if args.benchmark_report:
            # Timing only, not a physical production case: retain the full mesh
            # and finest dt while shortening both phases to a few thousand steps.
            dt = cfg.numerics.time_step
            pilot = replace(cfg, loading=replace(cfg.loading,
                            normal_phase_time=32768 * dt, normal_ramp_time=16384 * dt,
                            shear_phase_time=12800 * dt))
            start = time.monotonic()
            path = directory / 'full_mesh_timing.h5'
            execute(pilot, path, normal_frames=33, shear_frames=161)
            elapsed = time.monotonic() - start
            with h5py.File(path) as h5:
                timings = json.loads(h5.attrs['steady_chunk_timings_json'])
                history = np.asarray(h5['history'])
                assert np.isfinite(history).all()
            assert all(t['steps'] >= 5000 and t['seconds'] > 0 for t in timings.values())
            normal_s = timings['normal']['seconds'] / timings['normal']['steps']
            shear_s = timings['shear']['seconds'] / timings['shear']['steps']
            predicted = (.040 / dt * normal_s + .012 / dt * shear_s) / 3600
            from scripts.preflight_convergence_sweep import source_hash
            report = dict(passed=True, benchmark_only=True, dt=dt, mesh_size=cfg.numerics.mesh_size,
                          source_hash=source_hash(),
                          extension_mm=cfg.moving.loading_extension_length, timings=timings,
                          pilot_elapsed_seconds=elapsed, predicted_finest_hours=predicted,
                          hours_with_25_percent_margin=1.25 * predicted + .5,
                          caveat='Short kernel/I-O timing, not full-run guarantee; shared filesystem contention may add cost.')
            args.benchmark_report.parent.mkdir(parents=True, exist_ok=True)
            args.benchmark_report.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
