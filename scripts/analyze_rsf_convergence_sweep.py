#!/usr/bin/env python3
"""Stream every-step convergence diagnostics; never load the bulk dump."""

import argparse
import json
from pathlib import Path
import sys
import tomllib

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def analyze(run):
    cfg = tomllib.loads((run / 'input/case.toml').read_text())
    with h5py.File(run / 'data/simulation.h5') as h5:
        g = h5['integration_probes']
        n = int(g.attrs['saved_steps'])
        dt = float(g.attrs['dt'])
        y = np.asarray(g['y'])
        columns = [x.decode() for x in g['history_columns']]
        column = {name: i for i, name in enumerate(columns)}
        peak_speed = -1.
        peaks = np.zeros(len(y))
        integrated_slip = np.zeros(len(y))
        stop = None
        post_shear_work = 0.
        post_normal_work = 0.
        previous = None
        for start in range(0, n, 8192):
            v = np.asarray(g['values'][start:start + 8192], dtype=float)
            hist = np.asarray(g['history'][start:start + 8192], dtype=float)
            if not np.isfinite(v).all() or not np.isfinite(hist).all():
                raise ValueError('Nonfinite integration diagnostics')
            rate = v[:, :, 0]
            peaks = np.maximum(peaks, rate.max(axis=0))
            integrated_slip += rate.sum(axis=0) * dt
            i, j = np.unravel_index(np.argmax(rate), rate.shape)
            if rate[i, j] > peak_speed:
                peak_speed = float(rate[i, j])
                peak = dict(step=int(start + i + 1), shear_time_s=(start + i + 1) * dt,
                            y_mm=float(y[j]), slip_rate_mm_per_s=peak_speed,
                            mu=float(v[i, j, 3]), signed_tau_mpa=float(v[i, j, 4]),
                            normal_overlap_mm=float(v[i, j, 5]), sigma_n_mpa=float(v[i, j, 6]))
            stopped = hist[:, column['shear_loading_stopped']] > .5
            if stop is None and stopped.any():
                k = int(np.flatnonzero(stopped)[0])
                stop = dict(step=start + k + 1, shear_time_s=(start + k + 1) * dt,
                            displacement_mm=float(hist[k, column['applied_shear_displacement']]),
                            elastic_energy=float(hist[k, column['elastic_energy']]),
                            kinetic_energy=float(hist[k, column['kinetic_energy']]))
            rows = np.vstack((previous, hist)) if previous is not None else hist
            # Both interval endpoints must already be latched: exclude the
            # last increment that itself triggered stopping.
            mask = (rows[:-1, column['shear_loading_stopped']] > .5) & (rows[1:, column['shear_loading_stopped']] > .5)
            for force_name, disp_name, key in (
                ('shear_boundary_reaction', 'applied_shear_displacement', 'shear'),
                ('normal_external_force', 'normal_loading_coordinate', 'normal'),
            ):
                force = rows[:, column[force_name]]
                work = .5 * (force[:-1] + force[1:]) * np.diff(rows[:, column[disp_name]])
                if key == 'shear':
                    post_shear_work += float(work[mask].sum())
                else:
                    post_normal_work += float(work[mask].sum())
            previous = hist[-1:]
        if n == 0:
            raise ValueError('No shear diagnostic steps available')
        final_cumulative = np.asarray(g['values'][n - 1, :, 1])
        state_final = np.asarray(g['state'][n - 1])
        dc = cfg['rsf']['middle']['dc']
        result = dict(run_id=run.name, dt_s=dt, extension_mm=cfg['geometry']['moving']['loading_extension_length'],
                      mesh_mm=cfg['numerics']['mesh_size'], state_dtype=str(h5.attrs['rsf_state_integration_dtype']),
                      saved_shear_steps=n, complete_diagnostics=n == g['values'].shape[0],
                      stopped_before_cutoff=stop is not None, stop=stop, loading_end_peak=peak,
                      max_v_dt_over_dc=peak_speed * dt / dc,
                      post_stop_shear_work=post_shear_work if stop else None,
                      post_stop_normal_work=post_normal_work if stop else None,
                      probe_y_mm=y.tolist(), probe_peak_slip_rate_mm_per_s=peaks.tolist(),
                      probe_integrated_shear_slip_mm=integrated_slip.tolist(),
                      probe_final_total_cumulative_slip_mm=final_cumulative.tolist(),
                      probe_final_state_s=state_final.tolist(),
                      note='Short numerical diagnostic, not a full-rupture or rupture-speed measurement.')
    out = run / 'stats/rsf_convergence_metrics.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', nargs='?', type=Path)
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    if args.summarize:
        records = []
        for number in range(368, 384):
            path = args.summarize / f'runs/TS{number:04d}/stats/rsf_convergence_metrics.json'
            records.append(dict(run=f'TS{number:04d}', metrics=json.loads(path.read_text()) if path.exists() else None))
        out = args.summarize / 'stats/TS0368_TS0383_convergence_summary.json'
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(records, indent=2) + '\n')
    else:
        if not args.run_dir:
            parser.error('Specify a run directory or --summarize ROOT')
        out = analyze(args.run_dir)
    print(out)


if __name__ == '__main__':
    main()
