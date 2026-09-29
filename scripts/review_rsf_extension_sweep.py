#!/usr/bin/env python3
"""Journal-style extension-sweep comparison; never treat isolated peaks as a front."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def intervals(y, mask):
    padded = np.r_[False, mask, False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return [[float(y[a]), float(y[b])] for a, b in zip(starts, ends)]


def summarize(run):
    metrics = json.loads((run / 'stats/rsf_extension_sweep_metrics.json').read_text())
    stations = np.genfromtxt(run / 'stats/rsf_nucleation_station_arrivals.csv',
                            delimiter=',', names=True)
    y = stations['y_mm']
    peak = stations['peak_V_mm_s']
    slip = stations['post_normal_slip_Dc']
    regions = intervals(y, peak >= 500)
    connected = [r for r in regions if r[0] <= 25 and r[1] >= 5]
    main = max(connected, key=lambda r: r[1] - r[0]) if connected else None
    middle = (y >= 200) & (y < 470)
    leading = y >= 470
    row = dict(metrics)
    row.update(
        high_rate_intervals_mm=regions,
        loading_connected_500_interval_mm=main,
        main_500_end_mm=main[1] if main else None,
        global_peak_rate_mm_s=float(peak.max()),
        global_peak_station_mm=float(y[np.argmax(peak)]),
        middle_median_post_normal_slip_um=float(np.median(slip[middle]) * metrics['dc_mm'] * 1000),
        leading_median_post_normal_slip_um=float(np.median(slip[leading]) * metrics['dc_mm'] * 1000),
        leading_1dc_station_fraction=float(np.mean(slip[leading] >= 1)),
        final_500mm_post_normal_slip_Dc=float(slip[np.argmin(abs(y - 500))]),
        diagnostics_caution='V>=500 mm/s is an operational slip-rate threshold, not rupture speed; a 1Dc contour is not automatically a crack tip.',
    )
    return row, stations


def dump_diagnostics(run, row, stations):
    """Read scalar histories and one peak-station trace; never load bulk stress."""
    with h5py.File(run / 'data/simulation.h5') as h5:
        group = h5['interface_high_rate']
        phase = np.asarray(group['phase_id'])
        shear_rows = np.flatnonzero(phase == 2)
        normal_row = np.flatnonzero(phase == 1)[-1]
        columns = [x.decode() if isinstance(x, bytes) else str(x)
                   for x in group['history_columns']]
        history = np.asarray(group['history'], dtype=float)
        c = lambda name: history[:, columns.index(name)]
        stop = shear_rows[np.flatnonzero(c('shear_loading_stopped')[shear_rows] > .5)[0]]
        energy = c('elastic_energy') + c('interface_energy')
        minimum = stop + np.argmin(energy[stop:])
        times = np.asarray(group['step_id'], dtype=float) * float(h5.attrs['dt']) * 1000
        mu = np.asarray(group['friction_coefficient'][stop], dtype=float)
        y = stations['y_mm']
        j = int(np.argmax(stations['peak_V_mm_s']))
        # Chunked by one full interface frame in production HDF5. This selection
        # bounds resident memory, although it still reads the compressed frames.
        rates = np.asarray(group['slip_rate'][shear_rows[0]:shear_rows[-1] + 1, j])
        peak_row = int(shear_rows[0] + np.argmax(rates))
        peak_mu = float(group['friction_coefficient'][peak_row, j])
        peak_tau = float(group['friction_strength'][peak_row, j])
        bulk_energy_increment = energy[stop] - energy[normal_row]
        force = c('normal_external_force')[stop:minimum + 1]
        coordinate = c('normal_loading_coordinate')[stop:minimum + 1]
        normal_work = np.sum(.5 * (force[:-1] + force[1:]) * np.diff(coordinate))
        underflow = (np.asarray(group['friction_strength'][stop]) == 0) & (mu > 1e-4)
        return {
            'dt_s': float(h5.attrs['dt']),
            'projection': str(h5.attrs.get('rsf_velocity_projection', 'unknown')),
            'stop_time_in_shear_ms': float(times[stop]),
            'normal_end_energy': float(energy[normal_row]),
            'stop_energy': float(energy[stop]),
            'final_energy': float(energy[-1]),
            'energy_increment_normal_to_stop': float(bulk_energy_increment),
            'post_stop_peak_kinetic': float(c('kinetic_energy')[stop:].max()),
            'post_stop_peak_kinetic_over_increment': (float(c('kinetic_energy')[stop:].max() / bulk_energy_increment)
                                                     if bulk_energy_increment > 0 else None),
            'energy_minimum_time_in_shear_ms': float(times[minimum]),
            'normal_work_to_energy_minimum': float(normal_work),
            'extension_release_to_same_energy_minimum': float(c('extension_elastic_energy')[stop] - c('extension_elastic_energy')[minimum]),
            'mu_at_stop_middle_quantiles': np.quantile(mu[(y >= 200) & (y < 470)], [0, .5, 1]).tolist(),
            'mu_at_stop_leading_quantiles': np.quantile(mu[y >= 470], [0, .5, 1]).tolist(),
            'peak_time_in_shear_ms': float(times[peak_row]),
            'peak_time_after_stop_ms': float(times[peak_row] - times[stop]),
            'peak_friction_coefficient': peak_mu,
            'peak_friction_strength_MPa': peak_tau,
            'peak_inferred_compression_MPa': peak_tau / peak_mu if peak_mu > 0 else None,
            'zero_saved_strength_nonzero_mu_fraction_at_stop': float(np.mean(underflow)),
            'history_traction_caution': 'Older log-speed projection diagnostics multiply strength by sign(corrected velocity); speed underflow loses signed traction. Do not use that diagnostic for near-sticking prestress.',
        }


def plot_review(records, station_tables, plot_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'mathtext.fontset': 'stix', 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    length = np.array([r['extension_length_mm'] for r in records])
    get = lambda key: [r.get(key, np.nan) for r in records]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(length, get('loading_stop_displacement_mm'), 'o-', color='#234e70', ms=4)
    ax.set_ylabel('Displacement at arrest [mm]')
    ax = axes[0, 1]
    ax.plot(length, get('extension_energy_at_stop'), 'o-', label='At actuator arrest', ms=4)
    ax.plot(length, get('extension_max_release_after_stop'), 's-', label='Maximum subsequent release', ms=4)
    ax.set_ylabel('Extension energy [N mm / mm thickness]')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    ax.plot(length, 100*np.array(get('post_shear_1dc_station_fraction')), 'o-', label='Whole interface', ms=4)
    ax.plot(length, 100*np.array(get('leading_1dc_station_fraction')), 's-', label='Leading 30 mm', ms=4)
    ax.axhline(100, color='.5', lw=.8, ls=':')
    ax.set_ylabel(r'Stations with $\Delta\delta\geq D_c$ [%]')
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    ax.plot(length, get('main_500_end_mm'), 'o-', color='#985830', ms=4)
    ax.axhspan(470, 500, color='.85', zorder=0)
    ax.set_ylim(0, 505)
    ax.set_ylabel('End of connected peak-rate zone [mm]')
    ax.text(.04, .91, r'$\max_t V\geq500$ mm/s (not a front fit)',
            transform=ax.transAxes, fontsize=8)
    for i, ax in enumerate(axes.flat):
        ax.set_xlabel(r'Extension length, $L_{\mathrm{ext}}$ [mm]')
        ax.set_title(f'({chr(97+i)})', loc='left', fontsize=11)
        ax.grid(alpha=.15)
    fig.savefig(plot_dir / 'extension_sweep_summary.pdf')
    fig.savefig(plot_dir / 'extension_sweep_summary.png')
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.3), sharex=True, constrained_layout=True)
    for index, color in zip((0, 1, 3, 9, 14, 15), plt.cm.viridis(np.linspace(.03, .9, 6))):
        table = station_tables[index]
        label = f"{records[index]['extension_length_mm']:g} mm ({records[index]['run_id']})"
        axes[0].semilogy(table['y_mm'], np.maximum(table['peak_V_mm_s'], 1e-4), color=color, lw=1, label=label)
        axes[1].semilogy(table['y_mm'], np.maximum(table['post_normal_slip_Dc'], 1e-4), color=color, lw=1)
    axes[0].axhline(500, color='.25', lw=1, ls='--', label='500 mm/s threshold')
    axes[1].axhline(1, color='.25', lw=1, ls='--')
    axes[0].set_ylim(1e-1, 2e5)
    axes[1].set_ylim(1e-4, 2e3)
    axes[0].set_ylabel('Peak local slip rate [mm/s]')
    axes[1].set_ylabel(r'Post-normal slip, $\Delta\delta/D_c$')
    axes[1].text(.03, .04, r'Values below $10^{-4}$ clipped for display',
                 transform=axes[1].transAxes, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for i, ax in enumerate(axes):
        ax.axvspan(470, 500, color='.9', zorder=0)
        ax.set_xlim(0, 500)
        ax.grid(alpha=.15)
        ax.set_title(f'({chr(97+i)})', loc='left', fontsize=11)
    axes[1].set_xlabel('Station along fault, y [mm]')
    fig.savefig(plot_dir / 'extension_sweep_spatial_comparison.pdf')
    fig.savefig(plot_dir / 'extension_sweep_spatial_comparison.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dump-diagnostics', action='store_true')
    args = parser.parse_args()
    stats, plots = args.output_root / 'stats', args.output_root / 'Plot'
    stats.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    records, tables = [], []
    for number in range(352, 368):
        run = args.runs_root / f'TS{number:04d}'
        row, stations = summarize(run)
        if args.dump_diagnostics:
            row['dump_diagnostics'] = dump_diagnostics(run, row, stations)
        records.append(row)
        tables.append(stations)
        print(run.name, row['extension_length_mm'], row['main_500_end_mm'], flush=True)
    (stats / 'extension_sweep_review.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    scalar_keys = [k for k, v in records[0].items() if isinstance(v, (float, int)) or k == 'run_id']
    with (stats / 'extension_sweep_review.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in scalar_keys} for r in records)
    plot_review(records, tables, plots)


if __name__ == '__main__':
    main()
