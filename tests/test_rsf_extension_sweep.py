from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.generate_ts0352_extension_sweep_cases import (
    CASE_COUNT, LENGTHS_MM, TEMPLATE, case_path, render_case,
)
from scripts.analyze_rsf_extension_sweep import analyze
from tatva.pmma.config import load_case_config
from tatva.pmma.dynamics import build_case_model, run_simulation_dumped
from tatva.pmma.runner import make_case, make_run_config, preflight


def test_only_extension_changes_in_sixteen_cases():
    baseline = load_case_config(TEMPLATE)
    total = 0
    assert CASE_COUNT == 16
    for index, length in enumerate(LENGTHS_MM, 1):
        path = case_path(index)
        assert path.read_text() == render_case(TEMPLATE.read_text(), index)
        cfg = load_case_config(path)
        assert cfg.moving.loading_extension_length == length
        assert replace(cfg, name=baseline.name, moving=replace(cfg.moving, loading_extension_length=0.0)) == baseline
        estimate = preflight(cfg)
        assert estimate['fault_nodes'] == 1001
        assert estimate['within_dump_limit']
        total += estimate['estimated_uncompressed_bytes']
    assert total < 1.4e12
    with pytest.raises(ValueError):
        case_path(17)


def test_extension_has_no_normal_load_or_contact_and_preserves_profiles():
    cfg = load_case_config(case_path(16))
    cfg = replace(cfg, numerics=replace(cfg.numerics, mesh_size=10.0))
    extended = build_case_model(make_case(cfg), make_run_config(cfg))
    control = replace(cfg, moving=replace(cfg.moving, loading_extension_length=0.0))
    original = build_case_model(make_case(control), make_run_config(control))
    moving = extended['moving']
    coords = np.asarray(moving.mesh.coords)
    assert coords[:, 1].min() == -30.0 and coords[:, 1].max() == 500.0
    normal = np.asarray(extended['force_normal'])[:moving.n_nodes * 2].reshape(-1, 2)
    assert np.all(normal[coords[:, 1] < 0] == 0)
    assert normal[:, 0].sum() == pytest.approx(16 * 500)
    at_corner = (coords[:, 0] == 0) & (coords[:, 1] == 0)
    assert normal[at_corner, 0].item() == pytest.approx(16 * 5)
    master = np.asarray(extended['master_nodes'])
    assert coords[master, 1].min() == 0.0 and coords[master, 1].max() == 500.0
    assert np.sum(extended['interface_weights']) == pytest.approx(500)
    np.testing.assert_allclose(extended['interface_weights'], original['interface_weights'])
    for field in extended['rsf_parameters']:
        np.testing.assert_array_equal(extended['rsf_parameters'][field], original['rsf_parameters'][field])
    dofs = np.asarray(extended['moving_shear_loading_dofs'])
    np.testing.assert_array_equal(coords[dofs // 2, 1], -30)
    assert np.all(dofs % 2 == 1)
    np.testing.assert_array_equal(extended['shear_displacement_pressure'], 0.0)


def test_short_extension_run_records_energy_and_freezes_actuator(tmp_path):
    cfg = load_case_config(case_path(16))
    cfg = replace(cfg, numerics=replace(cfg.numerics, mesh_size=30.0, time_step=1e-7),
                  loading=replace(cfg.loading, normal_phase_time=4e-5, normal_ramp_time=1e-5,
                                  shear_phase_time=2e-5, shear_ramp_time=1e-5,
                                  shear_displacement_final=.003, stop_min_y=0, stop_max_y=200,
                                  stop_velocity=1e-8))
    out = tmp_path / 'extension.h5'
    run_simulation_dumped(make_case(cfg), make_run_config(cfg), out,
                          frames_per_phase=2, shear_frames_per_phase=4,
                          interface_frames_per_phase=400, shear_interface_frames_per_phase=200,
                          include_initial_frame=False, store_bulk_strain=False, store_bulk_velocity=False)
    with h5py.File(out) as h5:
        high = h5['interface_high_rate']
        columns = [x.decode() for x in high['history_columns']]
        history = np.asarray(high['history'])
        assert np.isfinite(history).all()
        extension = history[:, columns.index('extension_elastic_energy')]
        assert extension.max() > 0
        assert (extension >= 0).all()
        assert (extension <= history[:, columns.index('elastic_energy')] + 1e-6).all()
        stopped = history[:, columns.index('shear_loading_stopped')] > .5
        assert stopped.any()
        disp = history[:, columns.index('applied_shear_displacement')]
        np.testing.assert_array_equal(disp[stopped], disp[np.flatnonzero(stopped)[0]])
        shear = np.asarray(high['phase_id']) == 2
        np.testing.assert_allclose(history[shear, columns.index('normal_external_force')], 8000)


def test_scheduler_uses_one_gpu_each_and_no_mpi():
    script = (Path(__file__).resolve().parents[1] / 'slurm/PMMA-RSF-GB200-R1-EXTENSION-SWEEP.slurm').read_text()
    for token in ('#SBATCH --ntasks=16', '#SBATCH --time=16:00:00',
                  '--gpus-per-task=1', '--mpi=none', 'SWEEP_RUN_NUMBER=$((351 + SWEEP_INDEX))'):
        assert token in script


def test_analysis_distinguishes_normal_work_from_extension_release(tmp_path, monkeypatch):
    import scripts.analyze_rsf_extension_sweep as module
    monkeypatch.setattr(module, 'analyze_nucleation', lambda run: {'post_stop_boundary_work': 0.0})
    (tmp_path / 'input').mkdir()
    (tmp_path / 'data').mkdir()
    (tmp_path / 'input/case.toml').write_text(case_path(16).read_text())
    columns = ['shear_loading_stopped', 'extension_elastic_energy',
               'normal_loading_coordinate', 'normal_external_force']
    with h5py.File(tmp_path / 'data/simulation.h5', 'w') as h5:
        high = h5.create_group('interface_high_rate')
        high['history_columns'] = np.asarray(columns, dtype='S')
        high['phase_id'] = [1, 2, 2, 2, 2]
        high['history'] = [[0, 0, 0, 0], [0, 20, .5, 8000],
                           [1, 30, .6, 8000], [1, 10, .61, 8000], [1, 12, .62, 8000]]
    metrics = analyze(tmp_path)
    assert metrics['extension_length_mm'] == 30
    assert metrics['extension_energy_at_stop'] == 30
    assert metrics['extension_max_release_after_stop'] == 20
    assert metrics['normal_external_work_after_stop'] == pytest.approx(160)
    assert metrics['post_stop_boundary_work'] == 0
