from dataclasses import replace
import os
import signal
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.generate_ts0368_convergence_cases import VARIANTS, TEMPLATE, case_path, render_case
from tatva.friction import update_ageing_state
from tatva.pmma.config import load_case_config
from tatva.pmma.dynamics import build_case_model, run_simulation_dumped, SimulationCheckpointed
from tatva.pmma.runner import make_case, make_run_config, preflight
from tatva.pmma.integration_probes import IntegrationProbeWriter


def test_sixteen_variants_only_change_extension_dt_and_common_output():
    baseline = load_case_config(TEMPLATE)
    total = 0
    assert len(VARIANTS) == 16
    for index, (length, dt) in enumerate(VARIANTS, 1):
        cfg = load_case_config(case_path(index))
        assert case_path(index).read_text() == render_case(TEMPLATE.read_text(), index)
        assert cfg.moving.loading_extension_length == length
        assert cfg.numerics.time_step == pytest.approx(dt * 1e-9)
        assert cfg.numerics.rsf_state_dtype == "float64"
        assert replace(cfg, name=baseline.name,
                       moving=replace(cfg.moving, loading_extension_length=0.0),
                       numerics=replace(cfg.numerics, time_step=baseline.numerics.time_step, rsf_state_dtype=None),
                       output=baseline.output,
                       loading=replace(cfg.loading, shear_phase_time=baseline.loading.shear_phase_time)) == baseline
        estimate = preflight(cfg)
        assert estimate['within_dump_limit'] and estimate['integration_probe_nodes'] == 61
        total += estimate['estimated_uncompressed_bytes']
    assert total < 1.4e12
    with pytest.raises(ValueError):
        case_path(17)


@pytest.mark.parametrize('dt', [1e-8, 1.25e-9])
@pytest.mark.parametrize('velocity', [0.0, 1e-4, 2e-4])
def test_mixed_precision_ageing_constant_velocity(dt, velocity):
    jax.config.update('jax_enable_x64', True)
    initial = 4.285293684618383
    dc = jnp.asarray(.00042852936846183833, jnp.float32)
    v = jnp.asarray(velocity, jnp.float32)
    theta = jax.jit(lambda: jax.lax.fori_loop(
        0, 10000, lambda _, state: update_ageing_state(state, v, dt, dc),
        jnp.asarray(initial, jnp.float64)))()
    x = float(v) * dt * 10000 / float(dc)
    expected = initial + dt * 10000 if velocity == 0 else initial * np.exp(-x) + float(dc) / float(v) * (-np.expm1(-x))
    assert theta.dtype == jnp.float64
    assert float(theta) == pytest.approx(expected, abs=1e-11)


def tiny_config():
    cfg = load_case_config(case_path(16))
    return replace(cfg, numerics=replace(cfg.numerics, mesh_size=30., time_step=1e-7),
                   loading=replace(cfg.loading, normal_phase_time=4e-5, normal_ramp_time=1e-5,
                                   shear_phase_time=2e-5, shear_ramp_time=1e-5,
                                   shear_displacement_final=.003, stop_min_y=0, stop_max_y=200,
                                   stop_velocity=1e-8))


def smoke_run(path, **extra):
    cfg = tiny_config()
    return run_simulation_dumped(make_case(cfg), make_run_config(cfg), path,
                                 frames_per_phase=2, shear_frames_per_phase=4,
                                 interface_frames_per_phase=40, shear_interface_frames_per_phase=20,
                                 include_initial_frame=False, store_bulk_strain=False, store_bulk_velocity=False,
                                 rsf_state_dtype='float64', integration_probe_max_y=30., **extra)


def test_short_observation_does_not_speed_up_loading():
    cfg = tiny_config()
    short = replace(cfg, loading=replace(cfg.loading, shear_phase_time=5e-6))
    with jax.enable_x64(False):
        full_model = build_case_model(make_case(cfg), make_run_config(cfg))
        short_model = build_case_model(make_case(short), make_run_config(short))
    short_disp = np.asarray(short_model['shear_displacement_shear'])
    np.testing.assert_array_equal(short_disp, full_model['shear_displacement_shear'][:len(short_disp)])


@pytest.mark.parametrize('checkpoint_phase', ['normal', 'shear'])
def test_probe_simulation_and_checkpoint_resume(tmp_path, monkeypatch, checkpoint_phase):
    fresh = tmp_path / 'fresh.h5'
    smoke_run(fresh)
    resumed = tmp_path / 'resumed.h5'
    checkpoint = tmp_path / 'checkpoint.npz'
    kwargs = {}
    with monkeypatch.context() as patch:
        if checkpoint_phase == 'normal':
            kwargs['checkpoint_deadline_monotonic'] = time.monotonic() - 1
        else:
            append = IntegrationProbeWriter.append
            def request_stop(writer, *args):
                append(writer, *args)
                os.kill(os.getpid(), signal.SIGUSR1)
            patch.setattr(IntegrationProbeWriter, 'append', request_stop)
        with pytest.raises(SimulationCheckpointed):
            smoke_run(resumed, checkpoint_path=checkpoint, **kwargs)
    with np.load(checkpoint) as cp:
        # Verify that only theta, not the million-node mechanics, is promoted.
        assert cp['carry_0'].dtype == np.float32
        assert cp['carry_1'].dtype == np.float32
        assert cp['carry_4'].dtype == np.float64
    smoke_run(resumed, checkpoint_path=checkpoint, resume=True)
    assert not checkpoint.exists()
    with h5py.File(fresh) as a, h5py.File(resumed) as b:
        for name in ('values', 'state', 'history'):
            np.testing.assert_array_equal(a['integration_probes/' + name], b['integration_probes/' + name])
        g = a['integration_probes']
        assert g.attrs['saved_steps'] == 201 or g.attrs['saved_steps'] == 200
        assert g['state'].dtype == np.float64
        values = np.asarray(g['values'])
        assert np.isfinite(values).all()
        # Finite near-sticking traction must not disappear with underflowed V.
        np.testing.assert_allclose(np.abs(values[:, :, 4]), values[:, :, 3] * values[:, :, 6], atol=1e-5)


def test_probe_writer_resume_overwrites_only_after_checkpoint(tmp_path):
    with h5py.File(tmp_path / 'probes.h5', 'w') as h5:
        writer = IntegrationProbeWriter(h5, [0., .5], 10, 1e-9, 40, ['time'], 'lzf', buffer_steps=3)
        values = np.ones((7, 2, 8))
        writer.append(0, np.ones((7, 1)), values, np.ones((7, 2)))
        writer.flush()
        writer = IntegrationProbeWriter(h5, [0., .5], 10, 1e-9, 40, ['time'], 'lzf', resume_step=4, buffer_steps=3)
        writer.append(4, np.full((6, 1), 2), np.full((6, 2, 8), 2), np.full((6, 2), 2))
        writer.flush()
        np.testing.assert_array_equal(h5['integration_probes/values'][:4], 1)
        np.testing.assert_array_equal(h5['integration_probes/values'][4:], 2)
        assert writer.group.attrs['saved_steps'] == 10


def test_streaming_metrics_exclude_the_triggering_work_increment(tmp_path):
    import json
    from scripts.analyze_rsf_convergence_sweep import analyze
    (tmp_path / 'input').mkdir()
    (tmp_path / 'data').mkdir()
    (tmp_path / 'stats').mkdir()
    (tmp_path / 'input/case.toml').write_text(case_path(16).read_text())
    columns = ['shear_loading_stopped', 'applied_shear_displacement', 'elastic_energy',
               'kinetic_energy', 'shear_boundary_reaction', 'normal_external_force',
               'normal_loading_coordinate']
    with h5py.File(tmp_path / 'data/simulation.h5', 'w') as h5:
        h5.attrs['rsf_state_integration_dtype'] = 'float64'
        writer = IntegrationProbeWriter(h5, [0., .5], 3, 1e-9, 40, columns, 'lzf')
        values = np.ones((3, 2, 8))
        values[1, 1, 0] = 8.
        history = [[0, .5, 10, 1, 100, 1000, .01],
                   [1, .6, 11, 2, 100, 1000, .02],
                   [1, .6, 9, 1, 100, 1000, .021]]
        writer.append(0, np.asarray(history), values, np.ones((3, 2)))
        writer.flush()
    result = json.loads(analyze(tmp_path).read_text())
    assert result['stop']['step'] == 2
    assert result['loading_end_peak']['y_mm'] == .5
    assert result['post_stop_shear_work'] == 0
    assert result['post_stop_normal_work'] == pytest.approx(1., abs=1e-5)
    assert result['complete_diagnostics']


def test_walltime_keeps_margin_and_respects_partition_limit():
    from scripts.convergence_walltime import recommended_hours
    assert recommended_hours(11.2) == 16
    assert recommended_hours(13.4) == 18
    assert recommended_hours(15.2) == 20
    assert recommended_hours(18.0) == 24
    with pytest.raises(ValueError):
        recommended_hours(20)


def test_walltime_only_updates_owned_pilot_dependency(tmp_path, monkeypatch):
    import json
    import scripts.convergence_walltime as module
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'predicted_finest_hours': 13.4}))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k:
                        '101|PENDING|afterok:99(unfulfilled)\n102|PENDING|afterok:999(unfulfilled)\n')
    commands = []
    monkeypatch.setattr(module.subprocess, 'run', lambda cmd, **k: commands.append(cmd))
    module.configure_dependent_job(report, '99')
    assert commands == [['scontrol', 'update', 'JobId=101', 'TimeLimit=18:00:00']]
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: '1-00:00:00\n')
    assert module.slurm_budget_seconds('101') == 82800
    monkeypatch.setattr(module.time, 'time', lambda: 10000)
    assert module.slurm_budget_seconds('101', 20000) == 8200


def test_scheduler_rejection_does_not_invalidate_successful_pilot(tmp_path, monkeypatch):
    import json
    import scripts.convergence_walltime as module
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'passed': True, 'predicted_finest_hours': 11.56}))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k:
                        '101|PENDING|afterok:99(unfulfilled)\n')
    def rejected(cmd, **kwargs):
        raise module.subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(module.subprocess, 'run', rejected)
    module.configure_dependent_job(report, '99')
    result = json.loads(report.read_text())
    assert result['passed'] and result['recommended_wall_hours'] == 16
    assert 'walltime_update_error' in result
