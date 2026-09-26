#!/usr/bin/env python3
"""Small backend smoke test before allocating a production extension dump."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from tatva.friction import project_regularized_rate_state_velocity
from tatva.pmma.config import load_case_config
from tatva.pmma.dynamics import build_case_model, run_simulation_dumped
from tatva.pmma.runner import make_case, make_run_config


def main():
    print(f"Extension backend smoke test: {jax.devices()}", flush=True)
    a = jnp.asarray([.0025, .005, .01, .02], dtype=jnp.float32)
    factor = jnp.asarray(69.68641115, dtype=jnp.float32)
    free = jnp.full(a.shape, factor * 1.6)
    v, tau = jax.jit(project_regularized_rate_state_velocity)(
        free, jnp.asarray(16., dtype=jnp.float32), jnp.full(a.shape, 4.285293684618383), factor,
        reference_friction=.8, direct_effect=a, state_effect=a + .024123527228205266,
        reference_velocity=1e-4, characteristic_slip=.00042852936846183833,
    )
    np.testing.assert_allclose(tau, 1.6, rtol=2e-5)
    np.testing.assert_allclose(v + factor * tau, free, rtol=2e-5)
    cfg = load_case_config(ROOT / 'cases/rsf_0367_extension_16.toml')
    cfg = replace(cfg, numerics=replace(cfg.numerics, mesh_size=10., time_step=1e-7),
                  loading=replace(cfg.loading, normal_phase_time=4e-5, normal_ramp_time=1e-5,
                                  shear_phase_time=2e-5, shear_ramp_time=1e-5,
                                  shear_displacement_final=.003, stop_min_y=0, stop_max_y=200,
                                  stop_velocity=1e-8))
    model = build_case_model(make_case(cfg), make_run_config(cfg))
    coords = np.asarray(model['moving'].mesh.coords)
    np.testing.assert_allclose(np.sum(model['force_normal']), 8000, rtol=2e-6)
    np.testing.assert_allclose(np.sum(model['interface_weights']), 500, rtol=2e-6)
    assert coords[np.asarray(model['master_nodes']), 1].min() == 0
    assert np.all(coords[np.asarray(model['moving_shear_loading_dofs']) // 2, 1] == -30)
    with TemporaryDirectory(prefix='pmma-extension-smoke-') as directory:
        path = Path(directory) / 'smoke.h5'
        run_simulation_dumped(make_case(cfg), make_run_config(cfg), path,
                              frames_per_phase=2, shear_frames_per_phase=4,
                              interface_frames_per_phase=400, shear_interface_frames_per_phase=200,
                              include_initial_frame=False, store_bulk_strain=False, store_bulk_velocity=False)
        with h5py.File(path) as h5:
            high = h5['interface_high_rate']
            columns = [v.decode() for v in high['history_columns']]
            history = np.asarray(high['history'])
            assert np.isfinite(history).all()
            stopped = history[:, columns.index('shear_loading_stopped')] > .5
            assert stopped.any()
            displacement = history[:, columns.index('applied_shear_displacement')]
            np.testing.assert_array_equal(displacement[stopped], displacement[np.flatnonzero(stopped)[0]])
            assert history[:, columns.index('extension_elastic_energy')].max() > 0
    print("Extension backend smoke test passed; temporary dump removed.", flush=True)


if __name__ == '__main__':
    main()
