import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

SPEC = importlib.util.spec_from_file_location(
    'extension_review', Path(__file__).parents[1] / 'scripts/review_rsf_extension_sweep.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ReviewTests(unittest.TestCase):
    def test_isolated_peaks_do_not_bridge_gaps(self):
        self.assertEqual(review.intervals(np.arange(6), [1, 1, 0, 1, 0, 1]),
                         [[0., 1.], [3., 3.], [5., 5.]])
        self.assertEqual(review.intervals(np.arange(3), [0, 0, 0]), [])

    def test_phase_relative_time_and_matched_time_work(self):
        columns = ['shear_loading_stopped', 'elastic_energy', 'interface_energy',
                   'kinetic_energy', 'normal_external_force', 'normal_loading_coordinate',
                   'extension_elastic_energy']
        history = np.array([[0, 100, 0, 0, 10, 0, 0],
                            [0, 120, 0, 1, 10, 1, 5],
                            [1, 140, 0, 2, 10, 2, 8],
                            [1, 110, 0, 4, 14, 4, 3],
                            [1, 130, 0, 1, 10, 5, 6]])
        stations = np.array([(0., 100.), (200., 600.), (470., 20.), (500., 0.)],
                            dtype=[('y_mm', float), ('peak_V_mm_s', float)])
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'data').mkdir()
            with h5py.File(run / 'data/simulation.h5', 'w') as h5:
                h5.attrs['dt'] = .001
                group = h5.create_group('interface_high_rate')
                group['phase_id'] = [1, 2, 2, 2, 2]
                group['step_id'] = [40, 1, 2, 3, 4]
                group['history_columns'] = np.array(columns, dtype='S')
                group['history'] = history
                group['friction_coefficient'] = np.full((5, 4), .5)
                group['friction_strength'] = np.full((5, 4), 8.)
                group['slip_rate'] = np.array([[0]*4, [1]*4, [2]*4, [600]*4, [1]*4])
            result = review.dump_diagnostics(run, {}, stations)
        self.assertEqual(result['stop_time_in_shear_ms'], 2.)
        self.assertEqual(result['peak_time_in_shear_ms'], 3.)
        self.assertEqual(result['energy_minimum_time_in_shear_ms'], 3.)
        self.assertEqual(result['normal_work_to_energy_minimum'], 24.)
        self.assertEqual(result['extension_release_to_same_energy_minimum'], 5.)
        self.assertEqual(result['peak_inferred_compression_MPa'], 16.)
        self.assertAlmostEqual(result['post_stop_peak_kinetic_over_increment'], .1)


if __name__ == '__main__':
    unittest.main()
