"""Reject corrupted residual decompositions before declaring a panel complete."""
import copy
import unittest
from frozen_diffusion_recovery import validate_residuals


def fixture():
    fields = {}
    for name in ('micro', 'macro', 'memory'):
        rms = .5 if name == 'macro' else 0.
        field = dict(rms=rms, scalar_count=4, l2_energy=rms*rms*4,
                     initial_distance_normalized_rms=rms)
        if name != 'memory':
            field['spatial'] = dict(channel_mean_energy=rms*rms*4,
                                    spatial_band_energy_excluding_dc=[0., 0., 0.])
        fields[name] = field
    residual = dict(fields=fields, sum_rms=.5)
    return [dict(state_distance_ratio=[.5, .5], residuals=[residual, copy.deepcopy(residual)])]


class ResidualValidationTests(unittest.TestCase):
    def test_valid_partition(self):
        validate_residuals(fixture())

    def test_corrupted_distance_energy_and_partition_are_rejected(self):
        for key in ('rms', 'l2_energy', 'initial_distance_normalized_rms', 'spatial'):
            rows = fixture()
            field = rows[0]['residuals'][0]['fields']['macro']
            if key == 'spatial':
                field[key]['spatial_band_energy_excluding_dc'][0] = 1.
            else:
                field[key] += 1.
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_residuals(rows)


if __name__ == '__main__':
    unittest.main()
