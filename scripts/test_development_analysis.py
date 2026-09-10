#!/usr/bin/env python3
"""Analytical fixtures for dynamical measurements, independent of Titan learning."""
import unittest
import numpy as np
from development_analysis import dmd, events, lines, profile, response_slopes, rqa, lyapunov_label


class AnalysisTests(unittest.TestCase):
    def test_recurrence_fixed_periodic_and_no_returns(self):
        ages = np.arange(32)
        fixed = rqa(np.zeros((32,32)), ages, .01)
        self.assertEqual(fixed['recurrence_rate'], 1)
        self.assertGreater(fixed['determinism'], .99)
        self.assertGreater(fixed['laminarity'], .99)
        state = np.arange(32)%4
        periodic = rqa(abs(state[:,None]-state[None,:]), ages, .01)
        self.assertGreater(periodic['determinism'], .95)
        self.assertEqual(periodic['laminarity'], 0)
        self.assertIn(4, periodic['recurrence_time_steps'])
        empty = rqa(abs(ages[:,None]-ages[None,:]), ages, .01)
        self.assertEqual(empty['recurrence_rate'], 0)
        self.assertIsNone(empty['determinism'])
        self.assertEqual(lines([True,True,False,True]).tolist(), [2,1])

    def test_theiler_gap_is_not_a_return(self):
        ages=np.arange(32)
        r=rqa(abs(ages[:,None]-ages[None,:]), ages, 5)
        self.assertGreater(r['recurrence_rate'],0)
        self.assertEqual(r['recurrence_time_steps'],{})

    def test_fixed_window_qr_uses_accumulated_logs(self):
        from development_window_map import window_rates
        data=[dict(offset=t,finite_time_exponents=[.1,-.2]) for t in [4,8,12,16,20,24]]
        result=window_rates(data,16)
        self.assertEqual(result[-1]['start_offset'],8)
        np.testing.assert_allclose(result[-1]['exponents'],[.1,-.2],rtol=0,atol=1e-12)

    def test_dmd_known_decay_rotation_and_bad_holdout(self):
        t = np.arange(120)
        theta = .17
        x = .995**t[:,None] * np.c_[np.cos(theta*t), np.sin(theta*t)]
        result = dmd(x, rank=2)
        modes = result['eigenvalues']
        self.assertAlmostEqual(modes[0]['magnitude'], .995, places=10)
        self.assertAlmostEqual(abs(modes[0]['radians_per_step']), theta, places=10)
        self.assertLess(result['test_standardized_rmse'], 1e-10)
        self.assertLess(result['recursive_test_rmse'], 1e-10)
        bad = x.copy();bad[90:] = np.random.default_rng(4).normal(0, 30, (30,2))
        self.assertGreater(dmd(bad, 2)['test_standardized_rmse'], result['test_standardized_rmse']+1)
        self.assertFalse(dmd(np.ones((30,2)))['available'])

    def test_quadratic_and_noise_slopes(self):
        for power, resolved in [(2, True), (0, False)]:
            data = [dict(offset=0, band='high', epsilon=e, numerator_full_l2=e**power,
                         numerator_low_spatial_l2=.3*e**power, q_spatial_l2=.3*e**power/(2*e*e))
                    for e in [.01,.02,.04,.08]]
            g = response_slopes(data)['groups'][0]
            self.assertEqual(bool(g['resolved_quadratic_windows']), resolved)
            self.assertTrue(all(x['quadratic_candidate' if resolved else 'noise_like'] for x in g['intervals']))

    def test_gaussian_profile_translation_and_amplitude(self):
        y,x = np.indices((48,48))
        field = np.exp(-((x-24)**2+(y-24)**2)/32)
        a, b = profile(field), profile(np.roll(field*3, 7, axis=0))
        self.assertAlmostEqual(b['amplitude']/a['amplitude'], 3)
        self.assertAlmostEqual(a['ell_cells'], b['ell_cells'])
        qa, qb = np.array(a['normalized_profile'], dtype=float), np.array(b['normalized_profile'], dtype=float)
        self.assertLess(np.nanmax(abs(qa-qb)), 1e-12)
        self.assertIsNone(profile(np.ones((8,8))))

    def test_event_detection_and_collapse(self):
        u = np.ones(40)+.1*np.sin(np.arange(40))
        u[10]=8;u[28]=7
        trajectory = [dict(update_l2=float(x), developmental_age=i,
                     micro=dict(spectral_energy=[i+1.,i*2+1.,i*3+1.])) for i,x in enumerate(u)]
        yy,xx=np.indices((24,24));f=np.exp(-((xx-12)**2+(yy-12)**2)/16)
        spatial=np.stack([f]*40)
        result=events(trajectory, spatial)
        self.assertEqual([r['offset'] for r in result['events']], [10,28])
        self.assertTrue(result['collapse_available'])
        self.assertAlmostEqual(result['pairwise_collapse'][0]['normalized_profile_rmse'], 0)
        self.assertEqual(lyapunov_label([-.2,.1]), 'mixed_sampled_subspace')
        self.assertEqual(lyapunov_label([-.2,-.1]), 'strongly_contractive_sampled_subspace')


if __name__ == '__main__':
    unittest.main()
