"""Tests for signed_motion.py: the Kalman filter, the signed features and the
pull-back rule (fires for a pull-back BEHIND the guard spot that reverses, not for
a retraction that stops AT the guard spot).

Run: python -m pytest -q test_signed.py
"""
import unittest

import numpy as np

import signed_motion as sm

FPS = 30.


class KalmanTests(unittest.TestCase):
    def test_tracks_position_speed_and_acceleration(self):
        t = np.arange(90) / FPS
        z = 0.5 * 3. * t ** 2 + 0.2 * t                    # constant acceleration 3, start speed 0.2
        x, v, a = sm.kalman_ca(t, z, r=0.005)
        self.assertLess(abs(v[-1] - (3. * t[-1] + 0.2)), 0.05)
        self.assertLess(abs(a[-1] - 3.), 0.1)
        x, v, a = sm.kalman_ca(t, z + np.random.default_rng(0).normal(0, 0.005, len(t)), r=0.005)
        self.assertLess(abs(v[-1] - (3. * t[-1] + 0.2)), 0.5)            # noisy keypoints: rougher

    def test_causal_and_restarts_after_gaps(self):
        t = np.arange(60) / FPS
        z = np.sin(t * 4)
        full = sm.kalman_ca(t, z)
        cut = sm.kalman_ca(t[:30], z[:30])
        np.testing.assert_allclose(full[1][:30], cut[1], rtol=1e-9)   # the future changes nothing
        z2 = z.copy()
        z2[20:35] = np.nan                                    # 0.5 s gap: restart, no stale state
        x, v, _ = sm.kalman_ca(t, z2, reset=0.25)
        self.assertTrue(np.isnan(x[30]))
        self.assertEqual(v[35], 0.)


def still_boxer(n):
    kp = np.zeros((n, 17, 2))
    kp[:, 5], kp[:, 6] = (300., 200.), (400., 200.)
    kp[:, 9], kp[:, 10] = (320., 150.), (380., 150.)
    return kp, np.ones((n, 17))


class RuleTests(unittest.TestCase):
    def run_depth(self, div_k0):
        n = len(div_k0)
        t = np.arange(n) / FPS
        kp, cf = still_boxer(n)
        div = np.zeros((n, 2))
        div[:, 0] = div_k0
        F = sm.per_wrist(t, kp, cf, div)
        return F, sm.rule_scores(F)

    def test_pull_back_behind_guard_fires_before_the_reversal(self):
        div = np.zeros(120)
        div[60:66] = -6.                  # fist shrinks: pulled back behind guard (log size -0.6)
        div[66:70] = 3.                   # ...slowing down
        div[70:76] = 8.                   # then thrown forward
        F, r = self.run_depth(div)
        fired = np.flatnonzero(r["depth"][:, 0] > 0)
        self.assertTrue(len(fired), "no warning for a pull-back behind guard")
        self.assertLessEqual(fired[0], 72)                    # before the forward throw is well under way
        self.assertEqual(r["depth"][:, 1].max(), 0.)          # the other fist never moved

    def test_retraction_that_stops_at_guard_does_not_fire(self):
        div = np.zeros(120)
        div[40:46] = 6.                   # punch out (grows)
        div[46:58] = -3.                  # comes back and stops where it started
        F, r = self.run_depth(div)
        self.assertEqual(r["depth"][:, 0].max(), 0.)

    def test_dip_below_guard_fires(self):
        n = 120
        t = np.arange(n) / FPS
        kp, cf = still_boxer(n)
        kp[60:80, 9, 1] += 30 * np.sin(np.pi * np.arange(20) / 20)      # a smooth 0.3 sw dip and back up
        F = sm.per_wrist(t, kp, cf, None)
        r = sm.rule_scores(F)
        self.assertTrue((r["dip"][:, 0] > 0).any())
        self.assertTrue(np.all(np.isnan(F[:, :, sm.NAMES.index("depth")])))   # no flow: no depth


if __name__ == '__main__':
    unittest.main()
