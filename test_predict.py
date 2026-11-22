"""Tests for the punch predictor (predict_dataset.py, predict_trainer.py).

The two control tests matter most: on synthetic boxers whose head dips just before
every punch, the trainer must warn before far more punches than random warnings
at the same rate (a positive control); on the same boxers without the dip, it
must not (a negative control). That shows the pipeline can find a real telegraph
and doesn't invent one.

Run: python -m pytest -q test_predict.py
"""
import contextlib
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

import predict_dataset as pdx
import predict_trainer as pt

FPS = 30.


@contextlib.contextmanager
def workspace_temp():
    root = Path.cwd().resolve()
    path = root / ('pr_' + uuid4().hex[:8])
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != root:
            raise RuntimeError('Refusing cleanup outside workspace')
        shutil.rmtree(path)


def write(folder, name, kp, cf, starts, wrists, performer='p', commands=(), command_times=(), labels=None):
    npz = folder / f'{name}.npz'
    n = len(kp)
    np.savez_compressed(npz, timestamps=np.arange(n) / FPS, keypoints=kp, confidences=cf, performer=performer,
                        commands=np.array(commands, dtype='<U16'), command_times=np.array(command_times, float))
    if labels is None:
        labels = dict(reviewed=[dict(preparation=None, departure=int(s), family='JAB') for s in starts])
    npz.with_suffix('.labels.json').write_text(json.dumps(dict(labelers=labels)))
    cands = [dict(departure=int(s), peak=int(s) + 5, wrist='image-left' if k == 0 else 'image-right', guess='JAB',
                  family='JAB', quality='clean', lead_frames=5, disp=.7, espike=1., time=s / FPS)
             for s, k in zip(starts, wrists)]
    npz.with_suffix('.auto_candidates.json').write_text(json.dumps(dict(session=name, params={}, candidates=cands)))
    return npz


def boxer(n, seed, starts=None, telegraph=False, scale=1.0):
    """Synthetic keypoints of a boxer in guard: bouncing, noisy, punching at `starts`
    (random gaps of 1 s + exponential, mean 3 s, if None). With telegraph=True the head
    dips over the 8 frames (~270 ms) before every punch."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    w, cx, cy = 100. * scale, 320., 260.
    kp = np.zeros((n, 17, 2))
    cf = np.ones((n, 17))
    cf[:, 11:13] = 0.1                                           # hips out of view, as in real footage
    base = {0: (cx, cy - .8 * w), 5: (cx - w / 2, cy), 6: (cx + w / 2, cy), 7: (cx - .75 * w, cy + .55 * w),
            8: (cx + .75 * w, cy + .55 * w), 9: (cx - .25 * w, cy - .55 * w), 10: (cx + .25 * w, cy - .55 * w),
            11: (cx - .4 * w, cy + 1.5 * w), 12: (cx + .4 * w, cy + 1.5 * w)}
    for j, xy in base.items():
        kp[:, j] = xy
    kp[:, :, 1] += .03 * w * np.sin(2 * np.pi * 1.8 * t + rng.uniform(0, 6))[:, None]
    if starts is None:
        starts, s = [], int(3 * FPS)
        while s < n - 40:
            starts.append(s)
            s += int((1. + rng.exponential(2.)) * FPS)
    wrists = [int(rng.integers(2)) for _ in starts]
    out = np.r_[np.linspace(0, .7 * w, 6)[1:], np.linspace(.7 * w, 0, 8)[1:]]
    dip = np.r_[np.linspace(0, .2 * w, 8), np.linspace(.2 * w, 0, 8)]
    for s, k in zip(starts, wrists):
        m = max(0, min(len(out), n - s))
        kp[s:s + m, 9 + k, 0] += (-1 if k == 0 else 1) * out[:m]
        if telegraph:
            a = s - 8
            kp[a:a + 16, 0, 1] += dip[:max(0, min(16, n - a))]
    kp += rng.normal(0, 1.5 * scale, kp.shape)
    return kp, cf, starts, wrists


class TargetTests(unittest.TestCase):
    def test_targets_follow_the_rules(self):
        n = int(20 * FPS)
        with workspace_temp() as tmp:
            # 0-5 s IDLE (don't punch), 5-10 s JAB prompt, 10-14 s FEINT prompt, 14-20 s FREE
            kp, cf, starts, wrists = boxer(n, 1, starts=[int(7 * FPS), int(16 * FPS), int(16 * FPS) + 2])
            labels = dict(reviewed=[dict(preparation=None, departure=int(7 * FPS), family='JAB'),
                                    dict(preparation=None, departure=int(16 * FPS), family='FREE'),
                                    dict(preparation=None, departure=int(18.5 * FPS), family='FEINT')],
                          reviewed_messy=[dict(preparation=None, departure=int(16 * FPS) + 2, family='FREE')])
            npz = write(tmp, 's', kp, cf, starts, wrists, commands=('IDLE', 'JAB', 'FEINT', 'FREE'),
                        command_times=(0., 5., 10., 14.), labels=labels)
            s = pdx.load(npz)
        self.assertEqual(s['merged'], 1)                               # the same punch in both passes
        self.assertEqual([q['kind'] for q in s['punches']], [pdx.PROMPTED, pdx.NATURAL])
        self.assertEqual(s['feints'], [int(18.5 * FPS)])
        y, t = pdx.targets(s), s['t']
        free_start = 16.
        self.assertTrue(np.all(y[(t >= free_start - .3) & (t < free_start)] == 1))
        self.assertTrue(np.all(y[(t >= free_start - .6) & (t < free_start - .3)] == -1))   # too early to call wrong
        self.assertTrue(np.all(y[int(16 * FPS):int(16 * FPS) + 6] == -1))                  # fist in flight
        self.assertTrue(np.all(y[(t >= 5) & (t < 10)] == -1))         # prompted punch: screen cue, not learned
        self.assertTrue(np.all(y[(t >= 10) & (t < 14)] == -1))        # FEINT prompt
        self.assertTrue(np.all(y[(t >= 18.5 - .6) & (t <= 18.9)] == -1))   # around the reviewed feint
        self.assertTrue(np.all(y[(t >= 1) & (t < 4.5)] == 0))        # IDLE: no punch soon
        self.assertTrue(np.all(y[(t >= 14.2) & (t < 15.3)] == 0))    # free boxing between punches
        yp = pdx.targets(s, train_prompted=True)
        self.assertTrue(np.all(yp[(t >= 7 - .3) & (t < 7)] == 1))

    def test_combo_follow_up_is_kept_and_flagged(self):
        n = int(10 * FPS)
        with workspace_temp() as tmp:
            kp, cf, starts, wrists = boxer(n, 2, starts=[60, 66, 150])         # 0.2 s apart = a real combo
            s = pdx.load(write(tmp, 's', kp, cf, starts, wrists))
        self.assertEqual(s['merged'], 0)
        self.assertEqual([q['follow'] for q in s['punches']], [False, True, False])

    def test_features_are_causal(self):
        n = int(12 * FPS)
        kp, cf, starts, wrists = boxer(n, 3, telegraph=True)
        with workspace_temp() as tmp:
            s = pdx.load(write(tmp, 's', kp, cf, starts, wrists))
            cut = dict(s, t=s['t'][:200], kp=s['kp'][:200], cf=s['cf'][:200])
            X_full, names, _ = pdx.features(s)
            X_cut, _, _ = pdx.features(cut)
        # frames before the cut must not change when the future is removed
        np.testing.assert_allclose(X_full[:150], X_cut[:150], equal_nan=True, rtol=1e-5, atol=1e-5)


def fake_session(n=3000, punches=(), kinds=None):
    t = np.arange(n) / FPS
    s = dict(t=t, kind=np.full(n, pdx.NATURAL) if kinds is None else kinds, tracked=np.ones(n, bool),
             dt=np.full(n, 1 / FPS), feints=[], punches=[], flying=np.zeros(n, bool))
    prev = None
    for f, peak in punches:
        s['punches'].append(dict(frame=f, time=t[f], peak=peak, first_motion=f - 3, kind=s['kind'][f],
                                 follow=prev is not None and t[f] - t[prev] <= 1.))
        s['flying'][f:peak + 1] = True
        prev = f
    return s


class ScoringTests(unittest.TestCase):
    def test_credit_lead_false_and_in_flight(self):
        s = fake_session(punches=[(300, 306), (315, 321), (900, 906)])
        # 290: 10 frames before punch 1 (lead 333 ms); 303: while punch 1 flies (neither way);
        # 600: nothing near (false); 895: 5 frames before punch 3
        e = pt.evaluate(s, 0, 3000, np.array([290, 303, 600, 895]))
        leads = [None if r['lead'] is None else round(r['lead'] * FPS) for r in e['punches']]
        self.assertEqual(leads, [10, None, 5])           # the follow-up got no warning of its own
        self.assertEqual(e['during'], 1)
        self.assertEqual(int(e['false'][pdx.NATURAL]), 1)
        self.assertEqual([r['follow'] for r in e['punches']], [False, True, False])
        self.assertAlmostEqual(e['punches'][0]['motion'] * FPS, 7, places=5)   # first motion 3 frames earlier

    def test_one_warning_counts_once(self):
        s = fake_session(punches=[(300, 302), (310, 312)])
        e = pt.evaluate(s, 0, 3000, np.array([295]))
        self.assertEqual([r['lead'] is not None for r in e['punches']], [True, False])

    def test_random_warnings_match_the_rate(self):
        s = fake_session(n=int(60 * 60 * FPS))
        rng = np.random.default_rng(0)
        for rate in (2., 10., 40.):
            w = pt.random_warnings(s, 0, len(s['t']), np.full(5, rate), 0.3, rng)
            self.assertAlmostEqual(len(w) / 60., rate, delta=0.15 * rate)
            self.assertTrue(np.all(np.diff(s['t'][w]) >= 0.3 - 1e-9))

    def test_random_rate_per_kind(self):
        kinds = np.r_[np.full(54000, pdx.NATURAL), np.full(54000, pdx.DONT)]
        s = fake_session(n=len(kinds), kinds=kinds)
        w = pt.random_warnings(s, 0, len(kinds), np.array([20., 2., 0, 0, 0]), 0.3, np.random.default_rng(1))
        nat, dont = (kinds[w] == pdx.NATURAL).sum() / 30., (kinds[w] == pdx.DONT).sum() / 30.
        self.assertAlmostEqual(nat, 20., delta=3.)
        self.assertAlmostEqual(dont, 2., delta=.8)

    def test_wilson(self):
        lo, hi = pt.wilson(5, 10)
        self.assertAlmostEqual(lo, .237, places=2)
        self.assertAlmostEqual(hi, .763, places=2)


def run_controls(telegraph, sims=60):
    with workspace_temp() as tmp:
        paths = []
        for i, (performer, scale) in enumerate((('ann', 1.0), ('ben', 1.3))):
            kp, cf, starts, wrists = boxer(int(150 * FPS), 10 + i, telegraph=telegraph, scale=scale)
            paths.append(write(tmp, f's{i}', kp, cf, starts, wrists, performer=performer))
        sessions, names, groups = pt.build(paths, flow=False)
    plan, notes = pt.tests(sessions)
    name, _, folds = plan[0]
    rng = np.random.default_rng(0)
    res = pt.merge_folds([pt.run_fold(sessions, tr, te, list(range(len(names))), [2.], .3, .5, sims, rng)
                          for tr, te in folds], [2.])[2.]
    first = res['tally']['first punches']
    rand = res['rand']['c1'][:, 0]
    rand_pct = rand[:, 0] / np.maximum(rand[:, 1], 1)
    return name, first['hit'] / first['n'], np.percentile(rand_pct, 97.5), res['tally']


class ControlTests(unittest.TestCase):
    def test_positive_control_finds_the_telegraph_on_an_unseen_person(self):
        name, hit, rand97, tally = run_controls(telegraph=True)
        self.assertTrue(name.startswith('(a)'))
        self.assertGreaterEqual(hit, .6, tally['first punches'])
        self.assertGreater(hit, rand97 + .3)
        self.assertLess(tally['fa_natural'], 6.)

    def test_negative_control_finds_much_less(self):
        _, hit_tel, _, _ = run_controls(telegraph=True, sims=10)
        _, hit, rand97, tally = run_controls(telegraph=False, sims=60)
        self.assertLess(hit, hit_tel - .3, tally['first punches'])

    def test_plan_depends_on_performers(self):
        def sess(name, performer, natural):
            return dict(name=name, performer=performer, t=np.arange(600) / FPS,
                        punches=[dict(kind=pdx.NATURAL, frame=50 + 20 * i) for i in range(natural)])
        plan, notes = pt.tests([sess('a', 'ann', 30), sess('b', 'ann', 0)])
        self.assertEqual([p[0][:4] for p in plan], ["(b')"])
        self.assertTrue(any('at least 2 performers' in n for n in notes))
        plan, _ = pt.tests([sess('a', 'ann', 30), sess('b', 'ben', 30)])
        self.assertEqual([p[0][:3] for p in plan], ['(a)', '(b)'])


if __name__ == '__main__':
    unittest.main()
