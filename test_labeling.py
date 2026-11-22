"""Tests for the labeling tools: review_labels.py (start rule, feints, blind second
reviewer, same-punch warning), label_agreement.py and combo_candidates.py.

Run: python -m pytest -q test_labeling.py
"""
import contextlib
import io
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np

import combo_candidates as cc
import label_agreement as la
import review_labels as rl

FPS = 30.


@contextlib.contextmanager
def workspace_temp():
    root = Path.cwd().resolve()
    path = root / ('lb_' + uuid4().hex[:8])
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != root:
            raise RuntimeError('Refusing cleanup outside workspace')
        shutil.rmtree(path)


def cand(dep, family='JAB', quality='clean', wrist='image-left', fps=FPS):
    return dict(departure=dep, peak=dep + 5, wrist=wrist, guess='JAB', family=family, quality=quality,
                lead_frames=5, disp=0.6, espike=1.0, time=dep / fps)


def write_session(folder, cands, n=400, labels=None, commands=('JAB',), command_times=(0.,), kp=None):
    npz = folder / 'session_test.npz'
    t = np.arange(n) / FPS
    if kp is None:
        kp = np.zeros((n, 17, 2))
    np.savez_compressed(npz, timestamps=t, keypoints=kp, confidences=np.ones((n, 17)),
                        commands=np.array(commands, dtype='<U16'), command_times=np.array(command_times, float))
    npz.with_suffix('.auto_candidates.json').write_text(json.dumps(dict(session='x', params={}, candidates=cands)))
    if labels is not None:
        npz.with_suffix('.labels.json').write_text(json.dumps(dict(labelers=labels)))
    return npz


def run_review(npz, keys, extra=()):
    """keys: a string of key presses; a tuple ('click', x, y) in place of a key clicks the window there."""
    cap = Mock()
    cap.get.return_value = 400
    cap.read.return_value = (True, np.zeros((48, 64, 3), np.uint8))
    mouse = []
    presses = iter([k for k in keys] + ['q'] * 3)

    def wait(ms):
        k = next(presses)
        if isinstance(k, tuple) and k[0] == 'wheel':     # ('wheel', +1 = scroll up / -1 = down)
            mouse[0](rl.cv2.EVENT_MOUSEWHEEL, 0, 0, (120 * k[1]) << 16, None)
            return 255
        if isinstance(k, tuple):
            mouse[0](rl.cv2.EVENT_LBUTTONDOWN, k[1], k[2], 0, None)
            return 255
        return ord(k)
    with patch.object(sys, 'argv', ['review_labels.py', str(npz), *extra]), \
         patch.object(rl.cv2, 'VideoCapture', return_value=cap), \
         patch.object(rl.cv2, 'namedWindow'), patch.object(rl.cv2, 'imshow'), \
         patch.object(rl.cv2, 'setMouseCallback', side_effect=lambda win, fn: mouse.append(fn)), \
         patch.object(rl.cv2, 'destroyAllWindows'), \
         patch.object(rl.cv2, 'waitKey', side_effect=wait), \
         contextlib.redirect_stdout(io.StringIO()) as out:
        rl.main()
    return out.getvalue()


def labels_of(npz):
    return json.loads(npz.with_suffix('.labels.json').read_text())['labelers']


class ReviewToolTests(unittest.TestCase):
    def test_keys_rule_tag_feint_and_nudge(self):
        with workspace_temp() as tmp:
            npz = write_session(tmp, [cand(100), cand(200), cand(300)])
            log = run_review(npz, '.yfn')            # nudge +1 & punch, feint, not a punch
            self.assertIn('TOWARD the target', log)
            marks = labels_of(npz)['reviewed']
            self.assertEqual(marks, [dict(preparation=None, departure=101, family='JAB', rule='toward_target'),
                                     dict(preparation=None, departure=200, family='FEINT', rule='toward_target')])
            states = json.loads(npz.with_suffix('.review_reviewed.json').read_text())
            self.assertEqual([states[i]['decision'] for i in '012'], ['yes', 'feint', 'no'])
            self.assertEqual((states['0']['auto'], states['0']['initial']), (100, 100))
            self.assertIn('feint 1', log)

    def test_step_frames_and_click_to_set_start(self):
        with workspace_temp() as tmp:
            npz = write_session(tmp, [cand(100), cand(150), cand(200), cand(250), cand(300), cand(350)])
            # each candidate opens paused on its start frame
            # 1: 'd' 'd' step 2 frames forward, 's' = start there, 'y'
            # 2: click the first thumbnail (offset -2) = start there, 'y'
            # 3: scroll up 3 times (3 frames back), click the big picture = start, click the CROSS button
            # 4: click the timeline at its far left (the clip's first frame, -15), 's', then the NO button
            # 5: click the HOOK from-the-left button straight away
            cross = next(b for b in rl.BUTTONS if b[1] == 'CROSS')
            no = next(b for b in rl.BUTTONS if b[1] == 'no')
            hook = next(b for b in rl.BUTTONS if b[1] == 'HOOK_L')
            mid = lambda b: ('click', (b[2] + b[4]) // 2, (b[3] + b[5]) // 2)
            run_review(npz, ['d', 'd', 's', 'y',
                             ('click', 5, rl.STRIP_Y + 10), 'y',
                             ('wheel', 1), ('wheel', 1), ('wheel', 1), ('click', 100, 100), mid(cross),
                             ('click', 2, rl.TIMELINE[0] + 5), 's', mid(no),
                             mid(hook),
                             # 6: scroll down 4 frames and click the type straight away = start there
                             ('wheel', -1), ('wheel', -1), ('wheel', -1), ('wheel', -1), mid(cross)])
            marks = labels_of(npz)['reviewed']
            self.assertEqual([(m['departure'], m['family']) for m in marks],
                             [(102, 'JAB'), (148, 'JAB'), (197, 'CROSS'), (300, 'HOOK_L'), (354, 'CROSS')])
            states = json.loads(npz.with_suffix('.review_reviewed.json').read_text())
            self.assertEqual((states['3']['decision'], states['3']['departure']), ('no', 235))

    def test_blind_second_reviewer_is_kept_apart_and_jittered(self):
        main_marks = [dict(preparation=None, departure=100, family='JAB')]
        with workspace_temp() as tmp:
            npz = write_session(tmp, [cand(100), cand(200), cand(300)], labels=dict(reviewed=main_marks))
            run_review(npz, 'yyy', ['--reviewer', 'alice'])
            saved = labels_of(npz)
            self.assertEqual(saved['reviewed'], main_marks)                       # training labels untouched
            starts = [m['departure'] for m in saved['check_alice']]
            self.assertTrue(all(abs(s - a) <= 2 for s, a in zip(starts, (100, 200, 300))), starts)
            self.assertTrue(npz.with_suffix('.review_check_alice.json').exists())
            with self.assertRaises(SystemExit):                                     # no messy candidates here:
                run_review(npz, 'y', ['--reviewer', 'alice', '--messy'])           # saves an empty set and stops
            self.assertIn('check_alice_messy', labels_of(npz))
            self.assertEqual(labels_of(npz)['reviewed'], main_marks)

    def test_reviewer_name_checked(self):
        with workspace_temp() as tmp:
            npz = write_session(tmp, [cand(100)])
            for bad in ('reviewed', 'a b', ''):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    run_review(npz, 'y', ['--reviewer', bad])

    def test_jitter_is_reproducible_and_bounded(self):
        c = cand(50)
        firsts = [rl.start_frame(c, i, 2, 'seed') for i in range(200)]
        self.assertEqual(firsts, [rl.start_frame(c, i, 2, 'seed') for i in range(200)])
        self.assertEqual(set(firsts), {48, 49, 50, 51, 52})
        self.assertEqual(rl.start_frame(c, 3, 0, 'seed'), 50)

    def test_sibling_passes(self):
        self.assertEqual(rl.sibling_labeler('reviewed'), 'reviewed_messy')
        self.assertEqual(rl.sibling_labeler('reviewed_messy'), 'reviewed')
        self.assertEqual(rl.sibling_labeler('check_alice'), 'check_alice_messy')
        self.assertEqual(rl.sibling_labeler('check_alice_messy'), 'check_alice')

    def test_same_punch_warning(self):
        t = np.arange(400) / FPS
        states = {'0': dict(departure=100, decision='yes'), '1': dict(departure=103, decision=None),
                  '2': dict(departure=140, decision='yes')}
        self.assertEqual(rl.nearby_punch(1, states['1'], states, [], t), (100, 'this pass'))
        self.assertEqual(rl.nearby_punch(1, states['1'], states, [104], t), (104, 'other pass'))
        self.assertIsNone(rl.nearby_punch(2, states['2'], states, [], t))     # 1.2 s from the others

    def test_marks_keep_old_decisions_untagged(self):
        cands = [cand(100), cand(200)]
        states = {'0': dict(departure=100, family='JAB', decision='yes'),               # old review
                  '1': dict(departure=201, family='JAB', decision='yes', rule='toward_target')}
        self.assertEqual(rl.accepted_marks(cands, states),
                         [dict(preparation=None, departure=100, family='JAB'),
                          dict(preparation=None, departure=201, family='JAB', rule='toward_target')])


def write_progress(npz, labeler, decisions):
    states = {str(i): dict(departure=d, decision=k, family='JAB', auto=a)
              for i, (k, d, a) in enumerate(decisions) if k is not None}
    npz.with_suffix(f'.review_{labeler}.json').write_text(json.dumps(states))


class AgreementTests(unittest.TestCase):
    def test_counts_kappa_and_start_differences(self):
        with workspace_temp() as tmp:
            npz = write_session(tmp, [cand(d) for d in (100, 200, 300, 400 - 50, 380)])
            write_progress(npz, 'reviewed', [('yes', 100, 100), ('yes', 201, 200), ('no', 300, 300),
                                             ('feint', 350, 350), ('yes', 380, 380)])
            write_progress(npz, 'check_alice', [('yes', 101, 100), ('yes', 204, 200), ('no', 300, 300),
                                                ('no', 350, 350), (None, 0, 380)])
            rows, notes = la.compare(npz, 'reviewed', 'check_alice')
            self.assertEqual(len(rows), 4)                          # the 5th wasn't decided by alice
            with contextlib.redirect_stdout(io.StringIO()) as out:
                res = la.report(npz, 'reviewed', 'check_alice', rows, notes)
            self.assertEqual((res['agree'], res['start_n'], res['within1'], res['within2']), (3, 2, 1, 1))
            self.assertAlmostEqual(res['mean_diff'], 2.0)
            m = np.array([[2, 0, 0], [0, 0, 1], [0, 0, 1]])
            self.assertAlmostEqual(res['kappa'], la.kappa(m))
            self.assertIn('marks later', out.getvalue())
            self.assertEqual(res['disagreements'], 2)                # feint vs no, and the 3-frame gap
            rows, _ = la.compare(npz, 'reviewed', 'check_alice', t_from=5., t_to=9.)   # clip: 6.7 s only
            self.assertEqual([r['auto'] for r in rows], [200])

    def test_kappa(self):
        self.assertAlmostEqual(la.kappa(np.diag([5, 3, 2])), 1.0)
        self.assertAlmostEqual(la.kappa(np.array([[5, 5], [5, 5]])), 0.0)

    def test_changed_candidate_list_is_not_compared(self):
        with workspace_temp() as tmp:
            npz = write_session(tmp, [cand(100), cand(200)])
            write_progress(npz, 'reviewed', [('yes', 100, 100), ('yes', 200, 200)])
            write_progress(npz, 'check_bob', [('yes', 100, 90), ('yes', 200, 200)])     # made on another list
            rows, notes = la.compare(npz, 'reviewed', 'check_bob')
            self.assertEqual(len(rows), 1)
            self.assertTrue(any('candidate list changed' in n for n in notes))


def punch_keypoints(n, punches):
    """Shoulders still at x 300/400 (width 100 px), wrists resting in guard. Each punch:
    (start frame, wrist k) moves that wrist 70 px outward in 5 frames and back in 7."""
    kp = np.zeros((n, 17, 2))
    kp[:, 5], kp[:, 6] = (300., 200.), (400., 200.)
    kp[:, 9], kp[:, 10] = (320., 150.), (380., 150.)
    out = np.r_[np.linspace(0, 70, 6)[1:], np.linspace(70, 0, 8)[1:]]
    for s, k in punches:
        sign = -1 if k == 0 else 1
        kp[s:s + len(out), 9 + k, 0] += sign * out[:max(0, min(len(out), n - s))]
    return kp


class ComboTests(unittest.TestCase):
    PUNCHES = [(60, 0), (72, 1), (150, 0), (240, 1), (330, 0)]   # a jab-cross pair 0.4 s apart, then singles

    def test_launch_starts_found_within_two_frames(self):
        n = 400
        L = cc.find_launches(np.arange(n) / FPS, punch_keypoints(n, self.PUNCHES), np.ones((n, 17)))
        starts = {(e['start'], e['wrist']) for e in L}
        for s, k in self.PUNCHES:
            self.assertTrue(any(abs(a - s) <= 2 and w == k for a, w in starts), (s, k, sorted(starts)))
        self.assertEqual(len(L), len(self.PUNCHES), "retractions must not count as launches")

    def test_only_uncovered_follow_ups_in_free_blocks_are_appended(self):
        n = 400
        with workspace_temp() as tmp:
            # FREE block 1.0-9.0 s; auto_label only found the first punch of each group
            npz = write_session(tmp, [cand(60, 'FREE', 'combo'), cand(150, 'FREE', 'clean', 'image-left')],
                                n=n, commands=('IDLE', 'FREE', 'JAB'), command_times=(0., 1.0, 9.0),
                                kp=punch_keypoints(n, self.PUNCHES))
            before = json.loads(npz.with_suffix('.auto_candidates.json').read_text())['candidates']
            # the synthetic session has no video, so only the keypoint signal is used
            with patch.object(sys, 'argv', ['combo_candidates.py', str(npz), '--no-flow']), \
                 contextlib.redirect_stdout(io.StringIO()):
                cc.main()
            data = json.loads(npz.with_suffix('.auto_candidates.json').read_text())
            added = data['candidates'][len(before):]
            self.assertEqual(data['candidates'][:len(before)], before)        # appended at the end only
            # the cross 0.4 s after the jab is a follow-up; the punch at 8 s is > 1 s after any candidate,
            # and the one at 11 s is outside the FREE block
            self.assertEqual([(c['wrist'], c['quality'], c['family'], c['source']) for c in added],
                             [('image-right', 'combo', 'FREE', 'combo')])
            self.assertLessEqual(abs(added[0]['departure'] - 72), 2)
            self.assertEqual(rl.review_candidates(data['candidates'], messy=True)[-1], added[0])
            with patch.object(sys, 'argv', ['combo_candidates.py', str(npz), '--no-flow']):
                with self.assertRaises(SystemExit):
                    cc.main()                                                   # running twice adds nothing

    def test_sessions_without_free_blocks_are_left_alone(self):
        with workspace_temp() as tmp:
            npz = write_session(tmp, [cand(60)], commands=('JAB',), command_times=(0.,))
            text = npz.with_suffix('.auto_candidates.json').read_text()
            with patch.object(sys, 'argv', ['combo_candidates.py', str(npz)]):
                with self.assertRaises(SystemExit):
                    cc.main()
            self.assertEqual(npz.with_suffix('.auto_candidates.json').read_text(), text)


if __name__ == '__main__':
    unittest.main()
