"""Tests for free-boxing recording (collect.py --script --free, free_blocks.py)
and for free-block candidates reaching review (review_labels.review_candidates).

Run: python -m pytest -q test_free_blocks.py
"""
import contextlib
import io
import json
import random
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import cv2
import numpy as np

import free_blocks
from free_blocks import BASELINE, FREE, PROMPTS, FreeSchedule
from pose_features import COMMANDS
from review_labels import review_candidates

POOL = COMMANDS + ('GUARD_SHIFT',)
HERE = Path(__file__).resolve().parent


@contextlib.contextmanager
def workspace_temp():
    # same pattern as test_anticipation.py: a plain folder in the workspace (Windows ACLs)
    root = Path.cwd().resolve()
    path = root / ('fb_' + uuid4().hex[:8])
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != root:
            raise RuntimeError('Refusing cleanup outside workspace')
        shutil.rmtree(path)


def run_schedule(seconds, fps=30., **kw):
    """Drive a FreeSchedule like collect.py does; returns the timeline it produced."""
    sched = FreeSchedule(POOL, rng=random.Random(0), **kw)
    blocks, block_times, commands, command_times = [], [], [], []
    for t in np.arange(0, seconds, 1 / fps):
        block, command = sched.update(float(t))
        if block:
            blocks.append(block)
            block_times.append(float(t))
        if command:
            commands.append(command)
            command_times.append(float(t))
    return blocks, np.array(block_times), commands, np.array(command_times), sched


class FreeScheduleTests(unittest.TestCase):
    def test_layout_baseline_then_prompts_and_free_alternate(self):
        blocks, bt, cmds, ct, _ = run_schedule(400, baseline_s=20, prompt_s=60, free_s=60)
        self.assertEqual(blocks[:6], [BASELINE, PROMPTS, FREE, PROMPTS, FREE, PROMPTS])
        frame = 1 / 30 + 1e-9
        lengths = np.diff(bt)
        self.assertAlmostEqual(lengths[0], 20, delta=frame)                         # baseline
        for kind, length in zip(blocks[1:], lengths[1:]):
            if kind == FREE:
                self.assertAlmostEqual(length, 60, delta=frame)
            else:   # prompts end at the first prompt change after 60 s: never cut a prompt short
                self.assertGreaterEqual(length, 60 - frame)
                self.assertLessEqual(length, 64 + frame)
        self.assertTrue(np.all(np.diff(ct) > 0), "command times must strictly increase")
        # every block start carries its command; baseline = IDLE (a known don't-punch prompt)
        self.assertEqual((cmds[0], ct[0]), ('IDLE', bt[0]))
        for kind, t0 in zip(blocks, bt):
            j = int(np.flatnonzero(ct == t0)[0])
            if kind == FREE:
                self.assertEqual(cmds[j], 'FREE')
            elif kind == PROMPTS:
                self.assertIn(cmds[j], POOL)
        # inside FREE / BASELINE blocks no prompt appears after the block's own command
        ends = list(bt[1:]) + [np.inf]
        for kind, t0, t1 in zip(blocks, bt, ends):
            inside = [c for c, tc in zip(cmds, ct) if t0 < tc < t1]
            if kind in (FREE, BASELINE):
                self.assertEqual(inside, [])
            else:
                self.assertTrue(all(c in POOL for c in inside))
                gaps = np.diff([tc for tc in ct if t0 <= tc < t1])
                self.assertTrue(np.all((gaps >= 2 - frame) & (gaps <= 4 + frame)))

    def test_no_baseline(self):
        blocks, _, cmds, _, _ = run_schedule(200, baseline_s=0, prompt_s=60, free_s=60)
        self.assertEqual(blocks[:3], [PROMPTS, FREE, PROMPTS])
        self.assertIn(cmds[0], POOL)

    def test_switch_in_countdown(self):
        sched = FreeSchedule(POOL, baseline_s=20, prompt_s=60, free_s=30, rng=random.Random(1))
        sched.update(0.)
        self.assertAlmostEqual(sched.switch_in(5.), 15.)              # baseline: exact
        sched.update(20.)                                             # prompts start
        self.assertIsNone(sched.switch_in(21.))                       # not known yet
        t = 20.
        while sched.kind == PROMPTS:
            t += 1 / 30
            before = sched.switch_in(t)
            sched.update(t)
        self.assertEqual(sched.kind, FREE)
        self.assertIsNotNone(before)                                  # known during the last prompt
        self.assertAlmostEqual(sched.switch_in(t + 10), 20.)

    def test_bad_lengths_rejected(self):
        with self.assertRaises(ValueError):
            FreeSchedule(POOL, prompt_s=0)
        with self.assertRaises(ValueError):
            FreeSchedule(POOL, free_s=-1)

    def test_screens_draw_without_error_and_fit(self):
        sched = FreeSchedule(POOL, baseline_s=20, prompt_s=60, free_s=60, rng=random.Random(0))
        for t in (0., 17.5):   # baseline, then its last seconds (blinking border, countdown)
            sched.update(t)
            frame = np.zeros((480, 640, 3), np.uint8)
            self.assertTrue(free_blocks.draw(frame, sched, t))
            self.assertGreater(frame.sum(), 0)
        rows = [(line, free_blocks.LINE_SCALE, 2) for line in free_blocks.FREE_LINES + free_blocks.BASELINE_LINES]
        rows += [("FREE BOXING  60s", free_blocks.TITLE_SCALE, 3), ("FREE BOXING next, in 5s", free_blocks.NOTE_SCALE, 2)]
        for line, scale, thick in rows:
            (w, _), _ = cv2.getTextSize(line, free_blocks.FONT, scale, thick + 2)
            self.assertLess(12 + w + 10, 640, f"too wide for a 640 px frame: {line!r}")

    def test_summary(self):
        s = free_blocks.summary([BASELINE, PROMPTS, FREE, PROMPTS], [0., 20., 82., 142.], 200.)
        self.assertEqual(s, "Blocks: BASELINE 1 x 20 s | PROMPTS 2 x 60 s | FREE 1 x 60 s (60 s of free boxing)")
        self.assertEqual(free_blocks.summary([], [], 10.), "")


def load(path):
    # read everything and close the file (Windows can't delete an open .npz)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fake_video(n):
    cap = Mock()
    cap.read.side_effect = [(True, np.full((120, 160, 3), 60 + i % 100, np.uint8)) for i in range(n)] + [(False, None)]
    cap.get.return_value = 30.   # -> timestamps index / 30 (see collect.video_timestamp)
    return cap


def run_collect(argv, cap):
    import collect
    with patch.object(sys, 'argv', ['collect.py'] + argv), \
         patch.object(collect, 'open_source', return_value=(cap, False)), \
         patch.object(collect, 'PoseRunner', return_value=lambda frame: None), \
         patch.object(collect.cv2, 'imshow'), patch.object(collect.cv2, 'destroyAllWindows'), \
         patch.object(collect.cv2, 'waitKey', return_value=255), \
         contextlib.redirect_stdout(io.StringIO()) as out:
        collect.main()
    return out.getvalue()


class CollectFreeTests(unittest.TestCase):
    def test_free_session_saves_blocks_and_repose_keeps_them(self):
        random.seed(0)
        with workspace_temp() as tmp, contextlib.chdir(tmp):
            log = run_collect(['--source', 'fake.mp4', '--script', '--free', '--baseline-seconds', '0.1',
                               '--prompt-seconds', '0.1', '--free-seconds', '0.5',
                               '--performer', 'alice', '--stance', 'southpaw'], fake_video(160))
            self.assertIn('FREE-BOXING SESSION', log)
            self.assertIn('Blocks: BASELINE 1 x', log)
            path = next(Path('data').glob('*.npz'))
            z = load(path)
            t = z['timestamps']
            self.assertEqual(z['blocks'].tolist(), [BASELINE, PROMPTS, FREE, PROMPTS])
            bt = z['block_times']
            self.assertEqual(bt[0], t[0])
            self.assertTrue(np.all(np.diff(bt) > 0))
            cmds, ct = z['commands'].tolist(), z['command_times']
            self.assertEqual(cmds[0], 'IDLE')
            self.assertEqual(cmds[list(ct).index(bt[2])], 'FREE')
            self.assertAlmostEqual(bt[3] - bt[2], 0.5, delta=1 / 30 + 1e-9)
            self.assertTrue(np.all(np.diff(ct) > 0))
            self.assertEqual((str(z['performer']), str(z['stance'])), ('alice', 'southpaw'))
            video = cv2.VideoCapture(str(path.with_suffix('.mp4')))
            try:
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 160)
            finally:
                video.release()
            # re-posing a recorded video (--from-video) must keep the blocks and the stance
            run_collect(['--from-video', str(path.with_suffix('.mp4'))], fake_video(160))
            newpath = next(p for p in Path('data').glob('*.npz') if p != path)
            r = load(newpath)
            for key in ('blocks', 'block_times', 'commands', 'command_times', 'timestamps'):
                np.testing.assert_array_equal(r[key], z[key], err_msg=key)
            self.assertEqual(str(r['stance']), 'southpaw')

    def test_plain_script_session_has_no_blocks(self):
        random.seed(0)
        with workspace_temp() as tmp, contextlib.chdir(tmp):
            run_collect(['--source', 'fake.mp4', '--script'], fake_video(12))
            z = load(next(Path('data').glob('*.npz')))
            self.assertEqual(len(z['blocks']), 0)
            self.assertEqual(str(z['stance']), 'unknown')
            self.assertTrue(all(c in POOL for c in z['commands']))

    def test_light_check_records_nothing(self):
        with workspace_temp() as tmp, contextlib.chdir(tmp):
            log = run_collect(['--source', 'fake.mp4', '--check-light', '--script', '--free'], fake_video(60))
            self.assertIn('Light check done (nothing saved)', log)
            self.assertEqual(list(Path('data').glob('*')), [])

    def test_every_prompt_has_plain_words(self):
        for c in POOL:
            self.assertIn(c, free_blocks.COMMAND_HELP)
        self.assertEqual(free_blocks.light_level(30.)[1], 'light OK')
        self.assertEqual(free_blocks.light_level(20.)[1], 'too dark: add light')

    def test_hooks_named_by_hand_and_swap_note(self):
        import json
        import session_notes
        # HOOK_L = fist from image-left = the puncher's RIGHT hand (camera faces them)
        self.assertEqual(free_blocks.COMMAND_SHOWN['HOOK_L'], 'RIGHT-HAND HOOK')
        self.assertEqual(free_blocks.COMMAND_SHOWN['HOOK_R'], 'LEFT-HAND HOOK')
        with workspace_temp() as tmp:
            notes = Path(tmp) / 'notes.json'
            notes.write_text(json.dumps({'sessions': {'s_swapped': {'hooks_swapped': True}}}))
            cmds = np.array(['JAB', 'HOOK_L', 'HOOK_R', 'FREE'], dtype='<U16')
            self.assertEqual(session_notes.commands('data/s_swapped.npz', cmds, notes).tolist(),
                             ['JAB', 'HOOK_R', 'HOOK_L', 'FREE'])
            self.assertEqual(session_notes.commands('data/other.npz', cmds, notes).tolist(), cmds.tolist())
        real = session_notes.load()
        self.assertTrue(real['sessions']['session_20261121_213319_013537']['hooks_swapped'])

    def test_out_of_frame_alarm(self):
        xy, conf = np.full((17, 2), 300.), np.ones(17)
        xy[5, 0], xy[6, 0] = 250., 350.
        self.assertIsNone(free_blocks.framing_problem(xy, conf, 640))
        self.assertIn('NOBODY', free_blocks.framing_problem(None, None, 640))
        low = conf.copy(); low[6] = .1
        self.assertIn('SHOULDERS', free_blocks.framing_problem(xy, low, 640))
        edge = xy.copy(); edge[5, 0] = 5.
        self.assertIn('EDGE', free_blocks.framing_problem(edge, conf, 640))
        alarm = free_blocks.FramingAlarm(grace=.4, sound=False)
        self.assertIsNone(alarm.update(0., 'SHOULDERS LOST'))      # brief blur: no alarm yet
        self.assertIsNone(alarm.update(.3, 'SHOULDERS LOST'))
        self.assertEqual(alarm.update(.5, 'SHOULDERS LOST'), 'SHOULDERS LOST')
        self.assertIsNone(alarm.update(.6, None))                  # back in frame: stops at once
        self.assertIsNone(alarm.update(.7, 'SHOULDERS LOST'))      # grace starts over
        frame = np.zeros((480, 640, 3), np.uint8)
        free_blocks.draw_alarm(frame, 'SHOULDERS LOST', .3)
        self.assertGreater(frame.sum(), 0)

    def test_argument_checks(self):
        import collect
        bad = [
            ['--source', 'x.mp4', '--free'],                                          # needs --script
            ['--source', 'x.mp4', '--script', '--free', '--negatives-only'],
            ['--source', 'x.mp4', '--script', '--free', '--free-seconds', '0'],
            ['--source', '1', '--script', '--free', '--mount', 'head'],               # live, no --performer
        ]
        for argv in bad:
            with patch.object(sys, 'argv', ['collect.py'] + argv), \
                 patch.object(collect, 'PoseRunner', side_effect=AssertionError('got past the checks')), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit, msg=str(argv)):
                    collect.main()


def old_review_selection(all_cands, messy=False, everything=False):
    """review_labels.py's selection before free blocks existed (kept to prove nothing changed)."""
    punch = ("JAB", "CROSS", "HOOK_L", "HOOK_R", "UPPERCUT")
    cands = all_cands if everything else [c for c in all_cands if c["quality"] == "clean"]
    if messy:
        return [c for c in all_cands if c["quality"] != "clean" and ("guess" not in c or c["family"] in punch)]
    if not everything:
        cands = [c for c in cands if "guess" not in c or c["family"] in punch]
    return cands


class ReviewSelectionTests(unittest.TestCase):
    def test_free_block_candidates_are_reviewed_like_punch_prompts(self):
        def cand(family, quality, scripted=True):
            c = dict(departure=len(family), family=family, quality=quality)
            if scripted:
                c["guess"] = "JAB"
            return c
        cands = [cand('JAB', 'clean'), cand('FREE', 'clean'), cand('IDLE', 'clean'), cand('GUARD_SHIFT', 'clean'),
                 cand('FREE', 'combo'), cand('JAB', 'body'), cand('FEINT', 'combo'), cand('X', 'clean', False)]
        fam = lambda cs: [(c['family'], c['quality']) for c in cs]
        self.assertEqual(fam(review_candidates(cands)),
                         [('JAB', 'clean'), ('FREE', 'clean'), ('X', 'clean')])
        self.assertEqual(fam(review_candidates(cands, messy=True)), [('FREE', 'combo'), ('JAB', 'body')])
        self.assertEqual(len(review_candidates(cands, everything=True)), len(cands))

    def test_reactive_training_treats_free_like_a_punch_prompt(self):
        """Rename the JAB and CROSS prompts (and their punches) of a real session to FREE:
        the unchanged early_dataset.build must give exactly the same training labels."""
        import early_dataset
        stem = 'session_20261121_095106_097514'
        src = HERE / 'data' / f'{stem}.npz'
        if not (src.exists() and (HERE / 'features' / f'{stem}.energy.npy').exists()):
            self.skipTest('needs the committed session and its cached features')
        with contextlib.chdir(HERE), workspace_temp() as tmp:   # build() finds features/ from the cwd
            dst = tmp / src.name
            z = load(src)
            z['commands'] = np.where(np.isin(z['commands'], ['JAB', 'CROSS']), 'FREE', z['commands']).astype('<U16')
            np.savez_compressed(dst, **z)
            labels = json.loads(src.with_suffix('.labels.json').read_text())
            for marks in labels['labelers'].values():
                for m in marks:
                    if m['family'] in ('JAB', 'CROSS'):
                        m['family'] = 'FREE'
            dst.with_suffix('.labels.json').write_text(json.dumps(labels))
            shutil.copyfile(src.with_suffix('.auto_candidates.json'), dst.with_suffix('.auto_candidates.json'))
            before, after = early_dataset.build(src), early_dataset.build(dst)
        np.testing.assert_array_equal(before['y'], after['y'])
        np.testing.assert_array_equal(before['scripted_neg'], after['scripted_neg'])
        self.assertEqual([e['departure'] for e in before['events']], [e['departure'] for e in after['events']])
        self.assertIn('FREE', [e['family'] for e in after['events']])

    def test_existing_sessions_select_exactly_what_they_did_before(self):
        files = sorted(HERE.glob('data*/*.auto_candidates.json'))
        self.assertTrue(files, "no recorded sessions found")
        for f in files:
            all_cands = json.loads(f.read_text())["candidates"]
            if any(c.get("family") == "FREE" for c in all_cands):
                continue    # a --free session: the old selection never knew free boxing
            for kw in (dict(), dict(messy=True), dict(everything=True), dict(messy=True, everything=True)):
                self.assertEqual(review_candidates(all_cands, **kw), old_review_selection(all_cands, **kw),
                                 f"{f.name} {kw}")


if __name__ == '__main__':
    unittest.main()
