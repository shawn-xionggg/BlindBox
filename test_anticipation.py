"""Deterministic synthetic verification; no webcam, model download, or real-punch claims.

Run: python -m unittest -v test_anticipation
"""
import contextlib
import csv
import io
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
from uuid import uuid4
import joblib
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from pose_features import (FeatureStream, PoseRunner, ReactiveOnset, CueGate,
                           normalize, frame_features, feature_names, reactive_onset)
from collect import save_session
from train_anticipator import (build_labels, event_metrics, fit_candidates,
                               load_session, session_features, warnings_from_probabilities,
                               onsets, evaluate)


@contextlib.contextmanager
def workspace_temp():
    # TemporaryDirectory's restrictive Windows ACLs are incompatible with this
    # sandbox. A normal workspace directory inherits the permitted workspace ACL.
    root = Path.cwd().resolve()
    # Keep absolute MP4 paths below Windows/OpenCV's legacy MAX_PATH limit.
    path = root/('at_'+uuid4().hex[:8])
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != root:
            raise RuntimeError('Refusing cleanup outside workspace')
        shutil.rmtree(path)


def synthetic_session(seed=1, punches=18):
    """Idle noise plus a 200 ms shoulder/elbow wind-up and then fast wrist motion.

    Session-dependent camera translation, scale, frame jitter and noise ensure
    held-out evaluation exercises the causal normalized features.
    """
    rng = np.random.default_rng(seed)
    n = punches*60+40
    t = np.cumsum(rng.uniform(.031, .035, n))
    t -= t[0]
    base = np.zeros((17, 2))
    base[0] = (0, -.75)
    base[5], base[6] = (-.5, 0), (.5, 0)
    base[7], base[8] = (-.8, .45), (.8, .45)
    base[9], base[10] = (-.8, -.35), (.8, -.35)
    base[11], base[12] = (-.4, 1.2), (.4, 1.2)
    xy = np.repeat(base[None], n, axis=0)
    frames, sides = [], []
    for k in range(punches):
        idx = 35+k*60
        side = ('LEFT', 'RIGHT', 'CENTER')[k % 3]
        frames.append(idx)
        sides.append(side)
        sign = -1 if side == 'LEFT' else 1
        elbow, wrist = (7, 9) if sign < 0 else (8, 10)
        for j in range(idx-6, idx):
            strength = (j-(idx-6)+1)/6
            xy[j, 5, 1] -= .16*strength*sign
            xy[j, 6, 1] += .16*strength*sign
            xy[j, elbow, 0] += sign*.28*strength
            xy[j, elbow, 1] += .22*strength
            if side == 'CENTER':
                xy[j, 0, 0] += .18*strength
        # Keep wrists still during wind-up; move rapidly starting exactly at onset.
        for j in range(idx, idx+5):
            xy[j, wrist, 1] -= (j-idx+1)*.43
            if side == 'CENTER':
                xy[j, wrist, 0] = .1
        # Slow return avoids a second high-speed burst.
        for j in range(idx+5, idx+20):
            xy[j, wrist] += (xy[idx+4, wrist]-base[wrist])*(1-(j-idx-4)/15)
    xy += rng.normal(0, .0015, xy.shape)
    for i in range(n):
        scale = 110+seed*3+2*np.sin(t[i])
        offset = np.array([320+25*np.sin(t[i]*2), 160+12*np.cos(t[i])])
        xy[i] = xy[i]*scale+offset
    return dict(timestamps=t, keypoints=xy, confidences=np.ones((n, 17)),
                onset_frames=np.array(frames), onset_sides=np.array(sides))


class AnticipationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(limits=1)
        cls.train = synthetic_session(1)
        cls.test = synthetic_session(12)
        x, valid = session_features(cls.train)
        y, keep = build_labels(cls.train['timestamps'], onsets(cls.train))
        cls.models = fit_candidates(x[keep & valid], y[keep & valid])

    @classmethod
    def tearDownClass(cls):
        cls.limit.restore_original_limits()

    def test_normalization_camera_translation_and_scale(self):
        xy = self.train['keypoints'][0]
        np.testing.assert_allclose(normalize(xy), normalize(xy*2.4+np.array([81, -30])), atol=1e-12)

    def test_pose_runner_largest_person_and_empty(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)
            def cpu(self):
                return self
            def numpy(self):
                return self.value
            def __getitem__(self, index):
                return Tensor(self.value[index])
            def __len__(self):
                return len(self.value)

        class Boxes:
            xyxy = Tensor([[0, 0, 10, 20], [0, 0, 30, 40]])
            def __len__(self):
                return len(self.xyxy)

        runner = PoseRunner.__new__(PoseRunner)
        runner.imgsz = 640
        points = np.array([np.zeros((17, 2)), np.ones((17, 2))])
        result = SimpleNamespace(boxes=Boxes(), keypoints=SimpleNamespace(xy=Tensor(points), conf=Tensor(np.ones((2, 17)))))
        runner.model = Mock(return_value=[result])
        xy, conf = runner(np.zeros((2, 2, 3)))
        np.testing.assert_equal(xy, points[1])
        result.keypoints = None
        self.assertIsNone(runner(np.zeros((2, 2, 3))))

    def test_features_missing_person_imputation_and_flags(self):
        s = FeatureStream()
        x = self.train['keypoints'][0]
        good = s.update(0, x, np.ones(17))
        bad = s.update(.033, None)
        self.assertEqual(len(bad), len(feature_names()))
        self.assertTrue(np.isfinite(bad).all())
        np.testing.assert_equal(good[:18], bad[:18])
        np.testing.assert_equal(bad[36:45], 0)
        self.assertEqual(bad[-1], 0)
        c = np.ones(17)
        c[11:13] = 0
        moved = x.copy()
        moved[11:13] = 1e10
        f = s.update(.066, moved, c)
        np.testing.assert_equal(f[14:18], good[14:18])
        self.assertEqual(f[-1], 0)
        self.assertTrue(np.isfinite(frame_features([])).all())
        s2 = FeatureStream()
        self.assertTrue(np.isfinite(s2.update(0)).all())

    def test_velocity_uses_seconds(self):
        xy = self.train['keypoints'][0].copy()
        s = FeatureStream()
        s.update(0, xy, np.ones(17))
        width = np.linalg.norm(xy[6]-xy[5])
        xy[9, 0] += width*.2
        f = s.update(.1, xy, np.ones(17))
        self.assertAlmostEqual(f[18+5*2], 2., places=8)

    def test_features_are_causal(self):
        full, _ = session_features(self.train)
        prefix = {k: v[:100] for k, v in self.train.items() if k in ('timestamps', 'keypoints', 'confidences')}
        short, _ = session_features(prefix)
        np.testing.assert_equal(full[:100], short)

    def test_reactive_onsets_and_sides(self):
        s = self.train
        detected = reactive_onset(s['timestamps'], s['keypoints'], s['confidences'])
        self.assertEqual(len(detected), len(s['onset_frames']))
        for e, idx, side in zip(detected, s['onset_frames'], s['onset_sides']):
            self.assertLessEqual(abs(e['frame']-idx), 1)
            self.assertGreater(e['detected_time'], e['time'])
            self.assertEqual(e['side'], side)

    def test_idle_and_missing_pose_do_not_cue(self):
        s = self.train
        xy = np.repeat(s['keypoints'][0][None], 100, axis=0)
        conf = np.ones((100, 17))
        conf[30:40] = 0
        xy[30:40] = np.nan
        self.assertEqual(reactive_onset(np.arange(100)/30, xy, conf), [])

    def test_label_time_boundaries_and_recovery(self):
        t = np.array([.69, .7, .8, .999, 1., 1.1, 1.4, 1.401, 1.8])
        y, keep = build_labels(t, [dict(time=1., side='RIGHT')])
        np.testing.assert_equal(y, ['NONE', 'RIGHT', 'RIGHT', 'RIGHT', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'])
        np.testing.assert_equal(keep, [1, 1, 1, 1, 0, 0, 0, 1, 1])
        y, _ = build_labels([.9], [dict(time=1.1, side='LEFT'), dict(time=1., side='RIGHT')])
        self.assertEqual(y[0], 'RIGHT')

    def test_cue_gate_same_side_confirmation_cooldown_missing(self):
        gate = CueGate(.5)
        self.assertIsNone(gate.update(0, {'LEFT': .8}))
        self.assertIsNone(gate.update(.03, {'RIGHT': .8}))
        self.assertEqual(gate.update(.06, {'RIGHT': .8})['side'], 'RIGHT')
        self.assertIsNone(gate.update(.09, {'RIGHT': .8}))
        self.assertIsNone(gate.update(.66, {}))
        self.assertIsNone(gate.update(.70, {'LEFT': .8}))
        self.assertIsNotNone(gate.update(.73, {'LEFT': .8}))

    def test_event_metrics_one_to_one_leads_false_rate(self):
        events = [dict(time=1., side='LEFT'), dict(time=2., side='RIGHT'), dict(time=3., side='CENTER')]
        warnings = [dict(time=.8, side='LEFT'), dict(time=.9, side='LEFT'),
                    dict(time=1.95, side='LEFT'), dict(time=3.1, side='CENTER'), dict(time=4., side='LEFT')]
        m = event_metrics(warnings, events, 0, 5)
        self.assertEqual(m['warned'], 2)
        self.assertAlmostEqual(m['useful_pct'], 100/3)
        self.assertAlmostEqual(m['median_ms'], 125)
        self.assertAlmostEqual(m['p25_ms'], 87.5)
        self.assertAlmostEqual(m['p75_ms'], 162.5)
        self.assertEqual(m['side_accuracy'], 50)
        self.assertEqual(m['false_alarms'], 2)
        self.assertEqual(m['recovery_cues'], 1)
        self.assertAlmostEqual(m['false_per_min'], 120/3.8)
        b = event_metrics([dict(time=1.06, side='LEFT')], events, 0, 5, reactive=True)
        self.assertEqual(b['warned'], 0)
        self.assertAlmostEqual(b['median_ms'], -60)
        self.assertEqual(event_metrics([], [], 0, 0)['warned'], 0)

    def test_synthetic_heldout_warning_before_onset(self):
        s = self.test
        x, valid = session_features(s)
        validation = synthetic_session(6)
        vx, vv = session_features(validation)
        scores = {name: evaluate(model, validation, vx, vv, .5, .3)
                  for name, model in self.models.items()}
        chosen = max(scores, key=lambda name: scores[name]['useful_pct']+.25*scores[name]['warned_pct']
                     -5*scores[name]['false_per_min'])
        for name, model in self.models.items():
            m = evaluate(model, s, x, valid, .5, .3)
            print(f"\nSynthetic {name}: {m['warned']}/{m['punches']} warned, "
                  f"median lead {m['median_ms']:.1f} ms, useful {m['useful_pct']:.1f}%, false/min {m['false_per_min']:.2f}")
            if name == chosen:
                self.assertGreaterEqual(m['warned_pct'], 80)
                self.assertGreaterEqual(m['median_ms'], 100)
                self.assertGreaterEqual(m['side_accuracy'], 90)

    def test_session_roundtrip_and_model_serialization(self):
        s = self.train
        with workspace_temp() as tmp:
            path = Path(tmp)/'session.npz'
            labels = [dict(frame=int(i), side=str(side), source='auto')
                      for i, side in zip(s['onset_frames'], s['onset_sides'])]
            save_session(path, s['timestamps'], s['keypoints'], s['confidences'], labels, [2], 'synthetic', 'synthetic')
            loaded = load_session(path)
            np.testing.assert_equal(loaded['onset_frames'], s['onset_frames'])
            np.testing.assert_equal(loaded['rejected_frames'], [2])
            model = self.models['mlp']
            joblib.dump(model, Path(tmp)/'model.joblib')
            x, _ = session_features(s)
            np.testing.assert_equal(model.predict_proba(x[:4]), joblib.load(Path(tmp)/'model.joblib').predict_proba(x[:4]))

    def test_too_little_data_cli(self):
        with workspace_temp() as tmp:
            result = subprocess.run([sys.executable, 'train_anticipator.py', '--data', tmp], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Too little data: 0 onsets', result.stdout)

    def test_full_training_reporting_and_bundle(self):
        import train_anticipator
        with workspace_temp() as tmp:
            for name, seed in (('session_01', 1), ('session_02', 6), ('session_03', 12)):
                s = synthetic_session(seed)
                labels = [dict(frame=int(i), side=str(side), source='auto')
                          for i, side in zip(s['onset_frames'], s['onset_sides'])]
                save_session(tmp/(name+'.npz'), s['timestamps'], s['keypoints'],
                             s['confidences'], labels, [], 'synthetic', 'synthetic', name)
            # A re-pose copy must be deduplicated before any split.
            shutil.copyfile(tmp/'session_01.npz', tmp/'session_01_reposed.npz')
            output = tmp/'anticipator.joblib'
            console = io.StringIO()
            with patch.object(sys, 'argv', ['train_anticipator.py', '--data', str(tmp), '--output', str(output)]), \
                 patch.object(train_anticipator, 'fit_candidates', return_value=self.models), \
                 contextlib.redirect_stdout(console):
                train_anticipator.main()
            bundle = joblib.load(output)
            self.assertEqual(bundle['selected'], 'hist')
            self.assertEqual(bundle['feature_names'], feature_names())
            self.assertEqual(bundle['train_sessions'], ['session_01_reposed.npz'])
            self.assertEqual(bundle['validation_session'], 'session_02.npz')
            self.assertEqual(bundle['test_session'], 'session_03.npz')
            for message in ('test threshold 0.3', 'test threshold 0.5', 'test threshold 0.7',
                            'reactive baseline', 'Top 15 held-out', 'using session_01_reposed.npz only'):
                self.assertIn(message, console.getvalue())

    def test_collector_video_manual_and_rejected_labels(self):
        import collect
        frames = [np.full((120, 160, 3), 80+i*10, np.uint8) for i in range(4)]
        cap = Mock()
        cap.read.side_effect = [(True, f) for f in frames]+[(False, None)]
        cap.get.return_value = 30.
        detector = Mock()
        detector.update.side_effect = [dict(frame=0, time=0., side='LEFT', detected_time=0., speed=6.), None, None, None]
        with workspace_temp() as tmp, contextlib.chdir(tmp), \
             patch.object(sys, 'argv', ['collect.py', '--source', 'fake.mp4']), \
             patch.object(collect, 'open_source', return_value=(cap, False)), \
             patch.object(collect, 'PoseRunner', return_value=lambda frame: None), \
             patch.object(collect, 'LaunchDetector', return_value=detector), \
             patch.object(collect.cv2, 'imshow'), patch.object(collect.cv2, 'destroyAllWindows'), \
             patch.object(collect.cv2, 'waitKey', side_effect=[ord('l'), ord('p'), ord('x'), ord('q')]), \
             contextlib.redirect_stdout(io.StringIO()):
            collect.main()
            path = next(Path('data').glob('*.npz'))
            s = load_session(path)
            np.testing.assert_equal(s['onset_frames'], [1])
            np.testing.assert_equal(s['onset_sides'], ['LEFT'])
            np.testing.assert_equal(s['onset_sources'], ['manual'])
            np.testing.assert_equal(s['rejected_frames'], [0])
            video = cv2.VideoCapture(str(path.with_suffix('.mp4')))
            try:
                self.assertTrue(video.isOpened())
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
                ok, raw = video.read()
                self.assertTrue(ok)
                # A raw solid frame must have no skeleton/HUD painted into it.
                self.assertLess(float(raw.std()), 3.)
            finally:
                video.release()
            # Re-posing preserves original capture timestamps, manual corrections,
            # rejection records, and identity, rather than treating it as new data.
            cap.read.side_effect = [(True, np.full((120, 160, 3), 80, np.uint8)) for _ in range(4)]
            detector.update.side_effect = [None]*4
            with patch.object(sys, 'argv', ['collect.py', '--from-video', str(path)]):
                # --from-video points to the video; paired NPZ supplies metadata.
                sys.argv[-1] = str(path.with_suffix('.mp4'))
                with patch.object(collect.cv2, 'waitKey', side_effect=[255, 255, 255, ord('q')]):
                    collect.main()
            newpath = next(p for p in Path('data').glob('*.npz') if p != path)
            reposed = load_session(newpath)
            np.testing.assert_equal(reposed['timestamps'], s['timestamps'])
            np.testing.assert_equal(reposed['onset_frames'], [1])
            np.testing.assert_equal(reposed['onset_sources'], ['manual'])
            self.assertEqual(str(reposed['session_id']), str(s['session_id']))

    def test_live_early_cue_lead_log_and_reactive_backup(self):
        import anticipate_live
        cap = Mock()
        cap.get.return_value = 30.
        cap.read.side_effect = [(True, np.zeros((240, 400, 3), np.uint8)) for _ in range(30)]+[(False, None)]
        xy = self.train['keypoints'][0]
        predictor = Mock()
        predictor.classes_ = np.array(['NONE', 'LEFT'])
        predictor.predict_proba.side_effect = [np.array([[.1, .9]])]*2+[np.array([[.99, .01]])]*28
        detector = Mock()
        reactive_events = [None]*30
        reactive_events[5] = dict(frame=3, time=.1, side='LEFT', detected_time=5/30, speed=6.)
        reactive_events[25] = dict(frame=24, time=.8, side='RIGHT', detected_time=25/30, speed=6.)
        detector.update.side_effect = reactive_events
        beeper = Mock()
        cues = {'DODGE RIGHT': (1200, 120, (0, 200, 255)), 'DODGE LEFT': (600, 120, (255, 160, 0)), 'DUCK': (900, 300, (0, 0, 255))}
        original = SimpleNamespace(CUES=cues, beep=beeper)
        bundle = dict(model=predictor, window=8, horizon=.3, feature_names=feature_names())
        with workspace_temp() as tmp, contextlib.chdir(tmp), \
             patch.object(sys, 'argv', ['anticipate_live.py', '--source', 'fake.mp4', '--detector', 'reactive']), \
             patch.dict(sys.modules, {'punch_telegraph': original}), \
             patch.object(anticipate_live.joblib, 'load', return_value=bundle), \
             patch.object(anticipate_live, 'PoseRunner', return_value=lambda frame: (xy, np.ones(17))), \
             patch.object(anticipate_live, 'open_source', return_value=(cap, False)), \
             patch.object(anticipate_live, 'ReactiveOnset', return_value=detector), \
             patch.object(anticipate_live.cv2, 'imshow'), patch.object(anticipate_live.cv2, 'destroyAllWindows'), \
             patch.object(anticipate_live.cv2, 'waitKey', return_value=255), \
             contextlib.redirect_stdout(io.StringIO()):
            anticipate_live.main()
            with Path('anticipate_log.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r['event'] for r in rows], ['early', 'onset', 'onset', 'reactive_backup'])
            self.assertAlmostEqual(float(rows[1]['lead_ms']), 100-1000/30)
            self.assertEqual(beeper.call_count, 2)


class AddendumTests(unittest.TestCase):
    def session(self, n=30):
        base = synthetic_session(1, punches=1)['keypoints'][0]
        return dict(timestamps=np.arange(n)/30, keypoints=np.repeat(base[None], n, axis=0),
                    confidences=np.ones((n, 17)), onset_frames=np.array([], dtype=int),
                    onset_sides=np.array([], dtype='<U6'))

    def test_static_wrist_straight_growth_detected(self):
        from pose_features import MotionOnset
        s = self.session()
        stream, detector = FeatureStream(), MotionOnset()
        events = []
        for i, t in enumerate(s['timestamps']):
            xy = s['keypoints'][i]
            area = 400*np.exp(6*max(0, t-.2))
            stream.update(t, xy, np.ones(17), [[*xy[9], area]])
            event = detector.update(stream.history[-1], i)
            if event:
                events.append(event)
        self.assertTrue(events)
        self.assertEqual(events[0]['signal'], 'glove')
        self.assertLessEqual(abs(events[0]['time']-.2333), .034)
        self.assertEqual(reactive_onset(s['timestamps'], s['keypoints'], s['confidences']), [])

    def test_whole_body_step_in_cancels_glove_growth(self):
        from pose_features import MotionOnset
        s = self.session()
        stream, detector = FeatureStream(), MotionOnset()
        areas = []
        for i, t in enumerate(s['timestamps']):
            scale = 1+i*.015
            xy = s['keypoints'][i]*scale
            stream.update(t, xy, np.ones(17), [[*xy[9], 400*scale**2]])
            areas.append(stream.history[-1].glove_area[0])
            self.assertIsNone(detector.update(stream.history[-1], i))
        np.testing.assert_allclose(areas, areas[0])

    def test_jump_unknown_and_motion_reset(self):
        from pose_features import CandidateConfirm, MotionOnset
        s = self.session()
        xy = s['keypoints'][0]
        stream, detector, machine = FeatureStream(), MotionOnset(), CandidateConfirm()
        stream.update(0, xy, np.ones(17))
        machine.update(0, {'JAB': .9})
        stream.update(.033, xy+np.array([100, 0]), np.ones(17))
        f = stream.history[-1]
        self.assertTrue(f.unknown)
        self.assertEqual(len(stream.history), 1)
        self.assertIsNone(detector.update(f, 1))
        machine.update(.033, {'JAB': .9}, unknown=f.unknown)
        self.assertEqual(machine.state, 'UNKNOWN')
        self.assertIsNone(machine.candidate)
        stream.update(.066, xy+np.array([100, 0]), np.ones(17))
        self.assertFalse(stream.history[-1].unknown)
        self.assertTrue(np.isfinite(frame_features(stream.history)).all())
        self.assertIsNone(detector.update(stream.history[-1], 2))

    def test_shoulder_width_jump_and_missing_glove_flags(self):
        s = self.session()
        xy = s['keypoints'][0]
        stream = FeatureStream()
        stream.update(0, xy, np.ones(17), [[*xy[9], 400]])
        f = stream.update(.033, xy, np.ones(17))
        self.assertFalse(stream.history[-1].glove_mask.any())
        self.assertGreater(stream.history[-1].glove_area[0], 0)
        stream.update(.066, xy*1.5, np.ones(17))
        self.assertTrue(stream.history[-1].unknown)

    def test_scripted_negatives_are_false_even_near_onset(self):
        from train_anticipator import family_metrics
        s = self.session(60)
        s.update(onset_frames=np.array([30]), onset_sides=np.array(['CENTER']),
                 onset_families=np.array(['JAB']), commands=np.array(['FEINT', 'JAB', 'IDLE']),
                 command_times=np.array([0., 1., 1.5]))
        result = family_metrics([dict(time=.95, side='JAB'), dict(time=1.05, side='JAB'),
                                dict(time=1.8, side='JAB')], s)
        self.assertEqual(result['false_alarms'], 2)
        self.assertEqual(result['families']['JAB']['on_time_correct'], 1)
        self.assertAlmostEqual(result['negative_seconds'], 1.5)

    def test_leave_one_performer_out(self):
        from train_anticipator import performer_splits
        sessions = [dict(performer=p) for p in ('a', 'a', 'b', 'b', 'c')]
        folds = performer_splits(sessions)
        self.assertEqual(len(folds), 3)
        for train, test in folds:
            self.assertFalse({sessions[i]['performer'] for i in train} & {sessions[i]['performer'] for i in test})
            self.assertEqual(len(train)+len(test), 5)

    def test_human_departure_preferred_and_ranges_preserved(self):
        from label_onsets import save_labels
        s = self.session()
        with workspace_temp() as root:
            path = root/'session.npz'
            save_session(path, s['timestamps'], s['keypoints'], s['confidences'],
                         [dict(frame=5, side='CENTER', source='auto')], [], 'fake', 'fake')
            save_labels(path.with_suffix('.labels.json'), 'a', [dict(preparation=2, departure=10, family='JAB')])
            save_labels(path.with_suffix('.labels.json'), 'b', [dict(preparation=3, departure=12, family='JAB')])
            events = onsets(load_session(path))
            self.assertEqual(events[0]['family'], 'JAB')
            self.assertEqual(events[0]['departure_range'], (10/30, 12/30))
            self.assertAlmostEqual(events[0]['time'], 11/30)

    def test_candidate_confirm_silent_and_family_cue(self):
        from pose_features import CandidateConfirm, FAMILY_CUES
        machine = CandidateConfirm(mode='confirm')
        self.assertIsNone(machine.update(0, {'UPPERCUT': .8}))
        self.assertEqual(machine.state, 'CANDIDATE')
        self.assertIsNone(machine.update(.03, {'UPPERCUT': .8}))
        cue = machine.update(.06, {'UPPERCUT': .8}, dict(detected_time=.06))
        self.assertEqual(cue['family'], 'UPPERCUT')
        self.assertEqual(FAMILY_CUES[cue['family']], 'BACK')
        self.assertEqual(machine.state, 'CONFIRMED')
        self.assertIsNone(machine.update(.09, {'UPPERCUT': .8}, dict(detected_time=.09)))

    def test_prediction_detection_delays_separate(self):
        from train_anticipator import family_metrics
        s = self.session(90)
        s.update(onset_frames=np.array([30, 60]), onset_sides=np.array(['CENTER', 'CENTER']),
                 onset_families=np.array(['JAB', 'CROSS']), commands=np.array(['JAB']), command_times=np.array([0.]))
        m = family_metrics([dict(time=.9, side='JAB'), dict(time=2.05, side='CROSS')], s)
        self.assertAlmostEqual(m['families']['JAB']['predicted_delays_ms'][0], -100)
        self.assertAlmostEqual(m['families']['CROSS']['detection_delays_ms'][0], 50)
        self.assertFalse(m['families']['CROSS']['predicted_delays_ms'])

    def test_experiment_measurements(self):
        from first_experiment import measurements
        s = self.session()
        s['glove_blobs'] = np.full((30, 2, 3), np.nan)
        speeds, areas, events = measurements(s)
        self.assertEqual(speeds.shape, (30, 2))
        self.assertEqual(areas.shape, (30, 2))
        self.assertEqual(events, [])


class AddendumIntegrationTests(unittest.TestCase):
    def test_family_training_lopo_bundle_and_reporting(self):
        import train_anticipator as train
        from sklearn.dummy import DummyClassifier
        with workspace_temp() as root:
            for index, performer in enumerate(('one', 'two')):
                s = synthetic_session(index+1, punches=16)
                labels = [dict(frame=int(i), side=str(side), source='auto', family='JAB')
                          for i, side in zip(s['onset_frames'], s['onset_sides'])]
                save_session(root/f'{performer}.npz', s['timestamps'], s['keypoints'],
                             s['confidences'], labels, [], 'synthetic', 'synthetic',
                             performer=performer, commands=['JAB'], command_times=[0.])
            fitted = []
            def fake_fit(x, y):
                fitted.append(len(y))
                return {'hist': DummyClassifier(strategy='prior').fit(x, y)}
            output = root/'model.joblib'
            console = io.StringIO()
            with patch.object(train, 'fit_candidates', side_effect=fake_fit), \
                 patch.object(sys, 'argv', ['train_anticipator.py', '--data', str(root), '--output', str(output)]), \
                 contextlib.redirect_stdout(console):
                train.main()
            bundle = joblib.load(output)
            self.assertEqual(len(fitted), 2)
            self.assertFalse(set(bundle['train_sessions']) & set(bundle['test_sessions']))
            self.assertEqual(bundle['threshold'], .5)
            self.assertIn('UPPERCUT', bundle['family_cues'])
            for expected in ('Held out:', 'training-only selection', '(a) wrist only',
                             '(b) + relative glove area', 'candidate predict', 'candidate confirm',
                             'too few trials', 'negative footage'):
                self.assertIn(expected, console.getvalue())

    def test_glove_adapter_squares_blob_size(self):
        from glove_features import GloveSegmenter
        frame = np.zeros((200, 200, 3), np.uint8)
        cv2.rectangle(frame, (50, 50), (90, 90), (0, 0, 255), -1)
        blobs = GloveSegmenter('red')(frame)
        self.assertAlmostEqual(blobs[0, 2], 1600, delta=10)
        np.testing.assert_allclose(blobs[0, :2], [70, 70])

    def test_first_experiment_writes_png(self):
        import first_experiment
        with workspace_temp() as root:
            s = synthetic_session(1, punches=1)
            source = root/'straight.npz'
            np.savez(source, **s)
            with patch.object(sys, 'argv', ['first_experiment.py', str(source), '--output-dir', str(root)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                first_experiment.main()
            image = cv2.imread(str(root/'straight.png'))
            self.assertIsNotNone(image)
            self.assertGreater(image.shape[0], 400)


    def test_live_confirm_family_back_and_timing_log(self):
        import anticipate_live as live
        cap = Mock()
        cap.get.return_value = 30.
        cap.read.side_effect = [(True, np.zeros((240, 400, 3), np.uint8))]*3+[(False, None)]
        xy = synthetic_session(1, punches=1)['keypoints'][0]
        predictor = Mock(classes_=np.array(['NONE', 'UPPERCUT']))
        predictor.predict_proba.return_value = np.array([[.1, .9]])
        detector = Mock()
        detector.update.side_effect = [None, None, dict(frame=1, time=1/30, side='CENTER', detected_time=2/30)]
        beeper = Mock()
        bundle = dict(model=predictor, window=8, horizon=.3, feature_names=feature_names(), threshold=.7)
        original = SimpleNamespace(CUES={'DUCK': (900, 300, (0, 0, 255)),
            'DODGE LEFT': (600, 120, (0, 0, 0)), 'DODGE RIGHT': (1200, 120, (0, 0, 0))}, beep=beeper)
        def immediate_thread(target, **kwargs):
            return SimpleNamespace(start=target)
        with workspace_temp() as root, contextlib.chdir(root), \
             patch.object(sys, 'argv', ['anticipate_live.py', '--source', 'fake.mp4', '--mode', 'confirm', '--detector', 'reactive']), \
             patch.dict(sys.modules, {'punch_telegraph': original}), \
             patch.object(live.joblib, 'load', return_value=bundle), \
             patch.object(live, 'PoseRunner', return_value=lambda frame: (xy, np.ones(17))), \
             patch.object(live, 'open_source', return_value=(cap, False)), \
             patch.object(live, 'ReactiveOnset', return_value=detector), \
             patch.object(live.threading, 'Thread', side_effect=immediate_thread), \
             patch.object(live.time, 'sleep'), \
             patch.object(live.cv2, 'imshow'), patch.object(live.cv2, 'destroyAllWindows'), \
             patch.object(live.cv2, 'waitKey', return_value=255), \
             contextlib.redirect_stdout(io.StringIO()):
            live.main()
            with Path('anticipate_log.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            cues = [row for row in rows if row['event'] == 'early']
            self.assertEqual(len(cues), 1)
            self.assertEqual(cues[0]['side'], 'UPPERCUT')
            self.assertIn('EARLY DETECT', cues[0]['timing_status'])
            self.assertEqual(beeper.call_count, 2)
            self.assertTrue(all(call.args == (1500, 80) for call in beeper.call_args_list))


class LaunchDetectorTests(unittest.TestCase):
    def setUp(self):
        from launch_detector import LaunchDetector
        self.detector = LaunchDetector()
        self.xy = np.zeros((17, 2)) + [320., 200.]
        self.xy[5], self.xy[6] = (270., 200.), (370., 200.)
        self.xy[9], self.xy[10] = (240., 165.), (400., 165.)
        self.conf = np.ones(17)
        self.t = 0.

    def step(self, xy=None, conf=None, count=1):
        events = []
        for _ in range(count):
            e = self.detector.update(self.t, self.xy if xy is None else xy,
                                     self.conf if conf is None else conf)
            self.t += 1/30
            if e:
                events.append(e)
        return events

    def test_held_guard_launch_one_event_and_sides(self):
        from launch_detector import LaunchDetector
        for wrist, delta, side in ((9, (-70, 0), 'LEFT'),
                                    (10, (70, 0), 'RIGHT'),
                                    (9, (0, -70), 'CENTER')):
            with self.subTest(side=side):
                self.detector, self.t = LaunchDetector(), 0.
                self.assertEqual(self.step(count=20), [])
                moved = self.xy.copy()
                moved[wrist] += delta
                events = self.step(moved, count=10)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]['side'], side)
                self.assertEqual(events[0]['wrist'], wrist-9)

    def test_retraction_after_launch_does_not_fire(self):
        self.detector.cooldown = 0.  # Retraction suppression must not rely on cooldown.
        self.step(count=20)
        moved = self.xy.copy()
        moved[9, 0] -= 70
        self.assertEqual(len(self.step(moved)), 1)
        # Extended fist never settles into a new held guard before returning.
        for fraction in (.8, .5, .2, 0.):
            returning = self.xy + (moved-self.xy)*fraction
            self.assertEqual(self.step(returning), [])
        self.assertEqual(self.step(count=10), [])

    def test_whole_body_translation_and_scale_no_event(self):
        self.step(count=20)
        for i in range(100):
            moved = self.xy*(1.+i*.003)+np.array([i*2., -i*.7])
            self.assertEqual(self.step(moved), [])

    def test_lost_wrist_does_not_blind_other(self):
        conf = self.conf.copy()
        conf[9] = 0
        xy = self.xy.copy()
        xy[9] = np.nan
        self.step(xy, conf, count=20)
        xy[10, 0] += 70
        events = self.step(xy, conf)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['side'], 'RIGHT')

    def test_cooldown_across_wrists_then_rearm(self):
        self.step(count=20)
        first = self.xy.copy()
        first[9, 0] -= 70
        events = self.step(first)
        self.assertEqual(len(events), 1)
        second = first.copy()
        second[10, 0] += 70
        self.assertEqual(self.step(second), [])
        self.step(count=35)
        after = self.step(second)
        self.assertEqual(len(after), 1)
        self.assertGreaterEqual(after[0]['detected_time']-events[0]['detected_time'], .6)


class LiveRuntimeTests(unittest.TestCase):
    def test_latest_capture_skips_stale_and_keeps_timestamp(self):
        from live_runtime import LatestCapture
        cap = Mock()
        cap.read.side_effect = [(True, np.full((2, 2, 3), i, np.uint8)) for i in range(20)]+[(False, None)]
        worker = LatestCapture(cap)
        try:
            worker.thread.join(timeout=2)
            self.assertFalse(worker.thread.is_alive())
            frame, stamp = worker.read()
            self.assertTrue((frame == 19).all())
            self.assertIsInstance(stamp, float)
            self.assertGreater(stamp, 0.)
            self.assertEqual(worker.read(), (None, None))
        finally:
            worker.close()
        cap.release.assert_called_once()

    def test_writer_snapshots_raw_frames_and_drains_bounded_queue(self):
        from live_runtime import VideoSink
        import threading
        entered, proceed = threading.Event(), threading.Event()
        encoded = []
        backend = Mock()
        def encode(frame):
            entered.set()
            if not proceed.wait(timeout=2):
                raise RuntimeError('test stalled')
            encoded.append(frame.copy())
        backend.write.side_effect = encode
        with patch('live_runtime.cv2.VideoWriter', return_value=backend):
            sink = VideoSink('unused.mp4', 30., (4, 4), queue_size=1)
            frame = np.zeros((4, 4, 3), np.uint8)
            try:
                sink.write(frame)
                self.assertTrue(entered.wait(timeout=2))
                frame[:] = 100  # Simulate an overlay racing the encoder.
                sink.write(frame)
                frame[:] = 200
                self.assertEqual(sink.queue.maxsize, 1)
            finally:
                proceed.set()
                sink.close()
        self.assertEqual(sink.written, 2)
        self.assertTrue((encoded[0] == 0).all())
        self.assertTrue((encoded[1] == 100).all())
        backend.release.assert_called_once()

    def test_writer_failure_propagates_on_close(self):
        from live_runtime import VideoSink
        backend = Mock()
        backend.write.side_effect = OSError('disk failure')
        with patch('live_runtime.cv2.VideoWriter', return_value=backend):
            sink = VideoSink('unused.mp4', 30., (4, 4))
            try:
                sink.write(np.zeros((4, 4, 3), np.uint8))
            except RuntimeError:
                pass  # The worker may fail before write() returns.
            with self.assertRaisesRegex(RuntimeError, 'Video writer failed'):
                sink.close()
            self.assertFalse(sink.thread.is_alive())
        backend.release.assert_called_once()

    def test_headless_benchmark_does_not_touch_data_or_windows(self):
        import collect
        cap = Mock()
        cap.get.return_value = 30.
        cap.read.side_effect = [(True, np.zeros((120, 160, 3), np.uint8)) for _ in range(3)]
        with workspace_temp() as root, contextlib.chdir(root), \
             patch.object(sys, 'argv', ['collect.py', '--benchmark', 'fake.mp4', '--benchmark-frames', '3']), \
             patch.object(collect, 'open_source', return_value=(cap, False)), \
             patch.object(collect, 'PoseRunner', return_value=lambda frame: None), \
             patch.object(collect.cv2, 'imshow', side_effect=AssertionError('window opened')), \
             patch.object(collect.cv2, 'waitKey', side_effect=AssertionError('GUI wait')), \
             patch.object(collect.cv2, 'destroyAllWindows', side_effect=AssertionError('GUI cleanup')), \
             contextlib.redirect_stdout(io.StringIO()) as console:
            collect.main()
            self.assertFalse(Path('data').exists())
            self.assertEqual(list(root.iterdir()), [])
            self.assertIn('3 processed frames', console.getvalue())
            self.assertIn('achievable fps', console.getvalue())

    def test_default_live_launch_uses_raw_pose_and_side_cue(self):
        import anticipate_live as live
        xy = synthetic_session(1, punches=1)['keypoints'][0]
        conf = np.ones(17)
        # Use family classes with weak predictions: backup must still map LEFT
        # to DODGE RIGHT, rather than taking the classifier's family direction.
        predictor = Mock(classes_=np.array(['NONE', 'UPPERCUT']))
        predictor.predict_proba.return_value = np.array([[.99, .01]])
        bundle = dict(model=predictor, window=8, horizon=.3, feature_names=feature_names())
        detector = Mock(last_disp=[.7, 0.])
        detector.update.return_value = dict(time=0., detected_time=0., side='LEFT')
        cap = Mock()
        cap.get.return_value = 30.
        cap.read.side_effect = [(True, np.zeros((240, 400, 3), np.uint8)), (False, None)]
        beep = Mock()
        original = SimpleNamespace(CUES={'DUCK': (900, 300, (0, 0, 255)),
            'DODGE LEFT': (600, 120, (0, 0, 0)), 'DODGE RIGHT': (1200, 120, (0, 0, 0))}, beep=beep)
        with workspace_temp() as root, contextlib.chdir(root), \
             patch.object(sys, 'argv', ['anticipate_live.py', '--source', 'fake.mp4']), \
             patch.dict(sys.modules, {'punch_telegraph': original}), \
             patch.object(live.joblib, 'load', return_value=bundle), \
             patch.object(live, 'PoseRunner', return_value=lambda frame: (xy, conf)), \
             patch.object(live, 'LaunchDetector', return_value=detector) as launch, \
             patch.object(live, 'ReactiveOnset', side_effect=AssertionError('reactive default')), \
             patch.object(live, 'GloveOnset') as glove, \
             patch.object(live, 'open_source', return_value=(cap, False)), \
             patch.object(live.cv2, 'imshow'), patch.object(live.cv2, 'destroyAllWindows'), \
             patch.object(live.cv2, 'waitKey', return_value=255), \
             contextlib.redirect_stdout(io.StringIO()):
            live.main()
            launch.assert_called_once_with(thresh=.55)
            glove.return_value.update.assert_not_called()
            detector.update.assert_called_once()
            args = detector.update.call_args.args
            self.assertEqual(args[0], 0.)
            self.assertIs(args[1], xy)
            self.assertIs(args[2], conf)
            beep.assert_called_once_with(1200, 120)
            with Path('anticipate_log.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertIn('launch_backup', [r['event'] for r in rows])


if __name__ == '__main__':
    unittest.main()
