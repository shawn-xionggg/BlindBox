"""Live anticipation plus launch-from-guard backup. python anticipate_live.py --source 0.

Offline validation excludes compute/audio latency. Live warning times are measured
after inference. Live metrics use detector auto-labels, not independent ground truth;
feints and missed onsets require recorded, manually reviewed evaluation.
"""
import argparse
import csv
import json
import threading
from pathlib import Path
import time
import joblib
import cv2
import numpy as np
from pose_features import PoseRunner, FeatureStream, ReactiveOnset, feature_names
from collect import open_source, skeleton, text, video_timestamp
from pose_features import CandidateConfirm, GloveOnset, FAMILY_CUES, SAFETY, timing_label
from glove_features import GloveSegmenter, PRESETS
from launch_detector import LaunchDetector
from live_runtime import LatestCapture, StageTimings
from train_anticipator import event_metrics, print_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='0')
    ap.add_argument('--detector', choices=('launch', 'reactive'), default='launch')
    ap.add_argument('--launch-thresh', type=float, default=.55)
    ap.add_argument('--model', default='yolo11n-pose.pt', help='pose weights')
    ap.add_argument('--anticipator', default='models/anticipator.joblib')
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--thresh', type=float, default=None, help='Override training-selected probability threshold')
    ap.add_argument('--color', choices=PRESETS)
    ap.add_argument('--mode', choices=('predict', 'confirm'), default='predict')
    ap.add_argument('--low-thresh', type=float, default=.3)
    ap.add_argument('--glove-thresh', type=float, default=3.)
    ap.add_argument('--cue-map', help='JSON file overriding the family-to-game-command dictionary')
    args = ap.parse_args()
    if args.thresh is not None and not 0 <= args.thresh <= 1:
        ap.error('--thresh must be between 0 and 1')
    if args.launch_thresh <= 0:
        ap.error('--launch-thresh must be positive')
    bundle = joblib.load(args.anticipator)
    args.thresh = bundle.get('threshold', .5) if args.thresh is None else args.thresh
    if bundle['feature_names'] != feature_names():
        raise ValueError('Feature schema changed; retrain the anticipator')
    # Reuse the original cue definitions and non-blocking winsound implementation.
    from punch_telegraph import CUES as ORIGINAL_CUES, beep
    CUES = dict(ORIGINAL_CUES, BACK=(1500, 80, (200, 100, 255)))
    cue_names = {'LEFT': 'DODGE RIGHT', 'RIGHT': 'DODGE LEFT', 'CENTER': 'DUCK'}
    cue_names.update(bundle.get('family_cues', FAMILY_CUES))
    if args.cue_map:
        cue_names.update(json.loads(Path(args.cue_map).read_text()))
    if any(c not in CUES for c in cue_names.values()):
        ap.error('Unknown cue in --cue-map')
    cv2.setNumThreads(1)
    runner = PoseRunner(args.model, args.imgsz)
    model = bundle['model']
    stream = FeatureStream(bundle['window'])
    detector = LaunchDetector(thresh=args.launch_thresh) if args.detector == 'launch' else ReactiveOnset()
    machine = CandidateConfirm(args.thresh, args.low_thresh, args.mode)
    glove_detector, segmenter = GloveOnset(args.glove_thresh), GloveSegmenter(args.color)
    cap, live = open_source(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.
    warnings, punches, matched = [], [], set()
    last_early, last_cue, shown_until, last_lead = -np.inf, '', -np.inf, 'No matched punch yet'
    start, index, t = time.perf_counter(), 0, 0.
    previous_capture = None
    last_motion_detection = -np.inf
    path = Path('anticipate_log.csv')
    log_file = path.open('a', newline='')
    log = csv.writer(log_file)
    if log_file.tell() == 0:
        log.writerow(['run', 'event', 'time_s', 'side', 'onset_time_s', 'lead_ms', 'probability', 'timing_status', 'detection_time_s', 'classification_time_s'])
    run = time.strftime('%Y%m%d_%H%M%S')
    capture = LatestCapture(cap) if live else None
    timings = StageTimings()
    print(SAFETY)
    print('Live stats use detector auto-labels; verify with manually reviewed recordings.')
    try:
        while True:
            loop_start = time.perf_counter()
            accounted = sum(timings.totals.values())
            with timings.stage('capture wait'):
                if capture is not None:
                    frame, captured_at = capture.read()
                    ok = frame is not None
                else:
                    ok, frame = cap.read()
                    captured_at = time.perf_counter()
                if not ok:
                    break
                capture_t = captured_at-start if live else video_timestamp(cap, index, fps, previous_capture)
            previous_capture = capture_t
            with timings.stage('pose'):
                person = runner(frame)
            xy, conf = person if person is not None else (None, None)
            with timings.stage('glove segmentation'):
                blobs = segmenter(frame)
            features = stream.update(capture_t, xy, conf, blobs)
            f = stream.history[-1]
            valid = f.mask[5] and f.mask[6] and (f.mask[9] or f.mask[10])
            probs = dict(zip(model.classes_, model.predict_proba(features[None])[0])) if valid else {}
            t = time.perf_counter()-start if live else capture_t
            wrist_onset = detector.update(capture_t, xy, conf) if args.detector == 'launch' else detector.update(f, index)
            glove_onset = glove_detector.update(f, index) if args.detector == 'reactive' else None
            onset = min([e for e in (wrist_onset, glove_onset) if e], key=lambda e: e['time'], default=None)
            if onset and capture_t-last_motion_detection < .6:
                onset = None
            if onset:
                last_motion_detection = capture_t
            early = machine.update(t, probs, onset, f.unknown)
            if machine.last_transition and any(c in FAMILY_CUES for c in model.classes_):
                log.writerow([run, machine.last_transition.lower(), t, '', '', '', ''])
            if f.unknown:
                log.writerow([run, 'UNKNOWN', t, '', '', '', ''])
            cue = None
            if early:
                warnings.append(early)
                last_early = t
                cue = cue_names[early['side']]
                log.writerow([run, 'early', t, early['side'], '', '', probs[early['side']],
                              timing_label((t-onset['time'])*1000) if onset else 'UNVERIFIED: departure not labeled',
                              onset['detected_time'] if onset else '', t])
            if onset:
                punches.append(dict(time=onset['time'], side=onset['side']))
                candidates = [j for j, w in enumerate(warnings) if j not in matched
                              and 0 < onset['time']-w['time'] <= bundle['horizon']+1e-10]
                lead = ''
                if candidates:
                    j = candidates[0]
                    matched.add(j)
                    lead = (onset['time']-warnings[j]['time'])*1000
                    last_lead = timing_label(-lead)
                else:
                    last_lead = timing_label((t-onset['time'])*1000) if early else 'No matched cue; auto departure only'
                print(last_lead)
                log.writerow([run, 'onset', t, onset['side'], onset['time'], lead, '', last_lead, onset['detected_time'], t])
                if args.mode == 'predict' and t-last_early > .5:
                    # A motion signal alone cannot establish a gesture family.
                    family = max((c for c in probs if c != 'NONE'), key=probs.get, default=None)
                    cue = cue_names.get(family) if family and probs[family] > args.low_thresh else None
                    if args.detector == 'launch' or all(c not in FAMILY_CUES for c in model.classes_):
                        cue = cue_names[onset['side']]
                    last_lead = timing_label((t-onset['time'])*1000)
                    log.writerow([run, args.detector+'_backup', t, onset['side'], onset['time'], '', '', last_lead, onset['detected_time'], t])
                print_metrics('live auto-labels', event_metrics(warnings, punches, 0., t, bundle['horizon']))
            if cue:
                freq, ms, _ = CUES[cue]
                if cue == 'BACK':
                    def back_beep():
                        beep(1500, 80)
                        time.sleep(.15)
                        beep(1500, 80)
                    threading.Thread(target=back_beep, daemon=True).start()
                else:
                    beep(freq, ms)
                last_cue, shown_until = cue, t+.6
            with timings.stage('overlay'):
                skeleton(frame, xy, conf)
                for row, side in enumerate(model.classes_):
                    p = probs.get(side, 0.)
                    y = 25+row*30
                    text(frame, f'{side:6s} {p:.2f}', y)
                    cv2.rectangle(frame, (135, y-12), (335, y), (70, 70, 70), 1)
                    cv2.rectangle(frame, (135, y-12), (135+int(200*p), y), (0, 200, 200), -1)
                text(frame, last_lead, 160)
                text(frame, SAFETY, frame.shape[0]-30)
                text(frame, machine.state+' | '+args.mode+' | q/Esc quit', frame.shape[0]-10)
                if t < shown_until:
                    text(frame, last_cue, frame.shape[0]//2, CUES[last_cue][2], 1.)
                if args.detector == 'launch':
                    # Displacement in shoulder widths, capped at 2x threshold.
                    bar_max = max(1., args.launch_thresh*2)
                    for wrist, displacement in enumerate(detector.last_disp):
                        y = 190 + wrist*22
                        text(frame, f'W{wrist+1} {displacement:.2f}', y)
                        cv2.rectangle(frame, (100, y-10), (220, y), (70, 70, 70), 1)
                        end = 100+int(120*min(max(displacement, 0.), bar_max)/bar_max)
                        cv2.rectangle(frame, (100, y-10), (end, y), (0, 200, 200), -1)
                        threshold_x = 100+int(120*args.launch_thresh/bar_max)
                        cv2.line(frame, (threshold_x, y-12), (threshold_x, y+2), (0, 0, 255), 1)
            with timings.stage('imshow'):
                cv2.imshow('Anticipation', frame)
                segmenter.bind('Anticipation')
                key = cv2.waitKey(1) & 255
            log_file.flush()
            index += 1
            timings.frames += 1
            timings.totals['other'] += max(0., time.perf_counter()-loop_start-(sum(timings.totals.values())-accounted))
            if key in (27, ord('q')):
                break
    finally:
        if capture is not None:
            capture.close()
        else:
            cap.release()
        cv2.destroyAllWindows()
        timings.report()
        log_file.close()
        print_metrics('final auto-labels', event_metrics(warnings, punches, 0., t, bundle['horizon']))
        print('Warnings in the final prediction horizon may be unresolved; review anticipate_log.csv.')


if __name__ == '__main__':
    main()
