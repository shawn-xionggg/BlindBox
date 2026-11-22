"""Collect pose, editable onset labels, capture timestamps, and unannotated video.

q/Esc: save and quit; x: remove last auto label; p: manual onset;
l/r/c: choose image LEFT/RIGHT/CENTER for subsequent manual marks.
--from-video re-poses a video, preserving paired timestamps and manual edits.
"""
import argparse
from datetime import datetime
from pathlib import Path
import time
import os
import random
import shutil
from uuid import uuid4
import cv2
import numpy as np
from pose_features import PoseRunner, FeatureStream, GloveOnset, COMMANDS, NEGATIVES, SAFETY
from glove_features import GloveSegmenter, PRESETS
from launch_detector import LaunchDetector
from live_runtime import LatestCapture, VideoSink, StageTimings
import free_blocks

INSTRUCTIONS = 'Puncher: mix punches with idle guard movement, bouncing, feints. Aim for 100+ punches across sessions.'
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12))


def open_source(source):
    source = str(source)
    live = source.isdigit()
    cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY) if live else cv2.VideoCapture(source)
    if live:
        for prop, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
                            (cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                            (cv2.CAP_PROP_FPS, 30), (cv2.CAP_PROP_BUFFERSIZE, 1)):
            try:
                cap.set(prop, value)
            except cv2.error:
                pass
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Could not open {source}')
    return cap, live


def video_timestamp(cap, index, fps, previous=None):
    value = cap.get(cv2.CAP_PROP_POS_MSEC)
    # Some backends do not expose PTS; only then use the nominal frame clock.
    position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if (isinstance(value, (int, float, np.floating)) and np.isfinite(value)
            and isinstance(position, (int, float, np.floating)) and abs(position-index-1) < .5):
        value = float(value)/1000
        if (previous is None and value >= 0) or (previous is not None and value > previous):
            return value
    return max(index/fps, previous+1/fps if previous is not None else 0.)


def skeleton(frame, xy, conf):
    if xy is None:
        return
    for a, b in EDGES:
        if min(conf[a], conf[b]) >= .35 and np.isfinite(xy[[a, b]]).all():
            cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (0, 230, 0), 2)


def text(frame, message, y, color=(255, 255, 255), scale=.5):
    free_blocks.outlined(frame, message, (10, y), scale, color, 1)   # black outline: readable anywhere


def save_session(path, timestamps, keypoints, confidences, labels, rejected, model, source, session_id=None, glove_blobs=None, commands=(), command_times=(), performer="unknown", mount="desk", wearer="",
                 blocks=(), block_times=(), stance="unknown"):
    labels = sorted(labels, key=lambda e: e['frame'])
    np.savez_compressed(path, timestamps=np.asarray(timestamps),
        keypoints=np.asarray(keypoints).reshape(-1, 17, 2),
        confidences=np.asarray(confidences).reshape(-1, 17),
        onset_frames=np.asarray([e['frame'] for e in labels], dtype=int),
        onset_sides=np.asarray([e['side'] for e in labels], dtype='<U6'),
        onset_sources=np.asarray([e['source'] for e in labels], dtype='<U6'),
        detected_times=np.asarray([e.get('detected_time', timestamps[e['frame']]) for e in labels]),
        rejected_frames=np.asarray(rejected, dtype=int), pose_model=str(model), source=str(source),
        session_id=str(session_id or Path(path).stem), performer=str(performer),
        mount=str(mount), wearer=str(wearer), stance=str(stance),
        # --free sessions: BASELINE / PROMPTS / FREE blocks and their start times (free_blocks.py)
        blocks=np.asarray(blocks, dtype='<U16'), block_times=np.asarray(block_times, dtype=float),
        glove_blobs=np.asarray(glove_blobs if glove_blobs is not None else np.full((len(timestamps), 2, 3), np.nan)),
        commands=np.asarray(commands, dtype='<U16'), command_times=np.asarray(command_times, dtype=float),
        onset_families=np.asarray([e.get('family', 'UNKNOWN') for e in labels], dtype='<U16'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='0')
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--from-video')
    modes.add_argument('--benchmark', metavar='VIDEO', help='Headless processing and temporary recording; never writes data/')
    ap.add_argument('--benchmark-frames', type=int, default=600)
    ap.add_argument('--opencv-threads', type=int, default=1, help='OpenCV worker count; 1 avoids overhead on small frames')
    ap.add_argument('--sync-io', action='store_true', help='Synchronous capture/writer baseline for comparison')
    ap.add_argument('--launch-thresh', type=float, default=.55)
    ap.add_argument('--glove-onsets', action='store_true', help='Enable experimental glove auto-labels')
    ap.add_argument('--model', default='yolo11n-pose.pt')
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--manual-side', choices=('LEFT', 'RIGHT', 'CENTER'), default='CENTER')
    ap.add_argument('--color', choices=PRESETS)
    ap.add_argument('--script', action='store_true', help='Visual-only randomized commands every 2-4 s')
    ap.add_argument('--performer', default='unknown')
    ap.add_argument('--mount', choices=('head', 'desk'),
                    help="where the camera is: 'head' = on the wearer's hat (the real use), 'desk' = fixed. "
                         "Required for live recording; saved in the session so training can tell them apart")
    ap.add_argument('--wearer', default='', help="who wore the camera (head mount)")
    ap.add_argument('--negatives-only', action='store_true',
                    help="with --script: only don't-punch prompts (for wearer head-motion footage)")
    ap.add_argument('--free', action='store_true',
                    help="with --script: a calm BASELINE first, then the prompts alternate with self-paced "
                         "FREE BOXING blocks (natural setups and combos, for punch prediction; see free_blocks.py)")
    ap.add_argument('--baseline-seconds', type=float, default=20., help="with --free: baseline length (0 = none)")
    ap.add_argument('--prompt-seconds', type=float, default=60., help="with --free: length of each block of prompts")
    ap.add_argument('--free-seconds', type=float, default=60., help="with --free: length of each free-boxing block")
    ap.add_argument('--stance', choices=('orthodox', 'southpaw', 'unknown'), default='unknown',
                    help="the puncher's stance: orthodox = left hand leads, southpaw = right hand leads")
    ap.add_argument('--glove-thresh', type=float, default=3., help='Relative-area log growth per second')
    ap.add_argument('--check-light', action='store_true',
                    help="only show the camera and its live frame rate; NOTHING is recorded. Move lamps until it "
                         "reads ~30 fps (the C270 slows down in dim light), press q, then record")
    ap.add_argument('--no-beep', action='store_true',
                    help="don't beep when the puncher is out of frame (the red screen warning stays)")
    args = ap.parse_args()
    if args.check_light:
        args.script = args.free = args.negatives_only = False
    if args.benchmark_frames <= 0 or args.launch_thresh <= 0:
        ap.error("Frame limit and launch threshold must be positive")
    if args.opencv_threads < 1:
        ap.error("--opencv-threads must be positive")
    if not (args.benchmark or args.from_video or args.check_light) and str(args.source).isdigit() and args.mount is None:
        ap.error("say where the camera is: --mount head (on the wearer's hat) or --mount desk (fixed)")
    if args.negatives_only and not args.script:
        ap.error("--negatives-only needs --script")
    if args.free and not args.script:
        ap.error("--free needs --script")
    if args.free and args.negatives_only:
        ap.error("--free and --negatives-only don't mix: free-boxing blocks are for punching")
    if args.free and (args.prompt_seconds <= 0 or args.free_seconds <= 0 or args.baseline_seconds < 0):
        ap.error("--prompt-seconds and --free-seconds must be positive, --baseline-seconds 0 or more")
    if args.free and not (args.benchmark or args.from_video) and str(args.source).isdigit() and args.performer == 'unknown':
        # leave-one-performer-out tests depend on this name: a missing one silently merges people
        ap.error("--free records people for punch prediction: say who is punching with --performer <name>")
    cv2.setNumThreads(args.opencv_threads)
    runner = PoseRunner(args.model, args.imgsz)
    source = args.benchmark or args.from_video or args.source
    if args.benchmark and str(source).isdigit():
        ap.error("--benchmark requires a video file")
    paired = Path(source).with_suffix('.npz') if not str(source).isdigit() else None
    original = None
    if args.from_video and paired.exists():
        with np.load(paired, allow_pickle=False) as z:
            original = {k: z[k] for k in z.files}
    cap, live = open_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.
    benchmark_dir = Path('bench_'+uuid4().hex[:8]) if args.benchmark else None
    output_dir = benchmark_dir if benchmark_dir else Path('data')
    output_dir.mkdir(exist_ok=True)
    stem = output_dir / ('session_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    times, points, confidences, labels, rejected = [], [], [], [], []
    if original is not None:
        rejected = original.get('rejected_frames', np.array([], dtype=int)).tolist()
        for i, side, kind in zip(original['onset_frames'], original['onset_sides'], original['onset_sources']):
            if kind == 'manual':
                labels.append(dict(frame=int(i), side=str(side), source='manual',
                                   family=str(original.get('onset_families', np.full(len(original['onset_frames']), 'UNKNOWN'))[list(original['onset_frames']).index(i)])))
    stream, detector, writer = FeatureStream(), LaunchDetector(thresh=args.launch_thresh), None
    segmenter, glove_detector = GloveSegmenter(args.color), GloveOnset(args.glove_thresh)
    raw_blobs, commands, command_times = [], [], []
    blocks, block_times = [], []
    if original is not None:
        commands = original.get('commands', np.array([])).tolist()
        command_times = original.get('command_times', np.array([])).tolist()
        blocks = original.get('blocks', np.array([])).tolist()
        block_times = original.get('block_times', np.array([])).tolist()
    # --free: free_blocks.FreeSchedule picks the block and the prompt (same prompt pool as --script)
    schedule = (free_blocks.FreeSchedule(COMMANDS + ('GUARD_SHIFT',), args.baseline_seconds, args.prompt_seconds,
                                         args.free_seconds) if args.free and original is None else None)
    next_command = 0.
    last_auto_detection = -np.inf
    def active_command(t):
        j = np.searchsorted(command_times, t, side='right')-1
        return commands[j] if j >= 0 else 'UNKNOWN'
    flash_until, start, manual_side = -1., time.perf_counter(), args.manual_side
    capture = LatestCapture(cap) if live and not args.sync_io else None
    timings = StageTimings()
    # live only: beeps while the puncher's shoulders are lost or at the image edge, so the
    # wearer/puncher fix the framing at once (untracked frames are useless for training)
    alarm = free_blocks.FramingAlarm(sound=not args.no_beep) if live and not args.benchmark else None
    print(SAFETY)
    print(INSTRUCTIONS)
    print(__doc__)
    if schedule is not None:
        print(free_blocks.plan_text(args.baseline_seconds, args.prompt_seconds, args.free_seconds,
                                    args.performer, args.stance))
    try:
        while True:
            loop_start = time.perf_counter()
            accounted = sum(timings.totals.values())
            i = len(times)
            if args.benchmark and i >= args.benchmark_frames:
                break
            with timings.stage('capture wait'):
                if capture is not None:
                    frame, captured_at = capture.read()
                    ok = frame is not None
                else:
                    ok, frame = cap.read()
                    captured_at = time.perf_counter()
                if not ok:
                    break
                t = captured_at-start if live else video_timestamp(cap, i, fps, times[-1] if times else None)
            if original is not None:
                if i >= len(original['timestamps']):
                    raise ValueError('Paired timestamp count is shorter than video')
                t = float(original['timestamps'][i])
            with timings.stage('pose'):
                person = runner(frame)
            xy, conf = person if person is not None else (np.full((17, 2), np.nan), np.zeros(17))
            with timings.stage('video write'):
                if not args.check_light:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = VideoSink(stem.with_suffix('.mp4'), fps, (w, h), threaded=not args.sync_io)
                    writer.write(frame)  # Snapshot raw frame before overlay, once per NPZ row.
            times.append(t)
            points.append(xy.copy())
            confidences.append(conf.copy())
            if schedule is not None:
                block, command = schedule.update(t)
                if block:
                    blocks.append(block)
                    block_times.append(t)
                if command:
                    commands.append(command)
                    command_times.append(t)
            elif args.script and original is None and t >= next_command:
                # GUARD_SHIFT twice as often: sliding the guard sideways is the
                # top false beep live and the older sessions never recorded it
                pool = NEGATIVES if args.negatives_only else COMMANDS + ('GUARD_SHIFT',)
                commands.append(random.choice(pool))
                command_times.append(t)
                next_command = t+random.uniform(2, 4)
            with timings.stage('glove segmentation'):
                blobs = segmenter(frame)
            if original is not None and segmenter.color.hue is None and 'glove_blobs' in original:
                blobs = original['glove_blobs'][i]
            raw_blobs.append(blobs)
            if args.glove_onsets:
                stream.update(t, xy, conf, blobs)
            wrist_event = detector.update(t, xy, conf)
            if wrist_event:
                wrist_event['frame'] = max(0, int(np.searchsorted(times, wrist_event['time'], side='right'))-1)
            glove_event = glove_detector.update(stream.history[-1], i) if args.glove_onsets else None
            event = min([e for e in (wrist_event, glove_event) if e], key=lambda e: e['time'], default=None)
            if event and t-last_auto_detection < .6:
                event = None
            if event:
                last_auto_detection = t
                event['family'] = active_command(event['time'])
            if event and event['frame'] not in rejected and not any(e['frame'] == event['frame'] for e in labels):
                labels.append(dict(event, source='auto'))
                flash_until = t+.4
            with timings.stage('overlay'):
                skeleton(frame, xy, conf)
                text(frame, f'Onsets: {len(labels)} | manual side {manual_side} | q save / x undo auto / p mark', 22)
                text(frame, SAFETY, 44)
                # Saved labels stay HOOK_L/HOOK_R (= fist comes from that side of the IMAGE);
                # the screen spells out which hand, since the camera faces the puncher.
                # --free: BASELINE / FREE blocks get their own static screen instead of a prompt
                block_screen = schedule is not None and free_blocks.draw(frame, schedule, t)
                if args.check_light:
                    free_blocks.draw_command(frame, 'LIGHT CHECK')
                elif args.script and not block_screen:
                    # big outlined command + what it means (HOOK_L/HOOK_R name the IMAGE side the fist
                    # comes from: the screen spells out the hand)
                    free_blocks.draw_command(frame, active_command(t))
                elif not block_screen:
                    text(frame, 'COMMAND: '+active_command(t), 140, (0, 255, 255), .9)
                text(frame, 'Aim for 100+ punches across sessions. l/r/c selects manual side.', 66)
                if t < flash_until and not block_screen:   # nothing flashes during free boxing
                    text(frame, 'ONSET', 105, (0, 0, 255), 1.)
                # live camera frame rate over the last ~1 s: tells you if the light is good enough
                k = min(len(times), 31)
                free_blocks.draw_fps(frame, (k - 1)/(times[-1]-times[-k]) if k > 5 and times[-1] > times[-k] else None)
                if alarm is not None:
                    problem = free_blocks.framing_problem(xy if person is not None else None, conf, frame.shape[1])
                    free_blocks.draw_alarm(frame, alarm.update(t, problem), t)
                if segmenter.color.hue is None:
                    if not block_screen and not args.script:
                        text(frame, 'Click glove to sample color (optional)', 170)
                elif segmenter.hsv is not None:
                    # live glove mask preview: ONLY the gloves should be white
                    gm = segmenter.color.mask(segmenter.hsv)
                    fw = frame.shape[1]
                    frame[0:120, fw-160:fw] = cv2.resize(cv2.cvtColor(gm, cv2.COLOR_GRAY2BGR), (160, 120))
                    cv2.rectangle(frame, (fw-160, 0), (fw-1, 119), (0, 255, 255), 1)
                    cv2.putText(frame, 'glove mask', (fw-155, 135), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
            with timings.stage('imshow'):
                if args.benchmark:
                    key = 255
                else:
                    cv2.imshow('Collect anticipation', frame)
                    segmenter.bind('Collect anticipation')
                    key = cv2.waitKey(1 if live else max(1, round(1000/fps))) & 255
            timings.frames += 1
            timings.totals['other'] += max(0., time.perf_counter()-loop_start-(sum(timings.totals.values())-accounted))
            if key in (27, ord('q')):
                break
            if key in (ord('l'), ord('r'), ord('c')):
                manual_side = {ord('l'): 'LEFT', ord('r'): 'RIGHT', ord('c'): 'CENTER'}[key]
            if key == ord('x'):
                auto = [j for j, e in enumerate(labels) if e['source'] == 'auto']
                if auto:
                    rejected.append(labels.pop(auto[-1])['frame'])
            if key == ord('p'):
                labels = [e for e in labels if e['frame'] != i]
                labels.append(dict(frame=i, side=manual_side, source='manual', detected_time=t, family=active_command(t)))
                flash_until = t+.4
    finally:
        if alarm is not None:
            alarm.close()
        if capture is not None:
            capture.close()
        else:
            cap.release()
        try:
            if writer is not None:
                writer.close()  # Drain first: never publish NPZ after a failed write.
                if writer.written != len(times):
                    raise RuntimeError('Video/NPZ frame count mismatch')
        finally:
            if not args.benchmark:
                cv2.destroyAllWindows()
            timings.report(writer)
            if benchmark_dir is not None:
                shutil.rmtree(benchmark_dir)  # Unique direct child of the workspace.
        if args.check_light and times:
            k = min(len(times), 91)
            print(f'Light check done (nothing saved): camera {(k - 1)/max(times[-1]-times[-k], 1e-9):.1f} fps over the last ~3 s')
        if times and not args.benchmark and not args.check_light:
            labels = [e for e in labels if e['frame'] < len(times)]
            identity = str(original.get('session_id', paired.stem)) if original is not None else None
            save_session(stem.with_suffix('.npz'), times, points, confidences, labels, rejected, args.model, source, identity, raw_blobs, commands, command_times,
                         str(original.get('performer', args.performer)) if original is not None else args.performer,
                         str(original['mount']) if original is not None and 'mount' in original else (args.mount or 'desk'),
                         str(original['wearer']) if original is not None and 'wearer' in original else args.wearer,
                         blocks, block_times,
                         str(original['stance']) if original is not None and 'stance' in original else args.stance)
            if original is not None and paired.with_suffix('.labels.json').exists():
                shutil.copyfile(paired.with_suffix('.labels.json'), stem.with_suffix('.labels.json'))
            print(f'Saved {stem}.npz and .mp4: {len(times)} frames, {len(labels)} onsets. NPZ timestamps are authoritative (live inference may lower capture fps).')
            if blocks:
                print(free_blocks.summary(blocks, block_times, times[-1]))


if __name__ == '__main__':
    main()
