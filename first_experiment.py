"""Plot 1+ straight-reach, lateral-sweep and step-in clips/NPZ on identical axes.
Video input needs --color or saved glove data; NPZ needs no pose runtime.
"""
import argparse
import os
from pathlib import Path
import numpy as np
from pose_features import FeatureStream, PoseRunner, MotionOnset
from glove_features import GloveSegmenter, PRESETS
from collect import open_source, video_timestamp


def measurements(session, glove_thresh=3.):
    stream, detector = FeatureStream(), MotionOnset(glove_thresh=glove_thresh)
    speeds, areas, events = [], [], []
    previous = None
    for i, (t, xy, conf) in enumerate(zip(session['timestamps'], session['keypoints'], session['confidences'])):
        blobs = session.get('glove_blobs')
        stream.update(t, xy, conf, blobs[i] if blobs is not None else None)
        f = stream.history[-1]
        speed = np.zeros(2)
        if previous is not None and not f.unknown and not previous.unknown:
            valid = f.mask[[9, 10]] & previous.mask[[9, 10]]
            speed[valid] = np.linalg.norm(f.xy[[9, 10]][valid]-previous.xy[[9, 10]][valid], axis=1)/max(t-previous.t, 1e-9)
        speeds.append(speed)
        areas.append(np.where(f.glove_mask, f.glove_area, np.nan))
        event = detector.update(f, i)
        if event:
            events.append(event)
        previous = f
    return np.asarray(speeds), np.asarray(areas), events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('clips', nargs='+')
    ap.add_argument('--color', choices=PRESETS)
    ap.add_argument('--model', default='yolo11n-pose.pt')
    ap.add_argument('--output-dir', default='experiments')
    ap.add_argument('--glove-thresh', type=float, default=3.)
    args = ap.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str((output/'.matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runner = None
    for source in args.clips:
        path = Path(source)
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as z:
                session = {k: z[k] for k in z.files}
        else:
            runner = runner or PoseRunner(args.model)
            segment = GloveSegmenter(args.color)
            cap, _ = open_source(source)
            fps = cap.get(5) or 30
            session = {k: [] for k in ('timestamps', 'keypoints', 'confidences', 'glove_blobs')}
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    person = runner(frame)
                    xy, conf = person if person is not None else (np.full((17, 2), np.nan), np.zeros(17))
                    session['timestamps'].append(video_timestamp(cap, len(session['timestamps']), fps,
                        session['timestamps'][-1] if session['timestamps'] else None))
                    session['keypoints'].append(xy)
                    session['confidences'].append(conf)
                    session['glove_blobs'].append(segment(frame))
            finally:
                cap.release()
        speeds, areas, events = measurements(session, args.glove_thresh)
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        for j, side in enumerate(('image-left wrist', 'image-right wrist')):
            axes[0].plot(session['timestamps'], speeds[:, j], label=side)
            axes[1].plot(session['timestamps'], areas[:, j], label=side)
        for ax in axes:
            for event in events:
                ax.axvline(event['time'], color='red', alpha=.5)
            ax.legend()
            ax.grid(alpha=.2)
        axes[0].set(ylabel='Wrist speed (shoulder widths/s)', title=path.stem+' | red = auto-onset')
        axes[1].set(ylabel='Glove area / shoulder width squared', xlabel='Capture/video time (s)')
        fig.tight_layout()
        target = output/(path.stem+'.png')
        fig.savefig(target, dpi=150)
        plt.close(fig)
        print(target, '| auto-onsets:', len(events))


if __name__ == '__main__':
    main()
