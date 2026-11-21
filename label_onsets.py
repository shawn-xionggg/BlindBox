"""Frame-step independent onset labels. Each labeler's marks are preserved in JSON.

,/. single frame; j/l ten frames; space play; 1 preparation; 2 departure;
f cycle family; u undo; q save/quit. Repeat with --labeler another-name.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from pose_features import COMMANDS
from collect import text


def save_labels(path, labeler, marks):
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {'labelers': {}}
    data['labelers'][labeler] = marks
    # Keep independent marks; derived ranges align each labeler's chronological trials.
    groups = list(data['labelers'].values())
    data['ranges'] = []
    for i in range(max((len(g) for g in groups), default=0)):
        trials = [g[i] for g in groups if i < len(g)]
        row = {}
        for key in ('preparation', 'departure'):
            values = [m[key] for m in trials if m.get(key) is not None]
            row[key] = [min(values), max(values)] if values else None
        row['families'] = sorted({m['family'] for m in trials})
        data['ranges'].append(row)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    # Windows: antivirus / the search indexer can hold labels.json open for a moment,
    # making the atomic replace fail with "Access is denied". Retry instead of crashing.
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            time.sleep(0.1)
    path.write_text(temp.read_text())   # last resort: overwrite in place
    temp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('video')
    ap.add_argument('--labeler', required=True)
    args = ap.parse_args()
    video = Path(args.video)
    path = video.with_suffix('.labels.json')
    data = json.loads(path.read_text()) if path.exists() else {'labelers': {}}
    marks = data['labelers'].get(args.labeler, [])
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {video}')
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stamps = None
    if video.with_suffix('.npz').exists():
        with np.load(video.with_suffix('.npz')) as z:
            stamps = z['timestamps']
    index, family, preparation, play = 0, 0, None, False
    undo = []
    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                break
            t = float(stamps[index]) if stamps is not None and index < len(stamps) else index/fps
            text(frame, f'{args.labeler} | frame {index} | {t:.3f}s | {COMMANDS[family]}', 25)
            text(frame, ',/. frame j/l 10 space play 1 prep 2 departure f family u undo q save', 50)
            text(frame, f'Preparation: {preparation} | departures: {len(marks)}', 75)
            cv2.imshow('Label onsets', frame)
            key = cv2.waitKey(max(1, round(1000/fps)) if play else 0) & 255
            if key in (27, ord('q')):
                break
            if key == ord(' '):
                play = not play
            if key in map(ord, ',.jl'):
                index = max(0, min(count-1, index+{ord(','): -1, ord('.'): 1, ord('j'): -10, ord('l'): 10}[key]))
            if key == ord('f'):
                family = (family+1) % len(COMMANDS)
            if key in (ord('1'), ord('2')):
                undo.append((json.loads(json.dumps(marks)), preparation))
                if key == ord('1'):
                    preparation = index
                else:
                    marks.append(dict(preparation=preparation, departure=index, family=COMMANDS[family]))
                    marks.sort(key=lambda m: m['departure'])
                    preparation = None
                save_labels(path, args.labeler, marks)
            if key == ord('u') and undo:
                marks, preparation = undo.pop()
                save_labels(path, args.labeler, marks)
            if play:
                index = min(count-1, index+1)
                if index == count-1:
                    play = False
    finally:
        save_labels(path, args.labeler, marks)
        cap.release()
        cv2.destroyAllWindows()
    print('Saved', path, '; disagreements retained as ranges, not adjudicated truth.')


if __name__ == '__main__':
    main()
