"""
Does swapping the PyTorch pose model for its ONNX export change WHEN the
detector beeps? Replays a whole session through EarlyDetector twice -- once
with the saved PyTorch keypoints, once with fresh ONNX keypoints -- and
compares the cue times, plus how each version scores against the reviewed
punch starts.

  python check_onnx_detector.py [data/session_X.npz]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from early_live import EarlyDetector
from pose_features import NEGATIVES


def largest(r):
    if r.keypoints is None or r.boxes is None or not len(r.boxes):
        return np.full((17, 2), np.nan), np.zeros(17)
    b = r.boxes.xyxy.cpu().numpy()
    i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    return r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()


def main():
    npz = Path(sys.argv[1] if len(sys.argv) > 1 else "data/session_20261121_024335_748065.npz")
    z = dict(np.load(npz, allow_pickle=False))
    t = z["timestamps"]
    dt = float(np.median(np.diff(t)))
    lab = json.loads(npz.with_suffix(".labels.json").read_text())["labelers"]["reviewed"]
    deps = [m["departure"] for m in lab if m["family"] not in NEGATIVES]
    onnx = YOLO("yolo11n-pose.onnx", task="pose")
    cap = cv2.VideoCapture(str(npz.with_suffix(".mp4")))
    det_pt, det_ox = EarlyDetector(dt_est=dt), EarlyDetector(dt_est=dt)
    ev_pt, ev_ox = [], []
    for i in range(len(t)):
        ok, frame = cap.read()
        if not ok:
            break
        kx, cx = largest(onnx(frame, imgsz=320, verbose=False)[0])
        a = det_pt.update(t[i], frame, z["keypoints"][i], z["confidences"][i])["event"]
        b = det_ox.update(t[i], frame, kx, cx)["event"]
        if a:
            ev_pt.append(i)
        if b:
            ev_ox.append(i)
        if i and i % 1000 == 0:
            print(f"  {i}/{len(t)} frames")

    def match(x, y, tol=2):
        return sum(any(abs(a - b) <= tol for b in y) for a in x)

    def caught(ev):
        hits = [min((e - d for e in ev if d - 2 <= e <= d + 6), default=None) for d in deps]
        got = [h for h in hits if h is not None]
        return sum(h <= 2 for h in got), sum(h <= 4 for h in got), len(deps)

    print(f"\n{npz.name}: PyTorch-pose cues {len(ev_pt)}, ONNX-pose cues {len(ev_ox)}")
    print(f"cues within ±2 frames of each other: {match(ev_pt, ev_ox)}/{len(ev_pt)} (PyTorch side), "
          f"{match(ev_ox, ev_pt)}/{len(ev_ox)} (ONNX side)")
    for name, ev in (("PyTorch pose", ev_pt), ("ONNX pose", ev_ox)):
        c2, c4, n = caught(ev)
        print(f"{name:12s}: punches caught within +2 frames {c2}/{n}, within +4 frames {c4}/{n}  "
              f"(in-sample model -- only the comparison matters)")


if __name__ == "__main__":
    main()
