"""
Makes a test-frames file for the QNX vision program, plus Python's reference output for it.

Takes n consecutive frames of a recorded session (video + its capture timestamps), runs the same
pose the Pi model was trained on (yolo11n-pose.onnx through ultralytics at 320, largest person,
like repose_onnx.py) and the Python live detector (early_live.EarlyDetector with
models/early_detector_onnx.joblib, the model exported to qnx/model/early_model.c).

  python qnx/vision_pi/make_test_frames.py data/session_X.mp4 --start-time 60 --frames 180
      -> qnx/vision_pi/test_frames.bin   (BBXF: w,h,n; per frame: t f64, kp f32[17][2], conf f32[17], BGR)
      -> qnx/vision_pi/test_ref.csv      (same columns as `vision --csv`)

Then on the Pi:  vision --frames test_frames.bin --csv pi.csv  (and --file-pose for an exact feature check)
and here:        python qnx/vision_pi/compare_csv.py qnx/vision_pi/test_ref.csv pi.csv
"""
import argparse
import os
import struct
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from early_live import EarlyDetector, feature_matrix  # noqa: E402


def largest(r):
    if r.keypoints is None or r.boxes is None or not len(r.boxes):
        return np.full((17, 2), np.nan), np.zeros(17)
    b = r.boxes.xyxy.cpu().numpy()
    i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    conf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
    return r.keypoints.xy[i].cpu().numpy().astype(float), conf.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start-time", type=float, default=60.0, help="seconds into the session")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--out", default=os.path.join(ROOT, "qnx", "vision_pi", "test_frames.bin"))
    ap.add_argument("--ref", default=os.path.join(ROOT, "qnx", "vision_pi", "test_ref.csv"))
    args = ap.parse_args()

    from ultralytics import YOLO
    z = np.load(os.path.splitext(args.video)[0] + ".npz", allow_pickle=True)
    ts = z["timestamps"].astype(float)
    start = int(np.searchsorted(ts - ts[0], args.start_time))
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    pose = YOLO(os.path.join(ROOT, "yolo11n-pose.onnx"), task="pose")
    det = EarlyDetector(os.path.join(ROOT, "models", "early_detector_onnx.joblib"), budget=2.0)

    frames = []
    for i in range(args.frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append((float(ts[start + i] - ts[0]), f))
    h, w = frames[0][1].shape[:2]
    with open(args.out, "wb") as bf, open(args.ref, "w") as cf:
        bf.write(b"BBXF" + struct.pack("<3i", w, h, len(frames)))
        cf.write("i,t,found," + ",".join(f"w{k}_{n}" for k in range(2) for n in det.names) + ",score0,score1,event\n")
        events = 0
        for i, (t, f) in enumerate(frames):
            kps, conf = largest(pose(f, imgsz=320, verbose=False)[0])
            bf.write(struct.pack("<d", t) + kps.astype("<f4").tobytes() + conf.astype("<f4").tobytes() + f.tobytes())
            # the Pi reads the keypoints back as float32: give Python the same rounded values
            kps32, conf32 = kps.astype(np.float32).astype(float), conf.astype(np.float32).astype(float)
            out = det.update(t, f, kps32, conf32)
            X = feature_matrix(det.state.rows, det.names)
            ev = out["event"]["side"] if out["event"] else ""
            events += bool(ev)
            cf.write(f"{i},{t:.6f},{int(np.isfinite(kps32[5]).all())}," + ",".join(f"{v:.9g}" for v in X.ravel())
                     + f",{out['scores'][0]:.9g},{out['scores'][1]:.9g},{ev}\n")
    mb = os.path.getsize(args.out) / 1e6
    print(f"{len(frames)} frames {w}x{h} from t={frames[0][0]:.1f}s -> {args.out} ({mb:.0f} MB); "
          f"python cues: {events}; reference -> {args.ref}")


if __name__ == "__main__":
    main()
