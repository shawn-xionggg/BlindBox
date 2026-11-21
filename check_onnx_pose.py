"""
De-risk pose on the Pi/QNX: does the ONNX export of yolo11n-pose give the same
keypoints as the PyTorch model, and how fast is raw ONNX Runtime inference?

On QNX the plan is ONNX Runtime (C/C++ API) running yolo11n-pose.onnx, so the
model file must be verified first. Pre/post-processing (letterbox, decode,
NMS) is done here by ultralytics; on QNX it has to be reimplemented in C/C++.

  python check_onnx_pose.py [data/session_X.mp4] [--frames 300]
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


def largest(r):
    if r.keypoints is None or r.boxes is None or not len(r.boxes):
        return None, None
    b = r.boxes.xyxy.cpu().numpy()
    i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    return r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="data/session_20261121_030328_194149.mp4")
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, n - 1, args.frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    pt, ox = YOLO("yolo11n-pose.pt"), YOLO("yolo11n-pose.onnx", task="pose")
    diffs, conf_agree, both, only_one = [], [], 0, 0
    for f in frames:
        a, ca = largest(pt(f, imgsz=320, verbose=False)[0])
        b, cb = largest(ox(f, imgsz=320, verbose=False)[0])
        if a is None and b is None:
            continue
        if (a is None) != (b is None):
            only_one += 1
            continue
        both += 1
        m = (ca > 0.35) & (cb > 0.35)
        if m.any():
            diffs.append(np.linalg.norm(a[m] - b[m], axis=1))
        conf_agree.append(np.mean((ca > 0.35) == (cb > 0.35)))
    d = np.concatenate(diffs) if diffs else np.array([np.nan])
    print(f"{len(frames)} frames | person found by both: {both}, by only one: {only_one}")
    print(f"keypoint difference (px, 640x480 frame): median {np.median(d):.2f}, p99 {np.percentile(d, 99):.2f}, max {d.max():.2f}")
    print(f"keypoint visible/not-visible agreement: {100 * np.mean(conf_agree):.1f}%")

    import onnxruntime as ort
    x = np.random.rand(1, 3, 320, 320).astype(np.float32)
    for threads in (1, 2, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession("yolo11n-pose.onnx", so, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        for _ in range(5):
            s.run(None, {name: x})
        t0 = time.perf_counter()
        for _ in range(50):
            s.run(None, {name: x})
        ms = (time.perf_counter() - t0) / 50 * 1000
        print(f"raw ONNX Runtime inference, {threads} thread(s): {ms:.1f} ms/frame on this laptop")
    print(f"input {s.get_inputs()[0].shape}  output {[o.shape for o in s.get_outputs()]}")


if __name__ == "__main__":
    main()
