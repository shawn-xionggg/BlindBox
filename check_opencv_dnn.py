"""
Can OpenCV's DNN module run the pose model instead of ONNX Runtime?

QNX has an official OpenCV port (qnx-ports/build-files) but no known ONNX
Runtime port. If cv::dnn runs yolo11n-pose.onnx with the same output, the Pi
needs only OpenCV (pose + optical flow), with pose_decode.c on top.

Compares raw network outputs (OpenCV DNN vs ONNX Runtime) on real letterboxed
frames, then the decoded keypoints, and times both.

  python check_opencv_dnn.py [video.mp4] [--frames 100]
"""
import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics.data.augment import LetterBox


def decode_largest(out, w, h, conf=0.25, iou=0.7):
    """Python mirror of qnx/pose/pose_decode.c (verified identical to ultralytics)."""
    r = min(320 / h, 320 / w)
    px, py = round((320 - round(w * r)) / 2 - 0.1), round((320 - round(h * r)) / 2 - 0.1)
    s = out[4]
    idx = np.flatnonzero(s > conf)
    if not len(idx):
        return None
    idx = idx[np.argsort(-s[idx], kind="stable")]
    boxes = np.stack([out[0] - out[2] / 2, out[1] - out[3] / 2, out[0] + out[2] / 2, out[1] + out[3] / 2], 1)
    keep = []
    for a in idx:
        ok = True
        for b in keep:
            x1, y1 = max(boxes[a, 0], boxes[b, 0]), max(boxes[a, 1], boxes[b, 1])
            x2, y2 = min(boxes[a, 2], boxes[b, 2]), min(boxes[a, 3], boxes[b, 3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            ua = np.prod(boxes[a, 2:] - boxes[a, :2]) + np.prod(boxes[b, 2:] - boxes[b, :2]) - inter
            if ua > 0 and inter / ua > iou:
                ok = False
                break
        if ok:
            keep.append(a)
    pick = max(keep, key=lambda a: np.prod(boxes[a, 2:] - boxes[a, :2]))
    k = out[5:, pick].reshape(17, 3)
    xy = np.c_[np.clip((k[:, 0] - px) / r, 0, w), np.clip((k[:, 1] - py) / r, 0, h)]
    return xy, k[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="data/session_20261121_030328_194149.mp4")
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, args.frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    lb = LetterBox(new_shape=(320, 320), auto=False, stride=32)
    sess = ort.InferenceSession("yolo11n-pose.onnx", providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    net = cv2.dnn.readNetFromONNX("yolo11n-pose.onnx")
    raw_diff, kp_diff, vis_diff, found_mismatch = [], [], [], 0
    t_ort, t_cv = [], []
    for f in frames:
        x = np.ascontiguousarray(lb(image=f)[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        t0 = time.perf_counter()
        a = sess.run(None, {name: x})[0][0]
        t_ort.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        net.setInput(x)
        b = net.forward()[0]
        t_cv.append(time.perf_counter() - t0)
        raw_diff.append(np.abs(a - b).max())
        da, db = decode_largest(a, f.shape[1], f.shape[0]), decode_largest(b, f.shape[1], f.shape[0])
        if (da is None) != (db is None):
            found_mismatch += 1
        elif da is not None:
            kp_diff.append(np.abs(da[0] - db[0]).max())
            vis_diff.append(np.abs(da[1] - db[1]).max())
    print(f"OpenCV {cv2.__version__} DNN vs ONNX Runtime on {len(frames)} real frames")
    print(f"raw output max |diff|: median {np.median(raw_diff):.2e}, worst {max(raw_diff):.2e}")
    print(f"decoded: person found/not mismatches {found_mismatch}; keypoint max diff median "
          f"{np.median(kp_diff):.3f} px, worst {max(kp_diff):.3f} px; visibility worst {max(vis_diff):.4f}")
    print(f"speed on this laptop: ONNX Runtime {1000 * np.mean(t_ort[3:]):.1f} ms, OpenCV DNN {1000 * np.mean(t_cv[3:]):.1f} ms per frame")


if __name__ == "__main__":
    main()
