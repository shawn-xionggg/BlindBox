"""
Re-run pose on reviewed sessions with the ONNX model (the planned Pi/QNX pose
runtime), so the detector can be trained on the same keypoints it will see
when deployed. Originals in data/ are not touched; output goes to data_onnx/
as <stem>_onnx.* (npz with new keypoints, plus copies of the labels, the
auto-label candidates and the video).

  python repose_onnx.py data/session_A.npz data/session_B.npz
  python early_detector.py --data data_onnx --save models/early_detector_onnx.joblib
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

OUT = Path("data_onnx")


def largest(r):
    if r.keypoints is None or r.boxes is None or not len(r.boxes):
        return np.full((17, 2), np.nan), np.zeros(17)
    b = r.boxes.xyxy.cpu().numpy()
    i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    return r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()


def main():
    OUT.mkdir(exist_ok=True)
    model = YOLO("yolo11n-pose.onnx", task="pose")
    for a in sys.argv[1:]:
        src = Path(a)
        stem = src.stem + "_onnx"
        z = dict(np.load(src, allow_pickle=False))
        cap = cv2.VideoCapture(str(src.with_suffix(".mp4")))
        kp, cf = np.full_like(z["keypoints"], np.nan), np.zeros_like(z["confidences"])
        for i in range(len(z["timestamps"])):
            ok, f = cap.read()
            if not ok:
                break
            kp[i], cf[i] = largest(model(f, imgsz=320, verbose=False)[0])
            if i and i % 2000 == 0:
                print(f"  {src.name}: {i}/{len(kp)}")
        z["keypoints"], z["confidences"] = kp, cf
        z["pose_model"] = np.array("yolo11n-pose.onnx")
        np.savez_compressed(OUT / (stem + ".npz"), **z)
        for ext in (".mp4", ".labels.json", ".auto_candidates.json"):
            if src.with_suffix(ext).exists():
                shutil.copyfile(src.with_suffix(ext), OUT / (stem + ext))
        print(f"{src.name} -> {OUT / (stem + '.npz')}")


if __name__ == "__main__":
    main()
