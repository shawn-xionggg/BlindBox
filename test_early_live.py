"""
Check that the live EarlyDetector computes exactly the same features as the
offline training pipeline (early_dataset.py), by replaying a recorded session
frame by frame. If these differ, offline accuracy numbers would not hold live.

  python test_early_live.py [data/session_X.npz] [--frames N]
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import early_dataset
from early_live import EarlyDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default=None)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--model", default="models/early_detector.joblib")
    args = ap.parse_args()
    npz = Path(args.npz) if args.npz else sorted(Path("data").glob("*.npz"))[0]
    off = early_dataset.build(npz)
    z = np.load(npz, allow_pickle=False)
    t, kp, cf = z["timestamps"], z["keypoints"], z["confidences"]
    det = EarlyDetector(args.model, dt_est=float(np.median(np.diff(t))))
    cap = cv2.VideoCapture(str(npz.with_suffix(".mp4")))
    n = min(args.frames, len(t))
    names = off["names"]
    worst, bad = {}, {}
    ms = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        X = det.features(t[i], frame, kp[i], cf[i])
        ms.append((time.perf_counter() - t0) * 1000)
        Y = off["X"][i]
        for j, nm in enumerate(names):
            a, b = X[:, j], Y[:, j]
            both_nan = np.isnan(a) & np.isnan(b)
            diff = np.where(both_nan, 0, np.abs(a - b))
            mism = (np.isnan(a) != np.isnan(b)) | (diff > 1e-5 * (1 + np.abs(np.nan_to_num(b))))  # float rounding in the flow sums reaches ~2e-6
            if mism.any():
                bad[nm] = bad.get(nm, 0) + int(mism.sum())
                worst[nm] = max(worst.get(nm, 0), float(np.nanmax(np.where(np.isnan(diff), np.inf, diff))))
    print(f"{npz.name}: compared {n} frames x 2 wrists x {len(names)} features")
    print(f"live feature cost: {np.mean(ms):.1f} ms/frame (flow + energy + pose features, laptop CPU)")
    if not bad:
        print("OK: live features match offline features exactly")
    else:
        for nm in names:
            if nm in bad:
                print(f"  MISMATCH {nm:12s} {bad[nm]:5d} values, worst |diff| {worst[nm]:.3g}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
