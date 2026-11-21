"""
Optical-flow features around each wrist, from a recorded session video.

Why: a fist coming straight at the camera barely moves in 2D -- it gets
bigger. Optical flow (how every pixel moved since the last frame) sees that
as pixels spreading outward around the fist: positive DIVERGENCE
(du/dx + dv/dy). A whole-body step-in makes the torso expand too, so torso
flow is subtracted.

Per frame, per wrist (k=0: COCO kp 9, k=1: kp 10), all causal (frames i-1, i):
  div       mean divergence in a box around the wrist          (1/s)
  div_rel   div minus torso divergence (cancels step-in / camera zoom)
  mag_rel   mean |flow - torso flow| in the wrist box, in shoulder widths/s
  out_rel   mean flow (minus torso) pointing away from the shoulder midpoint,
            in shoulder widths/s
Plus per frame: torso_div, torso_mag.

Flow is computed on a half-size grayscale frame (Farneback). Results are
cached in features/<session>.flow.npz.

  python flow_features.py data/session_X.npz [more.npz ...]
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import os

L_SH, R_SH, WR = 5, 6, (9, 10)
KC = 0.35
# Flow resolution. 0.25 = quarter size (default: ~4x cheaper than half size and
# cross-validated as accurate or better); 0.5 = half size (~14 ms/frame on the
# laptop). Override with the FLOW_SCALE environment variable; the cache file
# name includes the scale, and early_live.py uses the same value.
SCALE = float(os.environ.get("FLOW_SCALE", "0.25"))
TAG = "" if SCALE == 0.5 else f"_s{SCALE:g}"


def box_mean(a, cx, cy, r, h, w):
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r + 1))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + r + 1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return a[y0:y1, x0:x1]


def compute(npz):
    z = np.load(npz, allow_pickle=False)
    t, kp, cf = z["timestamps"], z["keypoints"].astype(float), z["confidences"]
    n = len(t)
    cap = cv2.VideoCapture(str(npz.with_suffix(".mp4")))
    # u_rel / v_rel: mean sideways / vertical flow around the fist minus torso flow
    # (shoulder widths/s) -- used by the direction experiments, not by the detector
    out = {k: np.full((n, 2), np.nan) for k in ("div", "div_rel", "mag_rel", "out_rel", "u_rel", "v_rel")}
    torso_div = np.full(n, np.nan)
    torso_mag = np.full(n, np.nan)
    prev = None
    last_wr = [None, None]
    t0 = time.time()
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sh = cf[i, L_SH] > KC and cf[i, R_SH] > KC
        for k, w in enumerate(WR):
            if cf[i, w] > KC and np.isfinite(kp[i, w]).all():
                last_wr[k] = kp[i, w] * SCALE
        if prev is not None and sh and i > 0 and t[i] > t[i - 1]:
            dt = t[i] - t[i - 1]
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            u, v = flow[..., 0], flow[..., 1]
            div = np.gradient(u, axis=1) + np.gradient(v, axis=0)
            H, W = g.shape
            width = np.linalg.norm(kp[i, R_SH] - kp[i, L_SH]) * SCALE
            if width > 5:
                mid = (kp[i, L_SH] + kp[i, R_SH]) / 2 * SCALE
                # torso box: below the shoulder midpoint
                tr = 0.35 * width
                tcy = mid[1] + 0.6 * width
                tu = box_mean(u, mid[0], tcy, tr, H, W)
                tv = box_mean(v, mid[0], tcy, tr, H, W)
                td = box_mean(div, mid[0], tcy, tr, H, W)
                if tu is not None:
                    tfu, tfv = float(tu.mean()), float(tv.mean())
                    torso_div[i] = float(td.mean()) / dt
                    torso_mag[i] = np.hypot(tfu, tfv) / width / dt
                else:
                    tfu = tfv = 0.0
                    torso_div[i] = 0.0
                r = 0.35 * width
                for k in range(2):
                    if last_wr[k] is None:
                        continue
                    cx, cy = last_wr[k]
                    bu = box_mean(u, cx, cy, r, H, W)
                    if bu is None:
                        continue
                    bv = box_mean(v, cx, cy, r, H, W)
                    bd = box_mean(div, cx, cy, r, H, W)
                    du, dv = bu - tfu, bv - tfv
                    out["div"][i, k] = float(bd.mean()) / dt
                    out["div_rel"][i, k] = out["div"][i, k] - torso_div[i]
                    out["mag_rel"][i, k] = float(np.hypot(du, dv).mean()) / width / dt
                    d = np.array([cx, cy]) - mid
                    d = d / (np.linalg.norm(d) + 1e-6)
                    out["out_rel"][i, k] = float((du * d[0] + dv * d[1]).mean()) / width / dt
                    out["u_rel"][i, k] = float(du.mean()) / width / dt
                    out["v_rel"][i, k] = float(dv.mean()) / width / dt
        prev = g
        if i and i % 2000 == 0:
            print(f"  {npz.name}: {i}/{n} frames ({time.time() - t0:.0f}s)")
    cap.release()
    dst = Path("features") / (npz.stem + f".flow{TAG}.npz")
    dst.parent.mkdir(exist_ok=True)
    np.savez_compressed(dst, torso_div=torso_div, torso_mag=torso_mag, **out)
    print(f"{npz.name}: {n} frames in {time.time() - t0:.0f}s -> {dst}")
    return dst


def load(npz):
    p = Path("features") / (Path(npz).stem + f".flow{TAG}.npz")
    if not p.exists() or "u_rel" not in np.load(p).files:
        compute(Path(npz))
    return dict(np.load(p))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        compute(Path(a))
