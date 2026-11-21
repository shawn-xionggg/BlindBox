"""
Fake different lighting on recorded sessions, so the detector learns to cope with it.

Real bad lighting changes three things the pipeline sees:
  1. darker / brighter, lower-contrast images  -> the pose model's keypoints get worse
  2. more sensor noise (the camera boosts gain in the dark)  -> noisier optical flow
  3. motion blur (longer exposure)  -> blurred fists, smeared flow
Each preset re-renders the session video with those effects, re-runs the pose
model on the new frames, and saves a training copy in data_aug/:
  data_aug/<stem>_<preset>.npz / .mp4 / .labels.json / .auto_candidates.json
Timestamps and labels are unchanged (same frames, same punches).

Effects per pixel (values 0..1):  out = clip(gain * in**gamma * tint + noise)
  gamma > 1 crushes shadows (dim room), gamma < 1 washes out (overexposed)
  blur: out = (1 - b) * frame + b * previous output  (exposure smear)

  python augment_lighting.py data/session_X.npz [more.npz] --presets dim dim_blur bright
  python augment_lighting.py data/session_X.npz --presets --fps 15   # frame-rate copy only (no pose re-run)
  python augment_lighting.py --compare data/session_X.npz   # how much each preset hurts pose

A dim room often hurts most through the camera: auto-exposure lowers the frame rate
(C270: ~30 -> ~15 fps) and blurs moving fists. So there are frame-rate copies too.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from pose_features import PoseRunner

PRESETS = {
    # dim, warm indoor light: dark, crushed shadows, orange tint, sensor noise
    "dim":      dict(gain=0.40, gamma=1.5, tint=(0.80, 0.95, 1.10), noise=0.035, blur=0.0),
    # dim + long exposure: also smears motion
    "dim_blur": dict(gain=0.50, gamma=1.3, tint=(1.0, 1.0, 1.0), noise=0.025, blur=0.45),
    # bright / overexposed / backlit: washed out, cool tint
    "bright":   dict(gain=1.45, gamma=0.65, tint=(1.08, 1.0, 0.92), noise=0.01, blur=0.0),
}
OUT = Path("data_aug")


class Relight:
    def __init__(self, gain, gamma, tint, noise, blur, seed=0):
        self.gain, self.gamma, self.noise, self.blur = gain, gamma, noise, blur
        self.tint = np.array(tint, np.float32)   # B, G, R multipliers (OpenCV order)
        self.lut = np.clip(gain * (np.arange(256) / 255.0) ** gamma, 0, 1).astype(np.float32)
        self.rng = np.random.default_rng(seed)
        self.prev = None

    def __call__(self, frame):
        f = self.lut[frame] * self.tint
        if self.blur and self.prev is not None:
            f = (1 - self.blur) * f + self.blur * self.prev
        self.prev = f
        if self.noise:
            f = f + self.rng.normal(0, self.noise, f.shape).astype(np.float32)
        return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def augment(npz, preset, pose):
    src = Path(npz)
    stem = f"{src.stem}_{preset}"
    OUT.mkdir(exist_ok=True)
    z = dict(np.load(src, allow_pickle=False))
    cap = cv2.VideoCapture(str(src.with_suffix(".mp4")))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    relight = Relight(**PRESETS[preset])
    n = len(z["timestamps"])
    kp, cf = np.full_like(z["keypoints"], np.nan), np.zeros_like(z["confidences"])
    writer = None
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        g = relight(frame)
        if writer is None:
            writer = cv2.VideoWriter(str(OUT / (stem + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (g.shape[1], g.shape[0]))
        writer.write(g)
        person = pose(g)
        if person is not None:
            kp[i], cf[i] = person
        if i and i % 3000 == 0:
            print(f"  {stem}: {i}/{n}", flush=True)
    writer.release()
    cap.release()
    z["keypoints"], z["confidences"] = kp, cf
    np.savez_compressed(OUT / (stem + ".npz"), **z)
    for ext in (".labels.json", ".auto_candidates.json"):
        if src.with_suffix(ext).exists():
            shutil.copyfile(src.with_suffix(ext), OUT / (stem + ext))
    json.dump(dict(source=src.name, preset=preset, **PRESETS[preset]), open(OUT / (stem + ".aug.json"), "w"))
    print(f"{src.name} -> {OUT / (stem + '.npz')}", flush=True)


def subsample(npz, target_fps):
    """Lower frame rate: keep the frames closest to a target_fps grid. Pose points are reused
    (same frames), so no pose re-run; the video is rewritten so flow/pixel-change features
    see the bigger frame-to-frame jumps a slow camera gives. Frame-number labels are remapped."""
    src = Path(npz)
    stem = f"{src.stem}_fps{target_fps}"
    OUT.mkdir(exist_ok=True)
    z = dict(np.load(src, allow_pickle=False))
    t = z["timestamps"]
    n = len(t)
    grid = np.arange(t[0], t[-1], 1.0 / target_fps)
    keep = np.unique(np.clip(np.searchsorted(t, grid), 0, n - 1))
    for k, v in list(z.items()):
        if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n:
            z[k] = v[keep]
    z["keep_index"] = keep.astype(np.int64)   # which source frames were kept (the Pi stage reuses this)
    if "onset_frames" in z:
        z["onset_frames"] = np.array([int(np.argmin(np.abs(keep - f))) for f in z["onset_frames"]], dtype=int)
    np.savez_compressed(OUT / (stem + ".npz"), **z)

    def remap(f):
        return None if f is None else int(np.argmin(np.abs(keep - int(f))))
    lab = json.load(open(src.with_suffix(".labels.json")))
    for marks in lab["labelers"].values():
        for m in marks:
            m["departure"], m["preparation"] = remap(m["departure"]), remap(m.get("preparation"))
    lab.pop("ranges", None)
    json.dump(lab, open(OUT / (stem + ".labels.json"), "w"))
    cand = json.load(open(src.with_suffix(".auto_candidates.json")))
    for c in cand["candidates"]:
        c["departure"], c["peak"] = remap(c["departure"]), remap(c["peak"])
    json.dump(cand, open(OUT / (stem + ".auto_candidates.json"), "w"))

    cap = cv2.VideoCapture(str(src.with_suffix(".mp4")))
    writer, ks, i = None, set(keep.tolist()), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in ks:
            if writer is None:
                writer = cv2.VideoWriter(str(OUT / (stem + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), target_fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        i += 1
    writer.release()
    cap.release()
    json.dump(dict(source=src.name, preset=f"fps{target_fps}"), open(OUT / (stem + ".aug.json"), "w"))
    print(f"{src.name} -> {OUT / (stem + '.npz')} ({len(keep)}/{n} frames)", flush=True)


def compare(npz, pose, frames=300):
    """How much does each preset change the pose model's wrists and shoulders?"""
    src = Path(npz)
    z = np.load(src)
    cap = cv2.VideoCapture(str(src.with_suffix(".mp4")))
    n = len(z["timestamps"])
    idx = set(np.linspace(0, n - 1, frames).astype(int))
    raw = []
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        if i in idx:
            raw.append(f)
    base = [pose(f) for f in raw]
    for name, p in PRESETS.items():
        r = Relight(**p)
        lost, err = 0, []
        for f, b in zip(raw, base):
            a = pose(r(f))
            for k in (5, 6, 9, 10):
                if b is None or b[1][k] < 0.35:
                    continue
                if a is None or a[1][k] < 0.35:
                    lost += 1
                    continue
                sw = np.linalg.norm(b[0][5] - b[0][6]) or 1
                err.append(np.linalg.norm(a[0][k] - b[0][k]) / sw)
        tot = lost + len(err)
        print(f"{name:9s} shoulders+wrists lost {lost}/{tot} ({100 * lost / max(tot, 1):.0f}%), "
              f"moved median {np.median(err):.3f} / 90th pct {np.percentile(err, 90):.3f} shoulder widths")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--presets", nargs="*", default=list(PRESETS), choices=list(PRESETS))
    ap.add_argument("--fps", type=int, nargs="*", default=[], help="also make lower-frame-rate copies, e.g. --fps 15")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    pose = PoseRunner()
    for p in args.npz:
        if args.compare:
            compare(p, pose)
            continue
        for preset in args.presets:
            augment(p, preset, pose)
        for f in args.fps:
            subsample(p, f)


if __name__ == "__main__":
    main()
