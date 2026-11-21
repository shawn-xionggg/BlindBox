"""
Per-wrist, per-frame features + labels for EARLY punch detection, built from
reviewed sessions. Shared by early_detector.py (train/evaluate) and the
streaming early detector.

Everything here is causal (uses frame i and earlier), so it can run live.

Features per wrist (k=0: COCO kp 9, k=1: kp 10), lags 0..2 frames where noted:
  disp_rest  distance from the wrist's recent rest position (shoulder widths)  [lags 0-2]
  launch     LaunchDetector distance from the held guard (0 outside its window) [lags 0-2]
  speed      body-relative wrist speed (shoulder widths/s)                      [lags 0-1]
  relx, rely wrist position relative to the shoulders
  energy     pixel change around the wrist / its recent baseline                 [lags 0-1]
  div_rel, mag_rel, out_rel   optical-flow features (flow_features.py)        [lags 0-2 / 0-1 / 0-1]
  torso_div, torso_mag, width_rate   whole-body motion (step-ins, turns)
  other_speed  the other wrist's speed

Labels per wrist sample:
  1   punching wrist, frames departure .. departure+3 (first ~150 ms)
  0   scripted don't-punch windows except FEINT (>1 s from punches),
      calm frames >1 s from any auto-label candidate,
      and retractions: punching wrist, frames peak+2 .. peak+10
  -1  everything else (ignored: ambiguous)
"""
import json
from pathlib import Path

import numpy as np

import flow_features
from auto_label import motion_energy
from launch_detector import LaunchDetector
from pose_features import NEGATIVES

L_SH, R_SH, WR = 5, 6, (9, 10)
KC = 0.35
POS_FRAMES = 4
REF_DT = 0.048   # frame time the frame-count windows were tuned at (~20.8 fps)
EXTRA = False  # v2 features (rest history, both-hands, head motion): tested, no gain -> off


def _energy(npz, kp, cf, sh, width):
    p = Path("features") / (npz.stem + ".energy.npy")
    if p.exists():
        return np.load(p)
    n = len(kp)
    pix = np.full((n, 2, 2), np.nan)
    for k, w in enumerate(WR):
        last = None
        for i in range(n):
            if cf[i, w] > KC and np.isfinite(kp[i, w]).all():
                last = kp[i, w].copy()
            if last is not None:
                pix[i, k] = last
    has = np.isfinite(pix[:, :, 0]) & sh[:, None]
    e = motion_energy(npz.with_suffix(".mp4"), pix, has, np.where(sh, width, np.nanmedian(width[sh])) * 0.5)
    p.parent.mkdir(exist_ok=True)
    np.save(p, e)
    return e


def lag(a, k):
    out = np.full_like(a, np.nan)
    if k == 0:
        return a.copy()
    out[k:] = a[:-k]
    return out


def build(npz, labelers=("reviewed", "reviewed_messy")):
    npz = Path(npz)
    z = np.load(npz, allow_pickle=False)
    t, kp, cf = z["timestamps"], z["keypoints"].astype(float), z["confidences"]
    n = len(t)
    dtv = np.r_[np.nan, np.diff(t)]
    dt0 = float(np.nanmedian(dtv))
    sh = (cf[:, L_SH] > KC) & (cf[:, R_SH] > KC)
    width = np.linalg.norm(kp[:, R_SH] - kp[:, L_SH], axis=1)
    sh &= width > 10
    mid = (kp[:, L_SH] + kp[:, R_SH]) / 2

    rel = np.full((n, 2, 2), np.nan)
    wok = np.zeros((n, 2), bool)
    for k, w in enumerate(WR):
        g = sh & (cf[:, w] > KC)
        wok[:, k] = g
        rel[g, k] = (kp[g, w] - mid[g]) / width[g, None]
    speed = np.full((n, 2), np.nan)
    ok2 = wok[1:] & wok[:-1]
    sp = np.linalg.norm(np.diff(rel, axis=0), axis=2) / dtv[1:, None]
    speed[1:][ok2] = sp[ok2]

    # rest position (median of calm past frames) and distance from it
    win = max(3, int(round(0.6 / dt0)))
    disp_rest = np.full((n, 2), np.nan)
    rest_pos = np.full((n, 2, 2), np.nan)
    for k in range(2):
        spk = np.nan_to_num(speed[:, k], nan=99)
        for i in range(n):
            if not wok[i, k]:
                continue
            lo = max(0, i - win)
            m = wok[lo:i, k] & (spk[lo:i] < 1.5)
            if m.sum() >= 3:
                rest_pos[i, k] = np.median(rel[lo:i, k][m], axis=0)
                disp_rest[i, k] = np.linalg.norm(rel[i, k] - rest_pos[i, k])

    # launch detector distance per wrist (0 outside its post-rest window)
    det = LaunchDetector(thresh=1e9)  # never fires; we only read last_disp
    launch = np.zeros((n, 2))
    for i in range(n):
        det.update(t[i], kp[i], cf[i])
        launch[i] = det.last_disp

    e = _energy(npz, kp, cf, sh, width)
    energy = np.full((n, 2), np.nan)
    for k in range(2):
        for i in range(n):
            v = e[max(0, i - win):i, k]
            v = v[np.isfinite(v)]
            if len(v) >= 3 and np.isfinite(e[i, k]):
                energy[i, k] = e[i, k] / (np.median(v) + 1e-3)

    fl = flow_features.load(npz)
    width_rate = np.full(n, np.nan)
    width_rate[2:] = (width[2:] - width[:-2]) / np.maximum(width[:-2], 1) / (t[2:] - t[:-2])
    width_rate[~sh] = np.nan

    cols, names = [], []

    def add(name, a, lags=(0,)):
        for L in lags:
            cols.append(lag(a, L))
            names.append(f"{name}_l{L}" if len(lags) > 1 else name)

    add("disp_rest", disp_rest, (0, 1, 2))
    add("launch", launch, (0, 1, 2))
    add("speed", speed, (0, 1))
    add("relx", rel[:, :, 0])
    add("rely", rel[:, :, 1])
    add("energy", energy, (0, 1))
    add("div_rel", fl["div_rel"], (0, 1, 2))
    add("mag_rel", fl["mag_rel"], (0, 1))
    add("out_rel", fl["out_rel"], (0, 1))
    for name, a in (("torso_div", fl["torso_div"]), ("torso_mag", fl["torso_mag"]), ("width_rate", width_rate)):
        cols.append(np.repeat(a[:, None], 2, axis=1))
        names.append(name)
    cols.append(speed[:, ::-1].copy())
    names.append("other_speed")

    if EXTRA:
        # how long the fist had been resting, and frames since it last was at rest
        at_rest = wok & (np.nan_to_num(disp_rest, nan=9) < 0.15) & (np.nan_to_num(speed, nan=99) < 2.25)
        rest_run = np.zeros((n, 2))
        since_rest = np.full((n, 2), 30.0)
        for k in range(2):
            run, last_rest_run, since = 0, 0, 30
            for i in range(n):
                rest_run[i, k] = last_rest_run   # length of the most recent rest spell
                since_rest[i, k] = min(since, 30)
                if at_rest[i, k]:
                    run += 1
                    since = 0
                else:
                    if run:
                        last_rest_run = run
                    run = 0
                    since += 1
        add("rest_run", np.minimum(rest_run, 30))
        add("since_rest", since_rest)
        add("energy2", energy, (2,))
        add("mag_rel2", fl["mag_rel"], (2,))
        # both hands moving together (glove adjust, body moves) and head motion (head turns)
        both = np.minimum(np.nan_to_num(speed[:, 0], nan=0), np.nan_to_num(speed[:, 1], nan=0))
        cols.append(np.repeat(both[:, None], 2, axis=1))
        names.append("both_speed")
        nose_ok = sh & (cf[:, 0] > KC)
        nrel = np.full((n, 2), np.nan)
        nrel[nose_ok] = (kp[nose_ok, 0] - mid[nose_ok]) / width[nose_ok, None]
        nspeed = np.full(n, np.nan)
        nspeed[1:] = np.linalg.norm(np.diff(nrel, axis=0), axis=1) / dtv[1:]
        cols.append(np.repeat(nspeed[:, None], 2, axis=1))
        names.append("nose_speed")
    # relx sign flips meaning between wrists; give the model which wrist it is
    cols.append(np.tile(np.array([0.0, 1.0]), (n, 1)))
    names.append("wrist_id")
    X = np.stack(cols, axis=2)  # (n, 2, F)

    # ---------------- labels ----------------
    y = np.full((n, 2), -1, int)
    saved = json.loads(npz.with_suffix(".labels.json").read_text())["labelers"]
    lab = [m for name in labelers for m in saved.get(name, [])]
    punches = sorted((m for m in lab if m["family"] not in NEGATIVES), key=lambda m: int(m["departure"]))
    cands = json.loads(npz.with_suffix(".auto_candidates.json").read_text())["candidates"]
    by_dep = {c["departure"]: c for c in cands}
    events = []
    for m in punches:
        d = int(m["departure"])
        c = by_dep.get(d) or min(cands, key=lambda c: abs(c["departure"] - d))
        k = 0 if c["wrist"] == "image-left" else 1   # auto_label: k=0 -> kp 9
        peak = int(c["peak"]) if abs(c["departure"] - d) <= 3 else d + max(1, int(round(4 * REF_DT / dt0)))
        events.append(dict(departure=d, wrist=k, peak=peak, family=m["family"], time=float(t[d])))

    calm_ok = np.ones(n, bool)
    for c in cands:
        calm_ok &= np.abs(t - t[c["departure"]]) > 1.0
    for ev in events:
        calm_ok &= np.abs(t - ev["time"]) > 1.0
    y[calm_ok & sh] = 0

    scripted_neg = np.zeros(n, bool)
    feint = np.zeros(n, bool)
    if "commands" in z.files and len(z["commands"]):
        cmd, ct = z["commands"], z["command_times"]
        for j, cname in enumerate(cmd):
            cname = str(cname)
            if cname not in NEGATIVES:
                continue
            end = ct[j + 1] if j + 1 < len(ct) else t[-1]
            w = (t >= ct[j]) & (t < end)
            if cname == "FEINT":
                feint |= w
            else:
                scripted_neg |= w
        for ev in events:
            far = np.abs(t - ev["time"]) > 1.0
            scripted_neg &= far
            feint &= far
        y[scripted_neg & sh] = 0
        y[feint] = -1
        # A punch prompt means a punch happened in that window even when no reviewed
        # label sits there (auto-labeler missed it, or the review was rejected/unfinished).
        # Don't let those frames count as "calm" negatives.
        for j, cname in enumerate(cmd):
            cname = str(cname)
            if cname in NEGATIVES:
                continue
            end = ct[j + 1] if j + 1 < len(ct) else t[-1]
            w = (t >= ct[j]) & (t < end)
            blk = y[w]
            blk[blk == 0] = -1
            y[w] = blk

    # Windows below were tuned in frames at ~20.8 fps (48 ms/frame). Scale them so
    # they cover the same TIME at other frame rates (e.g. 30 fps sessions).
    fs = REF_DT / dt0
    def f(k):
        return max(1, int(round(k * fs)))
    for ev in events:
        d, k, p = ev["departure"], ev["wrist"], ev["peak"]
        lo, hi = max(0, d - f(3)), min(n, d + f(13))
        y[lo:hi, :] = -1
        y[p + f(2):min(n, p + f(11)), k] = 0      # retraction of the punching wrist
        y[d:min(n, d + f(POS_FRAMES)), k] = 1      # first ~190 ms

    return dict(t=t, X=X, y=y, names=names, events=events, scripted_neg=scripted_neg,
                feint=feint, sh=sh, launch=launch, name=npz.name, rel=rel, rest_pos=rest_pos,
                u_rel=fl["u_rel"], v_rel=fl["v_rel"])
