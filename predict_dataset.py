"""
Per-frame targets and features for punch PREDICTION: a warning before the fist
starts moving toward the target (the start rule is in review_labels.py).
Used by predict_trainer.py. Reuses the reviewed labels; changes no files.

Target for frame t:
   1  a reviewed punch starts within the next H s (default 0.3)
   0  no punch starts soon: don't-punch prompts (the BASELINE block is stored as
      IDLE, so it counts), wearer-motion sessions, and natural footage (free-boxing
      blocks, unscripted sessions) between punches
  -1  ignored, neither trained on nor counted:
      - each punch from its start to its peak (the fist is already moving toward the
        target: that is the reactive detector's job)
      - H to H+U s before a start (a warning that early isn't wrong; U default 0.3)
      - around feints (key f in review) and during FEINT prompts
      - prompts that ask for a punch: the puncher reacts to the screen (a fake cue)
      - frames where nobody is tracked
Punches are the reviewed marks (labelers reviewed and reviewed_messy) with
duplicates merged: starts < 0.12 s apart, or < 0.25 s apart inside one
single-punch prompt. A follow-up is a punch starting <= 1 s after another punch.

Features use frame i and earlier only, so they could run live:
  body     body-relative pose (nose, elbows, wrists, hips, shoulder angle and width,
           elbow angles, each fist's distance from its guard spot, optical flow
           around each fist if cached), each as (value - calm median) / calm spread
           of THIS person, at 0, 0.1, 0.2, 0.35, 0.6 and 1.0 s ago, plus its
           spread over the last 1 s
  rhythm   bounce of the shoulders: size over the last 1 s and 0.33 s, their ratio,
           phase and period
  context  label-free punch-like launches (combo_candidates.find_launches), used only
           once a live system could know them (2 frames after the movement ends, at
           least 0.15 s after it starts): time since the last one (any fist, each fist),
           how many in the last 2 s, the gap between the last two, and whether each
           fist is coming back
The calm baseline uses only frames where both fists rest near their guard spots,
from the session start (the BASELINE block in --free sessions) until 10 s of calm
frames are collected. It needs no labels, so a live system could calibrate the
same way. With --stance southpaw recorded, the body is mirrored so the lead hand
always has the same features.
"""
import json
import warnings
from pathlib import Path

import numpy as np

import combo_candidates as cc
from free_blocks import FREE_COMMAND
from pose_features import NEGATIVES

H, U = 0.3, 0.3            # target horizon; ignored band before it (s)
FOLLOW = 1.0               # a punch within this long after another is a follow-up (s)
LAGS = (0., .1, .2, .35, .6, 1.)
CONFIRM = 0.15             # launches become known this long after they start (s)
CALIBRATE = 10.            # seconds of calm frames that make the baseline
KC = 0.35
SINGLE_PUNCH = ("JAB", "CROSS", "HOOK_L", "HOOK_R", "UPPERCUT")
NATURAL, DONT, PROMPTED, FEINTW, OTHER = range(5)   # kinds of footage
JOINTS = (("nose", 0), ("elbow0", 7), ("elbow1", 8), ("wrist0", 9), ("wrist1", 10), ("hip0", 11), ("hip1", 12))
PAIRS = (("elbow0", "elbow1"), ("wrist0", "wrist1"), ("hip0", "hip1"), ("elbow_angle0", "elbow_angle1"),
         ("guard0", "guard1"), ("mag_rel0", "mag_rel1"), ("out_rel0", "out_rel1"), ("div_rel0", "div_rel1"))


def footage_kinds(t, commands, command_times):
    """Kind of footage per frame, from the prompt list (no prompts = unscripted = natural)."""
    kind = np.full(len(t), NATURAL if not len(commands) else OTHER, int)
    for j, c in enumerate(commands):
        c = str(c)
        lo = command_times[j]
        hi = command_times[j + 1] if j + 1 < len(commands) else np.inf
        w = (t >= lo) & (t < hi)
        if c == FREE_COMMAND:
            kind[w] = NATURAL
        elif c == "FEINT":
            kind[w] = FEINTW
        elif c in NEGATIVES:
            kind[w] = DONT
        elif c in SINGLE_PUNCH:
            kind[w] = PROMPTED
    return kind


def load(npz):
    """One reviewed session: keypoints, footage kinds, punches (deduplicated) and feints."""
    npz = Path(npz)
    with np.load(npz, allow_pickle=False) as z:
        s = dict(name=npz.stem, path=npz, t=z["timestamps"].astype(float),
                 kp=z["keypoints"].astype(float), cf=z["confidences"].astype(float),
                 commands=z["commands"] if "commands" in z.files else np.array([]),
                 command_times=z["command_times"] if "command_times" in z.files else np.array([]),
                 performer=str(z["performer"]) if "performer" in z.files else "unknown",
                 mount=str(z["mount"]) if "mount" in z.files else "desk",
                 stance=str(z["stance"]) if "stance" in z.files else "unknown")
    t = s["t"]
    s["kind"] = footage_kinds(t, s["commands"], s["command_times"])
    labels = json.loads(npz.with_suffix(".labels.json").read_text())["labelers"]
    cands = json.loads(npz.with_suffix(".auto_candidates.json").read_text())["candidates"]
    marks = [m for name in ("reviewed", "reviewed_messy") for m in labels.get(name, [])]
    s["feints"] = sorted(int(m["departure"]) for m in marks if m["family"] == "FEINT")
    starts = sorted(int(m["departure"]) for m in marks if m["family"] not in NEGATIVES)
    cmd_index = np.searchsorted(s["command_times"], t, side="right") - 1 if len(s["commands"]) else np.zeros(len(t), int)
    punches, merged = [], 0
    for d in starts:
        if punches:
            p = punches[-1]["frame"]
            gap = t[d] - t[p]
            one_prompt = (len(s["commands"]) and cmd_index[d] == cmd_index[p] and cmd_index[d] >= 0
                          and str(s["commands"][cmd_index[d]]) in SINGLE_PUNCH)
            if gap < 0.12 or (gap < 0.25 and one_prompt):
                merged += 1          # the same punch labeled twice: keep the earlier start
                continue
        punches.append(dict(frame=d))
    dt = float(np.median(np.diff(t)))
    cand_dep = np.array([c["departure"] for c in cands]) if cands else np.array([], int)
    for i, p in enumerate(punches):
        d = p["frame"]
        near = np.flatnonzero(np.abs(cand_dep - d) <= max(3, round(0.3 / dt))) if len(cand_dep) else []
        c = cands[int(near[np.argmin(np.abs(cand_dep[near] - d))])] if len(near) else None
        peak = int(c["peak"]) if c is not None and abs(c["departure"] - d) <= 3 else d + max(1, round(0.2 / dt))
        p.update(time=float(t[d]), peak=min(max(peak, d + 1), len(t) - 1),
                 first_motion=min(d, int(c["departure"])) if c is not None else d,
                 kind=int(s["kind"][d]),
                 follow=bool(i and t[d] - t[punches[i - 1]["frame"]] <= FOLLOW))
    s["punches"], s["merged"] = punches, merged
    return s


def targets(s, h=H, u=U, train_prompted=False):
    """1 / 0 / -1 per frame, as described at the top of this file."""
    t, kind = s["t"], s["kind"]
    sh = (s["cf"][:, 5] > KC) & (s["cf"][:, 6] > KC)
    y = np.full(len(t), -1, int)
    usable = (kind == NATURAL) | (kind == DONT)
    if train_prompted:
        usable |= kind == PROMPTED
    y[usable & sh] = 0
    if not train_prompted:
        y[kind == PROMPTED] = -1
    y[kind == FEINTW] = -1
    flying = np.zeros(len(t), bool)       # a fist already moving toward the target
    for p in s["punches"]:                # no frame near ANY punch counts as "no punch soon"
        ts = p["time"]
        y[(t >= ts - h - u) & (t < ts - h)] = -1
        flying[p["frame"]:p["peak"] + 1] = True
    y[flying] = -1
    for f in s["feints"]:
        y[(t >= t[f] - h - u) & (t <= t[f] + 0.4)] = -1
    for p in s["punches"]:
        if p["kind"] == PROMPTED and not train_prompted:
            continue
        ts = p["time"]
        y[(t >= ts - h) & (t < ts) & sh & ~flying] = 1
    return y


# ----------------------------------------------------------------------------- features
def _ema(x, t, tau, hold=0.2):
    """Causal exponential smoothing; holds the last value over gaps up to `hold` s."""
    out = np.full(len(x), np.nan)
    last, last_t = np.nan, -np.inf
    for i in range(len(x)):
        if np.isfinite(x[i]):
            if np.isfinite(last):
                a = 1 - np.exp(-(t[i] - last_t) / tau)
                last = last + a * (x[i] - last)
            else:
                last = x[i]
            last_t = t[i]
            out[i] = last
        elif t[i] - last_t <= hold:
            out[i] = last
    return out


def raw_channels(s, flow=None):
    """Per-frame channels (dict name -> (N,) with NaN where unknown), body-relative,
    mirrored for southpaws so channel 0 is always the lead side."""
    t, kp, cf = s["t"], s["kp"], s["cf"]
    sh = (cf[:, 5] > KC) & (cf[:, 6] > KC)
    width = np.linalg.norm(kp[:, 6] - kp[:, 5], axis=1)
    sh &= width > 10
    mid = (kp[:, 5] + kp[:, 6]) / 2
    ch = {}
    for name, j in JOINTS:
        ok = sh & (cf[:, j] > KC)
        rel = (kp[:, j] - mid) / np.maximum(width, 1)[:, None]
        ch[name + "_x"] = np.where(ok, rel[:, 0], np.nan)
        ch[name + "_y"] = np.where(ok, rel[:, 1], np.nan)
    d = kp[:, 6] - kp[:, 5]
    ch["shoulder_angle"] = np.where(sh, np.arctan2(d[:, 1], d[:, 0]), np.nan)
    ch["log_width"] = np.where(sh, np.log(np.maximum(width, 1)), np.nan)
    ch["mid_x"] = np.where(sh, mid[:, 0] / np.maximum(width, 1), np.nan)
    ch["mid_y"] = np.where(sh, mid[:, 1] / np.maximum(width, 1), np.nan)
    for k, (a, b, c) in enumerate(((5, 7, 9), (6, 8, 10))):
        ok = sh & (cf[:, b] > KC) & (cf[:, c] > KC)
        u, v = kp[:, a] - kp[:, b], kp[:, c] - kp[:, b]
        cos = (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-6)
        ch[f"elbow_angle{k}"] = np.where(ok, np.arccos(np.clip(cos, -1, 1)), np.nan)
    for k in range(2):
        r = np.stack([ch[f"wrist{k}_x"], ch[f"wrist{k}_y"]], 1)
        ok = np.isfinite(r).all(1)
        ch[f"guard{k}"] = np.linalg.norm(r - cc.guard_spot(t, r, ok), axis=1)
    if flow is not None:
        for name in ("mag_rel", "out_rel", "div_rel"):
            for k in range(2):
                ch[f"{name}{k}"] = flow[name][:, k].astype(float)
    if s["stance"] == "southpaw":
        for name in [n for n in ch if n.endswith("_x")] + ["shoulder_angle", "mid_x"]:
            ch[name] = -ch[name]
        for a, b in PAIRS:
            for suffix in ("_x", "_y", ""):
                if a + suffix in ch and b + suffix in ch:
                    ch[a + suffix], ch[b + suffix] = ch[b + suffix], ch[a + suffix]
    return ch


def calm_frames(s, ch):
    """Label-free calm: both fists tracked, near their guard spots and slow."""
    t = s["t"]
    calm = np.ones(len(t), bool)
    for k in range(2):
        x, y = ch[f"wrist{k}_x"], ch[f"wrist{k}_y"]
        speed = np.full(len(t), np.inf)
        speed[1:] = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-3)
        calm &= np.nan_to_num(ch[f"guard{k}"], nan=9) < 0.15
        calm &= np.nan_to_num(speed, nan=np.inf) < 1.0
    return calm


def baseline(t, M, calm, seconds=CALIBRATE, latest=60., step=0.5):
    """Per-frame calm median and spread of each column of M, from calm frames up to that
    frame (updated every `step` s), frozen once `seconds` of calm frames are collected or
    `latest` s have passed. NaN until 30 calm frames exist."""
    n, c = M.shape
    med, spread = np.full((n, c), np.nan), np.full((n, c), np.nan)
    calm_time = np.cumsum(np.where(calm, np.r_[0, np.diff(t)], 0.))
    enough = int(np.searchsorted(np.cumsum(calm), 30))           # first frame with 30 calm frames
    if enough >= n:
        return med, spread                                       # never calm: features stay NaN
    freeze = max(enough, min(int(np.searchsorted(calm_time, seconds)), int(np.searchsorted(t, t[0] + latest))))
    freeze = min(freeze, n - 1)
    points = sorted(set(np.searchsorted(t, np.arange(t[enough], t[freeze], step)).tolist()) | {enough, freeze})
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns (e.g. hips never seen)
        for a, b in zip(points, points[1:] + [n]):
            rows = M[:a + 1][calm[:a + 1]]
            m = np.nanmedian(rows, axis=0)
            mad = np.nanmedian(np.abs(rows - m), axis=0)
            end = n if a == freeze else b
            med[a:end], spread[a:end] = m, 1.4826 * mad
    return med, spread


def _lag_index(t, lag):
    idx = np.searchsorted(t, t - lag + 1e-9, side="right") - 1
    bad = (idx < 0) | (t - t[np.maximum(idx, 0)] > lag + 0.1)
    return np.where(bad, -1, idx)


def _window_sd(t, x, seconds):
    """Causal standard deviation of x over the last `seconds` (NaN-aware)."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.)
    c1, c2, cn = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(v * v)], np.r_[0, np.cumsum(ok)]
    lo = np.searchsorted(t, t - seconds, side="left")
    hi = np.arange(1, len(t) + 1)
    n = cn[hi] - cn[lo]
    mean = (c1[hi] - c1[lo]) / np.maximum(n, 1)
    var = (c2[hi] - c2[lo]) / np.maximum(n, 1) - mean ** 2
    return np.where(n >= 3, np.sqrt(np.maximum(var, 0)), np.nan)


def features(s, flow=None):
    """(X (N, F) float32, names, groups) for one session."""
    t = s["t"]
    ch = raw_channels(s, flow)
    names_c = list(ch)
    M = np.stack([_ema(ch[c], t, 0.05) for c in names_c], 1)
    calm = calm_frames(s, ch)
    med, spread = baseline(t, M, calm)
    Z = (M - med) / (spread + 0.05)
    cols, names, groups = [], [], []
    lag_idx = {lag: _lag_index(t, lag) for lag in LAGS}
    for j, c in enumerate(names_c):
        for lag, idx in lag_idx.items():
            cols.append(np.where(idx >= 0, Z[np.maximum(idx, 0), j], np.nan))
            names.append(f"{c}@{lag:g}s")
            groups.append("body")
        cols.append(_window_sd(t, Z[:, j], 1.0))
        names.append(f"{c}_sd1s")
        groups.append("body")
    # rhythm: band-passed vertical shoulder position (0.4-3 Hz), in calm-spread units
    y = M[:, names_c.index("mid_y")]
    bp = _ema(y, t, 0.05) - _ema(y, t, 0.4)
    scale = spread[:, names_c.index("mid_y")] + 0.05
    amp1, amp3 = _window_sd(t, bp, 1.0), _window_sd(t, bp, 0.33)
    up = np.flatnonzero((bp[1:] > 0) & (bp[:-1] <= 0)) + 1          # upward zero crossings
    last_up = np.searchsorted(up, np.arange(len(t)), side="right") - 1
    period = np.full(len(t), np.nan)
    since = np.full(len(t), np.nan)
    has2 = last_up >= 1
    period[has2] = np.clip(t[up[last_up[has2]]] - t[up[last_up[has2] - 1]], 0.25, 1.5)
    has1 = last_up >= 0
    since[has1] = t[has1] - t[up[last_up[has1]]]
    phase = 2 * np.pi * since / np.where(np.isfinite(period), period, 0.5)
    for name, col in (("bounce_amp1s", amp1 / scale), ("bounce_amp033s", amp3 / scale),
                      ("bounce_ratio", amp3 / (amp1 + 1e-3)), ("bounce_period", period),
                      ("bounce_phase_sin", np.sin(phase)), ("bounce_phase_cos", np.cos(phase))):
        cols.append(col)
        names.append(name)
        groups.append("rhythm")
    # context: launches found without labels. find_launches looks at the whole movement (and
    # smooths over +-1 frame), so a launch only counts as known 2 frames after its movement ends,
    # and never sooner than CONFIRM s after it starts: what a live system could know by then
    launches = cc.find_launches(t, s["kp"], s["cf"], mag=None if flow is None else flow["mag_rel"])
    known = [(max(t[e["start"]] + CONFIRM, t[min(e["end"] + 2, len(t) - 1)]), e["wrist"]) for e in launches]
    if s["stance"] == "southpaw":
        known = [(x, 1 - k) for x, k in known]
    kt = np.array(sorted(x for x, _ in known))

    def since_last(times):
        times = np.sort(np.asarray(times, float))
        if not len(times):
            return np.full(len(t), 3.)
        j = np.searchsorted(times, t, side="right") - 1
        return np.where(j >= 0, np.minimum(t - times[np.maximum(j, 0)], 3.), 3.)
    j = np.searchsorted(kt, t, side="right") - 1
    if len(kt) >= 2:
        prev_gap = np.where(j >= 1, np.minimum(kt[np.maximum(j, 1)] - kt[np.maximum(j - 1, 0)], 3.), 3.)
    else:
        prev_gap = np.full(len(t), 3.)
    ctx = {"ctx_since_launch": since_last(kt),
           "ctx_since_launch0": since_last([x for x, k in known if k == 0]),
           "ctx_since_launch1": since_last([x for x, k in known if k == 1]),
           "ctx_launches_2s": (j + 1 - np.searchsorted(kt, t - 2., side="left")).astype(float),
           "ctx_last_gap": prev_gap}
    idx01 = _lag_index(t, 0.1)
    for k in range(2):
        g = M[:, names_c.index(f"guard{k}")]
        back = g[np.maximum(idx01, 0)] - g
        ctx[f"ctx_coming_back{k}"] = np.where((idx01 >= 0) & np.isfinite(back), (back > 0.05).astype(float), np.nan)
    for name, col in ctx.items():
        cols.append(col)
        names.append(name)
        groups.append("context")
    X = np.stack(cols, 1).astype(np.float32)
    return X, names, groups
