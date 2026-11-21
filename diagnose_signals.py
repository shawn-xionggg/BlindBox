"""
Diagnose punch detection on reviewed sessions.

Uses the human-reviewed departure frames (data/*.labels.json) as ground truth
and answers two questions:

 1. How often is the live tracker "blind" (UNKNOWN) right when a punch starts,
    and why (a wrist lost, a shoulder lost, or the tracking-jump rule)?
 2. For each candidate signal, how many punches does it catch within
    +1/+2/+3/+4 frames of departure, at thresholds that keep false alarms on
    scripted don't-punch footage (feints, steps, nods, head turns, glove
    adjusts, idle) at <= 1, 2 or 4 per minute?

Signals (all per wrist, max over both wrists, causal = past frames only):
  speed   wrist speed relative to the shoulders (shoulder widths / s), 1 frame
  speed2  same, but averaged over 2 frames
  glove   d/dt log(glove area / shoulder width^2)   (looming, cancels step-in)
  energy  pixel change around the wrist / its recent baseline
  launch  distance from the guard position the fist had been holding, only in
          the first ~0.3 s after leaving a held rest (ignores retractions and
          whole-body moves). Caveat: the auto-labeler also defines departure by
          leaving rest, so this signal is partly favoured by how labels were made.
  any     OR of speed, glove and energy at their own thresholds

  python diagnose_signals.py
"""
import json
from pathlib import Path

import numpy as np

from auto_label import motion_energy
from pose_features import NEGATIVES

L_SH, R_SH, WR = 5, 6, (9, 10)
KC = 0.35


def load(npz):
    z = np.load(npz, allow_pickle=False)
    lab = json.loads(npz.with_suffix(".labels.json").read_text())["labelers"].get("reviewed", [])
    punches = [m for m in lab if m["family"] not in NEGATIVES]
    return z, punches


def signals(npz, z):
    t, kp, cf = z["timestamps"], z["keypoints"].astype(float), z["confidences"]
    n = len(t)
    dtv = np.r_[np.nan, np.diff(t)]
    sh = (cf[:, L_SH] > KC) & (cf[:, R_SH] > KC)
    width = np.linalg.norm(kp[:, R_SH] - kp[:, L_SH], axis=1)
    sh &= width > 10
    mid = (kp[:, L_SH] + kp[:, R_SH]) / 2
    wok = np.stack([cf[:, w] > KC for w in WR], 1) & sh[:, None]

    # --- why the live tracker goes UNKNOWN (replicates pose_features.FeatureStream) ---
    valid_live = sh & wok[:, 0] & wok[:, 1]
    jump = np.zeros(n, bool)
    for i in range(1, n):
        if valid_live[i] and valid_live[i - 1]:
            jump[i] = (np.linalg.norm(mid[i] - mid[i - 1]) / width[i - 1] > 0.5
                       or abs(width[i] / width[i - 1] - 1) > 0.30)
    unknown = ~valid_live | jump
    reason = np.where(~sh, "shoulder lost", np.where(~valid_live, "a wrist lost", np.where(jump, "jump rule", "ok")))

    # --- per-wrist signals ---
    rel = np.full((n, 2, 2), np.nan)
    for k, w in enumerate(WR):
        g = wok[:, k]
        rel[g, k] = (kp[g, w] - mid[g]) / width[g, None]
    speed = np.full((n, 2), np.nan)
    speed[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=2) / dtv[1:, None]
    speed2 = np.full((n, 2), np.nan)
    speed2[2:] = np.linalg.norm(rel[2:] - rel[:-2], axis=2) / (t[2:] - t[:-2])[:, None]

    glove = np.full((n, 2), np.nan)
    if "glove_blobs" in z.files and np.isfinite(z["glove_blobs"][:, :, 2]).any():
        gb = z["glove_blobs"]
        la = np.full((n, 2), np.nan)
        for i in range(n):
            if not sh[i]:
                continue
            for b in gb[i]:
                if not np.isfinite(b).all() or b[2] <= 0:
                    continue
                # nearest wrist (or nearest shoulder side if the wrist is lost)
                ref = np.array([kp[i, w] if wok[i, k] else kp[i, L_SH if k == 0 else R_SH] for k, w in enumerate(WR)])
                k = int(np.argmin(np.linalg.norm(ref - b[:2], axis=1)))
                if np.isnan(la[i, k]):
                    la[i, k] = np.log(b[2] / width[i] ** 2)
        glove[1:] = np.diff(la, axis=0) / dtv[1:, None]

    pix = np.stack([kp[:, w] for w in WR], 1)
    # fill lost wrists with last known pixel position so energy still looks there
    for k in range(2):
        last = None
        for i in range(n):
            if wok[i, k]:
                last = pix[i, k].copy()
            elif last is not None:
                pix[i, k] = last
    has = np.isfinite(pix[:, :, 0]) & sh[:, None]
    e = motion_energy(npz.with_suffix(".mp4"), pix, has, np.where(sh, width, np.nanmedian(width)) * 0.5)
    energy = np.full((n, 2), np.nan)
    win = max(3, int(round(0.6 / np.nanmedian(dtv))))
    for k in range(2):
        for i in range(n):
            v = e[max(0, i - win):i, k]
            v = v[np.isfinite(v)]
            if len(v) >= 3 and np.isfinite(e[i, k]):
                energy[i, k] = e[i, k] / (np.median(v) + 1e-3)

    # --- launch-from-guard (causal): distance of the wrist from where it had been
    # resting, reported only in the first ~0.3 s after leaving a rest held >= 0.15 s.
    # Ignores retractions (no held rest right before) and whole-body moves (relative).
    dt0 = np.nanmedian(dtv)
    win_r = max(3, int(round(0.6 / dt0)))
    hold_n = max(2, int(round(0.15 / dt0)))
    after_n = max(2, int(round(0.30 / dt0)))
    launch = np.full((n, 2), np.nan)
    for k in range(2):
        r, g = rel[:, k], wok[:, k]
        sp = np.nan_to_num(speed[:, k], nan=99)
        rest = np.full((n, 2), np.nan)
        for i in range(n):
            lo = max(0, i - win_r)
            m = g[lo:i] & (sp[lo:i] < 1.5)
            if m.sum() >= 3:
                rest[i] = np.median(r[lo:i][m], axis=0)
        disp = np.linalg.norm(r - rest, axis=1)
        at_rest = g & (np.nan_to_num(disp, nan=9) < 0.15) & (sp < 2.25)
        held = np.zeros(n, bool)  # frame i ends a hold of >= hold_n rest frames
        run = 0
        for i in range(n):
            run = run + 1 if at_rest[i] else 0
            held[i] = run >= hold_n
        last_held = -10**9
        armed_rest = None
        for i in range(n):
            if held[i]:
                last_held = i
                armed_rest = rest[i].copy()
            if g[i] and armed_rest is not None and 0 < i - last_held <= after_n and not at_rest[i]:
                launch[i, k] = np.linalg.norm(r[i] - armed_rest)

    sig = {name: np.nanmax(np.nan_to_num(a, nan=-np.inf), axis=1) for name, a in
           (("speed", speed), ("speed2", speed2), ("glove", glove), ("energy", energy), ("launch", launch))}
    for v in sig.values():
        v[~np.isfinite(v)] = np.nan
    return t, sig, unknown, reason


def negative_mask(z, t, punches, dep_pad=1.0):
    if "commands" not in z.files or not len(z["commands"]):
        return None
    cmd, ct = z["commands"], z["command_times"]
    neg = np.zeros(len(t), bool)
    for j, c in enumerate(cmd):
        if str(c) in NEGATIVES:
            end = ct[j + 1] if j + 1 < len(ct) else t[-1]
            neg |= (t >= ct[j]) & (t < end)
    for m in punches:
        td = t[m["departure"]]
        neg &= np.abs(t - td) > dep_pad
    return neg


def false_per_min(x, neg, t, thr, cooldown=0.6):
    hot = neg & (np.nan_to_num(x, nan=-np.inf) > thr)
    count, last = 0, -1e9
    for i in np.flatnonzero(hot):
        if not hot[i - 1] if i else True:
            if t[i] - last > cooldown:
                count += 1
                last = t[i]
    minutes = neg.sum() * np.nanmedian(np.diff(t)) / 60
    return count / minutes if minutes else np.nan


def main():
    sessions = sorted(p for p in Path("data").glob("*.npz") if p.with_suffix(".labels.json").exists())
    all_rows, neg_sets = [], []
    for npz in sessions:
        z, punches = load(npz)
        print(f"\n=== {npz.name}: {len(punches)} reviewed punches ===")
        t, sig, unknown, reason = signals(npz, z)
        fps = 1 / np.median(np.diff(t))
        print(f"fps {fps:.1f} (1 frame = {1000 / fps:.0f} ms)")
        # 1) blindness around departures
        win_frames = [range(m["departure"], min(len(t), m["departure"] + 4)) for m in punches]
        blind = [np.mean(unknown[list(w)]) for w in win_frames]
        rs = {}
        for w in win_frames:
            for i in w:
                rs[reason[i]] = rs.get(reason[i], 0) + 1
        tot = sum(rs.values())
        print(f"live tracker UNKNOWN in the first 4 frames of a punch: {100 * np.mean(blind):.0f}% of frames "
              f"(vs {100 * unknown.mean():.0f}% overall)")
        print("  why: " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(rs.items(), key=lambda x: -x[1])))
        neg = negative_mask(z, t, punches)
        for m in punches:
            all_rows.append((npz.name, m, t, sig))
        if neg is not None:
            neg_sets.append((t, sig, neg))
            print(f"don't-punch footage: {neg.sum() / fps / 60:.1f} min")

    if not neg_sets:
        print("No scripted negatives found; cannot set thresholds.")
        return

    # 2) detection vs false alarms. Thresholds are chosen on negatives (scripted session only).
    print("\n=== catch rate by frames after departure, at fixed false-alarm budgets ===")
    print("(false alarms measured on scripted don't-punch footage; punches from all reviewed sessions)")
    names = ["speed", "speed2", "glove", "energy", "launch"]
    budgets = (1.0, 2.0, 4.0)
    chosen = {}
    for name in names:
        vals = np.concatenate([s[name][neg] for t, s, neg in neg_sets])
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            print(f"{name:7s} no data")
            continue
        cands = np.unique(np.quantile(vals, np.linspace(0.80, 0.9999, 300)))
        for b in budgets:
            thr = None
            for c in cands:  # smallest threshold meeting the budget on every scripted session
                if all(false_per_min(s[name], neg, t, c) <= b for t, s, neg in neg_sets):
                    thr = c
                    break
            if thr is None:
                thr = cands[-1]
            chosen[(name, b)] = thr
            catches = []
            for k in (1, 2, 3, 4):
                hit = 0
                usable = 0
                for _, m, t, s in all_rows:
                    x = s[name]
                    if not np.isfinite(x).any():
                        continue
                    usable += 1
                    d = m["departure"]
                    seg = x[d:min(len(x), d + k + 1)]  # departure frame d .. d+k
                    hit += bool(np.nanmax(np.nan_to_num(seg, nan=-np.inf)) > thr)
                catches.append(f"{100 * hit / max(usable, 1):3.0f}%")
            ms = 1000 * np.median(np.diff(neg_sets[0][0]))
            print(f"{name:7s} <= {b:.0f} false/min  thr {thr:7.2f} | caught by +1 {catches[0]}  +2 {catches[1]}  "
                  f"+3 {catches[2]}  +4 {catches[3]}   (1 frame ~{ms:.0f} ms, n={usable})")

    # OR-combination at the 1/min budget for each (budget split three ways)
    print("\n=== combined: speed OR glove OR energy (each at <= 1/3 of the budget) ===")
    for b in budgets:
        thr = {}
        for name in ("speed", "glove", "energy"):
            vals = np.concatenate([s[name][neg] for t, s, neg in neg_sets])
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            for c in np.unique(np.quantile(vals, np.linspace(0.80, 0.9999, 300))):
                if all(false_per_min(s[name], neg, t, c) <= b / 3 for t, s, neg in neg_sets):
                    thr[name] = c
                    break
        fa = max(false_per_min(np.nanmax(np.stack([np.where(s[nm] > thr[nm], 1.0, 0.0) for nm in thr]), axis=0),
                               neg, t, 0.5) for t, s, neg in neg_sets)
        out = []
        for k in (1, 2, 3, 4):
            hit = 0
            for _, m, t, s in all_rows:
                d = m["departure"]
                hit += any(np.nanmax(np.nan_to_num(s[nm][d:d + k + 1], nan=-np.inf)) > thr[nm] for nm in thr)
            out.append(f"{100 * hit / len(all_rows):3.0f}%")
        print(f"budget {b:.0f}/min (actual {fa:.2f}/min) | +1 {out[0]}  +2 {out[1]}  +3 {out[2]}  +4 {out[3]}  "
              f"thresholds {', '.join(f'{k}={v:.2f}' for k, v in thr.items())}")

    # which don't-punch moves trigger the launch signal at the 4/min threshold?
    if ("launch", 4.0) in chosen:
        thr = chosen[("launch", 4.0)]
        print(f"\n=== launch false alarms by command (threshold {thr:.2f}) ===")
        for npz in sessions:
            z, punches = load(npz)
            if "commands" not in z.files or not len(z["commands"]):
                continue
            t = next(r[2] for r in all_rows if r[0] == npz.name)
            s = next(r[3] for r in all_rows if r[0] == npz.name)
            neg = negative_mask(z, t, punches)
            cmd, ct = z["commands"], z["command_times"]
            per = {}
            for j, c in enumerate(cmd):
                if str(c) not in NEGATIVES:
                    continue
                end = ct[j + 1] if j + 1 < len(ct) else t[-1]
                w = neg & (t >= ct[j]) & (t < end)
                per.setdefault(str(c), [0, 0])
                per[str(c)][0] += int(np.any(np.nan_to_num(s["launch"][w], nan=-np.inf) > thr))
                per[str(c)][1] += 1
            for c, (hit, tot) in sorted(per.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
                print(f"  {c:12s} buzzed in {hit}/{tot} windows")

    # pre-departure: does anything rise BEFORE the fist leaves? (anticipation signal)
    print("\n=== before departure: share of punches where a signal already exceeds its 2/min threshold ===")
    for name in names:
        if (name, 2.0) not in chosen:
            continue
        thr = chosen[(name, 2.0)]
        pre = []
        for _, m, t, s in all_rows:
            d = m["departure"]
            x = s[name][max(0, d - 4):d]
            if np.isfinite(x).any():
                pre.append(np.nanmax(x) > thr)
        if pre:
            print(f"{name:7s} in the 4 frames before departure: {100 * np.mean(pre):.0f}%  (n={len(pre)})")


if __name__ == "__main__":
    main()
