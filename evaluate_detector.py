"""
Replay recorded sessions through the live LaunchDetector and score it against
the human-reviewed punch starts (data/*.labels.json).

For each threshold it reports:
  caught +k   share of punches whose buzz came within k frames of the fist
              leaving guard (1 frame ~48 ms at ~20 fps)
  median      median delay from fist leaving guard to buzz
  false/min   buzzes during scripted don't-punch footage (feints, steps, head
              turns, nods, glove adjusts, idle), more than 1 s from any punch
  feints      share of FEINT windows that buzzed (a feint is a punch start,
              so some buzzing there is expected)

  python evaluate_detector.py
  python evaluate_detector.py --thresh 0.4 0.55 0.7
"""
import argparse
import json
from pathlib import Path

import numpy as np

from launch_detector import LaunchDetector
from pose_features import NEGATIVES


def replay(z, **kw):
    det = LaunchDetector(**kw)
    out = []
    for t, k, c in zip(z["timestamps"], z["keypoints"], z["confidences"]):
        e = det.update(t, k, c)
        if e:
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresh", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.7, 0.9])
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    sessions = sorted(p for p in Path(args.data).glob("*.npz") if p.with_suffix(".labels.json").exists())
    loaded = []
    for npz in sessions:
        z = dict(np.load(npz, allow_pickle=False))
        lab = json.loads(npz.with_suffix(".labels.json").read_text())["labelers"].get("reviewed", [])
        punches = [m for m in lab if m["family"] not in NEGATIVES]
        loaded.append((npz.name, z, punches))
        print(f"{npz.name}: {len(punches)} reviewed punches, scripted={bool(len(z.get('commands', [])))}")

    print(f"\n{'thresh':>6} | {'caught +1':>9} {'+2':>5} {'+3':>5} {'+4':>5} {'+6':>5} | {'median':>7} | {'false/min':>9} | feints")
    for thr in args.thresh:
        delays, n_p = [], 0
        caught = {k: 0 for k in (1, 2, 3, 4, 6)}
        false, neg_s, feint_hit, feint_n = 0, 0.0, 0, 0
        for name, z, punches in loaded:
            t = z["timestamps"]
            ev = replay(z, thresh=thr)
            et = np.array([e["detected_time"] for e in ev])
            used = set()
            for m in punches:
                n_p += 1
                d = m["departure"]
                lo = t[max(0, d - 2)]
                hi = t[min(len(t) - 1, d + 6)]
                idx = [i for i in np.flatnonzero((et >= lo) & (et <= hi)) if i not in used]
                if not idx:
                    continue
                i = idx[0]
                used.add(i)
                delays.append(1000 * (et[i] - t[d]))
                for k in caught:
                    if et[i] <= t[min(len(t) - 1, d + k)]:
                        caught[k] += 1
            if len(z.get("commands", [])):
                cmd, ct = z["commands"], z["command_times"]
                dts = [t[m["departure"]] for m in punches]
                dt = np.median(np.diff(t))
                for j, c in enumerate(cmd):
                    c = str(c)
                    if c not in NEGATIVES:
                        continue
                    end = ct[j + 1] if j + 1 < len(ct) else t[-1]
                    w = (t >= ct[j]) & (t < end)
                    for td in dts:
                        w &= np.abs(t - td) > 1.0
                    neg_s += w.sum() * dt
                    hits = [x for x in et if ct[j] <= x < end and all(abs(x - td) > 1.0 for td in dts)]
                    if c == "FEINT":
                        feint_n += 1
                        feint_hit += bool(hits)
                    false += len(hits)
        fpm = false / (neg_s / 60) if neg_s else float("nan")
        pc = " ".join(f"{100 * caught[k] / max(n_p, 1):4.0f}%" for k in (1, 2, 3, 4, 6))
        med = f"{np.median(delays):5.0f}ms" if delays else "    -  "
        print(f"{thr:6.2f} | {pc.split()[0]:>9} {' '.join(f'{x:>5}' for x in pc.split()[1:])} | {med:>7} | "
              f"{fpm:9.2f} | {feint_hit}/{feint_n}")


if __name__ == "__main__":
    main()
