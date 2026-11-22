"""Quality check for recorded sessions: length, frame rate, how much of the time the
puncher was tracked (per block), and why tracking was lost.

  python check_session.py data/session_X.npz [more.npz ...]
  python check_session.py --latest          # the newest session in data/

How to read it:
  tracked     share of frames with both shoulders seen. Training can only use those.
              Aim for 95%+. Below ~85%: the puncher was off-center or too close.
  fps         typical (median) and average camera frame rate. Aim for ~30.
              Below ~22: add light, or the laptop is too slow (plug it in, Best performance).
  lost gaps   < 0.3 s = blur from fast moves (normal); 0.3-2 s = leaning in / crashing
              forward; > 2 s = standing at the edge or out of the picture.
"""
import argparse
from pathlib import Path

import numpy as np


def check(path):
    with np.load(path, allow_pickle=False) as z:
        t, c, xy = z["timestamps"], z["confidences"], z["keypoints"]
        blocks = [str(b) for b in z["blocks"]] if "blocks" in z.files else []
        block_times = list(z["block_times"]) if "block_times" in z.files else []
        performer = str(z["performer"]) if "performer" in z.files else "?"
        stance = str(z["stance"]) if "stance" in z.files else "?"
    if len(t) < 2:
        return f"{Path(path).stem}: empty"
    ok = (np.minimum(c[:, 5], c[:, 6]) >= .35) & np.isfinite(xy[:, [5, 6]]).all((1, 2))
    minutes = (t[-1] - t[0]) / 60
    median_fps = 1 / np.median(np.diff(t))
    mean_fps = (len(t) - 1) / (t[-1] - t[0])
    per, free_s = {}, 0.
    ends = block_times[1:] + [t[-1] + 1]
    for kind, start, end in zip(blocks, block_times, ends):
        m = (t >= start) & (t < end)
        per.setdefault(kind, np.zeros(len(t), bool))
        per[kind] |= m
        if kind == "FREE":
            free_s += min(end, t[-1]) - start
    parts = " ".join(f"{k} {ok[m].mean():.0%}" for k, m in per.items() if m.any())
    gaps, i = [], 0
    while i < len(ok):
        if not ok[i]:
            j = i
            while j < len(ok) and not ok[j]:
                j += 1
            gaps.append(t[min(j, len(t) - 1)] - t[i])
            i = j
        else:
            i += 1
    g = np.asarray(gaps)
    lost = g.sum()
    share = lambda lo, hi: g[(g >= lo) & (g < hi)].sum() / lost if lost else 0.
    verdict = "GOOD" if ok.mean() >= .95 and median_fps >= 27 else "OK" if ok.mean() >= .85 and median_fps >= 20 else "CHECK"
    return (f"{Path(path).stem}  [{verdict}]  {performer} ({stance})\n"
            f"  {minutes:.1f} min, {free_s / 60:.1f} min free boxing, fps {median_fps:.0f} typical / {mean_fps:.0f} average\n"
            f"  tracked {ok.mean():.0%}  ({parts or 'no blocks'})\n"
            f"  lost {lost:.0f} s: <0.3 s {share(0, .3):.0%}, 0.3-2 s {share(.3, 2):.0%}, >2 s {share(2, 1e9):.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="*")
    ap.add_argument("--latest", action="store_true", help="check the newest session in data/")
    args = ap.parse_args()
    paths = list(args.npz)
    if args.latest:
        found = sorted(Path("data").glob("session_*.npz"), key=lambda p: p.stat().st_mtime)
        paths += [str(found[-1])] if found else []
    if not paths:
        ap.error("give a session .npz or --latest")
    for p in paths:
        print(check(p))


if __name__ == "__main__":
    main()
