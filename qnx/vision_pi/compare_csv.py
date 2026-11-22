"""
Compares the QNX vision program's per-frame CSV with Python's reference (make_test_frames.py).

  python qnx/vision_pi/compare_csv.py qnx/vision_pi/test_ref.csv pi.csv
"""
import csv
import sys

import numpy as np


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0].keys(), rows


def col(rows, c):
    return np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows])


def main(ref_path, pi_path):
    cols, a = load(ref_path)
    _, b = load(pi_path)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    print(f"{n} frames compared")
    worst = []
    for c in [c for c in cols if c.startswith("w")]:
        x, y = col(a, c), col(b, c)
        nan_mismatch = int((np.isnan(x) != np.isnan(y)).sum())
        m = ~np.isnan(x) & ~np.isnan(y)
        diff = np.abs(x[m] - y[m])
        rel = float((diff / np.maximum(np.abs(x[m]), 1e-3)).max()) if m.any() else 0.0
        worst.append((rel, c, float(diff.max()) if m.any() else 0.0, nan_mismatch))
    worst.sort(reverse=True)
    print("features, worst first (max relative diff, max abs diff, frames where only one side is NaN):")
    for rel, c, ab, nm in worst[:8]:
        print(f"  {c:22s} rel {rel:.2e}  abs {ab:.2e}  nan-mismatch {nm}")
    for k in (0, 1):
        d = np.abs(col(a, f"score{k}") - col(b, f"score{k}"))
        print(f"score{k}: max diff {np.nanmax(d):.2e}, mean {np.nanmean(d):.2e}")
    ea, eb = [r["event"] for r in a], [r["event"] for r in b]
    same = sum(1 for x, y in zip(ea, eb) if x and x == y)
    print(f"cues: python {sum(map(bool, ea))}, pi {sum(map(bool, eb))}, same frame + side {same}")
    for i, (x, y) in enumerate(zip(ea, eb)):
        if x != y:
            print(f"  frame {i}: python '{x}' vs pi '{y}'")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
