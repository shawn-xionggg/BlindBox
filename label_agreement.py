"""
Do two reviewers label punches the same way? Compares a blind second review
(review_labels.py --reviewer NAME) with the main review, candidate by candidate.

  python label_agreement.py data/session_X.npz --reviewer alice
  python label_agreement.py data/session_X.npz --reviewer alice --from 60 --to 180   # one clip (s)
  python label_agreement.py data/session_X.npz --reviewer bob --against alice      # two second reviewers

Only candidates BOTH reviewers decided are compared, so a partial second review
works. It prints:
  - decisions (punch / feint / no) side by side, % agreement and Cohen's kappa
    (agreement corrected for chance: 1 = perfect, 0 = no better than chance);
  - for candidates both called a punch: how far apart the two start frames are,
    and whether one reviewer marks systematically later;
  - every disagreement with its time in the session, to look at together.
Rule of thumb: kappa >= 0.8 and >= 90% of starts within 1 frame means the start
rule ("first frame moving TOWARD the target") is applied consistently.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from review_labels import review_candidates

KINDS = ("punch", "feint", "no")
AS_KIND = {"yes": "punch", "feint": "feint", "no": "no"}


def kappa(m):
    """Cohen's kappa of a square count matrix (rows: reviewer A, columns: reviewer B)."""
    n = m.sum()
    if not n:
        return float("nan")
    po = np.trace(m) / n
    pe = float((m.sum(0) * m.sum(1)).sum()) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def compare(npz, a, b, t_from=0., t_to=np.inf):
    """Pairs of decisions on the same candidates. a, b: labelers of the clean pass
    (their --messy pass is a + '_messy'). Returns (rows, notes)."""
    npz = Path(npz)
    all_cands = json.loads(npz.with_suffix(".auto_candidates.json").read_text())["candidates"]
    rows, notes = [], []
    for messy in (False, True):
        la, lb = (a + "_messy", b + "_messy") if messy else (a, b)
        pa, pb = npz.with_suffix(f".review_{la}.json"), npz.with_suffix(f".review_{lb}.json")
        name = "messy" if messy else "clean"
        if not (pa.exists() and pb.exists()):
            missing = ", ".join(p.name for p in (pa, pb) if not p.exists())
            notes.append(f"{name} pass: not compared (no {missing} yet)")
            continue
        sa, sb = json.loads(pa.read_text()), json.loads(pb.read_text())
        cands = review_candidates(all_cands, messy=messy)
        both, moved = 0, 0
        for i, c in enumerate(cands):
            x, y = sa.get(str(i), {}), sb.get(str(i), {})
            if not (x.get("decision") and y.get("decision")):
                continue
            # newer reviews store the candidate's own start ("auto"): it must match this list
            if any(s.get("auto", c["departure"]) != c["departure"] for s in (x, y)):
                moved += 1
                continue
            if not (t_from <= c["time"] < t_to):
                continue
            both += 1
            rows.append(dict(time=float(c["time"]), pass_=name, index=i, auto=int(c["departure"]),
                             a=AS_KIND[x["decision"]], b=AS_KIND[y["decision"]],
                             a_start=int(x["departure"]), b_start=int(y["departure"])))
        decided = (sum(1 for s in sa.values() if s.get("decision")), sum(1 for s in sb.values() if s.get("decision")))
        notes.append(f"{name} pass: {both} candidates decided by both in range "
                     f"({la} decided {decided[0]}, {lb} decided {decided[1]} of {len(cands)})")
        if moved:
            notes.append(f"  ! {moved} candidates skipped: the candidate list changed between the two reviews")
    return rows, notes


def report(npz, a, b, rows, notes, t_from=0., t_to=np.inf, show=30):
    npz = Path(npz)
    t = np.load(npz, allow_pickle=False)["timestamps"]
    ms = 1000 * float(np.median(np.diff(t)))
    clip = "" if (t_from, t_to) == (0., np.inf) else f"   clip {t_from:g}-{t_to:g} s"
    print(f"{npz.name}: {a} vs {b}{clip}")
    for n in notes:
        print("  " + n)
    if not rows:
        print("No candidate was decided by both reviewers here: nothing to compare.")
        return {}
    m = np.zeros((3, 3), int)
    for r in rows:
        m[KINDS.index(r["a"]), KINDS.index(r["b"])] += 1
    k = kappa(m)
    agree = int(np.trace(m))
    print(f"\nDecisions (rows {a}, columns {b})")
    print(f"{'':16s}" + "".join(f"{x:>8s}" for x in KINDS))
    for j, x in enumerate(KINDS):
        print(f"  {x:14s}" + "".join(f"{v:8d}" for v in m[j]))
    print(f"  agree {agree}/{len(rows)} ({100 * agree / len(rows):.0f}%), Cohen's kappa {k:.2f}")

    out = dict(n=len(rows), agree=agree, kappa=k)
    both = [r for r in rows if r["a"] == r["b"] == "punch"]
    if both:
        d = np.array([r["b_start"] - r["a_start"] for r in both])
        within1, within2 = int((np.abs(d) <= 1).sum()), int((np.abs(d) <= 2).sum())
        mean = float(d.mean())
        lean = "about the same" if abs(mean) < 0.5 else f"{b} marks {'later' if mean > 0 else 'earlier'}"
        print(f"\nStart frame where both said punch (n={len(both)}), {b} minus {a}:")
        print(f"  median |difference| {np.median(np.abs(d)):.0f} frames ({np.median(np.abs(d)) * ms:.0f} ms) | "
              f"within 1 frame {within1}/{len(both)} ({100 * within1 / len(both):.0f}%) | "
              f"within 2 frames {within2}/{len(both)} ({100 * within2 / len(both):.0f}%)")
        print(f"  mean difference {mean:+.1f} frames: {lean}")
        print(f"  moved from the computer's guess (median): {a} {np.median([r['a_start'] - r['auto'] for r in both]):+.0f}, "
              f"{b} {np.median([r['b_start'] - r['auto'] for r in both]):+.0f} frames")
        out.update(start_n=len(both), within1=within1, within2=within2, mean_diff=mean)

    bad = [r for r in rows if r["a"] != r["b"] or (r["a"] == "punch" and abs(r["b_start"] - r["a_start"]) > 1)]
    if bad:
        print(f"\nDisagreements ({len(bad)}; starts more than 1 frame apart count too). Look at these together:")
        for r in sorted(bad, key=lambda r: r["time"])[:show]:
            def say(kind, start):
                return f"{kind} at f{start}" if kind != "no" else "no"
            gap = f" ({r['b_start'] - r['a_start']:+d} frames)" if r["a"] == r["b"] == "punch" else ""
            print(f"  t={r['time']:8.2f}s  {r['pass_']} #{r['index']:<4d} {a}: {say(r['a'], r['a_start']):16s} "
                  f"{b}: {say(r['b'], r['b_start'])}{gap}")
        if len(bad) > show:
            print(f"  ... and {len(bad) - show} more")
    print("\nRule of thumb: kappa >= 0.8 and >= 90% of starts within 1 frame = the start rule is applied consistently.")
    out["disagreements"] = len(bad)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--reviewer", required=True, help="the second reviewer's name (review_labels.py --reviewer NAME)")
    ap.add_argument("--against", help="compare with this second reviewer instead of the main review")
    ap.add_argument("--from", dest="t_from", type=float, default=0., help="clip start, seconds into the session")
    ap.add_argument("--to", dest="t_to", type=float, default=np.inf, help="clip end, seconds into the session")
    args = ap.parse_args()
    a = f"check_{args.against}" if args.against else "reviewed"
    b = f"check_{args.reviewer}"
    rows, notes = compare(args.npz, a, b, args.t_from, args.t_to)
    report(args.npz, a, b, rows, notes, args.t_from, args.t_to)


if __name__ == "__main__":
    main()
