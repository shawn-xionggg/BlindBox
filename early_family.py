"""
Which way should the dodger move? Classify each detected punch EARLY into the
cue that matters:
    HOOK_L   fist swings in from image-left   -> DODGE RIGHT
    HOOK_R   fist swings in from image-right  -> DODGE LEFT
    STRAIGHT jab / cross                      -> DUCK
    UPPERCUT                                  -> BACK
using only what is known +f frames after the fist leaves guard (f = 1..4).

Compares:
  rule     the hand-made rule in early_live.py (sideways motion -> hook from
           the side the fist rested on, else STRAIGHT)
  wristside  hook from the side the punching fist rested on, always (no straights)
  learned  gradient boosting on the detector features + movement from rest,
           same contiguous-block cross-validation as early_detector.py

  python early_family.py           # evaluate + save models/early_family.joblib
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import early_dataset

GROUP = {"JAB": "STRAIGHT", "CROSS": "STRAIGHT", "HOOK_L": "HOOK_L", "HOOK_R": "HOOK_R", "UPPERCUT": "UPPERCUT"}
CUE = {"HOOK_L": "DODGE RIGHT", "HOOK_R": "DODGE LEFT", "STRAIGHT": "DUCK", "UPPERCUT": "BACK"}


def sample(s, ev, f):
    """Feature vector for punch ev, f frames after departure."""
    i = min(len(s["t"]) - 1, ev["departure"] + f)
    k = ev["wrist"]
    x = list(s["X"][i, k])
    rest = s["rest_pos"][ev["departure"], k]
    for L in (0, 1, 2):
        j = max(0, i - L)
        mv = s["rel"][j, k] - rest
        x += [mv[0], mv[1]]
    x += [rest[0], rest[1]]
    for L in (0, 1, 2):   # flow direction around the fist (body motion removed)
        j = max(0, i - L)
        x += [s["u_rel"][j, k], s["v_rel"][j, k]]
    return np.array(x, float)


def live_wrist(s, ev, f):
    """The wrist a live system would pick at +f frames: most flow motion right now."""
    i = min(len(s["t"]) - 1, ev["departure"] + f)
    m = s["X"][i, :, s["names"].index("mag_rel_l0")]
    return int(np.nanargmax(m)) if np.isfinite(m).any() else ev["wrist"]


def rule(s, ev, f):
    i = min(len(s["t"]) - 1, ev["departure"] + f)
    k = ev["wrist"]
    rest = s["rest_pos"][ev["departure"], k]
    mv = s["rel"][i, k] - rest
    if np.all(np.isfinite(mv)) and np.all(np.isfinite(rest)) and abs(mv[0]) > 1.3 * abs(mv[1]) and np.linalg.norm(mv) > 0.15:
        return "HOOK_L" if rest[0] < 0 else "HOOK_R"
    return "STRAIGHT"


def wristside(s, ev, f):
    rest = s["rest_pos"][ev["departure"], ev["wrist"]]
    if not np.all(np.isfinite(rest)):
        return "STRAIGHT"
    return "HOOK_L" if rest[0] < 0 else "HOOK_R"


def motion_truth(s, ev, horizon=6, lateral=0.3):
    """Self-supervised direction from what the fist physically did: the wrist
    that moved farthest from rest by +horizon frames, and its sideways travel."""
    d = ev["departure"]
    i = min(len(s["t"]) - 1, d + horizon)
    best_k, best_disp = None, -1
    for k in range(2):
        mv = s["rel"][i, k] - s["rest_pos"][d, k]
        if np.all(np.isfinite(mv)) and np.linalg.norm(mv) > best_disp:
            best_k, best_disp = k, float(np.linalg.norm(mv))
    if best_k is None:
        return None, None
    dx = s["rel"][i, best_k][0] - s["rest_pos"][d, best_k][0]
    lab = "LEFTWARD" if dx < -lateral else "RIGHTWARD" if dx > lateral else "STRAIGHT"
    return lab, best_k


def evaluate_motion(sessions, items):
    """Can the eventual sideways direction be predicted from the first frames?"""
    rows = []
    for si, b, ev in items:
        lab, k = motion_truth(sessions[si], ev)
        if lab is None:
            continue
        rows.append((si, b, dict(ev, wrist=k), lab))
    labs = np.array([r[3] for r in rows])
    blocks = sorted({r[1] for r in rows})
    u, c = np.unique(labs, return_counts=True)
    print(f"\n=== direction from the fist's own motion (no labels): {dict(zip(u, c))} ===")
    print(f"majority-class baseline {100 * c.max() / c.sum():.0f}%")
    print(f"{'frames after leaving guard':28s} | {'hindsight wrist':>15} | {'live wrist choice':>17} | {'live wrist picked right':>23}")
    for f in (1, 2, 3):
        accs = []
        for mode in ("hindsight", "live"):
            evs = [ev if mode == "hindsight" else dict(ev, wrist=live_wrist(sessions[si], ev, f))
                   for si, _, ev, _ in rows]
            X = np.stack([sample(sessions[si], ev, f) for (si, _, _, _), ev in zip(rows, evs)])
            pred = np.empty(len(rows), object)
            for b in blocks:
                te = np.array([r[1] == b for r in rows])
                clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=7,
                                                     min_samples_leaf=5, l2_regularization=1.0,
                                                     class_weight="balanced", random_state=0)
                clf.fit(X[~te], labs[~te])
                pred[te] = clf.predict(X[te])
            accs.append(100 * np.mean(pred == labs))
            if mode == "live":
                same = 100 * np.mean([e["wrist"] == r[2]["wrist"] for e, r in zip(evs, rows)])
        print(f"+{f} frames (~{48 * f} ms){'':14s} | {accs[0]:14.0f}% | {accs[1]:16.0f}% | {same:22.0f}%")
    return rows, labs


def main():
    paths = sorted(p for p in Path("data").glob("*.npz") if p.with_suffix(".labels.json").exists()
                   and p.with_suffix(".auto_candidates.json").exists())
    sessions = [early_dataset.build(p) for p in paths]
    items = []  # (session index, block id, event)
    for si, s in enumerate(sessions):
        nb = 4 if s["scripted_neg"].any() else 2
        edges = np.linspace(0, len(s["t"]), nb + 1).astype(int)
        for ev in s["events"]:
            if ev["family"] not in GROUP:
                continue
            b = int(np.searchsorted(edges, ev["departure"], side="right") - 1)
            items.append((si, (si, b), ev))
    labels = np.array([GROUP[ev["family"]] for _, _, ev in items])
    blocks = sorted({b for _, b, _ in items})
    print(f"{len(items)} punches: " + ", ".join(f"{g} {int((labels == g).sum())}" for g in CUE))
    print(f"\n{'frames after leaving guard':28s} | {'rule':>6} | {'wristside':>9} | {'learned':>7}")
    best = None
    for f in (1, 2, 3, 4):
        X = np.stack([sample(sessions[si], ev, f) for si, _, ev in items])
        pred = np.empty(len(items), object)
        for b in blocks:
            te = np.array([bb == b for _, bb, _ in items])
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=7,
                                                 min_samples_leaf=5, l2_regularization=1.0,
                                                 class_weight="balanced", random_state=0)
            clf.fit(X[~te], labels[~te])
            pred[te] = clf.predict(X[te])
        acc_l = np.mean(pred == labels)
        acc_r = np.mean([rule(sessions[si], ev, f) == lab for (si, _, ev), lab in zip(items, labels)])
        acc_w = np.mean([wristside(sessions[si], ev, f) == lab for (si, _, ev), lab in zip(items, labels)])
        print(f"+{f} frames (~{48 * f} ms){'':14s} | {100 * acc_r:5.0f}% | {100 * acc_w:8.0f}% | {100 * acc_l:6.0f}%")
        if f == 2:
            best = (X, pred)
    X, pred = best
    print("\nlearned, +2 frames, confusion (rows = truth):")
    groups = list(CUE)
    print(f"{'':10s}" + "".join(f"{g:>10s}" for g in groups))
    for g in groups:
        row = [int(np.sum((labels == g) & (pred == h))) for h in groups]
        print(f"{g:10s}" + "".join(f"{v:10d}" for v in row))
    evaluate_motion(sessions, items)
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=7, min_samples_leaf=5,
                                         l2_regularization=1.0, class_weight="balanced", random_state=0)
    clf.fit(X, labels)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(dict(model=clf, frames_after=2, cue=CUE), "models/early_family.joblib")
    print("\nsaved models/early_family.joblib (classify at +2 frames)")


if __name__ == "__main__":
    main()
