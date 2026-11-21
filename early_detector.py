"""
Train and honestly evaluate a learned EARLY punch detector.

Cross-validation: each session is cut into contiguous time blocks
(session with scripted negatives: 4 blocks, others: 2). Each block is held out
in turn. The alarm threshold is chosen on the TRAINING blocks only, to meet a
false-alarm budget on their scripted don't-punch footage. Then the held-out
block is scored with the same streaming rule as the live app (score > threshold,
0.6 s cooldown). The hand-made LaunchDetector baseline gets the identical
treatment (its threshold is also picked on training blocks), so the comparison is fair.

Reported per budget:
  caught +k   punches buzzed within k frames of the fist leaving guard (1 frame ~48 ms)
  median      median delay, fist leaving guard -> buzz
  false/min   buzzes in held-out scripted don't-punch footage (not feints)
  feint       share of held-out FEINT windows that buzzed (expected: feints are punch starts)

  python early_detector.py            # evaluate + save models/early_detector.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import early_dataset

COOLDOWN = 0.6
KS = (50, 100, 150, 200, 300)   # ms after the fist leaves guard (~1, 2, 3, 4, 6 frames at 20.8 fps)


def blocks_for(sess, n_blocks):
    n = len(sess["t"])
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(n_blocks)]


def events_from_scores(t, score, thr):
    ev, last = [], -1e9
    for i in np.flatnonzero(score > thr):
        if t[i] - last >= COOLDOWN:
            ev.append(i)
            last = t[i]
    return ev


def score_events(sess, lo, hi, ev_idx):
    """Match buzzes to punches inside [lo, hi). Returns per-punch delays (frames, or None),
    false alarm count, scripted-negative seconds, feint windows hit/total."""
    t = sess["t"]
    ev_idx = [i for i in ev_idx if lo <= i < hi]
    used, res = set(), []
    for e in sess["events"]:
        d = e["departure"]
        if not (lo <= d < hi):
            continue
        f = early_dataset.REF_DT / float(np.median(np.diff(t)))   # same TIME window at any frame rate
        cand = [i for i in ev_idx if d - round(2 * f) <= i <= d + round(6 * f) and i not in used]
        if cand:
            used.add(cand[0])
            res.append(cand[0] - d)
        else:
            res.append(None)
    neg = sess["scripted_neg"][lo:hi]
    dt = np.median(np.diff(t))
    neg_s = neg.sum() * dt
    false = sum(1 for i in ev_idx if sess["scripted_neg"][i])
    # feint windows: contiguous runs of the feint mask
    fm = sess["feint"].copy()
    fm[:lo] = False
    fm[hi:] = False
    runs, i = [], lo
    while i < hi:
        if fm[i]:
            j = i
            while j + 1 < hi and fm[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    fh = sum(any(a <= x <= b for x in ev_idx) for a, b in runs)
    return res, false, neg_s, fh, len(runs)


def threshold_for_budget(sessions_blocks, scorer, budget):
    """Smallest threshold whose false alarms/min on the given (sess, lo, hi) blocks <= budget."""
    vals = []
    for sess, lo, hi in sessions_blocks:
        s = scorer(sess)[lo:hi]
        m = sess["scripted_neg"][lo:hi]
        vals.append(s[m])
    vals = np.concatenate(vals) if vals else np.array([])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.inf
    cands = np.unique(np.quantile(vals, np.linspace(0.9, 1.0, 400)))
    for thr in cands:
        false, secs = 0, 0.0
        for sess, lo, hi in sessions_blocks:
            ev = events_from_scores(sess["t"], scorer(sess), thr)
            _, f, ns, _, _ = score_events(sess, lo, hi, ev)
            false += f
            secs += ns
        if secs == 0 or false / (secs / 60) <= budget:
            return thr
    return cands[-1] + 1e-9


def summarize(label, delays, false, neg_s, fh, fn, ms):
    n = len(delays)
    got = [d for d in delays if d is not None]
    cs = " ".join(f"{100 * sum(d <= k for d in got) / max(n, 1):4.0f}%" for k in KS)
    med = f"{np.median(got):5.0f}ms" if got else "   -   "
    fpm = false / (neg_s / 60) if neg_s else float("nan")
    print(f"{label:28s} | {cs} | {med} | {fpm:6.2f} | {fh}/{fn}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--save", default="models/early_detector.joblib")
    ap.add_argument("--mirror", action="store_true", help="add left-right mirrored training copies")
    ap.add_argument("--smooth", type=int, default=1, help="average each wrist's score over this many frames")
    ap.add_argument("--no-flow", action="store_true", help="drop optical-flow features (plain-C portability test)")
    ap.add_argument("--clean-train", action="store_true",
                    help="train only on clean-reviewed punches (still scored on all reviewed punches)")
    ap.add_argument("--aug-data", help="folder of augmented copies (augment_lighting.py); always scored, "
                    "held-out chunk by held-out chunk")
    ap.add_argument("--aug-train", action="store_true", help="also TRAIN on the augmented copies")
    ap.add_argument("--exit", action="store_true", help="also fire the fist-leaves-frame cue (exit_cue.py)")
    ap.add_argument("--pos-frames", type=int, default=early_dataset.POS_FRAMES,
                    help="frames after leaving guard labeled as 'early punch'")
    args = ap.parse_args()
    early_dataset.POS_FRAMES = args.pos_frames

    paths = sorted(p for p in Path(args.data).glob("*.npz") if p.with_suffix(".labels.json").exists()
                   and p.with_suffix(".auto_candidates.json").exists())
    sessions = []
    for p in paths:
        print("building", p.name)
        sessions.append(early_dataset.build(p))
    names = sessions[0]["names"]
    if args.clean_train:
        for p, s in zip(paths, sessions):
            s["y_train"] = early_dataset.build(p, labelers=("reviewed",))["y"]
    if args.no_flow:
        # drop optical-flow features (the only ones needing OpenCV's Farneback)
        keep = [j for j, nm in enumerate(names)
                if not nm.startswith(("div_rel", "mag_rel", "out_rel", "torso_div", "torso_mag"))]
        for s in sessions:
            s["X"] = s["X"][:, :, keep]
        names = [names[j] for j in keep]
        print(f"no-flow: {len(names)} features: {', '.join(names)}")
    ms = 1000 * np.median(np.diff(sessions[0]["t"]))

    # Augmented copies (other lighting / frame rate) of the same sessions. They're tied
    # to their source session by TIME, so when a chunk of a session is held out, the
    # same chunk of every copy is held out too (no peeking at the test footage).
    aug = {}
    if args.aug_data:
        import json as _json
        by_name = {s["name"]: si for si, s in enumerate(sessions)}
        for p in sorted(Path(args.aug_data).glob("*.npz")):
            info_p = p.with_suffix(".aug.json")
            if not info_p.exists():
                continue
            info = _json.loads(info_p.read_text())
            if info["source"] not in by_name:
                continue
            print("building", p.name)
            a = early_dataset.build(p)
            aug.setdefault(by_name[info["source"]], []).append((info["preset"], a))

    def tmap(si, a, lo, hi):
        """Same time range as sessions[si][lo:hi], in augmented copy a's frames."""
        t = sessions[si]["t"]
        return (int(np.searchsorted(a["t"], t[lo])),
                int(np.searchsorted(a["t"], t[hi - 1], side="right")))
    exit_ev = {}
    if args.exit:
        import exit_cue
        exit_ev = {s["name"]: exit_cue.session_events(p) for p, s in zip(paths, sessions)}

    def merged(sess, ev):
        """OR the learned buzzes with the exit cue's, under the shared cooldown."""
        extra = exit_ev.get(sess["name"])
        if not extra:
            return ev
        out, last = [], -1e9
        for i in sorted(set(ev) | set(extra)):
            if sess["t"][i] - last >= COOLDOWN:
                out.append(i)
                last = sess["t"][i]
        return out

    folds = []
    for si, s in enumerate(sessions):
        nb = 4 if s["scripted_neg"].any() else 2
        for lo, hi in blocks_for(s, nb):
            folds.append((si, lo, hi))

    def train(train_blocks):
        Xs, ys = [], []
        for si, lo, hi in train_blocks:
            s = sessions[si]
            X, y = s["X"][lo:hi].reshape(-1, len(names)), s.get("y_train", s["y"])[lo:hi].reshape(-1)
            m = y >= 0
            Xs.append(X[m])
            ys.append(y[m])
            if args.aug_train:
                for _, a in aug.get(si, []):
                    alo, ahi = tmap(si, a, lo, hi)
                    Xa, ya = a["X"][alo:ahi].reshape(-1, len(names)), a["y"][alo:ahi].reshape(-1)
                    Xs.append(Xa[ya >= 0])
                    ys.append(ya[ya >= 0])
        X, y = np.concatenate(Xs), np.concatenate(ys)
        if args.mirror:
            # mirror augmentation: a left-right flipped punch is still a punch. Only the
            # sideways position changes sign and the wrist identity swaps; all motion
            # features (speed, flow, divergence, pixel change) are mirror-invariant.
            Xm = X.copy()
            Xm[:, names.index("relx")] *= -1
            Xm[:, names.index("wrist_id")] = 1 - Xm[:, names.index("wrist_id")]
            X, y = np.concatenate([X, Xm]), np.concatenate([y, y])
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
                                             min_samples_leaf=30, l2_regularization=1.0,
                                             class_weight="balanced", random_state=0)
        clf.fit(X, y)
        return clf

    def model_scorer(clf):
        cache = {}

        def f(sess):
            key = sess["name"]
            if key not in cache:
                n = len(sess["t"])
                p = clf.predict_proba(sess["X"].reshape(-1, len(names)))[:, 1].reshape(n, 2)
                p[~sess["sh"]] = 0
                if args.smooth > 1:
                    # causal mean of the last `smooth` frames per wrist
                    q = p.copy()
                    for L in range(1, args.smooth):
                        q[L:] += p[:-L]
                        q[:L] += p[:L]
                    p = q / args.smooth
                cache[key] = p.max(axis=1)
            return cache[key]
        return f

    def launch_scorer(sess):
        return sess["launch"].max(axis=1)

    print(f"\n{'':28s} | caught <=50ms <=100 <=150 <=200 <=300 | median  | fa/min | feints")
    for budget in args.budgets:
        acc = {"learned": [[], 0, 0.0, 0, 0], "launch (hand-made)": [[], 0, 0.0, 0, 0]}
        hooks = {k: [] for k in acc}
        per_sess = {}
        per_aug = {}
        for fi, (si, lo, hi) in enumerate(folds):
            train_blocks = [f for j, f in enumerate(folds) if j != fi]
            clf = train(train_blocks)
            tb = [(sessions[a], b, c) for a, b, c in train_blocks]
            for label, scorer in (("learned", model_scorer(clf)), ("launch (hand-made)", launch_scorer)):
                thr = threshold_for_budget(tb, scorer, budget)
                s = sessions[si]
                ev = events_from_scores(s["t"], scorer(s), thr)
                if label == "learned":
                    ev = merged(s, ev)
                d, f, ns, fh, fn = score_events(s, lo, hi, ev)
                fams = [e["family"] for e in s["events"] if lo <= e["departure"] < hi]
                sdt = 1000 * float(np.median(np.diff(s["t"])))
                d = [None if x is None else x * sdt for x in d]       # frames -> ms
                hooks[label] += [x for x, fam in zip(d, fams) if fam.startswith("HOOK")]
                if label == "learned":
                    for preset, a in aug.get(si, []):
                        alo, ahi = tmap(si, a, lo, hi)
                        aev = events_from_scores(a["t"], scorer(a), thr)
                        ad, af, ans, _, _ = score_events(a, alo, ahi, aev)
                        adt = 1000 * float(np.median(np.diff(a["t"])))
                        pa = per_aug.setdefault(preset, [0, 0, 0, 0.0])
                        pa[0] += sum(x is not None and x * adt <= 200 for x in ad)
                        pa[1] += len(ad)
                        pa[2] += af
                        pa[3] += ans
                    per = per_sess.setdefault(s["name"], [0, 0, 0, 0.0])
                    per[0] += sum(x is not None and x <= 200 for x in d)
                    per[1] += len(d)
                    per[2] += f
                    per[3] += ns
                a = acc[label]
                a[0] += d
                a[1] += f
                a[2] += ns
                a[3] += fh
                a[4] += fn
        print(f"--- threshold picked on training blocks for <= {budget:g} false alarms/min ---")
        for label, a in acc.items():
            summarize(label, *a, ms)
            hk = hooks[label]
            print(f"{'':28s}   hooks only: caught <=200ms {sum(x is not None and x <= 200 for x in hk)}/{len(hk)}, "
                  f"<=300ms {sum(x is not None for x in hk)}/{len(hk)}")
        for preset, (c, n, f, ns) in sorted(per_aug.items()):
            print(f"    learned on {preset:9s} copies: caught <=200ms {c}/{n} ({100 * c / max(n, 1):.0f}%)"
                  + (f", {f / (ns / 60):.2f} fa/min" if ns else ""))
        for name, (c, n, f, ns) in per_sess.items():
            print(f"    learned on {name}: caught <=200ms {c}/{n}"
                  + (f", {f / (ns / 60):.2f} fa/min" if ns else ""))

    # final model on everything + importances
    clf = train(folds)
    thr = {b: threshold_for_budget([(sessions[a], b2, c) for a, b2, c in folds], model_scorer(clf), b)
           for b in args.budgets}
    Path(args.save).parent.mkdir(exist_ok=True)
    import flow_features
    joblib.dump(dict(model=clf, names=names, thresholds=thr, pos_frames=early_dataset.POS_FRAMES,
                     sessions=[s["name"] for s in sessions], flow_scale=flow_features.SCALE), args.save)
    print(f"\nsaved {args.save}; thresholds by budget {({k: round(float(v), 3) for k, v in thr.items()})}")

    from sklearn.inspection import permutation_importance
    s = sessions[-1]
    X, y = s["X"].reshape(-1, len(names)), s["y"].reshape(-1)
    m = y >= 0
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(m)
    idx = rng.choice(idx, size=min(len(idx), 6000), replace=False)
    imp = permutation_importance(clf, X[idx], y[idx], scoring="average_precision", n_repeats=3, random_state=0)
    print("top features (permutation importance, in-sample, indicative only):")
    for i in np.argsort(imp.importances_mean)[::-1][:12]:
        print(f"  {names[i]:14s} {imp.importances_mean[i]:.4f}")


if __name__ == "__main__":
    main()
