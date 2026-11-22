"""
Train and honestly evaluate a punch PREDICTOR: a warning before the fist starts
moving toward the target. Targets and features: predict_dataset.py.
The reactive detector (early_detector.py, models/early_detector*.joblib) is not touched.

Tests. The alarm threshold is always chosen on training data only, from scores of
models that did not see those frames (3-fold inside the training data):
  (a) leave one performer out: test on a person the model never trained on
  (b) personalization: train on everyone else + the first half of one performer's
      natural punches, test on the second half (a 2 s gap at the cut)
  (b') when only one performer exists: their first half -> second half, run as a
      pipeline check only. It is labeled as such: it is NOT a test on new people.
  (c) random warnings at the model's own rate, 500 runs: (c1) anywhere in the test
      footage, (c2) at the model's rate within each kind of footage (stricter: no
      credit for just noticing "this person is boxing now")
Warnings are replayed like live: score above the threshold, then a cooldown
(default 0.3 s, so a jab and the cross after it can each get one).
A warning counts for a punch if it comes 0-0.5 s before the start (--credit). A
warning while another punch is already flying toward the target counts neither way:
that is reacting, not predicting. Every other warning is a false warning.

Reported for first punches and follow-ups (<= 1 s after another punch) separately:
  warned 0-0.5 s before the start (x/y, 95% range), median lead, lead >= 100 ms,
  warned before ANY visible movement (the auto-labeler's first-motion frame),
  false warnings per minute of don't-punch footage and of natural footage
  (between punches), feints warned. Prompted punches (screen-glance cue) are left
  out of training and reported apart.

  python predict_trainer.py                    # every test the data allows, budgets 1 2 4
  python predict_trainer.py --quick            # budget 2 only, no ablations
  python predict_trainer.py --save models/candidates/predictor.joblib
  python predict_trainer.py --signed           # with vs without signed_motion.py's features
  python predict_trainer.py --rule depth       # the hand-written pull-back rule alone (depth/dip/either)
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import predict_dataset as pdx
from predict_dataset import DONT, FEINTW, NATURAL, OTHER, PROMPTED

KIND_NAMES = ("natural", "don't-punch", "prompted", "feint prompts", "other")
FEATURE_SETS = {"all": ("body", "rhythm", "context"), "body+rhythm": ("body", "rhythm"), "context only": ("context",)}


class NotEnoughData(Exception):
    pass


def wilson(k, n, z=1.96):
    """95% range for a proportion k/n."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0., c - h), min(1., c + h))


# ----------------------------------------------------------------------------- data
def build(paths, flow=True, train_prompted=False, horizon=pdx.H, signed=False):
    """signed=True appends signed_motion.py's per-fist features (group 'signed') and keeps
    its pull-back rule scores in s['rule']."""
    import flow_features
    import signed_motion as sm
    sessions, names, groups = [], None, None
    caches = {p: Path("features") / f"{Path(p).stem}.flow{flow_features.TAG}.npz" for p in paths}
    missing = [Path(p).stem for p, c in caches.items() if not c.exists()]
    if flow and missing:
        print(f"no optical-flow cache for {', '.join(missing)}: building WITHOUT flow features for all sessions")
        flow = False
    for p in paths:
        s = pdx.load(p)
        fl = dict(np.load(caches[p])) if flow else None
        s["X"], names, groups = pdx.features(s, fl)
        if signed:
            F = sm.per_wrist(s["t"], s["kp"], s["cf"], None if fl is None else fl["div_rel"])
            s["rule"] = {k: v.max(1) for k, v in sm.rule_scores(F).items()}
            if s["stance"] == "southpaw":
                F = F[:, ::-1]                               # lead hand first, as in predict_dataset
            s["X"] = np.concatenate([s["X"], F.reshape(len(F), -1).astype(np.float32)], 1)
            names = names + [f"signed_{n}{k}" for k in range(2) for n in sm.NAMES]
            groups = groups + ["signed"] * (2 * len(sm.NAMES))
        s["y"] = pdx.targets(s, h=horizon, train_prompted=train_prompted)
        s["cands"] = json.loads(Path(p).with_suffix(".auto_candidates.json").read_text())["candidates"]
        s["tracked"] = (s["cf"][:, 5] > pdx.KC) & (s["cf"][:, 6] > pdx.KC)
        s["dt"] = np.r_[np.median(np.diff(s["t"])), np.diff(s["t"])]
        s["flying"] = np.zeros(len(s["t"]), bool)
        for q in s["punches"]:
            s["flying"][q["frame"]:q["peak"] + 1] = True
        sessions.append(s)
    return sessions, names, groups


# ----------------------------------------------------------------------------- model
class Fitted:
    """A fitted classifier and the feature columns it uses. Columns that never vary in the
    training data (e.g. hips that are never in view) are dropped: the trees can't split them."""

    def __init__(self, clf, cols):
        self.clf, self.cols = clf, np.asarray(cols, int)

    def proba(self, X):
        return self.clf.predict_proba(X[:, self.cols])[:, 1]


def fit(sessions, parts, cols):
    cols = np.asarray(cols, int)
    X = np.concatenate([sessions[si]["X"][lo:hi][:, cols] for si, lo, hi in parts])
    y = np.concatenate([sessions[si]["y"][lo:hi] for si, lo, hi in parts])
    m = y >= 0
    if (y[m] == 1).sum() < 20 or (y[m] == 0).sum() < 100:
        raise NotEnoughData(f"{(y[m] == 1).sum()} 'punch soon' frames to learn from")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns
        varies = np.nanmax(X[m], axis=0) > np.nanmin(X[m], axis=0)
    # No class weights: they make scikit-learn bin every feature with weighted percentiles
    # (minutes per fit), and only the ranking of scores matters, since the threshold is set
    # separately for each warning budget.
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)
    return Fitted(clf.fit(X[m][:, varies], y[m]), cols[varies])


def predict(model, s, lo, hi):
    p = model.proba(s["X"][lo:hi])
    p[~s["tracked"][lo:hi]] = 0.
    return p


def inner_scores(sessions, parts, cols, k=3):
    """Out-of-fold scores for the training parts: each third of every part is scored by a
    model trained on the other two thirds. Falls back to in-sample scores if a fold is too small."""
    scores = [np.zeros(hi - lo) for _, lo, hi in parts]
    edges = [np.linspace(lo, hi, k + 1).astype(int) for _, lo, hi in parts]
    try:
        for f in range(k):
            train = [(si, e[g], e[g + 1]) for (si, _, _), e in zip(parts, edges) for g in range(k) if g != f]
            model = fit(sessions, train, cols)
            for j, ((si, lo, _), e) in enumerate(zip(parts, edges)):
                scores[j][e[f] - lo:e[f + 1] - lo] = predict(model, sessions[si], e[f], e[f + 1])
        return scores, True
    except NotEnoughData:
        model = fit(sessions, parts, cols)
        return [predict(model, sessions[si], lo, hi) for si, lo, hi in parts], False


# ----------------------------------------------------------------------------- warnings and scoring
def warnings_from(t, p, thr, cooldown):
    out, last = [], -np.inf
    for i in np.flatnonzero(p > thr):
        if t[i] - last >= cooldown:
            out.append(i)
            last = t[i]
    return np.asarray(out, int)


def evaluate(s, lo, hi, w, credit=0.5):
    """Score warnings (frame indices) on frames [lo, hi) of session s."""
    t, kind = s["t"], s["kind"]
    w = np.asarray([i for i in w if lo <= i < hi], int)
    flying = s["flying"][w] if len(w) else np.zeros(0, bool)
    during = int(flying.sum())
    w = w[~flying]                                    # reacting to a punch in flight: neither way
    tw = t[w]
    used = np.zeros(len(w), bool)
    punches = []
    for q in s["punches"]:
        if not lo <= q["frame"] < hi:
            continue
        ts = q["time"]
        j = np.flatnonzero((tw >= ts - credit) & (tw < ts) & ~used)
        row = dict(kind=q["kind"], follow=q["follow"], lead=None, motion=None)
        if len(j):
            used[j[0]] = True
            row.update(lead=ts - tw[j[0]], motion=t[q["first_motion"]] - tw[j[0]])
        punches.append(row)
    false = np.bincount(kind[w[~used]], minlength=5) if len(w) else np.zeros(5, int)
    tr = s["tracked"][lo:hi]
    minutes = np.bincount(kind[lo:hi][tr], weights=s["dt"][lo:hi][tr], minlength=5) / 60
    feints = [f for f in s["feints"] if lo <= f < hi]
    feint_hit = sum(bool(np.any((tw >= t[f] - credit) & (tw <= t[f] + 0.3))) for f in feints)
    return dict(punches=punches, false=false, minutes=minutes, during=during, feints=(feint_hit, len(feints)))


def random_warnings(s, lo, hi, rate_per_min, cooldown, rng):
    """Random warnings with the given rate per minute for each kind of footage (array of 5),
    thinned by the same cooldown (the per-frame chance is raised to make up for it)."""
    t, kind, dt = s["t"][lo:hi], s["kind"][lo:hi], s["dt"][lo:hi]
    r = np.asarray(rate_per_min, float)[kind] / 60.
    lam = r / np.maximum(1 - r * cooldown, 0.1)
    fire = (rng.random(len(t)) < lam * dt) & s["tracked"][lo:hi]
    out, last = [], -np.inf
    for i in np.flatnonzero(fire):
        if t[i] - last >= cooldown:
            out.append(i)
            last = t[i]
    return np.asarray(out, int) + lo


def pick_threshold(sessions, parts, scores, budget, cooldown, credit):
    """Lowest threshold whose false warnings per minute (don't-punch + natural footage, on
    the given scores) stay within budget."""
    vals = np.concatenate([sc[(np.isin(sessions[si]["kind"][lo:hi], (DONT, NATURAL))) & sessions[si]["tracked"][lo:hi]]
                           for (si, lo, hi), sc in zip(parts, scores)])
    if not len(vals):
        raise NotEnoughData("no don't-punch or natural footage to set the threshold on")

    def within_budget(thr):
        false = minutes = 0.
        for (si, lo, hi), sc in zip(parts, scores):
            s = sessions[si]
            e = evaluate(s, lo, hi, warnings_from(s["t"][lo:hi], sc, thr, cooldown) + lo, credit)
            false += e["false"][DONT] + e["false"][NATURAL]
            minutes += e["minutes"][DONT] + e["minutes"][NATURAL]
        return bool(minutes) and false / minutes <= budget

    cands = np.unique(np.quantile(vals, np.linspace(0.5, 1.0, 300)))
    a, b = 0, len(cands)            # false warnings fall as the threshold rises: binary search
    while a < b:
        mid = (a + b) // 2
        if within_budget(cands[mid]):
            b = mid
        else:
            a = mid + 1
    return float(cands[a]) if a < len(cands) else float(vals.max()) + 1e-9


def pick_threshold_calm(sessions, parts, scores, budget, cooldown):
    """Label-free threshold for a NEW person: the lowest one whose warnings during their own
    don't-punch footage (guard up, moving, no punching) stay within budget per minute.
    Uses no punch labels: only which footage was a don't-punch prompt."""
    vals = np.concatenate([sc[(sessions[si]["kind"][lo:hi] == DONT) & sessions[si]["tracked"][lo:hi]]
                           for (si, lo, hi), sc in zip(parts, scores)])
    if not len(vals):
        raise NotEnoughData("no don't-punch footage of this person to calibrate on")

    def rate(thr):
        n = minutes = 0.
        for (si, lo, hi), sc in zip(parts, scores):
            s = sessions[si]
            w = warnings_from(s["t"][lo:hi], sc, thr, cooldown) + lo
            n += np.sum(s["kind"][w] == DONT) if len(w) else 0
            m = (s["kind"][lo:hi] == DONT) & s["tracked"][lo:hi]
            minutes += s["dt"][lo:hi][m].sum() / 60
        return n / minutes if minutes else np.inf

    cands = np.unique(np.quantile(vals, np.linspace(0.5, 1.0, 300)))
    ok = [c for c in cands if rate(c) <= budget]
    return float(ok[0]) if ok else float(vals.max()) + 1e-9


def tally(evals):
    """Pool evaluate() results: per group of punches and the false-warning rates."""
    rows = [r for e in evals for r in e["punches"]]
    groups = {"first punches": [r for r in rows if r["kind"] == NATURAL and not r["follow"]],
              "follow-ups": [r for r in rows if r["kind"] == NATURAL and r["follow"]],
              "prompted": [r for r in rows if r["kind"] == PROMPTED]}
    out = {}
    for name, g in groups.items():
        hit = [r for r in g if r["lead"] is not None]
        out[name] = dict(n=len(g), hit=len(hit), leads=[r["lead"] for r in hit],
                         useful=sum(r["lead"] >= 0.1 for r in hit), premotion=sum(r["motion"] > 0 for r in hit))
    false = sum(e["false"] for e in evals)
    minutes = sum(e["minutes"] for e in evals)
    out["fa_dont"] = (false[DONT] / minutes[DONT]) if minutes[DONT] else float("nan")
    out["fa_natural"] = (false[NATURAL] / minutes[NATURAL]) if minutes[NATURAL] else float("nan")
    out["minutes"] = minutes
    out["during"] = sum(e["during"] for e in evals)
    out["feints"] = tuple(map(sum, zip(*[e["feints"] for e in evals]))) if evals else (0, 0)
    return out


# ----------------------------------------------------------------------------- one test
def run_fold(sessions, train_parts, test_parts, cols, budgets, cooldown, credit, sims, rng, rule=None,
             calm_threshold=False):
    """rule: score with signed_motion's hand-written rule (that variant) instead of a model;
    its threshold is still chosen on the training parts only.
    calm_threshold: the threshold comes from the TEST person's own don't-punch footage instead
    (label-free calibration, like "stand in guard for a minute" before a demo)."""
    if rule:
        def score(si, lo, hi):
            return sessions[si]["rule"][rule][lo:hi] * sessions[si]["tracked"][lo:hi]
        oof, honest = [score(*p) for p in train_parts], True
        test = [score(*p) for p in test_parts]
    else:
        model = fit(sessions, train_parts, cols)
        oof, honest = inner_scores(sessions, train_parts, cols)
        test = [predict(model, sessions[si], lo, hi) for si, lo, hi in test_parts]
    out = {}
    for b in budgets:
        thr = (pick_threshold_calm(sessions, test_parts, test, b, cooldown) if calm_threshold
               else pick_threshold(sessions, train_parts, oof, b, cooldown, credit))
        evals, kinds_w = [], np.zeros(5)
        for (si, lo, hi), sc in zip(test_parts, test):
            s = sessions[si]
            w = warnings_from(s["t"][lo:hi], sc, thr, cooldown) + lo
            evals.append(evaluate(s, lo, hi, w, credit))
            kinds_w += np.bincount(s["kind"][w], minlength=5) if len(w) else 0
        minutes = sum(e["minutes"] for e in evals)
        rate_any = np.full(5, kinds_w.sum() / max(minutes.sum(), 1e-9))
        rate_kind = np.where(minutes > 0, kinds_w / np.maximum(minutes, 1e-9), 0.)
        rand = {}
        for label, rate in (("c1", rate_any), ("c2", rate_kind)):
            res = []
            for _ in range(sims):
                ev = [evaluate(sessions[si], lo, hi, random_warnings(sessions[si], lo, hi, rate, cooldown, rng), credit)
                      for si, lo, hi in test_parts]
                tl = tally(ev)
                res.append([(tl[g]["hit"], tl[g]["n"]) for g in ("first punches", "follow-ups")])
            rand[label] = np.array(res)          # (sims, 2 groups, (hit, n))
        out[b] = dict(thr=thr, evals=evals, rand=rand, honest_threshold=honest)
    return out


def merge_folds(folds, budgets):
    """Pool the per-fold results of one test."""
    out = {}
    for b in budgets:
        evals = [e for f in folds for e in f[b]["evals"]]
        rand = {k: sum(f[b]["rand"][k] for f in folds) for k in ("c1", "c2")}
        out[b] = dict(tally=tally(evals), rand=rand, thr=[f[b]["thr"] for f in folds],
                      honest=all(f[b]["honest_threshold"] for f in folds))
    return out


def print_result(res, budgets):
    for b in budgets:
        r, tl = res[b], res[b]["tally"]
        print(f"--- budget {b:g} false warnings/min (threshold {', '.join(f'{x:.3f}' for x in r['thr'])}"
              f"{'' if r['honest'] else ' - chosen IN-SAMPLE: too little training data for 3 inner folds'}) ---")
        print(f"  {'':15s} {'warned 0-0.5 s before':24s} {'median lead':>11s} {'lead>=100ms':>12s} {'before any motion':>18s}")
        for g in ("first punches", "follow-ups"):
            x = tl[g]
            if not x["n"]:
                print(f"  {g:15s} none in the test footage")
                continue
            lo, hi = wilson(x["hit"], x["n"])
            med = f"{1000 * np.median(x['leads']):.0f} ms" if x["leads"] else "-"
            print(f"  {g:15s} {x['hit']:3d}/{x['n']:<3d} {100 * x['hit'] / x['n']:3.0f}% [{100 * lo:.0f}-{100 * hi:.0f}%]"
                  f"{'':4s} {med:>11s} {x['useful']:>8d}/{x['n']:<3d} {x['premotion']:>12d}/{x['n']}")
        for label, text in (("c1", "random, same rate anywhere"), ("c2", "random, same rate per footage")):
            parts = []
            for gi, g in enumerate(("first punches", "follow-ups")):
                h, n = r["rand"][label][:, gi, 0], r["rand"][label][:, gi, 1]
                if n.max() > 0:
                    pct = 100 * h / np.maximum(n, 1)
                    parts.append(f"{g} {pct.mean():.0f}% [{np.percentile(pct, 2.5):.0f}-{np.percentile(pct, 97.5):.0f}%]")
            print(f"  {text} ({len(r['rand'][label])} runs): {', '.join(parts) if parts else '-'}")
        m = tl["minutes"]
        print(f"  false warnings/min: don't-punch {tl['fa_dont']:.2f} ({m[DONT]:.1f} min) | between punches in natural "
              f"footage {tl['fa_natural']:.2f} ({m[NATURAL]:.1f} min) | while a punch was flying: {tl['during']} | "
              f"feints warned {tl['feints'][0]}/{tl['feints'][1]}")
        p = tl["prompted"]
        if p["n"]:
            print(f"  prompted punches (screen cue; not in the numbers above): warned {p['hit']}/{p['n']} "
                  f"({100 * p['hit'] / p['n']:.0f}%)")


# ----------------------------------------------------------------------------- tests
def natural_count(s):
    return sum(q["kind"] == NATURAL for q in s["punches"])


def split_performer(sessions, idx, gap=1.0):
    """Cut one performer's footage at the time of their median natural punch. Returns
    (first-half parts, second-half parts), with `gap` s left out on each side of the cut."""
    idx = sorted(idx, key=lambda i: sessions[i]["name"])
    nat = [(i, q["frame"]) for i in idx for q in sessions[i]["punches"] if q["kind"] == NATURAL]
    ci, cf = nat[len(nat) // 2]
    t = sessions[ci]["t"]
    a = int(np.searchsorted(t, t[cf] - gap))
    b = int(np.searchsorted(t, t[cf] + gap))
    first = [(i, 0, len(sessions[i]["t"])) for i in idx if sessions[i]["name"] < sessions[ci]["name"]] + [(ci, 0, a)]
    second = [(ci, b, len(t))] + [(i, 0, len(sessions[i]["t"])) for i in idx if sessions[i]["name"] > sessions[ci]["name"]]
    return [p for p in first if p[2] > p[1]], [p for p in second if p[2] > p[1]]


AUTO_QUALITIES = ("clean", "combo")     # auto_label.py / combo_candidates.py movements taken as punches


def start_offset(sessions, idx, near=10):
    """Median frames from the auto-labeler's departure to the reviewed start, over sessions idx."""
    diffs = []
    for i in idx:
        s = sessions[i]
        dep = np.array([c["departure"] for c in s["cands"]]) if s["cands"] else np.array([], int)
        for q in s["punches"]:
            if len(dep):
                j = int(np.argmin(np.abs(dep - q["frame"])))
                if abs(dep[j] - q["frame"]) <= near:
                    diffs.append(q["frame"] - dep[j])
    return int(round(np.median(diffs))) if diffs else 0


def auto_copy(s, offset, horizon):
    """The same footage with label-free punches: every auto-labeler movement of AUTO_QUALITIES,
    its start moved `offset` frames, no feints. What a live system could learn from by itself."""
    a = dict(s)
    t, n = s["t"], len(s["t"])
    starts = sorted({min(n - 1, max(0, int(c["departure"]) + offset)) for c in s["cands"] if c["quality"] in AUTO_QUALITIES})
    punches = []
    for d in starts:
        if punches and t[d] - t[punches[-1]["frame"]] < 0.12:
            continue
        c = min(s["cands"], key=lambda c: abs(int(c["departure"]) + offset - d))
        punches.append(dict(frame=d, time=float(t[d]), peak=min(max(int(c["peak"]), d + 1), n - 1),
                            first_motion=min(d, int(c["departure"])), kind=int(s["kind"][d]),
                            follow=bool(punches and t[d] - punches[-1]["time"] <= pdx.FOLLOW)))
    a.update(punches=punches, feints=[], name=s["name"] + "#auto")
    a["y"] = pdx.targets(a, h=horizon)
    a["flying"] = np.zeros(n, bool)
    for q in punches:
        a["flying"][q["frame"]:q["peak"] + 1] = True
    return a


def self_calibration_folds(sessions, horizon):
    """(b-auto): like (b), but this person's first half is learned from label-free punches
    (auto_copy); the threshold is set on that too. Only the scoring of the second half uses
    the reviewed labels. Appends the label-free copies to `sessions`."""
    by = {}
    for i, s in enumerate(list(sessions)):
        by.setdefault(s["performer"], []).append(i)
    with_nat = [p for p, idx in by.items() if sum(natural_count(sessions[i]) for i in idx) >= 10]
    full = lambda idx: [(i, 0, len(sessions[i]["t"])) for i in idx]
    folds, offsets = [], {}
    for p in with_nat:
        first, second = split_performer(sessions, by[p])
        others = [i for q, idx in by.items() if q != p for i in idx]
        # the gap comes from other people's head-camera sessions (labeled under the current start rule)
        same_cam = [i for i in others if sessions[i]["mount"] == sessions[by[p][0]]["mount"]] or others
        off = offsets[p] = start_offset(sessions, same_cam)
        copies = {}
        for si, lo, hi in first:
            if si not in copies:
                sessions.append(auto_copy(sessions[si], off, horizon))
                copies[si] = len(sessions) - 1
        folds.append((full(others) + [(copies[si], lo, hi) for si, lo, hi in first], second))
    return folds, offsets


def tests(sessions):
    """(name, explanation, [(train parts, test parts)]) for every test the data allows."""
    by = {}
    for i, s in enumerate(sessions):
        by.setdefault(s["performer"], []).append(i)
    with_nat = [p for p, idx in by.items() if sum(natural_count(sessions[i]) for i in idx) >= 10]
    full = lambda idx: [(i, 0, len(sessions[i]["t"])) for i in idx]
    out, notes = [], []
    if len(with_nat) >= 2:
        folds = [(full([i for q, idx in by.items() if q != p for i in idx]), full(by[p])) for p in with_nat]
        out.append(("(a) leave one performer out", "test on a person the model never trained on", folds))
        folds = []
        for p in with_nat:
            first, second = split_performer(sessions, by[p])
            folds.append((full([i for q, idx in by.items() if q != p for i in idx]) + first, second))
        out.append(("(b) personalization", "train on everyone else + the first half of this person's natural "
                    "punches, test on their second half", folds))
    else:
        notes.append(f"(a) and (b) need natural punches (free boxing or unscripted sessions) from at least 2 "
                     f"performers; this data has {len(with_nat)} ({', '.join(with_nat) or 'none'}). Not run.")
        if len(with_nat) == 1:
            p = with_nat[0]
            first, second = split_performer(sessions, by[p])
            out.append((f"(b') one performer ({p}): first half -> second half",
                        "PIPELINE CHECK ONLY: the model saw this same person, session and camera; "
                        "it says nothing about new people", [(first, second)]))
    return out, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--budgets", type=float, nargs="+", default=[1., 2., 4.])
    ap.add_argument("--horizon", type=float, default=pdx.H, help="train: a punch starts within this many s")
    ap.add_argument("--credit", type=float, default=0.5, help="a warning this long before a start counts for it (s)")
    ap.add_argument("--cooldown", type=float, default=0.3)
    ap.add_argument("--sims", type=int, default=500, help="random-baseline runs")
    ap.add_argument("--train-prompted", action="store_true", help="also learn from prompted punches (screen cue!)")
    ap.add_argument("--no-flow", action="store_true", help="leave out the optical-flow features")
    ap.add_argument("--quick", action="store_true", help="budget 2 only, 200 random runs, no ablations")
    ap.add_argument("--signed", action="store_true",
                    help="add signed_motion.py's per-fist features and compare with and without them "
                         "(saves models/candidates/predictor_signed.joblib)")
    ap.add_argument("--rule", choices=("depth", "dip", "either"),
                    help="score signed_motion.py's hand-written pull-back rule instead of a model")
    ap.add_argument("--self-calibrate", action="store_true",
                    help="add test (b-auto): personalization learned from label-free punches (the auto-labeler's "
                         "movements) instead of reviewed ones, as a live system could do in a warm-up minute")
    ap.add_argument("--only-mount", choices=("head", "desk"), help="use only sessions recorded with this camera mount")
    ap.add_argument("--calm-threshold", action="store_true",
                    help="add test (a-calm): new person, threshold from their own don't-punch footage (label-free)")
    ap.add_argument("--auto-qualities", nargs="+", default=list(AUTO_QUALITIES),
                    help="with --self-calibrate: which auto-labeler movements count as punches")
    ap.add_argument("--save", default="models/candidates/predictor.joblib")
    args = ap.parse_args()
    globals()["AUTO_QUALITIES"] = tuple(args.auto_qualities)
    save = Path(args.save)
    if save.resolve().parent != Path("models/candidates").resolve():
        ap.error("--save must be a file in models/candidates/ (installed models are never overwritten)")
    if args.quick:
        args.budgets, args.sims = [2.], min(args.sims, 200)
    paths = sorted(p for p in Path(args.data).glob("*.npz")
                   if p.with_suffix(".labels.json").exists() and p.with_suffix(".auto_candidates.json").exists())
    if args.only_mount:
        def mount_of(p):
            with np.load(p, allow_pickle=False) as z:
                return str(z["mount"]) if "mount" in z.files else "desk"
        paths = [p for p in paths if mount_of(p) == args.only_mount]
    if not paths:
        raise SystemExit(f"no reviewed sessions in {args.data}/")
    signed = args.signed or bool(args.rule)
    if args.signed and args.save == ap.get_default("save"):
        save = Path("models/candidates/predictor_signed.joblib")
    sessions, names, groups = build(paths, flow=not args.no_flow, train_prompted=args.train_prompted,
                                    horizon=args.horizon, signed=signed)
    print(f"{len(sessions)} sessions, {len(names)} features")
    for s in sessions:
        nat = [q for q in s["punches"] if q["kind"] == NATURAL]
        print(f"  {s['name']}: performer {s['performer']}, {s['mount']}, natural punches {len(nat)} "
              f"({sum(q['follow'] for q in nat)} follow-ups), prompted {sum(q['kind'] == PROMPTED for q in s['punches'])}"
              f", duplicates merged {s['merged']}")
    rng = np.random.default_rng(0)
    n_real = len(sessions)
    plan, notes = tests(sessions)
    if args.calm_threshold:
        a = [p for p in plan if p[0].startswith("(a)")]
        if a:
            plan.append(("(a-calm) new person, alarm level from their own calm footage",
                         "like (a), but the threshold is set on the test person's don't-punch footage "
                         "(no punch labels): the false-warning rate there is held to the budget", a[0][2], True))
    if args.self_calibrate:
        folds, offsets = self_calibration_folds(sessions, args.horizon)
        if folds:
            plan.append(("(b-auto) self-calibration",
                         "like (b), but this person's first half is learned from LABEL-FREE punches (auto-labeler "
                         "movements, start moved by the median auto->reviewed gap of the OTHER people: "
                         + ", ".join(f"{p} {o:+d} fr" for p, o in offsets.items())
                         + "); scored on the second half with reviewed labels", folds))
    for n in notes:
        print("\n" + n)
    for entry in plan:
        name, why, folds = entry[:3]
        calm = len(entry) > 3 and entry[3]
        print(f"\n=== {name} ===\n    {why}")
        if args.rule:
            sets, full = {f"rule '{args.rule}'": ()}, {f"rule '{args.rule}'"}
        elif args.signed:
            sets = {"all": FEATURE_SETS["all"], "all+signed": FEATURE_SETS["all"] + ("signed",),
                    "signed only": ("signed",)}
            full = {"all", "all+signed"}
        else:
            sets, full = FEATURE_SETS, {"all"}
        if args.quick:
            sets = {k: v for k, v in sets.items() if k in full}
        for set_name, use in sets.items():
            cols = [j for j, g in enumerate(groups) if g in use]
            budgets = args.budgets if set_name in full else [2.]
            try:
                res = merge_folds([run_fold(sessions, tr, te, cols, budgets, args.cooldown, args.credit,
                                            args.sims if set_name in full else min(args.sims, 200), rng,
                                            rule=args.rule, calm_threshold=calm)
                                   for tr, te in folds], budgets)
            except NotEnoughData as e:
                print(f"  [{set_name}] not enough data: {e}")
                continue
            if set_name in full:
                if len(full) > 1:
                    print(f"  [{set_name}]")
                print_result(res, budgets)
            else:
                tl = res[2.]["tally"]
                parts = [f"{g} {tl[g]['hit']}/{tl[g]['n']}" for g in ("first punches", "follow-ups") if tl[g]["n"]]
                print(f"  ablation [{set_name}] at budget 2: warned {', '.join(parts)}; "
                      f"false/min don't-punch {tl['fa_dont']:.2f}, natural {tl['fa_natural']:.2f}")

    if args.rule:
        return                      # a hand-written rule has nothing to save
    # final model on everything usable, threshold from out-of-fold scores
    cols = list(range(len(names)))
    sessions = sessions[:n_real]                # never the label-free copies
    parts = [(i, 0, len(s["t"])) for i, s in enumerate(sessions)]
    try:
        model = fit(sessions, parts, cols)
        oof, honest = inner_scores(sessions, parts, cols)
        thr = {b: pick_threshold(sessions, parts, oof, b, args.cooldown, args.credit) for b in args.budgets}
    except NotEnoughData as e:
        raise SystemExit(f"\nno model saved: {e}")
    save.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=model.clf, columns=model.cols, names=names, groups=groups, thresholds=thr,
                     horizon=args.horizon,
                     cooldown=args.cooldown, lags=pdx.LAGS, flow=not args.no_flow and "mag_rel0@0s" in names,
                     sessions=[s["name"] for s in sessions], train_prompted=args.train_prompted), save)
    print(f"\nsaved {save} (thresholds {', '.join(f'{b:g}/min: {v:.3f}' for b, v in thr.items())})")


if __name__ == "__main__":
    main()
