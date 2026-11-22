"""
Reactive-detector A/B test of signed_motion.py, using early_detector.py UNCHANGED.

  python early_detector_signed.py --budgets 2               # baseline: same trainer, no extra columns
  python early_detector_signed.py --signed --budgets 2      # + 12 signed motion features per fist
  python early_detector_signed.py --rule either --budgets 2 # the pull-back rule as the hand-made row

It wraps early_dataset.build to append the signed features as extra columns
(--signed), or to put the rule's score where early_detector.py reads its
hand-made baseline (--rule; it then appears as the 'launch (hand-made)' row).
Then it runs early_detector.main() as is: same folds, same thresholds, same
scoring, so the rows compare like for like. Any other early_detector.py option
passes through. Models go to models/candidates/reactive_<variant>.joblib only.

Offline only: early_live.py and the C model don't compute these features, so a
winner would need them ported before it could run live.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import early_dataset
import early_detector
import signed_motion as sm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signed", action="store_true", help="append the signed motion features")
    ap.add_argument("--rule", choices=("depth", "dip", "either"), help="score the pull-back rule as the hand-made row")
    args, rest = ap.parse_known_args()
    if "--save" in rest:
        ap.error("--save is chosen here: models/candidates/reactive_<variant>.joblib")
    variant = "signed" if args.signed else f"rule_{args.rule}" if args.rule else "baseline"
    save = Path("models/candidates") / f"reactive_{variant}.joblib"
    save.parent.mkdir(parents=True, exist_ok=True)

    original = early_dataset.build

    def build(npz, labelers=("reviewed", "reviewed_messy")):
        s = original(npz, labelers)
        F = sm.for_session(npz)                            # (N, 2, 12): same wrist order as s["X"]
        if args.signed:
            s["X"] = np.concatenate([s["X"], F.astype(s["X"].dtype)], axis=2)
            s["names"] = s["names"] + [f"signed_{n}" for n in sm.NAMES]
        if args.rule:
            s["launch"] = sm.rule_scores(F)[args.rule]
        return s

    early_dataset.build = build
    print(f"variant: {variant}" + (" (the 'launch (hand-made)' row below is the pull-back rule)" if args.rule else ""))
    sys.argv = [sys.argv[0]] + rest + ["--save", str(save)]
    early_detector.main()


if __name__ == "__main__":
    main()
