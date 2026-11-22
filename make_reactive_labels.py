"""
Labels for the REACTIVE detector from reviewed sessions, in a separate folder (data_reactive/).

Why: the reactive detector (early_detector.py, HEAD_MOUNTED_TRAINING.md) is built around
"start = first frame the fist LEAVES GUARD". The prediction work reviews starts as "first
frame the fist moves TOWARD the target" (later, after any wind-up). Training the reactive
detector on a mix of both definitions blurs what it learns. For every accepted punch the
review also saved the computer's leave-guard frame ("initial"), so the reactive labels can
use that without relabeling:
  - computer frame within 0-MAX_GAP frames before the reviewed start -> the computer frame
    (that is where this punch left guard)
  - otherwise (the computer had locked onto another movement, or the reviewer moved the start
    earlier) -> the reviewed start minus the median gap, never later than the reviewed start
Sessions without review progress files (older ones) are copied as they are.
data/ is never changed. Files are hard links where possible (no extra disk space).

  python make_reactive_labels.py
  python early_detector.py --data data_reactive --test-mount head --budgets 2 4 --save models/candidates/x.joblib
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np

import review_labels as rl
from label_onsets import save_labels

SRC, DST = Path("data"), Path("data_reactive")
MAX_GAP = 8


def link(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def reactive_marks(npz, labeler):
    """Leave-guard marks for one labeler pass, or None when there is no review progress."""
    prog = npz.with_suffix(f".review_{labeler}.json")
    if not prog.exists():
        return None
    states = json.loads(prog.read_text())
    cands = rl.review_candidates(json.loads(npz.with_suffix(".auto_candidates.json").read_text())["candidates"],
                                 messy=labeler.endswith("_messy"))
    pairs = [(int(v["departure"]), int(v.get("initial", v["departure"])), v) for v in states.values()
             if v.get("decision") in ("yes", "feint")]
    gaps = [d - i for d, i, _ in pairs if 0 <= d - i <= MAX_GAP]
    med = int(round(np.median(gaps))) if gaps else 0
    marks, used = [], {"computer": 0, "shifted": 0}
    for d, i, v in pairs:
        if v.get("rule") != "toward_target":
            start, how = d, "computer"        # reviewed under the old leave-guard rule: keep it exactly
        elif 0 <= d - i <= MAX_GAP:
            start, how = i, "computer"
        else:
            start, how = min(d, max(0, d - med)), "shifted"
        used[how] += 1
        marks.append(dict(preparation=None, departure=start,
                          family="FEINT" if v["decision"] == "feint" else v["family"], rule="leaves_guard"))
    marks.sort(key=lambda m: m["departure"])
    return marks, used, med


def main():
    DST.mkdir(exist_ok=True)
    for npz in sorted(SRC.glob("session_*.npz")):
        labels = npz.with_suffix(".labels.json")
        if not labels.exists():
            continue
        for suffix in (".npz", ".mp4", ".auto_candidates.json"):
            if npz.with_suffix(suffix).exists():
                link(npz.with_suffix(suffix), DST / (npz.stem + suffix))
        out = DST / labels.name
        saved = json.loads(labels.read_text())["labelers"]
        if out.exists():
            out.unlink()
        notes = []
        for labeler, marks in saved.items():
            r = reactive_marks(npz, labeler) if labeler in ("reviewed", "reviewed_messy") else None
            if r is None:
                save_labels(out, labeler, marks)
                notes.append(f"{labeler}: {len(marks)} as reviewed")
            else:
                new, used, med = r
                save_labels(out, labeler, new)
                notes.append(f"{labeler}: {len(new)} leave-guard ({used['computer']} computer frame, "
                             f"{used['shifted']} reviewed start {-med:+d} fr)")
        print(npz.stem, "|", "; ".join(notes))


if __name__ == "__main__":
    main()
