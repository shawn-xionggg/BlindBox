"""
Quick yes/no review of auto_label.py candidates.

Each candidate plays as a slow loop around the frame where the computer thinks
the punch starts (green border). You confirm it and fix the start frame:

  START = the first frame the fist moves TOWARD the target. A pull-back, dip or
  shoulder drop before that is the telegraph, not the start: nudge past it.

  y  punch, starting at the green frame        n  not a punch
  f  feint (a fake punch start: saved as type FEINT, never used as a punch)
  ,  .  nudge the start one frame earlier / later
  1 JAB  2 CROSS  3 HOOK_L  4 HOOK_R  5 UPPERCUT  6 STEP_IN   (fix the type)
     HOOK_L / HOOK_R = the fist comes from the image-left / image-right side
  b  back one       space  pause loop       q  save and quit

A warning appears when another accepted punch starts within 0.2 s of this one
(in this pass or the other pass): that is usually the same punch seen twice.

Progress is saved after every key, so you can quit and resume any time.
Accepted punches go to data/session_X.labels.json (labelers "reviewed" and
"reviewed_messy"), which training uses. Each new decision records the start
rule it was made under ("rule": "toward_target"); older labels have no rule.

  python review_labels.py data/session_X.npz          # clean punch candidates
                                                       # (scripted don't-punch moves skipped)
  python review_labels.py data/session_X.npz --messy  # punches right after another move,
                                                       # incl. combo follow-ups (combo_candidates.py)
  python review_labels.py data/session_X.npz --all    # also combo/body/glitch

Second reviewer, for a blind agreement check (never used for training):
  python review_labels.py data/session_X.npz --reviewer alice
  python review_labels.py data/session_X.npz --reviewer alice --messy
  python label_agreement.py data/session_X.npz --reviewer alice
"""
import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np

from free_blocks import FREE_COMMAND
from label_onsets import save_labels

FAMILIES = {ord("1"): "JAB", ord("2"): "CROSS", ord("3"): "HOOK_L",
            ord("4"): "HOOK_R", ord("5"): "UPPERCUT", ord("6"): "STEP_IN"}
# prompts during which a punch is expected; FREE = free-boxing block (collect.py --free)
PUNCH_PROMPTS = ("JAB", "CROSS", "HOOK_L", "HOOK_R", "UPPERCUT", FREE_COMMAND)
DECISIONS = {ord("y"): "yes", ord("n"): "no", ord("f"): "feint"}
RULE = "toward_target"   # stored with every new decision (see the module docstring)
RULE_LINES = ("START = first frame the fist moves",
              "TOWARD the target. A pull-back, dip",
              "or shoulder drop before it = telegraph,",
              "NOT the start (nudge with . past it).")
SAME_PUNCH_S = 0.2       # another accepted punch this close is probably the same punch
OFFSETS = range(-6, 11)   # frames shown in the loop, relative to departure
STRIP = range(-2, 4)      # thumbnails under the loop


def load_frames(cap, departure, n_frames):
    frames = {}
    for off in OFFSETS:
        i = departure + off
        if 0 <= i < n_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                frames[off] = f
    return frames


def blank(w, h):
    return np.zeros((h, w, 3), np.uint8)


def render(cand, state, frames, loop_off, pos, total, timestamps, labeler="reviewed", near=None):
    """Build the review canvas (960 x 480). near = (frame, where) of another accepted
    punch starting within SAME_PUNCH_S s, or None."""
    canvas = blank(960, 480)
    dep = state["departure"]
    # big looping view
    big = frames.get(loop_off)
    big = cv2.resize(big, (480, 360)) if big is not None else blank(480, 360)
    col = (0, 255, 0) if loop_off == 0 else (110, 110, 110)
    cv2.rectangle(big, (0, 0), (479, 359), col, 6 if loop_off == 0 else 1)
    dt_ms = 1000 * (timestamps[min(dep + loop_off, len(timestamps) - 1)] - timestamps[dep]) if 0 <= dep + loop_off < len(timestamps) else 0
    cv2.putText(big, f"{loop_off:+d} frames ({dt_ms:+.0f} ms)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    canvas[0:360, 0:480] = big
    # thumbnails
    for k, off in enumerate(STRIP):
        f = frames.get(off)
        f = cv2.resize(f, (160, 120)) if f is not None else blank(160, 120)
        c = (0, 255, 0) if off == 0 else (110, 110, 110)
        cv2.rectangle(f, (0, 0), (159, 119), c, 4 if off == 0 else 1)
        cv2.putText(f, f"{off:+d}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
        canvas[360:480, k * 160:(k + 1) * 160] = f
    # info panel
    decision = state.get("decision")
    first = state.get("initial", cand["departure"])   # where the green frame started for this reviewer
    warn = ""
    if near is not None:
        gap = near[0] - dep
        warn = f"! punch already at {gap:+d} fr ({near[1]}): same?"
    lines = [
        (f"Candidate {pos + 1} / {total}", (255, 255, 255), 0.7),
        (f"t = {cand['time']:.2f}s  {cand['quality']}  [{labeler}]", (200, 200, 200), 0.5),
        (f"type: {state['family']}" + ("" if state["family"] == cand["family"] else f"  (guess was {cand['family']})"),
         (0, 255, 255), 0.65),
        (f"start frame: {dep}" + ("" if dep == first else f"  (moved {dep - first:+d})"),
         (0, 255, 255), 0.55),
        (f"decision: {decision or '-'}", {"yes": (0, 255, 0), "no": (0, 0, 255), "feint": (0, 165, 255)}
         .get(decision, (200, 200, 200)), 0.65),
        (warn, (0, 140, 255), 0.5),
    ] + [(line, (255, 255, 255), 0.55) for line in RULE_LINES] + [
        ("y punch  f feint  n no  b back  q quit", (180, 255, 180), 0.55),
        (",  .  nudge start -1 / +1 frame", (180, 255, 180), 0.55),
        ("1 JAB 2 CROSS 3 HOOK_L 4 HOOK_R", (180, 255, 180), 0.55),
        ("5 UPPERCUT 6 STEP_IN  space pause", (180, 255, 180), 0.55),
    ]
    y = 34
    for text, c, s in lines:
        cv2.putText(canvas, text, (500, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1 if s < 0.6 else 2, cv2.LINE_AA)
        y += 24
    return canvas


def review_candidates(all_cands, messy=False, everything=False):
    """The candidates one review pass shows, in file order. Review progress is saved by
    position in this list, so the selection must stay stable."""
    if messy:
        # punches thrown right after another move, inside punch prompts (or unscripted sessions)
        return [c for c in all_cands if c["quality"] != "clean" and
                ("guess" not in c or c["family"] in PUNCH_PROMPTS)]
    if everything:
        return list(all_cands)
    # movements during scripted don't-punch commands are already known negatives
    # (scripted candidates carry a "guess" key; unscripted ones keep everything clean)
    return [c for c in all_cands if c["quality"] == "clean"
            and ("guess" not in c or c["family"] in PUNCH_PROMPTS)]


def sibling_labeler(labeler):
    """The same reviewer's other pass: reviewed <-> reviewed_messy, check_x <-> check_x_messy."""
    if labeler.endswith("_messy"):
        return labeler[:-len("_messy")]
    if labeler == "reviewed" or labeler.startswith("check_"):
        return labeler + "_messy"
    return None


def start_frame(cand, index, jitter, seed):
    """Where the green frame starts. A blind second reviewer gets a reproducible offset of up
    to `jitter` frames, so agreement can't come from both just accepting the computer's guess."""
    if not jitter:
        return int(cand["departure"])
    return max(0, int(cand["departure"]) + random.Random(f"{seed}:{index}").randint(-jitter, jitter))


def nearby_punch(pos, state, states, sibling_frames, timestamps):
    """(frame, where) of the closest other accepted punch starting within SAME_PUNCH_S s."""
    t0 = timestamps[state["departure"]]
    near = [(int(s["departure"]), "this pass") for i, s in states.items()
            if int(i) != pos and s.get("decision") == "yes"]
    near += [(d, "other pass") for d in sibling_frames]
    near = [(d, w) for d, w in near if abs(timestamps[d] - t0) <= SAME_PUNCH_S]
    return min(near, key=lambda x: abs(x[0] - state["departure"]), default=None)


def accepted_marks(cands, states):
    marks = []
    for i, c in enumerate(cands):
        s = states.get(str(i))
        if s and s.get("decision") in ("yes", "feint"):
            m = dict(preparation=None, departure=int(s["departure"]),
                     family="FEINT" if s["decision"] == "feint" else s["family"])
            if s.get("rule"):
                m["rule"] = s["rule"]
            marks.append(m)
    marks.sort(key=lambda m: m["departure"])
    return marks


def save(progress_path, labels_path, labeler, cands, states):
    progress_path.write_text(json.dumps(states, indent=1))
    save_labels(labels_path, labeler, accepted_marks(cands, states))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--all", action="store_true", help="also review combo/body/glitch candidates")
    ap.add_argument("--messy", action="store_true",
                    help="review the non-clean punches (thrown right after another move) in punch prompts; "
                         "saved as labeler 'reviewed_messy', which training also uses")
    ap.add_argument("--labeler", default="reviewed")
    ap.add_argument("--reviewer", help="a second reviewer's name, for a blind agreement check: saved as "
                                       "check_<name> (check_<name>_messy with --messy), never used for training")
    ap.add_argument("--jitter", type=int, help="start each green frame up to this many frames off the computer's "
                                               "guess (default: 2 with --reviewer, else 0)")
    ap.add_argument("--ms", type=int, default=140, help="ms per frame in the slow loop")
    args = ap.parse_args()
    if args.reviewer is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.reviewer) or args.reviewer.lower().startswith("reviewed"):
            ap.error("--reviewer takes a short name (letters, digits, _ or -), not 'reviewed'")
        args.labeler = f"check_{args.reviewer}" + ("_messy" if args.messy else "")
    elif args.messy:
        args.labeler = "reviewed_messy"
    jitter = args.jitter if args.jitter is not None else (2 if args.reviewer else 0)

    npz = Path(args.npz)
    cand_path = npz.with_suffix(".auto_candidates.json")
    if not cand_path.exists():
        raise SystemExit(f"No candidates yet: run  python auto_label.py {npz}")
    all_cands = json.loads(cand_path.read_text())["candidates"]
    cands = review_candidates(all_cands, messy=args.messy, everything=args.all)
    if not cands:
        # e.g. a --negatives-only session: no punches to review. Save an empty label set
        # so training still uses it (as known don't-punch footage).
        save_labels(npz.with_suffix(".labels.json"), args.labeler, [])
        raise SystemExit("No punch candidates to review; saved empty labels (session still used for training).")
    timestamps = np.load(npz, allow_pickle=False)["timestamps"]
    progress_path = npz.with_suffix(f".review_{args.labeler}{'_all' if args.all else ''}.json")
    labels_path = npz.with_suffix(".labels.json")
    states = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    sibling = sibling_labeler(args.labeler)
    saved = json.loads(labels_path.read_text())["labelers"] if labels_path.exists() else {}
    sibling_frames = [int(m["departure"]) for m in saved.get(sibling, []) if m["family"] != "FEINT"] if sibling else []

    print(f"Reviewing {len(cands)} candidates as '{args.labeler}'.")
    print("START = the first frame the fist moves TOWARD the target. A pull-back, dip or shoulder drop "
          "before it is the telegraph, not the start.")
    if args.reviewer is not None:
        print(f"Blind check: saved as '{args.labeler}', never used for training. Each green frame starts up to "
              f"{jitter} frames off the computer's guess. Afterwards: python label_agreement.py {npz} --reviewer {args.reviewer}")

    cap = cv2.VideoCapture(str(npz.with_suffix(".mp4")))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # resume at the first undecided candidate
    pos = next((i for i in range(len(cands)) if not states.get(str(i), {}).get("decision")), 0)
    win = "review punches"
    cv2.namedWindow(win)
    loaded_key, frames, loop_i, paused = None, {}, 0, False
    offs = list(OFFSETS)

    while True:
        c = cands[pos]
        first = start_frame(c, pos, jitter, f"{npz.stem}:{args.labeler}")
        st = states.setdefault(str(pos), dict(departure=first, initial=first, auto=int(c["departure"]),
                                              family=c["family"], decision=None))
        if loaded_key != (pos, st["departure"]):
            frames = load_frames(cap, st["departure"], n_frames)
            loaded_key, loop_i = (pos, st["departure"]), 0
        near = nearby_punch(pos, st, states, sibling_frames, timestamps)
        canvas = render(c, st, frames, offs[loop_i], pos, len(cands), timestamps, args.labeler, near)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(args.ms) & 0xFF
        if not paused:
            loop_i = (loop_i + 1) % len(offs)
        if key == 255:
            continue
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord(","):
            st["departure"] = max(0, st["departure"] - 1)
        elif key == ord("."):
            st["departure"] = min(n_frames - 1, st["departure"] + 1)
        elif key in FAMILIES:
            st["family"] = FAMILIES[key]
        elif key in DECISIONS:
            st["decision"] = DECISIONS[key]
            st["rule"] = RULE
            if pos + 1 < len(cands):
                pos += 1
            else:
                save(progress_path, labels_path, args.labeler, cands, states)
                print("All candidates reviewed.")
                break
        elif key == ord("b"):
            pos = max(0, pos - 1)
        save(progress_path, labels_path, args.labeler, cands, states)

    cap.release()
    cv2.destroyAllWindows()
    save(progress_path, labels_path, args.labeler, cands, states)
    dec = [s.get("decision") for s in states.values()]
    marks = accepted_marks(cands, states)
    fams = {}
    for m in marks:
        fams[m["family"]] = fams.get(m["family"], 0) + 1
    print(f"reviewed {sum(d is not None for d in dec)}/{len(cands)} | yes {dec.count('yes')} | "
          f"feint {dec.count('feint')} | no {dec.count('no')}")
    print(f"accepted by type: {fams}")
    print(f"labels -> {labels_path} (labeler '{args.labeler}')")


if __name__ == "__main__":
    main()
