"""
Quick yes/no review of auto_label.py candidates.

Each candidate opens PAUSED on the frame where the computer thinks the punch
starts (green border). The clip covers the whole punch: 0.5 s before to 1 s after.

  START = the first frame the fist moves TOWARD the target. A pull-back, dip or
  shoulder drop before that is the telegraph, not the start: move past it.

Fast way, with the mouse:
  space         play / pause the slow loop, to see the whole punch
  scroll wheel  one frame back / forward   (keys: a / d)
  click the timeline bar to jump there
  click the big picture (or a thumbnail)  = this frame is the START   (key: s)
  click a TYPE button (JAB, CROSS, HOOK..., UPPERCUT) = it's a punch of that type -> next
  click NO / FEINT = not a punch / a fake start -> next;  BACK = previous candidate

Keys:
  y  punch (keep the type)   n  not a punch   f  feint (saved as FEINT, never used as a punch)
  ,  .  move the start one frame earlier / later
  1 JAB  2 CROSS  3 HOOK_L  4 HOOK_R  5 UPPERCUT  6 STEP_IN   (set the type, then y)
     HOOK_L / HOOK_R = the fist comes from the image-left / image-right side
     (camera faces the puncher: image-left = their RIGHT hand)
  -  =  slower / faster loop     b  back one     q  save and quit

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
OFFSETS = range(-15, 31)  # frames in the clip, relative to the start: ~0.5 s before to ~1 s after
STRIP = range(-2, 4)      # thumbnails under the timeline
W, H = 960, 520           # review window
TIMELINE = (362, 392)     # y range of the clickable timeline bar
STRIP_Y = 396             # thumbnails' top
# clickable buttons in the right panel: (label, action, x0, y0, x1, y1)
BUTTONS = []
for _k, (_label, _act) in enumerate((("JAB", "JAB"), ("CROSS", "CROSS"), ("UPPERCUT", "UPPERCUT"))):
    BUTTONS.append((_label, _act, 495 + 155 * _k, 178, 640 + 155 * _k, 212))
BUTTONS += [("HOOK  <- from L (their R)", "HOOK_L", 495, 218, 717, 252),
            ("HOOK  -> from R (their L)", "HOOK_R", 723, 218, 950, 252),
            ("NO punch", "no", 495, 258, 717, 292), ("FEINT", "feint", 723, 258, 950, 292),
            ("< BACK", "back", 495, 298, 640, 332), ("PUNCH (keep type)", "yes", 650, 298, 950, 332)]


def load_frames(cap, departure, n_frames):
    """The clip around `departure`: one seek, then sequential reads (fast)."""
    frames = {}
    first = max(0, departure + OFFSETS[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for i in range(first, min(n_frames, departure + OFFSETS[-1] + 1)):
        ok, f = cap.read()
        if not ok:
            break
        frames[i - departure] = f
    return frames


def button_at(x, y):
    return next((b[1] for b in BUTTONS if b[2] <= x <= b[4] and b[3] <= y <= b[5]), None)


def blank(w, h):
    return np.zeros((h, w, 3), np.uint8)


def _put(canvas, text, xy, scale, color, thick=1):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(cand, state, frames, loop_off, pos, total, timestamps, labeler="reviewed", near=None,
           paused=True, ms=140):
    """Build the review canvas (W x H). near = (frame, where) of another accepted
    punch starting within SAME_PUNCH_S s, or None."""
    canvas = blank(W, H)
    dep = state["departure"]
    # big view: the frame on show
    big = frames.get(loop_off)
    big = cv2.resize(big, (480, 360)) if big is not None else blank(480, 360)
    col = (0, 255, 0) if loop_off == 0 else (200, 200, 200)
    cv2.rectangle(big, (0, 0), (479, 359), col, 8 if loop_off == 0 else 1)
    dt_ms = 1000 * (timestamps[min(dep + loop_off, len(timestamps) - 1)] - timestamps[dep]) if 0 <= dep + loop_off < len(timestamps) else 0
    _put(big, ("START  " if loop_off == 0 else "") + f"{loop_off:+d} fr ({dt_ms:+.0f} ms)", (10, 32), .8, col, 2)
    _put(big, "PAUSED - space to play" if paused else f"playing ({ms} ms/frame)", (10, 350), .55, (255, 255, 255))
    canvas[0:360, 0:480] = big
    # timeline: the whole clip; green = start, yellow = the computer's peak, white = on show
    y0, y1 = TIMELINE
    n = len(OFFSETS)
    cell = W / n
    peak = int(cand.get("peak", cand["departure"])) - dep
    for k, off in enumerate(OFFSETS):
        x0, x1 = int(k * cell), int((k + 1) * cell) - 1
        c = (0, 200, 0) if off == 0 else (0, 200, 255) if off == peak else (70, 70, 70) if off in frames else (25, 25, 25)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), c, -1)
        if off == loop_off:
            cv2.rectangle(canvas, (x0, y0 - 2), (x1, y1 + 2), (255, 255, 255), 2)
    # thumbnails around the start: click one = that frame is the start
    for k, off in enumerate(STRIP):
        f = frames.get(off)
        f = cv2.resize(f, (160, 120)) if f is not None else blank(160, 120)
        c = (0, 255, 0) if off == 0 else (110, 110, 110)
        cv2.rectangle(f, (0, 0), (159, 119), c, 4 if off == 0 else 1)
        _put(f, f"{off:+d}", (6, 20), 0.55, c)
        canvas[STRIP_Y:STRIP_Y + 120, k * 160:(k + 1) * 160] = f
    # info panel
    decision = state.get("decision")
    first = state.get("initial", cand["departure"])   # where the green frame started for this reviewer
    warn = f"! punch already at {near[0] - dep:+d} fr ({near[1]}): same?" if near is not None else ""
    lines = [
        (f"Candidate {pos + 1} / {total}   t = {cand['time']:.1f}s  {cand['quality']}", (255, 255, 255), 0.6, 1),
        (f"type: {state['family']}" + ("" if state["family"] == cand["family"] else f"  (was {cand['family']})"),
         (0, 255, 255), 0.6, 1),
        (f"start: frame {dep}" + ("" if dep == first else f"  (moved {dep - first:+d})") + f"   [{labeler}]",
         (0, 255, 255), 0.5, 1),
        (f"decision: {decision or '-'}", {"yes": (0, 255, 0), "no": (0, 0, 255), "feint": (0, 165, 255)}
         .get(decision, (200, 200, 200)), 0.6, 1),
        (warn, (0, 140, 255), 0.5, 1),
        ("START = first frame the fist moves TOWARD you.", (255, 255, 255), 0.47, 1),
        ("A pull-back / dip before it = telegraph, not start.", (255, 255, 255), 0.47, 1),
    ]
    y = 24
    for text, c, s, t in lines:
        _put(canvas, text, (495, y), s, c, t)
        y += 21
    for label, act, x0, by0, x1, by1 in BUTTONS:
        chosen = (act == state["family"] and decision == "yes") or act == decision
        fill = (0, 110, 0) if chosen else (60, 60, 60)
        cv2.rectangle(canvas, (x0, by0), (x1, by1), fill, -1)
        cv2.rectangle(canvas, (x0, by0), (x1, by1), (200, 200, 200), 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
        _put(canvas, label, (x0 + (x1 - x0 - tw) // 2, by0 + (by1 - by0 + th) // 2), .5, (255, 255, 255))
    _put(canvas, "scroll / a d = 1 frame   click picture or s = START", (495, 350), .45, (180, 255, 180))
    _put(canvas, "keys: y n f  , .  1-5 type  - = speed  b back  q quit", (495, 166), .45, (180, 255, 180))
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
    offs = list(OFFSETS)
    zero = offs.index(0)
    loaded_key, frames, loop_i, paused, ms = None, {}, zero, True, args.ms
    events = []    # mouse actions, handled in the main loop: ("start", offset) / ("show", index) / ("button", act)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xFFFF          # wheel delta = signed high 16 bits of flags
            delta = delta - 0x10000 if delta >= 0x8000 else delta
            events.append(("step", -1 if delta > 0 else 1))      # scroll up = earlier frame
        elif event == cv2.EVENT_LBUTTONDOWN:
            if TIMELINE[0] - 4 <= y <= TIMELINE[1] + 4:
                events.append(("show", int(np.clip(x * len(offs) // W, 0, len(offs) - 1))))
            elif y >= STRIP_Y:
                k = x // 160
                if 0 <= k < len(STRIP):
                    events.append(("start", STRIP[k]))
            elif x < 480 and y < 360:
                events.append(("start", None))       # the frame on show
            elif button_at(x, y):
                events.append(("button", button_at(x, y)))
    cv2.setMouseCallback(win, on_mouse)

    def decide(st, decision, family=None):
        """Record a decision; returns False when it was the last candidate."""
        nonlocal pos, paused
        if family:
            st["family"] = family
        st["decision"], st["rule"] = decision, RULE
        paused = True                          # the next candidate opens paused on its start
        # next undecided candidate (after this one first, then from the top); decided ones are skipped
        order = list(range(pos + 1, len(cands))) + list(range(0, pos))
        nxt = next((i for i in order if not states.get(str(i), {}).get("decision")), None)
        if nxt is not None:
            pos = nxt
            return True
        save(progress_path, labels_path, args.labeler, cands, states)
        print("All candidates reviewed.")
        return False

    while True:
        c = cands[pos]
        first = start_frame(c, pos, jitter, f"{npz.stem}:{args.labeler}")
        st = states.setdefault(str(pos), dict(departure=first, initial=first, auto=int(c["departure"]),
                                              family=c["family"], decision=None))
        if loaded_key != (pos, st["departure"]):
            frames = load_frames(cap, st["departure"], n_frames)
            loaded_key, loop_i = (pos, st["departure"]), zero      # always show the start first
        near = nearby_punch(pos, st, states, sibling_frames, timestamps)
        canvas = render(c, st, frames, offs[loop_i], pos, len(cands), timestamps, args.labeler, near, paused, ms)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(ms if not paused else 20) & 0xFF
        running = True
        while events:
            kind, val = events.pop(0)
            if kind == "step":
                paused, loop_i = True, int(np.clip(loop_i + val, 0, len(offs) - 1))
            elif kind == "show":
                paused, loop_i = True, val
            elif kind == "start":
                off = offs[loop_i] if val is None else val
                st["departure"] = int(np.clip(st["departure"] + off, 0, n_frames - 1))
                paused = True
            elif val == "back":
                pos, paused = max(0, pos - 1), True
            elif val in ("yes", "no", "feint"):
                running = decide(st, val)
            else:                                # a type button = a punch of that type
                running = decide(st, "yes", val)
            save(progress_path, labels_path, args.labeler, cands, states)
            if not running or (kind == "button"):
                events.clear()
                break
        if not running:
            break
        if key == 255:
            if not paused:
                loop_i = (loop_i + 1) % len(offs)
            continue
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key in (ord("a"), ord("d")):     # step the shown frame (pauses the loop)
            paused = True
            loop_i = int(np.clip(loop_i + (1 if key == ord("d") else -1), 0, len(offs) - 1))
        elif key in (ord("s"), 13):           # the frame on show becomes the start
            st["departure"] = int(np.clip(st["departure"] + offs[loop_i], 0, n_frames - 1))
        elif key == ord(","):
            st["departure"] = max(0, st["departure"] - 1)
        elif key == ord("."):
            st["departure"] = min(n_frames - 1, st["departure"] + 1)
        elif key in (ord("-"), ord("=")):     # loop speed
            ms = int(np.clip(ms + (40 if key == ord("-") else -40), 20, 500))
        elif key in FAMILIES:
            st["family"] = FAMILIES[key]
        elif key in DECISIONS:
            if not decide(st, DECISIONS[key]):
                break
        elif key == ord("b"):
            pos, paused = max(0, pos - 1), True
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
