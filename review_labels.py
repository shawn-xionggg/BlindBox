"""
Quick yes/no review of auto_label.py candidates.

Each candidate plays as a slow loop around the frame where the computer thinks
the fist left its resting position (green border). You just confirm:

  y  yes, that's a punch starting here        n  no, not a punch
  ,  .  nudge the start one frame earlier / later (if it's slightly off)
  1 JAB  2 CROSS  3 HOOK_L  4 HOOK_R  5 UPPERCUT  6 STEP_IN   (fix the type)
     HOOK_L / HOOK_R = the fist comes from the image-left / image-right side
  b  back one       space  pause loop       q  save and quit

Progress is saved after every key, so you can quit and resume any time.
Accepted punches go to data/session_X.labels.json (labeler "reviewed"),
which train_anticipator.py uses.

  python review_labels.py data/session_X.npz          # clean punch candidates
                                                       # (scripted don't-punch moves skipped)
  python review_labels.py data/session_X.npz --all    # also combo/body/glitch
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from label_onsets import save_labels

FAMILIES = {ord("1"): "JAB", ord("2"): "CROSS", ord("3"): "HOOK_L",
            ord("4"): "HOOK_R", ord("5"): "UPPERCUT", ord("6"): "STEP_IN"}
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


def render(cand, state, frames, loop_off, pos, total, timestamps):
    """Build the review canvas (960 x 480)."""
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
    lines = [
        (f"Candidate {pos + 1} / {total}", (255, 255, 255), 0.7),
        (f"t = {cand['time']:.2f}s   quality: {cand['quality']}", (200, 200, 200), 0.55),
        (f"type: {state['family']}" + ("" if state["family"] == cand["family"] else f"  (guess was {cand['family']})"),
         (0, 255, 255), 0.65),
        (f"start frame: {dep}" + ("" if dep == cand["departure"] else f"  (nudged {dep - cand['departure']:+d})"),
         (0, 255, 255), 0.55),
        (f"decision: {decision or '-'}", (0, 255, 0) if decision == "yes" else (0, 0, 255) if decision == "no" else (200, 200, 200), 0.65),
        ("", (0, 0, 0), 0.5),
        ("Does the fist START moving", (255, 255, 255), 0.55),
        ("at the GREEN frame?", (255, 255, 255), 0.55),
        ("", (0, 0, 0), 0.5),
        ("y yes   n no   b back   q quit", (180, 255, 180), 0.55),
        (",  .  nudge start -1 / +1 frame", (180, 255, 180), 0.55),
        ("1 JAB 2 CROSS 3 HOOK_L", (180, 255, 180), 0.55),
        ("4 HOOK_R 5 UPPERCUT 6 STEP_IN", (180, 255, 180), 0.55),
        ("space pause loop", (180, 255, 180), 0.55),
    ]
    y = 34
    for text, c, s in lines:
        cv2.putText(canvas, text, (500, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1 if s < 0.6 else 2, cv2.LINE_AA)
        y += 24
    return canvas


def accepted_marks(cands, states):
    marks = []
    for i, c in enumerate(cands):
        s = states.get(str(i))
        if s and s.get("decision") == "yes":
            marks.append(dict(preparation=None, departure=int(s["departure"]), family=s["family"]))
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
    ap.add_argument("--ms", type=int, default=140, help="ms per frame in the slow loop")
    args = ap.parse_args()

    npz = Path(args.npz)
    cand_path = npz.with_suffix(".auto_candidates.json")
    if not cand_path.exists():
        raise SystemExit(f"No candidates yet: run  python auto_label.py {npz}")
    all_cands = json.loads(cand_path.read_text())["candidates"]
    cands = all_cands if args.all else [c for c in all_cands if c["quality"] == "clean"]
    if args.messy:
        args.labeler = "reviewed_messy"
        cands = [c for c in all_cands if c["quality"] != "clean" and
                 ("guess" not in c or c["family"] in ("JAB", "CROSS", "HOOK_L", "HOOK_R", "UPPERCUT"))]
    elif not args.all:
        # movements during scripted don't-punch commands are already known negatives
        # (scripted candidates carry a "guess" key; unscripted ones keep everything clean)
        cands = [c for c in cands if "guess" not in c
                 or c["family"] in ("JAB", "CROSS", "HOOK_L", "HOOK_R", "UPPERCUT")]
    if not cands:
        raise SystemExit("No candidates to review.")
    timestamps = np.load(npz, allow_pickle=False)["timestamps"]
    progress_path = npz.with_suffix(f".review_{args.labeler}{'_all' if args.all else ''}.json")
    labels_path = npz.with_suffix(".labels.json")
    states = json.loads(progress_path.read_text()) if progress_path.exists() else {}

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
        st = states.setdefault(str(pos), dict(departure=c["departure"], family=c["family"], decision=None))
        if loaded_key != (pos, st["departure"]):
            frames = load_frames(cap, st["departure"], n_frames)
            loaded_key, loop_i = (pos, st["departure"]), 0
        canvas = render(c, st, frames, offs[loop_i], pos, len(cands), timestamps)
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
        elif key in (ord("y"), ord("n")):
            st["decision"] = "yes" if key == ord("y") else "no"
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
    print(f"reviewed {sum(d is not None for d in dec)}/{len(cands)} | yes {dec.count('yes')} | no {dec.count('no')}")
    print(f"accepted by type: {fams}")
    print(f"labels -> {labels_path} (labeler '{args.labeler}')")


if __name__ == "__main__":
    main()
