"""
Offline auto-labeler: find the frame where each fist LEAVES its resting (guard)
position, from a recorded session (data/session_X.npz + .mp4).

Offline we may look at future frames, so this is far more accurate than the
live detector. For each wrist:
  1. Position relative to the body: origin = shoulder midpoint, unit = shoulder
     width (cancels camera motion and distance).
  2. Rest position = median position over the previous ~0.6 s of calm frames.
  3. Motion energy = pixel change in a box around the wrist. This catches
     straight punches toward the lens, where the wrist barely moves in the
     image but the fist grows.
  4. A punch = displacement from rest > --disp OR motion energy spike.
     Departure = walk back from the peak to the last calm frame at rest;
     departure is the frame after it.
  5. Guess the family: whole body grew with arms still -> STEP_IN (negative),
     mostly sideways -> HOOK_L / HOOK_R (side the fist came from in the image),
     otherwise JAB (straight). Guesses -- review the contact sheets.

Writes data/session_X.auto_candidates.json (every candidate, with a quality
tag: clean / combo / body / glitch) and contact sheets in auto_label_review/.
Candidates are not training labels: confirm them with review_labels.py, which
writes the accepted ones to data/session_X.labels.json for train_anticipator.py.

  python auto_label.py data/session_X.npz
  python review_labels.py data/session_X.npz
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10
KCONF = 0.35


def fill_gaps(x, ok, max_gap):
    """Linearly fill short gaps (<= max_gap frames) in a (N, d) series."""
    x = x.copy()
    ok = ok.copy()
    idx = np.flatnonzero(ok)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for j in range(a + 1, b):
                f = (j - a) / (b - a)
                x[j] = (1 - f) * x[a] + f * x[b]
                ok[j] = True
    return x, ok


def smooth(x, ok, k=3):
    """Centered moving average over valid frames only."""
    out = x.copy()
    h = k // 2
    for i in np.flatnonzero(ok):
        lo, hi = max(0, i - h), min(len(x), i + h + 1)
        m = ok[lo:hi]
        out[i] = x[lo:hi][m].mean(axis=0)
    return out


def motion_energy(video, centers, ok, box):
    """Mean abs pixel change in a box around each wrist, per frame. (N, 2)."""
    cap = cv2.VideoCapture(str(video))
    n = len(ok)
    energy = np.full((n, 2), np.nan)
    prev = None
    for i in range(n):
        good, frame = cap.read()
        if not good:
            break
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            hgt, wid = gray.shape
            for k in range(2):
                if not ok[i, k]:
                    continue
                x, y = centers[i, k]
                r = int(max(20, box[i] / 2))
                x0, x1 = int(max(0, x - r)), int(min(wid, x + r))
                y0, y1 = int(max(0, y - r)), int(min(hgt, y + r))
                if x1 > x0 and y1 > y0:
                    energy[i, k] = diff[y0:y1, x0:x1].mean()
        prev = gray
    cap.release()
    return energy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--disp", type=float, default=0.45, help="peak displacement from rest that counts as a punch (shoulder widths)")
    ap.add_argument("--rest", type=float, default=0.10, help="displacement still counted as 'at rest'")
    ap.add_argument("--calm", type=float, default=1.5, help="speed still counted as calm (shoulder widths/s)")
    ap.add_argument("--energy", type=float, default=4.0, help="motion-energy spike vs local baseline that counts as a punch")
    ap.add_argument("--gap", type=float, default=0.5, help="min seconds between punches of the same hand")
    ap.add_argument("--hold", type=float, default=0.15, help="fist must rest this long (s) before departure to count as clean")
    ap.add_argument("--sheet-dir", default="auto_label_review")
    ap.add_argument("--verbose", action="store_true", help="list rejected movements too")
    args = ap.parse_args()

    npz = Path(args.npz)
    video = npz.with_suffix(".mp4")
    z = np.load(npz, allow_pickle=False)
    t = z["timestamps"]
    kp = z["keypoints"].astype(float)
    cf = z["confidences"]
    n = len(t)
    dt = np.median(np.diff(t))
    cmds = z["commands"] if "commands" in z.files else np.array([])
    # some punchers threw the opposite hook to the prompt (data/session_notes.json)
    import session_notes
    cmds = session_notes.commands(npz, cmds)
    if len(cmds) and session_notes.session(npz).get("hooks_swapped"):
        print("Note: this puncher swapped the hooks; HOOK_L/HOOK_R prompts are swapped back (session_notes.json)")
    cmd_t = z["command_times"] if "command_times" in z.files else np.array([])

    sh_ok = (cf[:, L_SH] > KCONF) & (cf[:, R_SH] > KCONF)
    width = np.linalg.norm(kp[:, R_SH] - kp[:, L_SH], axis=1)
    sh_ok &= width > 10
    mid = (kp[:, L_SH] + kp[:, R_SH]) / 2

    # per wrist: relative position (N, 2, 2), validity (N, 2), pixel centers (N, 2, 2)
    rel = np.full((n, 2, 2), np.nan)
    ok = np.zeros((n, 2), bool)
    pix = np.full((n, 2, 2), np.nan)
    for k, w in enumerate((L_WR, R_WR)):
        good = sh_ok & (cf[:, w] > KCONF)
        r = np.zeros((n, 2))
        r[good] = (kp[good, w] - mid[good]) / width[good, None]
        r, good2 = fill_gaps(r, good, 3)
        p, _ = fill_gaps(kp[:, w], good, 3)
        r = smooth(r, good2)
        rel[:, k], ok[:, k], pix[:, k] = r, good2, p

    width_s, wok = fill_gaps(width[:, None], sh_ok, 3)
    width_s = width_s[:, 0]

    print("measuring motion around the wrists ...")
    energy = motion_energy(video, pix, ok, width_s * 0.5)

    win = max(3, int(round(0.6 / dt)))
    events = []
    for k in range(2):
        r, good = rel[:, k], ok[:, k]
        speed = np.full(n, np.nan)
        speed[1:] = np.linalg.norm(np.diff(r, axis=0), axis=1) / np.diff(t)
        speed[1:][~(good[1:] & good[:-1])] = np.nan

        # rest position: median of calm, valid frames in the previous window
        rest = np.full((n, 2), np.nan)
        for i in range(n):
            lo = max(0, i - win)
            m = good[lo:i] & (np.nan_to_num(speed[lo:i], nan=99) < args.calm)
            if m.sum() >= 3:
                rest[i] = np.median(r[lo:i][m], axis=0)
        disp = np.linalg.norm(r - rest, axis=1)

        e = energy[:, k]
        base = np.full(n, np.nan)
        for i in range(n):
            lo = max(0, i - win)
            v = e[lo:i]
            v = v[np.isfinite(v)]
            if len(v) >= 3:
                base[i] = np.median(v) + 1e-3
        espike = e / base

        hot = (np.nan_to_num(disp) > args.disp) | (np.nan_to_num(espike) > args.energy)
        last_peak_t = -1e9
        i = 0
        while i < n:
            if not hot[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            seg = np.arange(i, j + 1)
            score = np.nan_to_num(disp[seg]) / args.disp + np.nan_to_num(espike[seg]) / args.energy
            peak = int(seg[np.argmax(score)])
            i = j + 1
            if t[peak] - last_peak_t < args.gap:
                continue
            # walk back to the last calm frame still at rest
            start = peak
            limit = max(0, peak - int(round(0.6 / dt)))
            while start - 1 >= limit:
                s = start - 1
                at_rest = good[s] and np.isfinite(disp[s]) and disp[s] < args.rest \
                    and np.nan_to_num(speed[s], nan=0) < args.calm \
                    and np.nan_to_num(espike[s], nan=1) < 2.0
                if at_rest:
                    break
                start = s
            departure = start
            # --- quality checks ---
            # 1) the fist must have been resting for >= --hold s before departure
            hold_n = max(2, int(round(args.hold / dt)))
            pre = np.arange(max(0, departure - hold_n), departure)
            rested = (start - 1 >= limit) and len(pre) == hold_n and all(
                good[s] and np.isfinite(disp[s]) and disp[s] < args.rest * 1.5
                and np.nan_to_num(speed[s], nan=0) < args.calm * 1.5 for s in pre)
            # 2) the arm must move more than the whole body
            seg2 = np.arange(departure, peak + 1)
            body = 0.0
            if len(seg2) > 1 and sh_ok[seg2].sum() >= 2:
                v = seg2[sh_ok[seg2]]
                body = np.linalg.norm(mid[v[-1]] - mid[v[0]]) / width_s[v[0]] + abs(width_s[v[-1]] / width_s[v[0]] - 1)
            arm = max(float(np.nan_to_num(disp[peak])), 0.1 * float(np.nan_to_num(espike[peak])))
            if np.nan_to_num(disp[peak]) > 4:
                quality = "glitch"   # wrist keypoint jumped (tracking error)
            elif not rested:
                quality = "combo"    # no clean rest before it (mid-combo or fidgeting)
            elif body > 0.6 * arm and body > 0.25:
                quality = "body"     # whole body moved (bob, lean, step)
            else:
                quality = "clean"
            # rough family guess
            a = rest[departure] if np.all(np.isfinite(rest[departure])) else r[departure]
            move = r[peak] - a
            lo, hi = max(0, departure - 2), min(n - 1, peak + 2)
            grow = width_s[hi] / max(width_s[lo], 1) - 1
            arm_moved = np.linalg.norm(move) > args.disp * 0.7
            if grow > 0.12 and not arm_moved and np.nan_to_num(espike[peak]) < args.energy * 2:
                family = "STEP_IN"
                if quality == "body":
                    quality = "clean"  # a clean step-in is a useful negative example
            elif arm_moved and abs(move[0]) > 1.3 * abs(move[1]):
                family = "HOOK_L" if a[0] < 0 else "HOOK_R"
            else:
                family = "JAB"
            guess = family
            if len(cmds):
                # scripted session: the on-screen command is the true type
                j = np.searchsorted(cmd_t, t[departure], side="right") - 1
                family = str(cmds[j]) if j >= 0 else "IDLE"
            events.append(dict(departure=int(departure), peak=peak, wrist="image-left" if k == 0 else "image-right",
                               guess=guess,
                               family=family, quality=quality, lead_frames=int(peak - departure),
                               disp=float(np.nan_to_num(disp[peak])), espike=float(np.nan_to_num(espike[peak]))))
            last_peak_t = t[peak]

    # merge events from both hands that are nearly simultaneous (keep the stronger)
    events.sort(key=lambda e: e["departure"])
    merged = []
    for e in events:
        if merged and t[e["departure"]] - t[merged[-1]["departure"]] < 0.25:
            if e["disp"] + e["espike"] > merged[-1]["disp"] + merged[-1]["espike"]:
                merged[-1] = e
            continue
        merged.append(e)

    # Candidates are NOT training labels yet: review them with review_labels.py,
    # which writes the accepted ones to session_X.labels.json.
    cand_path = npz.with_suffix(".auto_candidates.json")
    cand_path.write_text(json.dumps(dict(
        session=npz.stem, params=vars(args),
        candidates=[dict(e, time=float(t[e["departure"]] - t[0])) for e in merged]), indent=1))

    # contact sheets: frames departure-2 .. departure+3, departure boxed green
    out = Path(args.sheet_dir) / npz.stem
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    for n_e, e in enumerate(merged):
        tiles = []
        for off in range(-2, 4):
            fi = min(max(e["departure"] + off, 0), n - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            good, fr = cap.read()
            if not good:
                fr = np.zeros((480, 640, 3), np.uint8)
            fr = cv2.resize(fr, (320, 240))
            col = (0, 255, 0) if off == 0 else (0, 165, 255) if fi == e["peak"] else (200, 200, 200)
            cv2.rectangle(fr, (0, 0), (319, 239), col, 4 if off == 0 else 1)
            cv2.putText(fr, f"{off:+d} f{fi} {1000 * (t[fi] - t[e['departure']]):+.0f}ms", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            tiles.append(fr)
        sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        cv2.putText(sheet, f"#{n_e} {e['quality'].upper()} {e['family']} ({e['wrist']}) departure t={t[e['departure']] - t[0]:.2f}s",
                    (6, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"event_{n_e:03d}_{e['quality']}.jpg"), sheet)
    cap.release()

    if len(cmds):
        scored = [e for e in merged if e["quality"] == "clean" and e["family"] in ("JAB", "CROSS", "HOOK_L", "HOOK_R")]
        def coarse(f):
            return "STRAIGHT" if f in ("JAB", "CROSS") else f
        agree = sum(coarse(e["guess"]) == coarse(e["family"]) for e in scored)
        print(f"type guess vs script (straight/hook_l/hook_r): {agree}/{len(scored)} agree")
    quals, fams = {}, {}
    for e in merged:
        quals[e["quality"]] = quals.get(e["quality"], 0) + 1
        if e["quality"] == "clean":
            fams[e["family"]] = fams.get(e["family"], 0) + 1
    clean = [e for e in merged if e["quality"] == "clean"]
    print(f"{len(merged)} candidate movements: {quals}")
    print(f"clean candidates to review: {len(clean)} {fams}")
    if clean:
        print(f"clean departure -> peak median {1000 * np.median([t[e['peak']] - t[e['departure']] for e in clean]):.0f} ms")
    for n_e, e in enumerate(merged):
        if e["quality"] != "clean" and not args.verbose:
            continue
        print(f"  #{n_e:3d} t={t[e['departure']] - t[0]:7.2f}s  {e['quality']:6s} {e['family']:8s} {e['wrist']:12s} "
              f"departure->peak {1000 * (t[e['peak']] - t[e['departure']]):4.0f} ms  disp {e['disp']:.2f}  motion x{e['espike']:.1f}")
    print(f"candidates -> {cand_path} | contact sheets -> {out}")
    print(f"next: python review_labels.py {npz}")


if __name__ == "__main__":
    main()
