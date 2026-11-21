"""
Follow-up punch candidates for free-boxing blocks (sessions recorded with
collect.py --free). Run it after auto_label.py and before the --messy review.

Why: auto_label.py finds a punch by walking back from the fist's furthest point
to the last moment the fist rested. In a combo the fist never rests between
punches, and auto_label.py also skips a second punch from the same hand within
0.5 s and merges two-hand starts within 0.25 s. So the second and third punches
of a combo mostly never reach review.

This pass looks for LAUNCHES instead, from two signals per fist:
  (a) its distance from its guard spot (body-relative) starts growing fast, which
      catches hooks, uppercuts and anything with sideways or vertical travel;
  (b) the image around the fist moves fast (optical-flow magnitude minus the
      torso's), which catches straight punches at the camera that barely move
      sideways. Bursts where the fist comes back toward its guard spot are skipped.
A launch's start is where the fist begins moving out: (a) the first frame moving
out faster than 0.5 shoulder widths/s, (b) the quiet frame just before the burst
(the turnaround). On the 4 reviewed sessions, launches found 92% of reviewed punch
starts (337/368 within -150/+200 ms; median 32 ms early).

Only launches in FREE blocks that start within --follow s after another candidate
(a combo continues) and that no candidate already covers (none starts within
--near s) are appended to auto_candidates.json, with quality "combo", so they
show up in the --messy review. A launch of the same fist before its last punch's
peak + --retract s is that fist coming back, so it's skipped. Treating the old
sessions' punch prompts as free boxing, this adds about 12-24 candidates per
minute for the reviewer to accept or reject.

It only appends. Review progress is saved by position in the candidate list, so
existing candidates keep their places and a review in progress stays valid.
Running it a second time adds nothing.

  python auto_label.py data/session_X.npz
  python combo_candidates.py data/session_X.npz
  python review_labels.py data/session_X.npz
  python review_labels.py data/session_X.npz --messy
"""
import argparse
import json
from pathlib import Path

import numpy as np

from auto_label import fill_gaps, smooth
from free_blocks import FREE_COMMAND

L_SH, R_SH, WRISTS = 5, 6, (9, 10)
KCONF = 0.35


def wrist_tracks(t, kp, cf):
    """Body-relative wrist positions (origin = shoulder midpoint, unit = shoulder
    width), short gaps filled and lightly smoothed. Returns rel (N, 2, 2), ok (N, 2)."""
    n = len(t)
    sh = (cf[:, L_SH] > KCONF) & (cf[:, R_SH] > KCONF)
    width = np.linalg.norm(kp[:, R_SH] - kp[:, L_SH], axis=1)
    sh &= width > 10
    mid = (kp[:, L_SH] + kp[:, R_SH]) / 2
    rel = np.full((n, 2, 2), np.nan)
    ok = np.zeros((n, 2), bool)
    for k, w in enumerate(WRISTS):
        good = sh & (cf[:, w] > KCONF)
        r = np.zeros((n, 2))
        r[good] = (kp[good, w] - mid[good]) / width[good, None]
        r, good = fill_gaps(r, good, 3)
        r = smooth(r, good)
        r[~good] = np.nan
        rel[:, k], ok[:, k] = r, good
    return rel, ok


def guard_spot(t, r, ok, window=1.0, calm=1.5):
    """Per frame: median position over the past `window` s of calm frames (the fist's
    guard spot). Held at its last value while there are too few calm frames (mid-combo)."""
    n = len(t)
    speed = np.full(n, np.inf)
    speed[1:] = np.linalg.norm(np.diff(r, axis=0), axis=1) / np.diff(t)
    calm_ok = ok & (np.nan_to_num(speed, nan=np.inf) < calm)
    out = np.full((n, 2), np.nan)
    last = None
    lo = 0
    for i in range(n):
        while t[i] - t[lo] > window:
            lo += 1
        m = calm_ok[lo:i]
        if m.sum() >= 3:
            last = np.median(r[lo:i][m], axis=0)
        out[i] = last if last is not None else np.nan
    return out


def _smooth(x, ok, k=3):
    """Centered moving average over valid frames (NaN where invalid)."""
    v, c = np.where(ok, x, 0.), ok.astype(float)
    num, den = np.convolve(v, np.ones(k), "same"), np.convolve(c, np.ones(k), "same")
    return np.where(ok & (den > 0), num / np.maximum(den, 1), np.nan)


def _runs(mask, bridge=0):
    """(first, last) index of each run of True, bridging gaps of up to `bridge` frames."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > bridge + 1)
    return list(zip(idx[np.r_[0, cuts + 1]], idx[np.r_[cuts, len(idx) - 1]]))


def find_launches(t, kp, cf, mag=None, min_rise=0.2, min_speed=1.5, surge=1.3, back=0.2):
    """Launches of either fist, as dicts (start, peak, wrist k, rise) sorted by start.
    mag: optional (N, 2) optical-flow magnitude around each wrist minus the torso's
    (flow_features 'mag_rel', shoulder widths/s); without it only signal (a) is used."""
    rel, ok = wrist_tracks(t, kp, cf)
    out = []
    for k in range(2):
        e = np.linalg.norm(rel[:, k] - guard_spot(t, rel[:, k], ok[:, k]), axis=1)   # distance from guard
        fin = ok[:, k] & np.isfinite(e)
        e_s = _smooth(np.nan_to_num(e), fin)
        de = np.gradient(np.nan_to_num(e_s), t)
        de[~fin] = np.nan
        bursts = []
        # (a) moving out from the guard spot fast enough and far enough; the start is the
        #     first frame moving out faster than 0.5 shoulder widths/s
        for i, j in _runs(np.nan_to_num(de, nan=-1) > 0.5, bridge=1):
            rise = e_s[j] - e_s[max(0, i - 1)]
            if np.nanmax(de[i:j + 1]) >= min_speed and rise >= min_rise:
                bursts.append((int(i), int(j), float(rise)))
        # (b) fast motion around the fist that isn't a return to guard
        if mag is not None:
            m = np.nan_to_num(_smooth(np.nan_to_num(mag[:, k]), np.isfinite(mag[:, k])))
            for i, j in _runs(m > surge):
                rise = e_s[j] - e_s[max(0, i - 1)]
                if np.isfinite(rise) and rise < -0.15:
                    continue
                lo = int(np.searchsorted(t, t[i] - back))
                bursts.append((lo + int(np.argmin(m[lo:i + 1])), int(j), float(max(np.nan_to_num(rise), 0.)) + .01))
        for s, j, rise in bursts:
            hi = int(np.searchsorted(t, t[s] + 0.5))
            peak = s + int(np.nanargmax(e_s[s:max(hi, j + 1)])) if np.isfinite(e_s[s:max(hi, j + 1)]).any() else j
            out.append(dict(start=s, peak=max(peak, s + 1), wrist=k, rise=rise))
    # a punch turns the body, so both fists move: of two launches within 100 ms, keep the stronger
    out.sort(key=lambda e: e["start"])
    kept = []
    for e in out:
        if kept and t[e["start"]] - t[kept[-1]["start"]] < 0.1:
            if e["rise"] > kept[-1]["rise"]:
                kept[-1] = e
            continue
        kept.append(e)
    return kept


def free_windows(commands, command_times, t_end):
    """(start, end) times of the FREE blocks in a session's prompt list."""
    return [(float(command_times[j]), float(command_times[j + 1]) if j + 1 < len(command_times) else float(t_end) + 1)
            for j in range(len(commands)) if str(commands[j]) == FREE_COMMAND]


def new_candidates(npz, cands, near=0.15, follow=1.0, retract=0.25, flow=True, **kw):
    """Launches inside FREE blocks that continue a combo (start within `follow` s after
    another candidate) and that no candidate covers yet, in auto_label's format.
    A launch of the same fist before the previous candidate's peak + `retract` s is that
    fist coming back, not a new punch. Returns (new candidates, seconds searched)."""
    with np.load(npz, allow_pickle=False) as z:
        t, kp, cf = z["timestamps"], z["keypoints"].astype(float), z["confidences"]
        cmd = z["commands"] if "commands" in z.files else np.array([])
        ct = z["command_times"] if "command_times" in z.files else np.array([])
    windows = free_windows(cmd, ct, t[-1])
    if not windows:
        return [], 0.
    mag = None
    if flow:
        import flow_features     # cached in features/; computed from the .mp4 the first time
        mag = flow_features.load(npz)["mag_rel"]
    wrist_of = {"image-left": 0, "image-right": 1}
    taken = sorted((float(t[c["departure"]]), wrist_of.get(c["wrist"], -1), float(t[min(c["peak"], len(t) - 1)]))
                   for c in cands)    # (start time, fist, peak time) of every candidate so far
    added = []
    for e in find_launches(t, kp, cf, mag=mag, **kw):
        ts = float(t[e["start"]])
        if not any(a <= ts < b for a, b in windows):
            continue
        if any(abs(x[0] - ts) <= near for x in taken):
            continue        # auto_label.py (or an earlier launch) already proposed this start
        before = [x for x in taken if x[0] < ts]
        if not before or ts - before[-1][0] > follow:
            continue        # not a follow-up: auto_label.py handles punches that start a combo
        if any(x[1] == e["wrist"] and ts <= x[2] + retract for x in before):
            continue        # the same fist is still coming back from its last punch
        taken = sorted(taken + [(ts, e["wrist"], float(t[e["peak"]]))])
        added.append(dict(departure=int(e["start"]), peak=int(e["peak"]),
                          wrist="image-left" if e["wrist"] == 0 else "image-right",
                          guess="LAUNCH", family=FREE_COMMAND, quality="combo",
                          lead_frames=int(e["peak"] - e["start"]), disp=round(e["rise"], 3), espike=0.0,
                          time=ts - float(t[0]), source="combo"))
    free_s = sum(min(b, t[-1]) - a for a, b in windows)
    return added, free_s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--near", type=float, default=0.15, help="skip a launch if a candidate already starts this close (s)")
    ap.add_argument("--follow", type=float, default=1.0,
                    help="only launches starting this soon after another candidate (s): combo follow-ups")
    ap.add_argument("--retract", type=float, default=0.25,
                    help="ignore the same fist until this long after its last punch's peak (s): it's coming back")
    ap.add_argument("--min-rise", type=float, default=0.2, help="(a) how far the fist must move out (shoulder widths)")
    ap.add_argument("--min-speed", type=float, default=1.5, help="(a) how fast it must move out (shoulder widths/s)")
    ap.add_argument("--surge", type=float, default=1.3, help="(b) flow magnitude around the fist (shoulder widths/s)")
    ap.add_argument("--no-flow", action="store_true",
                    help="use only signal (a), the fist's position, when neither the video nor its flow cache "
                         "(features/) is on this machine; misses more straight punches")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be added")
    args = ap.parse_args()
    npz = Path(args.npz)
    cand_path = npz.with_suffix(".auto_candidates.json")
    if not cand_path.exists():
        raise SystemExit(f"No candidates yet: run  python auto_label.py {npz}  first")
    data = json.loads(cand_path.read_text())
    if "combo_pass" in data and not args.dry_run:
        raise SystemExit(f"Already done for {npz.name}: {data['combo_pass']['added']} combo candidates were added.")
    params = dict(near=args.near, follow=args.follow, retract=args.retract,
                  min_rise=args.min_rise, min_speed=args.min_speed, surge=args.surge)
    added, free_s = new_candidates(npz, data["candidates"], flow=not args.no_flow, **params)
    if not free_s:
        raise SystemExit(f"{npz.name} has no FREE blocks (it wasn't recorded with collect.py --free): nothing to do.")
    print(f"{len(added)} follow-up candidates in {free_s:.0f} s of free boxing "
          f"({60 * len(added) / free_s:.1f}/min), all quality 'combo' -> reviewed in the --messy pass")
    if args.dry_run:
        for c in added:
            print(f"  t={c['time']:7.2f}s  {c['wrist']:11s} start f{c['departure']}  out {c['disp']:.2f} sw")
        return
    data["candidates"] += added     # appended at the END: existing review progress stays valid
    data["combo_pass"] = dict(added=len(added), **params)
    cand_path.write_text(json.dumps(data, indent=1))
    print(f"appended to {cand_path}")
    print(f"next: python review_labels.py {npz} --messy")


if __name__ == "__main__":
    main()
