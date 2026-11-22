"""
Signed motion of each fist, and a hand-written pull-back rule.

The old speed feature is unsigned: it can't tell a fist going out from one coming
back. Here every fist gets, per frame (all causal, frame i and earlier):
  out_v, out_a    signed speed / acceleration AWAY FROM (+) or BACK TOWARD (-) its guard
                  spot, in the image, body-relative (shoulder widths/s, /s^2)
  down_y, down_v, down_a
                  position below its guard spot (+ = lower), vertical speed and
                  acceleration (+ = downward): a dip before an uppercut
  depth, depth_v, depth_a
                  looming: how far the fist is in front of (+) or behind (-) where it was
                  at guard, as log size from the optical-flow divergence around it
                  (flow_features 'div_rel'), and its speed and acceleration
  turn_t          time until the fist stops moving back (-velocity / acceleration while
                  moving back and slowing down), depth and vertical (s; 1 = not turning)
  turn_depth, turn_dip
                  where it will turn around: depth (negative = behind the guard spot) and
                  dip (positive = below the guard spot)
Smoothing: a constant-acceleration Kalman filter per signal. It estimates position,
velocity and acceleration together from noisy keypoints, handles dropped frames and
uneven frame times, and uses no future frames.

Caveat, measured on the reviewed sessions: the cached flow divergence around the
wrist runs NEGATIVE while punches extend (median about -0.7 /s), the opposite of a
fist growing. The depth signals inherit that. They are tested as specified anyway;
the numbers say whether they help.

Rule (rule_scores): warn for a fist that is already behind its guard spot (depth)
or below it (dip), still moving that way but slowing down, and will turn around
within TURN_S further behind / below. A retraction that stops at the guard spot
doesn't fire. Variants:
'depth', 'dip', 'either'.
"""
import math
from pathlib import Path

import numpy as np

from combo_candidates import guard_spot

KC = 0.35
TURN_S = 0.15          # warn when the turnaround is this close (s)
BEHIND = 0.05          # turnaround this far behind guard (log size) counts as a pull-back
BELOW = 0.10           # turnaround this far below guard (shoulder widths) counts as a dip
NAMES = ("out_v", "out_a", "down_y", "down_v", "down_a", "depth", "depth_v", "depth_a",
         "turn_t_depth", "turn_t_dip", "turn_depth", "turn_dip")


def kalman_ca(t, z, r=0.02, q=400., reset=0.25):
    """Causal constant-acceleration Kalman filter of one signal.
    z: measurements (NaN = missing). r: measurement noise (units of z). q: jerk noise.
    Returns (position, velocity, acceleration); NaN before the first measurement and after
    gaps longer than `reset` s (the filter restarts there)."""
    n = len(z)
    out = np.full((n, 3), np.nan)
    x = None
    last_t = last_meas = None
    rr = r * r
    for i in range(n):
        zi = z[i]
        have = zi == zi            # not NaN
        if x is None or (last_meas is not None and t[i] - last_meas > reset):
            if not have:
                continue
            x = [zi, 0., 0.]
            P = [[rr, 0., 0.], [0., 4., 0.], [0., 0., 400.]]
            last_t = last_meas = t[i]
            out[i] = x
            continue
        dt = t[i] - last_t
        last_t = t[i]
        if dt > 0:
            d2 = dt * dt / 2
            x = [x[0] + dt * x[1] + d2 * x[2], x[1] + dt * x[2], x[2]]
            F = [[1., dt, d2], [0., 1., dt], [0., 0., 1.]]
            FP = [[sum(F[a][k] * P[k][b] for k in range(3)) for b in range(3)] for a in range(3)]
            P = [[sum(FP[a][k] * F[b][k] for k in range(3)) for b in range(3)] for a in range(3)]
            d3, d4, d5 = dt ** 3, dt ** 4, dt ** 5
            Q = [[d5 / 20, d4 / 8, d3 / 6], [d4 / 8, d3 / 3, dt * dt / 2], [d3 / 6, dt * dt / 2, dt]]
            P = [[P[a][b] + q * Q[a][b] for b in range(3)] for a in range(3)]
        if have:
            s = P[0][0] + rr
            k = [P[0][0] / s, P[1][0] / s, P[2][0] / s]
            y = zi - x[0]
            x = [x[0] + k[0] * y, x[1] + k[1] * y, x[2] + k[2] * y]
            P = [[P[a][b] - k[a] * P[0][b] for b in range(3)] for a in range(3)]
            last_meas = t[i]
        out[i] = x
    return out[:, 0], out[:, 1], out[:, 2]


def _turn(v, a, back):
    """Time until a fist moving `back` (v of that sign) stops, if it is slowing down."""
    slowing = (np.sign(v) == back) & (np.sign(a) == -back) & (np.abs(a) > 1e-6)
    return np.where(slowing, np.clip(-v / np.where(slowing, a, 1.), 0, 1.), 1.)


def per_wrist(t, kp, cf, div_rel=None):
    """(N, 2, len(NAMES)) signed features for wrists k=0 (COCO 9) and k=1 (COCO 10)."""
    n = len(t)
    sh = (cf[:, 5] > KC) & (cf[:, 6] > KC)
    width = np.linalg.norm(kp[:, 6] - kp[:, 5], axis=1)
    sh &= width > 10
    mid = (kp[:, 5] + kp[:, 6]) / 2
    out = np.full((n, 2, len(NAMES)), np.nan)
    for k, w in enumerate((9, 10)):
        ok = sh & (cf[:, w] > KC)
        rel = np.where(ok[:, None], (kp[:, w] - mid) / np.maximum(width, 1)[:, None], np.nan)
        x, vx, ax = kalman_ca(t, rel[:, 0])
        y, vy, ay = kalman_ca(t, rel[:, 1])
        p = np.stack([x, y], 1)
        g = guard_spot(t, p, np.isfinite(p).all(1))
        d = p - g
        dist = np.linalg.norm(d, axis=1)
        u = d / np.maximum(dist, 0.03)[:, None]            # unit direction away from guard
        out_v = (u * np.stack([vx, vy], 1)).sum(1)
        out_a = (u * np.stack([ax, ay], 1)).sum(1)
        down_y = d[:, 1]
        # looming: log size relative to guard = leaky running sum of divergence / 2
        depth = depth_v = depth_a = np.full(n, np.nan)
        if div_rel is not None:
            raw = np.full(n, np.nan)
            acc, at_rest_prev = 0., False
            for i in range(n):
                dv = div_rel[i, k]
                if not (dv == dv) or not ok[i]:
                    acc = 0. if not ok[i] else acc
                    raw[i] = acc if ok[i] else np.nan
                    continue
                dt = t[i] - t[i - 1] if i else 0.
                acc += 0.5 * dv * dt
                if dist[i] == dist[i] and dist[i] < 0.12 and abs(out_v[i]) < 0.8 and abs(dv) < 0.5:
                    acc *= math.exp(-dt / 0.3)              # resting at guard: forget slowly
                raw[i] = acc
            depth, depth_v, depth_a = kalman_ca(t, raw, r=0.03)
        t_depth = _turn(depth_v, depth_a, -1)              # moving back = shrinking
        t_dip = _turn(vy, ay, +1)                          # moving down (image y grows)
        with np.errstate(all="ignore"):
            turn_depth = np.where(t_depth < 1, depth + depth_v * t_depth / 2, depth)
            turn_dip = np.where(t_dip < 1, down_y + vy * t_dip / 2, down_y)
        out[:, k] = np.stack([out_v, out_a, down_y, vy, ay, depth, depth_v, depth_a,
                              t_depth, t_dip, turn_depth, turn_dip], 1)
    return out


def rule_scores(F):
    """Hand-written pull-back rule from per_wrist features: (N, 2) strength per variant
    (0 = no warning). Larger = deeper pull-back / dip."""
    g = {name: F[:, :, j] for j, name in enumerate(NAMES)}
    # the fist must already be behind / below its guard spot (half the margin), not just be
    # predicted to overshoot: filter lag makes a retraction look like it overshoots a little
    with np.errstate(invalid="ignore"):
        depth = np.where((g["turn_t_depth"] < TURN_S) & (g["turn_depth"] < -BEHIND) & (g["depth"] < -BEHIND / 2),
                         -g["turn_depth"], 0.)
        dip = np.where((g["turn_t_dip"] < TURN_S) & (g["turn_dip"] > BELOW) & (g["down_y"] > BELOW / 2),
                       g["turn_dip"], 0.)
    depth, dip = np.nan_to_num(depth), np.nan_to_num(dip)
    return {"depth": depth, "dip": dip, "either": np.maximum(depth / BEHIND, dip / BELOW)}


def for_session(npz):
    """per_wrist for a session file, with the cached flow divergence if there is one."""
    import flow_features
    npz = Path(npz)
    with np.load(npz, allow_pickle=False) as z:
        t, kp, cf = z["timestamps"].astype(float), z["keypoints"].astype(float), z["confidences"].astype(float)
    cache = Path("features") / f"{npz.stem}.flow{flow_features.TAG}.npz"
    div = np.load(cache)["div_rel"] if cache.exists() else None
    return per_wrist(t, kp, cf, div)
