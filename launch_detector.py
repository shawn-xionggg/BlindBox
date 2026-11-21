"""
Launch-from-guard punch detector (streaming, past frames only).

A punch = a fist LEAVING a guard position it had been holding. Measured per
wrist, relative to the attacker's shoulders (origin = shoulder midpoint,
unit = shoulder width), so whole-body moves and camera motion cancel out.

Per wrist, every frame:
  1. rest  = median position over the last ~0.6 s of calm frames
  2. at rest = within --rest of that position and moving slower than --calm
  3. armed when the fist has been at rest for >= 0.15 s
  4. after it leaves rest, for up to 0.3 s: distance from the armed rest
     position > threshold  ->  PUNCH
Retractions never fire (the fist wasn't resting right before), and losing one
wrist doesn't blind the other.

Measured on 121 reviewed punches (diagnose_signals.py): the best pose signal so
far, ~43% caught within 4 frames (~200 ms) at 4 false alarms/min on scripted
don't-punch footage. Feints fire too: a feint IS a punch start.

Direction: mostly sideways motion -> hook from the side the fist started on
(LEFT/RIGHT = image side); otherwise CENTER (straight / uppercut).
"""
from collections import deque

import numpy as np

L_SH, R_SH, WRISTS = 5, 6, (9, 10)


class _Wrist:
    def __init__(self):
        self.hist = deque()          # (t, rel position, speed) of calm-or-not valid frames
        self.prev = None             # (t, rel position) of previous valid frame
        self.run = 0                 # consecutive at-rest frames
        self.run_start = None
        self.armed_rest = None
        self.armed_t = None          # time of the last held-rest frame
        self.leave_t = None          # time the fist first left the armed rest
        self.fired = False


class LaunchDetector:
    def __init__(self, thresh=0.55, rest=0.15, calm=1.5, hold=0.15, after=0.30,
                 window=0.6, cooldown=0.6, kconf=0.35):
        self.thresh, self.rest, self.calm = thresh, rest, calm
        self.hold, self.after, self.window = hold, after, window
        self.cooldown, self.kconf = cooldown, kconf
        self.w = [_Wrist(), _Wrist()]
        self.last_event = -np.inf
        self.last_disp = [0.0, 0.0]   # for display

    def reset(self):
        self.w = [_Wrist(), _Wrist()]

    def update(self, t, kps, conf):
        """kps (17,2) pixels, conf (17,). Returns an event dict or None."""
        t = float(t)
        self.last_disp = [0.0, 0.0]
        if kps is None or conf is None:
            return None
        kps, conf = np.asarray(kps, float), np.asarray(conf, float)
        if conf[L_SH] < self.kconf or conf[R_SH] < self.kconf:
            return None
        width = float(np.linalg.norm(kps[R_SH] - kps[L_SH]))
        if not np.isfinite(width) or width <= 10:
            return None
        mid = (kps[L_SH] + kps[R_SH]) / 2
        best = None
        for k, wi in enumerate(WRISTS):
            s = self.w[k]
            if conf[wi] < self.kconf or not np.isfinite(kps[wi]).all():
                s.run = 0
                continue
            p = (kps[wi] - mid) / width
            speed = 99.0
            frame_dt = 0.0
            if s.prev is not None and t > s.prev[0] and t - s.prev[0] < 0.25:
                frame_dt = t - s.prev[0]
                speed = float(np.linalg.norm(p - s.prev[1]) / frame_dt)
            s.prev = (t, p)
            # rest position from calm frames in the past window (excluding this one)
            while s.hist and t - s.hist[0][0] > self.window:
                s.hist.popleft()
            calm = [q for (_, q, sp) in s.hist if sp < self.calm]
            rest = np.median(calm, axis=0) if len(calm) >= 3 else None
            s.hist.append((t, p, speed))
            disp = float(np.linalg.norm(p - rest)) if rest is not None else np.inf
            at_rest = disp < self.rest and speed < self.calm * 1.5
            if at_rest:
                if s.run == 0:
                    s.run_start = t
                s.run += 1
                # hold length counts each rest frame's own duration
                if (t - s.run_start) + frame_dt >= self.hold - 1e-9 and s.run >= 2:
                    s.armed_rest, s.armed_t = rest, t
                    s.leave_t, s.fired = None, False
                continue
            s.run = 0
            if s.armed_rest is None or t - s.armed_t > self.after:
                continue
            if s.leave_t is None:
                s.leave_t = t
            d = float(np.linalg.norm(p - s.armed_rest))
            self.last_disp[k] = d
            if d > self.thresh and not s.fired:
                move = p - s.armed_rest
                if abs(move[0]) > 1.3 * abs(move[1]):
                    side = "LEFT" if s.armed_rest[0] < 0 else "RIGHT"
                else:
                    side = "CENTER"
                cand = dict(time=s.leave_t, detected_time=t, side=side, wrist=k, disp=d)
                if best is None or d > best["disp"]:
                    best = cand
        if best is not None:
            self.w[best["wrist"]].fired = True
            if t - self.last_event >= self.cooldown:
                self.last_event = t
                return best
        return None
