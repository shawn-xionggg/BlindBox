"""
Streaming (live) version of the learned early punch detector.

Computes, frame by frame from the camera image + pose keypoints, exactly the
features early_dataset.py builds offline (verified by test_early_live.py),
then scores each wrist with models/early_detector.joblib.

    det = EarlyDetector("models/early_detector.joblib", budget=2.0)
    out = det.update(t, frame_bgr, kps, conf)
    out["scores"]  -> per-wrist punch probability (k=0: kp 9, k=1: kp 10)
    out["event"]   -> None or dict(time, wrist, side, score) when it buzzes

side: LEFT / RIGHT = the fist comes from that side of the image (a hook),
      BACK = a fist dropped first (uppercut wind-up), CENTER = straight.
"""
from collections import deque

import cv2
import joblib
import numpy as np

from flow_features import SCALE  # must match the scale the model was trained with
from launch_detector import LaunchDetector

L_SH, R_SH, WR = 5, 6, (9, 10)
KC = 0.35
COOLDOWN = 0.6
UPPER_DROP = 0.10   # shoulder widths below the rest spot -> uppercut -> STEP BACK


class _FeatureState:
    """Holds the rolling state needed to compute one frame's features causally."""

    def __init__(self, dt_est):
        self.win = max(3, int(round(0.6 / dt_est)))
        self.launch = LaunchDetector(thresh=1e9)       # only used for last_disp
        self.prev_t = None
        self.prev_rel = [None, None]                    # rel position at previous frame (if valid)
        self.rel_hist = [deque(maxlen=self.win), deque(maxlen=self.win)]  # (rel, valid, speed)
        self.e_hist = [deque(maxlen=self.win), deque(maxlen=self.win)]    # raw energy values
        self.prev_gray_full = None
        self.prev_gray_half = None
        self.last_pix = [None, None]                    # last known wrist pixel (full res)
        self.last_pix_half = [None, None]
        self.widths_ok = []                             # widths of shoulder-valid frames (for energy box fallback)
        self.width_hist = deque(maxlen=3)               # (t, raw width, sh)
        self.rows = deque(maxlen=3)                     # per-frame base features for lags

    def base_features(self, t, frame, kps, conf):
        kps, conf = np.asarray(kps, float), np.asarray(conf, float)
        width = float(np.linalg.norm(kps[R_SH] - kps[L_SH])) if np.isfinite(kps[[L_SH, R_SH]]).all() else np.nan
        sh = bool(conf[L_SH] > KC and conf[R_SH] > KC and np.isfinite(width) and width > 10)
        mid = (kps[L_SH] + kps[R_SH]) / 2
        dt = (t - self.prev_t) if self.prev_t is not None else np.nan
        if sh:
            self.widths_ok.append(width)
            if len(self.widths_ok) > 2000:
                self.widths_ok = self.widths_ok[-2000:]

        f = {}
        rel, wok, speed = [None, None], [False, False], [np.nan, np.nan]
        for k, w in enumerate(WR):
            wok[k] = sh and conf[w] > KC
            if wok[k]:
                rel[k] = (kps[w] - mid) / width
            if wok[k] and self.prev_rel[k] is not None and np.isfinite(dt):
                speed[k] = float(np.linalg.norm(rel[k] - self.prev_rel[k]) / dt)
        # disp from rest: median of calm valid frames among the previous `win` frames
        disp = [np.nan, np.nan]
        for k in range(2):
            if wok[k]:
                calm = [r for (r, v, s) in self.rel_hist[k] if v and (s if np.isfinite(s) else 99) < 1.5]
                if len(calm) >= 3:
                    disp[k] = float(np.linalg.norm(rel[k] - np.median(calm, axis=0)))
        # launch detector
        self.launch.update(t, kps, conf)
        launch = list(self.launch.last_disp)

        # --- motion energy (full-res blurred gray diff around last known wrist pixel) ---
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        for k, w in enumerate(WR):
            if conf[w] > KC and np.isfinite(kps[w]).all():
                self.last_pix[k] = kps[w].copy()
        box = (width if sh else (np.median(self.widths_ok) if self.widths_ok else np.nan)) * 0.5
        energy = [np.nan, np.nan]
        raw_e = [np.nan, np.nan]
        if self.prev_gray_full is not None and np.isfinite(box):
            diff = cv2.absdiff(gray, self.prev_gray_full)
            H, W = gray.shape
            for k in range(2):
                if not sh or self.last_pix[k] is None:
                    continue
                x, y = self.last_pix[k]
                r = int(max(20, box / 2))
                x0, x1 = int(max(0, x - r)), int(min(W, x + r))
                y0, y1 = int(max(0, y - r)), int(min(H, y + r))
                if x1 > x0 and y1 > y0:
                    raw_e[k] = float(diff[y0:y1, x0:x1].mean())
        for k in range(2):
            past = [v for v in self.e_hist[k] if np.isfinite(v)]
            if len(past) >= 3 and np.isfinite(raw_e[k]):
                energy[k] = raw_e[k] / (np.median(past) + 1e-3)
        self.prev_gray_full = gray

        # --- optical flow (half-size, Farneback) ---
        g = cv2.cvtColor(cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        for k, w in enumerate(WR):
            if conf[w] > KC and np.isfinite(kps[w]).all():
                self.last_pix_half[k] = kps[w] * SCALE
        div_rel, mag_rel, out_rel = [np.nan] * 2, [np.nan] * 2, [np.nan] * 2
        torso_div = torso_mag = np.nan
        sh_conf = bool(conf[L_SH] > KC and conf[R_SH] > KC)   # flow_features checks confidence only
        if self.prev_gray_half is not None and sh_conf and np.isfinite(dt) and dt > 0 and np.isfinite(width):
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray_half, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            u, v = flow[..., 0], flow[..., 1]
            dv = np.gradient(u, axis=1) + np.gradient(v, axis=0)
            H, W = g.shape
            wh = width * SCALE
            if wh > 5:
                midh = mid * SCALE

                def bm(a, cx, cy, r):
                    x0, x1 = int(max(0, cx - r)), int(min(W, cx + r + 1))
                    y0, y1 = int(max(0, cy - r)), int(min(H, cy + r + 1))
                    return None if (x1 - x0 < 3 or y1 - y0 < 3) else a[y0:y1, x0:x1]
                tr, tcy = 0.35 * wh, midh[1] + 0.6 * wh
                tu = bm(u, midh[0], tcy, tr)
                if tu is not None:
                    tfu, tfv = float(tu.mean()), float(bm(v, midh[0], tcy, tr).mean())
                    torso_div = float(bm(dv, midh[0], tcy, tr).mean()) / dt
                    torso_mag = np.hypot(tfu, tfv) / wh / dt
                else:
                    tfu = tfv = 0.0
                    torso_div = 0.0
                r = 0.35 * wh
                for k in range(2):
                    if self.last_pix_half[k] is None:
                        continue
                    cx, cy = self.last_pix_half[k]
                    bu = bm(u, cx, cy, r)
                    if bu is None:
                        continue
                    bv, bd = bm(v, cx, cy, r), bm(dv, cx, cy, r)
                    du, dvv = bu - tfu, bv - tfv
                    div_k = float(bd.mean()) / dt
                    div_rel[k] = div_k - torso_div
                    mag_rel[k] = float(np.hypot(du, dvv).mean()) / wh / dt
                    d = np.array([cx, cy]) - midh
                    d = d / (np.linalg.norm(d) + 1e-6)
                    out_rel[k] = float((du * d[0] + dvv * d[1]).mean()) / wh / dt
        self.prev_gray_half = g

        # width rate over 2 frames (raw widths, like offline)
        self.width_hist.append((t, width, sh))
        width_rate = np.nan
        if len(self.width_hist) == 3 and sh:
            t2, w2, _ = self.width_hist[0]
            w2 = w2 if np.isfinite(w2) else np.nan
            if t > t2:
                width_rate = (width - w2) / max(w2, 1) / (t - t2)

        # update histories
        for k in range(2):
            self.rel_hist[k].append((rel[k] if wok[k] else None, wok[k], speed[k]))
            self.e_hist[k].append(raw_e[k])
            self.prev_rel[k] = rel[k] if wok[k] else None
        self.prev_t = t

        row = dict(disp_rest=disp, launch=launch, speed=speed,
                   relx=[rel[k][0] if wok[k] else np.nan for k in range(2)],
                   rely=[rel[k][1] if wok[k] else np.nan for k in range(2)],
                   energy=energy, div_rel=div_rel, mag_rel=mag_rel, out_rel=out_rel,
                   torso_div=torso_div, torso_mag=torso_mag, width_rate=width_rate, sh=sh,
                   rel=rel)
        self.rows.append(row)
        return row


def _lagged(rows, name, L, k):
    if len(rows) <= L:
        return np.nan
    v = rows[-1 - L][name]
    return v[k] if isinstance(v, list) else v


def feature_matrix(rows, names):
    """(2, F) feature rows for the newest frame, in the trained column order."""
    X = np.full((2, len(names)), np.nan)
    for j, nm in enumerate(names):
        for k in range(2):
            if nm in ("torso_div", "torso_mag", "width_rate"):
                X[k, j] = rows[-1][nm]
            elif nm == "other_speed":
                X[k, j] = rows[-1]["speed"][1 - k]
            elif nm == "wrist_id":
                X[k, j] = float(k)
            elif "_l" in nm and nm.rsplit("_l", 1)[1].isdigit():
                base, L = nm.rsplit("_l", 1)
                X[k, j] = _lagged(rows, base, int(L), k)
            else:
                X[k, j] = _lagged(rows, nm, 0, k)
    return X


class EarlyDetector:
    def __init__(self, model_path="models/early_detector.joblib", budget=2.0, threshold=None, dt_est=0.048):
        b = joblib.load(model_path)
        if float(b.get("flow_scale", 0.5)) != SCALE:
            raise ValueError(f"{model_path} was trained with FLOW_SCALE={b.get('flow_scale', 0.5)}, "
                             f"but this process uses {SCALE}; set the FLOW_SCALE environment variable to match")
        self.model, self.names = b["model"], b["names"]
        if threshold is None:
            ths = b["thresholds"]
            key = min(ths, key=lambda x: abs(float(x) - budget))
            threshold = float(ths[key])
        self.threshold = threshold
        self.state = _FeatureState(dt_est)
        self.last_event = -np.inf
        self.rest_pos = [None, None]

    def features(self, t, frame, kps, conf):
        self.state.base_features(float(t), frame, kps, conf)
        return feature_matrix(self.state.rows, self.names)

    def update(self, t, frame, kps, conf):
        X = self.features(t, frame, kps, conf)
        row = self.state.rows[-1]
        scores = np.zeros(2)
        if row["sh"]:
            scores = self.model.predict_proba(X)[:, 1]
        # remember where each fist was resting (for the cue direction)
        for k in range(2):
            d = row["disp_rest"][k]
            if row["rel"][k] is not None and np.isfinite(d) and d < 0.15:
                self.rest_pos[k] = row["rel"][k].copy()
        event = None
        k = int(np.argmax(scores))
        if scores[k] > self.threshold and t - self.last_event >= COOLDOWN:
            self.last_event = t
            side = "CENTER"
            # Uppercut tell: the fist DROPS first (wind-up) without swinging sideways.
            # 2-3 frames after leaving guard this flags ~78% of uppercuts (vs 16% jabs,
            # 28% crosses, 39% hooks). Checked on both fists, because early on the
            # detector often fires on the wrong hand. Stepping back is safe against
            # every punch, so false flags are harmless; ducking into an uppercut isn't.
            for j in range(2):
                if self.rest_pos[j] is not None and row["rel"][j] is not None:
                    dx, dy = row["rel"][j] - self.rest_pos[j]   # dy > 0 = down
                    if dy > UPPER_DROP and dy > abs(dx):
                        side = "BACK"
            if side == "BACK":
                pass
            elif self.rest_pos[k] is not None and row["rel"][k] is not None:
                move = row["rel"][k] - self.rest_pos[k]
                if abs(move[0]) > 1.3 * abs(move[1]) and np.linalg.norm(move) > 0.15:
                    # side from the direction of travel: a fist moving rightward came from
                    # the left. A hook rotates the body, so even the other hand moves that way.
                    side = "LEFT" if move[0] > 0 else "RIGHT"
            event = dict(time=float(t), wrist=k, side=side, score=float(scores[k]))
        return dict(scores=scores, event=event)
