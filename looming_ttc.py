"""
Time-to-contact (looming) punch detector.

Idea: a fist coming at your face grows in the image. If its size is s and it
is growing at ds/dt, then

        time to contact  TTC = s / (ds/dt)

No distance, no calibration -- and it gets MORE accurate the closer the fist
is, so short range is a strength. (Locusts dodge predators with the same trick.)

How it works:
  1. Find colored gloves (or anything brightly colored on the fists) by color.
     Click on a glove in the window to teach it the color.
  2. Size s = sqrt(blob area). Since s ~ 1/distance, 1/s falls in a straight
     line as the fist approaches. Fit a line to 1/s over the last few frames;
     TTC = time until that line hits zero (exact for steady speed, and much
     less noisy than a 2-frame difference).
  3. Cue when TTC < --ttc for --confirm frames in a row.
  4. Direction from where the glove is in the frame (camera is on the dodger's
     head, so frame center = their face):
        left third -> DODGE RIGHT, right third -> DODGE LEFT, middle -> DUCK

Keys:  click = set glove color | q/Esc quit | [ / ] lower/raise TTC threshold
       m toggle mask view | r record clip | space pause (clips only)

Examples:
  python looming_ttc.py                        # laptop webcam
  python looming_ttc.py --color red            # start with a red preset
  python looming_ttc.py --source clips/x.mp4   # tune on a recorded clip
"""

import argparse
import csv
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

# cue name -> (beep Hz, beep ms, BGR color)
CUES = {
    "DODGE RIGHT": (1200, 120, (0, 200, 255)),
    "DODGE LEFT": (600, 120, (255, 160, 0)),
    "DUCK": (900, 300, (0, 0, 255)),
}

# HSV presets (OpenCV hue is 0-179)
PRESETS = {"red": 0, "orange": 12, "yellow": 28, "green": 65, "blue": 110, "pink": 165}


def beep(freq, ms):
    try:
        import winsound

        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()
    except ImportError:
        sys.stdout.write("\a")
        sys.stdout.flush()


class ColorModel:
    """Hue band +/- tolerance, with minimum saturation and brightness."""

    def __init__(self, hue=None, hue_tol=12, s_min=90, v_min=60):
        self.hue, self.hue_tol, self.s_min, self.v_min = hue, hue_tol, s_min, v_min

    def sample(self, hsv, x, y):
        patch = hsv[max(0, y - 4): y + 5, max(0, x - 4): x + 5].reshape(-1, 3)
        h, s, v = np.median(patch, axis=0)
        self.hue = int(h)
        self.s_min = int(max(60, s - 70))
        self.v_min = int(max(40, v - 80))
        print(f"glove color set: hue={self.hue} s>={self.s_min} v>={self.v_min}")

    def mask(self, hsv):
        if self.hue is None:
            return None
        lo, hi = self.hue - self.hue_tol, self.hue + self.hue_tol
        if lo < 0:  # hue wraps around 0/179 (reds)
            bands = [(0, hi), (lo + 180, 179)]
        elif hi > 179:
            bands = [(lo, 179), (0, hi - 180)]
        else:
            bands = [(lo, hi)]
        m = None
        for a, b in bands:
            part = cv2.inRange(hsv, (a, self.s_min, self.v_min), (b, 255, 255))
            m = part if m is None else cv2.bitwise_or(m, part)
        k = np.ones((5, 5), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m


class GloveTrack:
    """Size history of one glove -> growth rate -> time to contact."""

    def __init__(self, c, s, t, hist):
        self.c, self.seen, self.hot = c, t, 0
        self.hist = deque(maxlen=hist)
        self.hist.append((t, s))

    def add(self, c, s, t):
        self.c, self.seen = c, t
        self.hist.append((t, s))

    def ttc(self):
        """Return (TTC seconds or inf, current size)."""
        s_now = self.hist[-1][1]
        if len(self.hist) < 3:
            return float("inf"), s_now
        ts = np.array([h[0] for h in self.hist])
        ss = np.array([h[1] for h in self.hist])
        if ts[-1] - ts[0] <= 0:
            return float("inf"), s_now
        # size ~ 1/distance, so 1/size falls in a straight line for a fist moving
        # at steady speed. Fit that line; TTC = when it reaches zero.
        slope, inv_now = np.polyfit(ts - ts[-1], 1.0 / ss, 1)
        if slope >= 0 or inv_now <= 0:
            return float("inf"), s_now
        return inv_now / -slope, s_now


def find_blobs(mask, min_area, k=2):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:k]:
        a = cv2.contourArea(c)
        if a < min_area:
            break
        m = cv2.moments(c)
        out.append((np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]), float(np.sqrt(a))))
    return out


def open_source(src):
    if src.isdigit():
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(src), backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, True
    return cv2.VideoCapture(src), False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--color", choices=PRESETS, help="start with a color preset instead of clicking")
    ap.add_argument("--ttc", type=float, default=0.35, help="cue when time-to-contact below this (s)")
    ap.add_argument("--confirm", type=int, default=2, help="frames below TTC threshold before cueing")
    ap.add_argument("--hist", type=int, default=5, help="frames used to fit growth rate")
    ap.add_argument("--min-area", type=int, default=400, help="ignore blobs smaller than this (px^2)")
    ap.add_argument("--side", type=float, default=0.18, help="off-center fraction of width that counts as left/right")
    ap.add_argument("--cooldown", type=float, default=0.6, help="seconds between cues")
    args = ap.parse_args()

    cap, live = open_source(args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open source {args.source!r}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    color = ColorModel(PRESETS.get(args.color))
    state = {"hsv": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["hsv"] is not None:
            color.sample(state["hsv"], x, y)

    win = "looming ttc"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    os.makedirs("clips", exist_ok=True)
    log_f = open("cues_ttc.csv", "a", newline="")
    log = csv.writer(log_f)
    if log_f.tell() == 0:
        log.writerow(["wall_time", "source", "cue", "ttc_ms", "size_px", "latency_ms"])

    tracks = []
    writer = None
    show_mask = False
    paused = False
    last_cue_t = -1e9
    shown_cue, shown_until = None, 0.0
    frame_idx = 0
    proc_ms, loop_ms, latencies = [], [], []
    prev_loop = time.perf_counter()
    frame = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            t_cap = time.perf_counter()
            t = t_cap if live else frame_idx / file_fps
            frame_idx += 1
            if writer is not None:
                writer.write(frame)

            t0 = time.perf_counter()
            hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
            state["hsv"] = hsv
            mask = color.mask(hsv)
            h, w = frame.shape[:2]

            # --- track gloves by nearest centroid ---
            blobs = find_blobs(mask, args.min_area) if mask is not None else []
            used = set()
            for c, s in blobs:
                best, best_d = None, 0.25 * w
                for i, tr in enumerate(tracks):
                    d = np.linalg.norm(tr.c - c)
                    if i not in used and d < best_d:
                        best, best_d = i, d
                if best is None:
                    tracks.append(GloveTrack(c, s, t, args.hist))
                    used.add(len(tracks) - 1)
                else:
                    tracks[best].add(c, s, t)
                    used.add(best)
            tracks = [tr for tr in tracks if t - tr.seen < 0.2]

            # --- time to contact + cue ---
            cue, cue_info = None, None
            for tr in tracks:
                ttc, s = tr.ttc()
                fresh = tr.seen == t
                tr.hot = tr.hot + 1 if (fresh and ttc < args.ttc) else 0
                x, y = tr.c.astype(int)
                hot = ttc < args.ttc
                col = (0, 0, 255) if hot else (0, 255, 0)
                cv2.circle(frame, (x, y), int(s / 2), col, 2)
                label = f"{ttc * 1000:.0f} ms" if ttc < 5 else "--"
                cv2.putText(frame, label, (x - 30, y - int(s / 2) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                if tr.hot >= args.confirm and (cue_info is None or ttc < cue_info[0]):
                    cue_info = (ttc, s, tr.c[0] / w)
            proc_ms.append((time.perf_counter() - t0) * 1000)

            if cue_info is not None and t - last_cue_t > args.cooldown:
                ttc, s, fx = cue_info
                if fx < 0.5 - args.side:
                    cue = "DODGE RIGHT"  # glove coming from image-left
                elif fx > 0.5 + args.side:
                    cue = "DODGE LEFT"  # glove coming from image-right
                else:
                    cue = "DUCK"  # straight at the face
                last_cue_t = t
                lat = (time.perf_counter() - t_cap) * 1000
                latencies.append(lat)
                freq, ms, _ = CUES[cue]
                beep(freq, ms)
                log.writerow([time.strftime("%H:%M:%S"), args.source, cue, f"{ttc * 1000:.0f}", f"{s:.0f}", f"{lat:.1f}"])
                log_f.flush()
                print(f"{cue:12s} impact in {ttc * 1000:4.0f} ms  size={s:4.0f}px  frame->cue {lat:4.1f} ms")
                shown_cue, shown_until = cue, time.perf_counter() + 0.6

            now = time.perf_counter()
            loop_ms.append((now - prev_loop) * 1000)
            prev_loop = now

            # --- overlay ---
            view = frame
            if show_mask and mask is not None:
                view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for xl in (0.5 - args.side, 0.5 + args.side):
                cv2.line(view, (int(xl * w), 0), (int(xl * w), h), (90, 90, 90), 1)
            fps = 1000 / np.mean(loop_ms[-30:]) if loop_ms else 0
            hud = f"fps {fps:4.1f} | proc {proc_ms[-1]:4.1f} ms | cue if TTC < {args.ttc * 1000:.0f} ms"
            cv2.putText(view, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if color.hue is None:
                cv2.putText(view, "CLICK ON A GLOVE to set its color", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if writer is not None:
                cv2.putText(view, "REC", (w - 70, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if shown_cue and time.perf_counter() < shown_until:
                cv2.putText(view, shown_cue, (30, h // 2), cv2.FONT_HERSHEY_DUPLEX, 2.0, CUES[shown_cue][2], 4)
            cv2.imshow(win, view)

        key = cv2.waitKey(1 if live else max(1, int(1000 / file_fps))) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("]"):
            args.ttc += 0.05
        if key == ord("["):
            args.ttc = max(0.05, args.ttc - 0.05)
        if key == ord("m"):
            show_mask = not show_mask
        if key == ord(" ") and not live:
            paused = not paused
        if key == ord("r") and live and frame is not None:
            if writer is None:
                path = time.strftime("clips/clip_%Y%m%d_%H%M%S.mp4")
                fh, fw = frame.shape[:2]
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (fw, fh))
                print("recording ->", path)
            else:
                writer.release()
                writer = None
                print("recording stopped")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    log_f.close()
    if proc_ms:
        print(f"\nframes {frame_idx} | avg processing {np.mean(proc_ms):.1f} ms | avg fps {1000 / np.mean(loop_ms):.1f}")
    if latencies:
        print(f"cues {len(latencies)} | frame->cue avg {np.mean(latencies):.1f} ms, p95 {np.percentile(latencies, 95):.1f} ms")


if __name__ == "__main__":
    main()
