"""
Punch telegraph prototype (laptop version).

Watches a webcam (or a recorded clip), finds the attacker's shoulders and
wrists with a tiny pose model, and beeps a cue the moment a wrist starts
moving fast:  DODGE LEFT / DODGE RIGHT / DUCK.

How it decides:
  1. Wrist position is measured relative to the attacker's shoulders
     (origin = shoulder midpoint, 1 unit = shoulder width). This cancels out
     camera motion, so it still works when the camera is on a moving hat.
  2. Wrist speed = change in that relative position per second (smoothed).
  3. If a wrist is above the speed threshold for N frames in a row -> cue.
  4. Direction comes from where that wrist is (image left/right/center).

Keys:  q/Esc quit | r start/stop recording a clip | [ / ] lower/raise
       threshold | space pause (clips only)

Examples:
  python punch_telegraph.py                      # webcam 0
  python punch_telegraph.py --source 1           # second webcam (e.g. C270)
  python punch_telegraph.py --source clips/x.mp4 # tune on a recorded clip
"""

import argparse
import csv
import os
import sys
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

# COCO keypoint indices
L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10

# cue name -> (beep Hz, beep ms, BGR color)
CUES = {
    "DODGE RIGHT": (1200, 120, (0, 200, 255)),
    "DODGE LEFT": (600, 120, (255, 160, 0)),
    "DUCK": (900, 300, (0, 0, 255)),
}


def beep(freq, ms):
    """Non-blocking beep so audio never slows the vision loop."""
    try:
        import winsound

        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()
    except ImportError:
        sys.stdout.write("\a")
        sys.stdout.flush()


class WristTrack:
    """Smoothed velocity of one wrist in shoulder-widths per second."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.prev = None
        self.prev_t = None
        self.v = np.zeros(2)
        self.hot = 0  # consecutive frames above threshold

    def update(self, p, t):
        if self.prev is not None and t > self.prev_t:
            raw = (p - self.prev) / (t - self.prev_t)
            self.v = self.alpha * raw + (1 - self.alpha) * self.v
        self.prev, self.prev_t = p, t
        return float(np.linalg.norm(self.v))


def pick_person(res):
    """Return keypoints of the biggest person (the attacker), or None."""
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(areas.argmax())
    xy = res.keypoints.xy[i].cpu().numpy()
    conf = (
        res.keypoints.conf[i].cpu().numpy()
        if res.keypoints.conf is not None
        else np.ones(len(xy))
    )
    return xy, conf


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
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=320, help="model input size (smaller = faster)")
    ap.add_argument("--thresh", type=float, default=5.0, help="wrist speed to cue, shoulder-widths/s")
    ap.add_argument("--confirm", type=int, default=2, help="frames above threshold before cueing")
    ap.add_argument("--side", type=float, default=0.6, help="how far off-center (shoulder widths) counts as left/right")
    ap.add_argument("--alpha", type=float, default=0.6, help="velocity smoothing, 1 = none")
    ap.add_argument("--kconf", type=float, default=0.35, help="min keypoint confidence")
    ap.add_argument("--cooldown", type=float, default=0.6, help="seconds between cues")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap, live = open_source(args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open source {args.source!r}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs("clips", exist_ok=True)
    log_f = open("cues.csv", "a", newline="")
    log = csv.writer(log_f)
    if log_f.tell() == 0:
        log.writerow(["wall_time", "source", "cue", "speed", "latency_ms"])

    tracks = {L_WR: WristTrack(args.alpha), R_WR: WristTrack(args.alpha)}
    writer = None
    last_cue_t = -1e9
    shown_cue, shown_until = None, 0.0
    frame_idx = 0
    paused = False
    infer_ms, loop_ms, latencies = [], [], []
    prev_loop = time.perf_counter()

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            t_cap = time.perf_counter()
            # clip time comes from frame count so tuning is repeatable
            t = t_cap if live else frame_idx / file_fps
            frame_idx += 1

            if writer is not None:
                writer.write(frame)

            t0 = time.perf_counter()
            res = model(frame, imgsz=args.imgsz, verbose=False)[0]
            infer_ms.append((time.perf_counter() - t0) * 1000)

            person = pick_person(res)
            speeds = {}
            cue = None
            if person is not None:
                xy, conf = person
                sw = np.linalg.norm(xy[L_SH] - xy[R_SH])
                if min(conf[L_SH], conf[R_SH]) > args.kconf and sw > 10:
                    mid = (xy[L_SH] + xy[R_SH]) / 2
                    cv2.line(frame, tuple(xy[L_SH].astype(int)), tuple(xy[R_SH].astype(int)), (200, 200, 200), 2)
                    best = None
                    for idx, tr in tracks.items():
                        if conf[idx] < args.kconf:
                            tr.reset()
                            continue
                        rel = (xy[idx] - mid) / sw
                        s = tr.update(rel, t)
                        speeds[idx] = s
                        tr.hot = tr.hot + 1 if s > args.thresh else 0
                        hotness = min(1.0, s / (args.thresh * 2))
                        color = (0, int(255 * (1 - hotness)), int(255 * hotness))
                        cv2.circle(frame, tuple(xy[idx].astype(int)), 8, color, -1)
                        if tr.hot >= args.confirm and (best is None or s > best[1]):
                            best = (rel, s)
                    if best is not None and t - last_cue_t > args.cooldown:
                        rel, s = best
                        if rel[0] < -args.side:
                            cue = "DODGE RIGHT"  # punch coming from image-left
                        elif rel[0] > args.side:
                            cue = "DODGE LEFT"  # punch coming from image-right
                        else:
                            cue = "DUCK"  # straight down the middle
                        last_cue_t = t
                        lat = (time.perf_counter() - t_cap) * 1000
                        latencies.append(lat)
                        freq, ms, _ = CUES[cue]
                        beep(freq, ms)
                        log.writerow([time.strftime("%H:%M:%S"), args.source, cue, f"{s:.1f}", f"{lat:.1f}"])
                        log_f.flush()
                        print(f"{cue:12s} speed={s:5.1f}  frame->cue {lat:5.1f} ms")
                        shown_cue, shown_until = cue, time.perf_counter() + 0.6
                else:
                    for tr in tracks.values():
                        tr.reset()
            else:
                for tr in tracks.values():
                    tr.reset()

            now = time.perf_counter()
            loop_ms.append((now - prev_loop) * 1000)
            prev_loop = now

            # overlay
            fps = 1000 / np.mean(loop_ms[-30:]) if loop_ms else 0
            hud = f"fps {fps:4.1f} | infer {infer_ms[-1]:4.1f} ms | thresh {args.thresh:.1f}"
            if speeds:
                hud += " | speed " + " ".join(f"{v:4.1f}" for v in speeds.values())
            cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if writer is not None:
                cv2.putText(frame, "REC", (frame.shape[1] - 70, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if shown_cue and time.perf_counter() < shown_until:
                _, _, color = CUES[shown_cue]
                cv2.putText(frame, shown_cue, (30, frame.shape[0] // 2), cv2.FONT_HERSHEY_DUPLEX, 2.0, color, 4)
            cv2.imshow("punch telegraph", frame)

        key = cv2.waitKey(1 if live else max(1, int(1000 / file_fps))) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("]"):
            args.thresh += 0.5
        if key == ord("["):
            args.thresh = max(0.5, args.thresh - 0.5)
        if key == ord(" ") and not live:
            paused = not paused
        if key == ord("r") and live:
            if writer is None:
                path = time.strftime("clips/clip_%Y%m%d_%H%M%S.mp4")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
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

    if infer_ms:
        print(f"\nframes {frame_idx} | avg infer {np.mean(infer_ms):.1f} ms | "
              f"avg fps {1000 / np.mean(loop_ms):.1f}")
    if latencies:
        print(f"cues {len(latencies)} | frame->cue avg {np.mean(latencies):.1f} ms, "
              f"p95 {np.percentile(latencies, 95):.1f} ms")


if __name__ == "__main__":
    main()
