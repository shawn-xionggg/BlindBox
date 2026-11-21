"""
Blindfolded-dodge demo: camera -> pose -> learned early punch detector -> beep.

Default cues: DODGE LEFT / DODGE RIGHT (a hook, judged by which way the fist
travels), STEP BACK (a fist drops first = uppercut wind-up), DUCK (anything else,
e.g. jab/cross). See early_live.py. --universal plays one INCOMING beep instead.

  python dodge_live.py --source 1                    # C270 is camera 1 on this laptop
  python dodge_live.py --budget 4                    # more sensitive (more false alarms)
  python dodge_live.py --source data/x.mp4 --headless --npz data/x.npz
        # replay a recorded session without camera/window, using its saved
        # pose (fast, exact) -- prints when it would have beeped
Keys: q/Esc quit.

Measured offline (cross-validated, 121 reviewed punches, ~20 fps):
  budget 2 -> ~38% of punches caught within ~100 ms, ~58% within ~200 ms, ~2.6 false/min
  budget 4 -> ~62% within ~100 ms, ~77% within ~200 ms, ~5.5 false/min
The model was trained at ~20 fps; at very different frame rates retrain it.
"""
import argparse
import csv
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from early_live import EarlyDetector

SAFETY = "Non-contact game: the performer stays beyond the wearer's reach."
DIRECTIONAL = {"LEFT": ("DODGE RIGHT", 1200, 120), "RIGHT": ("DODGE LEFT", 600, 120), "CENTER": ("DUCK", 900, 250),
               "BACK": ("STEP BACK", 400, 350)}   # uppercut: ducking would move into it
UNIVERSAL = ("INCOMING", 1000, 150)


def beep(freq, ms):
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()
    except ImportError:
        print("\a", end="", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="1")
    ap.add_argument("--model", default="models/early_detector.joblib")
    ap.add_argument("--budget", type=float, default=2.0, help="false alarms/min the threshold was tuned for (1, 2 or 4)")
    ap.add_argument("--threshold", type=float, default=None, help="override the probability threshold")
    ap.add_argument("--pose", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--directional", action="store_true", default=True,
                    help="DODGE LEFT / DODGE RIGHT / DUCK / STEP BACK cues (default)")
    ap.add_argument("--universal", dest="directional", action="store_false",
                    help="one INCOMING beep for every punch instead of direction cues")
    ap.add_argument("--ipc", action="store_true", help="send cues and per-frame heartbeats to portable C cue_driver")
    ap.add_argument("--no-beep", action="store_true", help="disable the local beep")
    ap.add_argument("--max-seconds", type=float, help="stop after this many seconds of source footage")
    ap.add_argument("--log", default="dodge_log.csv", help="cue CSV output path")
    ap.add_argument("--headless", action="store_true", help="no window or sound (for replaying videos)")
    ap.add_argument("--exit-cue", action="store_true",
                    help="experimental: also beep when a fist races out of the side of the frame (exit_cue.py; "
                         "measured +1/59 hooks for +0.2 false alarms/min, so off by default)")
    ap.add_argument("--npz", help="with a video source: use this session's saved pose instead of running the model")
    args = ap.parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        ap.error("--max-seconds must be positive")
    ipc = None
    if args.ipc:
        from cue_client import CueClient, capture_ns, monotonic_ns
        from cue_client import CUE_INCOMING, CUE_LEFT, CUE_RIGHT, CUE_DUCK
        ipc = CueClient()

    live = args.source.isdigit()
    # load arrays once (indexing an NpzFile re-decompresses the whole array every time)
    z = dict(np.load(args.npz, allow_pickle=False)) if args.npz else None
    runner = None
    if z is None:
        from pose_features import PoseRunner
        runner = PoseRunner(args.pose, args.imgsz)
    if live:
        from collect import open_source
        from live_runtime import LatestCapture
        cap, _ = open_source(args.source)
        capture = LatestCapture(cap)
    else:
        cap = cv2.VideoCapture(args.source)
        capture = None
    det = EarlyDetector(args.model, budget=args.budget, threshold=args.threshold,
                        dt_est=float(np.median(np.diff(z["timestamps"]))) if z is not None else 0.048)
    exit_cue = None
    if args.exit_cue:
        from exit_cue import ExitCue
        exit_cue = ExitCue()
    print(SAFETY)
    print(f"threshold {det.threshold:.3f} (budget {args.budget:g}/min) | cue: {'directional' if args.directional else 'universal INCOMING'}")

    log_path = Path(args.log)
    new = not log_path.exists()
    log_f = log_path.open("a", newline="")
    log = csv.writer(log_f)
    if new:
        log.writerow(["run", "time_s", "cue", "wrist", "score", "proc_ms"])
    run = time.strftime("%Y%m%d_%H%M%S")
    start = time.perf_counter()
    i, proc, events, shown = 0, [], [], ("", -1.0)
    try:
        while True:
            if capture is not None:
                frame, stamp = capture.read()
                if frame is None:
                    break
                t = stamp - start
                if ipc is not None:
                    t_capture_ns = capture_ns(stamp)
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                t = float(z["timestamps"][i]) if z is not None and i < len(z["timestamps"]) else i / 30.0
                if ipc is not None:
                    t_capture_ns = monotonic_ns()  # decoded frame receipt, not media time
            if args.max_seconds is not None and t >= args.max_seconds:
                break
            t0 = time.perf_counter()
            if z is not None:
                if i >= len(z["timestamps"]):
                    break
                kps, conf = z["keypoints"][i], z["confidences"][i]
            else:
                person = runner(frame)
                kps, conf = person if person is not None else (np.full((17, 2), np.nan), np.zeros(17))
            out = det.update(t, frame, kps, conf)
            if ipc is not None:
                ipc.send_heartbeat()  # exactly the processed-frame liveness signal
            ms = (time.perf_counter() - t0) * 1000
            proc.append(ms)
            ev = out["event"]
            if exit_cue is not None:
                xe = exit_cue.update(t, kps, conf, frame.shape[1])
                if xe and not ev and (not events or t - events[-1][0] >= 0.6):
                    ev = dict(time=t, wrist=xe["wrist"], side=xe["side"].upper(), score=1.0)
            if ev:
                name, freq, dur = DIRECTIONAL[ev["side"]] if args.directional else UNIVERSAL
                if not args.headless and not args.no_beep:
                    beep(freq, dur)
                if ipc is not None:
                    # STEP BACK has no code of its own: CUE_INCOMING already means "step back"
                    direction = {"DODGE LEFT": CUE_LEFT, "DODGE RIGHT": CUE_RIGHT,
                                 "DUCK": CUE_DUCK}.get(name, CUE_INCOMING)
                    try:
                        status = ipc.send_cue(direction, ev["score"], t_capture_ns, i)
                        if status:
                            print(f"IPC cue rejected frame={i} status={status}", flush=True)
                    except OSError as exc:
                        print(f"IPC cue failed frame={i}: {exc}", flush=True)
                events.append((t, name, ev))
                shown = (name, t + 0.6)
                log.writerow([run, f"{t:.3f}", name, ev["wrist"], f"{ev['score']:.3f}", f"{ms:.1f}"])
                print(f"{t:8.2f}s  {name:12s} wrist {ev['wrist']}  p={ev['score']:.2f}  ({ms:.0f} ms to decide)", flush=True)
            if not args.headless:
                for k, s in enumerate(out["scores"]):
                    y = 30 + 26 * k
                    cv2.putText(frame, f"wrist {k} {s:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    cv2.rectangle(frame, (130, y - 12), (330, y), (80, 80, 80), 1)
                    cv2.rectangle(frame, (130, y - 12), (130 + int(200 * s), y), (0, 200, 255), -1)
                    tx = 130 + int(200 * det.threshold)
                    cv2.line(frame, (tx, y - 14), (tx, y + 2), (0, 0, 255), 2)
                if t < shown[1]:
                    cv2.putText(frame, shown[0], (30, frame.shape[0] // 2), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255), 3)
                cv2.putText(frame, f"{np.mean(proc[-30:]):.0f} ms/frame | q quit", (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("dodge", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            i += 1
    finally:
        if ipc is not None:
            ipc.close()
        if capture is not None:
            capture.close()
        else:
            cap.release()
        cv2.destroyAllWindows()
        log_f.close()
    if proc:
        print(f"\n{i} frames | detector {np.mean(proc):.1f} ms/frame (p95 {np.percentile(proc, 95):.1f}) | "
              f"{len(events)} cues -> {log_path}")


if __name__ == "__main__":
    main()
