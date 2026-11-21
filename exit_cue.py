"""
Fist-leaves-the-frame cue (for hooks).

The C270's view is narrow (~55 deg diagonal), so a hook swings the fist out of
the side of the frame. Then the pose model loses the wrist and the learned
detector has nothing to score. But a fist racing toward the side edge, or
vanishing right after it, is itself strong evidence of a hook.

Streaming rule, per wrist (COCO kp 9 / 10):
  outward speed  v = change of the wrist's body-relative x toward the nearest
                 side edge, in shoulder widths per second. Body-relative, so the
                 wearer turning their head (whole image shifts) doesn't count.
  RACE  the wrist is inside the outer `band` of the image width, moving toward
        that edge with v > `speed` for `frames` frames in a row.
  LOST  the wrist was seen inside the band last frame moving outward with
        v > `lost_speed`, and this frame it's gone (confidence < kc).
A 0.6 s cooldown, like the other cues.

  python exit_cue.py              # replay on every reviewed session, print catches / false alarms
"""
import numpy as np

L_SH, R_SH, WR = 5, 6, (9, 10)


class ExitCue:
    def __init__(self, band=0.15, speed=4.0, frames=2, lost_speed=2.0, cooldown=0.6, kc=0.35):
        self.band, self.speed, self.frames, self.lost_speed = band, speed, frames, lost_speed
        self.cooldown, self.kc = cooldown, kc
        self.last_t = None
        self.mid = self.sw = None
        self.prev = [None, None]      # (relx, image-x fraction, outward speed) of last visible frame, or None
        self.run = [0, 0]
        self.last_fire = -1e9

    def update(self, t, kps, conf, width):
        dt = None if self.last_t is None else max(t - self.last_t, 1e-3)
        self.last_t = t
        if conf[L_SH] > self.kc and conf[R_SH] > self.kc and np.isfinite(kps[[L_SH, R_SH]]).all():
            sw = float(np.linalg.norm(kps[L_SH] - kps[R_SH]))
            if sw > 1:
                self.mid, self.sw = float((kps[L_SH, 0] + kps[R_SH, 0]) / 2), sw
        event = None
        for k, w in enumerate(WR):
            vis = conf[w] > self.kc and np.isfinite(kps[w]).all() and self.sw is not None
            prev = self.prev[k]
            if not vis:
                if prev is not None and event is None and self._edge(prev[1]) and prev[2] > self.lost_speed:
                    event = dict(time=t, wrist=k, side=self._edge(prev[1]), kind="LOST", speed=prev[2])
                self.prev[k], self.run[k] = None, 0
                continue
            relx, fx = (kps[w, 0] - self.mid) / self.sw, kps[w, 0] / width
            edge = self._edge(fx)
            v = 0.0
            if prev is not None and dt is not None and edge:
                v = (relx - prev[0]) / dt * (-1 if edge == "left" else 1)
            self.run[k] = self.run[k] + 1 if (edge and v > self.speed) else 0
            if self.run[k] >= self.frames and event is None:
                event = dict(time=t, wrist=k, side=edge, kind="RACE", speed=v)
            self.prev[k] = (relx, fx, v)
        if event and t - self.last_fire < self.cooldown:
            event = None
        if event:
            self.last_fire = t
        return event

    def _edge(self, fx):
        return "left" if fx < self.band else "right" if fx > 1 - self.band else None


def session_events(npz, width=640, **kw):
    """Replay the cue over a recorded session; returns frame indices where it fired."""
    z = np.load(npz, allow_pickle=True)
    kp, cf, t = z["keypoints"], z["confidences"], z["timestamps"]
    cue = ExitCue(**kw)
    return [i for i in range(len(t)) if cue.update(t[i], kp[i], cf[i], width)]


def main():
    from pathlib import Path
    import early_dataset
    from early_detector import score_events
    paths = sorted(p for p in Path("data").glob("*.npz") if p.with_suffix(".labels.json").exists())
    sessions = [early_dataset.build(p) for p in paths]
    for kw in (dict(), dict(band=0.12), dict(band=0.2), dict(speed=3.0), dict(speed=6.0),
               dict(lost_speed=1.0), dict(lost_speed=4.0), dict(frames=1)):
        caught, total, false, secs, fam = 0, 0, 0, 0.0, {}
        for p, s in zip(paths, sessions):
            ev = session_events(p, **kw)
            d, f, ns, _, _ = score_events(s, 0, len(s["t"]), ev)
            false += f
            secs += ns
            for e, x in zip(s["events"], d):
                hook = e["family"].startswith("HOOK")
                c = fam.setdefault("hooks" if hook else "other", [0, 0])
                c[0] += x is not None
                c[1] += 1
        print(f"{str(kw):24s} hooks caught {fam['hooks'][0]}/{fam['hooks'][1]}, "
              f"other punches {fam['other'][0]}/{fam['other'][1]}, false alarms {false / (secs / 60):.2f}/min")


if __name__ == "__main__":
    main()
