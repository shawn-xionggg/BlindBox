"""Causal pose features and shared streaming onset/cue logic. Times are seconds.

Image LEFT/RIGHT describe the attacker's position in the image, not anatomy.
Session format: timestamps[N], keypoints[N,17,2], confidences[N,17],
onset_frames[M], onset_sides[M], onset_sources[M], rejected_frames[K].
"""
from collections import deque
from dataclasses import dataclass, field
import numpy as np

SIDES = ('NONE', 'LEFT', 'RIGHT', 'CENTER')
JOINTS = (0, 5, 6, 7, 8, 9, 10, 11, 12)
NAMES = ('nose', 'left_shoulder', 'right_shoulder', 'left_elbow',
         'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip')
KCONF = .35
FAMILY_CUES = {'JAB': 'DUCK', 'CROSS': 'DUCK', 'HOOK_L': 'DODGE RIGHT',
               'HOOK_R': 'DODGE LEFT', 'UPPERCUT': 'BACK'}
NEGATIVES = ('FEINT', 'STEP_IN', 'HEAD_TURN', 'NOD', 'GLOVE_ADJUST', 'IDLE', 'GUARD_SHIFT')
COMMANDS = tuple(FAMILY_CUES) + NEGATIVES
SAFETY = "Non-contact game: the performer stays beyond the wearer's reach."



class PoseRunner:
    def __init__(self, model='yolo11n-pose.pt', imgsz=320):
        from ultralytics import YOLO
        self.model, self.imgsz = YOLO(model), imgsz

    def __call__(self, frame):
        r = self.model(frame, imgsz=self.imgsz, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or not len(r.boxes):
            return None
        b = r.boxes.xyxy.cpu().numpy()
        i = np.argmax((b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1]))
        return (r.keypoints.xy[i].cpu().numpy(),
                r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None
                else np.ones(17))


def normalize(kps):
    kps = np.asarray(kps, dtype=float)
    width = np.linalg.norm(kps[6]-kps[5])
    if not np.isfinite(width) or width < 1e-6:
        return np.full((17, 2), np.nan)
    return (kps-(kps[5]+kps[6])/2)/width


@dataclass
class PoseFrame:
    t: float
    xy: np.ndarray
    mask: np.ndarray
    width: float
    angle: float
    glove_area: np.ndarray = field(default_factory=lambda: np.zeros(2))
    glove_mask: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))
    unknown: bool = False


class FeatureStream:
    """Impute in normalized coordinates; never turn camera movement into velocity."""
    def __init__(self, window=8):
        if window < 2:
            raise ValueError('window must be >= 2')
        self.history = deque(maxlen=window)
        self.last = np.zeros((17, 2))
        self.center = None
        self.previous_width = None
        self.gloves = np.zeros(2)
        self.unknown = False

    def update(self, t, kps=None, conf=None, blobs=None):
        was_unknown = self.unknown
        mask = np.zeros(17, dtype=bool)
        width = self.history[-1].width if self.history else 0.
        angle = self.history[-1].angle if self.history else 0.
        if kps is not None:
            kps = np.asarray(kps)
            conf = np.ones(17) if conf is None else np.asarray(conf)
            mask = (conf >= KCONF) & np.isfinite(kps).all(axis=1)
            if mask[5] and mask[6] and np.linalg.norm(kps[6]-kps[5]) > 10:
                xy = normalize(kps)
                self.last[mask] = xy[mask]
                delta = kps[6]-kps[5]
                width, angle = float(np.linalg.norm(delta)), float(np.arctan2(delta[1], delta[0]))
            else:
                mask[:] = False
        valid = bool(mask[5] and mask[6] and mask[9] and mask[10])
        center = (kps[5]+kps[6])/2 if valid else None
        jump = valid and self.center is not None and (
            np.linalg.norm(center-self.center)/max(self.previous_width, 1) > .5
            or abs(width/self.previous_width-1) > .30)
        self.unknown = not valid or bool(jump)
        if self.unknown:
            self.history.clear()  # No derivatives may span a dropout or identity jump.
            self.gloves[:] = 0
            mask[:] = False
        if was_unknown:
            self.history.clear()
        self.center = center if valid else None
        self.previous_width = width if valid else None
        gm = np.zeros(2, dtype=bool)
        if valid and not jump and blobs is not None:
            # Each blob goes to its nearest wrist; do not reuse a wrist.
            for blob in np.asarray(blobs).reshape(-1, 3):
                if not np.isfinite(blob).all() or blob[2] <= 0:
                    continue
                j = int(np.argmin(np.linalg.norm(kps[[9, 10]]-blob[:2], axis=1)))
                if not gm[j]:
                    self.gloves[j] = blob[2]/width**2
                    gm[j] = True
        self.history.append(PoseFrame(float(t), self.last.copy(), mask.copy(), width, angle,
                                      self.gloves.copy(), gm, self.unknown))
        return frame_features(self.history)


def _angles(xy):
    out = []
    for a, b, c in ((5, 7, 9), (6, 8, 10)):
        u, v = xy[a]-xy[b], xy[c]-xy[b]
        out.append(np.arccos(np.clip(np.dot(u, v)/max(np.linalg.norm(u)*np.linalg.norm(v), 1e-9), -1, 1)))
    return np.array(out)


def feature_names():
    return ([f'{n}_{a}' for n in NAMES for a in ('x', 'y')]
            + [f'{n}_v{a}' for n in NAMES for a in ('x', 'y')]
            + [f'{n}_observed' for n in NAMES]
            + ['shoulder_angle', 'shoulder_angle_rate', 'shoulder_width_px', 'width_relative_change',
               'left_elbow_angle', 'right_elbow_angle', 'left_elbow_rate', 'right_elbow_rate',
               'left_guard_height', 'right_guard_height', 'lean_x', 'lean_y']
            + [f'glove_{j}_{n}' for n in ('area', 'rate', 'log_rate', 'observed') for j in ('left', 'right')]
            + ['hips_observed'])


def frame_features(history):
    """History contains normalized PoseFrames; all differences use elapsed time."""
    if not history:
        return np.zeros(len(feature_names()))
    a, b = history[0], history[-1]
    dt = max(b.t-a.t, 1e-6)
    ids = list(JOINTS)
    angles = _angles(b.xy)
    hips = b.mask[11] and b.mask[12]
    lean = -(b.xy[11]+b.xy[12])/2
    da = np.arctan2(np.sin(b.angle-a.angle), np.cos(b.angle-a.angle))
    out = np.r_[b.xy[ids].ravel(), ((b.xy[ids]-a.xy[ids])/dt).ravel(), b.mask[ids],
                b.angle, da/dt, b.width, (b.width-a.width)/max(a.width, 1),
                angles, (angles-_angles(a.xy))/dt,
                b.xy[9, 1]-b.xy[0, 1], b.xy[10, 1]-b.xy[0, 1], lean,
                b.glove_area, (b.glove_area-a.glove_area)/dt,
                np.where(a.glove_mask & b.glove_mask,
                         np.log(np.maximum(b.glove_area, 1e-9)/np.maximum(a.glove_area, 1e-9))/dt, 0),
                b.glove_mask, hips]
    return np.nan_to_num(out)


def side_from_x(x):
    return 'LEFT' if x < -.6 else 'RIGHT' if x > .6 else 'CENTER'


class ReactiveOnset:
    def __init__(self, thresh=5., lower=2., confirm=2, cooldown=.6, alpha=.6):
        self.thresh, self.lower, self.confirm = thresh, lower, confirm
        self.cooldown, self.alpha = cooldown, alpha
        self.last_event = -np.inf
        self.tracks = {}

    def update(self, frame, index):
        if frame.unknown:
            self.tracks.clear()
            return None
        candidates = []
        for wrist in (9, 10):
            if not (frame.mask[5] and frame.mask[6] and frame.mask[wrist]):
                self.tracks.pop(wrist, None)
                continue
            p = frame.xy[wrist]
            old = self.tracks.get(wrist)
            v, hot, start = np.zeros(2), 0, None
            if old is not None and frame.t > old['t']:
                v = self.alpha*(p-old['p'])/(frame.t-old['t'])+(1-self.alpha)*old['v']
                speed = np.linalg.norm(v)
                start = (old['start'] or (index, frame.t)) if speed > self.lower else None
                hot = old['hot']+1 if speed > self.thresh else 0
                if hot >= self.confirm and start is not None:
                    candidates.append((speed, start, side_from_x(p[0])))
            self.tracks[wrist] = dict(p=p.copy(), t=frame.t, v=v, hot=hot, start=start)
        if candidates and frame.t-self.last_event >= self.cooldown:
            speed, (idx, t), side = max(candidates)
            self.last_event = frame.t
            return dict(frame=idx, time=t, side=side, detected_time=frame.t, speed=float(speed))
        return None


def reactive_onset(timestamps, keypoints, confidences, **kwargs):
    stream, detector, events = FeatureStream(), ReactiveOnset(**kwargs), []
    for i, (t, xy, conf) in enumerate(zip(timestamps, keypoints, confidences)):
        stream.update(t, xy, conf)
        event = detector.update(stream.history[-1], i)
        if event:
            events.append(event)
    return events


class CueGate:
    def __init__(self, threshold=.5, confirm=2, cooldown=.6):
        self.threshold, self.confirm, self.cooldown = threshold, confirm, cooldown
        self.side, self.count, self.last = None, 0, -np.inf

    def update(self, t, probabilities):
        side = max((s for s in probabilities if s != 'NONE'), key=probabilities.get, default=None)
        if side is None or probabilities[side] <= self.threshold:
            self.side, self.count = None, 0
            return None
        self.count = self.count+1 if side == self.side else 1
        self.side = side
        if self.count >= self.confirm and t-self.last >= self.cooldown:
            self.last = t
            return dict(time=float(t), side=side)
        return None

class GloveOnset:
    """Log relative-area growth; missing observations break confirmation."""
    def __init__(self, thresh=3., lower=1., confirm=2, cooldown=.6):
        self.thresh, self.lower, self.confirm, self.cooldown = thresh, lower, confirm, cooldown
        self.tracks = {}
        self.last_event = -np.inf

    def update(self, frame, index):
        if frame.unknown:
            self.tracks.clear()
            return None
        events = []
        for j in range(2):
            if not frame.glove_mask[j]:
                self.tracks.pop(j, None)
                continue
            old = self.tracks.get(j)
            area = frame.glove_area[j]
            hot, start, rate = 0, None, 0.
            if old and frame.t > old['t']:
                rate = np.log(area/old['area'])/(frame.t-old['t'])
                start = (old['start'] or (index, frame.t)) if rate > self.lower else None
                hot = old['hot']+1 if rate > self.thresh else 0
                if hot >= self.confirm and start:
                    events.append(dict(frame=start[0], time=start[1], side=side_from_x(frame.xy[9+j, 0]),
                                       detected_time=frame.t, speed=float(rate), signal='glove'))
            self.tracks[j] = dict(t=frame.t, area=area, hot=hot, start=start)
        if events and frame.t-self.last_event >= self.cooldown:
            self.last_event = frame.t
            return min(events, key=lambda e: e['time'])
        return None


class MotionOnset:
    def __init__(self, thresh=5., glove_thresh=3., **kwargs):
        self.wrist = ReactiveOnset(thresh=thresh, **kwargs)
        self.glove = GloveOnset(thresh=glove_thresh,
                                confirm=kwargs.get('confirm', 2), cooldown=kwargs.get('cooldown', .6))
        self.last_event = -np.inf

    def update(self, frame, index):
        events = [e for e in (self.wrist.update(frame, index), self.glove.update(frame, index)) if e]
        if events and frame.t-self.last_event >= .6:
            self.last_event = frame.t
            return min(events, key=lambda e: e['time'])
        return None


class CandidateConfirm:
    """Shared causal state machine. Candidates are silent, stale candidates expire."""
    def __init__(self, threshold=.5, low=.3, mode='predict', cooldown=.6, ttl=.4):
        self.threshold, self.low, self.mode = threshold, low, mode
        self.cooldown, self.ttl = cooldown, ttl
        self.state, self.last_transition = 'IDLE', None
        self.gate = CueGate(threshold, cooldown=cooldown)
        self.candidate = None
        self.last = -np.inf

    def update(self, t, probabilities, signal=None, unknown=False):
        self.last_transition = None
        if unknown:
            self.state, self.candidate = 'UNKNOWN', None
            self.gate = CueGate(self.threshold, cooldown=self.cooldown)
            return None
        family = max((s for s in probabilities if s != 'NONE'), key=probabilities.get, default=None)
        if self.candidate and t-self.candidate['time'] > self.ttl:
            self.candidate, self.state = None, 'IDLE'
        if family and probabilities[family] > self.low:
            if self.candidate is None or family != self.candidate['side']:
                self.candidate = dict(time=float(t), side=family)
                self.state = self.last_transition = 'CANDIDATE'
        if self.state == 'UNKNOWN':
            self.state = 'IDLE'
        predicted = self.gate.update(t, probabilities)
        confirmed = signal is not None and self.candidate is not None
        if confirmed:
            self.state = self.last_transition = 'CONFIRMED'
        event = predicted if self.mode == 'predict' else (
            dict(time=float(t), side=self.candidate['side']) if confirmed else None)
        if event and t-self.last >= self.cooldown:
            self.last = t
            event.update(family=event['side'], detection_time=signal.get('detected_time') if signal else None,
                         classification_time=float(t), candidate_time=self.candidate['time'] if self.candidate else t)
            self.candidate = None
            return event
        return None


def timing_label(delay_ms):
    return f'PREDICTED ({delay_ms:.0f} ms)' if delay_ms < 0 else f'EARLY DETECT (+{delay_ms:.0f} ms)'
