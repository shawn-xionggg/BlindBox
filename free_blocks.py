"""
Free-boxing blocks for `collect.py --script --free` (data for punch PREDICTION).

Why: in plain --script mode every punch is a reaction to a prompt on the screen.
The puncher glances at the screen, then punches, so the footage has no natural
setup before a punch, and the glance itself is a fake cue. In a free-boxing
block the puncher boxes naturally and decides when to punch, so a predictor can
learn what really happens before a punch.

Session layout with --free:
  BASELINE  --baseline-seconds (default 20). Guard up, move like between punches,
            no punching. Gives each person's calm baseline. Saved as the prompt
            IDLE, so every existing tool counts it as don't-punch footage.
  PROMPTS   --prompt-seconds (default 60). The usual random prompts. The block
            ends at the first prompt change after its time is up, so no prompt
            is cut short.
  FREE      --free-seconds (default 60). Free boxing. Saved as the prompt FREE,
            which auto_label.py, review_labels.py and early_dataset.py handle like
            a punch prompt.
  then PROMPTS, FREE, PROMPTS, FREE, ... until q.

Saved in the session .npz (see collect.save_session):
  blocks       block kinds in order (BASELINE / PROMPTS / FREE)
  block_times  start time of each block (same clock as `timestamps`)
"""
import math
import random

import cv2
import numpy as np

BASELINE, PROMPTS, FREE = "BASELINE", "PROMPTS", "FREE"
FREE_COMMAND = "FREE"        # the prompt stored in `commands` during a free block
BASELINE_COMMAND = "IDLE"    # the baseline is don't-punch footage for every existing tool
HEADS_UP = 5.0               # seconds of warning before the block changes

FREE_LINES = (
    "Box naturally: bounce, step in/out, set up.",
    "YOU choose when: singles and combos",
    "(jab-cross, jab-jab-cross, hook-cross),",
    "a few feints. Pull every punch short.",
    "Eyes on the wearer, not this screen.",
)
BASELINE_LINES = (
    "Fighting stance, guard up. Move like you",
    "do between punches (bounce if you do).",
    "DON'T punch.",
)
GREEN, BLUE, WHITE, YELLOW, BLACK = (0, 220, 0), (255, 170, 0), (255, 255, 255), (0, 255, 255), (0, 0, 0)


class FreeSchedule:
    """Decides, frame by frame, which block is running and which prompt is shown."""

    def __init__(self, pool, baseline_s=20., prompt_s=60., free_s=60., rng=random):
        if prompt_s <= 0 or free_s <= 0 or baseline_s < 0:
            raise ValueError("prompt and free blocks need a positive length, the baseline >= 0")
        self.pool, self.rng = tuple(pool), rng
        self.baseline_s, self.prompt_s, self.free_s = float(baseline_s), float(prompt_s), float(free_s)
        self.kind = None        # current block
        self.start = None       # its start time
        self.end = None         # its planned end (exact for BASELINE and FREE)
        self.next_prompt = None

    def update(self, t):
        """Call once per frame, in time order. Returns (block, command): the block that
        starts at this frame and the prompt that starts at this frame (None if none)."""
        if self.kind is None:
            return self._begin(BASELINE if self.baseline_s > 0 else PROMPTS, t)
        if self.kind == PROMPTS:
            if t < self.next_prompt:
                return None, None
            if t - self.start >= self.prompt_s:
                return self._begin(FREE, t)
            return None, self._draw(t)
        if t >= self.end:
            return self._begin(PROMPTS, t)
        return None, None

    def switch_in(self, t):
        """Seconds until the next block starts, or None while that isn't known yet."""
        if self.kind in (BASELINE, FREE):
            return self.end - t
        if self.kind == PROMPTS and self.next_prompt - self.start >= self.prompt_s:
            return self.next_prompt - t     # the prompt on screen is the block's last one
        return None

    def _draw(self, t):
        self.next_prompt = t + self.rng.uniform(2, 4)
        return self.rng.choice(self.pool)

    def _begin(self, kind, t):
        self.kind, self.start = kind, t
        if kind == PROMPTS:
            self.end = t + self.prompt_s
            return kind, self._draw(t)
        self.end = t + (self.free_s if kind == FREE else self.baseline_s)
        return kind, FREE_COMMAND if kind == FREE else BASELINE_COMMAND


FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE, LINE_SCALE, NOTE_SCALE = 1.3, .65, .9
OUTLINE = 5                  # extra pixels of black around every letter


def _text_block(frame, rows):
    """rows of (message, y, color, scale, thickness). Dims one box behind all rows,
    then draws them, so the text stays readable over any background. Only the
    on-screen copy is drawn on: collect.py writes the video frame before this."""
    boxes = []
    for message, y, _, scale, thick in rows:
        (w, h), base = cv2.getTextSize(message, FONT, scale, thick + OUTLINE)
        boxes.append((12 + w, y - h, y + base))
    H, W = frame.shape[:2]
    x1 = min(W, max(b[0] for b in boxes) + 10)
    y0, y1 = max(0, min(b[1] for b in boxes) - 10), min(H, max(b[2] for b in boxes) + 8)
    roi = frame[y0:y1, 2:x1]
    roi[:] = (roi * .35).astype(np.uint8)
    for message, y, color, scale, thick in rows:
        outlined(frame, message, (12, y), scale, color, thick)


def outlined(frame, message, xy, scale, color, thick=1):
    """Text with a thick black outline, readable over a bright or busy background."""
    cv2.putText(frame, message, xy, FONT, scale, BLACK, thick + OUTLINE, cv2.LINE_AA)
    cv2.putText(frame, message, xy, FONT, scale, color, thick, cv2.LINE_AA)


def draw(frame, schedule, t):
    """Overlay for the current block. Returns True when it drew a full block screen
    (BASELINE or FREE), which replaces collect.py's usual COMMAND line."""
    left = schedule.switch_in(t)
    soon = left is not None and left <= HEADS_UP
    secs = int(math.ceil(max(0., left))) if left is not None else 0
    if schedule.kind == PROMPTS:
        if soon:
            _text_block(frame, [(f"FREE BOXING next, in {secs}s", 255, GREEN, NOTE_SCALE, 2)])
        return False
    h, w = frame.shape[:2]
    color = GREEN if schedule.kind == FREE else BLUE
    if not (soon and int(t * 4) % 2):   # the border blinks in the last few seconds
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 14)
    lines = FREE_LINES if schedule.kind == FREE else BASELINE_LINES
    rows = [(f"{'FREE BOXING' if schedule.kind == FREE else 'BASELINE'}  {secs}s", 140, color, TITLE_SCALE, 3)]
    rows += [(line, 184 + 32 * k, WHITE, LINE_SCALE, 2) for k, line in enumerate(lines)]
    if soon:
        rows.append((f"Prompts start in {secs}s", 204 + 32 * len(lines), YELLOW, NOTE_SCALE, 2))
    _text_block(frame, rows)
    return True


# what each prompt asks for, in plain words (HOOK_L / HOOK_R = the side of the IMAGE the
# fist comes from, so the hand is the opposite one)
COMMAND_HELP = {
    "JAB": "lead-hand straight punch, pulled short",
    "CROSS": "rear-hand straight punch, pulled short",
    "HOOK_L": "swing your RIGHT fist (saved as HOOK_L)",
    "HOOK_R": "swing your LEFT fist (saved as HOOK_R)",
    "UPPERCUT": "uppercut, either hand, pulled short",
    "FEINT": "FAKE a punch start, don't throw it",
    "STEP_IN": "step toward the wearer, NO punch",
    "HEAD_TURN": "turn your head, NO punch",
    "NOD": "nod your head, NO punch",
    "GLOVE_ADJUST": "fiddle with your hands/gloves, NO punch",
    "IDLE": "stay in guard, NO punch",
    "GUARD_SHIFT": "slide your fists left/right, NO punch",
    "LIGHT CHECK": "nothing recorded: get ~30 fps, then q",
}
RED = (90, 90, 255)          # light red: readable on the dark panel


# what the screen SAYS for a prompt. The saved names HOOK_L / HOOK_R mean the side of the
# image, which every puncher read as their own hand (and threw the wrong one), so the
# screen names the puncher's hand instead. The saved prompt names don't change.
COMMAND_SHOWN = {"HOOK_L": "RIGHT-HAND HOOK", "HOOK_R": "LEFT-HAND HOOK"}


def draw_command(frame, command):
    """The prompt in big outlined letters, with what it means underneath."""
    rows = [(COMMAND_SHOWN.get(command, command), 150, YELLOW, 1.6, 4)]
    if command in COMMAND_HELP:
        rows.append((COMMAND_HELP[command], 192, WHITE, .75, 2))
    _text_block(frame, rows)


def light_level(fps):
    """(color, words) for a camera frame rate: the C270 slows down when the scene is dark."""
    if fps >= 27:
        return GREEN, "light OK"
    if fps >= 22:
        return YELLOW, "add light"
    return RED, "too dark: add light"


def draw_fps(frame, fps, y=None):
    """Live camera frame rate, colored by whether the light is good enough for ~30 fps."""
    if fps is None:
        return
    color, words = light_level(fps)
    _text_block(frame, [(f"camera {fps:.0f} fps: {words}", y or frame.shape[0] - 18, color, .75, 2)])


def framing_problem(xy, conf, width, min_conf=.35, edge=.04):
    """Why the puncher can't be used right now, or None. Training needs both shoulders
    (every feature is measured from them), so a missing shoulder means the frame is lost."""
    if xy is None or conf is None:
        return "NOBODY IN VIEW"
    sh = np.asarray(xy)[[5, 6]]
    if min(conf[5], conf[6]) < min_conf or not np.isfinite(sh).all():
        return "SHOULDERS LOST: step back"
    if sh[:, 0].min() < edge * width or sh[:, 0].max() > (1 - edge) * width:
        return "AT THE EDGE: move to the middle"
    return None


class FramingAlarm:
    """Beeps continuously while the puncher is out of frame for longer than `grace`
    seconds (a brief blur during a fast punch doesn't count). The beeps run on their
    own thread, so they never slow the camera loop; only the screen shows it in the video."""

    def __init__(self, grace=.4, sound=True, tone=(1200, 120), gap=.08):
        self.grace, self.sound, self.tone, self.gap = grace, sound, tone, gap
        self.since = None           # when the current problem started
        self.reason = None
        self.on = False             # alarm sounding right now
        self._thread = None
        self._stop = False

    def update(self, t, reason):
        """reason: framing_problem(...) for this frame. Returns the reason while alarming, else None."""
        if reason is None:
            self.since = None
        elif self.since is None:
            self.since = t
        self.reason = reason
        self.on = self.since is not None and t - self.since >= self.grace
        if self.on and self.sound and self._thread is None:
            import threading
            self._thread = threading.Thread(target=self._beep, daemon=True)
            self._thread.start()
        return reason if self.on else None

    def _beep(self):
        import time as _time
        try:
            import winsound
            play = lambda: winsound.Beep(*self.tone)
        except ImportError:          # not Windows: the terminal bell
            play = lambda: (print("\a", end="", flush=True), _time.sleep(self.tone[1] / 1000))
        while not self._stop:
            if self.on:
                play()
                _time.sleep(self.gap)
            else:
                _time.sleep(.03)

    def close(self):
        self._stop = True


def draw_alarm(frame, reason, t):
    """Big blinking red warning plus a red border while the puncher is out of frame."""
    if reason is None:
        return
    h, w = frame.shape[:2]
    if int(t * 4) % 2:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 18)
    _text_block(frame, [("OUT OF FRAME", 395, RED, 1.2, 3), (reason, 430, WHITE, .75, 2)])


def plan_text(baseline_s, prompt_s, free_s, performer, stance):
    first = f"{baseline_s:g} s BASELINE, then " if baseline_s > 0 else ""
    return (f"FREE-BOXING SESSION: {first}{prompt_s:g} s of prompts and {free_s:g} s of FREE BOXING, "
            f"alternating. Puncher: {performer} (stance: {stance}).\n"
            "  BASELINE: guard up, move like between punches, no punching.\n"
            "  FREE BOXING: box naturally and choose when to punch (singles, combos, a few feints).\n"
            "  Eyes on the wearer, not the screen. Every punch stops short of the wearer.")


def summary(blocks, block_times, t_end):
    """One line: how many blocks of each kind ran, and for how long on average."""
    if not len(blocks):
        return ""
    ends = list(block_times[1:]) + [t_end]
    total, count = {}, {}
    for b, s, e in zip(blocks, block_times, ends):
        total[b] = total.get(b, 0.) + (e - s)
        count[b] = count.get(b, 0) + 1
    parts = [f"{b} {count[b]} x {total[b] / count[b]:.0f} s" for b in (BASELINE, PROMPTS, FREE) if b in count]
    return "Blocks: " + " | ".join(parts) + f" ({total.get(FREE, 0.):.0f} s of free boxing)"
