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


def _text_block(frame, rows):
    """rows of (message, y, color, scale, thickness). Dims one box behind all rows,
    then draws them, so the text stays readable over any background. Only the
    on-screen copy is drawn on: collect.py writes the video frame before this."""
    boxes = []
    for message, y, _, scale, thick in rows:
        (w, h), base = cv2.getTextSize(message, FONT, scale, thick + 2)
        boxes.append((12 + w, y - h, y + base))
    H, W = frame.shape[:2]
    x1 = min(W, max(b[0] for b in boxes) + 10)
    y0, y1 = max(0, min(b[1] for b in boxes) - 10), min(H, max(b[2] for b in boxes) + 8)
    roi = frame[y0:y1, 2:x1]
    roi[:] = (roi * .35).astype(np.uint8)
    for message, y, color, scale, thick in rows:
        cv2.putText(frame, message, (12, y), FONT, scale, BLACK, thick + 2, cv2.LINE_AA)
        cv2.putText(frame, message, (12, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw(frame, schedule, t):
    """Overlay for the current block. Returns True when it drew a full block screen
    (BASELINE or FREE), which replaces collect.py's usual COMMAND line."""
    left = schedule.switch_in(t)
    soon = left is not None and left <= HEADS_UP
    secs = int(math.ceil(max(0., left))) if left is not None else 0
    if schedule.kind == PROMPTS:
        if soon:
            _text_block(frame, [(f"FREE BOXING next, in {secs}s", 210, GREEN, NOTE_SCALE, 2)])
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
