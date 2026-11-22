"""Notes on people and sessions that the .npz files don't hold (data/session_notes.json):
skill level, sessions to leave out, and sessions where the puncher swapped the hooks.

Why a separate file: the recorded sessions are never edited, so a correction lives
next to them and the tools apply it when they read a session.
"""
import json
from pathlib import Path

import numpy as np

NOTES = Path(__file__).with_name("data") / "session_notes.json"
SWAP = {"HOOK_L": "HOOK_R", "HOOK_R": "HOOK_L"}


def load(path=NOTES):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"performers": {}, "sessions": {}}


def session(npz, path=NOTES):
    """The notes for one session (by file name), or {}."""
    return load(path).get("sessions", {}).get(Path(npz).stem, {})


def commands(npz, cmds, path=NOTES):
    """The prompts as the puncher actually performed them: HOOK_L and HOOK_R are swapped
    for sessions marked 'hooks_swapped' (the puncher threw the other hand)."""
    if not session(npz, path).get("hooks_swapped"):
        return cmds
    return np.asarray([SWAP.get(str(c), str(c)) for c in cmds], dtype=np.asarray(cmds).dtype if len(cmds) else "<U16")
