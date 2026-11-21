# Blindfolded Dodge — Project Brief for Agents

Read this first. It's the shared context for every agent (Claude, Codex, or a person) working on this repo. The full plan with timeline is in [PLAN-qnx-boxing-assistant.md](PLAN-qnx-boxing-assistant.md).

## What we're building
A hackathon project (HackWestern, **QNX track** — the main prize). A blindfolded person wears a hat with a **Logitech C270 webcam** and **buzzers**. A **Raspberry Pi 5 running QNX 8.0** (worn in a belt bag on a power bank) watches the attacker. It spots a punch *as it starts* and buzzes **DODGE LEFT / DODGE RIGHT / DUCK** in time for the dodger to move.

**Demo:** the puncher steps in from ~1.5–2 m and throws moderately fast punches (soft gloves). The blindfolded dodger avoids most of them.

**What judges must see (QNX strengths):**
1. **Real-time latency.** Priority-scheduled processes and a live readout of frame→cue latency (avg / p99 / max, jitter).
2. **Microkernel + IPC.** Separate processes talking via QNX message passing (MsgSend / pulses) and shared memory.
3. **Fault tolerance.** Kill the vision process live, a "system down" tone plays, it auto-restarts in ~1 s.

## Hard constraints
- **Timing budget:** a punch lands in ~200–300 ms and the C270 delivers a frame every ~33 ms (30 fps). **Target: cue ≤ 100 ms after the punch starts** (≈ 2–3 frames).
- **Everything runs on the Pi under QNX** for the final demo. A laptop is for prototyping only.
- **The camera is on a moving head.** All motion must be measured relative to the attacker's body, never raw image motion.
- **Hardware on hand:** Pi 5, C270, 1 buzzer (2nd likely soon). No vibration motors. The Pi 5 has no headphone jack.
- **Team:** 2 humans, ~30 h total. Prefer simple and working over clever.

## How detection works (reference: `punch_telegraph.py`)
1. Pose model (`yolo11n-pose`, COCO keypoints) → take the **largest person** = the attacker.
2. Normalize: origin = shoulder midpoint (kp 5, 6), unit = shoulder width.
3. Wrist (kp 9, 10) speed = Δ normalized position / Δt, EMA-smoothed (`alpha` 0.6).
4. Cue when a wrist speed > `thresh` (default **5 shoulder-widths/s**) for `confirm` (**2**) consecutive frames, with a **0.6 s** cooldown.
5. Direction from the wrist's normalized x: `< -0.6` → DODGE RIGHT, `> 0.6` → DODGE LEFT, else → DUCK.

**Looming / time-to-contact detector (reference: `looming_ttc.py`, now a lead approach, not just a fallback):** colored gloves found by HSV color (click to sample). Size s = √(blob area), and since s ∝ 1/distance, 1/s falls linearly as a fist approaches. A line is fit to 1/s over the last 5 frames, and **TTC = when that line hits zero**. Cue when TTC < **350 ms** for 2 frames. Direction from the glove's x in the frame (outer ~32% each side = left/right, middle = duck). It needs no ML runtime and no calibration, tolerates camera motion, and gets more accurate up close, which allows a demo at real punching range (~0.8–1 m). On a synthetic approaching blob it's within ~5 ms of the true TTC below 350 ms.

## Target architecture on QNX
```
capture ──(shared mem frame ring + pulse)──> vision ──(MsgSend cue_msg)──> cue_driver ──GPIO──> buzzers
   │                                            │ heartbeat pulse every frame      │
   └────────────── timestamps (ClockCycles) ────┴──────────> logger/dashboard <────┘
supervisor (HAM or custom): restarts vision; cue_driver plays "system down" if no heartbeat for 300 ms
```
Priorities (SCHED_FIFO, high → low): `cue_driver` > `vision` > `capture` > `logger`.

Shared definitions go in `qnx/common/protocol.h` (single owner, see below). Proposed:
```c
enum cue_dir { CUE_NONE=0, CUE_LEFT=1, CUE_RIGHT=2, CUE_DUCK=3, CUE_FAULT=4 };
typedef struct { uint16_t type; uint8_t dir; float speed; uint64_t t_capture_cycles; uint32_t frame_id; } cue_msg_t;
```

## Repo layout
**Now:** `punch_telegraph.py`, `requirements.txt`, `yolo11n-pose.pt`, `cues.csv` (created on run), `clips/` (recorded test clips), `PLAN-qnx-boxing-assistant.md`, `AGENTS.md`.

**Target (create as you go; don't move existing files without asking):**
```
prototype/      laptop Python prototype + tuning tools
models/         exported models (ONNX / TFLite) + benchmark results
fallback/       colored-glove looming detector
qnx/common/     protocol.h, shared helpers
qnx/capture/    C270 → shared memory
qnx/vision/     C++ pose inference + punch logic
qnx/cue_driver/ GPIO buzzer output
qnx/supervisor/ restart + heartbeat
qnx/logger/     latency stats + dashboard
hardware/       wiring notes, hat photos
```

## Workstreams (one agent per stream)
| # | Stream | Owns | Done when |
|---|---|---|---|
| A | **Detector tuning** | `prototype/`, `clips/` | On recorded hat-cam clips, ≥ 8/10 punches cue in ≤ 2 frames and < 1 false cue per minute of guard movement. Writes an eval script that replays clips and prints hit rate / false alarms / latency |
| B | **Model export + benchmark** | `models/` | `yolo11n-pose` exported to ONNX and TFLite at 192/256/320 px. Table of accuracy vs speed on the laptop CPU, with the Pi 5 estimate |
| C | **Glove fallback** | `fallback/` | Python looming detector with the same cue output and the same eval script as A |
| D | **QNX IPC skeleton** | `qnx/common/`, `qnx/capture/`, `qnx/cue_driver/`, `qnx/supervisor/` | Processes build with QNX SDP 8.0 (`qcc`), pass `cue_msg_t` end to end with a fake vision stub, supervisor restarts a killed process |
| E | **QNX vision port** | `qnx/vision/` | C++ port of the detector logic (unit-tested against Python on the same keypoint logs), inference via ONNX Runtime / TFLite from `qnx-ports` |
| F | **Latency dashboard** | `qnx/logger/` | Shows avg / p99 / max frame→cue latency, fps, and restart count, viewable from a laptop (SSH text UI or tiny web page) |

Stream D owns `qnx/common/protocol.h`. Others propose changes to it and don't edit it directly.

## Rules for agents
- **Stay in your stream's folders.** If you need a change elsewhere, write it down in your final report instead of making it.
- **Never** flash SD cards, format disks, or touch the Pi's boot media. Humans do hardware steps.
- **Don't commit** model weights or clips over ~20 MB, and never commit credentials or license keys (QNX license included).
- Keep the Python prototype runnable: `python punch_telegraph.py --source <cam index | clip.mp4>`.
- Report numbers, not vibes: latency in ms, hit rate as x/y, fps.
- Write C for QNX against SDP 8.0 APIs (`ChannelCreate`, `MsgSend`, `MsgReceivePulse`, `ClockCycles`, `pthread_setschedparam`). Note anything you couldn't compile or verify.

## Status (2026-11-21, overnight). Read `REPORT.md` for details
- ✅ **Current best detector: learned early detector** (`early_live.py`, `models/early_detector.joblib`, demo app `dodge_live.py`). Cross-validated on 121 human-reviewed punches: ~41% caught ≤ ~100 ms and ~60% ≤ ~200 ms after the fist leaves guard, at ~2 false alarms/min. Live features verified identical to training (`test_early_live.py`).
- ✅ Data pipeline: `collect.py --script` → `auto_label.py` → `review_labels.py` (y/n) → `early_detector.py` → `export_model_c.py` (C model for QNX, verified identical).
- ❌ Prediction before the fist moves: no signal found. ❌ Early direction (left/right/duck): near chance. Use one universal INCOMING cue.
- ❌ The orange glove matches skin, so the glove signal is useless. Use blue/green.
- ⏳ Needs more data: 2+ scripted sessions, ideally a second performer.
- ⏳ A wide-angle clip-on lens (~0.45x) for the C270 is wanted so the attacker fits in view at ~0.8 m.
- Old prototypes (`punch_telegraph.py`, `looming_ttc.py`, `anticipate_live.py` + `models/anticipator.joblib`) are superseded but kept.
- ⏳ QNX not flashed. The image is the **QNX SDP 8.0 Quick Start Target Image for Raspberry Pi 5** (`com.qnx.qnx800.quickstart.rpi5`, via QNX Software Center), flashed with Raspberry Pi Imager, 32 GB+ microSD.

- ✅ Pi path de-risked on the laptop: ONNX pose export verified (`check_onnx_pose.py`); retrain on deployment keypoints (`repose_onnx.py`, `models/early_detector_onnx.joblib`); C pose decoder + C model verified identical; OpenCV DNN runs the pose model identically but 4.6x slower. No ONNX Runtime port for QNX was found, which is the main risk.

## Unverified — check before relying on it
- Does the C270 (USB video) work under QNX 8.0 on the Pi 5?
- How to drive Pi 5 GPIO from QNX (resource manager vs direct RP1 access)?
- Which ML runtime (ONNX Runtime / TFLite) builds cleanly from `qnx-ports` for aarch64?
- Pose model fps on the Pi 5 CPU at 192/256 px.
