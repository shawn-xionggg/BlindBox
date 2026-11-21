# Plan: Blindfolded Dodge — QNX Real-Time Boxing Assistant

_Created 2026-11-21 from an interview with Claude._

## Goal
You want a blindfolded person to dodge moderately fast punches. A camera and buzzers on their hat warn them which way to move before each punch lands. You're entering it in the QNX track at HackWestern, so the build has to show off QNX's real-time timing, microkernel message passing and fault tolerance, not just "it runs on QNX".

## What success looks like
- The blindfolded dodger avoids most telegraphed punches (target: 7 of 10) from a puncher stepping in from ~1.5–2 m.
- A cue fires within **100 ms** of a punch starting, and a live readout shows the measured latency and jitter.
- You kill the vision process live, the dodger hears a "system down" tone, and the process restarts by itself within ~1 s.
- Everything runs on QNX on the Pi 5. No laptop does the inference.

## Scope
**In:** a hat-mounted C270, pose-based punch detection, 3 cues (dodge left / dodge right / duck), buzzers on the hat, a worn Pi 5 running on a power bank, a latency dashboard, auto-restart.
**Out (for now):** recognising punch types (jab vs hook), body vs head targets, vibration motors, phone app.
**Smallest version that counts as a win:** one buzzer, a single "incoming" cue, a measured latency readout, and the restart demo.

## Starting point
You're 2 people with C/C++ systems, ML/CV, wiring and Python skills. You have a Pi 5, a C270 webcam and 1 buzzer, and you can probably get a second buzzer tomorrow. QNX isn't flashed yet.

## Constraints
| Constraint | Detail |
|---|---|
| Time | ~30 hours total, 2 people. Lock the demo by about hour 24 |
| Budget | Hackathon hardware only |
| Hardware | 1 buzzer now (2 soon), no motors, no Pi headphone jack |
| Physics | A punch lands in ~200–300 ms and the C270 delivers a frame every 33 ms, so you have a 2–3 frame budget |
| Demo | The camera moves whenever the dodger dodges, and the C270's ~55° view is narrow |

## Approach
Build the system as small QNX processes that pass messages to each other. Give each one a real-time priority, with the buzzer process highest, so you can *prove* the cue path is predictable. The pose model measures the attacker's wrists **relative to the attacker's own shoulders**, which cancels out the hat's motion. A fast rise in wrist speed starts a cue. Side of frame picks the direction: punch from frame-left means dodge right, frame-right means dodge left, centered means duck. The colored-glove fallback uses "looming": a glove blob that grows quickly means a fist is coming at you. That signal still works with a moving camera.

## Tech stack
| Piece | Choice | Why |
|---|---|---|
| OS | QNX SDP 8.0 (free QNX Everywhere license) + Pi 5 image | Required for the track. Official Pi support |
| Language | C/C++ | Native QNX APIs (MsgSend, pulses, ClockCycles) |
| Pose model | MoveNet Lightning (192 px) or YOLO11n-pose (256–320 px) | Tiny and fast on the Pi 5 CPU, and gives wrist/elbow/shoulder points directly |
| Runtime | TFLite or ONNX Runtime from the `qnx-ports` repos | Already ported, so you avoid building from scratch |
| Fallback CV | OpenCV color threshold + blob area | Under 10 ms per frame, no ML runtime needed |
| Cues | Active buzzers on GPIO | Simple on/off. Cue timing comes from QNX priority scheduling |
| Power | USB-C power bank (5V/3A+, ideally 5A) in a belt bag | The dodger moves untethered |

## Milestones
| # | Milestone | Target | Done when... |
|---|---|---|---|
| 1 | QNX alive | Hour 4 | Pi 5 boots QNX, SSH works, buzzer beeps from your C code |
| 2 | Camera on QNX | Hour 8 | C270 frames are grabbed on QNX at 30 fps and timestamped |
| 3 | Detector works (laptop) | Hour 8 | The pose-to-punch logic fires correctly on recorded hat-cam clips |
| 4 | End-to-end on QNX | Hour 16 | Punch → buzzer, all on the Pi, with latency logged |
| 5 | QNX showcase | Hour 22 | Latency dashboard and kill/restart demo work |
| 6 | Demo locked | Hour 24 | 10 blindfolded rehearsal runs, then only bug fixes |

## Action steps
### Phase 1: Foundations (hours 0–8, in parallel)
**Person A (QNX/systems)**
- [ ] Get the QNX Everywhere license, flash the Pi 5 image, set up SSH and cross-compiling
- [ ] Wire the buzzer to GPIO and write `cue_driver`, which beeps on receiving a pulse
- [ ] Get C270 capture working on QNX (USB video), writing frames into shared memory
**Person B (CV)**
- [ ] Record 30–50 hat-cam clips of punches on the laptop, at the real step-in distance
- [ ] Prototype the detector in Python: keypoints → normalize by shoulders → wrist speed → cue + direction
- [ ] Tune the threshold until most clips cue in ≤ 2 frames with few false alarms

### Phase 2: Bring it onto QNX (hours 8–16)
- [ ] B: build TFLite/ONNX Runtime from `qnx-ports`. **If it isn't running by hour 12, switch to the glove fallback**
- [ ] B: port the detector to C++ as `vision` (reads shared memory, sends a cue message)
- [ ] A: set priorities (`cue_driver` > `vision` > `capture` > `logger`) and timestamp with `ClockCycles()`
- [ ] A: mount the camera and buzzer(s) on the hat, angled slightly down; put the Pi and power bank in a belt bag

### Phase 3: QNX showcase (hours 16–22)
- [ ] A: add a heartbeat. If `vision` goes silent for 300 ms, `cue_driver` plays a "system down" tone
- [ ] A: auto-restart `vision` (QNX High Availability Manager, or a simple supervisor process)
- [ ] A: live dashboard over SSH or a tiny web page: frame→cue latency (avg / p99 / max), fps, restarts
- [ ] B: add the second buzzer (left/right, both = duck). Stretch goal: USB headphones

### Phase 4: Rehearse (hours 22–30)
- [ ] Do 10+ blindfolded runs with soft gloves or pool noodles, and log hit rate
- [ ] Script the 2-minute pitch: dodge demo → latency stats → kill `vision` live → it recovers

## Risks & how to handle them
| Risk | Plan if it happens |
|---|---|
| ML runtime won't build for QNX | Switch to the colored-glove looming fallback at hour 12 |
| C270 doesn't work on QNX | Check this first, at hours 4–8. Ask the QNX mentors right away |
| Hat motion causes false cues | Normalize to the attacker's shoulders, require 2 consecutive frames, ignore frames when the shoulders move fast |
| Pose model too slow | Drop the input to 192 px and pin it to its own core |
| The 1-buzzer patterns are too slow | Fire a single "incoming" beep instantly until buzzer #2 arrives |
| Someone gets hit | Soft gloves or pool noodles only, and the puncher stays at ~half speed |

## Assumptions & open questions
- Judging is ~30 hours from now. Adjust the hour targets if not.
- Until buzzer #2 arrives: 1 short beep = left, 2 short beeps = right, 1 long beep = duck.
- Check whether the Pi 5 GPIO works from QNX's GPIO resource manager or needs a direct register write.

## First step (do this next)
Person A: download the QNX Pi 5 image and flash it now. Person B: record 10 hat-cam punch clips on a laptop with the C270 so detector work can start right away.
