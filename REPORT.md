# Overnight report: making punch detection more accurate

_2026-11-21, written while you slept. All numbers come from your two reviewed sessions (121 punches) and are cross-validated: scored on stretches of footage the model never trained on, with the alarm threshold picked without looking at that footage._

## TL;DR
- **Detection got much better:** a new learned detector catches punches **about twice as early** as the hand-made one, at the same false-alarm rate.
  - At ~2 false alarms/min: **41% of punches caught within ~100 ms** of the fist leaving guard, **60% within ~200 ms**. Median delay **96 ms**, down from 193 ms.
  - At ~5.6 false alarms/min: **63% within ~100 ms, 79% within ~200 ms**. Median delay **48 ms**.
- **True prediction (before the fist moves) is still not possible.** No signal rises before the fist leaves guard (0 of 121).
- **Direction (left / right / duck) is still not reliable.** Early direction guesses are near chance (~45–56% vs 45% for always guessing the most common one). **Use one universal "INCOMING" cue (step back) for the demo.** Then success depends only on detection, which works.
- **The live app is ready and verified against the offline numbers:** `python dodge_live.py --source 1`. Its live features match the training features exactly.
- **QNX progress:**
  - The model now exports to **plain C** and matches Python exactly (0 differences on 3,001 rows).
  - A QNX process skeleton (cue driver, supervisor, vision stub) exists with real QNX message-passing code. Its crash-recovery test passes here; QNX itself is untested. See its section below.

## Update (2026-11-21 late morning): 253 punches, messy punches added
- **More data:** session 4 (2.6 min, bare fists, new GUARD_SHIFT prompt), plus a new `review_labels.py --messy` pass over punches the old review skipped because the fist wasn't held still first. **121 → 253 reviewed punches.**
- **Harder, more honest test set:** the held-out punches now include the messy ones, which are harder to catch early. So all numbers are lower than the 121-punch numbers above. That's not a regression, the test got more realistic.
- **Same held-out punches, same false-alarm rate, training on clean-only vs all labels:**

| Train on | ≤1 fa/min: caught ≤4 frames | ≤2 fa/min: caught ≤4 frames | hooks ≤4 frames (2 fa/min) |
|---|---|---|---|
| clean punches only (old way) | 23% | 42% | 43/124 |
| **all reviewed punches (new model)** | **28%** | **45%** | **48/124** |

  The gain is small (error bar about ±6 points at n = 253) but it shows up at every budget. False alarms on held-out footage are 2.47/min against a budget of 2.
- **Installed:** `models/early_detector.joblib` (laptop) and `models/early_detector_onnx.joblib` (Pi keypoints: 44% ≤4 frames at 2.76 fa/min). C export refreshed from the ONNX one (159 trees, matches Python exactly). The 121-punch models are in `models/archive/`.
- **Training fix:** frames inside a punch prompt with no reviewed punch are no longer learned as "calm".
- **GUARD_SHIFT:** only 32 s recorded so far, too little to measure. The old model beeped 2 times in it (~4/min). Record more, with bigger and faster side-to-side moves like the ones that beep live.

- **Direction cues are back** (`dodge_live.py --directional`), with two new rules in `early_live.py`:
  - **Uppercut → STEP BACK:** an uppercut's fist drops first (wind-up). If either fist drops more than 0.10 shoulder widths below its rest spot without swinging sideways, the cue is STEP BACK. Replay of sessions 3–4: uppercuts got STEP BACK 12/15 (was DUCK 15/15). Some jabs, crosses and hooks also get STEP BACK, which is safe.
  - **Hook side from the direction of travel,** not the fist's starting side. A hook rotates the body, so even the wrong hand moves the same way. Wrong-way hook cues went from 5/6 to 0/4.
  - These replays use footage the model trained on, so they're optimistic.

## Why it got better (plain English)
1. **A fist coming at the camera barely moves in the picture; it grows.** Pose keypoints only see side-to-side and up-down motion, so early straight punches were invisible to them. **Optical flow** measures how every pixel moved between frames. Pixels spreading outward around the fist (*divergence*) means it's coming closer. That needs no glove.
2. **Pulling the fist back is as fast as throwing it.** The old speed rule buzzed on retractions. The learned model is trained with retractions as "not a punch".
3. **One model combines everything.** Instead of one signal with one threshold, a gradient-boosted classifier looks at each wrist over the last 3 frames:
   - distance from where the fist was resting
   - speed
   - pixel change around the fist
   - flow divergence and outward flow
   - whole-body motion (to ignore steps and turns)

   The strongest signals were pixel change near the fist, distance from rest, and flow magnitude.

## Results (held-out footage, ~20 fps, 1 frame ≈ 48 ms)
| Detector | False alarms/min | Caught ≤ +1 frame | ≤ +2 | ≤ +4 | Median delay |
|---|---|---|---|---|---|
| Old wrist-speed rule | ~4 | 3% | 14% | 32% | ~290 ms |
| Hand-made "fist leaves guard" | ~1.9 | 1% | 5% | 26% | 193 ms |
| **Learned (default, budget 2)** | **~2.1** | **25%** | **41%** | **60%** | **96 ms** |
| **Learned (budget 4)** | **~5.6** | **46%** | **63%** | **79%** | **48 ms** |

Feints buzz most of the time at budget 2 or more. That's expected: a feint *is* the start of a punch.

**What this means for the dodger:** in your session-3 punches, the fist went from leaving guard to full extension in a median of ~210 ms. A cue at ~50–100 ms, plus ~150–200 ms of human reaction, is tight at arm's length. It's workable if the puncher steps in from ~1.5 m, which adds travel time. Keep the puncher out of reach, as the guide says.

## Things I tried that did NOT help (so you don't redo them)
- **Orange glove:** its color matches skin, so the glove mask lit up on your face and hands. The glove signal was pure noise. Next time use **blue or green**, and check the new "glove mask" preview in the recorder.
- **Fist-leaves-the-frame cue for hooks** (`exit_cue.py`, `dodge_live.py --exit-cue`, off by default): hooks do reach the frame edge (median 2–5% of the width) and the pose model often loses the wrist there. But that happens 4–13 frames after the fist leaves guard, after the model already had its chance. Cross-validated: hooks caught ≤ 4 frames 28/59 → 29/59, false alarms 2.07 → 2.25/min. The real fix for hooks is a wider view (clip-on lens, or the puncher standing farther back).
- **The original anticipator** (predict before the fist moves): ~0 of 74 held-out punches warned early.
- **Extra features** (rest history, both-hands motion, head motion): no gain.
- **Mirror augmentation:** more catches, but equally more false alarms; the same trade-off curve.
- **Score smoothing over 2 frames,** and labeling 3 or 5 "early" frames instead of 4: no gain.
- **Direction from the fist's own motion** (no labels needed), even with optical-flow direction added: 48–51% when the wrist is chosen the way a live system must, vs 45% baseline. The deeper reason: **in the first ~100 ms, even *which hand* is punching is only picked right 49–58% of the time.** A punch rotates the whole upper body, so both fists start moving. That's why one universal cue is the right design.
- **Learned punch-type / direction classifier:** near chance. Partly a data problem:
  - In session 3, "HOOK_L" punches mostly travel leftward. You probably threw a *left* hook (your left hand), which comes from the *right* of the picture, the opposite of the label.
  - The auto-labeler sometimes picks the wrong hand during hooks.

## Recorder update: side-to-side guard
Live, sliding the guard left/right beeps falsely, because no session recorded it as "don't punch". The recorder (`collect.py --script`) now has a **GUARD_SHIFT** prompt ("slide your fists left/right, DON'T punch"), shown twice as often as the other prompts. Record bare-fisted, the way the demo is done.

## Bugs found and fixed
- Training paired different labelers' marks **by order**, so one skipped punch shifted every later pair. They're now paired **by time** (within 150 ms).
- The old tracker went "unknown" whenever *either* wrist was lost. The new detectors track each wrist on its own.
- The replay test re-read the compressed pose file every frame (37 ms per frame for nothing). Fixed in `dodge_live.py`.
- Optical flow was the main cost (14 ms per frame). **Quarter-resolution flow is 4x cheaper and just as accurate**, so it's now the default. The whole detector costs ~6 ms per frame on the laptop, plus pose.

## New files (what each does)
| File | Purpose |
|---|---|
| `dodge_live.py` | **The demo app.** Camera → pose → learned detector → beep. `--budget 1/2/4` sets sensitivity, `--directional` is experimental, `--headless --npz` replays a recording |
| `early_live.py` | Streaming learned detector (used by `dodge_live.py`) |
| `early_dataset.py`, `early_detector.py` | Build features/labels; train + cross-validate; save `models/early_detector.joblib` |
| `flow_features.py` | Optical-flow features (divergence, outward flow) around each wrist |
| `test_early_live.py` | Proves live features == training features (run after any feature change) |
| `early_family.py` | Direction/punch-type experiments (result: not reliable yet) |
| `diagnose_signals.py`, `evaluate_detector.py` | How fast each signal catches punches vs false alarms |
| `launch_detector.py` | Hand-made "fist leaves guard" detector (baseline; Codex wired it into `collect.py` / `anticipate_live.py`) |
| `auto_label.py`, `review_labels.py` | Find punch starts automatically, then confirm them with y/n |
| `export_model_c.py` → `qnx/model/` | The model as plain C for QNX, verified identical to Python |
| `live_runtime.py` (Codex) | Threaded capture, threaded video writing, per-stage timing |
| `qnx/` (Codex) | QNX process skeleton: see its README |
| `cue_client.py`, `test_bridge.py` (Codex) | Python → C cue-driver bridge and the full pipeline rehearsal test |
| `test_early.py` | Regression tests: live == training features, C model == Python, demo replay works |
| `check_onnx_pose.py`, `check_onnx_detector.py`, `repose_onnx.py` | ONNX pose checks; re-pose sessions into `data_onnx/` for training on deployment keypoints |
| `DEMO_SCRIPT.md` | 2-minute judge demo, QNX-first, with honest numbers |
| `qnx/pose/pose_decode.c`, `check_pose_decode.py` | Plain-C pose output decoder for QNX + its verification |
| `check_opencv_dnn.py` | OpenCV DNN vs ONNX Runtime for the pose model (same output, slower) |
| `check_tflite_pose.py`, `yolo11n-pose_tflite/` | TFLite pose model (plan B runtime for QNX) and its verification |

**Test status at the end of the night:** 45 Python tests pass (`python -m unittest test_anticipation test_early`), plus `qnx/test_qnx_skeleton.py` and `test_bridge.py`.

## QNX side (built by Codex, reviewed and re-tested by Claude)
`qnx/` holds the QNX process design from AGENTS.md:

| Process | Job |
|---|---|
| `cue_driver` | Highest priority (SCHED_FIFO). Receives cue messages, drives the buzzer, logs capture-to-cue latency. If no heartbeat arrives for 300 ms, it plays a "system down" pattern |
| `vision_stub` | Stands in for the vision process. Sends scripted cues plus a heartbeat every frame. Can simulate a crash |
| `supervisor` | Restarts the vision process if it dies. HAM is noted as the production alternative on QNX |

- **IPC:** on QNX it uses native message passing (`name_attach` / `MsgSend` / `MsgReceive` / `MsgReply`, heartbeat pulses, `TimerTimeout`). Here it falls back to localhost sockets with the same behavior, so the logic is testable on Windows.
- **Verified here** (`python qnx/test_qnx_skeleton.py`, built with warnings-as-errors):
  - 9 of 9 cues delivered through a simulated vision crash
  - the fault pattern fired at ~314 ms
  - the vision process restarted in ~663 ms
  - IPC cue latency ~0.5 ms average, 1.2 ms worst
- **Not verified** (needs the QNX SDK and the Pi): compiling with `qcc`, real QNX IPC and priorities, and **Pi 5 GPIO**. The buzzer is still a logged stub; see `qnx/README.md`.
- **The model as C:** `qnx/model/early_model.c` (80 trees, plain C99). It's exactly equal to Python. The remaining C work is the feature extraction (optical flow needs OpenCV on QNX).
- **Full pipeline rehearsal on the laptop (verified):** the Python detector now talks to the C cue driver (`dodge_live.py --ipc`, client in `cue_client.py`). `test_bridge.py` replays a session through both:
  - 7 of 7 cues delivered at **~6 ms** from frame capture to the cue driver
  - when the detector exits, the fault tone fires once at ~0.3 s
  - when the supervisor kills the Python detector, it **restarts it in ~0.66 s** and cues resume

  To run it yourself (MSYS2 gcc must be on PATH; launch the regular Python by full path, since MSYS2 has its own `python`):
  ```powershell
  $env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\usr\bin;" + $env:PATH; & "C:\Users\junio\AppData\Local\Programs\Python\Python313\python.exe" test_bridge.py
  ```
- **Judge demo idea:** kill the vision process mid-demo. The buzzer plays the fault pattern within ~0.3 s and detection resumes in under 1 s. That shows QNX-style fault tolerance live.

## Pose without PyTorch (de-risking the Pi)
QNX has no PyTorch, so the realistic route is **ONNX Runtime** running an ONNX export of the pose model. I checked this end to end:
- **Exported** `yolo11n-pose.onnx` (320 px, 11.7 MB).
- **Same keypoints:** the same person was found in 299 of 299 frames. Keypoints differ by a median of 1.6 px (on 640 px frames) because of padding, and visibility agrees 99.4% of the time (`check_onnx_pose.py`).
- **But the detector notices the difference:** with ONNX keypoints, only 56 of 71 cues land within 2 frames of the PyTorch-keypoint cues (`check_onnx_detector.py`). **Train on the pose model you deploy.**
- **Retrained on ONNX keypoints** (`repose_onnx.py` → `data_onnx/`, then `early_detector.py --data data_onnx`): just as accurate, within error bars.

  | ~False alarms/min | ≤ +2 frames | ≤ +4 frames |
  |---|---|---|
  | ~0.9 | 29% | 47% |
  | ~2.4 | 43% | 61% |
  | ~6.2 | 58% | 72% |

  Saved as `models/early_detector_onnx.joblib`.
- **Speed on this laptop:** raw ONNX pose takes 11 ms per frame with 4 threads. The full app with ONNX pose plus the detector takes ~19 ms per frame. **The Pi 5 will be several times slower. Measure it first thing.**
- **For the Pi:**
  - run `dodge_live.py --pose yolo11n-pose.onnx --model models/early_detector_onnx.joblib`
  - export the C model from the ONNX-trained detector: `python export_model_c.py models/early_detector_onnx.joblib`
  - **output decoding + NMS + largest person is done in plain C:** `qnx/pose/pose_decode.c`. Verified **identical** to ultralytics on 200 real frames (0.0000 px difference; `check_pose_decode.py`)
  - still to write in C/C++: grab a frame, letterbox it (resize + pad with gray 114), call ONNX Runtime's C API, then the feature extraction (optical flow needs OpenCV on QNX)

### Which libraries the Pi needs (the biggest QNX risk)
| Need | Option | Status |
|---|---|---|
| Optical flow (Farneback) | **OpenCV**: QNX ships an official port for QNX 8.0 aarch64 in [qnx-ports/build-files](https://deepwiki.com/qnx-ports/build-files/1.2-supported-qnx-versions-and-architectures) (see QNX's ["Porting OpenCV to QNX"](https://www.qnx.com/developers/docs/qnxeverywhere/com.qnx.doc.qpg/topic/porting_OpenCV.html)) | Likely available. Dropping flow costs ~10–15 points of catch rate (measured with `early_detector.py --no-flow`), so keep it |
| Pose inference, fast | ONNX Runtime | **No QNX port found.** It would need a cross-compile. 9–11 ms per frame on the laptop |
| Pose inference, plan B (likely fast) | **TensorFlow Lite**: QNX's ports reportedly include it for QNX 8.0 aarch64 ([DeepWiki summary](https://deepwiki.com/qnx-ports/build-files/6.2-tensorflow-lite), auto-generated, so confirm in `ports/tensorflow/README.md`) | **Done and verified.** `yolo11n-pose_tflite/yolo11n-pose_float32.tflite`, converted from the ONNX model with onnx2tf (ultralytics blocks TFLite export on Windows). Keypoints are **identical** to ONNX Runtime (0.002 px worst, 0 mismatches over 100 frames) and it's **about as fast** (12.3 ms per frame with 4 threads vs 11.4 ms). The ONNX-trained detector (`models/early_detector_onnx.joblib`) applies as is. Input is NHWC `[1,320,320,3]`, output `[1,56,2100]`, same as `pose_decode.c` expects. The float16 file won't run on a plain CPU; use float32 (`check_tflite_pose.py`) |
| Pose inference, fallback | **OpenCV DNN** (`cv::dnn::readNetFromONNX`) | **Verified identical** keypoints (0.001 px worst, `check_opencv_dnn.py`), but **4.6x slower** (42 ms vs 9 ms on the laptop), so probably too slow on the Pi. Use 256 px input or a lower frame rate if it's the only option |
| Pose output decoding | `qnx/pose/pose_decode.c` | Done, verified identical |
| Decision trees | `qnx/model/early_model.c` | Done, verified identical |

**First thing on the Pi:** build OpenCV (needed for flow) and TensorFlow Lite from QNX's ports, then run `yolo11n-pose_float32.tflite` and measure the frame rate. OpenCV DNN is the slower fallback; an ONNX Runtime cross-compile is the last resort.

## How to run the demo
```powershell
python dodge_live.py --source 1
```
- The C270 is camera 1 on your laptop.
- Bars show each wrist's punch probability. The red line is the threshold.
- `--budget 4` makes it more sensitive (more catches, more false alarms).

## Next steps (in order)
1. **Record 2 more scripted sessions**, ideally **with a second person punching** so you can test on someone the model never saw.
   - Use a **blue or green** glove (check the glove-mask preview), face the light, and use `--script`.
   - For each new session: `python auto_label.py data/<new>.npz`, then `python review_labels.py data/<new>.npz`, then `python early_detector.py`.
   - Then run `python export_model_c.py` to refresh the C model.
   - More data is the single biggest lever: the model learned from only ~120 punches by one person.
2. **Decide the cue:** keep **one universal INCOMING beep** for the demo unless direction gets reliable.
3. **Hook-label convention: done.** The recorder now shows "HOOK_L = hook with your RIGHT hand" (and the reverse for HOOK_R). Saved labels are unchanged.
4. **QNX / Pi:**
   - Flash QNX.
   - Build `qnx/` with qcc.
   - Test the cue driver on a real GPIO buzzer.
   - Measure pose speed on the Pi. That's the big unknown: if the Pi can't do ~15+ fps, the timing budget fails.
5. The model was trained at ~20 fps. If the live frame rate differs a lot (e.g. 30 fps), retrain on footage recorded at that rate.

## Caveats (be honest with judges)
- **Error bars:** with 121 punches, each catch rate is about **±9 percentage points** (95% margin ≈ 2·√(p(1−p)/n)). For example, "41% within ~100 ms" really means somewhere around 32–50%. Differences of a few points between settings are noise.
- Only **2 sessions and 1 performer**. Numbers on a new person will probably be lower.
- Punch-start labels came from the auto-labeler, which uses some of the same signals, and you confirmed them by eye. That can flatter the "+1 frame" numbers a little.
- False alarms on held-out footage run slightly above the tuning budget (2.07 vs 2 per minute), so expect a bit more live.
