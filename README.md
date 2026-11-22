# Boxing Assistant (Blindfolded Dodge)

A HackWestern project for the **QNX track**. A blindfolded person wears a hat with a webcam and a buzzer. The system watches the attacker, spots a punch **as it starts**, and beeps a cue (**DODGE LEFT / DODGE RIGHT / DUCK / STEP BACK**) in time for the wearer to move.

- **Hardware:** Raspberry Pi 5 running QNX 8.0 (worn in a belt bag with a power bank), Logitech C270 webcam and buzzer on the hat.
- **Software:** pose model (YOLO11n-pose) → a small learned "punch is starting" detector → cue rules → buzzer. It is prototyped in Python on a laptop and ported to C for QNX in `qnx/`.
- **Demo:** the puncher stands at step-in distance (~1.5–2 m) and throws moderately fast punches. The game is non-contact: the puncher always stays beyond the wearer's reach.

> Deeper docs:
> - [REPORT.md](REPORT.md): all experiments, numbers and what didn't work
> - [AGENTS.md](AGENTS.md): brief for AI agents and workstreams
> - [qnx/README.md](qnx/README.md): QNX processes
> - [DEMO_SCRIPT.md](DEMO_SCRIPT.md): judge demo
> - [PLAN-qnx-boxing-assistant.md](PLAN-qnx-boxing-assistant.md): original plan
> - **[HEAD_MOUNTED_TRAINING.md](HEAD_MOUNTED_TRAINING.md): the final training, with the camera on the wearer's head. Start here for new data.**
> - [TEAMMATE_AUGMENTATION_GUIDE.md](TEAMMATE_AUGMENTATION_GUIDE.md): run the lighting / frame-rate training on a fast machine
> - [PREDICTION.md](PREDICTION.md): experimental punch prediction (warning before the fist moves): how it's trained, tested and what it found so far

---

## 1. Setup (Windows laptop)

1. Install **Python 3.13** from python.org.
2. Clone the repo and install the requirements:
   ```powershell
   git clone https://github.com/shawn-xionggg/boxing-assistant
   cd boxing-assistant
   python -m pip install -r requirements.txt
   ```
3. Check it works. All tests should pass:
   ```powershell
   python -m pytest -q test_early.py test_anticipation.py
   ```

**Gotchas:**
- **"No module named cv2"** means a different Python ran (for example an MSYS one). Call the 3.13 one directly: `C:\Users\<you>\AppData\Local\Programs\Python\Python313\python.exe`.
- **Camera index:** run with `--source 0`, `1` or `2` until the window shows the C270. On the original laptop the C270 is `1` (`0` is the built-in camera). Only one program can use a camera at a time, so close the demo before recording.
- **Plug in the charger.** On battery, Windows slows the CPU and the pipeline drops from ~30 fps to ~7 fps. The model needs ~20–30 fps.

---

## 2. Run the live demo

```powershell
python dodge_live.py --source 1
```

- One bar per wrist shows its punch probability. When either bar crosses the red line, it beeps.
- `q` quits. Beeps are logged to `dodge_log.csv`.

| Option | What it does |
|---|---|
| `--budget 4` | More sensitive: more catches, more false beeps. `--budget 1` is less sensitive. The threshold is tuned for about this many false beeps per minute. |
| `--universal` | One "INCOMING" beep for every punch instead of direction cues. |
| `--source data/x.mp4 --npz data/x.npz --headless` | Replay a recorded session without a camera. |
| `--pose yolo11n-pose.onnx --model models/early_detector_onnx.joblib` | Run exactly what the Pi will run (ONNX pose + the detector trained on ONNX keypoints). |
| `--esp32 COM5` or `--esp32 udp:192.168.4.1` | Also buzz the **ESP32 hat buzzer** (wiring, flashing and testing: [esp32/README.md](esp32/README.md)). Add `--no-beep` to silence the laptop. |
| `--exit-cue` | Experimental: also beep when a fist races out the side of the frame. Measured as almost useless, so it's off by default. |

### What each cue means

| Beep | When | Why |
|---|---|---|
| **DODGE LEFT** (low) / **DODGE RIGHT** (high) | The triggering fist is swinging sideways (a hook). The side comes from **which way the fist is moving**. | A hook twists the whole upper body, so even the wrong hand moves the same way. Using the fist's start side sent people *into* hooks. |
| **STEP BACK** (deep, long) | Either fist **drops** below its guard spot first, without swinging sideways. | That's the wind-up of an uppercut. Ducking would move the head into it, and stepping back is safe against everything. |
| **DUCK** (middle, long) | Anything else (jab, cross). | |

---

## 3. How detection works (short version)

1. **Pose:** `yolo11n-pose` at 320 px finds the attacker (the largest person) and 17 body points. The wrists (COCO 9, 10) and shoulders (5, 6) are what matter.
2. **Body-relative:** all positions are measured from the shoulder midpoint in **shoulder widths**. The camera is on a moving head, so raw image motion would mostly be the wearer's own head.
3. **Features** (`early_dataset.py`), 24 per wrist per frame:
   - distance from the fist's rest spot in guard, and a hand-made "launch from guard" score
   - wrist speed and position
   - pixel change around the wrist
   - **optical flow**: whether pixels near the fist spread outward (looming toward the camera), with the attacker's torso motion subtracted
4. **Model:** a gradient-boosted tree classifier (`HistGradientBoostingClassifier`) outputs a "this fist just started a punch" probability. It beeps when the probability crosses the threshold, with a 0.6 s cooldown between beeps.
5. **Cue rules** in `early_live.py` (table above).

**Honest numbers** (cross-validated on footage the model never trained on; 378 reviewed punches from 1 performer; see REPORT.md): on 30 fps footage, which matches the live demo, **44% of punches are caught within 200 ms** of the fist leaving guard, at **1.5 false beeps per minute**.
- The model before the 30 fps session caught only 17% on that footage, because it had only ever seen 20 fps. **Record at the frame rate you demo at:** plugged in, ~30 fps.
- Prediction *before* the fist moves isn't possible: no signal was found.
- More data is the biggest lever.

---

## 4. Making training data (the important part)

The loop is **record → auto-label → review → train**. Every new session makes the model better, especially from a **new person**: so far all data is from one performer.

### 4.1 Record

```powershell
python collect.py --source 1 --script --mount head --performer <your-name> --wearer <wearer-name>
```

Yellow text on screen tells the puncher what to do every 2–4 s. **Do every action as shown, bare-fisted** (the demo uses bare fists).

| Prompt | Do this |
|---|---|
| `JAB`, `CROSS`, `UPPERCUT` | That punch. Pull it short: non-contact. |
| `HOOK_L = hook with your RIGHT hand` | Hook with your **right** hand. The fist comes from the image's left. |
| `HOOK_R = hook with your LEFT hand` | Hook with your **left** hand. |
| `FEINT` | Fake a punch start. |
| `GUARD_SHIFT` | Slide your fists left/right **without punching**. Make it big and fast: this is the main false-beep cause. |
| `STEP_IN`, `HEAD_TURN`, `NOD`, `GLOVE_ADJUST`, `IDLE` | That movement, **no punch**. |

**Free boxing (data for punch prediction):** add `--free --stance orthodox` (or `southpaw`); `--performer` is required with `--free`. The session starts with a 20 s calm baseline, then alternates 60 s of prompts with 60 s of free boxing, where the puncher boxes naturally and chooses when to punch (singles, combos, a few feints). Details: [HEAD_MOUNTED_TRAINING.md](HEAD_MOUNTED_TRAINING.md) section 4.

**Tips:**
- Face the light.
- Keep your whole upper body in frame.
- Record 5–10 minutes.
- Mix clean punches (held guard first) with messy ones (straight out of another move).
- Press `q` to save. You get `data/session_<time>.npz` (keypoints, timestamps, prompts) and `.mp4` (video).

### 4.2 Auto-label punch starts

```powershell
python auto_label.py data\session_<time>.npz
```

This finds every moment a fist leaves guard and writes `data/session_<time>.auto_candidates.json`.

For `--free` sessions, also run `python combo_candidates.py data\session_<time>.npz`. It adds the follow-up punches of combos (which `auto_label.py` mostly misses) to the `--messy` review pass.

### 4.3 Review (y/n), two passes

```powershell
python review_labels.py data\session_<time>.npz
python review_labels.py data\session_<time>.npz --messy
```

- The first pass shows **clean** punches (fist held still first). The second shows **messy** ones (thrown right after another move).
- It loops a slow clip around each auto-labeled start. Only punches from punch prompts are shown: the don't-punch prompts are already known negatives. Free-boxing blocks count as punch prompts (type `FREE`).

| Key | Action |
|---|---|
| `y` | Real punch, and the start frame is right |
| `f` | Feint: a fake punch start (saved as type FEINT, never trained as a punch) |
| `n` | Not a punch, or a wrong moment |
| `,` / `.` | Nudge the start one frame earlier / later |
| `1`–`6` | Fix the punch type (optional) |
| `b` | Go back |
| `space` | Pause |
| `q` | Save and quit (you can resume later) |

**The start frame** is the first frame the fist moves **toward the target**. A pull-back, dip or shoulder drop before it is the telegraph, not the start. That exact frame is what the models learn, so it matters. (Reviews before this rule used "the first frame the fist leaves guard"; new decisions are tagged `"rule": "toward_target"`.) Labels are saved to `data/session_<time>.labels.json`.

- An orange "punch already at ..." line means another accepted punch starts within 0.2 s: usually the same punch seen twice.
- To check that two people apply the rule the same way, the second one reviews blind with `--reviewer <name>`, then `python label_agreement.py data\session_<time>.npz --reviewer <name>` compares them. Details: [HEAD_MOUNTED_TRAINING.md](HEAD_MOUNTED_TRAINING.md) section 5.

### 4.4 Train

```powershell
python early_detector.py
```

- This cross-validates: each chunk of footage is held out in turn, and the alarm threshold is picked without looking at it.
- It prints:
  - catch rates (≤50/100/150/200/300 ms)
  - median delay
  - false beeps per minute
  - feints caught
  - a per-session breakdown
- Then it trains on everything and saves `models/early_detector.joblib`.
- **Compare against the previous run before keeping a new model.** Old models go in `models/archive/`.

### 4.5 Pi version of the model + C export

The Pi runs pose through ONNX/TFLite, whose keypoints differ very slightly, so the detector is retrained on those keypoints:

```powershell
python repose_onnx.py data\session_<time>.npz      # writes data_onnx\session_<time>_onnx.*
python early_detector.py --data data_onnx --budgets 2 --save models\early_detector_onnx.joblib
python export_model_c.py models\early_detector_onnx.joblib   # -> qnx/model/early_model.c (checked identical to Python)
```

`repose_onnx.py` copies labels at the time you run it. If you review more later, copy the new `labels.json` into `data_onnx\` (renamed `_onnx`) again.

### 4.6 Lighting / frame-rate augmentation (heavy, for a fast machine)

`run_augmented_training.py` makes darker+blurred, overexposed and 15 fps copies of every session, re-runs pose on them, and compares training with and without the copies. Follow [TEAMMATE_AUGMENTATION_GUIDE.md](TEAMMATE_AUGMENTATION_GUIDE.md) step by step.

### 4.7 Sharing data through git

**Videos are not in git.** They're too big (GitHub rejects files over 100 MB) and they show people. Training doesn't need them, because the expensive features are cached. After you record and review, commit:

```
data/session_<time>.npz
data/session_<time>.labels.json
data/session_<time>.auto_candidates.json
data/session_<time>.review_*.json
data_onnx/session_<time>_onnx.npz  (+ its .labels.json, .auto_candidates.json)
features/session_<time>*           # optical flow + pixel-change caches
```

Anyone can then retrain with `python early_detector.py`. Share the `.mp4` some other way (Drive or USB) only if someone needs to re-review or re-run pose.

---

## 5. QNX / Raspberry Pi

`qnx/` holds the QNX side. Details are in [qnx/README.md](qnx/README.md).

- **Processes:**
  - `cue_driver` drives the buzzer and plays a "system down" tone if heartbeats stop.
  - `vision_stub` stands in for vision.
  - `supervisor` restarts vision if it crashes.
- **IPC:** QNX message passing (`name_attach`/`MsgSend`/`MsgSendPulse`), with a TCP backend for testing on a laptop.
- `qnx/model/early_model.c` is the detector as plain C, generated by `export_model_c.py`.
- `qnx/pose/pose_decode.c` decodes the pose model's output. It's verified identical to ultralytics.
- **Pose runtime on the Pi:** `yolo11n-pose_tflite/yolo11n-pose_float32.tflite`.
  - Verified identical to ONNX Runtime and about as fast.
  - QNX's ports include TFLite and OpenCV. OpenCV is needed for optical flow.
  - Use float32, not float16.
- **Laptop rehearsal of the whole pipeline:** `python test_bridge.py`.

**Not done yet:** flash QNX on the Pi, build `qnx/` with `qcc`, wire the GPIO buzzer, and **measure pose fps on the Pi**. If it can't do ~15+ fps, the timing fails.

---

## 6. Repo map

| File | What it is |
|---|---|
| `dodge_live.py` | **The demo app** (camera → pose → detector → cue) |
| `early_live.py` | Streaming detector + cue rules (live features verified identical to training) |
| `collect.py` | Recorder with scripted prompts |
| `free_blocks.py` | Baseline and free-boxing blocks for `collect.py --free` (schedule and screens) |
| `auto_label.py` | Finds punch starts in a session |
| `review_labels.py` | y/n review UI (`--messy` for the second pass, `--reviewer` for a blind second opinion) |
| `combo_candidates.py` | Adds combo follow-up punches in free-boxing blocks to the review |
| `label_agreement.py` | How well two reviewers agree (decisions and start frames) |
| `predict_dataset.py`, `predict_trainer.py` | Experimental punch **prediction** (warn before the fist moves), tested on unseen people; see [PREDICTION.md](PREDICTION.md) |
| `signed_motion.py`, `early_detector_signed.py` | Signed (out vs back) fist motion, a pull-back rule, and a like-for-like reactive A/B test of them |
| `early_dataset.py` | Features + labels from reviewed sessions |
| `early_detector.py` | Cross-validated training and evaluation |
| `flow_features.py` | Optical-flow features (cached in `features/`) |
| `repose_onnx.py`, `export_model_c.py` | Pi version of the detector, and the C export |
| `exit_cue.py` | Fist-leaves-frame cue (experimental, off) |
| `launch_detector.py` | Hand-made "fist leaves guard" baseline |
| `check_*.py` | Pose runtime checks (ONNX, OpenCV DNN, TFLite, C decoder) |
| `cue_client.py`, `test_bridge.py`, `live_runtime.py` | Python ↔ QNX cue bridge and capture helpers |
| `punch_telegraph.py`, `looming_ttc.py`, `anticipate_live.py`, `train_anticipator.py`, `pose_features.py`, `glove_features.py`, `early_family.py` | Older approaches, kept for reference (REPORT.md says why each was dropped). `pose_features.py` still holds the prompt lists. |
| `models/` | Trained detectors (`early_detector.joblib` laptop, `early_detector_onnx.joblib` Pi). `archive/` holds older ones. |
| `data/`, `data_onnx/`, `features/` | Sessions (keypoints, prompts, labels) and cached features |

## 7. Conventions

- **HOOK_L / HOOK_R** = the side of the **image** the fist comes from (camera not mirrored). HOOK_L is thrown with the puncher's **right** hand.
- Wrist index `k=0` = COCO keypoint 9, `k=1` = keypoint 10.
- Distances are in shoulder widths, speeds in shoulder widths per second.
- Report numbers, not vibes: catch rate as x/y, latency in ms, false beeps per minute.
- Never commit videos, secrets or the QNX license key.
