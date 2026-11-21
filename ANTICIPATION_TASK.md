# Task: punch ANTICIPATION (predict a punch before the fist moves)

Read `AGENTS.md` first for project context. Existing reference code: `punch_telegraph.py` (pose + reactive wrist-speed detector) and `looming_ttc.py`. Do not modify those two files.

## Goal
The user wants a cue **before the punch starts**, not when the fist is already flying. Learn the attacker's **wind-up / telegraph** (shoulder rotation, weight shift, rear elbow chambering, guard drop, lean) from pose keypoints. Predict "a punch will start within the next ~100–300 ms, and from which side".

**Priority: make it WORK and measure it honestly. Do not optimize speed.** Default to the more accurate `yolo11s-pose.pt` (keep a `--model` flag; ultralytics auto-downloads weights). Accuracy and a trustworthy evaluation matter more than fps.

## Deliverables (new files in the repo root)

### 1. `pose_features.py` (shared module)
- `PoseRunner`: wraps an ultralytics YOLO pose model and returns the **largest person's** keypoints (17×2) + confidences, or None.
- `normalize(kps)`: origin = shoulder midpoint (COCO kp 5, 6), unit = shoulder width. Must work with a moving camera.
- `frame_features(history)`: from the last W frames (default W = 8, ~270 ms at 30 fps) of normalized keypoints and timestamps, build a feature vector including at least:
  - normalized positions of nose, shoulders, elbows, wrists, hips (use hips only if their confidence is high, else fill and add a mask flag)
  - velocities (per second, using real timestamps) of the above
  - shoulder-line angle and its rate of change, and raw shoulder width in pixels (relative change over the window = torso rotation / step-in proxy)
  - elbow angles (shoulder-elbow-wrist) and their rates
  - wrist height relative to the nose (guard drop)
  - torso lean (shoulder midpoint vs hip midpoint)
- Robust to missing keypoints (low confidence → impute the last value + a flag). Never crash on a frame with no person.
- `reactive_onset(...)`: the same wrist-speed rule as `punch_telegraph.py` (speed relative to the shoulders > 5 shoulder-widths/s for 2 frames). It labels the **onset frame** as the first frame of that burst where speed rose above a lower threshold (default 2.0), i.e. backtrack to where the motion really began. Also return the side (wrist image x relative to the shoulder midpoint: < -0.6 → LEFT-side punch, > 0.6 → RIGHT-side, else CENTER).

### 2. `collect.py`
- Live webcam (`--source 0`, DSHOW on Windows, 640×480@30) or a video file.
- Per frame: save the timestamp, keypoints and confidences. On quit, write `data/session_<timestamp>.npz` and also save the raw video to `data/session_<timestamp>.mp4` so sessions can be re-labeled or re-posed later.
- Show the live skeleton and the auto-detected onsets (flash + counter) so the user knows labels are being captured.
- Key `x` = mark "that last onset was wrong" (removes the most recent auto label). Key `p` = manual punch-onset mark at the current frame (for punches the auto rule missed). Store both in the npz.
- Print on screen what to do: "Puncher: mix punches with idle guard movement, bouncing, feints. Aim for 100+ punches across sessions."
- `--from-video data/x.mp4` mode: re-run pose on a saved video to regenerate the npz (for switching models).

### 3. `train_anticipator.py`
- Load all `data/*.npz`. Build per-frame features.
- Labels per frame t: class = side of the next onset if an onset occurs in `(t, t + H]`, where H defaults to 300 ms (by timestamps, not frame count). Otherwise NONE. **Exclude** frames from onset to onset + 400 ms (the punch itself / the recovery) from training, so the model learns the wind-up, not the punch.
- **Split by session** (leave-one-session-out, or the last session as test). Never a random frame split, which leaks.
- Model: sklearn `HistGradientBoostingClassifier` (class_weight balanced). Also try a small `MLPClassifier` with standardized features, and keep whichever scores better on the held-out sessions. Save to `models/anticipator.joblib` with the feature config (W, H, feature names, scaler).
- **Event-level evaluation** printed clearly (this is what matters, not frame accuracy). Run the live cue logic offline (prob > threshold for 2 frames, cooldown 0.6 s) on held-out sessions and report:
  - % of punches warned **before** their onset, and the median / p25 / p75 **lead time** (ms between first warning and onset)
  - % of punches warned in time to be useful (lead ≥ 100 ms)
  - false alarms per minute of non-punch time
  - side accuracy on correctly warned punches
  - the same metrics at several thresholds (e.g. 0.3 / 0.5 / 0.7), so the user can pick the trade-off
  - a baseline row: the reactive detector alone (its lead time is ≤ 0 by definition) for comparison
- Print feature importances (permutation importance on held-out data, top 15) so we learn which telegraphs matter.
- If there's too little data (< 30 onsets), say so clearly instead of training.

### 4. `anticipate_live.py`
- Live webcam: pose → features → model probability. **Two cues:**
  - **Early warning** (predictor, prob > `--thresh` for 2 frames): a beep pattern from `punch_telegraph.py` CUES by side (LEFT-side punch → "DODGE RIGHT", RIGHT-side → "DODGE LEFT", CENTER → "DUCK").
  - **Reactive backup** (wrist-speed onset) if no early warning fired in the last 0.5 s.
- When a reactive onset follows an early warning, print and overlay the **lead time**: "warned 140 ms before the punch started". Keep running stats (warned %, median lead, false alarms/min) and print them on quit.
- Overlay: skeleton, a probability bar per class, the cue text, and the lead time of the last punch.
- Log events to `anticipate_log.csv`.

### 5. Tests / verification (you can't use a webcam in the sandbox)
- Write `test_anticipation.py` with synthetic keypoint sequences: an idle "person" with noise, plus punches where the shoulder rotates and the rear elbow chambers for ~200 ms **before** the wrist shoots out. Check that `reactive_onset` finds onsets at the right frames, that labeling excludes punch frames, that a model trained on synthetic sessions warns before onset on a held-out synthetic session, and that the event metrics compute correctly. Run it and make it pass.
- Run `python -m py_compile` on every new file.
- Add `scikit-learn` and `joblib` to `requirements.txt`.

## Constraints
- Python 3.13, ultralytics 8.4, OpenCV, numpy, scikit-learn (all installed). No new heavy deps.
- Windows: use `winsound` for beeps in a thread, falling back to `\a`.
- Don't download datasets. Don't touch anything outside this folder.
- At the end, print a short summary: the files created, how to run each step, and test results.
