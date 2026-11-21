# Addendum to ANTICIPATION_TASK.md (pass 2)

The pass-1 files (`pose_features.py`, `collect.py`, `train_anticipator.py`, `anticipate_live.py`, `test_anticipation.py`) already exist. **Extend them; don't rewrite from scratch.** The user's research guide is `telegraphing-guide.md` (copied into this folder). This addendum turns it into requirements. Priority is still **working and honestly measured over fast.**

## 1. Fix the depth blind spot: relative glove area
A straight punch toward the lens barely moves the wrist in the image; it only grows. The wrist-speed rule alone misses jabs and crosses, and so does auto-labeling built on it.
- Add optional colored-glove segmentation by **importing** `ColorModel` and `find_blobs` from `looming_ttc.py` (don't modify that file). `collect.py` and `anticipate_live.py` take `--color <preset>` or click-to-sample, like `looming_ttc.py`.
- Per frame, assign each glove blob to the nearest wrist. Feature: **relative glove area = blob area / shoulder_width²** (cancels step-in and zoom: if everything scales, the ratio stays put), plus its rate of change and log-rate. Missing glove → impute + mask flag.
- Store raw glove blobs (centroid, area) per frame in the npz.
- **Auto-onset = the earliest of** (a) the wrist-speed burst and (b) a relative-glove-area growth burst (configurable threshold on the log-rate), backtracked to where motion began.

## 2. Gesture families and scripted sessions
- `collect.py --script` mode: shows a random **on-screen command** for the puncher every 2–4 s from: `JAB, CROSS, HOOK_L, HOOK_R, UPPERCUT, FEINT, STEP_IN, HEAD_TURN, NOD, GLOVE_ADJUST, IDLE`. **The prompt must be visual only, never audible**, so the blindfolded wearer can't hear it. Store the command and its timestamp, and attach each onset to the active command.
- `FEINT, STEP_IN, HEAD_TURN, NOD, GLOVE_ADJUST, IDLE` are **negatives**: any cue during them is a false alarm.
- `--performer <name>` is stored in every session.
- Cue = game command by family (configurable dict), scored against the scripted label, not the starting side: `JAB/CROSS → DUCK`, `HOOK_L → DODGE RIGHT`, `HOOK_R → DODGE LEFT`, `UPPERCUT → BACK`. Add a `BACK` cue (a distinct beep, e.g. two short 1500 Hz) wherever CUES are defined in the new files. HOOK_L / HOOK_R = the hook coming from image-left / image-right.
- The model predicts the family (or NONE); the cue maps from family.

## 3. Human onset labels: `label_onsets.py`
Auto labels are biased, so add a frame-stepping labeler for saved session videos:
- `,` / `.` = step one frame, `j` / `l` = step 10 frames, space = play/pause
- `1` = mark **first preparation** (if visible), `2` = mark **glove departure**, `f` = set family, `u` = undo
- Disagreements stay as a frame range: allow 2 labelers (`--labeler name`) and store both.
- Training prefers human departure labels over auto labels when present.

## 4. Candidate → confirm state machine (live + offline eval)
- States: `IDLE → CANDIDATE` (predictor probability > low threshold, logged silently) `→ CONFIRMED` (a glove/wrist signal meets the calibrated rule) `→ cue`.
- Output **UNKNOWN** and **clear motion history** when a required landmark disappears or the tracker jumps: person center moves > 0.5 shoulder widths in one frame, or shoulder width changes > 30% in one frame.
- `anticipate_live.py --mode predict` (cue at a confident candidate) vs `--mode confirm` (cue only at confirmation). Offline eval reports both.
- Timestamps: video timestamps for replay, capture timestamps live. Count detection time and classification time separately: the first sign of movement may come before its direction is clear.

## 5. Honest evaluation in `train_anticipator.py`
- **Leave-one-performer-out** when ≥ 2 performers exist. Choose thresholds on the training performers only, then score the held-out performer **without retuning**. Fall back to leave-one-session-out with a printed warning.
- Per family: on-time correct cues / attempts, wrong-direction cues, unknowns, and median + worst onset-to-cue delay.
- **Keep negative delays (cue before departure = prediction) separate from positive delays (after departure = early detection).** Only call something "prediction" if the cue comes before departure on unseen trials **and** stays quiet on negatives.
- False buzzes per minute of **negative footage** (scripted negatives + idle).
- **Ablation table on identical held-out clips:** (a) body-relative wrist speed only, (b) + relative glove area, (c) + shoulder/pose candidate predictor. Report the lead time bought and the false alarms added by each.
- Don't report a p99 with < 50 trials; say "too few trials" instead.

## 6. `first_experiment.py`
Input: 1+ clips (or npz) of a straight reach toward the camera, a lateral sweep, and a step-in with hands held at the chest. Output: a PNG per clip plotting body-relative wrist speed and relative glove area over time, with auto-onsets marked. This exposes the depth limitation (straight), checks lateral tracking (sweep), and tests whether a step false-alarms (step). matplotlib ships with ultralytics.

## 7. Wording in overlays and logs
Label each cue as **"PREDICTED (−X ms)"** if it fired before departure and **"EARLY DETECT (+X ms)"** if after. Never call a post-onset cue a prediction.

## 8. Safety
Non-contact game: the performer stays **beyond the wearer's reach**. Put that line on the `collect.py` and `anticipate_live.py` start screens.

## Tests
Extend `test_anticipation.py` with synthetic: a straight punch (wrist static, glove area grows → must be detected via glove area), a step-in (whole body scales → must NOT trigger), a tracker jump (→ UNKNOWN + history reset), scripted negatives counted as false alarms, and leave-one-performer-out splitting. Run it plus `py_compile` on all files and make them pass. Finish with a short summary.
