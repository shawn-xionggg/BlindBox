# Codex task: live app uses the launch-from-guard detector, and faster capture

Read `AGENTS.md`, then `launch_detector.py` and `evaluate_detector.py` (new, by Claude). The user is asleep: **no camera is available**. Verify everything offline with recorded sessions in `data/` (read-only) and with synthetic tests.

## Background (measured on 121 human-reviewed punches)
- The old reactive wrist-speed detector (`pose_features.ReactiveOnset`) fires ~250–350 ms after the fist leaves guard. It also buzzes on retractions (pulling the fist back is as fast as throwing it).
- The glove auto-onset (`GloveOnset`) floods false alarms: the orange glove matched skin.
- `launch_detector.LaunchDetector` (fist leaving a held guard position, body-relative, per wrist) is the best measured so far. `python evaluate_detector.py` prints its catch rate vs false alarms per threshold.
- The recorder only reached **~20.7 fps** (48 ms/frame) although pose takes ~14 ms. Something else in the loop costs ~30 ms.

## Do this
1. **`anticipate_live.py`:** add `--detector {launch,reactive}`, default `launch`, plus `--launch-thresh` (default 0.55). The launch detector takes the raw keypoints/confidences each frame (`det.update(t, kps, conf)`) and becomes the backup/confirm cue in place of `ReactiveOnset`. Side → cue: `LEFT → DODGE RIGHT`, `RIGHT → DODGE LEFT`, `CENTER → DUCK`. Overlay: draw `det.last_disp` per wrist as two small bars with the threshold line. Keep `--detector reactive` working as before.
2. **`collect.py`:** auto-onsets use `LaunchDetector` instead of `ReactiveOnset`. Disable glove auto-onsets by default (keep them behind `--glove-onsets`). Keep the glove mask preview Claude added.
3. **Speed:** add per-stage timing (capture wait, pose, glove segmentation, overlay, imshow, video write, other) to both apps, printed as averages on quit. Add `--benchmark <video.mp4>` to `collect.py` that runs the same per-frame processing on a recorded video **without** the camera or window and prints per-stage ms + achievable fps. Use it on `data/session_20261121_030328_194149.mp4` (first ~600 frames is enough). Then cut the overhead:
   - Grab frames in a background thread (always newest frame + its capture timestamp from `time.perf_counter()` at grab). In `collect.py`, keep video and npz **1:1**: write exactly the frames you process.
   - Move video writing to a writer thread with a bounded queue.
   - Anything else the benchmark shows. Report before/after numbers.
   - Also set `CAP_PROP_FOURCC` to MJPG at 640×480@30 when opening a webcam (fall back silently if refused). Note in the summary that the C270 may drop to ~15–20 fps in dim light (auto exposure), which code can't fix.
4. **Tests:** extend `test_anticipation.py` with LaunchDetector tests:
   - held guard then a launch → one event, with the right side
   - a retraction (fist returning to guard) → no event
   - a whole-body shift (all keypoints translate/scale together) → no event
   - one wrist lost → the other wrist still fires
   - cooldown respected

   Keep all 30 existing tests passing. Run the suite and `py_compile` on all files.

## Don't touch
`launch_detector.py` (if you find a bug, describe it in your summary instead of editing), `evaluate_detector.py`, `diagnose_signals.py`, `auto_label.py`, `review_labels.py`, `punch_telegraph.py`, `looming_ttc.py`, any new file whose name starts with `flow_` or `early_`, and everything under `data/` and `models/`.

## Finish with
A short summary: files changed, benchmark before/after (ms per stage, fps), test results, and anything you couldn't verify without a camera.
