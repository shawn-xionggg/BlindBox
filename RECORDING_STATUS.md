# Where we are: punch-prediction recordings (2026-11-21)

The live demo (reactive detector, `dodge_live.py`, `models/early_detector*.joblib`) is **unchanged**. Everything here is the separate attempt to *predict* punches; see [PREDICTION.md](PREDICTION.md).

## 1. Set up (Windows)

Tested with Python 3.12.10 and these versions: torch 2.11.0+cu128 (CUDA on), opencv 5.1.0, scikit-learn 1.9.1, ultralytics 8.4.216, numpy 2.5.4.
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
For the GPU, install the CUDA build of torch from pytorch.org first. Without it, pose runs on the CPU and the recorder gets slower (fewer fps).
Camera: the Logitech C270 was `--source 1` on our laptop (0 = built-in webcam).
Check it all works:
```powershell
.venv\Scripts\python.exe -m pytest -q test_free_blocks.py test_labeling.py test_predict.py test_signed.py
```

## 2. Record a person

1. **Light check** (records nothing). Puncher stands where they'll box, 1.5–2 m away, in the middle of the picture. Add light until the fps line is green (~30). If it beeps, the shoulders are lost or the puncher is at the edge: step back or center.
   ```powershell
   .venv\Scripts\python.exe collect.py --source 1 --check-light
   ```
   Press `q`.
2. **Record** (about 8 minutes, at least 3 free-boxing blocks, then `q`):
   ```powershell
   .venv\Scripts\python.exe collect.py --source 1 --script --free --mount head --performer NAME --stance orthodox --wearer WEARER
   ```
   - `--performer`: a fixed name per person (tests leave one person out, so the same person must always get the same name).
   - `--stance`: orthodox (left hand leads) or southpaw.
   - The screen alternates: 20 s calm BASELINE (no punches), 60 s of prompts, 60 s of FREE BOXING.
   - Hooks are shown as **RIGHT-HAND HOOK / LEFT-HAND HOOK** (saved as HOOK_L / HOOK_R = the image side the fist comes from).
   - It **beeps continuously** while the puncher is out of frame. Fix the framing; don't stop the recording.
3. Add a line for the session and person to `data/session_notes.json` (skill level, anything unusual).

Every punch stops short of the wearer. Soft gloves.

## 3. Data so far

`.npz` files (pose, timestamps, prompts) are in git. **The `.mp4` videos are not** (too big, and they show people). Labeling needs the videos, so copy them by Drive/USB into `data/` next to their `.npz`.

| Session | Puncher | Skill | Length | Free boxing | Tracked | Notes |
|---|---|---|---|---|---|---|
| session_20261121_192643_669947 | you (southpaw) | | 10.1 min | 4.5 min (10 s blocks) | 90% | |
| session_20261121_194856_263813 | volunteer1 | beginner | 4.5 min | 2 min | 90% | dim, ~21 fps |
| session_20261121_200608_314532 | volunteer1 | beginner | 6.5 min | 3 min | 97% | |
| session_20261121_210415_748239 | volunteer2 | intermediate–advanced (semi-pro Muay Thai) | 7.4 min | 3 min | 61% | off-center, leaned in; **hooks swapped** |
| session_20261121_213319_013537 | volunteer2 | intermediate–advanced | 8.4 min | 4 min | 99% | **hooks swapped**; ~21–23 fps |
| session_20261121_223419_058659 | volunteer3 (orthodox) | novice | 9.5 min | 4 min | 98% | ~29–31 fps |
| session_20261121_230741_390620 | volunteer4 (orthodox) | beginner | 3.1 min | 1 min | 99% | short; coming back for more |
| session_20261121_231908_874846 | volunteer5 (orthodox) | beginner | 5.4 min | 2 min | 99% | |
| session_20261121_233345_289835 | volunteer6 (orthodox) | beginner | 5.1 min | 3 min | 97% | 30 s prompt blocks; ~21 fps; **erratic** (e.g. 3 hooks in a row) |

A 36 s test run (`session_20261121_192413_312171`) is junk and isn't in git.

**Hooks swapped:** volunteer2 read the old "HOOK_L" prompt as his left hand, so his prompted hooks are the opposite hand. `data/session_notes.json` marks those sessions, and `auto_label.py` swaps HOOK_L/HOOK_R back for them (through `session_notes.py`). The recorded files are not edited.

## 4. Next steps

1. **Label today's sessions** (needs the `.mp4`s), per session, in this order (details in HEAD_MOUNTED_TRAINING.md sections 4–5):
   1. `auto_label.py data\SESSION.npz`
   2. `combo_candidates.py data\SESSION.npz` (adds follow-up punches of combos)
   3. `review_labels.py data\SESSION.npz` (clean pass), then `--messy`
   4. `repose_onnx.py` and the flow-feature cache
2. **Train and test the predictor** on 3 people: `python predict_trainer.py --signed > models\candidates\predictor.log`. Leaving one person out is now possible. Models only go to `models/candidates/`.
3. Optional: the reactive A/B with the signed features (`early_detector_signed.py`, PREDICTION.md section 6). It's CPU-heavy, so don't run it while recording.
4. More people. Each new person is worth more than more minutes from the same person.

## Rules

- Never delete sessions. Never commit `.mp4`.
- Models go to `models/candidates/`. Never overwrite `models/early_detector*.joblib`.
- Ask before changing anything the reactive demo uses.
