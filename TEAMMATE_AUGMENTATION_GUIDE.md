# Teammate guide: run the lighting / frame-rate training on your computer

This guide runs a heavy training job on **your** computer, because it's faster than Shawn's laptop. Follow it top to bottom. Every command is written out, so don't improvise.

**If you use Claude Code:** open this repo in it and paste the prompt from [section 11](#11-if-youre-using-claude-code). It should follow this guide exactly.

---

## 0. What this does and why (2-minute read)

The punch detector seems to get worse when the room lighting changes. Lighting can hurt it in two ways:
1. **The image itself changes:** darker, washed out, noisier. The pose model then places the wrists slightly differently.
2. **The camera slows down:** in dim light, the Logitech C270 automatically exposes longer. The frame rate drops from about 30 fps to about 15, and moving fists blur. Frame rate is already known to matter a lot: a model that had never seen 30 fps footage caught 17% of punches on it, versus 44% for one that had.

Since we can't re-record in every lighting, we **fake it.** The script takes the 4 recorded sessions and makes copies:

| Copy | What it simulates | How |
|---|---|---|
| `dim_blur` | Dim room, long exposure | Darker, crushed shadows, sensor noise, and each frame blended 45% with the previous one (motion smear). Then the pose model is re-run on it. |
| `bright` | Overexposed / backlit | Brighter, washed-out, slightly cool tint. Pose model re-run. |
| `fps15` | Camera slowed to 15 fps | Keeps only the frames closest to a 15 fps grid, with labels moved to match |

Then it trains the punch detector two ways and compares them:
- **baseline:** trained on the original footage only (how it's done now)
- **augmented:** trained on the original plus all the copies

**How the scoring stays honest:** every number is on footage the model didn't train on. Each session is cut into time chunks, and each chunk is held out in turn. When a chunk is held out, **all copies of that same chunk are held out too**, so the model never sees the test footage in another lighting. Both models are scored on the original held-out footage and on each kind of copy.

It does this twice: for the laptop model (PyTorch pose) and for the Raspberry Pi model (ONNX pose).

**Nothing you run here changes the live demo.** The script never writes to `models/`. Everything goes to a new folder, `runs/<run-name>/`, which you push back. Shawn decides whether to use the new model.

---

## 1. What you need

| Thing | Requirement |
|---|---|
| OS | Windows 10/11, macOS or Linux |
| Disk | **At least 8 GB free** inside the repo folder |
| RAM | 16 GB recommended |
| GPU | **Optional.** An NVIDIA GPU makes the pose step much faster (section 4.3). Without one it all runs on the CPU, which is fine, just slower. |
| Git | Any recent version, signed in to GitHub with access to `shawn-xionggg/boxing-assistant` |
| Python | **Exactly 3.13.x** (3.12 or 3.14 won't do; see below) |
| Time | Roughly **1.5–4 hours** of unattended running, depending on CPU/GPU. The script prints how long each step took. |
| Power | Plugged in, and set the computer **not to sleep** for the duration |

**Why exact versions matter:** the trained models are saved as Python pickles. They only load reliably on Shawn's demo laptop with **the same scikit-learn, numpy and joblib versions.** The script checks this and refuses to run if they differ.

Exact package versions (all installed in step 4):

| Package | Version | Must match? |
|---|---|---|
| Python | 3.13.x | yes |
| scikit-learn | 1.9.1 | **yes, exactly** |
| numpy | 2.2.6 | **yes, exactly** |
| joblib | 1.6.0 | **yes, exactly** |
| ultralytics | 8.4.216 | should (pose results differ slightly otherwise) |
| opencv-python | 4.13.0.92 | should (optical flow differs slightly otherwise) |
| onnxruntime | 1.26.0 | should |
| torch | installed by ultralytics (CUDA build optional) | any |
| pytest | any | no |

---

## 2. Get the 4 videos from Shawn

The session videos are **not on GitHub**: two are over GitHub's 100 MB limit, and they show a person. Shawn will send them, by Google Drive, USB or AirDrop. You need **all 4**, with **exactly these names**:

| File | Size (bytes) | Frames |
|---|---|---|
| `session_20261121_024335_748065.mp4` | 46,848,790 | 4,774 |
| `session_20261121_030328_194149.mp4` | 117,834,701 | 13,660 |
| `session_20261121_095106_097514.mp4` | 24,304,540 | 2,968 |
| `session_20261121_115113_071950.mp4` | 286,699,703 | 22,181 |

- **Put them in the `data/` folder of the repo,** next to the matching `.npz` files that are already there. For example: `boxing-assistant/data/session_20261121_115113_071950.mp4`.
- **Don't rename them.** If Google Drive downloads them as a `.zip`, unzip first.
- If a size doesn't match to the byte, the download is incomplete or wrong. Download it again.

Check the sizes:
- Windows PowerShell: `Get-ChildItem data\*.mp4 | Select-Object Name, Length`
- macOS/Linux: `ls -l data/*.mp4`

Total: about 476 MB.

---

## 3. Get the latest code

**If you've never cloned the repo:**
```
git clone https://github.com/shawn-xionggg/boxing-assistant
cd boxing-assistant
```

**If you already have it:**
```
cd boxing-assistant
git pull
```

Do the clone or pull **before** copying the videos in, or put the videos in afterwards. Git ignores `.mp4` files, so a pull won't touch them either way.

Make sure these files exist now (they arrive with the pull): `run_augmented_training.py`, `augment_lighting.py`, `early_detector.py`, `yolo11n-pose.pt`, `yolo11n-pose.onnx`, and the folders `data/`, `data_onnx/`, `features/`.

---

## 4. Install Python 3.13 and the packages

### 4.1 Python 3.13

- **Windows:** download Python **3.13.x** from https://www.python.org/downloads/windows/ (the "Windows installer (64-bit)"). In the installer, tick **"Add python.exe to PATH"**. Check it worked: `py -3.13 --version` should print `Python 3.13.x`.
- **macOS:** download the 3.13.x macOS installer from https://www.python.org/downloads/macos/. Check: `python3.13 --version`.
- **Linux:** use your package manager or pyenv to get 3.13. Check: `python3.13 --version`.

### 4.2 A private environment for this project (venv)

This keeps these exact versions away from anything else on your computer. Run these **inside the repo folder.**

**Windows (PowerShell):**
```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install scikit-learn==1.9.1 numpy==2.2.6 joblib==1.6.0 ultralytics==8.4.216 opencv-python==4.13.0.92 onnxruntime==1.26.0 pytest
```

**macOS / Linux:**
```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install scikit-learn==1.9.1 numpy==2.2.6 joblib==1.6.0 ultralytics==8.4.216 opencv-python==4.13.0.92 onnxruntime==1.26.0 pytest
```

> From here on, this guide writes **`PY`** for your venv's Python:
> - Windows: `.venv\Scripts\python.exe`
> - macOS/Linux: `.venv/bin/python`
>
> Always use it. Don't "activate" the venv, and don't use a bare `python`: calling the venv's Python by its path means the right versions are always used. On Windows this also avoids PowerShell's script-execution prompt. The `.venv` folder isn't committed.

If pip says a version doesn't exist for your platform, **stop and tell Shawn.** Don't substitute another version of scikit-learn, numpy or joblib.

### 4.3 Optional: use your NVIDIA GPU for the pose model

Skip this if you have no NVIDIA GPU, or on macOS.

1. Check what you have: `PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
   - If it already prints `True`, you're done.
   - If it prints `False`, continue with step 2.
2. Go to https://pytorch.org/get-started/locally/ and choose: Stable, your OS, **Pip**, **Python**, and the newest **CUDA** option. It shows a command like `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX`.
3. Run that command with **your venv's Python**, adding `--force-reinstall --no-deps`, for example:
   - Windows: `.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX --force-reinstall --no-deps`
   - Linux: the same, with `.venv/bin/python`

   `--no-deps` stops it from changing the numpy version.
4. Check again: `PY -c "import torch; print(torch.cuda.is_available())"` should print `True`.
5. Make sure numpy is still the right version: `PY -c "import numpy; print(numpy.__version__)"` should print `2.2.6`. If it doesn't, run `PY -m pip install numpy==2.2.6`.

Only the PyTorch pose step uses the GPU. The ONNX (Pi) pose step, optical flow and training run on the CPU, which is expected.

---

## 5. Check everything (about 10 seconds)

```
PY run_augmented_training.py --check-only
```

You should see `ok` on every line and **`all checks passed`** at the end:

```
  ok    Python 3.13.x (...)
  ok    scikit-learn 1.9.1
  ok    numpy 2.2.6
  ok    joblib 1.6.0
  ok    ultralytics 8.4.216
  ok    opencv-python 4.13.0.92
  ok    onnxruntime 1.26.0
  info  torch ...; NVIDIA GPU (CUDA): YES - <your GPU>     (or "no", which is fine)
  ok    session_20261121_024335_748065.mp4: 4774 frames (47 MB)
  ok    session_20261121_030328_194149.mp4: 13660 frames (118 MB)
  ok    session_20261121_095106_097514.mp4: 2968 frames (24 MB)
  ok    session_20261121_115113_071950.mp4: 22181 frames (287 MB)
  ok    ... GB free (need ~8 GB)
  all checks passed
```

- `FAIL` lines must be fixed before going on. Section 9 covers each one.
- `warn` lines (ultralytics, opencv, onnxruntime) are allowed, but better fixed with the `pip install ...==version` the line suggests.

---

## 6. Smoke test (about 1 minute)

```
PY run_augmented_training.py --smoke
```

This runs the whole pipeline on a 12-second slice of one session, in a temporary `smoke_test/` folder, and then deletes it. It ends with:

```
SMOKE TEST PASSED (pipeline runs end to end). ...
```

If it fails, stop and send Shawn the last ~30 lines of output.

---

## 7. The real run (1.5–4 hours, unattended)

1. Plug in the charger and set the computer not to sleep:
   - Windows: Settings → System → Power → Screen and sleep → "Never" while plugged in.
   - macOS: run the command as `caffeinate -i .venv/bin/python run_augmented_training.py` instead.
2. Run:
   ```
   PY run_augmented_training.py
   ```
3. Leave the terminal open. You can use the computer lightly, but heavy use slows it down.

What you'll see, in this order:
1. **CHECK:** the same as step 5.
2. **AUGMENT:** the slow part.
   - Lines like `session_..._dim_blur: 3000/22181`.
   - Each `done ... in Ns` line is one copy finished.
   - 8 lighting copies (4 sessions × 2), each re-run through the pose model, plus 4 quick `fps15` copies.
3. **FEATURES:** one `ok features for ...` line per copy (optical flow).
4. **PI COPIES:** the same lighting copies through the ONNX pose model, plus quick fps15 copies.
5. **TRAIN:** 4 trainings.
   - Each prints `run <name> ... (log: runs/<run>/<name>.log)`, then `done <name> in X min`.
   - The two `*_augmented` ones take the longest.
   - While a training runs, the screen stays quiet. That's normal; the detail goes to the `.log` file.
6. **RESULTS:** a table, also saved as `runs/<run>/summary.md`.
7. **`ALL DONE in N min`**

**If it stops halfway** (crash, sleep, closed terminal), just run the same command again. Finished pieces are skipped:
- copies with their `.aug.json` file written
- trainings whose log says `saved`

To continue the **same** run folder, pass its name: `PY run_augmented_training.py --run-name aug_YYYYMMDD_HHMM` (the name printed in the TRAIN step).

**Short on time?** Add `--skip-pi` to do only the laptop model. That's about half the time. The Pi model can be done later with the same command without `--skip-pi`.

Disk used when finished: about 3–6 GB, in `data_aug/`, `data_onnx_aug/` and `features/`. All of it is git-ignored.

---

## 8. Send the results back

1. Look at the results: open `runs/<run-name>/summary.md`. It looks like this (made-up numbers):

   | Training | Original footage | bright copies | dim_blur copies | fps15 copies |
   |---|---|---|---|---|
   | laptop_baseline | 52% caught <=200ms, 3.0 fa/min | 40% ... | 32% ... | 24% ... |
   | laptop_augmented | 53% caught <=200ms, 3.0 fa/min | 45% ... | 44% ... | 38% ... |

   - **What "better" looks like:** `*_augmented` catches more on the copy columns than `*_baseline`, while the "Original footage" column stays about the same at a similar false-beep rate (fa/min).
   - With about 380 punches, differences under about 5 points are noise.

2. Commit **only the `runs/` folder** and push. Replace `<run-name>` with your folder name:
   ```
   git pull
   git add runs/<run-name>
   git status
   ```
   `git status` must list **only** files under `runs/<run-name>/`:
   - `summary.md`
   - 4 `.log` files
   - 4 `.joblib` files, about 1–4 MB each

   If anything else shows up as staged, run `git restore --staged <that file>`.
   ```
   git commit -m "Augmentation run <run-name>: lighting + 15 fps copies"
   git push
   ```
   If the push is rejected because the remote has new commits, run `git pull --rebase` and then `git push`.

3. Tell Shawn it's pushed, and paste the summary table in your message.

**Do NOT:**
- copy anything into `models/`
- run `export_model_c.py`
- commit videos, `data_aug/`, `data_onnx_aug/` or `.venv/`
- edit any code to "make it work". If something fails, send the error instead.

Shawn installs the new model if it's better, which also involves regenerating the C model for the Pi.

---

## 9. Troubleshooting

| You see | Fix |
|---|---|
| `FAIL Python 3.12` (or 3.14) | You ran a different Python. Use the venv's Python (`PY`), and create the venv with 3.13 (section 4.1). |
| `FAIL scikit-learn 1.x, need exactly 1.9.1` (or numpy/joblib) | `PY -m pip install scikit-learn==1.9.1 numpy==2.2.6 joblib==1.6.0` |
| `FAIL ... .mp4 missing` | The video isn't in `data/`, or its name differs. Compare it with the table in section 2. |
| `FAIL ... N video frames but M pose rows` | Wrong or truncated video. Download it again and check the byte size. |
| `FAIL only X GB free` | Free up disk space, or move the repo to a bigger drive. |
| `FAIL folder data_onnx/ missing` | Run `git pull`. You have an old copy of the repo. |
| `No module named cv2` / `torch` / `ultralytics` | You ran a Python other than the venv's. Use `PY`. |
| `torch.cuda.is_available()` is `False` with an NVIDIA GPU | Redo section 4.3 and update the NVIDIA driver. Or just run on the CPU. |
| `CUDA out of memory` | Close other GPU programs and re-run. It resumes. |
| A training says `FAIL ... exited with 1` | Open that `.log` file and send Shawn the last 30 lines. |
| Very slow AUGMENT (over 1 s per frame) | Check Task Manager / Activity Monitor for other heavy programs. Without a GPU, ~20–40 ms per frame is normal. |
| The computer slept and the run died | Re-run the same command. It resumes. |
| `git push` rejected | `git pull --rebase`, then `git push`. |

---

## 10. What each new file is (for the curious)

| File | Role |
|---|---|
| `run_augmented_training.py` | The one command. Does checks, makes the copies, features, Pi copies, 4 trainings and the summary. Safe to re-run: it skips finished work. |
| `augment_lighting.py` | Makes the lighting copies. `PRESETS` holds the exact gain, gamma, tint, noise and blur values. `subsample()` makes the frame-rate copies. `--compare` measures how much a preset moves the pose points. |
| `early_detector.py --aug-data DIR [--aug-train]` | Cross-validated training. `--aug-data` scores on the copies; `--aug-train` also trains on them. Held-out time chunks are excluded across all copies. |
| `runs/<run>/` | Output: `laptop_*.joblib` and `pi_*.joblib` models, full logs, `summary.md` |

---

## 11. If you're using Claude Code

Paste this into Claude Code, opened in the repo folder:

> Follow `TEAMMATE_AUGMENTATION_GUIDE.md` exactly to run the augmentation training on this machine. Rules:
> - Use a Python 3.13 venv in `.venv`, and always call `.venv`'s Python by its full path, never a bare `python`.
> - Install exactly the pinned versions in section 4. Never substitute other versions of scikit-learn, numpy or joblib. If one can't be installed, stop and tell me.
> - The 4 session videos must already be in `data/` (section 2). If `--check-only` reports any FAIL, stop and tell me which one. Don't work around it.
> - Run `--check-only`, then `--smoke`, then the full run (`python run_augmented_training.py`) in the background, and report progress.
> - Do NOT edit any `.py` files, do NOT write to `models/`, do NOT run `export_model_c.py`, and do NOT commit anything except `runs/<run-name>/`.
> - When it's done, show me `runs/<run-name>/summary.md`, then commit only `runs/<run-name>/` and push (section 8). Don't add a Claude co-author line to the commit.
> - If anything fails, show me the error and the last 30 lines of the relevant log, and stop.
