# Recording guide (for the person running the recorder, and their Claude Code)

**If you are Claude Code:** read this whole file and follow it as the workflow. The human says things like "new volunteer, beginner, orthodox", "green, start recording" and "done, check it", and you run the steps below. Run the recorder in a terminal the human can see (it opens a camera window). Never delete sessions, never commit `.mp4`, and don't change code. If something breaks, say what happened and ask.

Background: [RECORDING_STATUS.md](RECORDING_STATUS.md) (what's recorded so far) and [PREDICTION.md](PREDICTION.md) (why). The live demo detector isn't touched by any of this.

## 0. Once per laptop

```powershell
git pull
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q test_free_blocks.py
```
- For the GPU (much faster pose, so more fps), first install the CUDA build of torch from pytorch.org. Check: `.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"` should print `True`.
- Find the C270: try `--source 1` in the light check below. Wrong camera → try `0` or `2`, and use that number everywhere.
- Plug the laptop in. Set Windows power mode to **Best performance**. In the NVIDIA Control Panel, set **Prefer maximum performance** for `.venv\Scripts\python.exe`. Without this, the recorder drops to ~21 fps.

## 1. Names (important)

- Each person gets **one fixed name**: `volunteer9`, `volunteer10`, ... (volunteers 1–8 are taken). Same person again = same name, e.g. **volunteer4** is coming back for a longer session. Two different people must never share a name: the tests leave one person out at a time.
- `--stance orthodox` (left foot and hand forward, most right-handers) or `--stance southpaw`.
- `--wearer` = whoever wears the head camera.
- Ask each person their **skill**: beginner / novice / some training / experienced. We have mostly beginners, so people who have trained, and southpaws, are the most valuable.

## 2. For each person

**a. Brief them (30 s):**
- Stand 1.5–2 m from the wearer, in the middle of the picture. **Never hit the wearer**: every punch stops short. Soft gloves.
- The screen shows prompts. Hooks say **RIGHT-HAND HOOK / LEFT-HAND HOOK**: use that hand.
- **BASELINE** (first 20 s, blue border): guard up, move, **no punches**.
- **FREE BOXING** (green border): box naturally and choose when to punch, singles and combos, a few feints. Eyes on the wearer, not the screen.

**b. Light check** (records nothing):
```powershell
.venv\Scripts\python.exe collect.py --source 1 --check-light
```
Puncher in position. The fps line at the bottom must be **green (~30)**. Yellow/red → add light. If it **beeps**, the shoulders are lost or the puncher is at the edge → move back or center. Press `q` (or Ctrl-C in the terminal).

**c. Record** (about 5 minutes):
```powershell
.venv\Scripts\python.exe collect.py --source 1 --script --free --mount head --performer volunteer9 --stance orthodox --wearer NAME --prompt-seconds 30
```
- Layout: 20 s baseline, then 30 s prompts / 60 s free boxing, repeating.
- Press **q** after the **third** free-boxing block (~5 min). Longer is fine; shorter than 2 free blocks isn't worth much.
- Beeping during the session = out of frame. Fix the position, **don't stop**.

**d. Check it:**
```powershell
.venv\Scripts\python.exe check_session.py --latest
```
- `[GOOD]`: tracked ≥ 95% and ~30 fps. `[OK]`: usable. `[CHECK]`: below 85% tracked or under 20 fps. Tell the human why (the "lost" line says: 0.3–2 s gaps = leaning in, > 2 s = at the edge) and fix it before the next person.

**e. Write it down:** add the person and the session to [data/session_notes.json](data/session_notes.json), the same way as the existing entries:
- under `performers`: `"volunteer9": {"skill": "beginner", "stance": "orthodox"}` (plus a `note` for anything odd: very erratic, injured, swapped hands)
- under `sessions`: `"session_YYYYMMDD_HHMMSS_ffffff": {"performer": "volunteer9", "wearer": "NAME", "note": "5.2 min, 3 min free boxing; 98% tracked"}`
- Check the JSON still loads: `.venv\Scripts\python.exe -c "import session_notes as s; print(s.load()['performers'].keys())"`
- Add a row to the table in RECORDING_STATUS.md section 3.
- If the puncher threw the **opposite hook** to the prompt all session, add `"hooks_swapped": true` to that session.

## 3. Game sessions (the leaderboard)

- Record them with the **same `collect.py` command**. Anything else won't save files the training tools can read.
- A helper presses **`p`** in the camera window on every real punch (a rough label, and the game's punch count).
- Add `"game": true` to the session's notes: people trying to beat a detector box differently, so these are kept apart in testing.
- With strangers, consider the camera on a tripod at head height (`--mount desk`) instead of on someone's head.
- Everyone knows they're being recorded and agrees. Leaderboard by nickname.

## 4. Send the data

After every 1–3 people:
```powershell
git add data/session_*.npz data/session_notes.json RECORDING_STATUS.md
git commit -m "volunteer9 (beginner, orthodox) free-boxing session"
git pull
git push
```
- **Only `.npz` and notes.** `.gitignore` already keeps videos out. Never commit `.mp4`.
- Send the `.mp4` files separately (Drive / USB) to whoever labels. Labeling needs the video next to its `.npz` in `data/`.
- `git pull` before pushing. The labeler pushes `*.labels.json` files, which don't clash with yours. If `session_notes.json` clashes, keep both sides' entries.

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| Camera window is black / wrong camera | another `--source` number; close other apps using the camera |
| fps red/yellow with good light | plug in, Best performance, NVIDIA "Prefer maximum performance"; nothing else heavy running |
| Constant beeping | the puncher is too close or off-center; the C270 sees little at < 1.5 m |
| `say where the camera is: --mount ...` | add `--mount head` |
| `say who is punching with --performer` | add `--performer volunteerN` |
| Closed the window by accident | the session is saved up to then; record a new one for the same name if it was short |
