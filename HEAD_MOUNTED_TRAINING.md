# Final training: camera on the wearer's head

This is the real training: data recorded the way the demo works, with the C270 on the wearer's hat, the puncher at step-in distance, and the wearer moving their head like they will in the demo. Everything recorded before this was "desk" data: a fixed camera on a table.

**Short version of the plan:**
1. Mount the camera on the hat.
2. Record 3–4 punch sessions (with `--free`, which adds free-boxing blocks) plus 1–2 "wearer moves" sessions, all with `--mount head`.
3. Label them (auto-label, then y/n review).
4. Train two candidate models and let the **head-mounted test score** pick the winner.
5. Install it, live-test it blindfolded, and tune the threshold.

---

## 1. Should we throw away the old data?

**No. Keep it, but stop trusting it for decisions.** From now on, models are judged only by how well they do on **held-out head-mounted footage.** Whether the old desk data stays in training is decided by that test, not by guessing.

### Why it might still help

The detector was built for a moving camera:
- Every position is measured **from the attacker's shoulders, in shoulder widths,** so if the whole image shifts or zooms, those numbers barely change.
- The optical-flow features **subtract the attacker's torso motion,** which removes most of the motion caused by the camera moving.

So "a fist leaving its guard spot" looks similar in desk and head footage. The 378 desk punches are a lot of examples of that.

### Why it might hurt

Head footage has things desk footage never had:

| Difference | Effect |
|---|---|
| **The wearer's head moves** (looking around, swaying, dodging) | A new source of false beeps the desk data never showed |
| **The puncher is farther away** (1.5–2 m) | The puncher looks smaller, so pose points are a bit noisier |
| **Different camera height and angle** | Body shapes look a little different |
| **Dodging right after a beep** | The camera jerks during the second half of punches |

### How to decide (built into the trainer)

Train two candidates and score **both on the same held-out head-mounted chunks:**

| Candidate | Trains on | Command flag |
|---|---|---|
| **All data** | desk + head sessions | `--test-mount head` |
| **Head only** | head sessions only | `--test-mount head --train-mount head` |

Pick the one that catches more punches within 200 ms at a similar false-beep rate. The alarm threshold is always picked from **head-mounted don't-punch footage,** so it accounts for head-motion false beeps.

What to expect:
- With only 1–2 head sessions, **"all data" will probably win,** because it has far more examples.
- As head data grows (about 300+ head punches), "head only" may catch up or win.
- **Re-run the comparison after every new session.**

Never delete the desk sessions. They stay in the repo.

---

## 2. Hardware setup

**The hat camera:**
- Mount the C270 on the front of the hat (brim or forehead), centered, lens level, pointing straight ahead.
- Tape or zip-tie it **solidly**: it must not wobble relative to the head. A camera wobbling on its own is motion the model can't explain.
- Aim check: with the puncher standing **1.5–2 m away** in guard, the preview must show their head to their waist, with both fists and some space on each side for hooks. If hooks still leave the frame, the puncher steps back a little.

**Cable and laptop (for recording and laptop testing):**
- The C270 plugs into the laptop by USB. Put the laptop on a table **behind or beside the wearer, with the screen facing the puncher,** because the puncher has to read the yellow prompts.
- If the cable is too short, use a USB extension. Over about 3 m, use an *active* (powered) extension.
- **Plug the laptop into the charger.** On battery it drops to about 7 fps and the data is useless.
- Find the camera index: run `python dodge_live.py --source 1`. If you see the wrong camera, try `0` or `2`.

**Room:**
- Normal indoor light. **Don't put a window or bright lamp behind the puncher** (backlight).
- If you can, record at least one session **in the demo room.**

**Safety:**
- It's a non-contact game. The puncher always stops short and stays beyond the wearer's reach.
- **During recording the wearer doesn't need a blindfold,** so keep eyes open. The camera can't see the wearer, so it makes no difference to the data.

---

## 3. Roles

| Person | Does |
|---|---|
| **Puncher** | Stands 1.5–2 m away, reads the yellow prompt on the laptop, and does exactly that move (section 4). During FREE BOXING, boxes naturally and picks their own moments instead. Pulls every punch short. |
| **Wearer** | Wears the hat. **Behaves like in the demo** (below). |

What the wearer should do **during every recording:**
- Stand in place facing the puncher. Every few seconds, move the head naturally: small look left or right, nod, lean, sway, shift weight.
- **Practice the dodge you'd do in the demo** at random times, not only when a punch comes: duck, lean left or right, step back.
- When a punch actually comes, react like in the demo, **after** the fist starts moving.

Why: in the demo the wearer's head is never still. The model has to see head motion during *both* punches and non-punches, so it learns that head motion alone doesn't mean a punch.

Swap roles for some sessions. **A second puncher is the most valuable data there is:** so far, every punch comes from one person.

---

## 4. Record

All recordings need `--mount head`. The recorder refuses to start without `--mount`, so this can't be forgotten.

**A. Punch sessions** (the main data). Record 8–10 minutes each, with `--free` so the session also has free-boxing blocks (data for punch prediction):
```powershell
python collect.py --source 1 --script --free --mount head --performer <puncher-name> --stance orthodox --wearer <wearer-name>
```
- `--performer` is required with `--free`. **Spell each person's name the same way every time** (lowercase first name): prediction is tested on people the model never trained on, so a misspelled name counts as a different person.
- `--stance orthodox` if the puncher's left hand leads (left foot forward), `--stance southpaw` if the right hand leads.
- Without `--free` the recorder works exactly as before (prompts only).

A `--free` session runs like this:

| Block | Screen | Puncher does |
|---|---|---|
| **BASELINE** (first 20 s) | blue border | Fighting stance, guard up, move like between punches (bounce if you bounce). **No punches.** |
| **Prompts** (60 s) | yellow `COMMAND` text | Exactly the prompted move (table below) |
| **FREE BOXING** (60 s) | green border and countdown | Box naturally: bounce, step in and out, set up, and **decide yourself** when to punch. Throw singles and combos (jab-cross, jab-jab-cross, hook-cross) and a few feints. **Eyes on the wearer, not the screen.** Every punch stops short; step back out after attacking. |
| then prompts and free boxing alternate until `q` | | |

- 5 s before a block changes, a countdown appears and the border blinks, so nobody needs to watch the screen during free boxing.
- Why free boxing: when every punch answers a prompt, nothing natural happens before it (the puncher just glances at the screen). Free boxing records real setups, combos and feints, which punch prediction needs.
- About half of a `--free` session is free boxing, so it has fewer prompted punches but adds natural ones. Block lengths can be changed with `--baseline-seconds`, `--prompt-seconds` and `--free-seconds`.

What each yellow prompt means:

| Prompt | Puncher does |
|---|---|
| `JAB` / `CROSS` / `UPPERCUT` | That punch, at demo speed, pulled short |
| `HOOK_L = hook with your RIGHT hand` | Hook with your **right** hand (comes from the image's left) |
| `HOOK_R = hook with your LEFT hand` | Hook with your **left** hand |
| `FEINT` | Fake a punch start |
| `GUARD_SHIFT` | Slide your fists left and right without punching. Big and fast. |
| `STEP_IN`, `HEAD_TURN`, `NOD`, `GLOVE_ADJUST`, `IDLE` | That move, **no punch** |

- Mix clean punches (guard held first) with messy ones (straight out of another move), and include some step-in punches.
- Press `q` to save. You get `data/session_<time>.npz` and `.mp4`.
- **Back up the `.mp4` right away** (USB drive or Drive). Labels for punch prediction may need redoing later, and only the video makes that possible. Videos are still never committed.

**B. Wearer-motion sessions** (teach "camera moving is not a punch"). Record 3–5 minutes:
```powershell
python collect.py --source 1 --script --negatives-only --mount head --performer <puncher-name> --wearer <wearer-name>
```
- Only don't-punch prompts appear. The puncher does them.
- **The wearer moves a lot:** looks around, ducks, leans, steps back, turns, as if dodging.

**How much to record:**

| Goal | Amount |
|---|---|
| Minimum to make a decision | 2 punch sessions + 1 wearer-motion session (about 25 min) → roughly 200 head punches |
| Good | 4 punch sessions (2 punchers) + 2 wearer-motion sessions → 400+ head punches |
| For punch prediction | `--free` sessions from **at least 3 different punchers**, 2 sessions each. Prediction is tested on people the model never saw, so the number of different people matters most. |

Each 10-minute punch session gives about 100–125 usable punches.

---

## 5. Label each new session

Run these for every new `data/session_<time>.npz`:
```powershell
python auto_label.py data\session_<time>.npz
python combo_candidates.py data\session_<time>.npz
python review_labels.py data\session_<time>.npz
python review_labels.py data\session_<time>.npz --messy
```
`combo_candidates.py` is for `--free` sessions (for any other session it says so and changes nothing). It adds the follow-up punches of combos, which `auto_label.py` mostly misses, to the `--messy` pass: expect roughly 12–24 extra candidates per minute of free boxing to accept or reject. It also builds the optical-flow cache, which makes the "cache the features" step below quicker.

**The start rule: the start frame is the first frame the fist moves TOWARD the target** (the wearer). A pull-back, dip or shoulder drop before that is the *telegraph*, not the start: nudge past it with `.`. Punch prediction learns from exactly that difference. Older reviews used "the first frame the fist leaves guard"; new decisions are saved with `"rule": "toward_target"`, so the two can be told apart.

Review keys:
- `y` = real punch, and the start frame is right
- `f` = feint, a fake punch start (saved as type FEINT, never trained as a punch)
- `n` = not a punch, or the wrong moment
- `,` / `.` = nudge one frame earlier / later
- `q` = save and quit (it resumes where you left off)

An orange **"punch already at ..."** line means another accepted punch starts within 0.2 s, in this pass or the other one. That is usually the same punch seen twice (once per fist, or once per pass): answer `n` unless it really is a second punch. (The older sessions have about 15 such duplicates.)

Free-boxing punches (`--free` sessions) show up in the review with type `FREE`, most of them in the `--messy` pass because the fist rarely rests before them. Leave the type as `FREE` or set it with `1`–`5`.

**Do two reviewers agree?** Check this once for each new person who labels, on one session or a 2-minute clip of it. The second person reviews blind: they start from slightly shifted frames, and their labels are saved separately and never used for training.
```powershell
python review_labels.py data\session_<time>.npz --reviewer <name>
python review_labels.py data\session_<time>.npz --reviewer <name> --messy
python label_agreement.py data\session_<time>.npz --reviewer <name>
```
They can stop early: only candidates both reviewers decided are compared, and `--from 60 --to 180` limits it to one clip (seconds into the session). Good agreement is kappa ≥ 0.8 and ≥ 90% of start frames within 1 frame. If it's lower, go through the listed disagreements together and agree on how to apply the start rule.

A wearer-motion session has no punches. The review then prints "No punch candidates to review; saved empty labels". That's expected, and the session is still used for training.

**Pi version of the keypoints:** run this **after** both reviews are done:
```powershell
python repose_onnx.py data\session_<time>.npz
```
It copies the labels at that moment. If you change labels later, copy `data\session_<time>.labels.json` to `data_onnx\session_<time>_onnx.labels.json`.

**Cache the features** (required before committing; takes about 3–5 minutes). These need the video, so they must be made on the computer that has the `.mp4`. The training machine then doesn't need the video at all:
```powershell
python -c "import early_dataset as d; d.build('data/session_<time>.npz'); d.build('data_onnx/session_<time>_onnx.npz')"
```

---

## 6. Train and decide

Training is heavy, about 20–40 minutes per run. **Run it on the fast machine.** The environment setup is in [TEAMMATE_AUGMENTATION_GUIDE.md](TEAMMATE_AUGMENTATION_GUIDE.md), sections 1–5. Training needs no videos, only the committed `.npz`, `.labels.json`, `.auto_candidates.json` and `features/` files, so commit those after labeling (section 8).

```powershell
mkdir models\candidates
python early_detector.py --test-mount head --budgets 2 --save models\candidates\all_data.joblib > models\candidates\all_data.log
python early_detector.py --test-mount head --train-mount head --budgets 2 --save models\candidates\head_only.joblib > models\candidates\head_only.log
```

Compare the `learned` line in the two logs, under "threshold picked ... <= 2 false alarms/min". It reads: caught ≤50 / ≤100 / ≤150 / **≤200** / ≤300 ms | median delay | **fa/min** | feints.

**The winner** catches more at ≤200 ms with fa/min about the same. With about 200 punches, differences under about 7 points are noise. If it's that close, prefer **all data** (more examples means steadier behavior).

**Pi version of the winner:** the same command with `--data data_onnx`. For example, if "all data" won:
```powershell
python early_detector.py --data data_onnx --test-mount head --budgets 2 --save models\candidates\all_data_onnx.joblib > models\candidates\all_data_onnx.log
```

**Install the winner:**
```powershell
copy models\early_detector.joblib models\archive\early_detector_before_head.joblib
copy models\early_detector_onnx.joblib models\archive\early_detector_onnx_before_head.joblib
copy models\candidates\all_data.joblib models\early_detector.joblib
copy models\candidates\all_data_onnx.joblib models\early_detector_onnx.joblib
python -m pytest -q test_early.py test_anticipation.py
python export_model_c.py models\early_detector_onnx.joblib
```
Use `head_only` instead of `all_data` if that one won. `export_model_c.py` must be given the **Pi** model (`early_detector_onnx.joblib`), because the C code runs on the Pi.

**Optional, after there's head data: lighting / frame-rate robustness.** Run the one-command augmentation on the fast machine, scored on head footage:
```powershell
python run_augmented_training.py --test-mount head
```
It needs every session's `.mp4` on that machine and `repose_onnx.py` done for each. Follow [TEAMMATE_AUGMENTATION_GUIDE.md](TEAMMATE_AUGMENTATION_GUIDE.md).

---

## 7. Live test (blindfolded)

```powershell
python dodge_live.py --source 1
```

1. The wearer puts on the hat and the blindfold. The puncher stands at 1.5–2 m. **Non-contact:** the puncher stays out of reach and stops every punch short.
2. Throw 20 punches of mixed types, plus 1 minute of don't-punch moves (guard shifts, step-ins). The wearer moves their head naturally throughout.
3. A third person counts each punch as:
   - dodged in time
   - beep came too late
   - no beep
   - wrong direction (for example DODGE LEFT for a hook from the left)

   Also count **false beeps** during the don't-punch minute.
4. Tune the threshold. The saved one is tuned for about 2 false beeps per minute on head footage.
   - Too many false beeps: `python dodge_live.py --source 1 --threshold 0.96`
   - Too many misses: `--threshold 0.94`
   - Steps of about 0.005 matter.
5. Every beep is logged in `dodge_log.csv`.

---

## 8. Commit (no videos)

After labeling a session, commit its small files so the fast machine can train:
```powershell
git add data\session_<time>.npz data\session_<time>.labels.json data\session_<time>.auto_candidates.json data\session_<time>.review_*.json
git add data_onnx\session_<time>_onnx.npz data_onnx\session_<time>_onnx.labels.json data_onnx\session_<time>_onnx.auto_candidates.json
git add features\session_<time>*
git commit -m "Head-mounted session <time>"
git push
```

- **Videos (`.mp4`) are never committed:** they're too big and show people. Git ignores them anyway. Share them by Drive or USB only if someone needs to re-review or run the augmentation.
- After installing a new model, also commit `models\early_detector.joblib`, `models\early_detector_onnx.joblib`, `models\archive\*` and `qnx\model\early_model.*`.

---

## 9. Checklist

- [ ] Camera solidly on the hat, aimed so the puncher's head to waist is visible at 1.5–2 m, laptop plugged in
- [ ] Punch session 1 (`--free --mount head --performer <name> --stance <stance>`, 8–10 min) → back up the `.mp4` → auto-label → `combo_candidates.py` → review → review `--messy` → `repose_onnx.py` → cache features → commit
- [ ] Punch session 2 (ideally the other person punching) → same
- [ ] More `--free` sessions from new punchers (at least 3 people in total, 2 sessions each) for punch prediction
- [ ] Wearer-motion session (`--negatives-only --mount head`, 3–5 min) → auto-label → review → `repose_onnx.py` → cache features → commit
- [ ] Train "all data" vs "head only" with `--test-mount head` on the fast machine, then pick the winner
- [ ] Train the Pi version of the winner → install → tests → `export_model_c.py` → commit
- [ ] Blindfolded live test, tune the threshold, then make that threshold the default
- [ ] More sessions (second puncher, the demo room) → repeat the comparison

---

## 10. Fixing a mistake

**Recorded a head session without `--mount head`?** That can't happen with the current recorder, which requires `--mount`. For an old file, set the tag with:
```powershell
python -c "import numpy as np; p='data/session_<time>.npz'; z=dict(np.load(p)); z['mount']=np.array('head'); np.savez_compressed(p, **z)"
```
Then do the same for `data_onnx/session_<time>_onnx.npz`.

**Using Claude Code for any of this?** Point it at this file and tell it:
- follow the file exactly
- never delete sessions
- never commit videos
- write candidate models to `models/candidates/`, and only install after comparing the two `--test-mount head` runs
