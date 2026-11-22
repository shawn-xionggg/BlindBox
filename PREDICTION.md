# Punch prediction (experimental)

The demo's detector is **reactive**: it beeps once the fist is moving. This is a separate attempt to **predict**: warn *before* the fist starts moving toward the target, from how the puncher sets up. The reactive detector, its models and its training are not changed by any of this.

The earlier attempt found no signal (REPORT.md). Every punch then was a reaction to a screen prompt, and the start label counted any wind-up as part of the punch, so there was nothing natural to see before a start. This version changes the data, the labels and the test.

| File | What it does |
|---|---|
| `collect.py --free` (`free_blocks.py`) | Records free-boxing blocks: the puncher sets up and chooses when to punch |
| `review_labels.py` | Start rule: the first frame the fist moves **toward the target**; a pull-back or dip before it is the telegraph |
| `combo_candidates.py` | Adds the follow-up punches of combos to the review |
| `label_agreement.py` | Checks that two reviewers apply the start rule the same way |
| `predict_dataset.py` | Per-frame targets and features |
| `predict_trainer.py` | Trains, tests (a), (b), (c), saves to `models/candidates/` |
| `test_predict.py` | Includes the positive and negative controls below |
| `signed_motion.py` | Signed motion of each fist (Kalman-smoothed) and a hand-written pull-back rule (section 7) |
| `early_detector_signed.py` | Runs the unchanged `early_detector.py` with the signed features added, for a like-for-like reactive comparison |

## 1. Run it

After recording and labeling new sessions (HEAD_MOUNTED_TRAINING.md sections 4–5, including the flow cache):
```powershell
python predict_trainer.py > models\candidates\predictor.log
```
About 1 minute on the current data. `--quick` runs budget 2 only and skips the ablations. The model goes to `models/candidates/predictor.joblib`; the trainer refuses to save anywhere else.

## 2. What it learns

- **Target:** for each frame, "a reviewed punch starts within the next 0.3 s".
- **Left out of training:**
  - each punch from its start to its peak (the fist is already flying, which is the reactive detector's job)
  - 0.3–0.6 s before a start (a warning that early isn't wrong)
  - feints (key `f` in review) and FEINT prompts
  - frames where nobody is tracked
  - **prompted punches**: the puncher reacts to the screen, a cue that doesn't exist in the demo. They are reported apart (`--train-prompted` includes them).
- **Duplicates:** the same punch labeled twice (starts < 0.12 s apart, or < 0.25 s inside one single-punch prompt) is counted once.
- **Features** (209). All use the current and past frames only, so they could run live:
  - **body:** pose relative to the shoulders (nose, elbows, wrists, hips, shoulder angle and width, elbow angles, each fist's distance from its guard spot, optical flow around each fist), sampled now and 0.1, 0.2, 0.35, 0.6 and 1 s ago, plus how much each varied over the last second.
  - **rhythm:** the size, phase and period of the shoulders' bounce, and whether it just stopped.
  - **context:** punch-like movements found without labels: time since the last one, how many in the last 2 s, whether a fist is coming back. A movement only counts once a live system could have confirmed it.
- **Each person's calm baseline:** every body feature is measured against the person's own calm (median and spread over frames where both fists rest in guard, collected at the start of the session; the BASELINE block of `--free` sessions). No labels are needed, so a live system could calibrate the same way.
- **Model:** gradient-boosted trees, the same fast kind as the reactive model. A GPU sequence model is only worth trying if this one beats random on people it never saw.

## 3. How it's tested

- **Thresholds come from training data only:** the warning threshold for each budget (1, 2, 4 false warnings per minute) is set on scores from models that never saw those training frames (3 folds inside the training data).
- **Replayed like live:** a warning fires when the score crosses the threshold, then a 0.3 s cooldown.
- **What counts:** a warning counts for a punch if it comes 0–0.5 s before the start. A warning while another punch is already flying counts neither way: that's reacting, not predicting. Every other warning is a false warning.
- **(a) Leave one performer out:** test on a person the model never trained on. This is the test that matters.
- **(b) Personalization:** train on everyone else plus the first half (by time) of one person's natural punches, test on their second half.
- **(b′) Pipeline check:** with only one performer, their first half → second half. It is labeled as such and says nothing about new people.
- **(c) Random warnings at the model's own rate,** 500 runs: (c1) anywhere, (c2) at the same rate within each kind of footage. (c2) stops a model from getting credit just for noticing that the person is boxing.

## 4. Reading the report

```
  first punches     5/52   10% [4-21%]          207 ms        4/52             5/52
  random, same rate per footage (500 runs): first punches 2% [0-7%]
```
- `5/52 10% [4-21%]`: warned 0–0.5 s before 5 of 52 first punches. The bracket is the 95% range: with 52 punches, the true rate could be anywhere from 4% to 21%.
- `207 ms`: median warning time before the start. `4/52`: warned at least 100 ms early. `5/52`: warned before the computer saw *any* movement (so not just a visible wind-up).
- `random ... 2% [0-7%]`: what random warnings at the same rate achieve.
- **A real signal** is a warned % on test **(a)** clearly above the random **(c2)** range, for first punches or follow-ups, at a false-warning rate you'd accept (≤ 2 per minute). Report first punches and follow-ups separately: follow-ups in a combo are expected to be easier.

## 5. Results so far (2026-11-21)

**Controls** (synthetic boxers, `test_predict.py`; test (a), budget 2):

| Synthetic data | Warned before the start | Random at the same rate |
|---|---|---|
| Head dips ~270 ms before every punch | 103/105 (98%), median lead 167 ms | ≤ 41% |
| No telegraph | 1/105 (1%) | ≤ 3% |

The pipeline finds a real telegraph on a person it never saw, and doesn't invent one.

**Real data:** all sessions are from one performer, so tests (a) and (b) can't run. Only one session (unscripted) has natural punches; the other 249 punches were prompted.
- (b′) pipeline check, budget 2: **5/52 first punches warned (10%, range 4–21%)** vs random 2% [0–7%]. Median lead 207 ms, ~0.6 false warnings per minute of don't-punch footage. Follow-ups: 0/3.
- Only the body and rhythm features carried it; context alone warned 0/52.
- Caveats: the same person, session and camera on both sides; a fixed desk camera; labels from before the start rule. **This is a hint at best. The real answer needs free-boxing sessions from at least 2–3 new people.**

## 6. Signed motion and the pull-back rule

The original speed feature can't tell a fist going out from one coming back. `signed_motion.py` adds, per fist and per frame (causal):
- signed speed and acceleration **away from (+) / back toward (−) its guard spot**
- how far it is **below** its guard spot, with vertical speed and acceleration (a dip before an uppercut)
- **looming**: how far it is in front of (+) or behind (−) where it was at guard, from the optical-flow divergence around it, with speed and acceleration
- **time to turnaround** (−speed ÷ acceleration while moving back and slowing down) and **where** it will turn around

All are smoothed with a constant-acceleration Kalman filter, which estimates position, speed and acceleration together and copes with dropped frames.

**The rule:** warn when a fist is already behind (or below) its guard spot, still moving back but slowing down, and will turn around within 0.15 s further behind. A retraction that stops at the guard spot doesn't fire (tested).

```powershell
python predict_trainer.py --signed          # predictor with vs without the signed features
python predict_trainer.py --rule either     # the rule alone (depth / dip / either)
python early_detector_signed.py --budgets 2              # reactive baseline
python early_detector_signed.py --signed --budgets 2     # reactive + signed features
python early_detector_signed.py --rule either --budgets 2   # rule as the reactive "hand-made" row
```

## 7. Known limits

- Older labels used "first frame the fist leaves guard" as the start, so some wind-ups are inside their punches. New decisions are tagged `"rule": "toward_target"`.
- The cached flow divergence (`div_rel`) runs *negative* during punches. As measured around the wrist joint, it isn't a clean "fist grows" signal, which matters for the looming features of the next step.
- On a head-mounted camera, the wearer's own head motion leaks into the bounce features.
- Nothing runs live yet. The features are causal, so a live version is possible once the tests show a signal worth it.
