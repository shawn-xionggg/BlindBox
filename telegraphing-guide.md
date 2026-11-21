# Reading movement onset with a head-mounted camera

## What the prototype can claim

In this non-contact QNX game, the hat camera sees a sequence of flat images. The pose model estimates where joints appear in each image. A separate rule compares those estimates over time and decides when to signal the buzzer.

A **telegraph** is a visible preparation that might precede a gesture. An **onset** is the beginning of the gesture itself. If the glove has already left guard when the buzzer sounds, report early detection. A prediction claim requires evidence that the cue arrives before glove departure on unseen trials and stays quiet during similar preparations that end without a gesture.

## Research findings

- Stanley and colleagues used 3D motion capture and force plates to study six punches from 15 amateur boxers striking a bag. They found differences among straights, hooks, and uppercuts over the course of the movement. This supports grouping the gestures by family. It leaves open how early a frontal camera can distinguish them. [Stanley et al., 2018](https://doi.org/10.1080/24748668.2018.1525651)
- Dinu and Louis found that different body segments contributed differently across punch families and between junior and elite boxers. A shoulder cue that works on one performer needs testing on another. [Dinu and Louis, 2020](https://doi.org/10.3389/fspor.2020.598861)
- In a VR boxing study, a foot feint always preceded an attacking sequence. That made a step informative within the experiment by design. Separately, fake cues affected responses in a boxing-specific attention task even among trained participants. Include steps and feints that end without a gesture in the test footage. [VR study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9557204/) · [Fake-punch timing study](https://pubmed.ncbi.nlm.nih.gov/37075655/)

## What the C270 might see

Lead and rear describe the performer's stance. Either hand can be the lead. The observations below are hypotheses for recorded, staged gestures. Their reliability and timing on this rig remain unverified.

| Basic gesture | Evidence to inspect in the image | Confusing example |
|---|---|---|
| Jab (lead straight) | One glove leaves guard. When it moves along the viewing axis, its outline can grow while its center barely travels. | Stepping closer enlarges the whole upper body. |
| Cross (rear straight) | Look for the same relative glove growth on the rear side. Log a shoulder-line change separately to see whether it precedes departure. | Turning in place changes the shoulder line even with both hands held still against the chest. |
| Lead or rear hook | The glove and elbow begin a lateral sweep relative to the chest. Several observations may be needed before the arc is visible. | A head turn shifts the whole scene sideways. |
| Lead or rear uppercut | The glove rises relative to its shoulder or chest. Check that it was visible before the rise. | A glove returning to guard also rises. A camera nod moves much of the scene vertically. |

A glove's starting side does not establish its eventual path. In particular, a central wrist can occur in both a forward and a rising gesture. For the demo, assign directions as game commands and score each against its scripted label. Keep the performer beyond reach of the wearer.

## How to make the signals measurable

1. **Establish a baseline.** Replay the same recordings through the existing YOLO11n-Pose wrist-speed rule. Watch the landmark overlay on missed gestures. A well-tracked point can remain nearly stationary in the image during motion toward the lens.
2. **Use the torso as a reference.** Subtract the shoulder midpoint from the wrist position and divide by shoulder width. This gives a rough measure of where the wrist sits relative to the person's body even as their image shifts or changes scale. Head roll and torso rotation can still alter the estimate.
3. **Measure glove growth separately.** Segment a colored glove and calculate its image area divided by squared shoulder width. For example, a 400-pixel glove region and shoulders 100 pixels apart give 400 / 100² = 0.04. If all image lengths double, glove area quadruples and the ratio stays the same. If only the glove grows, the ratio rises. Rotation, overlapping gloves, and partial visibility can also affect it.
4. **Keep a silent candidate state.** Log a possible preparation, such as a shoulder shift. Confirm only when the chosen glove signal meets your calibrated rule. Output “unknown” and discard stale motion history when a required landmark disappears or the tracker jumps.
5. **Time detection and classification independently.** The first sign of movement may arrive before its direction becomes clear. Two successive observations add one processed-frame interval after the first qualifying observation: about 33 ms at 30 fps or 67 ms at 15 fps. Use video timestamps for replay and capture timestamps on the Pi.
6. **Measure what the wearer receives.** Log capture-to-software-cue latency for QNX. Use a separate synchronized recording for visible movement onset to audible buzzer. If the one-buzzer patterns share a first beep, also measure when their sounds become distinguishable.

## Evaluation in one session

Record positive gestures, ordinary movement, and a fresh session with a second performer. Include head turns, nods, steps alone, glove adjustments, and movements that stop early. For straight reaches toward the lens, put the camera on an empty stand. A synchronized side-view phone recording makes the start of forward movement easier to label.

Have two viewers mark the first preparatory movement (if visible) and first glove departure independently. Preserve disagreements as a frame range. At 30 fps, one frame spans about 33 ms. Set thresholds on the first session and score the second without retuning.

| Comparison on identical clips | Decision it supports |
|---|---|
| Body-relative wrist speed alone | Establish which families the existing rule catches. |
| Add relative glove area | Check whether head-on gestures become detectable without extra false alarms from steps. |
| Add a shoulder-based candidate | Measure whether it buys lead time, and count preparations that lead nowhere. |

Report on-time correct cues out of attempts for each family, incorrect directions, unknowns, false buzzes per minute of negative footage, and median and worst observed onset-to-buzzer delay. The brief's 8/10 within two frames is a project target. Ten trials are too few for a meaningful gesture-latency p99. Keep negative delays (cue before labeled onset) separate when evaluating prediction.

**First experiment:** record a straight reach toward an empty camera, a lateral sweep, and a step forward with the hands held in place relative to the chest. Plot wrist speed and relative glove area for all three. The straight reach exposes the depth limitation, the sweep checks lateral tracking, and the step tests whether whole-body motion creates a false alarm.
