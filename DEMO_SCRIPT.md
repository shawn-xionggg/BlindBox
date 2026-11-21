# 2-minute judge demo: Blindfolded Dodge on QNX

_Draft, written overnight. Swap in the real Pi numbers once they're measured. Lines in quotes are what to say._

## Setup (before judges arrive)
- The dodger is blindfolded, wearing the hat (C270 + buzzer). The Pi 5 on QNX is in a belt bag.
- The puncher stands **~1.5 m away, out of reach**, wearing soft gloves.
- A laptop shows the live dashboard: per-stage latency, cue log, process list.

## 0:00 – 0:20 · The idea
> "Our dodger can't see. A camera on their hat watches the attacker, and the moment a fist leaves guard, the buzzer tells them to move. Everything runs on QNX on a Raspberry Pi 5."

## 0:20 – 0:50 · Live dodges (3–4 punches)
- The puncher throws a few punches from step-in distance. The dodger steps back on the beep.
> "The detector buzzes on average about 100 ms after the fist starts moving. That leaves time to react when the punch has to travel from 1.5 m."

## 0:50 – 1:20 · Why QNX (show the dashboard)
> "Vision, cue driving and supervision are separate processes. They talk through QNX message passing: MsgSend for cues, pulses for heartbeats. The buzzer process runs at the highest real-time priority, so a busy vision process can never delay a cue."
- Point at the latency numbers: capture → cue delivered (~6 ms through IPC in our rehearsal; use the Pi number once measured).

## 1:20 – 1:45 · Fault tolerance (the wow moment)
- Kill the vision process from the laptop (`slay` on QNX).
> "Watch: within 0.3 seconds the buzzer plays a 'system down' pattern, so the dodger knows not to trust it. The supervisor restarts vision in under a second, and the cues come back."

## 1:45 – 2:00 · Honest numbers + what's next
> "Over 121 reviewed punches, we catch about 60% within 200 ms at around 2 false alarms a minute. Predicting *before* the fist moves isn't possible from this camera yet: we measured it. Next is more training data and a second camera angle."

## Backup lines for questions
- **"Why one beep instead of left/right?"**
  > "We measured it. In the first 100 ms, even which hand is punching is a coin flip, because the whole body rotates. One reliable cue beats a fast wrong one."
- **"How do you know it isn't overfitting?"**
  > "Every number is cross-validated: the threshold is picked on separate footage from the footage it's scored on. And the live features are verified bit-identical to the training ones."
- **"What runs where?"**
  > "On the Pi: pose plus our features, then 80 decision trees exported to plain C, identical to the Python version."
