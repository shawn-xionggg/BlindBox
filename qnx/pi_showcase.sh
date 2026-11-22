#!/bin/sh
# Run ON THE PI (QNX), from the repo folder:   sh qnx/pi_showcase.sh
# The QNX fault-tolerance demo with the prebuilt Pi programs (qnx/build_aarch64/):
#   supervisor starts cue_driver + a scripted vision stub, the stub "crashes" at 0.65 s,
#   cue_driver notices the missing heartbeats (system-down cue), supervisor restarts vision,
#   cues resume. Prints every cue with its latency.
cd "$(dirname "$0")" || exit 1
chmod +x build_aarch64/* 2>/dev/null
./build_aarch64/supervisor --csv ./test_events.csv --duration 3.8 --crash-at 0.65
