#!/bin/sh
# Run ON THE PI, from the repo:   sh qnx/vision_pi/pi_live.sh [seconds]     (no seconds = until stopped)
# Starts BlindBox vision (camera -> pose -> early detector -> hat) detached from SSH, so a WiFi drop
# can't kill it (which would trip the hat's fault alarm).
#   live log:       tail -f /tmp/vision.log
#   per-frame CSV:  /tmp/vision_live.csv   (features, scores, cues)
#   stop:           sudo slay vision ; sudo qnx/build_aarch64/hat_usb X   (goodbye: hat alarm off)
# Before the first run after a reboot:  sh qnx/pi_camera_640.sh   (C270 at the 640x480 training size)
# Cue threshold (sensitivity) comes from /data/home/qnxuser/blindbox_threshold if it exists:
#   0.923 = catches more punches, ~4 false alarms/min (default)
#   0.949 = ~2 false alarms/min        0.967 = ~1 false alarm/min
#   change:  echo 0.949 > ~/blindbox_threshold   then run qnx/pi_start.sh again (or reboot)
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
chmod +x ../build_aarch64/vision 2>/dev/null
sudo slay -f vision 2>/dev/null
THRESH=$(cat /data/home/qnxuser/blindbox_threshold 2>/dev/null | tr -d ' \r\n')
[ -n "$THRESH" ] || THRESH=0.923
ARGS="--camera 5 --hat --ncnn pose_ncnn --fp16 --threads 3 --threshold $THRESH --csv /tmp/vision_live.csv --snapshot /tmp/vision_snap.ppm"
[ -n "$1" ] && ARGS="$ARGS --seconds $1"
sudo sh -c "nohup ../build_aarch64/vision $ARGS > /tmp/vision.log 2>&1 < /dev/null &"
sleep 2
cat /tmp/vision.log
