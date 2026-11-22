#!/bin/sh
# BlindBox: stop detection cleanly (ON THE PI):   sh ~/boxing-assistant/qnx/pi_stop.sh
# Stops vision, then sends the hat "goodbye" so it doesn't sound the missing-heartbeat fault alarm.
cd "$(dirname "$0")" || exit 1
sudo slay -f vision 2>/dev/null && echo "vision stopped" || echo "vision was not running"
sudo build_aarch64/hat_usb X >/dev/null 2>&1 && echo "hat: goodbye sent (alarm off)"
tail -8 /tmp/vision.log 2>/dev/null
