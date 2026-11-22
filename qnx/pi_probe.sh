#!/bin/sh
# Run ON THE PI (QNX), from the repo folder:   sh qnx/pi_probe.sh
# Lists everything we need to know to hook up the camera, the ESP32 buzzers and the network.
# Paste the whole output back to Claude / your teammate.

line() { echo; echo "================ $1"; }

line "QNX version"
uname -a

line "tools"
for t in python3 python git make qcc; do
  p=$(command -v $t 2>/dev/null)
  if [ -n "$p" ]; then echo "  $t: $p"; else echo "  $t: (not installed)"; fi
done
command -v python3 >/dev/null 2>&1 && python3 --version

line "USB devices (unplug/replug the ESP32 or camera and run again to see which line changes)"
if command -v usb >/dev/null 2>&1; then usb 2>&1; else echo "  ('usb' tool not found)"; fi

line "serial ports in /dev  (the ESP32 should appear as /dev/serusb1 or similar)"
ls /dev 2>/dev/null | grep -i -E "ser|tty|acm" || echo "  (none)"

line "camera-like devices in /dev"
ls /dev 2>/dev/null | grep -i -E "video|cam|uvc|sensor" || echo "  (none)"
ls /dev/camera* 2>/dev/null

line "GPIO"
ls /dev 2>/dev/null | grep -i gpio || echo "  (no gpio device in /dev)"

line "drivers / services running that matter (usb, serial, camera, gpio, wifi)"
pidin ar 2>/dev/null | grep -i -E "usb|serusb|devc-|camera|uvc|sensor|gpio|wpa|wifi|io-sock" | grep -v grep

line "network"
ifconfig 2>/dev/null | grep -E "flags|inet " || echo "  (ifconfig not available)"

line "free space here"
df -h . 2>/dev/null || df .

line "who am I"
id
echo
echo "done - paste everything above"
