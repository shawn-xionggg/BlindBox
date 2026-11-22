#!/bin/sh
# BlindBox: run ON THE PI after plugging it in (camera + hat connected):
#     sh ~/boxing-assistant/qnx/pi_start.sh          (runs until pi_stop.sh)
#     sh ~/boxing-assistant/qnx/pi_start.sh 120      (runs 120 s)
# 1. checks the C270 and the hat are on USB   2. camera -> 640x480 (training size)
# 3. one front beep = ready                    4. starts detection detached (log: /tmp/vision.log)
cd "$(dirname "$0")" || exit 1
chmod +x build_aarch64/* 2>/dev/null

USBLIST=$(usb 2>&1)
ok=1
echo "$USBLIST" | grep -q "0x046d" || { echo "!! camera (Logitech C270) not found on USB - plug it in"; ok=0; }
echo "$USBLIST" | grep -q "0x55d3" || { echo "!! buzzer hat (ESP32 UART port, CH343) not found on USB - plug it in"; ok=0; }
if [ ! -e /dev/usb/io-usb-otg ]; then echo "!! USB stack is down: sudo sh qnx/pi_usb_restart.sh"; ok=0; fi
[ $ok = 1 ] || exit 1

sh pi_camera_640.sh | head -1 || exit 1
sudo build_aarch64/hat_usb I >/dev/null && echo "hat: ready beep sent (front buzzer)"
sleep 1
sh vision_pi/pi_live.sh $1
echo
echo "BlindBox is running.  live log: tail -f /tmp/vision.log    stop: sh ~/boxing-assistant/qnx/pi_stop.sh"
