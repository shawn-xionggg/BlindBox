#!/bin/sh
# BlindBox auto-start at power-on. Called in the background by the Pi's startup through one line added
# to /data/var/etc/settings/network (see pi_autostart_install.sh). Log: /tmp/blindbox_boot.log
# Waits for the camera service and for the camera + hat on USB, then runs pi_start.sh (ready beep, detection).
# To skip auto-start for one boot: touch /data/home/qnxuser/NO_BLINDBOX (delete it to re-enable).
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f /data/home/qnxuser/NO_BLINDBOX ] && { echo "auto-start disabled (NO_BLINDBOX present)"; exit 0; }
echo "blindbox boot: waiting for the camera service and USB devices..."
i=0
while [ $i -lt 120 ]; do
    if [ -e /dev/sensor/camera1 ] && [ -e /dev/usb/io-usb-otg ]; then
        U=$(usb 2>&1)
        if echo "$U" | grep -q 0x046d && echo "$U" | grep -q 0x55d3; then break; fi
    fi
    sleep 1; i=$((i + 1))
done
echo "blindbox boot: ready after ${i}s"
sleep 8    # let the rest of the startup (sensor waits, screen, vnc) finish first
try=1
while [ $try -le 5 ]; do
    echo "blindbox boot: start attempt $try"
    if sh "$HERE/pi_start.sh"; then
        sleep 5
        # (QNX names the process by its path, so check the log, not pidin -P vision)
        if grep -q "first frame" /tmp/vision.log 2>/dev/null; then echo "blindbox boot: detection running"; exit 0; fi
    fi
    try=$((try + 1)); sleep 10
done
echo "blindbox boot: gave up after 5 attempts - check camera/hat cables, then run pi_start.sh by hand"
exit 1
