#!/bin/sh
# Run ON THE PI (QNX) after every boot:   sh qnx/pi_camera_640.sh
# Restarts QNX's camera service (sensor) so the USB C270 (camera unit 5) runs at 640x480,
# the resolution all training data was recorded at. Default config gives ~20 fps; this gives ~26.
# The system config is not edited: a copy with one extra line is written to ~/sensor_c270.conf.
# Undo: reboot.
SRC=/etc/config/sensor/sensor_rpi5.conf
DST="$HOME/sensor_c270.conf"

awk '{print} /^    name = front-camera$/{print "    default_video_resolution = 640, 480"}' "$SRC" > "$DST" || exit 1
grep -q "default_video_resolution = 640, 480" "$DST" || { echo "could not find the USB camera block in $SRC"; exit 1; }

sudo slay -f sensor 2>/dev/null
sleep 1
sudo sh -c "nohup sensor -U 521:521 -b external -r /data/share/sensor -c $DST > /tmp/sensor.log 2>&1 < /dev/null &"
sleep 2
if waitfor /dev/sensor/camera5 5; then
  echo "camera service restarted: C270 = camera unit 5 at 640x480"
  echo "test it:  camera_example1_callback -u 5   (any key stops it)"
else
  echo "camera5 did not appear. Is the C270 plugged in? Log: /tmp/sensor.log"; exit 1
fi
