#!/bin/sh
# Run ON THE PI (QNX) if the USB stack died (no /dev/usb/io-usb-otg; camera and hat gone):
#   sudo sh qnx/pi_usb_restart.sh && sh qnx/pi_camera_640.sh
# Restarts io-usb-otg detached, with the same arguments as /system/etc/usb_start.sh.
# Output goes to a file: io-usb-otg runs with -v, and writing to a closed SSH session kills it.
slay -f io-usb-otg 2>/dev/null
sleep 1
nohup /system/sbin/io-usb-otg -v -d dwc3-xhci ioport=0x1f00200000,irq=0xbf,iosize=0x100000,bmstr=0x1000000000,ioport=0x1f00300000,irq=0xc4,iosize=0x100000,bmstr=0x1000000000 > /tmp/io-usb.log 2>&1 < /dev/null &
if waitfor /dev/usb/io-usb-otg 10; then
  sleep 3
  echo "usb stack up: $(usb 2>&1 | grep -c Vendor) devices (hub + ESP32 x2 + C270 = 4)"
else
  echo "io-usb-otg did not come back: reboot the Pi"; exit 1
fi
