#!/bin/sh
# Installs / removes BlindBox auto-start on the Pi (QNX 8 Quick Start image).
#   sudo sh qnx/pi_autostart_install.sh          install
#   sudo sh qnx/pi_autostart_install.sh remove   uninstall
# The startup script /system/etc/startup/post_startup.sh sources /data/var/etc/settings/network (persistent)
# early in boot. We append ONE guarded line that starts pi_boot.sh in the background; a backup of the
# original file is kept next to it (network.before_blindbox). The line is syntax-checked before saving.
F=/data/var/etc/settings/network
BOOT=/data/home/qnxuser/boxing-assistant/qnx/pi_boot.sh
TAG="# blindbox-autostart"
LINE="[ -f $BOOT ] && ( sh $BOOT > /tmp/blindbox_boot.log 2>&1 < /dev/null & ) $TAG"
[ -f "$F" ] || { echo "no $F"; exit 1; }
[ -f "$F.before_blindbox" ] || cp -p "$F" "$F.before_blindbox"
grep -v "$TAG" "$F" > /tmp/network.new
if [ "$1" != "remove" ]; then echo "$LINE" >> /tmp/network.new; fi
sh -n /tmp/network.new || { echo "syntax check failed - nothing changed"; exit 1; }
cp /tmp/network.new "$F" && chmod 644 "$F"
echo "now $F is:"; cat "$F"
