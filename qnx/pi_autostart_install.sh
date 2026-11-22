#!/bin/sh
# Installs / removes BlindBox auto-start on the Pi (QNX 8 Quick Start image).
#   sudo sh qnx/pi_autostart_install.sh          install
#   sudo sh qnx/pi_autostart_install.sh remove   uninstall
# Adds ONE guarded line that starts pi_boot.sh in the background, just before the final `exit 0` of
# /usr/etc/startup/post_startup.sh (= /system/etc/startup/post_startup.sh, on the persistent /usr
# partition; runs last in boot, after the camera service is up). A backup of the original is kept
# (post_startup.sh.before_blindbox) and the edited script is syntax-checked before it replaces it.
# (/data/var/etc/settings/network is rebuilt at every boot, so a line there only survives one reboot.)
F=/usr/etc/startup/post_startup.sh
BOOT=/data/home/qnxuser/boxing-assistant/qnx/pi_boot.sh
TAG="# blindbox-autostart"
LINE="[ -f $BOOT ] && ( sh $BOOT > /tmp/blindbox_boot.log 2>&1 < /dev/null & ) $TAG"
[ -f "$F" ] || { echo "no $F"; exit 1; }
[ -f "$F.before_blindbox" ] || cp -p "$F" "$F.before_blindbox"

# also clean up the old one-boot-only line in the network settings file, if any
N=/data/var/etc/settings/network
if grep -q "$TAG" "$N" 2>/dev/null; then grep -v "$TAG" "$N" > /tmp/network.new && cp /tmp/network.new "$N"; fi

grep -v "$TAG" "$F" > /tmp/post_startup.new
if [ "$1" != "remove" ]; then
    awk -v line="$LINE" '{ a[NR] = $0 } /^exit 0/ { last = NR }
        END { for (i = 1; i <= NR; i++) { if (i == last) print line; print a[i] } if (!last) print line }' \
        /tmp/post_startup.new > /tmp/post_startup.new2 && mv /tmp/post_startup.new2 /tmp/post_startup.new
fi
sh -n /tmp/post_startup.new || { echo "syntax check failed - nothing changed"; exit 1; }
chmod u+w "$F"
cp /tmp/post_startup.new "$F" && chmod 555 "$F"
echo "end of $F is now:"; tail -4 "$F"
