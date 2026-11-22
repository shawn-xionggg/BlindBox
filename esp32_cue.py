"""
Laptop -> ESP32 hat buzzer (firmware: esp32/boxing_buzzer/boxing_buzzer.ino).

One byte per cue, so sending takes well under a millisecond:
  DODGE LEFT 'L', DODGE RIGHT 'R', DUCK 'D', STEP BACK 'B', INCOMING 'I'
  'H' heartbeat (every frame), 'P' ping (ESP32 replies 'K'), 'S' stop, 'F' fault sound,
  'X' goodbye (clean exit, so the ESP32 doesn't sound its "laptop gone" alarm)

Targets:
  COM5 / /dev/ttyUSB0        USB cable (needs:  pip install pyserial)
  udp:192.168.4.1            WiFi: connect the laptop to the ESP32's hotspot "dodge-hat" first
  udp:192.168.4.1:4210       (port optional, default 4210)

Test it on its own (plays every cue, measures the round trip):
  python esp32_cue.py COM5 --test
  python esp32_cue.py udp:192.168.4.1 --test
  python esp32_cue.py --list            # show serial ports (to find the ESP32's COM number)
"""
import argparse
import socket
import statistics
import sys
import time

CODES = {"DODGE LEFT": b"L", "DODGE RIGHT": b"R", "DUCK": b"D", "STEP BACK": b"B", "INCOMING": b"I"}
UDP_PORT = 4210


class Esp32Cue:
    def __init__(self, target):
        self.target = target
        self.sock = self.ser = None
        if target.startswith("udp:"):
            host, _, port = target[4:].partition(":")
            self.addr = (host, int(port or UDP_PORT))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
        else:
            try:
                import serial
            except ImportError:
                raise SystemExit("USB mode needs pyserial:  python -m pip install pyserial   "
                                 "(or use WiFi: --esp32 udp:192.168.4.1)")
            s = serial.Serial()
            s.port, s.baudrate, s.timeout, s.write_timeout = target, 115200, 0, 0
            s.dtr = s.rts = False          # stop the port opening from resetting the ESP32
            s.open()
            time.sleep(1.5)                # in case it reset anyway: let it boot
            s.reset_input_buffer()
            self.ser = s

    def _write(self, b):
        try:
            if self.sock is not None:
                self.sock.sendto(b, self.addr)
            else:
                self.ser.write(b)
        except Exception as exc:           # never let the buzzer link crash the demo
            print(f"ESP32 send failed ({exc})", flush=True)

    def send(self, cue_name):
        self._write(CODES.get(cue_name, b"I"))

    def heartbeat(self):
        self._write(b"H")

    def stop(self):
        self._write(b"S")

    def ping(self, timeout=0.5):
        """Round-trip time in ms (laptop -> ESP32 -> laptop), or None if no reply."""
        if self.sock is not None:
            try:
                while True:
                    self.sock.recvfrom(64)  # drop stale replies
            except (BlockingIOError, OSError):
                pass
        else:
            self.ser.reset_input_buffer()
        t0 = time.perf_counter()
        self._write(b"P")
        while time.perf_counter() - t0 < timeout:
            try:
                data = self.sock.recvfrom(64)[0] if self.sock is not None else self.ser.read(64)
            except (BlockingIOError, OSError):
                data = b""
            if b"K" in data:
                return (time.perf_counter() - t0) * 1000
            time.sleep(0.0002)
        return None

    def close(self):
        try:
            self._write(b"X")
            if self.ser is not None:
                self.ser.close()
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="COM5, /dev/ttyUSB0, or udp:192.168.4.1")
    ap.add_argument("--test", action="store_true", help="ping 20 times, then play every cue")
    ap.add_argument("--list", action="store_true", help="list serial ports")
    args = ap.parse_args()
    if args.list or not args.target:
        try:
            from serial.tools import list_ports
            for p in list_ports.comports():
                print(f"{p.device:10s} {p.description}")
        except ImportError:
            print("pyserial not installed:  python -m pip install pyserial")
        return
    esp = Esp32Cue(args.target)
    rtts = [r for r in (esp.ping() for _ in range(20)) if r is not None]
    if not rtts:
        print("No reply from the ESP32. Check: firmware flashed? right COM port / joined the 'dodge-hat' WiFi? "
              "Serial Monitor closed (only one program can use the port)?")
        sys.exit(1)
    print(f"ESP32 answered {len(rtts)}/20 pings: round trip median {statistics.median(rtts):.1f} ms, "
          f"worst {max(rtts):.1f} ms (one way is about half)")
    if args.test:
        # what you should hear with the firmware default (MOVE_TOWARD_BUZZ = false)
        expect = {"DODGE LEFT": "punch from your RIGHT -> RIGHT ear",
                  "DODGE RIGHT": "punch from your LEFT  -> LEFT ear",
                  "DUCK": "straight punch -> FRONT",
                  "STEP BACK": "uppercut -> ALL THREE",
                  "INCOMING": "one-beep mode -> FRONT"}
        for name in CODES:
            print(f"{name:12s} {expect[name]}", flush=True)
            esp.send(name)
            time.sleep(1.2)
    esp.close()


if __name__ == "__main__":
    main()
