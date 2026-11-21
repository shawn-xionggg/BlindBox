#!/usr/bin/env python3
"""Build and exercise real processes, acknowledgements, watchdog and restart.

Only Python's standard library is used. Run from any directory. The TCP edge
case checks apply to the portable backend; native QNX needs its own target run.
"""
import errno
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
NATIVE_QNX = sys.platform.startswith("qnx")
SUFFIX = ".exe" if os.name == "nt" else ""
TRANSCRIPT = []


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def binary(name):
    return str(ROOT / "bin" / (name + SUFFIX))


def kill_tree(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=5)


def start(args, output):
    return subprocess.Popen(args, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                            start_new_session=os.name != "nt")


def run_case(label, args, timeout=10):
    with (ROOT / "test-case-output.log").open("w+b") as output:
        proc = start(args, output)
        try:
            code = proc.wait(timeout=timeout)
        except BaseException:
            kill_tree(proc)
            raise
        output.seek(0)
        text = output.read().decode("utf-8", errors="replace")
    TRANSCRIPT.append(f"--- {label} ---\n{text}")
    check(code == 0, f"{label}: exit={code}\n{text}")
    return text


def build():
    env = os.environ.copy()
    if os.name == "nt":
        env["PATH"] = r"C:\msys64\ucrt64\bin;C:\msys64\usr\bin;" + env.get("PATH", "")
    # MSYS make may fail to create its signal pipe in a Windows sandbox;
    # native GNU mingw32-make uses the identical Makefile without MSYS runtime.
    make = shutil.which("mingw32-make" if os.name == "nt" else "make", path=env["PATH"])
    check(make is not None, "GNU make is required")
    command = [make, "-B", "all"]
    if NATIVE_QNX:
        command += ["TARGET=qnx", "CC=qcc"]
    subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=60)


def cue_rows(text):
    return re.findall(r"CUE frame=(\d+) dir=(\w+) latency_ms=([\d.]+) t_ns=(\d+)", text)


def crash_recovery(csv):
    text = run_case("crash recovery", [binary("supervisor"), "--csv", str(csv),
                    "--duration", "3.8", "--crash-at", "0.65"])
    rows = cue_rows(text)
    expected = ["INCOMING", "LEFT", "RIGHT", "INCOMING", "LEFT", "RIGHT", "DUCK", "NONE", "FAULT"]
    check([r[1] for r in rows] == expected, f"missing/extra/reordered cues: {rows}")
    acks = re.findall(r"STUB ack event=(\d+) dir=(\w+)", text)
    check([d for _, d in acks] == expected, f"not all sends acknowledged: {acks}")
    check("vision_exit status=77" in text, "simulated abrupt exit not observed")
    check(text.count("SUPERVISOR restart=") == 1, "expected one restart")
    check("SUPERVISOR stopped restarts=1" in text, "supervisor did not cleanly finish")
    check(text.count("WATCHDOG fault") == 1, "watchdog must fire once per heartbeat gap")
    crash = int(re.search(r"STUB simulated_crash t_ns=(\d+)", text)[1])
    fault = re.search(r"WATCHDOG fault gap_ms=([\d.]+) t_ns=(\d+)", text)
    recovered = int(re.search(r"WATCHDOG recovered t_ns=(\d+)", text)[1])
    restarted = re.search(r"SUPERVISOR restart=1 pid=\d+ delay_ms=([\d.]+) t_ns=(\d+)", text)
    check(crash < int(fault[2]) < int(restarted[2]) <= recovered,
          "fault/restart/recovery did not happen in the crash gap")
    check(300 <= float(fault[1]) < 500, "watchdog missed its deadline by >=200 ms")
    check(float(restarted[1]) < 1000, "restart took >=1 second after detected death")
    check((recovered - crash) / 1e6 < 1100, "heartbeats did not resume within ~1 second")
    gap_text = text[text.index("WATCHDOG fault"):text.index("SUPERVISOR restart=")]
    check("BUZZER pattern=FAULT" in gap_text, "watchdog logged without a fault pattern")
    latencies = [float(r[2]) for r in rows]
    check(all(0 <= v < 100 for v in latencies), "synthetic capture-to-cue latency >=100 ms")
    p99 = sorted(latencies)[-1]  # nearest-rank p99 for nine samples
    print(f"PASS crash: {len(rows)}/{len(expected)} cues + ACKs, one fault, one restart "
          f"after {float(restarted[1]):.1f} ms; watchdog {float(fault[1]):.1f} ms")
    print(f"Synthetic IPC/software cue latency: avg={statistics.mean(latencies):.3f} ms "
          f"p99={p99:.3f} ms max={max(latencies):.3f} ms (n={len(latencies)})")


def healthy_and_absent(csv):
    healthy = run_case("healthy heartbeats", [binary("supervisor"), "--csv", str(csv),
                       "--duration", "2.0"])
    check(len(cue_rows(healthy)) == 6, "healthy run lost cues")
    check("WATCHDOG fault" not in healthy, "healthy frame heartbeats triggered watchdog")
    check("SUPERVISOR stopped restarts=0" in healthy, "healthy run restarted")
    absent = run_case("missing heartbeats", [binary("cue_driver"), "--run-for", "0.9"])
    check(absent.count("WATCHDOG fault") == 1, "sustained outage must fault just once")
    check(absent.count("BUZZER pattern=FAULT") == 1, "fault pattern repeated during one outage")
    print("PASS healthy: 6/6 cues, no false watchdog fault; absent producer: exactly one fault")


def receive_ack(peer):
    data = bytearray()
    while len(data) < 4:
        part = peer.recv(4 - len(data))
        check(bool(part), "EOF before acknowledgement")
        data.extend(part)
    return struct.unpack("!i", data)[0]


def transport_edges():
    if NATIVE_QNX:
        return  # Native IPC cannot be exercised by a Python TCP client.
    with (ROOT / "test-case-output.log").open("w+b") as output:
        proc = start([binary("cue_driver"), "--run-for", "1.5"], output)
        try:
            deadline = time.monotonic() + 1
            while True:
                try:
                    peer = socket.create_connection(("127.0.0.1", 45873), timeout=0.2)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
            with peer:
                peer.settimeout(1)
                heartbeat = struct.pack("!8I", 2, 0, 0, 0, 0, 0, 0, 0)
                # Several receive timeouts must preserve a partially received record.
                peer.sendall(heartbeat[:5])
                time.sleep(0.06)
                peer.sendall(heartbeat[5:])
                packet = struct.pack("!8I", 1, 0x600, 2, 0x3F800000, 0, 0, 901, 0)
                peer.sendall(packet[:11])
                time.sleep(0.04)
                peer.sendall(packet[11:])
                check(receive_ack(peer) == 0, "fragmented cue was not acknowledged")
                # Future timestamp rejected by cue_driver, with application error ACK.
                peer.sendall(struct.pack("!8I", 1, 0x600, 3, 0, 0xFFFFFFFF, 0xFFFFFFFF, 902, 0))
                check(receive_ack(peer) == errno.EINVAL, "invalid cue was not rejected")
                peer.sendall(packet[:9])  # Disconnect midway through a fresh record.
            with socket.create_connection(("127.0.0.1", 45873), timeout=1) as peer:
                peer.settimeout(1)
                peer.sendall(heartbeat + struct.pack("!8I", 1, 0x600, 4, 0, 0, 0, 903, 0))
                check(receive_ack(peer) == 0, "new client retained stale partial record")
            check(proc.wait(timeout=4) == 0, "driver failed edge cases")
        except BaseException:
            kill_tree(proc)
            raise
        finally:
            output.seek(0)
            text = output.read().decode("utf-8", errors="replace")
            TRANSCRIPT.append(f"--- TCP edge cases ---\n{text}")
    check([r[0] for r in cue_rows(text)] == ["901", "903"], "invalid or partial cue was delivered")
    print("PASS transport: fragmented records, reply ACK, invalid timestamp, mid-record EOF, reconnect")


def main():
    try:
        build()
        # Exercise Windows process argument quoting with an absolute path containing spaces.
        csv = ROOT / "test events with spaces.csv"
        try:
            shutil.copyfile(ROOT / "test_events.csv", csv)
            crash_recovery(csv)
            healthy_and_absent(csv)
        finally:
            csv.unlink(missing_ok=True)
        transport_edges()
        print("PASS all QNX skeleton host checks")
    finally:
        (ROOT / "test-output.log").write_text("\n".join(TRANSCRIPT), encoding="utf-8")
        (ROOT / "test-case-output.log").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
