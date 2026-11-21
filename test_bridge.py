"""One-command host rehearsal: py -3.13 test_bridge.py.

Builds the portable C programs; replays ~60 source seconds without pacing;
then kills a supervised Python detector and verifies restart and resumed cues.
No camera, QNX SDK or hardware is required. Logs go to bridge-output.log.
"""
import ctypes
import errno
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess
import sys
import uuid
import time

from cue_client import CueClient, CueMessage, CUE_INCOMING, monotonic_ns, pack_cue
from qnx.test_qnx_skeleton import build, binary, cue_rows, kill_tree, check

ROOT = Path(__file__).resolve().parent
SESSION = ROOT / "data/session_20261121_030328_194149"
TRANSCRIPT = []


def read(path):
    return path.read_text(encoding="utf-8", errors="replace")


def wait_for(path, predicate, proc, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = read(path)
        if predicate(text):
            return text
        check(proc.poll() is None, f"process exited {proc.returncode}:\n{text}")
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for output:\n{read(path)}")


def launch(args, output):
    return subprocess.Popen(args, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                            start_new_session=os.name != "nt")


def stop(proc):
    if proc is not None and proc.poll() is None:
        if os.name == "nt":
            proc.kill()
            proc.wait(timeout=5)
        else:
            kill_tree(proc)


def kill_pid(pid):
    if os.name != "nt":
        os.kill(pid, signal.SIGKILL)
        return
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(1, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.TerminateProcess(handle, 99):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def vision_args(folder, seconds=60):
    return [sys.executable, "-u", str(ROOT / "dodge_live.py"),
            "--source", str(SESSION.with_suffix(".mp4")),
            "--npz", str(SESSION.with_suffix(".npz")), "--headless", "--no-beep",
            "--ipc", "--max-seconds", str(seconds), "--log", str(folder / "cues.csv")]


def client_checks(folder):
    # Exercise the real C struct ABI and its clock, independently of Python.
    source = folder / "abi.c"
    source.write_text('''#include "runtime.h"
#include "protocol.h"
#include <stddef.h>
int main(void) {
printf("%zu %zu %zu %zu %zu %zu %llu\\n", sizeof(cue_msg_t),
offsetof(cue_msg_t,type), offsetof(cue_msg_t,dir), offsetof(cue_msg_t,score),
offsetof(cue_msg_t,t_capture), offsetof(cue_msg_t,frame_id),
(unsigned long long)monotonic_ns()); return 0;
}
''', encoding="utf-8")
    exe = folder / ("abi.exe" if os.name == "nt" else "abi")
    gcc = r"C:\msys64\ucrt64\bin\gcc.exe" if os.name == "nt" else "gcc"
    subprocess.run([gcc, "-I", str(ROOT / "qnx/common"), str(source), "-o", str(exe)], check=True)
    before = monotonic_ns()
    values = [int(x) for x in subprocess.check_output([str(exe)], text=True).split()]
    after = monotonic_ns()
    check(values[:6] == [ctypes.sizeof(CueMessage)] +
          [getattr(CueMessage, name).offset for name in
           ("type", "dir", "score", "t_capture", "frame_id")], "C/Python native ABI differs")
    check(before <= values[6] <= after, "C/Python clocks have different epochs")
    check(pack_cue(2, 1.0, 0x123456789abcdef0, 7).hex() ==
          "0000000100000600000000023f800000123456789abcdef00000000700000000",
          "wire format differs")
    path = folder / "client.log"
    proc = None
    with path.open("wb") as output:
        try:
            proc = launch([binary("cue_driver")], output)
            wait_for(path, lambda t: "DRIVER ready" in t, proc)
            client = CueClient()
            try:
                client.connect()
                check(client.send_heartbeat(), "heartbeat send failed")
                check(client.send_cue(CUE_INCOMING, .75, monotonic_ns(), 801) == 0, "cue ACK failed")
                check(client.send_cue(CUE_INCOMING, .75, 2**64-1, 802) == errno.EINVAL,
                      "application error ACK lost")
                client.close()
                check(client.send_cue(CUE_INCOMING, .75, monotonic_ns(), 803) == 0,
                      "cue did not reconnect")
                stop(proc)
                try:
                    client.send_cue(CUE_INCOMING, .75, monotonic_ns(), 804)
                    raise AssertionError("dead peer unexpectedly acknowledged")
                except OSError:
                    pass
                check(client.sock is None, "failed cue retained socket")
                proc = launch([binary("cue_driver")], output)
                wait_for(path, lambda t: t.count("DRIVER ready") == 2, proc)
                deadline = time.monotonic() + 1
                while not client.send_heartbeat():
                    check(time.monotonic() < deadline, "nonblocking heartbeat reconnect failed")
                    time.sleep(.01)
                check(client.send_cue(CUE_INCOMING, .75, monotonic_ns(), 805) == 0,
                      "server restart reconnect failed")
            finally:
                client.close()
        finally:
            stop(proc)
            TRANSCRIPT.append("--- client checks ---\n" + read(path))
    print("PASS client: C ABI + clock, wire bytes, ACK/error ACK, dead peer, reconnect")


def replay(folder, directional=False):
    label = "directional" if directional else "universal"
    driver_path, vision_path = folder / f"driver-{label}.log", folder / f"vision-{label}.log"
    driver = vision = None
    with driver_path.open("wb") as driver_out, vision_path.open("wb") as vision_out:
        try:
            driver = launch([binary("cue_driver")], driver_out)
            wait_for(driver_path, lambda t: "DRIVER ready" in t, driver)
            vision = launch(vision_args(folder) + (["--directional"] if directional else []), vision_out)
            check(vision.wait(timeout=90) == 0, read(vision_path))
            exit_ns = monotonic_ns()
            # Allow the 300 ms watchdog and the complete 440 ms fault pattern.
            time.sleep(.85)
            text, printed = read(driver_path), read(vision_path)
            rows = cue_rows(text)
            events = re.findall(r"^\s*([\d.]+)s\s+(INCOMING|DODGE LEFT|DODGE RIGHT|DUCK)\s+wrist", printed, re.M)
            summary = re.search(r"(\d+) frames .*? (\d+) cues ->", printed)
            check(summary is not None and int(summary[2]) > 0, f"no replay cues:\n{printed}")
            check(len(rows) == len(events) == int(summary[2]), "printed/received cue counts differ")
            check("IPC cue" not in printed, f"IPC failed:\n{printed}")
            directions = {"INCOMING": "INCOMING", "DODGE LEFT": "LEFT",
                          "DODGE RIGHT": "RIGHT", "DUCK": "DUCK"}
            check([r[1] for r in rows] == [directions[name] for _, name in events],
                  "printed/received directions differ")
            if not directional:
                check(all(r[1] == "INCOMING" for r in rows), "default direction changed")
            check(all(float(t) < 60 for t, _ in events), "replay exceeded 60 source seconds")
            faults = re.findall(r"WATCHDOG fault gap_ms=([\d.]+) t_ns=(\d+)", text)
            after_exit = [f for f in faults if int(f[1]) >= exit_ns]
            check(len(after_exit) == 1, "expected exactly one watchdog fault after Python exit")
            tail = text[text.index("WATCHDOG fault", text.index("CUE frame=")):]
            check(tail.count("BUZZER pattern=FAULT") == 1, "post-replay fault pattern repeated")
            check(not any(int(rows[0][3]) < int(f[1]) < exit_ns for f in faults),
                  "watchdog fault during active replay")
            latencies = [float(r[2]) for r in rows]
            check(all(0 <= ms < 1000 for ms in latencies), "invalid frame receipt latency")
            print(f"PASS {label} replay: {summary[1]} frames, {len(rows)}/{len(events)} cues; "
                  f"latency avg={statistics.mean(latencies):.3f} ms max={max(latencies):.3f} ms; "
                  f"one exit fault at {float(after_exit[0][0]):.1f} ms heartbeat gap")
        finally:
            stop(vision)
            stop(driver)
            TRANSCRIPT.extend(["--- replay driver ---\n" + read(driver_path),
                               "--- replay vision ---\n" + read(vision_path)])


def supervised_kill(folder):
    args = vision_args(folder, 600)
    command = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
    path = folder / "supervisor.log"
    proc = None
    with path.open("wb") as output:
        try:
            proc = launch([binary("supervisor"), "--duration", "25", "--vision-cmd", command], output)
            text = wait_for(path, lambda t: "CUE frame=" in t, proc)
            pid = int(re.search(r"vision_started pid=(\d+)", text)[1])
            killed_ns = monotonic_ns()
            kill_pid(pid)
            text = wait_for(path, lambda t: "SUPERVISOR restart=1" in t and
                            "CUE frame=" in t.split("SUPERVISOR restart=1", 1)[1], proc)
            restart = re.search(r"SUPERVISOR restart=1 pid=(\d+) delay_ms=([\d.]+)", text)
            check(int(restart[1]) != pid, "PID did not change")
            check(float(restart[2]) < 1000, "restart took >=1 second")
            check("WATCHDOG recovered" in text.split("SUPERVISOR restart=1", 1)[1],
                  "restarted Python did not restore heartbeats")
            faults = re.findall(r"WATCHDOG fault gap_ms=([\d.]+) t_ns=(\d+)", text)
            check(any(int(ns) > killed_ns for _, ns in faults), "forced kill did not trigger watchdog")
            check(proc.wait(timeout=30) == 0, read(path))
            text = read(path)
            check("SUPERVISOR stopped restarts=1" in text, "unexpected extra restart or failed shutdown")
            print(f"PASS supervisor: killed Python PID {pid}, restarted as {restart[1]} "
                  f"after {float(restart[2]):.1f} ms; heartbeats + cues resumed; clean shutdown")
        finally:
            # On an assertion failure still let the finite supervisor clean up
            # its own children; forcibly killing a Windows parent orphans them.
            if proc is not None and proc.poll() is None:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            stop(proc)
            TRANSCRIPT.append("--- supervised kill ---\n" + read(path))


def main():
    os.chdir(ROOT)
    try:
        build()
        folder = ROOT / ("bridge-test-" + uuid.uuid4().hex[:8])
        folder.mkdir()
        try:
            client_checks(folder)
            replay(folder)
            replay(folder, directional=True)
            supervised_kill(folder)
        finally:
            for path in folder.iterdir():
                path.unlink()
            folder.rmdir()
        print("PASS all bridge host checks (logs: bridge-output.log)")
    finally:
        (ROOT / "bridge-output.log").write_text("\n".join(TRANSCRIPT), encoding="utf-8")


if __name__ == "__main__":
    main()
