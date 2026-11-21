# QNX process skeleton

Stream D's C99 process demo: scripted vision sends cues to a cue driver; a
supervisor restarts vision after an abrupt exit. All additions are under `qnx/`.
The separately owned `qnx/model/` is not part of this build.

## Files and process behavior

| File | Purpose |
| --- | --- |
| `common/protocol.h` | Shared directions, cue message, heartbeat code and channel name |
| `common/ipc.h` | One opaque server/client API, including cleanup and receive deadlines |
| `common/ipc_qnx.c` | QNX named channel, synchronous messages/replies and heartbeat pulses |
| `common/ipc_posix.c` | Loopback TCP test backend for Windows and POSIX hosts |
| `common/runtime.h` | Monotonic nanosecond clock, sleep and QNX FIFO scheduling |
| `cue_driver/cue_driver.c` | Buzzer pattern state machine, heartbeat watchdog, cue latency log |
| `vision_stub/vision_stub.c` | CSV replay, ~20 Hz heartbeats and optional abrupt crash |
| `supervisor/supervisor.c` | Child lifetime monitoring, delayed restart and cleanup |
| `Makefile` | Warning-clean C99 builds; native GCC and QNX qcc variants |
| `test_qnx_skeleton.py`, `test_events.csv` | Automated build, crash demo and transport checks |

QNX uses `name_attach`/`name_open`, `MsgSend`/`MsgReceive`/`MsgReply` and
`MsgSendPulse`. Message passing is native microkernel IPC; synchronous sends
and acknowledgements provide back-pressure. `TimerTimeout` bounds receive
waits without resetting the watchdog deadline for connection traffic. The
server handles name-open handshakes, disconnect/unblock pulses and malformed
messages, and never replies to heartbeat pulses. These follow the
[QNX name_attach reference](https://qnx.com/developers/docs/8.0/com.qnx.doc.neutrino.lib_ref/topic/n/name_attach.html)
and [TimerTimeout reference](https://qnx.com/developers/docs/8.0/com.qnx.doc.neutrino.lib_ref/topic/t/timertimeout.html).

FIFO application priorities are cue driver 30, vision 20, supervisor 10.
Because named QNX channels inherit the sender's priority, the driver restores
30 immediately after each IPC receive returns, before application processing.
Scheduling failures are printed; they must be resolved on the target before
claiming real-time behavior. QNX public named-channel creation may also require
the `PROCMGR_AID_PUBLIC_CHANNEL` ability in the target security policy. See
[QNX scheduling](https://qnx.com/developers/docs/8.0/com.qnx.doc.neutrino.sys_arch/topic/kernel_Scheduling_priority.html)
and [abilities](https://qnx.com/developers/docs/8.0/com.qnx.doc.security.system/topic/manual/abilities_reference.html).

On Windows the backend uses Winsock and child management uses `CreateProcess`
and `WaitForSingleObject`; POSIX/QNX use `posix_spawn` and `waitpid`. HAM is the
production QNX alternative to this custom supervisor. This demo restarts only
vision: a cue-driver exit makes the supervisor fail and stop its other child.

## Build and run here (PowerShell, MSYS2 UCRT)

From the repository root:

```powershell
$env:Path = "C:\msys64\ucrt64\bin;C:\msys64\usr\bin;" + $env:Path
mingw32-make -C qnx
python qnx/test_qnx_skeleton.py
qnx/bin/supervisor.exe --csv qnx/test_events.csv --duration 3.8 --crash-at 0.65
```

`make -C qnx` uses the same Makefile when MSYS make is runnable. On this machine
MSYS make failed to create its signal pipe (`Win32 error 5`), so the verified
build uses the installed native GNU `mingw32-make`. The test selects it on
Windows and forces a rebuild with `-Wall -Wextra -Wpedantic -Werror`.

On a POSIX development host:

```sh
make -C qnx
python3 qnx/test_qnx_skeleton.py
./qnx/bin/supervisor --csv qnx/test_events.csv --duration 3.8 --crash-at 0.65
```

The supervisor locates sibling executables relative to its invocation path.
Invoke it with an explicit path as above; when invoked without a directory,
the siblings must be in the current directory. `--duration` defaults to five
seconds. Normal shutdown stops/reaps both children. Unix SIGINT/SIGTERM also
cause cleanup; Windows finite runs terminate and wait for both children.
Forcibly killing the Windows supervisor itself is not a supported cleanup path.

The fixed test endpoint is **127.0.0.1:45873**. Run one test/demo at a time and
reserve that port. This is a single-producer backend; it does not authenticate
local peers or multiplex producers. It uses 32-byte network-order records,
preserves partial records across receive timeouts, and discards partial state
when a producer disconnects. Heartbeats have no ACK and do not wait for socket
space; they may be dropped when congested. A partial heartbeat write closes the
connection to avoid corrupting later messages. Cues block for their reply ACK.

## Python detector host rehearsal

From the repository root, with OpenCV, NumPy, joblib and the detector's Python
dependencies installed (on this laptop, use Python 3.13):

```powershell
py -3.13 test_bridge.py
```

This builds C with warnings as errors, checks the Python/C ABI and clock,
replays the first 60 source seconds of the supplied session in universal and
directional modes, compares printed and received cues, checks the single fault
pattern after each Python exit, then
forcibly kills a supervised Python detector and verifies restart, heartbeat
recovery and resumed cues. Logs are saved to `bridge-output.log`. Run it
separately from the skeleton tests: both use the same single-producer port.
Replay runs as fast as processing allows; its latencies measure decoded-frame
receipt through detector processing to software cue dispatch, not punch onset
or a real camera's exposure. Loading Python/model dependencies can trigger an
initial watchdog fault before the first frame; it recovers on the first
processed-frame heartbeat. This startup gap is separate from the exit/kill test.

For a camera run after building, from the repository root:

```powershell
qnx/bin/supervisor.exe --duration 120 --vision-cmd 'py -3.13 -u dodge_live.py --source 1 --ipc'
```

`--vision-cmd` replaces the default CSV vision stub and cannot be combined with
`--csv` or the stub's `--crash-at`. The command inherits the current directory
and environment. Windows uses CreateProcess command-line quoting and PATH
lookup; POSIX/QNX uses `/bin/sh -c 'exec ...'`. Supply one executable plus
arguments, with paths quoted for the host, so the supervisor directly owns the
vision process. Restart repeats that same command from the beginning. The
existing CSV mode is unchanged.

`dodge_live.py --ipc` sends a heartbeat after each processed frame and waits
for an ACK on each cue. `--directional` maps the experimental output to the C
LEFT/RIGHT/DUCK enum; the default sends INCOMING. Local beeps continue unless
`--no-beep` or `--headless` is set. `--max-seconds 60` limits source footage.
Transport failure closes the client connection and the next call reconnects;
failed cues are logged and never automatically resent because a lost ACK
could otherwise duplicate a buzz. Heartbeats never block for socket space
or connection completion.

The Python client packs the native `cue_msg_t` layout (24 bytes on this host,
including padding) and then serializes its fields to the portable backend's
**32-byte, eight-big-endian-word record**. It does not send raw struct bytes.
On Windows it calls `clock_gettime64(CLOCK_MONOTONIC)` from the same MSYS2
`libwinpthread-1.dll` used by the C build; override its default
`C:\msys64\ucrt64\bin\libwinpthread-1.dll` path with `DODGE_WINPTHREAD_DLL`.
Live `LatestCapture` perf-counter receipt timestamps are translated by
subtracting frame age from that clock, preserving queued-frame time to within
the clock sampling interval. Replay timestamps each decoded frame's receipt.
On POSIX Python uses `time.monotonic_ns()`. QNX vision would use the C library's
native MsgSend/pulses instead of this Python TCP client.

## QNX SDP 8.0 build and target run (not verified here)

In a shell configured for the installed QNX SDK:

```sh
make -C qnx TARGET=qnx CC="qcc -Vgcc_ntoaarch64le" EXE=
```

The build defines `_QNX_SOURCE` to expose native APIs under C99; QNX's pthread
functions are in libc. See [qcc options](https://qnx.com/developers/docs/8.0/com.qnx.doc.neutrino.utilities/topic/q/qcc.html).
Cross-building needs the SDK host's GNU make; this does not require Python.
Copy the three resulting `bin/` executables and the CSV to the Pi, keeping the
executables together, then on the target run:

```sh
./bin/supervisor --csv ./test_events.csv --duration 3.8 --crash-at 0.65
```

Switching toolchains in the same folder requires a forced rebuild (`make -B`)
or `make clean` first; `make clean` uses Python and only removes generated
files directly under this skeleton's `bin/` directory. It does not touch models.

## CSV and timing semantics

CSV rows are `time_s,dir`, ordered by time relative to each stub start. A header,
blank lines and `#` comment lines are accepted. Directions accept `NONE`,
`INCOMING`, `LEFT`, `RIGHT`, `DUCK`, `FAULT`, optional `CUE_` prefixes, or numeric
enum values 0 through 5. Times must be finite, nonnegative and at most 86400 s;
malformed/unsorted rows cause an error. Example:

```csv
time_s,dir
0.10,INCOMING
0.25,LEFT
0.45,RIGHT
0.90,DUCK
```

Standalone programs:

```sh
./bin/cue_driver --run-for 5
./bin/vision_stub --csv test_events.csv --run-for 4 --crash-at 0.65
```

Without `--run-for`, the driver runs until stopped and the stub runs for ten
seconds. The stub continues heartbeats after its final event. A crash uses
`_Exit(77)` without a graceful IPC close. The supervisor detects death on a
~10 ms polling loop and waits 650 ms before starting a replacement, leaving
enough silence to demonstrate the 300 ms heartbeat watchdog. It injects a
crash only into the first generation. Every replacement replays the CSV from
time zero; this is a scripted demo, not exactly-once event delivery or a
persistent vision checkpoint. Frame IDs also restart with each generation.

The watchdog starts when the driver starts, fires once after at least 300 ms
without a heartbeat, and rearms when a heartbeat arrives. Cue messages alone
do not keep it alive. The fault pattern preempts any active pattern. All new
patterns also preempt the previous pattern; bursts are not queued for later
playback. Buzzer durations advance on timed receive-loop iterations, without
sleeping through heartbeats: incoming 60 ms; left two 40 ms tones with a 45 ms
gap; right 130 ms; duck three 35 ms tones/gaps; fault two 180 ms tones with an
80 ms gap. `NONE` turns the output off. Pattern precision depends on actual
scheduling and timer granularity; these cues still need human/hardware tuning.

`cue_msg_t.t_capture` is **CLOCK_MONOTONIC nanoseconds on the same host**, not
ClockCycles. The stub timestamps immediately before sending a synthetic cue.
`latency_ms` measures that timestamp to software buzzer-pattern dispatch, and
includes IPC and software logging overhead. It does not measure camera
exposure, inference, punch onset, GPIO voltage or audible buzzer onset. The
QNX raw-struct IPC ABI requires all participants to rebuild from this header.
The direction enum intentionally follows this task (including `INCOMING`),
not the older enum draft in `AGENTS.md`.

## Verification and remaining work

The host test builds all three programs, checks nine cue deliveries and their
ACKs across an abrupt crash, and verifies a single watchdog fault before one
restart and subsequent heartbeat recovery. It also verifies a healthy run
with six cues and no false watchdog fault, a sustained outage with one fault,
fragmented TCP records, invalid-timestamp rejection and a reconnect after a
mid-record EOF. Paths containing spaces exercise Windows child argument
quoting. Full logs are saved to ignored `test-output.log`; the test prints
measured average/p99/max synthetic latency, watchdog gap and restart time.

Verified here using **Windows x64, UCRT GCC 15.2.0, GNU mingw32-make and Python
3.12.11**. A representative passing run delivered **9/9 cues**, restarted after
**669.3 ms**, and faulted at a **314.5 ms** heartbeat gap. Synthetic latency was
**0.465 ms average, 1.081 ms p99/max** (nine samples; nearest-rank p99).
These are host observations, not Pi performance guarantees.

**Unverified until SDK/target testing:** qcc compilation; QNX name registration,
message/pulse delivery and teardown; security abilities; actual FIFO priority
behavior and loaded-system deadline/jitter; Pi 5 GPIO and buzzer timing. The
QNX GPIO code is explicitly a logging-only `TODO: verify on hardware` stub:
**it does not drive a physical buzzer**. Linux/POSIX socket compilation is also
untested here. Camera capture, shared frame memory, real inference and the
latency dashboard are other streams, not implemented by this skeleton.
