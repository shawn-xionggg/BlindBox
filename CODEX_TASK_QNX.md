# Codex task: QNX process skeleton (AGENTS.md stream D)

Read `AGENTS.md` first. The user is asleep. **No QNX SDK or Pi is available**; this machine only has MSYS2 g++/gcc (`C:\msys64\ucrt64\bin`). Write code that is real QNX code where it matters, but also builds and runs here so the logic can be tested.

## Deliverables (all new, under `qnx/`)
- `qnx/common/protocol.h`: shared message format, as proposed in AGENTS.md:
  - `cue_msg_t` (type, dir, score, `t_capture` in ns, `frame_id`)
  - `CUE_NONE / CUE_INCOMING / CUE_LEFT / CUE_RIGHT / CUE_DUCK / CUE_FAULT`
  - a heartbeat pulse code
  - a fixed channel name
- `qnx/common/ipc.h` + `ipc_qnx.c` + `ipc_posix.c`: a tiny IPC layer with one API:
  - `ipc_server_open(name)`, `ipc_client_open(name)`
  - `ipc_send_cue(conn, const cue_msg_t*)` (blocking send + reply ack)
  - `ipc_receive(server, cue_msg_t*, int *is_pulse, int timeout_ms)`
  - `ipc_reply(server, rcvid, status)`
  - `ipc_send_heartbeat(conn)` (non-blocking pulse)

  On QNX (`#ifdef __QNX__`): `name_attach`/`name_open`, `MsgSend`/`MsgReceive`/`MsgReply`, `MsgSendPulse`, timeouts with `TimerTimeout`. Elsewhere: Unix-domain or localhost TCP sockets with the same semantics. Comment which QNX calls are used and why: message passing is the QNX microkernel's native IPC, and send-blocking gives back-pressure.
- `qnx/cue_driver/cue_driver.c`: highest-priority process. Sets `SCHED_FIFO` priority via `pthread_setschedparam` (on QNX), receives cue messages, and drives the buzzer through a `buzzer_on(ms)` / `buzzer_pattern(dir)` abstraction:
  - QNX/Pi 5 GPIO implementation behind `#ifdef __QNX__`, clearly marked `TODO: verify on hardware`. Don't invent APIs you're unsure of; leave a stub with a comment if needed.
  - A stdout implementation elsewhere.
  - Heartbeat watchdog: no heartbeat for 300 ms → play the fault pattern once and log it.
  - Logs capture-to-cue latency per message, using the timestamp in the message and `clock_gettime(CLOCK_MONOTONIC)`.
- `qnx/vision_stub/vision_stub.c`: stands in for the vision process. Reads a CSV of scripted cue events (time_s, dir), sends them at the right times, sends heartbeats every frame (~20 Hz), and can simulate a crash (`--crash-at <s>`: exit abruptly).
- `qnx/supervisor/supervisor.c`: starts `cue_driver` and `vision_stub`, restarts `vision_stub` within ~1 s if it dies (`posix_spawn` + `waitpid`), and logs restarts. Note in comments that on QNX the High Availability Manager (HAM) is the production alternative.
- `qnx/Makefile`: builds everything with gcc here (`make`), with a commented `qcc -Vgcc_ntoaarch64le` variant for QNX.
- `qnx/test_qnx_skeleton.sh` (or `.py`): builds, runs supervisor + stub with a short scripted CSV and a simulated crash, and checks the output:
  - all cues arrive
  - the fault pattern fires during the crash gap
  - the stub restarts
  - latencies are logged

  Run it and make it pass here.
- `qnx/README.md`: what runs where, how to build on QNX (qcc), what is verified here vs **unverified until real hardware** (GPIO, priorities, name_attach on the Pi).

## Rules
- Do not modify any existing Python file, `data/`, `models/`, or `features/`. A separate model-export task (by Claude) writes `qnx/model/`. Don't touch that folder.
- Keep it simple C99. No external libraries.
- Finish with a short summary: files, test result, and what is unverified.
