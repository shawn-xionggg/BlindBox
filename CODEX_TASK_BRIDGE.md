# Codex task: connect the Python detector to the C cue driver

Read `AGENTS.md`, `REPORT.md`, `qnx/README.md`, `qnx/common/protocol.h`, `qnx/common/ipc_posix.c`, and `dodge_live.py`. The user is asleep. No camera, no QNX. MSYS2 gcc is in `C:\msys64\ucrt64\bin`.

## Goal
Rehearse the full multi-process pipeline on the laptop: Python vision (`dodge_live.py`) sends cues + a heartbeat every frame to the C `cue_driver` over the **portable backend's exact wire format**. Killing the Python vision process makes `cue_driver` play the fault pattern, and the supervisor restarts the vision process.

## Do this
1. `cue_client.py` (new): a Python client for the portable IPC backend, byte-compatible with `ipc_posix.c`. It provides:
   - connect
   - `send_cue(dir, score, t_capture_ns, frame_id)` → waits for the reply ACK, returns the status
   - `send_heartbeat()` (non-blocking)
   - reconnect on failure

   Pack `cue_msg_t` with the same layout and padding as the C struct on this platform. Add a comment that on QNX the vision process would call `MsgSend` via the C library instead.
2. `dodge_live.py`: add `--ipc`. When set, send a heartbeat every processed frame and, on each cue, send `CUE_INCOMING` (or LEFT/RIGHT/DUCK with `--directional`). Use the capture timestamp in CLOCK_MONOTONIC-compatible nanoseconds, so cue_driver's latency numbers are meaningful: use the same clock the C side uses on Windows, and document the choice. Keep the local beep unless `--no-beep`. Don't change detection logic.
3. `qnx/supervisor/supervisor.c`: allow restarting an arbitrary vision command (e.g. `--vision-cmd "python dodge_live.py --source data/x.mp4 --npz data/x.npz --headless --ipc"`), keeping the existing default. Keep all existing qnx tests passing.
4. `test_bridge.py` (new): build qnx, start `cue_driver`, run `dodge_live.py --headless --ipc` replaying the first ~60 s of `data/session_20261121_030328_194149.mp4` with its npz, then check:
   - cue_driver received the same number of cues that dodge_live printed
   - latencies are logged
   - after dodge_live exits, cue_driver fires the fault pattern exactly once

   Also add a supervisor + `--vision-cmd` run with a forced kill of the Python process, and check that it restarts. Run it and make it pass.

## Don't touch
`early_*.py`, `flow_features.py`, `launch_detector.py`, `auto_label.py`, `review_labels.py`, `export_model_c.py`, `qnx/model/`, `data/`, `models/`, `features/`.

## Finish with
A short summary: files changed, test results, and how to run the rehearsal (one command if possible).
