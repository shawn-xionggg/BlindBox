"""Client for ipc_posix.c's single-producer loopback transport.

On QNX vision would call MsgSend/MsgSendPulse through the C library instead.
The portable wire is NOT the native struct: it is eight big-endian uint32s.
"""
import ctypes
import os
from pathlib import Path
import socket
import struct
import time

CUE_NONE, CUE_INCOMING, CUE_LEFT, CUE_RIGHT, CUE_DUCK, CUE_FAULT = range(6)
CUE_MSG_TYPE = 0x600


class CueMessage(ctypes.Structure):
    """Native cue_msg_t: offsets 0,2,4,8,16; size 24 including tail padding."""
    _fields_ = [("type", ctypes.c_uint16), ("dir", ctypes.c_uint8),
                ("score", ctypes.c_float), ("t_capture", ctypes.c_uint64),
                ("frame_id", ctypes.c_uint32)]


if os.name == "nt":
    # Match runtime.h's MinGW clock_gettime(CLOCK_MONOTONIC) exactly, rather
    # than assuming CPython's perf_counter has the same epoch/conversion.
    class _Timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_int64), ("tv_nsec", ctypes.c_int32)]

    _dll_path = Path(os.environ.get("DODGE_WINPTHREAD_DLL",
                                   r"C:\msys64\ucrt64\bin\libwinpthread-1.dll"))
    _clock_dll = ctypes.CDLL(str(_dll_path))
    _clock_gettime = _clock_dll.clock_gettime64
    _clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(_Timespec)]
    _clock_gettime.restype = ctypes.c_int

    def monotonic_ns():
        ts = _Timespec()
        if _clock_gettime(1, ctypes.byref(ts)):
            raise OSError("clock_gettime(CLOCK_MONOTONIC) failed")
        return ts.tv_sec * 1_000_000_000 + ts.tv_nsec
else:
    monotonic_ns = time.monotonic_ns


def capture_ns(perf_stamp):
    """Translate LatestCapture's perf_counter receipt stamp to the C clock.

    Sample both clocks at receipt consumption; subtract the frame's age. This
    includes queued-frame age, with only the clock sampling interval as error.
    Replay uses monotonic_ns directly at decoded frame receipt (not video time).
    """
    now = time.perf_counter()
    return monotonic_ns() - int((now - perf_stamp) * 1_000_000_000)


def pack_cue(direction, score, t_capture_ns, frame_id):
    msg = CueMessage(CUE_MSG_TYPE, direction, score, t_capture_ns, frame_id)
    bits = struct.unpack("!I", struct.pack("!f", msg.score))[0]
    return struct.pack("!8I", 1, msg.type, msg.dir, bits,
                       msg.t_capture >> 32, msg.t_capture & 0xffffffff,
                       msg.frame_id, 0)


class CueClient:
    """One processing thread owns this connection. Failed cues are not retried:
    an ACK can be lost after dispatch, so retrying could buzz twice. The next
    call reconnects. Heartbeats never wait for connection or socket space.
    """
    def __init__(self, host="127.0.0.1", port=45873, timeout=0.2):
        self.address = (host, port)
        self.timeout = timeout
        self.sock = None
        self.connecting = False

    def close(self):
        if self.sock is not None:
            self.sock.close()
        self.sock = None
        self.connecting = False

    def connect(self):
        self.close()
        try:
            self.sock = socket.create_connection(self.address, self.timeout)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            self.close()
            raise

    def send_cue(self, dir, score, t_capture_ns, frame_id):
        packet = pack_cue(dir, score, t_capture_ns, frame_id)
        try:
            if self.sock is None or self.connecting:
                self.connect()
            self.sock.settimeout(self.timeout)
            self.sock.sendall(packet)
            ack = bytearray()
            while len(ack) < 4:
                part = self.sock.recv(4 - len(ack))
                if not part:
                    raise ConnectionError("EOF before cue ACK")
                ack.extend(part)
            return struct.unpack("!i", ack)[0]
        except OSError:
            self.close()
            raise

    def send_heartbeat(self):
        try:
            if self.sock is None:
                import errno
                self.sock = socket.socket()
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.setblocking(False)
                rc = self.sock.connect_ex(self.address)
                if rc not in (0, errno.EINPROGRESS, errno.EWOULDBLOCK,
                              errno.EALREADY, 10035, 10036, 10037):
                    self.close()
                    return False
                self.connecting = rc != 0
            self.sock.setblocking(False)
            if self.connecting:
                import select
                _, writable, errors = select.select([], [self.sock], [self.sock], 0)
                if errors or (writable and self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)):
                    self.close()
                    return False
                if not writable:
                    return False
                self.connecting = False
            n = self.sock.send(struct.pack("!8I", 2, 0, 0, 0, 0, 0, 0, 0))
            if n != 32:
                self.close()  # Never leave a partial record on the stream.
                return False
            return True
        except BlockingIOError:
            return False  # Drop heartbeat without waiting.
        except OSError:
            self.close()
            return False
