"""Capture, lossless recording, and stage measurements shared by live apps."""
from collections import defaultdict
from contextlib import contextmanager
import queue
import threading
import time

import cv2


class StageTimings:
    names = ('capture wait', 'pose', 'glove segmentation', 'overlay', 'imshow', 'video write', 'other')

    def __init__(self):
        self.totals = defaultdict(float)
        self.frames = 0
        self.started = time.perf_counter()

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - start

    def report(self, writer=None):
        elapsed = time.perf_counter() - self.started
        n = max(1, self.frames)
        print(f'\nTiming averages ({self.frames} processed frames; ms/frame):')
        for name in self.names:
            print(f'  {name:20s} {1000*self.totals[name]/n:8.3f}')
        if writer is not None:
            print(f'  writer encode worker {1000*writer.encode_seconds/n:8.3f} (overlaps processing if threaded)')
        print(f'  wall/drain total     {1000*elapsed/n:8.3f}')
        print(f'  achievable fps       {self.frames/max(elapsed, 1e-9):8.2f}')


class LatestCapture:
    """One pending webcam frame; slow consumers skip stale frames, never replay them.

    The producer owns read/release. Timestamp is the monotonic read completion
    time (host receipt, not the camera's unknown exposure time).
    """
    def __init__(self, cap):
        self.cap = cap
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.pending = None
        self.done = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                stamp = time.perf_counter()
                if not ok:
                    break
                with self.condition:
                    self.pending = (frame, stamp)
                    self.condition.notify_all()
        except Exception as exc:
            self.error = exc
        finally:
            self.cap.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None or self.done)
            if self.error is not None:
                raise RuntimeError('Capture worker failed') from self.error
            item, self.pending = self.pending, None
            return item if item is not None else (None, None)

    def close(self):
        self.stop.set()
        # A broken backend can block read indefinitely; don't race release against
        # read on another thread. The daemon retains ownership in that case.
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            print('Capture backend has not returned from read; worker will release it on return.')


class VideoSink:
    """Bounded, lossless writer. Backpressure blocks processing; never drops rows.

    write() copies before returning, so the caller can overlay the input safely.
    close() drains and propagates worker failures before the NPZ is saved.
    """
    def __init__(self, path, fps, size, threaded=True, queue_size=8):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError('Could not open MP4 writer')
        self.encode_seconds = 0.
        self.written = 0
        self.error = None
        self.threaded = threaded
        self.closed = False
        self.queue = queue.Queue(maxsize=queue_size)
        if threaded:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _encode(self, frame):
        start = time.perf_counter()
        self.writer.write(frame)
        self.encode_seconds += time.perf_counter() - start
        self.written += 1

    def _check(self):
        if self.error is not None:
            raise RuntimeError('Video writer failed; paired session was not saved') from self.error

    def _put(self, frame):
        while True:
            self._check()
            try:
                self.queue.put(frame, timeout=.05)
                self._check()
                return
            except queue.Full:
                pass

    def write(self, frame):
        if self.closed:
            raise RuntimeError('Writer is closed')
        if self.threaded:
            self._put(frame.copy())
        else:
            self._encode(frame)

    def _run(self):
        try:
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                self._encode(frame)
        except Exception as exc:
            self.error = exc
        finally:
            self.writer.release()

    def close(self):
        if self.closed:
            self._check()
            return
        self.closed = True
        if self.threaded:
            try:
                self._put(None)
            finally:
                self.thread.join()
        else:
            self.writer.release()
        self._check()
