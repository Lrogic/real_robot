"""Background RealSense reader that keeps only the newest frame (no queue builds up)."""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pipeline.camera import FrameTiming, RealSenseStream


@dataclass(frozen=True)
class ReaderFrame:
    seq: int              # frames received by this reader, starting at 1
    rgb: np.ndarray
    timing: FrameTiming   # RealSense frame number and timestamp
    received: float       # time.perf_counter() when the frame was ready in host memory


class LatestFrameReader:
    def __init__(self, stream: RealSenseStream):
        self._stream = stream
        self._lock = threading.Condition()
        self._latest: Optional[ReaderFrame] = None
        self._error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="realsense-reader", daemon=True)
        self.received = 0
        self.first_frame_number: Optional[int] = None
        self.last_frame_number: Optional[int] = None

    def __enter__(self) -> "LatestFrameReader":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def camera_frames_missed(self) -> int:
        """Frame numbers the reader never received (dropped before reaching this process)."""
        if self.first_frame_number is None:
            return 0
        return self.last_frame_number - self.first_frame_number + 1 - self.received

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                rgb, _depth, timing = self._stream.read()
                with self._lock:
                    self.received += 1
                    if self.first_frame_number is None:
                        self.first_frame_number = timing.frame_number
                    self.last_frame_number = timing.frame_number
                    self._latest = ReaderFrame(self.received, rgb, timing, time.perf_counter())
                    self._lock.notify_all()
        except BaseException as error:  # surfaced to the main thread by latest()
            with self._lock:
                self._error = error
                self._lock.notify_all()

    def latest(self, newer_than: int = 0, timeout: float = 0.0) -> Optional[ReaderFrame]:
        """Newest frame with seq > newer_than, waiting up to timeout seconds; else None."""
        with self._lock:
            self._lock.wait_for(lambda: self._error is not None
                                or (self._latest is not None and self._latest.seq > newer_than),
                                timeout=timeout)
            if self._error is not None:
                raise RuntimeError("RealSense reader failed") from self._error
            if self._latest is None or self._latest.seq <= newer_than:
                return None
            return self._latest
