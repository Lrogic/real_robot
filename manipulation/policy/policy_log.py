"""JSON-lines log of the policy observation pipeline (--manipulation-logging) and its timing stats.

Every line has "t" (time.time()) and "mono" (time.monotonic()). Lines with "event" are
episode, trust and recovery events; lines with "tick" are one control step each.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

LOG_FILE = "policy_log.jsonl"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "value"):  # Enum
        return value.value
    return str(value)


class PolicyLog:
    """Writes nothing when path is None; the timing stats are kept either way."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self._file = open(path, "a") if path is not None else None
        self.ticks = 0
        self.first_tick_at: Optional[float] = None
        self.last_tick_at: Optional[float] = None
        self.overruns = 0
        self.recent = deque(maxlen=200)  # (tick_interval_s, build_s, frame_age_s)

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def event(self, kind: str, **data) -> None:
        self._write({"event": kind, **data})

    def tick(self, record: dict, tick_at: float, build_s: float, frame_age_s: Optional[float],
             overrun: bool) -> Optional[float]:
        """Log one control step; returns the interval since the previous tick."""
        interval = None if self.last_tick_at is None else tick_at - self.last_tick_at
        if self.first_tick_at is None:
            self.first_tick_at = tick_at
        self.last_tick_at = tick_at
        self.ticks += 1
        self.overruns += int(overrun)
        self.recent.append((interval, build_s, frame_age_s))
        self._write({"tick": self.ticks - 1, "tick_interval_s": interval, "rate_hz": self.recent_rate(),
                     "overall_rate_hz": self.overall_rate(), "build_s": build_s, "frame_age_s": frame_age_s,
                     "overrun": overrun, **record})
        return interval

    def recent_rate(self) -> Optional[float]:
        intervals = [i for i, _, _ in self.recent if i is not None]
        return None if not intervals else len(intervals) / max(sum(intervals), 1e-9)

    def overall_rate(self) -> Optional[float]:
        if self.ticks < 2:
            return None
        return (self.ticks - 1) / max(self.last_tick_at - self.first_tick_at, 1e-9)

    def stats_line(self) -> str:
        def ms(values):
            values = [v for v in values if v is not None]
            if not values:
                return "-"
            return f"mean {1000 * np.mean(values):.1f} / p95 {1000 * np.percentile(values, 95):.1f} ms"

        rate = self.recent_rate()
        overall = self.overall_rate()
        return (f"{self.ticks} ticks, recent {'-' if rate is None else f'{rate:.1f}'} Hz, overall "
                f"{'-' if overall is None else f'{overall:.1f}'} Hz, {self.overruns} overruns | "
                f"interval {ms(i for i, _, _ in self.recent)} | build {ms(b for _, b, _ in self.recent)} | "
                f"frame age {ms(a for _, _, a in self.recent)}")

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()

    def _write(self, record: dict) -> None:
        if self._file is None or self._file.closed:
            return
        line = {"t": time.time(), "mono": time.monotonic(), **record}
        self._file.write(json.dumps(line, default=_jsonable) + "\n")
        self._file.flush()
