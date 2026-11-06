"""Monotonic deadlines and state-message bookkeeping. Pure Python/numpy.

Deadlines are ``t0 + n * period``. A tick that starts late keeps the phase. If a whole
period was missed, the skipped slots are counted and the schedule jumps to the most recent
slot instead of running the missed ticks back to back (no catch-up bursts).

State freshness is decided by the driver's message header id, not by the getter call: a
read whose id equals the previous read's is a repeated sample.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class MonotonicClock:
    """time.monotonic with a sleep that wakes slightly early and spins to the deadline."""

    def __init__(self, spin_s: float = 0.0003):
        self.spin_s = spin_s

    @staticmethod
    def now() -> float:
        return time.monotonic()

    def sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining > self.spin_s:
            time.sleep(remaining - self.spin_s)
        while time.monotonic() < deadline:
            pass


class Deadlines:
    def __init__(self, period: float, t0: float):
        if period <= 0:
            raise ValueError("period must be positive")
        self.period, self.t0 = float(period), float(t0)
        self.index = 0
        self.skipped_total = 0

    def due(self, index: int) -> float:
        return self.t0 + index * self.period

    def advance(self, now: float) -> tuple[int, float, int]:
        """Move past the current tick. Returns (next index, its deadline, slots skipped)."""
        nxt = self.index + 1
        skipped = 0
        if now >= self.due(nxt) + self.period:
            latest = int(math.floor((now - self.t0) / self.period))
            skipped = latest - nxt
            nxt = latest
        self.index = nxt
        self.skipped_total += skipped
        return nxt, self.due(nxt), skipped


def interval_stats(values) -> dict:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None, "mean": None}
    return {"count": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "mean": float(a.mean())}


@dataclass
class StateTracker:
    """Fresh vs repeated driver messages, by header id."""

    last_id: Optional[int] = None
    last_ts_us: Optional[int] = None
    last_fresh_host: Optional[float] = None
    fresh: int = 0
    repeated: int = 0
    id_gaps: int = 0                  # message ids skipped between consecutive fresh reads
    out_of_order: int = 0
    header_intervals_s: list = field(default_factory=list)
    host_intervals_s: list = field(default_factory=list)

    def add(self, header_id: int, header_ts_us: int, host_time: float) -> bool:
        """Record one read; True when it carries a message not seen before."""
        if self.last_id is not None and header_id == self.last_id:
            self.repeated += 1
            return False
        if self.last_id is not None:
            if header_id < self.last_id:
                self.out_of_order += 1
            else:
                self.id_gaps += header_id - self.last_id - 1
                self.header_intervals_s.append((header_ts_us - self.last_ts_us) * 1e-6)
                self.host_intervals_s.append(host_time - self.last_fresh_host)
        self.last_id, self.last_ts_us, self.last_fresh_host = header_id, header_ts_us, host_time
        self.fresh += 1
        return True

    def summary(self, duration_s: float | None = None) -> dict:
        out = {
            "fresh_reads": self.fresh,
            "repeated_reads": self.repeated,
            "skipped_message_ids": self.id_gaps,
            "out_of_order_ids": self.out_of_order,
            "header_interval_s": interval_stats(self.header_intervals_s),
            "host_first_seen_interval_s": interval_stats(self.host_intervals_s),
        }
        if duration_s:
            out["fresh_rate_hz"] = self.fresh / duration_s
        return out
