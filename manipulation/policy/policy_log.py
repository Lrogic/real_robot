"""JSON-lines log of the policy observation pipeline (--manipulation-logging) and its timing stats.

Every line has "t" (time.time()) and "mono" (time.monotonic()). Lines with "event" are
episode, trust and recovery events; lines with "tick" are one control step each.

Each tick line has tick_interval_s (since the previous tick that passed the schedule), rate_hz
(over the last RECENT_TICKS ticks), overall_rate_hz (since the first tick) and build_s (from
passing the schedule to the end of the command send). episode_end lines carry "summary":
n / mean / median / p95 / worst of the configured tick fields (logging.summary_fields), and
control_state_ticks, the summarized ticks spent in each control state (running, hold_stale,
hold_tracking; nothing new is commanded on a held tick).
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .controller import CONTROL_STATES

LOG_FILE = "policy_log.jsonl"
RECENT_TICKS = 200
DEFAULT_SUMMARY_FIELDS = ("build_s", "tick_interval_s", "frame_age_s")
# Measured to the command send, so only ticks that reached the send are summarized.
SEND_FIELDS = frozenset({"build_s"})
CONTROL_STATE_TICKS = "control_state_ticks"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "value"):  # Enum
        return value.value
    return str(value)


def summarize(values: Sequence[float]) -> dict:
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "worst": float(np.max(values))}


def format_summary(summary: dict) -> str:
    """One line for the terminal; fields named *_s are shown in ms."""
    parts = []
    for name, s in summary.items():
        if name == CONTROL_STATE_TICKS:
            parts.append("ticks: " + ", ".join(f"{state} {count}" for state, count in s.items()))
            continue
        if not s["n"]:
            parts.append(f"{name}: no samples")
            continue
        scale, unit = (1000.0, " ms") if name.endswith("_s") else (1.0, "")
        parts.append(f"{name}: mean {scale * s['mean']:.2f} / median {scale * s['median']:.2f} / "
                     f"p95 {scale * s['p95']:.2f} / worst {scale * s['worst']:.2f}{unit} (n={s['n']})")
    return " | ".join(parts)


class PolicyLog:
    """Writes nothing when path is None; the timing stats are kept either way."""

    def __init__(self, path: Optional[Path], summary_fields: Sequence[str] = DEFAULT_SUMMARY_FIELDS):
        self.path = path
        self.summary_fields = tuple(summary_fields)
        self._episode_samples = defaultdict(list)
        self._episode_states = Counter()
        self._file = open(path, "a") if path is not None else None
        self.ticks = 0
        self.first_tick_at: Optional[float] = None
        self.last_tick_at: Optional[float] = None
        self.overruns = 0
        self.recent = deque(maxlen=RECENT_TICKS)  # (tick_interval_s, build_s, frame_age_s)

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def event(self, kind: str, **data) -> None:
        self._write({"event": kind, **data})

    def tick(self, record: dict, tick_at: float, build_s: float, frame_age_s: Optional[float],
             overrun: bool, in_summary: bool = False, reached_send: bool = False) -> Optional[float]:
        """Log one control step; returns the interval since the previous tick. in_summary adds it to
        the episode summary, reached_send also to the SEND_FIELDS."""
        interval = None if self.last_tick_at is None else tick_at - self.last_tick_at
        if self.first_tick_at is None:
            self.first_tick_at = tick_at
        self.last_tick_at = tick_at
        self.ticks += 1
        self.overruns += int(overrun)
        self.recent.append((interval, build_s, frame_age_s))
        row = {"tick": self.ticks - 1, "tick_interval_s": interval, "rate_hz": self.recent_rate(),
               "overall_rate_hz": self.overall_rate(), "build_s": build_s, "frame_age_s": frame_age_s,
               "overrun": overrun, **record}
        self._write(row)
        if in_summary:
            self._episode_states[row.get("control_state")] += 1
            for name in self.summary_fields:
                value = row.get(name)
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and (reached_send or name not in SEND_FIELDS)):
                    self._episode_samples[name].append(float(value))
        return interval

    def end_episode(self) -> dict:
        """Summary of each configured field and the control-state tick counts over the episode's
        ticks; starts the next episode's."""
        summary = {name: summarize(self._episode_samples[name]) for name in self.summary_fields}
        summary[CONTROL_STATE_TICKS] = {state: self._episode_states[state] for state in CONTROL_STATES}
        self._episode_samples.clear()
        self._episode_states.clear()
        return summary

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
