"""Summarize joint-step results; with two files, the second next to the first. Run from this directory.

    python analyze.py results/hardware_<stamp>.npz [results/sim_training_model.npz] [--pose ready]

Per joint, test, goal time and amplitude (both signs pooled, moved distance along the command):
  step / ramp / step_once   moved after 50 ms and at the end, as a fraction of the amplitude, and the
                time to cover 63% of the amplitude (- when it never does)
  relative      mean distance moved per 50 ms tick as a fraction of the amplitude
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KIND_ORDER = ("step", "ramp", "relative", "step_once", "relative_once")


class Session:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data = np.load(self.path)
        self.meta = json.loads(self.path.with_suffix(".json").read_text())
        self.trials = {t["name"]: t for t in self.meta["trials"]}
        self.period = float(self.meta["config"]["policy_period_s"])

    def metrics(self, name: str) -> dict:
        tr = self.trials[name]
        j, a = tr["joint"], tr["amplitude"]
        t, q = self.data[f"{name}/t"], self.data[f"{name}/q"][:, j]
        q0 = float(self.data[f"{name}/q_start"][j])
        moved = (q - q0) / a                       # fraction of the amplitude, along the command
        at = lambda s: float(np.interp(s, t, moved))
        if tr["kind"].startswith("relative"):
            ticks = tr["ticks"]
            per_tick = [at((k + 1) * self.period) - at(k * self.period) for k in range(ticks)]
            return {"per_tick": float(np.mean(per_tick)), "first_tick": per_tick[0], "last_tick": per_tick[-1]}
        final = float(np.mean(moved[t > t[-1] - 0.1]))
        reached = np.flatnonzero(moved >= 0.63)
        return {"at_50ms": at(self.period), "at_100ms": at(2 * self.period), "final": final,
                "t63_ms": float(t[reached[0]] * 1e3) if len(reached) else float("nan"),
                "peak": float(np.max(moved)), "final_error_mrad": float((1 - final) * abs(a) * 1e3)}


def pooled(session: Session, pose, joint, kind, goal_time, amplitude) -> dict | None:
    rows = [session.metrics(n) for n, t in session.trials.items()
            if t["pose"] == pose and t["joint"] == joint and t["kind"] == kind
            and abs(t["goal_time"] - goal_time) < 1e-9 and abs(abs(t["amplitude"]) - amplitude) < 1e-9]
    if not rows:
        return None
    return {k: float(np.nanmean([r[k] for r in rows])) if not all(np.isnan(r[k]) for r in rows) else float("nan")
            for k in rows[0]}


def fmt(m: dict | None, kind: str) -> str:
    if m is None:
        return " " * 26
    if kind.startswith("relative"):
        return f"{m['per_tick']:5.2f} ({m['first_tick']:4.2f} > {m['last_tick']:4.2f})  "
    t63 = f"{m['t63_ms']:4.0f}" if not np.isnan(m["t63_ms"]) else "   -"
    return f"{m['at_50ms']:5.2f} {m['final']:5.2f} {t63} ms {m['peak']:5.2f} "


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--pose", default=None)
    args = parser.parse_args()
    sessions = [Session(f) for f in args.files]
    first = sessions[0]
    poses = [args.pose] if args.pose else list(dict.fromkeys(t["pose"] for t in first.trials.values()))
    keys = sorted({(t["joint"], t["kind"], t["goal_time"], abs(t["amplitude"])) for t in first.trials.values()},
                  key=lambda k: (k[0], KIND_ORDER.index(k[1]), k[2], k[3]))
    print("columns per file: step/ramp = moved@50ms  final  t63  peak (fractions of the amplitude); "
          "relative = mean per tick (first > last tick)")
    print("files: " + "  |  ".join(s.path.name for s in sessions))
    for pose in poses:
        print(f"\n== pose {pose}")
        last = None
        for joint, kind, goal_time, amplitude in keys:
            if (joint, kind, goal_time) != last:
                print(f" joint_{joint} {kind} goal_time {goal_time * 1e3:.0f} ms")
                last = (joint, kind, goal_time)
            cells = [fmt(pooled(s, pose, joint, kind, goal_time, amplitude), kind) for s in sessions]
            print(f"   {amplitude * 1e3:4.0f} mrad   " + " | ".join(cells))


if __name__ == "__main__":
    main()
