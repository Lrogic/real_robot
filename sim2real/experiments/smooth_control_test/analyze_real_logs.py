"""Command timing and action statistics of real-robot policy runs, next to the sim rollout.

Reads vision/foundationpose_runs/run*/policy_log.jsonl (active-phase ticks only) and, for
comparison, results/substeps.npz (record_substeps.py). Per run: the IK mode, how regular
the 20 Hz commands were, how many ticks did not send a goal and why, and how much the
arm actions saturate and change from one step to the next while the arm moves.

  python analyze_real_logs.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2] / "vision" / "foundationpose_runs"
MOVING_RAD_S = 0.1


def action_stats(a: np.ndarray, moving: np.ndarray) -> str:
  a = np.clip(a[:, :6], -1, 1)
  da = np.abs(np.diff(a, axis=0))
  mv = moving[1:] & moving[:-1]
  flips = ((np.sign(a[1:]) != np.sign(a[:-1])) & (np.abs(a[1:]) > 0.3) & (np.abs(a[:-1]) > 0.3))[mv]
  return (f"moving steps {mv.sum():4d}  sat {np.mean(np.abs(a[1:][mv]) > 0.95):.2f}  "
          f"|da| {da[mv].mean():.3f}  rot|da| {da[mv][:, 3:].mean():.3f}  flips {flips.mean():.3f}")


def main() -> None:
  for log in sorted(RUNS.glob("run*/policy_log.jsonl"), key=lambda p: int(p.parent.name[3:])):
    cfg, ticks = {}, []
    for line in log.open():
      r = json.loads(line)
      if r.get("event") == "start":
        cfg = r["config"]["control"]
      elif r.get("event") is None and r.get("phase") == "active":
        ticks.append(r)
    if not ticks:
      continue
    iv = np.array([t["tick_interval_s"] for t in ticks if t.get("tick_interval_s")])
    sent = np.array([bool(t.get("sent")) for t in ticks])
    holds = Counter(t.get("hold_reason") for t in ticks if not t.get("sent"))
    age = np.array([t["frame_age_s"] for t in ticks if t.get("frame_age_s") is not None])
    acts = np.array([t["action_clipped"] for t in ticks if t.get("action_clipped") is not None])
    qd = np.array([t["robot"]["arm_qd_rad_s"] for t in ticks if t.get("action_clipped") is not None])
    sim_ik = [t["sim_ik"] for t in ticks if t.get("sim_ik")]
    print(f"\n{log.parent.name}: ik_solver={cfg.get('ik_solver')} driver_goal_time={cfg.get('driver_goal_time_s')} "
          f"mode={cfg.get('mode')} ticks={len(ticks)}")
    print(f"  tick interval ms: median {1e3 * np.median(iv):.1f}  p90 {1e3 * np.percentile(iv, 90):.1f}  "
          f"p99 {1e3 * np.percentile(iv, 99):.1f}  >75ms {np.mean(iv > 0.075):.2f}")
    print(f"  goal sent on {sent.mean():.2f} of ticks; not sent: {dict(holds)}")
    print(f"  frame age ms: median {1e3 * np.median(age):.1f}  p90 {1e3 * np.percentile(age, 90):.1f}")
    if sim_ik:
      rates = [s["rate_hz"] for s in sim_ik if s.get("rate_hz")]
      print(f"  sim-IK loop: median rate {np.median(rates):.0f} Hz  late ticks {sum(s['late_ticks'] for s in sim_ik)}  "
            f"reused state {sum(s['reused_state'] for s in sim_ik)} of {sum(s['ticks'] for s in sim_ik)}  "
            f"dq clamped {sum(s['max_dq_clamped'] for s in sim_ik)}")
    if len(acts) > 2:
      print("  actions: " + action_stats(acts, np.abs(qd).max(axis=1) > MOVING_RAD_S))

  sim = np.load(HERE / "results" / "substeps.npz")
  n = int(sim["decimation"])
  for e in range(3):
    a = sim[f"policy_{e}_actions"]
    qd = np.abs(sim[f"policy_{e}"][:, 9:15]).reshape(-1, n, 6)[:, -1].max(axis=1)
    print(f"\nsim policy_{e}: actions: " + action_stats(a, qd[: len(a)] > MOVING_RAD_S))


if __name__ == "__main__":
  main()
