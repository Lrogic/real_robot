"""How close to stopped the arm is when the next policy goal arrives, real vs sim.

For each policy period: the mean joint speed over the period, |q(next tick) - q(tick)| /
interval, and the instantaneous joint speed at the end of the period, |qd(next tick)|.
Their ratio is about 1 for continuous motion and near 0 when the arm has reached its goal
and is waiting. Uses the robot state logged at each real tick
(vision/foundationpose_runs/run*/policy_log.jsonl) and the 2 ms sim substeps
(results/substeps.npz). Real periods are split into normal ones (next goal 35-70 ms later)
and ones where the next tick sent no goal.

  python compare_end_of_period.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2] / "vision" / "foundationpose_runs"
MIN_MEAN_RAD_S = 0.1
WRIST = slice(3, 6)


def ratios(mean_qd: np.ndarray, end_qd: np.ndarray, joints) -> np.ndarray:
  m = np.linalg.norm(mean_qd[:, joints], axis=1)
  e = np.linalg.norm(end_qd[:, joints], axis=1)
  ok = m > MIN_MEAN_RAD_S
  return e[ok] / m[ok]


def fmt(name: str, r_all: np.ndarray, r_wrist: np.ndarray) -> str:
  return (f"{name:<34} n={len(r_all):4d}  all joints: median {np.median(r_all):.2f} p25 {np.percentile(r_all, 25):.2f}"
          f"   wrist: median {np.median(r_wrist):.2f} p25 {np.percentile(r_wrist, 25):.2f}  frac<0.2 {np.mean(r_wrist < 0.2):.2f}")


def real_periods(log: Path):
  ticks = [r for r in map(json.loads, log.open()) if r.get("event") is None and r.get("phase") == "active"]
  out = {"normal": ([], []), "next tick sent nothing": ([], [])}
  for a, b in zip(ticks, ticks[1:]):
    if not a.get("sent") or b["tick"] != a["tick"] + 1:
      continue
    dt = b["mono"] - a["mono"]
    qa, qb = np.array(a["robot"]["arm_q_rad"]), np.array(b["robot"]["arm_q_rad"])
    key = "normal" if b.get("sent") and 0.035 <= dt <= 0.07 else "next tick sent nothing" if not b.get("sent") else None
    if key:
      out[key][0].append((qb - qa) / dt)
      out[key][1].append(b["robot"]["arm_qd_rad_s"])
  return {k: (np.array(m), np.array(e)) for k, (m, e) in out.items() if m}


def main() -> None:
  sim = np.load(HERE / "results" / "substeps.npz")
  n, sdt = int(sim["decimation"]), float(sim["sim_dt"])
  ms, es = [], []
  for e in range(3):
    r = sim[f"policy_{e}"]
    q, qd = r[:, 3:9], r[:, 9:15]
    k = len(r) // n
    qs = q[: k * n].reshape(k, n, 6)
    ends = qs[:, -1]
    mean = np.diff(np.concatenate([qs[:1, 0], ends]), axis=0) / (n * sdt)
    jump = np.abs(np.diff(np.concatenate([qs[:1, 0], ends]), axis=0)).max(axis=1) > 0.5
    ms.append(mean[~jump])
    es.append(qd[: k * n].reshape(k, n, 6)[:, -1][~jump])
  m, e = np.concatenate(ms), np.concatenate(es)
  print(fmt("sim (end-of-period / mean)", ratios(m, e, slice(0, 6)), ratios(m, e, WRIST)))

  for log in sorted(RUNS.glob("run*/policy_log.jsonl"), key=lambda p: int(p.parent.name[3:])):
    cfg = next((r["config"]["control"] for r in map(json.loads, log.open()) if r.get("event") == "start"), {})
    if cfg.get("mode") != "full":
      continue
    for key, (m, e) in real_periods(log).items():
      r_all, r_w = ratios(m, e, slice(0, 6)), ratios(m, e, WRIST)
      if len(r_all) >= 5:
        print(fmt(f"{log.parent.name} {cfg.get('ik_solver')} {key}", r_all, r_w))


if __name__ == "__main__":
  main()
