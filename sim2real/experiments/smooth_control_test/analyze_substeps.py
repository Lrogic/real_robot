"""Within-control-period motion profile from results/substeps.npz (record_substeps.py).

For every 50 ms policy step it takes the 25 substep TCP speeds and reports how much of the
period the arm spends nearly stopped, and plots example traces and the mean profile.

  python analyze_substeps.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
# Columns written by SubstepRecorder.record.
TCP, Q, QD, CTRL = slice(0, 3), slice(3, 9), slice(9, 15), slice(15, 21)
MOVING_MM_S = 10.0
RESET_JUMP_M = 0.01


def periods(rows: np.ndarray, n: int, dt: float):
  """(tcp speed, max |joint vel|) shaped (steps, n), steps that contain a reset dropped."""
  tcp = rows[:, TCP]
  step = np.linalg.norm(np.diff(tcp, axis=0, prepend=tcp[:1]), axis=1)
  speed = step / dt * 1000.0
  qd = np.abs(rows[:, QD])
  k = len(rows) // n
  speed = speed[: k * n].reshape(k, n)
  jumps = step[: k * n].reshape(k, n)
  qd = qd[: k * n].reshape(k, n, 6)
  keep = jumps.max(axis=1) < RESET_JUMP_M
  keep[0] = False
  return speed[keep], qd[keep]


def stats(speed: np.ndarray) -> dict:
  moving = speed.mean(axis=1) > MOVING_MM_S
  s = speed[moving]
  peak = s.max(axis=1, keepdims=True)
  disp = np.cumsum(s, axis=1)
  half = s.shape[1] // 2
  return {
    "periods": int(len(speed)),
    "moving_periods": int(moving.sum()),
    "median_mean_speed_mm_s": float(np.median(s.mean(axis=1))),
    "median_end_over_peak": float(np.median(s[:, -1] / peak[:, 0])),
    "median_min_over_peak": float(np.median(s.min(axis=1) / peak[:, 0])),
    "median_frac_time_below_20pct_peak": float(np.median((s < 0.2 * peak).mean(axis=1))),
    "median_frac_disp_first_half": float(np.median(disp[:, half - 1] / disp[:, -1])),
    "median_peak_substep_ms": float(np.median(s.argmax(axis=1)) * 2),
  }


def main() -> None:
  data = np.load(RES / "substeps.npz")
  n, dt = int(data["decimation"]), float(data["sim_dt"])
  t_ms = np.arange(n) * dt * 1000

  policy_speed, policy_qd = [], []
  for key in sorted(k for k in data.files if k.startswith("policy_") and not k.endswith("_actions")):
    s, q = periods(data[key], n, dt)
    policy_speed.append(s)
    policy_qd.append(q)
  ps = np.concatenate(policy_speed)
  pq = np.concatenate(policy_qd)

  print(f"{'run':<10} " + " ".join(f"{k}" for k in stats(ps)))
  for name, s in [("policy", ps)] + [(k, periods(data[k], n, dt)[0]) for k in ("const_x1", "const_x05", "const_z1")]:
    st = stats(s)
    print(f"{name:<10} " + " ".join(f"{v:.3g}" for v in st.values()))

  moving = ps.mean(axis=1) > MOVING_MM_S
  names = [f"joint_{i}" for i in range(6)]
  print("\nper-joint (policy, moving periods): median end-of-period |qd| / in-period peak |qd|")
  for j in range(6):
    qj = pq[moving, :, j]
    pk = qj.max(axis=1)
    ok = pk > 0.02
    print(f"  {names[j]}: {np.median(qj[ok, -1] / pk[ok]):.3f}  (periods with peak > 0.02 rad/s: {ok.sum()})")

  fig, ax = plt.subplots(3, 1, figsize=(11, 10))
  rows = data["policy_0"]
  tcp = rows[:, TCP]
  sp = np.linalg.norm(np.diff(tcp, axis=0, prepend=tcp[:1]), axis=1) / dt * 1000
  t = np.arange(len(sp)) * dt
  i0 = int(np.argmax(np.convolve(sp, np.ones(500), "valid")))
  w = slice(i0, i0 + 500)
  ax[0].plot(t[w], sp[w], lw=1)
  for b in np.arange(np.ceil(t[w][0] / (n * dt)), t[w][-1] / (n * dt)) * n * dt:
    ax[0].axvline(b, color="gray", lw=0.5, ls=":")
  ax[0].set(title="Policy rollout, TCP speed per 2 ms substep (dotted = new 20 Hz action)",
            xlabel="time [s]", ylabel="TCP speed [mm/s]")

  for j in range(6):
    ax[1].plot(t[w], rows[w, QD][:, j], lw=1, label=names[j])
  ax[1].legend(ncol=6, fontsize=8)
  ax[1].set(title="Same window, arm joint velocities", xlabel="time [s]", ylabel="qd [rad/s]")

  norm = ps[moving] / ps[moving].max(axis=1, keepdims=True)
  ax[2].plot(t_ms, np.median(norm, axis=0), "k", lw=2, label="policy median")
  ax[2].fill_between(t_ms, *np.percentile(norm, [10, 90], axis=0), color="k", alpha=0.15, label="policy 10-90%")
  for k in ("const_x1", "const_z1"):
    s = periods(data[k], n, dt)[0][5:]
    ax[2].plot(t_ms, np.median(s / s.max(axis=1, keepdims=True), axis=0), lw=1.5, label=f"{k} (steady state)")
  ax[2].set(title="TCP speed within one 50 ms control period, normalized to the period's peak",
            xlabel="time since action [ms]", ylabel="speed / peak", ylim=(0, 1.05))
  ax[2].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(RES / "substep_profile.png", dpi=120)

  fig, ax = plt.subplots(figsize=(11, 3.5))
  for k in ("const_x1", "const_z1"):
    r = data[k]
    s = np.linalg.norm(np.diff(r[:, TCP], axis=0, prepend=r[:1, TCP]), axis=1) / dt * 1000
    ax.plot(np.arange(len(s))[:300] * dt, s[:300], label=k)
  ax.set(title="Constant action, TCP speed per substep", xlabel="time [s]", ylabel="mm/s")
  ax.legend()
  fig.tight_layout()
  fig.savefig(RES / "constant_action.png", dpi=120)
  print(f"\nplots: {RES / 'substep_profile.png'}, {RES / 'constant_action.png'}")


if __name__ == "__main__":
  main()
