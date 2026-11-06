"""Compare hardware trials with the simulated reference of the same plan. Offline.

Both trajectories are re-evaluated from their joint positions with the training-model FK
(one TCP definition, robot base frame) and put on one elapsed-time axis whose zero is
command onset. No time shift is applied, so hardware latency stays visible. Metrics use
only the overlap of both valid intervals, sampled on the sim's 2 ms grid (hardware
interpolated: linear position, normalized-linear quaternion between fresh samples).

Orientation differences are angles of relative rotations; displacements along the commanded
direction use base-frame projections (translation) or the base-frame rotation vector of
R(t) R(0)^T (rotation). FK agreement says nothing about link flex or calibration.

Launch from the wxai_mjlab environment:

  python compare.py results/hardware/single_tra_z+__cart_joint_gt0.05__*.json
  python compare.py results/hardware/scheduled_tra_x+__*.json --sim results/sim/scheduled_tra_x+__sim__....json
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from settings import RESULTS_DIR, load_settings
from common.poses import matrix_to_rotvec
from geometry import pose_from_json, quat_wxyz_to_matrix, rotation_angle
from records import load_result, save_result, to_jsonable
from timing import interval_stats


class Trajectory:
    def __init__(self, result: dict, fk):
        s = result["samples"]
        self.result = result
        self.t = np.asarray(s["t_s"], dtype=np.float64)
        self.q = np.asarray(s["arm_q_rad"], dtype=np.float64)
        self.qd = np.asarray(s["arm_qd_rad_s"], dtype=np.float64)
        keep = self.t >= 0.0
        if result["source"] == "hardware":
            first_neg = np.flatnonzero(~keep)
            if first_neg.size:
                keep[first_neg[-1]] = True          # the pre-onset sample is the state at onset
        self.t, self.q, self.qd = self.t[keep], self.q[keep], self.qd[keep]
        Ts = [fk.tcp(q) for q in self.q]
        self.p = np.array([T[:3, 3] for T in Ts])
        self.R = np.array([T[:3, :3] for T in Ts])
        vel = [fk.tcp_velocity(q, qd) for q, qd in zip(self.q, self.qd)]
        self.v = np.array([np.linalg.norm(v) for v, _ in vel])
        self.w = np.array([np.linalg.norm(w) for _, w in vel])
        self.label = result["condition"]["tag"] if result["source"] == "hardware" else "sim"

    def at(self, t):
        """Position and rotation at times t (interpolated between samples)."""
        t = np.clip(np.asarray(t, dtype=np.float64), self.t[0], self.t[-1])
        p = np.stack([np.interp(t, self.t, self.p[:, i]) for i in range(3)], axis=1)
        idx = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1)
        nxt = np.minimum(idx + 1, len(self.t) - 1)
        span = np.where(nxt > idx, self.t[nxt] - self.t[idx], 1.0)
        a = np.clip((t - self.t[idx]) / span, 0.0, 1.0)
        R = []
        for i, j, f in zip(idx, nxt, a):
            dR = self.R[i].T @ self.R[j]
            R.append(self.R[i] @ _exp(matrix_to_rotvec(dR) * f))
        return p, np.array(R)


def _exp(r):
    from common.poses import rotvec_to_matrix
    return rotvec_to_matrix(r)


def commanded_axis(plan: dict) -> tuple[str, np.ndarray]:
    m = plan.get("motion", {})
    if m.get("kind") in ("translation", "rotation"):
        return m["kind"], np.asarray(m["direction"], dtype=np.float64)
    gt, gr = np.asarray(m["goal_translation_m"]), np.asarray(m["goal_rotvec_rad"])
    if np.linalg.norm(gt) >= 1e-9:
        return "translation", gt / np.linalg.norm(gt)
    return "rotation", gr / np.linalg.norm(gr)


def projection(kind: str, u, p, R, p0, R0) -> np.ndarray:
    if kind == "translation":
        return (p - p0) @ u
    return np.array([matrix_to_rotvec(Ri @ R0.T) @ u for Ri in R])


def targets_over_time(result: dict) -> tuple[np.ndarray, list]:
    rows = [(o["actual_t_s"], pose_from_json(o["target"])) for o in result["outer_updates"] if o["new_target"]]
    return np.array([r[0] for r in rows]), [r[1] for r in rows]


def trajectory_metrics(tr: Trajectory, plan: dict, kind: str, u) -> dict:
    p0, R0 = tr.p[0], tr.R[0]
    proj = projection(kind, u, tr.p, tr.R, p0, R0)
    m = {"duration_s": float(tr.t[-1]), "peak_joint_speed_rad_s": float(np.max(np.abs(tr.qd))),
         "peak_tcp_linear_speed_m_s": float(tr.v.max()), "peak_tcp_angular_speed_rad_s": float(tr.w.max())}
    if tr.t[-1] >= 0.05:
        p50, R50 = tr.at([0.05])
        m["displacement_after_50ms"] = float(projection(kind, u, p50, R50, p0, R0)[0])
        m["displacement_after_50ms_norm"] = (float(np.linalg.norm(p50[0] - p0)) if kind == "translation"
                                             else rotation_angle(R0, R50[0]))
    t_tg, T_tg = targets_over_time(tr.result)
    if T_tg:
        tproj = projection(kind, u, np.array([T[:3, 3] for T in T_tg]), np.array([T[:3, :3] for T in T_tg]), p0, R0)
        peak_cmd = float(np.max(tproj)) if plan["test"] != "relative" or tproj.max() > 0 else float(np.max(tproj))
        m["peak_commanded"] = peak_cmd
        m["peak_reached"] = float(np.max(proj))
        m["overshoot"] = float(max(0.0, np.max(proj) - peak_cmd))
        m["overshoot_fraction"] = m["overshoot"] / abs(peak_cmd) if abs(peak_cmd) > 1e-12 else None
        T_last = T_tg[-1]
        m["final_error_to_last_target_m"] = float(np.linalg.norm(tr.p[-1] - T_last[:3, 3]))
        m["final_error_to_last_target_rad"] = rotation_angle(T_last[:3, :3], tr.R[-1])
        if plan["test"] == "single":
            step = plan["motion"]["step"]
            band = max(0.0005, 0.05 * step) if kind == "translation" else max(0.005, 0.05 * step)
            err = (np.linalg.norm(tr.p - T_last[:3, 3], axis=1) if kind == "translation"
                   else np.array([rotation_angle(T_last[:3, :3], Ri) for Ri in tr.R]))
            outside = np.flatnonzero(err > band)
            m["settling_band"] = band
            m["settling_time_s"] = (0.0 if outside.size == 0 else
                                    float(tr.t[outside[-1] + 1]) if outside[-1] + 1 < len(tr.t) else None)
    if plan["test"] == "relative":
        goal = pose_from_json(plan["relative"]["goal"])
        m["final_error_to_goal_m"] = float(np.linalg.norm(tr.p[-1] - goal[:3, 3]))
        m["final_error_to_goal_rad"] = rotation_angle(goal[:3, :3], tr.R[-1])
        reached = [o["actual_t_s"] for o in tr.result["outer_updates"] if o.get("info", {}).get("reached")]
        m["goal_tolerance_reached_t_s"] = reached[0] if reached else None
        m["relative_targets_sent"] = len(T_tg)
    m["stop_reason"] = tr.result.get("stop_reason")
    return m


def compare_pair(sim: Trajectory, hw: Trajectory, plan: dict) -> dict:
    kind, u = commanded_axis(plan)
    lo, hi = max(sim.t[0], hw.t[0], 0.0), min(sim.t[-1], hw.t[-1])
    grid = sim.t[(sim.t >= lo) & (sim.t <= hi)]
    ps, Rs = sim.at(grid)
    ph, Rh = hw.at(grid)
    dpos = np.linalg.norm(ph - ps, axis=1)
    dang = np.array([rotation_angle(a, b) for a, b in zip(Rs, Rh)])
    out = {
        "overlap_s": [float(lo), float(hi)], "grid_points": int(grid.size), "axis_kind": kind, "axis": u.tolist(),
        "position_discrepancy_m": {"mean": float(dpos.mean()), "max": float(dpos.max()),
                                   "at_end": float(dpos[-1])},
        "orientation_discrepancy_rad": {"mean": float(dang.mean()), "max": float(dang.max()),
                                        "at_end": float(dang[-1])},
        "sim": trajectory_metrics(sim, plan, kind, u),
        "hardware": trajectory_metrics(hw, plan, kind, u),
        "hardware_timing": hw.result.get("timing_summary"),
        "hardware_driver_tcp_vs_fk": hw.result.get("driver_tcp_vs_training_fk"),
        "start_offset_hw_minus_sim_m": (hw.p[0] - sim.p[0]).tolist(),
        "start_rotation_hw_vs_sim_rad": rotation_angle(sim.R[0], hw.R[0]),
    }
    if plan["test"] == "scheduled":
        out["scheduled_targets_identical"] = scheduled_consistency(sim.result, hw.result)
    return out, grid, dpos, dang


def scheduled_consistency(a: dict, b: dict) -> dict:
    ta = {o["index"]: pose_from_json(o["target"]) for o in a["outer_updates"] if o["new_target"]}
    tb = {o["index"]: pose_from_json(o["target"]) for o in b["outer_updates"] if o["new_target"]}
    common = sorted(set(ta) & set(tb))
    diff = max((float(np.abs(ta[k] - tb[k]).max()) for k in common), default=0.0)
    return {"common_indices": len(common), "only_sim": sorted(set(ta) - set(tb)),
            "only_hardware": sorted(set(tb) - set(ta)), "max_abs_difference": diff}


def plot(sim: Trajectory, hws: list, plan: dict, path: Path, discrepancies: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kind, u = commanded_axis(plan)
    scale, unit = (1000.0, "mm") if kind == "translation" else (np.degrees(1.0), "deg")
    fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    for tr, style in [(sim, dict(color="k", lw=2))] + [(h, dict(lw=1.2)) for h in hws]:
        proj = projection(kind, u, tr.p, tr.R, tr.p[0], tr.R[0]) * scale
        line, = ax[0].plot(tr.t, proj, label=tr.label, **style)
        t_tg, T_tg = targets_over_time(tr.result)
        if T_tg:
            tp = projection(kind, u, np.array([T[:3, 3] for T in T_tg]), np.array([T[:3, :3] for T in T_tg]),
                            tr.p[0], tr.R[0]) * scale
            ax[0].step(np.append(t_tg, tr.t[-1]), np.append(tp, tp[-1]), where="post", ls="--",
                       color=line.get_color(), alpha=0.6, lw=0.9)
        off = tr.p - tr.p[0] - np.outer((tr.p - tr.p[0]) @ u, u) if kind == "translation" else tr.p - tr.p[0]
        ax[1].plot(tr.t, np.linalg.norm(off, axis=1) * 1000.0, color=line.get_color(), **{k: v for k, v in style.items() if k != "color"})
        ax[3].plot(tr.t, tr.v * 1000.0 if kind == "translation" else np.degrees(tr.w), color=line.get_color(),
                   **{k: v for k, v in style.items() if k != "color"})
    for (grid, dpos, dang), h in zip(discrepancies, hws):
        ax[2].plot(grid, dpos * 1000.0, label=f"{h.label} position (mm)")
        ax[2].plot(grid, np.degrees(dang), ls=":", label=f"{h.label} orientation (deg)")
    for x in np.arange(0.0, max(t.t[-1] for t in [sim] + hws), plan["outer_period_s"]):
        ax[0].axvline(x, color="0.9", lw=0.5, zorder=0)
    ax[0].set_ylabel(f"along command ({unit})\n(dashed: commanded)")
    ax[1].set_ylabel("off-axis translation (mm)" if kind == "translation" else "TCP translation (mm)")
    ax[2].set_ylabel("hardware - sim")
    ax[3].set_ylabel("TCP speed (mm/s)" if kind == "translation" else "TCP angular speed (deg/s)")
    ax[3].set_xlabel("time since command onset (s)")
    ax[0].legend(fontsize=8)
    ax[2].legend(fontsize=7)
    fig.suptitle(f"{plan['name']} (plan {plan['sha256'][:10]})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def find_sim(plan_sha: str) -> Path:
    candidates = []
    for path in sorted((RESULTS_DIR / "sim").glob("*.json")):
        try:
            if load_result(path)["plan_sha256"] == plan_sha:
                candidates.append(path)
        except Exception:
            continue
    if not candidates:
        raise SystemExit(f"no sim result for plan {plan_sha[:12]}; run sim_reference.py on the plan first")
    return candidates[-1]


def main() -> None:
    from fk import TrainingFK

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hardware", type=Path, nargs="+", help="hardware result JSON(s) of one plan")
    parser.add_argument("--sim", type=Path, default=None, help="default: newest sim result of the same plan")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "compare")
    args = parser.parse_args()
    cfg = load_settings(args.config)
    fk = TrainingFK(cfg["training"]["scene"])
    hw_results = [load_result(p) for p in args.hardware]
    shas = {r["plan_sha256"] for r in hw_results}
    if len(shas) != 1:
        raise SystemExit("hardware results come from different plans; compare one plan at a time")
    sim_path = args.sim or find_sim(next(iter(shas)))
    sim_result = load_result(sim_path)
    if sim_result["plan_sha256"] not in shas:
        raise SystemExit(f"{sim_path} was run from a different plan than the hardware results")
    plan = sim_result["plan"]
    sim = Trajectory(sim_result, fk)
    hws = [Trajectory(r, fk) for r in hw_results]
    report = {"plan": plan["name"], "plan_sha256": plan["sha256"], "sim_file": str(sim_path), "pairs": []}
    curves = []
    rows = []
    for path, hw in zip(args.hardware, hws):
        metrics, grid, dpos, dang = compare_pair(sim, hw, plan)
        curves.append((grid, dpos, dang))
        report["pairs"].append({"hardware_file": str(path), "condition": hw.result["condition"], **metrics})
        hm, sm = metrics["hardware"], metrics["sim"]
        rows.append({"condition": hw.label, "stop": hm["stop_reason"],
                     "pos_disc_mean_mm": metrics["position_discrepancy_m"]["mean"] * 1e3,
                     "pos_disc_max_mm": metrics["position_discrepancy_m"]["max"] * 1e3,
                     "ori_disc_mean_deg": np.degrees(metrics["orientation_discrepancy_rad"]["mean"]),
                     "ori_disc_max_deg": np.degrees(metrics["orientation_discrepancy_rad"]["max"]),
                     "hw_50ms": hm.get("displacement_after_50ms"), "sim_50ms": sm.get("displacement_after_50ms"),
                     "hw_overshoot": hm.get("overshoot"), "sim_overshoot": sm.get("overshoot"),
                     "hw_settle_s": hm.get("settling_time_s"), "sim_settle_s": sm.get("settling_time_s"),
                     "hw_peak_qd": hm["peak_joint_speed_rad_s"], "sim_peak_qd": sm["peak_joint_speed_rad_s"],
                     "hw_goal_err_mm": (hm.get("final_error_to_goal_m") or 0) * 1e3 if plan["test"] == "relative" else None,
                     "outer_hz": (metrics["hardware_timing"] or {}).get("outer", {}).get("achieved_hz")})
    stem = f"{plan['name']}__{len(hws)}hw"
    plot(sim, hws, plan, args.out / f"{stem}.png", curves)
    save_result(to_jsonable(report), args.out / f"{stem}.json")
    with open(args.out / f"{stem}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    kind, _ = commanded_axis(plan)
    unit = "m" if kind == "translation" else "rad"
    print(f"{plan['name']} vs {sim_path.name}  (displacements in {unit} along the commanded axis)")
    for r in rows:
        print("  " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"Saved {args.out / stem}.png/.json/.csv")


if __name__ == "__main__":
    main()
