"""Fit models of the arm's height error ("sag") to ruler measurements and say which one holds.

The quantity modelled is   err = reported z - measured z   (mm): how much higher the arm's FK of
its measured joints puts the tool point than it really is. The candidate models:

  reach              a + b * reach                           (the line from the earlier runs)
  reach2             a + b * reach + c * reach^2
  reach+tilt         a + b * reach + c * x + d * y           (a table or base tilted against the arm)
  tilt               a + c * x + d * y
  compliance_j1      joint compliance, shoulder only
  compliance_j12     shoulder and elbow
  compliance_j123    shoulder, elbow and wrist pitch
  compliance_j12+tilt
  zero_j1, zero_j12, zero_j123  constant joint zero errors (no load dependence)

Joint compliance: each joint bends after its encoder by  delta_j = -c_j * tau_j,  tau_j the torque
the motor holds against gravity (the MuJoCo gravity torque of wxai_base_real_robot.xml, which the
driver's compensation effort matches to 0.03 Nm). The FK then misses  J_z . delta, so
err = sum_j c_j * J_zj * tau_j * 1000. c_j > 0 is physical (rad/Nm); its inverse is a stiffness.
Zero errors: the true angle is the reading plus o_j, so err = -sum_j o_j * J_zj * 1000.

Both depend on the pose only through the joints, so neither depends on the base direction. A tilt
does. Compliance and zero errors differ in how the error changes with height and tool pitch at a
fixed reach, which is what the sag_map grid varies.

Each results file (sag_map_*.json or hold_z_*.json) gets its own intercept, since the intercept
holds the offset between the arm's z = 0 and the table and the ruler's point against the tool
point; --shared-intercept uses one for all. Held-out error is reported leaving out one base
direction, one tool pitch, one height, and one stop at a time.

Run notes. sag_map_20261125_173202.json: the 45 deg stops (tool pitched halfway between level and
straight down) were hard to read with the ruler, and at 0.40 to 0.50 m reach they scatter 2 to 4 mm
about every model against about 1 mm elsewhere. Trust them less: compare with --exclude-pitch 45.
Its first reference visit (stop 0) also read 1.2 mm below the other four.

Env: vision or wxai_mjlab (needs mujoco and numpy), from any directory:

  python real_robot/sim2real/sag_test/analyze_sag.py real_robot/sim2real/sag_test/results/sag_map_*.json
  python real_robot/sim2real/sag_test/analyze_sag.py --include-hold-z results/...   (adds the hold_z runs)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REAL_ROBOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REAL_ROBOT / "sim2real" / "experiments"))


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARM_MODEL = REAL_ROBOT / "sim" / "mujoco" / "assets" / "robots" / "trossen_wxai" / "wxai_follower.xml"
GRAVITY_MODEL = (REAL_ROBOT / "sim" / "mujoco" / "assets" / "robots" / "trossen_wxai_base" / "wxai"
                 / "wxai_base_real_robot.xml")
TOOL_SITE = "ee_gripper_link"
RULER_SIGMA_MM = 0.5            # 1 mm resolution plus placement, for the design check

MODELS = {
    "reach": ["reach"],
    "reach2": ["reach", "reach_sq"],
    "reach+tilt": ["reach", "x", "y"],
    "tilt": ["x", "y"],
    "compliance_j1": ["comp1"],
    "compliance_j12": ["comp1", "comp2"],
    "compliance_j123": ["comp1", "comp2", "comp3"],
    "compliance_j12+tilt": ["comp1", "comp2", "x", "y"],
    "zero_j1": ["zero1"],
    "zero_j12": ["zero1", "zero2"],
    "zero_j123": ["zero1", "zero2", "zero3"],
}


class Kinematics:
    """Tool-point FK and Jacobian (driver tool frame) and gravity torques of the WXAI arm."""

    def __init__(self):
        import mujoco

        self.mj = mujoco
        self.arm = mujoco.MjModel.from_xml_path(str(ARM_MODEL))
        self.arm_data = mujoco.MjData(self.arm)
        self.site = self.arm.site(TOOL_SITE).id
        self.grav = mujoco.MjModel.from_xml_path(str(GRAVITY_MODEL))
        self.grav_data = mujoco.MjData(self.grav)

    def tool(self, q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position, rotation and the 3x6 translational Jacobian of the tool point."""
        m, d = self.arm, self.arm_data
        d.qpos[:] = 0
        d.qpos[:6] = q
        self.mj.mj_kinematics(m, d)
        self.mj.mj_comPos(m, d)
        jp = np.zeros((3, m.nv))
        self.mj.mj_jacSite(m, d, jp, None, self.site)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy(), jp[:, :6].copy()

    def gravity(self, q) -> np.ndarray:
        """Torque each joint's motor must hold against gravity (Nm), at rest."""
        m, d = self.grav, self.grav_data
        d.qpos[:] = 0
        d.qpos[:6] = q
        d.qvel[:] = 0
        self.mj.mj_forward(m, d)
        return d.qfrc_bias[:6].copy()


def features(kin: Kinematics, q) -> dict:
    p, R, J = kin.tool(q)
    tau = kin.gravity(q)
    f = {"x": p[0], "y": p[1], "z": p[2], "reach": float(np.hypot(p[0], p[1])),
         "pitch_deg": float(np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))),
         "yaw_deg": float(np.degrees(np.arctan2(p[1], p[0]))), "tau": tau, "Jz": J[2]}
    f["reach_sq"] = f["reach"] ** 2
    for j in (1, 2, 3):
        f[f"comp{j}"] = J[2, j] * tau[j] * 1000.0      # mm per (rad/Nm)
        f[f"zero{j}"] = -J[2, j] * 1000.0               # mm per rad
    return f


def load_samples(paths: list[Path], kin: Kinematics) -> list[dict]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        for s in payload.get("samples", []):
            if s.get("measured_mm") is None:
                continue
            q = np.array(s["arm_joint_positions_rad"])
            f = features(kin, q)
            reported = np.array(s["tool_m_rotvec_rad"])[:3] * 1000
            if abs(f["z"] * 1000 - reported[2]) > 0.5:
                print(f"warning: {path.name} stop {s['stop']}: model FK z {f['z'] * 1000:.2f} mm, driver "
                      f"{reported[2]:.2f} mm")
            rows.append({**f, "session": path.name, "stop": s["stop"], "kind": s.get("kind", "grid"),
                         "plan_yaw_deg": s.get("yaw_deg", 0.0), "plan_pitch_deg": s.get("pitch_deg", 90.0),
                         "plan_z_m": s.get("z_m"), "err": reported[2] - s["measured_mm"],
                         "measured_mm": s["measured_mm"]})
    return rows


def design_matrix(rows: list[dict], cols: list[str], sessions: list[str]) -> np.ndarray:
    inter = np.array([[r["session"] == s for s in sessions] for r in rows], dtype=float)
    feats = np.array([[r[c] for c in cols] for r in rows], dtype=float).reshape(len(rows), len(cols))
    return np.hstack([inter, feats])


def fit(rows, cols, sessions):
    X, y = design_matrix(rows, cols, sessions), np.array([r["err"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, y - X @ coef


def held_out(rows, cols, sessions, group) -> float | None:
    """RMS error on each group's stops, fit without that group. None if fewer than two groups."""
    keys = sorted({group(r) for r in rows})
    if len(keys) < 2:
        return None
    res = []
    for k in keys:
        train = [r for r in rows if group(r) != k]
        test = [r for r in rows if group(r) == k]
        if len({r["session"] for r in train}) < len({r["session"] for r in rows}):
            continue                                  # its session's intercept would be unknown
        coef, _ = fit(train, cols, sessions)
        Xt = design_matrix(test, cols, sessions)
        res.extend(np.array([r["err"] for r in test]) - Xt @ coef)
    return float(np.sqrt(np.mean(np.square(res)))) if res else None


GROUPS = {
    "direction": lambda r: round(r["plan_yaw_deg"] / 5) * 5,
    "pitch": lambda r: round(r["plan_pitch_deg"] / 5) * 5,
    "height": lambda r: None if r["plan_z_m"] is None else round(r["plan_z_m"] * 100),
    "stop": lambda r: (r["session"], r["stop"]),
}


def compare(rows: list[dict], sessions: list[str], verbose: bool = True) -> dict:
    out = {}
    if verbose:
        print(f"\n{'model':<22}{'params':>7}{'fit rms':>9}" + "".join(f"{'-' + g:>12}" for g in GROUPS)
              + "    (held-out rms, mm, leaving out one ...)")
    for name, cols in MODELS.items():
        coef, res = fit(rows, cols, sessions)
        cv = {g: held_out(rows, cols, sessions, fn) for g, fn in GROUPS.items()}
        out[name] = {"coef": dict(zip([f"intercept[{s}]" for s in sessions] + cols, coef.tolist())),
                     "fit_rms_mm": float(np.sqrt(np.mean(res ** 2))), "held_out_rms_mm": cv,
                     "params": len(cols) + len(sessions)}
        if verbose:
            print(f"{name:<22}{out[name]['params']:>7}{out[name]['fit_rms_mm']:>9.2f}"
                  + "".join(f"{'-' if v is None else f'{v:.2f}':>12}" for v in cv.values()))
    return out


def describe_physical(result: dict) -> None:
    for name in ("compliance_j1", "compliance_j12", "compliance_j123"):
        c = result[name]["coef"]
        parts = []
        for j, label in ((1, "shoulder"), (2, "elbow"), (3, "wrist pitch")):
            v = c.get(f"comp{j}")
            if v is not None:
                stiff = f"{np.radians(1) / v:.0f} Nm/deg" if v > 0 else "NEGATIVE (unphysical)"
                parts.append(f"{label} {v * 1000:.3f} mrad/Nm ({stiff})")
        print(f"  {name}: " + ", ".join(parts))
    for name in ("zero_j12", "zero_j123"):
        c = result[name]["coef"]
        print(f"  {name}: " + ", ".join(f"joint_{j} {np.degrees(c[f'zero{j}']):+.3f} deg"
                                        for j in (1, 2, 3) if f"zero{j}" in c))
    for name in ("reach", "reach+tilt", "tilt"):
        c = result[name]["coef"]
        parts = [f"{k} {v:+.1f} mm/m" for k, v in c.items() if k in ("reach", "x", "y")]
        if "x" in c:
            parts.append(f"tilt {np.degrees(np.arctan(np.hypot(c['x'], c['y']) / 1000)):.2f} deg")
        print(f"  {name}: " + ", ".join(parts))


def print_table(rows: list[dict], result: dict) -> None:
    best = min(result, key=lambda k: result[k]["held_out_rms_mm"]["direction"] or np.inf)
    cols = MODELS[best]
    sessions = sorted({r["session"] for r in rows})
    _, res = fit(rows, cols, sessions)
    print(f"\n  stop  kind       yaw  pitch  reach mm   z mm   tau1    tau2    tau3   err mm   resid ({best})")
    for r, e in zip(rows, res):
        print(f"  {r['stop']:4d}  {r['kind']:<9}{r['plan_yaw_deg']:5.0f}{r['plan_pitch_deg']:6.0f}"
              f"{r['reach'] * 1000:10.0f}{r['z'] * 1000:7.0f}{r['tau'][1]:7.2f}{r['tau'][2]:8.2f}"
              f"{r['tau'][3]:8.2f}{r['err']:9.1f}{e:+9.2f}")
    refs = [r for r in rows if r["kind"] == "reference"]
    if len(refs) >= 2:
        e = np.array([r["err"] for r in refs])
        print(f"\nReference pose, {len(refs)} visits: err {e.mean():.2f} mm, spread {np.ptp(e):.1f} mm, "
              f"std {e.std(ddof=1):.2f} mm (drift and ruler repeatability)")


def design_check(plan_q: list[np.ndarray], plan_meta: list[dict], kin: Kinematics, reference_files: list[Path],
                 trials: int = 20, seed: int = 0) -> None:
    """Can this grid tell the models apart with a ruler? Each single-cause hypothesis is fitted to the
    earlier hold_z run, used to predict the planned stops, ruler noise is added, and every model is
    fitted to that. A grid that discriminates puts the lowest held-out error on the true model."""
    ref_rows = load_samples(reference_files, kin)
    if not ref_rows:
        print("design check skipped: no earlier hold_z results to calibrate the hypotheses")
        return
    ref_sessions = sorted({r["session"] for r in ref_rows})
    plan_rows = []
    for q, meta in zip(plan_q, plan_meta):
        f = features(kin, q)
        plan_rows.append({**f, "session": "plan", "stop": meta["stop"], "kind": meta["kind"],
                          "plan_yaw_deg": meta["yaw_deg"], "plan_pitch_deg": meta["pitch_deg"],
                          "plan_z_m": meta["z_m"], "err": 0.0})
    truths = ["reach", "reach+tilt", "compliance_j1", "compliance_j12", "zero_j1"]
    fitted = ["reach", "reach2", "reach+tilt", "compliance_j1", "compliance_j12", "compliance_j123",
              "compliance_j12+tilt", "zero_j1", "zero_j123"]
    rng = np.random.default_rng(seed)
    print(f"\nDesign check: truths fitted to {', '.join(ref_sessions)}; ruler noise {RULER_SIGMA_MM} mm; "
          f"{trials} trials. Held-out rms (mm) leaving out one base direction:")
    print(f"{'truth':<17}{'range on grid':>14}  " + "".join(f"{m:>17}" for m in fitted))
    for truth in truths:
        cols = MODELS[truth]
        if truth == "compliance_j12":                 # the hold_z run cannot fit both; split the slope
            coef_s, _ = fit(ref_rows, ["comp1"], ref_sessions)
            c1 = coef_s[-1]
            coef = {"comp1": 0.6 * c1, "comp2": 0.6 * c1}
        elif truth == "reach+tilt":                   # the hold_z line plus a 0.3 deg tilt toward -y
            coef_s, _ = fit(ref_rows, ["reach"], ref_sessions)
            coef = {"reach": coef_s[-1], "x": 0.0, "y": float(np.tan(np.radians(0.3)) * 1000)}
        else:
            coef_v, _ = fit(ref_rows, cols, ref_sessions)
            coef = dict(zip(cols, coef_v[len(ref_sessions):]))
        clean = np.array([sum(coef[c] * r[c] for c in cols) for r in plan_rows])
        scores = {m: [] for m in fitted}
        for _ in range(trials):
            noisy = clean + rng.normal(0, RULER_SIGMA_MM, len(clean))
            trial = [{**r, "err": float(e)} for r, e in zip(plan_rows, noisy)]
            for m in fitted:
                scores[m].append(held_out(trial, MODELS[m], ["plan"], GROUPS["direction"]))
        means = {m: float(np.mean(v)) for m, v in scores.items()}
        best = min(means, key=means.get)
        print(f"{truth:<17}{np.ptp(clean):>11.1f} mm  "
              + "".join(f"{('*' if m == best else '') + f'{means[m]:.2f}':>17}" for m in fitted))
    print("(* lowest. Near " + f"{RULER_SIGMA_MM}" + " mm means the model explains everything but the ruler.)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, nargs="*", help="sag_map_*.json files (default: all in results/)")
    p.add_argument("--include-hold-z", action="store_true", help="also fit the hold_z_*.json runs")
    p.add_argument("--shared-intercept", action="store_true",
                   help="one intercept for all files (same session, ruler point and table)")
    p.add_argument("--exclude-pitch", type=float, nargs="*", default=[],
                   help="leave out stops planned at these tool pitches, deg (e.g. 45: hard to read)")
    p.add_argument("--output", type=Path, default=None, help="write the fits as JSON")
    args = p.parse_args()

    paths = list(args.results) or sorted(RESULTS_DIR.glob("sag_map_*.json"))
    if args.include_hold_z:
        paths += sorted(RESULTS_DIR.glob("hold_z_*.json"))
    if not paths:
        raise SystemExit("no results files")
    kin = Kinematics()
    rows = load_samples(paths, kin)
    if args.exclude_pitch:
        rows = [r for r in rows if not any(abs(r["plan_pitch_deg"] - p) < 1 for p in args.exclude_pitch)]
    if args.shared_intercept:
        for r in rows:
            r["session"] = "all"
    sessions = sorted({r["session"] for r in rows})
    print(f"{len(rows)} measured stops from {len(paths)} file(s)")
    if len(rows) < 6:
        raise SystemExit("too few measured stops to fit")
    result = compare(rows, sessions)
    print("\nFitted parameters:")
    describe_physical(result)
    print_table(rows, result)
    if args.output:
        args.output.write_text(json.dumps({"files": [str(p) for p in paths], "models": result}, indent=2))
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
