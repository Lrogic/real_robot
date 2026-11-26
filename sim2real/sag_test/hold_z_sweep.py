"""Hold a constant commanded height under position control while stepping out along +x, and
measure how high the fingertip really is at each stop.

The manipulator reports its fingertip higher than it is by about 30 mm per metre of reach
(docs/arm_joint_offsets.md, section 1, hand-guided in gravity compensation; the first run of
this script, results/hold_z_20261116_133146.json, found +29 mm/m under position control). In that
run the driver's Cartesian command (its own IK) left the joints settled where FK read up to 5 mm
above the goal, while the measured height stayed flat. This version sends the joint targets
itself, so the error of each joint against its target is known exactly, and logs the driver's
compensation efforts to see whether position mode adds a gravity feedforward.

The arm's own readings cannot show the real height, and neither can the camera (the calibration
absorbed the error). Measure the real height from the table at each stop with a ruler or
calipers and type it in. The same point on the gripper every time (for example the bottom of the
closed fingers).

Joint targets (--command joints, the default): IK of each stop's goal pose on the WXAI MuJoCo model
(sim/mujoco/assets/robots/trossen_wxai/wxai_follower.xml, site ee_gripper_link: the driver's tool
frame for wxai_v0_base; the manipulator is the same arm and end effector), solved to convergence
from the previous stop's solution. At connection the script checks the model's FK against the
driver's tool pose at the current joints and stops if they disagree. --command cartesian sends
the goal pose to the driver's IK instead (the first run's behaviour).

Sequence (--execute):
  1. Connect to the manipulator (the calibration config's robot section). The arm goes into
     gravity compensation: guide it near the start pose, tool roughly pointing down.
  2. Enter: position mode at the current joints, the gripper opened to --gripper-gap-mm between
     the fingertips (so a ruler fits between them at the tool point), IK for every stop, then a
     joint-space move to the start pose.
  3. Moves along +x, --step-m at a time, to start + --distance-m (and back with --back). The
     commanded orientation and z are constant. At each stop the script waits --settle-s, averages
     the readings for --average-s and asks for the measured height (Enter skips it).
  4. Back to the start pose, then gravity compensation, then idle: SUPPORT THE ARM when asked.
Ctrl-C at any point also ends in gravity compensation and idle.

Launch from the vision environment (it has both mujoco and trossen_arm), from any directory:

  /opt/conda-envs/vision/bin/python real_robot/sim2real/sag_test/hold_z_sweep.py              # dry run
  /opt/conda-envs/vision/bin/python real_robot/sim2real/sag_test/hold_z_sweep.py --execute
  ... --x0 0.25 --distance-m 0.18 --step-m 0.045 --z 0.10 --back --execute

Results go to sim2real/sag_test/results/hold_z_<time>.json, rewritten after every stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REAL_ROBOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REAL_ROBOT / "sim2real" / "experiments"
sys.path.insert(0, str(REAL_ROBOT))
sys.path.insert(0, str(REAL_ROBOT / "vision"))
sys.path.insert(0, str(EXPERIMENTS))

from calibration.board import DEFAULT_CONFIG as CALIBRATION_CONFIG  # noqa: E402
from calibration.board import load_config as load_calibration_config  # noqa: E402
from common.hardware import end_effector_record, hold_arm_with_gravity_compensation, release_to_idle, robot_identity  # noqa: E402
from common.io import write_json  # noqa: E402
from common.poses import matrix_to_rotvec, rotation_angle_deg, rotvec_to_matrix  # noqa: E402
from common.prompts import Quit, ask  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARM_MODEL = REAL_ROBOT / "sim" / "mujoco" / "assets" / "robots" / "trossen_wxai" / "wxai_follower.xml"
TOOL_SITE = "ee_gripper_link"
TOOL_DOWN_ROTVEC = (0.0, np.pi / 2, 0.0)    # tool x axis (through the fingers) along base -z
DRY_RUN_SEED = (0.0, 1.48, 1.18, -1.27, 0.0, 0.0)  # the first run's joints at the start pose, to seed the dry-run IK
GRIPPER_M_PER_GAP_MM = 0.0005               # driver gripper coordinate is half the jaw gap (gripper_width results)
MAX_REACH_M = 0.60                           # refuse plans further out than this
MIN_Z_M = 0.03                               # refuse plans closer to the table than this
MAX_FK_MISMATCH = (0.05, 0.5)                # deg, mm: model FK against the driver's tool pose
MAX_STEP_JOINT_RAD = 0.6                     # refuse an IK plan with a larger joint change between stops
REFERENCE_SLOPE_MM_PER_M = 32.0              # gravity-compensation touches, arm_joint_offsets.md section 1
REFERENCE_INTERCEPT_MM = -6.4


class ArmModel:
    """FK and IK of the WXAI arm joints to the driver's tool frame, on the MuJoCo model."""

    def __init__(self, path: Path = ARM_MODEL, site: str = TOOL_SITE):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site(site).id
        self.lo, self.hi = self.model.jnt_range[:6, 0].copy(), self.model.jnt_range[:6, 1].copy()

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        d.qpos[:] = 0
        d.qpos[:6] = q
        self.mj.mj_kinematics(self.model, d)
        self.mj.mj_comPos(self.model, d)                 # mj_jacSite needs it
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, p_goal, R_goal, q_seed, iterations: int = 300, damping: float = 1e-4) -> np.ndarray:
        """Damped least squares to convergence (0.001 mm, 0.0001 rad) within the joint ranges."""
        q = np.clip(np.asarray(q_seed, dtype=float), self.lo, self.hi)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iterations):
            p, R = self.fk(q)
            err = np.concatenate([p_goal - p, matrix_to_rotvec(R_goal @ R.T)])
            if np.linalg.norm(err[:3]) < 1e-6 and np.linalg.norm(err[3:]) < 1e-4:
                return q
            self.mj.mj_jacSite(self.model, self.data, jp, jr, self.site)
            J = np.vstack([jp[:, :6], jr[:, :6]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
            dq *= min(1.0, 0.2 / max(np.abs(dq).max(), 1e-12))     # at most 0.2 rad per iteration
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.fk(q)
        raise RuntimeError(f"IK did not converge: {np.linalg.norm(p_goal - p) * 1000:.2f} mm, "
                           f"{rotation_angle_deg(R_goal @ R.T):.3f} deg left")


def plan_targets(args) -> list[dict]:
    n = int(round(args.distance_m / args.step_m))
    xs = [args.x0 + i * args.distance_m / n for i in range(n + 1)]
    stops = [{"leg": "out", "x_m": x} for x in xs]
    if args.back:
        stops += [{"leg": "back", "x_m": x} for x in reversed(xs[:-1])]
    for s in stops:
        s.update(y_m=args.y, z_m=args.z, reach_m=float(np.hypot(s["x_m"], args.y)))
    return stops


def check_plan(stops: list[dict]) -> None:
    far = max(s["reach_m"] for s in stops)
    if far > MAX_REACH_M:
        raise SystemExit(f"plan reaches {far:.3f} m, more than {MAX_REACH_M} m")
    if stops[0]["z_m"] < MIN_Z_M:
        raise SystemExit(f"z {stops[0]['z_m']:.3f} m is closer to the table than {MIN_Z_M} m")


def solve_stops(arm: ArmModel, stops: list[dict], R_goal: np.ndarray, q_seed) -> list[np.ndarray]:
    """Joint targets for every stop, each seeded from the previous one so the branch stays the same."""
    targets, q = [], np.asarray(q_seed, dtype=float)
    for i, s in enumerate(stops):
        try:
            q_new = arm.ik(np.array([s["x_m"], s["y_m"], s["z_m"]]), R_goal, q)
        except RuntimeError as error:
            raise SystemExit(f"stop {i} (x {s['x_m']:.3f} m) is out of reach at this z and orientation: {error}. "
                             "The joints are clipped to their ranges (joint_3 reaches -90 deg first with the "
                             "tool straight down); shorten --distance-m or raise --z.") from None
        if i > 0 and np.abs(q_new - q).max() > MAX_STEP_JOINT_RAD:
            raise SystemExit(f"IK jumps {np.degrees(np.abs(q_new - q).max()):.0f} deg between stops {i - 1} and {i}")
        targets.append(q_new)
        q = q_new
    return targets


def read(driver) -> dict:
    out = driver.get_robot_output()
    arm = out.joint.arm
    return {"q": np.array(arm.positions, dtype=float),
            "effort": np.array(arm.efforts, dtype=float),
            "compensation_effort": np.array(arm.compensation_efforts, dtype=float),
            "external_effort": np.array(arm.external_efforts, dtype=float),
            "tool": np.array(out.cartesian.positions, dtype=float)}


def average(driver, seconds: float) -> dict:
    rows = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rows.append(read(driver))
        time.sleep(0.01)
    tool = np.array([r["tool"] for r in rows])
    mean_tool = tool.mean(axis=0)
    mean_tool[3:] = tool[len(tool) // 2, 3:]          # rotvecs do not average; take the middle one

    def mean(key):
        return np.mean([r[key] for r in rows], axis=0).tolist()

    return {"readings": len(rows),
            "arm_joint_positions_rad": mean("q"),
            "arm_joint_efforts": mean("effort"),
            "arm_compensation_efforts": mean("compensation_effort"),
            "arm_external_efforts": mean("external_effort"),
            "tool_m_rotvec_rad": mean_tool.tolist(),
            "tool_moved_mm": float(np.ptp(tool[:, :3], axis=0).max() * 1000)}


def ask_height(prompt: str) -> float | None:
    while True:
        text = ask(prompt).strip()
        if text == "":
            return None
        if text.lower() in {"q", "quit"}:
            raise Quit()
        try:
            return float(text)
        except ValueError:
            print("Type the height in mm, Enter to skip, or q to stop.")


def fit(samples: list[dict], key: str) -> dict | None:
    pts = [(s["reach_m"], s[key]) for s in samples if s.get(key) is not None]
    if len(pts) < 3:
        return None
    r, z = np.array(pts).T
    slope, intercept = np.polyfit(r, z, 1)
    res = z - (slope * r + intercept)
    return {"slope_mm_per_m": float(slope), "intercept_mm": float(intercept),
            "residual_rms_mm": float(np.sqrt((res ** 2).mean())), "points": len(pts)}


def summarize(samples: list[dict]) -> dict:
    if not samples:
        return {}
    print("\n  #  leg    reach mm   commanded z   reported z   measured   reported - measured   tool rot err")
    for s in samples:
        meas = "-" if s["measured_mm"] is None else f"{s['measured_mm']:.1f}"
        diff = "-" if s["measured_mm"] is None else f"{s['reported_z_mm'] - s['measured_mm']:+.1f}"
        print(f"{s['stop']:3d}  {s['leg']:<5} {s['reach_m'] * 1000:9.0f} {s['commanded_z_mm']:13.1f} "
              f"{s['reported_z_mm']:12.1f} {meas:>10} {diff:>21} {s['tool_rotation_error_deg']:12.2f} deg")
    if any(s.get("joint_error_deg") is not None for s in samples):
        print("\n  #   measured - target joints (deg), j0..j5                   effort (Nm) j1 j2"
              "       compensation effort (Nm) j1 j2")
        for s in samples:
            if s.get("joint_error_deg") is None:
                continue
            e, eff, comp = s["joint_error_deg"], s["arm_joint_efforts"], s["arm_compensation_efforts"]
            print(f"{s['stop']:3d}  " + " ".join(f"{v:+7.3f}" for v in e)
                  + f"   {eff[1]:+7.2f} {eff[2]:+7.2f}        {comp[1]:+7.2f} {comp[2]:+7.2f}")
    fits = {"reported_z_mm": fit(samples, "reported_z_mm"), "measured_mm": fit(samples, "measured_mm")}
    for s in samples:
        s["reported_minus_measured_mm"] = (None if s["measured_mm"] is None
                                           else s["reported_z_mm"] - s["measured_mm"])
    fits["reported_minus_measured_mm"] = fit(samples, "reported_minus_measured_mm")
    for key, f in fits.items():
        if f:
            print(f"{key:>26} vs reach: {f['slope_mm_per_m']:+.1f} mm per metre, intercept "
                  f"{f['intercept_mm']:.1f} mm, residual rms {f['residual_rms_mm']:.1f} mm ({f['points']} stops)")
    print(f"Gravity-compensation touches gave reported - true = {REFERENCE_SLOPE_MM_PER_M:+.0f} mm per metre "
          "(docs/arm_joint_offsets.md, section 1). Only the slopes compare: the intercepts hold the offset "
          "between the arm's z = 0 and the table, and the measured point against the tool frame.")
    return fits


def run(args, stops: list[dict]) -> None:
    import trossen_arm

    arm = ArmModel() if args.command == "joints" else None
    ident = robot_identity(load_calibration_config(args.calibration_config))
    output = args.output or RESULTS_DIR / f"hold_z_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    driver = trossen_arm.TrossenArmDriver()
    payload, samples = None, []
    try:
        driver.configure(getattr(trossen_arm.Model, ident["model"]),
                         getattr(trossen_arm.StandardEndEffector, ident["end_effector"]), ident["ip"], False)
        n_arm = driver.get_num_joints() - 1
        print(f"Connected to {ident['model']} at {ident['ip']} ({ident['end_effector']}).")
        fk_check = None
        if arm is not None:
            r = read(driver)
            p, R = arm.fk(r["q"])
            fk_check = {"rot_deg": rotation_angle_deg(R.T @ rotvec_to_matrix(r["tool"][3:])),
                        "trans_mm": float(np.linalg.norm(p - r["tool"][:3]) * 1000)}
            print(f"Model FK against the driver's tool pose: {fk_check['rot_deg']:.4f} deg, "
                  f"{fk_check['trans_mm']:.3f} mm")
            if fk_check["rot_deg"] > MAX_FK_MISMATCH[0] or fk_check["trans_mm"] > MAX_FK_MISMATCH[1]:
                raise SystemExit("The model's tool frame does not match the driver's; joint targets would be wrong.")

        ask("SUPPORT THE ARM and press Enter: it goes into gravity compensation. ")
        hold_arm_with_gravity_compensation(driver)
        ask("Guide the gripper near the start pose, tool roughly pointing down, then press Enter "
            "(position mode, then a move to the start pose). ")

        driver.set_arm_modes(trossen_arm.Mode.position)
        q_now = np.array(driver.get_arm_positions(), dtype=float)
        driver.set_arm_positions(q_now.tolist(), 0.5, True)
        gripper_cmd_m = args.gripper_gap_mm * GRIPPER_M_PER_GAP_MM
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(gripper_cmd_m, 1.0, True)
        gripper_m = float(driver.get_gripper_position())
        print(f"Gripper at {gripper_m * 1000:.2f} mm (commanded {gripper_cmd_m * 1000:.2f} mm, "
              f"a {gripper_m / GRIPPER_M_PER_GAP_MM:.1f} mm gap between the fingertips).")
        if args.orientation == "down":
            rotvec = np.array(TOOL_DOWN_ROTVEC)
        else:
            rotvec = np.array(driver.get_robot_output().cartesian.positions, dtype=float)[3:]
        R_goal = rotvec_to_matrix(rotvec)
        targets = solve_stops(arm, stops, R_goal, q_now) if arm is not None else None
        if targets is not None:
            print("Joint targets (deg):")
            for i, q in enumerate(targets):
                print(f"  {i:2d} " + " ".join(f"{v:7.2f}" for v in np.degrees(q)))

        payload = {
            "experiment": "sag_test/hold_z_sweep",
            "created": datetime.now().astimezone().isoformat(),
            "configuration": {**ident, "driver_version": driver.get_driver_version(),
                              "controller_version": driver.get_controller_version(),
                              "end_effector": {"name": ident["end_effector"],
                                               **end_effector_record(driver.get_end_effector())},
                              "num_arm_joints": n_arm,
                              "command": args.command,
                              "arm_control": ("position mode; set_arm_positions to IK joint targets, blocking"
                                              if arm is not None else
                                              "position mode; set_cartesian_positions (driver IK), blocking"),
                              "ik_model": None if arm is None else {"path": str(ARM_MODEL), "site": TOOL_SITE,
                                                                    "fk_check_at_connection": fk_check},
                              "start_move": f"joint space, {args.start_time_s} s",
                              "step_move": ("joint space" if arm is not None else f"{args.space} space")
                              + f", {args.step_time_s} s",
                              "gripper": {"commanded_gap_mm": args.gripper_gap_mm,
                                          "commanded_position_m": gripper_cmd_m, "measured_position_m": gripper_m,
                                          "gap_per_position": "gap = 2 x driver gripper position "
                                                              "(gripper_width hardware results)"},
                              "settle_s": args.settle_s, "average_s": args.average_s,
                              "orientation": args.orientation, "goal_rotvec_rad": rotvec.tolist()},
            "units": {"positions": "m", "heights_mm": "mm, measured from the table by the operator",
                      "efforts": "driver arm efforts, compensation efforts and external efforts (Nm)"},
            "plan": stops,
            "joint_targets_rad": None if targets is None else [q.tolist() for q in targets],
            "samples": samples,
        }

        def go(i: int, move_time: float, first_move: bool = False) -> None:
            s = stops[i]
            if targets is not None:
                driver.set_arm_positions(targets[i].tolist(), move_time, True)
            else:
                space = trossen_arm.InterpolationSpace.joint if first_move else \
                    getattr(trossen_arm.InterpolationSpace, args.space)
                driver.set_cartesian_positions([s["x_m"], s["y_m"], s["z_m"], *rotvec], space, move_time, True)

        first = stops[0]
        print(f"Moving to the start pose x {first['x_m']:.3f} y {first['y_m']:.3f} z {first['z_m']:.3f} m ...")
        go(0, args.start_time_s, first_move=True)
        ask("At the start pose. Check the gripper is clear of everything, then press Enter to begin. ")

        for i, stop in enumerate(stops):
            if i > 0:
                go(i, args.step_time_s)
            time.sleep(args.settle_s)
            reading = average(driver, args.average_s)
            tool = np.array(reading["tool_m_rotvec_rad"])
            reported_z_mm = tool[2] * 1000
            sample = {"stop": i, **stop, **reading,
                      "commanded_z_mm": stop["z_m"] * 1000, "reported_z_mm": float(reported_z_mm),
                      "position_error_mm": ((tool[:3] - np.array([stop["x_m"], stop["y_m"], stop["z_m"]])) * 1000).tolist(),
                      "tool_rotation_error_deg": rotation_angle_deg(R_goal.T @ rotvec_to_matrix(tool[3:])),
                      "joint_target_rad": None, "joint_error_deg": None}
            if targets is not None:
                err = np.degrees(np.array(reading["arm_joint_positions_rad"]) - targets[i])
                sample.update(joint_target_rad=targets[i].tolist(), joint_error_deg=err.tolist())
            print(f"\nStop {i} ({stop['leg']}): reach {stop['reach_m'] * 1000:.0f} mm, commanded z "
                  f"{stop['z_m'] * 1000:.1f} mm, reported z {reported_z_mm:.1f} mm, reported x {tool[0] * 1000:.1f} mm")
            if sample["joint_error_deg"] is not None:
                print("  measured - target (deg): " + " ".join(f"{v:+.3f}" for v in sample["joint_error_deg"]))
            print("  compensation efforts (Nm): " + " ".join(f"{v:+.2f}" for v in reading["arm_compensation_efforts"]))
            sample["measured_mm"] = ask_height("Measured height from the table (mm), Enter to skip, q to stop: ")
            samples.append(sample)
            write_json(output, payload)

        print("Back to the start pose ...")
        go(0, args.start_time_s)
    except (Quit, KeyboardInterrupt):
        print("\nStopped.")
    finally:
        if payload is not None:
            payload["fits"] = summarize(samples)
            write_json(output, payload)
            print(f"{len(samples)} stops saved to {output}")
        try:
            hold_arm_with_gravity_compensation(driver)
        except Exception as error:                       # not connected, or the driver already failed
            print(f"Could not switch to gravity compensation: {error}")
        release_to_idle(driver)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true", help="connect and move the arm (default: dry run)")
    p.add_argument("--command", choices=("joints", "cartesian"), default="joints",
                   help="joints: IK on the model, send joint targets; cartesian: the driver's IK")
    p.add_argument("--x0", type=float, default=0.25, help="start x, m")
    p.add_argument("--y", type=float, default=0.0, help="y held for the whole sweep, m")
    p.add_argument("--z", type=float, default=0.10, help="commanded z held for the whole sweep, m")
    p.add_argument("--distance-m", type=float, default=0.18, help="distance along +x, m")
    p.add_argument("--step-m", type=float, default=0.045, help="distance between stops, m")
    p.add_argument("--back", action="store_true", help="also stop on the way back (shows hysteresis from play)")
    p.add_argument("--orientation", choices=("down", "current"), default="down",
                   help="down: tool x axis straight down; current: the orientation the arm was guided to")
    p.add_argument("--space", choices=("cartesian", "joint"), default="cartesian",
                   help="--command cartesian only: interpolation space of the moves between stops")
    p.add_argument("--gripper-gap-mm", type=float, default=10.0,
                   help="gap between the fingertips while measuring, mm (0 closes the gripper)")
    p.add_argument("--start-time-s", type=float, default=4.0, help="move time to the start pose, s")
    p.add_argument("--step-time-s", type=float, default=2.0, help="move time between stops, s")
    p.add_argument("--settle-s", type=float, default=1.0, help="wait after each move before reading, s")
    p.add_argument("--average-s", type=float, default=0.5, help="readings averaged per stop, s")
    p.add_argument("--calibration-config", type=Path, default=CALIBRATION_CONFIG)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.step_m <= 0 or args.distance_m <= 0:
        raise SystemExit("--step-m and --distance-m must be positive")
    if not 0.0 <= args.gripper_gap_mm <= 80.0:
        raise SystemExit("--gripper-gap-mm must be between 0 and 80")

    stops = plan_targets(args)
    check_plan(stops)
    print(f"Command: {args.command}. Orientation: {args.orientation}. "
          f"Stops (commanded z {args.z * 1000:.0f} mm throughout):")
    for i, s in enumerate(stops):
        print(f"  {i:2d} {s['leg']:<4} x {s['x_m'] * 1000:6.1f} y {s['y_m'] * 1000:6.1f} mm, "
              f"reach {s['reach_m'] * 1000:5.0f} mm")
    if args.command == "joints" and args.orientation == "down":
        targets = solve_stops(ArmModel(), stops, rotvec_to_matrix(TOOL_DOWN_ROTVEC), DRY_RUN_SEED)
        print("IK joint targets from a nominal seed (deg; the run seeds from the guided pose):")
        for i, q in enumerate(targets):
            print(f"  {i:2d} " + " ".join(f"{v:7.2f}" for v in np.degrees(q)))
    if not args.execute:
        print("Dry run. Pass --execute to command the arm (the operator must be at the arm).")
        return
    run(args, stops)


if __name__ == "__main__":
    main()
