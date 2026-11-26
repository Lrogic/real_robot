"""Map the arm's height error ("sag") over base direction, reach, height and tool pitch under
position control, with a ruler at every stop.

hold_z_sweep.py found that the arm reports its tool point higher than it is by about 26 mm per metre
of reach, straight ahead, at one height, with the tool straight down. That run cannot say whether
the error depends on direction (a tilted table or base would), or whether it is really a function
of reach or of the gravity load on each joint (joint compliance after the encoders, which also
changes with height and tool pitch at a fixed reach), or a joint zero error (no load dependence).
This grid varies all four so analyze_sag.py can tell them apart.

The grid (per base direction, --yaws; GRID below): tool straight down (pitch 90) at z 0.10 m for
reaches 0.15 to 0.40 m (lower, the ruler between the fingertips hits the wrist housing first; the
design check found z 0.10 alone separates the models as well as z 0.05 and 0.10); tool pitched 45 deg (pointing down and
away from the base) at z 0.05 m for 0.30 to 0.50 m; tool level (pitch 0, pointing away) at z 0.12 m
for 0.45 and 0.50 m and at z 0.30 m for 0.30 to 0.50 m. Straight down cannot reach further or
higher (joint_3 hits -90 deg). The level rows are there for the elbow: in the other rows its gravity
load hardly changes, so its compliance could not be told from a line in reach. The design check
(dry run) showed that with the 45 deg z 0.18 m row instead, shoulder-plus-elbow compliance and the
reach line fitted a ruler's data almost equally well (held-out 0.50 against 0.56 mm); with the
level rows, 0.53 against 0.71 mm. A ruler of at least 300 mm is needed. The order inside each direction is shuffled (--seed) so the joints arrive
from varied sides. A reference pose (--reference) is visited at the start, between directions and
at the end, to show drift and how repeatable the ruler is.

Moves are joint-space (set_arm_positions, the driver's quintic) to IK targets on the MuJoCo model,
as in hold_z_sweep.py. Every move's straight joint path is checked on the model: if any point of
the gripper would pass below --min-path-z, the move goes through a safe pose (pitch 45, reach
0.30 m, z 0.20 m) at the start and goal directions. The dry run prints all of this.

WHAT TO MEASURE: the height above the table of the tool point, the middle of the fingertips' ends.
Read it on the side of one finger, at the centre of the fingertip (the tip is about 10 to 12 mm
wide; a small tape mark there helps if the centre is hard to judge). With the tool straight down
the tip is level and any point on it will do. Tilted, the tip's corners differ in height by about
8 mm (pitch 45) to 12 mm (level), so the centre matters. The side of the finger is offset from the
tool point only along the gripper's opening axis, which stays horizontal in every pose of this
grid, so its height is the tool point's. Ruler held vertical beside the finger; read to 0.5 mm
if you can. The same point for the whole run.

Sequence (--execute): as hold_z_sweep.py. Gravity compensation to guide the arm near the first pose
(straight ahead, tool roughly down), Enter for position mode and the move to the first stop, then
at each stop: settle, average the readings, ask for the height. q stops, Enter skips a stop. At the
end (or on Ctrl-C) back to the reference pose, gravity compensation, idle: SUPPORT THE ARM.
--start-at N resumes a stopped run from stop N into a new file (analyze both files with
--shared-intercept if the mark and table did not change).

Launch from the vision environment (mujoco and trossen_arm), from any directory:

  /opt/conda-envs/vision/bin/python real_robot/sim2real/sag_test/sag_map.py              # dry run + design check
  /opt/conda-envs/vision/bin/python real_robot/sim2real/sag_test/sag_map.py --execute

Results go to sim2real/sag_test/results/sag_map_<time>.json, rewritten after every stop. Then:

  /opt/conda-envs/vision/bin/python real_robot/sim2real/sag_test/analyze_sag.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hold_z_sweep as hz  # noqa: E402  (sets up the real_robot, vision and experiments paths)
from analyze_sag import Kinematics, design_check  # noqa: E402
from calibration.board import DEFAULT_CONFIG as CALIBRATION_CONFIG  # noqa: E402
from calibration.board import load_config as load_calibration_config  # noqa: E402
from common.hardware import end_effector_record, hold_arm_with_gravity_compensation, release_to_idle, robot_identity  # noqa: E402
from common.io import write_json  # noqa: E402
from common.poses import rotation_angle_deg, rotvec_to_matrix  # noqa: E402
from common.prompts import Quit, ask  # noqa: E402

RESULTS_DIR = HERE / "results"
GRID = [  # (pitch deg, z m, reaches m)
    (90, 0.10, (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)),   # not lower: the ruler must fit under the wrist
    (45, 0.05, (0.30, 0.40, 0.50)),
    (0, 0.12, (0.45, 0.50)),                  # tool level, low and high: the widest spread of elbow
    (0, 0.30, (0.30, 0.40, 0.50)),            # load at one reach (3.6 to 6.6 Nm) the arm allows
]
SAFE_POSE = (45, 0.30, 0.20)                 # pitch deg, reach m, z m: waypoint when a straight joint path dips
MAX_REACH_M = 0.52
MIN_YAW_DEG, MAX_YAW_DEG = -90.0, 0.0        # the operator's clear area; widen deliberately with --allow-any-yaw
PATH_SAMPLES = 40
# Points on the gripper in the tool frame (m), checked against the table along every move: the
# fingertip ends' corners (generous: the tips are about 12 mm wide along tool z, the fingers 60 mm at
# their base) and the housing's lower corners.
GRIPPER_POINTS = np.array([[0.0, sy * 0.04, sz * 0.03] for sy in (-1, 1) for sz in (-1, 1)]
                          + [[-0.156, sy * 0.092, sz * 0.029] for sy in (-1, 1) for sz in (-1, 1)]
                          + [[-0.077, sy * 0.092, sz * 0.029] for sy in (-1, 1) for sz in (-1, 1)])


def goal_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Tool x axis pitched pitch_deg below horizontal, pointing away from the base along yaw."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ rotvec_to_matrix((0.0, np.radians(pitch_deg), 0.0))


def goal_position(yaw_deg: float, reach_m: float, z_m: float) -> np.ndarray:
    a = np.radians(yaw_deg)
    return np.array([reach_m * np.cos(a), reach_m * np.sin(a), z_m])


def plan_stops(args) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    ref = {"kind": "reference", "yaw_deg": args.reference[0], "pitch_deg": args.reference[1],
           "reach_m": args.reference[2], "z_m": args.reference[3]}
    stops = [dict(ref)]
    for yaw in args.yaws:
        block = [{"kind": "grid", "yaw_deg": yaw, "pitch_deg": pitch, "reach_m": r, "z_m": z}
                 for pitch, z, reaches in GRID for r in reaches]
        stops += [block[i] for i in rng.permutation(len(block))]
        stops.append(dict(ref))
    for i, s in enumerate(stops):
        s["stop"] = i
        s["x_m"], s["y_m"], _ = goal_position(s["yaw_deg"], s["reach_m"], s["z_m"]).tolist()
    return stops


def check_plan(stops: list[dict], allow_any_yaw: bool) -> None:
    for s in stops:
        if s["reach_m"] > MAX_REACH_M:
            raise SystemExit(f"stop {s['stop']} reaches {s['reach_m']} m, more than {MAX_REACH_M} m")
        if s["z_m"] < hz.MIN_Z_M:
            raise SystemExit(f"stop {s['stop']} z {s['z_m']} m is closer to the table than {hz.MIN_Z_M} m")
        if not allow_any_yaw and not MIN_YAW_DEG <= s["yaw_deg"] <= MAX_YAW_DEG:
            raise SystemExit(f"stop {s['stop']} yaw {s['yaw_deg']} deg is outside the clear area "
                             f"{MIN_YAW_DEG:.0f} to {MAX_YAW_DEG:.0f} deg (--allow-any-yaw)")


def seed_for(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    q = np.array(hz.DRY_RUN_SEED, dtype=float)
    q[0] = np.radians(yaw_deg)
    if pitch_deg < 80:
        q[3] = np.radians(-30)                       # a pitched tool keeps the wrist nearer level
    return q


def solve(arm: hz.ArmModel, s: dict, q_seed=None) -> np.ndarray:
    p, R = goal_position(s["yaw_deg"], s["reach_m"], s["z_m"]), goal_rotation(s["yaw_deg"], s["pitch_deg"])
    seeds = ([q_seed] if q_seed is not None else []) + [seed_for(s["yaw_deg"], s["pitch_deg"])]
    for q0 in seeds:
        try:
            q = arm.ik(p, R, q0)
        except RuntimeError:
            continue
        if abs(q[4]) < np.radians(10) and abs(q[5]) < np.radians(10):   # the plain branch: no wrist roll
            return q
    raise SystemExit(f"stop {s.get('stop')} ({s['kind']}, yaw {s['yaw_deg']}, pitch {s['pitch_deg']}, "
                     f"reach {s['reach_m']}, z {s['z_m']}) has no plain IK solution")


def lowest_point_on_path(arm: hz.ArmModel, q_a: np.ndarray, q_b: np.ndarray) -> float:
    lowest = np.inf
    for t in np.linspace(0, 1, PATH_SAMPLES):
        p, R = arm.fk(q_a + t * (q_b - q_a))
        lowest = min(lowest, float((p + GRIPPER_POINTS @ R.T)[:, 2].min()))
    return lowest


def route(arm: hz.ArmModel, q_a: np.ndarray, q_b: np.ndarray, yaw_a: float, yaw_b: float,
          min_z: float) -> list[np.ndarray]:
    """Joint targets from q_a to q_b: direct if the straight joint path stays above min_z, else via
    the safe pose at each end's direction."""
    if lowest_point_on_path(arm, q_a, q_b) >= min_z:
        return [q_b]
    pitch, reach, z = SAFE_POSE
    safe = [solve(arm, {"kind": "safe", "yaw_deg": y, "pitch_deg": pitch, "reach_m": reach, "z_m": z})
            for y in (yaw_a, yaw_b)]
    path = [safe[0]] if abs(yaw_a - yaw_b) < 1e-6 else safe
    legs = [q_a, *path, q_b]
    for a, b in zip(legs[:-1], legs[1:]):
        low = lowest_point_on_path(arm, a, b)
        if low < min_z:
            raise SystemExit(f"no safe route: a leg through the safe pose dips to {low * 1000:.0f} mm")
    return [*path, q_b]


def build(arm: hz.ArmModel, stops: list[dict], min_z: float, q_start=None) -> tuple[list, list]:
    targets = [solve(arm, s) for s in stops]
    routes, q_prev = [], q_start
    yaw_prev = None if q_start is None else float(np.degrees(q_start[0]))
    for s, q in zip(stops, targets):
        if q_prev is None:
            routes.append([q])
        else:
            routes.append(route(arm, q_prev, q, yaw_prev, s["yaw_deg"], min_z))
        q_prev, yaw_prev = q, s["yaw_deg"]
    return targets, routes


def print_plan(arm: hz.ArmModel, stops, targets, routes) -> None:
    print("  #  kind        yaw  pitch  reach    z      joint targets (deg) j0..j5                  via")
    for s, q, r in zip(stops, targets, routes):
        p, _ = arm.fk(q)
        print(f"{s['stop']:3d}  {s['kind']:<10}{s['yaw_deg']:5.0f}{s['pitch_deg']:6.0f}{s['reach_m']:7.2f}"
              f"{s['z_m']:6.2f}   " + " ".join(f"{v:7.1f}" for v in np.degrees(q))
              + f"   {'safe pose' if len(r) > 1 else ''}")
    moves = sum(len(r) for r in routes)
    print(f"{len(stops)} stops, {moves} moves.")


def run(args, stops: list[dict]) -> None:
    import trossen_arm

    arm = hz.ArmModel()
    ident = robot_identity(load_calibration_config(args.calibration_config))
    output = args.output or RESULTS_DIR / f"sag_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    driver = trossen_arm.TrossenArmDriver()
    payload, samples = None, []
    reference = stops[0]
    q_reference = solve(arm, reference)
    try:
        driver.configure(getattr(trossen_arm.Model, ident["model"]),
                         getattr(trossen_arm.StandardEndEffector, ident["end_effector"]), ident["ip"], False)
        n_arm = driver.get_num_joints() - 1
        print(f"Connected to {ident['model']} at {ident['ip']} ({ident['end_effector']}).")
        r = hz.read(driver)
        p, R = arm.fk(r["q"])
        fk_check = {"rot_deg": rotation_angle_deg(R.T @ rotvec_to_matrix(r["tool"][3:])),
                    "trans_mm": float(np.linalg.norm(p - r["tool"][:3]) * 1000)}
        print(f"Model FK against the driver's tool pose: {fk_check['rot_deg']:.4f} deg, {fk_check['trans_mm']:.3f} mm")
        if fk_check["rot_deg"] > hz.MAX_FK_MISMATCH[0] or fk_check["trans_mm"] > hz.MAX_FK_MISMATCH[1]:
            raise SystemExit("The model's tool frame does not match the driver's; joint targets would be wrong.")

        ask("SUPPORT THE ARM and press Enter: it goes into gravity compensation. ")
        hold_arm_with_gravity_compensation(driver)
        ask("Guide the gripper above the table straight ahead, tool roughly pointing down, then press Enter "
            "(position mode, then the move to the first stop). ")
        driver.set_arm_modes(trossen_arm.Mode.position)
        q_now = np.array(driver.get_arm_positions(), dtype=float)
        driver.set_arm_positions(q_now.tolist(), 0.5, True)
        gripper_cmd_m = args.gripper_gap_mm * hz.GRIPPER_M_PER_GAP_MM
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(gripper_cmd_m, 1.0, True)
        gripper_m = float(driver.get_gripper_position())

        todo = stops[args.start_at:]
        targets, routes = build(arm, todo, args.min_path_z, q_start=q_now)
        payload = {
            "experiment": "sag_test/sag_map",
            "created": datetime.now().astimezone().isoformat(),
            "configuration": {**ident, "driver_version": driver.get_driver_version(),
                              "controller_version": driver.get_controller_version(),
                              "end_effector": {"name": ident["end_effector"],
                                               **end_effector_record(driver.get_end_effector())},
                              "num_arm_joints": n_arm,
                              "arm_control": "position mode; set_arm_positions to IK joint targets, blocking",
                              "ik_model": {"path": str(hz.ARM_MODEL), "site": hz.TOOL_SITE,
                                           "fk_check_at_connection": fk_check},
                              "move_time_s": args.move_time_s, "settle_s": args.settle_s,
                              "average_s": args.average_s, "seed": args.seed, "start_at": args.start_at,
                              "min_path_z_m": args.min_path_z, "safe_pose": SAFE_POSE,
                              "gripper": {"commanded_gap_mm": args.gripper_gap_mm,
                                          "commanded_position_m": gripper_cmd_m, "measured_position_m": gripper_m},
                              "measured_point": "side of a finger at the centre of the fingertip end "
                                                "(beside the tool point along tool y, which is horizontal)",
                              "pitch_meaning": "tool x axis angle below horizontal, pointing away from the base"},
            "units": {"positions": "m", "measured_mm": "mm, fingertip centre above the table, ruler",
                      "efforts": "driver arm efforts, compensation efforts and external efforts (Nm)"},
            "plan": stops,
            "samples": samples,
        }

        def go(legs: list[np.ndarray], move_time: float) -> None:
            for q in legs:
                q_from = np.array(driver.get_arm_positions(), dtype=float)
                swing = float(np.degrees(np.abs(q - q_from).max()))
                driver.set_arm_positions(q.tolist(), max(move_time, swing / args.max_joint_speed_dps), True)

        first = todo[0]
        print(f"Moving to stop {first['stop']} ...")
        go(routes[0], args.start_time_s)
        ask("Check the gripper is clear of everything, then press Enter to begin. ")

        for i, stop in enumerate(todo):
            if i > 0:
                go(routes[i], args.move_time_s)
            time.sleep(args.settle_s)
            reading = hz.average(driver, args.average_s)
            tool = np.array(reading["tool_m_rotvec_rad"])
            goal = goal_position(stop["yaw_deg"], stop["reach_m"], stop["z_m"])
            err = np.degrees(np.array(reading["arm_joint_positions_rad"]) - targets[i])
            sample = {**stop, **reading, "commanded_z_mm": stop["z_m"] * 1000,
                      "reported_z_mm": float(tool[2] * 1000),
                      "position_error_mm": ((tool[:3] - goal) * 1000).tolist(),
                      "tool_rotation_error_deg": rotation_angle_deg(
                          goal_rotation(stop["yaw_deg"], stop["pitch_deg"]).T @ rotvec_to_matrix(tool[3:])),
                      "joint_target_rad": targets[i].tolist(), "joint_error_deg": err.tolist()}
            print(f"\nStop {stop['stop']}/{len(stops) - 1} ({stop['kind']}): yaw {stop['yaw_deg']:.0f} deg, pitch "
                  f"{stop['pitch_deg']:.0f} deg, reach {stop['reach_m'] * 1000:.0f} mm, commanded z "
                  f"{stop['z_m'] * 1000:.0f} mm")
            print(f"  TCP (driver, base frame): x {tool[0] * 1000:7.1f}  y {tool[1] * 1000:7.1f}  "
                  f"z {tool[2] * 1000:7.1f} mm   (goal x {goal[0] * 1000:7.1f}  y {goal[1] * 1000:7.1f}  "
                  f"z {goal[2] * 1000:7.1f} mm)")
            print("  measured - target (deg): " + " ".join(f"{v:+.3f}" for v in err)
                  + f"   tip moved {reading['tool_moved_mm']:.2f} mm while reading")
            sample["measured_mm"] = hz.ask_height("Mark height above the table (mm), Enter to skip, q to stop: ")
            if sample["measured_mm"] is not None:
                print(f"  reported - measured: {sample['reported_z_mm'] - sample['measured_mm']:+.1f} mm")
            samples.append(sample)
            write_json(output, payload)

        print("Back to the reference pose ...")
        q_last = np.array(driver.get_arm_positions(), dtype=float)
        go(route(arm, q_last, q_reference, float(np.degrees(q_last[0])), reference["yaw_deg"], args.min_path_z),
           args.start_time_s)
    except (Quit, KeyboardInterrupt):
        print("\nStopped.")
    finally:
        if payload is not None:
            write_json(output, payload)
            print(f"{len(samples)} stops saved to {output}")
            print(f"Analyse with: /opt/conda-envs/vision/bin/python {HERE / 'analyze_sag.py'} {output}")
        try:
            hold_arm_with_gravity_compensation(driver)
        except Exception as error:                       # not connected, or the driver already failed
            print(f"Could not switch to gravity compensation: {error}")
        release_to_idle(driver)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true", help="connect and move the arm (default: dry run)")
    p.add_argument("--yaws", type=float, nargs="+", default=[0.0, -30.0, -60.0, -90.0],
                   help="base directions, deg (0 straight ahead, negative to the right)")
    p.add_argument("--reference", type=float, nargs=4, default=[-45.0, 90.0, 0.30, 0.10],
                   metavar=("YAW", "PITCH", "REACH", "Z"), help="reference pose, visited between directions")
    p.add_argument("--seed", type=int, default=0, help="shuffle of the stops inside each direction")
    p.add_argument("--start-at", type=int, default=0, help="resume: skip the stops before this one")
    p.add_argument("--allow-any-yaw", action="store_true", help="allow directions outside -90 to 0 deg")
    p.add_argument("--min-path-z", type=float, default=0.02,
                   help="lowest gripper point allowed along a direct move, m (else via the safe pose)")
    p.add_argument("--gripper-gap-mm", type=float, default=10.0, help="gap between the fingertips, mm")
    p.add_argument("--start-time-s", type=float, default=4.0, help="first and last move time, s")
    p.add_argument("--move-time-s", type=float, default=2.0, help="time of each move between stops, s")
    p.add_argument("--max-joint-speed-dps", type=float, default=25.0,
                   help="average joint speed limit, deg/s: moves with a bigger swing take longer")
    p.add_argument("--settle-s", type=float, default=1.0, help="wait after each move before reading, s")
    p.add_argument("--average-s", type=float, default=0.5, help="readings averaged per stop, s")
    p.add_argument("--no-design-check", action="store_true", help="skip the dry run's model discrimination check")
    p.add_argument("--calibration-config", type=Path, default=CALIBRATION_CONFIG)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    stops = plan_stops(args)
    check_plan(stops, args.allow_any_yaw)
    if not 0 <= args.start_at < len(stops):
        raise SystemExit(f"--start-at must be between 0 and {len(stops) - 1}")
    arm = hz.ArmModel()
    targets, routes = build(arm, stops, args.min_path_z)
    print_plan(arm, stops, targets, routes)
    if not args.execute:
        if not args.no_design_check:
            design_check(targets, stops, Kinematics(), sorted(RESULTS_DIR.glob("hold_z_*.json")))
        print("\nDry run. Pass --execute to command the arm (the operator must be at the arm).")
        return
    run(args, stops)


if __name__ == "__main__":
    main()
