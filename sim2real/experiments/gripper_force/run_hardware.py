"""Measure the real gripper's squeeze force on a jenga block in position mode.

The arm joints are not touched: they stay in whatever mode they are in (idle after power-on,
resting on their stops). Only the gripper is put in position mode. At exit the gripper opens
and the driver is cleaned up, which sets every joint to idle.

Trials, each started from a prompt:

  free     no block. The gripper closes to the policy's closed target. Gives the effort the
           gripper reports with no load at that command (offset plus its own friction).
  block    the operator holds a jenga block between the fingers, across its 25 mm face
           (the face the sim grasps). The gripper closes to each target in --targets, so the
           position error, and the force, differ between targets. Then it reopens.

The first target is the policy's closed command (gripper_closed + gripper_limit_margin_m =
0.002 m); the others give the force against position error. Each close uses the policy's
gripper goal time (0.5 s). The gripper position, velocity, effort and external effort are
logged at --rate-hz for --record-s after the close command. The summary averages the last
second.

The driver's gripper coordinate is half the jaw gap, in metres (gripper_width experiment).

Dry run unless --execute: prints the plan and exits without connecting.

Launch from the fp_robot environment:

  conda activate fp_robot
  python real_robot/sim2real/experiments/gripper_force/run_hardware.py --execute
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REAL_ROBOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REAL_ROBOT))
sys.path.insert(0, str(REAL_ROBOT / "vision"))
sys.path.insert(0, str(EXPERIMENTS))

from calibration.board import DEFAULT_CONFIG as CALIBRATION_CONFIG  # noqa: E402
from calibration.board import load_config as load_calibration_config  # noqa: E402
from common.hardware import robot_identity  # noqa: E402
from common.io import write_json  # noqa: E402
from common.prompts import Quit, ask  # noqa: E402

EXPERIMENT = "gripper_squeeze_force"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OPEN_M = 0.030  # 60 mm jaw gap: room to place a 25 mm block
OPEN_GOAL_TIME_S = 1.0
CLOSE_GOAL_TIME_S = 0.5  # manipulation/policy/jenga/config.yaml control.gripper_goal_time_s
POLICY_CLOSED_M = 0.002  # gripper_closed 0.0 + gripper_limit_margin_m 0.002
SUMMARY_WINDOW_S = 1.0


def _record(driver, duration_s: float, rate_hz: float) -> dict[str, list[float]]:
    log = {"t_s": [], "position_m": [], "velocity_m_s": [], "effort_n": [], "external_effort_n": []}
    period = 1.0 / rate_hz
    start = time.monotonic()
    next_t = start
    while True:
        now = time.monotonic()
        if now - start > duration_s:
            return log
        log["t_s"].append(now - start)
        log["position_m"].append(float(driver.get_gripper_position()))
        log["velocity_m_s"].append(float(driver.get_gripper_velocity()))
        log["effort_n"].append(float(driver.get_gripper_effort()))
        log["external_effort_n"].append(float(driver.get_gripper_external_effort()))
        next_t += period
        time.sleep(max(0.0, next_t - time.monotonic()))


def _summary(log: dict[str, list[float]], target_m: float) -> dict[str, float]:
    t = np.asarray(log["t_s"])
    tail = t >= t[-1] - SUMMARY_WINDOW_S
    out = {k: float(np.mean(np.asarray(v)[tail])) for k, v in log.items() if k != "t_s"}
    out["effort_std_n"] = float(np.std(np.asarray(log["effort_n"])[tail]))
    out["position_error_m"] = out["position_m"] - target_m
    return out


def _close_and_record(driver, target_m: float, args) -> dict:
    driver.set_gripper_position(target_m, CLOSE_GOAL_TIME_S, False)
    log = _record(driver, args.record_s, args.rate_hz)
    return {"target_m": target_m, "summary": _summary(log, target_m), "log": log}


def _open(driver) -> None:
    driver.set_gripper_position(OPEN_M, OPEN_GOAL_TIME_S, True)


def _confirm(prompt: str) -> None:
    while True:
        text = ask(f"{prompt} (Enter, or q to quit): ").strip().lower()
        if text == "":
            return
        if text in {"q", "quit"}:
            raise Quit()


def _print_summary(name: str, trial: dict, free_effort: float | None) -> None:
    s = trial["summary"]
    line = (f"  {name}: target {1e3 * trial['target_m']:.1f} mm, position {1e3 * s['position_m']:.2f} mm "
            f"(jaw gap {2e3 * s['position_m']:.1f} mm), error {1e3 * s['position_error_m']:+.2f} mm, "
            f"effort {s['effort_n']:+.2f} N (std {s['effort_std_n']:.2f}), "
            f"external {s['external_effort_n']:+.2f} N")
    if free_effort is not None:
        line += f", effort minus free {s['effort_n'] - free_effort:+.2f} N"
    print(line)


def run(args) -> None:
    import trossen_arm

    ident = robot_identity(load_calibration_config(args.calibration_config))
    output = args.output or RESULTS_DIR / f"gripper_force_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    driver = trossen_arm.TrossenArmDriver()
    payload = None
    try:
        driver.configure(getattr(trossen_arm.Model, ident["model"]),
                         getattr(trossen_arm.StandardEndEffector, ident["end_effector"]), ident["ip"], False)
        n = driver.get_num_joints()
        modes = [trossen_arm.MODE_NAME[m] for m in driver.get_modes()]
        print(f"Connected to {ident['model']} at {ident['ip']}. Joint modes: {modes}")
        limits = driver.get_joint_limits()[n - 1]
        for target in args.targets:
            if not float(limits.position_min) <= target <= float(limits.position_max):
                raise SystemExit(f"target {target} m is outside the gripper limits "
                                 f"[{limits.position_min}, {limits.position_max}]")
        characteristic = driver.get_joint_characteristics()[n - 1]
        gains = driver.get_motor_parameters()[n - 1][trossen_arm.Mode.position]
        payload = {
            "experiment": EXPERIMENT,
            "source": "hardware",
            "configuration": {
                "ip": ident["ip"], "model": ident["model"], "end_effector": ident["end_effector"],
                "driver_version": driver.get_driver_version(),
                "controller_version": driver.get_controller_version(),
                "arm_modes_at_start": modes,
                "gripper_position_mode_gains": {"position_kp": gains.position.kp, "velocity_kp": gains.velocity.kp},
                "gripper_effort_max_n": float(limits.effort_max),
                "gripper_friction_constant_term_n": float(characteristic.friction_constant_term),
                "gripper_effort_correction": float(characteristic.effort_correction),
                "open_m": OPEN_M, "close_goal_time_s": CLOSE_GOAL_TIME_S,
                "record_s": args.record_s, "rate_hz": args.rate_hz, "summary_window_s": SUMMARY_WINDOW_S,
                "block_note": args.block_note,
                "coordinate": "driver gripper position = half the jaw gap, m",
            },
            "trials": [],
        }
        write_json(output, payload)
        print(f"Saving to {output}\nThe arm joints are not commanded. Only the gripper moves.")

        _confirm("Keep fingers clear: the gripper goes to position mode and opens to 60 mm")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        _open(driver)

        free_effort = None
        for repeat in range(args.repeats):
            _confirm(f"[free {repeat + 1}/{args.repeats}] Nothing between the fingers; close")
            trial = _close_and_record(driver, args.targets[0], args)
            trial.update(kind="free", repeat=repeat)
            payload["trials"].append(trial)
            write_json(output, payload)
            _print_summary("free", trial, None)
            _open(driver)
        free = [t["summary"]["effort_n"] for t in payload["trials"] if t["kind"] == "free"]
        free_effort = float(np.mean(free))

        for repeat in range(args.repeats):
            for target in args.targets:
                _confirm(f"[block {repeat + 1}/{args.repeats}, target {1e3 * target:.1f} mm] "
                         "Hold the block between the fingers across its 25 mm face, centred; close")
                trial = _close_and_record(driver, target, args)
                trial.update(kind="block", repeat=repeat)
                payload["trials"].append(trial)
                write_json(output, payload)
                _print_summary("block", trial, free_effort)
                _open(driver)
                print("  Opened. Take the block.")

        blocks = [t for t in payload["trials"] if t["kind"] == "block"]
        print(f"\nFree-close effort at {1e3 * args.targets[0]:.1f} mm: {free_effort:+.2f} N (mean of {len(free)})")
        for target in args.targets:
            rows = [t["summary"] for t in blocks if t["target_m"] == target]
            effort = np.mean([r["effort_n"] for r in rows])
            error = np.mean([r["position_error_m"] for r in rows])
            print(f"Block, target {1e3 * target:.1f} mm: effort {effort:+.2f} N, minus free "
                  f"{effort - free_effort:+.2f} N, position error {1e3 * error:+.2f} mm (n={len(rows)})")
        print(f"Saved {output}")
    except Quit:
        print("\nStopped." + ("" if payload is None else f" {len(payload['trials'])} trial(s) saved."))
    finally:
        try:
            _open(driver)
        except Exception as exc:  # the driver may not be configured
            print(f"Could not reopen the gripper: {exc}")
        driver.cleanup()  # every joint to idle; the arm joints were never commanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="connect and move the gripper")
    parser.add_argument("--targets", type=float, nargs="+", default=[POLICY_CLOSED_M, 0.006, 0.010],
                        help="closed targets in m (driver coordinate, half the jaw gap); first is used for free closes")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--record-s", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--block-note", default="jenga block, 25 mm face")
    parser.add_argument("--calibration-config", type=Path, default=CALIBRATION_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.calibration_config.is_file():
        raise SystemExit(f"calibration config not found: {args.calibration_config}")
    if not args.execute:
        print("Dry run. With --execute: connect to the manipulator, leave the arm joints alone, put the "
              f"gripper in position mode, {args.repeats} free close(s) to {1e3 * args.targets[0]:.1f} mm, then "
              f"{args.repeats} round(s) of block closes to {[round(1e3 * t, 1) for t in args.targets]} mm, "
              f"each logged {args.record_s} s at {args.rate_hz} Hz; finally open and set every joint idle.")
        return
    run(args)


if __name__ == "__main__":
    main()
