"""Measure the WidowX AI inside-tip gap against the driver's gripper coordinate.

The arm joints are held in gravity compensation (zero external effort). Only the
gripper is position-commanded, through ``set_gripper_position``. Fully open and
fully closed are the gripper joint limits reported by the driver:

  open   = position_max
  closed = position_min
  half   = midpoint of those two

position_min is the closed endpoint. The gripper is not driven past it. The
driver coordinate is one joint position in metres, not jaw width. The move time
is the API default of 2 s.

Launch from the fp_robot environment:

  conda activate fp_robot
  python real_robot/sim2real/experiments/gripper_width/measure_gripper_hardware.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

REAL_ROBOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REAL_ROBOT))
sys.path.insert(0, str(REAL_ROBOT / "vision"))
sys.path.insert(0, str(EXPERIMENTS))

import trossen_arm  # noqa: E402

from calibration.board import DEFAULT_CONFIG as CALIBRATION_CONFIG  # noqa: E402
from calibration.board import load_config as load_calibration_config  # noqa: E402
from common.hardware import (  # noqa: E402
    end_effector_record,
    hold_arm_with_gravity_compensation,
    release_to_idle,
    robot_identity,
)
from common.io import write_json  # noqa: E402
from common.prompts import Quit, ask  # noqa: E402

EXPERIMENT = "gripper_inside_tip_gap"
SETTINGS = ("fully_open", "half_open", "fully_closed")
GRIPPER_GOAL_TIME_S = 2.0  # set_gripper_position default
SETTLE_VELOCITY_M_S = 0.002
SETTLE_TIMEOUT_S = 2.0
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _commands(limits) -> dict[str, float]:
    closed = float(limits.position_min)
    open_ = float(limits.position_max)
    if open_ <= closed:
        raise ValueError(f"gripper position_max {open_} m is not above position_min {closed} m")
    return {
        "fully_closed": closed,
        "half_open": 0.5 * (closed + open_),
        "fully_open": open_,
    }


def _prompt_gap_mm() -> float:
    print(
        "Measure the gap from the inside tip of one finger to the inside tip "
        "of the other, in millimetres."
    )
    while True:
        text = ask("Gap in mm (q to quit): ").strip()
        if text.lower() in {"q", "quit"}:
            raise Quit()
        value = text.lower().removesuffix("mm").strip()
        try:
            gap = float(value)
        except ValueError:
            print("Enter a number in millimetres, or q to quit.")
            continue
        if gap < 0:
            print("The gap cannot be negative.")
            continue
        return gap


def _prompt_continue() -> None:
    while True:
        text = ask("Press Enter for the next setting (q to quit): ")
        if text.strip() == "":
            return
        if text.strip().lower() in {"q", "quit"}:
            raise Quit()
        print("Press Enter to continue, or q to quit.")


def _command_gripper(driver, position_m: float, goal_time_s: float) -> None:
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(position_m, goal_time_s, True)


def _wait_until_slow(driver) -> bool:
    deadline = time.time() + SETTLE_TIMEOUT_S
    while True:
        velocity = abs(float(driver.get_gripper_velocity()))
        if velocity <= SETTLE_VELOCITY_M_S:
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.05)


def run(args: argparse.Namespace) -> None:
    calibration = load_calibration_config(args.calibration_config)
    identity = robot_identity(calibration)
    driver = trossen_arm.TrossenArmDriver()
    payload = None
    try:
        driver.configure(
            getattr(trossen_arm.Model, identity["model"]),
            getattr(trossen_arm.StandardEndEffector, identity["end_effector"]),
            identity["ip"],
            False,
        )
        limits = driver.get_joint_limits()[-1]
        commands = _commands(limits)
        print(
            f"Connected to {identity['model']} at {identity['ip']} ({identity['end_effector']}).\n"
            "Support the arm. The next step puts the arm in gravity compensation. "
            "Only the gripper will be position-commanded."
        )
        while True:
            text = ask("Press Enter to start (q to quit): ")
            if text.strip() == "":
                break
            if text.strip().lower() in {"q", "quit"}:
                raise Quit()
            print("Press Enter to start, or q to quit.")
        hold_arm_with_gravity_compensation(driver)
        output = args.output or (
            RESULTS_DIR / f"gripper_gap_hardware_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        payload = {
            "experiment": EXPERIMENT,
            "source": "hardware",
            "coordinate_units": {
                "commanded_gripper_position": "m",
                "actual_gripper_position": "m",
                "carriage_position": "m",
                "inside_tip_gap": "mm",
            },
            "configuration": {
                "ip": identity["ip"],
                "model": identity["model"],
                "end_effector": identity["end_effector"],
                "driver_version": driver.get_driver_version(),
                "controller_version": driver.get_controller_version(),
                "calibration_config": str(args.calibration_config),
                "gripper_joint_limit_min_m": float(limits.position_min),
                "gripper_joint_limit_max_m": float(limits.position_max),
                "gripper_move_time_s": GRIPPER_GOAL_TIME_S,
                "commands_m": commands,
                "command_definition": (
                    "fully_open is the driver gripper position_max and fully_closed "
                    "is position_min, sent with set_gripper_position. half_open is "
                    "their midpoint. The coordinate is the driver's single gripper "
                    "joint position in metres, not jaw width."
                ),
                "arm_control": "external_effort 0 (gravity compensation); no arm position commands",
                "end_effector": end_effector_record(driver.get_end_effector()),
            },
            "samples": [],
        }
        write_json(output, payload)
        print(
            "The arm is in gravity compensation. Keep supporting it.\n"
            f"Commands (m): open {commands['fully_open']:.6f}, "
            f"half {commands['half_open']:.6f}, closed {commands['fully_closed']:.6f}\n"
            f"Saving to {output}"
        )
        for index, name in enumerate(SETTINGS):
            command = commands[name]
            print(f"\nMoving the gripper to {name.replace('_', ' ')}: {command:.6f} m")
            _command_gripper(driver, command, GRIPPER_GOAL_TIME_S)
            settled = _wait_until_slow(driver)
            print(f"Setting: {name}")
            print(f"Commanded gripper position: {command:.6f} m")
            if not settled:
                print(
                    f"The gripper was still faster than {SETTLE_VELOCITY_M_S} m/s "
                    f"after {SETTLE_TIMEOUT_S:.0f} s. It will stay at this command while you measure."
                )
            gap_mm = _prompt_gap_mm()
            actual = float(driver.get_gripper_position())
            sample = {
                "setting": name,
                "commanded_gripper_position": command,
                "actual_gripper_position": actual,
                "actual_left_carriage_position": None,
                "actual_right_carriage_position": None,
                "inside_tip_gap_mm": gap_mm,
                "inside_tip_gap_source": "manual_inside_fingertip_measurement",
                "settled": settled,
                "timestamp": datetime.now().astimezone().isoformat(),
            }
            payload["samples"].append(sample)
            write_json(output, payload)
            print("Saved sample:")
            print(f"  setting: {name}")
            print(f"  commanded gripper position: {command:.6f} m")
            print(f"  hardware-reported gripper position: {actual:.6f} m")
            print(f"  measured inside-tip gap: {gap_mm:.3f} mm")
            print(f"  settled: {settled}")
            if index < len(SETTINGS) - 1:
                _prompt_continue()
        print(f"\nFinished. {len(payload['samples'])} samples in {output}")
    except Quit:
        saved = 0 if payload is None else len(payload["samples"])
        print(f"\nStopped. {saved} sample(s) saved.")
    finally:
        release_to_idle(driver)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration-config", type=Path, default=CALIBRATION_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.calibration_config.is_file():
        raise SystemExit(f"calibration config not found: {args.calibration_config}")
    run(args)


if __name__ == "__main__":
    main()
