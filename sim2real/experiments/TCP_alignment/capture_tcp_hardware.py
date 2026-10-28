"""Record hardware joint states and the driver tool pose for the TCP comparison.

The arm stays in gravity compensation (zero external effort). Nothing is
position-commanded. For each pose, guide the arm by hand, hold it still, and
press Enter. One ``get_robot_output()`` call supplies the joint positions,
joint velocities, Cartesian pose, and header timestamp.

Collect 10 to 15 poses. Change the base rotation, the reach, the height, and
the wrist orientation. Include poses where the wrist has turned a long way: a
TCP offset can agree in one wrist pose and disagree after the wrist rotates.

The file is the input to ``compare_tcp_fk.py``.

Launch from the fp_robot environment:

  conda activate fp_robot
  python real_robot/sim2real/experiments/TCP_alignment/capture_tcp_hardware.py
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

EXPERIMENT = "tcp_alignment"
TARGET_MIN = 10
TARGET_MAX = 15
WRIST_SPAN_NOTE_DEG = 90.0
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _snapshot(driver, n_arm: int) -> dict:
    """Joint, Cartesian, and header fields from one robot-output call."""
    out = driver.get_robot_output()
    host_time = time.time()
    arm_positions = [float(v) for v in out.joint.arm.positions]
    arm_velocities = [float(v) for v in out.joint.arm.velocities]
    cartesian = [float(v) for v in out.cartesian.positions]
    cartesian_velocity = [float(v) for v in out.cartesian.velocities]
    if len(arm_positions) != n_arm or len(arm_velocities) != n_arm:
        raise RuntimeError(
            f"expected {n_arm} arm joints, got {len(arm_positions)} positions "
            f"and {len(arm_velocities)} velocities"
        )
    if len(cartesian) != 6 or len(cartesian_velocity) != 6:
        raise RuntimeError(
            f"expected a 6-vector Cartesian pose and velocity, got {len(cartesian)} "
            f"and {len(cartesian_velocity)}"
        )
    return {
        "header_id": int(out.header.id),
        "header_timestamp_us": int(out.header.timestamp),
        "host_time_s": host_time,
        "host_time_iso": datetime.fromtimestamp(host_time).astimezone().isoformat(),
        "arm_joint_positions_rad": arm_positions,
        "arm_joint_velocities_rad_s": arm_velocities,
        "gripper_position_m": float(out.joint.gripper.position),
        "gripper_velocity_m_s": float(out.joint.gripper.velocity),
        "cartesian_positions_m_rotvec_rad": cartesian,
        "cartesian_velocities_m_s_rad_s": cartesian_velocity,
    }


def _prompt_capture(saved: int) -> None:
    if saved == 0:
        prompt = "Hold the arm still and press Enter to capture (q to quit): "
    elif saved >= TARGET_MAX:
        prompt = f"{saved} samples saved. Press Enter for another, or q to finish: "
    else:
        prompt = (
            f"{saved} saved, aim for {TARGET_MIN}-{TARGET_MAX}. "
            "Hold still and press Enter (q to finish): "
        )
    while True:
        text = ask(prompt)
        if text.strip() == "":
            return
        if text.strip().lower() in {"q", "quit"}:
            raise Quit()
        print("Press Enter to capture, or q to finish.")


def _print_sample(sample: dict) -> None:
    degrees = np.degrees(sample["arm_joint_positions_rad"])
    position_mm = np.array(sample["cartesian_positions_m_rotvec_rad"][:3]) * 1000.0
    joints = ", ".join(f"{value:.1f}" for value in degrees)
    xyz = ", ".join(f"{value:.1f}" for value in position_mm)
    print(f"Saved sample {sample['sample']}")
    print(f"  arm joints (deg): {joints}")
    print(f"  tool position (mm, base frame): {xyz}")
    print(f"  header id {sample['header_id']}, timestamp {sample['header_timestamp_us']} us")


def _wrist_note(samples: list[dict]) -> None:
    if len(samples) < 2:
        print("One wrist orientation cannot show a TCP offset that depends on wrist angle.")
        return
    wrist = [sample["arm_joint_positions_rad"][-1] for sample in samples]
    span = float(np.degrees(max(wrist) - min(wrist)))
    print(f"Wrist joint (last arm joint) spanned {span:.1f} deg across these samples.")
    if span < WRIST_SPAN_NOTE_DEG:
        print(
            f"That span is under {WRIST_SPAN_NOTE_DEG:.0f} deg. "
            "Add poses with a large wrist rotation before treating the alignment as checked."
        )


def _finish(payload: dict | None, output: Path | None, interrupted: bool) -> None:
    saved = 0 if payload is None else len(payload["samples"])
    if interrupted or saved == 0:
        where = "" if output is None else f" in {output}"
        print(f"\nStopped. {saved} sample(s) saved{where}.")
    else:
        print(f"\nFinished. {saved} samples in {output}.")
    if payload is not None and payload["samples"]:
        _wrist_note(payload["samples"])


def run(args: argparse.Namespace) -> None:
    calibration = load_calibration_config(args.calibration_config)
    identity = robot_identity(calibration)
    driver = trossen_arm.TrossenArmDriver()
    payload = None
    output = None
    try:
        driver.configure(
            getattr(trossen_arm.Model, identity["model"]),
            getattr(trossen_arm.StandardEndEffector, identity["end_effector"]),
            identity["ip"],
            False,
        )
        print(
            f"Connected to {identity['model']} at {identity['ip']} ({identity['end_effector']}).\n"
            "Support the arm. The next step puts the arm in gravity compensation.\n"
            "Guide it by hand. Vary the base rotation, reach, height, and wrist "
            "orientation, and include large wrist rotations.\n"
            f"Capture {TARGET_MIN} to {TARGET_MAX} poses. q finishes and keeps what was saved."
        )
        while True:
            text = ask("Press Enter to start (q to quit): ")
            if text.strip() == "":
                break
            if text.strip().lower() in {"q", "quit"}:
                raise Quit()
            print("Press Enter to start, or q to quit.")
        n_arm = hold_arm_with_gravity_compensation(driver)
        output = args.output or (
            RESULTS_DIR / f"tcp_hardware_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        payload = {
            "experiment": EXPERIMENT,
            "source": "hardware",
            "coordinate_units": {
                "arm_joint_position": "rad",
                "arm_joint_velocity": "rad/s",
                "gripper_position": "m",
                "gripper_velocity": "m/s",
                "cartesian_position": "m",
                "cartesian_orientation": "rotvec_rad",
                "cartesian_linear_velocity": "m/s",
                "cartesian_angular_velocity": "rad/s",
                "header_timestamp": "us_since_configuration",
            },
            "configuration": {
                "ip": identity["ip"],
                "model": identity["model"],
                "end_effector": identity["end_effector"],
                "driver_version": driver.get_driver_version(),
                "controller_version": driver.get_controller_version(),
                "calibration_config": str(args.calibration_config),
                "num_arm_joints": n_arm,
                "arm_control": "external_effort 0 (gravity compensation); no position commands",
                "snapshot_source": (
                    "Each sample is one get_robot_output() call: header, arm joints, "
                    "gripper joint, and Cartesian pose and velocity. host_time_s is the "
                    "host clock immediately after that call returned. Cartesian positions "
                    "are the driver end-effector frame in the driver base frame: "
                    "translation in metres, then angle-axis rotation in radians."
                ),
                "end_effector_properties": end_effector_record(driver.get_end_effector()),
            },
            "samples": [],
        }
        write_json(output, payload)
        print(f"The arm is in gravity compensation. Keep supporting it.\nSaving to {output}")
        while True:
            _prompt_capture(len(payload["samples"]))
            reading = _snapshot(driver, n_arm)
            sample = {"sample": len(payload["samples"]) + 1, **reading}
            payload["samples"].append(sample)
            write_json(output, payload)
            _print_sample(sample)
    except Quit:
        _finish(payload, output, interrupted=False)
    except KeyboardInterrupt:
        _finish(payload, output, interrupted=True)
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
