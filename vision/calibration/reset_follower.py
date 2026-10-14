"""Move the follower back to the joint pose stored in a calibration session.

    python -m calibration.reset_follower data/sessions/20261009_combined

Reads joint targets from session.json, connects using config follower (ip/model/EE),
commands the recorded arm_positions_rad in position mode, then disconnects (idle).
Use this to restore the parked pose your eye-to-hand result was collected with.
"""

import argparse
from pathlib import Path

import numpy as np

from .board import load_config
from .session import read_json


def session_follower_joints(session: dict) -> list[float]:
    """Joint targets (rad) from a session.json dict."""
    follower = session.get("follower")
    if follower is None:
        raise ValueError("session.json has no follower block")

    joints = follower.get("joint_positions_rad")
    if joints is None:
        cam = session.get("camera_arm") or {}
        joints = cam.get("joint_positions_rad")
    if joints is None or follower.get("status") == "unrecorded":
        raise ValueError("session has no recorded follower joint positions (was --follower unrecorded?)")
    return [float(q) for q in joints]


def move_follower_to_parked_pose(
    follower_cfg: dict,
    joint_positions_rad: list[float],
    move_time_s: float,
    *,
    prompt_before_idle: bool = True,
) -> None:
    import trossen_arm

    driver = trossen_arm.TrossenArmDriver()
    configured = False
    try:
        c = follower_cfg
        driver.configure(getattr(trossen_arm.Model, c["model"]),
                         getattr(trossen_arm.StandardEndEffector, c["end_effector"]),
                         c["ip"], False)
        configured = True
        n_arm = driver.get_num_joints() - 1
        if len(joint_positions_rad) != n_arm:
            raise ValueError(f"session has {len(joint_positions_rad)} arm joints, driver expects {n_arm}")

        print(f"Follower {c['ip']}: moving to session joints (rad) {joint_positions_rad}")
        driver.set_arm_modes(trossen_arm.Mode.position)
        driver.set_arm_positions(joint_positions_rad, move_time_s, True)

        reached = np.array(driver.get_arm_positions(), dtype=np.float64)
        target = np.asarray(joint_positions_rad, dtype=np.float64)
        err = float(np.abs(reached - target).max())
        print(f"Reached joints (rad): {reached.tolist()}")
        print(f"Max joint error vs session: {err:.4f} rad ({np.degrees(err):.2f} deg)")
    finally:
        if configured:
            if prompt_before_idle:
                try:
                    input("\nDisconnecting sets the follower to idle. SUPPORT THE ARM, then press Enter... ")
                except (EOFError, KeyboardInterrupt):
                    pass
            driver.cleanup()


def reset_follower_from_session_dir(
    session_dir: Path,
    calibration_config: Path | None = None,
    move_time_s: float = 2.0,
    *,
    prompt_before_idle: bool = True,
) -> None:
    """Move the config follower to the joint pose in session_dir/session.json."""
    session_path = Path(session_dir) / "session.json"
    if not session_path.is_file():
        raise FileNotFoundError(f"not a calibration session: {session_dir} (no session.json)")
    joints = session_follower_joints(read_json(session_path))
    move_follower_to_parked_pose(
        load_config(calibration_config)["follower"],
        joints,
        move_time_s,
        prompt_before_idle=prompt_before_idle,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path,
                   help="calibration session directory (must contain session.json)")
    p.add_argument("--config", default=None, help="calibration config (follower ip/model/EE)")
    p.add_argument("--move-time", type=float, default=2.0, metavar="SEC",
                   help="position move duration (default 2)")
    args = p.parse_args()

    try:
        reset_follower_from_session_dir(
            args.session.resolve(),
            args.config,
            args.move_time,
            prompt_before_idle=True,
        )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
