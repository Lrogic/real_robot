"""Read the follower once and save its parked pose.

    python -m calibration.record_follower [--out data/followers/NAME.json]

Connects to config follower.ip, reads joint and Cartesian state, and writes JSON.
Does not command the arm. Disconnecting sets it to idle, so support it.
The Cartesian pose is in the follower base, not the manipulator base.
"""

import argparse
import time
from pathlib import Path

from .board import DEFAULT_DATA_DIR, load_config
from .session import read_json, write_json

UNRECORDED = "unrecorded"
FOLLOWERS_DIR = DEFAULT_DATA_DIR / "followers"


def load_follower_choice(text: str) -> dict:
    """A saved profile path, or the keyword 'unrecorded'."""
    if text == UNRECORDED:
        return {"status": UNRECORDED, "profile_path": None, "profile": None,
                "joint_positions_rad": None}
    path = Path(text)
    if not path.is_file():
        raise SystemExit(f"Follower profile not found: {path} (or pass {UNRECORDED!r})")
    profile = read_json(path)
    try:
        joints = profile["state"]["arm_positions_rad"]
    except (KeyError, TypeError):
        raise SystemExit(f"{path} is not a follower profile (missing state.arm_positions_rad)")
    return {"status": "recorded", "profile_path": str(path.resolve()), "profile": profile,
            "joint_positions_rad": joints}


def read_follower(follower_cfg: dict) -> dict:
    """Configure the follower, read one state, then disconnect (arm goes idle)."""
    import trossen_arm

    from .robot import RobotState

    c = follower_cfg
    driver = trossen_arm.TrossenArmDriver()
    configured = False
    try:
        driver.configure(getattr(trossen_arm.Model, c["model"]),
                         getattr(trossen_arm.StandardEndEffector, c["end_effector"]),
                         c["ip"], False)
        configured = True
        state = RobotState.from_output(driver.get_robot_output())
        return {
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ip": c["ip"],
            "model": c["model"],
            "end_effector": c["end_effector"],
            "num_arm_joints": driver.get_num_joints() - 1,
            "driver_version": driver.get_driver_version(),
            "controller_version": driver.get_controller_version(),
            "state": state.to_json(),
            "frame": "follower base; cartesian and T_base_ee are this arm's FK, not T_base_camera",
            "note": "Parked pose the eye-to-hand result is valid for. The arm was not commanded.",
        }
    finally:
        if configured:
            try:
                input("\nDisconnecting sets the follower to idle. SUPPORT THE ARM, then press Enter... ")
            except (EOFError, KeyboardInterrupt):
                pass
            driver.cleanup()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="profile JSON (default data/followers/<timestamp>.json)")
    args = p.parse_args()
    config = load_config(args.config)
    profile = read_follower(config["follower"])
    out = args.out or FOLLOWERS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, profile)
    state = profile["state"]
    print(f"Wrote {out}")
    print(f"joints rad: {state['arm_positions_rad']}")
    print(f"cartesian xyz m, rotvec rad: {state['cartesian_xyz_m_rotvec_rad']}")
    print(f"Pass this to collect: --follower {out}")


if __name__ == "__main__":
    main()
