"""Reset the follower (camera) arm before RGB-D capture when --reset is set."""

import sys
from pathlib import Path


def run_follower_reset_if_requested(args) -> None:
    if not getattr(args, "reset", False):
        return
    from manipulation.session import find_calibration, reset_follower_for_calibration

    cal_path = find_calibration(getattr(args, "calibration", None))
    move_s = getattr(args, "follower_reset_move_time", 2.0)
    cal_cfg = getattr(args, "calibration_config", None)
    print(f"Resetting follower to parked pose from {cal_path.parent} (before camera capture)...")
    try:
        reset_follower_for_calibration(cal_path, cal_cfg, move_s)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e


def ensure_vision_on_path() -> None:
    """calibration and manipulation imports need real_robot/vision on sys.path."""
    vision = Path(__file__).resolve().parents[1]
    if str(vision) not in sys.path:
        sys.path.insert(0, str(vision))
