"""Reset the follower (camera) arm before RGB-D capture when --reset is set."""

from calibration.reset_follower import reset_follower_from_session_dir
from calibration.session import find_calibration


def run_follower_reset_if_requested(args) -> None:
    """Park the camera arm at the joints recorded in the calibration session (same folder as T_base_camera.json)."""
    if not getattr(args, "reset", False):
        return
    cal_path = find_calibration(getattr(args, "calibration", None))
    move_s = getattr(args, "follower_reset_move_time", 2.0)
    cal_cfg = getattr(args, "calibration_config", None)
    print(f"Resetting follower to parked pose from {cal_path.parent} (before camera capture)...")
    try:
        reset_follower_from_session_dir(cal_path.parent, cal_cfg, move_s, prompt_before_idle=False)
    except Exception as e:
        raise SystemExit(f"could not reset follower arm (--reset): {e}") from e
