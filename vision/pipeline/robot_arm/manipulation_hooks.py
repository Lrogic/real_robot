"""Choose the track_live hooks for --manipulation MODE (needs real_robot/ on PYTHONPATH; run.sh sets it)."""


def manipulation_hooks(args, recovery=None, camera=None):
    """Context manager yielding started hooks for args.manipulation: on_mouse, after_recovery
    (after SAM recovery, or after FoundationPose without --recovery), on_key, draw, close, and
    optionally before_recovery (after FoundationPose, before SAM recovery). camera is the held
    camera arm's CameraPose (pipeline/robot_arm/camera_arm.py): its T_base_camera places the objects."""
    if camera is None:
        raise ValueError("manipulation needs the camera pose from the camera arm setup (--camera)")
    if args.manipulation == "basic":
        from manipulation.basic_manipulation.session import manipulation_session

        return manipulation_session(args, camera)
    if args.manipulation == "policy":
        from manipulation.policy.session import policy_session

        return policy_session(args, recovery, camera)
    raise ValueError(f"Unknown manipulation mode {args.manipulation!r}")
