"""Choose the track_live hooks for --manipulation MODE (needs real_robot/ on PYTHONPATH; run.sh sets it)."""


def manipulation_hooks(args, recovery=None):
    """Context manager yielding started hooks for args.manipulation: on_mouse, after_recovery
    (after SAM recovery, or after FoundationPose without --recovery), on_key, draw, close, and
    optionally before_recovery (after FoundationPose, before SAM recovery)."""
    if args.manipulation == "basic":
        from manipulation.basic_manipulation.session import manipulation_session

        return manipulation_session(args)
    if args.manipulation == "policy":
        from manipulation.policy.session import policy_session

        return policy_session(args, recovery)
    raise ValueError(f"Unknown manipulation mode {args.manipulation!r}")
