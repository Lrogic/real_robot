"""Choose the track_live hooks for --manipulation MODE (needs real_robot/ on PYTHONPATH; run.sh sets it)."""


def manipulation_hooks(args):
    """Context manager yielding started hooks (on_mouse, update, on_key, draw, close) for args.manipulation."""
    if args.manipulation == "basic":
        from manipulation.basic_manipulation.session import manipulation_session

        return manipulation_session(args)
    raise ValueError(f"Unknown manipulation mode {args.manipulation!r}")
