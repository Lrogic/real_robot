"""Run follower reset from shared pipeline flags (used by run.sh before segment_objects in fp_robot)."""

from fp_capture.cli import pipeline_parser, parse_pipeline_args
from fp_capture.follower_reset import ensure_vision_on_path, run_follower_reset_if_requested

ensure_vision_on_path()


def main() -> int:
    parser = pipeline_parser(__doc__)
    args = parse_pipeline_args(parser)
    run_follower_reset_if_requested(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
