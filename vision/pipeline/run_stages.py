"""run.sh entry point: the setup stage, the track stage, or both, in one process.

    python -m pipeline.run_stages setup|track|setup-track --save-path RUN --camera parked|move [...]

The camera arm is set up first (pipeline/robot_arm/camera_arm.py) and held until the process
exits, so with setup-track the follower never lapses to idle between masking and tracking.
Every option after the stage is the shared pipeline option set (pipeline/pipeline_args.py).
"""

import sys

from .pipeline_args import camera_settings, parse_pipeline_args, pipeline_parser
from .robot_arm.camera_arm import held_camera
from .run_folder import load_run
from .setup_stage import check_setup_args, run_setup
from .track_stage import run_track, saved_camera_pose

STAGES = ("setup", "track", "setup-track")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in STAGES:
        sys.exit(f"usage: python -m pipeline.run_stages {{{','.join(STAGES)}}} [options]")
    stage, argv = argv[0], argv[1:]
    parser = pipeline_parser(__doc__)
    args = parse_pipeline_args(parser, argv)
    setup, track = stage in ("setup", "setup-track"), stage in ("track", "setup-track")
    saved = None
    if setup:
        check_setup_args(parser, args)
    else:
        try:
            run = load_run(args.save_path)
            saved = saved_camera_pose(args)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))

    with held_camera(args, camera_settings(args), saved) as camera:
        if setup:
            print("[setup] pipeline.setup_stage")
            if not run_setup(args, camera):
                return 0
        if track:
            if setup:
                run = load_run(args.save_path)
            print("[track] pipeline.track_stage")
            return run_track(args, run, camera)
    return 0


if __name__ == "__main__":
    sys.exit(main())
