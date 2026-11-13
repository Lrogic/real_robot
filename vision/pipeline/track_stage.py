"""Track every saved object that has a mesh, live, with FoundationPose.

Reads a run folder written by the setup stage, registers each object on the saved
RGB-D frame using its mask, then tracks all of them sequentially on each live frame.
Objects without a mesh are skipped. Run from real_robot/vision in the vision env
(run.sh --track does all of this).

    python -m pipeline.track_stage --save-path foundationpose_runs/run1 --camera parked
    python -m pipeline.track_stage --save-path foundationpose_runs/run1 --num-seconds 30 --save-video
    python -m pipeline.track_stage --save-path foundationpose_runs/run1 --fp-mode batched
    python -m pipeline.track_stage --save-path foundationpose_runs/run1 --manipulation basic [--manipulation-dry-run]
    python -m pipeline.track_stage --save-path foundationpose_runs/run1 --log-variation
    python -m pipeline.track_stage --save-path foundationpose_runs/run1 --recovery [--sam-model sam2-small]

--recovery also tracks every object with an HF SAM video tracker, started from the saved
frame and masks, and re-registers an object from its SAM mask after consecutive SAM-vs-FP
silhouette mismatches; see pipeline/recovery/sam_recovery.py and pipeline/recovery/recovery.yaml.

--manipulation basic adds the arm follow test (see
real_robot/manipulation/basic_manipulation/session.py for keys); it needs real_robot/ on
PYTHONPATH, which run.sh sets.

--camera must match the run's setup (camera_pose.json): parked re-parks the follower, move
returns it to the joints the setup locked; it is held until tracking ends. Do not move the
objects between setup and the start of tracking.
Press q or Esc in the window to stop.
"""

import sys
import tempfile
import time

from .camera import RealSenseStream
from .pipeline_args import camera_settings, parse_pipeline_args, pipeline_parser
from .robot_arm.camera_arm import held_camera
from .run_folder import RECOVERY_LOG_FILE, RGB_FILE, load_camera_pose, load_run
from .tracking_loop import check_live_camera, track_live, trackable_objects


def saved_camera_pose(args):
    """The run's camera_pose.json; raises ValueError if --camera disagrees with how the run was set up."""
    saved = load_camera_pose(args.save_path)
    if saved is None:
        if args.camera == "move":
            raise ValueError(f"{args.save_path} has no camera_pose.json (set up before --camera, so parked); "
                             "track it with --camera parked")
        print(f"Note: {args.save_path} has no camera_pose.json; assuming it was set up parked.")
    elif saved["mode"] != args.camera:
        raise ValueError(f"{args.save_path} was set up with --camera {saved['mode']}, not {args.camera}")
    return saved


def main():
    parser = pipeline_parser(__doc__)
    args = parse_pipeline_args(parser)
    try:
        run = load_run(args.save_path)
        saved = saved_camera_pose(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    with held_camera(args, camera_settings(args), saved) as camera:
        return run_track(args, run, camera)


def run_track(args, run, camera) -> int:
    """Track the run's objects live with the camera arm already held at camera (a CameraPose)."""
    objects = trackable_objects(run)
    if not objects:
        print(f"Nothing to track in {args.save_path}: no saved object has a usable mesh")
        return 0

    from .foundationpose.multi_object_fp_tracker import MultiObjectTracker

    with tempfile.TemporaryDirectory(prefix="fp_debug_") as debug_dir:
        tracker = MultiObjectTracker(objects, args.foundationpose_dir, debug_dir,
                                     batched=args.fp_mode == "batched",
                                     batched_render=args.fp_mode == "batched-render")
        print(f"FoundationPose tracking mode: {args.fp_mode}")
        registered_at = time.perf_counter()
        tracker.register(objects, run.capture.rgb, run.capture.depth, run.capture.K,
                         args.est_refine_iter)
        fp_register_s = time.perf_counter() - registered_at
        recovery = None
        if args.recovery:
            from .pipeline_args import resolved_recovery_settings
            from .recovery.sam_recovery import SamRecovery
            recovery = SamRecovery(args.recovery_cfg, tracker, objects, args.est_refine_iter,
                                   args.save_path / RECOVERY_LOG_FILE, resolved_recovery_settings(args))
            recovery.init_timing["fp_initial_registration_s"] = fp_register_s
            recovery.initialize(run.capture.rgb, objects, f"saved frame {args.save_path / RGB_FILE}")
        try:
            with RealSenseStream(camera_settings(args)) as stream:
                check_live_camera(run.capture.K, run.capture.depth.shape, run.capture.undistorted, stream)
                if args.manipulation:
                    from .robot_arm.manipulation_hooks import manipulation_hooks
                    with manipulation_hooks(args, recovery, camera) as session:
                        track_live(args, tracker, stream, hooks=session, recovery=recovery)
                else:
                    track_live(args, tracker, stream, recovery=recovery)
        finally:
            if recovery is not None:
                recovery.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
