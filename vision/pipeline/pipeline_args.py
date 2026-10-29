"""Command-line options shared by setup_stage.py and track_stage.py.

Both scripts accept the full option set so run.sh can forward one flat argument
list to each stage; every stage ignores the options that belong to the other.
"""

import argparse
from pathlib import Path

from .camera import CameraSettings
from .recovery.recovery_config import DEFAULT_RECOVERY_CONFIG, load_recovery_config
from .segmentation.sam_model_specs import SAM_MODEL_NAMES

RESEARCH_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MESHES_DIR = RESEARCH_DIR / "assets" / "meshes"
DEFAULT_FOUNDATIONPOSE_DIR = RESEARCH_DIR / "FoundationPose"
# Camera defaults without --recovery; --recovery takes them from the recovery config.
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS = 640, 480, 30
MANIPULATION_MODES = ("basic", "policy")
RECOVERY_ONLY_FLAGS = {"sam_model": "--sam-model", "sam_fps": "--sam-fps",
                       "iou_threshold": "--iou-threshold", "failure_limit": "--failure-limit",
                       "sam_hf_cache": "--sam-hf-cache", "recovery_config": "--recovery-config"}


def pipeline_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-path", type=Path, required=True,
                        help="Run folder: created by setup (must be missing or empty), read by tracking")

    camera = parser.add_argument_group("camera")
    camera.add_argument("--settle-frames", type=int, default=30,
                        help="Frames to discard after camera startup (default: 30)")
    camera.add_argument("--width", type=int, default=None,
                        help=f"(default: {DEFAULT_WIDTH}; with --recovery, the recovery config's)")
    camera.add_argument("--height", type=int, default=None,
                        help=f"(default: {DEFAULT_HEIGHT}; with --recovery, the recovery config's)")
    camera.add_argument("--fps", type=int, default=None,
                        help=f"(default: {DEFAULT_FPS}; with --recovery, the recovery config's)")
    camera.add_argument("--undistort", dest="undistort", action="store_true",
                        help="Undistort RGB and depth so FoundationPose's pinhole K is exact "
                             "(default: off, RealSense's distorted frames as before)")

    setup = parser.add_argument_group("setup (setup_stage.py)")
    setup.add_argument("--meshes-dir", type=Path, default=DEFAULT_MESHES_DIR,
                       help=f"Folder whose subfolders are mesh options (default: {DEFAULT_MESHES_DIR})")

    track = parser.add_argument_group("tracking (track_stage.py)")
    track.add_argument("--num-seconds", type=float, default=None,
                       help="Stop tracking after this many seconds (default: until q/Esc)")
    track.add_argument("--fp-mode", choices=("sequential", "batched", "batched-render"),
                       default="sequential",
                       help="sequential: FoundationPose track_one per object (default); "
                            "batched: one refiner pass per iteration for all objects; "
                            "batched-render: batched, plus one render call for all objects")
    track.add_argument("--est-refine-iter", type=int, default=5)
    track.add_argument("--track-refine-iter", type=int, default=2)
    track.add_argument("--save-video", action="store_true",
                       help="Buffer displayed frames in RAM and write save-path/tracking.mp4 at the end")
    track.add_argument("--stats-every", type=float, default=5.0,
                       help="Seconds between live timing reports (default: 5)")
    track.add_argument("--log-variation", action="store_true",
                       help="Record every tracked pose and write save-path/pose_variation.txt "
                            "when tracking stops (mean, std, percentiles, and a 1s pose log)")
    track.add_argument("--foundationpose-dir", type=Path, default=DEFAULT_FOUNDATIONPOSE_DIR,
                       help=f"FoundationPose checkout (default: {DEFAULT_FOUNDATIONPOSE_DIR})")

    recovery = parser.add_argument_group(
        "recovery (--recovery: config's SAM model at setup; SAM-vs-FP comparison and re-registration while tracking)")
    recovery.add_argument("--recovery", action="store_true",
                          help="Segment setup masks with the recovery config's HF SAM model instead of sam3 "
                               "and, while tracking, re-register an object after consecutive SAM/FP "
                               "silhouette mismatches")
    recovery.add_argument("--recovery-config", type=Path, default=None,
                          help=f"Recovery settings (default: {DEFAULT_RECOVERY_CONFIG})")
    recovery.add_argument("--sam-model", choices=SAM_MODEL_NAMES, default=None,
                          help="HF SAM video tracker (default: from the recovery config)")
    recovery.add_argument("--sam-fps", type=float, default=None,
                          help="Maximum SAM request rate while tracking (default: from the recovery config)")
    recovery.add_argument("--iou-threshold", type=float, default=None,
                          help="IoU below this counts as a mismatch (default: from the recovery config)")
    recovery.add_argument("--failure-limit", type=int, default=None,
                          help="Consecutive mismatches that trigger re-registration "
                               "(default: from the recovery config)")
    recovery.add_argument("--sam-hf-cache", type=Path, default=None,
                          help="Offline HF cache holding the SAM weights (default: from the recovery config)")

    camera_arm = parser.add_argument_group("camera arm (setup_stage.py --reset before capture)")
    camera_arm.add_argument("--reset", action="store_true",
                            help="Move the follower arm to the calibration session's recorded joints "
                                 "before the setup RGB-D frame (fails if unreachable)")
    camera_arm.add_argument("--calibration", type=Path, default=None,
                            help="T_base_camera.json (default: newest calibration session that has one)")
    camera_arm.add_argument("--calibration-config", type=Path, default=None,
                            help="Calibration config for follower ip/model (--reset)")
    camera_arm.add_argument("--follower-reset-move-time", type=float, default=2.0, metavar="SEC",
                            help="Follower position move duration when using --reset (default: 2)")

    manip = parser.add_argument_group("manipulation (track_stage.py --manipulation MODE)")
    manip.add_argument("--manipulation", choices=MANIPULATION_MODES, default=None,
                       help="basic: click a tracked object to make the manipulator move to and follow it; "
                            "policy: build the sim policy's observations at its control rate (needs --recovery)")
    manip.add_argument("--manipulation-dry-run", action="store_true",
                       help="With --manipulation: draw and log without connecting to the arm")
    manip.add_argument("--manipulation-config", type=Path, default=None,
                       help="Manipulation config (default: basic_manipulation/config.yaml for basic, "
                            "policy/jenga/config.yaml for policy, under real_robot/manipulation)")
    manip.add_argument("--manipulation-logging", action="store_true",
                       help="With --manipulation policy: write save-path/policy_log.jsonl")
    return parser


def parse_pipeline_args(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    args = parser.parse_args(argv)
    args.recovery_cfg = None
    if args.recovery:
        try:
            args.recovery_cfg = load_recovery_config(
                args.recovery_config, sam_model=args.sam_model, sam_fps=args.sam_fps,
                iou_threshold=args.iou_threshold, failure_limit=args.failure_limit,
                hf_cache=args.sam_hf_cache)
        except (OSError, ValueError) as error:
            parser.error(f"recovery config: {error}")
        defaults = (args.recovery_cfg.camera_width, args.recovery_cfg.camera_height,
                    args.recovery_cfg.camera_fps)
    else:
        given = [flag for dest, flag in RECOVERY_ONLY_FLAGS.items() if getattr(args, dest) is not None]
        if given:
            parser.error(f"{', '.join(given)} {'requires' if len(given) == 1 else 'require'} --recovery")
        defaults = (DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS)
    for dest, default in zip(("width", "height", "fps"), defaults):
        if getattr(args, dest) is None:
            setattr(args, dest, default)
    if args.settle_frames < 0 or args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("settle-frames must be >= 0; width, height, fps must be > 0")
    if args.num_seconds is not None and args.num_seconds <= 0:
        parser.error("num-seconds must be > 0")
    if args.stats_every <= 0:
        parser.error("stats-every must be > 0")
    if args.manipulation_dry_run and not args.manipulation:
        parser.error("--manipulation-dry-run requires --manipulation")
    if args.manipulation_logging and args.manipulation != "policy":
        parser.error("--manipulation-logging requires --manipulation policy")
    if args.manipulation == "policy" and not args.recovery:
        parser.error("--manipulation policy requires --recovery: pose trust comes from the SAM verdicts")
    if args.follower_reset_move_time <= 0:
        parser.error("--follower-reset-move-time must be > 0")
    if args.recovery_cfg is not None:
        if args.recovery_cfg.sam_fps > args.fps:
            parser.error(f"SAM fps {args.recovery_cfg.sam_fps} cannot exceed camera fps {args.fps}")
        print(f"Recovery configuration: {resolved_recovery_settings(args)}")
    return args


def resolved_recovery_settings(args: argparse.Namespace) -> dict:
    """Recovery config after CLI overrides, plus the camera and FP settings actually used."""
    settings = {k: v for k, v in args.recovery_cfg.describe().items() if not k.startswith("camera_")}
    return {**settings,
            "camera": {"width": args.width, "height": args.height, "fps": args.fps,
                       "undistort": args.undistort, "settle_frames": args.settle_frames},
            "fp_mode": args.fp_mode, "est_refine_iter": args.est_refine_iter,
            "track_refine_iter": args.track_refine_iter}


def camera_settings(args: argparse.Namespace) -> CameraSettings:
    return CameraSettings(args.width, args.height, args.fps, args.settle_frames, args.undistort)
