"""Command-line options shared by segment_objects.py and track_objects.py.

Both scripts accept the full option set so run.sh can forward one flat argument
list to each stage; every stage ignores the options that belong to the other.
"""

import argparse
from pathlib import Path

from .camera import CameraSettings

RESEARCH_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MESHES_DIR = RESEARCH_DIR / "assets" / "meshes"
DEFAULT_FOUNDATIONPOSE_DIR = RESEARCH_DIR / "FoundationPose"
DEFAULT_SAM3_CHECKPOINT = Path("/bigdata/luka/models/sam3/sam3.pt")


def pipeline_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-path", type=Path, required=True,
                        help="Run folder: created by setup (must be missing or empty), read by tracking")

    camera = parser.add_argument_group("camera")
    camera.add_argument("--settle-frames", type=int, default=30,
                        help="Frames to discard after camera startup (default: 30)")
    camera.add_argument("--width", type=int, default=640)
    camera.add_argument("--height", type=int, default=480)
    camera.add_argument("--fps", type=int, default=30)

    setup = parser.add_argument_group("setup (segment_objects.py)")
    setup.add_argument("--meshes-dir", type=Path, default=DEFAULT_MESHES_DIR,
                       help=f"Folder whose subfolders are mesh options (default: {DEFAULT_MESHES_DIR})")
    setup.add_argument("--checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT,
                       help=f"SAM3 checkpoint (default: {DEFAULT_SAM3_CHECKPOINT})")

    track = parser.add_argument_group("tracking (track_objects.py)")
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
    track.add_argument("--foundationpose-dir", type=Path, default=DEFAULT_FOUNDATIONPOSE_DIR,
                       help=f"FoundationPose checkout (default: {DEFAULT_FOUNDATIONPOSE_DIR})")
    return parser


def parse_pipeline_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if args.settle_frames < 0 or args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("settle-frames must be >= 0; width, height, fps must be > 0")
    if args.num_seconds is not None and args.num_seconds <= 0:
        parser.error("num-seconds must be > 0")
    if args.stats_every <= 0:
        parser.error("stats-every must be > 0")
    return args


def camera_settings(args: argparse.Namespace) -> CameraSettings:
    return CameraSettings(args.width, args.height, args.fps, args.settle_frames)
