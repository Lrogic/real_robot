"""Capture one RealSense RGB-D frame and interactively segment objects with SAM3.

Each saved object gets its own obj{i} folder under --save-path, optionally with a
copy of a mesh folder from --meshes-dir. See fp_capture/export.py for the layout.

    python segment_objects.py --save-path foundationpose_runs/run1
"""

import argparse
from pathlib import Path

from fp_capture.camera import CameraSettings, capture_frame
from fp_capture.export import RunWriter, ensure_empty_dir
from fp_capture.segmenter import SAM3_CHECKPOINT, Sam3Segmenter
from fp_capture.ui import SegmentationApp

DEFAULT_MESHES_DIR = Path(__file__).resolve().parent.parent / "assets" / "meshes"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-path", type=Path, required=True,
                        help="Run folder to create; must be missing or empty")
    parser.add_argument("--meshes-dir", type=Path, default=DEFAULT_MESHES_DIR,
                        help=f"Folder whose subfolders are mesh options (default: {DEFAULT_MESHES_DIR})")
    parser.add_argument("--checkpoint", type=Path, default=SAM3_CHECKPOINT,
                        help=f"SAM3 checkpoint (default: {SAM3_CHECKPOINT})")
    parser.add_argument("--settle-frames", type=int, default=30,
                        help="Frames to discard after camera startup (default: 30)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.settle_frames < 0 or args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("settle-frames must be >= 0; width, height, fps must be > 0")
    if not args.meshes_dir.is_dir():
        parser.error(f"--meshes-dir {args.meshes_dir} is not a directory")
    try:
        ensure_empty_dir(args.save_path)
    except FileExistsError as error:
        parser.error(str(error))
    return args


def mesh_folders(meshes_dir: Path):
    return sorted(p for p in meshes_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def main():
    args = arguments()
    segmenter = Sam3Segmenter(args.checkpoint)
    capture = capture_frame(CameraSettings(args.width, args.height, args.fps,
                                           args.settle_frames))
    print("Computing SAM3 image embedding...")
    segmenter.set_image(capture.rgb)

    writer = RunWriter(args.save_path, capture)
    app = SegmentationApp(capture.rgb, segmenter.predict, writer.save_object,
                          mesh_folders(args.meshes_dir))
    saved = app.run()
    if saved:
        print(f"Saved {saved} object(s) under {args.save_path}")
    else:
        print("No objects saved")


if __name__ == "__main__":
    main()
