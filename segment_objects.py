"""Capture one RealSense RGB-D frame and interactively segment objects with SAM3.

Each saved object gets its own obj{i} folder under --save-path, optionally with a
copy of a mesh folder from --meshes-dir. See fp_capture/run_folder.py for the layout.

    python segment_objects.py --save-path foundationpose_runs/run1
"""

from pathlib import Path

from fp_capture.camera import capture_frame
from fp_capture.cli import camera_settings, parse_pipeline_args, pipeline_parser
from fp_capture.run_folder import RunWriter, ensure_empty_dir
from fp_capture.segmenter import Sam3Segmenter
from fp_capture.ui import SegmentationApp


def mesh_folders(meshes_dir: Path):
    return sorted(p for p in meshes_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def main():
    parser = pipeline_parser(__doc__)
    args = parse_pipeline_args(parser)
    if not args.meshes_dir.is_dir():
        parser.error(f"--meshes-dir {args.meshes_dir} is not a directory")
    try:
        ensure_empty_dir(args.save_path)
    except FileExistsError as error:
        parser.error(str(error))

    segmenter = Sam3Segmenter(args.checkpoint)
    capture = capture_frame(camera_settings(args))
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
