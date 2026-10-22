"""Capture one RealSense RGB-D frame and interactively segment objects with SAM3.

Each saved object gets its own obj{i} folder under --save-path, optionally with a
copy of a mesh folder from --meshes-dir. See fp_capture/run_folder.py for the layout.
With --recovery the masks come from the recovery config's HF SAM model (--sam-model),
the same model that tracks the objects later; run that in the vision env.

    python segment_objects.py --save-path foundationpose_runs/run1
    python segment_objects.py --save-path foundationpose_runs/run1 --recovery
"""

import os
from functools import partial
from pathlib import Path

from fp_capture.camera import capture_frame
from fp_capture.cli import camera_settings, parse_pipeline_args, pipeline_parser
from fp_capture.follower_reset import ensure_vision_on_path, run_follower_reset_if_requested
from fp_capture.run_folder import RunWriter, ensure_empty_dir
from fp_capture.ui import SegmentationApp

ensure_vision_on_path()


def mesh_folders(meshes_dir: Path):
    return sorted(p for p in meshes_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def hf_predictor(tracker, rgb):
    """Click/box masks from the HF SAM video tracker on rgb, each in a throwaway session."""
    # The first request is several seconds slower; pay that cost before the UI opens.
    height, width = rgb.shape[:2]
    tracker.preview(rgb, [(width / 2, height / 2)], [1], None)
    return partial(tracker.preview, rgb)


def main():
    parser = pipeline_parser(__doc__)
    args = parse_pipeline_args(parser)
    if not args.meshes_dir.is_dir():
        parser.error(f"--meshes-dir {args.meshes_dir} is not a directory")
    try:
        ensure_empty_dir(args.save_path)
    except FileExistsError as error:
        parser.error(str(error))

    if not os.environ.get("REAL_ROBOT_FOLLOWER_RESET_DONE"):
        run_follower_reset_if_requested(args)

    if args.recovery:
        from experiments.segmentation.tracker import HfVideoTracker

        cfg = args.recovery_cfg
        print(f"Loading HF SAM model {cfg.sam_model} from {cfg.hf_cache}...")
        tracker = HfVideoTracker(cfg.sam_model, cache_dir=cfg.hf_cache)
        capture = capture_frame(camera_settings(args))
        predict = hf_predictor(tracker, capture.rgb)
        score_label = f"{cfg.sam_model} presence"
    else:
        from fp_capture.segmenter import Sam3Segmenter

        segmenter = Sam3Segmenter(args.checkpoint)
        capture = capture_frame(camera_settings(args))
        print("Computing SAM3 image embedding...")
        segmenter.set_image(capture.rgb)
        predict, score_label = segmenter.predict, "Mask score"

    writer = RunWriter(args.save_path, capture)
    app = SegmentationApp(capture.rgb, predict, writer.save_object,
                          mesh_folders(args.meshes_dir), score_label=score_label)
    saved = app.run()
    if saved:
        print(f"Saved {saved} object(s) under {args.save_path}")
    else:
        print("No objects saved")


if __name__ == "__main__":
    main()
