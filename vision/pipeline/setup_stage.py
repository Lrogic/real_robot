"""Capture one RealSense RGB-D frame and interactively segment objects with an HF SAM tracker.

Each saved object gets its own obj{i} folder under --save-path, optionally with a
copy of a mesh folder from --meshes-dir. See pipeline/run_folder.py for the layout.
Masks come from sam3 by default; with --recovery they come from the recovery config's
model (--sam-model), the same model that tracks the objects later. The camera arm is set up
first (--camera parked | move, pipeline/robot_arm/camera_arm.py) and its pose is saved as
camera_pose.json. Run from real_robot/vision in the vision env (run.sh --setup does all of
this; run.sh --setup-track runs tracking in the same process, with the follower still held).

    python -m pipeline.setup_stage --save-path foundationpose_runs/run1 --camera parked
    python -m pipeline.setup_stage --save-path foundationpose_runs/run1 --camera move --recovery
"""

from functools import partial
from pathlib import Path

from .camera import capture_frame
from .pipeline_args import camera_settings, parse_pipeline_args, pipeline_parser
from .robot_arm.camera_arm import held_camera
from .run_folder import RunWriter, ensure_empty_dir
from .segmentation.mask_selection_ui import SegmentationApp

SETUP_SAM_MODEL = "sam3"  # without --recovery


def mesh_folders(meshes_dir: Path):
    return sorted(p for p in meshes_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def hf_predictor(tracker, rgb):
    """Click/box masks from the HF SAM video tracker on rgb, each in a throwaway session."""
    # The first request is several seconds slower; pay that cost before the UI opens.
    height, width = rgb.shape[:2]
    tracker.preview(rgb, [(width / 2, height / 2)], [1], None)
    return partial(tracker.preview, rgb)


def check_setup_args(parser, args) -> None:
    """Fail before the camera arm moves."""
    if not args.meshes_dir.is_dir():
        parser.error(f"--meshes-dir {args.meshes_dir} is not a directory")
    try:
        ensure_empty_dir(args.save_path)
    except FileExistsError as error:
        parser.error(str(error))


def run_setup(args, camera) -> int:
    """Capture the frame with the camera arm already held, segment, and save; returns objects saved."""
    from .segmentation.sam_model_loader import HF_CACHE
    from .segmentation.sam_video_tracker import HfVideoTracker

    if args.recovery:
        sam_model, hf_cache = args.recovery_cfg.sam_model, args.recovery_cfg.hf_cache
    else:
        sam_model, hf_cache = SETUP_SAM_MODEL, HF_CACHE
    print(f"Loading HF SAM model {sam_model} from {hf_cache}...")
    tracker = HfVideoTracker(sam_model, cache_dir=hf_cache)
    capture = capture_frame(camera_settings(args))
    predict = hf_predictor(tracker, capture.rgb)
    score_label = f"{sam_model} presence"

    writer = RunWriter(args.save_path, capture, camera.to_json())
    app = SegmentationApp(capture.rgb, predict, writer.save_object,
                          mesh_folders(args.meshes_dir), score_label=score_label)
    saved = app.run()
    if saved:
        print(f"Saved {saved} object(s) under {args.save_path}")
    else:
        print("No objects saved")
    return saved


def main():
    parser = pipeline_parser(__doc__)
    args = parse_pipeline_args(parser)
    check_setup_args(parser, args)
    with held_camera(args, camera_settings(args)) as camera:
        run_setup(args, camera)


if __name__ == "__main__":
    main()
