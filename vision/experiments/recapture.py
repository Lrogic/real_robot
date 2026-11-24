"""Return the camera to a saved move-mode run's follower joints and recapture there at other resolutions.

For comparing captures of a scene that has not changed (e.g. SAM 3D input resolution): the
follower drives back to the source run's locked joints, as --track does, and holds there while
one frame is captured per --resolutions entry, back to back. Each frame then gets the usual mask
UI and is saved as its own run folder, <save-path>_<W>x<H>, in the run_folder.py layout. Its
camera_pose.json is the pose measured now, plus the source run's name. On exit the follower
parks, as with run.sh --camera move.

Run from real_robot/vision in the vision env, with the person at the arm:

    PYTHONPATH=..:. /opt/conda-envs/vision/bin/python -m experiments.recapture \
        --source foundationpose_runs/SAM3D-small_frame \
        --save-path foundationpose_runs/SAM3D-recapture --resolutions 640x480 1280x720
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from pipeline.camera import CameraSettings, capture_frame
from pipeline.pipeline_args import DEFAULT_MESHES_DIR
from pipeline.robot_arm.camera_arm import held_camera
from pipeline.run_folder import CAMERA_POSE_FILE, RunWriter, ensure_empty_dir
from pipeline.segmentation.mask_selection_ui import SegmentationApp
from pipeline.segmentation.sam_model_loader import HF_CACHE
from pipeline.setup_stage import SETUP_SAM_MODEL, hf_predictor, mesh_folders


def resolution(text: str):
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not WIDTHxHEIGHT")
    return width, height


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Run folder set up with --camera move")
    parser.add_argument("--save-path", type=Path, required=True,
                        help="Prefix for the new run folders; each gets _<W>x<H> appended")
    parser.add_argument("--resolutions", type=resolution, nargs="+", default=[(640, 480), (1280, 720)])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--settle-frames", type=int, default=30)
    parser.add_argument("--sam-model", default=SETUP_SAM_MODEL)
    parser.add_argument("--meshes-dir", type=Path, default=DEFAULT_MESHES_DIR)
    args = parser.parse_args()

    saved = json.loads((args.source / CAMERA_POSE_FILE).read_text())
    if saved.get("mode") != "move" or not saved.get("follower_joints_rad"):
        parser.error(f"{args.source} has no move-mode camera_pose.json with follower joints")
    source_K = json.loads((args.source / "camera_intrinsics.json").read_text())
    outputs = [(w, h, Path(f"{args.save_path}_{w}x{h}")) for w, h in args.resolutions]
    for _, _, path in outputs:  # fail before the arm moves
        try:
            ensure_empty_dir(path)
        except FileExistsError as error:
            parser.error(str(error))

    from pipeline.segmentation.sam_video_tracker import HfVideoTracker

    print(f"Loading HF SAM model {args.sam_model} from {HF_CACHE}...")
    tracker = HfVideoTracker(args.sam_model, cache_dir=HF_CACHE)

    # held_camera reads only these fields; None picks the default camera arm and calibration configs.
    arm_args = SimpleNamespace(camera="move", camera_config=None, calibration_config=None)
    undistort = bool(source_K.get("undistorted", False))
    first = CameraSettings(*args.resolutions[0], args.fps, args.settle_frames, undistort)
    with held_camera(arm_args, first, saved) as pose:
        pose.extra["recaptured_from"] = str(args.source.resolve())
        captures = []
        for width, height, path in outputs:  # capture everything first, so the pose is the same for all
            print(f"Capturing {width}x{height}...")
            captures.append(capture_frame(CameraSettings(width, height, args.fps, args.settle_frames, undistort)))
        for (width, height, path), capture in zip(outputs, captures):
            print(f"Segment the {width}x{height} frame ({path})")
            writer = RunWriter(path, capture, pose.to_json())
            app = SegmentationApp(capture.rgb, hf_predictor(tracker, capture.rgb), writer.save_object,
                                  mesh_folders(args.meshes_dir), score_label=f"{args.sam_model} presence")
            print(f"Saved {app.run()} object(s) under {path}")


if __name__ == "__main__":
    main()
