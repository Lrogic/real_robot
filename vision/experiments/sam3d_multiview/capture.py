"""Capture several views of one object with the camera arm, for the SAM 3D vs MV-SAM3D comparison.

The follower goes to gravity compensation with a live camera window, as run.sh --camera move does.
Guide the camera to a view and press Enter or space: the follower locks, settles, and one RGB-D
frame is captured with T_base_camera from the two-arm calibration. The follower is then free
again for the next view. Press q or Esc in the camera window when you have all the views you
want (the window's text says "abort"; here it means "done"). The follower then parks, and the
mask UI opens once per view: segment the object and save it with "No mesh".

Session layout (each view in the run_folder.py layout, so the other vision tools can read it):

    <save-path>/view_0/rgb.png, depth.npy, camera_intrinsics.json, camera_pose.json, obj1/mask.png
    <save-path>/view_1/...

Run from real_robot/vision in the vision env, with the person at the arm:

    PYTHONPATH=..:. /opt/conda-envs/vision/bin/python -m experiments.sam3d_multiview.capture \
        --save-path foundationpose_runs/mv_jenga1

Then reconstruct.py (sam3d env) on the same folder.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from calibration.board import load_config
from pipeline.camera import CameraSettings, capture_frame
from pipeline.robot_arm.camera_arm import (CameraArmConfig, CameraPose, _Follower, _locked_F, guide_follower,
                                           move_calibration, parked_calibration)
from pipeline.run_folder import RunWriter, ensure_empty_dir
from pipeline.segmentation.mask_selection_ui import SegmentationApp
from pipeline.segmentation.sam_model_loader import HF_CACHE
from pipeline.setup_stage import SETUP_SAM_MODEL, hf_predictor


def view_spread(poses):
    """Angle (deg) between the newest camera's optical axis and each earlier one."""
    axes = [np.asarray(p["T_base_camera"])[:3, 2] for p in poses]
    return [round(float(np.degrees(np.arccos(np.clip(axes[-1] @ a, -1, 1)))), 1) for a in axes[:-1]]


def capture_views(args, settings):
    """Guide/lock/capture until q/Esc; returns [(Capture, camera_pose json)]. Parks the follower at the end."""
    cfg = CameraArmConfig.load(args.camera_config)
    follower_cfg = load_config(args.calibration_config)["follower"]
    _, parked_joints = parked_calibration(cfg.parked_calibration)
    Z, X, identity = move_calibration(cfg.move_calibration)
    for key in ("model", "end_effector"):  # as held_camera checks
        if identity.get(key) and identity[key] != follower_cfg[key]:
            raise ValueError(f"follower {key} {follower_cfg[key]!r} differs from the two-arm calibration's "
                             f"{identity[key]!r} ({cfg.move_calibration})")

    views = []
    follower = _Follower(follower_cfg).__enter__()
    try:
        while args.max_views is None or len(views) < args.max_views:
            print(f"\nView {len(views)}: guide the camera, Enter/space captures, q/Esc finishes.")
            if not guide_follower(follower, settings):
                break
            F, joints = _locked_F(follower, cfg.lock_settle_s)
            pose = CameraPose("move", Z @ F @ X, cfg.move_calibration, cfg.parked_calibration, joints, F,
                              {"lock_settle_s": cfg.lock_settle_s, "view": len(views)})
            capture = capture_frame(settings)
            views.append((capture, pose.to_json()))
            spread = view_spread([p for _, p in views])
            print(f"Captured view {len(views) - 1}: camera at {np.round(pose.T_base_camera[:3, 3], 3)} m"
                  + (f", optical axis {spread} deg from the earlier views" if spread else ""))
    finally:
        print("Returning the follower to the parked joints before it goes idle...")
        try:
            follower.move_to(parked_joints, cfg.move_time_s)
        except Exception as error:
            print(f"Could not park the follower ({error}).")
            try:
                input("The follower will go idle where it is. SUPPORT IT, then press Enter... ")
            except (EOFError, KeyboardInterrupt):
                pass
        follower.close()
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-path", type=Path, required=True, help="Session folder (must be new or empty)")
    parser.add_argument("--max-views", type=int, default=None, help="Stop after this many views")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--settle-frames", type=int, default=30)
    parser.add_argument("--undistort", action="store_true")
    parser.add_argument("--sam-model", default=SETUP_SAM_MODEL)
    parser.add_argument("--camera-config", type=Path, default=None, help="Default: pipeline/camera_arm.yaml")
    parser.add_argument("--calibration-config", type=Path, default=None, help="Default: calibration/config.yaml")
    args = parser.parse_args()
    try:
        ensure_empty_dir(args.save_path)  # fail before the arm moves
    except FileExistsError as error:
        parser.error(str(error))

    from pipeline.segmentation.sam_video_tracker import HfVideoTracker

    print(f"Loading HF SAM model {args.sam_model} from {HF_CACHE}...")
    tracker = HfVideoTracker(args.sam_model, cache_dir=HF_CACHE)

    settings = CameraSettings(args.width, args.height, args.fps, args.settle_frames, args.undistort)
    views = capture_views(args, settings)
    if not views:
        raise SystemExit("No views captured")

    writers = []
    for i, (capture, pose) in enumerate(views):  # write every frame before masking, so none is lost
        writer = RunWriter(args.save_path / f"view_{i}", capture, pose)
        writer._write_scene()
        writers.append(writer)
    print(f"\nSaved {len(views)} views under {args.save_path}. Mask the object in each (save with 'No mesh').")

    for i, writer in enumerate(writers):
        print(f"Segment view {i} of {len(writers) - 1}")
        app = SegmentationApp(writer.capture.rgb, hf_predictor(tracker, writer.capture.rgb), writer.save_object,
                              [], score_label=f"{args.sam_model} presence")
        if app.run() == 0:
            print(f"WARNING: no mask saved for view {i}; reconstruct.py will skip it")
    (args.save_path / "session.json").write_text(json.dumps(
        {"views": len(views), "settings": vars(settings)}, indent=2) + "\n")
    print(f"Done: {args.save_path}")


if __name__ == "__main__":
    main()
