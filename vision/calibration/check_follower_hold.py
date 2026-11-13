"""Does the camera sit differently when the follower is idle versus held by its motors?

    cd vision && /opt/conda-envs/vision/bin/python -m calibration.check_follower_hold [--frames 60]

The eye-to-hand calibration (T_base_camera) was collected with the follower parked and idle:
vision/run.sh --reset drives it to the session's joints and then disconnects, so it rests on
itself. A follower that moves the camera will be held in position mode instead, with its weight
on the gearboxes. If the camera sits differently in the two states, that difference is a
constant error in any calibration that chains the follower's joints to the parked
T_base_camera. This measures it with the camera itself: a ChArUco board lying still on the
table is seen from each state, and any change in its pose is the camera moving.

Phases (the board and the manipulator stay still throughout):
    idle (start)            the follower as it is now: parked and idle (run --reset first)
    held at those joints    position mode at the joints it is resting at: the load change only
    held at recorded park   position mode at the calibration session's parked joints
    idle (end)              disconnected again from the recorded park, as --reset leaves it
The two idle phases also show how repeatable the idle rest is.

The follower is commanded (it holds, then moves a fraction of a degree to the recorded park),
so be at the arm. Its camera stream needs the tracking pipeline stopped. Writes
data/follower_hold/<timestamp>.json (git-ignored).

Reading the result
    For each phase it prints how far the camera moved from the starting idle pose (angle and
    translation) and how far that would shift tracked objects at the board, in the manipulator
    base frame. Shifts under about 0.5 mm and 0.05 deg are measurement noise (the script prints
    the frame-to-frame scatter): the October anchor can be reused. Larger shifts mean the
    follower should be held in position mode when parked, and re-anchored there.
"""

import argparse
import dataclasses
import datetime as dt
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline.camera import CameraSettings, RealSenseStream

from .board import DEFAULT_DATA_DIR, BoardConfig, board_corners_m, load_config, make_board, make_detector
from .check_depth import DEFAULT_CALIBRATION, MAX_RMS_PX, MIN_CORNERS, load_T_base_camera
from .detection import detect_raw, estimate_pose, quiet_opencv
from .reset_follower import session_follower_joints
from .session import read_json
from .transforms import T_from_rvec, inv_T, mean_T, transform_points

OUT_DIR = DEFAULT_DATA_DIR / "follower_hold"


def board_pose(stream, board, detector, frames):
    """Mean T_camera_board over `frames` frames, the per-frame poses, and the corners (board frame)."""
    K, no_dist = stream.K, np.zeros(5)
    Ts, tried = [], 0
    while len(Ts) < frames and tried < frames * 10:
        tried += 1
        rgb, _, _ = stream.read()
        corners, ids, _, _ = detect_raw(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), detector)
        if len(ids) < MIN_CORNERS:
            continue
        pose = estimate_pose(board, corners, ids, K, no_dist)
        if pose is None or pose[2] > MAX_RMS_PX:
            continue
        Ts.append(T_from_rvec(pose[0], pose[1]))
    if len(Ts) < frames // 2:
        raise SystemExit(f"Board seen in only {len(Ts)} of {tried} frames; is it in view and lit?")
    return mean_T(Ts), Ts


def camera_motion(T_cam0_board, T_cam_board, corners_board, R_base_cam):
    """How the camera moved from phase 0, and how far that shifts points at the board."""
    T_cam0_cam = T_cam0_board @ inv_T(T_cam_board)
    angle = float(np.degrees(np.arccos(np.clip((np.trace(T_cam0_cam[:3, :3]) - 1) / 2, -1, 1))))
    pts = transform_points(T_cam_board, corners_board)                    # board corners in the new camera frame
    shift_cam = transform_points(T_cam0_cam, pts) - pts                   # tracked point error if phase 0 is assumed
    shift_base = shift_cam @ R_base_cam.T
    return {"rotation_deg": angle, "translation_mm": float(np.linalg.norm(T_cam0_cam[:3, 3]) * 1e3),
            "object_shift_mean_mm": float(np.linalg.norm(shift_cam, axis=1).mean() * 1e3),
            "object_shift_max_mm": float(np.linalg.norm(shift_cam, axis=1).max() * 1e3),
            "object_shift_base_mean_mm": (shift_base.mean(axis=0) * 1e3).tolist()}


def scatter(Ts, T_mean):
    """Frame-to-frame scatter of the board pose: rotation (deg) and translation (mm), rms."""
    rot, trans = [], []
    for T in Ts:
        D = inv_T(T_mean) @ T
        rot.append(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))))
        trans.append(np.linalg.norm(D[:3, 3]) * 1e3)
    return float(np.sqrt(np.mean(np.square(rot)))), float(np.sqrt(np.mean(np.square(trans))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=60, help="board frames averaged per phase")
    parser.add_argument("--settle-s", type=float, default=3.0, help="wait after each change before measuring")
    parser.add_argument("--move-time-s", type=float, default=2.0, help="duration of the move to the recorded park")
    parser.add_argument("--session", default=str(DEFAULT_DATA_DIR / "sessions" / "20261009_combined"),
                        help="calibration session whose parked follower joints to use")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="T_base_camera for base-frame shifts")
    args = parser.parse_args()

    import trossen_arm

    quiet_opencv()
    config = load_config()
    board_cfg = BoardConfig.from_dict(config["board"])
    board, follower = make_board(board_cfg), config["follower"]
    detector = make_detector(board)
    corners_board = board_corners_m(board)
    parked = np.array(session_follower_joints(read_json(Path(args.session) / "session.json")))
    T_base_camera = load_T_base_camera(args.calibration)
    R_base_cam = T_base_camera[:3, :3] if T_base_camera is not None else np.eye(3)

    print(f"Follower {follower['ip']}. Recorded parked joints (deg): {np.round(np.degrees(parked), 2).tolist()}")
    print("Put the ChArUco board flat and still on the table in the camera's view. The follower should be\n"
          "parked and idle (run.sh --setup-track ... --reset leaves it that way). Stand at the follower:\n"
          "it will be held by its motors and moved a fraction of a degree.")
    if input("Press Enter to start (q to quit): ").strip().lower() == "q":
        return

    phases, driver, configured = [], trossen_arm.TrossenArmDriver(), False
    with RealSenseStream(CameraSettings(undistort=True)) as stream:
        def measure(name, joints=None):
            print(f"\n[{name}] measuring {args.frames} frames ...")
            T, Ts = board_pose(stream, board, detector, args.frames)
            rot_rms, trans_rms = scatter(Ts, T)
            phases.append({"phase": name, "T_camera_board": T.tolist(), "frames": len(Ts),
                           "scatter_rot_deg": rot_rms, "scatter_trans_mm": trans_rms,
                           "joints_rad": None if joints is None else list(map(float, joints))})
            print(f"  board at {np.linalg.norm(T[:3, 3]):.3f} m; frame-to-frame scatter {rot_rms:.3f} deg, {trans_rms:.2f} mm")

        try:
            measure("idle (start)")
            driver.configure(getattr(trossen_arm.Model, follower["model"]),
                             getattr(trossen_arm.StandardEndEffector, follower["end_effector"]), follower["ip"], False)
            configured = True
            q_idle = np.array(driver.get_arm_positions(), dtype=float)
            phases[0]["joints_rad"] = q_idle.tolist()
            print(f"  idle joints (deg): {np.round(np.degrees(q_idle), 2).tolist()}; "
                  f"off the recorded park by {np.degrees(np.abs(q_idle - parked).max()):.2f} deg at most")

            driver.set_arm_modes(trossen_arm.Mode.position)
            driver.set_arm_positions(q_idle.tolist(), 0.5, True)
            time.sleep(args.settle_s)
            measure("held at those joints", driver.get_arm_positions())

            driver.set_arm_positions(parked.tolist(), args.move_time_s, True)
            time.sleep(args.settle_s)
            measure("held at recorded park", driver.get_arm_positions())
        finally:
            if configured:
                try:
                    input("\nThe follower goes idle next (as --reset leaves it). SUPPORT THE ARM, then press Enter... ")
                except (EOFError, KeyboardInterrupt):
                    pass
                driver.cleanup()
        if len(phases) == 3:
            time.sleep(args.settle_s)
            measure("idle (end)")

    T0 = np.array(phases[0]["T_camera_board"])
    print("\nCamera movement relative to 'idle (start)'; object shift is at the board, in the manipulator base frame")
    print(f"{'phase':<24}{'rotation':>10}{'translation':>13}{'shift mean/max':>17}   shift x / y / z (base)")
    for p in phases[1:]:
        m = camera_motion(T0, np.array(p["T_camera_board"]), corners_board, R_base_cam)
        p["vs_idle_start"] = m
        sx, sy, sz = m["object_shift_base_mean_mm"]
        print(f"{p['phase']:<24}{m['rotation_deg']:>8.3f} deg{m['translation_mm']:>9.2f} mm"
              f"{m['object_shift_mean_mm']:>9.2f} / {m['object_shift_max_mm']:.2f} mm   {sx:+.2f} / {sy:+.2f} / {sz:+.2f} mm")
    noise = max(p["scatter_rot_deg"] for p in phases), max(p["scatter_trans_mm"] for p in phases)
    print(f"Frame-to-frame scatter up to {noise[0]:.3f} deg / {noise[1]:.2f} mm; averaging {args.frames} frames "
          "makes each phase's mean several times steadier than that.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"follower": follower, "session": args.session, "parked_joints_rad": parked.tolist(),
                               "calibration": args.calibration, "board": dataclasses.asdict(board_cfg),
                               "phases": phases}, indent=2, default=str))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
