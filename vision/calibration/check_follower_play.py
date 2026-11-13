"""Can the follower's camera move without its joint readings changing (play the encoders cannot see)?

    cd vision && /opt/conda-envs/vision/bin/python -m calibration.check_follower_play [--frames 40] [--dual PATH]

Why
    The two-arm calibration (calibrate_dual) left about 1 deg of scatter in where each arm really is
    versus what its joints report, with no pattern in joint angle or gravity. Gear backlash would do
    that: the encoders read the motor side, and the output can sit anywhere in the play depending on
    how it was last pushed. This measures it directly, in position mode and in idle (brake) mode.

What it does
    A ChArUco board lies still on the table in the camera's view. In each mode the follower is held
    where it is, and the script measures the board's pose (the camera's true motion) and the
    follower's joints and Cartesian pose (what the encoders report):
        rest                 before any push
        push/hold, release   for each of: up, down, left, right. Push the camera arm GENTLY in that
                             direction (on the wrist or forearm, not over the camera's view), hold it
                             while the script measures, then let go and it measures again.
    If the encoders saw all the motion, the camera's rotation would equal the tool's rotation from
    the joint readings (a rigid mount preserves rotation angles). The difference is hidden motion.
    With --dual (default: the newest data/sessions/*_dual/T_follower_camera.json) the camera mount
    is used to compare the full rotations rather than only their sizes.

Reading the result
    hidden rotation after release  play that stays: the camera ends up somewhere the joints do not show.
                                   A few tenths of a degree is a few mm at objects 0.4 m away.
    hidden rotation while held     compliance under load (gearbox and links bending).
    joint change                   what the encoders did register.
    Compare the two modes: if idle hides much less than position mode, idle is the mode to use.

The follower is commanded into position mode and idle at its current joints (it does not move by
itself). Stop the tracking pipeline first. Writes data/follower_play/<timestamp>.json (git-ignored).
"""

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline.camera import CameraSettings, RealSenseStream

from .board import DEFAULT_DATA_DIR, BoardConfig, board_corners_m, load_config, make_board, make_detector
from .check_depth import DEFAULT_CALIBRATION, MAX_RMS_PX, MIN_CORNERS, load_T_base_camera
from .check_follower_hold import camera_motion
from .detection import detect_raw, estimate_pose, quiet_opencv
from .robot import RobotState
from .transforms import T_from_rvec, inv_T, mean_T, rotation_between_deg

OUT_DIR = DEFAULT_DATA_DIR / "follower_play"
PUSHES = ("up", "down", "left", "right")


def newest_dual():
    found = sorted((DEFAULT_DATA_DIR / "sessions").glob("*_dual/T_follower_camera.json"))
    return found[-1] if found else None


def measure(stream, board, detector, driver, frames):
    """Mean board pose, follower tool pose and joints over `frames` frames with a board pose."""
    K, no_dist = stream.K, np.zeros(5)
    Ts, Fs, qs, tried = [], [], [], 0
    while len(Ts) < frames and tried < frames * 10:
        tried += 1
        rgb, _, _ = stream.read()
        state = RobotState.from_output(driver.get_robot_output())
        corners, ids, _, _ = detect_raw(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), detector)
        if len(ids) < MIN_CORNERS:
            continue
        pose = estimate_pose(board, corners, ids, K, no_dist)
        if pose is None or pose[2] > MAX_RMS_PX:
            continue
        Ts.append(T_from_rvec(pose[0], pose[1]))
        Fs.append(state.T_base_ee)
        qs.append(state.arm_positions)
    if len(Ts) < frames // 2:
        raise RuntimeError(f"board seen in only {len(Ts)} of {tried} frames (is it in view? hand in the way?)")
    return {"T_camera_board": mean_T(Ts), "F": mean_T(Fs), "q": np.mean(qs, axis=0), "frames": len(Ts)}


def compare(rest, m, X, corners_board, R_base_cam):
    """Camera motion (board), tool motion (joints) and the part the joints did not see."""
    cam = camera_motion(rest["T_camera_board"], m["T_camera_board"], corners_board, R_base_cam)
    dF = inv_T(rest["F"]) @ m["F"]                                       # tool motion from the joint readings
    out = {**cam, "tool_rotation_deg": rotation_between_deg(np.eye(3), dF[:3, :3]),
           "joint_change_deg": np.degrees(m["q"] - rest["q"]).tolist()}
    if X is not None:
        D_cam = rest["T_camera_board"] @ inv_T(m["T_camera_board"])     # camera motion, in the rest camera frame
        D_pred = inv_T(X) @ dF @ X                                       # the same, predicted from the joints
        out["hidden_rotation_deg"] = rotation_between_deg(D_pred[:3, :3], D_cam[:3, :3])
    else:
        out["hidden_rotation_deg"] = abs(out["rotation_deg"] - out["tool_rotation_deg"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=40, help="board frames averaged per measurement")
    parser.add_argument("--settle-s", type=float, default=2.0, help="wait after a mode change or a release")
    parser.add_argument("--modes", default="position,idle", help="modes to test, in order")
    parser.add_argument("--dual", type=Path, default=None, help="T_follower_camera.json for the camera mount")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="T_base_camera for base-frame shifts")
    args = parser.parse_args()

    import trossen_arm

    quiet_opencv()
    config = load_config()
    board = make_board(BoardConfig.from_dict(config["board"]))
    detector, corners_board, follower = make_detector(board), board_corners_m(board), config["follower"]
    dual = args.dual or newest_dual()
    X = np.asarray(json.loads(Path(dual).read_text())["T_fee_camera"]) if dual and Path(dual).exists() else None
    print(f"Camera mount for the hidden-rotation comparison: {dual if X is not None else 'none (sizes only)'}")
    T_base_camera = load_T_base_camera(args.calibration)
    R_base_cam = T_base_camera[:3, :3] if T_base_camera is not None else np.eye(3)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print("Board flat and still on the table, in the camera's view. Stand at the follower: it is held in each\n"
          f"mode ({', '.join(modes)}) where it is, and you push it gently when asked.")
    if input("Press Enter to start (q to quit): ").strip().lower() == "q":
        return

    results, driver, configured = [], trossen_arm.TrossenArmDriver(), False
    with RealSenseStream(CameraSettings(undistort=True)) as stream:
        try:
            driver.configure(getattr(trossen_arm.Model, follower["model"]),
                             getattr(trossen_arm.StandardEndEffector, follower["end_effector"]), follower["ip"], False)
            configured = True
            for mode in modes:
                q = list(driver.get_arm_positions())
                if mode == "position":
                    driver.set_arm_modes(trossen_arm.Mode.position)
                    driver.set_arm_positions(q, 0.5, True)
                elif mode == "idle":
                    driver.set_arm_modes(trossen_arm.Mode.idle)
                else:
                    raise SystemExit(f"unknown mode {mode}")
                time.sleep(args.settle_s)
                print(f"\n=== {mode} mode: measuring rest ...")
                rest = measure(stream, board, detector, driver, args.frames)
                results.append({"mode": mode, "step": "rest", "frames": rest["frames"], "q_deg": np.degrees(rest["q"]).tolist()})
                for push in PUSHES:
                    input(f"[{mode}] Push the camera arm gently {push.upper()} and keep holding, then press Enter... ")
                    try:
                        held = measure(stream, board, detector, driver, args.frames)
                        results.append({"mode": mode, "step": f"{push}, held", **compare(rest, held, X, corners_board, R_base_cam)})
                    except RuntimeError as e:
                        print(f"  skipped: {e}")
                    input(f"[{mode}] Let go, then press Enter... ")
                    time.sleep(args.settle_s)
                    rel = measure(stream, board, detector, driver, args.frames)
                    results.append({"mode": mode, "step": f"{push}, released", **compare(rest, rel, X, corners_board, R_base_cam)})
                    r = results[-1]
                    print(f"  after release: camera {r['rotation_deg']:.3f} deg, joints say {r['tool_rotation_deg']:.3f} deg, "
                          f"hidden {r['hidden_rotation_deg']:.3f} deg; objects shift {r['object_shift_mean_mm']:.2f} mm")
        finally:
            if configured:
                try:
                    input("\nThe follower goes idle next. SUPPORT IT, then press Enter... ")
                except (EOFError, KeyboardInterrupt):
                    pass
                driver.cleanup()

    print("\nRelative to each mode's rest pose. 'hidden' = camera rotation the joint readings do not account for.")
    print(f"{'mode':<9}{'step':<17}{'camera':>9}{'joints':>9}{'hidden':>9}{'object shift':>14}   largest joint change")
    for r in results:
        if r["step"] == "rest":
            continue
        jc = np.array(r["joint_change_deg"])
        j = int(np.argmax(np.abs(jc)))
        print(f"{r['mode']:<9}{r['step']:<17}{r['rotation_deg']:7.3f}d {r['tool_rotation_deg']:7.3f}d "
              f"{r['hidden_rotation_deg']:7.3f}d {r['object_shift_mean_mm']:9.2f} mm   joint {j}: {jc[j]:+.3f} deg")
    for mode in modes:
        rel = [r["hidden_rotation_deg"] for r in results if r["mode"] == mode and r["step"].endswith("released")]
        held = [r["hidden_rotation_deg"] for r in results if r["mode"] == mode and r["step"].endswith("held")]
        if rel:
            print(f"{mode}: hidden rotation after release mean {np.mean(rel):.3f} deg (max {np.max(rel):.3f}); "
                  f"while held mean {np.mean(held) if held else float('nan'):.3f} deg")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"follower": follower, "camera_mount": str(dual) if X is not None else None,
                               "results": results}, indent=2, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
