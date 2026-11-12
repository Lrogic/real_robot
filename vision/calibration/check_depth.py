"""Compare the D405 depth image with the ChArUco board's distance from the colour image.

    cd vision && /opt/conda-envs/vision/bin/python -m calibration.check_depth [--frames 30]

Camera only; no arm is touched. Stop the tracking pipeline first (one process per camera).

Why this exists
    The tracked jenga block read about 7 mm too high on the real arm compared with where the
    arm itself touched it (docs/block_height_offset.md). The hand-eye calibration
    (T_base_camera) is solved from ChArUco corners in the colour image and never uses depth,
    while FoundationPose places objects with the depth image. So a depth sensor that reads
    short or long shifts every tracked pose even when the calibration is right. This script
    separates the two: it measures the depth image against the colour image alone, with no
    arm, no calibration and no FoundationPose involved.

    First results (November 2026, this D405): depth read short by 4.6 mm at 0.25 m, 7.5 mm at
    0.31 m, 15.6 mm at 0.42 m and 19.9 mm at 0.52 m, growing roughly with distance squared,
    which is what a drifted stereo calibration gives.

What it does
    Uses the tracking pipeline's stream (640x480, depth aligned to colour, undistorted). For
    every frame with a board pose, each pixel inside the detected corners gets two depths:
      * the depth image's value at that pixel, and
      * where that pixel's line of sight meets the board plane, from the board pose that the
        colour image gives (PnP on the corners; the same measurement the calibration uses).
    Both are plane depth (distance along the camera's forward axis, as RealSense reports it),
    so a tilted board is handled. It prints the median of depth minus colour over all pixels
    and frames. With --calibration (default: the session the policy has used) it also prints
    how far that difference moves a point in the robot base frame.

How to use it
    Prop the board (the print in calibration/config.yaml) facing the camera, still, with the
    whole pattern in view. Run it at the object's distance and at one nearer and one farther
    distance: a constant percentage needs a scale, a constant offset an offset, and growth
    with distance squared a disparity correction (corrected = depth / (1 - c * depth)) or a
    RealSense tare calibration. The "board plane on the optical axis" line is the ground
    truth to give a tare calibration with the board left where it is.

Reading the output
    negative  the depth image puts the board closer to the camera than the colour image does
    The colour side is only as good as the board's printed square size (verify_board checks
    it) and the colour intrinsics.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.camera import CameraSettings, RealSenseStream

from .board import DEFAULT_DATA_DIR, BoardConfig, load_config, make_board, make_detector
from .detection import detect_raw, estimate_pose, quiet_opencv
from .transforms import T_from_rvec

DEFAULT_CALIBRATION = DEFAULT_DATA_DIR / "sessions" / "20261009_combined" / "T_base_camera.json"
MIN_CORNERS = 12
MAX_RMS_PX = 1.0


def compare(depth, corners, T_camera_board, K):
    """(pixels uv, depth-image z, board-plane z) for valid pixels inside the corners' hull."""
    hull = cv2.convexHull(corners.astype(np.float32))
    inside = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(inside, hull.astype(np.int32), 1)
    v, u = np.nonzero((inside > 0) & (depth > 0))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))], axis=1)
    normal, origin = T_camera_board[:3, 2], T_camera_board[:3, 3]
    plane_z = (normal @ origin) / (rays @ normal)       # rays have z = 1, so the scale is z
    return np.stack([u, v], axis=1), depth[v, u].astype(np.float64), plane_z, rays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=30, help="frames with a board pose to average")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--save", default=None, help="write the per-frame numbers to this JSON")
    args = parser.parse_args()

    quiet_opencv()
    cfg = BoardConfig.from_dict(load_config()["board"])
    board = make_board(cfg)
    detector = make_detector(board)
    T_base_camera = None
    if args.calibration and Path(args.calibration).exists():
        T_base_camera = np.array(json.loads(Path(args.calibration).read_text())["T_base_camera"])

    rows, tried = [], 0
    with RealSenseStream(CameraSettings(undistort=True)) as stream:
        K, no_dist = stream.K, np.zeros(5)
        while len(rows) < args.frames and tried < args.frames * 10:
            tried += 1
            rgb, depth, _ = stream.read()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            corners, ids, _, _ = detect_raw(gray, detector)
            if len(ids) < MIN_CORNERS:
                continue
            pose = estimate_pose(board, corners, ids, K, no_dist)
            if pose is None or pose[2] > MAX_RMS_PX:
                continue
            rvec, tvec, rms, _ = pose
            T = T_from_rvec(rvec, tvec)
            _, z_depth, z_plane, rays = compare(depth, corners, T, K)
            if len(z_depth) < 200:
                continue
            diff = z_depth - z_plane
            keep = np.abs(diff - np.median(diff)) < 0.02          # drop holes and edge pixels
            diff, rays, z_plane = diff[keep], rays[keep], z_plane[keep]
            centre = rays.mean(axis=0) * z_plane.mean()
            centre_ray = np.array([0.0, 0.0, 1.0])               # the optical axis, through the principal point
            row = {"corners": int(len(ids)), "rms_px": rms, "pixels": int(len(diff)),
                   "axis_plane_depth_mm": float((T[:3, 2] @ T[:3, 3]) / (centre_ray @ T[:3, 2]) * 1e3),
                   "centre_offset_px": float(np.linalg.norm(corners.mean(axis=0) - K[:2, 2])),
                   "board_distance_m": float(np.linalg.norm(centre)),
                   "tilt_deg": float(np.degrees(np.arccos(abs(T[:3, 2] @ centre) / np.linalg.norm(centre)))),
                   "z_diff_mm": float(np.median(diff) * 1e3), "z_diff_spread_mm": float(np.std(diff) * 1e3),
                   "z_diff_percent": float(np.median(diff / z_plane) * 100)}
            if T_base_camera is not None:
                shift = (rays * diff[:, None]).mean(axis=0)       # camera-frame move of the points
                row["shift_base_mm"] = (T_base_camera[:3, :3] @ shift * 1e3).tolist()
                row["board_centre_base_m"] = (T_base_camera @ np.append(centre, 1.0))[:3].tolist()
            rows.append(row)

    if not rows:
        print(f"No usable board pose in {tried} frames (need {MIN_CORNERS} corners and rms under {MAX_RMS_PX} px).")
        return
    med = lambda key: float(np.median([r[key] for r in rows]))
    print(f"{len(rows)} frames of {tried}; {med('corners'):.0f} corners, reprojection rms {med('rms_px'):.2f} px, "
          f"{med('pixels'):.0f} pixels compared")
    print(f"Board centre {med('board_distance_m'):.3f} m from the camera, tilted {med('tilt_deg'):.0f} deg from face-on")
    print(f"Depth image minus colour-image board plane: {med('z_diff_mm'):+.1f} mm "
          f"({med('z_diff_percent'):+.2f}% of depth), frame-to-frame std "
          f"{np.std([r['z_diff_mm'] for r in rows]):.1f} mm, within-board std {med('z_diff_spread_mm'):.1f} mm")
    print("  negative: the depth image puts the board closer to the camera than the colour image does")
    print(f"Board plane on the optical axis (colour): {med('axis_plane_depth_mm'):.1f} mm; board centre is "
          f"{med('centre_offset_px'):.0f} px from the image centre")
    print("  the first number is the ground truth for a RealSense tare calibration with the board left in place")
    if T_base_camera is not None:
        shift = np.median([r["shift_base_mm"] for r in rows], axis=0)
        centre = np.median([r["board_centre_base_m"] for r in rows], axis=0)
        print(f"Board centre in the base frame (colour): {np.round(centre, 3).tolist()} m")
        print(f"The depth image moves those points by x {shift[0]:+.1f}, y {shift[1]:+.1f}, z {shift[2]:+.1f} mm "
              "in the base frame")
    if args.save:
        Path(args.save).write_text(json.dumps({"kind": "depth_vs_board", "K": K.tolist(), "frames": rows}, indent=2))
        print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
