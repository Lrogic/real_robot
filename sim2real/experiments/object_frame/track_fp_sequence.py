"""Stage 2 of the synthetic FoundationPose-vs-MuJoCo object-pose test: track and compare.

Reads a run folder from ``render_sim_sequence.py``. FoundationPose is set up
through the pipeline's ``MultiObjectTracker`` with the scene's mesh file,
registered once on frame 0 with the simulator mask, and then only tracked:
no re-registration, no reset, no ground truth fed in. Ground truth is used
only to score the result.

For every frame:

  T_base_object_FP = T_base_camera @ center_pose @ to_origin   (FP's raw mesh-frame pose)
  T_base_object_GT = inv(T_world_base) @ T_world_object        (MuJoCo body state)

Position error is |t_FP - t_GT| in mm (plus per base axis); orientation error is
the angle of R_GT^T R_FP in degrees. With ``--symmetry box`` the four proper
symmetries of the mesh's oriented bounding box (identity and 180 deg about each
box axis, through the box centre) are also tried: T_GT @ S_k with
S_k = inv(to_origin) @ Rot180_k @ to_origin. The symmetry-aware error is the
one with the smallest rotation error, and its position error is reported
alongside. Ground truth never touches the tracker.

Writes ``fp_vs_sim_<time>.json`` (per frame and summary) and
``fp_vs_sim_<time>.mp4`` (thick axes = ground truth, thin axes with dots = FP)
into the run folder.

Launch from the fp_robot environment (GPU):

  conda activate fp_robot
  python real_robot/sim2real/experiments/object_frame/track_fp_sequence.py RUN_FOLDER --symmetry box
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(RESEARCH / "real_robot/vision"))

from common.io import write_json  # noqa: E402
from common.poses import quaternion_wxyz, rotation_angle_deg, rotvec_to_matrix  # noqa: E402
from pipeline.foundationpose.multi_object_fp_tracker import MultiObjectTracker  # noqa: E402
from pipeline.run_folder import SavedObject  # noqa: E402

DEFAULT_FOUNDATIONPOSE_DIR = RESEARCH / "FoundationPose"
AXIS_COLORS_BGR = ((0, 0, 255), (0, 200, 0), (255, 0, 0))


def load_frame(frames_dir: Path, i: int, depth_scale: float):
    bgr = cv2.imread(str(frames_dir / f"{i:05d}_rgb.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(frames_dir / f"{i:05d}_depth.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(frames_dir / f"{i:05d}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if bgr is None or depth is None or mask is None or depth.dtype != np.uint16:
        raise FileNotFoundError(f"frame {i} missing or malformed in {frames_dir}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth.astype(np.float64) * depth_scale, mask > 0


def box_symmetries(to_origin: np.ndarray) -> list[np.ndarray]:
    """Mesh-frame transforms S with T @ S showing the same box geometry as T."""
    out = []
    for axis in (None, (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        flip = np.eye(4)
        if axis is not None:
            flip[:3, :3] = rotvec_to_matrix(np.pi * np.asarray(axis, dtype=float))
        out.append(np.linalg.inv(to_origin) @ flip @ to_origin)
    return out


def errors(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    d = T_est[:3, 3] - T_gt[:3, 3]
    return {"position_error_mm": float(np.linalg.norm(d) * 1e3),
            "position_error_base_xyz_mm": (d * 1e3).tolist(),
            "rotation_error_deg": rotation_angle_deg(T_gt[:3, :3].T @ T_est[:3, :3])}


def project(K: np.ndarray, p_cam: np.ndarray):
    if p_cam[2] <= 1e-6:
        return None
    uv = K @ (p_cam / p_cam[2])
    return int(round(uv[0])), int(round(uv[1]))


def draw_axes(img, K, T_cam, length, thickness, dots):
    o = project(K, T_cam[:3, 3])
    for k, color in enumerate(AXIS_COLORS_BGR):
        tip = project(K, T_cam[:3, 3] + T_cam[:3, k] * length)
        if o is not None and tip is not None:
            cv2.line(img, o, tip, color, thickness, cv2.LINE_AA)
            if dots:
                cv2.circle(img, tip, 3, (255, 255, 255), -1, cv2.LINE_AA)
    if o is not None and dots:
        cv2.circle(img, o, 3, (255, 255, 255), -1, cv2.LINE_AA)
    return o


def video_frame(rgb, K, T_cam_gt, T_cam_fp, axis_len, text_lines, zoom_px):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    centre = draw_axes(bgr, K, T_cam_gt, axis_len, 5, dots=False)
    draw_axes(bgr, K, T_cam_fp, axis_len, 2, dots=True)
    h, w = bgr.shape[:2]
    cx, cy = centre if centre is not None else (w // 2, h // 2)
    half = zoom_px // 2
    padded = cv2.copyMakeBorder(bgr, half, half, half, half, cv2.BORDER_CONSTANT)
    crop = padded[cy:cy + zoom_px, cx:cx + zoom_px]
    zoom = cv2.resize(crop, (h, h), interpolation=cv2.INTER_NEAREST)
    canvas = np.concatenate([bgr, zoom], axis=1)
    for k, line in enumerate(text_lines):
        cv2.putText(canvas, line, (10, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    legend = "thick = MuJoCo GT mesh frame, thin + dots = FP mesh frame (x red, y green, z blue)"
    cv2.putText(canvas, legend, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    meta = json.loads((run_dir / "meta.json").read_text())
    if any(abs(s - 1.0) > 1e-12 for s in meta["mesh_scale"]):
        raise SystemExit("scene mesh scale is not 1; FP would need an identically scaled mesh file")
    frames_dir = run_dir / "frames"
    K = np.asarray(meta["K"])
    T_base_camera = np.asarray(meta["T_base_camera"])
    T_camera_base = np.linalg.inv(T_base_camera)
    n = len(meta["frames"]) if args.max_frames is None else min(args.max_frames, len(meta["frames"]))
    mesh_path = Path(meta["mesh_path"])

    rgb0, depth0, mask0 = load_frame(frames_dir, 0, meta["depth_scale_m"])
    saved = SavedObject(meta["object"], mask0, mesh_path)
    tracker = MultiObjectTracker([saved], args.foundationpose_dir, run_dir / "fp_debug")
    tracked = tracker.objects[0]
    to_origin = tracked.to_origin
    symmetries = box_symmetries(to_origin) if args.symmetry == "box" else [np.eye(4)]
    axis_len = 0.6 * float(np.max(tracked.bbox[1] - tracked.bbox[0]))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = run_dir / f"fp_vs_sim_{stamp}.json"
    out_video = run_dir / f"fp_vs_sim_{stamp}.mp4"
    zoom_px = args.zoom_px
    h, w = rgb0.shape[:2]
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (w + h, h))

    records = []
    for i in range(n):
        rgb, depth, _ = (rgb0, depth0, mask0) if i == 0 else load_frame(frames_dir, i, meta["depth_scale_m"])
        started = time.perf_counter()
        if i == 0:
            tracker.register([saved], rgb, depth, K, args.est_refine_iter)
        else:
            tracker.track(rgb, depth, K, args.track_refine_iter)
        fp_s = time.perf_counter() - started

        T_cam_mesh_fp = tracked.center_pose @ to_origin
        T_base_fp = T_base_camera @ T_cam_mesh_fp
        T_base_gt = np.asarray(meta["frames"][i]["T_base_object_gt"])
        raw = errors(T_base_fp, T_base_gt)
        candidates = [errors(T_base_fp, T_base_gt @ S) for S in symmetries]
        k_best = int(np.argmin([c["rotation_error_deg"] for c in candidates]))
        rec = {
            "index": i,
            "stage": "register" if i == 0 else "track",
            "fp_seconds": fp_s,
            "raw": raw,
            "symmetry_aware": {**candidates[k_best], "symmetry_index": k_best},
            "T_base_object_fp": T_base_fp.tolist(),
            "T_base_object_gt": T_base_gt.tolist(),
            "quat_wxyz_fp": quaternion_wxyz(T_base_fp[:3, :3]).tolist(),
            "quat_wxyz_gt": quaternion_wxyz(T_base_gt[:3, :3]).tolist(),
            "mask_pixels": meta["frames"][i]["mask_pixels"],
            "visible_fraction": meta["frames"][i]["visible_fraction"],
        }
        records.append(rec)

        lines = [f"frame {i}/{n - 1} ({rec['stage']})",
                 f"raw: {raw['position_error_mm']:.2f} mm, {raw['rotation_error_deg']:.2f} deg"]
        if args.symmetry != "none":
            sa = rec["symmetry_aware"]
            lines.append(f"sym-aware: {sa['position_error_mm']:.2f} mm, {sa['rotation_error_deg']:.2f} deg"
                         f" (S{k_best})")
        writer.write(video_frame(rgb, K, T_camera_base @ T_base_gt, T_cam_mesh_fp, axis_len, lines, zoom_px))
        if i % 30 == 0 or i == n - 1:
            print("  ".join(lines))

    writer.release()
    tracked_only = [r for r in records if r["stage"] == "track"]

    def summary(rs):
        if not rs:
            return None
        return {kind: {"position_error_mm": stats([r[kind]["position_error_mm"] for r in rs]),
                       "rotation_error_deg": stats([r[kind]["rotation_error_deg"] for r in rs])}
                for kind in ("raw", "symmetry_aware")}

    payload = {
        "experiment": "fp_vs_mujoco_object_pose",
        "created": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "render_meta": {k: meta[k] for k in ("scene", "task", "object", "object_body", "mesh_path", "base_body",
                                              "K", "T_base_camera", "camera_convention", "depth_convention",
                                              "trajectory", "frame0_backprojection_check")},
        "configuration": {
            "foundationpose_dir": str(args.foundationpose_dir),
            "est_refine_iter": args.est_refine_iter,
            "track_refine_iter": args.track_refine_iter,
            "symmetry": args.symmetry,
            "fp_pose": "T_camera_mesh = center_pose @ to_origin (FP's raw return, original mesh frame)",
            "registration": "frame 0 only, simulator mask; all later frames tracker.track, no reset, no GT",
            "symmetry_definition": "S_k = inv(to_origin) @ Rot180_k @ to_origin, k in {identity, OBB x, y, z}",
            "to_origin": to_origin.tolist(),
        },
        "coordinate_units": {"position_error": "mm", "rotation_error": "deg", "poses": "m, base frame",
                             "quaternion": "wxyz, w >= 0"},
        "summary_all_frames": summary(records),
        "summary_tracked_frames": summary(tracked_only),
        "video": str(out_video),
        "frames": records,
    }
    write_json(out_json, payload)
    for name, s in (("all frames", payload["summary_all_frames"]),
                    ("tracked frames", payload["summary_tracked_frames"])):
        if s is None:
            continue
        for kind in ("raw", "symmetry_aware") if args.symmetry != "none" else ("raw",):
            p, r = s[kind]["position_error_mm"], s[kind]["rotation_error_deg"]
            print(f"{name} {kind}: position mean {p['mean']:.2f} / max {p['max']:.2f} mm, "
                  f"rotation mean {r['mean']:.2f} / max {r['max']:.2f} deg")
    print(f"Saved {out_json}\nSaved {out_video}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="folder written by render_sim_sequence.py")
    parser.add_argument("--symmetry", choices=("none", "box"), default="none",
                        help="also score against the object's symmetric poses (box: 180 deg about each OBB axis)")
    parser.add_argument("--est-refine-iter", type=int, default=5, help="registration iterations (pipeline default)")
    parser.add_argument("--track-refine-iter", type=int, default=2, help="tracking iterations (pipeline default)")
    parser.add_argument("--foundationpose-dir", type=Path, default=DEFAULT_FOUNDATIONPOSE_DIR)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--zoom-px", type=int, default=160, help="side of the zoomed crop in the video")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
