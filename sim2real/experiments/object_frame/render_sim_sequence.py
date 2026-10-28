"""Stage 1 of the synthetic FoundationPose-vs-MuJoCo object-pose test: render.

Builds the training scene (robot at the scene base pose, table, manipulated
object) through the same mjlab ``Scene`` and entity configs that training
uses, adds one fixed camera, and moves the manipulated object along a smooth
prescribed trajectory. Each frame writes the object's free-joint qpos, zeroes
its velocity and calls ``mj_forward`` (no ``mj_step``), so physics never moves
it. Ground truth is then read back from MuJoCo's body state
(``xpos``/``xquat``, the same quantity as mjlab's ``root_link_pose_w``).

Camera and depth conventions written for FoundationPose:

* The pose saved is ``T_base_camera`` of the OpenCV optical frame (x right,
  y down, z forward). The MuJoCo camera looks along its -z with +y up, so
  ``T_world_cv = T_world_mjcam @ diag(1, -1, -1, 1)``.
* ``K`` is pinhole with square pixels from the camera ``fovy``:
  ``f = (H / 2) / tan(fovy / 2)``, principal point ``((W - 1) / 2, (H - 1) / 2)``
  in pixel-index coordinates (pixel centres at integers, as FP's
  ``depth2xyzmap`` uses).
* Depth is metric z-depth along the optical axis, saved as uint16 PNG in units
  of ``depth_scale_m`` (0.1 mm, the D405 default). Pixels with no geometry are 0
  (invalid for FP).
* The mask is MuJoCo's segmentation of the object's visual geom, so occlusion
  by the robot or table is respected.

The first frame is checked: depth pixels inside the mask are back-projected with
``K`` and the saved camera pose, mapped into the mesh frame with the ground
truth, and their distance to the mesh surface is reported. A wrong axis flip,
ray-distance depth or intrinsics shows up as millimetres to centimetres.

Launch from the wxai_mjlab environment (needs an OpenGL context; EGL works headless):

  conda activate wxai_mjlab
  MUJOCO_GL=egl python real_robot/sim2real/experiments/object_frame/render_sim_sequence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from mjlab.scene import Scene
from PIL import Image
from scipy.spatial import cKDTree
from tabletop_sim.scene import load_scene_config
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_TASK, make_grasp_env_cfg

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

from common.io import write_json  # noqa: E402
from common.poses import quaternion_wxyz, rotvec_to_matrix  # noqa: E402

RESEARCH = Path(__file__).resolve().parents[4]
DEFAULT_SCENE = RESEARCH / "real_robot/sim/mujoco/configs/scenes/tabletop_jenga_wxai.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CAMERA_NAME = "fp_test_camera"
DEPTH_SCALE_M = 1e-4
CV_FROM_MJ_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def body_T(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return make_T(data.xmat[body_id].reshape(3, 3), data.xpos[body_id])


def look_at_mj_camera(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World rotation of a MuJoCo camera at ``eye`` looking at ``target``: columns are
    the camera x (right), y (up) and z (backwards) axes."""
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(back, right), back], axis=1)


def trajectory(n: int, center_b: np.ndarray, R0_b: np.ndarray, amp_m, amp_deg) -> list[np.ndarray]:
    """Smooth base-frame poses: every component starts at 0 offset with zero velocity
    (1 - cos ramps and sin^2-shaped windows), with different frequencies per axis so
    translations along base x/y/z and rotations about base x/y/z mix."""
    s = np.linspace(0.0, 1.0, n)
    window = np.sin(np.pi * s) ** 2  # 0 at both ends, smooth
    phase = 2.0 * np.pi * s
    offsets = np.stack([
        amp_m[0] * window * np.sin(phase),
        amp_m[1] * window * np.sin(2.0 * phase),
        amp_m[2] * window * np.sin(1.5 * phase),
    ], axis=1)
    angles = np.radians(np.stack([
        amp_deg[0] * window * np.sin(1.5 * phase),
        amp_deg[1] * window * np.sin(phase + 0.7),
        amp_deg[2] * window * np.sin(0.5 * phase),
    ], axis=1))
    poses = []
    for d, a in zip(offsets, angles):
        # Rotations about the base axes, applied on top of the start orientation.
        R = rotvec_to_matrix([0, 0, a[2]]) @ rotvec_to_matrix([0, a[1], 0]) @ rotvec_to_matrix([a[0], 0, 0])
        poses.append(make_T(R @ R0_b, center_b + d))
    return poses


def build_model(scene_path: Path, task_path: Path, eye_b, target_b, fovy_deg: float):
    scene_cfg = load_scene_config(scene_path)
    env_cfg = make_grasp_env_cfg(scene_path, task_path)
    scene = Scene(env_cfg.scene, device="cpu")

    # The camera is placed in the base frame; the base pose comes from the scene config.
    base = scene_cfg.robot.base_pose
    T_world_base_cfg = make_T(_quat_to_R(base[3:]), np.asarray(base[:3]))
    eye_w = (T_world_base_cfg @ [*eye_b, 1.0])[:3]
    target_w = (T_world_base_cfg @ [*target_b, 1.0])[:3]
    R_mj = look_at_mj_camera(eye_w, target_w)
    cam = scene.spec.worldbody.add_camera(name=CAMERA_NAME, pos=eye_w.tolist(), fovy=fovy_deg)
    cam.quat = quaternion_wxyz(R_mj).tolist()
    model = scene.compile()
    # The robot root body (what mjlab's root_link_pose_w reads): its parent is the world.
    roots = [model.body(b).name for b in range(1, model.nbody)
             if model.body_parentid[b] == 0 and model.body(b).name.startswith("robot/")]
    if len(roots) != 1:
        raise RuntimeError(f"expected one robot root body, got {roots}")
    return model, scene_cfg, scene_cfg.manipulated_object, roots[0]


def _quat_to_R(q_wxyz) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=np.float64))
    return R.reshape(3, 3)


def intrinsics(width: int, height: int, fovy_deg: float) -> np.ndarray:
    f = 0.5 * height / np.tan(0.5 * np.radians(fovy_deg))
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def backprojection_check(depth_m, mask, K, T_cam_obj, mesh: trimesh.Trimesh) -> dict:
    v, u = np.nonzero(mask & (depth_m > 0))
    z = depth_m[v, u]
    pts_cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    T_obj_cam = np.linalg.inv(T_cam_obj)
    pts_obj = pts_cam @ T_obj_cam[:3, :3].T + T_obj_cam[:3, 3]
    # Distance to a dense surface sample (spacing well under 1 mm for hand-sized objects).
    surface, _ = trimesh.sample.sample_surface(mesh, 400_000, seed=0)
    dist, _ = cKDTree(surface).query(pts_obj)
    return {"num_points": int(len(pts_obj)), "median_mm": float(np.median(dist) * 1e3),
            "p95_mm": float(np.percentile(dist, 95) * 1e3), "max_mm": float(dist.max() * 1e3)}


def run(args: argparse.Namespace) -> None:
    model, scene_cfg, obj_name, base_body = build_model(
        args.scene, args.task, np.asarray(args.camera_eye_base), np.asarray(args.camera_target_base),
        args.fovy_deg)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state"))
    mujoco.mj_forward(model, data)

    obj = scene_cfg.manipulated
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{obj_name}/{obj_name}")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj_name}/{obj_name}_joint")
    qadr, vadr = int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
    visual_geoms = [g for g in range(model.ngeom)
                    if model.geom_bodyid[g] == body_id and model.geom_contype[g] == 0
                    and model.geom_conaffinity[g] == 0]

    T_world_base = body_T(data, base_id)
    T_world_cv = make_T(data.cam_xmat[cam_id].reshape(3, 3), data.cam_xpos[cam_id]) @ CV_FROM_MJ_CAMERA
    T_base_camera = np.linalg.inv(T_world_base) @ T_world_cv
    K = intrinsics(args.width, args.height, args.fovy_deg)

    T_base_obj0 = np.linalg.inv(T_world_base) @ body_T(data, body_id)
    center_b = T_base_obj0[:3, 3] + np.array([0.0, 0.0, args.lift_m])
    poses_b = trajectory(args.frames, center_b, T_base_obj0[:3, :3], args.amp_m, args.amp_deg)

    out_dir = args.output or RESULTS_DIR / f"sim_sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    mesh = trimesh.load(obj.asset_path, force="mesh")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    unoccluded = mujoco.MjvOption()
    unoccluded.geomgroup[:] = 0
    unoccluded.geomgroup[int(model.geom_group[visual_geoms[0]])] = 1

    frames = []
    for i, T_b in enumerate(poses_b):
        T_w = T_world_base @ T_b
        data.qpos[qadr:qadr + 3] = T_w[:3, 3]
        data.qpos[qadr + 3:qadr + 7] = quaternion_wxyz(T_w[:3, :3])
        data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        # Ground truth from MuJoCo's body state after forward kinematics.
        T_world_obj = body_T(data, body_id)
        T_base_obj = np.linalg.inv(body_T(data, base_id)) @ T_world_obj

        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_id)
        depth = renderer.render().astype(np.float64)
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam_id)
        seg = renderer.render()
        renderer.update_scene(data, camera=cam_id, scene_option=unoccluded)
        seg_alone = renderer.render()
        renderer.disable_segmentation_rendering()

        geom_ids, obj_types = seg[..., 0], seg[..., 1]
        is_geom = obj_types == int(mujoco.mjtObj.mjOBJ_GEOM)
        mask = is_geom & np.isin(geom_ids, visual_geoms)
        alone = (seg_alone[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg_alone[..., 0], visual_geoms)
        depth[~is_geom] = 0.0
        depth_u16 = np.clip(np.round(depth / DEPTH_SCALE_M), 0, 65535).astype(np.uint16)

        Image.fromarray(rgb).save(frames_dir / f"{i:05d}_rgb.png")
        Image.fromarray(depth_u16).save(frames_dir / f"{i:05d}_depth.png")
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(frames_dir / f"{i:05d}_mask.png")
        ys, xs = np.nonzero(alone)
        inside = bool(alone.any() and xs.min() > 0 and ys.min() > 0
                      and xs.max() < args.width - 1 and ys.max() < args.height - 1)
        frames.append({
            "index": i,
            "T_base_object_gt": T_base_obj.tolist(),
            "T_world_object_gt": T_world_obj.tolist(),
            "commanded_T_base_object": T_b.tolist(),
            "mask_pixels": int(mask.sum()),
            "visible_fraction": float(mask.sum() / max(alone.sum(), 1)),
            "fully_inside_image": inside,
        })
        if i == 0:
            T_cam_obj = np.linalg.inv(T_base_camera) @ T_base_obj
            check = backprojection_check(depth_u16 * DEPTH_SCALE_M, mask, K, T_cam_obj, mesh)
            print(f"frame 0 back-projection to mesh surface: median {check['median_mm']:.3f} mm, "
                  f"p95 {check['p95_mm']:.3f} mm, max {check['max_mm']:.3f} mm")
        if i % 30 == 0:
            print(f"frame {i}/{args.frames}: mask {frames[-1]['mask_pixels']} px, "
                  f"visible {frames[-1]['visible_fraction']:.2f}")

    renderer.close()
    commanded_err = max(np.abs(np.asarray(f["T_base_object_gt"]) - np.asarray(f["commanded_T_base_object"])).max()
                        for f in frames)
    meta = {
        "experiment": "fp_vs_mujoco_object_pose_render",
        "created": datetime.now().astimezone().isoformat(),
        "scene": str(args.scene),
        "task": str(args.task),
        "mujoco_version": mujoco.__version__,
        "object": obj_name,
        "object_body": f"{obj_name}/{obj_name}",
        "mesh_path": str(obj.asset_path),
        "mesh_scale": list(obj.scale),
        "base_body": base_body,
        "width": args.width,
        "height": args.height,
        "fovy_deg": args.fovy_deg,
        "K": K.tolist(),
        "depth_scale_m": DEPTH_SCALE_M,
        "fps": args.fps,
        "T_world_base": T_world_base.tolist(),
        "T_base_camera": T_base_camera.tolist(),
        "camera_convention": "OpenCV optical frame: x right, y down, z forward; T_world_cv = T_world_mjcam @ diag(1,-1,-1,1)",
        "depth_convention": "z-depth along the optical axis in metres = png * depth_scale_m; 0 where no geometry",
        "trajectory": {"frames": args.frames, "lift_m": args.lift_m, "amp_m": list(args.amp_m),
                       "amp_deg": list(args.amp_deg),
                       "description": "base-frame offsets and rotations about base axes on top of the reset pose"},
        "frame0_backprojection_check": check,
        "max_abs_gt_minus_commanded": float(commanded_err),
        "min_visible_fraction": min(f["visible_fraction"] for f in frames),
        "all_fully_inside_image": all(f["fully_inside_image"] for f in frames),
        "frames": frames,
    }
    write_json(out_dir / "meta.json", meta)
    print(f"min visible fraction {meta['min_visible_fraction']:.2f}, all inside image "
          f"{meta['all_fully_inside_image']}, max |GT - commanded| {commanded_err:.2e}")
    print(f"Saved {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--output", type=Path, default=None, help="output run folder (must not exist)")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fovy-deg", type=float, default=58.0, help="vertical field of view (D405-like)")
    parser.add_argument("--camera-eye-base", type=float, nargs=3, default=(0.58, -0.16, 0.26),
                        help="camera position in the robot base frame (m); ~0.31 m from the target")
    parser.add_argument("--camera-target-base", type=float, nargs=3, default=(0.365, 0.0, 0.10))
    parser.add_argument("--lift-m", type=float, default=0.07,
                        help="trajectory centre height above the reset pose, so tilts clear the table")
    parser.add_argument("--amp-m", type=float, nargs=3, default=(0.04, 0.04, 0.03))
    parser.add_argument("--amp-deg", type=float, nargs=3, default=(25.0, 25.0, 40.0),
                        help="rotation amplitudes about base x, y, z")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
