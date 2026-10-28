"""Temp check (fp_robot env, GPU): does FoundationPose return the pose of the
original OBJ frame for an off-centre, rotated mesh, and how does the wrapper's
center_pose relate to it?

1. Writes tmp_offset_mesh/jenga_offset.obj: jenga vertices mapped by a known
   T_off (4-6 cm translation, 35 deg rotation), so the OBJ origin is far from the
   geometry and the axes are not aligned with the block.
2. Renders RGB-D of that mesh at a known T_cam_obj with FP's own renderer.
3. Runs MultiObjectTracker.register on it, then compares the raw FP pose
   (center_pose @ to_origin) and center_pose with the ground truth.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[3]
FP_DIR = RESEARCH / "FoundationPose"
sys.path.insert(0, str(RESEARCH / "real_robot/vision"))
from pipeline.foundationpose.multi_object_fp_tracker import MultiObjectTracker  # noqa: E402
from pipeline.run_folder import SavedObject  # noqa: E402

np.set_printoptions(precision=5, suppress=True)
SRC = RESEARCH / "assets/meshes/jenga/jenga.obj"
OUT_DIR = HERE / "tmp_offset_mesh"
OUT = OUT_DIR / "jenga_offset.obj"

T_off = np.eye(4)
T_off[:3, :3] = R.from_rotvec(np.radians(35) * np.array([1, 2, 3]) / np.sqrt(14)).as_matrix()
T_off[:3, 3] = [0.04, -0.03, 0.06]


def make_offset_mesh() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    mesh = trimesh.load(SRC, process=False)
    mesh.apply_transform(T_off)
    mesh.export(OUT)


def rot_err_deg(Ra, Rb) -> float:
    return float(np.degrees(R.from_matrix(Ra.T @ Rb).magnitude()))


def box_sym_err_deg(T_est, T_gt, sym_axes_obj) -> float:
    """Smallest rotation error over the block's 180-degree symmetries (axes in OBJ frame)."""
    best = rot_err_deg(T_est[:3, :3], T_gt[:3, :3])
    for ax in sym_axes_obj:
        S = R.from_rotvec(np.pi * ax).as_matrix()
        best = min(best, rot_err_deg(T_est[:3, :3], T_gt[:3, :3] @ S))
    return best


def main() -> None:
    make_offset_mesh()
    mesh = trimesh.load(OUT, force="mesh")
    v = np.asarray(mesh.vertices)
    print("offset mesh AABB centre (FP model_center):", (v.min(0) + v.max(0)) / 2)

    H, W = 480, 640
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    # Ground truth: the OBJ-frame pose in the camera. Geometry sits at origin + offset,
    # so place it so the block itself is ~0.35 m in front of the camera.
    T_gt = np.eye(4)
    T_gt[:3, :3] = R.from_euler("xyz", [200, 20, 40], degrees=True).as_matrix()
    geom_centre_obj = T_off[:3, 3]
    T_gt[:3, 3] = np.array([0.01, -0.02, 0.35]) - T_gt[:3, :3] @ geom_centre_obj

    tracker = MultiObjectTracker(
        [SavedObject("obj1", np.zeros((H, W), bool), OUT)], FP_DIR, HERE / "tmp_debug"
    )
    import Utils  # importable once the tracker has put FoundationPose on sys.path

    tracked = tracker.objects[0]
    est = tracked.estimator
    color, depth, _ = Utils.nvdiffrast_render(
        K=K, H=H, W=W, ob_in_cams=torch.as_tensor(T_gt, dtype=torch.float, device="cuda")[None],
        glctx=est.glctx, mesh=mesh, use_light=True,
    )
    rgb = (color[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    depth = depth[0].cpu().numpy()
    mask = depth > 0
    print("rendered mask pixels:", int(mask.sum()))

    tracker.register([SavedObject("obj1", mask, OUT)], rgb, depth, K, iterations=5)
    center_pose = tracked.center_pose
    raw = center_pose @ tracked.to_origin

    # Block symmetry axes: the source mesh's x/y/z, carried into the offset OBJ frame.
    sym_axes = [T_off[:3, :3] @ e for e in np.eye(3)]
    for name, T in (("raw FP (center_pose @ to_origin)", raw), ("center_pose", center_pose)):
        print(f"\n{name}:")
        print("  translation error vs GT [mm]:", 1000 * (T[:3, 3] - T_gt[:3, 3]),
              "|.| =", 1000 * np.linalg.norm(T[:3, 3] - T_gt[:3, 3]))
        print("  rotation error vs GT [deg]:", rot_err_deg(T[:3, :3], T_gt[:3, :3]),
              " modulo box symmetry:", box_sym_err_deg(T, T_gt, sym_axes))
    pts = v[:: max(1, len(v) // 2000)]
    fit = np.linalg.norm((pts @ raw[:3, :3].T + raw[:3, 3]) - (pts @ T_gt[:3, :3].T + T_gt[:3, 3]), axis=1)
    print("\nraw FP: mean vertex error vs GT [mm]:", 1000 * fit.mean())
    print("to_origin rotation angle [deg]:", rot_err_deg(np.eye(3), tracked.to_origin[:3, :3]),
          " to_origin translation [mm]:", 1000 * tracked.to_origin[:3, 3])


if __name__ == "__main__":
    main()
