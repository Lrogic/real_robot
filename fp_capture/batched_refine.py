"""Refine the poses of several objects, each with its own mesh, in one network pass.

Mirrors PoseRefinePredictor.predict and FoundationPose.track_one from the
FoundationPose checkout without modifying them. Crops are still rendered per
object (the renderer takes one mesh per call), but depth preprocessing runs once
per frame and every refinement iteration is a single forward pass over all
objects. Import only after fp_capture.tracking has put the checkout on sys.path.
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

import learning.training.predict_pose_refine as fp_refine
import Utils as fp_utils


@dataclass(frozen=True)
class RefineTarget:
    mesh: object           # trimesh, centred as FoundationPose.mesh
    mesh_tensors: dict
    diameter: float
    ob_in_cam: torch.Tensor  # (4, 4) cuda, centred-mesh frame


@torch.inference_mode()
def refine_poses_batched(refiner, targets: Sequence[RefineTarget], rgb: np.ndarray,
                         depth: torch.Tensor, xyz_map: torch.Tensor, K: np.ndarray,
                         glctx, iterations: int) -> torch.Tensor:
    """Return (N, 4, 4) refined centred-mesh poses, one row per target."""
    cfg = refiner.cfg
    if cfg["trans_rep"] != "tracknet":
        raise NotImplementedError(f"trans_rep={cfg['trans_rep']} is not supported in batched mode")
    if cfg["use_normal"]:
        raise NotImplementedError("use_normal models are not supported in batched mode")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    rgb_tensor = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
    diameters = torch.as_tensor([t.diameter for t in targets], device="cuda",
                                dtype=torch.float).reshape(-1, 1)
    trans_normalizer = cfg["trans_normalizer"]
    if not isinstance(trans_normalizer, float):
        trans_normalizer = torch.as_tensor(list(trans_normalizer), device="cuda",
                                           dtype=torch.float).reshape(1, 3)

    poses = torch.stack([t.ob_in_cam.reshape(4, 4).float() for t in targets])
    for _ in range(iterations):
        rgb_a, rgb_b, xyz_a, xyz_b = [], [], [], []
        for target, pose in zip(targets, poses):
            data = fp_refine.make_crop_data_batch(
                cfg.input_resize, pose[None], target.mesh, rgb_tensor, depth, K,
                crop_ratio=cfg["crop_ratio"], normal_map=None, xyz_map=xyz_map, cfg=cfg,
                glctx=glctx, mesh_tensors=target.mesh_tensors, dataset=refiner.dataset,
                mesh_diameter=target.diameter)
            rgb_a.append(data.rgbAs)
            rgb_b.append(data.rgbBs)
            xyz_a.append(data.xyz_mapAs)
            xyz_b.append(data.xyz_mapBs)
        A = torch.cat([torch.cat(rgb_a), torch.cat(xyz_a)], dim=1).float()
        B = torch.cat([torch.cat(rgb_b), torch.cat(xyz_b)], dim=1).float()
        with torch.autocast("cuda", enabled=refiner.amp):
            output = refiner.model(A, B)
        output = {k: v.float() for k, v in output.items()}

        if cfg["normalize_xyz"]:
            trans_delta = output["trans"] * (diameters / 2)
        else:
            trans_delta = torch.tanh(output["trans"]) * trans_normalizer
        if cfg["rot_rep"] == "axis_angle":
            rot_delta = torch.tanh(output["rot"]) * cfg["rot_normalizer"]
            rot_mat_delta = fp_refine.so3_exp_map(rot_delta).permute(0, 2, 1)
        elif cfg["rot_rep"] == "6d":
            rot_mat_delta = fp_refine.rotation_6d_to_matrix(output["rot"]).permute(0, 2, 1)
        else:
            raise RuntimeError(f"Unknown rot_rep {cfg['rot_rep']}")
        poses = fp_utils.egocentric_delta_pose_to_pose(poses, trans_delta=trans_delta,
                                                       rot_mat_delta=rot_mat_delta)
    return poses


def track_batched(estimators: List[object], rgb: np.ndarray, depth: np.ndarray,
                  K: np.ndarray, iterations: int) -> List[np.ndarray]:
    """Batched equivalent of calling estimator.track_one on every estimator.

    All estimators must share one refiner and one rasterizer context. Updates each
    estimator's pose_last and returns its 4x4 pose in the original mesh frame.
    """
    first = estimators[0]
    if any(e.pose_last is None for e in estimators):
        raise RuntimeError("Every estimator must be registered before tracking")
    if any(e.refiner is not first.refiner or e.glctx is not first.glctx for e in estimators):
        raise ValueError("Batched tracking needs estimators that share a refiner and glctx")

    depth_t = torch.as_tensor(depth, device="cuda", dtype=torch.float)
    depth_t = fp_utils.erode_depth(depth_t, radius=2, device="cuda")
    depth_t = fp_utils.bilateral_filter_depth(depth_t, radius=2, device="cuda")
    K_t = torch.as_tensor(K, dtype=torch.float, device="cuda")
    xyz_map = fp_utils.depth2xyzmap_batch(depth_t[None], K_t[None], zfar=np.inf)[0]

    targets = [RefineTarget(mesh=e.mesh, mesh_tensors=e.mesh_tensors, diameter=e.diameter,
                            ob_in_cam=e.pose_last) for e in estimators]
    poses = refine_poses_batched(first.refiner, targets, rgb, depth_t, xyz_map, K,
                                 first.glctx, iterations)
    results = []
    for index, estimator in enumerate(estimators):
        estimator.pose_last = poses[index:index + 1]
        results.append((poses[index] @ estimator.get_tf_to_centered_mesh()).cpu().numpy())
    return results
