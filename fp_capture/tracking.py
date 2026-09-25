"""Track several objects with FoundationPose: one estimator per object, shared networks.

FoundationPose keeps the previous pose on the estimator instance, so each object
needs its own instance; the score/refine networks and the rasterizer are shared.
Requires the fp_robot env and a FoundationPose checkout.
"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .run_folder import SavedObject

# RGB colours for the 3D boxes, cycled per object.
PALETTE = [(0, 255, 0), (255, 64, 64), (64, 160, 255), (255, 200, 0),
           (255, 0, 255), (0, 255, 255), (255, 128, 0), (160, 96, 255)]


def _import_foundationpose(fp_dir: Path):
    if not (fp_dir / "estimater.py").is_file():
        raise FileNotFoundError(f"{fp_dir} is not a FoundationPose checkout")
    if str(fp_dir) not in sys.path:
        sys.path.insert(0, str(fp_dir))
    import estimater
    import nvdiffrast.torch as dr
    import Utils
    # Importing Utils sets INFO logging, which prints ~20 lines per object per frame.
    logging.getLogger().setLevel(logging.WARNING)
    return estimater, dr, Utils


@dataclass
class TrackedObject:
    name: str
    estimator: object
    to_origin: np.ndarray  # mesh frame -> oriented-bounds frame
    bbox: np.ndarray       # (2, 3) min/max in the oriented-bounds frame
    axis_length: float
    color: tuple
    center_pose: Optional[np.ndarray] = None


class MultiObjectTracker:
    def __init__(self, objects: Sequence[SavedObject], fp_dir: Path, debug_dir: Path,
                 batched: bool = False):
        import trimesh

        self.batched = batched
        estimater, dr, self._utils = _import_foundationpose(fp_dir)
        scorer = estimater.ScorePredictor()
        refiner = estimater.PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.objects: List[TrackedObject] = []
        for index, obj in enumerate(objects):
            mesh = trimesh.load(obj.mesh_path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
                raise ValueError(f"Expected a nonempty mesh at {obj.mesh_path}")
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            estimator = estimater.FoundationPose(
                model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                scorer=scorer, refiner=refiner, glctx=glctx,
                debug=0, debug_dir=str(debug_dir),
            )
            self.objects.append(TrackedObject(
                name=obj.name, estimator=estimator, to_origin=to_origin,
                bbox=np.stack([-extents / 2, extents / 2]).reshape(2, 3),
                axis_length=float(extents.max()) * 0.6,
                color=PALETTE[index % len(PALETTE)],
            ))

    def register(self, objects: Sequence[SavedObject], rgb, depth, K, iterations: int) -> None:
        """Estimate each object's initial pose from the saved frame and its mask."""
        for tracked, saved in zip(self.objects, objects):
            print(f"Registering {tracked.name} ({saved.mesh_path.name})...")
            pose = tracked.estimator.register(K=K, rgb=rgb, depth=depth,
                                              ob_mask=saved.mask, iteration=iterations)
            tracked.center_pose = pose @ np.linalg.inv(tracked.to_origin)

    def track(self, rgb, depth, K, iterations: int) -> Dict[str, float]:
        """Update every object's pose on one frame; return FP seconds per object."""
        if self.batched:
            return self._track_batched(rgb, depth, K, iterations)
        seconds = {}
        for tracked in self.objects:
            started = time.perf_counter()
            pose = tracked.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=iterations)
            seconds[tracked.name] = time.perf_counter() - started
            tracked.center_pose = pose @ np.linalg.inv(tracked.to_origin)
        return seconds

    def _track_batched(self, rgb, depth, K, iterations: int) -> Dict[str, float]:
        from .batched_refine import track_batched

        started = time.perf_counter()
        poses = track_batched([t.estimator for t in self.objects], rgb, depth, K, iterations)
        for tracked, pose in zip(self.objects, poses):
            tracked.center_pose = pose @ np.linalg.inv(tracked.to_origin)
        # One pass covers every object, so there is no per-object split to report.
        return {"all (batched)": time.perf_counter() - started}

    def draw(self, rgb: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Return rgb with each object's 3D box, axes, and name drawn at its current pose."""
        vis = rgb.copy()
        for tracked in self.objects:
            if tracked.center_pose is None:
                continue
            vis = self._utils.draw_posed_3d_box(K, img=vis, ob_in_cam=tracked.center_pose,
                                                bbox=tracked.bbox, line_color=tracked.color)
            _draw_axes(vis, tracked.center_pose, tracked.axis_length, K)
            origin = _project(tracked.center_pose[:3, 3], K)
            if origin is not None:
                cv2.putText(vis, tracked.name, (origin[0] + 8, origin[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracked.color, 2, cv2.LINE_AA)
        return vis


def _project(point_cam: np.ndarray, K: np.ndarray):
    if point_cam[2] <= 0:
        return None
    u, v = (K @ point_cam)[:2] / point_cam[2]
    return int(round(u)), int(round(v))


def _draw_axes(rgb: np.ndarray, pose: np.ndarray, length: float, K: np.ndarray) -> None:
    # Utils.draw_xyz_axis diffs the whole image per axis (~24 ms); plain lines are enough.
    origin = _project(pose[:3, 3], K)
    for axis, color in zip(np.eye(3), [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        tip = _project(pose[:3, 3] + pose[:3, :3] @ (axis * length), K)
        if origin is not None and tip is not None:
            cv2.line(rgb, origin, tip, color, 2, cv2.LINE_AA)
