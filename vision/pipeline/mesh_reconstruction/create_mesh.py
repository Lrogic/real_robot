"""The setup stage's "Create mesh" button: a textured mesh of the masked object from MV-SAM3D.

1. Three views are planned around the object's masked depth points (views.py, config.yaml):
   the front view looks down at the centroid from the setup camera's side, at the distance where
   the object fills bbox_fraction of the frame; the side views rotate it +/- side_angle_deg about
   the vertical through the centroid.
2. After Enter, the follower moves to the front view, then each side view, and home from the
   front view's stand-off, each move taking goal_time_s, and captures at the three views. Between
   views it backs out by retreat_m and goes round in arcs of at most arc_step_deg (views.transit). Home is where the camera arm was held (the
   locked move joints or the parked joints), so the run's camera pose stays valid. If the front
   view's automatic mask misses the target size by more than bbox_tolerance, the front view is
   moved once to correct it.
3. Each view's mask starts from the object's projected box: accept it (s) or fix it.
4. worker.py runs MV-SAM3D in the sam3d env and writes a textured mesh and a preview.
5. The preview is shown; name the mesh and choose whether to also save it to --meshes-dir.

Files go under <run>/mesh_reconstruction/<k>/: views/view_<i>/ (run_folder.py layout), out/
(worker mesh.glb, preview.png), worker.log, and <name>/ with <name>.obj, <name>.obj.mtl and
<name>.png, which the setup stage then copies into the object's obj<i>/ like any other mesh.
"""

import os
import re
import shutil
import subprocess
from functools import partial
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh
import yaml

from ..camera import capture_frame
from ..pipeline_args import camera_settings
from ..run_folder import RunWriter
from ..segmentation.mask_selection_ui import SegmentationApp
from . import views

CONFIG = Path(__file__).with_name("config.yaml")
WORKER = Path(__file__).with_name("worker.py")
PREVIEW_WINDOW = "MV-SAM3D mesh"


def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg["model"]["mvsam3d_dir"] = (path.parent / cfg["model"]["mvsam3d_dir"]).resolve()
    return cfg


def capture_views(arm, plan, points, settings, tracker, cfg):
    """Front, side +, side -, then back to the front stand-off; returns [(Capture, reached T_base_camera)].

    plan(distance=None) -> (pose(angle, distance), front distance); points: the object's, base frame.
    Between views the camera backs out and goes round (views.transit). The caller homes the arm.
    """
    t, view_cfg = cfg["arm"]["goal_time_s"], cfg["views"]

    def visit(goal):
        reached = arm.move_camera(goal, t)
        error = np.linalg.norm(reached[:3, 3] - goal[:3, 3])
        if error > view_cfg["max_reach_error_m"]:
            raise RuntimeError(f"camera stopped {1000 * error:.0f} mm from its goal (out of the follower's reach?)")
        return reached

    def shoot(goal):
        reached = visit(goal)
        return capture_frame(settings), reached

    pose, distance = plan()
    front = shoot(pose(0, distance))
    capture, reached = front
    auto = tracker.preview(capture.rgb, [], [], views.project_box(points, reached, capture.K, capture.rgb.shape))
    if auto is not None and auto[0].any():
        fraction = views.box_fraction(views.mask_box(auto[0]), capture.rgb.shape)
        if abs(fraction - view_cfg["bbox_fraction"]) > view_cfg["bbox_tolerance"]:
            print(f"Front view fills {fraction:.0%} of the frame; moving to correct it once.")
            pose, distance = plan(distance * fraction / view_cfg["bbox_fraction"])
            front = shoot(pose(0, distance))
    shots, angle = [front], 0.0
    for side in (view_cfg["side_angle_deg"], -view_cfg["side_angle_deg"]):
        *waypoints, goal = views.transit(pose, distance, angle, side, view_cfg)
        for waypoint in waypoints:
            visit(waypoint)
        shots.append(shoot(goal))
        angle = side
    for waypoint in views.transit(pose, distance, angle, 0.0, view_cfg)[:-1]:  # out and round to the front
        visit(waypoint)
    return shots


def export_obj(mesh: trimesh.Trimesh, folder: Path, name: str) -> None:
    """<name>.obj, <name>.obj.mtl and <name>.png, the layout of assets/meshes/."""
    text, files = trimesh.exchange.obj.export_obj(mesh, return_texture=True, mtl_name=f"{name}.obj.mtl")
    folder.mkdir(parents=True)
    (texture_name, texture), = [(k, v) for k, v in files.items() if not k.endswith(".mtl")]
    (folder / f"{name}.obj").write_text(text)
    (folder / f"{name}.obj.mtl").write_text(files[f"{name}.obj.mtl"].decode().replace(texture_name, f"{name}.png"))
    (folder / f"{name}.png").write_bytes(texture)


def ask_name(meshes_dir: Path):
    """(name, whether to also save it to meshes_dir), from the terminal."""
    while True:
        name = input("Mesh name (letters, digits, _ or -): ").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            print("Use only letters, digits, _ and -.")
            continue
        to_assets = input(f"Also save it to {meshes_dir / name}? [y/N]: ").strip().lower() == "y"
        if to_assets and (meshes_dir / name).exists():
            print(f"{meshes_dir / name} already exists; choose another name.")
            continue
        return name, to_assets


def run_worker(views_dir: Path, out_dir: Path, log: Path, cfg: dict) -> None:
    model = cfg["model"]
    prefix = str(Path(model["python"]).parents[1])  # sam3d env: its CUDA toolkit for gsplat/nvdiffrast
    env = {**os.environ, "PYTHONPATH": str(model["mvsam3d_dir"]), "HF_HUB_CACHE": model["hf_hub_cache"],
           "TORCH_HOME": model["torch_home"], "CONDA_PREFIX": prefix, "CUDA_HOME": prefix}
    cmd = [model["python"], str(WORKER), str(views_dir.resolve()), str(out_dir.resolve()), "--seed", str(model["seed"]),
           "--stage1-steps", str(model["stage1_steps"]), "--stage2-steps", str(model["stage2_steps"]),
           "--tile", str(cfg["preview_tile_px"]),
           "--edge-px", str(model["depth_edge_px"])]
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=model["mvsam3d_dir"], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def create_mesh(mask: np.ndarray, *, args, camera, capture, tracker) -> Optional[Path]:
    """Mesh folder for the masked object, or None if cancelled or failed (the follower is home again)."""
    cfg = load_config()
    points = views.masked_points(capture.depth, capture.K, mask, camera.T_base_camera)
    if len(points) < 100:
        print("Too few depth points in the mask to plan the mesh views.")
        return None
    plan = partial(views.plan, points, camera.T_base_camera, capture.K, mask.shape, cfg["views"])
    try:
        input(f"The camera arm will move by itself to 3 views around the object ({cfg['arm']['goal_time_s']} s "
              "per move). Press Enter to start, Ctrl-C to cancel... ")
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    try:
        shots = capture_views(camera.arm, plan, points, camera_settings(args), tracker, cfg)
    except Exception as error:  # driver IK, capture: give up, but always bring the camera home
        print(f"Mesh views failed: {error}")
        return None
    finally:
        print("Returning the camera arm to where it was held...")
        camera.arm.home(cfg["arm"]["goal_time_s"])

    root = args.save_path / "mesh_reconstruction"
    root.mkdir(parents=True, exist_ok=True)
    work = root / str(1 + sum(1 for p in root.iterdir() if p.is_dir()))
    for i, (shot, T_base_camera) in enumerate(shots):
        writer = RunWriter(work / "views" / f"view_{i}", shot, {"mode": "mesh_view", "T_base_camera": T_base_camera.tolist()})
        app = SegmentationApp(shot.rgb, partial(tracker.preview, shot.rgb), writer.save_object, None,
                              box=views.project_box(points, T_base_camera, shot.K, shot.rgb.shape),
                              window=f"Mesh view {i + 1}/{len(shots)}: s accepts, q cancels")
        if app.run() == 0:
            print("Mesh creation cancelled.")
            return None

    print(f"Running MV-SAM3D (about 1.5 min; log {work / 'worker.log'})...")
    try:
        run_worker(work / "views", work / "out", work / "worker.log", cfg)
    except subprocess.CalledProcessError:
        print(f"MV-SAM3D failed; see {work / 'worker.log'}")
        return None
    cv2.imshow(PREVIEW_WINDOW, cv2.imread(str(work / "out" / "preview.png")))
    cv2.waitKey(500)
    name, to_assets = ask_name(args.meshes_dir)
    cv2.destroyWindow(PREVIEW_WINDOW)
    mesh_dir = work / name
    export_obj(trimesh.load(work / "out" / "mesh.glb", force="mesh"), mesh_dir, name)
    if to_assets:
        shutil.copytree(mesh_dir, args.meshes_dir / name)
        print(f"Saved {args.meshes_dir / name}")
    return mesh_dir
