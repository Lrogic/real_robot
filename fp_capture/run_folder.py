"""On-disk layout for one scene and its segmented objects.

    <run>/camera_intrinsics.json
    <run>/depth.npy                aligned depth in metres
    <run>/rgb.png
    <run>/obj1/mask_overlay.png
    <run>/obj1/mask.png            one-channel uint8 0/255, as format_foundationpose_mask.py
    <run>/obj1/<mesh_folder>/...   only if a mesh was selected
    <run>/tracking.mp4             written by track_objects.py --save-video
"""

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .camera import Capture
from .drawing import overlay_mask

INTRINSICS_FILE = "camera_intrinsics.json"
DEPTH_FILE = "depth.npy"
RGB_FILE = "rgb.png"
MASK_FILE = "mask.png"
OVERLAY_FILE = "mask_overlay.png"
VIDEO_FILE = "tracking.mp4"
MESH_SUFFIXES = (".obj", ".stl", ".ply")
_OBJ_DIR = re.compile(r"obj(\d+)")


@dataclass(frozen=True)
class SavedObject:
    name: str
    mask: np.ndarray  # HxW bool
    mesh_path: Optional[Path]


@dataclass(frozen=True)
class SavedRun:
    root: Path
    capture: Capture
    objects: List[SavedObject]


def ensure_empty_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{path} already exists and is not an empty directory")


def format_foundationpose_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any() or mask.all():
        raise ValueError("Mask is entirely background or entirely foreground")
    return np.where(mask, 255, 0).astype(np.uint8)


def _write_image(path: Path, image: np.ndarray) -> None:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


class RunWriter:
    def __init__(self, root: Path, capture: Capture):
        ensure_empty_dir(root)
        self.root = root
        self.capture = capture
        self.num_objects = 0

    def _write_scene(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        height, width = self.capture.rgb.shape[:2]
        (self.root / INTRINSICS_FILE).write_text(json.dumps(
            {"width": width, "height": height, "K": self.capture.K.tolist()},
            indent=2) + "\n")
        np.save(self.root / DEPTH_FILE, self.capture.depth)
        _write_image(self.root / RGB_FILE, self.capture.rgb)

    def save_object(self, mask: np.ndarray, mesh_dir: Optional[Path]) -> Path:
        """Save one object's files into the next obj{i} folder and return it."""
        formatted = format_foundationpose_mask(mask)
        if self.num_objects == 0:
            self._write_scene()
        obj_dir = self.root / f"obj{self.num_objects + 1}"
        obj_dir.mkdir()
        _write_image(obj_dir / OVERLAY_FILE, overlay_mask(self.capture.rgb, mask))
        _write_image(obj_dir / MASK_FILE, formatted)
        if mesh_dir is not None:
            shutil.copytree(mesh_dir, obj_dir / mesh_dir.name)
        self.num_objects += 1
        print(f"Saved {obj_dir}" + (f" with mesh {mesh_dir.name}" if mesh_dir else ""))
        return obj_dir


def find_mesh_file(obj_dir: Path) -> Optional[Path]:
    """Return the single mesh file inside obj_dir's mesh folder, or None if there is none."""
    candidates = sorted(path for sub in obj_dir.iterdir() if sub.is_dir()
                        for path in sub.rglob("*")
                        if path.is_file() and path.suffix.lower() in MESH_SUFFIXES)
    if len(candidates) > 1:
        raise ValueError(f"{obj_dir} has several mesh files, expected one: "
                         + ", ".join(str(p.relative_to(obj_dir)) for p in candidates))
    return candidates[0] if candidates else None


def load_run(root: Path) -> SavedRun:
    for name in (INTRINSICS_FILE, DEPTH_FILE, RGB_FILE):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing {root / name}; run setup first")
    bgr = cv2.imread(str(root / RGB_FILE), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read {root / RGB_FILE}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(root / DEPTH_FILE, allow_pickle=False)
    K = np.asarray(json.loads((root / INTRINSICS_FILE).read_text())["K"], dtype=np.float64)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"{DEPTH_FILE} and {RGB_FILE} have different resolutions")

    obj_dirs = sorted((p for p in root.iterdir() if p.is_dir() and _OBJ_DIR.fullmatch(p.name)),
                      key=lambda p: int(_OBJ_DIR.fullmatch(p.name).group(1)))
    objects = []
    for obj_dir in obj_dirs:
        mask = cv2.imread(str(obj_dir / MASK_FILE), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Missing or unreadable {obj_dir / MASK_FILE}")
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"{obj_dir / MASK_FILE} does not match {RGB_FILE} resolution")
        objects.append(SavedObject(obj_dir.name, mask > 0, find_mesh_file(obj_dir)))
    return SavedRun(root, Capture(rgb=rgb, depth=depth, K=K), objects)
