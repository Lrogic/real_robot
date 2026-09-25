"""Write FoundationPose inputs for one scene and any number of segmented objects.

Layout:
    <run>/camera_intrinsics.json
    <run>/depth.npy
    <run>/rgb.png
    <run>/obj1/mask_overlay.png
    <run>/obj1/mask.png            one-channel uint8 0/255, as format_foundationpose_mask.py
    <run>/obj1/<mesh_folder>/...   only if a mesh was selected
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .camera import Capture
from .drawing import overlay_mask


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
        (self.root / "camera_intrinsics.json").write_text(json.dumps(
            {"width": width, "height": height, "K": self.capture.K.tolist()},
            indent=2) + "\n")
        np.save(self.root / "depth.npy", self.capture.depth)
        _write_image(self.root / "rgb.png", self.capture.rgb)

    def save_object(self, mask: np.ndarray, mesh_dir: Optional[Path]) -> Path:
        """Save one object's files into the next obj{i} folder and return it."""
        formatted = format_foundationpose_mask(mask)
        if self.num_objects == 0:
            self._write_scene()
        obj_dir = self.root / f"obj{self.num_objects + 1}"
        obj_dir.mkdir()
        _write_image(obj_dir / "mask_overlay.png", overlay_mask(self.capture.rgb, mask))
        _write_image(obj_dir / "mask.png", formatted)
        if mesh_dir is not None:
            shutil.copytree(mesh_dir, obj_dir / mesh_dir.name)
        self.num_objects += 1
        print(f"Saved {obj_dir}" + (f" with mesh {mesh_dir.name}" if mesh_dir else ""))
        return obj_dir
