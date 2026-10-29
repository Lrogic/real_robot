"""Load jenga/config.yaml and derive the training quantities from the sim scene and task JSONs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ..observations import make_T, quat_wxyz_to_matrix

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def _path(base: Path, value) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


@dataclass(frozen=True)
class JengaPolicyConfig:
    path: Path
    scene_path: Path
    task_path: Path
    embodiment: str
    object_name: str              # the scene's manipulated object
    mesh_path: Path               # its mesh file (FoundationPose must track the same file)
    tracked_name: Optional[str]
    box_center: np.ndarray
    T_base_initial: np.ndarray    # training reset pose (no yaw/xy randomization), base frame
    target_lift_height: float
    control_hz: float
    episode_steps: int
    gripper_open: float
    gripper_closed: float
    calibration: Optional[Path]
    robot: dict
    episode_key: str

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "JengaPolicyConfig":
        path = Path(path or DEFAULT_CONFIG).resolve()
        raw = yaml.safe_load(path.read_text())
        base = path.parent
        scene_path = _path(base, raw["sim"]["scene"])
        task_path = _path(base, raw["sim"]["task"])
        scene = json.loads(scene_path.read_text())
        task = json.loads(task_path.read_text())

        base_pose = np.asarray(scene["robot"].get("base_pose", [0, 0, 0, 1, 0, 0, 0]), dtype=np.float64)
        if not np.allclose(base_pose[3:], [1, 0, 0, 0]):
            # target_lift_height is along world z; with a rotated base it is not base z.
            raise ValueError(f"{scene_path}: robot base_pose rotation must be identity, got {base_pose[3:]}")
        T_world_base = make_T(np.eye(3), base_pose[:3])

        name = scene["manipulated_object"]
        obj = next(o for o in scene["objects"] if o["name"] == name)
        if obj.get("primitive") != "mesh" or not obj.get("asset_path"):
            raise ValueError(f"{scene_path}: manipulated object {name!r} is not a mesh")
        if any(abs(s - 1.0) > 1e-12 for s in obj.get("mesh_scale", [1.0, 1.0, 1.0])):
            raise ValueError(f"{scene_path}: {name} mesh scale must be 1 to match FoundationPose's mesh")
        pose = np.asarray(obj["initial_poses"][0], dtype=np.float64)
        T_world_obj = make_T(quat_wxyz_to_matrix(pose[3:]), pose[:3])

        if raw["object"].get("symmetry", "box") != "box":
            raise ValueError("object.symmetry: only 'box' is implemented")
        timing = task["timing"]
        return cls(
            path=path,
            scene_path=scene_path,
            task_path=task_path,
            embodiment=scene["robot"]["embodiment"],
            object_name=name,
            mesh_path=_path(scene_path.parent, obj["asset_path"]),
            tracked_name=raw["object"].get("tracked_name"),
            box_center=np.asarray(raw["object"].get("box_center", [0, 0, 0]), dtype=np.float64),
            T_base_initial=np.linalg.inv(T_world_base) @ T_world_obj,
            target_lift_height=float(task["target_lift_height"]),
            control_hz=float(timing["control_hz"]),
            episode_steps=int(timing["episode_steps"]),
            gripper_open=float(raw["embodiment"]["gripper_open"]),
            gripper_closed=float(raw["embodiment"]["gripper_closed"]),
            calibration=_path(base, raw.get("calibration", {}).get("T_base_camera")),
            robot={k: v for k, v in (raw.get("robot") or {}).items() if v},
            episode_key=str(raw.get("episode", {}).get("key", "s")),
        )

    def describe(self) -> dict:
        return {"config": str(self.path), "scene": str(self.scene_path), "task": str(self.task_path),
                "embodiment": self.embodiment, "object": self.object_name, "mesh": str(self.mesh_path),
                "tracked_name": self.tracked_name, "box_center": self.box_center.tolist(),
                "T_base_initial": self.T_base_initial.tolist(), "target_lift_height": self.target_lift_height,
                "control_hz": self.control_hz, "episode_steps": self.episode_steps,
                "gripper_open": self.gripper_open, "gripper_closed": self.gripper_closed,
                "episode_key": self.episode_key}
