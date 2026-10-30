"""Load jenga/config.yaml and derive the training quantities from the sim scene and task JSONs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ..observations import make_T, quat_wxyz_to_matrix

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
ACTION_SPACE = "ee_delta_pose_binary_gripper"
CONTROL_MODES = ("shadow", "step", "slow", "full")


def _path(base: Path, value) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


@dataclass(frozen=True)
class ControlConfig:
    mode: str
    ready_key: str
    kill_key: str
    step_key: str
    slow_time_dilation: int
    step_goal_time_s: float
    action_scale: float
    max_frame_age_s: float
    max_frame_gap_s: float
    gripper_goal_time_s: float
    gripper_limit_margin_m: float
    ready_move_time_s: float
    ready_tolerance_rad: float
    ready_gripper_tolerance: float
    stop_goal_time_s: float

    @classmethod
    def from_raw(cls, raw: dict) -> "ControlConfig":
        c = cls(
            mode=str(raw.get("mode", "shadow")),
            ready_key=str(raw.get("ready_key", "r")),
            kill_key=str(raw.get("kill_key", "k")),
            step_key=str(raw.get("step_key", "n")),
            slow_time_dilation=int(raw.get("slow_time_dilation", 4)),
            step_goal_time_s=float(raw.get("step_goal_time_s", 1.0)),
            action_scale=float(raw.get("action_scale", 1.0)),
            max_frame_age_s=float(raw.get("max_frame_age_s", 0.030)),
            max_frame_gap_s=float(raw.get("max_frame_gap_s", 0.030)),
            gripper_goal_time_s=float(raw.get("gripper_goal_time_s", 0.5)),
            gripper_limit_margin_m=float(raw.get("gripper_limit_margin_m", 0.002)),
            ready_move_time_s=float(raw.get("ready_move_time_s", 4.0)),
            ready_tolerance_rad=float(raw.get("ready_tolerance_rad", 0.02)),
            ready_gripper_tolerance=float(raw.get("ready_gripper_tolerance", 0.003)),
            stop_goal_time_s=float(raw.get("stop_goal_time_s", 0.2)),
        )
        c.validate()
        return c

    def validate(self) -> None:
        if self.mode not in CONTROL_MODES:
            raise ValueError(f"control.mode must be one of {CONTROL_MODES}, got {self.mode!r}")
        if self.slow_time_dilation < 1:
            raise ValueError("control.slow_time_dilation must be >= 1")
        if not 0.0 < self.action_scale <= 1.0:
            raise ValueError("control.action_scale must be in (0, 1]")
        if self.gripper_limit_margin_m < 0:
            raise ValueError("control.gripper_limit_margin_m must be >= 0")
        for name in ("step_goal_time_s", "max_frame_age_s", "max_frame_gap_s", "gripper_goal_time_s",
                     "ready_move_time_s", "ready_tolerance_rad", "ready_gripper_tolerance", "stop_goal_time_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"control.{name} must be > 0")

    def keys(self) -> dict:
        return {"ready": self.ready_key, "kill": self.kill_key, "step": self.step_key}


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
    ready_joint_pos: np.ndarray   # arm joints of EmbodimentCfg.ready_joint_pos, rad
    ready_gripper: float          # its carriage position, m
    action_space: str
    ee_pos_scale: float           # task ActionConfig
    ee_rot_scale: float
    checkpoint: Optional[Path]
    calibration: Optional[Path]
    robot: dict
    episode_key: str
    control: ControlConfig
    workspace_min: np.ndarray     # commanded TCP position box, base frame
    workspace_max: np.ndarray

    @property
    def control_period_s(self) -> float:
        return 1.0 / self.control_hz

    def with_mode(self, mode: Optional[str]) -> "JengaPolicyConfig":
        if mode is None:
            return self
        control = replace(self.control, mode=mode)
        control.validate()
        return replace(self, control=control)

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
        action_space = scene["robot"].get("action_space")
        if action_space != ACTION_SPACE:
            raise ValueError(f"{scene_path}: robot.action_space is {action_space!r}; only {ACTION_SPACE!r} "
                             "is deployed")
        action = task.get("action", {})
        emb_raw = raw["embodiment"]
        ready = np.asarray(emb_raw["ready_joint_pos"], dtype=np.float64)
        if ready.shape != (6,):
            raise ValueError("embodiment.ready_joint_pos must have 6 arm joint values")
        safety = raw.get("safety", {})
        workspace_min = np.asarray(safety["workspace_min"], dtype=np.float64)
        workspace_max = np.asarray(safety["workspace_max"], dtype=np.float64)
        if workspace_min.shape != (3,) or workspace_max.shape != (3,) or np.any(workspace_min >= workspace_max):
            raise ValueError("safety.workspace_min / workspace_max must be 3 values each, min < max")
        control = ControlConfig.from_raw(raw.get("control") or {})
        episode_key = str(raw.get("episode", {}).get("key", "s"))
        keys = [episode_key, *control.keys().values()]
        if len(set(keys)) != len(keys) or any(len(k) != 1 or k == "q" for k in keys):
            raise ValueError(f"episode and control keys must be distinct single characters other than q: {keys}")
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
            gripper_open=float(emb_raw["gripper_open"]),
            gripper_closed=float(emb_raw["gripper_closed"]),
            ready_joint_pos=ready,
            ready_gripper=float(emb_raw["ready_gripper"]),
            action_space=action_space,
            # ActionConfig defaults when the task leaves them out.
            ee_pos_scale=float(action.get("ee_pos_scale", 0.01)),
            ee_rot_scale=float(action.get("ee_rot_scale", 0.05)),
            checkpoint=_path(base, (raw.get("policy") or {}).get("checkpoint")),
            calibration=_path(base, raw.get("calibration", {}).get("T_base_camera")),
            robot={k: v for k, v in (raw.get("robot") or {}).items() if v},
            episode_key=episode_key,
            control=control,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
        )

    def describe(self) -> dict:
        return {"config": str(self.path), "scene": str(self.scene_path), "task": str(self.task_path),
                "embodiment": self.embodiment, "object": self.object_name, "mesh": str(self.mesh_path),
                "tracked_name": self.tracked_name, "box_center": self.box_center.tolist(),
                "T_base_initial": self.T_base_initial.tolist(), "target_lift_height": self.target_lift_height,
                "control_hz": self.control_hz, "episode_steps": self.episode_steps,
                "gripper_open": self.gripper_open, "gripper_closed": self.gripper_closed,
                "ready_joint_pos": self.ready_joint_pos.tolist(), "ready_gripper": self.ready_gripper,
                "action_space": self.action_space, "ee_pos_scale": self.ee_pos_scale,
                "ee_rot_scale": self.ee_rot_scale, "checkpoint": str(self.checkpoint),
                "episode_key": self.episode_key, "control": asdict(self.control),
                "workspace_min": self.workspace_min.tolist(), "workspace_max": self.workspace_max.tolist()}
