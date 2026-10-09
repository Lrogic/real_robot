"""Tool goals for a tracked object, workspace checks, and rate-limited stepping (pure math).

Poses are 4x4 T_base_tool (column vectors, metres). The tool frame is the driver's tool
frame for the configured end effector: origin at the fingertips, x along the gripper,
y the finger-opening direction, z = x cross y.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from calibration.transforms import make_T, rvec_from_R

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
UP = np.array([0.0, 0.0, 1.0])
HOVER, ENCLOSE = "hover", "enclose"


@dataclass(frozen=True)
class ManipulationConfig:
    robot: dict
    control_hz: float
    max_lin_speed_m_s: float
    max_ang_speed_rad_s: float
    max_tracking_error_m: float
    hover_standoff_m: float
    grasp_depth_m: float
    approach_pitch_rad: float
    gripper_open_margin_m: float
    gripper_open_time_s: float
    min_tool_z_m: float
    max_tool_z_m: float
    min_radius_m: float
    max_radius_m: float
    min_start_joint_margin_deg: float
    min_joint_margin_deg: float
    target_ema: float
    deadband_m: float
    max_jump_m: float
    stale_s: float

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ManipulationConfig":
        with open(path or DEFAULT_CONFIG) as f:
            c = yaml.safe_load(f)
        m, g, s, t = c["motion"], c["grasp"], c["safety"], c["target"]
        cfg = cls(robot=c.get("robot") or {},
                  control_hz=float(m["control_hz"]), max_lin_speed_m_s=float(m["max_lin_speed_m_s"]),
                  max_ang_speed_rad_s=float(m["max_ang_speed_rad_s"]),
                  max_tracking_error_m=float(m["max_tracking_error_m"]),
                  hover_standoff_m=float(g["hover_standoff_m"]), grasp_depth_m=float(g["grasp_depth_m"]),
                  approach_pitch_rad=float(np.radians(g["approach_pitch_deg"])),
                  gripper_open_margin_m=float(g["gripper_open_margin_m"]),
                  gripper_open_time_s=float(g["gripper_open_time_s"]),
                  min_tool_z_m=float(s["min_tool_z_m"]), max_tool_z_m=float(s["max_tool_z_m"]),
                  min_radius_m=float(s["min_radius_m"]), max_radius_m=float(s["max_radius_m"]),
                  min_start_joint_margin_deg=float(s["min_start_joint_margin_deg"]),
                  min_joint_margin_deg=float(s["min_joint_margin_deg"]),
                  target_ema=float(t["ema"]), deadband_m=float(t["deadband_m"]),
                  max_jump_m=float(t["max_jump_m"]), stale_s=float(t["stale_s"]))
        if cfg.control_hz <= 0 or cfg.max_lin_speed_m_s <= 0 or cfg.max_ang_speed_rad_s <= 0:
            raise ValueError("control_hz and speed limits must be > 0")
        if not 0 < cfg.target_ema <= 1:
            raise ValueError("target.ema must be in (0, 1]")
        return cfg


def approach_rotation(p_base_object: np.ndarray, pitch_down_rad: float = 0.0) -> np.ndarray:
    """Gripper pointing radially away from the base towards the object, tipped down by pitch_down_rad."""
    a = np.array([p_base_object[0], p_base_object[1], 0.0])
    n = np.linalg.norm(a)
    if n < 1e-6:
        raise ValueError("object is on the base z axis; no forward approach direction")
    a /= n
    c, s = np.cos(pitch_down_rad), np.sin(pitch_down_rad)
    return np.column_stack([c * a - s * UP, np.cross(UP, a), s * a + c * UP])


def tool_goal(p_base_object: np.ndarray, stage: str, cfg: ManipulationConfig) -> np.ndarray:
    """T_base_tool for the hover or enclose stage, with the tool height clamped to the floor."""
    R = approach_rotation(p_base_object, cfg.approach_pitch_rad)
    offset = -cfg.hover_standoff_m if stage == HOVER else cfg.grasp_depth_m
    t = np.asarray(p_base_object, dtype=np.float64) + R[:, 0] * offset
    t[2] = max(t[2], cfg.min_tool_z_m)
    return make_T(R, t)


def check_workspace(T_base_tool: np.ndarray, cfg: ManipulationConfig) -> Optional[str]:
    """Reason the goal is outside the allowed box, or None if it is inside."""
    x, y, z = T_base_tool[:3, 3]
    r = float(np.hypot(x, y))
    if z < cfg.min_tool_z_m - 1e-9:
        return f"z {z:.3f} below floor {cfg.min_tool_z_m:.3f}"
    if z > cfg.max_tool_z_m:
        return f"z {z:.3f} above {cfg.max_tool_z_m:.3f}"
    if r < cfg.min_radius_m:
        return f"radius {r:.3f} inside {cfg.min_radius_m:.3f}"
    if r > cfg.max_radius_m:
        return f"radius {r:.3f} beyond reach limit {cfg.max_radius_m:.3f}"
    return None


def step_toward(T_cur: np.ndarray, T_goal: np.ndarray, dt: float,
                cfg: ManipulationConfig) -> Tuple[np.ndarray, bool]:
    """Move T_cur towards T_goal by at most the speed limits times dt; return (pose, reached)."""
    dp = T_goal[:3, 3] - T_cur[:3, 3]
    dist = float(np.linalg.norm(dp))
    max_d = cfg.max_lin_speed_m_s * dt
    t = T_goal[:3, 3] if dist <= max_d else T_cur[:3, 3] + dp * (max_d / dist)

    rv = rvec_from_R(T_cur[:3, :3].T @ T_goal[:3, :3])
    angle = float(np.linalg.norm(rv))
    max_a = cfg.max_ang_speed_rad_s * dt
    if angle <= max_a:
        R = T_goal[:3, :3]
    else:
        R = T_cur[:3, :3] @ cv2.Rodrigues(rv * (max_a / angle))[0]
    return make_T(R, t), dist <= max_d and angle <= max_a


def joint_limit_margins(joints: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Per joint, distance (rad) to the nearer of its position limits; negative = beyond it."""
    joints = np.asarray(joints, dtype=np.float64)
    return np.minimum(joints - limits[:, 0], limits[:, 1] - joints)


def joint_margin_problem(joints, limits, min_margin_deg: float) -> Optional[str]:
    """Message naming the joint closest to a limit if it is within min_margin_deg, else None."""
    if limits is None or joints is None or not np.all(np.isfinite(joints)):
        return None
    margins = np.degrees(joint_limit_margins(joints, limits))
    j = int(np.argmin(margins))
    if margins[j] >= min_margin_deg:
        return None
    return f"joint {j + 1} is {margins[j]:.1f} deg from its limit (need {min_margin_deg:.0f})"


def pose_to_driver(T_base_tool: np.ndarray) -> list:
    """[x, y, z, rx, ry, rz] (angle-axis) as set_cartesian_positions expects."""
    return [float(v) for v in T_base_tool[:3, 3]] + [float(v) for v in rvec_from_R(T_base_tool[:3, :3])]
