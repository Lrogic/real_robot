"""The ee_delta_pose_binary_gripper action as a TCP goal for the Trossen driver. Pure numpy.

Mirrors tabletop_sim's ClippedDifferentialIKAction (mjlab DifferentialIKAction, relative
mode) and GripperAction (binary): the six arm values are clamped to [-1, 1], translation
scaled by ee_pos_scale (m) and rotation by ee_rot_scale (rad, axis-angle), and applied to
the TCP measured at the start of the step, both in the robot base frame:

    p_target = p + dp        R_target = R(dr) @ R

The gripper value opens at a >= 0 and closes below (no clamp). The sim's 25 damped IK
solves per step are replaced by the driver's own Cartesian IK on the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observations import make_T, quat_wxyz_from_matrix, rotvec_to_matrix


@dataclass(frozen=True)
class EeTarget:
    T_base_target: np.ndarray   # commanded TCP pose, base frame
    gripper: float              # carriage target, m
    gripper_open: bool
    clipped: np.ndarray         # the six arm values after the [-1, 1] clamp
    delta_pos: np.ndarray       # m, base frame, after scaling
    delta_rotvec: np.ndarray    # rad, base frame, after scaling


def matrix_to_rotvec(R) -> np.ndarray:
    """Axis-angle vector (angle in [0, pi]) of a rotation matrix."""
    w, x, y, z = quat_wxyz_from_matrix(R)
    v = np.array([x, y, z])
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.zeros(3)
    return v / s * (2.0 * np.arctan2(s, w))


def ee_delta_target(T_base_tcp, action, *, pos_scale: float, rot_scale: float, gripper_open: float,
                    gripper_closed: float, action_scale: float = 1.0) -> EeTarget:
    """The TCP goal and gripper target for one raw policy action (7 values)."""
    a = np.asarray(action, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"non-finite policy action {a.tolist()}")
    T = np.asarray(T_base_tcp, dtype=np.float64)
    clipped = np.clip(a[:6], -1.0, 1.0)
    dp = clipped[:3] * pos_scale * action_scale
    dr = clipped[3:] * rot_scale * action_scale
    target = make_T(rotvec_to_matrix(dr) @ T[:3, :3], T[:3, 3] + dp)
    opened = bool(a[6] >= 0.0)
    return EeTarget(T_base_target=target, gripper=gripper_open if opened else gripper_closed,
                    gripper_open=opened, clipped=clipped, delta_pos=dp, delta_rotvec=dr)


def driver_pose6(T) -> list:
    """[x, y, z, rx, ry, rz] (m, angle-axis rad), as set_cartesian_positions takes."""
    T = np.asarray(T, dtype=np.float64)
    return [float(v) for v in np.concatenate([T[:3, 3], matrix_to_rotvec(T[:3, :3])])]


def in_workspace(p, lo, hi) -> bool:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return bool(np.all(p >= lo) and np.all(p <= hi))
