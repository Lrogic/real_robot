"""Actor observation terms of the grasp task, computed from plain arrays.

Mirrors the actor group of tabletop_sim/tasks/grasp/env_cfg.py::_observations and the
term functions in tabletop_sim/tasks/grasp/mdp/observations.py. Everything is in the
robot base frame. No noise or corruption is applied, and is_grasped (critic only) is not
built. Pure numpy, so the vision env and the simulator env can both import it.

Quaternions are (w, x, y, z) with w >= 0, the sign rule of task_state.root_relative_pose.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# (name, size) in the order the observation manager concatenates the actor group.
ACTOR_TERMS = (
    ("qpos", 8),
    ("qvel", 8),
    ("ee_pose", 7),
    ("gripper_width", 1),
    ("object_pose", 7),
    ("tcp_to_obj_pos", 3),
    ("obj_to_goal_pos", 3),
)
ACTOR_DIM = sum(size for _, size in ACTOR_TERMS)


# ----- rotations ---------------------------------------------------------------------


def rotvec_to_matrix(rotvec) -> np.ndarray:
    """Angle-axis rotation (the driver's Cartesian orientation convention)."""
    rvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = rvec / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def quat_wxyz_from_matrix(rotation) -> np.ndarray:
    """Unit quaternion (w, x, y, z) with w >= 0."""
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    quat = np.asarray(q, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return -quat if quat[0] < 0.0 else quat


def quat_wxyz_to_matrix(quat) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def make_T(rotation, position) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return T


def pose7(T) -> np.ndarray:
    """4x4 pose -> [x, y, z, qw, qx, qy, qz] with qw >= 0."""
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[:3, 3], quat_wxyz_from_matrix(T[:3, :3])])


# ----- terms ---------------------------------------------------------------------------


def carriages_from_gripper(gripper_value: float) -> np.ndarray:
    """The hardware's single gripper coordinate (m) as the sim's (left, right) carriage
    joints: measured gaps match one-to-one (sim2real/experiments/gripper_width)."""
    return np.array([gripper_value, gripper_value], dtype=np.float64)


def gripper_closure(carriages, open_pos: float, closed_pos: float) -> float:
    """Normalised closure (closed = 1, open = 0), as task_state.gripper_closure."""
    left, right = np.asarray(carriages, dtype=np.float64).reshape(2)
    opening = ((left + right) * 0.5 - closed_pos) / (open_pos - closed_pos)
    return float(1.0 - np.clip(opening, 0.0, 1.0))


def joint_pos(arm_q, carriage_q) -> np.ndarray:
    """Arm joints then (left, right) carriage joints."""
    return np.concatenate([np.asarray(arm_q, dtype=np.float64).reshape(6),
                           np.asarray(carriage_q, dtype=np.float64).reshape(2)])


def joint_vel(arm_qd, carriage_qd) -> np.ndarray:
    return joint_pos(arm_qd, carriage_qd)


def ee_pose(T_base_tcp) -> np.ndarray:
    return pose7(T_base_tcp)


def gripper_width(carriage_q, open_pos: float, closed_pos: float) -> np.ndarray:
    return np.array([gripper_closure(carriage_q, open_pos, closed_pos)])


def object_pose(T_base_obj) -> np.ndarray:
    return pose7(T_base_obj)


def tcp_to_obj_pos(tcp_pos, obj_pos) -> np.ndarray:
    return np.asarray(obj_pos, dtype=np.float64).reshape(3) - np.asarray(tcp_pos, dtype=np.float64).reshape(3)


def obj_to_goal_pos(obj_pos, goal_pos) -> np.ndarray:
    return np.asarray(goal_pos, dtype=np.float64).reshape(3) - np.asarray(obj_pos, dtype=np.float64).reshape(3)


def actor_terms(*, arm_q=None, arm_qd=None, carriage_q=None, carriage_qd=None, T_base_tcp=None,
                T_base_obj=None, goal_pos=None, gripper_open: float, gripper_closed: float
                ) -> Dict[str, np.ndarray]:
    """Every actor term whose inputs are given; a term with a missing input is left out."""
    terms: Dict[str, np.ndarray] = {}
    robot = arm_q is not None and arm_qd is not None and carriage_q is not None and carriage_qd is not None
    if robot:
        terms["qpos"] = joint_pos(arm_q, carriage_q)
        terms["qvel"] = joint_vel(arm_qd, carriage_qd)
    if T_base_tcp is not None:
        terms["ee_pose"] = ee_pose(T_base_tcp)
    if carriage_q is not None:
        terms["gripper_width"] = gripper_width(carriage_q, gripper_open, gripper_closed)
    if T_base_obj is not None:
        obj_pos = np.asarray(T_base_obj)[:3, 3]
        terms["object_pose"] = object_pose(T_base_obj)
        if T_base_tcp is not None:
            terms["tcp_to_obj_pos"] = tcp_to_obj_pos(np.asarray(T_base_tcp)[:3, 3], obj_pos)
        if goal_pos is not None:
            terms["obj_to_goal_pos"] = obj_to_goal_pos(obj_pos, goal_pos)
    return terms


def build_actor_obs(terms: Dict[str, np.ndarray], dtype=np.float32) -> Optional[np.ndarray]:
    """The full actor vector in ACTOR_TERMS order, or None if any term is missing."""
    parts = []
    for name, size in ACTOR_TERMS:
        if name not in terms:
            return None
        value = np.asarray(terms[name], dtype=np.float64).reshape(-1)
        if value.size != size:
            raise ValueError(f"observation term {name} has {value.size} values, expected {size}")
        parts.append(value)
    return np.concatenate(parts).astype(dtype)
