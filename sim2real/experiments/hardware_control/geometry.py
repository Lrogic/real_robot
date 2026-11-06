"""TCP poses as 4x4 transforms in the robot base frame. Pure numpy.

Rotation convention (the training action's, mjlab ``apply_delta_pose``): a rotation
increment ``r`` (rotation vector, base-frame axes, rad) is applied on the left and about the
TCP origin, so the position does not change::

    p_target = p + dp        R_target = exp(r) @ R

The orientation error between poses is the mjlab ``compute_pose_error`` convention,
``r_err = log(R_goal @ R_current^T)`` in base-frame axes, so ``exp(r_err) @ R_current``
is exactly ``R_goal``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import settings  # noqa: F401  (sys.path for common)
from common.poses import matrix_to_rotvec, quaternion_wxyz, rotvec_to_matrix

AXES = {"x": 0, "y": 1, "z": 2}
STEP_LIMITS = ("per_axis", "euclidean")


def make_T(rotation, position) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return T


def axis_vector(axis: str, sign: int) -> np.ndarray:
    if axis not in AXES:
        raise ValueError(f"axis must be one of {tuple(AXES)}, got {axis!r}")
    if sign not in (-1, 1):
        raise ValueError(f"sign must be +1 or -1, got {sign!r}")
    v = np.zeros(3)
    v[AXES[axis]] = float(sign)
    return v


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def displaced(T_ref, translation=(0.0, 0.0, 0.0), rotvec=(0.0, 0.0, 0.0)) -> np.ndarray:
    """T_ref moved by a base-frame translation and rotated by exp(rotvec) about its origin."""
    T_ref = np.asarray(T_ref, dtype=np.float64)
    return make_T(rotvec_to_matrix(rotvec) @ T_ref[:3, :3], T_ref[:3, 3] + np.asarray(translation, dtype=np.float64))


def pose_error(T_current, T_goal) -> tuple[np.ndarray, np.ndarray]:
    """(goal - current position, rotation vector of R_goal @ R_current^T), base frame."""
    T_current = np.asarray(T_current, dtype=np.float64)
    T_goal = np.asarray(T_goal, dtype=np.float64)
    dp = T_goal[:3, 3] - T_current[:3, 3]
    dr = matrix_to_rotvec(T_goal[:3, :3] @ T_current[:3, :3].T)
    return dp, dr


def rotation_angle(R_a, R_b) -> float:
    """Geodesic angle (rad) of the relative rotation R_a^T R_b."""
    R = np.asarray(R_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    s = 0.5 * np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(np.arctan2(s, (np.trace(R) - 1.0) * 0.5))


@dataclass(frozen=True)
class Step:
    T_target: np.ndarray
    translation: np.ndarray     # applied base-frame translation, m
    rotvec: np.ndarray          # applied base-frame rotation vector, rad
    translation_limited: bool   # the step limit shortened the translation
    rotation_limited: bool
    final: bool                 # the target is the goal itself


def bounded_step(T_current, T_goal, max_translation: float, max_rotation: float, step_limit: str) -> Step:
    """New absolute target from T_current toward T_goal that never passes the goal.

    per_axis: each base-frame component of the translation and of the rotation vector is
    clipped to +-limit (training action convention). A component never exceeds its own
    error, so no axis passes the goal. For a multi-axis rotation the clipped rotation
    vector is not parallel to the error, so the path is not the geodesic.

    euclidean: the translation norm and the rotation angle are limited; the step is a
    fraction of the error along the straight line / geodesic and stops at the goal.
    """
    if step_limit not in STEP_LIMITS:
        raise ValueError(f"step_limit must be one of {STEP_LIMITS}, got {step_limit!r}")
    if max_translation <= 0 or max_rotation <= 0:
        raise ValueError("step limits must be positive")
    dp, dr = pose_error(T_current, T_goal)
    if step_limit == "per_axis":
        tp = np.clip(dp, -max_translation, max_translation)
        tr = np.clip(dr, -max_rotation, max_rotation)
        t_limited = bool(np.any(np.abs(dp) > max_translation))
        r_limited = bool(np.any(np.abs(dr) > max_rotation))
    else:
        dist, angle = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
        t_limited, r_limited = dist > max_translation, angle > max_rotation
        tp = dp * (max_translation / dist) if t_limited else dp
        tr = dr * (max_rotation / angle) if r_limited else dr
    final = not (t_limited or r_limited)
    if final:
        T_target = np.array(T_goal, dtype=np.float64)
    else:
        T_target = displaced(T_current, tp, tr)
    return Step(T_target, tp, tr, t_limited, r_limited, final)


def pose_json(T) -> dict:
    T = np.asarray(T, dtype=np.float64)
    return {
        "position_m": T[:3, 3].tolist(),
        "quaternion_wxyz": quaternion_wxyz(T[:3, :3]).tolist(),
        "rotvec_rad": matrix_to_rotvec(T[:3, :3]).tolist(),
        "rotation_matrix": T[:3, :3].tolist(),
    }


def pose_from_json(record: dict) -> np.ndarray:
    return make_T(np.asarray(record["rotation_matrix"], dtype=np.float64), record["position_m"])
