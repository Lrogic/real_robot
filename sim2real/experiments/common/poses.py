"""Rigid poses as position plus a rotation matrix. Rotations are angle-axis."""

from __future__ import annotations

import numpy as np


def rotvec_to_matrix(rotvec) -> np.ndarray:
    """Angle-axis rotation, the convention of the driver Cartesian orientation."""
    rvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-12:
        return np.eye(3)
    axis = rvec / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_rotvec(rotation) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cos_angle = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-12:
        return np.zeros(3)
    if angle < np.pi - 1e-6:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis /= 2.0 * np.sin(angle)
        return axis * angle
    # At 180 degrees the skew part vanishes: R = 2 a a^T - I.
    diagonal = np.diag(R)
    k = int(np.argmax(diagonal))
    axis = np.zeros(3)
    axis[k] = np.sqrt(max((diagonal[k] + 1.0) * 0.5, 0.0))
    for other in range(3):
        if other != k:
            axis[other] = R[k, other] / (2.0 * axis[k])
    return axis * angle


def rotation_angle_deg(rotation) -> float:
    """Geodesic angle of a rotation matrix, in degrees."""
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cos_angle = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def quaternion_wxyz(rotation) -> np.ndarray:
    """Unit quaternion, wxyz, with non-negative w: the sign rule of every training
    pose observation (``task_state.root_relative_pose``)."""
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (R[2, 1] - R[1, 2]) / scale
        y = (R[0, 2] - R[2, 0]) / scale
        z = (R[1, 0] - R[0, 1]) / scale
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        scale = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / scale
        x = 0.25 * scale
        y = (R[0, 1] + R[1, 0]) / scale
        z = (R[0, 2] + R[2, 0]) / scale
    elif R[1, 1] > R[2, 2]:
        scale = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / scale
        x = (R[0, 1] + R[1, 0]) / scale
        y = 0.25 * scale
        z = (R[1, 2] + R[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / scale
        x = (R[0, 2] + R[2, 0]) / scale
        y = (R[1, 2] + R[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    if quat[0] < 0.0:
        quat = -quat
    return quat / np.linalg.norm(quat)


def pose_record(position_m, rotation) -> dict:
    position = np.asarray(position_m, dtype=np.float64).reshape(3)
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    quat = quaternion_wxyz(matrix)
    return {
        "position_m": position.tolist(),
        "rotation_matrix": matrix.tolist(),
        "rotvec_rad": matrix_to_rotvec(matrix).tolist(),
        "quaternion_wxyz": quat.tolist(),
    }
