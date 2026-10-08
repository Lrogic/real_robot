"""Rigid-transform helpers. T_destination_source maps source coordinates into destination (metres)."""

import json
from pathlib import Path

import cv2
import numpy as np


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def T_from_rvec(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return make_T(R, tvec)


def rvec_from_R(R) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel()


def inv_T(T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rotation_angle_deg(R) -> float:
    c = (np.trace(np.asarray(R)[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rotation_axis_angle(R):
    """Unit rotation axis and angle (deg); axis is zeros for (near) identity."""
    rv = rvec_from_R(np.asarray(R)[:3, :3])
    angle = float(np.linalg.norm(rv))
    return (rv / angle if angle > 1e-9 else np.zeros(3)), float(np.degrees(angle))


def rotation_between_deg(Ra, Rb) -> float:
    return rotation_angle_deg(np.asarray(Ra)[:3, :3].T @ np.asarray(Rb)[:3, :3])


def mean_rotation(Rs) -> np.ndarray:
    """Chordal L2 mean: SVD projection of the arithmetic mean onto SO(3)."""
    U, _, Vt = np.linalg.svd(np.mean(np.asarray(Rs), axis=0))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def mean_T(Ts) -> np.ndarray:
    Ts = np.asarray(Ts)
    return make_T(mean_rotation(Ts[:, :3, :3]), Ts[:, :3, 3].mean(axis=0))


def transform_points(T, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


def load_T_base_camera(path) -> np.ndarray:
    """Load X = T_base_camera (4x4, metres) written by calibrate.py.

    Use as T_base_object = X @ T_camera_object for poses in the D405 color optical frame.
    """
    with open(Path(path)) as f:
        return np.asarray(json.load(f)["T_base_camera"], dtype=np.float64)
