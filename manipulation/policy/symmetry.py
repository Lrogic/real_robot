"""Object pose symmetries: pick the symmetric equivalent of a tracked pose closest to a reference.

A symmetry is a mesh-frame transform S with T @ S showing the same geometry as T. For a box
(a Jenga block) the proper symmetries are the identity and 180 deg turns about the three box
axes through the box centre. FoundationPose may lock onto any of them; the policy only saw
the orientations its training resets produced, so the one closest to a reference is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

BOX_SYMMETRY_NAMES = ("identity", "flip_x", "flip_y", "flip_z")


def rotation_angle(Ra, Rb) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    R = np.asarray(Ra, dtype=np.float64).T @ np.asarray(Rb, dtype=np.float64)
    return float(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))


def half_turn(axis) -> np.ndarray:
    """180 deg rotation about a unit axis: 2 a a^T - I."""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    a = a / np.linalg.norm(a)
    return 2.0 * np.outer(a, a) - np.eye(3)


def box_symmetries(center=(0.0, 0.0, 0.0), axes=np.eye(3)) -> Tuple[np.ndarray, ...]:
    """Identity, then 180 deg about each box axis, as 4x4 mesh-frame transforms.

    center is the box centre and the columns of axes are the box axes, both in the mesh
    frame. With C = [axes | center], each symmetry is S = C @ Rot180 @ C^-1 (the
    oriented-bounds form inv(to_origin) @ Rot180 @ to_origin of the object-frame experiment).
    """
    C = np.eye(4)
    C[:3, :3] = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    C[:3, 3] = np.asarray(center, dtype=np.float64).reshape(3)
    C_inv = np.linalg.inv(C)
    out = [np.eye(4)]
    for k in range(3):
        flip = np.eye(4)
        flip[:3, :3] = half_turn(np.eye(3)[k])
        out.append(C @ flip @ C_inv)
    return tuple(out)


def candidates(T, symmetries: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
    T = np.asarray(T, dtype=np.float64)
    return tuple(T @ S for S in symmetries)


@dataclass(frozen=True)
class SymmetryChoice:
    index: int
    T: np.ndarray                  # the chosen candidate, T_raw @ S_index
    angles_rad: Tuple[float, ...]  # rotation distance of every candidate to the reference


def select_closest(T_raw, symmetries: Sequence[np.ndarray], R_ref) -> SymmetryChoice:
    """The candidate whose rotation is closest to R_ref (ties go to the lower index)."""
    poses = candidates(T_raw, symmetries)
    angles = tuple(rotation_angle(R_ref, T[:3, :3]) for T in poses)
    k = int(np.argmin(angles))
    return SymmetryChoice(k, poses[k], angles)
